"""Jenga-style modular-design task (ManiSkill): stack 3x1x1 blocks into a target 5x5x5 design.

Importing this registers `JengaDesign-v1`; `api.maniskill.make_env` already imports it.
Spec: ai-notes/Design for Jenga.md. Design representation and scoring live in `api.jenga_core`.

World layout (table top is z = 0, robot base at [-0.615, 0, 0] facing +x, as in StackThree-v1):
    y < 0 half: the 5x5 design grid (visual-only marker), just in front of and to the right of the robot.
    y > 0 half: the randomly scattered blocks (SCATTER_* constants), on the robot's left.
    The robot stands on y = 0, between the two halves.

Indexing (same as jenga_core): design[i, j, k], i -> world +x, j -> world +y, k -> world +z.
    Cell (i, j, k) is the world box GRID_ORIGIN + CELL_SIZE * ([i, i+1] x [j, j+1] x [k, k+1]),
    i.e. GRID_ORIGIN is the min corner of cell [0, 0, 0] (k = 0 rests on the table). On the marker,
    cell [0, 0] (min x, min y: the robot-side corner furthest from the scatter half) is filled red,
    a red stripe runs along +i (x) and a green one along +j (y).

Block actors: 12 x 4 x 4 cm boxes whose long axis is their local x. Actors are
`env.blocks[0 .. n_design - 1]` for the design blocks (in `env.design_blocks` order, i.e. bottom-up)
followed by `num_spare_blocks` spares. Any actor may be used for any design slot: the score only
looks at which cells are occupied, not by which block.
"""

from __future__ import annotations

import colorsys
import time
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from api.jenga_core import BLOCK_LEN, CELL_SIZE, GRID, Block, design_to_blocks, empty_design, make_block, overlap_score

# ---- layout ----
ROBOT_BASE = np.array([-0.615, 0.0, 0.0])
# min corner of cell [0, 0, 0]: grid spans x in [-0.18, 0.02] (0.435-0.635 m in front of the base) and
# y in [-0.26, -0.06]. Close enough that top-down grasps reach every cell at every layer in both yaws.
GRID_ORIGIN = np.array([-0.18, -0.26, 0.0])
BLOCK_HALF_SIZE = np.array([BLOCK_LEN, 1, 1]) * CELL_SIZE / 2  # [0.06, 0.02, 0.02]
# block *centers* are sampled in this xy box (y > 0 half), further restricted to an annulus around the
# robot base so every block is reachable with a top-down grasp
SCATTER_XY_LOW = np.array([-0.42, 0.06])
SCATTER_XY_HIGH = np.array([0.12, 0.50])
SCATTER_RADIUS = (0.30, 0.70)  # allowed horizontal distance of a block center from the robot base
SCATTER_GAP = 0.025  # min clearance between scattered blocks, so the open gripper fingers fit beside a block
NUM_SPARE_BLOCKS = 2

# ---- stability check (spec section 5) ----
SETTLE_TIME = 1.0  # seconds of simulated time
STABILITY_TOL = 0.005  # max allowed displacement (m) of any block from its design pose
# blocks are placed this far above their resting height *per layer below them*, so stacked boxes never
# start interpenetrating; a k=4 block starts at most 5 * 0.2 mm = 1 mm high and simply settles
PLACEMENT_GAP = 2e-4


def _example_design() -> np.ndarray:
    """The worked example of spec section 2, as confirmed by the author: 11 horizontal blocks, 5 / 3 / 3 in layers k = 0 / 1 / 2.

    Sketch mapping (top-down sketch, rows 1-5 top to bottom, cols 1-5 left to right):
        i = col - 1, j = 5 - row, so `format_design` prints the layers exactly as sketched.
    k=1 (red, read confidently): rows 2 and 4 span cols 1-3, row 3 spans cols 3-5.
    k=0 (black): bars along j (sketch columns) in cols 1, 3, 5 over rows 2-4, plus rows 1 and 5 cols 2-4.
        Every k=1 bar rests on two k=0 bars, one under each end.
    k=2 (blue): bars along j in cols 1 and 2 over rows 2-4 (each bridging the k=1 bars of rows 2 and 4),
        plus row 3 cols 3-5 lying fully on its k=1 twin.
    """
    blocks = [
        # k = 0
        make_block(1, (0, 1, 0), axis=1),  # col 1, rows 2-4
        make_block(2, (2, 1, 0), axis=1),  # col 3, rows 2-4
        make_block(3, (4, 1, 0), axis=1),  # col 5, rows 2-4
        make_block(4, (1, 4, 0), axis=0),  # row 1, cols 2-4
        make_block(5, (1, 0, 0), axis=0),  # row 5, cols 2-4 (mirrors block 4)
        # k = 1
        make_block(6, (0, 3, 1), axis=0),  # row 2, cols 1-3
        make_block(7, (2, 2, 1), axis=0),  # row 3, cols 3-5
        make_block(8, (0, 1, 1), axis=0),  # row 4, cols 1-3
        # k = 2
        make_block(9, (0, 1, 2), axis=1),  # col 1, rows 2-4
        make_block(10, (1, 1, 2), axis=1),  # col 2, rows 2-4
        make_block(11, (2, 2, 2), axis=0),  # row 3, cols 3-5
    ]
    design = empty_design()
    for b in blocks:
        design[tuple(b.cells.T)] = b.id
    return design


EXAMPLE_DESIGN = _example_design()

# actor-local x (the block's long axis) -> world axis 0 / 1 / 2
_AXIS_QUATS = [
    np.array([1.0, 0.0, 0.0, 0.0]),
    euler2quat(0, 0, np.pi / 2),  # yaw 90 deg: local x -> world +y
    euler2quat(0, -np.pi / 2, 0),  # pitch -90 deg: local x -> world +z
]


def block_world_pose(block: Block, grid_origin: np.ndarray = GRID_ORIGIN) -> sapien.Pose:
    """World pose of a design block: center of its 3 cells, actor local x along the block's axis."""
    return sapien.Pose(p=grid_origin + block.center * CELL_SIZE, q=_AXIS_QUATS[block.axis])


def _segment_centers(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """[..., 3] positions, [..., 4] wxyz quats -> [..., 3 segments, 3] world centers of the block's
    3 unit cubes along its long (local x) axis."""
    w, x, y, z = q.unbind(-1)
    long_axis = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1)
    offsets = torch.arange(BLOCK_LEN, device=p.device, dtype=p.dtype) - (BLOCK_LEN - 1) / 2
    return p[..., None, :] + offsets[:, None] * CELL_SIZE * long_axis[..., None, :]


def _pose_cells(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    origin = torch.as_tensor(GRID_ORIGIN, device=p.device, dtype=p.dtype)
    return torch.floor((_segment_centers(p, q) - origin) / CELL_SIZE).long()


def pose_to_cells(pose: Union[sapien.Pose, Pose]) -> np.ndarray:
    """[3, 3] int (i, j, k) cells holding most of the mass of a block at `pose` (spec section 3): the cells
    containing the centers of its 3 unit cubes along its actual long axis. May lie outside the grid."""
    if isinstance(pose, sapien.Pose):
        p, q = torch.tensor(pose.p, dtype=torch.float64), torch.tensor(pose.q, dtype=torch.float64)
    else:
        p, q = pose.p[0].double().cpu(), pose.q[0].double().cpu()
    return _pose_cells(p, q).numpy()


def _occupancy_from_poses(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """[B, N, 3] block positions, [B, N, 4] quats -> [B, 5, 5, 5] bool occupancy (out-of-grid cells dropped)."""
    cells = _pose_cells(p, q).reshape(p.shape[0], -1, 3)  # [B, N * 3, 3]
    inside = ((cells >= 0) & (cells < GRID)).all(-1)
    b = torch.arange(p.shape[0], device=p.device)[:, None].expand_as(inside)
    occ = torch.zeros((p.shape[0], GRID, GRID, GRID), dtype=torch.bool, device=p.device)
    occ[b[inside], cells[..., 0][inside], cells[..., 1][inside], cells[..., 2][inside]] = True
    return occ


def built_occupancy(env, env_idx: int = 0) -> np.ndarray:
    """bool (5, 5, 5) occupancy of the structure currently built in sub-env `env_idx`, from all block poses."""
    return env.unwrapped._occupancy()[env_idx].cpu().numpy()


def _block_colors(n: int) -> list[list[float]]:
    """n visually distinct RGBA colors: golden-ratio hue steps, cycling saturation/value for large n."""
    colors = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        s, v = [(0.85, 0.95), (0.55, 0.85), (0.95, 0.6)][(i // 7) % 3]
        colors.append([*colorsys.hsv_to_rgb(h, s, v), 1.0])
    return colors


def _rects_overlap(a: np.ndarray, b: np.ndarray, gap: float) -> bool:
    """2D separating-axis test for two (x, y, yaw) block footprints, each inflated by gap / 2."""
    half = BLOCK_HALF_SIZE[:2] + gap / 2
    axes = []
    for yaw in (a[2], b[2]):
        c, s = np.cos(yaw), np.sin(yaw)
        axes.append(np.array([[c, s], [-s, c]]))  # rows: local x, local y in world
    d = b[:2] - a[:2]
    for R in axes:
        for ax in R:
            ra = np.abs(axes[0] @ ax) @ half
            rb = np.abs(axes[1] @ ax) @ half
            if abs(d @ ax) > ra + rb:
                return False
    return True


def sample_scatter(rng: np.random.RandomState, n: int, max_tries: int = 300, restarts: int = 100) -> np.ndarray:
    """n collision-free, reachable (x, y, yaw) footprints on the scatter half of the table. Sequential rejection
    sampling; if a block finds no free spot, start over (random sequential packing can jam itself)."""
    for _ in range(restarts):
        out: list[np.ndarray] = []
        for _ in range(n):
            for _ in range(max_tries):
                xy = rng.uniform(SCATTER_XY_LOW, SCATTER_XY_HIGH)
                r = np.linalg.norm(xy - ROBOT_BASE[:2])
                if not SCATTER_RADIUS[0] <= r <= SCATTER_RADIUS[1]:
                    continue
                cand = np.array([*xy, rng.uniform(-np.pi, np.pi)])
                if not any(_rects_overlap(cand, o, SCATTER_GAP) for o in out):
                    out.append(cand)
                    break
            else:
                break
        if len(out) == n:
            return np.stack(out)
    raise RuntimeError(f"could not scatter {n} blocks; enlarge the scatter region or use fewer blocks")


@register_env("JengaDesign-v1", max_episode_steps=2000)
class JengaDesignEnv(BaseEnv):
    """Build the target `design` out of 12x4x4 cm blocks scattered on the other half of the table.

    Args (besides BaseEnv's):
        design: (5, 5, 5) int design (jenga_core convention); nested lists are fine and keep the gym
            kwargs JSON-serialisable for RecordEpisode. Default EXAMPLE_DESIGN.
        num_spare_blocks: extra block actors beyond the design's block count.
        scatter: if False, every block is parked off the table at reset (used by `check_stability`).

    info: `score` = overlap_score(design, built occupancy), `max_score` = number of design cells,
    `success` = score == max_score. Reward: sparse = success, dense = score, normalized_dense = score / max.
    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["sparse", "dense", "normalized_dense", "none"]
    agent: Union[Panda, Fetch]
    TABLE_SCENE_BUILDER = TableSceneBuilder  # subclasses with other embodiments swap in a builder that knows their robot

    def __init__(
        self, *args, design=None, num_spare_blocks: int = NUM_SPARE_BLOCKS, scatter: bool = True,
        robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs,
    ):
        self.design = np.asarray(EXAMPLE_DESIGN if design is None else design, dtype=np.int64)
        self.design_blocks = design_to_blocks(self.design)  # bottom-up
        self.num_blocks = len(self.design_blocks) + num_spare_blocks
        self.scatter = scatter
        self.robot_init_qpos_noise = robot_init_qpos_noise
        kwargs.setdefault("reward_mode", "sparse")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        # identical to StackThree-v1
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # front-right corner of the table looking back at the robot: grid in the foreground, robot and
        # scattered blocks behind it, nothing occluded by the arm in its rest pose
        pose = sapien_utils.look_at([0.55, -0.7, 0.9], [-0.25, 0.05, 0.15])
        return CameraConfig("render_camera", pose, 512, 512, 1.1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=ROBOT_BASE))

    def _load_scene(self, options: dict):
        self.table_scene = self.TABLE_SCENE_BUILDER(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self._build_grid_marker()
        self.blocks = [
            actors.build_box(
                self.scene, half_sizes=BLOCK_HALF_SIZE.tolist(), color=c, name=f"block{n}",
                initial_pose=sapien.Pose(p=self._park_position(n)),
            )
            for n, c in enumerate(_block_colors(self.num_blocks))
        ]
        self.target_poses = [block_world_pose(b) for b in self.design_blocks]
        self._target_occ = torch.as_tensor(self.design != 0, device=self.device)

    def _build_grid_marker(self):
        """Visual-only (no collision shapes) 5x5 grid on the table, so it can never perturb blocks."""
        builder = self.scene.create_actor_builder()
        size, h, w = GRID * CELL_SIZE, 5e-4, 2.5e-3  # grid side, marker thickness, line width
        mat = lambda rgba: sapien.render.RenderMaterial(base_color=rgba)
        builder.add_box_visual(  # pale base plate
            sapien.Pose(p=[size / 2, size / 2, h / 2]), [size / 2, size / 2, h / 2], mat([0.92, 0.9, 0.8, 1])
        )
        for n in range(GRID + 1):
            builder.add_box_visual(sapien.Pose(p=[n * CELL_SIZE, size / 2, h]), [w / 2, size / 2 + w / 2, h], mat([0.15] * 3 + [1]))
            builder.add_box_visual(sapien.Pose(p=[size / 2, n * CELL_SIZE, h]), [size / 2 + w / 2, w / 2, h], mat([0.15] * 3 + [1]))
        # cell [0, 0] in red, +i (x) stripe red and +j (y) stripe green just outside the grid
        builder.add_box_visual(sapien.Pose(p=[CELL_SIZE / 2, CELL_SIZE / 2, h]), [CELL_SIZE / 2 - w, CELL_SIZE / 2 - w, h], mat([0.9, 0.2, 0.2, 1]))
        builder.add_box_visual(sapien.Pose(p=[size / 2, -0.012, h]), [size / 2, 0.004, h], mat([0.9, 0.2, 0.2, 1]))
        builder.add_box_visual(sapien.Pose(p=[-0.012, size / 2, h]), [0.004, size / 2, h], mat([0.2, 0.75, 0.2, 1]))
        builder.initial_pose = sapien.Pose(p=GRID_ORIGIN)
        self.grid_marker = builder.build_kinematic(name="grid_marker")

    @staticmethod
    def _park_position(n: int) -> np.ndarray:
        """Off-table resting spot for block n: on the floor (z = -0.9196) far behind the robot, 25 cm apart."""
        return np.array([-3.0 - 0.25 * (n // 8), 0.25 * (n % 8 - 3.5), -0.9196429 + BLOCK_HALF_SIZE[2]])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            b = len(env_idx)
            p = torch.zeros((b, self.num_blocks, 3))
            q = torch.zeros((b, self.num_blocks, 4))
            for row, e in enumerate(env_idx.tolist()):
                if self.scatter:
                    fp = sample_scatter(self._batched_episode_rng[e], self.num_blocks)
                    p[row, :, :2] = torch.as_tensor(fp[:, :2], dtype=torch.float32)
                    p[row, :, 2] = BLOCK_HALF_SIZE[2]  # lying flat on the table
                    q[row] = torch.as_tensor(np.stack([euler2quat(0, 0, yaw) for yaw in fp[:, 2]]), dtype=torch.float32)
                else:
                    p[row] = torch.as_tensor(np.stack([self._park_position(n) for n in range(self.num_blocks)]), dtype=torch.float32)
                    q[row, :, 0] = 1
            for n, blk in enumerate(self.blocks):
                blk.set_pose(Pose.create_from_pq(p=p[:, n], q=q[:, n]))

    def _occupancy(self) -> torch.Tensor:
        p = torch.stack([blk.pose.p for blk in self.blocks], dim=1)
        q = torch.stack([blk.pose.q for blk in self.blocks], dim=1)
        return _occupancy_from_poses(p, q)

    def evaluate(self):
        occ = self._occupancy()
        score = (occ & self._target_occ).flatten(1).sum(-1)
        max_score = int(self._target_occ.sum())
        return {
            "score": score,
            "max_score": torch.full_like(score, max_score),
            "success": score == max_score,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(block_poses=torch.stack([blk.pose.raw_pose for blk in self.blocks], dim=1).flatten(1))
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["success"].float()

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["score"].float()

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return info["score"].float() / info["max_score"].float()

    # ---- helpers for tests / visualisation / stability ----

    def place_design(self, design: np.ndarray | None = None, lift: float = 0.0):
        """Teleport blocks to the design poses of `design` (default: the env's own target) in sub-env 0,
        parking the remaining blocks off the table. Layer k is raised by (k + 1) * lift."""
        design_blocks = self.design_blocks if design is None else design_to_blocks(design)
        if len(design_blocks) > self.num_blocks:
            raise ValueError(f"design has {len(design_blocks)} blocks but the env only has {self.num_blocks}")
        for n, blk in enumerate(self.blocks):
            if n < len(design_blocks):
                pose = block_world_pose(design_blocks[n])
                pose.set_p(pose.p + [0, 0, (design_blocks[n].cells[:, 2].min() + 1) * lift])
            else:
                pose = sapien.Pose(p=self._park_position(n))
            blk.set_pose(pose)
            blk.set_linear_velocity(torch.zeros(3))
            blk.set_angular_velocity(torch.zeros(3))
        return design_blocks


@dataclass
class StabilityResult:
    stable: bool
    max_dev: np.ndarray  # [n_blocks] max displacement (m) of each block from its design pose, in design_to_blocks order
    max_rot: np.ndarray  # [n_blocks] max rotation (rad) from its design orientation
    blocks: list  # the design's Blocks, same order


_STABILITY_ENV = None


def _stability_env() -> JengaDesignEnv:
    """One cached CPU-sim env (num_envs=1, default sim config = the demo env's) with MAX_BLOCKS block actors,
    no rendering, and the robot moved far away."""
    global _STABILITY_ENV
    if _STABILITY_ENV is None:
        import gymnasium as gym

        from api.jenga_core import MAX_BLOCKS

        env = gym.make(
            "JengaDesign-v1", num_envs=1, design=empty_design().tolist(), num_spare_blocks=MAX_BLOCKS,
            scatter=False, obs_mode="state", sim_backend="physx_cpu", render_mode=None,
        ).unwrapped
        env.reset(seed=0)
        env.agent.robot.set_root_pose(sapien.Pose(p=[-3.0, 3.0, 0.0]))  # far from the grid and parked blocks
        _STABILITY_ENV = env
    return _STABILITY_ENV


def check_stability(
    design: np.ndarray, settle_time: float = SETTLE_TIME, tol: float = STABILITY_TOL, check_every: int = 5
) -> StabilityResult:
    """Spec section 5: place every design block exactly at its design pose (raised by PLACEMENT_GAP per layer
    to avoid initial interpenetration), run `settle_time` s of physics and record each block's max deviation
    from its (gap-free) design pose. Stable iff every block stays within `tol`.

    Uses a cached env with the same backend and sim settings as `make_env("JengaDesign-v1")` (CPU PhysX,
    num_envs=1). Poses are sampled every `check_every` physics steps; the run stops early once any block has
    moved > 10 * tol, so max_dev of a collapsing design is a lower bound.
    """
    env = _stability_env()
    blocks = env.place_design(np.asarray(design), lift=PLACEMENT_GAP)
    ents = [env.blocks[n]._objs[0] for n in range(len(blocks))]
    target_p = np.stack([block_world_pose(b).p for b in blocks])
    target_R = np.stack([quat2mat(_AXIS_QUATS[b.axis]) for b in blocks])
    for e in ents:
        e.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).wake_up()

    max_dev, max_rot = np.zeros(len(blocks)), np.zeros(len(blocks))
    n_steps = int(round(settle_time * env.sim_freq))
    for step in range(1, n_steps + 1):
        env.scene.step()
        if step % check_every == 0 or step == n_steps:
            poses = [e.pose for e in ents]
            p = np.stack([ps.p for ps in poses])
            R = np.stack([quat2mat(ps.q) for ps in poses])
            max_dev = np.maximum(max_dev, np.linalg.norm(p - target_p, axis=1))
            cos = (np.einsum("nij,nij->n", R, target_R) - 1) / 2  # trace(R_target^T R) = 1 + 2 cos(angle)
            max_rot = np.maximum(max_rot, np.arccos(np.clip(cos, -1, 1)))
            if max_dev.max() > 10 * tol:  # clearly collapsing, no need to keep simulating
                break
    return StabilityResult(bool((max_dev < tol).all()), max_dev, max_rot, blocks)


def time_stability_check(design: np.ndarray = EXAMPLE_DESIGN, n: int = 10) -> float:
    """Mean seconds per check_stability call (after the cached env is built)."""
    check_stability(design)
    t = time.perf_counter()
    for _ in range(n):
        check_stability(design)
    return (time.perf_counter() - t) / n
