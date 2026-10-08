"""Ground-truth 2D/3D point tracking in ManiSkill.

Pixels picked in one frame are lifted to 3D with the camera's position texture, pinned to the rigid body
they land on (via the segmentation id), and carried through the trajectory with that body's pose. This is
exact for rigid scenes (no learned tracker, no drift). Single env only (index 0).

    env = maniskill.make_env(TASK, obs_mode="rgb+position+segmentation", sensor_configs=dict(width=512, height=512))
    obs, _ = env.reset(seed=0)
    pts = tracking.grid_points(env, obs, "base_camera", grid_size=32)  # or sample_points / init_points
    uv, xyz, vis = tracking.locate(env, obs, pts)  # call again after every env.step
    tracks = tracking.track_demo("PickCube-v1", grid_size=32)  # or all of the above over a demo replay
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

OBS_MODE = "rgb+position+segmentation"  # textures the functions below need from the camera


@dataclass
class TrackedPoints:
    cam: str  # sensor camera uid the points are projected into, e.g. "base_camera"
    bodies: list  # [N] the Actor/Link each point is pinned to
    p_local: torch.Tensor  # [N, 3] each point in its body's frame

    @property
    def body_names(self) -> list[str]:
        return [b.name for b in self.bodies]


def _camera(obs: dict, cam: str):
    """Position (OpenGL camera frame, meters), segmentation id and camera params for env 0."""
    data, param = obs["sensor_data"][cam], obs["sensor_param"][cam]
    pos_gl = data["position"][0].float() / 1000.0  # [H, W, 3], raw texture is int16 millimeters
    seg = data["segmentation"][0, ..., 0].long()  # [H, W], 0 = nothing rendered
    return pos_gl, seg, param["cam2world_gl"][0], param["extrinsic_cv"][0], param["intrinsic_cv"][0]


def init_points(env, obs: dict, cam: str, uv) -> TrackedPoints:
    """Pin pixels `uv` ([N, 2] integer (u=col, v=row)) of the current frame to the bodies they land on."""
    uv = torch.as_tensor(np.asarray(uv), dtype=torch.long)
    pos_gl, seg, cam2world, _, _ = _camera(obs, cam)
    pos_gl, seg = pos_gl.to(uv.device), seg.to(uv.device)
    ids = seg[uv[:, 1], uv[:, 0]]
    assert (ids != 0).all(), "some pixels hit no geometry (segmentation id 0)"

    p_cam = torch.cat([pos_gl[uv[:, 1], uv[:, 0]], torch.ones(len(uv), 1)], dim=-1)  # [N, 4]
    p_world = p_cam @ cam2world.cpu().T

    id_map = env.unwrapped.segmentation_id_map
    bodies = [id_map[int(i)] for i in ids]
    p_local = torch.empty(len(uv), 3)
    for body in set(bodies):
        idx = [i for i, b in enumerate(bodies) if b is body]
        world2body = body.pose.inv().to_transformation_matrix()[0].cpu()
        p_local[idx] = (p_world[idx] @ world2body.T)[:, :3]
    return TrackedPoints(cam, bodies, p_local)


def sample_points(
    env, obs: dict, cam: str, per_body: int = 20, exclude=("ground",), seed: int = 0
) -> TrackedPoints:
    """Sample up to `per_body` random visible pixels on every body in view (except `exclude` by name)."""
    _, seg, _, _, _ = _camera(obs, cam)
    seg = seg.cpu().numpy()
    id_map = env.unwrapped.segmentation_id_map
    rng = np.random.default_rng(seed)
    uv = []
    for sid in np.unique(seg):
        if sid == 0 or id_map[int(sid)].name in exclude:
            continue
        v, u = np.nonzero(seg == sid)
        pick = rng.choice(len(u), size=min(per_body, len(u)), replace=False)
        uv.append(np.stack([u[pick], v[pick]], axis=-1))
    return init_points(env, obs, cam, np.concatenate(uv))


def grid_points(env, obs: dict, cam: str, grid_size: int = 32, exclude=()) -> TrackedPoints:
    """A uniform `grid_size` x `grid_size` grid of query pixels (cell centers) over the whole image, like the
    grid queries of foundation point trackers (e.g. CoTracker). Pixels on nothing or on `exclude` are dropped."""
    _, seg, _, _, _ = _camera(obs, cam)
    seg = seg.cpu().numpy()
    H, W = seg.shape
    u, v = np.meshgrid((np.arange(grid_size) + 0.5) * W / grid_size, (np.arange(grid_size) + 0.5) * H / grid_size)
    uv = np.stack([u.ravel(), v.ravel()], axis=-1).astype(int)
    id_map = env.unwrapped.segmentation_id_map
    ids = seg[uv[:, 1], uv[:, 0]]
    keep = np.array([i != 0 and id_map[int(i)].name not in exclude for i in ids], dtype=bool)
    return init_points(env, obs, cam, uv[keep])


def locate(env, obs: dict, pts: TrackedPoints, depth_tol: float = 0.005):
    """Where the tracked points are in the current frame.

    Returns (numpy):
        uv: [N, 2] float pixel coords (u=col, v=row, same convention as `init_points`); may fall outside the image.
        xyz: [N, 3] world coords.
        visible: [N] bool, in frame and not occluded (projected depth within `depth_tol` m of rendered depth).
    """
    pos_gl, seg, _, extrinsic, intrinsic = _camera(obs, pts.cam)
    pos_gl, seg, extrinsic, intrinsic = pos_gl.cpu(), seg.cpu(), extrinsic.cpu(), intrinsic.cpu()
    H, W = seg.shape

    p_world = torch.empty(len(pts.p_local), 4)
    p_local = torch.cat([pts.p_local, torch.ones(len(pts.p_local), 1)], dim=-1)
    for body in set(pts.bodies):
        idx = [i for i, b in enumerate(pts.bodies) if b is body]
        body2world = body.pose.to_transformation_matrix()[0].cpu()
        p_world[idx] = p_local[idx] @ body2world.T

    p_cv = p_world @ extrinsic.T  # [N, 3], OpenCV camera frame (+z forward)
    z = p_cv[:, 2]
    uv = (p_cv @ intrinsic.T)[:, :2] / z[:, None] - 0.5  # intrinsics put pixel (u, v)'s center at u+0.5, v+0.5

    u, v = uv[:, 0].round().long(), uv[:, 1].round().long()
    in_frame = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v = u.clamp(0, W - 1), v.clamp(0, H - 1)
    rendered_z = -pos_gl[v, u, 2]  # OpenGL camera looks down -z
    rendered_z[seg[v, u] == 0] = float("inf")  # nothing rendered there, so nothing can occlude
    visible = in_frame & (z <= rendered_z + depth_tol)
    return uv.numpy(), p_world[:, :3].numpy(), visible.numpy()


@dataclass
class Tracks:
    frames: np.ndarray  # [T, H, W, 3] uint8 camera frames
    uv: np.ndarray  # [T, N, 2] pixel coords
    xyz: np.ndarray  # [T, N, 3] world coords
    visible: np.ndarray  # [T, N] bool
    body_names: np.ndarray  # [N] body each point is pinned to
    success: bool  # task success at the end of the replay
    fps: float  # control frequency, for playback


def track_demo(
    task: str, cam: str = "base_camera", grid_size: int = 32, resolution: int = 512, eye=None, target=None, fov=None
) -> Tracks:
    """Replay `task`'s demo (`api.maniskill.load_demo`) and track a uniform `grid_size`^2 grid of frame-0 points
    through it in camera `cam`. `eye`/`target` (world xyz) and `fov` (rad) optionally re-place the camera."""
    from mani_skill.utils import sapien_utils

    from api import maniskill

    cam_cfg = {}
    if eye is not None:
        cam_cfg["pose"] = sapien_utils.look_at(eye=eye, target=target)
    if fov is not None:
        cam_cfg["fov"] = fov
    demo = maniskill.load_demo(task)
    env = maniskill.make_env(
        task, num_envs=1, obs_mode=OBS_MODE,
        sensor_configs={"width": resolution, "height": resolution, cam: cam_cfg},
        control_mode=demo.control_mode, max_episode_steps=len(demo.actions) + 1,
    )  # fmt: skip

    obs, _ = env.reset(**demo.reset_kwargs)
    pts = grid_points(env, obs, cam, grid_size=grid_size)
    frames, out = [], []
    for t in range(len(demo.actions) + 1):
        if t > 0:
            obs, _, _, _, info = env.step(torch.from_numpy(demo.actions[t - 1]).unsqueeze(0))
        frames.append(obs["sensor_data"][cam]["rgb"][0].cpu().numpy())
        out.append(locate(env, obs, pts))
    fps = env.unwrapped.control_freq
    env.close()
    uv, xyz, visible = (np.stack(x) for x in zip(*out))
    return Tracks(np.stack(frames), uv, xyz, visible, np.array(pts.body_names), bool(info["success"].item()), fps)


# ---- batched (all envs at once, on the sim device) ----
# Same math as init_points/locate above, but vectorized over envs and points with no host round trips, for
# throughput with GPU sim. Points live in a fixed [B, N] layout; `valid` masks queries that hit no geometry.


@dataclass
class BatchedPoints:
    cam: str
    bodies: list  # [K] every segmentable body (Actor/Link); each one's pose is batched over envs
    body_idx: torch.Tensor  # [B, N] long, index into `bodies`
    p_local: torch.Tensor  # [B, N, 4] homogeneous point in its body's frame
    valid: torch.Tensor  # [B, N] bool


def _body_transforms(bodies: list) -> torch.Tensor:
    """[K, B, 4, 4] body-to-world transforms of every body in every env."""
    return torch.stack([b.pose.to_transformation_matrix() for b in bodies])


def _gather(T: torch.Tensor, body_idx: torch.Tensor) -> torch.Tensor:
    """Per-point transforms [B, N, 4, 4] from T [K, B, 4, 4] and body_idx [B, N]."""
    return T[body_idx, torch.arange(body_idx.shape[0], device=body_idx.device)[:, None]]


def grid_points_batched(env, obs: dict, cam: str, grid_size: int = 32) -> BatchedPoints:
    """`grid_points` for every env at once: a `grid_size`^2 grid of pixel queries pinned to their bodies."""
    data, param = obs["sensor_data"][cam], obs["sensor_param"][cam]
    seg = data["segmentation"][..., 0].long()  # [B, H, W]
    B, H, W = seg.shape
    dev = seg.device
    u = ((torch.arange(grid_size, device=dev) + 0.5) * W / grid_size).long()
    v = ((torch.arange(grid_size, device=dev) + 0.5) * H / grid_size).long()
    v, u = (x.ravel() for x in torch.meshgrid(v, u, indexing="ij"))  # [N] row-major, same order as grid_points

    id_map = env.unwrapped.segmentation_id_map
    bodies = list(id_map.values())
    lut = torch.zeros(max(id_map) + 1, dtype=torch.long, device=dev)
    lut[torch.tensor(list(id_map), device=dev)] = torch.arange(len(bodies), device=dev)

    ids = seg[:, v, u]  # [B, N]
    valid = ids != 0
    body_idx = lut[ids]  # id 0 maps to body 0; masked out by `valid`
    # keep only bodies some query landed on, so locate_batched fetches as few poses per step as possible
    used, body_idx = torch.unique(body_idx, return_inverse=True)
    bodies = [bodies[i] for i in used.tolist()]

    pos_gl = data["position"][:, v, u].float() / 1000.0  # [B, N, 3]
    p_cam = torch.cat([pos_gl, torch.ones_like(pos_gl[..., :1])], dim=-1)
    p_world = p_cam @ param["cam2world_gl"].transpose(1, 2)  # [B, N, 4]
    world2body = torch.linalg.inv(_gather(_body_transforms(bodies), body_idx))
    p_local = (world2body @ p_world[..., None])[..., 0]
    return BatchedPoints(cam, bodies, body_idx, p_local, valid)


def locate_batched(obs: dict, pts: BatchedPoints, depth_tol: float = 0.005):
    """`locate` for every env at once. Returns torch tensors on the sim device:
    uv [B, N, 2], xyz [B, N, 3], visible [B, N] bool (False for invalid queries)."""
    data, param = obs["sensor_data"][pts.cam], obs["sensor_param"][pts.cam]
    pos_z = data["position"][..., 2]  # [B, H, W] int16 mm, OpenGL camera looks down -z
    seg = data["segmentation"][..., 0]
    B, H, W = seg.shape

    p_world = (_gather(_body_transforms(pts.bodies), pts.body_idx) @ pts.p_local[..., None])[..., 0]  # [B, N, 4]
    p_cv = p_world @ param["extrinsic_cv"].transpose(1, 2)  # [B, N, 3]
    z = p_cv[..., 2]
    uv = (p_cv @ param["intrinsic_cv"].transpose(1, 2))[..., :2] / z[..., None] - 0.5

    u, v = uv[..., 0].round().long(), uv[..., 1].round().long()
    in_frame = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    flat = (torch.arange(B, device=u.device)[:, None] * H + v.clamp(0, H - 1)) * W + u.clamp(0, W - 1)
    rendered_z = -pos_z.reshape(-1)[flat].float() / 1000.0
    rendered_z = torch.where(seg.reshape(-1)[flat] == 0, torch.inf, rendered_z)
    visible = pts.valid & in_frame & (z <= rendered_z + depth_tol)
    return uv, p_world[..., :3], visible
