"""Motion-planned scripted solver for `JengaDesign-v1` (spec section 9): builds the env's target design bottom-up
from the scattered blocks, using privileged simulator state (block poses), not vision.

    from api.jenga_solver import solve_jenga
    res = solve_jenga(env, seed=0)  # env made with control_mode="pd_joint_pos"
    res["score"], res["max_score"], res["blocks"]

Per design block: top-down grasp across the block's 4 cm side (two grasp yaws, the gripper being symmetric),
lift above everything built so far, carry it over the grid while turning its long axis onto the target axis,
lower it to just above its design pose and release, then retreat upward. The place pose is computed from the
block's *measured* pose in the gripper (re-measured right above the target), not from the nominal grasp.

Finger clearance (the reason for most of the logic below): a closed Panda fingertip sticks out ~1.5 cm beyond
the block's side, so a block cannot be lowered flush against an already-placed neighbour on its finger side.
  * Within a layer, blocks are ordered so that a block is placed before the blocks touching it on its finger
    sides (a topological sort of that "must come before" relation).
  * When that is impossible (two parallel, side-by-side blocks), the second block is released from just above
    the neighbour tops ("drop" release, ~2.5 cm), so the fingers never go beside the neighbour.
Vertical (axis = 2) design blocks are out of scope for v1 and are skipped.
"""

from __future__ import annotations

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver

from api.jenga import BLOCK_HALF_SIZE, CELL_SIZE, GRID_ORIGIN, pose_to_cells
from api.jenga_core import GRID, Block

# ---- gripper geometry / solver constants ----
GRASP_HEIGHT = 0.008  # TCP this far above the block centre when grasping (fingertips cover the upper ~2 cm)
FINGERTIP_BELOW_TCP = 0.0095  # Panda fingertip collision box extends this far below the TCP
FINGER_PREOPEN = 0.028  # finger joint position (m, 0.04 = fully open) used to approach and to release: leaves
#                        8 mm clearance on each side of the 4 cm block, so open fingers do not sweep neighbours
RELEASE_GAP = 0.003  # normal release: block bottom this far above its resting height
DROP_CLEARANCE = 0.004  # drop release: fingertips this far above the tops of same-layer neighbours
TRAVEL_CLEARANCE = 0.06  # carried block's bottom clears the target layer's top by this much while travelling
MIN_GAP = 0.0015  # placement keeps at least this gap to the measured faces of already-placed touching neighbours
REFINE_STEPS = 3  # extra steps holding the final waypoint of a precise move, so the PD controller converges
PLACE_TOL = 0.001  # corrective moves until the TCP / carried block is this close to its goal ...
N_CORRECT = 2  # ... or this many corrections were made
GRASP_HOVER = 0.03  # TCP height above the grasp pose where the gripper lines up before closing in on a block
HOVER = 0.005  # the carried block is lined up with its bottom this far above the target layer's top, then lowered
LINEAR_STEP = 0.006  # m per control step for straight-line moves into contact (12 cm/s at 20 Hz) ...
FAST_STEP = 0.015  # ... and for straight-line lifts / retreats / descents in free space (30 cm/s)
MAX_JOINT_STEP = 0.15  # rad; a larger jump between consecutive IK waypoints means the IK chain broke
SETTLE_STEPS = 10
# mplib's default Panda limits are 1 rad/s and 1 rad/s^2 (ManiSkill's solvers use 0.9x); the real arm allows
# ~2.2 rad/s. Faster motions keep the 10-block build within the env's 2000-step episode limit.
JOINT_VEL_SCALE = 2.0
JOINT_ACC_SCALE = 4.0  # idle steps at the end, before scoring


def _gripper_action(q: float) -> float:
    """Finger joint target (m) -> normalised pd_joint_pos gripper action (joint range [-0.01, 0.04])."""
    return 2 * (q + 0.01) / 0.05 - 1


def _yaw_quat(yaw: float) -> np.ndarray:
    return euler2quat(0, 0, yaw)


def _closing_dir(pose: sapien.Pose) -> np.ndarray:
    """World direction of the TCP's closing axis (local y)."""
    return pose.to_transformation_matrix()[:3, 1]


def _perp_axis(axis: int) -> int:
    return 1 - axis  # horizontal blocks only: x <-> y


def _finger_neighbours(block: Block, design: np.ndarray) -> set[int]:
    """Design block ids occupying the same-layer cells on either finger side of `block`'s middle cell."""
    mid = block.cells[1].copy()
    out = set()
    for s in (-1, 1):
        c = mid.copy()
        c[_perp_axis(block.axis)] += s
        if (0 <= c).all() and (c < GRID).all() and design[tuple(c)] not in (0, block.id):
            out.add(int(design[tuple(c)]))
    return out


def _clearance_shift(block: Block, target: sapien.Pose, design: np.ndarray, placed: dict[int, sapien.Pose]) -> np.ndarray:
    """World shift (m) of `block`'s target away from already-placed same-layer neighbours touching its sides or
    ends, using their *measured* poses: wherever a neighbour's actual face reaches closer than MIN_GAP to the
    target's face, the target moves away by the difference (so the descending block cannot catch it)."""
    shift = np.zeros(3)
    k = block.layer
    half = np.array([BLOCK_HALF_SIZE[0] if ax == block.axis else BLOCK_HALF_SIZE[1] for ax in (0, 1)])
    for ax in (0, 1):
        for s in (-1, 1):
            cells = [block.cells[0 if s < 0 else -1].copy()] if ax == block.axis else [c.copy() for c in block.cells]
            need = 0.0
            for c in cells:
                c[ax] += s
                if not ((0 <= c).all() and (c < GRID).all()) or design[c[0], c[1], k] not in placed:
                    continue
                nb = placed[int(design[c[0], c[1], k])]
                ext = np.abs(nb.to_transformation_matrix()[:3, :3]) @ BLOCK_HALF_SIZE  # neighbour's world AABB half size
                their_face = nb.p[ax] - s * ext[ax]
                our_face = target.p[ax] + s * half[ax]
                need = max(need, s * (our_face - their_face) + MIN_GAP)  # intrusion + gap
            shift[ax] -= s * need
    return shift


def plan_order(design: np.ndarray, blocks: list[Block]) -> list[Block]:
    """Bottom-up build order. Within a layer, block B goes before every block on B's finger sides (so B's fingers
    have room); cycles (side-by-side parallel blocks) are broken by taking the block with most such neighbours."""
    order = []
    for k in sorted({b.layer for b in blocks}):
        layer = [b for b in blocks if b.layer == k]
        horiz = [b for b in layer if b.axis != 2]
        before = {b.id: _finger_neighbours(b, design) & {o.id for o in horiz} for b in horiz}  # b before these
        remaining = list(horiz)
        while remaining:
            # blocks that nobody remaining needs to precede (no remaining block has them on its finger side)
            free = [b for b in remaining if not any(b.id in before[o.id] for o in remaining if o is not b)]
            pick = (free or remaining)
            pick = max(pick, key=lambda b: (len(before[b.id] & {o.id for o in remaining}), -remaining.index(b)))
            order.append(pick)
            remaining.remove(pick)
        order += [b for b in layer if b.axis == 2]
    return order


class _Builder:
    def __init__(self, env, planner: PandaArmMotionPlanningSolver, verbose: bool):
        self.env = env  # wrapped env (steps go through RecordEpisode)
        self.u = env.unwrapped
        self.planner = planner
        self.verbose = verbose
        self.rest_closing = _closing_dir(self.u.agent.tcp.pose.sp)
        self.placed_ids: set[int] = set()
        self.placed_actor: dict[int, object] = {}  # design id -> actor placed for it

    def log(self, *a):
        if self.verbose:
            print(*a)

    # ---- motion primitives ----
    def move(self, pose: sapien.Pose, dry_run: bool = False, refine: int = 0):
        """Straight-line (screw) motion. Returns -1 if unplannable (else the plan if dry_run, else 0)."""
        res = self.planner.move_to_pose_with_screw(pose, dry_run=True)
        if res == -1 or dry_run:
            return res
        if len(res["position"]):  # empty when already there; follow_path cannot handle that
            self.planner.follow_path(res, refine_steps=refine)
        return 0

    def travel(self, pose: sapien.Pose):
        res = self.move(pose)
        if res == -1:
            self.log("  screw travel failed, trying RRTConnect")
            res = self.planner.move_to_pose_with_RRTConnect(pose)
        return res

    def precise(self, pose: sapien.Pose, measure=None, want=None, step: float = LINEAR_STEP) -> int:
        """Linear move to `pose`, then up to N_CORRECT small corrective moves until the measured point (default the
        TCP position; e.g. the grasped block's centre) is within PLACE_TOL of `want` (default pose.p)."""
        measure = measure or (lambda: self.tcp().p)
        want = pose.p if want is None else want
        if self.linear(pose, step) == -1:
            return -1
        for _ in range(N_CORRECT):
            err = want - measure()
            if np.linalg.norm(err) < PLACE_TOL:
                break
            t = self.tcp()
            if self.linear(sapien.Pose(t.p + err, pose.q)) == -1:
                break
        return 0

    def _ik(self, pose: sapien.Pose, q_seed: np.ndarray):
        """CLIK inverse kinematics (full robot qpos) seeded at q_seed, or None if it did not converge."""
        pl = self.planner.planner
        goal = pl.transform_goal_to_wrt_base(np.concatenate([pose.p, pose.q]))
        q, ok, _ = pl.pinocchio_model.compute_IK_CLIK(pl.link_name_2_idx[pl.move_group], goal, q_seed, [])
        return q if ok else None

    def linear(self, pose: sapien.Pose, step: float = LINEAR_STEP, refine: int = REFINE_STEPS) -> int:
        """Straight-line Cartesian move: IK at waypoints `step` apart (each seeded with the previous solution),
        followed in joint space. Unlike the screw planner it ends where asked and does not drift sideways, so it is
        used for every move close to blocks. Falls back to a screw move if the IK chain breaks."""
        q0 = self.u.agent.robot.get_qpos()[0].cpu().numpy()
        start = self.tcp()
        n = max(4, int(np.ceil(np.linalg.norm(pose.p - start.p) / step)))
        q_start, q_goal = start.q, pose.q * (1 if float(np.dot(start.q, pose.q)) >= 0 else -1)
        n_arm = len(self.planner.planner.joint_vel_limits)
        path, q = [], q0
        for t in np.arange(1, n + 1) / n:
            quat = (1 - t) * q_start + t * q_goal  # nlerp: orientation changes are small here
            q_new = self._ik(sapien.Pose(start.p + t * (pose.p - start.p), quat / np.linalg.norm(quat)), q)
            if q_new is None or np.abs(q_new[:n_arm] - q[:n_arm]).max() > MAX_JOINT_STEP:
                self.log("  linear move IK failed, falling back to screw")
                return self.move(pose, refine=refine)
            path.append(q_new[:n_arm])
            q = q_new
        self.planner.follow_path({"position": np.array(path)}, refine_steps=refine)
        return 0

    def set_fingers(self, q: float, t: int = 8):
        return self.planner.open_gripper(t=t, gripper_state=_gripper_action(q))

    def tcp(self) -> sapien.Pose:
        return self.u.agent.tcp.pose.sp

    # ---- one block ----
    def grasp_candidates(self, actor) -> list[sapien.Pose]:
        """Top-down TCP grasp poses at the block centre, closing across the block's local y (two yaws)."""
        bp = actor.pose.sp
        R = bp.to_transformation_matrix()[:3, :3]
        closing = R[:, 1].copy()
        closing[2] = 0
        closing /= np.linalg.norm(closing)
        center = bp.p + np.array([0, 0, GRASP_HEIGHT])
        approaching = np.array([0.0, 0.0, -1.0])
        poses = [self.u.agent.build_grasp_pose(approaching, c, center) for c in (closing, -closing)]
        poses.sort(key=lambda p: -float(_closing_dir(p) @ self.rest_closing))  # least wrist twist first
        return poses

    def pick(self, actor, z_travel: float) -> str | None:
        """Grasp `actor` and lift it to z_travel. Returns an error string on failure."""
        grasp = None
        for g in self.grasp_candidates(actor):
            above = sapien.Pose([g.p[0], g.p[1], z_travel], g.q)
            if self.move(above, dry_run=True) != -1:
                grasp = g
                break
        if grasp is None:
            return "no plannable grasp"
        self.planner.gripper_state = _gripper_action(FINGER_PREOPEN)  # pre-open while travelling
        if self.travel(sapien.Pose([grasp.p[0], grasp.p[1], z_travel], grasp.q)) == -1:
            return "approach plan failed"
        # fast down to just above the block (fingertips ~1 cm above its top) and line up, then slowly onto it
        if self.precise(sapien.Pose(grasp.p + np.array([0, 0, GRASP_HOVER]), grasp.q), step=FAST_STEP) == -1 \
                or self.precise(grasp) == -1:
            self.move(sapien.Pose([grasp.p[0], grasp.p[1], z_travel], grasp.q))
            return "grasp descent plan failed"
        self.planner.close_gripper(t=8)
        if self.linear(sapien.Pose([grasp.p[0], grasp.p[1], z_travel], grasp.q), FAST_STEP) == -1:
            return "lift plan failed"
        if not bool(self.u.agent.is_grasping(actor)[0]):
            self.set_fingers(FINGER_PREOPEN)
            return "grasp slipped"
        return None

    def place_tcp(self, actor, target: sapien.Pose, z_offset: float) -> list[sapien.Pose]:
        """TCP poses that put the grasped block at `target` raised by z_offset, using the measured block-in-TCP
        transform. Two options (the block is symmetric under a 180 deg yaw), least wrist twist first."""
        tcp = self.tcp()
        tcp_T_block = tcp.inv() * actor.pose.sp
        out = []
        for flip in (0.0, np.pi):
            goal = sapien.Pose(target.p + np.array([0, 0, z_offset]), target.q) * sapien.Pose(q=_yaw_quat(flip))
            out.append(goal * tcp_T_block.inv())
        cur = _closing_dir(tcp)
        out.sort(key=lambda p: -float(_closing_dir(p) @ cur))
        return out

    def place(self, actor, block: Block, target: sapien.Pose, z_travel: float, design: np.ndarray) -> tuple[str | None, str]:
        drop = bool(_finger_neighbours(block, design) & self.placed_ids)
        if drop:
            # fingertips must end above the neighbours' tops: tip z = block centre + GRASP_HEIGHT - FINGERTIP_BELOW_TCP
            top = (block.layer + 1) * CELL_SIZE + GRID_ORIGIN[2]
            z_off = top + DROP_CLEARANCE - (target.p[2] + GRASP_HEIGHT - FINGERTIP_BELOW_TCP)
        else:
            z_off = RELEASE_GAP
        mode = "drop" if drop else "normal"
        placed_poses = {bid: a.pose.sp for bid, a in self.placed_actor.items()}
        target = sapien.Pose(target.p + _clearance_shift(block, target, design, placed_poses), target.q)

        # carry over the target at travel height (pick the place yaw that is plannable with least twist)
        above_tcp = None
        for g in self.place_tcp(actor, target, z_off):
            cand = sapien.Pose([g.p[0], g.p[1], z_travel], g.q)
            if self.move(cand, dry_run=True) != -1:
                above_tcp = cand
                break
        if above_tcp is None:
            return "no plannable place pose", mode
        if self.travel(above_tcp) == -1:
            return "carry plan failed", mode
        # re-measure the grasp, line the block up just above its goal, then descend (both with block-based correction)
        block_p = lambda: actor.pose.sp.p
        goal = min(self.place_tcp(actor, target, z_off), key=lambda p: -float(_closing_dir(p) @ _closing_dir(self.tcp())))
        # hover: block bottom HOVER above the tops of the target layer, so lining up cannot catch a neighbour
        hover_dz = max(0.0, (block.layer + 1) * CELL_SIZE + HOVER - (target.p[2] + z_off - BLOCK_HALF_SIZE[2]))
        hover = sapien.Pose(goal.p + np.array([0, 0, hover_dz]), goal.q)
        if self.precise(hover, block_p, target.p + np.array([0, 0, z_off + hover_dz]), FAST_STEP) == -1:
            return "place descent plan failed", mode
        if self.precise(goal, block_p, target.p + np.array([0, 0, z_off])) == -1:
            return "place descent plan failed", mode
        self.set_fingers(FINGER_PREOPEN, t=8)
        t = self.tcp()
        if self.linear(sapien.Pose([t.p[0], t.p[1], z_travel], t.q), FAST_STEP) == -1:
            return "retreat plan failed", mode
        return None, mode


def solve_jenga(env, seed: int | None = None, debug: bool = False, vis: bool = False, verbose: bool = False) -> dict:
    """Build the env's target design with a motion-planned scripted policy. Requires control_mode="pd_joint_pos".

    Resets `env` (with `seed`), then for each design block (bottom-up, finger-clearance-aware order within a
    layer, see `plan_order`) picks its earmarked actor (falling back to an unused spare if the pick fails) and
    places it at its target pose. Failures are recorded and skipped, never raised.

    Returns dict with
        blocks: per design block (in build order) {id, axis, layer, actor, status, mode, pos_err, cells_ok, ...}
            status "placed" or a failure reason; pos_err = final distance (m) of the actor from its target pose;
            cells_ok = whether the actor's final 3 cells equal the design block's cells.
        score, max_score, success: from env.evaluate() at the end.
        failed: ids of design blocks that did not end up on their cells.
        steps: env steps taken.
    """
    env.reset(seed=seed)
    u = env.unwrapped
    assert u.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"], u.control_mode
    planner = PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis, base_pose=u.agent.robot.pose,
        visualize_target_grasp_pose=vis, print_env_info=False,
        joint_vel_limits=JOINT_VEL_SCALE, joint_acc_limits=JOINT_ACC_SCALE,
    )
    b = _Builder(env, planner, verbose)
    design = u.design
    n_design = len(u.design_blocks)
    actor_of = {blk.id: n for n, blk in enumerate(u.design_blocks)}  # earmarked actor per design id
    spares = list(range(n_design, len(u.blocks)))

    results = []
    for block in plan_order(design, u.design_blocks):
        rec = dict(id=block.id, axis=block.axis, layer=block.layer, actor=None, status=None, mode=None)
        results.append(rec)
        if block.axis == 2:
            rec["status"] = "skipped: vertical blocks not supported in v1"
            b.log(f"block {block.id}: {rec['status']}")
            continue
        target = u.target_poses[actor_of[block.id]]
        z_travel = (block.layer + 1) * CELL_SIZE + GRASP_HEIGHT + BLOCK_HALF_SIZE[2] + TRAVEL_CLEARANCE
        for n in [actor_of[block.id]] + spares:
            actor = u.blocks[n]
            err = b.pick(actor, z_travel)
            if err is None:
                break
            b.log(f"block {block.id}: pick of actor {n} failed ({err})")
            rec.setdefault("pick_failures", []).append((n, err))
            if bool(u.agent.is_grasping(actor)[0]):  # still holding something: put it back down where we are
                b.set_fingers(FINGER_PREOPEN)
        else:
            rec["status"] = "pick failed"
            continue
        if n in spares:
            spares.remove(n)
        rec["actor"] = n
        err, rec["mode"] = b.place(actor, block, target, z_travel, design)
        if err is not None:
            rec["status"] = err
            b.log(f"block {block.id}: {err}")
            b.set_fingers(FINGER_PREOPEN)
            continue
        rec["status"] = "placed"
        b.placed_ids.add(block.id)
        b.placed_actor[block.id] = actor
        b.log(f"block {block.id}: placed actor {n} ({rec['mode']}), step {planner.elapsed_steps}")

    # settle, then measure
    for _ in range(SETTLE_STEPS):
        planner.open_gripper(t=1, gripper_state=_gripper_action(FINGER_PREOPEN))
    for rec in results:
        if rec["actor"] is None:
            rec["pos_err"], rec["cells_ok"] = None, False
            continue
        blk = u.design_blocks[actor_of[rec["id"]]]
        pose = u.blocks[rec["actor"]].pose.sp
        rec["pos_err"] = float(np.linalg.norm(pose.p - u.target_poses[actor_of[rec["id"]]].p))
        cells = pose_to_cells(pose)
        rec["cells_ok"] = sorted(map(tuple, cells.tolist())) == sorted(map(tuple, blk.cells.tolist()))
    info = u.evaluate()
    planner.close()
    return dict(
        blocks=results,
        score=int(info["score"][0]),
        max_score=int(info["max_score"][0]),
        success=bool(info["success"][0]),
        failed=[r["id"] for r in results if not r["cells_ok"]],
        steps=planner.elapsed_steps,
    )


def format_results(res: dict) -> str:
    lines = [f"score {res['score']}/{res['max_score']}  success={res['success']}  steps={res['steps']}"]
    for r in res["blocks"]:
        err = "  -  " if r["pos_err"] is None else f"{r['pos_err'] * 1000:5.1f}"
        lines.append(
            f"  block {r['id']:2d} (k={r['layer']}, axis={'xyz'[r['axis']]}) actor={r['actor']}  {r['status']:<10s}"
            f" mode={r['mode']}  err={err} mm  cells_ok={r['cells_ok']}"
        )
    if res["failed"]:
        lines.append(f"  failed blocks: {res['failed']}")
    return "\n".join(lines)
