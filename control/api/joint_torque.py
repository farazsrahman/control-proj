"""Ground-truth contact torque on a robot hand's joints, from the object's contact forces (Isaac Lab / PhysX).

    from api import isaac
    from api.joint_torque import HandContactTorque, enable_contact_reporting

    env = isaac.make_env("Isaac-Repose-Cube-Shadow-Direct-v0", num_envs=2048, cfg_fn=enable_contact_reporting)
    sensor = HandContactTorque(env)
    obs, *_ = env.step(action)
    sensor.tau_ext        # [num_envs, 20] N m on the actuated joints (order: sensor.joint_names), averaged over the env step
    sensor.link_force     # [num_envs, 25, 3] N, force ON each hand link BY the object (world frame), averaged over the env step

tau_ext = sum over links of J_p^T f, with f the object's contact force on the link (normal + friction) and J_p the link's linear
Jacobian shifted to the average contact point. It is the torque the object exerts on the joints, i.e. the "external" part of the
joint dynamics  M qdd + c + g = tau_act + tau_ext  (no dynamics model or finite-difference acceleration needed).

Why this is not Isaac Lab's ContactSensor: the in-hand tasks clone envs with `copy_from_source=False`, so only env_0 has USD prims and
ContactSensor sees num_envs = 1. And PhysX pairs each sensor body with ONE filter prim, so there is one raw PhysX contact view per link.

Things that matter (all found by probing on the Shadow repose env, see ai-notes/Work Log.md):
  * the force matrix is the NORMAL force only; friction comes from get_friction_data and is added. Without it the object's
    z-impulse was 21% short.
  * `view.get_jacobians()` is in the world frame and referenced at the link COM (not the link frame origin): the contact-point shift
    uses r = p - (origin + R com_local).
  * forces are read after every physics substep and averaged over the env step (decimation substeps); the first env step after a
    reset is zeroed for the envs that reset (the object was teleported, `tau_ext` belongs to the previous episode's last step).

Also in this module: HandTouch (cheap binary per-link touch bits, normal force > 0.01 N) and HandContactGeometry (HandTouch plus per-link contact
force / point / normal in the hand frame). The scripts that produced the validation numbers below are in scripts/diagnostics/.

Validation (Shadow hand, 256 envs, teacher policy in hand, 46k samples):
  * object Newton check: mean z force by the hand 2.121 N = m(a - g) 2.121 N (corr 1.000); near-static samples 2.119 N vs weight 2.119 N.
  * windowed impulse check, K = 5/15/30 env steps: sum(F) dt = m dv - m g T with slope 1.000 and relative error 0.000.
  * the per-link sum equals the object's own net contact force exactly (normal part), so no contacts are missed.
  * power identity sum_j tau_j qdot_j = sum_b f_b . v_p,b: 1.8% median relative error.
  * kinematic Jacobian (finite difference of every joint): 2-5% median per finger link, 8.6% palm. NOT validated: the passive/mimic
    `J0` joints and the wrist link (the finite-difference test could not move them independently).
  * a held-out MLP predicts tau_ext from the existing 157-d observation with R^2 0.83 (0.82 from joint pos/vel + last action alone),
    so tau_ext is mostly redundant with those observations.
"""

from __future__ import annotations

import torch


def _just_reset(u) -> torch.Tensor:
    """[N] bool: the env was reset on the last step (after env.step) or by a manual env.reset(): its episode counter is 0. More robust than
    `reset_buf`, which a manual reset does not update."""
    return u.episode_length_buf == 0


def enable_contact_reporting(cfg) -> None:
    """`cfg_fn` for isaac.make_env: puts PhysX's contact-report API on the hand's links (needed for the contact views)."""
    cfg.robot_cfg.spawn.activate_contact_sensors = True


def _skew(r: torch.Tensor) -> torch.Tensor:
    """[..., 3] -> [..., 3, 3] cross-product matrices."""
    z = torch.zeros_like(r[..., 0])
    return torch.stack([z, -r[..., 2], r[..., 1], r[..., 2], z, -r[..., 0], -r[..., 1], r[..., 0], z], -1).reshape(*r.shape[:-1], 3, 3)


class HandContactTorque:
    """Per-env-step contact torque on the actuated joints of a fixed-base hand (built for the Shadow in-hand repose envs).

    COST: it reads every contact view and the Jacobian after EVERY physics substep, whether or not tau_ext is queried, which made env stepping
    about 5.6x slower at 2048 envs (32 s vs 5.7 s per 100 steps, measured). Call close() or use HandTouch when the signal is not needed.

    Create it after `isaac.make_env(..., cfg_fn=enable_contact_reporting)`. It hooks the env's `sim.step` (to read the contact forces
    after every physics substep) and `_pre_physics_step` (to restart the average each env step); `close()` removes the hooks.
    Expects the env to expose `hand` (Articulation), `actuated_dof_indices`, and the object at `cfg.object_cfg.prim_path`.
    """

    def __init__(self, env):
        from isaacsim.core.simulation_manager import SimulationManager

        u = env.unwrapped
        self._u = u
        self._view = u.hand.root_physx_view
        self.device = u.device
        self.num_envs = u.num_envs
        names = list(u.hand.body_names)
        self.link_names = names[1:]  # body 0 is the fixed base: it has no Jacobian row and the object never touches it
        self._body_idx = torch.arange(1, len(names), device=self.device)
        self._act = torch.as_tensor(list(u.actuated_dof_indices), device=self.device)
        self.joint_names = [u.hand.joint_names[i] for i in u.actuated_dof_indices]
        self._dt = u.cfg.sim.dt
        self._com_local = self._view.get_coms().clone().to(self.device)[:, self._body_idx, :3]  # [N, B, 3] link COM in the link frame

        robot_path = u.cfg.robot_cfg.prim_path.replace(".*", "*")
        object_path = u.cfg.object_cfg.prim_path.replace(".*", "*")
        sim_view = SimulationManager.get_physics_sim_view()
        self._views = [
            sim_view.create_rigid_contact_view(
                f"{robot_path}/{name}", filter_patterns=[object_path], max_contact_data_count=16 * self.num_envs
            )
            for name in self.link_names
        ]
        for v in self._views:
            if v.sensor_count != self.num_envs:
                raise RuntimeError(f"contact view found {v.sensor_count} bodies, expected {self.num_envs}: did you pass cfg_fn=enable_contact_reporting?")

        self._reset_acc()
        self._orig_sim_step = u.sim.step
        self._orig_pre_physics = u._pre_physics_step
        u.sim.step = self._sim_step
        u._pre_physics_step = self._pre_physics_step

    # -- hooks ---------------------------------------------------------------------------------------------------------

    def _pre_physics_step(self, actions):
        self._reset_acc()
        return self._orig_pre_physics(actions)

    def _sim_step(self, *args, **kwargs):
        out = self._orig_sim_step(*args, **kwargs)
        with torch.no_grad():
            tau, force = self._read_substep()
        self._tau_sum = tau if self._tau_sum is None else self._tau_sum + tau
        self._force_sum = force if self._force_sum is None else self._force_sum + force
        self._n += 1
        return out

    def _reset_acc(self):
        self._tau_sum, self._force_sum, self._n = None, None, 0

    def close(self):
        """Remove the hooks from the env."""
        self._u.sim.step = self._orig_sim_step
        self._u._pre_physics_step = self._orig_pre_physics

    # -- physics reads -------------------------------------------------------------------------------------------------

    def _read_contacts(self):
        """Force ON each link BY the object [N, B, 3] (normal + friction) and the average contact point [N, B, 3] (nan: no contact)."""
        N = self.num_envs
        forces, points = [], []
        for v in self._views:
            normal = v.get_contact_force_matrix(dt=self._dt).reshape(N, 3)
            fr, _, fr_cnt, fr_start = v.get_friction_data(dt=self._dt)
            fr_cnt, fr_start = fr_cnt.reshape(N).long(), fr_start.reshape(N).long()
            cs = torch.cat([torch.zeros(1, 3, device=fr.device), fr.reshape(-1, 3).cumsum(0)])
            forces.append(normal + (cs[fr_start + fr_cnt] - cs[fr_start]))
            _, pts, _, _, cnt, start = v.get_contact_data(dt=self._dt)
            cnt, start = cnt.reshape(N).long(), start.reshape(N).long()
            cs = torch.cat([torch.zeros(1, 3, device=pts.device), pts.reshape(-1, 3).cumsum(0)])
            mean = (cs[start + cnt] - cs[start]) / cnt.clamp(min=1)[:, None]
            points.append(torch.where((cnt > 0)[:, None], mean, torch.full_like(mean, float("nan"))))
        return torch.stack(forces, 1), torch.stack(points, 1)

    def _read_substep(self):
        from isaaclab.utils.math import convert_quat, quat_apply

        f, p = self._read_contacts()
        has = torch.isfinite(p).all(-1) & (f.norm(dim=-1) > 0)
        f = torch.where(has[..., None], f, torch.zeros_like(f))
        p = torch.where(has[..., None], p, torch.zeros_like(p))
        tr = self._view.get_link_transforms()
        x_com = tr[:, self._body_idx, :3] + quat_apply(convert_quat(tr[:, self._body_idx, 3:], to="wxyz"), self._com_local)
        J = self._view.get_jacobians()[:, self._body_idx - 1]  # [N, B, 6, nj]: world frame, referenced at the link COM
        Jp = J[:, :, :3] - _skew(p - x_com) @ J[:, :, 3:]  # linear Jacobian at the contact point
        tau = torch.einsum("nbjk,nbj->nk", Jp, f)  # [N, nj]
        return tau, f

    # -- results (valid after an env.step) -----------------------------------------------------------------------------

    @property
    def tau_ext(self) -> torch.Tensor:
        """[num_envs, 20] contact torque on the actuated joints (order: `joint_names`), averaged over the last env step's substeps."""
        if self._n == 0:
            return torch.zeros(self.num_envs, len(self._act), device=self.device)
        tau = (self._tau_sum / self._n)[:, self._act]
        return tau.masked_fill(_just_reset(self._u)[:, None], 0.0)

    @property
    def link_force(self) -> torch.Tensor:
        """[num_envs, 25, 3] force on each link (order: `link_names`) by the object, world frame, averaged over the last env step."""
        if self._n == 0:
            return torch.zeros(self.num_envs, len(self._body_idx), 3, device=self.device)
        return (self._force_sum / self._n).masked_fill(_just_reset(self._u)[:, None, None], 0.0)

    @property
    def in_contact(self) -> torch.Tensor:
        """[num_envs, 25] bool: the object pushed on this link during the last env step."""
        return self.link_force.norm(dim=-1) > 0


class HandTouch:
    """Binary per-link touch signals, as in Yin et al. 2023 ("Rotating without Seeing"): link k reads 1 when the object's contact force
    on it exceeds `threshold` N (they use 0.01 N in sim), else 0. Reads the normal contact force of the LAST physics substep of the
    env step (one PhysX call per link, no Jacobian, no hooks), so it is much cheaper than HandContactTorque.

        env = isaac.make_env(TASK, num_envs=..., cfg_fn=enable_contact_reporting)
        touch = HandTouch(env)
        obs, *_ = env.step(action)
        touch.contacts        # [num_envs, 25] float in {0, 1} (order: touch.link_names); read it once per env step, after the step

    Rows of envs that just reset are zeroed (the object was teleported). All 25 links are sensed; restrict with a column mask for a
    sparser sensor layout. No dropout or delay is modeled (Yin et al. add both for sim-to-real).
    """

    def __init__(self, env, threshold: float = 0.01):
        from isaacsim.core.simulation_manager import SimulationManager

        u = env.unwrapped
        self._u = u
        self.threshold = threshold
        self.num_envs = u.num_envs
        self.link_names = list(u.hand.body_names)[1:]  # the fixed-base root never touches the object
        self._dt = u.cfg.sim.dt
        robot_path = u.cfg.robot_cfg.prim_path.replace(".*", "*")
        object_path = u.cfg.object_cfg.prim_path.replace(".*", "*")
        sim_view = SimulationManager.get_physics_sim_view()
        self._views = [
            sim_view.create_rigid_contact_view(f"{robot_path}/{name}", filter_patterns=[object_path], max_contact_data_count=16 * self.num_envs)
            for name in self.link_names
        ]
        for v in self._views:
            if v.sensor_count != self.num_envs:
                raise RuntimeError(f"contact view found {v.sensor_count} bodies, expected {self.num_envs}: did you pass cfg_fn=enable_contact_reporting?")

    @property
    def contacts(self) -> torch.Tensor:
        with torch.no_grad():
            f = torch.stack([v.get_contact_force_matrix(dt=self._dt).reshape(self.num_envs, 3) for v in self._views], 1)
        return (f.norm(dim=-1) > self.threshold).float().masked_fill(_just_reset(self._u)[:, None], 0.0)


class HandContactGeometry(HandTouch):
    """HandTouch plus the contact GEOMETRY per link, all expressed in the hand-root frame (the hand is fixed-base, so this frame is
    constant and does not depend on the env's origin):

        geom = HandContactGeometry(env)         # after isaac.make_env(..., cfg_fn=enable_contact_reporting)
        obs, *_ = env.step(action)
        g = geom.read()                         # dict of [num_envs, 25, ...] tensors, read once per env step, after the step
        g["touch"]   [N, 25]    1 if the object's force on the link (normal + friction) > threshold. NOTE: HandTouch.contacts thresholds
                                the NORMAL force only, so the two definitions differ slightly (friction-only contacts)
        g["force"]   [N, 25, 3] force ON the link BY the object (normal + friction), N
        g["point"]   [N, 25, 3] average contact point, m
        g["normal"]  [N, 25, 3] average contact normal (PhysX convention, same for every link; not unit length if contacts disagree)

    Everything is zero for links with no contact and for envs that just reset. Reads the last physics substep of the env step.
    """

    def read(self) -> dict[str, torch.Tensor]:
        from isaaclab.utils.math import quat_apply_inverse

        N, B = self.num_envs, len(self._views)
        forces, points, normals = [], [], []
        with torch.no_grad():
            for v in self._views:
                f = v.get_contact_force_matrix(dt=self._dt).reshape(N, 3)
                fr, _, fr_cnt, fr_start = v.get_friction_data(dt=self._dt)
                fr_cnt, fr_start = fr_cnt.reshape(N).long(), fr_start.reshape(N).long()
                cs = torch.cat([torch.zeros(1, 3, device=fr.device), fr.reshape(-1, 3).cumsum(0)])
                forces.append(f + (cs[fr_start + fr_cnt] - cs[fr_start]))
                _, pts, nrm, _, cnt, start = v.get_contact_data(dt=self._dt)
                cnt, start = cnt.reshape(N).long(), start.reshape(N).long()
                den = cnt.clamp(min=1)[:, None]
                csp = torch.cat([torch.zeros(1, 3, device=pts.device), pts.reshape(-1, 3).cumsum(0)])
                csn = torch.cat([torch.zeros(1, 3, device=nrm.device), nrm.reshape(-1, 3).cumsum(0)])
                points.append((csp[start + cnt] - csp[start]) / den)
                normals.append((csn[start + cnt] - csn[start]) / den)
        force, point_w, normal_w = torch.stack(forces, 1), torch.stack(points, 1), torch.stack(normals, 1)
        keep = (force.norm(dim=-1) > self.threshold) & ~_just_reset(self._u)[:, None]
        hand = self._u.hand.data
        q = hand.root_quat_w[:, None, :].expand(-1, B, -1)
        mask = keep[..., None]
        return dict(
            touch=keep.float(),
            force=quat_apply_inverse(q, force) * mask,
            point=quat_apply_inverse(q, point_w - hand.root_pos_w[:, None, :]) * mask,
            normal=quat_apply_inverse(q, normal_w) * mask,
        )
