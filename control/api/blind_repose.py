"""Blind in-hand reorientation: the Shadow-hand repose task (same dynamics, rewards, resets and success logic as
Isaac-Repose-Cube-Shadow-Direct-v0) with an actor that never sees the object, and a privileged critic.

    import api.blind_repose            # registers the three tasks below (after Isaac Sim has launched)

    Isaac-Repose-Cube-Shadow-Blind-Proprio-v0   actor: 133 object-free proprio dims + 4-d rotation command, over a 16-step window
    Isaac-Repose-Cube-Shadow-Blind-Touch-v0     same + 25 binary per-link touch bits (api.joint_torque.HandTouch)
    Isaac-Repose-Cube-Shadow-Privileged-v0      pipeline check: the stock 157-d privileged actor observation, same asymmetric critic wiring

The critic always gets the env's 187-d privileged state (`compute_full_state`: object pose / velocity, goal, fingertip force-torque, ...)
under the "critic" group, which Isaac Lab's rl_games wrapper maps to rl_games' central value network. The rotation command is the
(axis xyz, angle/pi) of the world-frame rotation that takes the object to the goal, derived from the observed relative rotation: the
task specification, not a measurement of the object's pose. Train with scripts/train_blind.py; the agent config is blind_ppo_cfg.yaml.
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_mul
from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv
from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg

WINDOW = 16
# layout of the 157-d "full" observation: 0:48 joint pos/vel | 48:61 object pose+velocity | 61:72 goal pos/rot + relative rotation |
# 72:137 fingertip pos/rot/vel (forward kinematics of the joints) | 137:157 last action
PROPRIO_IDX = list(range(0, 48)) + list(range(72, 157))  # 133 object-free dims
QDIFF = slice(68, 72)
N_LINKS = 25


def rotation_command(qdiff: torch.Tensor) -> torch.Tensor:
    """[N, 4] = (axis xyz, angle / pi) of the world-frame rotation taking the object to the goal. The observation holds
    q_diff = q_obj * conj(q_goal) (wxyz), the rotation goal -> object, so the needed rotation is its conjugate; shortest
    representation (angle in [0, pi]); the axis shrinks to 0 with the angle instead of dividing by a tiny norm."""
    q = qdiff * qdiff.new_tensor([1.0, -1.0, -1.0, -1.0])
    q = torch.where(q[:, :1] < 0, -q, q)
    xyz = q[:, 1:]
    s = xyz.norm(dim=-1, keepdim=True)
    return torch.cat([xyz / s.clamp(min=1e-6), 2.0 * torch.atan2(s, q[:, :1]) / math.pi], dim=-1)


def _quat_from_rotvec(v: torch.Tensor) -> torch.Tensor:
    """[..., 3] rotation vector -> [..., 4] wxyz quaternion."""
    ang = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return torch.cat([torch.cos(ang / 2), torch.sin(ang / 2) * v / ang], dim=-1)


class MeasurementModel:
    """Gearbox-style measurement errors on the joint encoders, applied to the ACTOR's observation only (physics and the critic stay exact).
    The schemes are those of Origami Robotics' "The Dexterity Deadlock" (backlash, stiction / dead zones, a time-varying "chaos" term),
    modeled on the measurement side and scaled with the gear ratio N (the post only says friction and backlash grow with N; the linear
    scaling and the reference magnitudes below are our assumptions):

      backlash   measured position = a "play" operator on the true joint position: it only follows once the true position leaves a band of
                 half-width h around it (hysteresis; the sign of the error flips when the motion reverses).  h = (N / 15) * 0.1 deg
      stiction   the information wall: the measured DEFLECTION of an actuated joint from its commanded target is dead-zoned by d_s, so
                 contact deflections below d_s are erased (a gear train stuck by static friction reads the commanded position). The joint-side
                 stiction torque grows with N: tau_s = (N / 15) * 0.0017 N m, and the dead zone is d_s = tau_s / K_j with K_j the joint's PD
                 stiffness (1 N m/rad fingers, 5 wrist), i.e. 0.1 deg at N = 15 for a finger
      chaos      the time-variant term: h and d_s are multiplied, per joint and per episode, by exp(N(0, 0.5)) (unit-to-unit variation) and by
                 a slow Ornstein-Uhlenbeck drift exp(x), std 0.3, time constant 300 control steps (temperature / wear)

    The measured joint velocity is the finite difference of the measured positions (as on hardware), and the fingertip position / rotation /
    velocity observations change to first order through the Jacobian (world frame, link COM): dp = J_v dq, rotation vector dtheta = J_w dq.
    `set(n=0)` switches everything off (the default; the clean policies were trained that way).
    """

    N_REF, H0_DEG, TAU0 = 15.0, 0.1, math.radians(0.1)  # TAU0 [N m]: hides 0.1 deg of deflection at K = 1 N m/rad
    CHAOS_SIGMA, DRIFT_SIGMA, DRIFT_TAU = 0.5, 0.3, 300.0

    def __init__(self, env):
        self.env = env
        self.act = torch.as_tensor(env.actuated_dof_indices, device=env.device)
        # PD stiffness per joint, read from the actuator groups (passive joints: 1.0, unused); guarded against the values the schemes were designed with
        self.K = torch.ones(env.num_hand_dofs, device=env.device)
        for actuator in env.hand.actuators.values():
            self.K[actuator.joint_indices] = actuator.stiffness[0].to(self.K)
        expected = torch.tensor([5.0 if "WRJ" in nm else 1.0 for nm in env.hand.joint_names], device=env.device)
        assert torch.allclose(self.K[self.act], expected[self.act]), f"actuator stiffness changed ({self.K[self.act].tolist()}): revisit the stiction dead-zone widths"
        self.rows = torch.as_tensor(env.finger_bodies, device=env.device) - 1  # fixed base: no Jacobian row for body 0
        self.set(n=0.0)

    def set(self, n: float = 0.0, backlash: bool = False, stiction: bool = False, chaos: bool = False):
        self.n, self.backlash, self.stiction, self.chaos = float(n), backlash, stiction, chaos
        self.active = self.n > 0 and (backlash or stiction)
        self.play = None  # all internal state restarts at the next observation
        self.fresh = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)

    def mark_reset(self, env_ids):
        self.fresh[env_ids] = True

    def apply(self):
        """(dof_pos, dof_vel, fingertip_pos, fingertip_rot, fingertip_vel) as measured, for the current true state of the env."""
        env = self.env
        q, qd = env.hand_dof_pos, env.hand_dof_vel
        nd = q.shape[1]
        if self.play is None:
            self.play, self.prev = q.clone(), q.clone()
            self.m0, self.x = torch.ones_like(q), torch.zeros_like(q)
        fresh = self.fresh
        if fresh.any():
            k = int(fresh.sum())
            self.play[fresh], self.prev[fresh] = q[fresh], q[fresh]
            if self.chaos:
                self.m0[fresh] = torch.exp(self.CHAOS_SIGMA * torch.randn(k, nd, device=q.device))
                self.x[fresh] = self.DRIFT_SIGMA * torch.randn(k, nd, device=q.device)
            else:
                self.m0[fresh], self.x[fresh] = 1.0, 0.0
        if self.chaos:
            rho = math.exp(-1.0 / self.DRIFT_TAU)
            keep = ~fresh
            self.x[keep] = rho * self.x[keep] + math.sqrt(1 - rho**2) * self.DRIFT_SIGMA * torch.randn(int(keep.sum()), nd, device=q.device)
            mult = self.m0 * torch.exp(self.x)
        else:
            mult = torch.ones_like(q)
        s_n = self.n / self.N_REF
        q_m = q
        if self.backlash:
            h = s_n * math.radians(self.H0_DEG) * mult
            self.play = torch.maximum(torch.minimum(self.play, q + h), q - h)
            q_m = self.play
        if self.stiction:
            ds = (s_n * self.TAU0 / self.K)[None] * mult
            target = env.cur_targets[:, self.act]
            d = q_m[:, self.act] - target
            q_m = q_m.clone()
            q_m[:, self.act] = target + torch.sign(d) * torch.clamp(d.abs() - ds[:, self.act], min=0.0)
        qd_m = torch.where(fresh[:, None], torch.zeros_like(qd), (q_m - self.prev) / env.step_dt)
        self.prev = q_m.clone()
        self.fresh[:] = False
        dq, dqd = q_m - q, qd_m - qd
        J = env.hand.root_physx_view.get_jacobians()[:, self.rows]  # [N, 5, 6, nd]
        fp = env.fingertip_pos + torch.einsum("nfjk,nk->nfj", J[:, :, :3], dq)
        fr = quat_mul(_quat_from_rotvec(torch.einsum("nfjk,nk->nfj", J[:, :, 3:], dq)), env.fingertip_rot)
        fv = env.fingertip_velocities + torch.cat(
            [torch.einsum("nfjk,nk->nfj", J[:, :, :3], dqd), torch.einsum("nfjk,nk->nfj", J[:, :, 3:], dqd)], dim=-1
        )
        return q_m, qd_m, fp, fr, fv


@configclass
class BlindShadowHandEnvCfg(ShadowHandEnvCfg):
    asymmetric_obs = True
    state_space = 187
    observation_space = WINDOW * (len(PROPRIO_IDX) + 4)
    variant: str = "proprio"  # "proprio" | "touch" | "privileged"
    window: int = WINDOW


@configclass
class BlindTouchEnvCfg(BlindShadowHandEnvCfg):
    observation_space = WINDOW * (len(PROPRIO_IDX) + 4 + N_LINKS)
    variant = "touch"  # needs cfg.robot_cfg.spawn.activate_contact_sensors = True (api.joint_torque.enable_contact_reporting)


@configclass
class PrivilegedActorEnvCfg(BlindShadowHandEnvCfg):
    observation_space = 157
    variant = "privileged"
    window = 1


class BlindShadowHandEnv(InHandManipulationEnv):
    """The stock in-hand env with the actor observation replaced (see the module docstring); everything else is inherited."""

    cfg: BlindShadowHandEnvCfg
    _hist = None  # [N, window, feat] sliding window of the actor's per-step features
    _fresh = None  # [N] bool: env was reset since the last observation, so its window restarts from the new first step
    _meas = None  # MeasurementModel (off by default); switch with set_measurement()
    _stic = None  # api.stiction_dynamics.StictionDynamics (created on first set_stiction_dynamics; off by default)

    def __init__(self, cfg: BlindShadowHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        feat = len(PROPRIO_IDX) + 4 + (N_LINKS if cfg.variant == "touch" else 0)
        expected_obs = 157 if cfg.variant == "privileged" else cfg.window * feat
        assert cfg.observation_space == expected_obs, f"observation_space {cfg.observation_space} != {expected_obs} for variant {cfg.variant!r} (window {cfg.window})"
        self._proprio_idx = torch.as_tensor(PROPRIO_IDX, device=self.device)
        self._meas = MeasurementModel(self)
        self._touch = None
        if cfg.variant == "touch":
            from api.joint_torque import HandTouch

            self._touch = HandTouch(self)

    def set_measurement(self, n: float = 0.0, backlash: bool = False, stiction: bool = False, chaos: bool = False):
        """Gearbox measurement errors on the actor's joint / fingertip observations (see MeasurementModel); n = gear ratio, 0 = off."""
        self._meas.set(n=n, backlash=backlash, stiction=stiction, chaos=chaos)

    def set_stiction_dynamics(self, n: float = 0.0, **kwargs):
        """Gearbox stiction as a change of the DYNAMICS (api.stiction_dynamics; off by default, independent of set_measurement)."""
        from api.stiction_dynamics import StictionDynamics

        if self._stic is None:
            self._stic = StictionDynamics(self)
        self._stic.set(n=n, **kwargs)

    def _apply_action(self) -> None:
        super()._apply_action()
        if self._stic is not None and self._stic.active:
            self._stic.apply()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if self._fresh is not None:
            self._fresh[env_ids] = True
        if self._meas is not None:
            self._meas.mark_reset(env_ids)

    def _push(self, feat: torch.Tensor) -> torch.Tensor:
        K = self.cfg.window
        if self._hist is None:
            self._hist = feat[:, None, :].repeat(1, K, 1)
            self._fresh = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            self._hist = torch.cat([self._hist[:, 1:], feat[:, None, :]], dim=1)
            if self._fresh.any():
                self._hist[self._fresh] = feat[self._fresh][:, None, :].repeat(1, K, 1)
            self._fresh[:] = False
        return self._hist.flatten(1)

    def _get_observations(self) -> dict:
        self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[:, self.finger_bodies]
        states = self.compute_full_state()  # [N, 187] critic (exact)
        if self._meas is not None and self._meas.active:
            true = (self.hand_dof_pos, self.hand_dof_vel, self.fingertip_pos, self.fingertip_rot, self.fingertip_velocities)
            (self.hand_dof_pos, self.hand_dof_vel, self.fingertip_pos, self.fingertip_rot, self.fingertip_velocities) = self._meas.apply()
            full = self.compute_full_observations()  # the stock privileged actor observation computed from the MEASURED hand state
            (self.hand_dof_pos, self.hand_dof_vel, self.fingertip_pos, self.fingertip_rot, self.fingertip_velocities) = true
        else:
            full = self.compute_full_observations()  # the stock privileged actor observation, [N, 157]
        if self.cfg.variant == "privileged":
            return {"policy": full, "critic": states}
        parts = [full[:, self._proprio_idx], rotation_command(full[:, QDIFF])]
        if self._touch is not None:
            parts.append(self._touch.contacts)
        return {"policy": self._push(torch.cat(parts, dim=-1)), "critic": states}


_CFG_YAML = str(Path(__file__).with_name("blind_ppo_cfg.yaml"))
for _id, _cfg in [
    ("Isaac-Repose-Cube-Shadow-Blind-Proprio-v0", BlindShadowHandEnvCfg),
    ("Isaac-Repose-Cube-Shadow-Blind-Touch-v0", BlindTouchEnvCfg),
    ("Isaac-Repose-Cube-Shadow-Privileged-v0", PrivilegedActorEnvCfg),
]:
    if _id not in gym.registry:
        gym.register(
            id=_id,
            entry_point=f"{__name__}:BlindShadowHandEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": f"{__name__}:{_cfg.__name__}", "rl_games_cfg_entry_point": _CFG_YAML},
        )
