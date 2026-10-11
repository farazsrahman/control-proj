"""Per-env domain randomization for Isaac-Cartpole-Direct-v0: actuator gain, pole mass and action delay.

    rand = CartpoleRandomizer(env)        # after isaac.make_env(...)
    rand.resample()                       # fresh draw for every env (or only `env_ids`)
    rand.set(1.0, 1.0, 0)                 # the nominal system
    env.step(rand.apply_action(action))   # gain and delay act on the action, so call this on every step

Gain and pole mass are scale factors on the nominal values (the env's 100 N action scale, the masses authored in
the USD), drawn log-uniformly. The delay is an integer number of control steps (1 step = 1/60 s), drawn uniformly:
the force applied at step t is the one chosen at step t - delay. The true values stay available in `rand.gain`,
`rand.pole_mass` and `rand.delay` for logging and probing. Pole masses are written through the PhysX view and the
pole's inertia is scaled by the same ratio (as Isaac Lab's `randomize_rigid_body_mass` does).

Parameters are per env and only change when `set`/`resample` is called, so resample the envs that just finished an
episode to randomize per episode. `set`/`resample` also clear those envs' action history; after a full
`env.reset()` call `reset_buffer()` so the new episode doesn't start by replaying old actions.
"""

from __future__ import annotations

import math

import torch


class CartpoleRandomizer:
    def __init__(self, env, gain_range=(0.5, 2.0), pole_mass_range=(0.2, 5.0), delay_range=(0, 3)):
        base = env.unwrapped
        self.device = base.device
        self.num_envs = base.num_envs
        self.gain_range, self.pole_mass_range, self.delay_range = gain_range, pole_mass_range, delay_range

        self._view = base.cartpole.root_physx_view
        body_ids, names = base.cartpole.find_bodies("pole")
        assert len(body_ids) == 1, f"expected one pole body, found {names}"
        self._pole = body_ids[0]
        # PhysX view tensors live on the CPU. Nominal values come from the asset defaults, not the live view, so
        # rebuilding a randomizer after masses were already randomized still scales the true nominal system.
        data = base.cartpole.data
        self._nom_mass = getattr(data, "default_mass", self._view.get_masses()).clone()
        self._nom_inertia = getattr(data, "default_inertia", self._view.get_inertias()).clone()
        self._mass = self._nom_mass.clone()
        self._inertia = self._nom_inertia.clone()

        self.gain = torch.ones(self.num_envs, device=self.device)
        self.pole_mass = torch.ones(self.num_envs, device=self.device)
        self.delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._hist = None  # [N, max_delay + 1, action_dim]; slot 0 is the newest action (built on first use)

    @staticmethod
    def _log_uniform(rng, n, device):
        lo, hi = math.log(rng[0]), math.log(rng[1])
        return torch.exp(torch.empty(n, device=device).uniform_(lo, hi))

    def sample(self, n):
        """n independent (gain, pole_mass, delay) draws from the distribution."""
        lo, hi = self.delay_range
        return (
            self._log_uniform(self.gain_range, n, self.device),
            self._log_uniform(self.pole_mass_range, n, self.device),
            torch.randint(lo, hi + 1, (n,), device=self.device),
        )

    def set(self, gain, pole_mass, delay, env_ids=None):
        """Apply given values (scalars or tensors matching `env_ids`) to `env_ids` (default: all envs)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        gain = torch.as_tensor(gain, dtype=torch.float, device=self.device).expand(n)
        pole_mass = torch.as_tensor(pole_mass, dtype=torch.float, device=self.device).expand(n)
        delay = torch.as_tensor(delay, dtype=torch.long, device=self.device).expand(n)
        assert int(delay.max()) <= self.delay_range[1], "delay above delay_range[1]"
        self.gain[env_ids] = gain
        self.pole_mass[env_ids] = pole_mass
        self.delay[env_ids] = delay

        ids = env_ids.cpu()
        ratio = pole_mass.cpu()
        self._mass[ids, self._pole] = self._nom_mass[ids, self._pole] * ratio
        self._inertia[ids, self._pole] = self._nom_inertia[ids, self._pole] * ratio[:, None]
        self._view.set_masses(self._mass, ids)
        self._view.set_inertias(self._inertia, ids)
        self.reset_buffer(env_ids)

    def resample(self, env_ids=None):
        n = self.num_envs if env_ids is None else len(env_ids)
        self.set(*self.sample(n), env_ids=env_ids)

    def reset_buffer(self, env_ids=None):
        """Forget the action history (applied delayed actions restart from zero force)."""
        if self._hist is not None:
            if env_ids is None:
                self._hist.zero_()
            else:
                self._hist[env_ids] = 0.0

    def apply_action(self, action):
        """[N, A] env action -> the action actually applied: delayed by each env's `delay`, scaled by its gain."""
        if self._hist is None:
            self._hist = torch.zeros(self.num_envs, self.delay_range[1] + 1, action.shape[1], device=self.device)
        self._hist = torch.roll(self._hist, 1, dims=1)
        self._hist[:, 0] = action
        delayed = self._hist[torch.arange(self.num_envs, device=self.device), self.delay]
        return delayed * self.gain[:, None]

    def applied_pole_mass_ratio(self):
        """Read the pole mass back from the physics engine as a ratio to nominal, [N] (to verify `set` took effect)."""
        m = self._view.get_masses()[:, self._pole]
        return (m / self._nom_mass[:, self._pole]).to(self.device)
