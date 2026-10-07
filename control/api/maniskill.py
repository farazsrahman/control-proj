"""The one place notebooks and scripts touch ManiSkill startup.

    from api import maniskill
    env = maniskill.make_env("PickCube-v1", num_envs=16, render=True)
    demo = maniskill.load_demo("StackCube-v1")
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from utils.path import ROOT_DIR

# demos for our own custom tasks (e.g. StackThree-v1) are committed here, since they have no official
# download source and we don't want a fresh machine to need to regenerate (and re-verify) them.
REPO_DEMO_DIR = ROOT_DIR / "demos"


def make_env(task: str, num_envs: int = 1, render: bool = False, **kwargs) -> gym.Env:
    """Create a registered ManiSkill task.

    Unlike Isaac Lab, ManiSkill has no persistent app/process to launch first and supports multiple
    envs per process; this just wraps `gym.make`.

    Args:
        task: gym id, e.g. "PickCube-v1".
        num_envs: parallel envs. `sim_backend` auto-selects GPU sim for num_envs > 1, CPU sim for 1.
        render: enable env.render() -> RGB frames.
        kwargs: forwarded to gym.make (e.g. obs_mode, control_mode, robot_uids).
    """
    import mani_skill.envs  # noqa: F401  (registers tasks)

    import api.maniskill_tasks  # noqa: F401  (registers our custom tasks, e.g. StackThree-v1)

    return gym.make(task, num_envs=num_envs, render_mode="rgb_array" if render else None, **kwargs)


@dataclass
class Demo:
    actions: np.ndarray  # [T, action_dim], the raw action logged at each step
    control_mode: str  # pass to make_env(..., control_mode=...) to match the actions' semantics
    reset_kwargs: dict  # pass to env.reset(**reset_kwargs) to reproduce the initial state actions were recorded for
    success: bool  # whether this episode succeeded when it was recorded


def load_demo(task: str, episode_id: int = 0, source: str = "motionplanning") -> Demo:
    """Load one episode of a ManiSkill demonstration for `task`: our own committed demos (e.g. for custom
    tasks) under `<repo>/demos/`, if present, else ManiSkill's official dataset, downloaded to
    `~/.maniskill/demos` on first use (`mani_skill.utils.download_demo`).

    Demos contain no observations, only actions + reset_kwargs, so replaying them is necessarily open-loop.

    Args:
        task: gym id, e.g. "StackCube-v1".
        episode_id: which recorded episode to load.
        source: "motionplanning" (scripted solver; most tasks) or "rl" (trained PPO; fewer tasks have it).
    """
    import h5py

    traj_dir = REPO_DEMO_DIR / task / source
    if not (traj_dir / "trajectory.h5").exists():
        from mani_skill import DEMO_DIR
        from mani_skill.utils.download_demo import main as download_demo, parse_args as download_demo_args

        traj_dir = DEMO_DIR / task / source
        if not (traj_dir / "trajectory.h5").exists():
            download_demo(download_demo_args([task]))

    meta = json.load(open(traj_dir / "trajectory.json"))
    episode = meta["episodes"][episode_id]
    with h5py.File(traj_dir / "trajectory.h5", "r") as f:
        actions = f[f"traj_{episode['episode_id']}"]["actions"][:]
    return Demo(actions, episode["control_mode"], episode["reset_kwargs"], episode["success"])
