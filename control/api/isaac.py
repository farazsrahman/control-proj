"""The one place notebooks and scripts touch Isaac Sim / Isaac Lab startup.

    from api import isaac
    app = isaac.launch(cameras=True)
    env = isaac.make_env("Isaac-Cartpole-Direct-v0", num_envs=4096, render=True)
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Callable

import gymnasium as gym

from utils.path import ROOT_DIR

# libGLU.so.1 (libglu1-mesa, not installed on the dev node), extracted locally. Isaac Sim's MDL/iray libs need it.
LIBGLU = ROOT_DIR / "deps/sysroot/usr/lib/x86_64-linux-gnu/libGLU.so.1"

_app = None
_cameras = False
_env = None


def launch(headless: bool = True, cameras: bool = False, **kwargs):
    """Start Isaac Sim (once per process; later calls return the running app).

    Must run before importing anything else from isaaclab. `kwargs` go to isaaclab's AppLauncher.
    """
    global _app, _cameras
    if _app is not None:
        return _app
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    # loading it globally first lets the dynamic loader resolve later dlopen("libGLU.so.1") calls by soname,
    # so no LD_LIBRARY_PATH is needed
    if LIBGLU.exists():
        ctypes.CDLL(str(LIBGLU), mode=ctypes.RTLD_GLOBAL)

    from isaaclab.app import AppLauncher

    stdout = sys.stdout
    try:
        _app = AppLauncher(headless=headless, enable_cameras=cameras, **kwargs).app
    finally:
        # AppLauncher silences stdout during startup, then sets it to sys.__stdout__ rather than the previous
        # stream. In Jupyter that is the kernel's terminal, so print() would stop reaching the notebook.
        sys.stdout = stdout
    _cameras = cameras
    return _app


def make_env(
    task: str,
    num_envs: int | None = None,
    render: bool = False,
    cfg_fn: Callable | None = None,
    device: str = "cuda:0",
) -> gym.Env:
    """Create a registered Isaac Lab task. Only one env per process (creating a second one hangs).

    Args:
        task: gym id, e.g. "Isaac-Cartpole-Direct-v0".
        num_envs: overrides the task default.
        render: enable env.render() -> RGB frames (requires launch(cameras=True)).
        cfg_fn: called with the task's env cfg before creation, to edit it in place (camera, reward weights, ...).
        device: simulation device.
    """
    global _env
    if _app is None:
        raise RuntimeError("Call api.isaac.launch() before make_env().")
    if _env is not None:
        raise RuntimeError(
            "An Isaac Lab env already exists in this process, and creating a second one hangs. "
            "Restart the kernel to switch tasks or settings."
        )
    if render and not _cameras:
        raise RuntimeError("render=True needs launch(cameras=True).")

    import isaaclab_tasks  # noqa: F401  (registers tasks)
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    if cfg_fn is not None:
        cfg_fn(cfg)
    _env = gym.make(task, cfg=cfg, render_mode="rgb_array" if render else None)
    return _env
