"""Throughput benchmarks for ManiSkill GPU-sim rollouts (frames/s = num_envs * env steps / s).

Modes:
    "state": physics only, no rendering (obs_mode="state").
    "rgb":   physics + rendering one RGB sensor camera.
    "track": physics + rendering rgb/position/segmentation + GT grid point tracking (api.tracking.locate_batched).

    from api import profiling
    res = profiling.run(task="StackCube-v1", mode="rgb", num_envs=1024, resolution=128)  # isolated subprocess

Each `run` is its own process: GPU PhysX is initialized once per process, and an OOM at large num_envs
shouldn't take down the notebook kernel. Everything stays on the GPU (no .cpu() of observations).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

CAM = "base_camera"


def bench(
    task: str = "StackCube-v1",
    mode: str = "state",
    num_envs: int = 1024,
    resolution: int = 128,
    steps: int = 200,
    warmup: int = 20,
    grid_size: int = 32,
    breakdown_steps: int = 50,
) -> dict:
    """Time `steps` env steps (after `warmup`) with random actions. Run via `run()` for process isolation."""
    import torch

    from api import maniskill, tracking

    obs_mode = {"state": "state", "rgb": "rgb", "track": tracking.OBS_MODE}[mode]
    kwargs = dict(sensor_configs=dict(width=resolution, height=resolution)) if mode != "state" else {}
    t0 = time.perf_counter()
    env = maniskill.make_env(
        task, num_envs=num_envs, obs_mode=obs_mode,
        sim_backend="physx_cuda",  # GPU sim even at num_envs=1, so every point of the sweep is the same backend
        robot_uids="panda",  # no wrist camera: render exactly one camera (base_camera) per frame
        max_episode_steps=10**6,  # no truncation mid-benchmark
        **kwargs,
    )  # fmt: skip
    obs, _ = env.reset(seed=0)
    setup_s = time.perf_counter() - t0

    # random actions pre-generated on the GPU (sampling 4096-env actions on the host every step would be timed too)
    low = torch.as_tensor(env.action_space.low, device="cuda")
    high = torch.as_tensor(env.action_space.high, device="cuda")
    actions = low + (high - low) * torch.rand((16, *low.shape), device="cuda")

    pts = tracking.grid_points_batched(env, obs, CAM, grid_size) if mode == "track" else None

    def step(i):
        obs = env.step(actions[i % len(actions)])[0]
        if pts is not None:
            tracking.locate_batched(obs, pts)

    for i in range(warmup):
        step(i)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(steps):
        step(i)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gpu_mem = _process_gpu_mem_gb()

    out = dict(
        task=task, mode=mode, num_envs=num_envs, resolution=resolution if mode != "state" else None,
        fps=num_envs * steps / dt, ms_per_step=1000 * dt / steps, setup_s=setup_s,
        gpu_mem_gb=gpu_mem,  # this process, from nvidia-smi (PhysX/SAPIEN allocations aren't visible to torch)
    )  # fmt: skip

    if pts is not None:  # how much of a tracked step is the tracking itself (synced per part, so slightly pessimistic)
        t_step = t_locate = 0.0
        for i in range(breakdown_steps):
            t0 = time.perf_counter()
            obs = env.step(actions[i % len(actions)])[0]
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            tracking.locate_batched(obs, pts)
            torch.cuda.synchronize()
            t_step, t_locate = t_step + t1 - t0, t_locate + time.perf_counter() - t1
        out.update(ms_step_render=1000 * t_step / breakdown_steps, ms_locate=1000 * t_locate / breakdown_steps)
    env.close()
    return out


def _process_gpu_mem_gb(pid: int | None = None) -> float | None:
    import os

    pid = pid or os.getpid()
    q = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )  # fmt: skip
    for line in q.stdout.splitlines():
        p, mem = (x.strip() for x in line.split(","))
        if int(p) == pid:
            return float(mem) / 1024
    return None


def gpu_status() -> str:
    """GPU name/memory and every process holding GPU memory: other jobs on a shared GPU cap num_envs and skew fps."""
    smi = lambda *a: subprocess.run(["nvidia-smi", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    return "\n".join([
        smi("--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv"),
        smi("--query-compute-apps=pid,used_memory,process_name", "--format=csv"),
    ])  # fmt: skip


def run(timeout: float = 900, **kwargs) -> dict:
    """`bench(**kwargs)` in a fresh subprocess. On failure (e.g. OOM) returns kwargs + {"error": <last stderr line>}."""
    code = f"import json; from api.profiling import bench; print('RESULT', json.dumps(bench(**{kwargs!r})))"
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {**kwargs, "error": f"timeout after {timeout}s"}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    err = [l for l in p.stderr.strip().splitlines() if l.strip()]
    return {**kwargs, "error": err[-1] if err else f"exit code {p.returncode}"}
