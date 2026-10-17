"""Dump the data for the offline hand renderer (utils/hand_render.py): link meshes + backlash rollouts + FK of the true / measured hand.

Run on the node from control/ (GPU via the GPU env var, default 0):
    GPU=0 .venv/bin/python scripts/diagnostics/dump_hand_viz.py                       # N = 0 (clean), 100, 288, 1200
    GPU=1 NS=0,288 STEPS=100 .venv/bin/python scripts/diagnostics/dump_hand_viz.py
Env vars: GPU, NS (gear ratios, 0 = clean), STEPS, NUM_ENVS (also the FK batch size), PPO_CKPT_PATH (relative to control/, default: the
blind-proprio checkpoint of the measurement sweep), OUT (default control/.cache/tmp/hand_viz).
Writes meshes.npz and rollout_N<n>.npz (arrays [T, 4 envs, ...]: q_true, q_meas, sim link poses, FK link poses at q_true and at q_meas, cube).
Prints the consistency checks (FK vs the sim's own link poses; fingertip positions).
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")
from api import isaac

app = isaac.launch(headless=True)

import time
from pathlib import Path

import numpy as np
import torch

import api.blind_repose  # noqa: F401  (registers the blind tasks)
from api import hand_viz, policies
from utils import hand_render as hr
from utils.path import CACHE_DIR, ROOT_DIR

NS = [float(x) for x in os.environ.get("NS", "0,100,288,1200").split(",")]
STEPS = int(os.environ.get("STEPS", "200"))
NUM_ENVS = int(os.environ.get("NUM_ENVS", "64"))
CKPT = os.environ.get("PPO_CKPT_PATH", "logs/rl_games/blind_repose/blind-proprio/nn/last_blind_repose_ep_5000_rew_3925.8337.pth")
OUT = Path(os.environ.get("OUT", CACHE_DIR / "tmp" / "hand_viz"))
OUT.mkdir(parents=True, exist_ok=True)
TASK = "Isaac-Repose-Cube-Shadow-Blind-Proprio-v0"

env = isaac.make_env(TASK, num_envs=NUM_ENVS)
u = env.unwrapped
policy = policies.load_rl_games(TASK, checkpoint=ROOT_DIR / CKPT, device=u.device)
print(f"{TASK}, {NUM_ENVS} envs, joints {u.hand.joint_names}\nbodies {u.hand.body_names}", flush=True)

meshes = hand_viz.export_meshes(env)
np.savez_compressed(OUT / "meshes.npz", **{f"{k}/v": v for k, (v, f) in meshes.items()}, **{f"{k}/f": f for k, (v, f) in meshes.items()},
                    joint_names=np.array(u.hand.joint_names), body_names=np.array(u.hand.body_names),
                    finger_bodies=np.array(u.finger_bodies), env_origins=u.scene.env_origins.cpu().numpy())
rolls = {}
for n in NS:
    t0 = time.time()
    rolls[n] = hand_viz.record_rollout(env, policy, n=n, backlash=True, steps=STEPS)
    print(f"rollout N={n:g}: {time.time() - t0:.0f}s", flush=True)

# FK pass (destroys the env state, so after all rollouts): link poses at the true and at the measured joint positions
for n, r in rolls.items():
    T, E, nd = r["q_true"].shape
    for key in ["true", "meas"]:
        q = torch.as_tensor(r[f"q_{key}"].reshape(T * E, nd), device=u.device)
        P, Q = [], []
        for i in range(0, len(q), NUM_ENVS):
            p, qu = hand_viz.link_poses(env, q[i : i + NUM_ENVS])
            P.append(p.cpu().numpy())
            Q.append(qu.cpu().numpy())
        r[f"fk_{key}_pos"] = np.concatenate(P).reshape(T, E, -1, 3)
        r[f"fk_{key}_quat"] = np.concatenate(Q).reshape(T, E, -1, 4)
    # consistency: FK at the true joint positions vs the link poses the sim reported in the rollout (frames that are not a reset frame)
    ok = r["ep"] > 1
    dp = np.linalg.norm(r["fk_true_pos"] - r["pos"], axis=-1)[ok]
    dq = 2 * np.arccos(np.clip(np.abs((r["fk_true_quat"] * r["quat"]).sum(-1)), 0, 1))[ok]
    tips = u.finger_bodies
    print(f"N={n:g}: FK(q_true) vs sim link poses over {int(ok.sum())} frames: pos err max {dp.max():.2e} m, mean {dp.mean():.2e}; "
          f"rot err max {np.degrees(dq.max()):.3f} deg; fingertip bodies {tips}; "
          f"|FK(q_meas) - FK(q_true)| at fingertips: mean {1e3 * np.linalg.norm(r['fk_meas_pos'] - r['fk_true_pos'], axis=-1)[..., tips].mean():.2f} mm, "
          f"max {1e3 * np.linalg.norm(r['fk_meas_pos'] - r['fk_true_pos'], axis=-1)[..., tips].max():.2f} mm", flush=True)
    np.savez_compressed(OUT / f"rollout_N{n:g}.npz", **r)
# post-hoc backlash: the play operator applied to the CLEAN trajectory (same motion, any h), FK'd like the others; and a check that this numpy
# re-implementation reproduces what MeasurementModel recorded in the rollouts of the same N (frames before each env's first reset)
if 0.0 in rolls:
    r0 = rolls[0.0]
    T, E, nd = r0["q_true"].shape
    for n in [n for n in NS if n > 0]:
        qm = hr.play_operator(r0["q_true"], hr.half_width(n), r0["q_reset"], r0["ep"] == 0)
        q = torch.as_tensor(qm.reshape(T * E, nd), device=u.device)
        P, Q = [], []
        for i in range(0, len(q), NUM_ENVS):
            p_, qu = hand_viz.link_poses(env, q[i : i + NUM_ENVS])
            P.append(p_.cpu().numpy()); Q.append(qu.cpu().numpy())  # noqa: E702
        r0[f"q_ph{n:g}"] = qm
        r0[f"fk_ph{n:g}_pos"] = np.concatenate(P).reshape(T, E, -1, 3)
        r0[f"fk_ph{n:g}_quat"] = np.concatenate(Q).reshape(T, E, -1, 4)
        if n in rolls:
            r = rolls[n]
            mine = hr.play_operator(r["q_true"], hr.half_width(n), r["q_reset"], r["ep"] == 0)
            first_reset = np.where((r["ep"] == 0).any(0), (r["ep"] == 0).argmax(0), T)  # per env
            err = [np.abs(mine[: first_reset[e], e] - r["q_meas"][: first_reset[e], e]).max() for e in range(E) if first_reset[e] > 0]
            print(f"play_operator vs MeasurementModel at N={n:g}: max |diff| {max(err):.2e} rad over frames before the first reset ({[int(x) for x in first_reset]})", flush=True)
    np.savez_compressed(OUT / "rollout_N0.npz", **r0)
print("wrote", sorted(p.name for p in OUT.iterdir()), "to", OUT)
import sys

sys.stdout.flush()
os._exit(0)  # app.close() can leave the Kit process alive at ~100% CPU holding GPU memory, so leave without it (the outputs are written)
