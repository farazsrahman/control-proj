"""Dump for the single-joint backlash demo (notebooks/repose/repose-backlash-joint-demo.ipynb): ONE joint driven through a waypoint path, the backlash
model's readings of it (the sim's MeasurementModel), and the link poses of the true and the perceived hand.

Run on the node from control/:   GPU=0 PYTHONPATH=. .venv/bin/python scripts/diagnostics/dump_joint_demo.py
Env vars: GPU (default 0), JOINT (default robot0_FFJ2: index-finger proximal flexion, checked in the printout), N (gear ratio, default 200),
WAYPOINTS (degrees, default 0,50,32,50,0), FRAMES_PER_SEG (default 60 env steps = 2 s), NUM_ENVS (FK batch), OUT (default control/.cache/tmp/joint_demo).
Writes meshes.npz and joint_demo.npz; prints the checks (numpy play operator vs MeasurementModel, which links turn with the joint).
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU", "0")
from api import isaac

app = isaac.launch(headless=True)

from pathlib import Path

import numpy as np

import api.blind_repose  # noqa: F401  (registers the blind tasks)
from api import hand_viz
from utils import hand_render as hr
from utils.path import CACHE_DIR

JOINT = os.environ.get("JOINT", "robot0_FFJ2")
N = float(os.environ.get("N", "200"))
WAYPOINTS = [float(x) for x in os.environ.get("WAYPOINTS", "0,50,32,50,0").split(",")]
FPS_ = int(os.environ.get("FRAMES_PER_SEG", "60"))
NUM_ENVS = int(os.environ.get("NUM_ENVS", "64"))
OUT = Path(os.environ.get("OUT", CACHE_DIR / "tmp" / "joint_demo"))
OUT.mkdir(parents=True, exist_ok=True)
TASK = "Isaac-Repose-Cube-Shadow-Blind-Proprio-v0"

env = isaac.make_env(TASK, num_envs=NUM_ENVS)
u = env.unwrapped
env.reset(seed=1)
meshes = hand_viz.export_meshes(env)
np.savez_compressed(OUT / "meshes.npz", **{f"{k}/v": v for k, (v, f) in meshes.items()}, **{f"{k}/f": f for k, (v, f) in meshes.items()},
                    joint_names=np.array(u.hand.joint_names), body_names=np.array(u.hand.body_names),
                    finger_bodies=np.array(u.finger_bodies), env_origins=u.scene.env_origins.cpu().numpy())
d = hand_viz.drive_joint(env, JOINT, WAYPOINTS, FPS_, N)
j = int(d["joint_index"])
print(f"joint {JOINT} (index {j}), soft limits {np.degrees(d['limits']).round(1)} deg, N={N:g}, h={np.degrees(hr.half_width(N)):.3f} deg, {len(d['q_true'])} frames")

# 1. the numpy play operator (utils/hand_render.py) against the MeasurementModel readings of the same trajectory
mine = hr.play_operator(d["q_true"], hr.half_width(N), d["q_true"][0], np.zeros(d["q_true"].shape[:2], bool))
print(f"numpy play operator vs MeasurementModel: max |diff| {np.abs(mine - d['q_meas']).max():.2e} rad")
# 2. only the driven joint's error is non-zero
err = d["q_meas"] - d["q_true"]
print(f"max |error| per joint (deg), driven joint {np.degrees(np.abs(err[..., j]).max()):.3f}; all other joints {np.degrees(np.abs(np.delete(err, j, axis=-1)).max()):.2e}")
# 3. which links turn when the joint moves: the joint's CHILD link and everything below it rotate (a link's own origin sits on its joint axis, so
#    its position alone would not show it; use orientation), in kinematic order
k = int(np.abs(d["q_true"][:, 0, j]).argmax())
turn = np.degrees(hr.rot_angle(d["fk_true_quat"][0, 0], d["fk_true_quat"][k, 0]))
print("links that rotate with the joint, frame 0 -> max (deg):", {b: round(float(a), 1) for b, a in zip(u.hand.body_names, turn) if a > 0.1})
np.savez_compressed(OUT / "joint_demo.npz", **d)
print("wrote", sorted(p.name for p in OUT.iterdir()), "to", OUT)
import sys

sys.stdout.flush()
os._exit(0)  # app.close() can leave the Kit process alive at ~100% CPU holding GPU memory, so leave without it (the outputs are written)
