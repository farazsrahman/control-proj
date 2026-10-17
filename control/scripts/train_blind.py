"""Train the blind repose policies with rl_games PPO and a privileged critic (api/blind_repose.py).

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/train_blind.py --variant proprio --name blind-proprio [--max_epochs 5000]

    --variant proprio | touch | privileged   (privileged = pipeline check with the stock 157-d actor observation)
Logs + checkpoints: logs/rl_games/blind_repose/<name>/ (nn/*.pth, summaries/ for tensorboard). Same PPO settings as the stock Shadow-hand
teacher (api/blind_ppo_cfg.yaml), 8192 envs by default. Evaluate checkpoints with the notebook protocol, not the training reward.
"""

import argparse
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=["proprio", "touch", "privileged"], required=True)
parser.add_argument("--name", default=None, help="run name (log sub-directory); default: blind-<variant>")
parser.add_argument("--num_envs", type=int, default=8192)
parser.add_argument("--max_epochs", type=int, default=5000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint", default=None, help="resume from this rl_games checkpoint")
args = parser.parse_args()

from api import isaac

isaac.launch(headless=True)

import gymnasium as gym
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

import api.blind_repose  # noqa: F401  (registers the tasks)
import isaaclab_tasks  # noqa: F401
from api.joint_torque import enable_contact_reporting
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

TASK = {
    "proprio": "Isaac-Repose-Cube-Shadow-Blind-Proprio-v0",
    "touch": "Isaac-Repose-Cube-Shadow-Blind-Touch-v0",
    "privileged": "Isaac-Repose-Cube-Shadow-Privileged-v0",
}[args.variant]
name = args.name or f"blind-{args.variant}"

env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args.num_envs)
env_cfg.seed = args.seed
if args.variant == "touch":
    enable_contact_reporting(env_cfg)
agent_cfg = load_cfg_from_registry(TASK, "rl_games_cfg_entry_point")
cfg = agent_cfg["params"]
cfg["seed"] = args.seed
cfg["config"]["max_epochs"] = args.max_epochs
cfg["config"]["num_actors"] = args.num_envs
log_root = os.path.abspath(os.path.join("logs", "rl_games", cfg["config"]["name"]))
cfg["config"]["train_dir"] = log_root
cfg["config"]["full_experiment_name"] = name
if args.checkpoint:
    cfg["load_checkpoint"], cfg["load_path"] = True, args.checkpoint
dump_yaml(os.path.join(log_root, name, "params", "env.yaml"), env_cfg)
dump_yaml(os.path.join(log_root, name, "params", "agent.yaml"), agent_cfg)
print(f"[train_blind] task {TASK}, run {name}, {args.num_envs} envs, {args.max_epochs} epochs, logs {log_root}/{name}", flush=True)

env = gym.make(TASK, cfg=env_cfg)
env = RlGamesVecEnvWrapper(
    env, cfg["config"]["device"], cfg["env"].get("clip_observations", math.inf), cfg["env"].get("clip_actions", math.inf),
    cfg["env"].get("obs_groups"), cfg["env"].get("concate_obs_groups", True),
)
vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

runner = Runner(IsaacAlgoObserver())
runner.load(agent_cfg)
runner.reset()
if args.checkpoint:
    runner.run({"train": True, "play": False, "sigma": None, "checkpoint": args.checkpoint})
else:
    runner.run({"train": True, "play": False, "sigma": None})
env.close()
