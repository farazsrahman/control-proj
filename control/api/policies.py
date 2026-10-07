"""Load trained rl_games policies as plain torch modules (no rl_games runner needed).

    from api import policies
    policy = policies.load_rl_games(TASK, device=env.unwrapped.device)   # NVIDIA-published checkpoint
    action = policy(obs)                                                # obs: env observation dict or tensor
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from utils.path import ROOT_DIR

# downloaded checkpoints are cached here regardless of the caller's working directory
CHECKPOINT_DIR = ROOT_DIR / ".pretrained_checkpoints"

_ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU, "sigmoid": nn.Sigmoid}


def pretrained_checkpoint(task: str, workflow: str = "rl_games") -> Path:
    """Path to NVIDIA's published checkpoint for `task`, downloaded into CHECKPOINT_DIR on first use."""
    from isaaclab.utils.assets import retrieve_file_path
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint_path

    remote = get_published_pretrained_checkpoint_path(workflow, task)
    local = CHECKPOINT_DIR / workflow / task / Path(remote).name
    if not local.exists():
        try:
            retrieve_file_path(remote, download_dir=str(local.parent))
        except Exception as e:
            raise FileNotFoundError(f"No published {workflow} checkpoint for {task} ({remote})") from e
    return local


class RlGamesMlpPolicy(nn.Module):
    """Deterministic (mean-action) actor of an rl_games `actor_critic` MLP, with rl_games' obs/action handling."""

    def __init__(self, net, obs_mean, obs_var, clip_obs, clip_actions):
        super().__init__()
        self.net = net
        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_var", obs_var)
        self.clip_obs = clip_obs
        self.clip_actions = clip_actions

    @torch.no_grad()
    def forward(self, obs):
        x = obs["policy"] if isinstance(obs, dict) else obs
        x = x.clamp(-self.clip_obs, self.clip_obs)  # RlGamesVecEnvWrapper clip_observations
        if self.obs_mean is not None:
            x = ((x - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5)).clamp(-5.0, 5.0)  # RunningMeanStd
        mu = self.net(x)
        if math.isfinite(self.clip_actions):
            mu = mu.clamp(-1.0, 1.0) * self.clip_actions  # rl_games player clamps, then rescales to the action box
        return mu


def load_rl_games(task: str, checkpoint: str | Path | None = None, device: str = "cuda:0") -> RlGamesMlpPolicy:
    """Build the policy for `task` from an rl_games checkpoint (default: NVIDIA's published one).

    Architecture and clipping come from the task's registered rl_games agent cfg. Only plain MLP actors are
    supported (no RNN/CNN), which covers most Isaac Lab state-based tasks.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    params = load_cfg_from_registry(task, "rl_games_cfg_entry_point")["params"]
    mlp_cfg = params["network"]["mlp"]
    if params["network"].get("rnn") or params["network"].get("cnn") or mlp_cfg.get("d2rl"):
        raise NotImplementedError(f"{task}: only plain MLP rl_games networks are supported")
    activation = _ACTIVATIONS[mlp_cfg["activation"]]

    path = Path(checkpoint) if checkpoint is not None else pretrained_checkpoint(task)
    sd = torch.load(path, map_location=device, weights_only=False)["model"]

    layers = []
    for i in sorted({int(k.split(".")[2]) for k in sd if k.startswith("a2c_network.actor_mlp.")}):
        w, b = sd[f"a2c_network.actor_mlp.{i}.weight"], sd[f"a2c_network.actor_mlp.{i}.bias"]
        lin = nn.Linear(w.shape[1], w.shape[0])
        lin.load_state_dict({"weight": w, "bias": b})
        layers += [lin, activation()]
    mu_w, mu_b = sd["a2c_network.mu.weight"], sd["a2c_network.mu.bias"]
    mu = nn.Linear(mu_w.shape[1], mu_w.shape[0])
    mu.load_state_dict({"weight": mu_w, "bias": mu_b})

    normalize = params["config"].get("normalize_input", False)
    obs_mean = sd["running_mean_std.running_mean"].float() if normalize else None  # stored as float64
    obs_var = sd["running_mean_std.running_var"].float() if normalize else None
    env_params = params.get("env", {})
    policy = RlGamesMlpPolicy(
        nn.Sequential(*layers, mu),
        obs_mean,
        obs_var,
        clip_obs=env_params.get("clip_observations", math.inf),
        clip_actions=env_params.get("clip_actions", math.inf),
    )
    return policy.to(device).eval()
