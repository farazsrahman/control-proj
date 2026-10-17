"""The full-episode evaluation protocol shared by the repose experiments.

    from api.repose_eval import evaluate_policy, summarize
    r, g, l = evaluate_policy(env, lambda obs: policy(obs), episodes=2, seed=1)
    print(summarize(r, g, l, env.unwrapped.max_episode_length))

Greedy rollouts until every env has finished 1 + `episodes` episodes; each env's FIRST episode is discarded (envs do not all start at the
same episode phase) and the next `episodes` are counted, so every counted episode is complete and starts from a fresh reset. Per counted
episode: return = raw env reward summed over the episode, goals = env.successes at the episode's end, length in env steps. The env resets
`successes` on the step an episode ends, so the value from the PREVIOUS step is used (the convention of scripts/eval_isaac.py). Timeouts
end an episode after max_episode_length - 1 steps, so "dropped the cube" means length < max_episode_length - 1.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def evaluate_policy(env, act_fn, episodes: int = 2, seed: int = 1, max_steps: int | None = None):
    """(return, goals, length) numpy arrays over the counted episodes of every env. `act_fn(obs) -> action` (obs as the env returns it)."""
    u = env.unwrapped
    device, n = u.device, u.num_envs
    torch.manual_seed(seed)
    obs, _ = env.reset(seed=seed)
    ret = torch.zeros(n, device=device)
    length = torch.zeros(n, device=device)
    last_succ = torch.zeros(n, device=device)
    n_done = torch.zeros(n, dtype=torch.long, device=device)
    R, G, L = [], [], []
    steps = 0
    while n_done.min() < 1 + episodes and (max_steps is None or steps < max_steps):
        obs, rew, term, trunc, _ = env.step(act_fn(obs))
        steps += 1
        ret += rew
        length += 1
        done = term | trunc
        if done.any():
            counted = done & (n_done >= 1) & (n_done < 1 + episodes)
            R += ret[counted].tolist()
            G += last_succ[counted].tolist()
            L += length[counted].tolist()
            n_done[done] += 1
            ret[done] = 0
            length[done] = 0
        last_succ = u.successes.float().clone()
        last_succ[done] = 0
    return np.array(R), np.array(G), np.array(L)


def summarize(r, g, l, max_episode_length: int) -> dict:
    """Mean and standard error of return and goals, and the fraction of episodes that ended early (the cube was dropped)."""
    se = lambda a: float(a.std(ddof=1) / np.sqrt(len(a)))
    return dict(n=len(r), ret=(float(r.mean()), se(r)), goals=(float(g.mean()), se(g)), dropped=float((l < max_episode_length - 1).mean()))
