"""On-disk cache for experiment results, keyed by (experiment name, hyperparameters).

A result is a dict of named numpy arrays, saved as one .npz per cache entry; the file name is a hash of the experiment
name and its parameters, so re-running an unchanged experiment is a cache hit. Domain-agnostic: it knows nothing about
Isaac, ManiSkill, or any other task.

    from utils.experiment_cache import ExperimentCache

    cache = ExperimentCache()   # -> control/data/experiment_cache/ (gitignored)

    key    = cache.make_key("cartpole_markov_ppo", dict(ctx=1, iterations=150, seed=0))
    result = cache.get(key, expected={"return": (150,), "length": (150,)})
    if result is None:                       # miss, OR a hit whose shapes don't match
        ret, length = ...                    # (re)compute
        cache.set(key, {"return": ret, "length": length})
    else:
        ret, length = result["return"], result["length"]

or, equivalently, in one call via `compute`:

    result = cache.compute("cartpole_markov_ppo", dict(ctx=1, iterations=150, seed=0),
                           lambda: {"return": ret, "length": length}, expected={"return": (150,), "length": (150,)})

Model weights and training curves are just more named arrays: `pack_state_dict` / `unpack_state_dict` convert a torch
state dict to/from cache entries, so one entry can hold a whole training run (weights + history):

    run = cache.compute("cartpole_markov_ppo", params, lambda: {
        **pack_state_dict(ac.state_dict(), "ac."), **{f"history.{k}": np.asarray(v) for k, v in history.items()}})
    ac.load_state_dict(unpack_state_dict(run, "ac.", device=device))

`expected` is the recompute signal for a changed output shape/structure: when a cached entry's array shapes (or the set
of array names) don't match what the caller now expects, `get` returns None exactly as for a plain miss.

Anything that changes the result must be in `params` (including a seed, and a version string you bump when the
experiment's code changes meaning) -- the hash cannot see code, only the parameters you hand it.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from utils.path import ROOT_DIR

DEFAULT_CACHE_DIR = ROOT_DIR / "data" / "experiment_cache"

Params = Mapping[str, Any]
Result = Mapping[str, np.ndarray]


def _json_default(obj: Any) -> Any:
    """Fallback for values json can't encode: Paths and numpy scalars become plain values. Arrays and tensors are
    rejected -- str() of a large one is truncated ("[1 2 ... 9]"), so distinct params would silently share a key."""
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "shape"):  # np.ndarray, torch.Tensor, ...
        raise TypeError(f"param of type {type(obj).__name__} can't be hashed reliably; pass a plain value or a digest")
    return str(obj)


def _make_key(experiment: str, params: Params) -> str:
    """Stable hash over {experiment name, parameters}. Values must be JSON-serializable (numbers, strings, bools, None,
    lists/dicts thereof), or a Path / numpy scalar."""
    blob = json.dumps({"experiment": experiment, "params": params}, sort_keys=True, default=_json_default)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class ExperimentCache:
    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def make_key(self, experiment: str, params: Params) -> str:
        """The cache key for (experiment, params): the same inputs always hash to the same key."""
        return _make_key(experiment, params)

    def get(self, key: str, expected: Mapping[str, tuple[int, ...]] | None = None) -> dict[str, np.ndarray] | None:
        """The cached arrays for `key`, or None on a miss. If `expected` is given (array name -> shape), an entry
        missing a name or holding the wrong shape for one is ALSO a miss, so a changed output shape triggers a
        recompute instead of a silent mismatch downstream."""
        path = self._npz_path(key)
        if not path.exists():
            return None
        with np.load(path) as data:
            result = {name: data[name] for name in data.files}
        if expected is not None:
            for name, shape in expected.items():
                if name not in result or tuple(result[name].shape) != tuple(shape):
                    return None
        return result

    def set(self, key: str, result: Result, experiment: str | None = None, params: Params | None = None) -> None:
        """Cache `result` (name -> array) under `key`. `experiment`/`params`, if given, are written to a sidecar .json
        purely so the cache directory is human-inspectable (which hash is which run); they play no role in lookup.
        The .npz is written to a temp file and renamed, so an interrupted kernel never leaves a corrupt entry."""
        path = self._npz_path(key)
        tmp = path.with_name(f"{key}.tmp.npz")
        np.savez(tmp, **{k: np.asarray(v) for k, v in result.items()})
        os.replace(tmp, path)
        if experiment is not None or params is not None:
            meta = {"experiment": experiment, "params": params}
            self._meta_path(key).write_text(json.dumps(meta, indent=2, default=_json_default))

    def compute(
        self,
        experiment: str,
        params: Params,
        fn: Callable[[], Result],
        expected: Mapping[str, tuple[int, ...]] | None = None,
    ) -> dict[str, np.ndarray]:
        """get-or-compute-and-cache in one call: `fn()` runs (and its result is cached) only on a miss, including a
        cached entry whose shape no longer matches `expected`."""
        key = self.make_key(experiment, params)
        cached = self.get(key, expected)
        if cached is not None:
            return cached
        result = fn()
        self.set(key, result, experiment=experiment, params=params)
        return {k: np.asarray(v) for k, v in result.items()}

    def delete(self, key: str) -> None:
        self._npz_path(key).unlink(missing_ok=True)
        self._meta_path(key).unlink(missing_ok=True)

    def _npz_path(self, key: str) -> Path:
        return self.dir / f"{key}.npz"

    def _meta_path(self, key: str) -> Path:
        return self.dir / f"{key}.json"


def pack_state_dict(state_dict: Mapping[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    """torch state dict -> {prefix + name: numpy array}, ready to merge into a cache entry. bfloat16 (which numpy
    can't hold) is upcast to float32, which is lossless; `load_state_dict` casts back to the module's dtype."""
    import torch

    out = {}
    for name, t in state_dict.items():
        t = t.detach().cpu()
        out[prefix + name] = (t.float() if t.dtype == torch.bfloat16 else t).numpy()
    return out


def unpack_state_dict(result: Result, prefix: str = "", device: Any = "cpu") -> dict[str, Any]:
    """Inverse of `pack_state_dict`: the entries of `result` starting with `prefix`, as a torch state dict."""
    import torch

    return {
        name[len(prefix):]: torch.from_numpy(np.array(arr)).to(device)
        for name, arr in result.items()
        if name.startswith(prefix)
    }
