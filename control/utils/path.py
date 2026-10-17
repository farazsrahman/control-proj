from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent.resolve()
# every cache (package managers, model hubs, compiled kernels, generated assets, temp files) lives here: gitignored, excluded
# from syncs, on the same filesystem as the repo (shared machines have small home quotas). Shell side: scripts/env.sh.
CACHE_DIR = ROOT_DIR / ".cache"


if __name__ == "__main__":
    print("ROOT_DIR is:", ROOT_DIR)
