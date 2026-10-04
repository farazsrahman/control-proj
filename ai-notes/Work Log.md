### September 30th, 2026

- Isaac Lab 2.3.2 + Isaac Sim 5.1 installed as the uv project in `control/`. Isaac Lab is an editable clone in `control/deps/IsaacLab`. Always run with `uv run --env-file .env` from `control/`; it adds a locally extracted `libGLU.so.1` (`deps/sysroot`), without which Kit hangs at startup.
- Eureka pen spinning ported to `control/deps/eureka_pen_spin`: `Eureka-Reorient-Pen-Shadow-Direct-v0` (pre-train) and `Eureka-Spin-Pen-Shadow-Direct-v0` (spin waypoints). Uses the human reward from the released code, not a GPT reward. The released `EurekaPenSpinning.pth` does not transfer (different hand asset and observations). See its README for other deviations.

### October 2nd, 2026

- Pretrained `Isaac-Repose-Cube-Shadow-Direct-v0` (rl_games) plays correctly headless; videos in `control/logs/videos/pretrained-cube-shadow/`. Default camera is too far away. For a close-up use `--num_envs 1 'env.viewer.origin_type=env' 'env.viewer.eye=[0.45,-0.85,0.95]' 'env.viewer.lookat=[0.0,-0.35,0.55]'`.
- Quirk in `play.py --use_pretrained_checkpoint`: once the checkpoint is cached, videos go to `logs/rl_games/<name>/.pretrained_checkpoints/...` instead of `.pretrained_checkpoints/.../videos/`.
- Notebooks: `control/scripts/jupyter.sh [port]` starts JupyterLab with the Isaac Sim environment variables (kernels inherit them). JupyterLab is pinned `<4.4` because isaaclab-rl pins `packaging<24`.
- One Isaac Lab env per kernel: making a second env after `env.close()` hung indefinitely (cube, then pen). Restart the kernel to switch tasks.
- Removed the Eureka pen-spin port (`control/deps/eureka_pen_spin`, and its `train.py`/`play.py` wrappers). Using only Isaac Lab's built-in tasks for now. Run Isaac Lab's scripts directly: `deps/IsaacLab/scripts/reinforcement_learning/<lib>/{train,play}.py`.
- Deleted all one-off videos and run outputs (`control/logs/videos/`, stray mp4s). The notebook `notebooks/isaac-lab-quickstart/cube_video.ipynb` embeds its own videos and replaces them.
- Isaac Sim takes over `sys.stdout` on launch, so `print()` output never appears in notebooks (it goes to the kernel log). Save `sys.stdout`/`sys.stderr` before `AppLauncher(...)` and restore them right after (done in `cartpole_ppo.ipynb`).
- `control` is now an installable package (`api`, `utils`). Notebooks should start with `from api import sim; app = sim.launch(cameras=True)`, which restores `sys.stdout`, accepts the EULA and preloads libGLU, so no env vars are needed. Create envs with `sim.make_env(...)`, which raises an error on a second env instead of hanging. `api.video.show(frames, fps)` embeds videos in the notebook.
- `api.policies.load_rl_games(task, checkpoint=None, device)` loads an rl_games MLP checkpoint (default: NVIDIA-published) as a torch module, with network size and clipping read from the task's agent cfg. Downloads are cached in `control/.pretrained_checkpoints/`. Verified on the cube and cartpole tasks.
