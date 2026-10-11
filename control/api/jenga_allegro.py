"""JengaDesign with a single free-floating Allegro hand instead of a Panda arm. Importing this registers `JengaAllegro-v1`.

Identical task, scoring, block scatter and design grid to `JengaDesign-v1` (see `api.jenga`); only the embodiment
changes. The hand (`api.floating_allegro`) has a 6-DoF virtual root, so it can reach anywhere: the scatter region
is unchanged from the Panda env (its reach annulus is simply not a constraint here). Actions are 6 root + 16
finger dims (22), `pd_joint_delta_pos` by default.

Not yet tuned: the init pose, root gains and grasp strategy have only been written, not exercised in simulation.
"""

from __future__ import annotations

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder

import api.floating_allegro  # noqa: F401  (registers floating_allegro_hand_right)
from api.jenga import JengaDesignEnv

# Hand base_link origin at reset. Fingers extend ~0.2 m along local -z, so at identity orientation the fingertips
# hang ~0.1-0.15 m above the table. y = 0 sits between the design grid (y < 0) and the scatter half (y > 0).
HAND_INIT_POS = np.array([-0.15, 0.0, 0.35])
HAND_INIT_EULER = (0.0, 0.0, 0.0)  # identity: fingers down, palm normal horizontal (the stock `palm_side` pose)


class FloatingHandTableSceneBuilder(TableSceneBuilder):
    """TableSceneBuilder only knows a fixed list of robot uids; this one places the floating hand itself."""

    def initialize(self, env_idx: torch.Tensor):
        self.table.set_pose(sapien.Pose(p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2)))
        agent = self.env.agent
        agent.robot.set_pose(sapien.Pose(p=HAND_INIT_POS, q=euler2quat(*HAND_INIT_EULER)))
        qpos = torch.zeros((len(env_idx), agent.robot.max_dof), device=self.env.device)
        qpos += torch.randn_like(qpos) * self.robot_init_qpos_noise
        agent.reset(qpos)


@register_env("JengaAllegro-v1", max_episode_steps=2000)
class JengaAllegroEnv(JengaDesignEnv):
    SUPPORTED_ROBOTS = ["floating_allegro_hand_right"]
    TABLE_SCENE_BUILDER = FloatingHandTableSceneBuilder

    def __init__(self, *args, robot_uids="floating_allegro_hand_right", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        # skip JengaDesignEnv._load_agent, which spawns the robot at the Panda base position
        super(JengaDesignEnv, self)._load_agent(options, sapien.Pose(p=HAND_INIT_POS, q=euler2quat(*HAND_INIT_EULER)))

    @property
    def _default_human_render_camera_configs(self):
        from mani_skill.sensors.camera import CameraConfig
        from mani_skill.utils import sapien_utils

        # same framing as JengaDesign-v1, pulled back a little so the whole hand stays in view
        pose = sapien_utils.look_at([0.6, -0.8, 0.9], [-0.2, 0.0, 0.15])
        return CameraConfig("render_camera", pose, 512, 512, 1.1, 0.01, 100)
