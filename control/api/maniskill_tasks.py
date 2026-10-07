"""Custom ManiSkill tasks not in the upstream library. Importing this registers their gym ids;
`api.maniskill.make_env` already imports it, so notebooks never need to import this directly.

Adapted from mani_skill's built-in StackCube-v1 (mani_skill/envs/tasks/tabletop/stack_cube.py).
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("StackThree-v1", max_episode_steps=100)
class StackThreeEnv(BaseEnv):
    """Stack three cubes into a tower: cubeB onto cubeC, then cubeA onto cubeB.

    Sparse reward only (two milestones, no dense shaping): +5 once cubeB is stacked on cubeC (resting,
    static, released), +10 total once cubeA is also stacked on cubeB. `reward_mode="dense"` is not
    implemented, since the task was asked for sparse rewards only -- only "sparse"/"normalized_dense" work.
    """

    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        kwargs.setdefault("reward_mode", "sparse")
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene, half_size=0.02, color=[1, 0, 0, 1], name="cubeA", initial_pose=sapien.Pose(p=[0, 0, 0.1])
        )
        self.cubeB = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 0, 1, 1], name="cubeB", initial_pose=sapien.Pose(p=[1, 0, 0.1])
        )
        self.cubeC = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 1, 0, 1], name="cubeC", initial_pose=sapien.Pose(p=[2, 0, 0.1])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.02
            xy = torch.rand((b, 2)) * 0.2 - 0.1
            region = [[-0.1, -0.2], [0.1, 0.2]]
            sampler = randomization.UniformPlacementSampler(bounds=region, batch_size=b, device=self.device)
            radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001

            # sampled sequentially so each cube avoids every cube already placed this call
            for cube in (self.cubeA, self.cubeB, self.cubeC):
                xyz[:, :2] = xy + sampler.sample(radius, 100, verbose=False)
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
                cube.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

    def _is_on(self, top, bottom):
        offset = top.pose.p - bottom.pose.p
        xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
        z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
        return torch.logical_and(xy_flag, z_flag)

    def evaluate(self):
        # lin_thresh loosened from StackCube-v1's 1e-2: a cube bearing another cube's weight has one more
        # contact point, and GPU PhysX's contact jitter alone sits around 0.015-0.017 and never decays
        # below 1e-2 even after 100+ idle steps (checked empirically) -- 1e-2 would make success unreachable.
        is_B_grasped = self.agent.is_grasping(self.cubeB)
        b_stacked = (
            self._is_on(self.cubeB, self.cubeC)
            & self.cubeB.is_static(lin_thresh=2e-2, ang_thresh=0.5)
            & ~is_B_grasped
        )
        is_A_grasped = self.agent.is_grasping(self.cubeA)
        a_stacked = (
            self._is_on(self.cubeA, self.cubeB)
            & self.cubeA.is_static(lin_thresh=2e-2, ang_thresh=0.5)
            & ~is_A_grasped
        )
        success = b_stacked & a_stacked
        return {
            "is_cubeA_grasped": is_A_grasped,
            "is_cubeB_grasped": is_B_grasped,
            "b_stacked": b_stacked,
            "a_stacked": a_stacked,
            "success": success.bool(),
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if "state" in self.obs_mode:
            obs.update(
                cubeA_pose=self.cubeA.pose.raw_pose,
                cubeB_pose=self.cubeB.pose.raw_pose,
                cubeC_pose=self.cubeC.pose.raw_pose,
            )
        return obs

    def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
        reward = torch.zeros(self.num_envs, device=self.device)
        reward[info["b_stacked"]] = 5.0
        reward[info["success"]] = 10.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_sparse_reward(obs=obs, action=action, info=info) / 10.0


def _pick_and_stack(planner: PandaArmMotionPlanningSolver, env: StackThreeEnv, obj, goal_pose: sapien.Pose):
    """Grasp `obj` and place it so it lands at `goal_pose`. Lifted from mani_skill's StackCube-v1 solution
    (examples/motionplanning/panda/solutions/stack_cube.py), factored out since StackThree needs it twice."""
    FINGER_LENGTH = 0.025
    obb = get_actor_obb(obj)
    approaching = np.array([0, 0, -1])
    target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH)
    grasp_pose = env.agent.build_grasp_pose(approaching, grasp_info["closing"], grasp_info["center"])

    # search for a reachable grasp angle, as the cube's z-rotation is randomized
    angles = np.arange(0, np.pi * 2 / 3, np.pi / 2)
    angles = np.repeat(angles, 2)
    angles[1::2] *= -1
    for angle in angles:
        grasp_pose2 = grasp_pose * sapien.Pose(q=euler2quat(0, 0, angle))
        if planner.move_to_pose_with_screw(grasp_pose2, dry_run=True) != -1:
            grasp_pose = grasp_pose2
            break
    else:
        print("Fail to find a valid grasp pose")

    planner.move_to_pose_with_screw(grasp_pose * sapien.Pose([0, 0, -0.05]))  # reach
    planner.move_to_pose_with_screw(grasp_pose)  # grasp
    planner.close_gripper()

    lift_pose = sapien.Pose([0, 0, 0.1]) * grasp_pose
    planner.move_to_pose_with_screw(lift_pose)

    offset = (goal_pose.p - obj.pose.p).cpu().numpy()[0]  # ManiSkill poses are batched tensors
    planner.move_to_pose_with_screw(sapien.Pose(lift_pose.p + offset, lift_pose.q))

    return planner.open_gripper()


def solve_stack_three(env: StackThreeEnv, seed: int | None = None, debug: bool = False, vis: bool = False):
    """Motion-planned solution for StackThree-v1, for generating a one-off demonstration: grasp cubeB and
    place it on cubeC, then grasp cubeA and place it on cubeB. Requires control_mode="pd_joint_pos".
    """
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis, base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis, print_env_info=False,
    )
    env = env.unwrapped
    cube_height = (env.cube_half_size[2] * 2).item()

    goal_B = env.cubeC.pose * sapien.Pose([0, 0, cube_height])
    _pick_and_stack(planner, env, env.cubeB, goal_B)

    goal_A = env.cubeB.pose * sapien.Pose([0, 0, cube_height])  # read after cubeB has actually been placed
    res = _pick_and_stack(planner, env, env.cubeA, goal_A)

    planner.close()
    return res
