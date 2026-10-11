"""Floating (free-base) Allegro hand for ManiSkill. Importing this registers `floating_allegro_hand_right`.

ManiSkill ships `allegro_hand_right` / `allegro_hand_left` (fixed base) and floating variants of other hands
(Inspire, Ability), but no floating Allegro. This builds one the same way the Ability hand does: a chain of 6
massless virtual joints (x/y/z prismatic, then x/y/z revolute) between a `root` link and the Allegro `base_link`.
The URDF is generated from ManiSkill's own Allegro URDF at import time (no vendored copy), into a cache dir that
symlinks back to the stock meshes.

Joint layout: qpos = [6 root joints, 16 finger joints] (22 total). Root qpos 0 means "at the agent's pose"; root
joint targets are offsets from that pose. Hand frame (`base_link`): fingers extend along local -z, so the identity
orientation is the stock `palm_side` pose (fingers pointing down, palm normal horizontal).

Camera: `palm_camera` (128x128 RGB-D, rides on the palm link). Tasks add their own fixed workspace camera.

Controllers (`pd_joint_delta_pos` is the default, `pd_joint_pos` takes absolute targets):
    root_pos   xyz offset (m)        root_rot  xyz euler offset (rad)        fingers  16 joint angles (rad)
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.allegro_hand.allegro import AllegroHandRight
from mani_skill.sensors.camera import CameraConfig

ROOT_JOINT_NAMES = [
    "root_x_axis_joint",
    "root_y_axis_joint",
    "root_z_axis_joint",
    "root_x_rot_joint",
    "root_y_rot_joint",
    "root_z_rot_joint",
]
_ROOT_LINKS = ["root"] + [f"root_arm_1_link_{i}" for i in range(1, 6)]
_ROOT_JOINT_SPECS = [  # (name, type, axis)
    (ROOT_JOINT_NAMES[0], "prismatic", "1 0 0"),
    (ROOT_JOINT_NAMES[1], "prismatic", "0 1 0"),
    (ROOT_JOINT_NAMES[2], "prismatic", "0 0 1"),
    (ROOT_JOINT_NAMES[3], "revolute", "1 0 0"),
    (ROOT_JOINT_NAMES[4], "revolute", "0 1 0"),
    (ROOT_JOINT_NAMES[5], "revolute", "0 0 1"),
]

_STOCK_URDF = Path(PACKAGE_ASSET_DIR) / "robots/allegro/allegro_hand_right_glb.urdf"
_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "control-proj" / "floating_allegro"


def _floating_urdf_text(stock: str) -> str:
    """Stock Allegro URDF with the 6-DoF virtual root chain spliced in above `base_link`."""
    zero_inertial = '<inertial><mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>'
    parts = [f'<link name="{name}">{zero_inertial}</link>' for name in _ROOT_LINKS]
    chain = _ROOT_LINKS[1:] + ["base_link"]
    for (name, kind, axis), parent, child in zip(_ROOT_JOINT_SPECS, _ROOT_LINKS, chain):
        parts.append(
            f'<joint name="{name}" type="{kind}"><parent link="{parent}"/><child link="{child}"/>'
            f'<axis xyz="{axis}"/><limit effort="100" lower="-20" upper="20" velocity="0.5"/>'
            f'<dynamics damping="10" friction="0"/></joint>'
        )
    marker = '<link name="base_link">'
    assert stock.count(marker) == 1, "unexpected stock Allegro URDF layout"
    return stock.replace(marker, "\n  ".join(parts) + "\n  " + marker, 1)


def _write_floating_urdf() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meshes = _CACHE_DIR / "meshes"
    if meshes.is_symlink() or meshes.exists():
        meshes.unlink()
    meshes.symlink_to(_STOCK_URDF.parent / "meshes")  # URDF mesh paths are relative to the URDF
    out = _CACHE_DIR / "allegro_hand_right_floating.urdf"
    out.write_text(_floating_urdf_text(_STOCK_URDF.read_text()))
    return out


@register_agent()
class FloatingAllegroHandRight(AllegroHandRight):
    uid = "floating_allegro_hand_right"
    urdf_path = str(_write_floating_urdf())
    fix_root_link = False
    # AllegroHandRight's urdf_config (fingertip friction), tip/palm links and proprioception are inherited.

    keyframes = dict(
        palm_side=Keyframe(qpos=[0.0] * 22, pose=sapien.Pose(p=[0, 0, 0.35])),
    )

    # Palm camera, mounted on the `palm` link (origin 6.5 cm in front of base_link along its -z, fingers along palm -z).
    # Looks down the fingers (camera x -> palm -z, camera up -> palm +x) from 4.5 cm off the palm along +x, which clears
    # the palm geometry whichever face of the hand +x is. Starting guess, not yet viewed in sim: tune the pose first.
    palm_camera_pose = sapien.Pose(p=[0.045, 0.0, 0.0], q=euler2quat(0, np.pi / 2, 0))
    palm_camera_res = 128
    palm_camera_fov = np.pi / 2

    root_pos_range = 0.02  # max |delta| per control step, metres
    root_rot_range = 0.05  # radians
    finger_delta_range = 0.1  # radians (matches the stock Allegro delta controller)

    @property
    def _controller_configs(self):
        root_kw = dict(stiffness=1e3, damping=1e2, force_limit=100)
        finger_kw = dict(stiffness=self.joint_stiffness, damping=self.joint_damping, force_limit=self.joint_force_limit)
        pos, rot = ROOT_JOINT_NAMES[:3], ROOT_JOINT_NAMES[3:]

        abs_cfg = dict(
            root_pos=PDJointPosControllerConfig(pos, None, None, normalize_action=False, **root_kw),
            root_rot=PDJointPosControllerConfig(rot, None, None, normalize_action=False, **root_kw),
            fingers=PDJointPosControllerConfig(self.joint_names, None, None, normalize_action=False, **finger_kw),
        )
        delta_cfg = dict(
            root_pos=PDJointPosControllerConfig(pos, -self.root_pos_range, self.root_pos_range, use_delta=True, use_target=True, **root_kw),
            root_rot=PDJointPosControllerConfig(rot, -self.root_rot_range, self.root_rot_range, use_delta=True, use_target=True, **root_kw),
            fingers=PDJointPosControllerConfig(self.joint_names, -self.finger_delta_range, self.finger_delta_range, use_delta=True, **finger_kw),
        )
        return deepcopy(dict(pd_joint_delta_pos=delta_cfg, pd_joint_pos=abs_cfg))

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                "palm_camera", self.palm_camera_pose, self.palm_camera_res, self.palm_camera_res,
                self.palm_camera_fov, 0.01, 100, mount=self.robot.links_map[self.palm_link_name],
            )
        ]

    @property
    def tcp(self):
        """The palm link, so tasks written for arms (`agent.tcp`) work unchanged."""
        return self.palm_link
