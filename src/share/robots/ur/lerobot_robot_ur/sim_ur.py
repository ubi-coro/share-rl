"""Kinematic-only UR simulator: no physics/collision, just enough to preview an env config's
teleop/primitive-graph wiring (and, for multi-arm primitives, run a full closed loop) before
touching real hardware. Not a substitute for testing on the real robot."""

import time
from dataclasses import dataclass, field

import numpy as np
from lerobot.robots import RobotConfig
from lerobot.utils.errors import DeviceNotConnectedError
from scipy.spatial.transform import Rotation

from share.envs.manipulation_primitive.task_frame import PolicyMode, TASK_FRAME_AXIS_NAMES
from share.robots.ur.lerobot_robot_ur.config_mock_ur import MockURConfig
from share.robots.ur.lerobot_robot_ur.mock_ur import MockUR
from share.utils.transformation_utils import rotation_from_extrinsic_xyz


@RobotConfig.register_subclass("sim_ur")
@dataclass
class SimURConfig(MockURConfig):
    robot_ip: str = "sim"
    initial_pose: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.3, 0.0, 0.0, 0.0])
    # first-order settle time for ABSOLUTE POS targets, seconds
    pos_time_constant_s: float = 0.15


class SimUR(MockUR):
    """MockUR with get_observation()/send_action() actually integrating a moving pose."""

    config_class = SimURConfig
    name = "sim_ur"

    def __init__(self, config: SimURConfig):
        super().__init__(config)
        self._position = list(config.initial_pose[:3])
        self._rotation = Rotation.from_euler("xyz", config.initial_pose[3:], degrees=False)
        self._gripper_pos = 0.0
        self._last_step_time: float | None = None

    def get_observation(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        obs = {key: 0.0 for key in self._motors_ft}
        rotvec = self._rotation.as_rotvec()
        for i, axis in enumerate(TASK_FRAME_AXIS_NAMES):
            obs[f"{axis}.ee_pos"] = float((self._position + list(rotvec))[i])
        if self.config.use_gripper:
            obs["gripper.pos"] = self._gripper_pos
        for cam_name, shape in self._cameras_ft.items():
            obs[cam_name] = np.zeros(shape, dtype=np.uint8)
        return obs

    def send_action(self, action):
        now = time.monotonic()
        dt = 0.0 if self._last_step_time is None else now - self._last_step_time
        self._last_step_time = now

        super().send_action(action)  # updates self.task_frame.target/control_mode as usual

        alpha = 1.0 - np.exp(-dt / max(self.config.pos_time_constant_s, 1e-6)) if dt > 0 else 0.0
        for i in range(3):
            key = f"{TASK_FRAME_AXIS_NAMES[i]}.ee_pos"
            if key not in action:
                continue
            if self.task_frame.policy_mode[i] == PolicyMode.RELATIVE:
                self._position[i] += float(action[key]) * dt
            else:
                self._position[i] += alpha * (float(action[key]) - self._position[i])

        rot_keys = [f"{TASK_FRAME_AXIS_NAMES[i]}.ee_pos" for i in range(3, 6)]
        if any(key in action for key in rot_keys):
            drx, dry, drz = (float(action.get(key, 0.0)) for key in rot_keys)
            if self.task_frame.policy_mode[3] == PolicyMode.RELATIVE:
                self._rotation = Rotation.from_rotvec(np.array([drx, dry, drz]) * dt) * self._rotation
            else:
                target = rotation_from_extrinsic_xyz(drx, dry, drz)
                step = (target * self._rotation.inv()).as_rotvec() * alpha
                self._rotation = Rotation.from_rotvec(step) * self._rotation

        return action

    def send_gripper_action(self, gripper_action: float) -> None:
        self._gripper_pos = float(gripper_action)
