# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains logic to instantiate a robot, read information from its motors and cameras,
and send orders to its motors.
"""
import logging
import time
from dataclasses import asdict
from functools import cached_property
from multiprocessing.managers import SharedMemoryManager
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import MotorCalibration
from lerobot.processor.hil_processor import GRIPPER_KEY
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TASK_FRAME_AXIS_NAMES, TaskFrame
from share.robots.ur.lerobot_robot_ur.config_ur import URConfig
from share.robots.ur.lerobot_robot_ur.controller import (
    TaskFrameCommand,
    RTDETaskFrameController,
)
from share.robots.ur.lerobot_robot_ur.wrench_monitor import WrenchMonitorProcess

from share.grippers.robotiq_controller import RTDERobotiqController
from share.utils.transformation_utils import euler_xyz_from_rotvec

logger = logging.getLogger(__name__)


class UR(Robot):
    config_class = URConfig
    name = "ur"

    joint_names = state_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    def __init__(self, config: URConfig):
        super().__init__(config)
        self.config = config
        self.task_frame = TaskFrameCommand(controller_overrides=self._default_controller_overrides())

        self.shm = SharedMemoryManager()
        self.shm.start()
        config.shm_manager = self.shm

        self.controller = RTDETaskFrameController(config)
        self.wrench_monitor = None

        if self.config.use_gripper:
            gripper_range = self._gripper_range_from_calibration()
            min_position, max_position = gripper_range if gripper_range is not None else (None, None)
            self.gripper = RTDERobotiqController(
                hostname=config.robot_ip,
                shm_manager=self.shm,
                frequency=config.gripper_frequency,
                soft_real_time=config.gripper_soft_real_time,
                rt_core=config.gripper_rt_core,
                auto_calibrate=not self.is_calibrated,
                min_position=min_position,
                max_position=max_position,
                verbose=config.verbose
            )
        else:
            self.gripper = None

        self.cameras = make_cameras_from_configs(config.cameras)

        # runtime vars
        self.logs = {}
        self.last_robot_action = TaskFrameCommand()
        self._active_control_space: ControlSpace | None = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        ft = {}
        for i, ax in enumerate(TASK_FRAME_AXIS_NAMES):
            ft[f"{ax}.ee_pos"] = float
            ft[f"{ax}.ee_vel"] = float
            ft[f"{ax}.ee_wrench"] = float

        for i, joint_name in enumerate(self.joint_names):
            ft[f"{joint_name}.q_pos"] = float

        if self.gripper is not None:
            ft["gripper.pos"] = float

        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict:
        """
        A dictionary describing the structure and types of the observations produced by the robot.
        Its structure (keys) should match the structure of what is returned by :pymeth:`get_observation`.
        Values for the dict should either be:
            - The type of the value if it's a simple value, e.g. `float` for single proprioceptive value (a joint's position/velocity)
            - A tuple representing the shape if it's an array-type value, e.g. `(height, width, channel)` for images

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        """
        A dictionary describing the structure and types of the actions expected by the robot. Its structure
        (keys) should match the structure of what is passed to :pymeth:`send_action`. Values for the dict
        should be the type of the value if it's a simple value, e.g. `float` for single proprioceptive value
        (a joint's goal position/velocity)

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        ft = {key: float for key in self.task_frame.to_robot_action()}

        if self.gripper is not None:
            ft["gripper.pos"] = float

        return ft

    @property
    def is_connected(self) -> bool:
        """
        Whether the robot is currently connected or not. If `False`, calling :pymeth:`get_observation` or
        :pymeth:`send_action` should raise an error.
        """
        _is_connected = self.controller.is_ready
        if self.gripper is not None:
            _is_connected &= self.gripper.is_ready
        return _is_connected

    def connect(self, calibrate: bool = True):
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.controller.start()
        self.controller.zero_ft()
        if self.config.wrench_monitor_enabled:
            self.wrench_monitor = WrenchMonitorProcess(
                self.controller.wrench_monitor_rb,
                controller_unexpected_exit_event=self.controller.unexpected_exit_event,
                wrench_monitor_refresh_hz=self.config.wrench_monitor_refresh_hz,
                wrench_monitor_history_s=self.config.wrench_monitor_history_s,
                wrench_monitor_sample_hz=self.config.frequency,
            )
            self.wrench_monitor.start()
        if self.gripper is not None:
            self.gripper.auto_calibrate = calibrate and not self.is_calibrated
            self.gripper.start()
            if self.gripper.auto_calibrate:
                self.calibrate()

        for name in self.cameras:
            self.cameras[name].connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Whether the Robotiq gripper range is stored, or no gripper is configured."""
        if not self.config.use_gripper:
            return True
        return self._gripper_range_from_calibration() is not None

    def calibrate(self) -> None:
        """
        Persist the Robotiq open/closed encoder range after gripper auto-calibration.
        """
        if self.gripper is None:
            return None

        state = self.gripper.get_state()
        min_position = int(state["min_position"])
        max_position = int(state["max_position"])
        if min_position >= max_position:
            raise RuntimeError(
                f"Invalid Robotiq calibration range: min={min_position}, max={max_position}"
            )

        self.calibration["gripper"] = MotorCalibration(
            id=0,
            drive_mode=0,
            homing_offset=0,
            range_min=min_position,
            range_max=max_position,
        )
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")
        return None

    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the robot.
        This may include setting motor parameters, control modes, or initial state.
        """
        return None

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        obs_dict = {}
        controller_data = self.controller.get_robot_state()

        origin = controller_data.get("TaskFrameOrigin")
        origin_rpy = None
        if origin is not None:
            origin_rpy = [*origin[:3], *euler_xyz_from_rotvec(origin[3:6])]

        for i, ax in enumerate(TASK_FRAME_AXIS_NAMES):
            obs_dict[f"{ax}.ee_pos"] = controller_data['ActualTCPPose'][i]
            obs_dict[f"{ax}.ee_vel"] = controller_data['ActualTCPSpeed'][i]
            obs_dict[f"{ax}.ee_wrench"] = controller_data['ActualTCPForce'][i]
            if origin_rpy is not None:
                # Origin the pose sample is expressed in (same ring-buffer sample),
                # so pose and frame stay consistent across task-frame switches.
                obs_dict[f"{ax}.task_frame_origin"] = float(origin_rpy[i])

        for i, joint_name in enumerate(self.joint_names):
            obs_dict[f"{joint_name}.pos"] = controller_data['ActualQ'][i]
            obs_dict[f"{joint_name}.vel"] = controller_data['ActualQd'][i]

        if self.gripper is not None:
            obs_dict["gripper.pos"] = float(self.gripper.get_state()["width"]) / 255.0

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Pass task-frame command to the low-level controller.



        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """

        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action_space = self._space_from_action(action)
        if action_space is not None:
            self._ensure_control_space(action_space)
            if self.task_frame.space != action_space:
                self.task_frame.space = action_space
                self.task_frame.origin = None if action_space == ControlSpace.JOINT else [0.0] * 6
                if action_space == ControlSpace.JOINT:
                    self.task_frame.control_mode = [ControlMode.POS] * len(self.task_frame.target)

        if self.task_frame.space == ControlSpace.JOINT:
            for i in range(len(self.joint_names)):
                canonical_key = f"joint_{i + 1}.pos"
                named_key = f"{self.joint_names[i]}.pos"
                if canonical_key in action:
                    self.task_frame.target[i] = action[canonical_key]
                    self.task_frame.control_mode[i] = ControlMode.POS
                elif named_key in action:
                    self.task_frame.target[i] = action[named_key]
                    self.task_frame.control_mode[i] = ControlMode.POS
        else:
            for i, ax in enumerate(TASK_FRAME_AXIS_NAMES):
                if f"{ax}.ee_pos" in action:
                    self.task_frame.target[i] = action[f"{ax}.ee_pos"]
                    self.task_frame.control_mode[i] = ControlMode.POS
                elif f"{ax}.ee_vel" in action:
                    self.task_frame.target[i] = action[f"{ax}.ee_vel"]
                    self.task_frame.control_mode[i] = ControlMode.VEL
                    self.task_frame.policy_mode[i] = PolicyMode.ABSOLUTE
                elif f"{ax}.ee_wrench" in action:
                    self.task_frame.target[i] = action[f"{ax}.ee_wrench"]
                    self.task_frame.control_mode[i] = ControlMode.WRENCH
                    self.task_frame.policy_mode[i] = PolicyMode.ABSOLUTE

        if self.gripper is not None and f"{GRIPPER_KEY}.pos" in action:
            self.send_gripper_action(action[f"{GRIPPER_KEY}.pos"])

        self.controller.send_cmd(self.task_frame)

        return action

    def send_gripper_action(self, gripper_action: float):
        self.gripper.move(gripper_action, vel=self.config.gripper_vel, force=self.config.gripper_force)

    def set_task_frame(self, new_task_frame: TaskFrameCommand | TaskFrame):
        new_task_frame = self._task_frame_command_from_frame(new_task_frame)

        self._ensure_control_space(new_task_frame.space)
        self.task_frame = new_task_frame

    def _task_frame_command_from_frame(self, frame: TaskFrameCommand | TaskFrame) -> TaskFrameCommand:
        if isinstance(frame, TaskFrameCommand):
            command = TaskFrameCommand(**asdict(frame))
        else:
            command = TaskFrameCommand(**asdict(frame))
        command.controller_overrides = self._merged_controller_overrides(command.controller_overrides)
        return command

    def _default_controller_overrides(self) -> dict[str, Any]:
        overrides = {
            "kp": list(self.config.kp),
            "kd": list(self.config.kd),
            "min_pose": list(self.config.min_pose_rpy),
            "max_pose": list(self.config.max_pose_rpy),
            "wrench_limits": list(self.config.wrench_limits),
            "compliance_adaptive_limit_enable": list(self.config.compliance_adaptive_limit_enable),
            "compliance_reference_limit_enable": list(self.config.compliance_reference_limit_enable),
            "compliance_desired_wrench": list(self.config.compliance_desired_wrench),
            "compliance_adaptive_limit_min": list(self.config.compliance_adaptive_limit_min),
        }
        if getattr(self.config, "force_mode_damping", None) is not None:
            overrides["force_mode_damping"] = float(self.config.force_mode_damping)
        return overrides


    def _merged_controller_overrides(self, overrides: dict[str, Any] | None) -> dict[str, Any]:
        unknown = set(overrides or {}) - TaskFrameCommand.SUPPORTED_CONTROLLER_OVERRIDE_KEYS
        if unknown:
            raise ValueError(f"Unsupported UR task-frame controller overrides: {', '.join(sorted(unknown))}")
        merged = dict(self.task_frame.controller_overrides or self._default_controller_overrides())
        if not overrides:
            return merged
        merged.update(overrides)
        return merged

    def _ensure_control_space(self, space: ControlSpace | int) -> ControlSpace:
        """Lock the robot wrapper to its first commanded control space."""
        resolved = ControlSpace(int(space))
        if self._active_control_space is None:
            self._active_control_space = resolved
            return resolved
        if resolved != self._active_control_space:
            raise ValueError(
                "UR robot does not support switching between task-space and joint-space control"
            )
        return resolved

    @staticmethod
    def _space_from_action(action: dict[str, Any]) -> ControlSpace | None:
        """Infer whether an action dict targets joint or task space."""
        has_task_keys = any(
            key.endswith(".ee_pos") or key.endswith(".ee_vel") or key.endswith(".ee_wrench")
            for key in action
        )
        has_joint_keys = any(
            key.endswith(".pos") and ".ee_" not in key and key != f"{GRIPPER_KEY}.pos"
            for key in action
        )

        if has_task_keys and has_joint_keys:
            raise ValueError("UR robot actions cannot mix task-space and joint-space keys")
        if has_task_keys:
            return ControlSpace.TASK
        if has_joint_keys:
            return ControlSpace.JOINT
        return None

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.wrench_monitor is not None:
            self.wrench_monitor.stop()
            self.wrench_monitor = None
        self.controller.stop()
        if self.gripper is not None:
            self.gripper.stop()
        for cam in self.cameras.values():
            cam.disconnect()
        self.shm.shutdown()

        logger.info(f"{self} disconnected.")

    def _gripper_range_from_calibration(self) -> tuple[int, int] | None:
        gripper_calibration = self.calibration.get("gripper")
        if gripper_calibration is None:
            return None
        return int(gripper_calibration.range_min), int(gripper_calibration.range_max)
