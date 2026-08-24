from __future__ import annotations

from dataclasses import asdict
from functools import cached_property
from typing import Any

import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    PolicyMode,
    TASK_FRAME_AXIS_NAMES,
    TaskFrame,
)
from share.robots.task_frame_command import merge_controller_overrides

from .command import FR3_JOINT_NAMES, FrankaTaskFrameCommand
from .config_franka import MockFrankaConfig


class MockFranka(Robot):
    """FR3-compatible mock that does not import Franky."""

    config_class = MockFrankaConfig
    name = "mock_franka"
    joint_names = state_names = FR3_JOINT_NAMES

    def __init__(self, config: MockFrankaConfig):
        super().__init__(config)
        self.config = config
        self.task_frame = FrankaTaskFrameCommand(
            controller_overrides=self._default_controller_overrides()
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_connected = False
        self._active_control_space: ControlSpace | None = None
        self._pose = np.zeros(6, dtype=np.float64)
        self._velocity = np.zeros(6, dtype=np.float64)
        self._wrench = np.zeros(6, dtype=np.float64)
        self._q = np.zeros(7, dtype=np.float64)
        self._dq = np.zeros(7, dtype=np.float64)
        self._gripper_position = 0.0

    @property
    def _motors_ft(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for axis_name in TASK_FRAME_AXIS_NAMES:
            features[f"{axis_name}.ee_pos"] = float
            features[f"{axis_name}.ee_vel"] = float
            features[f"{axis_name}.ee_wrench"] = float
            features[f"{axis_name}.task_frame_origin"] = float
        for joint_name in self.joint_names:
            features[f"{joint_name}.pos"] = float
            features[f"{joint_name}.vel"] = float
        if self.config.use_gripper:
            features["gripper.pos"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return {
            name: (camera.height, camera.width, 3)
            for name, camera in self.config.cameras.items()
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict[str, type]:
        features = {key: float for key in self.task_frame.to_robot_action()}
        if self.config.use_gripper:
            features["gripper.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        for camera in self.cameras.values():
            camera.connect()
        self._is_connected = True

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        for camera in self.cameras.values():
            camera.disconnect()
        self._is_connected = False

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def zero_ft(self) -> None:
        self._wrench.fill(0.0)

    def get_observation(self) -> dict[str, Any]:
        self._require_connected()
        observation: dict[str, Any] = {}
        origin = np.asarray(self.task_frame.origin or [0.0] * 6)
        for axis, axis_name in enumerate(TASK_FRAME_AXIS_NAMES):
            observation[f"{axis_name}.ee_pos"] = float(self._pose[axis])
            observation[f"{axis_name}.ee_vel"] = float(self._velocity[axis])
            observation[f"{axis_name}.ee_wrench"] = float(self._wrench[axis])
            observation[f"{axis_name}.task_frame_origin"] = float(origin[axis])
        for index, joint_name in enumerate(self.joint_names):
            observation[f"{joint_name}.pos"] = float(self._q[index])
            observation[f"{joint_name}.vel"] = float(self._dq[index])
        if self.config.use_gripper:
            observation["gripper.pos"] = self._gripper_position
        for name, camera in self.cameras.items():
            observation[name] = camera.async_read()
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self._require_connected()
        space = self._space_from_action(action)
        if space is not None:
            self._ensure_control_space(space)
            self._ensure_frame_for_space(space)

        if self.task_frame.space == ControlSpace.JOINT:
            for index, joint_name in enumerate(self.joint_names):
                value = action.get(
                    f"{joint_name}.pos", action.get(f"joint_{index + 1}.pos")
                )
                if value is not None:
                    self.task_frame.target[index] = float(value)
                    self._q[index] = float(value)
        else:
            for axis, axis_name in enumerate(TASK_FRAME_AXIS_NAMES):
                for suffix in ("ee_vel", "ee_wrench"):
                    if f"{axis_name}.{suffix}" in action:
                        raise ValueError(
                            "Franka task-space control is position-only "
                            f"(got {axis_name}.{suffix})"
                        )
                key = f"{axis_name}.ee_pos"
                if key in action:
                    value = float(action[key])
                    self.task_frame.target[axis] = value
                    self.task_frame.control_mode[axis] = ControlMode.POS
                    self._pose[axis] = value
        if self.config.use_gripper and "gripper.pos" in action:
            self._gripper_position = float(np.clip(action["gripper.pos"], 0.0, 1.0))
        return dict(action)

    def set_task_frame(self, new_task_frame: FrankaTaskFrameCommand | TaskFrame) -> None:
        command = FrankaTaskFrameCommand(**asdict(new_task_frame))
        self._ensure_control_space(command.space)
        if command.space == ControlSpace.JOINT:
            if len(command.target) != 7:
                raise ValueError("FR3 joint task frames require seven targets")
            command.joint_names = list(command.joint_names or self.joint_names)
        command.controller_overrides = self._merged_controller_overrides(
            command.controller_overrides
        )
        self.task_frame = command

    def _default_controller_overrides(self) -> dict[str, Any]:
        return {
            "translational_stiffness": float(self.config.translational_stiffness),
            "rotational_stiffness": float(self.config.rotational_stiffness),
            "min_pose": list(self.config.min_pose_rpy),
            "max_pose": list(self.config.max_pose_rpy),
            "rotation_interval_modes": list(self.config.rotation_interval_modes),
            "compliance_reference_limit_enable": list(
                self.config.compliance_reference_limit_enable
            ),
            "joint_stiffness": list(self.config.joint_stiffness),
            "joint_damping": list(self.config.joint_damping),
            "joint_error_clip": list(self.config.joint_error_clip),
        }

    def _merged_controller_overrides(
        self, overrides: dict[str, Any] | None
    ) -> dict[str, Any]:
        return merge_controller_overrides(
            self.task_frame.controller_overrides,
            overrides,
            FrankaTaskFrameCommand.SUPPORTED_CONTROLLER_OVERRIDE_KEYS,
            "Franka",
            self._default_controller_overrides,
        )

    def _ensure_control_space(self, space: ControlSpace | int) -> None:
        resolved = ControlSpace(int(space))
        if self._active_control_space is None:
            self._active_control_space = resolved
        elif resolved != self._active_control_space:
            raise ValueError(
                "MockFranka does not support switching control space during a connection"
            )

    def _ensure_frame_for_space(self, space: ControlSpace) -> None:
        if self.task_frame.space == space:
            return
        if space == ControlSpace.JOINT:
            self.task_frame = FrankaTaskFrameCommand(
                target=self._q.tolist(),
                space=ControlSpace.JOINT,
                policy_mode=[PolicyMode.ABSOLUTE] * 7,
                control_mode=[ControlMode.POS] * 7,
                origin=None,
                joint_names=list(self.joint_names),
                controller_overrides=self._default_controller_overrides(),
            )

    @staticmethod
    def _space_from_action(action: dict[str, Any]) -> ControlSpace | None:
        has_task = any(
            key.endswith((".ee_pos", ".ee_vel", ".ee_wrench")) for key in action
        )
        has_joint = any(
            key.endswith(".pos")
            and ".ee_" not in key
            and key != "gripper.pos"
            for key in action
        )
        if has_task and has_joint:
            raise ValueError("MockFranka actions cannot mix task and joint control")
        if has_task:
            return ControlSpace.TASK
        if has_joint:
            return ControlSpace.JOINT
        return None

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
