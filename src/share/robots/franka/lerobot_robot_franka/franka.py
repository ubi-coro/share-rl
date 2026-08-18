from __future__ import annotations

import logging
from dataclasses import asdict
from functools import cached_property
from multiprocessing.managers import SharedMemoryManager
from typing import Any

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

from .command import FR3_JOINT_NAMES, FrankaTaskFrameCommand
from .config_franka import FrankaConfig
from .controller import FrankaControllerProcess
from .hand import FrankaHandWorker


logger = logging.getLogger(__name__)


class Franka(Robot):
    """LeRobot-compatible FR3 using a ROS-free Franky torque bridge."""

    config_class = FrankaConfig
    name = "franka"
    joint_names = state_names = FR3_JOINT_NAMES

    def __init__(self, config: FrankaConfig):
        super().__init__(config)
        self.config = config
        self.task_frame = FrankaTaskFrameCommand(
            controller_overrides=self._default_controller_overrides()
        )
        self.shm = SharedMemoryManager()
        self.shm.start()
        self.config.shm_manager = self.shm
        self.controller = FrankaControllerProcess(config)
        self.hand = FrankaHandWorker(config) if config.use_gripper else None
        self.cameras = make_cameras_from_configs(config.cameras)
        self._active_control_space: ControlSpace | None = None
        self._started = False
        self.last_robot_action: dict[str, Any] = {}
        self.logs: dict[str, Any] = {}

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
        if self.hand is not None:
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
        if self.hand is not None:
            features["gripper.pos"] = float
        return features

    @property
    def is_connected(self) -> bool:
        connected = self.controller.is_ready
        if self.hand is not None:
            connected = connected and self.hand.is_ready
        return connected

    @property
    def is_calibrated(self) -> bool:
        return self.hand is None or self.hand.is_ready

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self._started = True
        try:
            self.controller.start()
            self.controller.zero_ft()
            if self.hand is not None:
                self.hand.start(home=calibrate)
            for camera in self.cameras.values():
                camera.connect()
        except BaseException:
            self._disconnect_components()
            raise
        logger.info("%s connected", self)

    def disconnect(self) -> None:
        if not self._started and not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        self._disconnect_components()
        logger.info("%s disconnected", self)

    def _disconnect_components(self) -> None:
        self.controller.stop()
        if self.hand is not None:
            self.hand.stop()
        for camera in self.cameras.values():
            try:
                camera.disconnect()
            except BaseException:
                pass
        if self._started:
            self.shm.shutdown()
        self._started = False

    def calibrate(self) -> None:
        if self.hand is None:
            return
        if not self.hand.is_ready:
            raise DeviceNotConnectedError("The Franka Hand is not connected")
        self.hand.home()

    def configure(self) -> None:
        return None

    def zero_ft(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
        self.controller.zero_ft()

    def get_observation(self) -> dict[str, Any]:
        self._require_connected()
        data = self.controller.get_robot_state()
        observation: dict[str, Any] = {}
        for axis, axis_name in enumerate(TASK_FRAME_AXIS_NAMES):
            observation[f"{axis_name}.ee_pos"] = float(data["ActualTCPPose"][axis])
            observation[f"{axis_name}.ee_vel"] = float(data["ActualTCPSpeed"][axis])
            observation[f"{axis_name}.ee_wrench"] = float(data["ActualTCPForce"][axis])
            observation[f"{axis_name}.task_frame_origin"] = float(
                data["TaskFrameOrigin"][axis]
            )
        for index, joint_name in enumerate(self.joint_names):
            observation[f"{joint_name}.pos"] = float(data["ActualQ"][index])
            observation[f"{joint_name}.vel"] = float(data["ActualQd"][index])
        if self.hand is not None:
            observation["gripper.pos"] = float(self.hand.get_state()["position"])
        for name, camera in self.cameras.items():
            observation[name] = camera.async_read()
        self.logs["controller_loop_duration_s"] = float(data["loop_duration_s"])
        self.logs["controller_deadline_missed"] = bool(data["deadline_missed"])
        self.logs["fci_control_command_success_rate"] = float(
            data["control_command_success_rate"]
        )
        return observation

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self._require_connected()
        action_space = self._space_from_action(action)
        if action_space is not None:
            self._ensure_control_space(action_space)
            self._ensure_frame_for_space(action_space)

        if self.task_frame.space == ControlSpace.JOINT:
            for index, joint_name in enumerate(self.joint_names):
                key = f"{joint_name}.pos"
                canonical_key = f"joint_{index + 1}.pos"
                if key in action:
                    self.task_frame.target[index] = float(action[key])
                elif canonical_key in action:
                    self.task_frame.target[index] = float(action[canonical_key])
        else:
            for axis, axis_name in enumerate(TASK_FRAME_AXIS_NAMES):
                for suffix, mode in (
                    ("ee_pos", ControlMode.POS),
                    ("ee_vel", ControlMode.VEL),
                    ("ee_wrench", ControlMode.WRENCH),
                ):
                    key = f"{axis_name}.{suffix}"
                    if key not in action:
                        continue
                    self.task_frame.target[axis] = float(action[key])
                    self.task_frame.control_mode[axis] = mode
                    if mode is not ControlMode.POS:
                        self.task_frame.policy_mode[axis] = PolicyMode.ABSOLUTE
                    break

        if self.hand is not None and "gripper.pos" in action:
            self.hand.move(float(action["gripper.pos"]))

        self.controller.send_cmd(self.task_frame)
        self.last_robot_action = dict(action)
        return dict(action)

    def set_task_frame(self, new_task_frame: FrankaTaskFrameCommand | TaskFrame) -> None:
        command = self._task_frame_command_from_frame(new_task_frame)
        self._ensure_control_space(command.space)
        self.task_frame = command
        if self.is_connected:
            self.controller.send_cmd(command)

    def _task_frame_command_from_frame(
        self, frame: FrankaTaskFrameCommand | TaskFrame
    ) -> FrankaTaskFrameCommand:
        command = FrankaTaskFrameCommand(**asdict(frame))
        if command.space == ControlSpace.JOINT:
            if len(command.target) != 7:
                raise ValueError("FR3 joint task frames require seven targets")
            command.joint_names = list(command.joint_names or self.joint_names)
        command.controller_overrides = self._merged_controller_overrides(
            command.controller_overrides
        )
        return command

    def _default_controller_overrides(self) -> dict[str, Any]:
        return {
            "kp": list(self.config.kp),
            "kd": list(self.config.kd),
            "min_pose": list(self.config.min_pose_rpy),
            "max_pose": list(self.config.max_pose_rpy),
            "rotation_interval_modes": list(self.config.rotation_interval_modes),
            "wrench_limits": list(self.config.wrench_limits),
            "compliance_reference_limit_enable": list(
                self.config.compliance_reference_limit_enable
            ),
            "compliance_adaptive_limit_enable": list(
                self.config.compliance_adaptive_limit_enable
            ),
            "compliance_desired_wrench": list(
                self.config.compliance_desired_wrench
            ),
            "compliance_adaptive_limit_min": list(
                self.config.compliance_adaptive_limit_min
            ),
            "nullspace_stiffness": list(self.config.nullspace_stiffness),
            "nullspace_damping": list(self.config.nullspace_damping),
            "nullspace_max_torque": float(self.config.nullspace_max_torque),
            "joint_stiffness": list(self.config.joint_stiffness),
            "joint_damping": list(self.config.joint_damping),
            "joint_error_clip": list(self.config.joint_error_clip),
        }

    def _merged_controller_overrides(
        self, overrides: dict[str, Any] | None
    ) -> dict[str, Any]:
        unknown = set(overrides or {}) - FrankaTaskFrameCommand.SUPPORTED_CONTROLLER_OVERRIDE_KEYS
        if unknown:
            raise ValueError(
                "Unsupported Franka controller overrides: "
                + ", ".join(sorted(unknown))
            )
        merged = dict(
            self.task_frame.controller_overrides
            or self._default_controller_overrides()
        )
        if overrides:
            merged.update(overrides)
        return merged

    def _ensure_control_space(self, space: ControlSpace | int) -> ControlSpace:
        resolved = ControlSpace(int(space))
        if self._active_control_space is None:
            self._active_control_space = resolved
        elif resolved != self._active_control_space:
            raise ValueError(
                "Franka does not support switching control space during a connection"
            )
        return resolved

    def _ensure_frame_for_space(self, space: ControlSpace) -> None:
        if self.task_frame.space == space:
            return
        if space == ControlSpace.JOINT:
            state = self.controller.get_robot_state()
            initial_target = [float(value) for value in state["ActualQ"]]

            self.task_frame = FrankaTaskFrameCommand(
                target=initial_target,
                space=ControlSpace.JOINT,
                policy_mode=[PolicyMode.ABSOLUTE] * 7,
                control_mode=[ControlMode.POS] * 7,
                origin=None,
                joint_names=list(self.joint_names),
                controller_overrides=self._default_controller_overrides(),
            )
        else:
            self.task_frame = FrankaTaskFrameCommand(
                controller_overrides=self._default_controller_overrides()
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
            raise ValueError("Franka actions cannot mix task and joint control")
        if has_task:
            return ControlSpace.TASK
        if has_joint:
            return ControlSpace.JOINT
        return None

    def _require_connected(self) -> None:
        self.controller.check_health()
        if self.hand is not None:
            self.hand.check_health()
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")
