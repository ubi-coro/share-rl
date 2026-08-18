from __future__ import annotations

import math
from dataclasses import dataclass, field
from multiprocessing.managers import SharedMemoryManager
from typing import ClassVar

import draccus
from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

from share.robots.adaptive_limits import (
    adaptive_scale_and_derivative,
    compute_adaptive_limit_theta,
    validate_adaptive_fixed_point,
)


@dataclass
class FrankaControllerConfig(draccus.ChoiceRegistry):
    """Configuration and factory contract for a torque strategy."""

    strategy_name: ClassVar[str] = "base"

    def make_strategy(self, robot_config: "FrankaConfig"):
        raise NotImplementedError


@FrankaControllerConfig.register_subclass("adaptive")
@dataclass
class AdaptiveFrankaControllerConfig(FrankaControllerConfig):
    """SHARE mixed-axis Cartesian impedance controller."""

    strategy_name: ClassVar[str] = "adaptive"

    def make_strategy(self, robot_config: "FrankaConfig"):
        from .control_law import AdaptiveTaskFrameController

        return AdaptiveTaskFrameController(robot_config)


@RobotConfig.register_subclass("franka")
@dataclass
class FrankaConfig(RobotConfig):
    """ROS-free FR3 configuration backed by Franky 2 and libfranka."""

    robot_ip: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    controller: FrankaControllerConfig = field(default_factory=AdaptiveFrankaControllerConfig)

    frequency: float = 500.0
    launch_timeout: float = 10.0
    get_max_k: int = 128
    command_timeout_s: float = 0.25
    torque_signal_timeout_s: float = 0.05
    enforce_realtime: bool = True
    rt_core: int | None = 3
    shm_manager: SharedMemoryManager | None = None

    kp: list[float] = field(
        default_factory=lambda: [500.0, 500.0, 500.0, 50.0, 50.0, 50.0]
    )
    kd: list[float] = field(
        default_factory=lambda: [2.0 * math.sqrt(500.0)] * 3 + [2.0 * math.sqrt(50.0)] * 3
    )
    gains_time_constant_s: float = 0.1

    min_pose_rpy: list[float] = field(default_factory=lambda: [-math.inf] * 6)
    max_pose_rpy: list[float] = field(default_factory=lambda: [math.inf] * 6)
    rotation_interval_modes: list[str] = field(default_factory=lambda: ["linear"] * 6)
    wrench_limits: list[float] = field(
        default_factory=lambda: [30.0, 30.0, 30.0, 3.0, 3.0, 3.0]
    )
    compliance_adaptive_limit_enable: list[bool] = field(default_factory=lambda: [False] * 6)
    compliance_reference_limit_enable: list[bool] = field(default_factory=lambda: [False] * 6)
    compliance_desired_wrench: list[float] = field(
        default_factory=lambda: [5.0, 5.0, 5.0, 0.5, 0.5, 0.5]
    )
    compliance_adaptive_limit_min: list[float] = field(default_factory=lambda: [0.1] * 6)
    compliance_adaptive_limit_theta: list[float] | None = None

    nullspace_stiffness: list[float] = field(default_factory=lambda: [20.0] * 7)
    nullspace_damping: list[float] | None = None
    nullspace_max_torque: float = 5.0

    joint_stiffness: list[float] = field(default_factory=lambda: [50.0] * 7)
    joint_damping: list[float] | None = None
    joint_error_clip: list[float] = field(default_factory=lambda: [0.5] * 7)

    max_delta_tau: float = 1.0
    joint_limit_margin: float = 0.15
    joint_limit_potential: float = 20.0
    joint_limit_max_torque: float = 10.0

    payload_mass: float | None = None
    payload_center_of_mass: list[float] | None = None
    payload_inertia: list[float] | None = None
    end_effector_transform: list[float] | None = None

    use_gripper: bool = False
    gripper_speed: float = 0.05
    gripper_force: float = 20.0
    gripper_epsilon_inner: float = 0.005
    gripper_epsilon_outer: float = 0.005
    gripper_frequency: float = 20.0

    lower_torque_thresholds_nominal: list[float] = field(default_factory=lambda: [20.0] * 7)
    upper_torque_thresholds_nominal: list[float] = field(default_factory=lambda: [20.0] * 7)
    lower_force_thresholds_nominal: list[float] = field(default_factory=lambda: [30.0] * 6)
    upper_force_thresholds_nominal: list[float] = field(default_factory=lambda: [30.0] * 6)

    verbose: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < float(self.frequency) <= 500.0:
            raise ValueError("FrankaConfig.frequency must be in (0, 500]")
        if float(self.command_timeout_s) <= 0.0:
            raise ValueError("FrankaConfig.command_timeout_s must be positive")
        if float(self.torque_signal_timeout_s) <= 0.0:
            raise ValueError("FrankaConfig.torque_signal_timeout_s must be positive")
        if float(self.gains_time_constant_s) < 0.0:
            raise ValueError("FrankaConfig.gains_time_constant_s must be non-negative")
        self._validate_length("kp", self.kp, 6)
        self._validate_length("kd", self.kd, 6)
        self._validate_length("min_pose_rpy", self.min_pose_rpy, 6)
        self._validate_length("max_pose_rpy", self.max_pose_rpy, 6)
        self._validate_length("rotation_interval_modes", self.rotation_interval_modes, 6)
        self._validate_length("wrench_limits", self.wrench_limits, 6)
        self._validate_length(
            "compliance_adaptive_limit_enable", self.compliance_adaptive_limit_enable, 6
        )
        self._validate_length(
            "compliance_reference_limit_enable", self.compliance_reference_limit_enable, 6
        )
        self._validate_length("compliance_desired_wrench", self.compliance_desired_wrench, 6)
        self._validate_length(
            "compliance_adaptive_limit_min", self.compliance_adaptive_limit_min, 6
        )
        self._validate_length("nullspace_stiffness", self.nullspace_stiffness, 7)
        self._validate_length("joint_stiffness", self.joint_stiffness, 7)
        self._validate_length("joint_error_clip", self.joint_error_clip, 7)
        if self.nullspace_damping is None:
            self.nullspace_damping = [2.0 * math.sqrt(value) for value in self.nullspace_stiffness]
        if self.joint_damping is None:
            self.joint_damping = [2.0 * math.sqrt(value) for value in self.joint_stiffness]
        self._validate_length("nullspace_damping", self.nullspace_damping, 7)
        self._validate_length("joint_damping", self.joint_damping, 7)
        for name in (
            "lower_torque_thresholds_nominal",
            "upper_torque_thresholds_nominal",
        ):
            self._validate_length(name, getattr(self, name), 7)
        for name in ("lower_force_thresholds_nominal", "upper_force_thresholds_nominal"):
            self._validate_length(name, getattr(self, name), 6)
        if self.payload_center_of_mass is not None:
            self._validate_length("payload_center_of_mass", self.payload_center_of_mass, 3)
        if self.payload_inertia is not None:
            self._validate_length("payload_inertia", self.payload_inertia, 9)
        if self.end_effector_transform is not None:
            self._validate_length("end_effector_transform", self.end_effector_transform, 16)

        for mode in self.rotation_interval_modes:
            if mode not in {"linear", "ccw_arc"}:
                raise ValueError("rotation_interval_modes entries must be 'linear' or 'ccw_arc'")

        if self.compliance_adaptive_limit_theta is None:
            self.compliance_adaptive_limit_theta = [1.0] * 6
            for axis, enabled in enumerate(self.compliance_adaptive_limit_enable):
                if not enabled:
                    continue
                if math.isinf(self.wrench_limits[axis]):
                    self.wrench_limits[axis] = 2.0 * self.compliance_desired_wrench[axis]
                theta = compute_adaptive_limit_theta(
                    self.wrench_limits[axis],
                    self.compliance_desired_wrench[axis],
                    self.compliance_adaptive_limit_min[axis],
                )
                validate_adaptive_fixed_point(
                    self.wrench_limits[axis],
                    self.compliance_desired_wrench[axis],
                    self.compliance_adaptive_limit_min[axis],
                    theta,
                )
                self.compliance_adaptive_limit_theta[axis] = theta
        self._validate_length(
            "compliance_adaptive_limit_theta", self.compliance_adaptive_limit_theta, 6
        )

    @staticmethod
    def _validate_length(name: str, value: object, length: int) -> None:
        if value is None or len(value) != length:
            raise ValueError(f"FrankaConfig.{name} must have length {length}")

    @staticmethod
    def compute_theta(wrench_limit: float, desired_wrench: float, minimum_scale: float) -> float:
        return compute_adaptive_limit_theta(wrench_limit, desired_wrench, minimum_scale)

    @staticmethod
    def exp_scale_and_derivative(
        force: float, theta: float, minimum_scale: float
    ) -> tuple[float, float]:
        return adaptive_scale_and_derivative(force, theta, minimum_scale)


@RobotConfig.register_subclass("mock_franka")
@dataclass
class MockFrankaConfig(FrankaConfig):
    """Hardware-free Franka configuration."""

    robot_ip: str = "mock"
