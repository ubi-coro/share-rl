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
)
from share.robots.task_frame_command import require_positive_or_inf


@dataclass
class FrankaControllerConfig(draccus.ChoiceRegistry):
    """Configuration and factory contract for a torque strategy."""

    strategy_name: ClassVar[str] = "base"

    def make_strategy(self, robot_config: "FrankaConfig"):
        raise NotImplementedError


@FrankaControllerConfig.register_subclass("cartesian_reference")
@dataclass
class CartesianReferenceControllerConfig(FrankaControllerConfig):
    """Integrates task-frame targets and streams them into Franky's native
    Cartesian impedance controller -- see control_law.CartesianReferenceController."""

    strategy_name: ClassVar[str] = "cartesian_reference"

    def make_strategy(self, robot_config: "FrankaConfig"):
        from .control_law import CartesianReferenceController

        return CartesianReferenceController(robot_config)


@RobotConfig.register_subclass("franka")
@dataclass
class FrankaConfig(RobotConfig):
    """ROS-free FR3 configuration backed by Franky 2 and libfranka.

    Task-space control wraps Franky's native Cartesian impedance controller
    (position-only -- see lerobot_robot_franka.command.FrankaTaskFrameCommand)
    rather than computing torque in the 500 Hz Python bridge, so the tunable
    gains here are Franky's own: a scalar translational and rotational
    stiffness plus a per-axis force ceiling, not the six independent
    kp/kd/wrench_limits axes UR's RTDE force-mode controller uses.
    """

    robot_ip: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    controller: FrankaControllerConfig = field(default_factory=CartesianReferenceControllerConfig)

    frequency: float = 500.0
    launch_timeout: float = 10.0
    get_max_k: int = 128
    command_timeout_s: float = 0.25
    enforce_realtime: bool = True
    rt_core: int | None = 3
    shm_manager: SharedMemoryManager | None = None

    # Franky's own defaults are 2000 / 200; SHARE starts an order of
    # magnitude softer for first contact with a new robot/cell.
    translational_stiffness: float = 200.0
    rotational_stiffness: float = 20.0
    gains_time_constant_s: float = 0.1

    min_pose_rpy: list[float] = field(default_factory=lambda: [-math.inf] * 6)
    max_pose_rpy: list[float] = field(default_factory=lambda: [math.inf] * 6)
    rotation_interval_modes: list[str] = field(default_factory=lambda: ["linear"] * 6)
    # Per-axis force/torque ceiling passed straight through to Franky's
    # `force_constraints`; math.inf marks an axis unconstrained.
    force_constraints: list[float] = field(
        default_factory=lambda: [30.0, 30.0, 30.0, 3.0, 3.0, 3.0]
    )
    compliance_reference_limit_enable: list[bool] = field(default_factory=lambda: [False] * 6)

    # Redundancy-resolution posture task, projected into the Cartesian
    # nullspace by Franky's CartesianImpedanceTrackingMotion. Fixed for the
    # life of that motion (captured once when task-space control starts),
    # so it lives here rather than as a per-command override.
    nullspace_stiffness: list[float] = field(default_factory=lambda: [20.0] * 7)
    nullspace_max_torque: float = 5.0

    joint_stiffness: list[float] = field(default_factory=lambda: [50.0] * 7)
    joint_damping: list[float] | None = None
    joint_error_clip: list[float] = field(default_factory=lambda: [0.5] * 7)

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
        if float(self.gains_time_constant_s) < 0.0:
            raise ValueError("FrankaConfig.gains_time_constant_s must be non-negative")
        if not math.isfinite(self.translational_stiffness) or self.translational_stiffness < 0.0:
            raise ValueError("FrankaConfig.translational_stiffness must be finite and non-negative")
        if not math.isfinite(self.rotational_stiffness) or self.rotational_stiffness < 0.0:
            raise ValueError("FrankaConfig.rotational_stiffness must be finite and non-negative")
        self._validate_length("min_pose_rpy", self.min_pose_rpy, 6)
        self._validate_length("max_pose_rpy", self.max_pose_rpy, 6)
        self._validate_length("rotation_interval_modes", self.rotation_interval_modes, 6)
        self._validate_length("force_constraints", self.force_constraints, 6)
        self._validate_length(
            "compliance_reference_limit_enable", self.compliance_reference_limit_enable, 6
        )
        self._validate_length("nullspace_stiffness", self.nullspace_stiffness, 7)
        if not math.isfinite(self.nullspace_max_torque) or self.nullspace_max_torque < 0.0:
            raise ValueError("FrankaConfig.nullspace_max_torque must be finite and non-negative")
        self._validate_length("joint_stiffness", self.joint_stiffness, 7)
        self._validate_length("joint_error_clip", self.joint_error_clip, 7)
        if self.joint_damping is None:
            self.joint_damping = [2.0 * math.sqrt(value) for value in self.joint_stiffness]
        self._validate_length("joint_damping", self.joint_damping, 7)
        for name in ("lower_torque_thresholds_nominal", "upper_torque_thresholds_nominal"):
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

        require_positive_or_inf("force_constraints", self.force_constraints)

    @staticmethod
    def _validate_length(name: str, value: object, length: int) -> None:
        if value is None or len(value) != length:
            raise ValueError(f"FrankaConfig.{name} must have length {length}")

    @staticmethod
    def compute_theta(wrench_limit: float, desired_wrench: float, minimum_scale: float) -> float:
        return compute_adaptive_limit_theta(wrench_limit, desired_wrench, minimum_scale)

    @staticmethod
    def exp_scale_and_derivative(force: float, theta: float, minimum_scale: float) -> tuple[float, float]:
        return adaptive_scale_and_derivative(force, theta, minimum_scale)


@RobotConfig.register_subclass("mock_franka")
@dataclass
class MockFrankaConfig(FrankaConfig):
    """Hardware-free Franka configuration."""

    robot_ip: str = "mock"
