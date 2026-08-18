from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode
from share.robots.adaptive_limits import (
    adaptive_wrench_scales,
    compute_adaptive_limit_theta,
    reference_error_limit,
)
from share.utils.transformation_utils import (
    RotationIntervalMode,
    signed_error_to_nearest_arc_endpoint,
    wrap_to_pi,
)


@dataclass(slots=True)
class FrankaState:
    """Hardware-neutral state consumed by torque strategies."""

    q: np.ndarray
    dq: np.ndarray
    T_base_ee: np.ndarray
    T_ee_stiffness: np.ndarray
    twist_base_ee: np.ndarray
    wrench_base_at_stiffness: np.ndarray
    jacobian_base_ee: np.ndarray
    timestamp: float


@dataclass(slots=True)
class ControllerOutput:
    torque: np.ndarray
    pose_task_rpy: np.ndarray
    twist_task: np.ndarray
    measured_wrench_task: np.ndarray
    desired_wrench_task: np.ndarray
    adaptive_scale: np.ndarray
    holding: bool


class FrankaControllerStrategy(Protocol):
    """Stable strategy surface for Python and future pybind controllers."""

    def step(
        self,
        state: FrankaState,
        command: dict[str, Any] | None,
        model: Any,
        dt: float,
    ) -> np.ndarray | ControllerOutput:
        ...

    def zero_wrench(self, state: FrankaState) -> None:
        ...


def pose_rpy_to_transform(pose: np.ndarray | list[float]) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError("pose must have shape (6,)")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def transform_to_pose_rpy(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    return np.concatenate(
        (
            transform[:3, 3],
            Rotation.from_matrix(transform[:3, :3]).as_euler("xyz"),
        )
    )


def task_pose(T_base_ee: np.ndarray, T_base_task: np.ndarray) -> np.ndarray:
    return transform_to_pose_rpy(np.linalg.inv(T_base_task) @ T_base_ee)


def rotate_twist(twist: np.ndarray, rotation_target_source: np.ndarray) -> np.ndarray:
    """Rotate a twist between coincident origins without a point shift."""
    twist = np.asarray(twist, dtype=np.float64)
    rotation = np.asarray(rotation_target_source, dtype=np.float64)
    return np.concatenate((rotation @ twist[:3], rotation @ twist[3:]))


def transform_wrench(
    wrench_source: np.ndarray,
    rotation_target_source: np.ndarray,
    source_origin_in_target: np.ndarray,
) -> np.ndarray:
    """Change wrench axes and reference point.

    source_origin_in_target is the source reference point minus the target
    reference point, expressed in target axes.
    """
    wrench = np.asarray(wrench_source, dtype=np.float64)
    rotation = np.asarray(rotation_target_source, dtype=np.float64)
    offset = np.asarray(source_origin_in_target, dtype=np.float64)
    force = rotation @ wrench[:3]
    moment = rotation @ wrench[3:] + np.cross(offset, force)
    return np.concatenate((force, moment))


def measured_wrench_in_task(
    wrench_base_at_stiffness: np.ndarray,
    T_base_ee: np.ndarray,
    T_ee_stiffness: np.ndarray,
    T_base_task: np.ndarray,
) -> np.ndarray:
    T_base_stiffness = T_base_ee @ T_ee_stiffness
    rotation_task_base = T_base_task[:3, :3].T
    offset = rotation_task_base @ (
        T_base_stiffness[:3, 3] - T_base_task[:3, 3]
    )
    return transform_wrench(wrench_base_at_stiffness, rotation_task_base, offset)


def task_wrench_at_ee_in_base(
    wrench_task_at_task: np.ndarray,
    T_base_task: np.ndarray,
    T_base_ee: np.ndarray,
) -> np.ndarray:
    rotation_base_task = T_base_task[:3, :3]
    offset = T_base_task[:3, 3] - T_base_ee[:3, 3]
    return transform_wrench(wrench_task_at_task, rotation_base_task, offset)


def so3_error(desired: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Logarithmic orientation error, expressed in the common parent frame."""
    return Rotation.from_matrix(desired @ actual.T).as_rotvec()


def nullspace_torque(
    jacobian: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    q_reference: np.ndarray,
    stiffness: np.ndarray,
    damping: np.ndarray,
    max_torque: float,
) -> np.ndarray:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    projector = np.eye(7) - jacobian.T @ np.linalg.pinv(jacobian.T)
    posture = stiffness * (q_reference - q) - damping * dq
    return np.clip(projector @ posture, -max_torque, max_torque)


def smooth_values(
    current: np.ndarray,
    target: np.ndarray,
    dt: float,
    time_constant: float,
) -> np.ndarray:
    if time_constant <= 0.0:
        return np.array(target, dtype=np.float64, copy=True)
    alpha = 1.0 - math.exp(-max(float(dt), 0.0) / time_constant)
    return current + alpha * (target - current)


def apply_workspace_and_contact_limits(
    pose_rpy: np.ndarray,
    desired_wrench: np.ndarray,
    measured_wrench: np.ndarray,
    stiffness: np.ndarray,
    min_pose: np.ndarray,
    max_pose: np.ndarray,
    rotation_interval_modes: np.ndarray,
    wrench_limits: np.ndarray,
    adaptive_enable: np.ndarray,
    adaptive_minimum: np.ndarray,
    adaptive_theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bound nominal wrench, then add an uncapped inward workspace spring."""
    bounded = np.array(desired_wrench, dtype=np.float64, copy=True)
    scales = adaptive_wrench_scales(
        bounded,
        measured_wrench,
        adaptive_enable,
        adaptive_minimum,
        adaptive_theta,
    )
    scaled_limits = scales * wrench_limits
    bounded = np.clip(bounded, -scaled_limits, scaled_limits)

    for axis in range(3):
        correction = 0.0
        if pose_rpy[axis] > max_pose[axis]:
            correction = max_pose[axis] - pose_rpy[axis]
            if bounded[axis] > 0.0:
                bounded[axis] = 0.0
        elif pose_rpy[axis] < min_pose[axis]:
            correction = min_pose[axis] - pose_rpy[axis]
            if bounded[axis] < 0.0:
                bounded[axis] = 0.0
        bounded[axis] += stiffness[axis] * correction

    wrapped_rpy = np.asarray(wrap_to_pi(pose_rpy[3:]), dtype=np.float64)
    for local_axis, axis in enumerate(range(3, 6)):
        if not np.isfinite(min_pose[axis]) and not np.isfinite(max_pose[axis]):
            continue
        correction = 0.0
        mode = RotationIntervalMode(int(rotation_interval_modes[axis]))
        if mode is RotationIntervalMode.CCW_ARC:
            correction = signed_error_to_nearest_arc_endpoint(
                wrapped_rpy[local_axis],
                min_pose[axis],
                max_pose[axis],
            )
        elif wrapped_rpy[local_axis] > max_pose[axis]:
            correction = max_pose[axis] - wrapped_rpy[local_axis]
        elif wrapped_rpy[local_axis] < min_pose[axis]:
            correction = min_pose[axis] - wrapped_rpy[local_axis]

        if correction > 0.0 and bounded[axis] < 0.0:
            bounded[axis] = 0.0
        elif correction < 0.0 and bounded[axis] > 0.0:
            bounded[axis] = 0.0
        bounded[axis] += stiffness[axis] * correction

    return bounded, scales, scaled_limits


class AdaptiveTaskFrameController:
    """Stateful mixed-axis impedance law used by the 500 Hz bridge."""

    HOLD_KP = np.array([200.0, 200.0, 200.0, 20.0, 20.0, 20.0])
    HOLD_KD = 2.0 * np.sqrt(HOLD_KP)

    def __init__(self, config: Any):
        self.config = config
        self.T_base_task = np.eye(4, dtype=np.float64)
        self.virtual_position = np.zeros(3, dtype=np.float64)
        self.virtual_rotation = np.eye(3, dtype=np.float64)
        self.wrench_bias_base = np.zeros(6, dtype=np.float64)
        self.q_reference: np.ndarray | None = None
        self._sequence = -1
        self._holding = False
        self._initialized = False
        self._control_mode = np.full(6, int(ControlMode.POS), dtype=np.int8)
        self._policy_mode = np.full(6, int(PolicyMode.RELATIVE), dtype=np.int8)
        self._kp = np.asarray(config.kp, dtype=np.float64)
        self._kd = np.asarray(config.kd, dtype=np.float64)
        self._target_kp = self._kp.copy()
        self._target_kd = self._kd.copy()
        self._nullspace_kp = np.asarray(config.nullspace_stiffness, dtype=np.float64)
        self._nullspace_kd = np.asarray(config.nullspace_damping, dtype=np.float64)
        self._target_nullspace_kp = self._nullspace_kp.copy()
        self._target_nullspace_kd = self._nullspace_kd.copy()
        self._command: dict[str, Any] | None = None
        self._adaptive_theta_values = np.ones(6, dtype=np.float64)

    def zero_wrench(self, state: FrankaState) -> None:
        self.wrench_bias_base = np.asarray(
            state.wrench_base_at_stiffness, dtype=np.float64
        ).copy()

    def step(
        self,
        state: FrankaState,
        command: dict[str, Any] | None,
        model: Any,
        dt: float,
    ) -> ControllerOutput:
        del model
        if self.q_reference is None:
            self.q_reference = np.asarray(state.q, dtype=np.float64).copy()

        if command is None:
            self._enter_hold(state)
        elif self._holding or int(command["sequence"]) != self._sequence:
            self._accept_command(command, state)

        pose, rotation_task_ee = self._pose_components(state)
        rotation_task_base = self.T_base_task[:3, :3].T
        twist = rotate_twist(state.twist_base_ee, rotation_task_base)
        measured_wrench = measured_wrench_in_task(
            state.wrench_base_at_stiffness - self.wrench_bias_base,
            state.T_base_ee,
            state.T_ee_stiffness,
            self.T_base_task,
        )

        self._kp = smooth_values(
            self._kp, self._target_kp, dt, self.config.gains_time_constant_s
        )
        self._kd = smooth_values(
            self._kd, self._target_kd, dt, self.config.gains_time_constant_s
        )
        self._nullspace_kp = smooth_values(
            self._nullspace_kp,
            self._target_nullspace_kp,
            dt,
            self.config.gains_time_constant_s,
        )
        self._nullspace_kd = smooth_values(
            self._nullspace_kd,
            self._target_nullspace_kd,
            dt,
            self.config.gains_time_constant_s,
        )

        desired_wrench = self._nominal_wrench(
            pose,
            rotation_task_ee,
            twist,
            max(float(dt), 0.0),
        )
        command_data = self._command
        assert command_data is not None
        theta = self._adaptive_theta_values
        bounded_wrench, scales, _ = apply_workspace_and_contact_limits(
            pose,
            desired_wrench,
            measured_wrench,
            self._kp,
            command_data["min_pose"],
            command_data["max_pose"],
            command_data["rotation_interval_modes"],
            command_data["wrench_limits"],
            command_data["compliance_adaptive_limit_enable"],
            command_data["compliance_adaptive_limit_min"],
            theta,
        )

        wrench_base_at_ee = task_wrench_at_ee_in_base(
            bounded_wrench, self.T_base_task, state.T_base_ee
        )
        task_torque = state.jacobian_base_ee.T @ wrench_base_at_ee
        posture_torque = nullspace_torque(
            state.jacobian_base_ee,
            state.q,
            state.dq,
            self.q_reference,
            self._nullspace_kp,
            self._nullspace_kd,
            float(command_data["nullspace_max_torque"]),
        )
        return ControllerOutput(
            torque=task_torque + posture_torque,
            pose_task_rpy=pose,
            twist_task=twist,
            measured_wrench_task=measured_wrench,
            desired_wrench_task=bounded_wrench,
            adaptive_scale=scales,
            holding=self._holding,
        )

    def _pose_components(self, state: FrankaState) -> tuple[np.ndarray, np.ndarray]:
        T_task_ee = np.linalg.inv(self.T_base_task) @ state.T_base_ee
        pose = transform_to_pose_rpy(T_task_ee)
        return pose, T_task_ee[:3, :3]

    def _accept_command(self, command: dict[str, Any], state: FrankaState) -> None:
        command = self._copy_command(command)
        new_origin = pose_rpy_to_transform(command["origin"])
        old_control_mode = self._control_mode.copy()
        old_policy_mode = self._policy_mode.copy()

        if self._initialized and not np.allclose(new_origin, self.T_base_task):
            T_base_virtual = self.T_base_task @ self._virtual_transform()
            T_task_virtual = np.linalg.inv(new_origin) @ T_base_virtual
            self.virtual_position = T_task_virtual[:3, 3].copy()
            self.virtual_rotation = T_task_virtual[:3, :3].copy()

        self.T_base_task = new_origin
        pose, rotation_task_ee = self._pose_components(state)
        if not self._initialized:
            self.virtual_position = pose[:3].copy()
            self.virtual_rotation = rotation_task_ee.copy()
            self._initialized = True
        new_control_mode = command["control_mode"][:6]
        new_policy_mode = command["policy_mode"][:6]

        reanchor_rotation = False
        for axis in range(6):
            enters_relative_position = (
                new_control_mode[axis] == int(ControlMode.POS)
                and new_policy_mode[axis] == int(PolicyMode.RELATIVE)
                and (
                    self._holding
                    or old_control_mode[axis] != int(ControlMode.POS)
                    or old_policy_mode[axis] != int(PolicyMode.RELATIVE)
                )
            )
            if not enters_relative_position:
                continue
            if axis < 3:
                self.virtual_position[axis] = pose[axis]
            else:
                reanchor_rotation = True

        if reanchor_rotation:
            virtual_rpy = Rotation.from_matrix(self.virtual_rotation).as_euler("xyz")
            measured_rpy = Rotation.from_matrix(rotation_task_ee).as_euler("xyz")
            for axis in range(3, 6):
                if (
                    new_control_mode[axis] == int(ControlMode.POS)
                    and new_policy_mode[axis] == int(PolicyMode.RELATIVE)
                ):
                    virtual_rpy[axis - 3] = measured_rpy[axis - 3]
            self.virtual_rotation = Rotation.from_euler("xyz", virtual_rpy).as_matrix()

        self._control_mode = new_control_mode.copy()
        self._policy_mode = new_policy_mode.copy()
        self._target_kp = command["kp"].copy()
        self._target_kd = command["kd"].copy()
        self._target_nullspace_kp = command["nullspace_stiffness"].copy()
        self._target_nullspace_kd = command["nullspace_damping"].copy()
        self._command = command
        self._sequence = int(command["sequence"])
        self._adaptive_theta_values = self._compute_adaptive_theta(command)
        self._holding = False

    def _enter_hold(self, state: FrankaState) -> None:
        if self._holding:
            return
        pose, rotation_task_ee = self._pose_components(state)
        if self._command is None:
            self._command = self._default_hold_command()
        else:
            self._command = self._copy_command(self._command)
        self.virtual_position = pose[:3].copy()
        self.virtual_rotation = rotation_task_ee.copy()
        self._control_mode = np.full(6, int(ControlMode.POS), dtype=np.int8)
        self._policy_mode = np.full(6, int(PolicyMode.ABSOLUTE), dtype=np.int8)
        self._command["target"][:6] = pose
        self._command["control_mode"][:6] = self._control_mode
        self._command["policy_mode"][:6] = self._policy_mode
        self._command["compliance_reference_limit_enable"][:] = False
        self._command["compliance_adaptive_limit_enable"][:] = False
        self._adaptive_theta_values.fill(1.0)
        self._target_kp = self.HOLD_KP.copy()
        self._target_kd = self.HOLD_KD.copy()
        self._holding = True
        self._initialized = True

    def _nominal_wrench(
        self,
        pose: np.ndarray,
        rotation_task_ee: np.ndarray,
        twist: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        command = self._command
        assert command is not None
        target = command["target"]

        for axis in range(3):
            if self._control_mode[axis] != int(ControlMode.POS):
                continue
            if self._policy_mode[axis] == int(PolicyMode.RELATIVE):
                self.virtual_position[axis] += target[axis] * dt
            else:
                self.virtual_position[axis] = target[axis]

        angular_step = np.zeros(3, dtype=np.float64)
        virtual_rpy = Rotation.from_matrix(self.virtual_rotation).as_euler("xyz")
        for local_axis, axis in enumerate(range(3, 6)):
            if self._control_mode[axis] != int(ControlMode.POS):
                continue
            if self._policy_mode[axis] == int(PolicyMode.RELATIVE):
                angular_step[local_axis] = target[axis] * dt
            else:
                virtual_rpy[local_axis] = target[axis]
        if np.any(angular_step):
            self.virtual_rotation = (
                Rotation.from_rotvec(angular_step).as_matrix() @ self.virtual_rotation
            )
            virtual_rpy = Rotation.from_matrix(self.virtual_rotation).as_euler("xyz")
        for local_axis, axis in enumerate(range(3, 6)):
            if (
                self._control_mode[axis] == int(ControlMode.POS)
                and self._policy_mode[axis] == int(PolicyMode.ABSOLUTE)
            ):
                virtual_rpy[local_axis] = target[axis]
        self.virtual_rotation = Rotation.from_euler("xyz", virtual_rpy).as_matrix()

        translation_error = self.virtual_position - pose[:3]
        rotation_error = so3_error(self.virtual_rotation, rotation_task_ee)
        pose_error = np.concatenate((translation_error, rotation_error))

        limited_error = pose_error.copy()
        relative_rotation_was_clipped = False
        for axis in range(6):
            if (
                self._control_mode[axis] != int(ControlMode.POS)
                or self._policy_mode[axis] != int(PolicyMode.RELATIVE)
            ):
                continue
            limit = reference_error_limit(
                command["wrench_limits"][axis],
                self._kp[axis],
                bool(command["compliance_reference_limit_enable"][axis]),
            )
            clipped = float(np.clip(limited_error[axis], -limit, limit))
            if axis < 3:
                self.virtual_position[axis] = pose[axis] + clipped
            elif clipped != limited_error[axis]:
                relative_rotation_was_clipped = True
            limited_error[axis] = clipped

        if relative_rotation_was_clipped:
            self.virtual_rotation = (
                Rotation.from_rotvec(limited_error[3:]).as_matrix() @ rotation_task_ee
            )

        wrench = np.zeros(6, dtype=np.float64)
        for axis in range(6):
            mode = ControlMode(int(self._control_mode[axis]))
            if mode is ControlMode.POS:
                wrench[axis] = self._kp[axis] * limited_error[axis] - self._kd[axis] * twist[axis]
            elif mode is ControlMode.VEL:
                wrench[axis] = self._kd[axis] * (target[axis] - twist[axis])
            else:
                wrench[axis] = target[axis]
        return wrench

    def _compute_adaptive_theta(self, command: dict[str, Any]) -> np.ndarray:
        theta = np.ones(6, dtype=np.float64)
        for axis, enabled in enumerate(command["compliance_adaptive_limit_enable"]):
            if enabled:
                theta[axis] = compute_adaptive_limit_theta(
                    command["wrench_limits"][axis],
                    command["compliance_desired_wrench"][axis],
                    command["compliance_adaptive_limit_min"][axis],
                )
        return theta

    def _virtual_transform(self) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.virtual_rotation
        transform[:3, 3] = self.virtual_position
        return transform

    @staticmethod
    def _copy_command(command: dict[str, Any]) -> dict[str, Any]:
        return {
            key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
            for key, value in command.items()
        }

    def _default_hold_command(self) -> dict[str, Any]:
        return {
            "sequence": np.int64(-1),
            "target": np.zeros(7),
            "origin": np.zeros(6),
            "control_mode": np.full(7, int(ControlMode.POS), dtype=np.int8),
            "policy_mode": np.full(7, int(PolicyMode.ABSOLUTE), dtype=np.int8),
            "min_pose": np.asarray(self.config.min_pose_rpy, dtype=np.float64),
            "max_pose": np.asarray(self.config.max_pose_rpy, dtype=np.float64),
            "rotation_interval_modes": np.asarray(
                [
                    int(RotationIntervalMode.from_name(mode))
                    for mode in self.config.rotation_interval_modes
                ],
                dtype=np.int8,
            ),
            "kp": self.HOLD_KP.copy(),
            "kd": self.HOLD_KD.copy(),
            "wrench_limits": np.asarray(self.config.wrench_limits, dtype=np.float64),
            "compliance_reference_limit_enable": np.zeros(6, dtype=np.bool_),
            "compliance_adaptive_limit_enable": np.zeros(6, dtype=np.bool_),
            "compliance_desired_wrench": np.asarray(
                self.config.compliance_desired_wrench, dtype=np.float64
            ),
            "compliance_adaptive_limit_min": np.asarray(
                self.config.compliance_adaptive_limit_min, dtype=np.float64
            ),
            "nullspace_stiffness": np.asarray(
                self.config.nullspace_stiffness, dtype=np.float64
            ),
            "nullspace_damping": np.asarray(
                self.config.nullspace_damping, dtype=np.float64
            ),
            "nullspace_max_torque": float(self.config.nullspace_max_torque),
        }
