from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode
from share.robots.adaptive_limits import reference_error_limit
from share.utils.transformation_utils import RotationIntervalMode, clip_angle_to_ccw_arc, wrap_to_pi


@dataclass(slots=True)
class FrankaState:
    """Hardware-neutral state consumed by the reference-tracking strategy."""

    q: np.ndarray
    dq: np.ndarray
    T_base_ee: np.ndarray
    T_ee_stiffness: np.ndarray
    twist_base_ee: np.ndarray
    wrench_base_at_stiffness: np.ndarray
    timestamp: float


@dataclass(slots=True)
class ReferenceOutput:
    """One control tick's result: the pose Franky should track, plus diagnostics.

    Franky's native ``CartesianImpedanceTrackingMotion`` turns
    ``target_pose_task_rpy`` into torque on its own real-time thread; this
    strategy never computes torque or a wrench itself. Stiffness gains are
    reported raw (not smoothed here) -- Franky smooths ``set_gains`` updates
    itself via its own ``gains_time_constant``.
    """

    target_pose_task_rpy: np.ndarray
    pose_task_rpy: np.ndarray
    twist_task: np.ndarray
    measured_wrench_task: np.ndarray
    translational_stiffness: float
    rotational_stiffness: float
    holding: bool


class FrankaControllerStrategy(Protocol):
    """Stable strategy surface for Python and future pybind controllers."""

    def step(self, state: FrankaState, command: dict[str, Any] | None, dt: float) -> ReferenceOutput:
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
    offset = rotation_task_base @ (T_base_stiffness[:3, 3] - T_base_task[:3, 3])
    return transform_wrench(wrench_base_at_stiffness, rotation_task_base, offset)


def so3_error(desired: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Logarithmic orientation error, expressed in the common parent frame."""
    return Rotation.from_matrix(desired @ actual.T).as_rotvec()


def clip_pose_to_workspace(
    pose_rpy: np.ndarray,
    min_pose: np.ndarray,
    max_pose: np.ndarray,
    rotation_interval_modes: np.ndarray,
) -> np.ndarray:
    """Clip a target pose directly into the configured workspace box.

    Translation is clipped per axis. Rotation is clipped in wrapped RPY,
    either to a linear [min, max] range or, for ``ccw_arc`` axes, to the
    nearest point on the allowed counterclockwise arc.
    """
    out = np.array(pose_rpy, dtype=np.float64, copy=True)
    out[:3] = np.clip(out[:3], np.asarray(min_pose[:3]), np.asarray(max_pose[:3]))

    wrapped = np.asarray(wrap_to_pi(out[3:6]), dtype=np.float64)
    for local_axis, axis in enumerate(range(3, 6)):
        if not np.isfinite(min_pose[axis]) and not np.isfinite(max_pose[axis]):
            continue
        mode = RotationIntervalMode(int(rotation_interval_modes[axis]))
        if mode is RotationIntervalMode.CCW_ARC:
            wrapped[local_axis] = clip_angle_to_ccw_arc(
                float(wrapped[local_axis]), float(min_pose[axis]), float(max_pose[axis])
            )
        else:
            wrapped[local_axis] = np.clip(wrapped[local_axis], min_pose[axis], max_pose[axis])
    out[3:6] = wrapped
    return out


class CartesianReferenceController:
    """Integrates a POS-only task-frame command into a target pose for Franky.

    Franky's native ``CartesianImpedanceTrackingMotion`` owns position-error
    -> torque (and its own gain smoothing); this class owns everything
    upstream of that: virtual-target integration (translation as a velocity,
    rotation via proper SO(3) composition for RELATIVE axes), origin
    re-anchoring when the task frame moves mid-run, workspace clamping, and a
    reference-error anti-windup clamp against the measured pose. It never
    computes a wrench or a torque, and it never touches Franky directly --
    ``force_constraints``/nullspace are fixed for the whole connection on
    ``FrankaConfig`` because Franky fixes them at motion-construction time
    (see controller.py); only ``translational_stiffness``/
    ``rotational_stiffness`` are live-updatable per command.
    """

    HOLD_TRANSLATIONAL_STIFFNESS = 200.0
    HOLD_ROTATIONAL_STIFFNESS = 20.0

    def __init__(self, config: Any):
        self.config = config
        self.T_base_task = np.eye(4, dtype=np.float64)
        self.virtual_position = np.zeros(3, dtype=np.float64)
        self.virtual_rotation = np.eye(3, dtype=np.float64)
        self.wrench_bias_base = np.zeros(6, dtype=np.float64)
        self._holding = False
        self._initialized = False
        self._control_mode = np.full(6, int(ControlMode.POS), dtype=np.int8)
        self._policy_mode = np.full(6, int(PolicyMode.RELATIVE), dtype=np.int8)
        self._sequence = -1
        self._command: dict[str, Any] | None = None

    def zero_wrench(self, state: FrankaState) -> None:
        self.wrench_bias_base = np.asarray(state.wrench_base_at_stiffness, dtype=np.float64).copy()

    def step(self, state: FrankaState, command: dict[str, Any] | None, dt: float) -> ReferenceOutput:
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

        command_data = self._command
        assert command_data is not None
        target_pose = self._integrate_target(pose, rotation_task_ee, dt)
        target_pose = clip_pose_to_workspace(
            target_pose,
            command_data["min_pose"],
            command_data["max_pose"],
            command_data["rotation_interval_modes"],
        )

        return ReferenceOutput(
            target_pose_task_rpy=target_pose,
            pose_task_rpy=pose,
            twist_task=twist,
            measured_wrench_task=measured_wrench,
            translational_stiffness=float(command_data["translational_stiffness"]),
            rotational_stiffness=float(command_data["rotational_stiffness"]),
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
        self._command = command
        self._sequence = int(command["sequence"])
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
        self._command["translational_stiffness"] = self.HOLD_TRANSLATIONAL_STIFFNESS
        self._command["rotational_stiffness"] = self.HOLD_ROTATIONAL_STIFFNESS
        self._holding = True
        self._initialized = True

    def _integrate_target(self, pose: np.ndarray, rotation_task_ee: np.ndarray, dt: float) -> np.ndarray:
        """Advance the virtual target and return it, clamped against windup.

        Relative POS axes are a velocity integrated at the control
        frequency -- translation directly, rotation via an SO(3) composition
        so a mix of relative axes still yields a proper 3D rotation rather
        than an Euler-angle sum. Absolute POS axes are then imposed directly.
        """
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
                Rotation.from_rotvec(angular_step) * Rotation.from_matrix(self.virtual_rotation)
            ).as_matrix()
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
        limited_error = np.concatenate((translation_error, rotation_error))

        translational_stiffness = float(command["translational_stiffness"])
        rotational_stiffness = float(command["rotational_stiffness"])
        force_constraints = self.config.force_constraints
        # The clamp bounds the force we command *this tick* (stiffness * error <=
        # force_constraint); it must not overwrite the persistent virtual target
        # itself. self.virtual_position/self.virtual_rotation carry the true,
        # unclamped intended target across ticks -- mutating them here would
        # permanently re-anchor the reference to wherever the arm currently is
        # the first time tracking error saturates, discarding the real target
        # and turning a one-off saturation into commanding max force in a fixed
        # direction forever (even once the arm stops moving and error would
        # otherwise settle back to zero). Only the value returned/sent for this
        # tick is clamped.
        output_position = self.virtual_position.copy()
        output_rotation = self.virtual_rotation
        relative_rotation_was_clipped = False
        for axis in range(6):
            if (
                self._control_mode[axis] != int(ControlMode.POS)
                or self._policy_mode[axis] != int(PolicyMode.RELATIVE)
            ):
                continue
            stiffness = translational_stiffness if axis < 3 else rotational_stiffness
            limit = reference_error_limit(
                float(force_constraints[axis]),
                stiffness,
                bool(command["compliance_reference_limit_enable"][axis]),
            )
            clipped = float(np.clip(limited_error[axis], -limit, limit))
            if axis < 3:
                output_position[axis] = pose[axis] + clipped
            elif clipped != limited_error[axis]:
                relative_rotation_was_clipped = True
            limited_error[axis] = clipped

        if relative_rotation_was_clipped:
            output_rotation = (
                Rotation.from_rotvec(limited_error[3:]) * Rotation.from_matrix(rotation_task_ee)
            ).as_matrix()

        return np.concatenate(
            (output_position, Rotation.from_matrix(output_rotation).as_euler("xyz"))
        )

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
                [int(RotationIntervalMode.from_name(mode)) for mode in self.config.rotation_interval_modes],
                dtype=np.int8,
            ),
            "translational_stiffness": self.HOLD_TRANSLATIONAL_STIFFNESS,
            "rotational_stiffness": self.HOLD_ROTATIONAL_STIFFNESS,
            "compliance_reference_limit_enable": np.zeros(6, dtype=np.bool_),
        }
