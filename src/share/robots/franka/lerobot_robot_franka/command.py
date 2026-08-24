from __future__ import annotations

import enum
import time
from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy as np

from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    TASK_FRAME_AXIS_NAMES,
    TaskFrame,
)
from share.robots.task_frame_command import (
    OverrideField,
    reject_unknown_overrides,
    resolve_override_fields,
    resolve_rotation_interval_modes,
)
from share.utils.transformation_utils import RotationIntervalMode


FR3_JOINT_NAMES = [f"fr3_joint{index}" for index in range(1, 8)]


class FrankaCommand(enum.IntEnum):
    SET = 0
    STOP = 1
    ZERO_FT = 2


def _require_positive(name: str, value: np.ndarray) -> None:
    if np.any(~np.isfinite(value)) or np.any(value < 0.0):
        raise ValueError(f"{name} entries must be finite and non-negative")


def _require_joint_error_clip(name: str, value: np.ndarray) -> None:
    if np.any(~np.isfinite(value)) or np.any((value <= 0.0) | (value > 0.5)):
        raise ValueError(f"{name} entries must be in (0, 0.5]")


# Task-space overrides: Franky's native Cartesian impedance controller takes
# a single translational and a single rotational stiffness (not six
# independent axis gains) plus an optional per-axis force ceiling.
# translational_stiffness/rotational_stiffness are live-updatable every tick
# (CartesianImpedanceTrackingMotion.set_gains, smoothed internally by Franky
# via gains_time_constant) so they're per-command overrides. force_constraints,
# nullspace_stiffness, and nullspace_max_torque are not: Franky fixes them at
# CartesianImpedanceTrackingMotion construction time (no live setter), so
# they live on FrankaConfig only -- see controller.py. Joint-space overrides
# (joint_stiffness/joint_damping/joint_error_clip) are unrelated -- that path
# already wraps the real JointImpedanceTrackingMotion API and is untouched.
# min_pose/max_pose are deliberately not schema entries: unlike the other
# overrides, their default is dynamic -- self.min_pose/self.max_pose (the
# TaskFrame field set directly on the command, e.g. TaskFrame(min_pose=...))
# -- not a fixed constant, so they're resolved by hand below.
OVERRIDE_SCHEMA: dict[str, OverrideField] = {
    "rotation_interval_modes": OverrideField(default=["linear"] * 6, shape=None, dtype=list),
    "translational_stiffness": OverrideField(default=200.0, shape=None, dtype=float, validate=_require_positive),
    "rotational_stiffness": OverrideField(default=20.0, shape=None, dtype=float, validate=_require_positive),
    "compliance_reference_limit_enable": OverrideField(default=[False] * 6, shape=(6,), dtype=np.bool_),
    "joint_stiffness": OverrideField(default=[50.0] * 7, shape=(7,), validate=_require_positive),
    "joint_damping": OverrideField(default=[14.14213562] * 7, shape=(7,), validate=_require_positive),
    "joint_error_clip": OverrideField(default=[0.5] * 7, shape=(7,), validate=_require_joint_error_clip),
}


@dataclass
class FrankaTaskFrameCommand(TaskFrame):
    """Complete, atomically applied Franka controller snapshot.

    Task-space control is position-only: Franky's native Cartesian impedance
    controller does the low-level pose tracking, so there is no SHARE-owned
    wrench law to feed a VEL or WRENCH axis into. ``policy_mode`` still
    distinguishes RELATIVE (velocity, integrated by the bridge with proper
    SO(3) composition for rotation) from ABSOLUTE (a fixed target) per axis.
    """

    cmd: FrankaCommand = FrankaCommand.SET

    SUPPORTED_CONTROLLER_OVERRIDE_KEYS: ClassVar[frozenset[str]] = frozenset(OVERRIDE_SCHEMA) | {
        "min_pose",
        "max_pose",
    }

    def to_queue_dict(
        self,
        *,
        sequence: int = 0,
        timestamp: float | None = None,
    ) -> dict[str, np.ndarray | int | float | bool]:
        raw = asdict(self)
        raw_overrides = raw.pop("controller_overrides", None) or {}
        reject_unknown_overrides(raw_overrides, self.SUPPORTED_CONTROLLER_OVERRIDE_KEYS, "Franka")

        width = len(self.target)
        if self.space == ControlSpace.TASK:
            if width != 6:
                raise ValueError("Franka task-space commands require six axes")
            if any(mode != ControlMode.POS for mode in self.control_mode):
                raise ValueError(
                    "Franka task-space control is position-only -- Franky's native Cartesian "
                    "impedance controller owns the wrench law, so there is no VEL/WRENCH axis "
                    "to send it"
                )
        elif self.space == ControlSpace.JOINT and width != 7:
            raise ValueError("FR3 joint-space commands require seven axes")

        target = np.zeros(7, dtype=np.float64)
        target[:width] = np.asarray(self.target, dtype=np.float64)
        control_mode = np.full(7, -1, dtype=np.int8)
        policy_mode = np.full(7, -1, dtype=np.int8)
        control_mode[:width] = [
            int(mode) if mode is not None else -1 for mode in self.control_mode
        ]
        policy_mode[:width] = [
            int(mode) if mode is not None else -1 for mode in self.policy_mode
        ]

        origin = np.zeros(6, dtype=np.float64)
        if self.origin is not None:
            origin[:] = np.asarray(self.origin, dtype=np.float64)

        if not np.all(np.isfinite(target[:width])):
            raise ValueError("Franka targets must be finite")
        if not np.all(np.isfinite(origin)):
            raise ValueError("Franka task-frame origins must be finite")

        fields = resolve_override_fields(raw_overrides, OVERRIDE_SCHEMA)
        # self.min_pose/self.max_pose are only meaningful as Cartesian bounds
        # for TASK space -- for JOINT space they're 7-long joint bounds set
        # by TaskFrame's own __post_init__, not a fallback for this 6-wide
        # queue field, so only use them as the default when space == TASK.
        is_task = self.space == ControlSpace.TASK
        min_pose = np.asarray(
            raw_overrides.get("min_pose", self.min_pose if is_task else [-np.inf] * 6),
            dtype=np.float64,
        )
        max_pose = np.asarray(
            raw_overrides.get("max_pose", self.max_pose if is_task else [np.inf] * 6),
            dtype=np.float64,
        )
        rotation_interval_modes = resolve_rotation_interval_modes(fields.pop("rotation_interval_modes"))

        if min_pose.shape != (6,) or max_pose.shape != (6,):
            raise ValueError("min_pose and max_pose must have shape (6,)")
        for axis in range(3):
            if min_pose[axis] > max_pose[axis]:
                raise ValueError("Translational min_pose must not exceed max_pose")
        for axis in range(3, 6):
            mode = RotationIntervalMode(int(rotation_interval_modes[axis]))
            if mode == RotationIntervalMode.CCW_ARC:
                if not np.isfinite(min_pose[axis]) or not np.isfinite(max_pose[axis]):
                    raise ValueError("ccw_arc rotational intervals require finite endpoints")
            elif min_pose[axis] > max_pose[axis]:
                raise ValueError("Linear rotational min_pose must not exceed max_pose")

        snapshot = {
            "cmd": int(self.cmd),
            "sequence": np.int64(sequence),
            "command_timestamp": float(time.monotonic() if timestamp is None else timestamp),
            "space": np.int8(self.space),
            "width": np.int8(width),
            "target": target,
            "control_mode": control_mode,
            "policy_mode": policy_mode,
            "origin": origin,
            "min_pose": min_pose,
            "max_pose": max_pose,
            "rotation_interval_modes": rotation_interval_modes,
            **fields,
        }
        return snapshot

    def to_robot_action(self) -> dict[str, float]:
        if self.space == ControlSpace.JOINT:
            if len(self.target) != 7:
                raise ValueError("FR3 joint-space control requires seven axes")
            names = self.joint_names or FR3_JOINT_NAMES
            return {f"{name}.pos": float(self.target[index]) for index, name in enumerate(names)}

        if any(mode != ControlMode.POS for mode in self.control_mode):
            raise ValueError("Franka task-space control is position-only")
        return {
            f"{axis_name}.ee_pos": float(self.target[axis])
            for axis, axis_name in enumerate(TASK_FRAME_AXIS_NAMES)
        }

    @classmethod
    def stop_command(cls) -> "FrankaTaskFrameCommand":
        return cls(cmd=FrankaCommand.STOP)

    @classmethod
    def zero_ft_command(cls) -> "FrankaTaskFrameCommand":
        return cls(cmd=FrankaCommand.ZERO_FT)
