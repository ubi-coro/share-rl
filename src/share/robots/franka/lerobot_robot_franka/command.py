from __future__ import annotations

import enum
import time
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import numpy as np

from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    PolicyMode,
    TASK_FRAME_AXIS_NAMES,
    TaskFrame,
)
from share.utils.transformation_utils import RotationIntervalMode


FR3_JOINT_NAMES = [f"fr3_joint{index}" for index in range(1, 8)]


class FrankaCommand(enum.IntEnum):
    SET = 0
    STOP = 1
    ZERO_FT = 2


@dataclass
class FrankaTaskFrameCommand(TaskFrame):
    """Complete, atomically applied Franka controller snapshot."""

    cmd: FrankaCommand = FrankaCommand.SET

    SUPPORTED_CONTROLLER_OVERRIDE_KEYS: ClassVar[set[str]] = {
        "kp",
        "kd",
        "min_pose",
        "max_pose",
        "rotation_interval_modes",
        "wrench_limits",
        "compliance_reference_limit_enable",
        "compliance_adaptive_limit_enable",
        "compliance_desired_wrench",
        "compliance_adaptive_limit_min",
        "nullspace_stiffness",
        "nullspace_damping",
        "nullspace_max_torque",
        "joint_stiffness",
        "joint_damping",
        "joint_error_clip",
    }

    def to_queue_dict(
        self,
        *,
        sequence: int = 0,
        timestamp: float | None = None,
    ) -> dict[str, np.ndarray | int | float | bool]:
        raw = asdict(self)
        raw_overrides = raw.pop("controller_overrides", None) or {}
        unknown = set(raw_overrides) - self.SUPPORTED_CONTROLLER_OVERRIDE_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unsupported Franka controller overrides: {names}")

        width = len(self.target)
        if self.space == ControlSpace.TASK and width != 6:
            raise ValueError("Franka task-space commands require six axes")
        if self.space == ControlSpace.JOINT and width != 7:
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

        min_pose = np.asarray(
            raw_overrides.get(
                "min_pose",
                self.min_pose if self.space == ControlSpace.TASK else [-np.inf] * 6,
            ),
            dtype=np.float64,
        )
        max_pose = np.asarray(
            raw_overrides.get(
                "max_pose",
                self.max_pose if self.space == ControlSpace.TASK else [np.inf] * 6,
            ),
            dtype=np.float64,
        )
        rotation_modes = np.asarray(
            [
                int(RotationIntervalMode.from_name(str(mode)))
                for mode in raw_overrides.get("rotation_interval_modes", ["linear"] * 6)
            ],
            dtype=np.int8,
        )

        if min_pose.shape != (6,) or max_pose.shape != (6,):
            raise ValueError("min_pose and max_pose must have shape (6,)")
        if not np.all(np.isfinite(target[:width])):
            raise ValueError("Franka targets must be finite")
        if not np.all(np.isfinite(origin)):
            raise ValueError("Franka task-frame origins must be finite")
        for axis in range(3):
            if min_pose[axis] > max_pose[axis]:
                raise ValueError("Translational min_pose must not exceed max_pose")
        for axis in range(3, 6):
            mode = RotationIntervalMode(int(rotation_modes[axis]))
            if mode == RotationIntervalMode.CCW_ARC:
                if not np.isfinite(min_pose[axis]) or not np.isfinite(max_pose[axis]):
                    raise ValueError(
                        "ccw_arc rotational intervals require finite endpoints"
                    )
            elif min_pose[axis] > max_pose[axis]:
                raise ValueError(
                    "Linear rotational min_pose must not exceed max_pose"
                )

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
            "rotation_interval_modes": rotation_modes,
            "kp": self._array(raw_overrides, "kp", [500.0] * 3 + [50.0] * 3, 6),
            "kd": self._array(
                raw_overrides,
                "kd",
                [44.72135955] * 3 + [14.14213562] * 3,
                6,
            ),
            "wrench_limits": self._array(
                raw_overrides, "wrench_limits", [30.0] * 3 + [3.0] * 3, 6
            ),
            "compliance_reference_limit_enable": self._array(
                raw_overrides,
                "compliance_reference_limit_enable",
                [False] * 6,
                6,
                dtype=np.bool_,
            ),
            "compliance_adaptive_limit_enable": self._array(
                raw_overrides,
                "compliance_adaptive_limit_enable",
                [False] * 6,
                6,
                dtype=np.bool_,
            ),
            "compliance_desired_wrench": self._array(
                raw_overrides,
                "compliance_desired_wrench",
                [5.0] * 3 + [0.5] * 3,
                6,
            ),
            "compliance_adaptive_limit_min": self._array(
                raw_overrides, "compliance_adaptive_limit_min", [0.1] * 6, 6
            ),
            "nullspace_stiffness": self._array(
                raw_overrides, "nullspace_stiffness", [20.0] * 7, 7
            ),
            "nullspace_damping": self._array(
                raw_overrides, "nullspace_damping", [8.94427191] * 7, 7
            ),
            "nullspace_max_torque": float(
                raw_overrides.get("nullspace_max_torque", 5.0)
            ),
            "joint_stiffness": self._array(
                raw_overrides, "joint_stiffness", [50.0] * 7, 7
            ),
            "joint_damping": self._array(
                raw_overrides, "joint_damping", [14.14213562] * 7, 7
            ),
            "joint_error_clip": self._array(
                raw_overrides, "joint_error_clip", [0.5] * 7, 7
            ),
        }
        for name in (
            "kp",
            "kd",
            "nullspace_stiffness",
            "nullspace_damping",
            "joint_stiffness",
            "joint_damping",
        ):
            values = snapshot[name]
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} entries must be finite and non-negative")
        if np.any(np.isnan(snapshot["wrench_limits"])) or np.any(
            snapshot["wrench_limits"] <= 0.0
        ):
            raise ValueError("wrench_limits entries must be positive")
        if np.any(~np.isfinite(snapshot["compliance_desired_wrench"])) or np.any(
            snapshot["compliance_desired_wrench"] <= 0.0
        ):
            raise ValueError("compliance_desired_wrench entries must be finite and positive")
        adaptive_minimum = snapshot["compliance_adaptive_limit_min"]
        if np.any(~np.isfinite(adaptive_minimum)) or np.any(
            (adaptive_minimum < 0.0) | (adaptive_minimum >= 1.0)
        ):
            raise ValueError("compliance_adaptive_limit_min entries must be in [0, 1)")
        if not np.isfinite(snapshot["nullspace_max_torque"]) or (
            snapshot["nullspace_max_torque"] < 0.0
        ):
            raise ValueError("nullspace_max_torque must be finite and non-negative")
        error_clip = snapshot["joint_error_clip"]
        if np.any(~np.isfinite(error_clip)) or np.any(
            (error_clip <= 0.0) | (error_clip > 0.5)
        ):
            raise ValueError("joint_error_clip entries must be in (0, 0.5]")
        return snapshot

    @staticmethod
    def _array(
        overrides: dict[str, Any],
        name: str,
        default: list[Any],
        length: int,
        *,
        dtype: np.dtype = np.float64,
    ) -> np.ndarray:
        value = np.asarray(overrides.get(name, default), dtype=dtype)
        if value.shape != (length,):
            raise ValueError(f"{name} must have shape ({length},)")
        return value

    def to_robot_action(self) -> dict[str, float]:
        if self.space == ControlSpace.JOINT:
            if len(self.target) != 7:
                raise ValueError("FR3 joint-space control requires seven axes")
            names = self.joint_names or FR3_JOINT_NAMES
            return {f"{name}.pos": float(self.target[index]) for index, name in enumerate(names)}

        action: dict[str, float] = {}
        for axis, axis_name in enumerate(TASK_FRAME_AXIS_NAMES):
            suffix = {
                ControlMode.POS: "ee_pos",
                ControlMode.VEL: "ee_vel",
                ControlMode.WRENCH: "ee_wrench",
            }[self.control_mode[axis]]
            action[f"{axis_name}.{suffix}"] = float(self.target[axis])
        return action

    @classmethod
    def stop_command(cls) -> "FrankaTaskFrameCommand":
        return cls(cmd=FrankaCommand.STOP)

    @classmethod
    def zero_ft_command(cls) -> "FrankaTaskFrameCommand":
        return cls(cmd=FrankaCommand.ZERO_FT)
