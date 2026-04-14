#!/usr/bin/env python

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

from dataclasses import dataclass, field

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.teleoperators import TeleoperatorConfig

GELLO_ANGLE_NORM_MODE = getattr(MotorNormMode, "RADIANS", MotorNormMode.DEGREES)
GELLO_OUTPUTS_ARE_NATIVE_RADIANS = hasattr(MotorNormMode, "RADIANS")


def _default_gello_motors() -> dict[str, Motor]:
    return {
        "waist": Motor(1, "xl330-m288", GELLO_ANGLE_NORM_MODE),
        "shoulder": Motor(2, "xl330-m288", GELLO_ANGLE_NORM_MODE),
        "elbow": Motor(3, "xl330-m288", GELLO_ANGLE_NORM_MODE),
        "forearm_roll": Motor(4, "xl330-m288", GELLO_ANGLE_NORM_MODE),
        "wrist_angle": Motor(5, "xl330-m288", GELLO_ANGLE_NORM_MODE),
        "wrist_rotate": Motor(6, "xl330-m288", GELLO_ANGLE_NORM_MODE),
        "gripper": Motor(7, "xl330-m077", GELLO_ANGLE_NORM_MODE),
    }


def _default_gello_calibration() -> dict[str, MotorCalibration]:
    return {
        "waist": MotorCalibration(id=1, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
        "shoulder": MotorCalibration(id=2, drive_mode=1, homing_offset=0, range_min=0, range_max=4095),
        "elbow": MotorCalibration(id=3, drive_mode=1, homing_offset=0, range_min=0, range_max=4095),
        "forearm_roll": MotorCalibration(id=4, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
        "wrist_angle": MotorCalibration(id=5, drive_mode=1, homing_offset=0, range_min=0, range_max=4095),
        "wrist_rotate": MotorCalibration(id=6, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
        "gripper": MotorCalibration(id=7, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
    }


def _ur_output_name_map() -> dict[str, str]:
    return {
        "waist": "shoulder_pan_joint",
        "shoulder": "shoulder_lift_joint",
        "elbow": "elbow_joint",
        "forearm_roll": "wrist_1_joint",
        "wrist_angle": "wrist_2_joint",
        "wrist_rotate": "wrist_3_joint",
        "gripper": "gripper",
    }


@TeleoperatorConfig.register_subclass("gello")
@dataclass
class GelloConfig(TeleoperatorConfig):
    port: str  # Port to connect to the arm

    motors: dict[str, Motor] = field(default_factory=lambda: {})
    output_name_map: dict[str, str] = field(default_factory=dict)

    default_calibration: dict[str, MotorCalibration] | None = None

    # The duration of the velocity-based time profile
    # Higher values lead to smoother motions, but increase lag.
    moving_time: float = 0.1

    def output_motor_names(self) -> dict[str, str]:
        missing_motors = sorted(set(self.output_name_map) - set(self.motors))
        if missing_motors:
            raise ValueError(
                "Gello output_name_map contains unknown motors: "
                + ", ".join(missing_motors)
            )

        output_names = {
            motor_name: self.output_name_map.get(motor_name, motor_name)
            for motor_name in self.motors
        }
        duplicates = sorted({
            output_name
            for output_name in output_names.values()
            if list(output_names.values()).count(output_name) > 1
        })
        if duplicates:
            raise ValueError(
                "Gello output_name_map must produce unique output names, got duplicates: "
                + ", ".join(duplicates)
            )

        return output_names


@TeleoperatorConfig.register_subclass("gello_viperx")
@dataclass
class GelloViperXConfig(GelloConfig):
    motors: dict[str, Motor] = field(default_factory=_default_gello_motors)
    default_calibration: dict[str, MotorCalibration] | None = field(default_factory=_default_gello_calibration)


@TeleoperatorConfig.register_subclass("gello_ur")
@dataclass
class GelloURConfig(GelloViperXConfig):
    output_name_map: dict[str, str] = field(default_factory=_ur_output_name_map)


@TeleoperatorConfig.register_subclass("gelloha")
@dataclass
class GellohaConfig(GelloViperXConfig):
    output_name_map: dict[str, str] = field(default_factory=lambda: {"gripper": "finger"})
