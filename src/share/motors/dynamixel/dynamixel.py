import math
from enum import Enum

from lerobot.motors.dynamixel import DynamixelMotorsBus, OperatingMode
from lerobot.motors.motors_bus import MotorNormMode, MotorCalibration


class TrossenNormMode(str, Enum):
    RANGE_0_100 = "range_0_100"
    RANGE_M100_100 = "range_m100_100"
    DEGREES = "degrees"
    RANGE_0_1 = "range_0_1"
    RADIANS = "radians"


class TrossenDynamixelBus(DynamixelMotorsBus):

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        for motor, calibration in calibration_dict.items():
            if self.read("Homing_Offset", motor) != calibration.homing_offset:
                self.write("Homing_Offset", motor, calibration.homing_offset)
            if self.read("Min_Position_Limit", motor) != calibration.range_min:
                self.write("Min_Position_Limit", motor, calibration.range_min)
            if self.read("Max_Position_Limit", motor) != calibration.range_max:
                self.write("Max_Position_Limit", motor, calibration.range_max)
            if self.read("Drive_Mode", motor) != calibration.drive_mode:
                self.write("Drive_Mode", motor, calibration.drive_mode)

        if cache:
            self.calibration = calibration_dict

    def _unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        unnormalized_values = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            if self.motors[motor].norm_mode is TrossenNormMode.RANGE_M100_100:
                val = -val if drive_mode else val
                bounded_val = min(100.0, max(-100.0, val))
                unnormalized_values[id_] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is TrossenNormMode.RANGE_0_100:
                val = 100 - val if drive_mode else val
                bounded_val = min(100.0, max(0.0, val))
                unnormalized_values[id_] = int((bounded_val / 100) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is TrossenNormMode.RANGE_0_1:
                val = 1.0 - val if drive_mode else val
                bounded_val = min(1.0, max(0.0, val))
                unnormalized_values[id_] = int((bounded_val) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is TrossenNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                unnormalized_values[id_] = int((val * max_res / 360) + mid)
            elif self.motors[motor].norm_mode is TrossenNormMode.RADIANS:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                unnormalized_values[id_] = int((val * max_res / (2 * math.pi)) + mid)
            else:
                raise NotImplementedError

        return unnormalized_values

    def _normalize(self, ids_values: dict[int, int]) -> dict[int, float]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized_values = {}
        for id_, val in ids_values.items():
            motor = self._id_to_name(id_)
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            bounded_val = min(max_, max(min_, val))
            if self.motors[motor].norm_mode is TrossenNormMode.RANGE_M100_100:
                norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
                normalized_values[id_] = -norm if drive_mode else norm
            elif self.motors[motor].norm_mode is TrossenNormMode.RANGE_0_100:
                norm = ((bounded_val - min_) / (max_ - min_)) * 100
                normalized_values[id_] = 100 - norm if drive_mode else norm
            elif self.motors[motor].norm_mode is TrossenNormMode.RANGE_0_1:
                norm = ((bounded_val - min_) / (max_ - min_))
                normalized_values[id_] = 1.0 - norm if drive_mode else norm
            elif self.motors[motor].norm_mode is TrossenNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                normalized_values[id_] = (val - mid) * 360 / max_res
            elif self.motors[motor].norm_mode is TrossenNormMode.RADIANS:
                mid = (min_ + max_) / 2
                max_res = self.model_resolution_table[self._id_to_model(id_)] - 1
                normalized_values[id_] = (val - mid) * 2 * math.pi / max_res
            else:
                raise NotImplementedError

        return normalized_values


