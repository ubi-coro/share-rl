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

from lerobot.robots import RobotConfig
from lerobot.cameras import CameraConfig


@dataclass
class BotaConfig:
    """Bota force/torque sensor between the wrist and the gripper, used to diagnose contact forces.

    When enabled, the observation carries the gravity-compensated wrench as `{x,y,z,rx,ry,rz}.ee_wrench`
    (N / Nm, sensor frame, NaN until the payload is calibrated). It only reaches a policy if the env's
    `add_ee_wrench_to_observation` is set. The payload fit is stored in the arm's calibration file and made
    by `ViperX.calibrate()`.
    """

    enable: bool = False
    # Stable /dev/serial/by-id/usb-Bota_Systems_... path of the sensor
    port: str = ""
    # Calibration switches the torque off and records the trajectory while you move the arm slowly by hand through
    # as many orientations as you can, for at most this many seconds. Press Enter to finish earlier once enough
    # orientations are covered. Hold the arm by the wrist or forearm, not the gripper, so your grip does not
    # enter the sensor.
    calibration_duration_s: float = 90.0
    # Seconds of still readings averaged to re-zero the bias on connect. The arm must be still and untouched.
    tare_duration_s: float = 1.0
    # Also fit a small net for what depends on the pose and the weight model cannot know, e.g. cable pull. It is
    # kept only if it beats the weight model on held-out orientations, and needs more orientations (30 instead of
    # 12). The trajectory is stored next to the calibration file, so a later calibration can refit it (e.g. to
    # turn this on) without recording a new one. Connecting with this on and a stored fit that has no net runs the
    # calibration, unless a net was already tried and rejected. A stored net is used whatever this says.
    residual: bool = False


@RobotConfig.register_subclass("viperx")
@dataclass
class ViperXConfig(RobotConfig):
    port: str  # Port to connect to the arm

    disable_torque_on_disconnect: bool = False

    # /!\ FOR SAFETY, READ THIS /!\
    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    # For Aloha, for every goal position request, motor rotations are capped at 5 degrees by default.
    # When you feel more confident with teleoperation or running the policy, you can extend
    # this safety limit and even removing it by setting it to `null`.
    max_relative_target: float | None = 5.0

    # The duration of the velocity-based time profile
    # Higher values lead to smoother motions, but increase lag.
    moving_time: float = 0.1

    bota: BotaConfig = field(default_factory=BotaConfig)

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Troubleshooting: If one of your IntelRealSense cameras freeze during
    # data recording due to bandwidth limit, you might need to plug the camera
    # on another USB hub or PCIe card.
