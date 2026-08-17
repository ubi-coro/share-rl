#!/usr/bin/env python
import logging
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from multiprocessing.managers import SharedMemoryManager
from typing import Literal, Optional, Sequence

import draccus
import numpy as np

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig

draccus.encode.register(SharedMemoryManager, lambda _: None)

@RobotConfig.register_subclass("ur")
@dataclass
class URConfig(RobotConfig):
    robot_ip: str
    model: str = "ur5e"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # gripper
    use_gripper: bool = False  # attempts to initialize gripper from RTDEControlInterface
    gripper_frequency: float = 50.0
    gripper_vel: float = 1.0  # [0-1]
    gripper_force: float = 1.0  # [0-1]
    gripper_soft_real_time: bool = False
    gripper_rt_core: int = 4

    # controller parameters
    frequency: float = 500.0
    payload_mass: Optional[float] = None
    payload_cog: Optional[Sequence[float]] = None
    tcp_offset_pose: Optional[list[float]] = None
    soft_real_time: bool = False
    rt_core: int = 3
    launch_timeout: float = 10.0
    get_max_k: int = 128
    shm_manager: Optional[SharedMemoryManager] = None
    ft_filter_cutoff_hz: Optional[float] = None  # Hz, EMA low-pass cutoff for f/t sensor
    force_mode_gain_scaling: float = 1.0
    force_mode_damping: Optional[float] = None  # None keeps the UR controller default.
    use_force_mode: bool = True  # Set False to use direct pose servoing instead of UR force mode.
    simple_pose_use_servo: bool = False  # True → servoL (streaming 1 kHz), False → moveL (default)
    simple_pose_servo_time: float = 0.002  # [s] servoL control duration per command.
    simple_pose_lookahead_time: float = 0.2  # [s] range [0.03, 0.2], larger is smoother/slower.
    simple_pose_gain: float = 100.0  # range [100, 2000], lower is gentler/slower.
    simple_pose_max_speed: list[float] = field(default_factory=lambda: [0.2, 0.2, 0.2, 0.3, 0.3, 0.3])  # [m/s, m/s, m/s, rad/s, rad/s, rad/s] — only used for servoL
    simple_pose_move_speed: float = 0.2  # [m/s] moveL translation speed
    simple_pose_move_accel: float = 0.1   # [m/s²] moveL acceleration
    simple_pose_stop_accel: float = 0.5  # [m/s²] stopL deceleration when leaving moveL
    kp: list[float] = field(default_factory=lambda: [2500.0, 2500.0, 2500.0, 150.0, 150.0, 150.0])
    kd: list[float] = field(default_factory=lambda: [80.0, 80.0, 80.0, 8.0, 8.0, 8.0])

    # safety
    max_pose_rpy: list[float] = field(default_factory = lambda: [float("inf")] * 6)
    min_pose_rpy: list[float] = field(default_factory = lambda: [-float("inf")] * 6)
    wrench_limits: list[float] = field(default_factory = lambda: [30.0, 30.0, 30.0, 3.0, 3.0, 3.0])
    speed_limits: list[float] = field(default_factory = lambda: [5.0, 5.0, 5.0, 0.5, 0.5, 0.5])

    # compliance / anti-windup behavior
    compliance_adaptive_limit_enable: list[bool] = field(default_factory=lambda: [False] * 6)
    compliance_reference_limit_enable: list[bool] = field(default_factory=lambda: [False] * 6)
    compliance_desired_wrench: list[float] = field(default_factory=lambda: [5.0, 5.0, 5.0, 0.5, 0.5, 0.5])
    compliance_adaptive_limit_theta: Optional[list[float]] = None
    compliance_adaptive_limit_min: list[float] = field(default_factory=lambda: [0.1] * 6)
    
    # flag
    use_degrees: bool = False  # Set to `True` for backward compatibility with previous policies/dataset
    verbose: bool = False
    mock: bool = False
    debug: bool = False
    debug_hz: float = 10.0
    debug_axis: int = 0

    # Asynchronous controller-rate wrench visualization.
    wrench_monitor_enabled: bool = False
    wrench_monitor_refresh_hz: float = 30.0
    wrench_monitor_history_s: float = 10.0

    def __post_init__(self):
        if float(self.debug_hz) <= 0.0:
            raise ValueError("URConfig.debug_hz must be > 0.")
        if float(self.wrench_monitor_refresh_hz) <= 0.0:
            raise ValueError("URConfig.wrench_monitor_refresh_hz must be > 0.")
        if float(self.wrench_monitor_history_s) <= 0.0:
            raise ValueError("URConfig.wrench_monitor_history_s must be > 0.")
        if self.force_mode_damping is not None and not (0.0 <= float(self.force_mode_damping) <= 1.0):
            raise ValueError("URConfig.force_mode_damping must be in [0, 1].")
        if not (0.03 <= float(self.simple_pose_lookahead_time) <= 0.2):
            raise ValueError("URConfig.simple_pose_lookahead_time must be in [0.03, 0.2].")
        if not (100.0 <= float(self.simple_pose_gain) <= 2000.0):
            raise ValueError("URConfig.simple_pose_gain must be in [100, 2000].")
        if float(self.simple_pose_servo_time) <= 0.0:
            raise ValueError("URConfig.simple_pose_servo_time must be > 0.")
        if len(self.simple_pose_max_speed) != 6 or any(v <= 0 for v in self.simple_pose_max_speed):
            raise ValueError("URConfig.simple_pose_max_speed must be a length-6 list of positive values.")
        if len(self.kp) != 6:
            raise ValueError("URConfig.kp must be a length-6 list.")
        if len(self.kd) != 6:
            raise ValueError("URConfig.kd must be a length-6 list.")
        if len(self.compliance_adaptive_limit_enable) != 6:
            raise ValueError("URConfig.compliance_adaptive_limit_enable must be a length-6 list.")
        if len(self.compliance_reference_limit_enable) != 6:
            raise ValueError("URConfig.compliance_reference_limit_enable must be a length-6 list.")
        if self.compliance_adaptive_limit_theta is None:
            if self.verbose:
                logging.info(f"=== Compute parameters for exponential contact force limit scaling: ===")

            self.compliance_adaptive_limit_theta = [0.0] * 6
            for i in range(6):
                if not self.compliance_adaptive_limit_enable[i]:
                    continue

                if self.wrench_limits[i] == float("inf"):
                    self.wrench_limits[i] = 2.0 * self.compliance_desired_wrench[i]

                # Compute theta
                theta = self.compute_theta(
                    self.wrench_limits[i],  # assume uniform limits
                    self.compliance_desired_wrench[i],
                    self.compliance_adaptive_limit_min[i],
                )

                # Evaluate scale and derivative at f_star
                s_star, ds_df_star = self.exp_scale_and_derivative(
                    self.compliance_desired_wrench[i],
                    theta,
                    self.compliance_adaptive_limit_min[i]
                )
                g_prime = self.wrench_limits[i] * ds_df_star

                if self.verbose:
                    logging.info(f" {['X', 'Y', 'Z', 'A', 'B', 'C'][i]}-Axis:")
                    logging.info(f"  Computed θ = {theta:.4f}")
                    logging.info(f"  At f* = {self.compliance_desired_wrench[i]} N:")
                    logging.info(f"    s(f*) = {s_star:.4f}")
                    logging.info(f"    s'(f*) = {ds_df_star:.4f}")
                    logging.info(f"    g'(f*) = F_max * s'(f*) = {g_prime:.4f}")

                    # Bifurcation check
                    if abs(g_prime) < 1.0:
                        logging.info("  --> Stable fixed point (|g'(f*)| < 1)")
                    else:
                        logging.warning("  --> Unstable: bifurcation/oscillation likely (|g'(f*)| >= 1)")

                if abs(g_prime) >= 1.0:
                    raise ValueError(f"Likely oscillation on {['X', 'Y', 'Z', 'A', 'B', 'C'][i]}-axis contact "
                                     f"force limiter, run again with verbose=True and check parameters!")

                self.compliance_adaptive_limit_theta[i] = theta

    @staticmethod
    def compute_theta(F_max: float, f_star: float, s_min: float) -> float:
        """
        Compute the decay constant θ so that the fixed-point condition
            f_star = F_max * [s_min + (1 - s_min) * exp(-f_star/θ)]
        is satisfied exactly.
        """
        s_star = f_star / F_max
        if not (s_min < s_star < 1.0):
            raise ValueError("Require s_min < f_star/F_max < 1.0")
        ratio = (s_star - s_min) / (1.0 - s_min)
        return -f_star / np.log(ratio)

    @staticmethod
    def exp_scale_and_derivative(f: float, theta: float, s_min: float) -> tuple:
        """
        Returns the scale s(f) and its derivative s'(f) for exponential-decay-to-floor.
        s(f) = s_min + (1 - s_min)*exp(-f/theta)
        s'(f) = -(1 - s_min)/theta * exp(-f/theta)
        """
        exp_term = np.exp(-f / theta)
        s = s_min + (1 - s_min) * exp_term
        ds_df = -(1 - s_min) / theta * exp_term
        return s, ds_df
