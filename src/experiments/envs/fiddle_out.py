"""Experiment MP-Net config for teleop plus an EE-frame fiddle-out trajectory."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.envs import EnvConfig
from pynput import keyboard

from share.envs.manipulation_primitive.config_manipulation_primitive import (
    EventConfig,
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    ObservationConfig,
    OpenLoopTrajectoryPrimitiveConfig,
    OpenLoopTrajectorySpec,
)
from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    PolicyMode,
    TaskFrame,
)
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (
    ManipulationPrimitiveNetConfig,
)
from share.envs.manipulation_primitive_net.transitions import OnSuccess, OnTargetPoseReached
from share.envs.utils import copy_per_robot
from share.robots.ur import URConfig
from share.teleoperators import TeleopEvents
from share.teleoperators.spacemouse import SpaceMouseConfig
from share.utils.transformation_utils import compose_delta_pose, task_pose_to_world_pose, world_pose_to_task_pose


def _shared_processor() -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=30.0,
        observation=ObservationConfig(
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=True,
            add_ee_pos_to_observation=False,
            add_joint_position_to_observation=False,
        ),
        gripper=GripperConfig(enable=False),
        events=EventConfig(
            key_mapping={
                TeleopEvents.RERECORD_EPISODE: keyboard.Key.left,
                TeleopEvents.STOP_RECORDING: keyboard.Key.down,
            },
            foot_switch_mapping={
                (TeleopEvents.SUCCESS,): {"device": 20, "toggle": False},
            },
        ),
    )


@ManipulationPrimitiveConfig.register_subclass("ee_fiddle_out_circle")
@dataclass
class EEFiddleOutCirclePrimitiveConfig(OpenLoopTrajectoryPrimitiveConfig):
    """Open-loop EE-frame lift with a small XY circle while holding orientation fixed."""

    circle_radius_m: float | dict[str, float] = 0.005
    circle_frequency_hz: float | dict[str, float] = 2.0
    lift_duration_s: float | dict[str, float] = 1.0

    def validate(self, robot_dict, teleop_dict):
        super().validate(robot_dict, teleop_dict)
        robot_names = list(robot_dict)
        self.circle_radius_m = copy_per_robot(self.circle_radius_m, robot_names)
        self.circle_frequency_hz = copy_per_robot(self.circle_frequency_hz, robot_names)
        self.lift_duration_s = copy_per_robot(self.lift_duration_s, robot_names)

        if self.trajectory.delta is None:
            raise ValueError("ee_fiddle_out_circle requires trajectory.delta.")

        for name in robot_names:
            radius = float(self.circle_radius_m[name])
            frequency_hz = float(self.circle_frequency_hz[name])
            lift_duration_s = float(self.lift_duration_s[name])
            total_duration_s = float(self.trajectory.duration_s[name])
            delta = [float(v) for v in self.trajectory.delta[name]]
            frame = self.trajectory.frame[name]

            if radius < 0.0:
                raise ValueError("ee_fiddle_out_circle requires circle_radius_m >= 0.")
            if frequency_hz < 0.0:
                raise ValueError("ee_fiddle_out_circle requires circle_frequency_hz >= 0.")
            if lift_duration_s <= 0.0:
                raise ValueError("ee_fiddle_out_circle requires lift_duration_s > 0.")
            if total_duration_s < lift_duration_s:
                raise ValueError("ee_fiddle_out_circle requires trajectory.duration_s >= lift_duration_s.")
            if frame != "ee_current":
                raise ValueError("ee_fiddle_out_circle requires trajectory.frame == 'ee_current'.")
            if any(abs(delta[axis]) > 1e-9 for axis in (0, 1, 3, 4, 5)):
                raise ValueError("ee_fiddle_out_circle only supports a pure local-Z trajectory.delta.")
            if delta[2] <= 0.0:
                raise ValueError("ee_fiddle_out_circle requires a positive local-Z trajectory.delta.")

    def target_pose_at(
        self,
        alpha: float,
        start_pose: dict[str, list[float]],
        goal_pose: dict[str, list[float]],
    ) -> dict[str, list[float]]:
        alpha = max(0.0, float(alpha))
        z_alpha = min(1.0, alpha)
        pose_by_robot: dict[str, list[float]] = {}
        task_frames = self.task_frame if isinstance(self.task_frame, dict) else {
            name: self.task_frame for name in start_pose
        }
        circle_radius = self.circle_radius_m if isinstance(self.circle_radius_m, dict) else {
            name: self.circle_radius_m for name in start_pose
        }
        circle_frequency = self.circle_frequency_hz if isinstance(self.circle_frequency_hz, dict) else {
            name: self.circle_frequency_hz for name in start_pose
        }
        trajectory_delta = self.trajectory.delta if isinstance(self.trajectory.delta, dict) else {
            name: self.trajectory.delta for name in start_pose
        }
        trajectory_duration = self.trajectory.duration_s if isinstance(self.trajectory.duration_s, dict) else {
            name: self.trajectory.duration_s for name in start_pose
        }
        lift_duration = self.lift_duration_s if isinstance(self.lift_duration_s, dict) else {
            name: self.lift_duration_s for name in start_pose
        }

        for name, frame in task_frames.items():
            start_world = task_pose_to_world_pose(start_pose[name], frame.origin)
            delta_z = float(trajectory_delta[name][2])
            radius = float(circle_radius[name])
            elapsed_s = alpha * float(trajectory_duration[name])
            z_alpha = min(1.0, elapsed_s / float(lift_duration[name]))
            theta = 2.0 * math.pi * float(circle_frequency[name]) * elapsed_s
            scripted_world = compose_delta_pose(
                start_pose_world=start_world,
                delta=[
                    float(radius * (math.cos(theta) - 1.0)),
                    float(radius * math.sin(theta)),
                    float(z_alpha * delta_z),
                    0.0,
                    0.0,
                    0.0,
                ],
                frame_name="ee_current",
            )
            pose_by_robot[name] = world_pose_to_task_pose(scripted_world, frame.origin)

        return pose_by_robot


@EnvConfig.register_subclass("demo_ur3e_teleop_fiddle_out")
@dataclass
class DemoUR3eTeleopFiddleOutEnvConfig(ManipulationPrimitiveNetConfig):
    """Teleop in 6-DoF until success, then execute a small EE-frame fiddle-out."""

    fps: int = 30
    start_primitive: str = "teleop"
    reset_primitive: str = "teleop"

    fiddle_out_height_m: float = 0.02
    fiddle_out_radius_m: float = 0.005
    fiddle_out_circle_frequency_hz: float = 2.0
    fiddle_out_lift_duration_s: float = 1.0
    fiddle_out_duration_s: float = 3.0
    teleop_wrench_limits: list[float] = field(default_factory=lambda: [25.0, 25.0, 25.0, 1.0, 1.0, 1.0])
    fiddle_out_wrench_limits: list[float] = field(default_factory=lambda: [6.0, 6.0, 6.0, 0.4, 0.4, 0.4])

    def __post_init__(self):
        processor = _shared_processor()
        processor.fps = float(self.fps)

        self.robot = URConfig(
            robot_ip="172.22.22.2",
            kp=[3000, 3000, 3000, 200, 200, 200],
            soft_real_time=True,
            rt_core=3,
            wrench_limits=list(self.teleop_wrench_limits),
            compliance_reference_limit_enable=[False] * 6,
            debug=True,
            debug_axis=2,
        )
        self.teleop = SpaceMouseConfig()
        self.teleop.action_scale = [0.05, 0.05, 0.2, 0.1, 0.1, 0.1]
        self.cameras = {
            "wrist": RealSenseCameraConfig(serial_number_or_name="323743071487"),
        }

        teleop_primitive = ManipulationPrimitiveConfig(
            task_frame=TaskFrame(
                target=[0.0] * 6,
                space=ControlSpace.TASK,
                control_mode=[ControlMode.POS] * 6,
                policy_mode=[PolicyMode.RELATIVE] * 6,
                controller_overrides={"wrench_limits": list(self.teleop_wrench_limits)},
            ),
            processor=processor,
            notes="6-DoF relative task-space teleop primitive.",
        )
        fiddle_out_primitive = EEFiddleOutCirclePrimitiveConfig(
            task_frame=TaskFrame(
                target=[0.0] * 6,
                space=ControlSpace.TASK,
                control_mode=[ControlMode.POS] * 6,
                policy_mode=[None] * 6,
                controller_overrides={"wrench_limits": list(self.fiddle_out_wrench_limits)},
            ),
            trajectory=OpenLoopTrajectorySpec(
                delta=[0.0, 0.0, float(self.fiddle_out_height_m), 0.0, 0.0, 0.0],
                frame="ee_current",
                duration_s=float(self.fiddle_out_duration_s),
            ),
            circle_radius_m=float(self.fiddle_out_radius_m),
            circle_frequency_hz=float(self.fiddle_out_circle_frequency_hz),
            lift_duration_s=float(self.fiddle_out_lift_duration_s),
            processor=processor,
            notes=(
                "Open-loop fiddle-out: ramp up in the current EE frame, keep orientation fixed, "
                "then keep drawing a local XY circle while holding the final Z command until the "
                "primitive times out or the Z-based transition fires."
            ),
        )

        self.primitives = {
            "teleop": teleop_primitive,
            "fiddle_out": fiddle_out_primitive,
        }
        self.transitions = [
            OnSuccess(source="teleop", target="fiddle_out"),
            OnTargetPoseReached(source="fiddle_out", target="teleop", axes=["z"]),
        ]

        super().__post_init__()


__all__ = [
    "EEFiddleOutCirclePrimitiveConfig",
    "DemoUR3eTeleopFiddleOutEnvConfig",
]
