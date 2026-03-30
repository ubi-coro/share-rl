"""Experiment-only scripted MP-Net primitives and example configs."""

from __future__ import annotations

import math
from dataclasses import dataclass

from lerobot.cameras import Camera
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator

from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig, PrimitiveEntryContext
from share.envs.manipulation_primitive.env_manipulation_primitive import (
    ManipulationPrimitive,
    OpenLoopTrajectoryPrimitive,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (
    ManipulationPrimitiveNetConfig,
)
from share.envs.utils import resolve_entry_start_pose
from share.utils.constants import DEFAULT_ROBOT_NAME
from share.utils.transformation_utils import (
    task_pose_to_world_pose,
    world_pose_to_task_pose,
)


class WorldLiftCirclePrimitive(OpenLoopTrajectoryPrimitive):
    """Scripted primitive that lifts in world Z while tracing a world-frame XY circle."""

    def _interpolated_action(self, alpha: float) -> dict[str, dict[str, float]]:
        action: dict[str, dict[str, float]] = {}
        theta = 2.0 * math.pi * alpha
        radius = float(self.open_loop_config.circle_radius_m)

        for name, frame in self.task_frame.items():
            start_pose = self._start_pose.get(name, [float(v) for v in frame.target])
            target_pose = self._target_pose.get(name, [float(v) for v in frame.target])
            start_world = task_pose_to_world_pose(start_pose, frame.origin)
            target_world = task_pose_to_world_pose(target_pose, frame.origin)

            world_pose = list(target_world)
            world_pose[0] = float(target_world[0] + radius * (math.cos(theta) - 1.0))
            world_pose[1] = float(target_world[1] + radius * math.sin(theta))
            world_pose[2] = float(start_world[2] + alpha * (target_world[2] - start_world[2]))
            world_pose[3:6] = [float(v) for v in start_world[3:6]]

            pose = world_pose_to_task_pose(world_pose, frame.origin)
            action[name] = {
                frame.action_key_for_axis(axis): float(pose[axis])
                for axis in range(len(frame.target))
            }

        return action


@ManipulationPrimitiveConfig.register_subclass("world_lift_circle")
@dataclass
class WorldLiftCirclePrimitiveConfig(ManipulationPrimitiveConfig):
    """Scripted primitive that lifts 5 cm in world Z and draws an XY circle."""

    circle_radius_m: float = 0.02
    lift_height_m: float = 0.05
    duration_substeps: int = 60
    substeps_per_step: int = 5
    controller_hz: float | None = None
    substep_dt_s: float | None = None

    def validate(self, robot_dict, teleop_dict):
        super().validate(robot_dict, teleop_dict)

        if self.circle_radius_m < 0.0:
            raise ValueError("world_lift_circle requires circle_radius_m >= 0.")
        if self.duration_substeps <= 0:
            raise ValueError("world_lift_circle requires duration_substeps > 0.")
        if self.substeps_per_step <= 0:
            raise ValueError("world_lift_circle requires substeps_per_step > 0.")
        if self.lift_height_m <= 0.0:
            raise ValueError("world_lift_circle requires lift_height_m > 0.")

        if self.controller_hz is None and self.substep_dt_s is None:
            self.substep_dt_s = 1.0 / self.processor.fps if self.processor.fps > 0 else 0.0
        elif self.substep_dt_s is None:
            self.substep_dt_s = 1.0 / self.controller_hz

        if self.policy is not None:
            raise ValueError("world_lift_circle is scripted-only and must not configure a policy.")

        for name, frame in self.task_frame.items():
            if frame.space != ControlSpace.TASK:
                raise ValueError(f"world_lift_circle requires TASK-space frames, got '{name}'.")
            if frame.is_adaptive:
                raise ValueError(f"world_lift_circle does not support learnable axes, got '{name}'.")
            if any(mode != ControlMode.POS for mode in frame.control_mode):
                raise ValueError(f"world_lift_circle requires POS control on every axis, got '{name}'.")

    def make(
        self,
        robot_dict: dict[str, Robot],
        teleop_dict: dict[str, Teleoperator],
        cameras: dict[str, Camera],
        device: str = "cpu",
    ):
        self.validate(robot_dict, teleop_dict)
        self.infer_features(robot_dict, cameras)

        display_cameras = self.processor.image_preprocessing is not None and self.processor.image_preprocessing.display_cameras
        env = WorldLiftCirclePrimitive(
            task_frame=self.task_frame,
            robot_dict=robot_dict,
            cameras=cameras,
            open_loop_config=self,
            display_cameras=display_cameras,
        )
        env_processor = self.make_env_processor(device)
        action_processor = self.make_action_processor(robot_dict, teleop_dict, device)
        return env, env_processor, action_processor

    def on_entry(self, env: ManipulationPrimitive, entry_context: PrimitiveEntryContext | None) -> None:
        start_pose, target_pose = self.resolve_targets(entry_context)
        env.set_target_pose(target_pose, info_key=self.target_pose_info_key)
        if isinstance(env, OpenLoopTrajectoryPrimitive):
            env.configure_trajectory(start_pose=start_pose, target_pose=target_pose)

    def resolve_targets(
        self,
        entry_context: PrimitiveEntryContext | None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        start_pose: dict[str, list[float]] = {}
        target_pose: dict[str, list[float]] = {}

        for name, frame in self.task_frame.items():
            start_pose[name] = resolve_entry_start_pose(entry_context, name, frame)
            start_world = task_pose_to_world_pose(start_pose[name], frame.origin)
            target_world = list(start_world)
            target_world[2] = float(start_world[2] + self.lift_height_m)
            target_world[3:6] = [float(v) for v in start_world[3:6]]
            target_pose[name] = world_pose_to_task_pose(target_world, frame.origin)

        return start_pose, target_pose


def make_world_lift_circle_mpnet(
    *,
    circle_radius_m: float = 0.02,
    duration_substeps: int = 60,
    substeps_per_step: int = 5,
    primitive_name: str = "lift_circle",
    robot_name: str = DEFAULT_ROBOT_NAME,
) -> ManipulationPrimitiveNetConfig:
    """Build a minimal MP-Net using the custom lift-and-circle primitive."""

    primitive = WorldLiftCirclePrimitiveConfig(
        task_frame={
            robot_name: TaskFrame(
                target=[0.0] * 6,
                origin=[0.0] * 6,
                control_mode=[ControlMode.POS] * 6,
            )
        },
        circle_radius_m=circle_radius_m,
        duration_substeps=duration_substeps,
        substeps_per_step=substeps_per_step,
        notes=(
            "Experiment primitive: lift 5 cm in world Z from the current pose, "
            "draw one world-frame XY circle, and keep orientation fixed."
        ),
        is_terminal=True,
    )
    return ManipulationPrimitiveNetConfig(
        start_primitive=primitive_name,
        reset_primitive=primitive_name,
        primitives={primitive_name: primitive},
        transitions=[],
    )


WORLD_LIFT_CIRCLE_MPNET = make_world_lift_circle_mpnet()


__all__ = [
    "WorldLiftCirclePrimitive",
    "WorldLiftCirclePrimitiveConfig",
    "make_world_lift_circle_mpnet",
    "WORLD_LIFT_CIRCLE_MPNET",
]
