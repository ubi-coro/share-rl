from typing import Any

import gymnasium
import numpy as np
from gymnasium.core import ObsType

from lerobot.utils.constants import OBS_IMAGES
from lerobot.cameras import Camera
from lerobot.robots import Robot

from share.envs.manipulation_primitive.task_frame import TaskFrame
from share.envs.utils import check_task_frame_robot
from share.teleoperators import TeleopEvents


class ManipulationPrimitive(gymnasium.Env):
    """Minimal gym env wiring robots, cameras, and task-frame commands."""

    def __init__(
        self,
        task_frame: dict[str, TaskFrame],
        robot_dict: dict[str, Robot],
        cameras: dict[str, Camera],
        display_cameras: bool = False
    ):
        """Initialize robot/camera handles and primitive runtime state.

        Args:
            task_frame: Per-robot task-frame command objects mutated by the
                primitive config and consumed by the robot interfaces.
            robot_dict: Connected robots keyed by primitive robot name.
            cameras: Connected cameras exposed through the env observation dict.
            display_cameras: Whether ``render()`` should open live camera views.
        """
        self.robot_dict = robot_dict
        self.task_frame = task_frame
        self.cameras = cameras
        self.display_cameras = display_cameras

        self.current_step = 0
        self._motor_keys: set[str] = set()
        self._is_task_frame_robot: dict[str, bool] = check_task_frame_robot(robot_dict)
        self.reset_runtime_state()
        self.apply_task_frames()
        for name, robot in self.robot_dict.items():
            self._motor_keys.update([f"{name}.{key}" for key in robot._motors_ft])

    def step(self, action: dict[str, dict[str, float]]) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply one outer primitive step.

        Args:
            action: Nested action dictionary keyed first by robot name and then
                by low-level command key such as ``x.ee_pos`` or ``joint_1.pos``.

        Returns:
            Standard gym ``(observation, reward, terminated, truncated, info)``
            values for one primitive step.
        """
        self.apply_task_frames()
        for name, robot in self.robot_dict.items():
            robot.send_action(action.get(name, {}))

        obs = self._get_observation()

        if self.display_cameras:
            self.render()

        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset counters and clear env-owned runtime state.

        Args:
            seed: Optional gym reset seed.
            options: Optional reset payload forwarded by the MP-Net runtime.

        Returns:
            A fresh raw observation and default info payload for the primitive.
        """
        super().reset(seed=seed, options=options)

        # Manipulation primitives do not execute an autonomous reset trajectory by default.
        # We only re-send the configured task frame to robots that support task-frame commands.
        self.apply_task_frames()

        self.current_step = 0
        self.reset_runtime_state()
        obs = self._get_observation()
        return obs, self._get_info()

    def render(self) -> None:
        """Display camera observations using OpenCV windows."""
        import cv2
        current_observation = self._get_observation()
        if current_observation is not None and "pixels" in current_observation:
            for key, img in current_observation["pixels"].items():
                cv2.imshow(key, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self) -> None:
        """Disconnect any connected robots."""
        for robot_dict in self.robot_dict.values():
            if robot_dict.is_connected:
                robot_dict.disconnect()

    def apply_task_frames(self) -> None:
        """Re-send the current task frame to every robot that supports it."""
        for name, robot in self.robot_dict.items():
            if self._is_task_frame_robot.get(name, False):
                robot.set_task_frame(self.task_frame[name])

    def reset_runtime_state(self) -> None:
        """Clear runtime-only target/progress state owned by this env."""
        self._target_pose_info_key: str | None = None
        self._target_pose: dict[str, list[float]] = {}
        self._primitive_complete = False
        self._trajectory_progress = 0.0

    def set_target_pose(
        self,
        target_pose: dict[str, list[float]],
        info_key: str | None,
        *,
        update_task_frame: bool = True,
    ) -> None:
        """Store the current runtime target pose and update task-frame targets.

        Args:
            target_pose: Per-robot 6D target poses in this primitive's task
                frame. These values become the live task-frame targets.
            info_key: Optional info key used to publish the target pose.
            update_task_frame: Whether these targets should also become the live
                task-frame command stored on the env.
        """
        self._target_pose = {name: list(pose) for name, pose in target_pose.items()}
        self._target_pose_info_key = info_key
        if not update_task_frame:
            return
        for name, pose in self._target_pose.items():
            if name not in self.task_frame:
                continue
            for axis in range(min(len(self.task_frame[name].target), len(pose))):
                self.task_frame[name].target[axis] = float(pose[axis])

    def _get_observation(self):
        """Collect raw camera pixels and robot observations into one dict."""
        obs_dict = {}

        for cam_key, cam in self.cameras.items():
            obs_dict[f"{OBS_IMAGES}.{cam_key}"] = cam.async_read()

        for name in self.robot_dict:
            robot_dict_obs_dict = self.robot_dict[name].get_observation()
            obs_dict |= {f"{name}.{key}": robot_dict_obs_dict[key] for key in robot_dict_obs_dict}

        return obs_dict

    def _get_info(self):
        """Return the primitive runtime info payload for the current step."""
        info = {
            TeleopEvents.IS_INTERVENTION: False,
            "primitive_complete": bool(self._primitive_complete),
            "trajectory_progress": float(self._trajectory_progress),
        }
        if self._target_pose_info_key and self._target_pose:
            info[self._target_pose_info_key] = {
                name: list(pose) for name, pose in self._target_pose.items()
            }
        return info


class OpenLoopTrajectoryPrimitive(ManipulationPrimitive):
    """Primitive env variant that executes a scripted chunk per outer step."""

    def __init__(
        self,
        task_frame: dict[str, TaskFrame],
        robot_dict: dict[str, Robot],
        cameras: dict[str, Camera],
        open_loop_config: Any,
        display_cameras: bool = False,
    ):
        """Initialize the scripted open-loop primitive env.

        Args:
            task_frame: Per-robot task-frame command objects.
            robot_dict: Connected robots keyed by primitive robot name.
            cameras: Connected cameras exposed through the env observation dict.
            open_loop_config: Config object supplying trajectory timing params.
            display_cameras: Whether ``render()`` should open live camera views.
        """
        self.open_loop_config = open_loop_config
        self._trajectory_substeps = 0
        super().__init__(
            task_frame=task_frame,
            robot_dict=robot_dict,
            cameras=cameras,
            display_cameras=display_cameras,
        )

    @property
    def uses_autonomous_step(self) -> bool:
        return True

    def reset_runtime_state(self) -> None:
        super().reset_runtime_state()
        self._start_pose: dict[str, list[float]] = {}
        self._duration_substeps = 0
        self._substeps_per_step = 1
        self._trajectory_substeps = 0

    def configure_trajectory(
        self,
        start_pose: dict[str, list[float]],
        target_pose: dict[str, list[float]],
        info_key: str | None,
    ) -> None:
        """Initialize one scripted trajectory from entry-time poses.

        Args:
            start_pose: Per-robot 6D start poses in this primitive's task frame.
            target_pose: Per-robot 6D target poses in this primitive's task
                frame. These are also published as the env target pose.
        """
        self._start_pose = {name: list(pose) for name, pose in start_pose.items()}
        self._duration_substeps, self._substeps_per_step = self.open_loop_config.trajectory_timing(self.robot_dict)
        self.set_target_pose(
            target_pose=target_pose,
            info_key=info_key,
            update_task_frame=False,
        )
        self._set_live_task_frame_pose(start_pose)
        self._primitive_complete = False
        self._trajectory_progress = 0.0
        self._trajectory_substeps = 0

    def step(self, action: dict[str, dict[str, float]]) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Run a scripted chunk of the open-loop trajectory.

        Args:
            action: Ignored in v1. The primitive follows its scripted internal
                trajectory instead of consuming an external policy action.

        Returns:
            Standard gym ``(observation, reward, terminated, truncated, info)``
            values after executing up to the configured internal robot
            steps.
        """
        substeps = max(1, int(self._substeps_per_step))
        obs = self._get_observation()
        reward = 0.0
        terminated = False
        truncated = False
        for _ in range(substeps):
            self._trajectory_substeps += 1
            alpha = self._trajectory_substeps / float(self._duration_substeps)
            scripted_pose = self.open_loop_config.target_pose_at(
                alpha=alpha,
                start_pose=self._start_pose,
                goal_pose=self._target_pose,
            )
            self._set_live_task_frame_pose(scripted_pose)
            scripted_action = self._action_from_pose(scripted_pose)
            obs, step_reward, terminated, truncated, _info = super().step(scripted_action)
            reward += step_reward
            self._trajectory_progress = min(1.0, alpha)
            self._primitive_complete = alpha >= 1.0
            if terminated or truncated:
                break

        return obs, reward, terminated, truncated, self._get_info()

    def _set_live_task_frame_pose(self, pose_by_robot: dict[str, list[float]]) -> None:
        """Update the live task-frame setpoint used by the scripted runner."""
        for name, pose in pose_by_robot.items():
            if name not in self.task_frame:
                continue
            for axis in range(min(len(self.task_frame[name].target), len(pose))):
                self.task_frame[name].target[axis] = float(pose[axis])

    def _action_from_pose(self, pose_by_robot: dict[str, list[float]]) -> dict[str, dict[str, float]]:
        """Build one low-level action dict from the current scripted target pose.

        Args:
            pose_by_robot: Per-robot scripted task-space pose for this substep.

        Returns:
            Nested low-level robot commands matching the current scripted pose.
        """
        action: dict[str, dict[str, float]] = {}
        for name, pose in pose_by_robot.items():
            frame = self.task_frame[name]
            action[name] = {
                frame.action_key_for_axis(axis): float(pose[axis])
                for axis in range(len(frame.target))
            }
        return action
