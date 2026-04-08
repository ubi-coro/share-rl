from collections import deque
from dataclasses import dataclass, field
from typing import Any

import einops
import numpy as np
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import EnvTransition, TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry, ObservationProcessorStep
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from share.envs.manipulation_primitive.task_frame import TASK_FRAME_AXIS_NAMES
from share.utils.transformation_utils import (
    rotation_from_extrinsic_xyz,
    euler_xyz_from_rotation,
    get_robot_pose_from_observation,
)

for _registry_name in (
    "default_observation_processor",
    "joints_to_ee_observation",
    "relative_frame_observation",
    "joint_velocity_processor",
    "current_processor",
):
    ProcessorStepRegistry.unregister(_registry_name)


@dataclass
@ProcessorStepRegistry.register("default_observation_processor")
class DefaultObservationProcessor(ProcessorStep):
    """Build ``observation.state`` from normalized per-robot modality config.

    All boolean, axis-selection, and frame-stacking settings are expected to be
    per-robot dicts, matching the normalized manipulation-primitive config.
    """

    device: str = "cpu"

    gripper_enable: dict[str, bool] = field(default_factory=dict)
    add_joint_position_to_observation: dict[str, bool] = field(default_factory=dict)
    add_joint_velocity_to_observation: dict[str, bool] = field(default_factory=dict)
    add_current_to_observation: dict[str, bool] = field(default_factory=dict)

    add_ee_pos_to_observation: dict[str, bool] = field(default_factory=dict)
    ee_pos_axes: dict[str, list[str]] = field(default_factory=dict)

    add_ee_velocity_to_observation: dict[str, bool] = field(default_factory=dict)
    ee_velocity_axes: dict[str, list[str]] = field(default_factory=dict)

    add_ee_wrench_to_observation: dict[str, bool] = field(default_factory=dict)
    ee_wrench_axes: dict[str, list[str]] = field(default_factory=dict)

    stack_frames: dict[str, int] = field(default_factory=dict)

    _prev_obs: dict[str, dict[str, float]] = field(default_factory=dict, init=False)
    _state_buffer: deque[torch.Tensor] = field(init=False)

    def __post_init__(self):
        self._state_buffer = deque(maxlen=self._resolved_stack_frames())

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            return transition

        new_transition = transition.copy()
        new_observation = dict(observation)

        state_values = self._collect_state_values(observation)
        if state_values:
            state_tensor = torch.tensor(state_values, dtype=torch.float32)
            stack_frames = self._resolved_stack_frames()

            if stack_frames > 1:
                if not self._state_buffer:
                    for _ in range(stack_frames):
                        self._state_buffer.append(state_tensor)
                else:
                    self._state_buffer.append(state_tensor)
                state_tensor = torch.cat(list(self._state_buffer), dim=-1)

            new_observation[OBS_STATE] = state_tensor

        for key, value in observation.items():
            if "image" in key:
                new_observation[key] = self._process_image(value)

        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def _collect_state_values(self, observation: dict[str, Any]) -> list[float]:
        values: list[float] = []

        for name in sorted(self._robot_names(observation)):
            if self._is_enabled(self.add_joint_position_to_observation, name):
                values.extend(self._joint_values(observation, name, "pos"))

            if self._is_enabled(self.add_joint_velocity_to_observation, name):
                vals = self._joint_values(observation, name, "vel")
                if not vals:
                    vals = self._differentiate(name, observation, self._joint_keys(observation, name, "pos"))
                values.extend(vals)

            if self._is_enabled(self.add_current_to_observation, name):
                values.extend(self._joint_values(observation, name, "current"))

            if self._is_enabled(self.add_ee_pos_to_observation, name):
                values.extend(
                    self._ee_values(
                        observation,
                        name,
                        self._axes(self.ee_pos_axes, name, ".ee_pos"),
                    )
                )

            if self._is_enabled(self.add_ee_velocity_to_observation, name):
                axes = self._axes(self.ee_velocity_axes, name, ".ee_vel")
                vals = self._ee_values(observation, name, axes)
                if not vals:
                    vals = self._differentiate(
                        name,
                        observation,
                        [f"{name}.{axis}.ee_pos" for axis in axes],
                    )
                values.extend(vals)

            if self._is_enabled(self.add_ee_wrench_to_observation, name):
                values.extend(
                    self._ee_values(
                        observation,
                        name,
                        self._axes(self.ee_wrench_axes, name, ".ee_wrench"),
                    )
                )

            if self._is_enabled(self.gripper_enable, name):
                key = f"{name}.gripper.pos"
                if key in observation:
                    values.append(self._to_float(observation[key]))

        self._update_prev_obs(observation)
        return values

    def _process_image(self, image: Any) -> torch.Tensor:
        img = image if isinstance(image, torch.Tensor) else torch.from_numpy(np.asarray(image))

        if img.ndim == 3:
            h, w, c = img.shape
            if c < h and c < w:
                img = einops.rearrange(img, "h w c -> c h w")
        elif img.ndim == 4:
            _, h, w, c = img.shape
            if c < h and c < w:
                img = einops.rearrange(img, "b h w c -> b c h w")
        else:
            raise ValueError(f"Expected image tensor with 3 or 4 dimensions, got shape {tuple(img.shape)}")

        img = img.to(torch.float32)
        return img / 255.0 if img.max() > 1.0 else img

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features = {ft: dict(bucket) for ft, bucket in features.items()}
        obs_features = new_features.get(PipelineFeatureType.OBSERVATION, {})

        state_dim = 0
        for name in sorted(self._robot_names(obs_features)):
            if self._is_enabled(self.add_joint_position_to_observation, name):
                state_dim += len(self._joint_keys(obs_features, name, "pos"))

            if self._is_enabled(self.add_joint_velocity_to_observation, name):
                vel_keys = self._joint_keys(obs_features, name, "vel")
                state_dim += len(vel_keys) if vel_keys else len(self._joint_keys(obs_features, name, "pos"))

            if self._is_enabled(self.add_current_to_observation, name):
                state_dim += len(self._joint_keys(obs_features, name, "current"))

            if self._is_enabled(self.add_ee_pos_to_observation, name):
                state_dim += len(self._ee_keys(obs_features, name, self._axes(self.ee_pos_axes, name, ".ee_pos")))

            if self._is_enabled(self.add_ee_velocity_to_observation, name):
                filter_vel = self._axes(self.ee_velocity_axes, name, ".ee_vel")
                filter_pos = self._axes(self.ee_pos_axes, name, ".ee_pos")
                vel_keys = self._ee_keys(obs_features, name, filter_vel)
                state_dim += len(vel_keys) if vel_keys else len(self._ee_keys(obs_features, name, filter_pos))

            if self._is_enabled(self.add_ee_wrench_to_observation, name):
                state_dim += len(self._ee_keys(obs_features, name, self._axes(self.ee_wrench_axes, name, ".ee_wrench")))

            if self._is_enabled(self.gripper_enable, name) and f"{name}.gripper.pos" in obs_features:
                state_dim += 1

        if state_dim > 0:
            obs_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(state_dim * self._resolved_stack_frames(),),
            )

        for name, feature in obs_features.items():
            if feature.type == FeatureType.VISUAL:
                h, w, c = feature.shape
                if c < h and c < w:
                    obs_features[name].shape = (c, h, w)

        return new_features

    @staticmethod
    def _robot_names(observation: dict[str, Any]) -> set[str]:
        return {
            key.split(".", 1)[0]
            for key in observation
            if "." in key and not key.startswith(OBS_IMAGES)
        }

    @staticmethod
    def _is_enabled(flag_dict: dict[str, bool], name: str) -> bool:
        return bool(flag_dict.get(name, False))

    @staticmethod
    def _axes(axis_dict: dict[str, list[str]], name: str, suffix: str = ".pos") -> list[str]:
        return list(axis_dict.get(name, [f"{ax}{suffix}" for ax in TASK_FRAME_AXIS_NAMES]))

    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.item()) if value.ndim == 0 else float(value.flatten()[0].item())
        return float(value)

    def _joint_keys(self, observation: dict[str, Any], robot_name: str, suffix: str) -> list[str]:
        prefix = f"{robot_name}."
        return [
            key
            for key in observation
            if key.startswith(prefix)
            and key.endswith(f".{suffix}")
            and ".ee_" not in key
            and ".gripper." not in key
        ]

    def _joint_values(self, observation: dict[str, Any], robot_name: str, suffix: str) -> list[float]:
        return [self._to_float(observation[key]) for key in self._joint_keys(observation, robot_name, suffix)]

    @staticmethod
    def _ee_keys(
        observation: dict[str, Any],
        robot_name: str,
        axis_names: list[str],
    ) -> list[str]:
        return [f"{robot_name}.{axis}" for axis in axis_names if f"{robot_name}.{axis}" in observation]

    def _ee_values(
        self,
        observation: dict[str, Any],
        robot_name: str,
        axis_names: list[str],
    ) -> list[float]:
        return [self._to_float(observation[key]) for key in self._ee_keys(observation, robot_name, axis_names)]

    def _differentiate(self, robot_name: str, observation: dict[str, Any], keys: list[str]) -> list[float]:
        prev = self._prev_obs.get(robot_name, {})
        return [
            self._to_float(observation[key]) - prev.get(key, self._to_float(observation[key]))
            for key in keys
            if key in observation
        ]

    def _update_prev_obs(self, observation: dict[str, Any]) -> None:
        for name in self._robot_names(observation):
            prefix = f"{name}."
            self._prev_obs[name] = {
                key: self._to_float(value)
                for key, value in observation.items()
                if key.startswith(prefix) and "image" not in key
            }

    def _resolved_stack_frames(self) -> int:
        unique = {int(v) for v in self.stack_frames.values()}
        if not unique:
            return 1
        if len(unique) > 1:
            raise ValueError("VanillaMPObservationProcessorStep requires uniform stack_frames across robots.")
        return max(1, unique.pop())

    def reset(self) -> None:
        self._prev_obs.clear()
        self._state_buffer = deque(maxlen=self._resolved_stack_frames())


@dataclass
@ProcessorStepRegistry.register("joints_to_ee_observation")
class JointsToEEObservation(ProcessorStep):
    """Append deterministic end-effector pose channels from joint observations."""

    kinematics: dict[str, Any] = field(default_factory=dict)
    motor_names: dict[str, list[str]] = field(default_factory=dict)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            return transition

        is_batched = any(isinstance(v, torch.Tensor) and v.ndim > 0 for v in observation.values())
        batch_size = None
        if is_batched:
            for value in observation.values():
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    batch_size = int(value.shape[0])
                    break

        new_transition = transition.copy()
        new_observation = dict(observation)

        for robot_name, solver in self.kinematics.items():
            joints = self.motor_names.get(robot_name, [])
            if not joints:
                continue

            if is_batched:
                if batch_size is None:
                    continue
                axis_values = [[] for _ in range(6)]
                for b in range(batch_size):
                    joint_state = self._extract_joint_state(observation, robot_name, joints, index=b)
                    pose = solver.forward_kinematics(joint_state)
                    for axis in range(6):
                        axis_values[axis].append(float(pose[axis]))

                for axis, axis_name in enumerate(TASK_FRAME_AXIS_NAMES):
                    new_observation[f"{robot_name}.{axis_name}.ee_pos"] = torch.tensor(axis_values[axis], dtype=torch.float32)
            else:
                joint_state = self._extract_joint_state(observation, robot_name, joints, index=None)
                pose = solver.forward_kinematics(joint_state)
                for axis, axis_name in enumerate(TASK_FRAME_AXIS_NAMES):
                    new_observation[f"{robot_name}.{axis_name}.ee_pos"] = float(pose[axis])

        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Leave feature specs unchanged."""
        return features

    @staticmethod
    def _extract_joint_state(
        observation: dict[str, Any],
        robot_name: str,
        joints: list[str],
        index: int | None,
    ) -> dict[str, float]:
        state: dict[str, float] = {}
        for joint_name in joints:
            key = f"{robot_name}.{joint_name}.pos"
            if key not in observation:
                raise ValueError(f"Missing joint observation key '{key}' for robot '{robot_name}'")
            value = observation[key]
            if isinstance(value, torch.Tensor):
                if index is None:
                    state[joint_name] = float(value.item()) if value.ndim == 0 else float(value[0].item())
                else:
                    state[joint_name] = float(value[index].item())
            else:
                state[joint_name] = float(value)
        return state


@dataclass
@ProcessorStepRegistry.register("relative_frame_observation")
class RelativeFrameObservationProcessor(ProcessorStep):
    """Re-express absolute EE pose channels relative to a per-episode reference pose."""

    enable: bool | dict[str, bool] = True

    _reference_pose: dict[str, list[float]] = field(default_factory=dict, init=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION)
        if not isinstance(observation, dict):
            return transition

        new_transition = transition.copy()
        new_observation = dict(observation)

        robot_names = self._robot_names(observation)
        for name in robot_names:
            if not self._enabled(name):
                continue
            try:
                pose = get_robot_pose_from_observation(observation, name)
            except KeyError:
                continue
            reference = self._reference_pose.setdefault(name, pose)

            # Positions are simple vector offsets; orientations are composed on SO(3).
            relative_position = [pose[i] - reference[i] for i in range(3)]
            pose_rot = rotation_from_extrinsic_xyz(*pose[3:6])
            ref_rot = rotation_from_extrinsic_xyz(*reference[3:6])
            relative_orientation = euler_xyz_from_rotation(pose_rot * ref_rot.inv())

            relative_pose = relative_position + relative_orientation
            for axis, axis_name in enumerate(TASK_FRAME_AXIS_NAMES):
                new_observation[f"{name}.{axis_name}.ee_pos"] = relative_pose[axis]

        new_transition[TransitionKey.OBSERVATION] = new_observation
        return new_transition

    def _enabled(self, name: str) -> bool:
        if isinstance(self.enable, dict):
            return bool(self.enable.get(name, False))
        return bool(self.enable)

    @staticmethod
    def _robot_names(observation: dict[str, Any]) -> set[str]:
        names: set[str] = set()
        for key in observation:
            if "." in key:
                names.add(key.split(".", 1)[0])
        return names

    def reset(self) -> None:
        self._reference_pose.clear()

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("joint_velocity_processor")
class JointVelocityProcessorStep(ObservationProcessorStep):
    """
    Calculates and appends joint velocity information to the observation state.

    This step computes the velocity of each joint by calculating the finite
    difference between the current and the last observed joint positions. The
    resulting velocity vector is then concatenated to the original state vector.

    Attributes:
        dt: The time step (delta time) in seconds between observations, used for
            calculating velocity.
        last_joint_positions: Stores the joint positions from the previous step
                              to enable velocity calculation.
    """

    enable: dict[str, bool]
    dt: float = 0.1
    last_joint_positions: dict[str, torch.Tensor] | None = None

    def observation(self, observation: dict) -> dict:
        """
        Computes joint velocities and adds them to the observation state.

        Args:
            observation: The input observation dictionary, expected to contain
                         an `observation.state` key with joint positions.

        Returns:
            A new observation dictionary with the `observation.state` tensor
            extended to include joint velocities.

        Raises:
            ValueError: If `observation.state` is not found in the observation.
        """
        new_observation = dict(observation)

        if any(self.enable.values()):
            # Get current joint positions (assuming they're in observation.state)
            current_positions = observation.get(OBS_STATE)
            if current_positions is None:
                raise ValueError(f"{OBS_STATE} is not in observation")

            # Initialize last joint positions if not already set
            if self.last_joint_positions is None:
                self.last_joint_positions = current_positions.clone()
                joint_velocities = torch.zeros_like(current_positions)
            else:
                # Compute velocities
                joint_velocities = (current_positions - self.last_joint_positions) / self.dt

            self.last_joint_positions = current_positions.clone()

            # Extend observation with velocities
            extended_state = torch.cat([current_positions, joint_velocities], dim=-1)

            # Create new observation dict
            new_observation[OBS_STATE] = extended_state

        return new_observation

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the time step `dt`.
        """
        return {
            "dt": self.dt,
            "enable": self.enable
        }

    def reset(self) -> None:
        """Resets the internal state, clearing the last known joint positions."""
        self.last_joint_positions = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the `observation.state` feature to reflect the added velocities.

        This method doubles the size of the first dimension of the `observation.state`
        shape to account for the concatenation of position and velocity vectors.

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary.
        """
        if OBS_STATE in features[PipelineFeatureType.OBSERVATION] and any(self.enable.values()):
            original_feature = features[PipelineFeatureType.OBSERVATION][OBS_STATE]
            # Double the shape to account for positions + velocities
            new_shape = (original_feature.shape[0] * 2,) + original_feature.shape[1:]

            features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                type=original_feature.type, shape=new_shape
            )
        return features


@dataclass
@ProcessorStepRegistry.register("current_processor")
class MotorCurrentProcessorStep(ObservationProcessorStep):
    """
    Reads motor currents from a robot and appends them to the observation state.

    This step queries the robot's hardware interface to get the present current
    for each motor and concatenates this information to the existing state vector.

    Attributes:
        robot: An instance of a `lerobot` Robot class that provides access to
               the hardware bus.
    """

    enable: dict[str, bool]
    robot_dict: dict[str, "Robot"] | None = None

    def observation(self, observation: dict) -> dict:
        """
        Fetches motor currents and adds them to the observation state.

        Args:
            observation: The input observation dictionary.

        Returns:
            A new observation dictionary with the `observation.state` tensor
            extended to include motor currents.

        Raises:
            ValueError: If the `robot` attribute has not been set.
        """
        # Get current values from robot state
        if self.robot_dict is None:
            raise ValueError("Robot is not set")

        current_state = observation.get(OBS_STATE)
        if current_state is None:
            return observation

        motor_currents = []
        for name, robot in self.robot_dict.items():
            if self.enable[name] and hasattr(robot, "bus") and hasattr(robot.bus, "motors"):
                present_current_dict = robot.bus.sync_read("Present_Current")  # type: ignore[attr-defined]
                motor_currents.extend([present_current_dict[name] for name in robot.bus.motors])

        motor_currents = torch.tensor(
            motor_currents,  # type: ignore[attr-defined]
            dtype=current_state.dtype,
            device=current_state.device
        )

        extended_state = torch.cat([current_state, motor_currents], dim=-1)

        # Create new observation dict
        new_observation = dict(observation)
        new_observation[OBS_STATE] = extended_state

        return new_observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the `observation.state` feature to reflect the added motor currents.

        This method increases the size of the first dimension of the `observation.state`
        shape by the number of motors in the robot.

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary.
        """
        if OBS_STATE in features[PipelineFeatureType.OBSERVATION] and self.robot_dict is not None:
            original_feature = features[PipelineFeatureType.OBSERVATION][OBS_STATE]
            # Add motor current dimensions to the original state shape
            num_motors = 0

            for name, robot in self.robot_dict.items():
                if self.enable[name] and hasattr(robot, "bus") and hasattr(robot.bus, "motors"):  # type: ignore[attr-defined]
                    num_motors += len(self.robot.bus.motors)  # type: ignore[attr-defined]

            if num_motors > 0:
                new_shape = (original_feature.shape[0] + num_motors,) + original_feature.shape[1:]
                features[PipelineFeatureType.OBSERVATION][OBS_STATE] = PolicyFeature(
                    type=original_feature.type, shape=new_shape
                )
        return features

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the time step `dt`.
        """
        return {
            "enable": self.enable
        }
