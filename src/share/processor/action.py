import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from lerobot.types import EnvTransition, TransitionKey
from scipy.spatial.transform import Rotation
from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY, GRIPPER_KEY
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry

from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    PolicyMode,
    TaskFrame,
    TASK_FRAME_AXIS_NAMES
)
from share.envs.utils import check_delta_teleoperator
from share.utils.transformation_utils import rotation_from_extrinsic_xyz, rotation_component_keys
from share.processor.utils import policy_action_keys_for_robot, flatten_nested_policy_action
from share.teleoperators.utils import TeleopEvents

for _registry_name in (
    "to_nested_action",
    "match_teleop_to_policy_action",
    "task_frame_intervention_action_processor",
    "discretize_gripper_processor",
    "to_joint_action_processor",
    "relative_frame_action",
):
    ProcessorStepRegistry.unregister(_registry_name)

@dataclass
@ProcessorStepRegistry.register("to_nested_action")
class ToNestedActionProcessorStep(ProcessorStep):
    """Convert the flat policy action tensor into a per-robot keyed dict."""

    task_frame: dict[str, TaskFrame] = field(default_factory=dict)
    gripper_enable: dict[str, bool] = field(default_factory=dict)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, dict):
            return transition

        action_tensor = torch.as_tensor(action)
        nested_action: dict[str, dict[str, torch.Tensor]] = {}
        idx = 0
        for name, frame in self.task_frame.items():
            robot_action: dict[str, torch.Tensor] = {}
            for key in policy_action_keys_for_robot(frame, self.gripper_enable[name]):
                if idx >= action_tensor.numel():
                    raise ValueError("Policy action tensor is shorter than expected for the configured action schema")
                robot_action[key] = action_tensor[idx]
                idx += 1
            nested_action[name] = robot_action

        if idx != action_tensor.numel():
            raise ValueError("Policy action tensor has trailing values beyond the configured action schema")

        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = nested_action
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("match_teleop_to_policy_action")
class MatchTeleopToPolicyActionProcessorStep(ProcessorStep):
    """Map raw teleop commands into the keyed policy learning-space action format."""

    teleoperators: dict[str, Any] = field(default_factory=dict)
    task_frame: dict[str, TaskFrame] = field(default_factory=dict)
    kinematics: dict[str, Any] = field(default_factory=dict)
    use_virtual_reference: dict[str, bool] = field(default_factory=dict)
    gripper_enable: dict[str, bool] = field(default_factory=dict)

    _is_delta_teleoperator: dict[str, bool] = field(default_factory=dict, init=False)
    _virtual_task_pose: dict[str, list[float]] = field(default_factory=dict, init=False)
    _virtual_joint_target: dict[str, dict[str, float]] = field(default_factory=dict, init=False)
    _prev_fk_pose: dict[str, list[float]] = field(default_factory=dict, init=False)
    _prev_joint_state: dict[str, dict[str, float]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._is_delta_teleoperator = check_delta_teleoperator(self.teleoperators)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        teleop_action_dict = complementary_data.get(TELEOP_ACTION_KEY)
        if not isinstance(teleop_action_dict, dict):
            return new_transition

        converted_actions: dict[str, dict[str, float]] = {}
        for name, teleop_action in teleop_action_dict.items():
            frame = self.task_frame.get(name)
            if frame is None:
                continue

            if self._is_delta_teleoperator.get(name, False):
                converted = self._map_delta_teleop(name, frame, teleop_action, transition)
            else:
                converted = self._map_absolute_joint_teleop(name, frame, teleop_action)

            if self.gripper_enable[name] and f"{GRIPPER_KEY}.pos" in teleop_action:
                converted[f"{GRIPPER_KEY}.pos"] = teleop_action[f"{GRIPPER_KEY}.pos"]

            converted_actions[name] = converted

        complementary_data[TELEOP_ACTION_KEY] = converted_actions
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def _map_delta_teleop(self, name: str, frame: TaskFrame, teleop_action: Any, transition: EnvTransition) -> dict[str, float]:
        deltas = self._extract_delta_action(teleop_action)

        # ik to get from integrated deltas to joints
        if frame.space == ControlSpace.JOINT:
            solver = self._require_solver(name)
            base_pose = self._integration_base_pose(name, frame, transition)
            pose_target = [base_pose[i] + deltas[i] for i in range(6)]
            joint_target = solver.inverse_kinematics(pose_target)
            base_joint_state = self._integration_base_joint_state(name, frame, transition)
            encoded: dict[str, float] = {}
            for axis in frame.learnable_axis_indices:
                joint_name = frame.action_key_for_axis(axis)
                joint_value = joint_target[joint_name]
                if frame.policy_mode[axis] == PolicyMode.RELATIVE:
                    joint_value -= float(base_joint_state.get(joint_name, joint_value))
                encoded[frame.action_key_for_axis(axis)] = joint_value

            # update virtual
            if any(frame.policy_mode[axis] == PolicyMode.ABSOLUTE for axis in frame.learnable_axis_indices):
                learnable_joints = [frame.action_key_for_axis(axis) for axis in frame.learnable_axis_indices]
                self._virtual_joint_target[name] = {name: joint_target[name] for name in learnable_joints}

            return encoded

        source_pose = deltas
        if any(
            frame.control_mode[axis] == ControlMode.POS and frame.policy_mode[axis] == PolicyMode.ABSOLUTE
            for axis in frame.learnable_axis_indices
        ):
            base_pose = self._integration_base_pose(name, frame, transition)
            source_pose = [base_pose[i] + deltas[i] for i in range(6)]
            self._virtual_task_pose[name] = source_pose

        return self._encode_learning_space(frame, source_pose)

    def _map_absolute_joint_teleop(self, name: str, frame: TaskFrame, teleop_action: Any) -> dict[str, float]:
        joint_state = self._extract_joint_action(teleop_action)
        if frame.space == ControlSpace.JOINT:
            prev_joint_state = self._prev_joint_state.get(name, joint_state)
            self._prev_joint_state[name] = dict(joint_state)
            encoded: dict[str, float] = {}
            for axis in frame.learnable_axis_indices:
                joint_name = frame.action_key_for_axis(axis)
                joint_value = float(joint_state[joint_name])
                if frame.policy_mode[axis] == PolicyMode.RELATIVE:
                    joint_value -= float(prev_joint_state.get(joint_name, joint_value))
                encoded[joint_name] = joint_value
            return encoded

        solver = self._require_solver(name)
        pose = solver.forward_kinematics(joint_state)
        prev_pose = self._prev_fk_pose.get(name, pose)
        self._prev_fk_pose[name] = pose

        source = []
        for axis in range(6):
            if frame.policy_mode[axis] == PolicyMode.RELATIVE:
                source.append(pose[axis] - prev_pose[axis])
            else:
                source.append(pose[axis])

        return self._encode_learning_space(frame, source)

    def _encode_learning_space(self, frame: TaskFrame, source_pose: list[float]) -> dict[str, float]:
        values: list[float] = []
        absolute_rot_axes = [
            axis for axis in frame.learnable_axis_indices if frame.is_absolute_rotation_axis(axis)
        ]

        for axis in frame.learnable_axis_indices:
            control_mode = frame.control_mode[axis]
            policy_mode = frame.policy_mode[axis]
            if axis in absolute_rot_axes:
                continue
            if control_mode in {ControlMode.VEL, ControlMode.WRENCH}:
                values.append(source_pose[axis])
            elif axis < 3 or policy_mode == PolicyMode.RELATIVE:
                values.append(source_pose[axis])

        if absolute_rot_axes:
            rot = [source_pose[3], source_pose[4], source_pose[5]]
            if len(absolute_rot_axes) == 1:
                angle = rot[absolute_rot_axes[0] - 3]
                values.extend([math.cos(angle), math.sin(angle)])
            elif len(absolute_rot_axes) == 2:
                matrix = rotation_from_extrinsic_xyz(*rot).as_matrix()
                values.extend(matrix[:, 0].tolist())
            else:
                matrix = rotation_from_extrinsic_xyz(*rot).as_matrix()
                values.extend(np.concatenate([matrix[:, 0], matrix[:, 1]]).tolist())

        keys = frame.policy_action_keys()
        if len(keys) != len(values):
            raise ValueError("Learning-space key/value mismatch while encoding teleop action")
        return {key: float(value) for key, value in zip(keys, values, strict=True)}

    def _integration_base_pose(self, name: str, frame: TaskFrame, transition: EnvTransition) -> list[float]:
        use_virtual = self.use_virtual_reference[name] if isinstance(self.use_virtual_reference, dict) else self.use_virtual_reference
        if use_virtual and name in self._virtual_task_pose:
            return self._virtual_task_pose[name]

        observation = transition.get(TransitionKey.OBSERVATION)
        if isinstance(observation, dict):
            obs_pose = []
            for axis_name in TASK_FRAME_AXIS_NAMES:
                key = f"{name}.{axis_name}.ee_pos"
                if key not in observation:
                    obs_pose = []
                    break
                obs_pose.append(observation[key])
            if len(obs_pose) == 6:
                return obs_pose

        return list(frame.target)

    def _integration_base_joint_state(self, name: str, frame: TaskFrame, transition: EnvTransition) -> dict[str, float]:
        use_virtual = self.use_virtual_reference[name] if isinstance(self.use_virtual_reference, dict) else self.use_virtual_reference
        if use_virtual and name in self._virtual_joint_target:
            return dict(self._virtual_joint_target[name])

        observation = transition.get(TransitionKey.OBSERVATION)
        if isinstance(observation, dict):
            joint_state: dict[str, float] = {}
            for joint_name in frame.joint_names:
                key = f"{name}.{joint_name}.pos"
                if key in observation:
                    joint_state[joint_name] = observation[key]
            if joint_state:
                return joint_state

        return dict(self._prev_joint_state.get(name, {}))

    def _require_solver(self, name: str) -> Any:
        solver = self.kinematics.get(name)
        if solver is None:
            raise ValueError(f"Missing kinematics solver for '{name}'")
        return solver

    @staticmethod
    def _extract_delta_action(teleop_action: Any) -> list[float]:
        if isinstance(teleop_action, dict):
            return [
                float(teleop_action.get("delta_x", teleop_action.get("x.vel", 0.0))),
                float(teleop_action.get("delta_y", teleop_action.get("y.vel", 0.0))),
                float(teleop_action.get("delta_z", teleop_action.get("z.vel", 0.0))),
                float(teleop_action.get("delta_rx", teleop_action.get("wx.vel", 0.0))),
                float(teleop_action.get("delta_ry", teleop_action.get("wy.vel", 0.0))),
                float(teleop_action.get("delta_rz", teleop_action.get("wz.vel", 0.0))),
            ]
        return [float(v) for v in teleop_action][:6]

    @staticmethod
    def _extract_joint_action(teleop_action: dict) -> dict[str, float]:
        joint_state: dict[str, float] = {}
        for key, value in teleop_action.items():
            if key.endswith(".q"):
                key = key.removesuffix(".q")
            if "." not in key:
                key = f"{key}.pos"
            joint_state[key] = float(value)
        return joint_state

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("task_frame_intervention_action_processor")
class InterventionActionProcessorStep(ProcessorStep):
    """Merge keyed learning-space actions and project them into full robot actions."""

    teleoperators: dict[str, Any] = field(default_factory=dict)
    task_frame: dict[str, TaskFrame] = field(default_factory=dict)
    gripper_enable: bool | dict[str, bool] = False
    gripper_static_pos: float | dict[str, float] = 0.0

    def __post_init__(self):
        self._disable_torque_on_intervention = {name: hasattr(teleop, "bus") for name, teleop in self.teleoperators.items()}
        self._intervention_occurred = False

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        policy_actions = transition.get(TransitionKey.ACTION)
        if not isinstance(policy_actions, dict):
            raise TypeError(f"Action should be a dict, got {type(policy_actions)}")

        new_transition = transition.copy()
        info = dict(new_transition.get(TransitionKey.INFO) or {})
        complementary_data = dict(new_transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        teleop_action_dict = complementary_data.get(TELEOP_ACTION_KEY)
        is_intervention = bool(info.get(TeleopEvents.IS_INTERVENTION, False))

        self._intervention_occurred = self._intervention_occurred | is_intervention
        if self._intervention_occurred and not is_intervention:
            info[TeleopEvents.INTERVENTION_COMPLETED] = True

        if is_intervention and isinstance(teleop_action_dict, dict):
            source_actions = teleop_action_dict
            for name in teleop_action_dict:
                if self._disable_torque_on_intervention.get(name, False):
                    self.teleoperators[name].disable_torque()
        else:
            source_actions = policy_actions
            if self._intervention_occurred:
                for name, teleop in self.teleoperators.items():
                    if self._disable_torque_on_intervention.get(name, False):
                        teleop.enable_torque()
            else:
                for teleop_name, teleop in self.teleoperators.items():
                    teleop.send_feedback(self._map_to_teleop_action(source_actions, teleop_name))

        full_action_dict: dict[str, dict[str, float]] = {}
        for name, frame in self.task_frame.items():
            full_action = self._project_policy_action(frame, source_actions[name])
            if self.gripper_enable[name] and f"{GRIPPER_KEY}.pos" in source_actions[name]:
                full_action[f"{GRIPPER_KEY}.pos"] = source_actions[name][f"{GRIPPER_KEY}.pos"]
            full_action_dict[name] = full_action

        complementary_data[TELEOP_ACTION_KEY] = flatten_nested_policy_action(
            source_actions,
            task_frame=self.task_frame,
            gripper_enable=self.gripper_enable,
            like=policy_actions,
        )
        new_transition[TransitionKey.ACTION] = full_action_dict
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        new_transition[TransitionKey.INFO] = info
        return new_transition

    def _project_policy_action(self, frame: TaskFrame, encoded_action: dict[str, Any]) -> dict[str, float]:
        full_target = {
            frame.action_key_for_axis(axis): float(frame.target[axis])
            for axis in range(len(frame.target))
        }
        absolute_rot_axes = [axis for axis in frame.learnable_axis_indices if frame.is_absolute_rotation_axis(axis)]

        for axis in frame.learnable_axis_indices:
            if axis in absolute_rot_axes:
                continue
            key = frame.action_key_for_axis(axis)
            if key not in encoded_action:
                raise ValueError(f"Missing learning-space action key '{key}' for task-frame projection")
            full_target[key] = float(encoded_action[key])

        if absolute_rot_axes:
            rotation_keys = rotation_component_keys(frame, absolute_rot_axes)
            rotation_raw = []
            for key in rotation_keys:
                if key not in encoded_action:
                    raise ValueError(f"Missing rotation learning-space key '{key}' for task-frame projection")
                rotation_raw.append(encoded_action[key])
            rotation_values, _ = self._decode_absolute_rotation(absolute_rot_axes, rotation_raw)
            for axis in absolute_rot_axes:
                full_target[frame.action_key_for_axis(axis)] = float(rotation_values[axis - 3])

        return full_target

    def _map_to_teleop_action(self, policy_action: dict[str, dict[str, Any]], name: str) -> dict[str, float]:
        if name not in policy_action or name not in self.teleoperators:
            return {}

        action_features = getattr(self.teleoperators[name], "action_features", {})
        if isinstance(action_features, dict) and isinstance(action_features.get("names"), dict):
            feature_names = list(action_features["names"].keys())
        elif isinstance(action_features, dict):
            feature_names = list(action_features.keys())
        else:
            feature_names = []

        aliases = {
            "delta_x": "x.vel",
            "delta_y": "y.vel",
            "delta_z": "z.vel",
            "delta_rx": "rx.vel",
            "delta_ry": "ry.vel",
            "delta_rz": "rz.vel",
            "gripper": f"{GRIPPER_KEY}.pos",
        }
        robot_action = policy_action[name]
        teleop_action: dict[str, float] = {}
        for feature_name in feature_names:
            key = aliases.get(feature_name, feature_name)
            if key in robot_action:
                teleop_action[feature_name] = robot_action[key]
        return teleop_action

    def _decode_absolute_rotation(self, absolute_rot_axes: list[int], raw: list[float]) -> tuple[list[float], int]:
        rot = [0.0, 0.0, 0.0]
        if len(absolute_rot_axes) == 1:
            if len(raw) < 2:
                raise ValueError("S1 rotation representation requires 2 values")
            rot[absolute_rot_axes[0] - 3] = math.atan2(raw[1], raw[0])
            return rot, 2
        if len(absolute_rot_axes) == 2:
            if len(raw) < 3:
                raise ValueError("S2 rotation representation requires 3 values")
            direction = np.asarray(raw[:3], dtype=float)
            norm = np.linalg.norm(direction)
            direction = np.array([1.0, 0.0, 0.0], dtype=float) if norm < 1e-8 else direction / norm
            reference = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(reference, direction))) > 0.95:
                reference = np.array([0.0, 1.0, 0.0], dtype=float)
            col1 = direction
            col2 = np.cross(reference, col1)
            col2_norm = np.linalg.norm(col2)
            col2 = np.array([0.0, 1.0, 0.0], dtype=float) if col2_norm < 1e-8 else col2 / col2_norm
            col3 = np.cross(col1, col2)
            matrix = np.column_stack([col1, col2, col3])
            rx, ry, rz = Rotation.from_matrix(matrix).as_euler("xyz", degrees=False)
            return [float(rx), float(ry), float(rz)], 3
        if len(absolute_rot_axes) == 3:
            if len(raw) < 6:
                raise ValueError("SO(3) 6D representation requires 6 values")
            matrix = self._rotation_6d_to_matrix(raw[:6])
            euler = Rotation.from_matrix(matrix).as_euler("xyz", degrees=False)
            return euler.tolist(), 6
        raise ValueError(f"Expected 1..3 absolute rotation axes, got {len(absolute_rot_axes)}")

    @staticmethod
    def _rotation_6d_to_matrix(raw: list[float]) -> list[list[float]]:
        a1 = np.asarray(raw[:3], dtype=float)
        a2 = np.asarray(raw[3:6], dtype=float)

        def normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
            n = float(np.linalg.norm(v))
            if n < 1e-8:
                return fallback
            return v / n

        b1 = normalize(a1, np.array([1.0, 0.0, 0.0], dtype=float))
        u2 = a2 - float(np.dot(a2, b1)) * b1
        fallback = np.array([0.0, 1.0, 0.0], dtype=float) if abs(float(b1[0])) > 0.9 else np.array([1.0, 0.0, 0.0], dtype=float)
        b2 = normalize(u2, normalize(fallback - float(np.dot(fallback, b1)) * b1, np.array([0.0, 1.0, 0.0], dtype=float)))
        b3 = np.cross(b1, b2)
        return np.column_stack([b1, b2, b3]).tolist()

    def _resolved_gripper_static_pos(self, name: str) -> float:
        if isinstance(self.gripper_static_pos, dict):
            return float(self.gripper_static_pos.get(name, 0.0))
        return float(self.gripper_static_pos)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def reset(self) -> None:
        self._intervention_occurred = False


@dataclass
@ProcessorStepRegistry.register("discretize_gripper_processor")
class DiscretizeGripperProcessorStep(ProcessorStep):
    """Discretize gripper actions using a per-robot internal gripper state."""

    discretize: dict[str, bool] = field(default_factory=dict)
    min_pos: dict[str, float] = field(default_factory=dict)
    max_pos: dict[str, float] = field(default_factory=dict)
    threshold: dict[str, float] = field(default_factory=dict)
    mode: dict[str, Literal["state", "pulse"]] = field(default_factory=dict)

    _robot_names: list[str] = field(default_factory=list, init=False)
    _gripper_state: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        all_robot_keys = set(self.discretize) | set(self.min_pos) | set(self.max_pos) | set(self.threshold) | set(self.mode)
        self._robot_names = sorted(all_robot_keys)
        self.reset()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, dict):
            return transition

        new_transition = transition.copy()
        new_action: dict[str, dict[str, Any]] = {}
        for name, robot_action in action.items():
            if not isinstance(robot_action, dict) or not self.discretize.get(name, False):
                new_action[name] = robot_action
                continue

            robot_action_out = dict(robot_action)
            if f"{GRIPPER_KEY}.pos" not in robot_action_out:
                new_action[name] = robot_action_out
                continue

            # initialize gripper state
            if name not in self._gripper_state:
                self._gripper_state[name] = self.min_pos.get(name, 0.0)
                if name not in self._robot_names:
                    self._robot_names.append(name)

            # update gripper state
            input_val = robot_action_out[f"{GRIPPER_KEY}.pos"]
            mode = self.mode.get(name, "state")
            threshold = self.threshold.get(name, 0.5)
            min_pos = self.min_pos.get(name, 0.0)
            max_pos = self.max_pos.get(name, 1.0)
            if mode == "pulse":
                if input_val > threshold:
                    self._gripper_state[name] = max_pos
                elif input_val < -threshold:
                    self._gripper_state[name] = min_pos
            elif mode == "state":
                if input_val > threshold:
                    self._gripper_state[name] = max_pos
                elif input_val < threshold:
                    self._gripper_state[name] = min_pos
            else:
                raise ValueError(f"Unsupported gripper discretization mode '{mode}'")

            robot_action_out[f"{GRIPPER_KEY}.pos"] = float(self._gripper_state[name])
            new_action[name] = robot_action_out

        new_transition[TransitionKey.ACTION] = new_action
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "discretize": self.discretize,
            "min_pos": self.min_pos,
            "max_pos": self.max_pos,
            "threshold": self.threshold,
            "mode": self.mode,
        }

    def reset(self) -> None:
        self._gripper_state = {name: self.min_pos.get(name, 0.0) for name in self._robot_names}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("to_joint_action_processor")
class ToJointActionProcessorStep(ProcessorStep):
    """Convert nested task-frame robot actions into nested joint robot actions when needed."""

    is_task_frame_robot: dict[str, bool] = field(default_factory=dict)
    task_frame: dict[str, TaskFrame] = field(default_factory=dict)
    kinematics: dict[str, Any] = field(default_factory=dict)
    use_virtual_reference: bool | dict[str, bool] = True

    _virtual_task_pose: dict[str, list[float]] = field(default_factory=dict, init=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Convert task-space robot action dicts to joint-space robot action dicts."""
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, dict):
            return transition

        new_transition = transition.copy()
        joint_action: dict[str, dict[str, float]] = {}

        for name, robot_action in action.items():
            frame = self.task_frame.get(name)
            if frame is None:
                joint_action[name] = dict(robot_action) if isinstance(robot_action, dict) else robot_action
                continue

            if self.is_task_frame_robot.get(name, False):
                joint_action[name] = dict(robot_action)
                continue

            if not isinstance(robot_action, dict):
                raise TypeError(f"Task-frame action for '{name}' must be a dict, got {type(robot_action)}")

            task_target = self._task_target_from_action(name, frame, robot_action)
            absolute_target = self._integrate_relative_axes(name, frame, task_target, transition)
            bounded_target = self._clamp_target(frame, absolute_target)

            solver = self.kinematics.get(name)
            if solver is None:
                raise ValueError(f"Missing kinematics solver for joint-only robot '{name}'")

            try:
                ik_solution = solver.inverse_kinematics(bounded_target)
            except Exception as exc:  # pragma: no cover - exercised with mock solver in unit tests
                raise ValueError(f"IK failed for '{name}': {exc}") from exc

            robot_joint_action: dict[str, float] = {}
            for joint_name in frame.joint_names:
                if joint_name not in ik_solution:
                    raise ValueError(f"IK solution for '{name}' missing joint '{joint_name}'")
                robot_joint_action[f"{joint_name}.pos"] = float(ik_solution[joint_name])

            if f"{GRIPPER_KEY}.pos" in robot_action:
                robot_joint_action[f"{GRIPPER_KEY}.pos"] = robot_action[f"{GRIPPER_KEY}.pos"]

            joint_action[name] = robot_joint_action
            self._virtual_task_pose[name] = bounded_target

        new_transition[TransitionKey.ACTION] = joint_action
        return new_transition

    def _task_target_from_action(self, name: str, frame: TaskFrame, robot_action: dict[str, Any]) -> list[float]:
        task_target: list[float] = []
        for axis in range(len(frame.target)):
            key = frame.action_key_for_axis(axis)
            if key not in robot_action:
                raise ValueError(f"Missing task-frame action key '{name}.{key}' for joint conversion")
            task_target.append(robot_action[key])
        return task_target

    def _integrate_relative_axes(
        self,
        name: str,
        frame: TaskFrame,
        task_target: list[float],
        transition: EnvTransition,
    ) -> list[float]:
        """Integrate relative POS axes on top of the current/base task pose."""
        base_pose = self._base_pose(name, frame, transition)
        out = list(task_target)
        for axis in frame.learnable_axis_indices:
            if frame.control_mode[axis] == ControlMode.POS and frame.policy_mode[axis] == PolicyMode.RELATIVE:
                out[axis] = base_pose[axis] + task_target[axis]
        return out

    def _base_pose(self, name: str, frame: TaskFrame, transition: EnvTransition) -> list[float]:
        """Resolve integration base pose from virtual state, observation, or default target."""
        use_virtual = self.use_virtual_reference[name] if isinstance(self.use_virtual_reference, dict) else self.use_virtual_reference
        if use_virtual and name in self._virtual_task_pose:
            return list(self._virtual_task_pose[name])

        observation = transition.get(TransitionKey.OBSERVATION)
        if isinstance(observation, dict):
            obs_pose = []
            for axis_name in TASK_FRAME_AXIS_NAMES:
                key = f"{name}.{axis_name}.ee_pos"
                if key not in observation:
                    obs_pose = []
                    break
                obs_pose.append(observation[key])
            if len(obs_pose) == 6:
                return obs_pose

        return list(frame.target)

    @staticmethod
    def _clamp_target(frame: TaskFrame, target: list[float]) -> list[float]:
        """Clamp task target to configured min/max bounds when available."""
        if frame.min_target is None or frame.max_target is None:
            return target
        return [max(frame.min_target[i], min(frame.max_target[i], target[i])) for i in range(len(target))]

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Leave feature specs unchanged."""
        return features


@dataclass
@ProcessorStepRegistry.register("relative_frame_action")
class RelativeFrameActionProcessor(ProcessorStep):
    """Pass-through placeholder for relative action transforms (currently identity)."""

    enable: bool | dict[str, bool] = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, dict):
            return transition
        if not any(self._enabled(name) for name in action):
            return transition

        # Current implementation intentionally no-ops numerically for kinematic axis channels.
        # It preserves gripper/non-kinematic channels exactly and remains invertible.
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = dict(action)
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Leave feature specs unchanged."""
        return features

    def _enabled(self, name: str) -> bool:
        if isinstance(self.enable, dict):
            return bool(self.enable.get(name, False))
        return bool(self.enable)
