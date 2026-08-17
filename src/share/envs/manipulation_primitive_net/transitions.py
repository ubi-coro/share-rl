import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Literal

from draccus import ChoiceRegistry

from share.envs.manipulation_primitive.config_manipulation_primitive import PRIMITIVE_TARGET_POSE_INFO_KEY
from share.envs.manipulation_primitive.task_frame import TASK_FRAME_AXIS_NAMES
from share.envs.utils import to_scalar, resolve_value, compare, axis_to_index
from share.teleoperators import TeleopEvents
from share.utils.constants import DEFAULT_ROBOT_NAME
from share.utils.transformation_utils import get_robot_pose_from_observation, task_pose_to_world_pose

logger = logging.getLogger(__name__)

DEFAULT_TARGET_POSE_AXES_INFO_KEY = "_primitive_target_pose_axes"


@dataclass
class Outcome:
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    reason: str | None = None


@dataclass
class Transition(ChoiceRegistry):
    source: str
    target: str

    additional_reward: float = 0.0
    reason: str | None = None

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        raise NotImplementedError

    def check(self, obs: dict, info: dict) -> bool:
        result = self.evaluate(obs=obs, info=info)
        return result.terminated or result.truncated


@Transition.register_subclass("always")
@dataclass
class Always(Transition):
    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        return Outcome(
            terminated=True,
            reward=self.additional_reward,
            reason="always" if self.reason is None else self.reason
        )


@Transition.register_subclass("on_event")
@dataclass
class OnEvent(Transition):
    """Fire when a named boolean flag is truthy in ``info``.

    The general operator-signal edge. A processor ``EventConfig`` maps a key or
    footswitch to a flag name -- a ``TeleopEvents`` member *or any plain string* --
    which ``AddKeyboardEventsAsInfoStep`` writes into ``info``; this edge routes on it.
    So an optional, operator-triggered branch (re-zero F/T, re-home, re-scan, ...) is
    pure graph data: name a flag, bind a key to it, and add an ``OnEvent`` edge -- no
    new event enum member and no bespoke code. ``OnSuccess``/``OnFailure`` are presets.
    """

    event_key: Any = ""
    default_reason: ClassVar[str] = "event"

    def _resolve_event_key(self) -> Any:
        return self.event_key

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        return Outcome(
            terminated=bool(info.get(self._resolve_event_key(), False)),
            reward=self.additional_reward,
            reason=self.default_reason if self.reason is None else self.reason,
        )


@Transition.register_subclass("on_success")
@dataclass
class OnSuccess(OnEvent):
    success_key: str = TeleopEvents.SUCCESS
    additional_reward: float = 1.0
    default_reason: ClassVar[str] = "success"

    def _resolve_event_key(self) -> Any:
        return self.success_key


@Transition.register_subclass("on_failure")
@dataclass
class OnFailure(OnEvent):
    failure_key: str = TeleopEvents.FAILURE
    additional_reward: float = 0.0
    default_reason: ClassVar[str] = "failure"

    def _resolve_event_key(self) -> Any:
        return self.failure_key


@Transition.register_subclass("on_observation_threshold")
@dataclass
class OnObservationThreshold(Transition):
    obs_key: str = ""
    threshold: float = 0.0
    operator: Literal["ge", "gt", "le", "lt", "eq", "ne"] = "ge"

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        value = to_scalar(resolve_value(obs, self.obs_key))
        fired = compare(value, self.threshold, self.operator)
        print(f"THRESHOLD {self.obs_key}: {value:.1f}", self.operator, self.threshold, "?", fired)
        return Outcome(
            terminated=fired,
            reward=self.additional_reward,
            reason="observation_threshold" if self.reason is None else self.reason
        )


@Transition.register_subclass("on_time_limit")
@dataclass
class OnTimeLimit(Transition):
    max_steps: int = 0
    step_key: str = "step"

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        current_steps = int(to_scalar(resolve_value(info, self.step_key)))
        fired = current_steps >= self.max_steps
        return Outcome(
            terminated=False,
            truncated=fired,
            reward=self.additional_reward,
            reason="time_limit" if self.reason is None else self.reason
        )


@Transition.register_subclass("reward_classifier")
@dataclass
class RewardClassifierTransition(Transition):
    """Fire when a learned reward classifier detects success on the current observation.

    Runs a pretrained lerobot reward classifier (trained with
    ``share/scripts/train_reward_classifier.py``) on the processed image observations
    (plus ``observation.state`` for the state-augmented variant, auto-detected from the
    checkpoint config). The model and its normalization preprocessor are loaded lazily
    and cached per (path, device), so several edges can share one classifier instance.
    """

    pretrained_path: str = ""
    threshold: float = 0.9
    operator: Literal["ge", "gt", "le", "lt", "eq", "ne"] = "ge"
    additional_reward: float = 1.0
    device: str = "cuda"
    # The stock classifier head assumes a 4x4 backbone feature map, i.e. 128x128 inputs.
    image_size: int = 128
    prob_info_key: str = "reward_classifier_prob"

    _CACHE: ClassVar[dict[tuple[str, str], tuple[Any, Any]]] = {}

    def _load(self) -> tuple[Any, Any]:
        key = (self.pretrained_path, self.device)
        if key not in self._CACHE:
            if not self.pretrained_path:
                raise ValueError("RewardClassifierTransition requires 'pretrained_path'.")
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
            from lerobot.processor import PolicyProcessorPipeline
            from lerobot.processor.converters import batch_to_transition, transition_to_batch

            from share.policies.reward_classifier import StateRewardClassifier

            config = PreTrainedConfig.from_pretrained(self.pretrained_path)
            policy_cls = StateRewardClassifier if config.type == "state_reward_classifier" else Classifier
            policy = policy_cls.from_pretrained(self.pretrained_path)
            policy.config.device = self.device
            policy.to(self.device)
            policy.eval()
            preprocessor = PolicyProcessorPipeline.from_pretrained(
                self.pretrained_path,
                config_filename="classifier_preprocessor.json",
                overrides={"device_processor": {"device": self.device}},
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            )
            self._CACHE[key] = (policy, preprocessor)
        return self._CACHE[key]

    def _success_probability(self, obs: dict[str, Any]) -> float:
        import torch

        from lerobot.utils.constants import OBS_IMAGE, OBS_STATE

        from share.policies.reward_classifier import StateRewardClassifier

        policy, preprocessor = self._load()
        image_keys = [key for key in policy.config.input_features if key.startswith(OBS_IMAGE)]
        needs_state = isinstance(policy, StateRewardClassifier)

        batch: dict[str, Any] = {}
        for key in image_keys + ([OBS_STATE] if needs_state else []):
            value = obs[key]
            value = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if key in image_keys:
                if value.dtype == torch.uint8:
                    value = value.float() / 255.0
                if value.dim() == 3:
                    value = value.unsqueeze(0)
            elif value.dim() == 1:
                value = value.unsqueeze(0)
            batch[key] = value

        with torch.no_grad():
            batch = preprocessor(batch)
            for key in image_keys:
                if batch[key].shape[-2:] != (self.image_size, self.image_size):
                    batch[key] = torch.nn.functional.interpolate(
                        batch[key],
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    )
            images = [batch[key] for key in image_keys]
            if needs_state:
                output = policy.predict(images, state=batch[OBS_STATE])
            else:
                output = policy.predict(images)
        return float(output.probabilities.reshape(-1)[0].item())

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        probability = self._success_probability(obs)
        info[self.prob_info_key] = probability
        fired = compare(probability, self.threshold, self.operator)
        return Outcome(
            terminated=fired,
            truncated=False,
            reward=self.additional_reward if fired else 0.0,
            reason="reward_classifier" if fired and self.reason is None else self.reason if fired else None,
        )


@Transition.register_subclass("on_info_equals")
@dataclass
class OnInfoEquals(Transition):
    """Fire when one ``info`` entry equals a fixed value.

    Generic building block for branching on categorical state a primitive publishes
    through its ``info`` payload (e.g. which of several outcomes it resolved to),
    without needing a dedicated Transition subclass per category.
    """
    key: str = ""
    value: Any = None

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        fired = resolve_value(info, self.key) == self.value
        return Outcome(
            terminated=fired,
            reward=self.additional_reward if fired else 0.0,
            reason="info_equals" if fired and self.reason is None else self.reason if fired else None,
        )


@Transition.register_subclass("all_of")
@dataclass
class AllOf(Transition):
    """Fire only when every nested condition fires this step.

    Generic AND-combinator so two independent conditions (e.g. "pose reached" and
    "this categorical outcome was selected") can gate a single edge without a
    bespoke Transition subclass per combination. Nested conditions' own
    ``source``/``target``/``reward`` are ignored; only ``evaluate()`` is used.
    """
    conditions: list[Transition] = field(default_factory=list)

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        outcomes = [condition.evaluate(obs=obs, info=info) for condition in self.conditions]
        fired = bool(outcomes) and all(o.terminated or o.truncated for o in outcomes)
        return Outcome(
            terminated=fired,
            reward=self.additional_reward if fired else 0.0,
            reason="all_of" if fired and self.reason is None else self.reason if fired else None,
        )


@Transition.register_subclass("record_world_pose")
@dataclass
class RecordWorldPoseTransition(Transition):
    """Wrap another transition; whenever it fires, append the current world EE pose to a JSONL file.

    The pose channels in the observation are expressed in the controller's active task
    frame, alongside the ``task_frame_origin`` channels that name that frame (both come
    from the same controller state sample), so the world pose is reconstructed exactly
    even while a task-space primitive is active. Used to collect plugged-pose samples
    for ``share/scripts/compute_resample_volume.py``.

    Like ``AllOf``, the nested condition's own ``source``/``target``/``reward`` are
    ignored; this wrapper's edge fields and the condition's ``Outcome`` are what count.
    """

    condition: Transition | None = None
    output_file: str = ""
    robot_name: str = DEFAULT_ROBOT_NAME

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        outcome = self.condition.evaluate(obs=obs, info=info)
        if outcome.terminated or outcome.truncated:
            world_pose = self._world_pose(obs)
            record = {"pose": world_pose, "time": datetime.now().isoformat(timespec="seconds")}
            with open(self.output_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            logger.info(f"Recorded world pose {[round(v, 4) for v in world_pose]} to {self.output_file}")
        return outcome

    def _world_pose(self, obs: dict[str, Any]) -> list[float]:
        task_pose = get_robot_pose_from_observation(obs, self.robot_name)
        origin_keys = [f"{self.robot_name}.{axis}.task_frame_origin" for axis in TASK_FRAME_AXIS_NAMES]
        origin = [float(obs[key]) for key in origin_keys] if all(key in obs for key in origin_keys) else None
        return task_pose_to_world_pose(task_pose, origin)


@Transition.register_subclass("on_target_pose_reached")
@dataclass
class OnTargetPoseReached(Transition):
    robot_name: str | None = None
    axes: list[int | str] | None = None
    tolerance: float | list[float] = 0.01
    target_key: str = PRIMITIVE_TARGET_POSE_INFO_KEY

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        """Check whether the current EE pose has reached the target pose.

        Args:
            obs: Processed observation dictionary. The current EE pose is read
                from here using the shared observation-pose utility.
            info: Processed info dictionary. The target pose is read from
                ``target_key``.

        Returns:
            ``Outcome`` indicating whether the pose condition fired.
        """
        targets = resolve_value(info, self.target_key)
        robot_names = [self.robot_name] if self.robot_name is not None else sorted(targets)
        fired = bool(robot_names)
        for robot_name in robot_names:
            current_pose = get_robot_pose_from_observation(obs, robot_name)
            target_pose = [float(v) for v in targets[robot_name]]
            # print(f"TARGET POSE {robot_name}: {[round(v, 4) for v in target_pose]}"
            #       f" \nCURRENT POSE: {[round(v, 4) for v in current_pose]}")
            axes = self._resolved_axes(info=info, robot_name=robot_name)
            tolerances = self._resolved_tolerances()
            for axis in axes:
                error = current_pose[axis] - target_pose[axis]
                if axis >= 3:
                    error = math.atan2(math.sin(error), math.cos(error))
                if abs(error) > tolerances[axis]:
                    fired = False
                    break
            if not fired:
                break

        return Outcome(
            terminated=fired,
            reward=self.additional_reward if fired else 0.0,
            reason="target_pose_reached" if fired and self.reason is None else self.reason if fired else None,
        )

    def _resolved_axes(self, info: dict[str, Any], robot_name: str) -> list[int]:
        if self.axes is not None:
            return [axis_to_index(axis) for axis in self.axes]
        inferred_axes = info.get(DEFAULT_TARGET_POSE_AXES_INFO_KEY, {})
        if robot_name in inferred_axes and inferred_axes[robot_name]:
            return [int(axis) for axis in inferred_axes[robot_name]]
        return [0, 1, 2, 3, 4, 5]

    def _resolved_tolerances(self) -> list[float]:
        if isinstance(self.tolerance, (int, float)):
            return [float(self.tolerance)] * 6
        if len(self.tolerance) != 6:
            raise ValueError("OnTargetPoseReached.tolerance must be a scalar or length-6 list.")
        return [float(v) for v in self.tolerance]


