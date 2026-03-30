import math
from dataclasses import dataclass
from typing import Any, Literal

from draccus import ChoiceRegistry

from share.envs.manipulation_primitive.config_manipulation_primitive import PRIMITIVE_TARGET_POSE_INFO_KEY
from share.envs.utils import to_scalar, resolve_value, compare, axis_to_index
from share.teleoperators import TeleopEvents
from share.utils.transformation_utils import get_robot_pose_from_observation

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


@Transition.register_subclass("on_success")
@dataclass
class OnSuccess(Transition):
    success_key: str = TeleopEvents.SUCCESS

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        return Outcome(
            terminated=info.get(self.success_key, False),
            reward=self.additional_reward,
            reason="success" if self.reason is None else self.reason
        )


@Transition.register_subclass("on_observation_threshold")
@dataclass
class OnObservationThreshold(Transition):
    obs_key: str = ""
    threshold: float = 0.0
    operator: Literal["ge", "gt", "le", "lt", "eq", "ne"] = "ge"

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:
        value = to_scalar(resolve_value(obs, self.obs_key))
        fired = compare(value, self.threshold, self.operator)
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
    metric_key: str = "success"
    threshold: float = 0.5
    operator: Literal["ge", "gt", "le", "lt", "eq", "ne"] = "ge"
    additional_reward: float = 1.0

    def evaluate(self, obs: dict[str, Any], info: dict[str, Any]) -> Outcome:

        # todo: run the classifier here

        if self.metric_key in info:
            metric = resolve_value(info, self.metric_key)
        else:
            metric = resolve_value(obs, self.metric_key)

        value = to_scalar(metric)
        fired = compare(value, self.threshold, self.operator)
        return Outcome(
            terminated=fired,
            truncated=False,
            reward=self.additional_reward if fired else 0.0,
            reason="reward_classifier" if fired and self.reason is None else self.reason if fired else None,
        )


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


