import copy
import types
from typing import get_origin, Union, get_args, Any

import numpy as np
from lerobot.configs.types import PolicyFeature, FeatureType
from lerobot.processor.hil_processor import GRIPPER_KEY
from lerobot.utils.constants import REWARD, DONE

from share.envs.manipulation_primitive.task_frame import ControlSpace
from share.utils.transformation_utils import get_robot_pose_from_observation, task_pose_to_world_pose, world_pose_to_task_pose

DELTA_TRANSLATION_ACTION_NAMES = {
    "delta_x",
    "delta_y",
    "delta_z",
    "x.vel",
    "y.vel",
    "z.vel",
}
DELTA_ROTATION_ACTION_NAMES = {
    "delta_rx",
    "delta_ry",
    "delta_rz",
    "rx.vel",
    "ry.vel",
    "rz.vel",
    "wx.vel",
    "wy.vel",
    "wz.vel",
}
DELTA_AUXILIARY_ACTION_NAMES = {
    GRIPPER_KEY,
    f"{GRIPPER_KEY}.pos",
}
DELTA_ACTION_NAMES = (
    DELTA_TRANSLATION_ACTION_NAMES |
    DELTA_ROTATION_ACTION_NAMES |
    DELTA_AUXILIARY_ACTION_NAMES
)


def check_task_frame_robot(robot_dict: dict[str, "Robot"]):
    is_task_frame_robot = {}
    for name, r in robot_dict.items():
        is_task_frame_robot[name] = hasattr(r, "set_task_frame")

    return is_task_frame_robot


def get_teleoperator_action_names(teleoperator: "Teleoperator") -> set[str]:
    action_features = getattr(teleoperator, "action_features", {})
    if not isinstance(action_features, dict):
        return set()

    feature_names = action_features.get("names")
    if isinstance(feature_names, dict):
        return {str(name) for name in feature_names}

    return {str(name) for name in action_features}


def check_delta_teleoperator(teleop_dict: dict[str, "Teleoperator"]):
    is_delta_teleoperator = {}
    for name, t in teleop_dict.items():
        action_names = get_teleoperator_action_names(t)
        is_delta_teleoperator[name] = (
            bool(action_names) and
            action_names.issubset(DELTA_ACTION_NAMES) and
            bool(action_names & (DELTA_TRANSLATION_ACTION_NAMES | DELTA_ROTATION_ACTION_NAMES))
        )

    return is_delta_teleoperator


def is_union_with_dict(field_type) -> bool:
    origin = get_origin(field_type)
    if origin is types.UnionType or origin is Union:
        return any(get_origin(arg) is dict for arg in get_args(field_type))
    return False


def env_to_dataset_features(env_features: dict[str, PolicyFeature]) -> dict:
    ds_features = {}
    for key, ft in env_features.items():
        new_ft = {"shape": ft.shape}
        if ft.type == FeatureType.VISUAL:
            new_ft["dtype"] = "video"
            new_ft["names"] = ["channels", "height", "width"]
        else:
            new_ft["dtype"] = "float32"
            new_ft["names"] = None
        ds_features[key] = new_ft

    ds_features[REWARD] = {"dtype": "float32", "shape": (1,), "names": None}
    ds_features[DONE] = {"dtype": "bool", "shape": (1,), "names": None}
    return ds_features


def copy_per_robot(value: Any, robot_names: list[str]) -> dict[str, Any]:
    if isinstance(value, dict):
        return {
            name: copy.deepcopy(value[name] if name in value else next(iter(value.values()), None))
            for name in robot_names
        }
    return {name: copy.deepcopy(value) for name in robot_names}


def any_enabled(value: bool | dict[str, bool]) -> bool:
    if isinstance(value, dict):
        return any(bool(v) for v in value.values())
    return bool(value)


def resolve_entry_start_pose(
        entry_context: "PrimitiveEntryContext | None",
        robot_name: str,
        frame: "TaskFrame",
    ) -> list[float]:
    """Resolve the incoming EE pose for one robot in this primitive's frame.

    Args:
        entry_context: Optional processed observation and previous origin.
        robot_name: Robot whose entry pose should be resolved.
        frame: The task frame used by this primitive for that robot.

    Returns:
        A 6D pose expressed as [xyzrpy] in ``frame`` coordinates. Joint-space primitives
        fall back to their static configured target.
    """
    if frame.space != ControlSpace.TASK:
        return [float(v) for v in frame.target]

    if entry_context is None or not entry_context.observation:
        return [float(v) for v in frame.target]

    observed_pose = get_robot_pose_from_observation(entry_context.observation, robot_name)
    previous_origin = entry_context.task_frame_origin.get(robot_name)
    world_pose = task_pose_to_world_pose(observed_pose, previous_origin)
    return world_pose_to_task_pose(world_pose, frame.origin)


def task_frame_origins(primitive: Any) -> dict[str, list[float] | None]:
        """Extract per-robot task-frame origins from one primitive config.

        Args:
            primitive: Primitive config exposing a ``task_frame`` mapping.

        Returns:
            Per-robot task-frame origins copied into plain lists for entry
            context hand-off across primitive boundaries.
        """
        task_frames = getattr(primitive, "task_frame", {})
        if not isinstance(task_frames, dict):
            return {}
        return {
            name: None if frame.origin is None else [float(v) for v in frame.origin]
            for name, frame in task_frames.items()
        }


def axis_to_index(axis: int | str) -> int:
    if isinstance(axis, int):
        return axis
    axis_names = ["x", "y", "z", "rx", "ry", "rz"]
    if axis not in axis_names:
        raise ValueError(f"Unknown task-frame axis '{axis}'.")
    return axis_names.index(axis)


def resolve_value(source: dict[str, Any], key: str) -> Any:
    """Resolve a possibly dotted key from a transition source dictionary.

    Args:
        source: Observation or info dictionary passed to a transition.
        key: Plain key or dotted path to resolve.

    Returns:
        The resolved nested value.
    """
    current: Any = source
    if key in source:
        return current[key]

    for piece in key.split("."):
        if piece not in current:
            raise KeyError(f"Key '{key}' not found in transition source.")
        current = current[piece]

    return current


def to_scalar(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)

    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected scalar-like value for transition comparison, received shape {arr.shape}.")
    return float(arr.reshape(-1)[0])


def compare(lhs: float, rhs: float, operator: str) -> bool:
    if operator == "ge":
        return lhs >= rhs
    if operator == "gt":
        return lhs > rhs
    if operator == "le":
        return lhs <= rhs
    if operator == "lt":
        return lhs < rhs
    if operator == "eq":
        return lhs == rhs
    if operator == "ne":
        return lhs != rhs
    raise ValueError(f"Unsupported comparison operator '{operator}'.")
