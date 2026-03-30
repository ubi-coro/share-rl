from typing import Any

from scipy.spatial.transform import Rotation

from share.envs.manipulation_primitive.task_frame import TASK_FRAME_AXIS_NAMES


ROTATION_AXIS_ALIASES: dict[str, tuple[str, str]] = {
    "rx": ("rx", "wx"),
    "ry": ("ry", "wy"),
    "rz": ("rz", "wz"),
}


def rotation_from_extrinsic_xyz(rx: float, ry: float, rz: float) -> Rotation:
    """Build a rotation from extrinsic XYZ angles using explicit axis composition."""

    # Extrinsic XYZ composition applies X then Y then Z in the world frame.
    # Rotation multiplication order in scipy is right-to-left application.
    rot_x = Rotation.from_rotvec([rx, 0.0, 0.0])
    rot_y = Rotation.from_rotvec([0.0, ry, 0.0])
    rot_z = Rotation.from_rotvec([0.0, 0.0, rz])
    return rot_z * rot_y * rot_x


def euler_xyz_from_rotation(rotation: Rotation) -> list[float]:
    """Convert a ``Rotation`` back to XYZ Euler angles in radians."""

    return rotation.as_euler("xyz", degrees=False).tolist()


def euler_xyz_from_rotvec(rotvec: list[float]) -> list[float]:
    """Convert a rotation vector into user-facing XYZ roll-pitch-yaw angles."""

    return euler_xyz_from_rotation(Rotation.from_rotvec(rotvec))


def task_pose_to_world_pose(pose: list[float], origin: list[float] | None) -> list[float]:
    """Express a task-frame pose in world coordinates.

    Args:
        pose: 6D pose expressed relative to ``origin``.
        origin: Optional task-frame origin in world coordinates.

    Returns:
        The same pose represented in world coordinates.
    """
    if origin is None:
        return [float(v) for v in pose]

    origin_rot = rotation_from_extrinsic_xyz(*origin[3:6])
    pose_rot = rotation_from_extrinsic_xyz(*pose[3:6])
    world_position = origin_rot.apply(pose[:3]).tolist()
    world_rot = origin_rot * pose_rot
    return [
        float(origin[0] + world_position[0]),
        float(origin[1] + world_position[1]),
        float(origin[2] + world_position[2]),
        *[float(v) for v in euler_xyz_from_rotation(world_rot)],
    ]


def world_pose_to_task_pose(world_pose: list[float], origin: list[float] | None) -> list[float]:
    """Express a world pose in one task frame.

    Args:
        world_pose: 6D pose represented in world coordinates.
        origin: Optional task-frame origin in world coordinates.

    Returns:
        The pose re-expressed relative to ``origin``.
    """
    if origin is None:
        return [float(v) for v in world_pose]

    origin_rot = rotation_from_extrinsic_xyz(*origin[3:6])
    world_rot = rotation_from_extrinsic_xyz(*world_pose[3:6])
    relative_position = origin_rot.inv().apply(
        [
            float(world_pose[0] - origin[0]),
            float(world_pose[1] - origin[1]),
            float(world_pose[2] - origin[2]),
        ]
    ).tolist()
    relative_rot = origin_rot.inv() * world_rot
    return [
        *[float(v) for v in relative_position],
        *[float(v) for v in euler_xyz_from_rotation(relative_rot)],
    ]


def compose_delta_pose(
    start_pose_world: list[float],
    delta: list[float],
    frame_name: str,
) -> list[float]:
    """Apply a task-space delta in the requested frame.

    Args:
        start_pose_world: Absolute 6D world pose used as the delta reference.
        delta: 6D Cartesian delta to apply.
        frame_name: Delta frame selector. ``"world"`` applies the delta
            directly; ``"ee_current"`` rotates the translational component by
            the current EE orientation before composing it.

    Returns:
        The resolved target pose in world coordinates.
    """
    if frame_name == "world":
        return [float(start_pose_world[i] + delta[i]) for i in range(6)]
    if frame_name != "ee_current":
        raise ValueError(f"Unsupported delta frame '{frame_name}'.")

    start_rot = rotation_from_extrinsic_xyz(*start_pose_world[3:6])
    delta_rot = rotation_from_extrinsic_xyz(*delta[3:6])
    translated = start_rot.apply(delta[:3]).tolist()
    target_rot = start_rot * delta_rot
    return [
        float(start_pose_world[0] + translated[0]),
        float(start_pose_world[1] + translated[1]),
        float(start_pose_world[2] + translated[2]),
        *[float(v) for v in euler_xyz_from_rotation(target_rot)],
    ]


def rotation_component_keys(frame: "TaskFrame", absolute_rot_axes: list[int]) -> list[str]:
    if len(absolute_rot_axes) == 1:
        axis_name = frame.action_key_for_axis(absolute_rot_axes[0]).removesuffix(".pos")
        return [f"{axis_name}.pos.cos", f"{axis_name}.pos.sin"]
    if len(absolute_rot_axes) == 2:
        return ["rotation.s2.x", "rotation.s2.y", "rotation.s2.z"]
    if len(absolute_rot_axes) == 3:
        return [
            "rotation.so3.a1.x",
            "rotation.so3.a1.y",
            "rotation.so3.a1.z",
            "rotation.so3.a2.x",
            "rotation.so3.a2.y",
            "rotation.so3.a2.z",
        ]
    return []


def get_robot_pose_from_observation(observation: dict[str, Any], robot_name: str) -> list[float]:
    """Fetch one robot EE pose from processed observation channels.

    Args:
        observation: Processed observation dictionary containing per-robot pose
            channels such as ``arm.x.ee_pos`` or FK-generated equivalents.
        robot_name: Robot name prefix to extract.

    Returns:
        A 6D pose ordered as ``[x, y, z, rx, ry, rz]``.

    Raises:
        KeyError: If any required pose axis is missing from the observation.
    """
    def _to_float(value: Any) -> float:
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except ValueError:
                pass
        if hasattr(value, "reshape"):
            return float(value.reshape(-1)[0])
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            return float(list(value)[0])
        return float(value)

    position: list[float] = []
    raw_rotvec: list[float] = []
    missing: list[str] = []
    suffixes = (".ee_pos", ".pos")
    for axis_name in TASK_FRAME_AXIS_NAMES[:3]:
        aliases = (axis_name,)
        value: float | None = None
        for alias in aliases:
            for suffix in suffixes:
                key = f"{robot_name}.{alias}{suffix}"
                if key in observation:
                    value = _to_float(observation[key])
                    break
            if value is not None:
                break
        if value is None:
            missing.append(axis_name)
            continue
        position.append(value)

    for axis_name in TASK_FRAME_AXIS_NAMES[3:6]:
        aliases = ROTATION_AXIS_ALIASES.get(axis_name, (axis_name,))
        value: float | None = None
        for alias in aliases:
            for suffix in suffixes:
                key = f"{robot_name}.{alias}{suffix}"
                if key in observation:
                    value = _to_float(observation[key])
                    break
            if value is not None:
                break
        if value is None:
            missing.append(axis_name)
            continue
        raw_rotvec.append(value)

    if missing:
        raise KeyError(
            f"Observation is missing EE pose axes for robot '{robot_name}': {', '.join(missing)}."
        )
    return [*position, *euler_xyz_from_rotvec(raw_rotvec)]
