import collections
import enum
import math
import numpy as np
from typing import Any
from dataclasses import dataclass

from scipy.spatial.transform import Rotation

from share.envs.manipulation_primitive.task_frame import TASK_FRAME_AXIS_NAMES
from share.utils.constants import DEFAULT_ROBOT_NAME

ROTATION_AXIS_ALIASES: dict[str, tuple[str, str]] = {
    "rx": ("rx", "wx"),
    "ry": ("ry", "wy"),
    "rz": ("rz", "wz"),
}

TAU = 2.0 * math.pi


class RotationIntervalMode(enum.IntEnum):
    """Encoded interpretation for one rotational bound interval."""

    LINEAR = 0
    CCW_ARC = 1

    @classmethod
    def from_name(cls, value: str) -> "RotationIntervalMode":
        normalized = str(value).strip().lower()
        if normalized == "linear":
            return cls.LINEAR
        if normalized == "ccw_arc":
            return cls.CCW_ARC
        raise ValueError(f"Unsupported rotation interval mode '{value}'.")

    def to_name(self) -> str:
        if self is type(self).CCW_ARC:
            return "ccw_arc"
        return "linear"


def seconds_to_ms(value: float) -> float:
    """Convert seconds to milliseconds for debug logging."""

    return 1000.0 * float(value)


@dataclass(slots=True)
class RollingPerfWindow:
    """Fixed-size rolling timing window with percentile summaries."""

    buf: collections.deque

    @classmethod
    def create(cls, maxlen: int) -> "RollingPerfWindow":
        return cls(buf=collections.deque(maxlen=maxlen))

    def add(self, duration_s: float) -> None:
        self.buf.append(float(duration_s))

    def stats(self) -> dict[str, float] | None:
        if not self.buf:
            return None
        samples = np.fromiter(self.buf, dtype=np.float64)
        return {
            "n": int(samples.size),
            "mean": float(samples.mean()),
            "std": float(samples.std()),
            "p50": float(np.percentile(samples, 50)),
            "p90": float(np.percentile(samples, 90)),
            "p99": float(np.percentile(samples, 99)),
            "max": float(samples.max()),
            "min": float(samples.min()),
        }


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to (-pi, pi], mapping -pi to +pi for consistency."""
    out = (np.asarray(angle) + math.pi) % TAU - math.pi
    out = np.asarray(out)
    out[np.isclose(out, -math.pi)] = math.pi
    if np.isscalar(angle):
        return float(out.item())
    return out


def ccw_distance(a: float, b: float) -> float:
    """Counterclockwise distance from a to b on S1, in [0, 2pi)."""
    return float((b - a) % TAU)


def circular_distance(a: float, b: float) -> float:
    """Shortest unsigned angular distance on S1, in [0, pi]."""
    d = abs(float(wrap_to_pi(b - a)))
    return min(d, TAU - d)


def is_in_ccw_arc(x: float, start: float, end: float, eps: float = 1e-12) -> bool:
    """Whether x lies on the allowed counterclockwise arc [start -> end]."""
    return ccw_distance(start, x) <= ccw_distance(start, end) + eps


def clip_angle_to_ccw_arc(x: float, start: float, end: float) -> float:
    """Project x to the nearest point on the allowed CCW arc [start -> end]."""
    x = float(wrap_to_pi(x))
    start = float(wrap_to_pi(start))
    end = float(wrap_to_pi(end))

    if is_in_ccw_arc(x, start, end):
        return x

    d_start = circular_distance(x, start)
    d_end = circular_distance(x, end)
    return start if d_start <= d_end else end


def penetration_to_ccw_arc(x: float, start: float, end: float) -> float:
    """Shortest angular distance from x to the allowed CCW arc [start -> end]."""
    x = float(wrap_to_pi(x))
    start = float(wrap_to_pi(start))
    end = float(wrap_to_pi(end))

    if is_in_ccw_arc(x, start, end):
        return 0.0
    return min(circular_distance(x, start), circular_distance(x, end))


def signed_error_to_nearest_arc_endpoint(x: float, start: float, end: float) -> float:
    """Signed shortest angular correction from x to the nearest allowed endpoint.

    Positive means increase the angle, negative means decrease it.
    Returns 0 if x already lies in the allowed arc.
    """
    x = float(wrap_to_pi(x))
    start = float(wrap_to_pi(start))
    end = float(wrap_to_pi(end))

    if is_in_ccw_arc(x, start, end):
        return 0.0

    d_start = circular_distance(x, start)
    d_end = circular_distance(x, end)
    target = start if d_start <= d_end else end
    return float(wrap_to_pi(target - x))


def unwrap_angle_near_reference(angle: float, reference: float) -> float:
    """Unwrap angle to the representation nearest to reference."""
    angle = float(angle)
    reference = float(reference)
    while angle - reference > math.pi:
        angle -= TAU
    while angle - reference < -math.pi:
        angle += TAU
    return angle


# rotation handling
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


def rotvec_to_euler_xyz(rotvec: list[float] | np.ndarray) -> np.ndarray:
    """Convert a rotation vector into XYZ Euler angles as a NumPy array."""

    return Rotation.from_rotvec(rotvec).as_euler("xyz", degrees=False)


def euler_xyz_to_rotvec(euler_xyz: list[float] | np.ndarray) -> np.ndarray:
    """Convert XYZ Euler angles into a rotation vector."""

    return Rotation.from_euler("xyz", euler_xyz, degrees=False).as_rotvec()


def homogeneous_to_sixvec(transform: np.ndarray) -> list[float]:
    """Convert a 4x4 homogeneous transform into ``[x, y, z, rx, ry, rz]``."""

    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("Input must be a 4x4 matrix.")

    translation = transform[:3, 3]
    rotation_vector = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    return list(np.concatenate((translation, rotation_vector)))


def sixvec_to_homogeneous(six_vec: list[float] | np.ndarray) -> np.ndarray:
    """Convert ``[x, y, z, rx, ry, rz]`` into a 4x4 homogeneous transform."""

    six = np.asarray(six_vec, dtype=float)
    if six.shape != (6,):
        raise ValueError(f"Expected 6-vector, got shape {six.shape}")

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = Rotation.from_rotvec(six[3:]).as_matrix()
    transform[:3, 3] = six[:3]
    return transform


def exp_scale(f_meas: float, f_thresh: float, s_min: float = 0.2, theta: float = 0.1) -> float:
    """Exponentially scale a magnitude toward ``s_min`` as measurement increases.

    ``f_thresh`` is kept for API compatibility with existing controller call sites.
    """

    del f_thresh
    return float(s_min + (1 - s_min) * np.exp(-f_meas / theta))


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
            directly; ``"ee"`` rotates the translational component by
            the current EE orientation before composing it.

    Returns:
        The resolved target pose in world coordinates.
    """
    start_rot = rotation_from_extrinsic_xyz(*start_pose_world[3:6])
    delta_rot = rotation_from_extrinsic_xyz(*delta[3:6])

    if frame_name == "world":
        target_rot = delta_rot * start_rot
        return [
            float(start_pose_world[0] + delta[0]),
            float(start_pose_world[1] + delta[1]),
            float(start_pose_world[2] + delta[2]),
            *[float(v) for v in euler_xyz_from_rotation(target_rot)],
        ]

    if frame_name != "ee":
        raise ValueError(f"Unsupported delta frame '{frame_name}'.")

    translated = start_rot.apply(delta[:3]).tolist()
    target_rot = start_rot * delta_rot
    return [
        float(start_pose_world[0] + translated[0]),
        float(start_pose_world[1] + translated[1]),
        float(start_pose_world[2] + translated[2]),
        *[float(v) for v in euler_xyz_from_rotation(target_rot)],
    ]


# getting information (keys, poses) from observations
def rotation_component_keys(frame: "TaskFrame", absolute_rot_axes: list[int]) -> list[str]:
    if len(absolute_rot_axes) == 1:
        axis_name = TASK_FRAME_AXIS_NAMES[absolute_rot_axes[0]]
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


def get_robot_pose_from_observation(observation: dict[str, Any], robot_name: str | None = None) -> list[float]:
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

    if robot_name is None:
        robot_name = DEFAULT_ROBOT_NAME

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
