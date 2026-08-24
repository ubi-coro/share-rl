"""Shared scaffolding for per-robot TaskFrameCommand subclasses.

Every SHARE robot backend adds its own controller-tunable knobs on top of the
shared ``TaskFrame`` contract (see ``task_frame.py``) -- UR's per-axis
kp/kd/wrench budget, Franka's scalar Cartesian stiffness -- and each robot
must keep validating *its own* set of knobs; a Franka override is never
meaningful on a UR and vice versa. What's identical across robots is the
*mechanism*: reject unknown override keys, default a missing key, coerce it
to the right shape/dtype, and fail loudly on a bad shape. This module owns
that mechanism so each robot only has to declare a schema, not re-implement
the coercion loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from share.utils.transformation_utils import RotationIntervalMode


@dataclass(frozen=True)
class OverrideField:
    """One ``controller_overrides`` key: its default, shape, and dtype.

    ``shape=None`` means the value is a Python scalar (coerced via ``dtype``,
    e.g. ``float`` or ``bool``) rather than a NumPy array.
    """

    default: Any
    shape: tuple[int, ...] | None
    dtype: Any = np.float64
    validate: Callable[[str, Any], None] | None = None


def reject_unknown_overrides(overrides: Mapping[str, Any], supported: frozenset[str], robot_name: str) -> None:
    """Raise if ``overrides`` contains a key outside ``supported``."""
    unknown = set(overrides) - supported
    if unknown:
        raise ValueError(f"Unsupported {robot_name} controller overrides: {', '.join(sorted(unknown))}")


def resolve_override_fields(overrides: Mapping[str, Any], schema: Mapping[str, OverrideField]) -> dict[str, Any]:
    """Default, coerce, and validate every key in ``schema`` from ``overrides``.

    Unknown keys in ``overrides`` are ignored here -- callers should run
    :func:`reject_unknown_overrides` first so an unknown key is reported
    rather than silently dropped.
    """
    resolved: dict[str, Any] = {}
    for key, field in schema.items():
        raw = overrides.get(key, field.default)
        if field.shape is None:
            value = field.dtype(raw)
        else:
            value = np.asarray(raw, dtype=field.dtype)
            if value.shape != field.shape:
                raise ValueError(f"{key} must have shape {field.shape}")
        if field.validate is not None:
            field.validate(key, value)
        resolved[key] = value
    return resolved


def require_positive_or_inf(name: str, values: Any) -> None:
    """Raise unless every entry is a positive finite number or +inf (unconstrained)."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if np.any(values[finite] <= 0.0) or np.any(np.isnan(values)):
        raise ValueError(f"{name} entries must be positive, or inf for unconstrained")


def resolve_rotation_interval_modes(raw_modes: Any) -> np.ndarray:
    """Convert a list of rotation-interval-mode names into their int8 codes."""
    return np.array(
        [int(RotationIntervalMode.from_name(str(mode))) for mode in raw_modes],
        dtype=np.int8,
    )


def merge_controller_overrides(
    current: Mapping[str, Any] | None,
    new: Mapping[str, Any] | None,
    supported: frozenset[str],
    robot_name: str,
    default_factory: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge a partial override dict onto the current one, rejecting unknown keys.

    Used by the robot wrapper (``Robot.set_task_frame`` / ``send_action``)
    where a primitive supplies only the overrides it wants to change.
    ``default_factory`` is called lazily -- only once an unknown-key rejection
    hasn't already raised -- since building the defaults can require a fully
    populated config that a caller checking for a rejection may not have.
    """
    reject_unknown_overrides(new or {}, supported, robot_name)
    merged = dict(current) if current else dict(default_factory())
    if new:
        merged.update(new)
    return merged
