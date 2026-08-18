from __future__ import annotations

import math

import numpy as np


def compute_adaptive_limit_theta(wrench_limit: float, desired_wrench: float, minimum_scale: float) -> float:
    """Compute the exponential decay constant for an adaptive wrench limit."""
    wrench_limit = float(wrench_limit)
    desired_wrench = float(desired_wrench)
    minimum_scale = float(minimum_scale)
    if not math.isfinite(wrench_limit) or wrench_limit <= 0.0:
        raise ValueError("wrench_limit must be finite and positive")
    if not math.isfinite(desired_wrench) or desired_wrench <= 0.0:
        raise ValueError("desired_wrench must be finite and positive")
    if not 0.0 <= minimum_scale < 1.0:
        raise ValueError("minimum_scale must be in [0, 1)")

    fixed_point_scale = desired_wrench / wrench_limit
    if not minimum_scale < fixed_point_scale < 1.0:
        raise ValueError("Require minimum_scale < desired_wrench/wrench_limit < 1")
    ratio = (fixed_point_scale - minimum_scale) / (1.0 - minimum_scale)
    return -desired_wrench / math.log(ratio)


def adaptive_scale_and_derivative(force: float, theta: float, minimum_scale: float) -> tuple[float, float]:
    """Return exponential adaptive scale and its force derivative."""
    force = abs(float(force))
    theta = float(theta)
    minimum_scale = float(minimum_scale)
    if not math.isfinite(theta) or theta <= 0.0:
        raise ValueError("theta must be finite and positive")
    if not 0.0 <= minimum_scale < 1.0:
        raise ValueError("minimum_scale must be in [0, 1)")
    exp_term = math.exp(-force / theta)
    scale = minimum_scale + (1.0 - minimum_scale) * exp_term
    derivative = -(1.0 - minimum_scale) * exp_term / theta
    return scale, derivative


def validate_adaptive_fixed_point(
    wrench_limit: float,
    desired_wrench: float,
    minimum_scale: float,
    theta: float,
) -> None:
    """Reject adaptive parameters whose fixed point is locally unstable."""
    _, derivative = adaptive_scale_and_derivative(desired_wrench, theta, minimum_scale)
    if abs(float(wrench_limit) * derivative) >= 1.0:
        raise ValueError("Adaptive wrench-limit fixed point is unstable (|F_max * s'(f*)| >= 1)")


def reference_error_limit(wrench_limit: float, stiffness: float, enabled: bool) -> float:
    """Return the stored-reference anti-windup limit used by SHARE controllers."""
    if not enabled:
        return math.inf
    wrench_limit = float(wrench_limit)
    stiffness = float(stiffness)
    if wrench_limit <= 0.0:
        return 0.0
    if stiffness <= 0.0:
        return math.inf
    return wrench_limit / stiffness


def adaptive_wrench_scales(
    desired_wrench: np.ndarray,
    measured_wrench: np.ndarray,
    enabled: np.ndarray,
    minimum_scale: np.ndarray,
    theta: np.ndarray,
) -> np.ndarray:
    """Compute per-axis scales, reacting only to contact opposing the command."""
    desired = np.asarray(desired_wrench, dtype=np.float64)
    measured = np.asarray(measured_wrench, dtype=np.float64)
    enabled = np.asarray(enabled, dtype=bool)
    minimum_scale = np.asarray(minimum_scale, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if any(value.shape != desired.shape for value in (measured, enabled, minimum_scale, theta)):
        raise ValueError("Adaptive wrench-limit arrays must have identical shapes")

    scales = np.ones_like(desired)
    for axis in np.flatnonzero(enabled):
        opposing_force = measured[axis]
        if np.sign(desired[axis]) == np.sign(opposing_force):
            opposing_force = 0.0
        scales[axis], _ = adaptive_scale_and_derivative(
            opposing_force,
            theta[axis],
            minimum_scale[axis],
        )
    return scales
