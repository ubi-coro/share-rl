"""Focused tests for the current fiddle-out experiment config."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[3] / "src" / "experiments" / "envs" / "fiddle_out.py"
_SPEC = importlib.util.spec_from_file_location("fiddle_out_experiment", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

DemoUR3eTeleopFiddleOutEnvConfig = _MODULE.DemoUR3eTeleopFiddleOutEnvConfig
EEFiddleOutCirclePrimitiveConfig = _MODULE.EEFiddleOutCirclePrimitiveConfig
OnSuccess = _MODULE.OnSuccess
OnTargetPoseReached = _MODULE.OnTargetPoseReached
TaskFrame = _MODULE.TaskFrame
ControlMode = _MODULE.ControlMode
ControlSpace = _MODULE.ControlSpace


def test_demo_fiddle_out_config_builds_expected_graph():
    """The shipped experiment should expose the teleop -> fiddle-out loop."""
    cfg = DemoUR3eTeleopFiddleOutEnvConfig()

    assert cfg.start_primitive == "teleop"
    assert cfg.reset_primitive == "teleop"
    assert set(cfg.primitives) == {"teleop", "fiddle_out"}
    assert isinstance(cfg.transitions[0], OnSuccess)
    assert isinstance(cfg.transitions[1], OnTargetPoseReached)

    teleop_frame = cfg.primitives["teleop"].task_frame
    fiddle_frame = cfg.primitives["fiddle_out"].task_frame

    assert teleop_frame.policy_action_dim == 6
    assert teleop_frame.space == ControlSpace.TASK
    assert teleop_frame.control_mode == [ControlMode.POS] * 6
    assert fiddle_frame.policy_action_dim == 0
    assert cfg.primitives["fiddle_out"].trajectory.frame == "ee"


def test_fiddle_out_circle_target_pose_applies_xy_circle_and_local_z_lift():
    """The scripted target should draw an EE-frame XY circle while ramping local Z."""
    primitive = EEFiddleOutCirclePrimitiveConfig(
        task_frame={"main": TaskFrame(target=[0.0] * 6, origin=[0.0] * 6)},
        trajectory=_MODULE.OpenLoopTrajectorySpec(
            delta={"main": [0.0, 0.0, -0.02, 0.0, 0.0, 0.0]},
            frame={"main": "ee"},
            duration_s={"main": 1.0},
        ),
        circle_radius_m={"main": 0.01},
        circle_frequency_hz={"main": 1.0},
    )

    target = primitive.target_pose_at(
        alpha=0.25,
        start_pose={"main": [0.0] * 6},
        goal_pose={"main": [0.0, 0.0, -0.02, 0.0, 0.0, 0.0]},
    )["main"]

    assert target[0] == pytest.approx(0.01 * (math.cos(math.pi / 2.0) - 1.0))
    assert target[1] == pytest.approx(0.01 * math.sin(math.pi / 2.0))
    assert target[2] == pytest.approx(-0.005)
    assert target[3:] == pytest.approx([0.0, 0.0, 0.0])
