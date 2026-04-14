"""Validation tests for ``ManipulationPrimitiveConfig`` teleoperator compatibility rules."""

from types import SimpleNamespace

import pytest

from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame
from share.utils.mock_utils import (
    MockAbsoluteJointTeleoperator,
    MockTaskFrameRobot,
    MockVelocityDeltaTeleoperator,
)


def test_validate_allows_velocity_style_delta_teleop_for_adaptive_vel_force():
    """Config validation: SpaceMouse-style EE velocity teleops should satisfy adaptive VEL/FORCE rules."""
    config = ManipulationPrimitiveConfig(
        task_frame=TaskFrame(
            target=[0.0] * 6,
            space=ControlSpace.TASK,
            policy_mode=[PolicyMode.ABSOLUTE, None, None, None, None, None],
            control_mode=[ControlMode.VEL, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS],
        ),
    )

    config.validate(
        robot_dict={"arm": MockTaskFrameRobot()},
        teleop_dict={"arm": MockVelocityDeltaTeleoperator()},
    )


def test_validate_rejects_absolute_joint_teleop_for_adaptive_vel_force():
    """Config validation: absolute-joint leaders should still fail adaptive VEL/FORCE task-frame configs."""
    config = ManipulationPrimitiveConfig(
        task_frame=TaskFrame(
            target=[0.0] * 6,
            space=ControlSpace.TASK,
            policy_mode=[PolicyMode.ABSOLUTE, None, None, None, None, None],
            control_mode=[ControlMode.VEL, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS],
        ),
    )

    with pytest.raises(ValueError, match="require a delta teleoperator"):
        config.validate(
            robot_dict={"arm": MockTaskFrameRobot()},
            teleop_dict={"arm": MockAbsoluteJointTeleoperator()},
        )


def test_validate_initializes_kinematics_for_task_frame_ur_when_enabled(monkeypatch):
    captured = {}

    def _fake_get_kinematics(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "share.envs.manipulation_primitive.config_manipulation_primitive.get_kinematics",
        _fake_get_kinematics,
    )

    class _MockURRobot(MockTaskFrameRobot):
        def __init__(self):
            super().__init__()
            self.name = "ur"
            self.config = SimpleNamespace(model="ur5e")
            self.bus = SimpleNamespace(motors={f"joint_{i + 1}": None for i in range(6)})

    config = ManipulationPrimitiveConfig(
        task_frame=TaskFrame(
            target=[0.0] * 6,
            space=ControlSpace.TASK,
            policy_mode=[PolicyMode.ABSOLUTE] * 6,
            control_mode=[ControlMode.POS] * 6,
        ),
    )
    config.processor.kinematics.enable = True
    config.processor.kinematics.urdf_path = None
    config.processor.kinematics.target_frame_name = "tool0"

    config.validate(
        robot_dict={"arm": _MockURRobot()},
        teleop_dict={"arm": MockAbsoluteJointTeleoperator()},
    )

    assert captured["robot_name"] == "ur"
    assert captured["robot_model"] == "ur5e"
    assert captured["target_frame_name"] == "tool0"
