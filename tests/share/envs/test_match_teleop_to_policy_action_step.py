"""Focused tests for teleop-to-policy action normalization."""

from __future__ import annotations

import pytest
import torch
from scipy.spatial.transform import Rotation

from lerobot.processor.core import TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame
from share.processor.action import MatchTeleopToPolicyActionProcessorStep
from share.utils.transformation_utils import wrap_to_pi
from tests.share.envs.mock_pipeline_entities import (
    MockAbsoluteJointTeleoperator,
    MockComplexKinematicsSolver,
    MockComplexObservationRobot,
    MockDeltaTeleoperator,
    MockKinematicsSolver,
    MockVelocityDeltaTeleoperator,
)


def _transition_with_teleop_action(robot_name: str, action: dict[str, float], observation: dict | None = None):
    return {
        TransitionKey.OBSERVATION: observation or {},
        TransitionKey.ACTION: torch.zeros(1),
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {TELEOP_ACTION_KEY: {robot_name: action}},
    }


def test_delta_teleop_maps_differential_targets_directly():
    """Delta teleop mapping: VEL/WRENCH task-frame axes should pass through unchanged."""
    step = MatchTeleopToPolicyActionProcessorStep(
        teleoperators={"arm": MockDeltaTeleoperator()},
        task_frame={
            "arm": TaskFrame(
                policy_mode=[PolicyMode.ABSOLUTE, PolicyMode.ABSOLUTE, None, None, None, None],
                control_mode=[ControlMode.VEL, ControlMode.WRENCH, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS],
                target=[0.0] * 6,
            )
        },
        gripper_enable={"arm": False},
    )

    out = step(_transition_with_teleop_action("arm", {"delta_x": 0.4, "delta_y": -0.2}))
    converted = out[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]

    assert converted == {"x.ee_vel": pytest.approx(0.4), "y.ee_wrench": pytest.approx(-0.2)}


def test_delta_teleop_mixed_rotation_keeps_relative_axis_and_accumulates_absolute_axis():
    """Delta teleop mapping: mixed rotation should keep relative channels in SO(3) delta coordinates."""
    frame = TaskFrame(
        target=[0.0, 0.0, 0.0, 0.2, 0.0, -0.4],
        policy_mode=[None, None, None, PolicyMode.RELATIVE, None, PolicyMode.ABSOLUTE],
        control_mode=[ControlMode.POS] * 6,
        space=ControlSpace.TASK,
    )
    observation = {
        "arm.x.ee_pos": 0.0,
        "arm.y.ee_pos": 0.0,
        "arm.z.ee_pos": 0.0,
        "arm.rx.ee_pos": 0.2,
        "arm.ry.ee_pos": 0.0,
        "arm.rz.ee_pos": -0.4,
    }
    step = MatchTeleopToPolicyActionProcessorStep(
        teleoperators={"arm": MockDeltaTeleoperator()},
        task_frame={"arm": frame},
        gripper_enable={"arm": False},
        use_virtual_reference=False,
    )

    out = step(
        _transition_with_teleop_action(
            "arm",
            {"delta_rx": 0.1, "delta_rz": 0.2},
            observation=observation,
        )
    )
    converted = out[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]

    expected_rotation = Rotation.from_euler("xyz", [0.2, 0.0, -0.4], degrees=False)
    expected_rotation = Rotation.from_rotvec([0.1, 0.0, 0.0]) * expected_rotation
    expected_rpy = wrap_to_pi(expected_rotation.as_euler("xyz", degrees=False))
    expected_rpy[2] = wrap_to_pi(-0.4 + 0.2)

    assert converted["rx.ee_pos"] == pytest.approx(0.1)
    assert converted["rz.pos.cos"] == pytest.approx(float(torch.cos(torch.tensor(expected_rpy[2]))))
    assert converted["rz.pos.sin"] == pytest.approx(float(torch.sin(torch.tensor(expected_rpy[2]))))


def test_absolute_joint_teleop_uses_fk_and_so3_relative_rotation_delta():
    """Absolute-joint mapping: rotational relative policy channels should use SO(3) log deltas."""
    frame = TaskFrame(
        policy_mode=[None, None, None, PolicyMode.RELATIVE, None, None],
        control_mode=[ControlMode.POS] * 6,
        target=[0.0] * 6,
        space=ControlSpace.TASK,
    )
    step = MatchTeleopToPolicyActionProcessorStep(
        teleoperators={"arm": MockAbsoluteJointTeleoperator()},
        task_frame={"arm": frame},
        kinematics={"arm": MockComplexKinematicsSolver()},
        gripper_enable={"arm": False},
    )

    first = step(
        _transition_with_teleop_action(
            "arm",
            {"joint_1.pos": 0.4, "joint_2.pos": -0.2, "joint_3.pos": 0.3},
        )
    )
    second = step(
        _transition_with_teleop_action(
            "arm",
            {"joint_1.pos": 0.5, "joint_2.pos": -0.2, "joint_3.pos": 0.3},
        )
    )

    first_val = first[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]
    second_val = second[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]

    pose_1 = MockComplexKinematicsSolver().forward_kinematics({"joint_1": 0.4, "joint_2": -0.2, "joint_3": 0.3})
    pose_2 = MockComplexKinematicsSolver().forward_kinematics({"joint_1": 0.5, "joint_2": -0.2, "joint_3": 0.3})
    rot_delta = (
        Rotation.from_euler("xyz", pose_2[3:6], degrees=False)
        * Rotation.from_euler("xyz", pose_1[3:6], degrees=False).inv()
    ).as_rotvec()

    assert first_val["rx.ee_pos"] == pytest.approx(0.0)
    assert second_val["rx.ee_pos"] == pytest.approx(rot_delta[0])


def test_delta_teleop_absolute_translation_uses_observation_base_when_virtual_reference_disabled():
    """Delta teleop mapping: absolute task-frame translation should integrate against observation when virtual reference is off."""
    frame = TaskFrame(
        target=[0.0] * 6,
        space=ControlSpace.TASK,
        policy_mode=[PolicyMode.ABSOLUTE, None, None, None, None, None],
        control_mode=[ControlMode.POS] * 6,
    )
    step = MatchTeleopToPolicyActionProcessorStep(
        teleoperators={"arm": MockVelocityDeltaTeleoperator()},
        task_frame={"arm": frame},
        gripper_enable={"arm": False},
        use_virtual_reference=False,
    )

    out = step(
        _transition_with_teleop_action(
            "arm",
            {"x.vel": 0.25},
            observation={
                "arm.x.ee_pos": 0.5,
                "arm.y.ee_pos": 0.0,
                "arm.z.ee_pos": 0.0,
                "arm.rx.ee_pos": 0.0,
                "arm.ry.ee_pos": 0.0,
                "arm.rz.ee_pos": 0.0,
            },
        )
    )
    converted = out[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]

    assert converted["x.ee_pos"] == pytest.approx(0.75)


def test_match_step_uses_complex_fk_for_relative_translation_channels():
    """Absolute-joint mapping: FK-based relative translation deltas should stay componentwise for xyz."""
    robot = MockComplexObservationRobot()
    obs = robot.get_observation(prefix="arm")
    step = MatchTeleopToPolicyActionProcessorStep(
        teleoperators={"arm": MockAbsoluteJointTeleoperator()},
        task_frame={
            "arm": TaskFrame(
                policy_mode=[PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, None, None, None],
                control_mode=[ControlMode.POS] * 6,
                target=[0.0] * 6,
                space=ControlSpace.TASK,
            )
        },
        kinematics={"arm": MockComplexKinematicsSolver()},
        gripper_enable={"arm": False},
    )

    joint_action_1 = {
        "joint_1.pos": obs["arm.joint_1.pos"],
        "joint_2.pos": obs["arm.joint_2.pos"],
        "joint_3.pos": obs["arm.joint_3.pos"],
    }
    step(_transition_with_teleop_action("arm", joint_action_1))

    joint_action_2 = {
        "joint_1.pos": joint_action_1["joint_1.pos"] + 0.1,
        "joint_2.pos": joint_action_1["joint_2.pos"] - 0.05,
        "joint_3.pos": joint_action_1["joint_3.pos"] + 0.02,
    }
    out = step(_transition_with_teleop_action("arm", joint_action_2))
    converted = out[TransitionKey.COMPLEMENTARY_DATA][TELEOP_ACTION_KEY]["arm"]

    expected_dx = 0.5 * 0.1 + 0.2 * (-0.05) - 0.1 * 0.02
    expected_dy = -0.3 * 0.1 + 0.4 * (-0.05) + 0.2 * 0.02
    assert converted["x.ee_pos"] == pytest.approx(expected_dx, abs=1e-6)
    assert converted["y.ee_pos"] == pytest.approx(expected_dy, abs=1e-6)
