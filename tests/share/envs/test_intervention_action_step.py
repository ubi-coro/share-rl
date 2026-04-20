"""Focused tests for intervention-side action projection."""

from __future__ import annotations

import math

import pytest

from lerobot.processor.core import TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.processor.action import InterventionActionProcessorStep
from share.teleoperators.utils import TeleopEvents
from share.utils.transformation_utils import rotation_from_extrinsic_xyz


def _base_transition(action: dict, info: dict | None = None, complementary_data: dict | None = None):
    return {
        TransitionKey.OBSERVATION: {},
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: info or {},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data or {},
    }


def test_intervention_action_processor_projects_and_merges_task_frame_targets():
    """Projection: learnable axes should override only their task-frame slots."""
    frame = TaskFrame(
        target=[1.0, 2.0, 3.0, 0.1, 0.2, 0.3],
        policy_mode=[PolicyMode.ABSOLUTE, None, PolicyMode.RELATIVE, PolicyMode.ABSOLUTE, None, None],
        control_mode=[ControlMode.VEL, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS],
    )
    step = InterventionActionProcessorStep(task_frame={"arm": frame}, gripper_enable={"arm": False})

    out = step(
        _base_transition(
            {
                "arm": {
                    "x.ee_vel": 2.0,
                    "z.ee_pos": -0.5,
                    "rx.pos.cos": 0.0,
                    "rx.pos.sin": 1.0,
                }
            }
        )
    )

    action = out[TransitionKey.ACTION]["arm"]
    assert action["x.ee_vel"] == pytest.approx(2.0)
    assert action["y.ee_pos"] == pytest.approx(2.0)
    assert action["z.ee_pos"] == pytest.approx(-0.5)
    assert action["rx.ee_pos"] == pytest.approx(math.pi / 2.0)
    assert action["ry.ee_pos"] == pytest.approx(0.2)
    assert action["rz.ee_pos"] == pytest.approx(0.3)


def test_intervention_action_processor_prefers_teleop_during_intervention_and_marks_completion():
    """Teleop override: intervention actions should win and then emit a completion marker on release."""
    frame = TaskFrame(
        target=[0.0] * 6,
        policy_mode=[PolicyMode.ABSOLUTE, None, None, None, None, None],
        control_mode=[ControlMode.POS] * 6,
    )
    step = InterventionActionProcessorStep(task_frame={"arm": frame}, gripper_enable={"arm": False})

    first = step(
        _base_transition(
            {"arm": {"x.ee_pos": 0.1}},
            info={TeleopEvents.IS_INTERVENTION: True},
            complementary_data={TELEOP_ACTION_KEY: {"arm": {"x.ee_pos": 0.4}}},
        )
    )
    assert first[TransitionKey.ACTION]["arm"]["x.ee_pos"] == pytest.approx(0.4)

    second = step(
        _base_transition(
            {"arm": {"x.ee_pos": 0.2}},
            info={TeleopEvents.IS_INTERVENTION: False},
            complementary_data={},
        )
    )
    assert second[TransitionKey.INFO][TeleopEvents.INTERVENTION_COMPLETED] is True


def test_intervention_action_processor_decodes_so3_6d_representation():
    """Rotation manifold decoding: SO(3) 6D actions should decode back to Euler task-frame targets."""
    expected_euler = [0.2, -0.3, 0.4]
    matrix = rotation_from_extrinsic_xyz(*expected_euler).as_matrix()
    encoded = {
        "rotation.so3.a1.x": matrix[0][0],
        "rotation.so3.a1.y": matrix[1][0],
        "rotation.so3.a1.z": matrix[2][0],
        "rotation.so3.a2.x": matrix[0][1],
        "rotation.so3.a2.y": matrix[1][1],
        "rotation.so3.a2.z": matrix[2][1],
    }

    frame = TaskFrame(
        target=[0.0] * 6,
        policy_mode=[None, None, None, PolicyMode.ABSOLUTE, PolicyMode.ABSOLUTE, PolicyMode.ABSOLUTE],
        control_mode=[ControlMode.POS] * 6,
    )
    step = InterventionActionProcessorStep(task_frame={"arm": frame}, gripper_enable={"arm": False})

    out = step(_base_transition({"arm": encoded}))
    action = out[TransitionKey.ACTION]["arm"]

    assert action["rx.ee_pos"] == pytest.approx(expected_euler[0], abs=1e-5)
    assert action["ry.ee_pos"] == pytest.approx(expected_euler[1], abs=1e-5)
    assert action["rz.ee_pos"] == pytest.approx(expected_euler[2], abs=1e-5)
