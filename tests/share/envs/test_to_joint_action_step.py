"""Focused tests for task-frame to joint-space conversion."""

from __future__ import annotations

import pytest
from scipy.spatial.transform import Rotation

from lerobot.processor.core import TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.processor.action import (
    InterventionActionProcessorStep,
    MatchTeleopToPolicyActionProcessorStep,
    ToJointActionProcessorStep,
)
from share.teleoperators.utils import TeleopEvents
from share.utils.transformation_utils import wrap_to_pi
from tests.share.envs.mock_pipeline_entities import (
    MockComplexKinematicsSolver,
    MockComplexObservationRobot,
    MockDeltaTeleoperator,
    MockKinematicsSolver,
)


class RecordingSolver:
    def __init__(self):
        self.last_pose: list[float] | None = None

    def inverse_kinematics(self, pose: list[float]) -> dict[str, float]:
        self.last_pose = [float(v) for v in pose]
        return {"joint_1": float(pose[0]), "joint_2": float(pose[1]), "joint_3": float(pose[2])}


def _transition(action, observation=None, info=None, complementary_data=None):
    return {
        TransitionKey.OBSERVATION: observation or {},
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False,
        TransitionKey.TRUNCATED: False,
        TransitionKey.INFO: info or {},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data or {},
    }


def test_to_joint_integrates_relative_translation_and_clamps_limits():
    """Joint conversion: relative translation should integrate against the base pose before IK."""
    frame = TaskFrame(
        target=[0.0] * 6,
        policy_mode=[PolicyMode.RELATIVE, None, None, None, None, None],
        control_mode=[ControlMode.POS] * 6,
        min_pose=[-0.2] * 6,
        max_pose=[0.2] * 6,
    )
    step = ToJointActionProcessorStep(
        is_task_frame_robot={"arm": False},
        task_frame={"arm": frame},
        kinematics={"arm": MockKinematicsSolver()},
        use_virtual_reference={"arm": True},
    )

    out1 = step(
        _transition(
            {"arm": {"x.ee_pos": 0.5, "y.ee_pos": 0.0, "z.ee_pos": 0.0, "rx.ee_pos": 0.0, "ry.ee_pos": 0.0, "rz.ee_pos": 0.0}},
            observation={
                "arm.x.ee_pos": 0.1,
                "arm.y.ee_pos": 0.0,
                "arm.z.ee_pos": 0.0,
                "arm.rx.ee_pos": 0.0,
                "arm.ry.ee_pos": 0.0,
                "arm.rz.ee_pos": 0.0,
            },
        )
    )
    assert out1[TransitionKey.ACTION]["arm"]["joint_1.pos"] == pytest.approx(0.2)

    out2 = step(
        _transition(
            {"arm": {"x.ee_pos": -0.1, "y.ee_pos": 0.0, "z.ee_pos": 0.0, "rx.ee_pos": 0.0, "ry.ee_pos": 0.0, "rz.ee_pos": 0.0}}
        )
    )
    assert out2[TransitionKey.ACTION]["arm"]["joint_1.pos"] == pytest.approx(0.1)


def test_to_joint_mixed_rotation_uses_masked_so3_then_absolute_rpy():
    """Joint conversion: mixed rotation should mirror the controller's masked SO(3) semantics."""
    solver = RecordingSolver()
    frame = TaskFrame(
        target=[0.0, 0.0, 0.0, 0.2, 0.0, -0.4],
        policy_mode=[None, None, None, PolicyMode.RELATIVE, None, PolicyMode.ABSOLUTE],
        control_mode=[ControlMode.POS] * 6,
        min_pose=[-2.0] * 6,
        max_pose=[2.0] * 6,
    )
    step = ToJointActionProcessorStep(
        is_task_frame_robot={"arm": False},
        task_frame={"arm": frame},
        kinematics={"arm": solver},
        use_virtual_reference={"arm": False},
    )
    observation = {
        "arm.x.ee_pos": 0.0,
        "arm.y.ee_pos": 0.0,
        "arm.z.ee_pos": 0.0,
        "arm.rx.ee_pos": 0.2,
        "arm.ry.ee_pos": 0.0,
        "arm.rz.ee_pos": -0.4,
    }

    step(
        _transition(
            {
                "arm": {
                    "x.ee_pos": 0.0,
                    "y.ee_pos": 0.0,
                    "z.ee_pos": 0.0,
                    "rx.ee_pos": 0.1,
                    "ry.ee_pos": 0.0,
                    "rz.ee_pos": -0.2,
                }
            },
            observation=observation,
        )
    )

    expected_rotation = Rotation.from_euler("xyz", [0.2, 0.0, -0.4], degrees=False)
    expected_rotation = Rotation.from_rotvec([0.1, 0.0, 0.0]) * expected_rotation
    expected_rpy = wrap_to_pi(expected_rotation.as_euler("xyz", degrees=False))
    expected_rpy[1] = 0.0
    expected_rpy[2] = -0.2

    assert solver.last_pose is not None
    np_pose = solver.last_pose
    assert np_pose[3] == pytest.approx(expected_rpy[0], abs=1e-6)
    assert np_pose[4] == pytest.approx(expected_rpy[1], abs=1e-6)
    assert np_pose[5] == pytest.approx(expected_rpy[2], abs=1e-6)


def test_processor_chain_teleop_to_task_frame_to_joint_action():
    """Processor chain: intervention teleop should preserve mixed semantics end to end."""
    frame = TaskFrame(
        target=[0.0, 0.0, 0.0, 0.2, 0.0, -0.4],
        policy_mode=[None, None, None, PolicyMode.RELATIVE, None, PolicyMode.ABSOLUTE],
        control_mode=[ControlMode.POS] * 6,
        min_pose=[-2.0] * 6,
        max_pose=[2.0] * 6,
    )
    solver = RecordingSolver()
    match = MatchTeleopToPolicyActionProcessorStep(
        teleoperators={"arm": MockDeltaTeleoperator()},
        task_frame={"arm": frame},
        gripper_enable={"arm": False},
        use_virtual_reference={"arm": False},
    )
    intervention = InterventionActionProcessorStep(task_frame={"arm": frame}, gripper_enable={"arm": False})
    to_joint = ToJointActionProcessorStep(
        is_task_frame_robot={"arm": False},
        task_frame={"arm": frame},
        kinematics={"arm": solver},
        use_virtual_reference={"arm": False},
    )

    tr = _transition(
        {"arm": {}},
        observation={
            "arm.x.ee_pos": 0.0,
            "arm.y.ee_pos": 0.0,
            "arm.z.ee_pos": 0.0,
            "arm.rx.ee_pos": 0.2,
            "arm.ry.ee_pos": 0.0,
            "arm.rz.ee_pos": -0.4,
        },
        info={TeleopEvents.IS_INTERVENTION: True},
        complementary_data={TELEOP_ACTION_KEY: {"arm": {"delta_rx": 0.1, "delta_rz": 0.2}}},
    )

    to_joint(intervention(match(tr)))

    expected_rotation = Rotation.from_euler("xyz", [0.2, 0.0, -0.4], degrees=False)
    expected_rotation = Rotation.from_rotvec([0.1, 0.0, 0.0]) * expected_rotation
    expected_rpy = wrap_to_pi(expected_rotation.as_euler("xyz", degrees=False))
    expected_rpy[1] = 0.0
    expected_rpy[2] = -0.2

    assert solver.last_pose is not None
    assert solver.last_pose[3] == pytest.approx(expected_rpy[0], abs=1e-6)
    assert solver.last_pose[4] == pytest.approx(expected_rpy[1], abs=1e-6)
    assert solver.last_pose[5] == pytest.approx(expected_rpy[2], abs=1e-6)


def test_to_joint_step_consumes_ee_observation_for_relative_integration():
    """Joint conversion: observed EE pose should be the integration base when virtual reference is disabled."""
    robot = MockComplexObservationRobot()
    obs = robot.get_observation(prefix="arm")

    frame = TaskFrame(
        target=[0.0] * 6,
        policy_mode=[PolicyMode.RELATIVE, None, None, None, None, None],
        control_mode=[ControlMode.POS] * 6,
        min_pose=[-2.0] * 6,
        max_pose=[2.0] * 6,
    )
    step = ToJointActionProcessorStep(
        is_task_frame_robot={"arm": False},
        task_frame={"arm": frame},
        kinematics={"arm": MockComplexKinematicsSolver()},
        use_virtual_reference={"arm": False},
    )

    out = step(
        _transition(
            {
                "arm": {
                    "x.ee_pos": 0.1,
                    "y.ee_pos": obs["arm.y.ee_pos"],
                    "z.ee_pos": obs["arm.z.ee_pos"],
                    "rx.ee_pos": obs["arm.wx.ee_pos"],
                    "ry.ee_pos": obs["arm.wy.ee_pos"],
                    "rz.ee_pos": obs["arm.wz.ee_pos"],
                }
            },
            observation={
                "arm.x.ee_pos": obs["arm.x.ee_pos"],
                "arm.y.ee_pos": obs["arm.y.ee_pos"],
                "arm.z.ee_pos": obs["arm.z.ee_pos"],
                "arm.rx.ee_pos": obs["arm.wx.ee_pos"],
                "arm.ry.ee_pos": obs["arm.wy.ee_pos"],
                "arm.rz.ee_pos": obs["arm.wz.ee_pos"],
            },
        )
    )

    integrated_target = [obs["arm.x.ee_pos"] + 0.1, obs["arm.y.ee_pos"], obs["arm.z.ee_pos"], obs["arm.wx.ee_pos"], obs["arm.wy.ee_pos"], obs["arm.wz.ee_pos"]]
    expected_joints = MockComplexKinematicsSolver().inverse_kinematics(integrated_target)
    result = out[TransitionKey.ACTION]["arm"]
    assert result["joint_1.pos"] == pytest.approx(expected_joints["joint_1"], abs=1e-6)
    assert result["joint_2.pos"] == pytest.approx(expected_joints["joint_2"], abs=1e-6)
    assert result["joint_3.pos"] == pytest.approx(expected_joints["joint_3"], abs=1e-6)
