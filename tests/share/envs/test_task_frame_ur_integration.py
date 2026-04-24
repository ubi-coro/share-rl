"""Focused integration tests for the current UR task-frame wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.robots.ur.lerobot_robot_ur.controller import TaskFrameCommand
from share.robots.ur.lerobot_robot_ur.ur import UR
from share.utils.transformation_utils import RotationIntervalMode


def test_task_frame_command_queue_dict_encodes_current_override_schema():
    """Queue payloads should follow the current override and rotvec contract."""
    command = TaskFrameCommand(
        origin=[0.1, 0.2, 0.3, 0.2, -0.1, 0.3],
        target=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        control_mode=[
            ControlMode.POS,
            ControlMode.VEL,
            ControlMode.WRENCH,
            ControlMode.POS,
            ControlMode.POS,
            ControlMode.POS,
        ],
        policy_mode=[
            PolicyMode.RELATIVE,
            None,
            None,
            PolicyMode.ABSOLUTE,
            PolicyMode.RELATIVE,
            PolicyMode.ABSOLUTE,
        ],
        controller_overrides={
            "kp": [1.0] * 6,
            "rotation_interval_modes": ["linear", "ccw_arc", "linear", "ccw_arc", "linear", "ccw_arc"],
        },
    )

    queued = command.to_queue_dict()

    np.testing.assert_allclose(
        queued["origin"][3:6],
        TaskFrameCommand(origin=[0.0, 0.0, 0.0, 0.2, -0.1, 0.3]).to_queue_dict()["origin"][3:6],
    )
    np.testing.assert_array_equal(
        queued["delta_mode"],
        [
            int(PolicyMode.RELATIVE),
            int(PolicyMode.ABSOLUTE),
            int(PolicyMode.ABSOLUTE),
            int(PolicyMode.ABSOLUTE),
            int(PolicyMode.RELATIVE),
            int(PolicyMode.ABSOLUTE),
        ],
    )
    np.testing.assert_array_equal(
        queued["rotation_interval_modes"],
        [
            int(RotationIntervalMode.LINEAR),
            int(RotationIntervalMode.CCW_ARC),
            int(RotationIntervalMode.LINEAR),
            int(RotationIntervalMode.CCW_ARC),
            int(RotationIntervalMode.LINEAR),
            int(RotationIntervalMode.CCW_ARC),
        ],
    )


def test_ur_action_features_follow_the_current_task_frame_schema():
    """The UR wrapper should expose action keys from its current task frame."""
    robot = object.__new__(UR)
    robot.controller = SimpleNamespace(is_ready=True)
    robot.gripper = None
    robot.cameras = {}
    robot.config = SimpleNamespace(
        kp=[2500.0] * 3 + [150.0] * 3,
        kd=[80.0] * 3 + [8.0] * 3,
        min_pose_rpy=[-float("inf")] * 6,
        max_pose_rpy=[float("inf")] * 6,
        wrench_limits=[30.0] * 6,
        compliance_adaptive_limit_enable=[False] * 6,
        compliance_reference_limit_enable=[False] * 6,
        compliance_desired_wrench=[5.0] * 6,
        compliance_adaptive_limit_min=[0.1] * 6,
    )
    robot.task_frame = TaskFrameCommand(
        target=[0.0] * 6,
        control_mode=[
            ControlMode.POS,
            ControlMode.VEL,
            ControlMode.WRENCH,
            ControlMode.POS,
            ControlMode.VEL,
            ControlMode.WRENCH,
        ],
        policy_mode=[None] * 6,
    )
    robot._active_control_space = None

    assert robot.action_features == {
        "x.ee_pos": float,
        "y.ee_vel": float,
        "z.ee_wrench": float,
        "rx.ee_pos": float,
        "ry.ee_vel": float,
        "rz.ee_wrench": float,
    }

    robot.set_task_frame(
        TaskFrame(
            target=[0.1] * 6,
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[PolicyMode.ABSOLUTE] * 6,
        )
    )

    assert robot.task_frame.target == [0.1] * 6
    assert set(robot.action_features) == {
        "x.ee_pos",
        "y.ee_pos",
        "z.ee_pos",
        "rx.ee_pos",
        "ry.ee_pos",
        "rz.ee_pos",
    }
