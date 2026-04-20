"""Focused tests for non-hardware UR controller helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode
from share.robots.ur.lerobot_robot_ur.controller import Command, RTDETaskFrameController, TaskFrameCommand


def test_task_frame_command_delta_mode_treats_only_relative_axes_as_deltas():
    """Delta mode should be derived from policy_mode only."""
    command = TaskFrameCommand(
        policy_mode=[
            PolicyMode.RELATIVE,
            PolicyMode.ABSOLUTE,
            None,
            PolicyMode.RELATIVE,
            None,
            PolicyMode.ABSOLUTE,
        ]
    )

    assert command.delta_mode == [
        PolicyMode.RELATIVE,
        PolicyMode.ABSOLUTE,
        PolicyMode.ABSOLUTE,
        PolicyMode.RELATIVE,
        PolicyMode.ABSOLUTE,
        PolicyMode.ABSOLUTE,
    ]


def test_task_frame_command_rejects_unknown_controller_override_keys():
    """Unknown override keys should fail before queueing."""
    with pytest.raises(ValueError, match="Unsupported UR task-frame controller overrides"):
        TaskFrameCommand(
            control_mode=[ControlMode.POS] * 6,
            controller_overrides={"mystery_limit": [1.0] * 6},
        ).to_queue_dict()


def test_controller_zero_ft_reuses_last_command_layout_with_new_opcode():
    """`zero_ft()` should reuse the last command payload with a new opcode."""
    queued_items: list[dict[str, np.ndarray]] = []
    controller = object.__new__(RTDETaskFrameController)
    controller._last_cmd = TaskFrameCommand(
        target=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[PolicyMode.ABSOLUTE] * 6,
    )
    controller.robot_cmd_queue = SimpleNamespace(put=queued_items.append)

    controller.zero_ft()

    assert len(queued_items) == 1
    queued = queued_items[0]
    assert queued["cmd"] == int(Command.ZERO_FT)
    np.testing.assert_allclose(queued["target"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
