"""Mock example: two-arm baton handover using synchronized task-frame configs."""

from __future__ import annotations

import torch
from lerobot.processor import TransitionKey

from _mock_mp_net_utils import make_mock_connect
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    ManipulationPrimitiveConfig,
    MoveDeltaPrimitiveConfig,
    OpenLoopTrajectorySpec,
    OpenLoopTrajectoryPrimitiveConfig,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.envs.manipulation_primitive_net.transitions import OnSuccess, OnTargetPoseReached


def _frame(target: list[float]) -> TaskFrame:
    return TaskFrame(target=target, control_mode=[ControlMode.POS] * 6)


staging_cfg = ManipulationPrimitiveConfig(
    notes="Both arms wait in mirrored staging poses.",
    task_frame={
        "giver": _frame([-0.18, 0.00, 0.20, 0.0, 0.0, 0.0]),
        "receiver": _frame([0.18, 0.00, 0.20, 0.0, 0.0, 0.0]),
    },
)

meet_cfg = MoveDeltaPrimitiveConfig(
    notes="Both arms move toward the shared handover point.",
    task_frame={
        "giver": _frame([-0.18, 0.00, 0.20, 0.0, 0.0, 0.0]),
        "receiver": _frame([0.18, 0.00, 0.20, 0.0, 0.0, 0.0]),
    },
    delta={
        "giver": [0.12, 0.00, -0.02, 0.0, 0.0, 0.0],
        "receiver": [-0.12, 0.00, -0.02, 0.0, 0.0, 0.0],
    },
    delta_frame={"giver": "world", "receiver": "world"},
)

transfer_cfg = OpenLoopTrajectoryPrimitiveConfig(
    notes="Receiver lifts the baton as the giver releases and peels away.",
    task_frame={
        "giver": _frame([-0.06, 0.00, 0.18, 0.0, 0.0, 0.0]),
        "receiver": _frame([0.06, 0.00, 0.18, 0.0, 0.0, 0.0]),
    },
    trajectory=OpenLoopTrajectorySpec(
        delta={
            "giver": [-0.08, -0.06, 0.04, 0.0, 0.0, 0.0],
            "receiver": [0.10, 0.04, 0.08, 0.0, 0.0, 0.0],
        },
        frame={"giver": "world", "receiver": "world"},
        duration_s={"giver": 0.6, "receiver": 0.6},
    ),
)

reset_cfg = MoveDeltaPrimitiveConfig(
    notes="Return both arms to their staging positions after the handover.",
    task_frame={
        "giver": _frame([-0.14, -0.06, 0.22, 0.0, 0.0, 0.0]),
        "receiver": _frame([0.16, 0.04, 0.26, 0.0, 0.0, 0.0]),
    },
    delta={
        "giver": [-0.04, 0.06, -0.02, 0.0, 0.0, 0.0],
        "receiver": [0.02, -0.04, -0.06, 0.0, 0.0, 0.0],
    },
    delta_frame={"giver": "world", "receiver": "world"},
)


config = ManipulationPrimitiveNetConfig(
    start_primitive="staging",
    reset_primitive="staging",
    primitives={
        "staging": staging_cfg,
        "meet": meet_cfg,
        "transfer": transfer_cfg,
        "reset": reset_cfg,
    },
    transitions=[
        OnTargetPoseReached(source="staging", target="meet", tolerance=0.005),
        OnTargetPoseReached(source="meet", target="transfer", tolerance=0.01),
        OnSuccess(source="transfer", target="reset", success_key="primitive_complete"),
        OnTargetPoseReached(source="reset", target="staging", tolerance=0.01),
    ],
)


class DualArmHandoverDemoNet(ManipulationPrimitiveNet):
    connect = staticmethod(
        make_mock_connect(
            {
                "giver": [-0.18, 0.00, 0.20, 0.0, 0.0, 0.0],
                "receiver": [0.18, 0.00, 0.20, 0.0, 0.0, 0.0],
            }
        )
    )


def run_demo(steps: int = 14) -> None:
    net = DualArmHandoverDemoNet(config)
    transition = net.reset()
    print(f"start -> {net.active_primitive}")

    for step in range(steps):
        transition = net.step(torch.zeros(net.action_dim))
        if transition[TransitionKey.DONE] or transition[TransitionKey.TRUNCATED]:
            net.reset()
        info = transition[TransitionKey.INFO]
        print(
            f"{step:02d} {info['transition_from']} -> {info['transition_to']} "
            f"reason={info['transition_reason']}"
        )


if __name__ == "__main__":
    run_demo()
