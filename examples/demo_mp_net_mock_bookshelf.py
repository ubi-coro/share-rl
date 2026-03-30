"""Mock example: browse a bookshelf with world-frame slides and scripted retreat."""

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


home_cfg = ManipulationPrimitiveConfig(
    notes="Hover in front of the shelf.",
    task_frame={"arm": TaskFrame(target=[0.00, 0.00, 0.22, 0.0, 0.0, 0.0], control_mode=[ControlMode.POS] * 6)},
)

scan_right_cfg = MoveDeltaPrimitiveConfig(
    notes="Slide along the spines in world X while keeping orientation fixed.",
    task_frame={"arm": TaskFrame(target=[0.00, 0.00, 0.22, 0.0, 0.0, 0.0], control_mode=[ControlMode.POS] * 6)},
    delta={"arm": [0.18, 0.0, 0.0, 0.0, 0.0, 0.0]},
    delta_frame={"arm": "world"},
)

lean_in_cfg = MoveDeltaPrimitiveConfig(
    notes="Lean in using current-EE coordinates as if peeking between books.",
    task_frame={"arm": TaskFrame(target=[0.18, 0.00, 0.22, 0.0, 0.0, 0.0], control_mode=[ControlMode.POS] * 6)},
    delta={"arm": [0.00, 0.03, -0.04, 0.0, 0.0, 0.0]},
    delta_frame={"arm": "ee_current"},
)

retreat_cfg = OpenLoopTrajectoryPrimitiveConfig(
    notes="Back away and lift before returning to the shelf start.",
    task_frame={"arm": TaskFrame(target=[0.18, 0.03, 0.18, 0.0, 0.0, 0.0], control_mode=[ControlMode.POS] * 6)},
    trajectory=OpenLoopTrajectorySpec(
        delta={"arm": [-0.18, -0.03, 0.10, 0.0, 0.0, 0.0]},
        frame={"arm": "world"},
        duration_s={"arm": 0.8},
    ),
)


config = ManipulationPrimitiveNetConfig(
    start_primitive="home",
    reset_primitive="home",
    primitives={
        "home": home_cfg,
        "scan_right": scan_right_cfg,
        "lean_in": lean_in_cfg,
        "retreat": retreat_cfg,
    },
    transitions=[
        OnTargetPoseReached(source="home", target="scan_right", robot_name="arm", axes=["x", "z"], tolerance=0.005),
        OnTargetPoseReached(source="scan_right", target="lean_in", robot_name="arm", axes=["x"], tolerance=0.005),
        OnTargetPoseReached(source="lean_in", target="retreat", robot_name="arm", axes=["y", "z"], tolerance=0.005),
        OnSuccess(source="retreat", target="home", success_key="primitive_complete"),
    ],
)


class BookshelfDemoNet(ManipulationPrimitiveNet):
    connect = staticmethod(make_mock_connect({"arm": [0.00, 0.00, 0.22, 0.0, 0.0, 0.0]}))


def run_demo(steps: int = 100) -> None:
    net = BookshelfDemoNet(config)
    transition = net.reset()
    print(f"start -> {net.active_primitive}")

    for step in range(steps):
        transition = net.step(torch.zeros(net.action_dim))
        if transition[TransitionKey.DONE] or transition[TransitionKey.TRUNCATED]:
            net.reset()
        info = transition[TransitionKey.INFO]
        print(f"{step:02d} {info['transition_from']} -> {info['transition_to']} reason={info['transition_reason']}")


if __name__ == "__main__":
    run_demo()
