"""Mock example: tilt, pour, and recover like a tiny tea service routine."""

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


upright_cfg = ManipulationPrimitiveConfig(
    notes="Start upright above the cup.",
    task_frame={"arm": TaskFrame(target=[0.0, 0.0, 0.20, 0.0, 0.0, 0.0], control_mode=[ControlMode.POS] * 6)},
)

tilt_cfg = MoveDeltaPrimitiveConfig(
    notes="Rotate around the wrist in current-EE coordinates to begin pouring.",
    task_frame={"arm": TaskFrame(target=[0.0, 0.0, 0.20, 0.0, 0.0, 0.0], control_mode=[ControlMode.POS] * 6)},
    delta={"arm": [0.00, 0.00, -0.02, 0.0, 0.35, 0.0]},
    delta_frame={"arm": "ee_current"},
)

pour_arc_cfg = OpenLoopTrajectoryPrimitiveConfig(
    notes="Trace a gentle pouring arc while maintaining the wrist tilt.",
    task_frame={"arm": TaskFrame(target=[0.0, 0.0, 0.18, 0.0, 0.35, 0.0], control_mode=[ControlMode.POS] * 6)},
    trajectory=OpenLoopTrajectorySpec(
        delta={"arm": [0.08, 0.00, -0.02, 0.0, 0.0, 0.20]},
        frame={"arm": "world"},
        duration_s={"arm": 1.0},
    ),
)

recover_cfg = MoveDeltaPrimitiveConfig(
    notes="Return upright and slide back over the tray.",
    task_frame={"arm": TaskFrame(target=[0.08, 0.0, 0.16, 0.0, 0.35, 0.20], control_mode=[ControlMode.POS] * 6)},
    delta={"arm": [-0.08, 0.00, 0.04, 0.0, -0.35, -0.20]},
    delta_frame={"arm": "world"},
)


config = ManipulationPrimitiveNetConfig(
    start_primitive="upright",
    reset_primitive="upright",
    primitives={
        "upright": upright_cfg,
        "tilt": tilt_cfg,
        "pour_arc": pour_arc_cfg,
        "recover": recover_cfg,
    },
    transitions=[
        OnTargetPoseReached(source="upright", target="tilt", robot_name="arm", axes=["z"], tolerance=0.005),
        OnTargetPoseReached(source="tilt", target="pour_arc", robot_name="arm", axes=["ry", "z"], tolerance=0.01),
        OnSuccess(source="pour_arc", target="recover", success_key="primitive_complete"),
        OnTargetPoseReached(source="recover", target="upright", robot_name="arm", axes=["x", "z", "ry", "rz"], tolerance=0.01),
    ],
)


class TeaPourDemoNet(ManipulationPrimitiveNet):
    connect = staticmethod(make_mock_connect({"arm": [0.0, 0.0, 0.20, 0.0, 0.0, 0.0]}))


def run_demo(steps: int = 16) -> None:
    net = TeaPourDemoNet(config)
    transition = net.reset()
    print(f"start -> {net.active_primitive}")

    for step in range(steps):
        transition = net.step(torch.zeros(net.action_dim))
        if transition[TransitionKey.DONE] or transition[TransitionKey.TRUNCATED]:
            net.reset()
        info = transition[TransitionKey.INFO]
        print(
            f"{step:02d} {info['transition_from']} -> {info['transition_to']} "
            f"reason={info['transition_reason']} progress={info.get('trajectory_progress', 0.0):.2f}"
        )


if __name__ == "__main__":
    run_demo()
