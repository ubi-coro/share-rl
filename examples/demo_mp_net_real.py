"""Real UR example showing home, move-delta, adaptive, and open-loop primitives."""

from __future__ import annotations

import time

import torch
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.processor import TransitionKey
from lerobot.teleoperators import TeleopEvents
from lerobot.utils.robot_utils import precise_sleep

from share.envs.manipulation_primitive.config_manipulation_primitive import (
    EventConfig,
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    MoveDeltaPrimitiveConfig,
    OpenLoopTrajectorySpec,
    OpenLoopTrajectoryPrimitiveConfig,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.envs.manipulation_primitive_net.transitions import OnSuccess, OnTargetPoseReached
from share.robots.ur import URConfig
from share.teleoperators.spacemouse import SpacemouseConfig


EVENTS = EventConfig(
    foot_switch_mapping={
        (TeleopEvents.SUCCESS,): {"device": 4, "toggle": False},
    }
)

GRIPPER = GripperConfig(enable=True, discretize=True)


home_cfg = ManipulationPrimitiveConfig(
    notes="Drive to a comfortable inspection hover pose.",
    task_frame=TaskFrame(
        origin=[0.45, -0.10, 0.22, 0.0, 0.0, 0.0],
        target=[0.0, 0.0, 0.20, 0.0, 0.0, 0.0],
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    ),
    processor=ManipulationPrimitiveProcessorConfig(events=EVENTS, gripper=GRIPPER),
)

descend_cfg = MoveDeltaPrimitiveConfig(
    notes="Approach the work surface in world coordinates without changing yaw.",
    task_frame=TaskFrame(
        origin=[0.45, -0.10, 0.22, 0.0, 0.0, 0.0],
        target=[0.0, 0.0, 0.20, 0.0, 0.0, 0.0],
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    ),
    delta=[0.0, 0.0, -0.08, 0.0, 0.0, 0.0],
    delta_frame="world",
    processor=ManipulationPrimitiveProcessorConfig(events=EVENTS, gripper=GRIPPER),
)

inspect_cfg = ManipulationPrimitiveConfig(
    notes="Human/policy cooperatively sweeps the contact patch while keeping attitude fixed.",
    task_frame=TaskFrame(
        origin=[0.45, -0.10, 0.22, 0.0, 0.0, 0.0],
        target=[0.0, 0.0, 0.12, 0.0, 0.0, 0.0],
        policy_mode=[PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, None, None, None],
        control_mode=[ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS, ControlMode.POS],
    ),
    processor=ManipulationPrimitiveProcessorConfig(events=EVENTS, gripper=GRIPPER),
)

retract_cfg = OpenLoopTrajectoryPrimitiveConfig(
    notes="Scripted retreat that lifts and backs away before the next cycle.",
    task_frame=TaskFrame(
        origin=[0.45, -0.10, 0.22, 0.0, 0.0, 0.0],
        target=[0.0, 0.0, 0.12, 0.0, 0.0, 0.0],
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    ),
    trajectory=OpenLoopTrajectorySpec(
        delta=[0.0, -0.05, 0.14, 0.0, 0.0, 0.0],
        frame="world",
        duration_s=0.6,
    ),
    processor=ManipulationPrimitiveProcessorConfig(events=EVENTS, gripper=GRIPPER),
)


net_cfg = ManipulationPrimitiveNetConfig(
    fps=30,
    start_primitive="home",
    reset_primitive="home",
    primitives={
        "home": home_cfg,
        "descend": descend_cfg,
        "inspect": inspect_cfg,
        "retract": retract_cfg,
    },
    transitions=[
        OnTargetPoseReached(source="home", target="descend", axes=["z"], tolerance=0.01),
        OnTargetPoseReached(source="descend", target="inspect", axes=["z"], tolerance=0.01),
        OnSuccess(source="inspect", target="retract"),
        OnSuccess(source="retract", target="home", success_key="primitive_complete"),
    ],
    robot=URConfig(
        robot_ip="172.22.22.2",
        frequency=500,
        soft_real_time=True,
        rt_core=3,
        use_gripper=True,
    ),
    teleop=SpacemouseConfig(action_scale=[0.25, 0.25, 0.20, 0.50, 0.50, 0.50]),
    cameras={
        "main": RealSenseCameraConfig(serial_number_or_name="352122273250"),
    },
)


def run_demo() -> None:
    """Run one continuous inspection loop on the configured UR setup."""
    net = ManipulationPrimitiveNet(net_cfg)
    transition = net.reset()

    print(f"start -> {net.active_primitive}")

    for _step in range(100_000):
        loop_t0 = time.perf_counter()
        action = torch.zeros(net.action_dim, dtype=torch.float32)
        transition = net.step(action)

        if transition[TransitionKey.DONE] or transition[TransitionKey.TRUNCATED]:
            transition = net.reset()

        info = transition[TransitionKey.INFO]
        dt = time.perf_counter() - loop_t0
        precise_sleep(1 / net.config.fps - dt)
        print(
            f"[{net.active_primitive}] "
            f"primitive_step={info.get('primitive_step', 0):04d} "
            f"reason={info.get('transition_reason')} "
            f"progress={info.get('trajectory_progress', 0.0):.2f}"
        )


if __name__ == "__main__":
    run_demo()
