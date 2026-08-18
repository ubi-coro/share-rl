"""Conservative first task-frame motion for an already commissioned FR3."""

import time

from lerobot.utils.errors import DeviceNotConnectedError

from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    PolicyMode,
    TaskFrame,
)
from share.robots.franka import Franka, FrankaConfig


robot = Franka(
    FrankaConfig(
        robot_ip="172.16.0.2",
        enforce_realtime=True,
        # Set all three payload fields together before contact experiments.
        # payload_mass=...,
        # payload_center_of_mass=[...],
        # payload_inertia=[...],  # row-major 3x3 matrix
    )
)

try:
    robot.connect(calibrate=False)
    robot.zero_ft()
    observation = robot.get_observation()
    current_pose = [
        observation[f"{axis}.ee_pos"]
        for axis in ("x", "y", "z", "rx", "ry", "rz")
    ]

    robot.set_task_frame(
        TaskFrame(
            space=ControlSpace.TASK,
            target=[0.0] * 6,
            policy_mode=[PolicyMode.RELATIVE] * 6,
            control_mode=[ControlMode.POS] * 6,
            origin=[0.0] * 6,
            min_pose=[
                current_pose[0] - 0.03,
                current_pose[1] - 0.03,
                current_pose[2] - 0.03,
                -3.14159,
                -3.14159,
                -3.14159,
            ],
            max_pose=[
                current_pose[0] + 0.03,
                current_pose[1] + 0.03,
                current_pose[2] + 0.03,
                3.14159,
                3.14159,
                3.14159,
            ],
            controller_overrides={
                "kp": [200.0, 200.0, 200.0, 20.0, 20.0, 20.0],
                "kd": [28.3, 28.3, 28.3, 8.9, 8.9, 8.9],
                "wrench_limits": [15.0, 15.0, 15.0, 2.0, 2.0, 2.0],
                "compliance_reference_limit_enable": [True] * 6,
            },
        )
    )

    # Relative POS targets are velocities integrated by the 500 Hz controller.
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        robot.send_action({"x.ee_pos": 0.01})
        time.sleep(0.01)

    for _ in range(20):
        robot.send_action({"x.ee_pos": 0.0})
        time.sleep(0.01)
finally:
    try:
        robot.disconnect()
    except DeviceNotConnectedError:
        pass
