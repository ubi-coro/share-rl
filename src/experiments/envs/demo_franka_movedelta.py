"""Smoke-test env for the Franka backend: SpaceMouse teleop plus a scripted
p2p delta move, transitioning back and forth.

Mirrors demo_movedelta.py's UR pattern, adapted for Franka's position-only
task-space contract (no use_force_mode/simple_pose_use_servo -- those are
UR-specific; Franka's controller_overrides are translational_stiffness/
rotational_stiffness/compliance_reference_limit_enable, see command.py).

Exercises both control paths landed this session in one place: the
teleop primitive drives CartesianReferenceController's RELATIVE/velocity
integration, and the delta primitive drives its ABSOLUTE/p2p path (target
resolved once at primitive entry, held until the arm converges).

Complete wiki/franka_setup.rst's acceptance progression before running this
against real hardware -- replace robot_ip below first.
"""

import os
from dataclasses import dataclass

from lerobot.envs import EnvConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from pynput import keyboard

from share.envs.manipulation_primitive.config_manipulation_primitive import (
    EventConfig,
    GripperConfig,
    ImagePreprocessingConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    MoveDeltaPrimitiveConfig,
    ObservationConfig,
)
from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    PolicyMode,
    TaskFrame,
)
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (
    ManipulationPrimitiveNetConfig,
)
from share.envs.manipulation_primitive_net.transitions import OnSuccess, OnTargetPoseReached
from share.robots.franka import FrankaConfig
from share.teleoperators import TeleopEvents
from share.teleoperators.spacemouse import SpaceMouseConfig

processor = ManipulationPrimitiveProcessorConfig(
    observation=ObservationConfig(
        add_ee_velocity_to_observation=True,
        add_ee_wrench_to_observation=True,
        add_ee_pos_to_observation=True,
        add_joint_position_to_observation=True,
    ),
    # Required for any configured camera: this is the only step that converts a raw
    # numpy camera frame into a torch.Tensor (DeviceProcessorStep, which runs right
    # after, only moves tensors -- it silently passes non-tensor values through).
    # Without it record_loop()'s v.squeeze().cpu() fails on the raw numpy image.
    image_preprocessing=ImagePreprocessingConfig(),
    # Franka Hand, hardcoded like UR's Robotiq gripper -- see FrankaConfig.use_gripper
    # below. Franka(hand=FrankaHandWorker(...)) already exposes it as the standard
    # "gripper.pos" action/observation key, so enabling it here is the only wiring
    # this demo needs.
    gripper=GripperConfig(enable=True),
    events=EventConfig(
        key_mapping={
            TeleopEvents.RERECORD_EPISODE: keyboard.Key.left,
            TeleopEvents.STOP_RECORDING: keyboard.Key.down,
            TeleopEvents.SUCCESS: keyboard.Key.right,
        },
    ),
)

# Same conservative profile as franka_first_motion.py / franka_teleop_spacemouse.py.
_CONTROLLER_OVERRIDES = {
    "translational_stiffness": 200.0,
    "rotational_stiffness": 20.0,
    "compliance_reference_limit_enable": [False] * 6,
}

mp_teleop = ManipulationPrimitiveConfig(
    task_frame=TaskFrame(
        target=[0.0] * 6,
        space=ControlSpace.TASK,
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[PolicyMode.RELATIVE] * 6,
        controller_overrides=_CONTROLLER_OVERRIDES,
    ),
    processor=processor,
    notes="SpaceMouse-driven relative teleop -- exercises the RELATIVE/velocity path.",
)

mp_delta = MoveDeltaPrimitiveConfig(
    delta_frame="world",
    delta=[0.0, 0.0, 0.02, 0.0, 0.0, 0.0],
    task_frame=TaskFrame(
        target=[0.0] * 6,
        space=ControlSpace.TASK,
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[None] * 6,
        controller_overrides=_CONTROLLER_OVERRIDES,
    ),
    processor=processor,
    notes="Scripted 2 cm lift resolved once at entry -- exercises the ABSOLUTE/p2p path.",
)


@EnvConfig.register_subclass("demo_franka_movedelta")
@dataclass
class DemoFrankaMoveDeltaEnvConfig(ManipulationPrimitiveNetConfig):
    fps: int = 30
    start_primitive: str = "teleop"
    reset_primitive: str = "teleop"

    def __post_init__(self):
        # Set FRANKA_ROBOT_IP to test against a local franky-sim instance instead of
        # real hardware -- see examples/robots/run_franky_sim.py, which prints the
        # hostname to export. Also relaxes enforce_realtime, since a dev machine
        # testing against sim usually lacks the realtime-kernel setup a real
        # connection requires (unrelated to real-vs-simulated robot, just local
        # thread scheduling permissions).
        sim_robot_ip = os.environ.get("FRANKA_ROBOT_IP")

        self.robot = FrankaConfig(
            # TODO: replace with the FR3's actual IP (see wiki/franka_setup.rst).
            robot_ip=sim_robot_ip or "172.16.0.2",
            enforce_realtime=sim_robot_ip is None,
            translational_stiffness=200.0,
            rotational_stiffness=20.0,
            force_constraints=[15.0, 15.0, 15.0, 2.0, 2.0, 2.0],
            compliance_reference_limit_enable=[True] * 6,
            use_gripper=True,
        )

        self.teleop = SpaceMouseConfig(
            action_scale=[0.1, 0.1, 0.1, 0.5, 0.5, 0.5],
            button_mapping={0: {"event": TeleopEvents.SUCCESS, "toggle": False}},
        )

        self.cameras = {
            "main": OpenCVCameraConfig(index_or_path="/dev/video0")
        }

        self.primitives = {
            "teleop": mp_teleop,
            "delta": mp_teleop,
        }

        self.transitions = [
            OnSuccess(source="teleop", target="delta"),
            OnTargetPoseReached(source="delta", target="teleop"),
        ]

        super().__post_init__()
