"""Simple pure 6-DoF SpaceMouse teleop with two switchable gain profiles."""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.envs import EnvConfig
from pynput import keyboard

from share.envs.manipulation_primitive.config_manipulation_primitive import (
    EventConfig,
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    ObservationConfig,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.transitions import OnSuccess
from share.robots.ur import URConfig
from share.teleoperators import TeleopEvents
from share.teleoperators.spacemouse import SpaceMouseConfig


def _shared_processor(fps: float) -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=fps,
        observation=ObservationConfig(
            add_ee_pos_to_observation=True,
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=True,
            add_joint_position_to_observation=False,
        ),
        gripper=GripperConfig(
            enable=True,
            discretize=True,
            min_pos=0.5,
        ),
        events=EventConfig(
            # Press space to swap gain profiles; left/down remain the usual
            # dataset-recording controls.
            key_mapping={
                TeleopEvents.SUCCESS: keyboard.Key.space,
                TeleopEvents.RERECORD_EPISODE: keyboard.Key.left,
                TeleopEvents.STOP_RECORDING: keyboard.Key.down,
            },
            pulse_events=(TeleopEvents.SUCCESS,),
        ),
    )


def _teleop_primitive(
    processor: ManipulationPrimitiveProcessorConfig,
    kp: list[float],
    kd: list[float],
    wrench_limits: list[float],
    notes: str,
) -> ManipulationPrimitiveConfig:
    return ManipulationPrimitiveConfig(
        notes=notes,
        processor=processor,
        task_frame=TaskFrame(
            target=[0.0] * 6,
            space=ControlSpace.TASK,
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[PolicyMode.RELATIVE] * 6,  # all 6 axes enabled
            origin=[0.0] * 6,
            controller_overrides={
                "use_force_mode": True,  # defaults False regardless of URConfig.use_force_mode
                "kp": list(kp),
                "kd": list(kd),
                "wrench_limits": list(wrench_limits),
            },
        ),
    )


@EnvConfig.register_subclass("teleop_spacemouse_6dof")
@dataclass
class TeleopSpaceMouse6DofEnvConfig(ManipulationPrimitiveNetConfig):
    """Pure 6-DoF SpaceMouse teleop, no autonomy.

    Two identical task frames ("teleop_soft" and "teleop_stiff") differing
    only in UR controller gains, so the same SpaceMouse input can be driven
    compliantly or precisely. Press the SUCCESS key (default: space) to swap
    between them without leaving teleop.
    """

    robot_ip: str = "172.22.22.2"
    fps: int = 30
    start_primitive: str = "teleop_soft"
    reset_primitive: str = "teleop_soft"

    # Low translation gains keep small SpaceMouse deflections from producing
    # large Cartesian jumps; rotation is scaled independently.
    translation_action_scale: float = 0.1
    rotation_action_scale: float = 0.5

    soft_kp: list[float] = field(default_factory=lambda: [1500.0, 1500.0, 1500.0, 100.0, 100.0, 100.0])
    soft_kd: list[float] = field(default_factory=lambda: [40.0, 40.0, 40.0, 3.0, 3.0, 3.0])
    soft_wrench_limits: list[float] = field(default_factory=lambda: [15.0, 15.0, 15.0, 2.0, 2.0, 2.0])

    stiff_kp: list[float] = field(default_factory=lambda: [3500.0, 3500.0, 3500.0, 220.0, 220.0, 220.0])
    stiff_kd: list[float] = field(default_factory=lambda: [90.0, 90.0, 90.0, 10.0, 10.0, 10.0])
    stiff_wrench_limits: list[float] = field(default_factory=lambda: [30.0, 30.0, 30.0, 3.0, 3.0, 3.0])

    def __post_init__(self) -> None:
        processor = _shared_processor(float(self.fps))

        self.robot = URConfig(
            robot_ip=self.robot_ip,
            frequency=125,
            soft_real_time=True,
            rt_core=3,
            use_gripper=True,
            compliance_reference_limit_enable=[True] * 6,
        )
        self.teleop = SpaceMouseConfig(
            action_scale=[
                self.translation_action_scale,
                self.translation_action_scale,
                self.translation_action_scale,
                self.rotation_action_scale,
                self.rotation_action_scale,
                self.rotation_action_scale,
            ],
        )

        self.primitives = {
            "teleop_soft": _teleop_primitive(
                processor,
                self.soft_kp,
                self.soft_kd,
                self.soft_wrench_limits,
                notes="6-DoF relative task-space teleop, soft/compliant gains.",
            ),
            "teleop_stiff": _teleop_primitive(
                processor,
                self.stiff_kp,
                self.stiff_kd,
                self.stiff_wrench_limits,
                notes="6-DoF relative task-space teleop, stiff/precise gains.",
            ),
        }
        self.transitions = [
            OnSuccess(source="teleop_soft", target="teleop_stiff"),
            OnSuccess(source="teleop_stiff", target="teleop_soft"),
        ]

        super().__post_init__()


__all__ = [
    "TeleopSpaceMouse6DofEnvConfig",
]
