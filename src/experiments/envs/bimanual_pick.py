"""Bimanual pick: teleop+grasp left, teleop+grasp right, then teleop the shared midpoint.

One SpaceMouse drives whichever arm is active via teleop_mapping. Uses env_class to plug a
custom cooperative-frame env into a plain ManipulationPrimitiveConfig -- no bespoke Config
subclass needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.envs import EnvConfig
from lerobot.robots import Robot
from pynput import keyboard
from scipy.spatial.transform import Rotation

from share.envs.manipulation_primitive.config_manipulation_primitive import (
    EventConfig,
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    MoveDeltaPrimitiveConfig,
    ObservationConfig,
)
from share.envs.manipulation_primitive.env_manipulation_primitive import ManipulationPrimitive
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.transitions import OnSuccess
from share.robots.ur import SimURConfig, URConfig
from share.teleoperators import TeleopEvents
from share.teleoperators.spacemouse import SpaceMouseConfig
from share.utils.transformation_utils import (
    euler_xyz_from_rotation,
    euler_xyz_from_rotvec,
    rotation_from_extrinsic_xyz,
    task_pose_to_world_pose,
    world_pose_to_task_pose,
)


def _pose_from_raw_obs(obs: dict[str, float]) -> list[float]:
    rotvec = [obs["rx.ee_pos"], obs["ry.ee_pos"], obs["rz.ee_pos"]]
    return [obs["x.ee_pos"], obs["y.ee_pos"], obs["z.ee_pos"], *euler_xyz_from_rotvec(rotvec)]


class CooperativeFramePrimitive(ManipulationPrimitive):
    """Drives N robots from one teleop input via a shared virtual midpoint frame.

    On first step, captures each robot's fixed offset from the midpoint of all robot poses.
    Each step, integrates the driver's raw teleop delta into the midpoint, then re-projects
    every robot's target from it. The driver's task frame must be RELATIVE POS (its per-step
    delta is read from the incoming action); other robots must be POS with policy_mode=None.
    Anti-windup is left to controller.py's compliance_reference_limit_enable -- set it via
    controller_overrides on every robot, this primitive does not clamp anything itself.
    """

    def __init__(
        self,
        task_frame: dict[str, TaskFrame],
        robot_dict: dict[str, Robot],
        cameras: dict[str, Any],
        display_cameras: bool = False,
        driver: str = "left",
        fps: float = 30.0,
    ):
        super().__init__(task_frame, robot_dict, cameras, display_cameras)
        self.driver = driver
        self.fps = fps
        self._vtcp_world: list[float] | None = None
        self._offset_from_vtcp: dict[str, list[float]] = {}
        self._prev_target_world: dict[str, list[float]] = {}

    def reset_runtime_state(self) -> None:
        super().reset_runtime_state()
        self._vtcp_world = None
        self._offset_from_vtcp = {}
        self._prev_target_world = {}

    def _poses_world(self) -> dict[str, list[float]]:
        return {
            name: task_pose_to_world_pose(_pose_from_raw_obs(robot.get_observation()), self.task_frame[name].origin)
            for name, robot in self.robot_dict.items()
        }

    def step(self, action: dict[str, dict[str, float]]):
        pose_world = self._poses_world()

        if self._vtcp_world is None:
            names = list(self.robot_dict)
            midpoint = [sum(pose_world[n][i] for n in names) / len(names) for i in range(3)]
            self._vtcp_world = [*midpoint, 0.0, 0.0, 0.0]
            for name in names:
                self._offset_from_vtcp[name] = world_pose_to_task_pose(pose_world[name], self._vtcp_world)
                self._prev_target_world[name] = list(pose_world[name])

        dt = 1.0 / self.fps
        driver_delta = action.get(self.driver, {})
        d = [float(driver_delta.get(f"{ax}.ee_pos", 0.0)) for ax in ("x", "y", "z", "rx", "ry", "rz")]

        self._vtcp_world[0] += d[0] * dt
        self._vtcp_world[1] += d[1] * dt
        self._vtcp_world[2] += d[2] * dt
        new_rot = rotation_from_extrinsic_xyz(*[v * dt for v in d[3:]]) * rotation_from_extrinsic_xyz(*self._vtcp_world[3:])
        self._vtcp_world[3:] = euler_xyz_from_rotation(new_rot)

        cooperative_action: dict[str, dict[str, float]] = {}
        for name, frame in self.task_frame.items():
            target_world = task_pose_to_world_pose(self._offset_from_vtcp[name], self._vtcp_world)
            prev_world = self._prev_target_world[name]

            if name == self.driver:
                # RELATIVE POS axes: send a velocity the controller integrates over its own
                # dt, converging toward target_world by roughly the next outer step.
                lin_vel = [(target_world[i] - prev_world[i]) / dt for i in range(3)]
                rot_vel = (
                    rotation_from_extrinsic_xyz(*target_world[3:]) * rotation_from_extrinsic_xyz(*prev_world[3:]).inv()
                ).as_rotvec() / dt
                values = [*lin_vel, *rot_vel]
            else:
                values = world_pose_to_task_pose(target_world, frame.origin)

            cooperative_action[name] = dict(zip(("x.ee_pos", "y.ee_pos", "z.ee_pos", "rx.ee_pos", "ry.ee_pos", "rz.ee_pos"), values))
            if "gripper.pos" in action.get(name, {}):
                cooperative_action[name]["gripper.pos"] = action[name]["gripper.pos"]
            self._prev_target_world[name] = target_world

        return super().step(cooperative_action)


def _processor(fps: float, gripper: GripperConfig) -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=fps,
        observation=ObservationConfig(
            add_ee_pos_to_observation=True,
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=True,
            add_joint_position_to_observation=False,
        ),
        gripper=gripper,
        events=EventConfig(
            key_mapping={
                TeleopEvents.SUCCESS: keyboard.Key.space,
                TeleopEvents.RERECORD_EPISODE: keyboard.Key.left,
                TeleopEvents.STOP_RECORDING: keyboard.Key.down,
            },
            pulse_events=(TeleopEvents.SUCCESS,),
        ),
    )


def _gripper(live: str, held: dict[str, float]) -> GripperConfig:
    enable = {live: True}
    static_pos: dict[str, float | None] = {live: None}
    for name, pos in held.items():
        enable[name] = False
        static_pos[name] = pos
    return GripperConfig(enable=enable, discretize=True, min_pos=0.5, static_pos=static_pos)


def _single_arm_primitive(
    active: str,
    passive: str,
    processor: ManipulationPrimitiveProcessorConfig,
    controller_overrides: dict[str, Any],
) -> MoveDeltaPrimitiveConfig:
    return MoveDeltaPrimitiveConfig(
        notes=f"Teleop {active} 6-DoF; {passive} holds its entry pose.",
        processor=processor,
        delta={active: [0.0] * 6, passive: [0.0] * 6},
        task_frame={
            active: TaskFrame(
                target=[0.0] * 6,
                control_mode=[ControlMode.POS] * 6,
                policy_mode=[PolicyMode.RELATIVE] * 6,
                controller_overrides=controller_overrides,
            ),
            passive: TaskFrame(
                target=[0.0] * 6,
                control_mode=[ControlMode.POS] * 6,
                policy_mode=[None] * 6,
                controller_overrides=controller_overrides,
            ),
        },
    )


@EnvConfig.register_subclass("bimanual_pick")
@dataclass
class BimanualPickEnvConfig(ManipulationPrimitiveNetConfig):
    """teleop_left (grasp) -> teleop_right (grasp) -> teleop_midpoint (cooperative), looping."""

    fps: int = 30
    left_robot_ip: str = "172.22.22.5"
    right_robot_ip: str = "172.22.22.2"
    # right arm's base pose expressed in left arm's base frame -- calibrate for your cell
    right_base_pose_in_left_base: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    translation_action_scale: float = 0.1
    rotation_action_scale: float = 0.5
    open_gripper_position: float = 0.0
    closed_gripper_position: float = 1.0
    start_primitive: str = "teleop_left"
    reset_primitive: str = "teleop_left"
    # kinematic-only simulated arms instead of real UR hardware -- for previewing this graph
    # with a real SpaceMouse before running on the robots
    mock: bool = False

    def __post_init__(self) -> None:
        controller_overrides = {
            "use_force_mode": True,
            "compliance_reference_limit_enable": [True] * 6,
            "kp": [2000.0, 2000.0, 2000.0, 150.0, 150.0, 150.0],
            "kd": [60.0, 60.0, 60.0, 6.0, 6.0, 6.0],
            "wrench_limits": [15.0, 15.0, 15.0, 2.0, 2.0, 2.0],
        }

        if self.mock:
            self.robot = {
                "left": SimURConfig(use_gripper=True, initial_pose=[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]),
                "right": SimURConfig(use_gripper=True, initial_pose=[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]),
            }
        else:
            self.robot = {
                "left": URConfig(robot_ip=self.left_robot_ip, frequency=125, soft_real_time=True, rt_core=3, use_gripper=True),
                "right": URConfig(robot_ip=self.right_robot_ip, frequency=125, soft_real_time=True, rt_core=3, use_gripper=True),
            }
        self.teleop = {
            "left": SpaceMouseConfig(
                action_scale=[self.translation_action_scale] * 3 + [self.rotation_action_scale] * 3,
            ),
        }

        left_processor = _processor(self.fps, _gripper("left", {"right": self.open_gripper_position}))
        right_processor = _processor(self.fps, _gripper("right", {"left": self.closed_gripper_position}))
        midpoint_processor = _processor(
            self.fps,
            GripperConfig(enable=False, static_pos={"left": self.closed_gripper_position, "right": self.closed_gripper_position}),
        )

        midpoint_primitive = ManipulationPrimitiveConfig(
            notes="Teleop the shared midpoint; both arms track it cooperatively.",
            processor=midpoint_processor,
            env_class=CooperativeFramePrimitive,
            env_kwargs={"driver": "left", "fps": float(self.fps)},
            task_frame={
                "left": TaskFrame(
                    target=[0.0] * 6,
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=[PolicyMode.RELATIVE] * 6,
                    controller_overrides=controller_overrides,
                ),
                "right": TaskFrame(
                    target=[0.0] * 6,
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=[None] * 6,
                    origin=list(self.right_base_pose_in_left_base),
                    controller_overrides=controller_overrides,
                ),
            },
        )

        self.primitives = {
            "teleop_left": _single_arm_primitive("left", "right", left_processor, controller_overrides),
            "teleop_right": _single_arm_primitive("right", "left", right_processor, controller_overrides),
            "teleop_midpoint": midpoint_primitive,
        }
        self.primitives["teleop_right"].teleop_mapping = {"right": "left"}

        self.transitions = [
            OnSuccess(source="teleop_left", target="teleop_right"),
            OnSuccess(source="teleop_right", target="teleop_midpoint"),
            OnSuccess(source="teleop_midpoint", target="teleop_left"),
        ]

        super().__post_init__()


__all__ = [
    "CooperativeFramePrimitive",
    "BimanualPickEnvConfig",
]
