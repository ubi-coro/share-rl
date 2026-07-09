"""Experiment config for bimanual cooperative UR robot control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

from lerobot.envs import EnvConfig
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from share.teleoperators import TeleopEvents

from share.envs.manipulation_primitive.config_manipulation_primitive import (
    EventConfig,
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    ObservationConfig,
)
from share.envs.manipulation_primitive.env_manipulation_primitive import ManipulationPrimitive
from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    PolicyMode,
    TaskFrame,
    TASK_FRAME_AXIS_NAMES,
)
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (
    ManipulationPrimitiveNetConfig,
)
from share.envs.manipulation_primitive_net.transitions import OnSuccess
from share.robots.ur import URConfig
from share.utils.transformation_utils import (
    homogeneous_to_sixvec,
    sixvec_to_homogeneous,
    rotvec_to_euler_xyz,
)


def _shared_processor() -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=30.0,
        observation=ObservationConfig(
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=True,
            add_ee_pos_to_observation=True,
            add_joint_position_to_observation=False,
        ),
        gripper=GripperConfig(enable=True, discretize=True),
        events=EventConfig(
            key_mapping={
                TeleopEvents.SUCCESS: "s",
            },
        ),
    )


@ManipulationPrimitiveConfig.register_subclass("mapped_primitive")
@dataclass
class MappedManipulationPrimitiveConfig(ManipulationPrimitiveConfig):
    teleop_mapping: dict[str, str] = field(default_factory=dict)

    def make(self, robot_dict, teleop_dict, cameras, device="cpu"):
        mapped_teleop = {}
        for target, source in self.teleop_mapping.items():
            if source in teleop_dict:
                mapped_teleop[target] = teleop_dict[source]
        return super().make(robot_dict, mapped_teleop, cameras, device=device)


class SynchronousArmPrimitive(ManipulationPrimitive):
    """Primitive env that moves both arms in unison relative to a V-TCP."""

    def __init__(
        self,
        task_frame: dict[str, TaskFrame],
        robot_dict: dict[str, Robot],
        cameras: dict[str, Any],
        display_cameras: bool = False,
        right_arm_base_pose_in_left_base: list[float] | None = None,
        fps: float = 30.0,
    ):
        super().__init__(
            task_frame=task_frame,
            robot_dict=robot_dict,
            cameras=cameras,
            display_cameras=display_cameras,
        )
        if right_arm_base_pose_in_left_base is None:
            # Default right base is offset by 80cm along world X
            right_arm_base_pose_in_left_base = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Rigid offset transform between the two robot base frames
        self.T_leftbase_rightbase = sixvec_to_homogeneous(right_arm_base_pose_in_left_base)
        self.fps = fps
        self.reset_runtime_state()

    def reset_runtime_state(self) -> None:
        super().reset_runtime_state()
        self._T_v_tcp_left: np.ndarray | None = None
        self._T_v_tcp_right: np.ndarray | None = None
        self._T_world_v_tcp: np.ndarray | None = None
        self._prev_left_target: list[float] | None = None
        self._prev_right_target: list[float] | None = None
        self._initialized = False

    def step(self, action: dict[str, dict[str, float]]) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        # Read observations
        left_obs = self.robot_dict["left"].get_observation()
        right_obs = self.robot_dict["right"].get_observation()

        if not self._initialized:
            left_pose_raw = [left_obs[f"{ax}.ee_pos"] for ax in TASK_FRAME_AXIS_NAMES]
            right_pose_raw = [right_obs[f"{ax}.ee_pos"] for ax in TASK_FRAME_AXIS_NAMES]

            T_world_left = sixvec_to_homogeneous(left_pose_raw)
            T_rightbase_right = sixvec_to_homogeneous(right_pose_raw)
            T_world_right = self.T_leftbase_rightbase @ T_rightbase_right

            # Initialize V-TCP at the midpoint
            p_v_tcp = 0.5 * (T_world_left[:3, 3] + T_world_right[:3, 3])
            R_v_tcp = np.eye(3)
            T_world_v_tcp = np.eye(4)
            T_world_v_tcp[:3, :3] = R_v_tcp
            T_world_v_tcp[:3, 3] = p_v_tcp

            # Store V-TCP and relative offsets
            self._T_world_v_tcp = T_world_v_tcp
            self._T_v_tcp_left = np.linalg.inv(T_world_v_tcp) @ T_world_left
            self._T_v_tcp_right = np.linalg.inv(T_world_v_tcp) @ T_world_right
            self._prev_left_target = left_pose_raw
            self._prev_right_target = right_pose_raw
            self._initialized = True

        # Extract commands from the action dict (which are relative deltas when policy_mode is RELATIVE)
        left_cmd = action.get("left", {})
        dx = float(left_cmd.get("x.ee_pos", 0.0))
        dy = float(left_cmd.get("y.ee_pos", 0.0))
        dz = float(left_cmd.get("z.ee_pos", 0.0))
        drx = float(left_cmd.get("rx.ee_pos", 0.0))
        dry = float(left_cmd.get("ry.ee_pos", 0.0))
        drz = float(left_cmd.get("rz.ee_pos", 0.0))

        print(f"[DEBUG COOP STEP] dx={dx:.5f}, dy={dy:.5f}, dz={dz:.5f} | left_obs_x={left_obs['x.ee_pos']:.4f}, right_obs_x={right_obs['x.ee_pos']:.4f}", flush=True)

        # Update V-TCP position and orientation (scaled by dt to convert velocities to step displacements)
        dt = 1.0 / self.fps
        if self._T_world_v_tcp is not None:
            self._T_world_v_tcp[:3, 3] += np.array([dx, dy, dz]) * dt
            rot = R.from_euler("xyz", np.array([drx, dry, drz]) * dt).as_matrix()
            self._T_world_v_tcp[:3, :3] = rot @ self._T_world_v_tcp[:3, :3]

        # Compute individual arm target transformations
        T_world_target_left = self._T_world_v_tcp @ self._T_v_tcp_left
        T_world_target_right = self._T_world_v_tcp @ self._T_v_tcp_right

        # Left target is in left base coordinates directly
        left_target_pose = homogeneous_to_sixvec(T_world_target_left)
        
        # Right target needs to be converted back to the right base coordinate system
        T_rightbase_target_right = np.linalg.inv(self.T_leftbase_rightbase) @ T_world_target_right
        right_target_pose = homogeneous_to_sixvec(T_rightbase_target_right)

        # Convert target absolute pose to command format expected by controller (cancel dt if relative)
        left_mode = self.task_frame["left"].policy_mode
        left_prev_rot = [self._prev_left_target[i] for i in range(3, 6)]
        left_delta_rot = (R.from_rotvec(left_target_pose[3:]) * R.from_rotvec(left_prev_rot).inv()).as_rotvec()

        left_act_dict = {}
        for i, ax in enumerate(TASK_FRAME_AXIS_NAMES):
            if i < 3:
                val = left_target_pose[i]
                if left_mode[i] == PolicyMode.RELATIVE:
                    val = (val - self._prev_left_target[i]) * self.fps
            else:
                if left_mode[i] == PolicyMode.RELATIVE:
                    val = left_delta_rot[i - 3] * self.fps
                else:
                    left_target_euler = rotvec_to_euler_xyz(left_target_pose[3:])
                    val = left_target_euler[i - 3]
            left_act_dict[f"{ax}.ee_pos"] = val

        # Update the tracked left target pose for relative delta command calculations on next step
        self._prev_left_target = left_target_pose

        right_mode = self.task_frame["right"].policy_mode
        right_prev_rot = [self._prev_right_target[i] for i in range(3, 6)]
        right_delta_rot = (R.from_rotvec(right_target_pose[3:]) * R.from_rotvec(right_prev_rot).inv()).as_rotvec()
        right_target_euler = rotvec_to_euler_xyz(right_target_pose[3:])

        right_act_dict = {}
        for i, ax in enumerate(TASK_FRAME_AXIS_NAMES):
            if i < 3:
                val = right_target_pose[i]
                if right_mode[i] == PolicyMode.RELATIVE:
                    val = (val - self._prev_right_target[i]) * self.fps
            else:
                if right_mode[i] == PolicyMode.RELATIVE:
                    val = right_delta_rot[i - 3] * self.fps
                else:
                    val = right_target_euler[i - 3]
            right_act_dict[f"{ax}.ee_pos"] = val

        # Update the tracked right target pose for relative delta command calculations on next step
        self._prev_right_target = right_target_pose

        # Assemble the action dict for both robot arms
        cooperative_action = {
            "left": left_act_dict,
            "right": right_act_dict,
        }

        # Command both grippers in sync using the left arm's teleoperated gripper action
        if "gripper.pos" in action.get("left", {}):
            cooperative_action["left"]["gripper.pos"] = action["left"]["gripper.pos"]
            cooperative_action["right"]["gripper.pos"] = action["left"]["gripper.pos"]

        return super().step(cooperative_action)


@ManipulationPrimitiveConfig.register_subclass("synchronous_arm")
@dataclass
class SynchronousArmPrimitiveConfig(ManipulationPrimitiveConfig):
    """Primitive config that builds the SynchronousArmPrimitive environment."""

    right_arm_base_pose_in_left_base: list[float] = field(
        default_factory=lambda: [0.8, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    def make(
        self,
        robot_dict: dict[str, Robot],
        teleop_dict: dict[str, Teleoperator],
        cameras: dict[str, Any],
        device: str = "cpu",
    ):
        self.validate(robot_dict, teleop_dict)
        self.infer_features(robot_dict, cameras)

        display_cameras = (
            self.processor.image_preprocessing is not None
            and self.processor.image_preprocessing.display_cameras
        )
        
        env = SynchronousArmPrimitive(
            task_frame=self.task_frame,
            robot_dict=robot_dict,
            cameras=cameras,
            display_cameras=display_cameras,
            right_arm_base_pose_in_left_base=self.right_arm_base_pose_in_left_base,
            fps=self.processor.fps,
        )

        env_processor = self.make_env_processor(device)
        action_processor = self.make_action_processor(robot_dict, teleop_dict, device)
        return env, env_processor, action_processor


@EnvConfig.register_subclass("demo_ur_bimanual_cooperative")
@dataclass
class DemoURBimanualCooperativeEnvConfig(ManipulationPrimitiveNetConfig):
    """Orchestrates left-only teleop, right-only teleop, and cooperative dual-arm control."""

    fps: int = 30
    start_primitive: str = "left_arm"
    reset_primitive: str = "left_arm"

    def __post_init__(self):
        processor = _shared_processor()
        processor.fps = float(self.fps)

        # 1. Define both robot configs
        self.robot = {
            "left": URConfig(
                robot_ip="172.22.22.5",
                kp=[3000, 3000, 3000, 200, 200, 200],
                soft_real_time=True,
                rt_core=3,
                use_gripper=True,
            ),
            "right": URConfig(
                robot_ip="172.22.22.2",
                kp=[3000, 3000, 3000, 200, 200, 200],
                soft_real_time=True,
                rt_core=3,
                use_gripper=True,
            ),
        }

        # 2. Map teleoperation to the Left arm's control space
        self.teleop = {
            "left": SpaceMouseConfig(
                action_scale=[0.05, 0.05, 0.2, 0.1, 0.1, 0.1]
            )
        }

        # 3. Define the primitives
        # Left-only controls
        left_arm_primitive = ManipulationPrimitiveConfig(
            task_frame={
                "left": TaskFrame(
                    target=[0.0] * 6,
                    space=ControlSpace.TASK,
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=[PolicyMode.RELATIVE] * 6,
                ),
                "right": TaskFrame(
                    target=[0.0] * 6,
                    space=ControlSpace.TASK,
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=[None] * 6, # Stiff/Lock
                ),
            },
            processor=processor,
            notes="Move the left arm independently using the SpaceMouse.",
        )

        # Right-only controls
        right_arm_primitive = MappedManipulationPrimitiveConfig(
            teleop_mapping={"right": "left"},
            task_frame={
                "left": TaskFrame(
                    target=[0.0] * 6,
                    space=ControlSpace.TASK,
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=[None] * 6, # Stiff/Lock
                ),
                "right": TaskFrame(
                    target=[0.0] * 6,
                    space=ControlSpace.TASK,
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=[PolicyMode.RELATIVE] * 6,
                ),
            },
            processor=processor,
            notes="Move the right arm independently using the SpaceMouse.",
        )

        # Cooperative controls
        cooperative_primitive = SynchronousArmPrimitiveConfig(
            task_frame={
                "left": TaskFrame(
                    target=[0.0] * 6,
                    space=ControlSpace.TASK,
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=[PolicyMode.RELATIVE] * 6,
                ),
                "right": TaskFrame(
                    target=[0.0] * 6,
                    space=ControlSpace.TASK,
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=[None] * 6,
                ),
            },
            processor=processor,
            notes="Move both arms in unison relative to the Virtual TCP.",
        )

        self.primitives = {
            "left_arm": left_arm_primitive,
            "right_arm": right_arm_primitive,
            "cooperative": cooperative_primitive,
        }

        # 4. Transitions cycling in a continuous loop: left -> right -> cooperative -> left
        self.transitions = [
            OnSuccess(source="left_arm", target="right_arm"),
            OnSuccess(source="right_arm", target="cooperative"),
            OnSuccess(source="cooperative", target="left_arm"),
        ]

        super().__post_init__()


# Avoid pylint/IDE warnings
from share.teleoperators.spacemouse import SpaceMouseConfig

__all__ = [
    "SynchronousArmPrimitive",
    "SynchronousArmPrimitiveConfig",
    "DemoURBimanualCooperativeEnvConfig",
]
