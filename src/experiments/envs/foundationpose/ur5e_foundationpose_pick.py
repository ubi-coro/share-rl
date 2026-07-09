"""UR5e pick pipeline with explicit FoundationPose and grasp-pose TODO hooks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import Camera
from lerobot.envs import EnvConfig
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator

from experiments.envs.foundationpose.primitives import FoundationPosePrimitive, RuntimeFrameTargetPrimitiveConfig, \
    FoundationPosePrimitiveConfig
from share.cameras.configuration_realsense_depth import RealSenseDepthCameraConfig
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    MoveDeltaPrimitiveConfig,
    ObservationConfig, EventConfig,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, TaskFrame, PolicyMode, ControlSpace
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.transitions import (
    OnFailure,
    OnSuccess,
    OnTargetPoseReached,
    OnTimeLimit,
)
from pose_estimation import GraspObjectSpec
from share.robots.ur import URConfig
from share.teleoperators import TeleopEvents
from share.teleoperators.spacemouse import SpaceMouseConfig

import logging

logger = logging.getLogger(__name__)


def _shared_processor(gripper_pos: float=0.0) -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=30,
        observation=ObservationConfig(
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=True,
            add_ee_pos_to_observation=True,
            add_joint_position_to_observation=False,
        ),
        gripper=GripperConfig(
            enable=False,
            static_pos=gripper_pos,
        ),
    )



_NO_FORCE_OVERRIDES = {"use_force_mode": False}


def get_target_prim_cfg(target: list[float], processor: ManipulationPrimitiveProcessorConfig) -> ManipulationPrimitiveConfig:
    return ManipulationPrimitiveConfig(
        notes="Move to a known safe start pose.",
        processor=processor,
        task_frame=TaskFrame(
            target=target,
            policy_mode=[None] * 6,
            control_mode=[ControlMode.POS] * 6,
            controller_overrides=_NO_FORCE_OVERRIDES,
        ),
    )


def get_object_relative_grasp_prim_cfg(
    grasp_pose: list[float],
    processor: ManipulationPrimitiveProcessorConfig,
) -> RuntimeFrameTargetPrimitiveConfig:
    return RuntimeFrameTargetPrimitiveConfig(
        notes="Move to the fixed grasp pose expressed in the estimated object frame.",
        processor=processor,
        task_frame=TaskFrame(
            target=list(grasp_pose),
            policy_mode=[None] * 6,
            control_mode=[ControlMode.POS] * 6,
            controller_overrides=_NO_FORCE_OVERRIDES,
        ),
        frame_origin_runtime_key="object_pose",
    )



@EnvConfig.register_subclass("ur5e_foundationpose_pick")
@dataclass
class UR5eFoundationPosePickEnvConfig(ManipulationPrimitiveNetConfig):
    """UR5e pipeline: scan pose, FoundationPose, grasp target, close gripper."""

    robot_ip: str = "172.22.22.2"
    fps: int = 30
    offline: bool = False
    start_primitive: str = "move_to_scan_pose"
    reset_primitive: str = "move_to_scan_pose"
    camera_serial_number: str = "352122271533"
    object_dir: str = ""
    calibration_file: str = ""
    target_tolerance: list[float] = field(default_factory=lambda: [0.01, 0.01, 0.01, 0.10, 0.10, 0.10])
    closed_gripper_position: float = 1.0
    open_gripper_position: float = 0.0
    gripper_hold_steps: int = 15
    mock_initial_pose: list[float] = field(default_factory=lambda: [0.45, -0.20, 0.35, 3.14, 0.0, 0.0])

    def make(self):
        return super().make()

    def __post_init__(self) -> None:
        object_dir = Path(self.object_dir)
        grasp_obj = str(object_dir / "object_spec.json")

        poses = json.loads((object_dir / "poses.json").read_text())
        scan_pose = poses["estimation_pose_top"]
        scan_pose_2 = poses["estimation_pose_stretched"]
        stretch_pose = poses["stretch_pose"]
        plug_pose = poses["plug_pose"]
        grasp_pose_hanging = poses["grasp_pose_hanging"]
        grasp_pose_insert = poses["grasp_pose_insert"]

        move_processor = _shared_processor(self.open_gripper_position)
        close_gripper_processor = _shared_processor(self.closed_gripper_position)
        open_gripper_processor = _shared_processor(self.open_gripper_position)

        self.robot = URConfig(
            robot_ip=self.robot_ip,
            frequency=500,
            soft_real_time=True,
            rt_core=3,
            use_gripper=True,
            use_force_mode=False,
            simple_pose_use_servo=False
        )
        self.teleop = SpaceMouseConfig(action_scale=[0.25, 0.25, 0.20, 0.50, 0.50, 0.50])
        self.cameras = {
            "main": RealSenseDepthCameraConfig(
                serial_number_or_name=self.camera_serial_number,
                use_depth=True
            ),
        }

        self.primitives = {
            "move_to_scan_pose": get_target_prim_cfg(scan_pose, move_processor),
            "estimate_object_pose": FoundationPosePrimitiveConfig(
                notes="Move the UR5e to the predefined scan pose before running FoundationPose.",
                task_description="estimate object pose",
                grasp_obj=grasp_obj,
                calibration_file=self.calibration_file,
            ),
            "move_to_grasp_pose": get_object_relative_grasp_prim_cfg(grasp_pose_hanging, move_processor),
            "close_gripper": get_object_relative_grasp_prim_cfg(grasp_pose_hanging, close_gripper_processor),
            "move_to_stretch_pose": get_target_prim_cfg(stretch_pose, close_gripper_processor),
            "open_gripper": get_object_relative_grasp_prim_cfg(grasp_pose_hanging, open_gripper_processor),

            "move_to_scan_pose_2": get_target_prim_cfg(scan_pose_2, open_gripper_processor),
            "estimate_object_pose_2": FoundationPosePrimitiveConfig(
                notes="Move the UR5e to the predefined scan pose before running FoundationPose.",
                task_description="estimate object pose",
                grasp_obj=grasp_obj,
                calibration_file=self.calibration_file,
            ),
            "move_to_grasp_pose_2": get_object_relative_grasp_prim_cfg(grasp_pose_insert, move_processor),
            "close_gripper_2": get_object_relative_grasp_prim_cfg(grasp_pose_insert, close_gripper_processor),
            "move_to_plug_pose": get_target_prim_cfg(plug_pose, close_gripper_processor),
            "move_to_scan_pose_3": get_target_prim_cfg(scan_pose, open_gripper_processor),
            "open_gripper_2": get_object_relative_grasp_prim_cfg(grasp_pose_insert, open_gripper_processor),
        }
        self.transitions = [
            OnTargetPoseReached(
                source="move_to_scan_pose",
                target="estimate_object_pose",
                tolerance=list(self.target_tolerance),
            ),
            OnSuccess(
                source="estimate_object_pose",
                target="move_to_grasp_pose",
            ),
            OnFailure(
                source="estimate_object_pose",
                target="move_to_scan_pose",
            ),
            OnTargetPoseReached(
                source="move_to_grasp_pose",
                target="close_gripper",
                tolerance=list(self.target_tolerance),
            ),
            OnTimeLimit(
                source="close_gripper",
                target="move_to_stretch_pose",
                max_steps=int(self.gripper_hold_steps),
            ),
            OnTargetPoseReached(
                source="move_to_stretch_pose",
                target="move_to_scan_pose_2",
                tolerance=list(self.target_tolerance),
            ),
            OnTimeLimit(
                source="open_gripper",
                target="move_to_scan_pose_2",
                max_steps=int(self.gripper_hold_steps),
            ),


            OnTimeLimit(
                source="move_to_scan_pose_2",
                target="estimate_object_pose_2",
                max_steps=int(90),
            ),
            OnSuccess(
                source="estimate_object_pose_2",
                target="move_to_grasp_pose_2",
            ),
            OnFailure(
                source="estimate_object_pose_2",
                target="move_to_scan_pose_2",
            ),
            OnTargetPoseReached(
                source="move_to_grasp_pose_2",
                target="close_gripper_2",
                tolerance=list(self.target_tolerance),
            ),
            OnTimeLimit(
                source="close_gripper_2",
                target="move_to_plug_pose",
                max_steps=int(self.gripper_hold_steps),
            ),
            OnTimeLimit(
                source="move_to_plug_pose",
                target="move_to_scan_pose_3",
                max_steps=int(80),
            ),
            OnTimeLimit(
                source="move_to_scan_pose_3",
                target="move_to_scan_pose",
                max_steps=int(150),
            ),
            OnTimeLimit(
                source="open_gripper_2",
                target="move_to_scan_pose",
                max_steps=int(self.gripper_hold_steps),
            ),
        ]

        super().__post_init__()


__all__ = [
    "UR5eFoundationPosePickEnvConfig",
]
