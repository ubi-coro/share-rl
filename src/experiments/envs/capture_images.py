"""Free 6-DoF teleop that snapshots camera+depth / robot-pose pairs to disk on a keypress.

Used to collect real-hardware test data for the FoundationPose pipeline (see
``experiments.envs.foundationpose``) before trusting pose estimates from it: teleop the
arm to wherever the object/scene should be, press the capture key, and one numbered
snapshot is appended to ``output_dir``. Deliberately has no dependency on the optional
``pose_estimation`` package -- it only needs a RealSense depth camera and a UR arm, so it
keeps working even on a workstation where FoundationPose itself isn't installed.

Each snapshot under ``output_dir/snapshots/<index>_*`` contains:
  - ``<index>_rgb.png``   -- raw color frame, straight off the camera (no crop/resize).
  - ``<index>_depth.npy`` -- raw depth frame in meters (float32), same resolution as the RGB.
  - ``<index>_meta.json`` -- robot pose (every raw ``main.*`` observation channel, so both
    the rotvec EE pose and joint positions are covered), camera intrinsics (3x3), depth
    scale, and the camera->gripper extrinsics loaded from ``calibration_file``.
``output_dir/snapshots.jsonl`` accumulates one line per snapshot (the same dict as the
per-snapshot meta file) so the whole set can be scanned without opening every file. The
calibration file itself is copied once into ``output_dir/calibration/`` so a capture set
is self-contained even if the source calibration file later changes.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.envs import EnvConfig
from lerobot.robots import Robot
from lerobot.utils.constants import OBS_IMAGES
from pynput import keyboard

from share.cameras.camera_realsense_depth import RealSenseDepthCamera
from share.cameras.configuration_realsense_depth import RealSenseDepthCameraConfig
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    EventConfig,
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    ObservationConfig,
)
from share.envs.manipulation_primitive.env_manipulation_primitive import ManipulationPrimitive
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.transitions import OnSuccess
from share.processor.info import AddKeyboardEventsAsInfoStep
from share.robots.ur import URConfig
from share.teleoperators import TeleopEvents, has_event
from share.teleoperators.spacemouse import SpaceMouseConfig
from share.utils.constants import DEFAULT_ROBOT_NAME

logger = logging.getLogger(__name__)

# Local to this env, not a share.teleoperators.TeleopEvents member -- nothing outside this
# module needs to recognize it, so it doesn't belong in the shared/canonical event enum.
CAPTURE_IMAGES_EVENT = "capture_images"


def _load_camera_to_gripper_transform(calibration_path: str | Path) -> dict[str, Any] | None:
    """Read the ``camera_to_gripper`` block straight out of a hand-eye calibration file.

    Kept independent of ``experiments.envs.foundationpose.primitives`` (which needs the
    optional ``pose_estimation`` package) so this recorder has no such dependency.
    """
    if not calibration_path:
        return None
    payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    return payload.get("camera_to_gripper")


class CaptureImagesPrimitive(ManipulationPrimitive):
    """Plain teleop primitive that also dumps camera+depth/robot-pose snapshots on a keypress."""

    def __init__(
        self,
        task_frame: dict[str, TaskFrame],
        robot_dict: dict[str, Robot],
        cameras: dict[str, Any],
        display_cameras: bool = False,
        output_dir: str = "",
        calibration_file: str = "",
        camera_key: str = "main",
        robot_name: str = DEFAULT_ROBOT_NAME,
        capture_key: keyboard.Key | str = keyboard.Key.enter,
    ):
        super().__init__(task_frame, robot_dict, cameras, display_cameras)

        if not output_dir:
            raise ValueError("CaptureImagesPrimitive requires a non-empty output_dir.")
        self.output_dir = Path(output_dir)
        self.snapshots_dir = self.output_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.output_dir / "snapshots.jsonl"

        self.camera_key = camera_key
        self.robot_name = robot_name
        self.calibration_file = str(calibration_file) if calibration_file else None
        self.camera_to_gripper = _load_camera_to_gripper_transform(calibration_file)
        self._copy_calibration_file()

        # Independent of the per-primitive AddKeyboardEventsAsInfoStep the action processor
        # installs for the shared EventConfig.key_mapping -- this one is scoped to the capture
        # key only, but shares the same process-wide pynput listener (see share/processor/info.py),
        # so it doesn't spawn a second competing listener. The event name is local to this
        # module (CAPTURE_IMAGES_EVENT), not a share.teleoperators.TeleopEvents member.
        self._capture_key_step = AddKeyboardEventsAsInfoStep(
            mapping={CAPTURE_IMAGES_EVENT: capture_key},
            pulse_events=(CAPTURE_IMAGES_EVENT,),
        )
        self._next_index = self._count_existing_snapshots()

    def _count_existing_snapshots(self) -> int:
        if not self.index_path.exists():
            return 0
        with self.index_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def _copy_calibration_file(self) -> None:
        if not self.calibration_file:
            return
        src = Path(self.calibration_file)
        if not src.exists():
            logger.warning("Calibration file '%s' not found; snapshots will have no extrinsics.", src)
            return
        dst_dir = self.output_dir / "calibration"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)

    def step(self, action: dict[str, dict[str, float]]):
        obs, reward, terminated, truncated, info = super().step(action)

        key_info = self._capture_key_step.info({})
        if has_event(key_info, CAPTURE_IMAGES_EVENT):
            # StdinKeyboardListener (the permission-free fallback used whenever this process
            # has no /dev/input access -- see share/processor/utils.py) never fires a release
            # event. AddKeyboardEventsAsInfoStep's pulse only re-arms on release, so without
            # one the capture key would fire once and then stay silent for the rest of the
            # run. Force a re-arm here instead of waiting for a release that may never come.
            self._capture_key_step._pressed[CAPTURE_IMAGES_EVENT] = False
            saved_path = self._save_snapshot(obs)
            info["record_status"] = f"saved snapshot -> {saved_path.name}"
            logger.info("Saved snapshot to %s", saved_path)

        return obs, reward, terminated, truncated, info

    def _save_snapshot(self, obs: dict[str, Any]) -> Path:
        cam = self.cameras.get(self.camera_key)
        if not isinstance(cam, RealSenseDepthCamera):
            raise TypeError(
                f"Camera '{self.camera_key}' must be a RealSenseDepthCamera to snapshot depth, "
                f"got {type(cam).__name__}."
            )

        index = self._next_index
        self._next_index += 1
        stem = f"{index:06d}"

        rgb = obs[f"{OBS_IMAGES}.{self.camera_key}"]
        rgb = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
        depth = cam.read_depth(timeout_ms=200, in_meters=True)

        rgb_path = self.snapshots_dir / f"{stem}_rgb.png"
        depth_path = self.snapshots_dir / f"{stem}_depth.npy"
        meta_path = self.snapshots_dir / f"{stem}_meta.json"

        from PIL import Image
        Image.fromarray(rgb).save(rgb_path)
        np.save(depth_path, depth.astype(np.float32))

        robot_pose = {
            key.removeprefix(f"{self.robot_name}."): _to_jsonable(value)
            for key, value in obs.items()
            if key.startswith(f"{self.robot_name}.")
        }

        meta = {
            "index": index,
            "timestamp": time.time(),
            "rgb_path": str(rgb_path.relative_to(self.output_dir)),
            "depth_path": str(depth_path.relative_to(self.output_dir)),
            "robot_name": self.robot_name,
            "robot_pose": robot_pose,
            "camera_key": self.camera_key,
            "camera_intrinsics": cam.get_camera_intrinsics().tolist(),
            "depth_scale": cam.get_depth_scale(),
            "camera_to_gripper": self.camera_to_gripper,
            "calibration_file": self.calibration_file,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")

        return meta_path


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _teleop_processor(fps: float) -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=fps,
        observation=ObservationConfig(
            add_ee_pos_to_observation=True,
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=True,
            add_joint_position_to_observation=True,
        ),
        gripper=GripperConfig(enable=False),
        events=EventConfig(
            key_mapping={
                TeleopEvents.STOP_RECORDING: keyboard.Key.down,
            },
        ),
    )


@EnvConfig.register_subclass("capture_images")
@dataclass
class CaptureImagesEnvConfig(ManipulationPrimitiveNetConfig):
    """Free-teleop the UR arm and snapshot camera+depth/robot-pose pairs on a keypress.

    Single-primitive graph, no dataset/policy involved -- run it with ``record.py`` and
    ``--dataset.root`` left unset. See the module docstring for the on-disk layout.
    """

    fps: int = 30
    robot_ip: str = "172.22.22.2"  # the right UR5e -- matches calibration_file's robot_ip below
    camera_serial_number: str = "352122273250"
    calibration_file: str = "calibration/hand_eye_calibration_result_ur5e.json"
    output_dir: str = "data/capture_images"
    capture_key: str = "enter"
    translation_action_scale: float = 0.25
    rotation_action_scale: float = 0.8
    # Soft/compliant gains, same profile as teleop_spacemouse_6dof's "soft" preset. Only
    # used when use_servo=False (force mode); ignored in the servoing setup.
    kp: list[float] = field(default_factory=lambda: [1500.0, 1500.0, 1500.0, 100.0, 100.0, 100.0])
    kd: list[float] = field(default_factory=lambda: [40.0, 40.0, 40.0, 3.0, 3.0, 3.0])
    wrench_limits: list[float] = field(default_factory=lambda: [15.0, 15.0, 15.0, 2.0, 2.0, 2.0])
    # True (default): direct task-space pose streaming via servoL, no force mode -- simpler
    # and stiffer, good for lining the arm up for a snapshot. False: compliant force-mode
    # teleop (see teleop_spacemouse_6dof), driven by kp/kd/wrench_limits above.
    use_servo: bool = True
    start_primitive: str = "teleop_and_capture"
    reset_primitive: str = "teleop_and_capture"

    def __post_init__(self) -> None:
        self.robot = URConfig(
            # RobotConfig.id defaults to None, and lerobot's connect-log line interpolates
            # it unguarded ("f'{self.id} {cls}'") -- leaving it unset prints "None UR
            # connected." Give it a real id so the log reads sensibly.
            id="capture_images_ur5e",
            robot_ip=self.robot_ip,
            frequency=125,
            soft_real_time=True,
            rt_core=3,
            use_gripper=True,
            simple_pose_use_servo=self.use_servo,
            # Anti-windup for force mode only; irrelevant (and left at its URConfig
            # default) in the servoing setup.
            compliance_reference_limit_enable=[not self.use_servo] * 6,
        )
        self.teleop = SpaceMouseConfig(
            id="capture_images_spacemouse",
            action_scale=[
                self.translation_action_scale,
                self.translation_action_scale,
                self.translation_action_scale,
                self.rotation_action_scale,
                self.rotation_action_scale,
                self.rotation_action_scale,
            ]
        )
        self.cameras = {
            "main": RealSenseDepthCameraConfig(
                serial_number_or_name=self.camera_serial_number,
                use_depth=True,
            ),
        }

        capture_key = getattr(keyboard.Key, self.capture_key, None) or self.capture_key

        self.primitives = {
            "teleop_and_capture": ManipulationPrimitiveConfig(
                notes="Free 6-DoF teleop; press the capture key to snapshot camera+depth+pose.",
                processor=_teleop_processor(float(self.fps)),
                task_frame=TaskFrame(
                    target=[0.0] * 6,
                    space=ControlSpace.TASK,
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=[PolicyMode.RELATIVE] * 6,
                    origin=[0.0] * 6,
                    controller_overrides=(
                        # RTDETaskFrameController.to_queue_dict() defaults
                        # simple_pose_use_servo to False for any command that omits it --
                        # URConfig.simple_pose_use_servo only applies before the first
                        # command arrives -- so this has to be set here explicitly, not
                        # just on URConfig, or every command after the first silently
                        # falls back to moveL.
                        {"use_force_mode": False, "simple_pose_use_servo": True}
                        if self.use_servo
                        else {
                            "use_force_mode": True,
                            "kp": list(self.kp),
                            "kd": list(self.kd),
                            "wrench_limits": list(self.wrench_limits),
                        }
                    ),
                ),
                env_class=CaptureImagesPrimitive,
                env_kwargs={
                    "output_dir": self.output_dir,
                    "calibration_file": self.calibration_file,
                    "camera_key": "main",
                    "capture_key": capture_key,
                },
            ),
        }
        # MP-Net rejects a primitive with no outgoing edges. A single free-teleop primitive
        # has nowhere to go, so give it a harmless self-loop gated on SUCCESS -- a key nothing
        # here maps to, so it never actually fires.
        self.transitions = [
            OnSuccess(source="teleop_and_capture", target="teleop_and_capture"),
        ]

        super().__post_init__()


__all__ = [
    "CaptureImagesEnvConfig",
    "CaptureImagesPrimitive",
]
