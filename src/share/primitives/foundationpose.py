import json
import queue
import threading
from pathlib import Path
from dataclasses import dataclass, field
from PIL import Image

import numpy as np
from lerobot.teleoperators import Teleoperator
from scipy.spatial.transform import Rotation as SciRotation
from lerobot.cameras import Camera
from lerobot.robots import Robot
from lerobot.utils.rotation import Rotation

from share.cameras import RealSenseDepthCamera
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    ManipulationPrimitiveConfig,
    PrimitiveEntryContext,
)
from share.envs.manipulation_primitive.env_manipulation_primitive import ManipulationPrimitive
from share.envs.manipulation_primitive.task_frame import TaskFrame
from share.teleoperators import TeleopEvents
import logging

from pose_estimation import GraspObjectSpec, PoseEstimator, PoseEstimatorError, create_pose_estimator
from share.utils.constants import DEFAULT_ROBOT_NAME
from share.utils.transformation_utils import get_robot_pose_from_observation, task_pose_to_world_pose

logger = logging.getLogger(__name__)

EE_POSE_KEYS = ("x.ee_pos", "y.ee_pos", "z.ee_pos", "rx.ee_pos", "ry.ee_pos", "rz.ee_pos")


def _transform_from_translation_rotation(
    translation_m: list[float] | np.ndarray,
    rotation_matrix: list[list[float]] | np.ndarray,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation_m, dtype=np.float64).reshape(3)
    return transform


def load_camera_to_gripper_transform(calibration_path: str | Path) -> np.ndarray:
    payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    camera_to_gripper = payload["camera_to_gripper"]
    return _transform_from_translation_rotation(
        translation_m=camera_to_gripper["translation_m"],
        rotation_matrix=camera_to_gripper["rotation_matrix"],
    )


def object_pose_camera_to_robot_base(
    object_pose_camera: np.ndarray,
    tcp_pose_base_to_gripper: list[float] | np.ndarray,
    camera_to_gripper_transform: np.ndarray,
) -> np.ndarray:
    base_to_gripper = np.eye(4, dtype=np.float64)
    base_to_gripper[:3, :3] = Rotation.from_rotvec(np.asarray(tcp_pose_base_to_gripper[3:], dtype=np.float64)).as_matrix()
    base_to_gripper[:3, 3] = np.asarray(tcp_pose_base_to_gripper[:3], dtype=np.float64)
    camera_to_object = np.asarray(object_pose_camera, dtype=np.float64).reshape(4, 4)
    return base_to_gripper @ camera_to_gripper_transform @ camera_to_object


def object_pose_camera_to_tcp_frame(
    object_pose_camera: np.ndarray,
    camera_to_gripper_transform: np.ndarray,
) -> np.ndarray:
    camera_to_object = np.asarray(object_pose_camera, dtype=np.float64).reshape(4, 4)
    return camera_to_gripper_transform @ camera_to_object


def transform_to_pose_xyzrpy(transform: np.ndarray) -> list[float]:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = SciRotation.from_matrix(transform[:3, :3])
    translation = transform[:3, 3].tolist()
    return [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        *[float(value) for value in rotation.as_euler("xyz", degrees=False).tolist()],
    ]


def pose_xyzrpy_to_transform(pose: list[float] | np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = SciRotation.from_euler("xyz", pose[3:], degrees=False).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def pose_within_tolerance(
    current_pose: list[float] | np.ndarray,
    expected_pose: list[float] | np.ndarray,
    translation_tolerance: float | list[float],
    rotation_tolerance: float,
) -> bool:
    """Check whether ``current_pose`` matches ``expected_pose`` within tolerance.

    Translation is checked per-axis (xyz). Rotation is checked as the geodesic
    angular distance between the two orientations, avoiding the singularities and
    non-uniqueness issues of per-axis Euler angle comparison.
    """
    current_pose = np.asarray(current_pose, dtype=np.float64).reshape(6)
    expected_pose = np.asarray(expected_pose, dtype=np.float64).reshape(6)

    if isinstance(translation_tolerance, (int, float)):
        trans_tols = [float(translation_tolerance)] * 3
    else:
        trans_tols = [float(v) for v in translation_tolerance]
        if len(trans_tols) != 3:
            raise ValueError("translation_tolerance must be a scalar or length-3 list.")

    for axis in range(3):
        if abs(current_pose[axis] - expected_pose[axis]) > trans_tols[axis]:
            return False

    R_current = SciRotation.from_euler("xyz", current_pose[3:])
    R_expected = SciRotation.from_euler("xyz", expected_pose[3:])
    angle = (R_expected.inv() * R_current).magnitude()
    return angle <= rotation_tolerance


def tcp_pose_rotvec_to_transform(pose: list[float] | np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = SciRotation.from_rotvec(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


class FoundationPosePrimitive(ManipulationPrimitive):
    def __init__(
            self,
            task_frame: dict[str, TaskFrame],
            robot_dict: dict[str, Robot],
            cameras: dict[str, Camera],
            grasp_object: GraspObjectSpec | str | Path,
            calibration_file: str | Path,
            display_cameras: bool = False,
            pose_key: str = "pose",
            expected_pose: list[float] | None = None,
            translation_tolerance: float | list[float] = 0.02,
            rotation_tolerance: float = 0.2,
            estimation_timeout_s: float = 30.0,
    ):
        super().__init__(task_frame, robot_dict, cameras, display_cameras)
        self.expected_pose = expected_pose
        self.translation_tolerance = translation_tolerance
        self.rotation_tolerance = rotation_tolerance
        if estimation_timeout_s <= 0:
            raise ValueError("estimation_timeout_s must be greater than zero.")
        self.estimation_timeout_s = estimation_timeout_s

        if isinstance(grasp_object, GraspObjectSpec):
            self.object_spec = grasp_object
        else:
            self.object_spec = GraspObjectSpec.from_json_file(grasp_object)

        # self.pose_estimator = PoseEstimator()
        self.pose_estimator = create_pose_estimator(
            segmentation_provider="yolo",
            pose_provider="foundationpose",
            mesh_path=self.object_spec.mesh_path,
            prompt=self.object_spec.yolo_class_name,
            confidence_threshold=self.object_spec.confidence_threshold,
            image_format="rgb",
        )
        self._pose_estimator_initialized = False
        self.camera_to_gripper_transform = load_camera_to_gripper_transform(calibration_file)

        self._pose_estimator_config = {
            "mesh_path": self.object_spec.mesh_path,
            "prompt": self.object_spec.yolo_class_name,
            "confidence_threshold": self.object_spec.confidence_threshold,
        }
        self.debug_output_dir = Path("tmp")
        self.pose_key = pose_key

    def step(self, action: dict[str, dict[str, float]]):
        obs, reward, terminated, truncated, info = super().step(action)
        cam = self.cameras.get("wrist")
        if not isinstance(cam, RealSenseDepthCamera):
            raise TypeError(f"Expected RealSenseDepthCamera, got {type(cam).__name__}")

        if not self._pose_estimator_initialized:
            self.pose_estimator.configure(
                **self._pose_estimator_config,
                camera_intrinsics=cam.get_camera_intrinsics().tolist(),
            )
            self.pose_estimator.restart_tracking()
            self._pose_estimator_initialized = True
        logger.debug("configuration:", self._pose_estimator_config)
        image = obs.get('observation.images.wrist')
        depth = cam.read_depth(timeout_ms=False, in_meters=True)

        try:
            estimation = self._estimate_pose_with_timeout(image, depth)
        except (PoseEstimatorError, TimeoutError) as e:
            logger.warning("pose estimation failed, routing to OnFailure: %s", e)
            info[TeleopEvents.SUCCESS] = False
            info[TeleopEvents.FAILURE] = True
            self._pose_estimator_initialized = False
            return obs, reward, terminated, truncated, info

        pose = estimation.get('pose')
        tcp_pose = [obs[f"{DEFAULT_ROBOT_NAME}.{key}"] for key in EE_POSE_KEYS]
        pose_base = object_pose_camera_to_robot_base(
            object_pose_camera=pose,
            tcp_pose_base_to_gripper=tcp_pose,
            camera_to_gripper_transform=self.camera_to_gripper_transform,
        )
        object_pose_world = transform_to_pose_xyzrpy(pose_base)

        debug_prints = False
        if debug_prints:
            pose_tcp = object_pose_camera_to_tcp_frame(
                object_pose_camera=pose,
                camera_to_gripper_transform=self.camera_to_gripper_transform,
            )

            tcp_world = np.eye(4, dtype=np.float64)
            tcp_world[:3, :3] = Rotation.from_rotvec(np.asarray(tcp_pose[3:], dtype=np.float64)).as_matrix()
            tcp_world[:3, 3] = np.asarray(tcp_pose[:3], dtype=np.float64)
            camera_world = tcp_world @ self.camera_to_gripper_transform

            logger.info("object pose in world frame:\n%s", pose_base)
            logger.info("object pose in tcp frame:\n%s", pose_tcp)
            logger.info("object pose in camera frame:\n%s", pose)
            logger.info("tcp pose in world frame:\n%s", tcp_world)
            logger.info("camera pose in world frame:\n%s", camera_world)

            camera_distance_m = float(np.linalg.norm(np.asarray(pose, dtype=np.float64)[:3, 3]))
            tcp_distance_m = float(np.linalg.norm(pose_tcp[:3, 3]))
            base_distance_m = float(np.linalg.norm(pose_base[:3, 3]))
            logger.info(f"object translation distance in camera frame [m]: {camera_distance_m:.4f}")
            logger.info(f"object translation distance in tcp frame [m]: {tcp_distance_m:.4f}")
            logger.info(f"object translation distance in base frame [m]: {base_distance_m:.4f}")

            gripper_pose_object = np.linalg.inv(pose_base) @ tcp_world
            logger.info(f"OBJECT POSE IN WORLD FRAME: {transform_to_pose_xyzrpy(pose_base)}")
            logger.info(f"OBJECT POSE IN TCP FRAME: {transform_to_pose_xyzrpy(pose_tcp)}")
            logger.info(f"GRIPPER POSE IN OBJECT FRAME: {transform_to_pose_xyzrpy(gripper_pose_object)}")

        self.set_runtime_value(self.pose_key, object_pose_world)

        # No expected pose configured: treat the estimate as trusted so OnSuccess
        # behaves like the unconditional Always transition used before this check
        # existed. Set expected_pose to actually gate on tolerance.
        pose_ok = True
        if self.expected_pose is not None:
            pose_ok = pose_within_tolerance(
                object_pose_world, self.expected_pose,
                self.translation_tolerance, self.rotation_tolerance,
            )
            logger.info(
                "pose estimation tolerance check %s (estimated=%s, expected=%s, trans_tol=%s, rot_tol=%.3f rad)",
                "passed" if pose_ok else "failed",
                object_pose_world,
                self.expected_pose,
                self.translation_tolerance,
                self.rotation_tolerance,
            )
        info[TeleopEvents.SUCCESS] = pose_ok
        info[TeleopEvents.FAILURE] = not pose_ok

        self._pose_estimator_initialized = False
        return obs, reward, terminated, truncated, info


    def estimate_pose(self, image, depth):
        self.debug_output_dir.mkdir(parents=True, exist_ok=True)

        estimation = self.pose_estimator.estimate_pose(image=image, depth=depth)

        if 'mask' in estimation.keys() and estimation.get('mask') is not None:
            Image.fromarray(image).save(self.debug_output_dir / "obs_wrist.png")
            Image.fromarray(estimation.get('mask')).save(self.debug_output_dir / "mask.png")

        return estimation

    def _estimate_pose_with_timeout(self, image, depth):
        """Run pose estimation without letting a missing worker block forever."""
        result_queue = queue.Queue(maxsize=1)

        def estimate():
            try:
                result_queue.put((True, self.estimate_pose(image, depth)))
            except Exception as exc:
                result_queue.put((False, exc))

        threading.Thread(target=estimate, name="foundationpose-estimation", daemon=True).start()

        try:
            succeeded, result = result_queue.get(timeout=self.estimation_timeout_s)
        except queue.Empty as exc:
            raise TimeoutError(
                f"Pose estimation did not finish within {self.estimation_timeout_s:g} seconds."
            ) from exc

        if not succeeded:
            raise result
        return result

@ManipulationPrimitiveConfig.register_subclass("foundation_pose")
@dataclass
class FoundationPosePrimitiveConfig(ManipulationPrimitiveConfig):
    grasp_obj: GraspObjectSpec|str|None = None
    calibration_file: str = ""
    # Optimal/expected object pose (world frame, xyzrpy). When set, the primitive
    # checks the estimated pose against it and reports TeleopEvents.SUCCESS /
    # TeleopEvents.FAILURE in info so OnSuccess/OnFailure transitions can route
    # to different next primitives depending on whether the estimate is trusted.
    expected_pose: list[float] | None = None
    translation_tolerance: float | list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    rotation_tolerance: float = 0.5
    # Maximum time to wait for the pose estimator background worker.
    estimation_timeout_s: float = 30.0

    def validate(self, robot_dict, teleop_dict):
        super().validate(robot_dict, teleop_dict)

    def make(
            self,
            robot_dict: dict[str, Robot],
            teleop_dict: dict[str, Teleoperator],
            cameras: dict[str, Camera],
            device: str = "cpu"
    ):
        self.validate(robot_dict, teleop_dict)
        self.infer_features(robot_dict, cameras)  # todo: fix initial_features

        display_cameras = self.processor.image_preprocessing is not None and self.processor.image_preprocessing.display_cameras
        env = FoundationPosePrimitive(task_frame=self.task_frame,
                                      robot_dict=robot_dict,
                                      cameras=cameras,
                                      display_cameras=display_cameras,
                                      grasp_object=self.grasp_obj,
                                      calibration_file=self.calibration_file,
                                      pose_key="object_pose",
                                      expected_pose=self.expected_pose,
                                      translation_tolerance=self.translation_tolerance,
                                      rotation_tolerance=self.rotation_tolerance,
                                      estimation_timeout_s=self.estimation_timeout_s)

        env_processor = self.make_env_processor(device)
        action_processor = self.make_action_processor(robot_dict, teleop_dict, device)
        return env, env_processor, action_processor

@ManipulationPrimitiveConfig.register_subclass("cached_object_pose_in_tcp_frame")
@dataclass
class CachedObjectPoseInTcpFramePrimitiveConfig(ManipulationPrimitiveConfig):
    """Compute current TCP-frame object pose from a saved world-frame estimate."""

    object_pose_runtime_key: str = "object_pose"
    output_runtime_key: str = "object_pose_in_tcp_frame"
    gripper_pose_output_runtime_key: str = "gripper_pose_in_object_frame"
    robot_name: str = DEFAULT_ROBOT_NAME

    def on_entry(self, env: ManipulationPrimitive, entry_context: PrimitiveEntryContext | None) -> None:
        super().on_entry(env, entry_context)

        object_pose_world = env.get_runtime_value(self.object_pose_runtime_key)
        if object_pose_world is None:
            raise RuntimeError(
                f"Missing runtime object pose '{self.object_pose_runtime_key}'. "
                "Run the FoundationPose primitive before entering this primitive."
            )

        current_tcp_pose_world = self._current_tcp_pose_world(env, entry_context)
        object_pose = self._pose_for_robot(object_pose_world)
        T_world_tcp = pose_xyzrpy_to_transform(current_tcp_pose_world)
        T_world_object = pose_xyzrpy_to_transform(object_pose)
        # T_tcp_object: object pose expressed in TCP frame
        T_tcp_object = np.linalg.inv(T_world_tcp) @ T_world_object
        # T_object_tcp: gripper/TCP pose expressed in object frame — this is the grasp pose
        T_object_tcp = np.linalg.inv(T_world_object) @ T_world_tcp
        object_pose_tcp = transform_to_pose_xyzrpy(T_tcp_object)
        gripper_pose_object = transform_to_pose_xyzrpy(T_object_tcp)

        env.set_runtime_value(self.output_runtime_key, object_pose_tcp)
        env.set_runtime_value(self.gripper_pose_output_runtime_key, gripper_pose_object)
        env._primitive_complete = True
        logger.info("CachedObjectPoseInTcpFramePrimitiveConfig on entry.")
        logger.info("OBJECT POSE IN TCP FRAME: %s", object_pose_tcp)
        logger.info("GRIPPER POSE IN OBJECT FRAME: %s", gripper_pose_object)
        print()

    def _current_tcp_pose_world(
        self,
        env: ManipulationPrimitive,
        entry_context: PrimitiveEntryContext | None,
    ) -> list[float]:
        if entry_context is not None:
            try:
                return get_robot_pose_from_observation(entry_context.observation, self.robot_name)
            except KeyError:
                logger.debug("Entry observation has no TCP pose for '%s'; reading env observation.", self.robot_name)

        obs = env._get_observation()
        raw_tcp_pose = [obs[f"{self.robot_name}.{key}"] for key in EE_POSE_KEYS]
        return transform_to_pose_xyzrpy(tcp_pose_rotvec_to_transform(raw_tcp_pose))

    def _pose_for_robot(self, runtime_pose) -> list[float]:
        if isinstance(runtime_pose, dict):
            if self.robot_name in runtime_pose:
                return [float(value) for value in runtime_pose[self.robot_name]]
            if {"x", "y", "z", "rx", "ry", "rz"}.issubset(runtime_pose):
                return [float(runtime_pose[axis]) for axis in ("x", "y", "z", "rx", "ry", "rz")]

        if isinstance(runtime_pose, (list, tuple, np.ndarray)) and len(runtime_pose) == 6:
            return [float(value) for value in runtime_pose]

        raise ValueError(
            f"Runtime object pose '{self.object_pose_runtime_key}' must be a 6D pose or per-robot pose dict, "
            f"got {type(runtime_pose).__name__}."
        )


@ManipulationPrimitiveConfig.register_subclass("runtime_frame_target")
@dataclass
class RuntimeFrameTargetPrimitiveConfig(ManipulationPrimitiveConfig):
    """Primitive that resolves its task-frame origin from a saved runtime pose."""

    frame_origin_runtime_key: str = "object_pose"

    def on_entry(self, env: ManipulationPrimitive, entry_context) -> None:
        runtime_origin = env.get_runtime_value(self.frame_origin_runtime_key)
        if runtime_origin is None:
            raise RuntimeError(
                f"Missing runtime frame origin '{self.frame_origin_runtime_key}'. "
                "Run the pose-estimation primitive before entering this primitive."
            )

        origin_by_robot = self._origin_by_robot(runtime_origin)
        for name, frame in self.task_frame.items():
            frame.origin = [float(value) for value in origin_by_robot[name]]

        super().on_entry(env, entry_context)

    def _origin_by_robot(self, runtime_origin) -> dict[str, list[float]]:
        if isinstance(runtime_origin, dict):
            if set(self.task_frame).issubset(runtime_origin):
                return {
                    name: [float(value) for value in runtime_origin[name]]
                    for name in self.task_frame
                }
            if {"x", "y", "z", "rx", "ry", "rz"}.issubset(runtime_origin):
                pose = [float(runtime_origin[axis]) for axis in ("x", "y", "z", "rx", "ry", "rz")]
                return {name: list(pose) for name in self.task_frame}

        if isinstance(runtime_origin, (list, tuple, np.ndarray)) and len(runtime_origin) == 6:
            pose = [float(value) for value in runtime_origin]
            return {name: list(pose) for name in self.task_frame}

        raise ValueError(
            f"Runtime frame origin '{self.frame_origin_runtime_key}' must be a 6D pose or per-robot pose dict, "
            f"got {type(runtime_origin).__name__}."
        )

@ManipulationPrimitiveConfig.register_subclass("relruntime_frame_target")
@dataclass
class RelativeRuntimeFrameTargetPrimitiveConfig(ManipulationPrimitiveConfig):
    """Primitive that resolves its task-frame origin from a saved runtime pose."""

    frame_origin_runtime_key: str = "object_pose"

    def on_entry(self, env: ManipulationPrimitive, entry_context) -> None:
        runtime_origin = env.get_runtime_value(self.frame_origin_runtime_key).copy()
        if runtime_origin is None:
            raise RuntimeError(
                f"Missing runtime frame origin '{self.frame_origin_runtime_key}'. "
                "Run the pose-estimation primitive before entering this primitive."
            )
        runtime_origin[2] += 0.02
        origin_by_robot = self._origin_by_robot(runtime_origin)
        for name, frame in self.task_frame.items():
            frame.origin = [float(value) for value in origin_by_robot[name]]

        super().on_entry(env, entry_context)

    def _origin_by_robot(self, runtime_origin) -> dict[str, list[float]]:
        if isinstance(runtime_origin, dict):
            if set(self.task_frame).issubset(runtime_origin):
                return {
                    name: [float(value) for value in runtime_origin[name]]
                    for name in self.task_frame
                }
            if {"x", "y", "z", "rx", "ry", "rz"}.issubset(runtime_origin):
                pose = [float(runtime_origin[axis]) for axis in ("x", "y", "z", "rx", "ry", "rz")]
                return {name: list(pose) for name in self.task_frame}

        if isinstance(runtime_origin, (list, tuple, np.ndarray)) and len(runtime_origin) == 6:
            pose = [float(value) for value in runtime_origin]
            return {name: list(pose) for name in self.task_frame}

        raise ValueError(
            f"Runtime frame origin '{self.frame_origin_runtime_key}' must be a 6D pose or per-robot pose dict, "
            f"got {type(runtime_origin).__name__}."
        )