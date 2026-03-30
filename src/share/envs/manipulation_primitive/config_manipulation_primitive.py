import copy
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from draccus import ChoiceRegistry
from pynput import keyboard

from lerobot.cameras import Camera
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import PREFIXES_TO_STRIP, strip_prefix, create_initial_features
from lerobot.envs import EnvConfig
from lerobot.teleoperators import Teleoperator
from lerobot.robots import Robot
from lerobot.processor import DataProcessorPipeline, DeviceProcessorStep, ImageCropResizeProcessorStep
from lerobot.processor.converters import identity_transition
from lerobot.processor.hil_processor import GRIPPER_KEY
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from share.envs.manipulation_primitive.env_manipulation_primitive import (
    ManipulationPrimitive,
    OpenLoopTrajectoryPrimitive,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, TaskFrame, TASK_FRAME_AXIS_NAMES
from share.envs.utils import (
    check_task_frame_robot,
    check_delta_teleoperator,
    is_union_with_dict,
    resolve_entry_start_pose,
    any_enabled,
    copy_per_robot
)
from share.utils.kinematics import get_kinematics
from share.processor.action import (
    ToNestedActionProcessorStep,
    MatchTeleopToPolicyActionProcessorStep,
    InterventionActionProcessorStep,
    DiscretizeGripperProcessorStep,
    RelativeFrameActionProcessor,
    ToJointActionProcessorStep
)
from share.processor.info import (
    AddKeyboardEventsAsInfoStep,
    AddFootswitchEventsAsInfoStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep
)
from share.processor.observation import (
    JointsToEEObservation,
    RelativeFrameObservationProcessor,
    DefaultObservationProcessor
)
from share.teleoperators import TeleopEvents
from share.utils.transformation_utils import task_pose_to_world_pose, compose_delta_pose, world_pose_to_task_pose

PRIMITIVE_TARGET_POSE_INFO_KEY = "primitive_target_pose"
PRIMITIVE_COMPLETE_INFO_KEY = "primitive_complete"
TRAJECTORY_PROGRESS_INFO_KEY = "trajectory_progress"


@dataclass(slots=True)
class PrimitiveEntryContext:
    """Processed observation and task-frame origin forwarded on primitive entry."""

    source_primitive: str | None = None
    target_primitive: str | None = None
    observation: dict[str, Any] = field(default_factory=dict)
    task_frame_origin: dict[str, list[float] | None] = field(default_factory=dict)


@dataclass
class ImagePreprocessingConfig:
    """Camera crop/resize options applied inside the env processor."""

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None  # cam_name -> (top, left, height, width)
    resize_size: tuple[int, int] | None = None
    filter_keys: list[str] | None = None
    display_cameras: bool = False


@dataclass
class KinematicsConfig:
    """Configuration for inverse kinematics processing."""

    enable: bool | dict[str, bool] = False
    use_virtual_reference: bool | dict[str, bool] = True
    urdf_path: str | dict[str, str | None] | None = None
    target_frame_name: str | dict[str, str | None] | None = None
    end_effector_bounds: dict[str, list[float]] | dict[str, dict[str, list[float]]] | None = None
    end_effector_step_sizes: dict[str, float] | dict[str, dict[str, float]] | None = None


@dataclass
class ObservationConfig:
    """Configuration for observation processing."""

    add_joint_position_to_observation: bool | dict[str, bool] = True
    add_joint_velocity_to_observation: bool | dict[str, bool] = False
    add_current_to_observation: bool | dict[str, bool] = False

    add_ee_pos_to_observation: bool | dict[str, bool] = False
    add_ee_velocity_to_observation: bool | dict[str, bool] = False
    add_ee_wrench_to_observation: bool | dict[str, bool] = False

    ee_pos_axes: list[str] | dict[str, list[str]] | None = field(default_factory=lambda: [f"{ax}.ee_pos" for ax in TASK_FRAME_AXIS_NAMES])
    ee_velocity_axes: list[str] | dict[str, list[str]] | None = field(default_factory=lambda: [f"{ax}.ee_vel" for ax in TASK_FRAME_AXIS_NAMES])
    ee_wrench_axes: list[str] | dict[str, list[str]] | None = field(default_factory=lambda: [f"{ax}.ee_wrench" for ax in TASK_FRAME_AXIS_NAMES])

    stack_frames: int | dict[str, int] = 0
    relative_ee_pos: bool | dict[str, bool] = True


@dataclass
class GripperConfig:
    """Configuration for gripper control, discretization, and penalties."""

    enable: bool | dict[str, bool] = False
    discretize: bool | dict[str, bool] = False
    threshold: float | dict[str, float] = 0.5
    mode: Literal["state", "pulse"] | dict[str, Literal["state", "pulse"]] = "state"
    max_pos: float | dict[str, float] = 1.0
    min_pos: float | dict[str, float] = 0.0
    static_pos: float | dict[str, float] = 0.0
    penalty: float | dict[str, float | None] | None = None


@dataclass
class EventConfig:
    """Mappings from teleop inputs to structured intervention events."""

    key_mapping: dict[TeleopEvents, dict | keyboard.Key] = field(default_factory=lambda: {})
    foot_switch_mapping: dict[tuple[TeleopEvents], dict] = field(default_factory=lambda: {})


@dataclass
class HookConfig:
    """Optional processor timing hook configuration."""

    time_env_processor: bool = False
    time_action_processor: bool = False
    log_every: int = 10


@dataclass
class ManipulationPrimitiveProcessorConfig:
    """Top-level processor settings shared across robots and per robot."""

    # for all arms
    control_time_s: float = 10.0
    fps: float = 10.0
    image_preprocessing: ImagePreprocessingConfig | None = None
    events: EventConfig = field(default_factory=EventConfig)
    hooks: HookConfig = field(default_factory=HookConfig)

    # per arm
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)


@dataclass
class ManipulationPrimitiveConfig(EnvConfig, ChoiceRegistry):
    """Shared config and entry hooks for one primitive in an MP-Net."""
    task_frame: TaskFrame | dict[str, TaskFrame] = field(default_factory=TaskFrame)
    processor: ManipulationPrimitiveProcessorConfig = field(default_factory=ManipulationPrimitiveProcessorConfig)
    policy: PreTrainedConfig | None = None
    policy_overwrites: dict = field(default_factory=dict)
    notes: str | None = None
    is_terminal: bool = False
    task_description: str | None = None
    target_pose_info_key: str | None = PRIMITIVE_TARGET_POSE_INFO_KEY

    def __post_init__(self):
        self._kinematics_solver = {}
        self._joint_names = {}

    @property
    def gym_kwargs(self) -> dict:
        """Extra kwargs forwarded to gym environment creation."""
        return {}

    @property
    def is_adaptive(self) -> bool:
        task_frames = self.task_frame.values() if isinstance(self.task_frame, dict) else [self.task_frame]
        return any(tf.policy_action_dim > 0 for tf in task_frames)

    @property
    def num_cameras(self) -> int:
        if self.features is None:
            return 0
        else:
            return len([ft for ft in self.features.values() if ft.type == FeatureType.VISUAL])

    def make(
        self,
        robot_dict: dict[str, Robot],
        teleop_dict: dict[str, Teleoperator],
        cameras: dict[str, Camera],
        device: str = "cpu"
    ):
        """Build one primitive env together with its processors.

        Args:
            robot_dict: Connected robot handles keyed by primitive robot name.
            teleop_dict: Connected teleoperators keyed by robot name.
            cameras: Connected camera handles available to the primitive.
            device: Torch device used by observation-side processors.

        Returns:
            A tuple of ``(env, env_processor, action_processor)`` ready to be
            installed into the MP-Net runtime.
        """
        self.validate(robot_dict, teleop_dict)
        self.infer_features(robot_dict, cameras)  # todo: fix initial_features

        display_cameras = self.processor.image_preprocessing is not None and self.processor.image_preprocessing.display_cameras
        env = ManipulationPrimitive(task_frame=self.task_frame, robot_dict=robot_dict, cameras=cameras, display_cameras=display_cameras)

        env_processor = self.make_env_processor(device)
        action_processor = self.make_action_processor(robot_dict, teleop_dict, device)
        return env, env_processor, action_processor

    def make_action_processor(self, robot_dict, teleop_dict, device) -> DataProcessorPipeline:
        """Create the action-side processing pipeline.

        Args:
            robot_dict: Connected robot handles keyed by robot name.
            teleop_dict: Connected teleoperators keyed by robot name.
            device: Torch device hint forwarded to processor steps when needed.

        Returns:
            A ``DataProcessorPipeline`` that normalizes teleop/policy actions
            into the nested low-level action dict expected by the primitive env.
        """
        action_pipeline_steps = []

        # events
        if self.processor.events.key_mapping:
            action_pipeline_steps.append(AddKeyboardEventsAsInfoStep(mapping=self.processor.events.key_mapping))

        if self.processor.events.foot_switch_mapping:
            action_pipeline_steps.append(AddFootswitchEventsAsInfoStep(mapping=self.processor.events.foot_switch_mapping))

        try:
            action_pipeline_steps.append(AddTeleopEventsAsInfoStep(teleoperators=teleop_dict))
        except TypeError:
            pass

        action_pipeline_steps.extend([
            AddTeleopActionAsComplimentaryDataStep(teleoperators=teleop_dict),  # this checks events and should come after Add*EventsAsInfoStep's
            ToNestedActionProcessorStep(
                task_frame=self.task_frame,
                gripper_enable=self.processor.gripper.enable,
            ),

            # make teleop action match policy based on task frame (treat delta ee / vel / force the same):
            # teleop Q:
            # policy delta ee / vel / force: FK + differentiate
            # abs ee: FK
            # delta Q: differentiate
            # abs Q. noop
            #
            # teleop delta ee:
            # policy delta ee / vel / force: noop
            # abs ee: integrate
            # delta Q: IK
            # abs Q: integrate + IK
            MatchTeleopToPolicyActionProcessorStep(
                teleoperators=teleop_dict,
                task_frame=self.task_frame,
                kinematics=self._kinematics_solver,
                use_virtual_reference=self.processor.kinematics.use_virtual_reference,
                joint_names=self._joint_names,
                gripper_enable=self.processor.gripper.enable,
            ),

            # scatter policy / teleop action (depending on is-intervention event) into full task frame action target
            # send feedback to teleoperators if they need it
            InterventionActionProcessorStep(
                teleoperators=teleop_dict,
                task_frame=self.task_frame,
                gripper_enable=self.processor.gripper.enable,
            ),
            DiscretizeGripperProcessorStep(
                min_pos=self.processor.gripper.min_pos,
                max_pos=self.processor.gripper.max_pos,
                threshold=self.processor.gripper.threshold, 
                discretize=self.processor.gripper.discretize
            ),
        ])

        # action in ee frame instead of in world frame
        if any_enabled(self.processor.observation.relative_ee_pos):
            action_pipeline_steps.append(
                RelativeFrameActionProcessor(
                    enable=self.processor.observation.relative_ee_pos
                )
            )

        # todo: fix this
        is_task_frame_robot = check_task_frame_robot(robot_dict)
        if not all(is_task_frame_robot.values()) and False:
            # after this processor, the action must a dictionary of joint names
            # policy_action: delta vel ->

            action_pipeline_steps.append(
                ToJointActionProcessorStep(
                    is_task_frame_robot=is_task_frame_robot,
                    task_frame=self.task_frame,
                    kinematics=self._kinematics_solver,
                    joint_names=self._joint_names,
                    use_virtual_reference=self.processor.kinematics.use_virtual_reference
                )
            )

        # timing hooks
        if self.processor.hooks.time_action_processor:
            from share.utils.control_utils import make_step_timing_hooks
            action_before_hooks, action_after_hooks = make_step_timing_hooks(
                pipeline_steps=action_pipeline_steps,
                label="action",
                log_every=self.processor.hooks.log_every,
                ema_alpha=0.2,
                also_print=False
            )
        else:
            action_before_hooks, action_after_hooks = [], []

        return DataProcessorPipeline(
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition,
            before_step_hooks=action_before_hooks, after_step_hooks=action_after_hooks
        )

    def make_env_processor(self, device: str = "cpu") -> DataProcessorPipeline:
        """Create the observation-side processing pipeline.

        Args:
            device: Torch device used for tensor conversion and stacked state
                outputs in downstream processor steps.

        Returns:
            A ``DataProcessorPipeline`` that augments raw env observations with
            FK-derived EE poses, relative-frame channels, state tensors, and any
            configured image preprocessing.
        """
        env_pipeline_steps = []

        # obs is dict with keys {robot_name}.{axis/joint}.{pos/vel/ee_pos/ee_vel/ee_wrench} | {OBS_IMAGES}{camera_name}
        # {axis} is in {x,y,z,wx,wy,wz}
        if self._kinematics_solver:
            # for all robots that have a solver, we want to fetch their joints and add {robot_name}.{axis}.ee_pos to the obs
            env_pipeline_steps.append(
                JointsToEEObservation(
                    kinematics=self._kinematics_solver,
                    motor_names=self._joint_names,
                )
            )

        # action relative to starting pose
        if any_enabled(self.processor.observation.relative_ee_pos):
            env_pipeline_steps.append(
                RelativeFrameObservationProcessor(
                    enable=self.processor.observation.relative_ee_pos
                )
            )

        if self.processor.image_preprocessing:
            env_pipeline_steps.append(
                ImageCropResizeProcessorStep(
                    crop_params_dict=self.processor.image_preprocessing.crop_params_dict,
                    resize_size=self.processor.image_preprocessing.resize_size
                )
            )

        #env_pipeline_steps.append(
        #    GripperPenaltyProcessorStep(
        #        max_gripper_pos=self.processor.gripper.max_pos,
        #        penalty=self.processor.gripper.penalty,
        #    )
        #)

        env_pipeline_steps.extend([
            # builds OBS_STATE based on what we want to have in there
            # if obs has no joint vel and we want it, compute numerically
            # same for ee_vel
            DefaultObservationProcessor(
                device=device,
                gripper_enable=self.processor.gripper.enable,
                add_joint_position_to_observation=self.processor.observation.add_joint_position_to_observation,
                add_joint_velocity_to_observation=self.processor.observation.add_joint_velocity_to_observation,
                add_current_to_observation=self.processor.observation.add_current_to_observation,
                add_ee_pos_to_observation=self.processor.observation.add_ee_pos_to_observation,
                ee_pos_axes=self.processor.observation.ee_pos_axes,
                add_ee_velocity_to_observation=self.processor.observation.add_ee_velocity_to_observation,
                ee_velocity_axes=self.processor.observation.ee_velocity_axes,
                add_ee_wrench_to_observation=self.processor.observation.add_ee_wrench_to_observation,
                ee_wrench_axes=self.processor.observation.ee_wrench_axes,
                stack_frames=self.processor.observation.stack_frames,
            ),
            DeviceProcessorStep(device=device)
        ])

        # timing hooks
        if self.processor.hooks.time_env_processor:
            from share.utils.control_utils import make_step_timing_hooks
            env_before_hooks, env_after_hooks = make_step_timing_hooks(
                pipeline_steps=env_pipeline_steps,
                label="env",
                log_every=self.processor.hooks.log_every,
                ema_alpha=0.2,
                also_print=False,
            )
        else:
            env_before_hooks, env_after_hooks = [], []

        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition,
            before_step_hooks=env_before_hooks, after_step_hooks=env_after_hooks
        )

    def validate(self, robot_dict, teleop_dict):
        """Validate one primitive against the connected robot/teleop setup.

        Args:
            robot_dict: Connected robot handles keyed by robot name.
            teleop_dict: Connected teleoperator handles keyed by robot name.

        This method normalizes per-robot config fields, initializes any needed
        kinematics solvers, and enforces the task-frame compatibility rules used
        throughout the action and observation pipelines.
        """

        is_task_frame_robot = check_task_frame_robot(robot_dict)
        is_delta_teleoperator = check_delta_teleoperator(teleop_dict)

        # go through each per-robot attribute and check if we need to turn scalar configs into configs for each robot
        if not isinstance(self.task_frame, dict):
            self.task_frame = {name: copy.deepcopy(self.task_frame) for name in robot_dict}

        for attr in ["observation", "gripper", "kinematics"]:
            _attr = getattr(self.processor, attr)
            for fn in fields(_attr):
                if is_union_with_dict(fn.type) and not isinstance(getattr(_attr, fn.name), dict):
                    setattr(_attr, fn.name, {name: getattr(_attr, fn.name) for name in robot_dict})
            setattr(self.processor, attr, _attr)

        # Set up kinematics solver if inverse kinematics is configured
        for name, robot in robot_dict.items():
            if not is_task_frame_robot[name]:
                assert hasattr(robot, "bus")
                self._joint_names[name] = list(robot.bus.motors.keys())

                if self.processor.kinematics.enable[name]:
                    self._kinematics_solver[name] = get_kinematics(
                        robot_name=robot.name,
                        urdf_path=self.processor.kinematics.urdf_path[name],
                        target_frame_name=self.processor.kinematics.target_frame_name[name],
                        joint_names=self._joint_names[name],
                    )

        # checks per robot
        for name, frame in self.task_frame.items():
            if name not in robot_dict:
                raise ValueError(f"Missing robot for task-frame entry '{name}'.")

            # ENV-101: learnable VEL/FORCE axes require delta teleoperator input.
            for axis in frame.learnable_axis_indices:
                if frame.control_mode[axis] in {ControlMode.VEL, ControlMode.WRENCH} and not is_delta_teleoperator[name]:
                    raise ValueError(
                        "Adaptive task-frame axes with VEL/FORCE control require a delta teleoperator. "
                        f"Got robot='{name}', axis={axis}, control_mode={frame.control_mode[axis].name}, "
                        "teleoperator_kind='absolute'."
                    )

            # ENV-102: JOINT-space and joint-only robots must only receive POS axis modes.
            if frame.space == ControlSpace.JOINT:
                non_pos_axes = [i for i, mode in enumerate(frame.control_mode) if mode != ControlMode.POS]
                if non_pos_axes:
                    raise ValueError(
                        "ControlSpace.JOINT only supports POS axis modes. "
                        f"Got robot='{name}', non_pos_axes={non_pos_axes}."
                    )

            if not is_task_frame_robot[name]:
                non_pos_axes = [i for i, mode in enumerate(frame.control_mode) if mode != ControlMode.POS]
                if non_pos_axes:
                    raise ValueError(
                        "Joint-only robots only support POS axis modes in this pipeline. "
                        f"Got robot='{name}', non_pos_axes={non_pos_axes}."
                    )

            # ENV-102: TASK-space with absolute-joint teleop or joint-only robot requires kinematics.
            requires_kinematics = (
                    frame.space == ControlSpace.TASK and
                    (not is_delta_teleoperator[name] or not is_task_frame_robot[name])
            )
            if requires_kinematics and not self.processor.kinematics.enable[name]:
                raise ValueError(
                    "Kinematics must be enabled for TASK-space control when teleop/robot modalities require FK/IK. "
                    f"Set processor.kinematics.enable['{name}']=True."
                )

            if requires_kinematics and name not in self._kinematics_solver:
                raise ValueError(
                    "Kinematics are required but no solver was initialized. "
                    f"Check kinematics config and robot interface for '{name}' "
                    "(urdf_path/target_frame_name/bus availability)."
                )

        # if gripper.enable but the robot has no GRIPPER_KEY action feature, disable
        for name, robot in robot_dict.items():
            if self.processor.gripper.enable[name]:
                if not f"{GRIPPER_KEY}.pos" in robot.action_features:
                    raise ValueError(
                        f"Gripper processing enabled for robot '{name}' but no gripper action feature found. "
                        "Expected an action key like '{GRIPPER_KEY}.pos'."
                    )

    def infer_features(self, robot_dict, cameras):
        """Infer policy-visible feature specs from the configured pipelines.

        Args:
            robot_dict: Connected robots used to sample the raw observation
                schema exposed by the primitive env.
            cameras: Connected cameras used to sample visual observation shapes.

        The inferred features mirror the observation/action contract after the
        env processor has transformed the raw env outputs.
        """
        # process features with respective pipeline
        # get initial obs features from robot_dict instead
        initial_features = {}
        for cam_key, cam in cameras.items():
            initial_features[f"{OBS_IMAGES}.{cam_key}"] = PolicyFeature(type=FeatureType.VISUAL, shape=cam.async_read().shape)

        for name in robot_dict:
            for k, v in robot_dict[name].get_observation().items():
                if isinstance(v, float):
                    shape = (1, )
                elif hasattr(v, "shape"):
                    shape = v.shape
                elif hasattr(v, "__iter__"):
                    shape = (len(v), )
                else:
                    raise ValueError(f"Unknown type for observation {name}.{k}: {type(v)}")
                initial_features[f"{name}.{k}"] = PolicyFeature(type=FeatureType.STATE, shape=shape)

        initial_features = create_initial_features(observation=initial_features)
        env_processor = self.make_env_processor()
        pipeline_features = env_processor.transform_features(initial_features)
        obs_features = pipeline_features[PipelineFeatureType.OBSERVATION]

        action_dim = sum(frame.policy_action_dim for frame in self.task_frame.values())
        action_dim += sum(bool(enable) for enable in self.processor.gripper.enable.values())  # add gripper action dim if enabled

        # expose state, action and visual features
        self.features = {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
            OBS_STATE: pipeline_features[PipelineFeatureType.OBSERVATION][OBS_STATE]
        }
        for key, ft in obs_features.items():
            if ft.type == FeatureType.VISUAL:
                key = strip_prefix(key, PREFIXES_TO_STRIP)
                self.features[f"{OBS_IMAGES}.{key}"] = PolicyFeature(type=FeatureType.VISUAL, shape=ft.shape)

    def on_entry(self, env: ManipulationPrimitive, entry_context: PrimitiveEntryContext | None) -> None:
        """Initialize env-owned runtime state when the primitive becomes active.

        Args:
            env: The primitive env that should own all runtime target state for
                this activation.
            entry_context: Optional processed observation and prior task-frame
                origin from the primitive that just terminated.
        """
        env.set_target_pose(
            {
                name: [float(v) for v in frame.target]
                for name, frame in self.task_frame.items()
            },
            info_key=self.target_pose_info_key,
        )

    def resolve_targets(
        self,
        entry_context: PrimitiveEntryContext | None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """Resolve start and target poses in this primitive's task frame.

        Args:
            entry_context: Optional processed observation and prior task-frame
                origin describing the primitive boundary that just fired.

        Returns:
            A pair ``(start_pose, target_pose)`` where both dictionaries are
            keyed by robot name and expressed in this primitive's task frame.
        """
        start_pose: dict[str, list[float]] = {}
        target_pose: dict[str, list[float]] = {}
        for name, frame in self.task_frame.items():
            start_pose[name] = resolve_entry_start_pose(entry_context, name, frame)
            target_pose[name] = [float(v) for v in frame.target]
        return start_pose, target_pose

@ManipulationPrimitiveConfig.register_subclass("move_delta")
@dataclass
class MoveDeltaPrimitiveConfig(ManipulationPrimitiveConfig):
    """Primitive that resolves fixed task-space position axes from an entry-time delta."""

    delta: list[float] | dict[str, list[float]] = field(default_factory=lambda: [0.0] * 6)
    delta_frame: Literal["world", "ee_current"] | dict[str, Literal["world", "ee_current"]] = "world"
    publish_target_info: bool | dict[str, bool] = True

    def validate(self, robot_dict, teleop_dict):
        """Validate move-delta-specific config fields.

        Args:
            robot_dict: Connected robot handles keyed by robot name.
            teleop_dict: Connected teleoperator handles keyed by robot name.
        """
        super().validate(robot_dict, teleop_dict)
        robot_names = list(robot_dict)
        self.delta = copy_per_robot(self.delta, robot_names)
        self.delta_frame = copy_per_robot(self.delta_frame, robot_names)
        self.publish_target_info = copy_per_robot(self.publish_target_info, robot_names)

        for name, frame in self.task_frame.items():
            if frame.space != ControlSpace.TASK:
                raise ValueError(f"move_delta primitives require TASK-space task frames, got '{name}'.")
            if len(self.delta[name]) != 6:
                raise ValueError(f"move_delta delta for '{name}' must be a 6-vector.")

    def on_entry(self, env: ManipulationPrimitive, entry_context: PrimitiveEntryContext | None) -> None:
        """Resolve entry-time delta targets and publish them into the env.

        Args:
            env: Primitive env that should own the resolved runtime target.
            entry_context: Optional processed observation and previous origin
                used to interpret the prior primitive pose.
        """
        _start_pose, target_pose = self.resolve_targets(entry_context)
        env.set_target_pose(
            target_pose,
            info_key=self.target_pose_info_key if any(self.publish_target_info.values()) else None,
        )

    def resolve_targets(
        self,
        entry_context: PrimitiveEntryContext | None,
    ) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
        """Resolve entry-time start and target poses for a move-delta primitive.

        Args:
            entry_context: Optional processed observation and previous origin
                describing the pose at the primitive switch boundary.

        Returns:
            A pair ``(start_pose, target_pose)`` in this primitive's task frame.
            Fixed ``POS`` axes resolve from ``entry_pose + delta``; learnable
            and non-``POS`` axes keep their configured targets.
        """
        start_pose, target_pose = super().resolve_targets(entry_context)
        for name, frame in self.task_frame.items():
            start_world = task_pose_to_world_pose(start_pose[name], frame.origin)
            target_world = compose_delta_pose(
                start_pose_world=start_world,
                delta=[float(v) for v in self.delta[name]],
                frame_name=self.delta_frame[name],
            )
            resolved_target = world_pose_to_task_pose(target_world, frame.origin)
            target_pose[name] = [float(v) for v in frame.target]
            for axis in self._fixed_pos_axes(frame):
                target_pose[name][axis] = float(resolved_target[axis])
        return start_pose, target_pose

    @staticmethod
    def _fixed_pos_axes(frame: TaskFrame) -> list[int]:
        return [
            axis
            for axis in range(len(frame.target))
            if frame.control_mode[axis] == ControlMode.POS and frame.policy_mode[axis] is None
        ]


@ManipulationPrimitiveConfig.register_subclass("open_loop_trajectory")
@dataclass
class OpenLoopTrajectoryPrimitiveConfig(MoveDeltaPrimitiveConfig):
    """Scripted primitive that executes an internal trajectory over substeps."""

    duration_substeps: int = 10
    substeps_per_step: int = 1
    controller_hz: float | None = None
    substep_dt_s: float | None = None
    interruption_policy: Literal["stop_on_events"] = "stop_on_events"

    def validate(self, robot_dict, teleop_dict):
        """Validate open-loop-specific constraints.

        Args:
            robot_dict: Connected robot handles keyed by robot name.
            teleop_dict: Connected teleoperator handles keyed by robot name.
        """
        super().validate(robot_dict, teleop_dict)
        if self.duration_substeps <= 0:
            raise ValueError("open_loop_trajectory requires duration_substeps > 0.")
        if self.substeps_per_step <= 0:
            raise ValueError("open_loop_trajectory requires substeps_per_step > 0.")
        if self.controller_hz is None and self.substep_dt_s is None:
            self.substep_dt_s = 1.0 / self.processor.fps if self.processor.fps > 0 else 0.0
        elif self.substep_dt_s is None:
            self.substep_dt_s = 1.0 / self.controller_hz

        if self.policy is not None:
            raise ValueError("open_loop_trajectory primitives are scripted-only and must not configure a policy.")
        for name, frame in self.task_frame.items():
            if frame.is_adaptive:
                raise ValueError(f"open_loop_trajectory primitives must be non-adaptive, got learnable axes for '{name}'.")

    def make(
        self,
        robot_dict: dict[str, Robot],
        teleop_dict: dict[str, Teleoperator],
        cameras: dict[str, Camera],
        device: str = "cpu",
    ):
        """Build the scripted open-loop env and its processors.

        Args:
            robot_dict: Connected robot handles keyed by robot name.
            teleop_dict: Connected teleoperators keyed by robot name.
            cameras: Connected camera handles available to the primitive.
            device: Torch device used by observation-side processors.

        Returns:
            A tuple of ``(env, env_processor, action_processor)`` where ``env``
            is the scripted open-loop env subclass for this primitive.
        """
        self.validate(robot_dict, teleop_dict)
        self.infer_features(robot_dict, cameras)

        display_cameras = self.processor.image_preprocessing is not None and self.processor.image_preprocessing.display_cameras
        env = OpenLoopTrajectoryPrimitive(
            task_frame=self.task_frame,
            robot_dict=robot_dict,
            cameras=cameras,
            open_loop_config=self,
            display_cameras=display_cameras,
        )
        env_processor = self.make_env_processor(device)
        action_processor = self.make_action_processor(robot_dict, teleop_dict, device)
        return env, env_processor, action_processor

    def on_entry(self, env: ManipulationPrimitive, entry_context: PrimitiveEntryContext | None) -> None:
        """Seed one scripted open-loop rollout from the primitive entry context.

        Args:
            env: Primitive env that owns the scripted trajectory runtime state.
            entry_context: Optional processed observation and previous origin
                used to interpret the pose at the primitive boundary.
        """
        start_pose, target_pose = self.resolve_targets(entry_context)
        env.set_target_pose(
            target_pose,
            info_key=self.target_pose_info_key if any(self.publish_target_info.values()) else None,
        )
        if isinstance(env, OpenLoopTrajectoryPrimitive):
            env.configure_trajectory(start_pose=start_pose, target_pose=target_pose)
