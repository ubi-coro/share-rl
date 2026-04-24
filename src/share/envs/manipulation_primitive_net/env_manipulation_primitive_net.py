import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.processor import create_transition, TransitionKey, EnvTransition
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.utils.constants import ACTION
from lerobot.cameras import Camera
from lerobot.teleoperators import Teleoperator
from lerobot.robots import Robot
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.transition import Transition

from share.envs.manipulation_primitive.config_manipulation_primitive import PrimitiveEntryContext
from share.envs.manipulation_primitive.task_frame import ControlMode
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.transitions import DEFAULT_TARGET_POSE_AXES_INFO_KEY
from share.envs.utils import task_frame_origins
from share.teleoperators import TeleopEvents


class ManipulationPrimitiveNet(gym.Env):
    """Gym wrapper that chains manipulation primitives using typed transitions."""

    def __init__(self, config: ManipulationPrimitiveNetConfig):

        self.config = config

        # initialize hardware environments
        self.robot_dict, self.teleop_dict, self.cameras = self.connect()

        self._envs = {}
        self._env_processors = {}
        self._action_processors = {}
        self._transitions: dict[str, list[Transition]] = {}

        for name, primitive in self.config.primitives.items():
            env, env_processor, action_processor = primitive.make(
                self.robot_dict,
                self.teleop_dict,
                self.cameras,
                device=getattr(self.config, "device", "cpu")
            )
            self._envs[name] = env
            self._env_processors[name] = env_processor
            self._action_processors[name] = action_processor
            self._transitions[name] = []

        for transition in self.config.transitions:
            self._transitions[transition.source].append(transition)

        self._active = self.config.reset_primitive
        self._last_reset_info: dict[str, Any] = {}
        self._pending_entry_context: PrimitiveEntryContext | None = None
        self._episode_step_count = 0
        self._primitive_step_count = 0
        self._needs_full_reset = True
        self._step_info: dict[str, Any] = {}

    @property
    def active_primitive(self) -> str:
        return self._active

    @property
    def action_dim(self) -> int:
        return self.config.primitives[self._active].features[ACTION].shape[0]

    @property
    def in_terminal(self):
        return self.config.primitives[self._active].is_terminal

    def connect(self) -> tuple[dict[str, "Robot"], dict[str, "Teleoperator"], dict[str, "Camera"]]:
        """Connect all hardware/configured IO needed by this MP-Net.

        Returns:
            A tuple ``(robot_dict, teleop_dict, cameras)`` keyed by configured
            names and ready to be shared across all primitive envs.
        """
        assert self.config.robot is not None, "Robot config must be provided for real robot environment"

        from lerobot.cameras import make_cameras_from_configs
        from lerobot.teleoperators import make_teleoperator_from_config
        from lerobot.robots import make_robot_from_config

        # Handle multi robot configuration
        robot_dict = {}
        for name in self.config.robot:
            robot_dict[name] = make_robot_from_config(self.config.robot[name])
            robot_dict[name].connect()

        # Handle multi teleop configuration
        teleop_dict = {}
        for name in self.config.teleop:
            teleop_dict[name] = make_teleoperator_from_config(self.config.teleop[name])
            teleop_dict[name].connect()

        # Handle cameras
        cameras = make_cameras_from_configs(self.config.cameras)
        for name in cameras:
            cameras[name].connect()

        return robot_dict, teleop_dict, cameras

    def step(self, action: np.ndarray | torch.Tensor) -> EnvTransition:
        """Step the active primitive once and evaluate outgoing transitions.

        Args:
            action: Flat policy action for the currently active primitive.

        Returns:
            The processed transition emitted by the active primitive step. If an
            outgoing transition fires, the transition metadata indicates the next
            primitive and ``reset()`` must be called to enter it.
        """
        if self._needs_full_reset:
            raise RuntimeError("step() called after MP-Net episode finished; call reset() before stepping again.")

        transition = self._step_env_and_check_transitions(action)
        self._needs_full_reset |= self.in_terminal

        return transition

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> EnvTransition:
        """Reset the MP-Net or enter the next primitive after a transition.

        Args:
            seed: Optional gym reset seed.
            options: Optional reset payload forwarded to primitive envs.

        Returns:
            The processed reset transition for the primitive that becomes active.
        """
        super().reset(seed=seed)
        self._primitive_step_count = 0
        if self._needs_full_reset:
            transition = self._full_reset(seed=seed, options=options)
            self._needs_full_reset = False
            self._episode_step_count = 0
            return transition

        return self._enter_active_primitive(seed=seed, options=options, entry_context=self._pending_entry_context)

    def close(self):
        for camera in self.cameras.values():
            camera.disconnect()
        for robot in self.robot_dict.values():
            robot.disconnect()
        for teleop in self.teleop_dict.values():
            teleop.disconnect()

        keys = list(self._envs.keys())
        for k in keys:
            self._envs[k].close()
            del self._envs[k]
            del self._action_processors[k]
            del self._env_processors[k]

    def set_step_info(self, info: dict[str, Any] | None) -> None:
        """Set fixed info flags to seed every outer `step(action)` call."""

        self._step_info = {} if info is None else dict(info)

    def _step_env_and_check_transitions(self, action: torch.Tensor) -> EnvTransition:
        """Execute one primitive step and evaluate at most one transition edge.

        Args:
            action: Flat policy action for the current primitive.

        Returns:
            The processed transition after action processing, env stepping,
            observation processing, and optional transition routing.
        """
        self._episode_step_count += 1
        self._primitive_step_count += 1
        active = self._active
        if active not in self._envs:
            raise KeyError(f"Unknown active primitive '{active}'.")
        primitive = self.config.primitives[active]

        # 1) Process action
        info = dict(self._step_info)
        #if primitive.policy is None and not getattr(self._envs[active], "uses_autonomous_step", False):
        #    info[TeleopEvents.IS_INTERVENTION] = True

        action_transition = create_transition(action=action, info=info)
        processed_action_transition = self._action_processors[active](action_transition)

        # todo: find a better solution to exit on intervention end
        #if processed_action_transition[TransitionKey.INFO].get(TeleopEvents.INTERVENTION_COMPLETED, False):
        #    return processed_action_transition

        # 2) Step environment
        raw_obs, reward, terminated, truncated, info = self._envs[active].step(processed_action_transition[TransitionKey.ACTION])

        # 3) Read out info and possibly overwrite action
        complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
        info.update(processed_action_transition[TransitionKey.INFO].copy())

        if info.get(TeleopEvents.IS_INTERVENTION, False) and TELEOP_ACTION_KEY in complementary_data:
            action_to_record = complementary_data[TELEOP_ACTION_KEY]
        else:
            action_to_record = action

        # 4) Process observation
        transition = create_transition(
            observation=raw_obs,
            action=action_to_record,
            reward=reward + processed_action_transition[TransitionKey.REWARD],
            done=terminated or processed_action_transition[TransitionKey.DONE],
            truncated=truncated or processed_action_transition[TransitionKey.TRUNCATED],
            info=info,
            complementary_data=complementary_data,
        )
        processed_transition = self._env_processors[active](transition)
        processed_obs = processed_transition[TransitionKey.OBSERVATION]

        # 5) Build info
        info = processed_transition.get(TransitionKey.INFO, {})
        info["step"] = self._primitive_step_count
        info["primitive_step"] = self._primitive_step_count
        info["episode_step"] = self._episode_step_count
        info["transition_from"] = active
        info["transition_to"] = active
        info["transition_reason"] = None
        info[DEFAULT_TARGET_POSE_AXES_INFO_KEY] = self._default_target_pose_axes(primitive)

        # 6) Check for transitions
        for transition in self._transitions[self._active]:
            result = transition.evaluate(obs=processed_obs, info=info)
            if not (result.terminated or result.truncated):
                continue

            # condition has fired
            target = transition.target
            self._pending_entry_context = PrimitiveEntryContext(
                source_primitive=active,
                target_primitive=target,
                observation=dict(processed_obs),
                task_frame_origin=task_frame_origins(primitive),
            )
            self._primitive_step_count = 0
            self._active = target

            processed_transition[TransitionKey.REWARD] += result.reward
            processed_transition[TransitionKey.DONE] |= result.terminated
            processed_transition[TransitionKey.TRUNCATED] |= result.truncated
            info["transition_to"] = target
            info["transition_reason"] = result.reason
            break

        info.pop(DEFAULT_TARGET_POSE_AXES_INFO_KEY, None)
        processed_transition[TransitionKey.INFO] = info
        return processed_transition

    @staticmethod
    def _default_target_pose_axes(primitive: Any) -> dict[str, list[int]]:
        default_axes: dict[str, list[int]] = {}
        for name, frame in primitive.task_frame.items():
            axes = [
                axis
                for axis in range(len(frame.target))
                if frame.control_mode[axis] == ControlMode.POS and frame.policy_mode[axis] is None
            ]
            default_axes[name] = axes
        return default_axes

    def _step_reset_path_until_start(self, obs: dict[str, np.ndarray], info: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Walk the reset graph until ``start_primitive`` becomes active.

        Args:
            obs: Latest processed observation from the reset rollout.
            info: Latest processed info dictionary from the reset rollout.

        Returns:
            The most recent processed ``(obs, info)`` pair observed while
            traversing the reset path toward ``start_primitive``.
        """
        while self._active != self.config.start_primitive:
            start_loop_t = time.perf_counter()

            action = self.sample_action()
            transition = self._step_env_and_check_transitions(action)
            obs = transition[TransitionKey.OBSERVATION]
            info.update(transition[TransitionKey.INFO])  # keep at least prior info dict if env returns empty

            # todo: is this necessary?
            if self._pending_entry_context is not None and self._active != self.config.start_primitive:
                entered = self._enter_active_primitive(
                    seed=None,
                    options=None,
                    entry_context=self._pending_entry_context,
                )
                obs = entered[TransitionKey.OBSERVATION]
                info.update(entered[TransitionKey.INFO])

            dt_load = time.perf_counter() - start_loop_t
            precise_sleep(1 / self.config.fps - dt_load)
        return obs, info

    def _full_reset(self, seed: int | None, options: dict[str, Any] | None) -> EnvTransition:
        """Perform an episode reset beginning from ``reset_primitive``.

        Args:
            seed: Optional reset seed used to derive per-primitive env seeds.
            options: Optional reset payload forwarded to primitive envs.

        Returns:
            The processed reset transition for the primitive that should be
            active when user-facing stepping resumes.
        """
        self._pending_entry_context = None
        self._active = self.config.reset_primitive
        for name, primitive in self.config.primitives.items():
            if name == self._active:
                continue
            self._env_processors[name].reset()
            self._action_processors[name].reset()
            env_seed = None if seed is None else seed + sum(ord(c) for c in name)
            self._envs[name].reset(seed=env_seed, options={} if options is None else dict(options))

        transition = self._enter_active_primitive(seed=seed, options=options, entry_context=None)
        if self._active != self.config.start_primitive:
            self._step_reset_path_until_start(
                obs=transition[TransitionKey.OBSERVATION],
                info=transition[TransitionKey.INFO],
            )
            transition = self._enter_active_primitive(
                seed=seed,
                options=options,
                entry_context=self._pending_entry_context,
            )
        self._episode_step_count = 0
        return transition

    def _enter_active_primitive(
        self,
        seed: int | None,
        options: dict[str, Any] | None,
        entry_context: PrimitiveEntryContext | None,
    ) -> EnvTransition:
        """Enter the active primitive and run its entry hook.

        Args:
            seed: Optional reset seed used to derive this env's seed.
            options: Optional reset payload forwarded to the primitive env.
            entry_context: Optional processed observation and previous
                task-frame origin captured when the prior primitive terminated.

        Returns:
            The processed reset transition for the newly entered primitive.
        """
        primitive = self.config.primitives[self._active]
        self._env_processors[self._active].reset()
        self._action_processors[self._active].reset()

        env_seed = None if seed is None else seed + sum(ord(c) for c in self._active)
        raw_obs, raw_info = self._envs[self._active].reset(
            seed=env_seed,
            options={} if options is None else dict(options),
        )
        transition = create_transition(observation=raw_obs, info=raw_info)
        processed_transition = self._env_processors[self._active](transition)
        processed_obs = processed_transition[TransitionKey.OBSERVATION]
        if entry_context is None:
            entry_context = PrimitiveEntryContext(
                source_primitive=None,
                target_primitive=self._active,
                observation=dict(processed_obs),
                task_frame_origin=task_frame_origins(primitive),
            )

        self._envs[self._active].reset_runtime_state()

        # run entry hook
        primitive.on_entry(self._envs[self._active], entry_context)

        if hasattr(self._envs[self._active], "apply_task_frames"):
            self._envs[self._active].apply_task_frames()

        processed_transition[TransitionKey.INFO] = {
            **processed_transition.get(TransitionKey.INFO, {}),
            **getattr(self._envs[self._active], "_get_info", lambda: {})(),
        }

        self._last_reset_info = processed_transition[TransitionKey.INFO]
        self._pending_entry_context = None
        return processed_transition

    def sample_action(self, primitive: str | None = None) -> Any:
        if primitive is None:
            primitive = self._active

        ft = self.config.primitives[primitive].features[ACTION]
        return 2 * torch.rand(size=ft.shape) - 1
