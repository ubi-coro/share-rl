from __future__ import annotations

import importlib.metadata
import multiprocessing as mp
import os
import queue
import time
import traceback
from typing import Any

import numpy as np

from share.envs.manipulation_primitive.task_frame import ControlSpace
from share.utils.shared_memory import Empty, SharedMemoryQueue, SharedMemoryRingBuffer

from .command import FrankaCommand, FrankaTaskFrameCommand
from .control_law import (
    ControllerOutput,
    FrankaState,
    measured_wrench_in_task,
    pose_rpy_to_transform,
    rotate_twist,
    task_pose,
)


FR3_LOWER_JOINT_LIMITS = np.array(
    [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508],
    dtype=np.float64,
)
FR3_UPPER_JOINT_LIMITS = np.array(
    [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508],
    dtype=np.float64,
)


class FrankaControllerError(RuntimeError):
    pass


def load_franky() -> Any:
    """Import and validate the optional Franky dependency inside the worker."""
    try:
        import franky
    except ImportError as error:
        raise ImportError(
            "The Franka backend requires Franky 2.x. Install the wheel built for "
            "your robot server/libfranka version with the share-rl[franka] extra."
        ) from error

    version = getattr(franky, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version("franky-control")
        except importlib.metadata.PackageNotFoundError:
            version = None
    if version is not None and str(version).split(".", maxsplit=1)[0] != "2":
        raise RuntimeError(
            f"Unsupported Franky version {version}; SHARE requires the MIT-licensed 2.x line"
        )
    return franky


def normalize_franky_state(raw_state: Any, model: Any, franky: Any) -> FrankaState:
    """Convert Franky/libfranka objects into NumPy-only controller state."""
    twist = getattr(raw_state, "O_dP_EE_est", None)
    if twist is None:
        twist = getattr(raw_state, "O_dP_EE_c", raw_state.O_dP_EE_d)
    jacobian = model.zero_jacobian(franky.Frame.EndEffector, raw_state)
    return FrankaState(
        q=np.asarray(raw_state.q, dtype=np.float64).reshape(7),
        dq=np.asarray(raw_state.dq, dtype=np.float64).reshape(7),
        T_base_ee=np.asarray(raw_state.O_T_EE, dtype=np.float64).reshape(4, 4),
        T_ee_stiffness=np.asarray(raw_state.EE_T_K, dtype=np.float64).reshape(4, 4),
        twist_base_ee=np.asarray(twist, dtype=np.float64).reshape(6),
        wrench_base_at_stiffness=np.asarray(
            raw_state.O_F_ext_hat_K, dtype=np.float64
        ).reshape(6),
        jacobian_base_ee=np.asarray(jacobian, dtype=np.float64).reshape(6, 7),
        timestamp=time.monotonic(),
    )


class FrankaControllerProcess(mp.Process):
    """500 Hz SHARE bridge around Franky's 1 kHz torque motion."""

    def __init__(self, config: Any):
        if config.shm_manager is None:
            raise ValueError("FrankaConfig.shm_manager must be a running SharedMemoryManager")
        super().__init__(name="FrankaControllerProcess")
        self.config = config
        self.ready_event = mp.Event()
        self.stop_requested_event = mp.Event()
        self.unexpected_exit_event = mp.Event()
        self.error_queue: mp.Queue[str] = mp.Queue(maxsize=4)
        self._sequence = 0

        command_example = self._default_command().to_queue_dict()
        self.robot_cmd_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=config.shm_manager,
            examples=command_example,
            buffer_size=256,
        )
        output_example = {
            "ActualTCPPose": np.zeros(6, dtype=np.float64),
            "ActualTCPSpeed": np.zeros(6, dtype=np.float64),
            "ActualTCPForce": np.zeros(6, dtype=np.float64),
            "SetTCPForce": np.zeros(6, dtype=np.float64),
            "TaskFrameOrigin": np.zeros(6, dtype=np.float64),
            "ActualQ": np.zeros(7, dtype=np.float64),
            "ActualQd": np.zeros(7, dtype=np.float64),
            "timestamp": 0.0,
            "monotonic_timestamp": 0.0,
            "command_sequence": np.int64(-1),
            "loop_duration_s": 0.0,
            "deadline_missed": False,
            "control_command_success_rate": 0.0,
            "holding": True,
        }
        self.robot_out_rb = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=config.shm_manager,
            examples=output_example,
            get_max_k=config.get_max_k,
            get_time_budget=0.4,
            put_desired_frequency=config.frequency,
        )

    @property
    def is_ready(self) -> bool:
        return (
            self.ready_event.is_set()
            and self.is_alive()
            and not self.unexpected_exit_event.is_set()
        )

    def start(self, wait: bool = True) -> None:
        super().start()
        if wait and not self.ready_event.wait(timeout=self.config.launch_timeout):
            self.check_health()
            raise TimeoutError("Timed out while connecting to the Franka controller")

    def send_cmd(self, command: FrankaTaskFrameCommand) -> None:
        self.check_health()
        self._sequence += 1
        self.robot_cmd_queue.put(
            command.to_queue_dict(sequence=self._sequence, timestamp=time.monotonic())
        )

    def zero_ft(self) -> None:
        self.send_cmd(FrankaTaskFrameCommand.zero_ft_command())

    def get_robot_state(self) -> dict[str, Any]:
        self.check_health()
        if self.robot_out_rb.count == 0:
            raise FrankaControllerError("The Franka controller has not published state yet")
        return self.robot_out_rb.get()

    def check_health(self) -> None:
        try:
            message = self.error_queue.get_nowait()
        except queue.Empty:
            message = None
        if message is not None:
            raise FrankaControllerError(message)
        if self.unexpected_exit_event.is_set():
            raise FrankaControllerError("The Franka controller process exited unexpectedly")

    def stop(self, wait: bool = True) -> None:
        if self.is_alive():
            try:
                self.send_cmd(FrankaTaskFrameCommand.stop_command())
            except (FrankaControllerError, queue.Full):
                self.stop_requested_event.set()
            if wait:
                self.join(timeout=self.config.launch_timeout + 3.0)
                if self.is_alive():
                    self.terminate()
                    self.join(timeout=2.0)

    def run(self) -> None:
        robot = None
        motion = None
        normal_stop = False
        try:
            if self.config.rt_core is not None:
                os.sched_setaffinity(0, {int(self.config.rt_core)})

            franky = load_franky()
            realtime = (
                franky.RealtimeConfig.Enforce
                if self.config.enforce_realtime
                else franky.RealtimeConfig.Ignore
            )
            robot = franky.Robot(
                self.config.robot_ip,
                realtime_config=realtime,
                default_torque_threshold=20.0,
                default_force_threshold=30.0,
            )
            self._configure_robot(robot)
            model = robot.model
            strategy = self.config.controller.make_strategy(self.config)
            wrench_bias_base = np.zeros(6, dtype=np.float64)
            raw_state = robot.state
            state = normalize_franky_state(raw_state, model, franky)

            motion = self._make_joint_motion(
                franky,
                state.q,
                np.asarray(self.config.joint_stiffness, dtype=np.float64),
                np.asarray(self.config.joint_damping, dtype=np.float64),
            )
            motion.set_reference(franky.JointReference(q=state.q))
            robot.move(motion, asynchronous=True)

            active_space: ControlSpace | None = None
            current_command: dict[str, Any] | None = None
            last_command_time = -np.inf
            last_sequence = -1
            period = 1.0 / float(self.config.frequency)
            next_tick = time.perf_counter()
            last_tick = next_tick
            joint_hold_target: np.ndarray | None = None
            joint_holding = False
            self.ready_event.set()

            while not self.stop_requested_event.is_set():
                parent = mp.parent_process()
                if parent is not None and not parent.is_alive():
                    normal_stop = True
                    break

                loop_start = time.perf_counter()
                dt = max(loop_start - last_tick, np.finfo(np.float64).eps)
                last_tick = loop_start
                raw_state = robot.state
                state = normalize_franky_state(raw_state, model, franky)
                stop, zero_requested, latest = self._drain_commands()
                if stop:
                    normal_stop = True
                    break
                if zero_requested:
                    wrench_bias_base = state.wrench_base_at_stiffness.copy()
                    strategy.zero_wrench(state)
                if latest is not None:
                    space = ControlSpace(int(latest["space"]))
                    if active_space is None:
                        active_space = space
                        if space == ControlSpace.TASK:
                            motion = self._make_torque_motion(franky)
                            robot.move(motion, asynchronous=True)
                    elif space != active_space:
                        raise FrankaControllerError(
                            "Franka control space is fixed for each connection"
                        )
                    current_command = latest
                    last_command_time = float(latest["command_timestamp"])
                    last_sequence = int(latest["sequence"])
                    if space == ControlSpace.JOINT:
                        motion.set_gains(
                            franky.JointImpedanceGains(
                                stiffness=latest["joint_stiffness"],
                                damping=latest["joint_damping"],
                            )
                        )
                        joint_hold_target = None
                        joint_holding = False

                stale = loop_start - last_command_time > self.config.command_timeout_s
                output: ControllerOutput | None = None
                if active_space == ControlSpace.TASK:
                    strategy_result = strategy.step(
                        state,
                        None if stale else current_command,
                        model,
                        dt,
                    )
                    if isinstance(strategy_result, ControllerOutput):
                        output = strategy_result
                        torque = strategy_result.torque
                    else:
                        torque = np.asarray(strategy_result, dtype=np.float64)
                        if torque.shape != (7,):
                            raise ValueError(
                                "Franka controller strategies must return seven torques"
                            )
                    motion.set_torque(torque)
                elif active_space == ControlSpace.JOINT:
                    if stale:
                        if not joint_holding:
                            joint_hold_target = state.q.copy()
                            hold_stiffness = np.full(7, 20.0, dtype=np.float64)
                            motion.set_gains(
                                franky.JointImpedanceGains(
                                    stiffness=hold_stiffness,
                                    damping=2.0 * np.sqrt(hold_stiffness),
                                )
                            )
                            joint_holding = True
                        target = joint_hold_target
                    else:
                        joint_holding = False
                        requested = current_command["target"][:7]
                        error_clip = current_command["joint_error_clip"]
                        target = state.q + np.clip(
                            requested - state.q, -error_clip, error_clip
                        )
                    motion.set_reference(franky.JointReference(q=target))

                if robot.poll_motion():
                    raise FrankaControllerError("Franky torque motion ended unexpectedly")

                loop_duration = time.perf_counter() - loop_start
                deadline_missed = loop_duration > period
                self._publish_state(
                    state,
                    output,
                    wrench_bias_base,
                    current_command,
                    stale,
                    last_sequence,
                    loop_duration,
                    deadline_missed,
                    float(getattr(raw_state, "control_command_success_rate", 0.0)),
                )

                next_tick += period
                remaining = next_tick - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
                elif remaining < -period:
                    next_tick = time.perf_counter()

            normal_stop = True
        except BaseException:
            self.unexpected_exit_event.set()
            message = traceback.format_exc()
            try:
                self.error_queue.put_nowait(message)
            except queue.Full:
                pass
        finally:
            self.ready_event.clear()
            if robot is not None:
                self._graceful_stop(robot)
            if not normal_stop and not self.unexpected_exit_event.is_set():
                self.unexpected_exit_event.set()

    def _configure_robot(self, robot: Any) -> None:
        robot.set_collision_behavior(
            self.config.lower_torque_thresholds_nominal,
            self.config.upper_torque_thresholds_nominal,
            self.config.lower_force_thresholds_nominal,
            self.config.upper_force_thresholds_nominal,
        )
        if self.config.end_effector_transform is not None:
            robot.set_ee(
                np.asarray(self.config.end_effector_transform, dtype=np.float64).reshape(4, 4)
            )
        load_fields = (
            self.config.payload_mass,
            self.config.payload_center_of_mass,
            self.config.payload_inertia,
        )
        if any(value is not None for value in load_fields):
            if not all(value is not None for value in load_fields):
                raise ValueError(
                    "payload_mass, payload_center_of_mass, and payload_inertia must be set together"
                )
            robot.set_load(
                float(self.config.payload_mass),
                np.asarray(self.config.payload_center_of_mass, dtype=np.float64),
                np.asarray(self.config.payload_inertia, dtype=np.float64).reshape(3, 3),
            )

    def _make_torque_motion(self, franky: Any) -> Any:
        return franky.SimpleTorqueMotion(
            initial_torque=np.zeros(7),
            signal_timeout=float(self.config.torque_signal_timeout_s),
            compensate_coriolis=True,
            max_delta_tau=float(self.config.max_delta_tau),
            lower_joint_limits=FR3_LOWER_JOINT_LIMITS,
            upper_joint_limits=FR3_UPPER_JOINT_LIMITS,
            joint_limit_activation_distance=float(self.config.joint_limit_margin),
            joint_limit_stiffness=float(self.config.joint_limit_potential),
            joint_limit_damping=2.0 * np.sqrt(float(self.config.joint_limit_potential)),
            joint_limit_max_torque=float(self.config.joint_limit_max_torque),
        )

    def _make_joint_motion(
        self,
        franky: Any,
        q: np.ndarray,
        stiffness: np.ndarray,
        damping: np.ndarray,
    ) -> Any:
        del q
        return franky.JointImpedanceTrackingMotion(
            stiffness=stiffness,
            damping=damping,
            compensate_coriolis=True,
            max_delta_tau=float(self.config.max_delta_tau),
            lower_joint_limits=FR3_LOWER_JOINT_LIMITS,
            upper_joint_limits=FR3_UPPER_JOINT_LIMITS,
            joint_limit_activation_distance=float(self.config.joint_limit_margin),
            joint_limit_stiffness=float(self.config.joint_limit_potential),
            joint_limit_damping=2.0 * np.sqrt(float(self.config.joint_limit_potential)),
            joint_limit_max_torque=float(self.config.joint_limit_max_torque),
            gains_time_constant=float(self.config.gains_time_constant_s),
        )

    def _drain_commands(
        self,
    ) -> tuple[bool, bool, dict[str, Any] | None]:
        try:
            batch = self.robot_cmd_queue.get_all()
        except Empty:
            return False, False, None
        stop = False
        zero_requested = False
        latest = None
        count = len(batch["cmd"])
        for index in range(count):
            command_type = FrankaCommand(int(batch["cmd"][index]))
            if command_type is FrankaCommand.STOP:
                stop = True
            elif command_type is FrankaCommand.ZERO_FT:
                zero_requested = True
            else:
                latest = {key: value[index].copy() for key, value in batch.items()}
        return stop, zero_requested, latest

    def _publish_state(
        self,
        state: FrankaState,
        output: ControllerOutput | None,
        wrench_bias_base: np.ndarray,
        command: dict[str, Any] | None,
        stale: bool,
        sequence: int,
        loop_duration: float,
        deadline_missed: bool,
        control_success_rate: float,
    ) -> None:
        if output is None:
            T_base_task = (
                pose_rpy_to_transform(command["origin"])
                if command is not None
                else np.eye(4)
            )
            pose = task_pose(state.T_base_ee, T_base_task)
            twist = rotate_twist(
                state.twist_base_ee, T_base_task[:3, :3].T
            )
            wrench = measured_wrench_in_task(
                state.wrench_base_at_stiffness - wrench_bias_base,
                state.T_base_ee,
                state.T_ee_stiffness,
                T_base_task,
            )
            desired_wrench = np.zeros(6)
            holding = stale or command is None
            origin = (
                command["origin"].copy()
                if command is not None
                else np.zeros(6, dtype=np.float64)
            )
        else:
            pose = output.pose_task_rpy
            twist = output.twist_task
            wrench = output.measured_wrench_task
            desired_wrench = output.desired_wrench_task
            holding = output.holding
            origin = np.asarray(command["origin"], dtype=np.float64)

        self.robot_out_rb.put(
            {
                "ActualTCPPose": pose,
                "ActualTCPSpeed": twist,
                "ActualTCPForce": wrench,
                "SetTCPForce": desired_wrench,
                "TaskFrameOrigin": origin,
                "ActualQ": state.q,
                "ActualQd": state.dq,
                "timestamp": time.time(),
                "monotonic_timestamp": time.monotonic(),
                "command_sequence": np.int64(sequence),
                "loop_duration_s": float(loop_duration),
                "deadline_missed": bool(deadline_missed),
                "control_command_success_rate": float(control_success_rate),
                "holding": bool(holding),
            },
            wait=False,
        )

    @staticmethod
    def _graceful_stop(robot: Any) -> None:
        try:
            franky = load_franky()
            robot.move(franky.TorqueStopMotion(), asynchronous=False)
        except BaseException:
            try:
                robot.stop()
            except BaseException:
                pass

    def _default_command(self) -> FrankaTaskFrameCommand:
        command = FrankaTaskFrameCommand()
        command.controller_overrides = {
            "kp": list(self.config.kp),
            "kd": list(self.config.kd),
            "min_pose": list(self.config.min_pose_rpy),
            "max_pose": list(self.config.max_pose_rpy),
            "rotation_interval_modes": list(self.config.rotation_interval_modes),
            "wrench_limits": list(self.config.wrench_limits),
            "compliance_reference_limit_enable": list(
                self.config.compliance_reference_limit_enable
            ),
            "compliance_adaptive_limit_enable": list(
                self.config.compliance_adaptive_limit_enable
            ),
            "compliance_desired_wrench": list(
                self.config.compliance_desired_wrench
            ),
            "compliance_adaptive_limit_min": list(
                self.config.compliance_adaptive_limit_min
            ),
            "nullspace_stiffness": list(self.config.nullspace_stiffness),
            "nullspace_damping": list(self.config.nullspace_damping),
            "nullspace_max_torque": float(self.config.nullspace_max_torque),
            "joint_stiffness": list(self.config.joint_stiffness),
            "joint_damping": list(self.config.joint_damping),
            "joint_error_clip": list(self.config.joint_error_clip),
        }
        return command
