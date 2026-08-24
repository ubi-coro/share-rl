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


def normalize_franky_state(raw_state: Any) -> FrankaState:
    """Convert Franky/libfranka objects into NumPy-only controller state.

    ``O_T_EE`` is libfranka's flat 16-element *column-major* transform, so it
    needs ``order="F"`` (equivalently ``.reshape(4, 4).T``) -- a plain
    ``.reshape(4, 4)`` silently returns the transpose. ``EE_T_K`` and the
    ``O_dP_EE_*`` twists are already-typed Franky objects (``Affine``,
    ``Twist``), not raw arrays, so they're unwrapped via their own
    ``.matrix``/``.linear``/``.angular`` accessors rather than ``np.asarray``.
    """
    twist = raw_state.O_dP_EE_est if raw_state.O_dP_EE_est is not None else raw_state.O_dP_EE_c
    return FrankaState(
        q=np.asarray(raw_state.q, dtype=np.float64).reshape(7),
        dq=np.asarray(raw_state.dq, dtype=np.float64).reshape(7),
        T_base_ee=np.asarray(raw_state.O_T_EE, dtype=np.float64).reshape(4, 4, order="F"),
        T_ee_stiffness=np.asarray(raw_state.EE_T_K.matrix, dtype=np.float64),
        twist_base_ee=np.concatenate((np.asarray(twist.linear), np.asarray(twist.angular))),
        wrench_base_at_stiffness=np.asarray(raw_state.O_F_ext_hat_K, dtype=np.float64).reshape(6),
        timestamp=time.monotonic(),
    )


class FrankaControllerProcess(mp.Process):
    """500 Hz SHARE bridge around Franky's native Cartesian/joint impedance motions."""

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
            strategy = self.config.controller.make_strategy(self.config)
            wrench_bias_base = np.zeros(6, dtype=np.float64)
            raw_state = robot.state
            state = normalize_franky_state(raw_state)

            active_space: ControlSpace | None = None
            current_command: dict[str, Any] | None = None
            last_command_time = -np.inf
            last_sequence = -1
            period = 1.0 / float(self.config.frequency)
            next_tick = time.perf_counter()
            last_tick = next_tick
            joint_hold_target: np.ndarray | None = None
            joint_holding = False
            last_translational_stiffness: float | None = None
            last_rotational_stiffness: float | None = None
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
                state = normalize_franky_state(raw_state)
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
                            motion = self._make_task_motion(franky, state)
                            # Seed the reference to the current pose before
                            # handing control to the motion -- matches
                            # franky's own CartesianImpedanceTracker, which
                            # does the same to avoid a jump if the RT loop
                            # reads a reference before this process's first
                            # real strategy.step() call lands.
                            motion.set_reference(
                                franky.CartesianReference(target=franky.Affine(state.T_base_ee))
                            )
                            robot.move(motion, asynchronous=True)
                        else:
                            motion = self._make_joint_motion(
                                franky,
                                np.asarray(self.config.joint_stiffness, dtype=np.float64),
                                np.asarray(self.config.joint_damping, dtype=np.float64),
                            )
                            motion.set_reference(franky.JointReference(q=state.q))
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
                output = None
                if active_space == ControlSpace.TASK:
                    output = strategy.step(state, None if stale else current_command, dt)
                    T_base_task = pose_rpy_to_transform(current_command["origin"])
                    target_matrix = T_base_task @ pose_rpy_to_transform(output.target_pose_task_rpy)
                    motion.set_reference(
                        franky.CartesianReference(target=franky.Affine(target_matrix))
                    )
                    # set_reference is meant to be called every tick (that's
                    # the whole point of a tracking motion), but gains only
                    # change on a new command or a stale/hold transition --
                    # skip the call otherwise rather than pushing an
                    # unchanged CartesianImpedanceGains through the RT loop
                    # 500 times a second.
                    if (
                        output.translational_stiffness != last_translational_stiffness
                        or output.rotational_stiffness != last_rotational_stiffness
                    ):
                        motion.set_gains(
                            franky.CartesianImpedanceGains.isotropic(
                                output.translational_stiffness, output.rotational_stiffness
                            )
                        )
                        last_translational_stiffness = output.translational_stiffness
                        last_rotational_stiffness = output.rotational_stiffness
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
                    raise FrankaControllerError("Franky motion ended unexpectedly")

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
            # set_ee wants a flat length-16 sequence, not a (4, 4) array.
            robot.set_ee(list(self.config.end_effector_transform))
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
                list(self.config.payload_center_of_mass),
                # set_load wants a flat length-9 sequence, not a (3, 3) array.
                list(self.config.payload_inertia),
            )

    def _make_task_motion(self, franky: Any, state: FrankaState) -> Any:
        return franky.CartesianImpedanceTrackingMotion(
            translational_stiffness=float(self.config.translational_stiffness),
            rotational_stiffness=float(self.config.rotational_stiffness),
            force_constraints=self._force_constraints(),
            posture_task=franky.PostureTask(
                target=state.q,
                stiffness=np.asarray(self.config.nullspace_stiffness, dtype=np.float64),
                max_torque=float(self.config.nullspace_max_torque),
            ),
            lower_joint_limits=FR3_LOWER_JOINT_LIMITS,
            upper_joint_limits=FR3_UPPER_JOINT_LIMITS,
            gains_time_constant=float(self.config.gains_time_constant_s),
        )

    def _force_constraints(self) -> list[float | None]:
        return [
            None if not np.isfinite(value) else float(value)
            for value in self.config.force_constraints
        ]

    def _make_joint_motion(
        self,
        franky: Any,
        stiffness: np.ndarray,
        damping: np.ndarray,
    ) -> Any:
        return franky.JointImpedanceTrackingMotion(
            stiffness=stiffness,
            damping=damping,
            compensate_coriolis=True,
            lower_joint_limits=FR3_LOWER_JOINT_LIMITS,
            upper_joint_limits=FR3_UPPER_JOINT_LIMITS,
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
        output: Any,
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
            holding = True
            origin = (
                command["origin"].copy()
                if command is not None
                else np.zeros(6, dtype=np.float64)
            )
        else:
            pose = output.pose_task_rpy
            twist = output.twist_task
            wrench = output.measured_wrench_task
            holding = output.holding
            origin = np.asarray(command["origin"], dtype=np.float64)

        self.robot_out_rb.put(
            {
                "ActualTCPPose": pose,
                "ActualTCPSpeed": twist,
                "ActualTCPForce": wrench,
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
        """Ramp the last commanded torque down via TorqueStopMotion.

        Robot.stop() preempts a torque motion's control loop with a
        ControlException instead of ramping down -- that's the whole reason
        TorqueStopMotion exists (see franky's own CartesianImpedanceTracker.stop(),
        which follows the same pattern). So a "Move command preempted!"
        ControlException here is an expected, tolerable outcome -- e.g.
        control already ended some other way -- not a failure to fall back
        from. Only escalate to the abrupt robot.stop() if TorqueStopMotion
        itself couldn't even be attempted (e.g. franky failed to load) or
        failed for a genuinely different reason.
        """
        try:
            franky = load_franky()
        except BaseException:
            return
        try:
            robot.move(franky.TorqueStopMotion(), asynchronous=False)
            return
        except BaseException as error:
            if "preempt" in str(error).lower():
                return
        try:
            robot.stop()
        except BaseException:
            pass

    def _default_command(self) -> FrankaTaskFrameCommand:
        command = FrankaTaskFrameCommand()
        command.controller_overrides = {
            "translational_stiffness": float(self.config.translational_stiffness),
            "rotational_stiffness": float(self.config.rotational_stiffness),
            "min_pose": list(self.config.min_pose_rpy),
            "max_pose": list(self.config.max_pose_rpy),
            "rotation_interval_modes": list(self.config.rotation_interval_modes),
            "compliance_reference_limit_enable": list(
                self.config.compliance_reference_limit_enable
            ),
            "joint_stiffness": list(self.config.joint_stiffness),
            "joint_damping": list(self.config.joint_damping),
            "joint_error_clip": list(self.config.joint_error_clip),
        }
        return command
