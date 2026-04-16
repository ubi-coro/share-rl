import math
import gc
import os
import time
import enum
import multiprocessing as mp
from dataclasses import dataclass, asdict, replace
from multiprocessing.managers import SharedMemoryManager
from typing import Any, ClassVar

import numpy as np
from scipy.spatial.transform import Rotation as R

from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame
from share.utils.shared_memory import SharedMemoryRingBuffer, SharedMemoryQueue, Empty
from share.utils.transformation_utils import (
    clip_angle_to_ccw_arc,
    euler_xyz_to_rotvec,
    exp_scale,
    homogeneous_to_sixvec,
    RollingPerfWindow,
    RotationIntervalMode,
    rotvec_to_euler_xyz,
    seconds_to_ms,
    signed_error_to_nearest_arc_endpoint,
    sixvec_to_homogeneous,
    wrap_to_pi,
)

# ---------------------------------------------------------------------------
# Internal enums
# ---------------------------------------------------------------------------
DeltaMode = PolicyMode

class Command(enum.IntEnum):
    SET = 0
    STOP = 1
    OPEN = 2
    CLOSE = 3
    ZERO_FT = 4


@dataclass
class TaskFrameCommand(TaskFrame):
    """Controller command with user-facing rotational pose inputs in RPY.

    The rotational parts of ``origin`` and absolute rotational ``target`` entries
    are specified as XYZ roll-pitch-yaw angles [rad] at the interface. Internally,
    the controller converts them to rotation vectors before use.
    """
    cmd: Command = Command.SET
    SUPPORTED_CONTROLLER_OVERRIDE_KEYS: ClassVar[set[str]] = {
        "kp",
        "kd",
        "min_pose",
        "max_pose",
        "wrench_limits",
        "compliance_reference_limit_enable",
        "compliance_adaptive_limit_enable",
        "compliance_desired_wrench",
        "compliance_adaptive_limit_min",
        "rotation_interval_modes"
    }

    @property
    def delta_mode(self) -> list[DeltaMode]:
        """Per-axis absolute/relative interpretation derived from ``policy_mode``."""
        return [DeltaMode.RELATIVE if m  == PolicyMode.RELATIVE else DeltaMode.ABSOLUTE for m in self.policy_mode]

    def to_queue_dict(self):
        """Convert the command to a queue-friendly dict of NumPy arrays and ints.

        The rotational part of ``origin`` is interpreted as user-facing XYZ
        roll-pitch-yaw [rad] and converted to an internal rotation vector.
        """
        d = asdict(self)
        d.pop("joint_names")
        raw_overrides = d.pop("controller_overrides", None) or {}
        unknown = set(raw_overrides) - self.SUPPORTED_CONTROLLER_OVERRIDE_KEYS
        if unknown:
            raise ValueError(f"Unsupported UR task-frame controller overrides: {', '.join(sorted(unknown))}")
        try:
            d["cmd"] = self.cmd.value
            d.pop("policy_mode", None)
            d["space"] = np.asarray(self.space).astype(np.int8)
            d["control_mode"] = np.array([int(m) if m is not None else -1 for m in self.control_mode])
            d["policy_mode"] = np.array([int(m) if m is not None else -1 for m in self.policy_mode])
            d["delta_mode"] = np.array([int(m) if m is not None else -1 for m in self.delta_mode])
            d["target"] = np.asarray(self.target).astype(np.float64)
            d["origin"] = np.asarray(self.origin).astype(np.float64)
            d["origin"][3:6] = R.from_euler("xyz", d["origin"][3:6], degrees=False).as_rotvec()
            d["max_pose"] = np.asarray(raw_overrides.get("max_pose", self.max_pose)).astype(np.float64)
            d["min_pose"] = np.asarray(raw_overrides.get("min_pose", self.min_pose)).astype(np.float64)
            d["rotation_interval_modes"] = np.array(
                [
                    int(RotationIntervalMode.from_name(str(mode)))
                    for mode in raw_overrides.get("rotation_interval_modes", ["linear"] * 6)
                ],
                dtype=np.int8,
            )
            d["kp"] = np.asarray(raw_overrides.get("kp", [2500.0, 2500.0, 2500.0, 150.0, 150.0, 150.0])).astype(np.float64)
            d["kd"] = np.asarray(raw_overrides.get("kd", [80.0, 80.0, 80.0, 8.0, 8.0, 8.0])).astype(np.float64)
            d["wrench_limits"] = np.asarray(raw_overrides.get("wrench_limits", [30.0, 30.0, 30.0, 3.0, 3.0, 3.0])).astype(np.float64)
            d["compliance_adaptive_limit_enable"] = np.asarray(raw_overrides.get("compliance_adaptive_limit_enable", [False] * 6)).astype(np.bool_)
            d["compliance_reference_limit_enable"] = np.asarray(raw_overrides.get("compliance_reference_limit_enable", [False] * 6)).astype(np.bool_)
            d["compliance_desired_wrench"] = np.asarray(raw_overrides.get("compliance_desired_wrench", [5.0, 5.0, 5.0, 0.5, 0.5, 0.5])).astype(np.float64)
            d["compliance_adaptive_limit_min"] = np.asarray(raw_overrides.get("compliance_adaptive_limit_min", [0.1] * 6)).astype(np.float64)
        except Exception as e:
            raise ValueError(f"TaskFrameCommand seems to be missing fields: {e}")

        return d

    def to_robot_action(self):
        action_dict = {}
        if self.space == ControlSpace.JOINT:
            for i in range(len(self.target)):
                if self.control_mode[i] != ControlMode.POS:
                    raise ValueError("UR joint-space control only supports POS axes")
                action_dict[f"joint_{i + 1}.pos"] = self.target[i]
            return action_dict

        for i, ax in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
            if self.control_mode[i] == ControlMode.POS:
                action_dict[f"{ax}.ee_pos"] = self.target[i]
            elif self.control_mode[i] == ControlMode.VEL:
                action_dict[f"{ax}.ee_vel"] = self.target[i]
            elif self.control_mode[i] == ControlMode.WRENCH:
                action_dict[f"{ax}.ee_wrench"] = self.target[i]
        return action_dict
    

class RTDETaskFrameController(mp.Process):
    """RTDE task-frame controller with per-axis modes and 6D impedance.

    Runs a 1 kHz loop that:
      • Reads commands from shared memory (pose/vel/force modes per axis)
      • Estimates current state in the task frame
      • Integrates virtual targets (for IMPEDANCE_VEL)
      • Computes and bounds a wrench, then applies it via `forceMode(...)`

    Notes:
        - Translation bounds are enforced directly; rotation bounds are applied
          in RPY space but the controller operates internally on rot-vectors.
        - Automatically (re)enters `forceMode` as needed.

    Attributes:
        config (URConfig): Runtime configuration (RTDE IP, gains, limits, etc.).
        ready_event (mp.Event): Set once the control loop is alive.
        robot_cmd_queue (SharedMemoryQueue): Incoming `TaskFrameCommand`s.
        robot_out_rb (SharedMemoryRingBuffer): Outgoing robot state samples.
    """

    def __init__(self, config: 'URConfig'):
        """Initialize controller processes, queues, and default internal state.

        Args:
            config (URConfig): Configuration (frequency, limits, payload/TCP, etc.).

        Raises:
            AssertionError: If `config` fields are inconsistent (validated/normalized).
        """

        config = _validate_config(config)
        super().__init__(name="RTDETaskFrameController")
        self.config = config
        self.ready_event = mp.Event()  # “ready” event to signal when the loop has started successfully
        self.force_on = False  # are we currently in forceMode?
        self._receive_keys = [
            'ActualTCPPose',
            'ActualTCPSpeed',
            'ActualTCPForce',
            'ActualQ',
            'ActualQd',
        ]

        # 1) Build the command queue (TaskFrameCommand messages)
        self.robot_cmd_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=config.shm_manager,
            examples=TaskFrameCommand().to_queue_dict(),
            buffer_size=256
        )

        # 2) Build the ring buffer for streaming back pose/vel/force
        if self.config.mock:
            raise ValueError("UR does not support mocks")
        else:
            from rtde_receive import RTDEReceiveInterface
        rtde_r = RTDEReceiveInterface(hostname=config.robot_ip)

        example = dict()
        for key in self._receive_keys:
            example[key] = np.array(getattr(rtde_r, 'get' + key)())
        example["ActualTCPForceFiltered"] = np.array([0.0] * 6)
        example["SetTCPForce"] = np.array([0.0] * 6)
        example['timestamp'] = time.time()
        self.robot_out_rb = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=config.shm_manager,
            examples=example,
            get_max_k=config.get_max_k,
            get_time_budget=0.4,
            put_desired_frequency=config.frequency
        )

        # 3) Controller state: last TaskFrameCommand, task‐frame state, gains, etc.
        self._last_cmd = TaskFrameCommand(
            controller_overrides={
                "kp": np.array(self.config.kp, dtype=np.float64),
                "kd": np.array(self.config.kd, dtype=np.float64),
                "wrench_limits": np.array(self.config.wrench_limits, dtype=np.float64),
                "compliance_adaptive_limit_enable": np.array(self.config.compliance_adaptive_limit_enable, dtype=bool),
                "compliance_reference_limit_enable": np.array(self.config.compliance_reference_limit_enable, dtype=bool),
                "compliance_desired_wrench": np.array(self.config.compliance_desired_wrench, dtype=np.float64),
                "compliance_adaptive_limit_min": np.array(self.config.compliance_adaptive_limit_min, dtype=np.float64),
                "rotation_interval_modes": ["linear"] * 6
            },
        )
        self.origin = self._last_cmd.origin
        self.control_mode = self._last_cmd.control_mode
        self.delta_mode = self._last_cmd.delta_mode
        self.target = self._last_cmd.target
        self.max_pose = self._last_cmd.max_pose
        self.min_pose = self._last_cmd.min_pose
        self._resolve_compliance_settings(**self._last_cmd.controller_overrides)
        self._active_space: ControlSpace | None = None

    # =========== launch & shutdown =============
    def connect(self):
        """Spawn the control process and block until the first iteration completes."""
        self.start()

    def start(self, wait=True):
        """Start the control process.

        Args:
            wait (bool, optional): If True, block until the loop signals readiness.
        """
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        """Request a graceful shutdown of the control loop.

        Args:
            wait (bool, optional): If True, join the process before returning.
        """
        # Send a STOP command
        stop_cmd = replace(self._last_cmd)
        stop_cmd.cmd = Command.STOP
        self.robot_cmd_queue.put(stop_cmd.to_queue_dict())
        if wait:
            self.stop_wait()

    def start_wait(self):
        """Block until the controller signals ready or the launch timeout elapses.

        Raises:
            AssertionError: If the process is not alive after waiting.
        """
        self.ready_event.wait(self.config.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        """Join the control process (blocks until termination)."""
        self.join()

    @property
    def is_ready(self):
        """bool: True once the control loop completed its first successful cycle."""
        return self.ready_event.is_set()

    # =========== context manager ============
    def __enter__(self):
        """Context: start controller and return self (blocks until ready)."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context: stop controller on exit, regardless of exceptions."""
        self.stop()

    # =========== sending a new TaskFrameCommand ============
    def send_cmd(self, cmd: TaskFrameCommand):
        """Merge cmd into the last command and push the result to the queue.
        The first call stores a full copy; subsequent calls update only fields
        that are provided (non-None).

        Args:
            cmd (TaskFrameCommand): Partial or full command to apply.
        """
        self._ensure_control_space(cmd.space)
        self._last_cmd = cmd
        self.robot_cmd_queue.put(cmd.to_queue_dict())

    @property
    def task_frame(self) -> TaskFrameCommand:
        return replace(self._last_cmd)

    def zero_ft(self):
        """Re-zero the force-torque sensor in the control loop."""
        # We only need the cmd field for ZERO_FT, everything else can be None
        zero_cmd = replace(self._last_cmd)
        zero_cmd.cmd = Command.ZERO_FT
        self.robot_cmd_queue.put(zero_cmd.to_queue_dict())

    # =========== get robot state from ring buffer ============
    def get_robot_state(self, k=None, out=None):
        """Get the latest (or last k) robot state sample(s).

        Args:
            k (int, optional): If `None`, return the latest sample. If an integer,
                return the last `k` samples.
            out (dict, optional): Optional preallocated output buffer.

        Returns:
            dict or tuple[dict,...]: State dict(s) including:
                - ``'ActualTCPPose'`` (6, ) task-frame pose (x,y,z, rx,ry,rz)
                - ``'ActualTCPSpeed'`` (6, ) task-frame twist
                - ``'ActualTCPForce'`` (6, ) task-frame wrench
                - any additional keys requested via `config.receive_keys`
                - ``'SetTCPForce'`` (6, ) last commanded wrench in the task frame
                - ``'timestamp'`` (float)
        """
        if k is None:
            return self.robot_out_rb.get(out=out)
        else:
            return self.robot_out_rb.get_last_k(k=k, out=out)

    def get_all_robot_states(self):
        """Return all buffered robot states currently stored in the ring buffer.
        Returns:
            list[dict]: Chronologically ordered state samples.
        """
        return self.robot_out_rb.get_all()

    # ========= main loop in process ============
    def run(self):
        """Run the RTDE control loop until a stop command or transport failure.

        The loop follows one fixed order on every iteration: apply pending
        commands, read the latest robot state, update the virtual reference,
        compute the commanded wrench/torque, send it, and then wait for the
        next control period. Rotational task-space targets remain user-facing
        XYZ Euler angles at the interface and are converted internally only
        where the controller needs SO(3) operations.
        """
        self._configure_realtime()
        robot_ip = self.config.robot_ip
        dt = 1.0 / self.config.frequency
        rtde_c, rtde_r = self._connect_rtde_interfaces()
        ft_alpha = self._force_filter_alpha(dt)
        wrench_F = np.zeros(6, dtype=np.float64)
        measured_wrench_F = np.zeros(6, dtype=np.float64)

        try:
            if self.config.verbose:
                print(f"[RTDETaskFrameController] Connecting to {robot_ip}...")

            self._configure_robot_session(rtde_c)

            pose_F = self.read_current_state(rtde_r)["ActualTCPPose"]
            x_cmd = pose_F.copy()
            q_cmd = np.array(rtde_r.getActualQ(), dtype=np.float64)
            active_space: ControlSpace | None = None
            keep_running = True
            iter_idx = 0
            perf = self._init_perf_tracking(dt)

            while keep_running:
                t_loop_start = rtde_c.initPeriod()
                t_iter0 = time.monotonic()
                dt_loop = t_iter0 - perf["t_prev"]
                perf["t_prev"] = t_iter0
                perf["dt_win"].add(dt_loop)

                section_start = time.monotonic()
                msgs, n_cmd = self._get_pending_commands()
                perf["sec_wins"]["queue_get"].add(time.monotonic() - section_start)

                section_start = time.monotonic()
                keep_running, active_space, x_cmd, q_cmd = self._apply_pending_commands(
                    msgs=msgs,
                    n_cmd=n_cmd,
                    rtde_c=rtde_c,
                    rtde_r=rtde_r,
                    active_space=active_space,
                    x_cmd=x_cmd,
                    q_cmd=q_cmd,
                )
                perf["sec_wins"]["cmd_apply"].add(time.monotonic() - section_start)
                if not keep_running:
                    break

                section_start = time.monotonic()
                current_state = self.read_current_state(rtde_r)
                pose_F = current_state["ActualTCPPose"]
                v_F = current_state["ActualTCPSpeed"]
                measured_wrench_F = self._update_filtered_wrench(
                    current_state["ActualTCPForce"], measured_wrench_F, ft_alpha
                )
                perf["sec_wins"]["read_state"].add(time.monotonic() - section_start)

                section_start = time.monotonic()
                q_actual, qd_actual = self._populate_current_state(
                    rtde_r=rtde_r,
                    current_state=current_state,
                    measured_wrench_F=measured_wrench_F,
                    wrench_F=wrench_F,
                )
                perf["sec_wins"]["recv_extra"].add(time.monotonic() - section_start)

                section_start = time.monotonic()
                self.robot_out_rb.put(current_state)
                perf["sec_wins"]["rb_put"].add(time.monotonic() - section_start)

                section_start = time.monotonic()
                x_cmd, q_cmd = self._update_virtual_targets(
                    active_space=active_space,
                    x_cmd=x_cmd,
                    q_cmd=q_cmd,
                    pose_F=pose_F,
                    dt=dt,
                )
                perf["sec_wins"]["virt_update"].add(time.monotonic() - section_start)

                section_start = time.monotonic()
                wrench_F, torque_cmd = self._compute_output_command(
                    active_space=active_space,
                    x_cmd=x_cmd,
                    q_cmd=q_cmd,
                    pose_F=pose_F,
                    v_F=v_F,
                    measured_wrench_F=measured_wrench_F,
                    q_actual=q_actual,
                    qd_actual=qd_actual,
                )
                perf["sec_wins"]["wrench"].add(time.monotonic() - section_start)

                section_start = time.monotonic()
                self._send_output_command(active_space, rtde_c, wrench_F, torque_cmd)
                perf["sec_wins"]["forcemode"].add(time.monotonic() - section_start)

                compute_time = time.monotonic() - t_iter0
                perf["compute_win"].add(compute_time)

                section_start = time.monotonic()
                rtde_c.waitPeriod(t_loop_start)
                perf["sec_wins"]["waitPeriod"].add(time.monotonic() - section_start)

                iter_idx += 1
                self._maybe_log_timing(
                    perf=perf,
                    iter_idx=iter_idx,
                    dt_loop=dt_loop,
                    compute_time=compute_time,
                    n_cmd=n_cmd,
                    t_iter0=t_iter0,
                )

                if not self.ready_event.is_set():
                    self.ready_event.set()
        finally:
            self._cleanup_rtde(rtde_c, rtde_r)
            self.ready_event.set()
            if self.config.verbose:
                print(f"[RTDETaskFrameController] Disconnected from robot {robot_ip}")

    def _configure_realtime(self) -> None:
        """Enable the optional soft real-time scheduler settings for the child process."""
        if not self.config.soft_real_time:
            return
        os.sched_setaffinity(0, {self.config.rt_core})
        os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))

    def _connect_rtde_interfaces(self):
        """Create RTDE control/receive interfaces for the configured UR robot."""
        if self.config.mock:
            raise ValueError("UR does not support mocks")

        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        frequency = self.config.frequency
        return (
            RTDEControlInterface(self.config.robot_ip, frequency),
            RTDEReceiveInterface(self.config.robot_ip),
        )

    def _configure_robot_session(self, rtde_c) -> None:
        """Apply one-time TCP and payload configuration after connecting."""
        if self.config.tcp_offset_pose is not None:
            rtde_c.setTcp(self.config.tcp_offset_pose)

        if self.config.payload_mass is None:
            return

        if self.config.payload_cog is not None:
            assert rtde_c.setPayload(self.config.payload_mass, self.config.payload_cog)
        else:
            assert rtde_c.setPayload(self.config.payload_mass)

    def _force_filter_alpha(self, dt: float) -> float | None:
        """Return the low-pass coefficient for the measured wrench filter."""
        if self.config.ft_filter_cutoff_hz is None:
            return None

        cutoff_hz = float(self.config.ft_filter_cutoff_hz)
        tau = 1.0 / (2.0 * np.pi * max(cutoff_hz, 1e-6))
        return dt / (tau + dt)

    def _init_perf_tracking(self, dt: float) -> dict[str, Any]:
        """Create rolling windows and thresholds for optional verbose timing logs."""
        win_secs = 2.0
        win_len = int(win_secs * self.config.frequency)
        sec_names = [
            "queue_get", "cmd_apply", "read_state", "recv_extra",
            "rb_put", "virt_update", "wrench", "forcemode", "waitPeriod",
        ]
        t_prev = time.monotonic()
        return {
            "dt_nom": dt,
            "dt_win": RollingPerfWindow.create(win_len),
            "compute_win": RollingPerfWindow.create(win_len),
            "sec_names": sec_names,
            "sec_wins": {name: RollingPerfWindow.create(win_len) for name in sec_names},
            "t_prev": t_prev,
            "log_interval": 5.0,
            "next_log_time": t_prev + 5.0,
            "spike_abs_s": max(0.002, 3.0 * dt),
            "spike_rel": 3.0,
            "spike_compute_s": max(0.0015, 2.0 * dt),
        }

    def _update_filtered_wrench(
        self,
        raw_wrench: np.ndarray,
        measured_wrench_F: np.ndarray,
        ft_alpha: float | None,
    ) -> np.ndarray:
        """Update the filtered task-frame wrench estimate."""
        if ft_alpha is None:
            return np.asarray(raw_wrench, dtype=np.float64)
        return measured_wrench_F + ft_alpha * (raw_wrench - measured_wrench_F)

    def _populate_current_state(
        self,
        *,
        rtde_r,
        current_state: dict[str, np.ndarray],
        measured_wrench_F: np.ndarray,
        wrench_F: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fill extra receive keys, publish filtered wrench, and return joint state arrays."""
        for key in self._receive_keys:
            if key not in current_state:
                current_state[key] = np.array(getattr(rtde_r, "get" + key)())

        current_state["ActualTCPForceFiltered"] = np.array(measured_wrench_F)
        current_state["SetTCPForce"] = np.array(wrench_F)
        current_state["timestamp"] = time.time()
        q_actual = np.asarray(current_state["ActualQ"], dtype=np.float64)
        qd_actual = np.asarray(current_state["ActualQd"], dtype=np.float64)
        return q_actual, qd_actual

    def _update_virtual_targets(
        self,
        *,
        active_space: ControlSpace | None,
        x_cmd: np.ndarray,
        q_cmd: np.ndarray,
        pose_F: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance the stored virtual target for the active control space."""
        if active_space == ControlSpace.TASK:
            for i in range(3):
                control_mode_i = ControlMode(self.control_mode[i])
                delta_mode_i = DeltaMode(self.delta_mode[i])
                if control_mode_i == ControlMode.POS and delta_mode_i == DeltaMode.ABSOLUTE:
                    x_cmd[i] = self.target[i]
                elif control_mode_i == ControlMode.POS and delta_mode_i == DeltaMode.RELATIVE:
                    x_cmd[i] += float(self.target[i]) * dt

            x_cmd[3:6] = self._integrate_virtual_target_rotation(x_cmd[3:6], dt)
            x_cmd = self.clip_pose(x_cmd)
            x_cmd = self._clamp_virtual_target_error_task(x_cmd, pose_F)
            return x_cmd, q_cmd

        if active_space == ControlSpace.JOINT:
            for i in range(len(q_cmd)):
                if DeltaMode(self.delta_mode[i]) == DeltaMode.ABSOLUTE:
                    q_cmd[i] = float(self.target[i])
                else:
                    q_cmd[i] += float(self.target[i]) * dt

        return x_cmd, q_cmd

    def _compute_output_command(
        self,
        *,
        active_space: ControlSpace | None,
        x_cmd: np.ndarray,
        q_cmd: np.ndarray,
        pose_F: np.ndarray,
        v_F: np.ndarray,
        measured_wrench_F: np.ndarray,
        q_actual: np.ndarray,
        qd_actual: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the wrench or torque command for the active control space."""
        wrench_F = np.zeros(6, dtype=np.float64)
        torque_cmd = np.zeros(6, dtype=np.float64)
        if active_space == ControlSpace.TASK:
            wrench_F = self._compute_task_wrench(
                x_cmd=x_cmd,
                pose_F=pose_F,
                v_F=v_F,
                measured_wrench_F=measured_wrench_F,
            )
        elif active_space == ControlSpace.JOINT:
            torque_cmd = self._compute_joint_torque(
                q_cmd=q_cmd,
                q_actual=q_actual,
                qd_actual=qd_actual,
            )
        return wrench_F, torque_cmd

    def _send_output_command(
        self,
        active_space: ControlSpace | None,
        rtde_c,
        wrench_F: np.ndarray,
        torque_cmd: np.ndarray,
    ) -> None:
        """Send the already-computed output command through the appropriate RTDE API."""
        if active_space == ControlSpace.TASK:
            self._send_task_wrench(rtde_c, wrench_F)
        elif active_space == ControlSpace.JOINT:
            self._send_joint_torque(rtde_c, torque_cmd)

    def _maybe_log_timing(
        self,
        *,
        perf: dict[str, Any],
        iter_idx: int,
        dt_loop: float,
        compute_time: float,
        n_cmd: int,
        t_iter0: float,
    ) -> None:
        """Emit optional rolling timing summaries and spike diagnostics."""
        if self.config.verbose and t_iter0 >= perf["next_log_time"] and perf["dt_win"].buf:
            self._log_perf_summary(perf)
            perf["next_log_time"] = t_iter0 + perf["log_interval"]

        is_dt_spike = (dt_loop > perf["spike_abs_s"]) or (dt_loop > perf["spike_rel"] * perf["dt_nom"])
        is_compute_spike = compute_time > perf["spike_compute_s"]
        if self.config.verbose and (is_dt_spike or is_compute_spike):
            self._log_perf_spike(perf, iter_idx, dt_loop, compute_time, n_cmd)

    def _log_perf_summary(self, perf: dict[str, Any]) -> None:
        """Print a short rolling summary of loop timing and hottest sections."""
        dt_stats = perf["dt_win"].stats()
        compute_stats = perf["compute_win"].stats()
        sec_lines = []
        for name in perf["sec_names"]:
            stats = perf["sec_wins"][name].stats()
            if stats is None:
                continue
            sec_lines.append((name, stats["p99"], stats["max"]))
        sec_lines.sort(key=lambda item: item[1], reverse=True)
        top = "  ".join(
            f"{name}:p99={seconds_to_ms(p99):.2f} max={seconds_to_ms(mx):.2f}"
            for name, p99, mx in sec_lines[:5]
        )
        print(
            f"[RTDETaskFrameController] dt_loop(ms) p50={seconds_to_ms(dt_stats['p50']):.2f} "
            f"p90={seconds_to_ms(dt_stats['p90']):.2f} p99={seconds_to_ms(dt_stats['p99']):.2f} "
            f"max={seconds_to_ms(dt_stats['max']):.2f} | compute(ms) p50={seconds_to_ms(compute_stats['p50']):.2f} "
            f"p99={seconds_to_ms(compute_stats['p99']):.2f} max={seconds_to_ms(compute_stats['max']):.2f} | top: {top}"
        )

    def _log_perf_spike(
        self,
        perf: dict[str, Any],
        iter_idx: int,
        dt_loop: float,
        compute_time: float,
        n_cmd: int,
    ) -> None:
        """Print one detailed spike diagnostic when a loop iteration overruns badly."""
        last_secs = {
            name: (perf["sec_wins"][name].buf[-1] if perf["sec_wins"][name].buf else float("nan"))
            for name in perf["sec_names"]
        }
        culprit = max(last_secs.items(), key=lambda item: (0.0 if math.isnan(item[1]) else item[1]))
        print(
            f"[RTDETaskFrameController][SPIKE] iter={iter_idx} "
            f"dt_loop={seconds_to_ms(dt_loop):.2f}ms (dt={seconds_to_ms(perf['dt_nom']):.2f}ms) "
            f"compute={seconds_to_ms(compute_time):.2f}ms n_cmd={n_cmd} "
            f"culprit={culprit[0]}:{seconds_to_ms(culprit[1]):.2f}ms secs(ms)="
            + " ".join(f"{name}={seconds_to_ms(last_secs[name]):.2f}" for name in perf["sec_names"])
            + f" gc_count={gc.get_count()}"
        )

    def _ensure_control_space(self, space: ControlSpace | int) -> ControlSpace:
        """Lock the controller to its first commanded control space."""
        resolved = ControlSpace(int(space))
        if self._active_space is None:
            self._active_space = resolved
            return resolved
        if resolved != self._active_space:
            raise ValueError(
                "UR controller does not support switching between task-space and joint-space control"
            )
        return resolved

    def _compute_joint_torque(
        self,
        q_cmd: np.ndarray,
        q_actual: np.ndarray,
        qd_actual: np.ndarray,
    ) -> np.ndarray:
        """Compute joint torques from a simple impedance control law."""
        kp = np.asarray(self.kp, dtype=np.float64)
        kd = np.asarray(self.kd, dtype=np.float64)
        return kp * (q_cmd - q_actual) - kd * qd_actual

    @staticmethod
    def _send_joint_torque(rtde_c, torque_cmd: np.ndarray) -> None:
        """Send torques through the available UR RTDE joint-torque API."""
        torque = np.asarray(torque_cmd, dtype=np.float64).tolist()
        if hasattr(rtde_c, "directTorque"):
            rtde_c.directTorque(torque, True)
        elif hasattr(rtde_c, "torqueCommand"):
            rtde_c.torqueCommand(torque, True)
        else:
            raise AttributeError("RTDEControlInterface does not expose directTorque/torqueCommand")

    def _compute_task_wrench(
        self,
        x_cmd: np.ndarray,
        pose_F: np.ndarray,
        v_F: np.ndarray,
        measured_wrench_F: np.ndarray,
    ) -> np.ndarray:
        """Compute a bounded task-space wrench from the impedance control law."""
        wrench_F = np.zeros(6, dtype=np.float64)
        err_vec = np.zeros(6, dtype=np.float64)
        err_vec[:3] = x_cmd[:3] - np.array(pose_F[:3])

        R_cmd = R.from_rotvec(x_cmd[3:6])
        R_act = R.from_rotvec(pose_F[3:6])
        R_err = R_cmd * R_act.inv()
        err_vec[3:6] = R_err.as_rotvec()

        for i in range(6):
            control_mode_i = ControlMode(self.control_mode[i])

            if control_mode_i == ControlMode.WRENCH:
                wrench_F[i] = float(self.target[i])
                continue

            if control_mode_i == ControlMode.POS:
                e = float(err_vec[i])
                edot = float(-v_F[i])
            elif control_mode_i == ControlMode.VEL:
                e = 0.0
                edot = float(self.target[i] - v_F[i])
            else:
                e = 0.0
                edot = 0.0

            wrench_F[i] = self.kp[i] * e + self.kd[i] * edot

        self.apply_wrench_bounds(pose_F, desired_wrench=wrench_F, measured_wrench=measured_wrench_F)
        return wrench_F

    def _send_task_wrench(self, rtde_c, wrench_F: np.ndarray) -> None:
        """Send a task-space wrench through UR force mode."""
        rtde_c.forceMode(
            self.origin.tolist(),
            [1, 1, 1, 1, 1, 1],
            np.asarray(wrench_F, dtype=np.float64).tolist(),
            2,
            self.config.speed_limits,
        )
        self.force_on = True

    def _enter_task_force_mode(self, rtde_c) -> None:
        """Start forceMode once for task-space control."""
        rtde_c.forceModeSetGainScaling(self.config.force_mode_gain_scaling)
        self._send_task_wrench(rtde_c, np.zeros(6, dtype=np.float64))

    def _get_pending_commands(self) -> tuple[dict[str, np.ndarray] | None, int]:
        """Drain the shared-memory queue and return all pending command payloads."""
        try:
            msgs = self.robot_cmd_queue.get_all()
            return msgs, len(msgs["cmd"])
        except Empty:
            return None, 0

    @classmethod
    def _transform_task_pose_between_frames(
        cls,
        pose: np.ndarray,
        source_origin: np.ndarray,
        target_origin: np.ndarray,
    ) -> np.ndarray:
        """Re-express one internal task-frame pose in a different task frame."""
        T_world_source = sixvec_to_homogeneous(source_origin)
        T_source_pose = sixvec_to_homogeneous(pose)
        T_world_pose = T_world_source @ T_source_pose

        T_world_target = sixvec_to_homogeneous(target_origin)
        T_target_pose = np.linalg.inv(T_world_target) @ T_world_pose
        return np.asarray(homogeneous_to_sixvec(T_target_pose), dtype=np.float64)

    def _apply_pending_commands(
        self,
        msgs: dict[str, np.ndarray] | None,
        n_cmd: int,
        rtde_c,
        rtde_r,
        active_space: ControlSpace | None,
        x_cmd: np.ndarray,
        q_cmd: np.ndarray,
    ) -> tuple[bool, ControlSpace | None, np.ndarray, np.ndarray]:
        """Apply queued commands and update controller state and virtual targets."""
        keep_running = True
        if msgs is None:
            return keep_running, active_space, x_cmd, q_cmd

        for i in range(n_cmd):
            single = {k: msgs[k][i] for k in msgs}
            cmd_id = int(single["cmd"])
            if cmd_id == Command.STOP.value:
                keep_running = False
                break

            if cmd_id == Command.ZERO_FT.value:
                rtde_c.zeroFtSensor()
                continue

            if cmd_id != Command.SET.value:
                keep_running = False
                break

            new_space = self._ensure_control_space(single["space"])
            if active_space is None:
                active_space = new_space
            elif new_space != active_space:
                raise ValueError(
                    "UR controller does not support switching between task-space and joint-space control"
                )

            previous_origin = np.asarray(self.origin, dtype=np.float64).copy()
            new_origin = np.asarray(single["origin"], dtype=np.float64).copy()
            if new_space == ControlSpace.TASK and not np.allclose(previous_origin, new_origin):
                x_cmd = self._transform_task_pose_between_frames(
                    pose=x_cmd,
                    source_origin=previous_origin,
                    target_origin=new_origin,
                )

            self.origin = new_origin
            self.target = single["target"].copy()
            self.max_pose = single["max_pose"].copy()
            self.min_pose = single["min_pose"].copy()
            self._resolve_compliance_settings(**single)

            pose_F = self.read_current_state(rtde_r)["ActualTCPPose"]
            q_now = np.array(rtde_r.getActualQ(), dtype=np.float64)
            new_control_mode = single["control_mode"]
            new_delta_mode = single["delta_mode"]

            if new_space == ControlSpace.JOINT and np.any(new_control_mode != ControlMode.POS):
                raise ValueError("UR joint-space control only supports POS axes")

            for axis in range(6):
                became_relative_pos = (
                    new_control_mode[axis] != self.control_mode[axis]
                    and new_control_mode[axis] == ControlMode.POS
                    and new_delta_mode[axis] == DeltaMode.RELATIVE
                )
                if not became_relative_pos:
                    continue
                if new_space == ControlSpace.TASK:
                    x_cmd[axis] = pose_F[axis]
                else:
                    q_cmd[axis] = q_now[axis]

            self.control_mode = new_control_mode.copy()
            self.delta_mode = new_delta_mode.copy()
            if new_space == ControlSpace.TASK and not self.force_on:
                self._enter_task_force_mode(rtde_c)

        return keep_running, active_space, x_cmd, q_cmd

    def _resolve_compliance_settings(
        self,
        kp,
        kd,
        wrench_limits,
        compliance_adaptive_limit_enable,
        compliance_reference_limit_enable,
        compliance_desired_wrench,
        compliance_adaptive_limit_min,
        rotation_interval_modes,
        **kwargs
    ):
        """Normalize active compliance settings and compute derived adaptive scales."""
        self.kp = kp
        self.kd = kd
        self.wrench_limits = np.asarray(wrench_limits, dtype=np.float64).copy()
        self.compliance_desired_wrench = np.asarray(compliance_desired_wrench, dtype=np.float64).copy()
        self.compliance_adaptive_limit_enable = np.asarray(compliance_adaptive_limit_enable, dtype=bool).copy()
        self.compliance_reference_limit_enable = np.asarray(compliance_reference_limit_enable, dtype=bool).copy()
        self.compliance_adaptive_limit_min = np.asarray(compliance_adaptive_limit_min, dtype=np.float64).copy()
        self.compliance_adaptive_limit_theta = np.zeros(6, dtype=np.float64)
        self.rotation_interval_modes = rotation_interval_modes

        for axis in range(6):
            if not self.compliance_adaptive_limit_enable[axis]:
                continue
            if not np.isfinite(self.wrench_limits[axis]):
                self.wrench_limits[axis] = 2.0 * self.compliance_desired_wrench[axis]
            self.compliance_adaptive_limit_theta[axis] = type(self.config).compute_theta(
                float(self.wrench_limits[axis]),
                float(self.compliance_desired_wrench[axis]),
                float(self.compliance_adaptive_limit_min[axis]),
            )

    def _get_reference_error_limit(self, axis: int) -> float:
        """Return max allowed reference error for a POS axis from soft wrench budget."""
        if not self.compliance_reference_limit_enable[axis]:
            return np.inf

        f_soft = float(self.wrench_limits[axis])
        if f_soft <= 0.0:
            return 0.0

        kp = float(self.kp[axis])
        if kp <= 0.0:
            return np.inf

        return f_soft / kp

    def _integrate_virtual_target_rotation(self, rotvec_cmd: np.ndarray, dt: float) -> np.ndarray:
        """Update rotational virtual targets in either free SO(3) or constrained XYZ RPY semantics."""
        out = np.asarray(rotvec_cmd, dtype=np.float64).copy()
        target_rpy = np.asarray(self.target[3:6], dtype=np.float64)
        mask_abs_pos = np.array(
            [
                ControlMode(self.control_mode[i]) == ControlMode.POS
                and DeltaMode(self.delta_mode[i]) == DeltaMode.ABSOLUTE
                for i in range(3, 6)
            ],
            dtype=bool,
        )
        mask_delta_pos = np.array(
            [
                ControlMode(self.control_mode[i]) == ControlMode.POS
                and DeltaMode(self.delta_mode[i]) == DeltaMode.RELATIVE
                for i in range(3, 6)
            ],
            dtype=bool,
        )

        if np.any(mask_delta_pos):
            if np.all(mask_delta_pos) and not np.any(mask_abs_pos):
                # Fully relative 3-axis rotation can follow free SO(3) integration.
                omega = np.zeros(3, dtype=np.float64)
                omega[mask_delta_pos] = target_rpy[mask_delta_pos]
                return (R.from_rotvec(omega * dt) * R.from_rotvec(out)).as_rotvec()

            # Mixed/partial rotational targets use the controller's constrained
            # XYZ Euler chart so absolute axes stay locked while relative axes
            # integrate in the same semantics exposed at the task-frame API.
            rpy_cmd = wrap_to_pi(rotvec_to_euler_xyz(out).astype(np.float64))
            if np.any(mask_abs_pos):
                rpy_cmd[mask_abs_pos] = target_rpy[mask_abs_pos]
            rpy_cmd[mask_delta_pos] = wrap_to_pi(
                rpy_cmd[mask_delta_pos] + target_rpy[mask_delta_pos] * dt
            )
            return euler_xyz_to_rotvec(rpy_cmd)

        if np.any(mask_abs_pos):
            rpy_cmd = wrap_to_pi(rotvec_to_euler_xyz(out).astype(np.float64))
            rpy_cmd[mask_abs_pos] = target_rpy[mask_abs_pos]
            return euler_xyz_to_rotvec(rpy_cmd)

        return out

    def _clamp_virtual_target_error_task(self, x_cmd: np.ndarray, pose_F: np.ndarray) -> np.ndarray:
        """Clamp stored task-space virtual target error relative to current pose.

        Translation is clipped directly per axis.
        Rotation is clipped in wrapped RPY coordinates, consistent with the controller's
        constrained-orientation interface.
        """
        out = np.asarray(x_cmd, dtype=np.float64).copy()

        if not np.any(self.compliance_reference_limit_enable):
            return out

        # --- translation ---
        for i in range(3):
            if not (
                self.compliance_reference_limit_enable[i]
                and ControlMode(self.control_mode[i]) == ControlMode.POS
                and DeltaMode(self.delta_mode[i]) == DeltaMode.RELATIVE
            ):
                continue

            e_max = self._get_reference_error_limit(i)
            if not np.isfinite(e_max):
                continue

            err = float(out[i] - pose_F[i])
            out[i] = pose_F[i] + np.clip(err, -e_max, e_max)

        # --- rotation: clip relative stored error in wrapped RPY ---
        rot_axes = [
            i for i in range(3, 6)
            if (
                self.compliance_reference_limit_enable[i]
                and ControlMode(self.control_mode[i]) == ControlMode.POS
                and DeltaMode(self.delta_mode[i]) == DeltaMode.RELATIVE
            )
        ]
        if rot_axes:
            cmd_rpy = wrap_to_pi(rotvec_to_euler_xyz(out[3:6]).astype(np.float64))
            pose_rpy = wrap_to_pi(rotvec_to_euler_xyz(pose_F[3:6]).astype(np.float64))

            rpy_err = wrap_to_pi(cmd_rpy - pose_rpy)
            for axis in rot_axes:
                j = axis - 3
                e_max = self._get_reference_error_limit(axis)
                if not np.isfinite(e_max):
                    continue
                rpy_err[j] = np.clip(rpy_err[j], -e_max, e_max)

            out[3:6] = euler_xyz_to_rotvec(wrap_to_pi(pose_rpy + rpy_err))

        return out

    def _rotation_interval_mode(self, axis: int) -> str:
        """Return the configured rotational bound interpretation for one axis."""
        return RotationIntervalMode(int(self.rotation_interval_modes[axis])).to_name()

    def _clip_rotational_axis(self, angle: float, axis: int) -> float:
        """Clip one wrapped RPY angle according to its configured interval mode."""
        lo = float(self.min_pose[axis])
        hi = float(self.max_pose[axis])
        if self._rotation_interval_mode(axis) == "ccw_arc":
            return float(clip_angle_to_ccw_arc(angle, lo, hi))
        return float(np.clip(angle, lo, hi))


    def _cleanup_rtde(self, rtde_c, rtde_r) -> None:
        """Best-effort shutdown of RTDE force mode, script, and connections."""
        try:
            if self.force_on:
                rtde_c.forceModeStop()
        except Exception:
            pass
        try:
            rtde_c.stopScript()
        except Exception:
            pass
        try:
            rtde_c.disconnect()
        except Exception:
            pass
        try:
            rtde_r.disconnect()
        except Exception:
            pass

    def read_current_state(self, rtde_r):
        """Read world state from RTDE and express pose/twist/wrench in the task frame.

        Args:
            rtde_r: `RTDEReceiveInterface` (or mock) used to query current state.

        Returns:
            dict: ``{'ActualTCPPose','ActualTCPSpeed','ActualTCPForce'}`` in task frame.
        """
        # 1) get the world→frame 4×4
        T = np.linalg.inv(sixvec_to_homogeneous(self.origin))
        R_fw = T[:3, :3]        # rotation: world → frame
        t_fw = T[:3,  3]        # translation: world origin in frame coords

        # 2) pose in world and speed
        pose_W = np.array(rtde_r.getActualTCPPose())   # [x,y,z, Rx,Ry,Rz]
        v_W    = np.array(rtde_r.getActualTCPSpeed())  # [vx,vy,vz, ωx,ωy,ωz]

        # 3) pose in frame
        p_W_h = np.hstack((pose_W[:3], 1.0))
        p_F   = T.dot(p_W_h)[:3]
        R_W   = R.from_rotvec(pose_W[3:6]).as_matrix()
        R_F   = R_fw.dot(R_W)
        rotvec_F = R.from_matrix(R_F).as_rotvec()
        pose_F   = np.concatenate((p_F, rotvec_F))

        # 4) twist in frame
        v_F = np.empty(6)
        v_F[:3]  = R_fw.dot(v_W[:3])
        v_F[3:6] = R_fw.dot(v_W[3:6])

        # 5) wrench in world
        wrench_W = np.array(rtde_r.getActualTCPForce())  # [Fx,Fy,Fz, Mx,My,Mz]
        f_W = wrench_W[:3]
        m_TCP = wrench_W[3:]

        # compute frame origin in world (base) coords
        p_frame = -R_fw.T.dot(t_fw) 

        # TCP position in world coords
        p_TCP = pose_W[:3]

        # vector from TCP to frame origin
        r = p_frame - p_TCP

        # shift the moment from the TCP to your frame origin
        m_frame = m_TCP + np.cross(r, f_W)

        # now express in your frame axes
        f_F = R_fw.dot(f_W)
        m_F = R_fw.dot(m_frame)

        wrench_F = np.concatenate((f_F, m_F))

        return {
            "ActualTCPPose": pose_F,
            "ActualTCPSpeed": v_F,
            "ActualTCPForce": wrench_F
        }

    def clip_pose(self, pose: np.ndarray) -> np.ndarray:
        """Clamp translation per-axis and rotation in RPY space.

        Translation is clipped directly.
        Rotation is clipped per axis either as a standard numeric interval
        ("linear") or as a wrapped CCW arc on S1 ("ccw_arc").
        """
        out = pose.copy()

        # --- translation ---
        out[:3] = np.clip(
            out[:3],
            np.array(self.min_pose[:3]),
            np.array(self.max_pose[:3])
        )

        # --- rotation ---
        rpy = rotvec_to_euler_xyz(out[3:6]).astype(np.float64)
        rpy = wrap_to_pi(rpy)

        for j, axis in enumerate(range(3, 6)):
            rpy[j] = self._clip_rotational_axis(float(rpy[j]), axis)

        out[3:6] = euler_xyz_to_rotvec(rpy)
        return out

    def apply_wrench_bounds(self, pose: np.ndarray, desired_wrench: np.ndarray, measured_wrench: np.ndarray):
        """Contact-aware wrench limiting and boundary protection (in-place).

        Zeroes or scales components that would push the TCP further outside
        position/orientation limits and applies exponential scaling near contact.

        Args:
            pose (np.ndarray): Current task-frame pose (6,).
            desired_wrench (np.ndarray): Computed wrench to be bounded (modified).
            measured_wrench (np.ndarray): Measured task-frame wrench from RTDE.
        """

        scale_vec = np.array([1.0] * 6)
        for i in range(6):
            if not self.compliance_adaptive_limit_enable[i]:
                continue

            f_measured = measured_wrench[i]

            if np.sign(desired_wrench[i]) == np.sign(f_measured):
                f_measured = 0.0

            scale_vec[i] = exp_scale(
                abs(f_measured),
                self.wrench_limits[i],
                self.compliance_adaptive_limit_min[i],
                self.compliance_adaptive_limit_theta[i],
            )

        scaled_wrench_limits = scale_vec * np.array(self.wrench_limits)

        # ----- translation axes -----
        for i in range(3):
            # hard clip wrench
            desired_wrench[i] = np.clip(desired_wrench[i], -scaled_wrench_limits[i], scaled_wrench_limits[i])

            # 2) if outside bounds, project away outward component and add spring back toward bound
            if pose[i] > self.max_pose[i]:
                # remove outward push (positive wrench on + side)
                if desired_wrench[i] > 0.0:
                    desired_wrench[i] = 0.0
                penetration = pose[i] - self.max_pose[i]  # > 0
                desired_wrench[i] += -self.kp[i] * penetration

            elif pose[i] < self.min_pose[i]:
                # remove outward push (negative wrench on - side)
                if desired_wrench[i] < 0.0:
                    desired_wrench[i] = 0.0
                penetration = self.min_pose[i] - pose[i]  # > 0
                desired_wrench[i] += +self.kp[i] * penetration

        # ----- rotation axes (operate in wrapped RPY coordinates) -----
        rpy = wrap_to_pi(rotvec_to_euler_xyz(pose[3:6]).astype(np.float64))
        min_rpy = np.array(self.min_pose[3:6], dtype=np.float64)
        max_rpy = np.array(self.max_pose[3:6], dtype=np.float64)

        for j, i in enumerate(range(3, 6)):
            desired_wrench[i] = np.clip(desired_wrench[i], -scaled_wrench_limits[i], scaled_wrench_limits[i])

            if self._rotation_interval_mode(i) == "ccw_arc":
                correction = signed_error_to_nearest_arc_endpoint(
                    float(rpy[j]),
                    float(min_rpy[j]),
                    float(max_rpy[j]),
                )
                penetration = abs(correction)

                if penetration > 0.0:
                    # If commanded torque points away from the nearest allowed endpoint,
                    # suppress it before adding restoring torque.
                    if desired_wrench[i] * correction < 0.0:
                        desired_wrench[i] = 0.0
                    desired_wrench[i] += self.kp[i] * correction

            else:
                if rpy[j] > max_rpy[j]:
                    if desired_wrench[i] > 0.0:
                        desired_wrench[i] = 0.0
                    penetration = rpy[j] - max_rpy[j]
                    desired_wrench[i] += -self.kp[i] * penetration

                elif rpy[j] < min_rpy[j]:
                    if desired_wrench[i] < 0.0:
                        desired_wrench[i] = 0.0
                    penetration = min_rpy[j] - rpy[j]
                    desired_wrench[i] += +self.kp[i] * penetration

            desired_wrench[i] = np.clip(desired_wrench[i], -scaled_wrench_limits[i], scaled_wrench_limits[i])

    def clip_reference_errors(self, e: float, edot: float, i: int) -> tuple[float, float]:
        """
        Limit position/orientation error e and velocity error edot so that
        kp*e and kd*edot cannot exceed +/- fmax (HIL-SERL style reference limiting).
        """
        _kp = self.kp[i]
        _kd = self.kd[i]
        _fmax = self.compliance_desired_wrench[i]

        if _fmax <= 0:
            return 0.0, 0.0

        if _kp > 0:
            e = float(np.clip(e, -_fmax / _kp, _fmax / _kp))
        if _kd > 0:
            edot = float(np.clip(edot, -_fmax / _kd, _fmax / _kd))
        return e, edot

def _validate_config(config: 'URConfig') -> 'URConfig':
    """Normalize and validate controller configuration.

    Checks frequency range, TCP/payload shapes, instantiates a shared memory
    manager if missing, and enforces simple physical bounds.

    Args:
        config (URConfig): User-provided configuration.

    Returns:
        URConfig: Possibly modified/normalized config.

    Raises:
        AssertionError: On invalid frequency, payload/TCP shapes, or types.
    """
    assert 0 < config.frequency <= 500
    if config.tcp_offset_pose is not None:
        config.tcp_offset_pose = np.array(config.tcp_offset_pose)
        assert config.tcp_offset_pose.shape == (6,)
    if config.payload_mass is not None:
        assert 0 <= config.payload_mass <= 5
    if config.payload_cog is not None:
        config.payload_cog = np.array(config.payload_cog)
        assert config.payload_cog.shape == (3,)
        assert config.payload_mass is not None
    if config.shm_manager is None:
        config.shm_manager = SharedMemoryManager()
        config.shm_manager.start()
    assert isinstance(config.shm_manager, SharedMemoryManager)
    return config
