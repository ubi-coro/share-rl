import threading
import time
from dataclasses import field, dataclass

import numpy as np
from lerobot.robots import Robot, RobotConfig
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig
from lerobot.processor import RobotAction, RobotObservation
from scipy.spatial.transform import Rotation as R

from share.envs.manipulation_primitive.task_frame import TaskFrame


class MockRobot(Robot):
    config_class = RobotConfig
    name = "mock_robot"

    def __init__(self, name="mock_robot", is_task_frame=True):
        # Initialize with a dummy config
        cfg = RobotConfig(id=name)
        super().__init__(cfg)
        self.config = cfg
        self._is_task_frame = is_task_frame
        self.current_joints = np.zeros(6)
        # Mock bus for configuration validation (looks for motors dict)
        self.bus = type("MockBus", (), {"motors": {f"joint_{i+1}": None for i in range(6)}})
        self.current_frame = TaskFrame()

    @property
    def observation_features(self) -> dict:
        return {f"joint_{i+1}.pos": float for i in range(6)}

    @property
    def action_features(self) -> dict:
        return {f"joint_{i+1}.pos": float for i in range(6)}

    @property
    def _motors_ft(self):
        return self.observation_features

    @property
    def is_connected(self) -> bool: return True
    @property
    def is_calibrated(self) -> bool: return True

    def connect(self, calibrate: bool = True): pass
    def disconnect(self): pass
    def calibrate(self): pass
    def configure(self): pass

    def get_observation(self) -> RobotObservation:
        return {f"joint_{i+1}.pos": float(self.current_joints[i]) for i in range(6)}

    def send_action(self, action: RobotAction) -> RobotAction:
        for i in range(6):
            key = f"joint_{i+1}.pos"
            if key in action: self.current_joints[i] = action[key]
        return action

    def set_task_frame(self, frame):
        if not self._is_task_frame:
            raise AttributeError("Hardware does not support task frames.")
        self.current_frame = frame


class MockTeleoperator(Teleoperator):
    config_class = TeleoperatorConfig
    name = "mock_teleop"

    def __init__(self, name="mock_teleop", is_delta=True):
        cfg = TeleoperatorConfig(id=name)
        super().__init__(cfg)
        self._is_delta = is_delta
        if is_delta:
            self._features = {f"delta_{ax}": float for ax in ["x", "y", "z", "rx", "ry", "rz"]}
        else:
            self._features = {f"joint_{i+1}.pos": float for i in range(6)}

    @property
    def action_features(self) -> dict: return self._features
    @property
    def is_connected(self) -> bool: return True
    @property
    def is_calibrated(self) -> bool: return True
    @property
    def feedback_features(self) -> dict: return {}
    def connect(self): pass
    def disconnect(self): pass
    def calibrate(self): pass
    def configure(self): pass
    def send_feedback(self, feedback_action): pass
    def get_action(self) -> RobotAction: return {key: 0.25 for key in self._features}

class MockKinematicsSolver:
    """Mock that mimics RobotKinematics but uses simple identity/vector math."""
    def forward_kinematics(self, joints: np.ndarray | dict) -> np.ndarray:
        if isinstance(joints, dict):
            return np.array([joints.get(f"joint_{i+1}.pos", 0.0) for i in range(6)])
        return joints[:6]

    def inverse_kinematics(self, current_joint_pos: np.ndarray, desired_ee_pose: np.ndarray, **kwargs) -> np.ndarray:
        # In mock, pose == joints.
        # Round-trip check: IK(FK(q)) -> q
        return desired_ee_pose



@dataclass
class MockComplexObservationRobot:
    """Robot stub emitting a richer observation dictionary for env-pipeline tests."""

    name: str = "mock_complex_robot"
    joint_names: list[str] = field(default_factory=lambda: ["joint_1", "joint_2", "joint_3"])

    def get_observation(self, prefix: str = "arm") -> dict[str, float]:
        joints = {"joint_1": 0.35, "joint_2": -0.25, "joint_3": 0.55}
        ee_x = 0.5 * joints["joint_1"] + 0.2 * joints["joint_2"] - 0.1 * joints["joint_3"]
        ee_y = -0.3 * joints["joint_1"] + 0.4 * joints["joint_2"] + 0.2 * joints["joint_3"]
        ee_z = joints["joint_1"] + joints["joint_2"] + joints["joint_3"]
        ee_wx = 0.1 * joints["joint_1"]
        ee_wy = -0.05 * joints["joint_2"]
        ee_wz = 0.2 * joints["joint_3"]

        return {
            f"{prefix}.joint_1.pos": joints["joint_1"],
            f"{prefix}.joint_2.pos": joints["joint_2"],
            f"{prefix}.joint_3.pos": joints["joint_3"],
            f"{prefix}.joint_1.vel": 0.03,
            f"{prefix}.joint_2.vel": -0.01,
            f"{prefix}.joint_3.vel": 0.02,
            f"{prefix}.joint_1.current": 0.4,
            f"{prefix}.joint_2.current": 0.2,
            f"{prefix}.joint_3.current": 0.1,
            f"{prefix}.x.ee_pos": ee_x,
            f"{prefix}.y.ee_pos": ee_y,
            f"{prefix}.z.ee_pos": ee_z,
            f"{prefix}.wx.ee_pos": ee_wx,
            f"{prefix}.wy.ee_pos": ee_wy,
            f"{prefix}.wz.ee_pos": ee_wz,
        }


@dataclass
class MockJointOnlyRobot:
    """Minimal robot stub without task-frame capability."""

    name: str = "mock_joint_robot"
    joint_names: list[str] = field(default_factory=lambda: ["joint_1", "joint_2", "joint_3"])


@dataclass
class MockTaskFrameRobot(MockJointOnlyRobot):
    """Minimal robot stub exposing task-frame capability."""

    last_task_frame_command: list[float] | None = None

    def set_task_frame(self, command: list[float]) -> None:
        self.last_task_frame_command = list(command)


@dataclass
class MockDeltaTeleoperator:
    """Delta teleoperator stub (SpaceMouse/keyboard style)."""

    action_features: dict[str, type] = field(
        default_factory=lambda: {
            "delta_x": float,
            "delta_y": float,
            "delta_z": float,
            "delta_rx": float,
            "delta_ry": float,
            "delta_rz": float,
        }
    )


@dataclass
class MockVelocityDeltaTeleoperator:
    """Delta teleoperator stub exposing Cartesian velocity keys directly."""

    action_features: dict[str, type] = field(
        default_factory=lambda: {
            "x.vel": float,
            "y.vel": float,
            "z.vel": float,
            "wx.vel": float,
            "wy.vel": float,
            "wz.vel": float,
        }
    )


@dataclass
class MockKeyboardStyleDeltaTeleoperator:
    """Delta teleoperator stub exposing metadata-style action names."""

    action_features: dict = field(
        default_factory=lambda: {
            "dtype": "float32",
            "shape": (4,),
            "names": {"x.vel": 0, "y.vel": 1, "z.vel": 2, "gripper": 3},
        }
    )


@dataclass
class MockGamepadStyleDeltaTeleoperator:
    """Delta teleoperator stub exposing legacy metadata-style delta names."""

    action_features: dict = field(
        default_factory=lambda: {
            "dtype": "float32",
            "shape": (4,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
        }
    )


@dataclass
class MockPhoneLikeTeleoperator:
    """Special-schema teleoperator stub that should not be treated as delta-like."""

    action_features: dict[str, type] = field(
        default_factory=lambda: {
            "phone.pos": object,
            "phone.rot": object,
            "phone.raw_inputs": dict,
            "phone.enabled": bool,
        }
    )


@dataclass
class MockAbsoluteJointTeleoperator:
    """Absolute-joint teleoperator stub (leader-arm style)."""

    action_features: dict[str, type] = field(
        default_factory=lambda: {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
        }
    )


@dataclass
class MockKinematicsSolver:
    """Deterministic FK/IK mock used by processor-pipeline unit tests."""

    joint_names: list[str] = field(default_factory=lambda: ["joint_1", "joint_2", "joint_3"])

    def forward_kinematics(self, joint_positions: dict[str, float]) -> list[float]:
        """Map joints to a deterministic 6D task-frame pose."""
        x = sum(joint_positions[name] for name in self.joint_names)
        y = joint_positions[self.joint_names[0]]
        z = joint_positions[self.joint_names[-1]]
        rx = 0.1 * x
        ry = 0.1 * y
        rz = 0.1 * z
        return [x, y, z, rx, ry, rz]

    def inverse_kinematics(self, pose: list[float]) -> dict[str, float]:
        """Map a task-frame pose back to deterministic joint targets."""
        x, y, z, _, _, _ = pose
        return {
            self.joint_names[0]: y,
            self.joint_names[1]: x - y - z,
            self.joint_names[2]: z,
        }


@dataclass
class MockComplexKinematicsSolver(MockKinematicsSolver):
    """Kinematics mock with a richer affine mapping for FK/IK tests."""

    def forward_kinematics(self, joint_positions: dict[str, float]) -> list[float]:
        q1 = joint_positions[self.joint_names[0]]
        q2 = joint_positions[self.joint_names[1]]
        q3 = joint_positions[self.joint_names[2]]
        return [
            0.5 * q1 + 0.2 * q2 - 0.1 * q3,
            -0.3 * q1 + 0.4 * q2 + 0.2 * q3,
            q1 + q2 + q3,
            0.1 * q1,
            -0.05 * q2,
            0.2 * q3,
        ]

    def inverse_kinematics(self, pose: list[float]) -> dict[str, float]:
        x, y, z, _, _, _ = pose
        q2 = (5.0 * x + y + 0.1 * z) / 2.7
        q1 = 2.0 * x - 0.4 * q2 + 0.2 * z
        q3 = z - q1 - q2
        return {
            self.joint_names[0]: q1,
            self.joint_names[1]: q2,
            self.joint_names[2]: q3,
        }


@dataclass(slots=True)
class _MockPeriodToken:
    start_time: float
    waited: bool = False


_MOCK_RTDE_STATES: dict[str, dict] = {}
_MOCK_RTDE_STATES_LOCK = threading.Lock()


def _get_or_create_mock_rtde_state(hostname: str, frequency: float | None = None) -> dict:
    with _MOCK_RTDE_STATES_LOCK:
        state = _MOCK_RTDE_STATES.get(hostname)
        if state is None:
            state = {
                "pose": np.zeros(6, dtype=np.float64),
                "vel": np.zeros(6, dtype=np.float64),
                "commanded_wrench": np.zeros(6, dtype=np.float64),
                "measured_wrench": np.zeros(6, dtype=np.float64),
                "ft_bias": np.zeros(6, dtype=np.float64),
                "task_frame": np.zeros(6, dtype=np.float64),
                "selection_vector": np.ones(6, dtype=np.float64),
                "speed_limits": np.full(6, np.inf, dtype=np.float64),
                "mode": "idle",
                "gain_scaling": 1.0,
                "payload_mass": 0.0,
                "payload_cog": np.zeros(3, dtype=np.float64),
                "tcp_offset": np.zeros(6, dtype=np.float64),
                "frequency": float(frequency or 125.0),
                "last_update": time.monotonic(),
                "lock": threading.Lock(),
            }
            _MOCK_RTDE_STATES[hostname] = state
        elif frequency is not None:
            state["frequency"] = float(frequency)
        return state


def _advance_mock_rtde_state(state: dict) -> None:
    now = time.monotonic()
    with state["lock"]:
        dt = max(0.0, min(now - state["last_update"], 0.05))
        if dt <= 0.0:
            return

        if state["mode"] == "force":
            applied_wrench = state["selection_vector"] * state["gain_scaling"] * state["commanded_wrench"]
        else:
            applied_wrench = np.zeros(6, dtype=np.float64)

        # A lightly damped diagonal rigid-body model is enough for controller tests.
        linear_mass = 8.0 + max(float(state["payload_mass"]), 0.0)
        angular_inertia = 1.2 + 0.05 * max(float(state["payload_mass"]), 0.0)
        linear_damping = 5.0
        angular_damping = 2.5

        acc = np.zeros(6, dtype=np.float64)
        acc[:3] = applied_wrench[:3] / linear_mass - linear_damping * state["vel"][:3]
        acc[3:] = applied_wrench[3:] / angular_inertia - angular_damping * state["vel"][3:]

        state["vel"] += acc * dt
        finite_speed_limits = np.where(np.isfinite(state["speed_limits"]), state["speed_limits"], np.inf)
        state["vel"] = np.clip(state["vel"], -finite_speed_limits, finite_speed_limits)
        state["pose"][:3] += state["vel"][:3] * dt
        state["pose"][3:6] = (
            R.from_rotvec(state["vel"][3:6] * dt) * R.from_rotvec(state["pose"][3:6])
        ).as_rotvec()

        alpha = np.clip(12.0 * dt, 0.0, 1.0)
        state["measured_wrench"] += alpha * (applied_wrench - state["measured_wrench"])
        state["last_update"] = now


class MockRTDEControlInterface:
    """Small UR RTDE control mock used by the controller in hardware-free mode."""

    def __init__(self, hostname: str | None = None, frequency: float = 125.0, *args):
        self.hostname = hostname or "mock-ur"
        self.frequency = float(frequency)
        self._state = _get_or_create_mock_rtde_state(self.hostname, self.frequency)

    def initPeriod(self):
        _advance_mock_rtde_state(self._state)
        return _MockPeriodToken(start_time=time.monotonic())

    def waitPeriod(self, t0):
        if isinstance(t0, _MockPeriodToken):
            if t0.waited:
                _advance_mock_rtde_state(self._state)
                return
            deadline = t0.start_time + (1.0 / max(self.frequency, 1e-6))
            t0.waited = True
        else:
            deadline = float(t0) + (1.0 / max(self.frequency, 1e-6))

        remaining = deadline - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)
        _advance_mock_rtde_state(self._state)

    def setTcp(self, tcp_pose):
        with self._state["lock"]:
            self._state["tcp_offset"] = np.asarray(tcp_pose, dtype=np.float64)

    def setPayload(self, mass, cog=None):
        with self._state["lock"]:
            self._state["payload_mass"] = float(mass)
            if cog is not None:
                self._state["payload_cog"] = np.asarray(cog, dtype=np.float64)
        return True

    def moveJ(self, joints, speed, acceleration):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            joints = np.asarray(joints, dtype=np.float64)
            self._state["pose"][3:6] = 0.0
            self._state["vel"][:] = 0.0
            self._state["commanded_wrench"][:] = 0.0
            self._state["measured_wrench"][:] = 0.0
            self._state["mode"] = "idle"
            self._state["pose"][:3] = 0.01 * joints[:3]
        time.sleep(0.02)
        return True

    def servoL(self, pose, vel, acc, dt, lookahead_time, gain):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            self._state["pose"] = np.asarray(pose, dtype=np.float64)
            self._state["vel"][:] = 0.0
            self._state["commanded_wrench"][:] = 0.0
            self._state["measured_wrench"][:] = 0.0
            self._state["mode"] = "idle"
        return True

    def speedL(self, speed6, acc, dt):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            self._state["vel"] = np.asarray(speed6, dtype=np.float64)
            self._state["mode"] = "idle"
            self._state["commanded_wrench"][:] = 0.0
            self._state["measured_wrench"][:] = 0.0
        _advance_mock_rtde_state(self._state)
        return True

    def forceModeSetGainScaling(self, scaling):
        with self._state["lock"]:
            self._state["gain_scaling"] = float(scaling)

    def forceMode(self, *args):
        if len(args) < 2:
            raise TypeError("forceMode expects at least a selection vector and wrench")

        if len(args) >= 5:
            candidate_selection = np.asarray(args[1], dtype=np.float64)
            looks_like_v2_signature = candidate_selection.shape == (6,) and np.all(
                np.isin(candidate_selection, [0.0, 1.0])
            )
        else:
            looks_like_v2_signature = False

        if looks_like_v2_signature:
            task_frame, selection_vector, wrench, _force_type, limits = args[:5]
        else:
            task_frame = np.zeros(6, dtype=np.float64)
            selection_vector = args[0]
            wrench = args[1]
            limits = args[3] if len(args) > 3 else np.full(6, np.inf, dtype=np.float64)

        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            self._state["task_frame"] = np.asarray(task_frame, dtype=np.float64)
            self._state["selection_vector"] = np.asarray(selection_vector, dtype=np.float64)
            self._state["commanded_wrench"] = np.asarray(wrench, dtype=np.float64)
            self._state["speed_limits"] = np.asarray(limits, dtype=np.float64)
            self._state["mode"] = "force"
        return True

    def zeroFtSensor(self):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            self._state["ft_bias"] = self._state["measured_wrench"].copy()

    def forceModeStop(self):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            self._state["mode"] = "idle"
            self._state["commanded_wrench"][:] = 0.0

    def speedStop(self):
        with self._state["lock"]:
            self._state["vel"][:] = 0.0

    def servoStop(self):
        with self._state["lock"]:
            self._state["vel"][:] = 0.0

    def stopScript(self):
        self.forceModeStop()

    def disconnect(self):
        pass


class MockRTDEReceiveInterface:
    """Receive side of the in-process UR RTDE mock."""

    def __init__(self, hostname: str | None = None, *args, **kwargs):
        self.hostname = hostname or "mock-ur"
        self._state = _get_or_create_mock_rtde_state(self.hostname)

    def getActualTCPPose(self):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            return self._state["pose"].copy()

    def getActualTCPSpeed(self):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            return self._state["vel"].copy()

    def getActualTCPForce(self):
        _advance_mock_rtde_state(self._state)
        with self._state["lock"]:
            return (self._state["measured_wrench"] - self._state["ft_bias"]).copy()

    def getActualQ(self):
        return np.zeros(6, dtype=np.float64)

    def getActualQd(self):
        return np.zeros(6, dtype=np.float64)

    def disconnect(self):
        pass
