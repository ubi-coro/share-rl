from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass, field
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

from share.utils.mock_utils import MockRTDEControlInterface, MockRTDEReceiveInterface

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MODULE_PATH = ROOT / "src/share/robots/ur/lerobot_robot_ur/controller.py"
SPEC = importlib.util.spec_from_file_location("tf_controller_v2_test_module", MODULE_PATH)
tf_controller = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(tf_controller)


@dataclass
class _ControllerConfig:
    robot_ip: str
    frequency: float = 100.0
    payload_mass: float | None = None
    payload_cog: list[float] | None = None
    tcp_offset_pose: list[float] | None = None
    soft_real_time: bool = False
    rt_core: int = 0
    launch_timeout: float = 1.0
    get_max_k: int = 16
    shm_manager: SharedMemoryManager | None = None
    ft_filter_cutoff_hz: float | None = None
    force_mode_gain_scaling: float = 1.0
    default_stiffness_mode: tf_controller.StiffnessMode = tf_controller.StiffnessMode.COMPLIANT
    servo_lookahead_time: float = 0.1
    servo_gain: float = 300.0
    kp: list[float] = field(default_factory=lambda: [2500.0, 2500.0, 2500.0, 150.0, 150.0, 150.0])
    kd: list[float] = field(default_factory=lambda: [80.0, 80.0, 80.0, 8.0, 8.0, 8.0])
    max_pose_rpy: list[float] = field(default_factory=lambda: [float("inf")] * 6)
    min_pose_rpy: list[float] = field(default_factory=lambda: [-float("inf")] * 6)
    wrench_limits: list[float] = field(default_factory=lambda: [30.0] * 6)
    speed_limits: list[float] = field(default_factory=lambda: [1.0] * 6)
    compliance_adaptive_limit_enable: list[bool] = field(default_factory=lambda: [False] * 6)
    compliance_reference_limit_enable: list[bool] = field(default_factory=lambda: [False] * 6)
    compliance_desired_wrench: list[float] = field(default_factory=lambda: [5.0] * 6)
    compliance_adaptive_limit_theta: list[float] = field(default_factory=lambda: [1.0] * 6)
    compliance_adaptive_limit_min: list[float] = field(default_factory=lambda: [0.1] * 6)
    use_degrees: bool = False
    verbose: bool = False
    mock: bool = True
    debug: bool = False
    debug_axis: int = 0

    @staticmethod
    def compute_theta(F_max: float, f_star: float, s_min: float) -> float:
        return max(0.1, F_max + f_star + s_min)


def test_mock_rtde_interfaces_generate_motion_from_force_mode():
    hostname = f"pytest-mock-rtde-{time.time_ns()}"
    rtde_c = MockRTDEControlInterface(hostname, frequency=200.0)
    rtde_r = MockRTDEReceiveInterface(hostname)

    rtde_c.forceModeSetGainScaling(0.5)
    for _ in range(12):
        period = rtde_c.initPeriod()
        rtde_c.forceMode([0.0] * 6, [1, 1, 1, 1, 1, 1], [12.0, 0.0, 0.0, 0.0, 0.0, 1.5], 2, [1.0] * 6)
        rtde_c.waitPeriod(period)

    pose = rtde_r.getActualTCPPose()
    speed = rtde_r.getActualTCPSpeed()
    wrench = rtde_r.getActualTCPForce()

    assert pose[0] > 0.0
    assert abs(pose[5]) > 0.0
    assert speed[0] > 0.0
    assert wrench[0] > 0.0


def test_mock_rtde_interfaces_generate_motion_from_servo_and_torque_modes():
    hostname = f"pytest-mock-rtde-joints-{time.time_ns()}"
    rtde_c = MockRTDEControlInterface(hostname, frequency=200.0)
    rtde_r = MockRTDEReceiveInterface(hostname)

    rtde_c.servoJ([0.3, -0.2, 0.1, 0.0, 0.0, 0.0], 0.5, 0.5, 0.01, 0.1, 300.0)
    for _ in range(8):
        period = rtde_c.initPeriod()
        rtde_c.waitPeriod(period)

    assert rtde_r.getActualQ()[0] > 0.0

    rtde_c.directTorque([2.0, 0.0, 0.0, 0.0, 0.0, 0.0], True)
    for _ in range(8):
        period = rtde_c.initPeriod()
        rtde_c.waitPeriod(period)

    assert rtde_r.getActualQd()[0] > 0.0

    rtde_c.servoL([0.05, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5, 0.5, 0.01, 0.1, 300.0)
    assert rtde_r.getActualTCPPose()[0] >= 0.05


def test_controller_mock_mode_starts_quickly_and_streams_state():
    config = _ControllerConfig(robot_ip=f"pytest-controller-{time.time_ns()}")
    controller = tf_controller.RTDETaskFrameController(config)

    start = time.perf_counter()
    controller.start(wait=True)
    try:
        assert time.perf_counter() - start < 0.5

        cmd = tf_controller.TaskFrameCommand(
            origin=[0.0] * 6,
            target=[10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            control_mode=[tf_controller.ControlMode.WRENCH] + [tf_controller.ControlMode.VEL] * 5,
            policy_mode=[None] * 6,
            max_pose=[float("inf")] * 6,
            min_pose=[-float("inf")] * 6,
        )
        controller.send_cmd(cmd)

        state = controller.get_robot_state()
        deadline = time.time() + 0.5
        while time.time() < deadline and state["ActualTCPPose"][0] <= 0.0:
            time.sleep(0.02)
            state = controller.get_robot_state()

        assert state["ActualTCPPose"][0] > 0.0
        assert state["SetTCPForce"][0] > 0.0
    finally:
        controller.stop(wait=True)
        if config.shm_manager is not None:
            config.shm_manager.shutdown()
