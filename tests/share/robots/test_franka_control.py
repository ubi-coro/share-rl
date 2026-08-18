from __future__ import annotations

import math
import sys
from types import SimpleNamespace
import time
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from share.cameras import MockCameraConfig
from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    PolicyMode,
    TaskFrame,
)
from share.robots.adaptive_limits import (
    adaptive_wrench_scales,
    reference_error_limit,
)
from share.robots.franka import MockFranka, MockFrankaConfig
from share.robots.franka.lerobot_robot_franka.command import (
    FR3_JOINT_NAMES,
    FrankaTaskFrameCommand,
)
from share.robots.franka.lerobot_robot_franka.config_franka import FrankaConfig
from share.robots.franka.lerobot_robot_franka.control_law import (
    AdaptiveTaskFrameController,
    FrankaState,
    apply_workspace_and_contact_limits,
    nullspace_torque,
    pose_rpy_to_transform,
    smooth_values,
    so3_error,
    transform_wrench,
)
from share.robots.franka.lerobot_robot_franka.controller import (
    FrankaControllerProcess,
    load_franky,
)
from share.robots.franka.lerobot_robot_franka.hand import FrankaHandWorker
from share.robots.ur.lerobot_robot_ur.config_ur import URConfig


def make_state(
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    wrench: np.ndarray | None = None,
) -> FrankaState:
    transform = pose_rpy_to_transform([*position, *rpy])
    jacobian = np.zeros((6, 7), dtype=np.float64)
    jacobian[:, :6] = np.eye(6)
    return FrankaState(
        q=np.zeros(7),
        dq=np.zeros(7),
        T_base_ee=transform,
        T_ee_stiffness=np.eye(4),
        twist_base_ee=np.zeros(6),
        wrench_base_at_stiffness=np.zeros(6) if wrench is None else wrench,
        jacobian_base_ee=jacobian,
        timestamp=0.0,
    )


def command_dict(
    config: FrankaConfig,
    *,
    sequence: int = 1,
    target: list[float] | None = None,
    control_mode: list[ControlMode] | None = None,
    policy_mode: list[PolicyMode] | None = None,
    origin: list[float] | None = None,
    min_pose: list[float] | None = None,
    max_pose: list[float] | None = None,
) -> dict:
    command = FrankaTaskFrameCommand(
        target=target or [0.0] * 6,
        control_mode=control_mode or [ControlMode.POS] * 6,
        policy_mode=policy_mode or [PolicyMode.RELATIVE] * 6,
        origin=origin or [0.0] * 6,
        min_pose=min_pose,
        max_pose=max_pose,
        controller_overrides={
            "kp": list(config.kp),
            "kd": list(config.kd),
            "min_pose": list(min_pose or config.min_pose_rpy),
            "max_pose": list(max_pose or config.max_pose_rpy),
            "rotation_interval_modes": list(config.rotation_interval_modes),
            "wrench_limits": list(config.wrench_limits),
            "compliance_reference_limit_enable": list(
                config.compliance_reference_limit_enable
            ),
            "compliance_adaptive_limit_enable": list(
                config.compliance_adaptive_limit_enable
            ),
            "compliance_desired_wrench": list(config.compliance_desired_wrench),
            "compliance_adaptive_limit_min": list(
                config.compliance_adaptive_limit_min
            ),
            "nullspace_stiffness": list(config.nullspace_stiffness),
            "nullspace_damping": list(config.nullspace_damping),
            "nullspace_max_torque": config.nullspace_max_torque,
            "joint_stiffness": list(config.joint_stiffness),
            "joint_damping": list(config.joint_damping),
            "joint_error_clip": list(config.joint_error_clip),
        },
    )
    return command.to_queue_dict(sequence=sequence, timestamp=0.0)


def test_adaptive_helpers_match_ur_semantics() -> None:
    franka_theta = FrankaConfig.compute_theta(30.0, 5.0, 0.1)
    ur_theta = URConfig.compute_theta(30.0, 5.0, 0.1)
    assert franka_theta == pytest.approx(ur_theta)

    desired = np.array([10.0, -10.0])
    measured = np.array([-5.0, -5.0])
    scales = adaptive_wrench_scales(
        desired,
        measured,
        np.array([True, True]),
        np.array([0.1, 0.1]),
        np.array([franka_theta, franka_theta]),
    )
    assert scales[0] < 1.0
    assert scales[1] == 1.0
    assert reference_error_limit(30.0, 500.0, True) == pytest.approx(0.06)


def test_wrench_reference_point_shift_and_inverse() -> None:
    rotation_target_source = Rotation.from_euler("z", 0.4).as_matrix()
    offset = np.array([0.3, -0.2, 0.1])
    source = np.array([1.0, 2.0, -0.5, 0.2, 0.1, -0.4])
    target = transform_wrench(source, rotation_target_source, offset)
    recovered = transform_wrench(
        target,
        rotation_target_source.T,
        -rotation_target_source.T @ offset,
    )
    np.testing.assert_allclose(recovered, source, atol=1e-12)

    shifted = transform_wrench(
        np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
        np.eye(3),
        np.array([1.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(shifted[3:], [0.0, 0.0, 2.0])


def test_so3_error_uses_log_map() -> None:
    desired = Rotation.from_euler("xyz", [0.3, -0.2, 0.4]).as_matrix()
    actual = Rotation.from_euler("xyz", [-0.1, 0.2, 0.1]).as_matrix()
    error = so3_error(desired, actual)
    reconstructed = Rotation.from_rotvec(error).as_matrix() @ actual
    np.testing.assert_allclose(reconstructed, desired, atol=1e-12)


def test_nullspace_projection_and_torque_cap() -> None:
    rng = np.random.default_rng(4)
    jacobian = rng.normal(size=(6, 7))
    torque = nullspace_torque(
        jacobian,
        np.zeros(7),
        np.zeros(7),
        np.ones(7),
        np.full(7, 20.0),
        np.full(7, 2.0 * math.sqrt(20.0)),
        5.0,
    )
    assert np.max(np.abs(torque)) <= 5.0
    np.testing.assert_allclose(jacobian @ torque, np.zeros(6), atol=1e-10)


def test_workspace_suppresses_outward_wrench_and_restores_inward() -> None:
    bounded, scales, _ = apply_workspace_and_contact_limits(
        pose_rpy=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        desired_wrench=np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        measured_wrench=np.zeros(6),
        stiffness=np.full(6, 10.0),
        min_pose=np.full(6, -0.5),
        max_pose=np.full(6, 0.5),
        rotation_interval_modes=np.zeros(6, dtype=np.int8),
        wrench_limits=np.full(6, 100.0),
        adaptive_enable=np.zeros(6, dtype=bool),
        adaptive_minimum=np.full(6, 0.1),
        adaptive_theta=np.ones(6),
    )
    assert bounded[0] == pytest.approx(-5.0)
    np.testing.assert_allclose(scales, np.ones(6))



def test_ccw_arc_workspace_recovers_across_wrapped_boundary() -> None:
    min_pose = np.full(6, -np.inf)
    max_pose = np.full(6, np.inf)
    min_pose[5] = 2.5
    max_pose[5] = -2.5
    rotation_modes = np.zeros(6, dtype=np.int8)
    rotation_modes[5] = 1
    bounded, _, _ = apply_workspace_and_contact_limits(
        pose_rpy=np.zeros(6),
        desired_wrench=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
        measured_wrench=np.zeros(6),
        stiffness=np.full(6, 10.0),
        min_pose=min_pose,
        max_pose=max_pose,
        rotation_interval_modes=rotation_modes,
        wrench_limits=np.full(6, 100.0),
        adaptive_enable=np.zeros(6, dtype=bool),
        adaptive_minimum=np.full(6, 0.1),
        adaptive_theta=np.ones(6),
    )
    assert bounded[5] == pytest.approx(25.0)
def test_gain_smoothing_has_configured_time_constant() -> None:
    result = smooth_values(np.zeros(2), np.ones(2), 0.1, 0.1)
    np.testing.assert_allclose(result, np.full(2, 1.0 - math.exp(-1.0)))


def test_relative_reference_limit_prevents_windup() -> None:
    config = FrankaConfig(
        robot_ip="mock",
        kp=[100.0] * 6,
        kd=[0.0] * 6,
        wrench_limits=[10.0] * 6,
        compliance_reference_limit_enable=[True] * 6,
        gains_time_constant_s=0.0,
    )
    strategy = AdaptiveTaskFrameController(config)
    command = command_dict(config, target=[100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    output = strategy.step(make_state(), command, model=None, dt=0.01)
    assert strategy.virtual_position[0] == pytest.approx(0.1)
    assert output.desired_wrench_task[0] == pytest.approx(10.0)


def test_mixed_axis_control_law() -> None:
    config = FrankaConfig(
        robot_ip="mock",
        kp=[100.0] * 6,
        kd=[2.0] * 6,
        wrench_limits=[100.0] * 6,
        gains_time_constant_s=0.0,
    )
    strategy = AdaptiveTaskFrameController(config)
    command = command_dict(
        config,
        target=[0.1, 0.2, 3.0, 0.0, 0.0, 0.0],
        control_mode=[
            ControlMode.POS,
            ControlMode.VEL,
            ControlMode.WRENCH,
            ControlMode.POS,
            ControlMode.POS,
            ControlMode.POS,
        ],
        policy_mode=[PolicyMode.ABSOLUTE] * 6,
    )
    output = strategy.step(make_state(), command, model=None, dt=0.002)
    assert output.desired_wrench_task[:3] == pytest.approx([10.0, 0.4, 3.0])


def test_origin_change_preserves_physical_virtual_target() -> None:
    config = FrankaConfig(robot_ip="mock", gains_time_constant_s=0.0)
    strategy = AdaptiveTaskFrameController(config)
    state = make_state(position=(0.5, 0.0, 0.0))
    first = command_dict(config, sequence=1)
    strategy.step(state, first, model=None, dt=0.002)
    virtual_base_before = (
        strategy.T_base_task @ strategy._virtual_transform()
    )
    second = command_dict(
        config,
        sequence=2,
        origin=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    output = strategy.step(state, second, model=None, dt=0.002)
    virtual_base_after = strategy.T_base_task @ strategy._virtual_transform()
    np.testing.assert_allclose(virtual_base_after, virtual_base_before)
    np.testing.assert_allclose(output.desired_wrench_task, np.zeros(6), atol=1e-10)


def test_stale_hold_and_fresh_recovery() -> None:
    config = FrankaConfig(robot_ip="mock", gains_time_constant_s=0.0)
    strategy = AdaptiveTaskFrameController(config)
    state = make_state(position=(0.2, 0.0, 0.0))
    command = command_dict(config, sequence=7)
    strategy.step(state, command, model=None, dt=0.002)
    held = strategy.step(state, None, model=None, dt=0.002)
    assert held.holding
    recovered = strategy.step(state, command, model=None, dt=0.002)
    assert not recovered.holding


def test_command_snapshot_and_mock_franka_schema() -> None:
    joint_frame = FrankaTaskFrameCommand(
        target=[0.0] * 7,
        space=ControlSpace.JOINT,
        policy_mode=[PolicyMode.ABSOLUTE] * 7,
        control_mode=[ControlMode.POS] * 7,
        origin=None,
        joint_names=list(FR3_JOINT_NAMES),
    )
    snapshot = joint_frame.to_queue_dict(sequence=3, timestamp=4.0)
    assert snapshot["target"].shape == (7,)
    assert int(snapshot["width"]) == 7
    assert snapshot["sequence"] == 3

    robot = MockFranka(
        MockFrankaConfig(
            use_gripper=True,
            cameras={"wrist": MockCameraConfig(width=4, height=3, fps=30)},
        )
    )
    assert "fr3_joint7.pos" in robot.observation_features
    assert "gripper.pos" in robot.observation_features
    robot.connect()
    robot.set_task_frame(joint_frame)
    action = {"fr3_joint7.pos": 0.25, "gripper.pos": 1.0}
    assert robot.send_action(action) == action
    observation = robot.get_observation()
    assert observation["fr3_joint7.pos"] == pytest.approx(0.25)
    assert observation["gripper.pos"] == pytest.approx(1.0)
    assert observation["wrist"].shape == (3, 4, 3)
    with pytest.raises(ValueError, match="switching control space"):
        robot.send_action({"x.ee_pos": 0.0})
    robot.disconnect()

    partial = MockFranka(MockFrankaConfig())
    partial._q = np.arange(7, dtype=np.float64)
    partial.connect()
    partial.send_action({"fr3_joint7.pos": 0.25})
    partial_observation = partial.get_observation()
    assert partial_observation["fr3_joint1.pos"] == pytest.approx(0.0)
    assert partial_observation["fr3_joint6.pos"] == pytest.approx(5.0)
    partial.disconnect()



def test_hand_uses_share_open_closed_convention() -> None:
    worker = FrankaHandWorker(SimpleNamespace())
    worker.max_width.value = 0.08
    worker.width.value = 0.08
    assert worker.get_state()["position"] == pytest.approx(0.0)
    worker.width.value = 0.0
    assert worker.get_state()["position"] == pytest.approx(1.0)
    worker.move(2.0)
    assert worker.command_queue.get(timeout=0.2) == pytest.approx(1.0)
    worker.command_queue.close()
    worker.error_queue.close()


class _FakeFrame:
    EndEffector = object()


class _FakeRealtimeConfig:
    Enforce = object()
    Ignore = object()


class _FakeJointReference:
    def __init__(self, q):
        self.q = np.asarray(q)


class _FakeJointImpedanceGains:
    def __init__(self, stiffness, damping):
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)


class _FakeMotion:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.reference = None
        self.torque = np.zeros(7)

    def set_reference(self, reference):
        self.reference = reference

    def set_gains(self, gains):
        self.gains = gains

    def set_torque(self, torque):
        self.torque = np.asarray(torque)



class _FakeTorqueMotion(_FakeMotion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.torque_published = False

    def set_torque(self, torque):
        super().set_torque(torque)
        self.torque_published = True

class _FakeTorqueStopMotion(_FakeMotion):
    pass


class _FakeModel:
    def zero_jacobian(self, frame, state):
        del frame, state
        jacobian = np.zeros((6, 7))
        jacobian[:, :6] = np.eye(6)
        return jacobian


class _FakeRawState:
    def __init__(self):
        self.q = np.zeros(7)
        self.dq = np.zeros(7)
        self.O_T_EE = np.eye(4)
        self.EE_T_K = np.eye(4)
        self.O_dP_EE_est = np.zeros(6)
        self.O_dP_EE_d = np.zeros(6)
        self.O_F_ext_hat_K = np.array([1.0, -2.0, 3.0, 0.1, -0.2, 0.3])
        self.control_command_success_rate = 1.0


class _FakeRobot:
    def __init__(self, hostname, **kwargs):
        self.hostname = hostname
        self.kwargs = kwargs
        self.model = _FakeModel()
        self.state = _FakeRawState()
        self.current_motion = None
        self.is_in_control = False

    def set_collision_behavior(self, *args):
        self.collision_args = args

    def set_ee(self, transform):
        self.ee = transform

    def set_load(self, mass, center, inertia):
        self.load = (mass, center, inertia)

    def move(self, motion, asynchronous=False):
        del asynchronous
        self.current_motion = motion
        self.is_in_control = not isinstance(motion, _FakeTorqueStopMotion)

    def poll_motion(self):
        if isinstance(self.current_motion, _FakeTorqueMotion):
            if not self.current_motion.torque_published:
                raise RuntimeError("torque was not published")
        return False

    def stop(self):
        self.is_in_control = False


class _FailingFakeRobot(_FakeRobot):
    def poll_motion(self):
        raise RuntimeError("simulated torque watchdog")


def _fake_franky_module(robot_class: type[_FakeRobot] = _FakeRobot) -> SimpleNamespace:
    return SimpleNamespace(
        __version__="2.0.0",
        Frame=_FakeFrame,
        RealtimeConfig=_FakeRealtimeConfig,
        Robot=robot_class,
        JointReference=_FakeJointReference,
        JointImpedanceGains=_FakeJointImpedanceGains,
        JointImpedanceTrackingMotion=_FakeMotion,
        SimpleTorqueMotion=_FakeTorqueMotion,
        TorqueStopMotion=_FakeTorqueStopMotion,
    )


def _wait_for_state(
    controller: FrankaControllerProcess,
    predicate,
    timeout: float = 2.0,
) -> dict:
    deadline = time.monotonic() + timeout
    latest = None
    while time.monotonic() < deadline:
        if controller.robot_out_rb.count:
            latest = controller.get_robot_state()
            if predicate(latest):
                return latest
        time.sleep(0.005)
    raise AssertionError(f"Timed out waiting for controller state; latest={latest}")


class _RawTorqueStrategy:
    def zero_wrench(self, state):
        del state

    def step(self, state, command, model, dt):
        del state, command, model, dt
        return np.zeros(7, dtype=np.float64)


class _RawTorqueControllerConfig:
    def make_strategy(self, robot_config):
        del robot_config
        return _RawTorqueStrategy()


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="The fake worker test relies on Linux fork semantics",
)
def test_controller_process_with_fake_franky(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "franky", _fake_franky_module())
    shared_memory = SharedMemoryManager()
    shared_memory.start()
    config = FrankaConfig(
        robot_ip="fake",
        controller=_RawTorqueControllerConfig(),
        shm_manager=shared_memory,
        rt_core=None,
        enforce_realtime=False,
        launch_timeout=2.0,
        command_timeout_s=0.1,
        gains_time_constant_s=0.0,
    )
    controller = FrankaControllerProcess(config)
    try:
        controller.start()
        idle = _wait_for_state(controller, lambda value: bool(value["holding"]))
        assert idle["ActualQ"].shape == (7,)

        controller.send_cmd(controller._default_command())
        active = _wait_for_state(
            controller,
            lambda value: int(value["command_sequence"]) == 1
            and not bool(value["holding"]),
        )
        assert active["control_command_success_rate"] == pytest.approx(1.0)

        controller.zero_ft()
        zeroed = _wait_for_state(
            controller,
            lambda value: np.linalg.norm(value["ActualTCPForce"]) < 1e-12,
        )
        np.testing.assert_allclose(zeroed["ActualTCPForce"], np.zeros(6))

        stale = _wait_for_state(controller, lambda value: bool(value["holding"]))
        assert stale["command_sequence"] == 1

        controller.send_cmd(controller._default_command())
        recovered = _wait_for_state(
            controller,
            lambda value: int(value["command_sequence"]) == 3
            and not bool(value["holding"]),
        )
        assert float(recovered["loop_duration_s"]) >= 0.0
    finally:
        controller.stop()
        shared_memory.shutdown()
    assert not controller.unexpected_exit_event.is_set()



@pytest.mark.skipif(
    sys.platform != "linux",
    reason="The fake worker test relies on Linux fork semantics",
)
def test_controller_propagates_motion_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "franky",
        _fake_franky_module(_FailingFakeRobot),
    )
    shared_memory = SharedMemoryManager()
    shared_memory.start()
    config = FrankaConfig(
        robot_ip="fake",
        shm_manager=shared_memory,
        rt_core=None,
        enforce_realtime=False,
        launch_timeout=2.0,
    )
    controller = FrankaControllerProcess(config)
    try:
        controller.start()
        assert controller.unexpected_exit_event.wait(timeout=2.0)
        time.sleep(0.05)
        with pytest.raises(RuntimeError, match="simulated torque watchdog"):
            controller.check_health()
    finally:
        controller.stop()
        shared_memory.shutdown()


def test_franky_major_version_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "franky", SimpleNamespace(__version__="1.2.3"))
    with pytest.raises(RuntimeError, match="requires.*2.x"):
        load_franky()
