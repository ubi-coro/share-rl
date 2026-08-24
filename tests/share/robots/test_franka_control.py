from __future__ import annotations

import sys
import time
from multiprocessing.managers import SharedMemoryManager
from types import SimpleNamespace

import numpy as np
import pytest

from share.cameras import MockCameraConfig
from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    PolicyMode,
)
from share.robots.adaptive_limits import reference_error_limit
from share.robots.franka import MockFranka, MockFrankaConfig
from share.robots.franka.lerobot_robot_franka.command import (
    FR3_JOINT_NAMES,
    FrankaTaskFrameCommand,
)
from share.robots.franka.lerobot_robot_franka.config_franka import FrankaConfig
from share.robots.franka.lerobot_robot_franka.control_law import (
    CartesianReferenceController,
    FrankaState,
    clip_pose_to_workspace,
    pose_rpy_to_transform,
    so3_error,
    transform_wrench,
)
from share.robots.franka.lerobot_robot_franka.controller import (
    FrankaControllerProcess,
    load_franky,
    normalize_franky_state,
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
    return FrankaState(
        q=np.zeros(7),
        dq=np.zeros(7),
        T_base_ee=transform,
        T_ee_stiffness=np.eye(4),
        twist_base_ee=np.zeros(6),
        wrench_base_at_stiffness=np.zeros(6) if wrench is None else wrench,
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
            "translational_stiffness": config.translational_stiffness,
            "rotational_stiffness": config.rotational_stiffness,
            "min_pose": list(min_pose or config.min_pose_rpy),
            "max_pose": list(max_pose or config.max_pose_rpy),
            "rotation_interval_modes": list(config.rotation_interval_modes),
            "compliance_reference_limit_enable": list(
                config.compliance_reference_limit_enable
            ),
            "joint_stiffness": list(config.joint_stiffness),
            "joint_damping": list(config.joint_damping),
            "joint_error_clip": list(config.joint_error_clip),
        },
    )
    return command.to_queue_dict(sequence=sequence, timestamp=0.0)


def test_adaptive_helper_matches_ur_semantics() -> None:
    franka_theta = FrankaConfig.compute_theta(30.0, 5.0, 0.1)
    ur_theta = URConfig.compute_theta(30.0, 5.0, 0.1)
    assert franka_theta == pytest.approx(ur_theta)
    assert reference_error_limit(30.0, 500.0, True) == pytest.approx(0.06)


def test_wrench_reference_point_shift_and_inverse() -> None:
    from scipy.spatial.transform import Rotation

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


def test_so3_error_uses_log_map() -> None:
    from scipy.spatial.transform import Rotation

    desired = Rotation.from_euler("xyz", [0.3, -0.2, 0.4]).as_matrix()
    actual = Rotation.from_euler("xyz", [-0.1, 0.2, 0.1]).as_matrix()
    error = so3_error(desired, actual)
    reconstructed = Rotation.from_rotvec(error).as_matrix() @ actual
    np.testing.assert_allclose(reconstructed, desired, atol=1e-12)


def test_workspace_clip_bounds_translation_and_linear_rotation() -> None:
    clipped = clip_pose_to_workspace(
        pose_rpy=np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
        min_pose=np.array([-0.5, -0.5, -np.inf, -np.inf, -np.inf, -np.inf]),
        max_pose=np.array([0.5, 0.5, np.inf, np.inf, np.inf, np.inf]),
        rotation_interval_modes=np.zeros(6, dtype=np.int8),
    )
    assert clipped[0] == pytest.approx(0.5)
    assert clipped[1] == pytest.approx(-0.5)


def test_workspace_clip_ccw_arc_projects_to_nearest_endpoint() -> None:
    min_pose = np.full(6, -np.inf)
    max_pose = np.full(6, np.inf)
    min_pose[5] = 2.5
    max_pose[5] = -2.5
    rotation_modes = np.zeros(6, dtype=np.int8)
    rotation_modes[5] = 1  # ccw_arc
    clipped = clip_pose_to_workspace(
        pose_rpy=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        min_pose=min_pose,
        max_pose=max_pose,
        rotation_interval_modes=rotation_modes,
    )
    assert clipped[5] == pytest.approx(2.5) or clipped[5] == pytest.approx(-2.5)


def test_relative_translation_is_integrated_as_velocity() -> None:
    config = FrankaConfig(robot_ip="mock", gains_time_constant_s=0.0)
    strategy = CartesianReferenceController(config)
    command = command_dict(config, target=[0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    output = strategy.step(make_state(), command, dt=0.1)
    assert output.target_pose_task_rpy[0] == pytest.approx(0.005)


def test_mixed_relative_and_absolute_axes_use_proper_so3_composition() -> None:
    config = FrankaConfig(robot_ip="mock", gains_time_constant_s=0.0)
    strategy = CartesianReferenceController(config)
    command = command_dict(
        config,
        target=[0.02, 0.0, 0.0, 0.3, 0.1, 0.0],
        policy_mode=[
            PolicyMode.RELATIVE,
            PolicyMode.RELATIVE,
            PolicyMode.RELATIVE,
            PolicyMode.ABSOLUTE,
            PolicyMode.RELATIVE,
            PolicyMode.RELATIVE,
        ],
    )
    output = strategy.step(make_state(), command, dt=0.1)
    assert output.target_pose_task_rpy[0] == pytest.approx(0.002)
    assert output.target_pose_task_rpy[3] == pytest.approx(0.3)


def test_relative_reference_limit_prevents_windup() -> None:
    config = FrankaConfig(
        robot_ip="mock",
        translational_stiffness=100.0,
        force_constraints=[10.0] * 3 + [1.0] * 3,
        compliance_reference_limit_enable=[True] * 6,
        gains_time_constant_s=0.0,
    )
    strategy = CartesianReferenceController(config)
    command = command_dict(config, target=[100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    strategy.step(make_state(), command, dt=0.01)
    # force_constraints[0] / translational_stiffness == 10.0 / 100.0
    assert strategy.virtual_position[0] == pytest.approx(0.1)


def test_origin_change_preserves_physical_virtual_target() -> None:
    config = FrankaConfig(robot_ip="mock", gains_time_constant_s=0.0)
    strategy = CartesianReferenceController(config)
    state = make_state(position=(0.5, 0.0, 0.0))
    first = command_dict(config, sequence=1)
    strategy.step(state, first, dt=0.002)
    virtual_base_before = strategy.T_base_task @ strategy._virtual_transform()

    second = command_dict(config, sequence=2, origin=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    output = strategy.step(state, second, dt=0.002)
    virtual_base_after = strategy.T_base_task @ strategy._virtual_transform()
    np.testing.assert_allclose(virtual_base_after, virtual_base_before)
    # The virtual target didn't move physically (0.5 in base frame); with the
    # origin now at 0.1, that's 0.4 expressed in the new task frame.
    assert output.target_pose_task_rpy[0] == pytest.approx(0.4)


def test_stale_hold_and_fresh_recovery() -> None:
    config = FrankaConfig(robot_ip="mock", gains_time_constant_s=0.0)
    strategy = CartesianReferenceController(config)
    state = make_state(position=(0.2, 0.0, 0.0))
    command = command_dict(config, sequence=7)
    strategy.step(state, command, dt=0.002)
    held = strategy.step(state, None, dt=0.002)
    assert held.holding
    recovered = strategy.step(state, command, dt=0.002)
    assert not recovered.holding


def test_task_space_is_position_only() -> None:
    with pytest.raises(ValueError, match="position-only"):
        FrankaTaskFrameCommand(
            target=[0.0] * 6,
            control_mode=[ControlMode.VEL] * 6,
            policy_mode=[PolicyMode.ABSOLUTE] * 6,
        ).to_queue_dict()


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


def test_mock_franka_task_space_rejects_vel_and_wrench() -> None:
    robot = MockFranka(MockFrankaConfig())
    robot.connect()
    with pytest.raises(ValueError, match="position-only"):
        robot.send_action({"x.ee_vel": 0.1})
    robot.disconnect()


def test_min_max_pose_top_level_kwarg_is_respected_without_override_key() -> None:
    """Regression test: workspace bounds set via TaskFrame(min_pose=...) must
    survive to_queue_dict() even when controller_overrides carries no
    min_pose/max_pose key of its own (the franka_first_motion.py pattern)."""
    command = FrankaTaskFrameCommand(
        target=[0.0] * 6,
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[PolicyMode.RELATIVE] * 6,
        origin=[0.0] * 6,
        min_pose=[-0.1, -0.1, -0.1, -3.14, -3.14, -3.14],
        max_pose=[0.1, 0.1, 0.1, 3.14, 3.14, 3.14],
        controller_overrides={"translational_stiffness": 200.0, "rotational_stiffness": 20.0},
    )
    queued = command.to_queue_dict()
    np.testing.assert_allclose(queued["min_pose"], [-0.1] * 3 + [-3.14] * 3)
    np.testing.assert_allclose(queued["max_pose"], [0.1] * 3 + [3.14] * 3)


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


class _FakeAffine:
    def __init__(self, matrix=None, translation=None, quaternion=None):
        if matrix is not None:
            self.matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        else:
            self.matrix = np.eye(4)


class _FakeTwist:
    def __init__(self, linear=None, angular=None):
        self.linear = np.zeros(3) if linear is None else np.asarray(linear)
        self.angular = np.zeros(3) if angular is None else np.asarray(angular)


class _FakeRealtimeConfig:
    Enforce = object()
    Ignore = object()


class _FakeReferenceType:
    Absolute = 0
    Relative = 1


class _FakeCartesianReference:
    def __init__(self, target):
        self.target = target


class _FakeCartesianImpedanceGains:
    def __init__(self, translational_stiffness=500, rotational_stiffness=50):
        self.translational_stiffness = translational_stiffness
        self.rotational_stiffness = rotational_stiffness

    @staticmethod
    def isotropic(translational_stiffness, rotational_stiffness, **kwargs):
        return _FakeCartesianImpedanceGains(translational_stiffness, rotational_stiffness)


class _FakePostureTask:
    def __init__(self, target, stiffness, damping=None, max_torque=None):
        self.target = np.asarray(target)
        self.stiffness = np.asarray(stiffness)
        self.max_torque = max_torque


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
        self.gains = None

    def set_reference(self, reference):
        self.reference = reference

    def set_gains(self, gains):
        self.gains = gains


class _FakeCartesianImpedanceTrackingMotion(_FakeMotion):
    pass


class _FakeJointImpedanceTrackingMotion(_FakeMotion):
    pass


class _FakeTorqueStopMotion(_FakeMotion):
    pass


class _FakeModel:
    pass


class _FakeRawState:
    def __init__(self):
        self.q = np.zeros(7)
        self.dq = np.zeros(7)
        self.O_T_EE = list(np.eye(4).flatten(order="F"))
        self.EE_T_K = _FakeAffine(np.eye(4))
        self.O_dP_EE_est = _FakeTwist()
        self.O_dP_EE_c = _FakeTwist()
        self.O_dP_EE_d = _FakeTwist()
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
        return False

    def stop(self):
        self.is_in_control = False


class _FailingFakeRobot(_FakeRobot):
    def poll_motion(self):
        raise RuntimeError("simulated torque watchdog")


def _fake_franky_module(robot_class: type[_FakeRobot] = _FakeRobot) -> SimpleNamespace:
    return SimpleNamespace(
        __version__="2.0.0",
        RealtimeConfig=_FakeRealtimeConfig,
        ReferenceType=_FakeReferenceType,
        Robot=robot_class,
        Affine=_FakeAffine,
        CartesianReference=_FakeCartesianReference,
        CartesianImpedanceGains=_FakeCartesianImpedanceGains,
        CartesianImpedanceTrackingMotion=_FakeCartesianImpedanceTrackingMotion,
        PostureTask=_FakePostureTask,
        JointReference=_FakeJointReference,
        JointImpedanceGains=_FakeJointImpedanceGains,
        JointImpedanceTrackingMotion=_FakeJointImpedanceTrackingMotion,
        TorqueStopMotion=_FakeTorqueStopMotion,
    )


def test_normalize_franky_state_handles_column_major_O_T_EE_and_typed_accessors() -> None:
    """Regression test: O_T_EE is libfranka's flat *column-major* 16-vector
    (a plain .reshape(4, 4) silently returns the transpose), and EE_T_K /
    O_dP_EE_* are typed Franky objects (Affine / Twist), not raw arrays."""
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_euler("xyz", [0, 0, 0.5]).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [1.0, 2.0, 3.0]

    raw_state = SimpleNamespace(
        q=np.zeros(7),
        dq=np.zeros(7),
        O_T_EE=list(transform.flatten(order="F")),
        EE_T_K=_FakeAffine(np.eye(4)),
        O_dP_EE_est=_FakeTwist(linear=[0.1, 0.2, 0.3], angular=[0.01, 0.02, 0.03]),
        O_dP_EE_c=_FakeTwist(),
        O_F_ext_hat_K=np.zeros(6),
    )
    state = normalize_franky_state(raw_state)
    np.testing.assert_allclose(state.T_base_ee, transform, atol=1e-12)
    np.testing.assert_allclose(state.twist_base_ee, [0.1, 0.2, 0.3, 0.01, 0.02, 0.03])


def test_configure_robot_passes_flat_sequences_to_set_ee_and_set_load() -> None:
    """Regression test: set_ee/set_load want flat length-16/length-9
    sequences, not (4, 4)/(3, 3) arrays."""
    config = FrankaConfig(
        robot_ip="mock",
        end_effector_transform=list(np.eye(4).flatten()),
        payload_mass=1.2,
        payload_center_of_mass=[0.0, 0.0, 0.05],
        payload_inertia=list(np.eye(3).flatten()),
    )
    process = object.__new__(FrankaControllerProcess)
    process.config = config
    fake_robot = _FakeRobot("mock")
    process._configure_robot(fake_robot)
    assert isinstance(fake_robot.ee, list) and len(fake_robot.ee) == 16
    mass, center, inertia = fake_robot.load
    assert isinstance(center, list) and len(center) == 3
    assert isinstance(inertia, list) and len(inertia) == 9


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
