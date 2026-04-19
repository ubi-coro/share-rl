from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_task_frame_module():
    module_path = ROOT / "src/share/envs/manipulation_primitive/task_frame.py"
    module_name = "share_task_frame_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


task_frame_module = _load_task_frame_module()
ControlMode = task_frame_module.ControlMode
ControlSpace = task_frame_module.ControlSpace
PolicyMode = task_frame_module.PolicyMode
TaskFrame = task_frame_module.TaskFrame


def _load_controller_module():
    share_envs_pkg = types.ModuleType("share.envs")
    share_envs_pkg.__path__ = []
    mp_pkg = types.ModuleType("share.envs.manipulation_primitive")
    mp_pkg.__path__ = []
    shared_memory_module = types.ModuleType("share.utils.shared_memory")

    class _UnusedRingBuffer:
        pass

    class _UnusedQueue:
        pass

    shared_memory_module.SharedMemoryRingBuffer = _UnusedRingBuffer
    shared_memory_module.SharedMemoryQueue = _UnusedQueue
    shared_memory_module.Empty = type("Empty", (Exception,), {})
    sys.modules["share.envs"] = share_envs_pkg
    sys.modules["share.envs.manipulation_primitive"] = mp_pkg
    sys.modules["share.envs.manipulation_primitive.task_frame"] = task_frame_module
    sys.modules["share.utils.shared_memory"] = shared_memory_module

    module_path = ROOT / "src/share/robots/ur/lerobot_robot_ur/controller.py"
    module_name = "share_ur_controller_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_ur_module(controller_module):
    cameras_module = types.ModuleType("lerobot.cameras")
    cameras_module.make_cameras_from_configs = lambda configs: {}

    hil_processor_module = types.ModuleType("lerobot.processor.hil_processor")
    hil_processor_module.GRIPPER_KEY = "gripper"

    robots_module = types.ModuleType("lerobot.robots")
    robots_module.Robot = object

    errors_module = types.ModuleType("lerobot.utils.errors")

    class DeviceNotConnectedError(RuntimeError):
        pass

    class DeviceAlreadyConnectedError(RuntimeError):
        pass

    errors_module.DeviceNotConnectedError = DeviceNotConnectedError
    errors_module.DeviceAlreadyConnectedError = DeviceAlreadyConnectedError

    config_module = types.ModuleType("share.robots.ur.lerobot_robot_ur.config_ur")
    config_module.URConfig = object

    gripper_module = types.ModuleType("share.grippers.robotiq_controller")
    gripper_module.RTDERobotiqController = object

    ur_parent = types.ModuleType("share.robots.ur")
    ur_parent.__path__ = []
    ur_pkg = types.ModuleType("share.robots.ur.lerobot_robot_ur")
    ur_pkg.__path__ = []

    sys.modules.setdefault("lerobot.cameras", cameras_module)
    sys.modules.setdefault("lerobot.processor.hil_processor", hil_processor_module)
    sys.modules.setdefault("lerobot.robots", robots_module)
    sys.modules.setdefault("lerobot.utils.errors", errors_module)
    sys.modules["share.robots.ur"] = ur_parent
    sys.modules["share.robots.ur.lerobot_robot_ur"] = ur_pkg
    sys.modules["share.robots.ur.lerobot_robot_ur.config_ur"] = config_module
    sys.modules["share.robots.ur.lerobot_robot_ur.controller"] = controller_module
    sys.modules["share.grippers.robotiq_controller"] = gripper_module

    module_path = ROOT / "src/share/robots/ur/lerobot_robot_ur/ur.py"
    module_name = "share_ur_robot_test"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _batched_queue_dict(payload: dict) -> dict[str, np.ndarray]:
    batched = {}
    for key, value in payload.items():
        array = np.asarray(value)
        batched[key] = array[None] if array.ndim > 0 else np.asarray([array])
    return batched


class _ReadyController:
    def __init__(self):
        self.is_ready = True
        self.commands = []

    def send_cmd(self, cmd):
        self.commands.append(cmd)


def test_task_frame_command_joint_space_maps_joint_position_keys():
    controller_module = _load_controller_module()
    command = controller_module.TaskFrameCommand(
        space=ControlSpace.JOINT,
        target=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[PolicyMode.ABSOLUTE] * 6,
    )

    assert command.to_robot_action() == {
        "joint_1.pos": 0.1,
        "joint_2.pos": 0.2,
        "joint_3.pos": 0.3,
        "joint_4.pos": 0.4,
        "joint_5.pos": 0.5,
        "joint_6.pos": 0.6,
    }


def test_task_frame_command_converts_origin_rpy_to_internal_rotvec():
    controller_module = _load_controller_module()
    command = controller_module.TaskFrameCommand(
        origin=[0.0, 0.0, 0.0, 0.2, -0.1, 0.3],
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[PolicyMode.ABSOLUTE] * 6,
    )

    queued = command.to_queue_dict()
    expected_rotvec = controller_module.R.from_euler("xyz", [0.2, -0.1, 0.3], degrees=False).as_rotvec()
    np.testing.assert_allclose(queued["origin"][3:6], expected_rotvec)


def test_controller_rejects_task_to_joint_switches():
    controller_module = _load_controller_module()
    controller = object.__new__(controller_module.RTDETaskFrameController)
    controller._active_space = None
    controller._last_cmd = controller_module.TaskFrameCommand()
    controller.robot_cmd_queue = _Queue()

    controller.send_cmd(
        controller_module.TaskFrameCommand(
            space=ControlSpace.TASK,
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[PolicyMode.ABSOLUTE] * 6,
        )
    )

    with pytest.raises(ValueError, match="switching between task-space and joint-space"):
        controller.send_cmd(
            controller_module.TaskFrameCommand(
                space=ControlSpace.JOINT,
                control_mode=[ControlMode.POS] * 6,
                policy_mode=[PolicyMode.ABSOLUTE] * 6,
            )
        )


def test_controller_joint_impedance_uses_direct_torque_interface():
    controller_module = _load_controller_module()
    controller = object.__new__(controller_module.RTDETaskFrameController)
    controller.kp = np.array([10.0] * 6)
    controller.kd = np.array([2.0] * 6)

    torque = controller._compute_joint_torque(
        q_cmd=np.array([1.0, -0.5, 0.2, 0.0, 0.0, 0.3]),
        q_actual=np.array([0.8, -0.4, 0.0, 0.1, 0.0, 0.0]),
        qd_actual=np.array([0.2, -0.1, 0.0, 0.3, 0.0, -0.2]),
    )
    np.testing.assert_allclose(torque, [1.6, -0.8, 2.0, -1.6, 0.0, 3.4])

    class DirectTorqueOnly:
        def __init__(self):
            self.calls = []

        def directTorque(self, torque_cmd, friction_comp):
            self.calls.append((torque_cmd, friction_comp))

    rtde_c = DirectTorqueOnly()
    controller_module.RTDETaskFrameController._send_joint_torque(rtde_c, torque)
    assert rtde_c.calls == [(torque.tolist(), True)]


def test_controller_reexpresses_virtual_target_when_task_frame_origin_changes():
    controller_module = _load_controller_module()
    controller = object.__new__(controller_module.RTDETaskFrameController)
    controller._active_space = None
    controller.origin = np.zeros(6, dtype=np.float64)
    controller.control_mode = np.array([int(ControlMode.POS)] * 6, dtype=np.int64)
    controller.delta_mode = np.array([int(PolicyMode.ABSOLUTE)] * 6, dtype=np.int64)
    controller.force_on = True
    controller._resolve_compliance_settings = lambda **kwargs: None
    controller.read_current_state = lambda rtde_r: {"ActualTCPPose": np.zeros(6, dtype=np.float64)}
    controller._enter_task_force_mode = lambda rtde_c: None

    class _Receive:
        @staticmethod
        def getActualQ():
            return [0.0] * 6

    command = controller_module.TaskFrameCommand(
        origin=[0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2.0],
        target=[0.0] * 6,
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[PolicyMode.ABSOLUTE] * 6,
    )
    msgs = _batched_queue_dict(command.to_queue_dict())

    keep_running, active_space, x_cmd, _ = controller._apply_pending_commands(
        msgs=msgs,
        n_cmd=1,
        rtde_c=object(),
        rtde_r=_Receive(),
        active_space=None,
        x_cmd=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        q_cmd=np.zeros(6, dtype=np.float64),
    )

    assert keep_running is True
    assert active_space == ControlSpace.TASK
    np.testing.assert_allclose(x_cmd[:3], [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(x_cmd[3:6], [0.0, 0.0, -np.pi / 2.0], atol=1e-6)


def test_controller_partial_rotation_uses_masked_so3_delta_then_applies_absolute_rpy():
    controller_module = _load_controller_module()
    controller = object.__new__(controller_module.RTDETaskFrameController)
    controller.control_mode = np.array([int(ControlMode.POS)] * 6, dtype=np.int64)
    controller.delta_mode = np.array(
        [
            int(PolicyMode.ABSOLUTE),
            int(PolicyMode.ABSOLUTE),
            int(PolicyMode.ABSOLUTE),
            int(PolicyMode.ABSOLUTE),
            int(PolicyMode.RELATIVE),
            int(PolicyMode.ABSOLUTE),
        ],
        dtype=np.int64,
    )
    controller.target = np.array([0.0, 0.0, 0.0, 0.4, 0.2, -0.3], dtype=np.float64)

    start_rpy = np.array([0.1, -0.2, 0.25], dtype=np.float64)
    updated_rotvec = controller._integrate_virtual_target_rotation(
        controller_module.R.from_euler("xyz", start_rpy, degrees=False).as_rotvec(),
        dt=0.5,
    )

    after_delta = controller_module.R.from_rotvec([0.0, 0.1, 0.0]) * controller_module.R.from_euler(
        "xyz",
        start_rpy,
        degrees=False,
    )
    expected_rpy = controller_module.wrap_to_pi(after_delta.as_euler("xyz", degrees=False))
    expected_rpy[[0, 2]] = [0.4, -0.3]

    updated_rpy = controller_module.wrap_to_pi(
        controller_module.R.from_rotvec(updated_rotvec).as_euler("xyz", degrees=False)
    )
    np.testing.assert_allclose(updated_rpy, expected_rpy, atol=1e-6)
    np.testing.assert_allclose(updated_rpy[[0, 2]], [0.4, -0.3], atol=1e-6)


def test_ur_wrapper_locks_joint_space_and_rejects_task_space_afterwards():
    controller_module = _load_controller_module()
    ur_module = _load_ur_module(controller_module)
    robot = object.__new__(ur_module.URV2)
    robot.controller = _ReadyController()
    robot.gripper = None
    robot.cameras = {}
    robot.config = types.SimpleNamespace(
        kp=[2500.0] * 3 + [150.0] * 3,
        kd=[80.0] * 3 + [8.0] * 3,
        wrench_limits=[30.0] * 6,
        compliance_adaptive_limit_enable=[False] * 6,
        compliance_reference_limit_enable=[False] * 6,
        compliance_desired_wrench=[5.0] * 6,
        compliance_adaptive_limit_min=[0.1] * 6,
    )
    robot.task_frame = controller_module.TaskFrameCommand(
        space=ControlSpace.JOINT,
        target=[0.0] * 6,
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[PolicyMode.RELATIVE] * 6,
    )
    robot._active_control_space = None

    robot.set_task_frame(robot.task_frame)
    robot.send_action({"joint_1.pos": 0.1, "joint_2.pos": -0.2})

    sent = robot.controller.commands[-1]
    assert sent.space == ControlSpace.JOINT
    assert sent.target[:2] == [0.1, -0.2]

    with pytest.raises(ValueError, match="switching between task-space and joint-space"):
        robot.send_action({"x.ee_pos": 0.05})

    with pytest.raises(ValueError, match="switching between task-space and joint-space"):
        robot.set_task_frame(
            TaskFrame(
                space=ControlSpace.TASK,
                target=[0.0] * 6,
                control_mode=[ControlMode.POS] * 6,
                policy_mode=[PolicyMode.ABSOLUTE] * 6,
            )
        )


def test_ur_wrapper_applies_task_frame_controller_overrides():
    controller_module = _load_controller_module()
    ur_module = _load_ur_module(controller_module)
    robot = object.__new__(ur_module.UR)
    robot.controller = _ReadyController()
    robot.gripper = None
    robot.cameras = {}
    robot.config = types.SimpleNamespace(
        kp=[2500.0] * 3 + [150.0] * 3,
        kd=[80.0] * 3 + [8.0] * 3,
        wrench_limits=[30.0] * 6,
        compliance_adaptive_limit_enable=[False] * 6,
        compliance_reference_limit_enable=[False] * 6,
        compliance_desired_wrench=[5.0] * 6,
        compliance_adaptive_limit_min=[0.1] * 6,
    )
    robot.task_frame = controller_module.TaskFrameCommand()
    robot._active_control_space = None

    robot.set_task_frame(
        TaskFrame(
            target=[0.0] * 6,
            control_mode=[ControlMode.POS] * 6,
            controller_overrides={
                "kp": [11.0] * 6,
                "kd": [4.0] * 6,
                "min_pose": [-0.3] * 6,
                "max_pose": [0.4] * 6,
                "wrench_limits": [10.0] * 6,
                "compliance_adaptive_limit_enable": [False] * 6,
                "compliance_reference_limit_enable": [True] * 6,
                "compliance_desired_wrench": [3.0] * 6,
                "compliance_adaptive_limit_min": [0.2] * 6,
            },
        )
    )

    assert robot.task_frame.controller_overrides["kp"] == [11.0] * 6
    assert robot.task_frame.controller_overrides["kd"] == [4.0] * 6
    assert robot.task_frame.controller_overrides["min_pose"] == [-0.3] * 6
    assert robot.task_frame.controller_overrides["max_pose"] == [0.4] * 6
    assert robot.task_frame.controller_overrides["wrench_limits"] == [10.0] * 6
    assert robot.task_frame.controller_overrides["compliance_adaptive_limit_enable"] == [False] * 6
    assert robot.task_frame.controller_overrides["compliance_reference_limit_enable"] == [True] * 6
    assert robot.task_frame.controller_overrides["compliance_desired_wrench"] == [3.0] * 6
    assert robot.task_frame.controller_overrides["compliance_adaptive_limit_min"] == [0.2] * 6

    queue_dict = robot.task_frame.to_queue_dict()
    np.testing.assert_allclose(queue_dict["kp"], [11.0] * 6)
    np.testing.assert_allclose(queue_dict["kd"], [4.0] * 6)
    np.testing.assert_allclose(queue_dict["min_pose"], [-0.3] * 6)
    np.testing.assert_allclose(queue_dict["max_pose"], [0.4] * 6)
    np.testing.assert_allclose(queue_dict["wrench_limits"], [10.0] * 6)
    np.testing.assert_array_equal(queue_dict["compliance_adaptive_limit_enable"], [False] * 6)
    np.testing.assert_array_equal(queue_dict["compliance_reference_limit_enable"], [True] * 6)
    np.testing.assert_allclose(queue_dict["compliance_desired_wrench"], [3.0] * 6)
    np.testing.assert_allclose(queue_dict["compliance_adaptive_limit_min"], [0.2] * 6)

    robot.set_task_frame(
        TaskFrame(
            target=[0.7] * 6,
            control_mode=[ControlMode.POS] * 6,
        )
    )
    assert robot.task_frame.target == [0.7] * 6
    assert robot.task_frame.controller_overrides["kp"] == [11.0] * 6
    assert robot.task_frame.controller_overrides["kd"] == [4.0] * 6
    assert robot.task_frame.controller_overrides["wrench_limits"] == [10.0] * 6
    assert robot.task_frame.controller_overrides["compliance_reference_limit_enable"] == [True] * 6


def test_ur_wrapper_rejects_unknown_task_frame_controller_override():
    controller_module = _load_controller_module()
    ur_module = _load_ur_module(controller_module)
    robot = object.__new__(ur_module.UR)
    robot.controller = _ReadyController()
    robot.gripper = None
    robot.cameras = {}
    robot.config = types.SimpleNamespace()
    robot.task_frame = controller_module.TaskFrameCommand()
    robot._active_control_space = None

    with pytest.raises(ValueError, match="Unsupported UR task-frame controller overrides"):
        robot.set_task_frame(
            TaskFrame(
                target=[0.0] * 6,
                control_mode=[ControlMode.POS] * 6,
                controller_overrides={"mystery_limit": [1.0] * 6},
            )
        )
