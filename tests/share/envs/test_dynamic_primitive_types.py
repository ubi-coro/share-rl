"""Focused tests for dynamic-target and open-loop primitive behavior."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
from scipy.spatial.transform import Rotation

import draccus
from lerobot.processor import TransitionKey

from share.debug.mpnet_debug import MPNetDebugConfig, MPNetDebugger
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    ManipulationPrimitiveProcessorConfig,
    MoveDeltaPrimitiveConfig,
    OpenLoopTrajectorySpec,
    OpenLoopTrajectoryPrimitiveConfig,
    PrimitiveEntryContext,
    ManipulationPrimitiveConfig,
)
from share.envs.manipulation_primitive.env_manipulation_primitive import ManipulationPrimitive
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.envs.manipulation_primitive_net.transitions import (
    DEFAULT_TARGET_POSE_AXES_INFO_KEY,
    OnTargetPoseReached,
)
from share.teleoperators import TeleopEvents
from share.utils.mock_utils import MockRobot, MockTeleoperator
from share.utils.transformation_utils import compose_delta_pose


class _CustomEnvForEnvClassTest(ManipulationPrimitive):
    """Module-level so the draccus type decoder can resolve it by dotted path."""

    pass


class IdentityProcessor:
    """Processor stub that keeps transitions unchanged while tracking resets."""

    def __init__(self):
        self.reset_count = 0

    def __call__(self, transition):
        return transition

    def reset(self):
        self.reset_count += 1


class DummyPrimitiveEnv:
    """Minimal env stub for primitive-entry and scripted-step tests."""

    uses_autonomous_step = False

    def __init__(self, observation: dict[str, float]):
        self.observation = dict(observation)
        self.reset_calls = 0
        self.applied_task_frames = 0
        self.actions: list[dict[str, dict[str, float]]] = []
        self.target_pose = {}
        self.target_pose_info_key = None

    def reset(self, *, seed=None, options=None):
        self.reset_calls += 1
        return dict(self.observation), {"reset_seed": seed}

    def apply_task_frames(self):
        self.applied_task_frames += 1

    def reset_runtime_state(self):
        self.target_pose = {}
        self.target_pose_info_key = None

    def set_target_pose(self, target_pose, info_key, update_task_frame=True):
        self.target_pose = {name: list(pose) for name, pose in target_pose.items()}
        self.target_pose_info_key = info_key

    def step(self, action):
        self.actions.append(action)
        robot_action = action.get("arm", {})
        updated = dict(self.observation)
        for axis_name in ["x", "y", "z", "rx", "ry", "rz"]:
            key = f"{axis_name}.ee_pos"
            if key in robot_action:
                updated[f"arm.{axis_name}.ee_pos"] = robot_action[key]
        self.observation = updated
        return dict(self.observation), 0.0, False, False, self._get_info()

    def _get_observation(self):
        return dict(self.observation)

    def _get_info(self):
        info = {
            "primitive_complete": False,
            "trajectory_progress": 0.0,
        }
        if self.target_pose_info_key is not None:
            info[self.target_pose_info_key] = {name: list(pose) for name, pose in self.target_pose.items()}
        return info


def _task_frame(origin=None) -> TaskFrame:
    return TaskFrame(
        target=[0.0] * 6,
        origin=[0.0] * 6 if origin is None else list(origin),
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    )


def _validated_move_delta(delta, delta_frame="world", origin=None) -> MoveDeltaPrimitiveConfig:
    config = MoveDeltaPrimitiveConfig(
        task_frame={"arm": _task_frame(origin=origin)},
        delta={"arm": delta},
        delta_frame={"arm": delta_frame},
    )
    config.validate(
        robot_dict={"arm": MockRobot(name="arm", is_task_frame=True)},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
    )
    return config


def _validated_open_loop(
    *,
    target=None,
    delta=None,
    frame="task",
    duration_s=1.0,
    origin=None,
    fps=10.0,
) -> OpenLoopTrajectoryPrimitiveConfig:
    config = OpenLoopTrajectoryPrimitiveConfig(
        task_frame={"arm": _task_frame(origin=origin)},
        processor=ManipulationPrimitiveProcessorConfig(fps=fps),
        trajectory=OpenLoopTrajectorySpec(
            target={"arm": target} if target is not None else None,
            delta={"arm": delta} if delta is not None else None,
            frame={"arm": frame},
            duration_s={"arm": duration_s},
        ),
    )
    config.validate(
        robot_dict={"arm": MockRobot(name="arm", is_task_frame=True)},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
    )
    return config


def _rpy_from_rotvec(rotvec) -> list[float]:
    return Rotation.from_rotvec(rotvec).as_euler("xyz", degrees=False).tolist()


def test_move_delta_primitive_resolves_world_target_on_entry():
    config = _validated_move_delta([0.1, -0.2, 0.3, 0.0, 0.0, 0.0], delta_frame="world")
    env = DummyPrimitiveEnv({})

    config.on_entry(
        env,
        PrimitiveEntryContext(
            observation={
                "arm.x.ee_pos": 1.0,
                "arm.y.ee_pos": 2.0,
                "arm.z.ee_pos": 3.0,
                "arm.rx.ee_pos": 0.1,
                "arm.ry.ee_pos": 0.2,
                "arm.rz.ee_pos": 0.3,
            },
            task_frame_origin={"arm": [0.0] * 6},
        ),
    )

    assert env.target_pose["arm"][:3] == pytest.approx([1.1, 1.8, 3.3])
    assert env._get_info()["primitive_target_pose"]["arm"][:3] == pytest.approx([1.1, 1.8, 3.3])


def test_move_delta_primitive_resolves_ee_relative_translation_on_entry():
    config = _validated_move_delta([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], delta_frame="ee")
    env = DummyPrimitiveEnv({})

    config.on_entry(
        env,
        PrimitiveEntryContext(
            observation={
                "arm.x.ee_pos": 0.0,
                "arm.y.ee_pos": 0.0,
                "arm.z.ee_pos": 0.0,
                "arm.rx.ee_pos": 0.0,
                "arm.ry.ee_pos": 0.0,
                "arm.rz.ee_pos": math.pi / 2,
            },
            task_frame_origin={"arm": [0.0] * 6},
        ),
    )

    assert env.target_pose["arm"][0] == pytest.approx(0.0, abs=1e-6)
    assert env.target_pose["arm"][1] == pytest.approx(0.1, abs=1e-6)


def test_move_delta_uses_processed_pose_channels_when_relative_view_is_present():
    config = _validated_move_delta([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], delta_frame="ee", origin=[0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    env = DummyPrimitiveEnv({})

    config.on_entry(
        env,
        PrimitiveEntryContext(
            observation={
                "arm.x.ee_pos": 0.0,
                "arm.y.ee_pos": 0.0,
                "arm.z.ee_pos": 0.0,
                "arm.rx.ee_pos": 0.0,
                "arm.ry.ee_pos": 0.0,
                "arm.rz.ee_pos": 0.0,
            },
            task_frame_origin={"arm": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]},
        ),
    )

    assert env.target_pose["arm"][0] == pytest.approx(0.1, abs=1e-6)
    assert env.target_pose["arm"][1] == pytest.approx(0.0, abs=1e-6)
    assert env.target_pose["arm"][2] == pytest.approx(0.0, abs=1e-6)


def test_move_delta_zero_delta_holds_entry_pose_instead_of_static_target():
    config = _validated_move_delta([0.0] * 6, delta_frame="world")
    config.task_frame["arm"].target = [9.0, 8.0, 7.0, -0.4, 0.5, -0.6]
    env = DummyPrimitiveEnv({})
    expected_orientation = _rpy_from_rotvec([0.1, 0.2, 0.3])

    config.on_entry(
        env,
        PrimitiveEntryContext(
            observation={
                "arm.x.ee_pos": 1.0,
                "arm.y.ee_pos": 2.0,
                "arm.z.ee_pos": 3.0,
                "arm.rx.ee_pos": 0.1,
                "arm.ry.ee_pos": 0.2,
                "arm.rz.ee_pos": 0.3,
            },
            task_frame_origin={"arm": [0.0] * 6},
        ),
    )

    assert env.target_pose["arm"] == pytest.approx([1.0, 2.0, 3.0, *expected_orientation])
    assert env._get_info()["primitive_target_pose"]["arm"] == pytest.approx([1.0, 2.0, 3.0, *expected_orientation])


def test_move_delta_only_resolves_fixed_pos_axes_from_entry_delta():
    config = MoveDeltaPrimitiveConfig(
        task_frame={
            "arm": TaskFrame(
                target=[9.0, 5.0, 7.0, 0.8, 0.9, 1.0],
                origin=[0.0] * 6,
                policy_mode=[None, PolicyMode.RELATIVE, None, None, None, None],
                control_mode=[
                    ControlMode.POS,
                    ControlMode.POS,
                    ControlMode.VEL,
                    ControlMode.POS,
                    ControlMode.POS,
                    ControlMode.POS,
                ],
            )
        },
        delta={"arm": [0.25, 0.4, 0.3, 0.05, 0.1, -0.15]},
        delta_frame={"arm": "world"},
    )
    config.validate(
        robot_dict={"arm": MockRobot(name="arm", is_task_frame=True)},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
    )
    env = DummyPrimitiveEnv({})
    expected_orientation = _rpy_from_rotvec([0.1, 0.2, 0.3])

    config.on_entry(
        env,
        PrimitiveEntryContext(
            observation={
                "arm.x.ee_pos": 1.0,
                "arm.y.ee_pos": 2.0,
                "arm.z.ee_pos": 3.0,
                "arm.rx.ee_pos": 0.1,
                "arm.ry.ee_pos": 0.2,
                "arm.rz.ee_pos": 0.3,
            },
            task_frame_origin={"arm": [0.0] * 6},
        ),
    )

    expected_pose = compose_delta_pose(
        [1.0, 2.0, 3.0, *expected_orientation],
        [0.25, 0.4, 0.3, 0.05, 0.1, -0.15],
        "world",
    )
    assert env.target_pose["arm"] == pytest.approx(
        [expected_pose[0], 5.0, 7.0, expected_pose[3], expected_pose[4], expected_pose[5]]
    )


def test_move_delta_resolves_partial_fixed_rotation_axes_independently():
    start_rpy = [0.2, 0.3, -0.1]
    start_rotvec = Rotation.from_euler("xyz", start_rpy, degrees=False).as_rotvec()
    config = MoveDeltaPrimitiveConfig(
        task_frame={
            "arm": TaskFrame(
                target=[0.0, 0.0, 0.0, 9.0, 8.0, 7.0],
                origin=[0.0] * 6,
                policy_mode=[None, None, None, None, PolicyMode.RELATIVE, PolicyMode.RELATIVE],
                control_mode=[ControlMode.POS] * 6,
            )
        },
        delta={"arm": [0.0, 0.0, 0.0, 0.4, 0.2, 0.5]},
        delta_frame={"arm": "ee"},
    )
    config.validate(
        robot_dict={"arm": MockRobot(name="arm", is_task_frame=True)},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
    )
    env = DummyPrimitiveEnv({})

    config.on_entry(
        env,
        PrimitiveEntryContext(
            observation={
                "arm.x.ee_pos": 0.0,
                "arm.y.ee_pos": 0.0,
                "arm.z.ee_pos": 0.0,
                "arm.rx.ee_pos": float(start_rotvec[0]),
                "arm.ry.ee_pos": float(start_rotvec[1]),
                "arm.rz.ee_pos": float(start_rotvec[2]),
            },
            task_frame_origin={"arm": [0.0] * 6},
        ),
    )

    expected_x_only = compose_delta_pose(
        [0.0, 0.0, 0.0, *start_rpy],
        [0.0, 0.0, 0.0, 0.4, 0.0, 0.0],
        "ee",
    )
    assert env.target_pose["arm"][3] == pytest.approx(expected_x_only[3])
    assert env.target_pose["arm"][4] == pytest.approx(8.0)
    assert env.target_pose["arm"][5] == pytest.approx(7.0)


def test_move_delta_zero_rotation_delta_publishes_current_orientation_as_rpy():
    config = _validated_move_delta([0.0] * 6, delta_frame="world")
    env = DummyPrimitiveEnv({})
    expected_orientation = _rpy_from_rotvec([0.15, -0.1, 0.25])

    config.on_entry(
        env,
        PrimitiveEntryContext(
            observation={
                "arm.x.ee_pos": 0.0,
                "arm.y.ee_pos": 0.0,
                "arm.z.ee_pos": 0.0,
                "arm.rx.ee_pos": 0.15,
                "arm.ry.ee_pos": -0.1,
                "arm.rz.ee_pos": 0.25,
            },
            task_frame_origin={"arm": [0.0] * 6},
        ),
    )

    assert env.target_pose["arm"][3:6] == pytest.approx(expected_orientation)


def test_target_pose_transition_reads_current_pose_from_observation():
    target_transition = OnTargetPoseReached(
        source="move",
        target="next",
        robot_name="arm",
        axes=["x"],
        tolerance=[0.02] * 6,
    )
    outcome = target_transition.evaluate(
        obs={
            "arm.x.ee_pos": 0.51,
            "arm.y.ee_pos": 0.0,
            "arm.z.ee_pos": 0.0,
            "arm.rx.ee_pos": 0.0,
            "arm.ry.ee_pos": 0.0,
            "arm.rz.ee_pos": 0.0,
        },
        info={
            "primitive_target_pose": {"arm": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]},
        },
    )
    assert outcome.terminated is True


def test_target_pose_transition_defaults_to_fixed_pos_axes_from_info():
    target_transition = OnTargetPoseReached(
        source="move",
        target="next",
        robot_name="arm",
        tolerance=[0.02] * 6,
    )
    outcome = target_transition.evaluate(
        obs={
            "arm.x.ee_pos": 0.51,
            "arm.y.ee_pos": 0.3,
            "arm.z.ee_pos": 0.0,
            "arm.rx.ee_pos": 0.4,
            "arm.ry.ee_pos": 0.0,
            "arm.rz.ee_pos": 0.0,
        },
        info={
            "primitive_target_pose": {"arm": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]},
            DEFAULT_TARGET_POSE_AXES_INFO_KEY: {"arm": [0]},
        },
    )
    assert outcome.terminated is True


def test_mp_net_reset_uses_pending_entry_context_for_new_primitive():
    move_delta = _validated_move_delta([0.25, 0.0, 0.0, 0.0, 0.0, 0.0], delta_frame="world")
    env = DummyPrimitiveEnv(
        {
            "arm.x.ee_pos": 0.0,
            "arm.y.ee_pos": 0.0,
            "arm.z.ee_pos": 0.0,
            "arm.rx.ee_pos": 0.0,
            "arm.ry.ee_pos": 0.0,
            "arm.rz.ee_pos": 0.0,
        }
    )

    net = ManipulationPrimitiveNet.__new__(ManipulationPrimitiveNet)
    net._envs = {"move": env}
    net._env_processors = {"move": IdentityProcessor()}
    net._action_processors = {"move": IdentityProcessor()}
    net._transitions = {"move": []}
    net.config = SimpleNamespace(
        primitives={"move": move_delta},
        start_primitive="move",
        reset_primitive="move",
        fps=10,
        terminals=[],
    )
    net._active = "move"
    net._last_reset_info = {}
    net._pending_entry_context = PrimitiveEntryContext(
        observation={
            "arm.x.ee_pos": 0.4,
            "arm.y.ee_pos": 0.0,
            "arm.z.ee_pos": 0.0,
            "arm.rx.ee_pos": 0.0,
            "arm.ry.ee_pos": 0.0,
            "arm.rz.ee_pos": 0.0,
        },
        task_frame_origin={"arm": [0.0] * 6},
    )
    net._episode_step_count = 0
    net._primitive_step_count = 0
    net._needs_full_reset = False

    transition = net.reset()

    assert env.applied_task_frames == 1
    assert transition[TransitionKey.INFO]["primitive_target_pose"]["arm"][0] == pytest.approx(0.65)


def test_open_loop_trajectory_runs_chunked_substeps_and_reports_progress():
    config = _validated_open_loop(delta=[0.4, 0.0, 0.0, 0.0, 0.0, 0.0], frame="world", duration_s=0.4, fps=5.0)
    robot = MockRobot(name="arm", is_task_frame=True)
    robot.config.frequency = 10.0

    env, _, _ = config.make(
        robot_dict={"arm": robot},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
        cameras={},
    )
    env.robot_dict["arm"].get_observation = lambda: {
        "x.ee_pos": 0.0,
        "y.ee_pos": 0.0,
        "z.ee_pos": 0.0,
        "rx.ee_pos": 0.0,
        "ry.ee_pos": 0.0,
        "rz.ee_pos": 0.0,
    }
    env.on_step_callback = None
    config.on_entry(
        env,
        PrimitiveEntryContext(
            observation={
                "arm.x.ee_pos": 0.0,
                "arm.y.ee_pos": 0.0,
                "arm.z.ee_pos": 0.0,
                "arm.rx.ee_pos": 0.0,
                "arm.ry.ee_pos": 0.0,
                "arm.rz.ee_pos": 0.0,
            },
            task_frame_origin={"arm": [0.0] * 6},
        ),
    )

    first = env.step({})
    assert env._trajectory_substeps == 2
    assert first[4]["trajectory_progress"] == pytest.approx(0.5)

    second = env.step({})
    assert env._trajectory_substeps == 4
    assert second[4]["trajectory_progress"] == pytest.approx(1.0)
    assert second[4]["primitive_complete"] is True


def test_open_loop_trajectory_keeps_sampling_after_nominal_completion():
    config = _validated_open_loop(delta=[0.4, 0.0, 0.0, 0.0, 0.0, 0.0], frame="world", duration_s=0.4, fps=5.0)
    robot = MockRobot(name="arm", is_task_frame=True)
    robot.config.frequency = 10.0

    env, _, _ = config.make(
        robot_dict={"arm": robot},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
        cameras={},
    )
    env.robot_dict["arm"].get_observation = lambda: {
        f"{axis}.ee_pos": float(env.robot_dict["arm"].current_frame.target[index])
        for index, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"])
    }
    config.on_entry(
        env,
        PrimitiveEntryContext(
            observation={
                "arm.x.ee_pos": 0.0,
                "arm.y.ee_pos": 0.0,
                "arm.z.ee_pos": 0.0,
                "arm.rx.ee_pos": 0.0,
                "arm.ry.ee_pos": 0.0,
                "arm.rz.ee_pos": 0.0,
            },
            task_frame_origin={"arm": [0.0] * 6},
        ),
    )

    env.step({})
    env.step({})
    third = env.step({})

    assert env._trajectory_substeps == 6
    assert third[4]["trajectory_progress"] == pytest.approx(1.0)
    assert third[4]["primitive_complete"] is True
    assert env.task_frame["arm"].target == pytest.approx([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_open_loop_trajectory_uses_entry_pose_for_fixed_pos_axes():
    config = _validated_open_loop(delta=[0.0] * 6, frame="world", duration_s=0.4, fps=5.0)
    robot = MockRobot(name="arm", is_task_frame=True)
    robot.config.frequency = 10.0

    env, _, _ = config.make(
        robot_dict={"arm": robot},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
        cameras={},
    )
    env.robot_dict["arm"].get_observation = lambda: {
        "x.ee_pos": 1.0,
        "y.ee_pos": 2.0,
        "z.ee_pos": 3.0,
        "rx.ee_pos": 0.1,
        "ry.ee_pos": 0.2,
        "rz.ee_pos": 0.3,
    }
    expected_orientation = _rpy_from_rotvec([0.1, 0.2, 0.3])

    config.on_entry(
        env,
        PrimitiveEntryContext(
            observation={
                "arm.x.ee_pos": 1.0,
                "arm.y.ee_pos": 2.0,
                "arm.z.ee_pos": 3.0,
                "arm.rx.ee_pos": 0.1,
                "arm.ry.ee_pos": 0.2,
                "arm.rz.ee_pos": 0.3,
            },
            task_frame_origin={"arm": [0.0] * 6},
        ),
    )

    assert env._target_pose["arm"] == pytest.approx([1.0, 2.0, 3.0, *expected_orientation])


def test_open_loop_trajectory_config_owns_current_target_sampling():
    config = _validated_open_loop(target=[0.4, 0.0, 0.2, 0.0, 0.0, 0.0], frame="task", duration_s=1.0)
    sampled = config.target_pose_at(
        alpha=0.25,
        start_pose={"arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        goal_pose={"arm": [0.4, 0.0, 0.2, 0.0, 0.0, 0.0]},
    )
    assert sampled["arm"] == pytest.approx([0.1, 0.0, 0.05, 0.0, 0.0, 0.0])


def test_open_loop_trajectory_keeps_final_target_info_for_target_pose_transition():
    config = _validated_open_loop(delta=[0.4, 0.0, 0.0, 0.0, 0.0, 0.0], frame="world", duration_s=0.4, fps=5.0)
    robot = MockRobot(name="arm", is_task_frame=True)
    robot.config.frequency = 10.0

    env, _, _ = config.make(
        robot_dict={"arm": robot},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
        cameras={},
    )
    env.robot_dict["arm"].get_observation = lambda: {
        f"{axis}.ee_pos": float(env.robot_dict["arm"].current_frame.target[index])
        for index, axis in enumerate(["x", "y", "z", "rx", "ry", "rz"])
    }
    config.on_entry(
        env,
        PrimitiveEntryContext(
            observation={
                "arm.x.ee_pos": 0.0,
                "arm.y.ee_pos": 0.0,
                "arm.z.ee_pos": 0.0,
                "arm.rx.ee_pos": 0.0,
                "arm.ry.ee_pos": 0.0,
                "arm.rz.ee_pos": 0.0,
            },
            task_frame_origin={"arm": [0.0] * 6},
        ),
    )

    final_step = env.step({})
    final_step = env.step({})
    transition = OnTargetPoseReached(source="scripted", target="done", robot_name="arm", axes=["x"])
    outcome = transition.evaluate(
        obs=final_step[0],
        info={
            **final_step[4],
            DEFAULT_TARGET_POSE_AXES_INFO_KEY: {"arm": [0]},
        },
    )
    assert final_step[4]["primitive_target_pose"]["arm"][0] == pytest.approx(0.4)
    assert outcome.terminated is True


def test_static_primitive_publishes_target_pose_info_on_entry():
    config = ManipulationPrimitiveConfig(task_frame={"arm": _task_frame()})
    config.validate(
        robot_dict={"arm": MockRobot(name="arm", is_task_frame=True)},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
    )
    env = DummyPrimitiveEnv({})

    config.on_entry(env, None)

    assert env.target_pose["arm"] == pytest.approx([0.0] * 6)
    assert env._get_info()["primitive_target_pose"]["arm"] == pytest.approx([0.0] * 6)


def test_open_loop_trajectory_info_matches_debugger_target_visualization(tmp_path):
    config = _validated_open_loop(delta=[0.4, 0.0, 0.0, 0.0, 0.0, 0.0], frame="world", duration_s=0.4, fps=5.0)
    robot = MockRobot(name="arm", is_task_frame=True)
    robot.config.frequency = 10.0

    env, _, _ = config.make(
        robot_dict={"arm": robot},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
        cameras={},
    )
    env.robot_dict["arm"].get_observation = lambda: {
        "x.ee_pos": 0.0,
        "y.ee_pos": 0.0,
        "z.ee_pos": 0.0,
        "rx.ee_pos": 0.0,
        "ry.ee_pos": 0.0,
        "rz.ee_pos": 0.0,
    }
    config.on_entry(
        env,
        PrimitiveEntryContext(
            observation={
                "arm.x.ee_pos": 0.0,
                "arm.y.ee_pos": 0.0,
                "arm.z.ee_pos": 0.0,
                "arm.rx.ee_pos": 0.0,
                "arm.ry.ee_pos": 0.0,
                "arm.rz.ee_pos": 0.0,
            },
            task_frame_origin={"arm": [0.0] * 6},
        ),
    )

    step = env.step({})
    config.is_terminal = True
    net_config = ManipulationPrimitiveNetConfig(
        start_primitive="scripted",
        reset_primitive="scripted",
        primitives={"scripted": config},
        transitions=[],
    )
    debugger = MPNetDebugger.start(
        MPNetDebugConfig(
            enabled=True,
            live_rerun=False,
            trace_path=tmp_path / "trace.jsonl",
            flush_interval_s=0.01,
        ),
        net_config,
    )
    debugger.log_step(
        SimpleNamespace(active_primitive="scripted", config=net_config),
        {
            TransitionKey.OBSERVATION: step[0],
            TransitionKey.INFO: {
                **step[4],
                "primitive_step": 1,
                "episode_step": 1,
                "transition_from": "scripted",
                "transition_to": "scripted",
                "transition_reason": None,
            },
        },
    )
    debugger.close()

    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    step_event = next(event for event in events if event["kind"] == "step")
    assert step_event["trajectory_progress"] == pytest.approx(0.5)
    assert step_event["robots"]["arm"]["target_pose"][0] == pytest.approx(0.4)


def test_resolve_teleop_dict_is_identity_without_mapping():
    config = ManipulationPrimitiveConfig(task_frame={"arm": _task_frame()})
    teleop_dict = {"arm": MockTeleoperator(name="arm")}

    resolved = config.resolve_teleop_dict(teleop_dict)

    assert resolved is teleop_dict


def test_resolve_teleop_dict_overrides_mapped_targets_and_keeps_unmapped_identity():
    left = MockTeleoperator(name="left")
    config = ManipulationPrimitiveConfig(
        task_frame={"left": _task_frame(), "right": _task_frame()},
        teleop_mapping={"right": "left"},
    )
    teleop_dict = {"left": left}

    resolved = config.resolve_teleop_dict(teleop_dict)

    assert resolved["right"] is left
    assert resolved["left"] is left
    assert "right" not in teleop_dict  # original not mutated


def test_resolve_teleop_dict_raises_clear_error_for_unknown_source():
    config = ManipulationPrimitiveConfig(
        task_frame={"right": _task_frame()},
        teleop_mapping={"right": "nonexistent"},
    )

    with pytest.raises(ValueError, match="nonexistent"):
        config.resolve_teleop_dict({"left": MockTeleoperator(name="left")})


def test_make_instantiates_default_env_class():
    config = ManipulationPrimitiveConfig(task_frame={"arm": _task_frame()})
    env, _, _ = config.make(
        robot_dict={"arm": MockRobot(name="arm", is_task_frame=True)},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
        cameras={},
    )
    assert type(env) is ManipulationPrimitive


def test_make_instantiates_custom_env_class_via_env_class_field():
    config = ManipulationPrimitiveConfig(
        task_frame={"arm": _task_frame()},
        env_class=_CustomEnvForEnvClassTest,
    )
    env, _, _ = config.make(
        robot_dict={"arm": MockRobot(name="arm", is_task_frame=True)},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
        cameras={},
    )
    assert isinstance(env, _CustomEnvForEnvClassTest)


def test_make_raises_clear_error_on_env_kwargs_reserved_key_collision():
    config = ManipulationPrimitiveConfig(
        task_frame={"arm": _task_frame()},
        env_kwargs={"robot_dict": {}},
    )
    with pytest.raises(ValueError, match="reserved"):
        config.make(
            robot_dict={"arm": MockRobot(name="arm", is_task_frame=True)},
            teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
            cameras={},
        )


def test_env_class_round_trips_through_draccus_encode_decode():
    """Regression test for save_mpnet_config()'s draccus.dump(), used by record.py's
    dataset snapshot. Scoped to encode/decode directly, not the full net-level round-trip:
    that currently fails for an unrelated pre-existing reason (EventConfig.key_mapping's
    pynput.keyboard.Key has no draccus decoder at all, regardless of env_class)."""
    config = ManipulationPrimitiveConfig(
        task_frame={"arm": _task_frame()},
        env_class=_CustomEnvForEnvClassTest,
    )

    encoded = draccus.encode(config)
    assert encoded["env_class"] == (
        f"{_CustomEnvForEnvClassTest.__module__}.{_CustomEnvForEnvClassTest.__qualname__}"
    )

    decoded_cls = draccus.decode(type, encoded["env_class"])
    assert decoded_cls is _CustomEnvForEnvClassTest


def _locked_task_frame(target=None) -> TaskFrame:
    return TaskFrame(
        target=[0.0] * 6 if target is None else list(target),
        origin=[0.0] * 6,
        policy_mode=[None] * 6,
        control_mode=[ControlMode.POS] * 6,
    )


def test_validate_allows_locked_robot_absent_from_teleop_dict():
    """A robot in task_frame but absent from teleop_dict must validate, not KeyError."""
    config = ManipulationPrimitiveConfig(
        task_frame={"driven": _task_frame(), "locked": _locked_task_frame()},
    )

    config.validate(
        robot_dict={
            "driven": MockRobot(name="driven", is_task_frame=True),
            "locked": MockRobot(name="locked", is_task_frame=True),
        },
        teleop_dict={"driven": MockTeleoperator(name="driven", is_delta=True)},
    )
    assert config._has_teleop == {"driven": True, "locked": False}


def test_validate_still_rejects_vel_axis_missing_teleop_with_clear_error():
    """A learnable VEL axis still requires a delta teleoperator, ValueError not KeyError."""
    frame = TaskFrame(
        # RELATIVE only supports POS; a learnable VEL axis must be ABSOLUTE
        target=[0.0] * 6,
        policy_mode=[PolicyMode.ABSOLUTE, None, None, None, None, None],
        control_mode=[ControlMode.VEL] + [ControlMode.POS] * 5,
    )
    config = ManipulationPrimitiveConfig(task_frame={"locked": frame})

    with pytest.raises(ValueError, match="require a delta teleoperator"):
        config.validate(
            robot_dict={"locked": MockRobot(name="locked", is_task_frame=True)},
            teleop_dict={},
        )


def test_on_entry_reports_actual_pose_for_locked_robot_with_no_teleoperator():
    """A robot with no teleoperator reports its current pose as target, not the placeholder."""
    config = ManipulationPrimitiveConfig(
        task_frame={"driven": _task_frame(), "locked": _locked_task_frame(target=[0.0] * 6)},
    )
    config.validate(
        robot_dict={
            "driven": MockRobot(name="driven", is_task_frame=True),
            "locked": MockRobot(name="locked", is_task_frame=True),
        },
        teleop_dict={"driven": MockTeleoperator(name="driven", is_delta=True)},
    )

    env = DummyPrimitiveEnv({})
    entry_context = PrimitiveEntryContext(
        observation={
            "driven.x.ee_pos": 0.0, "driven.y.ee_pos": 0.0, "driven.z.ee_pos": 0.0,
            "driven.rx.ee_pos": 0.0, "driven.ry.ee_pos": 0.0, "driven.rz.ee_pos": 0.0,
            "locked.x.ee_pos": 1.5, "locked.y.ee_pos": -0.4, "locked.z.ee_pos": 0.3,
            "locked.rx.ee_pos": 0.0, "locked.ry.ee_pos": 0.0, "locked.rz.ee_pos": 0.0,
        },
        task_frame_origin={"driven": [0.0] * 6, "locked": [0.0] * 6},
    )

    config.on_entry(env, entry_context)

    assert env.target_pose["locked"] == pytest.approx([1.5, -0.4, 0.3, 0.0, 0.0, 0.0])


def test_on_entry_keeps_static_target_for_scripted_robot_that_has_a_teleoperator():
    """A robot with a teleoperator declared keeps its real scripted target (e.g.
    get_target_prim_cfg()-style move-to-pose primitives), not the current pose."""
    scripted_target = [0.4, 0.1, 0.2, 0.0, 0.0, 0.0]
    config = ManipulationPrimitiveConfig(
        task_frame={"arm": _locked_task_frame(target=scripted_target)},
    )
    config.validate(
        robot_dict={"arm": MockRobot(name="arm", is_task_frame=True)},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
    )

    env = DummyPrimitiveEnv({})
    entry_context = PrimitiveEntryContext(
        observation={
            "arm.x.ee_pos": 0.0, "arm.y.ee_pos": 0.0, "arm.z.ee_pos": 0.0,
            "arm.rx.ee_pos": 0.0, "arm.ry.ee_pos": 0.0, "arm.rz.ee_pos": 0.0,
        },
        task_frame_origin={"arm": [0.0] * 6},
    )

    config.on_entry(env, entry_context)

    assert env.target_pose["arm"] == pytest.approx(scripted_target)
