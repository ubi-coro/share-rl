"""Focused tests for dynamic-target and open-loop primitive behavior."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from lerobot.processor import TransitionKey

from share.debug.mpnet_debug import MPNetDebugConfig, MPNetDebugger
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    MoveDeltaPrimitiveConfig,
    OpenLoopTrajectoryPrimitiveConfig,
    PrimitiveEntryContext,
    ManipulationPrimitiveConfig,
)
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.envs.manipulation_primitive_net.transitions import OnTargetPoseReached
from share.utils.mock_utils import MockRobot, MockTeleoperator


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

    def set_target_pose(self, target_pose, info_key):
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
    config = _validated_move_delta([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], delta_frame="ee_current")
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


def test_move_delta_zero_delta_holds_entry_pose_instead_of_static_target():
    config = _validated_move_delta([0.0] * 6, delta_frame="world")
    config.task_frame["arm"].target = [9.0, 8.0, 7.0, -0.4, 0.5, -0.6]
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

    assert env.target_pose["arm"] == pytest.approx([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
    assert env._get_info()["primitive_target_pose"]["arm"] == pytest.approx([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])


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

    assert env.target_pose["arm"] == pytest.approx([1.25, 5.0, 7.0, 0.15, 0.3, 0.15])


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
    config = OpenLoopTrajectoryPrimitiveConfig(
        task_frame={"arm": _task_frame()},
        delta={"arm": [0.4, 0.0, 0.0, 0.0, 0.0, 0.0]},
        duration_substeps=4,
        substeps_per_step=2,
    )
    config.validate(
        robot_dict={"arm": MockRobot(name="arm", is_task_frame=True)},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
    )

    env, _, _ = config.make(
        robot_dict={"arm": MockRobot(name="arm", is_task_frame=True)},
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


def test_open_loop_trajectory_uses_entry_pose_for_fixed_pos_axes():
    config = OpenLoopTrajectoryPrimitiveConfig(
        task_frame={"arm": _task_frame()},
        delta={"arm": [0.0] * 6},
        duration_substeps=4,
        substeps_per_step=2,
    )
    config.task_frame["arm"].target = [3.0, 4.0, 5.0, -0.4, 0.5, -0.6]
    config.validate(
        robot_dict={"arm": MockRobot(name="arm", is_task_frame=True)},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
    )

    env, _, _ = config.make(
        robot_dict={"arm": MockRobot(name="arm", is_task_frame=True)},
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

    assert env._target_pose["arm"] == pytest.approx([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])


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
    config = OpenLoopTrajectoryPrimitiveConfig(
        task_frame={"arm": _task_frame()},
        delta={"arm": [0.4, 0.0, 0.0, 0.0, 0.0, 0.0]},
        duration_substeps=4,
        substeps_per_step=2,
    )
    config.validate(
        robot_dict={"arm": MockRobot(name="arm", is_task_frame=True)},
        teleop_dict={"arm": MockTeleoperator(name="arm", is_delta=True)},
    )

    env, _, _ = config.make(
        robot_dict={"arm": MockRobot(name="arm", is_task_frame=True)},
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
