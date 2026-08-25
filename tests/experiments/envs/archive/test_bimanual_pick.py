"""Tests for the archived experiments.envs.archive.bimanual_pick -- not actively maintained.
rail_bimanual_grasp.py now carries its own self-contained copy of CooperativeFramePrimitive
(tests in test_rail_bimanual_grasp.py); this file only guards the archived original."""

import math
import os

os.environ.setdefault("PYNPUT_BACKEND", "dummy")

import pytest
from pynput import keyboard

from lerobot.processor import TransitionKey, create_transition

import numpy as np

from experiments.envs.archive.bimanual_pick import (
    OFFSET_FROM_VTCP_RUNTIME_KEY,
    PIVOT_OFFSET_X_RUNTIME_KEY,
    BimanualPickEnvConfig,
    CooperativeFramePrimitive,
    PivotOffsetXCalibrationPrimitive,
    _origin_from_raw_obs,
)
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    ManipulationPrimitiveConfig,
    MoveDeltaPrimitiveConfig,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.robots.ur import SimUR, SimURConfig
from share.teleoperators import TeleopEvents
from share.utils.mock_utils import MockRobot, MockTeleoperator


class _FakeRobot:
    """Minimal robot double exposing EE-pose observations and recording send_action calls."""

    def __init__(self, pose: list[float]):
        self.pose = list(pose)
        self.last_action: dict[str, float] = {}
        self.last_task_frame = None
        self._motors_ft: dict[str, type] = {}

    def get_observation(self) -> dict[str, float]:
        return dict(zip(("x.ee_pos", "y.ee_pos", "z.ee_pos", "rx.ee_pos", "ry.ee_pos", "rz.ee_pos"), self.pose))

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.last_action = dict(action)
        return action

    def set_task_frame(self, frame) -> None:
        self.last_task_frame = frame


def test_bimanual_pick_config_builds_expected_graph():
    cfg = BimanualPickEnvConfig()

    assert list(cfg.primitives) == [
        "teleop_left", "teleop_right", "calibrate_pivot_offset_x", "teleop_midpoint", "reset", "policy",
    ]
    assert isinstance(cfg.primitives["teleop_left"], MoveDeltaPrimitiveConfig)
    assert isinstance(cfg.primitives["teleop_right"], MoveDeltaPrimitiveConfig)
    assert cfg.primitives["teleop_right"].teleop_mapping == {"right": "left"}
    assert cfg.primitives["calibrate_pivot_offset_x"].env_class is PivotOffsetXCalibrationPrimitive
    assert cfg.primitives["teleop_midpoint"].env_class is CooperativeFramePrimitive
    assert cfg.primitives["reset"].env_class is CooperativeFramePrimitive
    assert cfg.primitives["policy"].env_class is CooperativeFramePrimitive
    for name in ("calibrate_pivot_offset_x", "teleop_midpoint", "reset", "policy"):
        assert cfg.primitives[name].env_kwargs["robot_base_pose_in_world"] == {
            "right": cfg.right_base_pose_in_left_base
        }
        assert cfg.primitives[name].env_kwargs["pivot_offset_x"] == pytest.approx(0.0)
    lower_dimensional_rotation = [PolicyMode.RELATIVE, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, PolicyMode.RELATIVE, None]
    fixed_targets = {
        "left": [0.0, 0.0, 0.0, np.pi, 0.0, np.pi],
        "right": [0.0, 0.0, 0.0, np.pi, 0.0, 0.0],
    }
    for primitive in cfg.primitives.values():
        for arm, frame in primitive.task_frame.items():
            assert frame.target == pytest.approx(fixed_targets[arm])
    assert cfg.primitives["teleop_left"].task_frame["left"].policy_mode == lower_dimensional_rotation
    assert cfg.primitives["teleop_right"].task_frame["right"].policy_mode == lower_dimensional_rotation
    # calibrate_pivot_offset_x locks every axis on both arms -- only the ','/'.' keyboard
    # nudge (not the SpaceMouse) can move the pivot there.
    assert cfg.primitives["calibrate_pivot_offset_x"].task_frame["left"].policy_mode == [None] * 6
    for name in ("teleop_midpoint", "reset", "policy"):
        assert cfg.primitives[name].task_frame["left"].policy_mode == lower_dimensional_rotation
    assert [(t.source, t.target, type(t).__name__) for t in cfg.transitions] == [
        ("teleop_left", "teleop_right", "OnSuccess"),
        ("teleop_right", "calibrate_pivot_offset_x", "OnSuccess"),
        ("calibrate_pivot_offset_x", "teleop_midpoint", "OnSuccess"),
        ("teleop_midpoint", "reset", "OnSuccess"),
        ("reset", "policy", "OnSuccess"),
        ("policy", "reset", "OnSuccess"),
    ]
    # "reset" is teleop-driven too (drive back into the initial state by hand, then space
    # starts "policy"; on success there, teleop back up in "reset" again) -- so, like
    # teleop_left/teleop_midpoint, it has learnable axes but no policy attached, and
    # build_adaptive_registry (which requires both is_adaptive and policy is not None) skips
    # it. Only "policy" has both, so only it is picked up for HIL-RL.
    assert cfg.primitives["teleop_left"].policy is None
    assert cfg.primitives["calibrate_pivot_offset_x"].policy is None
    assert cfg.primitives["reset"].is_adaptive
    assert cfg.primitives["reset"].policy is None
    assert cfg.primitives["policy"].is_adaptive
    assert cfg.primitives["policy"].policy is not None
    # start_primitive/reset_primitive stay at teleop_left: only the very first full reset of a
    # run goes there -- reset->policy->reset is a plain loop, not a second full reset.
    assert cfg.start_primitive == "teleop_left"
    assert cfg.reset_primitive == "teleop_left"


def test_bimanual_pick_config_with_known_pivot_offset_x_skips_calibration():
    """A known-good pivot_offset_x drops calibrate_pivot_offset_x out of the graph entirely
    and seeds every cooperative primitive with it directly."""
    cfg = BimanualPickEnvConfig(pivot_offset_x=0.037)

    assert list(cfg.primitives) == ["teleop_left", "teleop_right", "teleop_midpoint", "reset", "policy"]
    assert [(t.source, t.target, type(t).__name__) for t in cfg.transitions] == [
        ("teleop_left", "teleop_right", "OnSuccess"),
        ("teleop_right", "teleop_midpoint", "OnSuccess"),
        ("teleop_midpoint", "reset", "OnSuccess"),
        ("reset", "policy", "OnSuccess"),
        ("policy", "reset", "OnSuccess"),
    ]
    for name in ("teleop_midpoint", "reset", "policy"):
        assert cfg.primitives[name].env_kwargs["pivot_offset_x"] == pytest.approx(0.037)

def test_teleop_left_moves_sim_robot_through_the_real_pipeline():
    """SimUR driven through the actual .make()-built processor pipeline, not a hand-rolled
    harness -- confirms the sim robot behaves inside the real action/env processors too."""
    cfg = BimanualPickEnvConfig(mock=True)
    primitive = cfg.primitives["teleop_left"]

    left = SimUR(SimURConfig(use_gripper=True, cameras={}))
    right = SimUR(SimURConfig(use_gripper=True, cameras={}))
    for robot in (left, right):
        robot.connect()

    teleop = MockTeleoperator(name="left", is_delta=True)
    teleop.get_action = lambda: {**{k: 0.25 for k in teleop._features}, "gripper.pos": 0.5}
    env, env_processor, action_processor = primitive.make(
        robot_dict={"left": left, "right": right},
        teleop_dict={"left": teleop},
        cameras={},
    )
    primitive.on_entry(env, None)

    start_x = left.get_observation()["x.ee_pos"]
    for _ in range(10):
        transition = create_transition(
            action={"left": {}, "right": {}},
            observation=env._get_observation(),
            info={TeleopEvents.IS_INTERVENTION: True},
        )
        processed = action_processor(transition)
        env.step(processed[TransitionKey.ACTION])

    assert left.get_observation()["x.ee_pos"] != pytest.approx(start_x)


def test_origin_from_raw_obs_prefers_reported_origin_over_fallback():
    """Every robot in this primitive keeps task-frame origin=[0]*6 for its whole lifetime, so
    this should never fire in practice -- but it's the only thing standing between a stale
    controller-side origin and a silently wrong pose if that ever changes."""
    obs = dict(zip(("x", "y", "z", "rx", "ry", "rz"), range(6)))
    for ax, value in zip(("x", "y", "z", "rx", "ry", "rz"), (1.0, 2.0, 3.0, 0.0, 0.0, 0.0)):
        obs[f"{ax}.task_frame_origin"] = value
    assert _origin_from_raw_obs(obs, fallback=[9.0] * 6) == [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]


def test_origin_from_raw_obs_falls_back_when_absent():
    assert _origin_from_raw_obs({}, fallback=[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]) == [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _task_frame(policy_mode) -> TaskFrame:
    return TaskFrame(
        target=[0.0] * 6,
        control_mode=[ControlMode.POS] * 6,
        policy_mode=policy_mode,
    )


def test_cooperative_frame_relates_right_base_to_left_base_correctly():
    """Both TCPs are actually gripping the same physical point in space: left's base defines
    world, right's base sits 1m away along x. Right's own-native reading of that shared point
    (-0.5) differs from left's (0.5) purely because of the base offset -- the primitive must
    account for that via robot_base_pose_in_world, not average the two raw readings as if they
    were already in the same frame (that was the actual bug: it silently dropped the offset and
    commanded the follower toward a point roughly one base-separation away from reality)."""
    left = _FakeRobot([0.5, 0.0, 0.8, 0.0, 0.0, 0.0])
    right = _FakeRobot([-0.5, 0.0, 0.8, 0.0, 0.0, 0.0])
    task_frame = {
        "left": _task_frame([PolicyMode.RELATIVE] * 6),
        "right": _task_frame([None] * 6),
    }
    env = CooperativeFramePrimitive(
        task_frame,
        {"left": left, "right": right},
        {},
        driver="left",
        fps=30.0,
        robot_base_pose_in_world={"right": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    )

    env.step({"left": {}})

    # Driver reports its own absolute pose too, same as the follower -- nothing has moved yet.
    assert left.last_action["x.ee_pos"] == pytest.approx(0.5, abs=1e-9)
    assert right.last_action["x.ee_pos"] == pytest.approx(-0.5, abs=1e-9)


def test_cooperative_frame_first_step_holds_current_poses():
    """With zero teleop delta, the midpoint hasn't moved -- both arms hold their entry pose."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # in right's own base frame
    task_frame = {
        "left": _task_frame([PolicyMode.RELATIVE] * 6),
        "right": _task_frame([None] * 6),
    }
    env = CooperativeFramePrimitive(
        task_frame,
        {"left": left, "right": right},
        {},
        driver="left",
        fps=30.0,
        robot_base_pose_in_world={"right": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    )

    env.step({"left": {}})

    assert left.last_action["x.ee_pos"] == pytest.approx(0.0, abs=1e-9)
    assert right.last_action["x.ee_pos"] == pytest.approx(0.0, abs=1e-9)
    assert right.last_action["z.ee_pos"] == pytest.approx(1.0, abs=1e-9)


def test_cooperative_frame_driver_translation_moves_both_arms_together():
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    task_frame = {
        "left": _task_frame([PolicyMode.RELATIVE] * 6),
        "right": _task_frame([None] * 6),
    }
    fps = 30.0
    env = CooperativeFramePrimitive(
        task_frame,
        {"left": left, "right": right},
        {},
        driver="left",
        fps=fps,
        robot_base_pose_in_world={"right": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    )
    env.step({"left": {}})  # captures the initial midpoint offsets

    dz = 0.6  # m/s along z
    env.step({"left": {"z.ee_pos": dz}})

    expected_world_dz = dz / fps
    # follower (right) reports an absolute target in its own base frame; x is unaffected by a
    # pure z move regardless of the 1m x base offset
    assert right.last_action["z.ee_pos"] == pytest.approx(1.0 + expected_world_dz, abs=1e-9)
    assert right.last_action["x.ee_pos"] == pytest.approx(0.0, abs=1e-9)
    # driver (left) reports the same kind of absolute target as the follower, not a velocity
    assert left.last_action["z.ee_pos"] == pytest.approx(1.0 + expected_world_dz, abs=1e-9)
    assert left.last_action["x.ee_pos"] == pytest.approx(0.0, abs=1e-9)


def test_cooperative_frame_keeps_fixed_grasp_offset_between_arms():
    """The arms' relative offset (grasp geometry) must stay constant as the midpoint moves."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.2, 0.0, 1.0, 0.0, 0.0, 0.0])  # 0.2m ahead of left along x, own frame
    task_frame = {
        "left": _task_frame([PolicyMode.RELATIVE] * 6),
        "right": _task_frame([None] * 6),
    }
    fps = 30.0
    env = CooperativeFramePrimitive(
        task_frame,
        {"left": left, "right": right},
        {},
        driver="left",
        fps=fps,
        robot_base_pose_in_world={"right": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    )
    env.step({"left": {}})

    for _ in range(5):
        env.step({"left": {"x.ee_pos": 0.3}})

    world_left_dx = 0.3 * 5 / fps
    # a pure x translation with an x-only base offset shifts right's own-frame x by the same
    # world delta the driver applied, regardless of the base offset (it cancels out)
    assert right.last_action["x.ee_pos"] == pytest.approx(world_left_dx + 0.2, abs=1e-6)


def test_cooperative_frame_rotation_swings_arms_about_the_midpoint():
    """A yaw rate about the shared midpoint should swing both arms along a circle."""
    left = _FakeRobot([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])  # same shared frame: no base offset
    task_frame = {
        "left": _task_frame([PolicyMode.RELATIVE] * 6),
        "right": _task_frame([None] * 6),
    }
    fps = 30.0
    env = CooperativeFramePrimitive(task_frame, {"left": left, "right": right}, {}, driver="left", fps=fps)
    env.step({"left": {}})

    drz = 3.0  # rad/s yaw about the midpoint
    env.step({"left": {"rz.ee_pos": drz}})

    theta = drz / fps
    assert right.last_action["x.ee_pos"] == pytest.approx(0.1 * math.cos(theta), abs=1e-6)
    assert right.last_action["y.ee_pos"] == pytest.approx(0.1 * math.sin(theta), abs=1e-6)
    assert right.last_action["rz.ee_pos"] == pytest.approx(0.0, abs=1e-6)
    # driver's own reported absolute target, mirrored about the midpoint, reaches the same yaw
    assert left.last_action["x.ee_pos"] == pytest.approx(-0.1 * math.cos(theta), abs=1e-6)
    assert left.last_action["y.ee_pos"] == pytest.approx(-0.1 * math.sin(theta), abs=1e-6)
    assert left.last_action["rz.ee_pos"] == pytest.approx(theta, abs=1e-6)


def test_cooperative_frame_follower_keeps_ry_when_driver_locks_rx_and_rz():
    """A passive follower still follows the driver's live ry rotation."""
    left = _FakeRobot([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    task_frame = {
        "left": TaskFrame(
            target=[0.0, 0.0, 0.0, np.pi, 0.0, np.pi],
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[PolicyMode.RELATIVE, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, PolicyMode.RELATIVE, None],
        ),
        "right": TaskFrame(
            target=[0.0, 0.0, 0.0, np.pi, 0.0, 0.0],
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[None] * 6,
        ),
    }
    fps = 30.0
    env = CooperativeFramePrimitive(task_frame, {"left": left, "right": right}, {}, driver="left", fps=fps)
    env.step({"left": {}})

    env.step({"left": {"ry.ee_pos": 1.0}})

    assert right.last_action["ry.ee_pos"] != pytest.approx(0.0)
    assert right.last_action["rx.ee_pos"] == pytest.approx(np.pi)
    assert right.last_action["rz.ee_pos"] == pytest.approx(0.0)


def test_driver_task_frame_policy_mode_none_locks_an_axis_regardless_of_live_input():
    """A DOF restriction for the cooperative driver (e.g. free xyz, rotation restricted to y
    only for an insertion task) needs no support from CooperativeFramePrimitive at all --
    set policy_mode=None on the disabled axes directly in the driver's TaskFrame.
    InterventionActionProcessorStep._project_policy_action seeds every axis from the frame's
    static target and only overwrites the ones in learnable_axis_indices (policy_mode is not
    None) with live teleop/policy input, so a locked axis stays at its static value (0.0,
    never touched by CooperativeFramePrimitive) no matter what the teleop reports. This is
    the mechanism the class docstring's DOF-restriction guidance relies on -- this test
    guards that claim directly, independent of which env_class ends up using it."""
    task_frame = {
        "left": TaskFrame(
            target=[0.0] * 6,
            control_mode=[ControlMode.POS] * 6,
            # x, y, z, ry live; rx, rz locked -- the exact insertion-task restriction
            policy_mode=[PolicyMode.RELATIVE, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, PolicyMode.RELATIVE, None],
        ),
    }
    cfg = ManipulationPrimitiveConfig(task_frame=task_frame)
    robot = MockRobot(name="left")
    teleop = MockTeleoperator(name="left", is_delta=True)
    # large, distinct values on every axis, keyed by the teleop's own feature convention
    teleop.get_action = lambda: {
        "delta_x": 0.3, "delta_y": -0.2, "delta_z": 0.1, "delta_rx": 5.0, "delta_ry": 2.0, "delta_rz": 5.0,
    }

    env, env_processor, action_processor = cfg.make(
        robot_dict={"left": robot}, teleop_dict={"left": teleop}, cameras={},
    )
    transition = create_transition(
        action={"left": {}},
        observation=env._get_observation(),
        info={TeleopEvents.IS_INTERVENTION: True},
    )
    left_action = action_processor(transition)[TransitionKey.ACTION]["left"]

    assert left_action["x.ee_pos"] == pytest.approx(0.3)
    assert left_action["y.ee_pos"] == pytest.approx(-0.2)
    assert left_action["z.ee_pos"] == pytest.approx(0.1)
    assert left_action["ry.ee_pos"] == pytest.approx(2.0)
    assert left_action["rx.ee_pos"] == pytest.approx(0.0)
    assert left_action["rz.ee_pos"] == pytest.approx(0.0)


def test_cooperative_frame_picks_up_pivot_offset_x_from_shared_runtime_on_capture():
    """A pivot_offset_x measured by an earlier calibration primitive (or nudged live in a
    previously-active cooperative primitive) overrides this instance's own configured
    starting value the moment it first activates."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    task_frame = {
        "left": _task_frame([PolicyMode.RELATIVE] * 6),
        "right": _task_frame([None] * 6),
    }
    env = CooperativeFramePrimitive(
        task_frame,
        {"left": left, "right": right},
        {},
        driver="left",
        fps=30.0,
        robot_base_pose_in_world={"right": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        pivot_offset_x=0.0,
    )
    env.attach_shared_runtime_values({PIVOT_OFFSET_X_RUNTIME_KEY: 0.4})

    env.step({"left": {}})

    assert env.pivot_offset_x == pytest.approx(0.4)
    # zero driver delta -> both arms hold their current pose, round-tripping correctly
    # through whatever offset is in play.
    assert left.last_action["x.ee_pos"] == pytest.approx(0.0, abs=1e-9)
    assert right.last_action["x.ee_pos"] == pytest.approx(0.0, abs=1e-9)


def test_cooperative_frame_rotates_about_offset_pivot_not_raw_midpoint():
    """A calibrated x-offset elsewhere in space, not the natural arm midpoint, must be what
    rotation actually swings around."""
    left = _FakeRobot([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    task_frame = {
        "left": _task_frame([PolicyMode.RELATIVE] * 6),
        "right": _task_frame([None] * 6),
    }
    fps = 30.0
    env = CooperativeFramePrimitive(task_frame, {"left": left, "right": right}, {}, driver="left", fps=fps)
    pivot_offset_x = 1.0  # 1m off to the side of both arms, not their raw midpoint (x=0)
    env.attach_shared_runtime_values({PIVOT_OFFSET_X_RUNTIME_KEY: pivot_offset_x})
    env.step({"left": {}})

    drz = 3.0
    env.step({"left": {"rz.ee_pos": drz}})

    theta = drz / fps
    pivot = [pivot_offset_x, 0.0]
    dx0, dy0 = 0.1 - pivot[0], 0.0 - pivot[1]
    expected_x = pivot[0] + dx0 * math.cos(theta) - dy0 * math.sin(theta)
    expected_y = pivot[1] + dx0 * math.sin(theta) + dy0 * math.cos(theta)
    assert right.last_action["x.ee_pos"] == pytest.approx(expected_x, abs=1e-6)
    assert right.last_action["y.ee_pos"] == pytest.approx(expected_y, abs=1e-6)


def test_cooperative_frame_captures_offset_from_vtcp_once_and_reuses_it():
    """The grasp geometry between robots must be captured once and reused, not re-derived
    every activation -- otherwise a robot's small compliant give under contact gets baked in
    as the new "true" offset next time, and the relative pose between the two arms creeps a
    little further every loop iteration (the actual bug this guards against).

    left (the driver) stays put; right drifts 5cm closer between activations, simulating a
    slight compliant give under contact. midpoint legitimately still reacts to the current
    (drifted) poses each activation -- only the *offset from that midpoint* must stay frozen.
    First activation: midpoint=(0.0+0.2)/2=0.1, so right's offset is captured as 0.2-0.1=0.1.
    Second activation, offset correctly reused: midpoint=(0.0+0.15)/2=0.075, target=
    0.075+0.1=0.175. Had the offset instead been re-derived fresh from the drifted pose (the
    bug), it would recover right's raw current pose exactly -- 0.15, not 0.175 -- with no
    protection against the drift compounding on every subsequent loop iteration."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.2, 0.0, 1.0, 0.0, 0.0, 0.0])
    task_frame = {
        "left": _task_frame([PolicyMode.RELATIVE] * 6),
        "right": _task_frame([None] * 6),
    }
    shared_runtime_values = {}

    first = CooperativeFramePrimitive(task_frame, {"left": left, "right": right}, {}, driver="left", fps=30.0)
    first.attach_shared_runtime_values(shared_runtime_values)
    first.step({"left": {}})

    assert shared_runtime_values[OFFSET_FROM_VTCP_RUNTIME_KEY]["right"][0] == pytest.approx(0.1, abs=1e-6)

    # Simulate compliant give: right's actual pose drifts closer to left before the next
    # activation (e.g. a slight give under contact during cooperative_insert).
    right.pose = [0.15, 0.0, 1.0, 0.0, 0.0, 0.0]

    second = CooperativeFramePrimitive(task_frame, {"left": left, "right": right}, {}, driver="left", fps=30.0)
    second.attach_shared_runtime_values(shared_runtime_values)
    second.step({"left": {}})

    # Offset unchanged -- reused, not re-derived from the drifted pose.
    assert shared_runtime_values[OFFSET_FROM_VTCP_RUNTIME_KEY]["right"][0] == pytest.approx(0.1, abs=1e-6)
    assert right.last_action["x.ee_pos"] == pytest.approx(0.175, abs=1e-6)


def test_cooperative_frame_freezes_a_newly_locked_axis_at_its_actual_entry_value():
    """freeze_driver_axes_at_entry holds whatever the driver actually measures on capture,
    not the driver's static configured target -- e.g. demoting an axis (like rail_bimanual_
    grasp.py's ry in cooperative_insert) from live to locked without hardcoding an assumed
    value for it: whatever the operator left it at during the preceding manual primitive is
    what gets held."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.5, 0.0])  # actual ry = 0.5, far from target's 0.0
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    task_frame = {
        "left": TaskFrame(
            target=[0.0] * 6,  # static target's ry = 0.0
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[PolicyMode.RELATIVE, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, None, None],
        ),
        "right": _task_frame([None] * 6),
    }
    env = CooperativeFramePrimitive(
        task_frame, {"left": left, "right": right}, {}, driver="left", fps=30.0,
        freeze_driver_axes_at_entry=(4,),
    )

    env.step({"left": {}})

    assert env._vtcp_world[4] == pytest.approx(0.5)
    # Zero driver delta -> the driver still just holds its current (actual) pose, exactly like
    # any other capture -- freezing ry doesn't perturb anything on the very first step. Sent
    # as the frozen/reprojected value (0.5), NOT the static config target (0.0): regression
    # guard for a real bug where the driver's own frame.target[axis] override silently
    # clobbered the freeze right after computing it correctly.
    assert left.last_action["ry.ee_pos"] == pytest.approx(0.5, abs=1e-9)


def test_cooperative_frame_freeze_does_not_clobber_the_follower_either():
    """The same clobbering bug affected the follower too: whenever a freeze_driver_axes_at_entry
    axis is also in the driver's own locked set, follower_fixed_rotation_axes includes it, and
    the override used to blindly overwrite the follower's own already-correct reprojection with
    the static target as well."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.5, 0.0])  # actual ry = 0.5
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.5, 0.0])  # same ry, symmetric setup
    task_frame = {
        "left": TaskFrame(
            target=[0.0] * 6,
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[PolicyMode.RELATIVE, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, None, None],
        ),
        "right": _task_frame([None] * 6),
    }
    env = CooperativeFramePrimitive(
        task_frame, {"left": left, "right": right}, {}, driver="left", fps=30.0,
        freeze_driver_axes_at_entry=(4,),
    )

    env.step({"left": {}})

    # Symmetric setup (both arms at the same actual ry, zero driver delta) -> both should hold
    # their own actual pose, not snap to the static target's ry=0.0.
    assert right.last_action["ry.ee_pos"] == pytest.approx(0.5, abs=1e-9)


def test_cooperative_frame_driver_axis_override_uses_the_supplied_value_not_the_measured_pose():
    """driver_axis_overrides supplies an exact, reproducible value instead of reading each
    robot's actual (possibly drifted/mis-homed) measured pose -- for a caller that wants the
    same numeric ry every run rather than whatever a fresh grasp happens to land at. Both
    robots must converge on the override, not just the driver: pose_world is one shared world
    frame, so the override means the same declared value for everyone in it -- ry is
    nominally 0.0 for both arms here (only rz differs, mirrored), exactly like live ry teleop
    already rotates both arms together through this same offset+vtcp mechanism.

    Regression guard for a real bug (found twice): capturing a robot's offset_from_vtcp from
    its *raw* measured pose against a vtcp deliberately overridden to a different value
    encodes exactly that mismatch into the offset -- target_world's later offset+vtcp
    recomposition then silently reconstructs the original measured value, undoing the
    override for that robot's own commanded action, even though _vtcp_world itself correctly
    held the override the whole time. First found and fixed for the driver alone; the
    follower had the exact same bug (left correctly snapped to the override, right silently
    kept holding its own old measured value) until this test caught it too."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.9, 0.0])  # actual ry = 0.9, deliberately "wrong"
    right = _FakeRobot([0.2, 0.0, 1.0, 0.0, 0.4, 0.0])  # actual ry = 0.4, also "wrong", differently
    task_frame = {
        "left": TaskFrame(
            target=[0.0] * 6,
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[PolicyMode.RELATIVE, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, None, None],
        ),
        "right": _task_frame([None] * 6),
    }
    env = CooperativeFramePrimitive(
        task_frame, {"left": left, "right": right}, {}, driver="left", fps=30.0,
        freeze_driver_axes_at_entry=(4,),
        driver_axis_overrides={4: 0.25},
    )

    env.step({"left": {}})

    assert env._vtcp_world[4] == pytest.approx(0.25)
    assert left.last_action["ry.ee_pos"] == pytest.approx(0.25, abs=1e-9)
    assert right.last_action["ry.ee_pos"] == pytest.approx(0.25, abs=1e-9)
    # Translation is untouched by the rotation override -- the follower still holds its own
    # actual x/y (nothing has moved yet on this same capturing step, exactly like the
    # ordinary "first step holds current poses" case).
    assert right.last_action["x.ee_pos"] == pytest.approx(0.2, abs=1e-9)
    assert right.last_action["y.ee_pos"] == pytest.approx(0.0, abs=1e-9)


def test_cooperative_frame_without_freeze_uses_the_static_target_for_a_locked_axis():
    """Sanity check for the test above: with freeze_driver_axes_at_entry left at its default
    (empty), a locked axis falls back to the driver's static configured target exactly as
    before this feature existed."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.5, 0.0])  # actual ry = 0.5, static target's is 0.0
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    task_frame = {
        "left": TaskFrame(
            target=[0.0] * 6,
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[PolicyMode.RELATIVE, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, None, None],
        ),
        "right": _task_frame([None] * 6),
    }
    env = CooperativeFramePrimitive(task_frame, {"left": left, "right": right}, {}, driver="left", fps=30.0)

    env.step({"left": {}})

    assert env._vtcp_world[4] == pytest.approx(0.0)


def test_cooperative_frame_driver_live_axis_sends_a_reprojected_target_not_a_velocity():
    """Every robot, driver included, is always commanded with an absolute reprojected target
    -- there is no velocity/RELATIVE path at the controller for anyone, even on the driver's
    own live axes. The axis still moves from teleop input via _vtcp_world, it's just reported
    as a position, not a raw rad/s value."""
    left = _FakeRobot([0.0] * 6)
    right = _FakeRobot([0.0] * 6)
    task_frame = {
        "left": TaskFrame(
            target=[0.0] * 6,
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[None, None, None, None, PolicyMode.RELATIVE, None],
        ),
        "right": _task_frame([None] * 6),
    }
    fps = 30.0
    env = CooperativeFramePrimitive(task_frame, {"left": left, "right": right}, {}, driver="left", fps=fps)
    env.step({"left": {}})  # captures

    dry = 3.0  # rad/s, teleop input on the live axis
    env.step({"left": {"ry.ee_pos": dry}})

    # An absolute angle (~dry/fps, small) was sent, not the raw rad/s velocity (dry, large) --
    # the axis still moved (it's not frozen/static), just reported differently.
    assert left.last_action["ry.ee_pos"] == pytest.approx(dry / fps, abs=1e-6)
    assert abs(left.last_action["ry.ee_pos"]) < abs(dry)


def test_cooperative_frame_apply_task_frames_reports_the_driver_as_fully_absolute():
    """apply_task_frames() always reports the driver's whole task frame as ABSOLUTE
    (policy_mode=None on every axis) to its own controller via a per-step copy --
    self.task_frame[driver] itself, and therefore the action processor's live-input
    passthrough, is untouched. The follower's frame is sent unmodified (already ABSOLUTE)."""
    left = _FakeRobot([0.0] * 6)
    right = _FakeRobot([0.0] * 6)
    left_frame = TaskFrame(
        target=[0.0] * 6,
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[PolicyMode.RELATIVE, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, PolicyMode.RELATIVE, None],
    )
    right_frame = _task_frame([None] * 6)
    task_frame = {"left": left_frame, "right": right_frame}
    env = CooperativeFramePrimitive(task_frame, {"left": left, "right": right}, {}, driver="left", fps=30.0)

    env.apply_task_frames()

    assert left.last_task_frame is not left_frame
    assert left.last_task_frame.policy_mode == [None] * 6
    # self.task_frame[driver] itself is untouched -- the action processor still sees it live.
    assert left_frame.policy_mode == [
        PolicyMode.RELATIVE, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, PolicyMode.RELATIVE, None,
    ]
    # The follower was already fully ABSOLUTE, so it's sent as the same object, unmodified.
    assert right.last_task_frame is right_frame


def _reference_limited_overrides(wrench_limit: float, kp: float) -> dict:
    return {
        "compliance_reference_limit_enable": [True] * 6,
        "wrench_limits": [wrench_limit] * 6,
        "kp": [kp] * 6,
    }


def test_cooperative_frame_clamps_vtcp_to_the_drivers_own_reference_budget():
    """A driver that never physically moves (simulating a blocked/lagging arm) must not have
    its own commanded target run away from its actual measured pose beyond
    wrench_limits/kp -- the same budget controller.py's compliance_reference_limit_enable
    would give it directly, just enforced on the shared vtcp instead."""
    left = _FakeRobot([0.0] * 6)  # pose never updates -- simulates a fully blocked arm
    right = _FakeRobot([0.0] * 6)
    budget = 10.0 / 100.0  # wrench_limits/kp = 0.1 m
    task_frame = {
        "left": TaskFrame(
            target=[0.0] * 6,
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[PolicyMode.RELATIVE] * 6,
            controller_overrides=_reference_limited_overrides(wrench_limit=10.0, kp=100.0),
        ),
        "right": _task_frame([None] * 6),
    }
    fps = 30.0
    env = CooperativeFramePrimitive(task_frame, {"left": left, "right": right}, {}, driver="left", fps=fps)
    env.step({"left": {}})  # captures

    for _ in range(50):
        env.step({"left": {"x.ee_pos": 1.0}})  # would integrate to ~1.67 m unclamped

    assert left.last_action["x.ee_pos"] == pytest.approx(budget, abs=1e-6)

    # Releasing input doesn't cause a jump -- the vtcp was already sitting at the budget edge.
    env.step({"left": {}})
    assert left.last_action["x.ee_pos"] == pytest.approx(budget, abs=1e-6)


def test_cooperative_frame_reference_clamp_uses_the_most_restrictive_robot():
    """The follower's own (tighter) budget must bound the shared vtcp too, even though the
    driver's own budget alone would allow more -- clamping only the driver's target and
    letting the follower's run further ahead would pull the two robots' targets apart again,
    exactly what commanding both off one shared vtcp was meant to prevent."""
    left = _FakeRobot([0.0] * 6)
    right = _FakeRobot([0.0] * 6)  # also never moves -- its own budget is the binding one
    task_frame = {
        "left": TaskFrame(
            target=[0.0] * 6,
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[PolicyMode.RELATIVE] * 6,
            controller_overrides=_reference_limited_overrides(wrench_limit=10.0, kp=10.0),  # budget 1.0 m
        ),
        "right": TaskFrame(
            target=[0.0] * 6,
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[None] * 6,
            controller_overrides=_reference_limited_overrides(wrench_limit=1.0, kp=100.0),  # budget 0.01 m
        ),
    }
    fps = 30.0
    env = CooperativeFramePrimitive(task_frame, {"left": left, "right": right}, {}, driver="left", fps=fps)
    env.step({"left": {}})  # captures

    for _ in range(50):
        env.step({"left": {"x.ee_pos": 1.0}})

    # Bound by the follower's tighter 0.01 m budget, not the driver's own looser 1.0 m one.
    assert left.last_action["x.ee_pos"] == pytest.approx(0.01, abs=1e-6)
    assert right.last_action["x.ee_pos"] == pytest.approx(0.01, abs=1e-6)


def test_cooperative_frame_without_controller_overrides_reference_limiting_is_a_noop():
    """No wrench_limits/kp/compliance_reference_limit_enable configured -- exactly today's
    default env configs -- means no budget is known, so nothing gets clamped, unchanged from
    before this feature existed."""
    left = _FakeRobot([0.0] * 6)
    right = _FakeRobot([0.0] * 6)
    task_frame = {
        "left": _task_frame([PolicyMode.RELATIVE] * 6),
        "right": _task_frame([None] * 6),
    }
    fps = 30.0
    env = CooperativeFramePrimitive(task_frame, {"left": left, "right": right}, {}, driver="left", fps=fps)
    env.step({"left": {}})

    for _ in range(50):
        env.step({"left": {"x.ee_pos": 1.0}})

    assert left.last_action["x.ee_pos"] == pytest.approx(1.0 * 50 / fps, abs=1e-6)


def test_pivot_offset_x_calibration_locks_out_spacemouse_axes():
    """The driver's policy_mode is all None here -- InterventionActionProcessorStep's
    _project_policy_action seeds every axis from the frame's static target regardless of
    what the teleoperator reports (see the DOF-restriction test above), so a nonzero action
    on the driver's x is simply ignored: only the ','/'.' keyboard nudge can move the pivot
    during calibration."""
    task_frame = {"left": _task_frame([None] * 6), "right": _task_frame([None] * 6)}
    env = PivotOffsetXCalibrationPrimitive(
        task_frame,
        {"left": _FakeRobot([0.0] * 6), "right": _FakeRobot([0.0] * 6)},
        {},
        driver="left",
        fps=30.0,
    )

    env.step({"left": {"x.ee_pos": 5.0}})

    assert env._vtcp_world[0] == pytest.approx(0.0)


def test_pivot_offset_x_calibration_reports_the_live_value_every_step():
    task_frame = {"left": _task_frame([None] * 6), "right": _task_frame([None] * 6)}
    env = PivotOffsetXCalibrationPrimitive(
        task_frame,
        {"left": _FakeRobot([0.0] * 6), "right": _FakeRobot([0.0] * 6)},
        {},
        driver="left",
        fps=30.0,
        pivot_offset_x=0.123,
    )

    _obs, _reward, _terminated, _truncated, info = env.step({"left": {}})

    assert info["record_status"] == "pivot_offset_x = +0.1230 m  (,/. to nudge, space to lock)"


def test_pivot_offset_x_calibration_nudge_publishes_to_shared_runtime_immediately():
    """The whole point of publishing every step (not just on capture) is that whatever is
    last nudged/printed here is exactly what teleop_midpoint picks up on its own first step,
    without the two primitives sharing a Python instance."""
    task_frame = {"left": _task_frame([None] * 6), "right": _task_frame([None] * 6)}
    env = PivotOffsetXCalibrationPrimitive(
        task_frame,
        {"left": _FakeRobot([0.0] * 6), "right": _FakeRobot([0.0] * 6)},
        {},
        driver="left",
        fps=30.0,
        pivot_offset_x_step=0.005,
    )
    env.attach_shared_runtime_values({})
    env.step({"left": {}})  # captures the initial pivot

    env._pivot_offset_x_event._on_key_press(keyboard.KeyCode.from_char("."))
    env.step({"left": {}})
    env._pivot_offset_x_event._on_key_release(keyboard.KeyCode.from_char("."))

    assert env.pivot_offset_x == pytest.approx(0.005)
    assert env.get_runtime_value(PIVOT_OFFSET_X_RUNTIME_KEY) == pytest.approx(0.005)
