import json
import os

os.environ.setdefault("PYNPUT_BACKEND", "dummy")

import pytest

from experiments.envs.rail_bimanual_grasp import (
    COOPERATIVE_INSERT_LEFT_POLICY,
    COOPERATIVE_LEFT_POLICY,
    DEFAULT_GRASP_ORIENTATION_RPY,
    DEFAULT_RAIL_Y_M,
    RECALIBRATE_RY_KEY,
    RECALIBRATE_RY_REQUEST_FLAG,
    CooperativeFramePrimitive,
    CooperativeInsertPrimitive,
    FROZEN_AXES_RUNTIME_KEY,
    RailBimanualGraspEnvConfig,
    CalibratePrimitive,
    XZ_RELATIVE_POLICY,
)
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    MoveDeltaPrimitiveConfig,
    PrimitiveEntryContext,
    ZeroFTPrimitiveConfig,
)
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.envs.manipulation_primitive_net.transitions import OnEvent, OnSuccess, OnTargetPoseReached, OnTimeLimit, RewardClassifierTransition
from share.utils.mock_utils import MockRobot


def _entry_context() -> PrimitiveEntryContext:
    observation = {}
    for name, x, y, z in (
        ("left", 0.2, -0.1, 0.4),
        ("right", -0.3, -0.2, 0.5),
    ):
        values = [x, y, z, 0.1, -0.2, 0.3]
        for axis, value in zip(("x", "y", "z", "rx", "ry", "rz"), values):
            observation[f"{name}.{axis}.ee_pos"] = value
    return PrimitiveEntryContext(
        observation=observation,
        task_frame_origin={"left": [0.0] * 6, "right": [0.0] * 6},
    )


def _robot_dict() -> dict:
    return {
        "left": MockRobot(name="left", is_task_frame=True),
        "right": MockRobot(name="right", is_task_frame=True),
    }


def test_rail_env_graph_and_task_frame_contract():
    cfg = RailBimanualGraspEnvConfig(mock=True)

    assert list(cfg.primitives) == [
        "align_to_rail",
        "teleop_left",
        "teleop_right",
        "zero_ft",
        "calibrate",
        "zero_ft_before_cooperative",
        "cooperative_reset",
        "cooperative_insert",
        "pull_out",
        "pushdown",
    ]
    assert isinstance(cfg.primitives["zero_ft"], ZeroFTPrimitiveConfig)
    assert cfg.primitives["zero_ft"].settle_duration_s == pytest.approx(cfg.zero_ft_settle_duration_s)
    assert isinstance(cfg.primitives["align_to_rail"], MoveDeltaPrimitiveConfig)
    assert cfg.alignment_linear_speed_mps == pytest.approx(0.05)
    for name, frame in cfg.primitives["align_to_rail"].task_frame.items():
        assert frame.target[1] == pytest.approx(DEFAULT_RAIL_Y_M[name])
        assert frame.target[3:6] == pytest.approx(DEFAULT_GRASP_ORIENTATION_RPY[name])
        assert frame.policy_mode == [None] * 6

    assert cfg.primitives["teleop_left"].task_frame["left"].policy_mode == XZ_RELATIVE_POLICY
    assert cfg.primitives["teleop_right"].task_frame["right"].policy_mode == XZ_RELATIVE_POLICY
    assert cfg.primitives["teleop_right"].teleop_mapping == {"right": "main"}
    assert cfg.primitives["calibrate"].env_class is CalibratePrimitive
    # Regression: without this mapping, the driver ("left") never resolves a teleoperator at
    # all -- self.teleop is a single SpaceMouseConfig, wrapped under DEFAULT_ROBOT_NAME
    # ("main") by ManipulationPrimitiveNetConfig, not under "left" -- so the SpaceMouse would
    # silently produce zero motion in calibrate, exactly like every other primitive
    # here that drives "left" needs the same mapping (see teleop_right's "right": "main" above).
    assert cfg.primitives["calibrate"].teleop_mapping == {"left": "main"}
    # calibrate is the only primitive that still exposes ry live -- that's the entire
    # calibration: teleop it to the right value, then space moves on.
    assert cfg.primitives["calibrate"].task_frame["left"].policy_mode == COOPERATIVE_LEFT_POLICY
    # cooperative_reset and cooperative_insert are both translation-only: x and ry (indices 0
    # and 4) are locked in both, frozen to whatever they measure on entry rather than a
    # hardcoded target -- they differ only in whether a policy is attached.
    for name in ("cooperative_reset", "cooperative_insert"):
        assert cfg.primitives[name].task_frame["left"].policy_mode == COOPERATIVE_INSERT_LEFT_POLICY
        assert cfg.primitives[name].task_frame["right"].policy_mode == [None] * 6
        assert cfg.primitives[name].env_kwargs["freeze_driver_axes_at_entry"] == (0, 4)
    assert cfg.primitives["cooperative_reset"].policy is None
    assert cfg.primitives["cooperative_insert"].policy is not None

    for name in ("teleop_left", "teleop_right", "zero_ft", "calibrate", "zero_ft_before_cooperative", "cooperative_reset", "cooperative_insert"):
        for robot_name, frame in cfg.primitives[name].task_frame.items():
            assert frame.target[1] == pytest.approx(DEFAULT_RAIL_Y_M[robot_name])
            assert frame.target[3:6] == pytest.approx(DEFAULT_GRASP_ORIENTATION_RPY[robot_name])

    assert isinstance(cfg.transitions[0], OnTargetPoseReached)
    assert [(edge.source, edge.target) for edge in cfg.transitions] == [
        ("align_to_rail", "teleop_left"),
        ("teleop_left", "teleop_right"),
        ("teleop_right", "zero_ft"),
        ("zero_ft", "calibrate"),
        ("calibrate", "zero_ft_before_cooperative"),
        ("zero_ft_before_cooperative", "cooperative_reset"),
        ("cooperative_reset", "calibrate"),  # OnEvent -- up arrow, re-teleop ry
        ("cooperative_reset", "cooperative_insert"),
        ("cooperative_insert", "cooperative_reset"),  # OnTimeLimit -- no pull_out on timeout
        ("cooperative_insert", "pull_out"),  # OnSuccess -- demo=False (default)
        ("pull_out", "cooperative_reset"),
        ("pushdown", "cooperative_reset"),  # OnSuccess -- only reachable via demo=True
    ]
    for name in ("zero_ft", "zero_ft_before_cooperative"):
        edge = next(e for e in cfg.transitions if e.source == name)
        assert isinstance(edge, OnEvent)
    time_limit_edge = cfg.transitions[-4]
    assert isinstance(time_limit_edge, OnTimeLimit)
    assert time_limit_edge.max_steps == int(cfg.fps * cfg.cooperative_insert_episode_duration_s)
    assert isinstance(cfg.transitions[-3], OnSuccess)
    assert isinstance(cfg.transitions[-2], OnTargetPoseReached)
    assert isinstance(cfg.transitions[-1], OnSuccess)
    # No reward_classifier_path -> the loop only closes manually (OnSuccess); no
    # RewardClassifierTransition edge should be present at all.
    assert not any(isinstance(edge, RewardClassifierTransition) for edge in cfg.transitions)

    pull_out = cfg.primitives["pull_out"]
    assert isinstance(pull_out, MoveDeltaPrimitiveConfig)
    assert pull_out.delta == pytest.approx([0.0, 0.0, cfg.pull_out_height_m, 0.0, 0.0, 0.0])
    assert pull_out.delta_frame == "world"
    for name, frame in pull_out.task_frame.items():
        assert frame.policy_mode == [None] * 6


def test_rail_env_reward_classifier_path_adds_an_additional_auto_trigger():
    """reward_classifier_path adds a second, automatic edge alongside -- not instead of -- the
    manual OnSuccess trigger from cooperative_insert to pull_out."""
    cfg = RailBimanualGraspEnvConfig(
        mock=True,
        reward_classifier_path="/tmp/does-not-need-to-exist-for-this-test/pretrained_model",
        reward_classifier_threshold=0.75,
        reward_classifier_device="cpu",
    )

    edges = [(edge.source, edge.target, type(edge).__name__) for edge in cfg.transitions]
    assert ("cooperative_insert", "pull_out", "OnSuccess") in edges
    assert ("cooperative_insert", "pull_out", "RewardClassifierTransition") in edges

    classifier_edge = next(e for e in cfg.transitions if isinstance(e, RewardClassifierTransition))
    assert classifier_edge.pretrained_path == "/tmp/does-not-need-to-exist-for-this-test/pretrained_model"
    assert classifier_edge.threshold == pytest.approx(0.75)
    assert classifier_edge.device == "cpu"


def test_skip_grasp_enters_through_calibrate():
    """skip_grasp jumps straight into calibrate -- deliberately skipping
    zero_ft_before_cooperative's F/T re-zero for faster interactive iteration on ry/
    pivot_offset_x (a temporary trade-off, not yet the final semantics)."""
    cfg = RailBimanualGraspEnvConfig(mock=True, skip_grasp=True)

    assert cfg.start_primitive == "calibrate"
    assert cfg.reset_primitive == "calibrate"


def test_cooperative_reset_servo_recovery_swaps_force_mode_for_servo():
    """A force-mode singularity leaves the arms stuck under cooperative_reset's normal
    compliant teleop -- force mode itself is what's stuck. cooperative_reset_servo_recovery
    makes cooperative_reset rigid/servo-position-controlled instead (same as align_to_rail/
    zero_ft) so the SpaceMouse can drive clear; reference-limiting (a wrench_limits/kp
    budget, meaningless outside force mode) is off too. Nothing else -- cooperative_insert,
    calibrate, etc. -- is affected."""
    cfg = RailBimanualGraspEnvConfig(mock=True, skip_grasp=True, cooperative_reset_servo_recovery=True)

    for name, frame in cfg.primitives["cooperative_reset"].task_frame.items():
        overrides = frame.controller_overrides
        assert overrides["use_force_mode"] is False
        assert overrides["simple_pose_use_servo"] is True
        assert overrides["compliance_reference_limit_enable"] == [False] * 6

    for name, frame in cfg.primitives["cooperative_insert"].task_frame.items():
        assert frame.controller_overrides["use_force_mode"] is True


def test_cooperative_reset_servo_recovery_requires_skip_grasp():
    with pytest.raises(ValueError, match="skip_grasp"):
        RailBimanualGraspEnvConfig(mock=True, skip_grasp=False, cooperative_reset_servo_recovery=True)


def test_cooperative_reset_recalibrate_ry_key_routes_to_calibrate():
    """Up arrow in cooperative_reset jumps back to calibrate (which unlocks ry for
    live teleop again) without a full reset -- for re-calibrating ry mid-session. Nowhere
    else has this key bound; calibrate's own OnSuccess edge already routes back
    through zero_ft_before_cooperative -> cooperative_reset afterward."""
    cfg = RailBimanualGraspEnvConfig(mock=True)

    assert cfg.primitives["cooperative_reset"].processor.events.key_mapping[RECALIBRATE_RY_REQUEST_FLAG] == RECALIBRATE_RY_KEY
    for name in ("cooperative_insert", "calibrate", "zero_ft"):
        assert RECALIBRATE_RY_REQUEST_FLAG not in cfg.primitives[name].processor.events.key_mapping

    edge = next(e for e in cfg.transitions if e.source == "cooperative_reset" and e.target == "calibrate")
    assert isinstance(edge, OnEvent)
    assert edge.event_key == RECALIBRATE_RY_REQUEST_FLAG


def test_ry_angle_supplies_a_driver_axis_override_to_both_cooperative_primitives():
    """ry_angle gives calibrate's live teleop-and-measure a reproducible alternative
    -- an exact, known-good value instead of freezing whatever the arm's actual pose happens
    to measure. Wired to both cooperative_reset and cooperative_insert, since either one may
    be the first to activate and capture it (see CooperativeFramePrimitive.driver_axis_overrides
    for why the exact value -- not just the measured pose -- is what actually gets used)."""
    cfg = RailBimanualGraspEnvConfig(mock=True, skip_grasp=True, ry_angle=0.1234)

    for name in ("cooperative_reset", "cooperative_insert"):
        assert cfg.primitives[name].env_kwargs["driver_axis_overrides"] == {4: pytest.approx(0.1234)}


def test_ry_angle_defaults_to_no_override():
    cfg = RailBimanualGraspEnvConfig(mock=True)

    for name in ("cooperative_reset", "cooperative_insert"):
        assert cfg.primitives[name].env_kwargs["driver_axis_overrides"] == {}


def test_ry_angle_requires_skip_grasp():
    with pytest.raises(ValueError, match="skip_grasp"):
        RailBimanualGraspEnvConfig(mock=True, skip_grasp=False, ry_angle=0.1)


def test_x_offset_supplies_a_driver_axis_override_to_both_cooperative_primitives():
    """Same idea as ry_angle, for the locked x axis -- no skip_grasp requirement, since unlike
    ry there's no dedicated calibration primitive this would make redundant."""
    cfg = RailBimanualGraspEnvConfig(mock=True, x_offset=0.02)

    for name in ("cooperative_reset", "cooperative_insert"):
        assert cfg.primitives[name].env_kwargs["driver_axis_overrides"] == {0: pytest.approx(0.02)}


def test_x_offset_and_ry_angle_combine():
    cfg = RailBimanualGraspEnvConfig(mock=True, skip_grasp=True, x_offset=0.02, ry_angle=0.1234)

    for name in ("cooperative_reset", "cooperative_insert"):
        assert cfg.primitives[name].env_kwargs["driver_axis_overrides"] == {
            0: pytest.approx(0.02),
            4: pytest.approx(0.1234),
        }


def test_load_calibration_requires_the_file_to_exist():
    with pytest.raises(ValueError, match="doesn't exist"):
        RailBimanualGraspEnvConfig(
            mock=True, load_calibration=True, calibration_path="/tmp/definitely-not-there.json"
        )


def test_load_calibration_populates_pivot_x_offset_ry_angle_from_file(tmp_path):
    """load_calibration doesn't require skip_grasp -- the loaded values fill in exactly like
    x_offset/ry_angle set directly, and the relaxed validation (skip_grasp OR
    load_calibration) is what makes that legal."""
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({"pivot_offset_x": 0.11, "x_offset": 0.02, "ry_angle": 0.33}))

    cfg = RailBimanualGraspEnvConfig(
        mock=True, load_calibration=True, calibration_path=str(calibration_path)
    )

    assert cfg.pivot_offset_x == pytest.approx(0.11)
    assert cfg.x_offset == pytest.approx(0.02)
    assert cfg.ry_angle == pytest.approx(0.33)
    for name in ("cooperative_reset", "cooperative_insert"):
        assert cfg.primitives[name].env_kwargs["driver_axis_overrides"] == {
            0: pytest.approx(0.02),
            4: pytest.approx(0.33),
        }


def test_load_calibration_skips_calibrate_in_the_graph(tmp_path):
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({"pivot_offset_x": 0.0, "x_offset": 0.0, "ry_angle": 0.0}))

    cfg = RailBimanualGraspEnvConfig(
        mock=True, load_calibration=True, calibration_path=str(calibration_path)
    )

    assert any(e.source == "zero_ft" and e.target == "zero_ft_before_cooperative" for e in cfg.transitions)
    assert not any(e.source == "zero_ft" and e.target == "calibrate" for e in cfg.transitions)
    # calibrate itself still exists and is still reachable manually (the up-arrow escape
    # hatch from cooperative_reset), just not on the normal entry path.
    assert "calibrate" in cfg.primitives
    assert any(e.source == "cooperative_reset" and e.target == "calibrate" for e in cfg.transitions)


def test_load_calibration_with_skip_grasp_enters_through_zero_ft_before_cooperative(tmp_path):
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps({"pivot_offset_x": 0.0, "x_offset": 0.0, "ry_angle": 0.0}))

    cfg = RailBimanualGraspEnvConfig(
        mock=True, skip_grasp=True, load_calibration=True, calibration_path=str(calibration_path)
    )

    assert cfg.start_primitive == "zero_ft_before_cooperative"
    assert cfg.reset_primitive == "zero_ft_before_cooperative"


def test_calibration_path_flows_to_calibrate_env_kwargs():
    cfg = RailBimanualGraspEnvConfig(mock=True, calibration_path="/tmp/my_calibration.json")

    assert cfg.primitives["calibrate"].env_kwargs["calibration_path"] == "/tmp/my_calibration.json"


def test_calibrate_primitive_saves_calibration_every_step(tmp_path):
    """Written every step, not on some detected "finish" -- CalibratePrimitive has no
    visibility into the SUCCESS keypress that ends it (handled by the outer action processor
    after this env's own step() already returned). The last write before the operator moves
    on is what sticks, which is functionally the same as an explicit "save on finish"."""
    calibration_path = tmp_path / "calibration.json"
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.3, 0.0])  # actual ry = 0.3
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    env = CalibratePrimitive(
        _ry_calibration_task_frame(), {"left": left, "right": right}, {}, driver="left", fps=30.0,
        calibration_path=str(calibration_path),
    )
    try:
        env.step({"left": {}})

        saved = json.loads(calibration_path.read_text())
        assert saved["pivot_offset_x"] == pytest.approx(0.0)
        assert saved["x_offset"] == pytest.approx(0.0)
        assert saved["ry_angle"] == pytest.approx(0.3)

        env._slider_x.set_val(0.07)
        env.step({"left": {}})

        saved = json.loads(calibration_path.read_text())
        assert saved["pivot_offset_x"] == pytest.approx(0.07)
    finally:
        import matplotlib.pyplot as plt

        plt.close(env._slider_fig)


def test_driver_axis_overrides_pins_a_translation_axis_for_every_robot():
    """driver_axis_overrides isn't rotation-only -- a translation override must reach the
    follower's own offset capture too, not just the driver's (same requirement ry's override
    already has to satisfy). No freeze_driver_axes_at_entry entry needed for a translation
    axis: it's already locked via policy_mode=None."""
    left = _FakeRobot([0.10, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([-0.10, 0.0, 1.0, 0.0, 0.0, 0.0])
    env = CooperativeFramePrimitive(
        _cooperative_insert_task_frame(), {"left": left, "right": right}, {},
        driver="left", fps=30.0, driver_axis_overrides={0: 0.02},
    )

    env.step({"left": {}})

    assert left.last_action["x.ee_pos"] == pytest.approx(0.02)
    assert right.last_action["x.ee_pos"] == pytest.approx(0.02)


def test_pivot_offset_x_survives_freeze_driver_axes_at_entry_without_compounding():
    """Regression test for a real bug: cooperative_reset/cooperative_insert/pushdown all
    freeze x via freeze_driver_axes_at_entry, which runs *after* midpoint (where
    pivot_offset_x used to be added) is built -- so the freeze branch silently overwrote and
    discarded a calibrated pivot_offset_x the moment x was also a frozen axis. Fixing that by
    re-publishing FROZEN_AXES_RUNTIME_KEY with the pivot already baked in would have swapped
    "discarded" for "compounds a little further on every hop" instead -- this checks a chain
    of three primitives (calibrate -> cooperative_reset -> pushdown) ends up with the exact
    same pivot every time, not zero times and not growing."""
    shared_runtime_values = {}

    # calibrate: x is live, not frozen -- pivot_offset_x set directly, as if just calibrated.
    left1 = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right1 = _FakeRobot([0.2, 0.0, 1.0, 0.0, 0.0, 0.0])
    calibrate_env = CooperativeFramePrimitive(
        _ry_calibration_task_frame(), {"left": left1, "right": right1}, {},
        driver="left", fps=30.0, pivot_offset_x=0.05,
    )
    calibrate_env.attach_shared_runtime_values(shared_runtime_values)
    calibrate_env.step({"left": {}})
    assert calibrate_env._vtcp_world[0] == pytest.approx(0.15)  # raw midpoint 0.1 + 0.05

    # cooperative_reset: x IS frozen -- pivot_offset_x adopted from shared runtime state, not
    # passed directly, same as a real primitive switch.
    left2 = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right2 = _FakeRobot([0.2, 0.0, 1.0, 0.0, 0.0, 0.0])
    reset_env = CooperativeFramePrimitive(
        _cooperative_insert_task_frame(), {"left": left2, "right": right2}, {},
        driver="left", fps=30.0, freeze_driver_axes_at_entry=(0,),
    )
    reset_env.attach_shared_runtime_values(shared_runtime_values)
    reset_env.step({"left": {}})
    assert reset_env.pivot_offset_x == pytest.approx(0.05)
    assert reset_env._vtcp_world[0] == pytest.approx(0.05)  # driver's own x (0.0) + pivot, not discarded

    # pushdown: also freezes x, chained after cooperative_reset -- must not compound.
    left3 = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right3 = _FakeRobot([0.2, 0.0, 1.0, 0.0, 0.0, 0.0])
    pushdown_env = CooperativeFramePrimitive(
        _cooperative_insert_task_frame(), {"left": left3, "right": right3}, {},
        driver="left", fps=30.0, freeze_driver_axes_at_entry=(0,),
    )
    pushdown_env.attach_shared_runtime_values(shared_runtime_values)
    pushdown_env.step({"left": {}})
    assert pushdown_env._vtcp_world[0] == pytest.approx(0.05)  # still 0.05, not 0.10


def test_freeze_driver_axes_at_entry_persists_across_activations():
    """A freeze_driver_axes_at_entry axis is captured once, on the first activation of the
    whole loop, and reused after that -- not re-measured every activation (which would let it
    silently drift along with whatever small pose changes happen between episodes, the same
    failure OFFSET_FROM_VTCP_RUNTIME_KEY already guards against for grasp geometry). Answers
    "keep this pose without specifying exact values": just let the first activation capture
    it -- no ry_angle/x_offset needed."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.3, 0.0])  # ry = 0.3
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    shared_runtime_values = {}

    first = CooperativeInsertPrimitive(
        _cooperative_insert_task_frame(), {"left": left, "right": right}, {},
        driver="left", fps=30.0, freeze_driver_axes_at_entry=(4,),
    )
    first.attach_shared_runtime_values(shared_runtime_values)
    first.step({"left": {}})

    assert shared_runtime_values[FROZEN_AXES_RUNTIME_KEY][4] == pytest.approx(0.3)

    # ry drifts before the next activation (e.g. a slight give under contact).
    left.pose = [0.0, 0.0, 1.0, 0.0, 0.45, 0.0]

    second = CooperativeInsertPrimitive(
        _cooperative_insert_task_frame(), {"left": left, "right": right}, {},
        driver="left", fps=30.0, freeze_driver_axes_at_entry=(4,),
    )
    second.attach_shared_runtime_values(shared_runtime_values)
    second.step({"left": {}})

    # Still the first activation's value -- not re-measured from the drifted pose.
    assert left.last_action["ry.ee_pos"] == pytest.approx(0.3)


def test_midpoint_offset_nudges_the_captured_translation_for_every_robot():
    """midpoint_offset (used by pushdown for its gentle z push) applies on top of the averaged
    midpoint, same as pivot_offset_x already does for x -- and reaches every robot's target,
    not just the driver's, since it shifts the shared vtcp itself."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    env = CooperativeFramePrimitive(
        _cooperative_insert_task_frame(), {"left": left, "right": right}, {},
        driver="left", fps=30.0, midpoint_offset=[0.0, 0.0, -0.005],
    )

    env.step({"left": {}})

    assert left.last_action["z.ee_pos"] == pytest.approx(0.995)
    assert right.last_action["z.ee_pos"] == pytest.approx(0.995)


def test_alignment_holds_entry_xz_and_targets_plane_and_orientation():
    cfg = RailBimanualGraspEnvConfig(mock=True)
    alignment = cfg.primitives["align_to_rail"]
    alignment.validate(robot_dict=_robot_dict(), teleop_dict={})

    start_pose, goal_pose = alignment.resolve_targets(_entry_context())

    assert start_pose["left"][:3] == pytest.approx([0.2, -0.1, 0.4])
    assert goal_pose["left"][:3] == pytest.approx([0.2, DEFAULT_RAIL_Y_M["left"], 0.4])
    assert goal_pose["left"][3:6] == pytest.approx(DEFAULT_GRASP_ORIENTATION_RPY["left"])
    assert goal_pose["right"][:3] == pytest.approx([-0.3, DEFAULT_RAIL_Y_M["right"], 0.5])
    assert goal_pose["right"][3:6] == pytest.approx(DEFAULT_GRASP_ORIENTATION_RPY["right"])


def test_cooperative_primitives_use_a_softer_translation_wrench_limit():
    """cooperative_reset/cooperative_insert are the only primitives that actually push the
    workpiece around against the rail -- they get a lower translation wrench_limits than the
    30 N shared by align_to_rail/teleop_left/teleop_right/calibrate. Rotation stays
    at the shared limit everywhere (it's locked/absolute in the cooperative primitives, not
    the live contact interface)."""
    cfg = RailBimanualGraspEnvConfig(mock=True)

    for name in ("cooperative_reset", "cooperative_insert"):
        for robot_name in ("left", "right"):
            wrench_limits = cfg.primitives[name].task_frame[robot_name].controller_overrides["wrench_limits"]
            assert wrench_limits[:3] == pytest.approx([cfg.cooperative_translation_wrench_limit_n] * 3)
            assert wrench_limits[3:] == pytest.approx([4.0, 4.0, 4.0])

    for name in ("align_to_rail", "teleop_left", "teleop_right", "calibrate"):
        for robot_name, frame in cfg.primitives[name].task_frame.items():
            assert frame.controller_overrides["wrench_limits"][:3] == pytest.approx([30.0, 30.0, 30.0])


def test_cooperative_translation_wrench_limit_is_configurable():
    cfg = RailBimanualGraspEnvConfig(mock=True, cooperative_translation_wrench_limit_n=8.0)

    for name in ("cooperative_reset", "cooperative_insert"):
        wrench_limits = cfg.primitives[name].task_frame["left"].controller_overrides["wrench_limits"]
        assert wrench_limits[:3] == pytest.approx([8.0, 8.0, 8.0])


def test_calibrate_pivot_offset_range_is_configurable():
    cfg = RailBimanualGraspEnvConfig(mock=True, pivot_offset_range_m=0.08)

    assert cfg.primitives["calibrate"].env_kwargs["pivot_offset_range_m"] == pytest.approx(0.08)


def test_calibrate_pivot_offset_range_defaults_to_0_3m():
    """6x the original 0.05m default (4x, then 1.5x again) -- a long rail's contact point can
    sit noticeably far from the raw grasp midpoint."""
    cfg = RailBimanualGraspEnvConfig(mock=True)

    assert cfg.primitives["calibrate"].env_kwargs["pivot_offset_range_m"] == pytest.approx(0.3)


def test_calibrate_slider_relocates_the_pivot_without_moving_either_robot():
    """Regression test for a real bug: moving pivot_offset_x must not itself move either
    robot -- only relocate where later rotation happens about. The previous implementation (a
    plain vtcp_world[0] += delta, leaving offset_from_vtcp untouched) instead translated the
    whole rigid assembly by delta on every live change, since rotation composition
    (task_pose_to_world_pose) always rotates the offset about vtcp_world's position -- exactly
    the "changing pivot_x visibly moves the rail" symptom this was found from. Slider.set_val
    is the same call a real mouse drag makes."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.2, 0.0, 1.0, 0.0, 0.0, 0.0])
    env = CalibratePrimitive(
        _ry_calibration_task_frame(), {"left": left, "right": right}, {}, driver="left", fps=30.0,
    )
    try:
        env.step({"left": {}})  # creates the slider window, captures at the raw midpoint (0.1)

        env._slider_x.set_val(0.05)
        env.step({"left": {}})

        # The relocation alone must be invisible -- neither robot actually moved.
        assert env.pivot_offset_x == pytest.approx(0.05)
        assert left.last_action["x.ee_pos"] == pytest.approx(0.0, abs=1e-9)
        assert right.last_action["x.ee_pos"] == pytest.approx(0.2, abs=1e-9)

        # Rotation now sweeps around the *relocated* pivot (0.15): left (radius 0.15) should
        # move 3x as far as right (radius 0.05) for the same rotation, not equally (which is
        # what rotating around the old raw midpoint, 0.1, would give both).
        env.step({"left": {"ry.ee_pos": 1.0}})
        left_delta = abs(left.last_action["x.ee_pos"] - 0.0)
        right_delta = abs(right.last_action["x.ee_pos"] - 0.2)
        assert left_delta > 1e-6  # actually moved, not a false-positive from a zero rotation
        assert left_delta == pytest.approx(3.0 * right_delta, rel=1e-3)
    finally:
        import matplotlib.pyplot as plt

        plt.close(env._slider_fig)


def test_cooperative_frame_primitive_reports_pivot_offset_every_step():
    """Every CooperativeFramePrimitive (not just CalibratePrimitive) reports pivot_offset_x in
    info["record_status"] every step -- a slider change should be visible in
    cooperative_reset/cooperative_insert/pushdown too, not just calibrate."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    env = CooperativeFramePrimitive(
        _cooperative_insert_task_frame(), {"left": left, "right": right}, {},
        driver="left", fps=30.0, pivot_offset_x=0.012,
    )

    _obs, _reward, _terminated, _truncated, info = env.step({"left": {}})

    assert info["record_status"] == "pivot_x = +0.0120 m"


def test_calibrate_disables_reference_limiting_on_every_axis():
    """calibrate needs real range on everything it manually adjusts -- ry (teleop) and
    pivot_offset_x/y (`,`/`.`/`[`/`]`). CooperativeFramePrimitive's vtcp reference-limit clamp
    would otherwise cap either at wrench_limits[i]/kp[i] of instantaneous lead over the arm's
    actual pose (~1.5 deg for ry, 1cm for a pivot offset) -- past which further nudges silently
    stop moving the arm while the printed offset keeps climbing regardless (a real bug found
    this way). Disabled there and only there; wrench_limits/kp themselves (the actual torque/
    stiffness) are untouched, and cooperative_reset/cooperative_insert keep full
    reference-limiting on every axis."""
    cfg = RailBimanualGraspEnvConfig(mock=True)

    for name, frame in cfg.primitives["calibrate"].task_frame.items():
        overrides = frame.controller_overrides
        assert overrides["compliance_reference_limit_enable"] == [False] * 6
        assert overrides["wrench_limits"] == pytest.approx([30.0, 30.0, 30.0, 4.0, 4.0, 4.0])
        assert overrides["kp"] == pytest.approx([3000.0, 3000.0, 3000.0, 150.0, 150.0, 150.0])

    for name in ("cooperative_reset", "cooperative_insert"):
        for robot_name, frame in cfg.primitives[name].task_frame.items():
            assert frame.controller_overrides["compliance_reference_limit_enable"] == [True] * 6


def test_demo_flag_routes_cooperative_insert_success_to_pushdown():
    """demo=False (default) keeps the normal train/record loop -- a genuine success goes to
    pull_out. demo=True redirects both success triggers (manual and, if configured, the
    reward classifier) to pushdown instead."""
    default_cfg = RailBimanualGraspEnvConfig(mock=True)
    assert any(
        e.source == "cooperative_insert" and e.target == "pull_out" for e in default_cfg.transitions
    )
    assert not any(e.source == "cooperative_insert" and e.target == "pushdown" for e in default_cfg.transitions)

    demo_cfg = RailBimanualGraspEnvConfig(mock=True, demo=True, reward_classifier_path="/tmp/fake")
    assert any(
        e.source == "cooperative_insert" and e.target == "pushdown" for e in demo_cfg.transitions
    )
    assert not any(e.source == "cooperative_insert" and e.target == "pull_out" for e in demo_cfg.transitions)
    classifier_edge = next(e for e in demo_cfg.transitions if isinstance(e, RewardClassifierTransition))
    assert classifier_edge.target == "pushdown"


def test_pushdown_task_frame_only_exposes_ry_live():
    """x/y/z/rx/rz are all locked in pushdown -- x and y hold wherever cooperative_insert left
    them (x persisted the same way as elsewhere, y just doesn't move since nothing captures it
    fresh either), z gets a gentle push via midpoint_offset, rx/rz stay at their static target
    (rz softened via controller overrides, not policy_mode). ry alone is live, for the operator
    to teleop down toward 0deg by hand -- captured from wherever cooperative_insert froze it
    (freeze_driver_axes_at_entry includes 4, matching cooperative_insert_env_kwargs), not from
    the static DEFAULT_GRASP_ORIENTATION_RPY target: see
    test_pushdown_freezes_ry_instead_of_snapping_to_static_target for the real-hardware
    regression (ry snapping to 0deg/parallel plus a z lift the instant pushdown activated)."""
    cfg = RailBimanualGraspEnvConfig(mock=True, pushdown_z_offset_m=-0.007)
    pushdown = cfg.primitives["pushdown"]

    assert pushdown.task_frame["left"].policy_mode == [None, None, None, None, PolicyMode.RELATIVE, None]
    assert pushdown.task_frame["right"].policy_mode == [None] * 6
    assert pushdown.env_kwargs["freeze_driver_axes_at_entry"] == (0, 4)
    assert pushdown.env_kwargs["midpoint_offset"] == pytest.approx([0.0, 0.0, -0.007])


def test_pushdown_freezes_ry_instead_of_snapping_to_static_target():
    """Regression test for a real-hardware bug: pushdown's freeze_driver_axes_at_entry omitted
    4 (ry), so CooperativeFramePrimitive's capture (no per-axis freeze/override -> static
    task_frame target) snapped ry to its static 0.0 (parallel) the instant pushdown activated,
    instead of continuing from the calibrated angle cooperative_insert had it frozen at -- and,
    since rotation composes about the pivot, swung the driven arm in z and lost contact as a
    side effect. Chains cooperative_insert (which freezes ry from the robot's own measured
    pose, same as production's ry_angle-less path) into pushdown, exactly like production's
    primitive graph always does."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.35, 0.0])  # calibrated ry = 0.35, not 0.0
    right = _FakeRobot([0.2, 0.0, 1.0, 0.0, 0.0, 0.0])
    shared_runtime_values = {}

    insert_env = CooperativeInsertPrimitive(
        _cooperative_insert_task_frame(), {"left": left, "right": right}, {},
        driver="left", fps=30.0, freeze_driver_axes_at_entry=(0, 4),
    )
    insert_env.attach_shared_runtime_values(shared_runtime_values)
    insert_env.step({"left": {}})
    assert shared_runtime_values[FROZEN_AXES_RUNTIME_KEY][4] == pytest.approx(0.35)

    pushdown_env = CooperativeFramePrimitive(
        _cooperative_insert_task_frame(), {"left": left, "right": right}, {},
        driver="left", fps=30.0, freeze_driver_axes_at_entry=(0, 4),
    )
    pushdown_env.attach_shared_runtime_values(shared_runtime_values)
    pushdown_env.step({"left": {}})
    assert pushdown_env._vtcp_world[4] == pytest.approx(0.35)  # not 0.0 (the static target)


def test_pushdown_softens_yaw_and_frees_rotation_reference_limiting():
    cfg = RailBimanualGraspEnvConfig(mock=True, pushdown_yaw_kp=12.0, pushdown_yaw_wrench_limit_nm=0.8)

    for robot_name, frame in cfg.primitives["pushdown"].task_frame.items():
        overrides = frame.controller_overrides
        assert overrides["kp"][5] == pytest.approx(12.0)
        assert overrides["wrench_limits"][5] == pytest.approx(0.8)
        assert overrides["compliance_reference_limit_enable"] == [True, True, True, False, False, False]
        # Only yaw softened -- the other axes keep cooperative_insert's usual values.
        assert overrides["kp"][:5] == pytest.approx([3000.0, 3000.0, 3000.0, 150.0, 150.0])




def test_cooperative_insert_policy_has_real_action_dataset_stats():
    """cooperative_insert's action space is 2-dim (x/z translation only, y locked) -- an unset
    dataset_stats would silently fall back to SACConfig's own placeholder shape/range instead
    (see share/rl/runtime.py resolve_policy_dataset_stats)."""
    cfg = RailBimanualGraspEnvConfig(mock=True, translation_action_scale=0.05)

    stats = cfg.primitives["cooperative_insert"].policy.dataset_stats
    assert stats["action"]["min"] == pytest.approx([-0.05, -0.05])
    assert stats["action"]["max"] == pytest.approx([0.05, 0.05])


def test_cooperative_insert_policy_matches_insertion_sac_preset():
    """cooperative_insert's SACConfig mirrors InsertionSACConfig (the hoermann connector
    env's own SAC preset) field-for-field, with one deliberate deviation: a real pretrained
    vision backbone (helper2424/resnet10, same as train_reward_classifier.py's own default)
    frozen rather than trained from scratch, instead of that preset's unset
    vision_encoder_name + freeze_vision_encoder=False."""
    cfg = RailBimanualGraspEnvConfig(mock=True)
    policy = cfg.primitives["cooperative_insert"].policy

    assert policy.vision_encoder_name == "helper2424/resnet10"
    assert policy.freeze_vision_encoder is True
    assert policy.device == "cuda"
    assert policy.storage_device == "cpu"
    assert policy.use_amp is False
    assert policy.online_steps == int(1e8)
    assert policy.async_prefetch is True
    assert policy.online_step_before_learning == 300
    assert policy.online_buffer_capacity == 30000
    assert policy.offline_buffer_capacity == 10000
    assert policy.utd_ratio == 3
    assert policy.shared_encoder is True
    assert policy.num_critics == 2
    assert policy.target_entropy == pytest.approx(-1.5)
    assert policy.critic_target_update_weight == pytest.approx(0.003)
    assert policy.use_backup_entropy is False


def test_grasp_stages_never_drive_the_gripper_live():
    """Neither grasp stage's gripper is teleop-driven -- the SpaceMouse never touches the
    gripper at all (both buttons are SUCCESS, see button_mapping on self.teleop). left_grasp
    holds both open; right_grasp's static target for "left" is already closed, so left snaps
    shut on its own the instant right_grasp activates -- no close_grippers_on_exit needed."""
    cfg = RailBimanualGraspEnvConfig(mock=True)

    left_gripper = cfg.primitives["teleop_left"].processor.gripper
    right_gripper = cfg.primitives["teleop_right"].processor.gripper

    assert left_gripper.enable == {"left": False, "right": False}
    assert right_gripper.enable == {"left": False, "right": False}
    assert left_gripper.static_pos == {"left": cfg.open_gripper_position, "right": cfg.open_gripper_position}
    assert right_gripper.static_pos == {"left": cfg.closed_gripper_position, "right": cfg.open_gripper_position}
    assert cfg.primitives["teleop_left"].close_grippers_on_exit is False
    assert cfg.primitives["teleop_right"].close_grippers_on_exit is False

    # cooperative_reset and cooperative_insert share the same processor (both grippers held
    # closed, static) -- both arms are already holding the workpiece by this point.
    for name in ("cooperative_reset", "cooperative_insert"):
        cooperative_gripper = cfg.primitives[name].processor.gripper
        assert cooperative_gripper.enable == {"left": False, "right": False}
        assert cooperative_gripper.static_pos == {
            "left": cfg.closed_gripper_position,
            "right": cfg.closed_gripper_position,
        }


def test_alignment_uses_controller_side_velocity_limits():
    cfg = RailBimanualGraspEnvConfig(mock=True, alignment_linear_speed_mps=0.02, alignment_angular_speed_rad_s=0.2)
    alignment_frame = cfg.primitives["align_to_rail"].task_frame["right"]
    assert cfg.alignment_linear_speed_mps == pytest.approx(0.02)
    assert cfg.alignment_angular_speed_rad_s == pytest.approx(0.2)
    assert alignment_frame.controller_overrides["use_force_mode"] is False


def test_cooperative_primitives_observe_driver_only_xyz_velocity():
    """Both arms are rigidly coupled through the workpiece -- their xyz velocity is mostly
    the same signal up to inertia, and rotation is locked entirely in every cooperative
    primitive. observation.state should be a clean, near-Markovian driver-only xyz velocity
    -- not two robots' worth of largely-redundant 6-axis velocity in two different native
    frames. align_to_rail (not policy-trained, unaffected) keeps the old broad default."""
    cfg = RailBimanualGraspEnvConfig(mock=True)

    for name in ("zero_ft", "zero_ft_before_cooperative", "calibrate", "cooperative_reset"):
        obs = cfg.primitives[name].processor.observation
        assert obs.add_ee_velocity_to_observation == {"left": True, "right": False}
        assert obs.ee_velocity_axes == ["x.ee_vel", "y.ee_vel", "z.ee_vel"]

    assert cfg.primitives["align_to_rail"].processor.observation.add_ee_velocity_to_observation is True


def test_cooperative_insert_observes_delta_position_and_previous_action_too():
    """cooperative_insert alone adds two extra channels on top of the driver-only y/z
    velocity every other cooperative primitive gets: delta position from wherever the driver
    was when the episode started (position slot, custom dy/dz names), and the previous step's
    action (velocity slot, alongside the current velocity) -- see CooperativeInsertPrimitive.
    x is excluded throughout (locked, always ~0). Neither matters for cooperative_reset/
    calibrate/etc, which have no policy and aren't recorded."""
    cfg = RailBimanualGraspEnvConfig(mock=True)
    obs = cfg.primitives["cooperative_insert"].processor.observation

    assert obs.add_ee_pos_to_observation == {"left": True, "right": False}
    assert obs.ee_pos_axes == ["dy.ee_pos", "dz.ee_pos"]
    assert obs.add_ee_velocity_to_observation == {"left": True, "right": False}
    assert obs.ee_velocity_axes == ["y.ee_vel", "z.ee_vel", "prev_y.ee_vel", "prev_z.ee_vel"]

    stats = cfg.primitives["cooperative_insert"].policy.dataset_stats["observation.state"]
    half_square = cfg.cooperative_insert_position_range_m / 2
    assert stats["min"] == pytest.approx([-half_square] * 2 + [-cfg.translation_action_scale] * 4)
    assert stats["max"] == pytest.approx([half_square] * 2 + [cfg.translation_action_scale] * 4)


def test_cooperative_insert_infers_the_full_six_dim_state_shape():
    """Regression test for a real dataset-creation bug: infer_features() samples each robot's
    raw get_observation() once, before any primitive has run, to build the static feature-shape
    snapshot record.py's LeRobotDataset.create() writes into a fresh dataset's metadata. dx/dz
    and prev_x/prev_z only ever exist because CooperativeInsertPrimitive.step() injects them
    into the *runtime* observation -- a plain robot double (like real UR robots) never exposes
    them, so the inferred shape used to silently come out as whatever placeholder was already
    there instead of the real 6, and record.py crashed on the first real frame."""
    cfg = RailBimanualGraspEnvConfig(mock=True)
    primitive = cfg.primitives["cooperative_insert"]
    robot_dict = {
        "left": _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        "right": _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    }

    primitive.validate(robot_dict, {})  # normalizes scalar processor config into per-robot dicts, as make() does
    primitive.infer_features(robot_dict, cameras={})

    assert primitive.features["observation.state"].shape == (6,)


class _FakeRobot:
    """Minimal robot double exposing EE-pose observations and recording send_action calls."""

    def __init__(self, pose: list[float]):
        self.pose = list(pose)
        self.last_action: dict[str, float] = {}
        self._motors_ft: dict[str, type] = {}

    def get_observation(self) -> dict[str, float]:
        return dict(zip(("x.ee_pos", "y.ee_pos", "z.ee_pos", "rx.ee_pos", "ry.ee_pos", "rz.ee_pos"), self.pose))

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.last_action = dict(action)
        return action

    def set_task_frame(self, frame) -> None:
        pass


def _ry_calibration_task_frame() -> dict:
    return {
        "left": TaskFrame(target=[0.0] * 6, control_mode=[ControlMode.POS] * 6, policy_mode=COOPERATIVE_LEFT_POLICY),
        "right": TaskFrame(target=[0.0] * 6, control_mode=[ControlMode.POS] * 6, policy_mode=[None] * 6),
    }


def _cooperative_insert_task_frame() -> dict:
    return {
        "left": TaskFrame(target=[0.0] * 6, control_mode=[ControlMode.POS] * 6, policy_mode=COOPERATIVE_INSERT_LEFT_POLICY),
        "right": TaskFrame(target=[0.0] * 6, control_mode=[ControlMode.POS] * 6, policy_mode=[None] * 6),
    }


def test_cooperative_insert_reset_observation_already_carries_the_full_six_dim_state():
    """Regression test for a real recording crash: record.py stores the *pre*-step observation
    alongside each action (o_t paired with a_t), so the very first frame of an episode comes
    from reset()'s observation, not step()'s. Injecting dy/dz and prev_y/prev_z only from
    step() left reset()'s observation without them, which is exactly what LeRobotDataset's
    per-frame shape check caught."""
    left = _FakeRobot([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    env = CooperativeInsertPrimitive(
        _cooperative_insert_task_frame(), {"left": left, "right": right}, {}, driver="left", fps=30.0,
    )

    obs, _info = env.reset()

    for axis in ("dy", "dz"):
        assert obs[f"left.{axis}.ee_pos"] == pytest.approx(0.0)
    for axis in ("prev_y", "prev_z"):
        assert obs[f"left.{axis}.ee_vel"] == pytest.approx(0.0)


def test_cooperative_insert_prev_action_is_not_lagged_by_one_step():
    """The observation returned by step(a_t) must carry a_t itself (the action that just
    produced it), not a_{t-1} -- ManipulationPrimitive.step() calls _get_observation()
    internally right after send_action(), so _prev_action has to be updated before
    super().step() runs, not after."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    env = CooperativeInsertPrimitive(
        _cooperative_insert_task_frame(), {"left": left, "right": right}, {}, driver="left", fps=30.0,
    )
    env.reset()

    obs, _reward, _terminated, _truncated, _info = env.step(
        {"left": {"y.ee_pos": 0.11, "z.ee_pos": 0.33}}
    )

    assert obs["left.prev_y.ee_vel"] == pytest.approx(0.11)
    assert obs["left.prev_z.ee_vel"] == pytest.approx(0.33)


def test_ry_angle_calibration_reports_the_actual_measured_ry():
    """The printed ry tracks the arm's actual physical pose (what the operator is teleoping
    it to) -- there is no target/lock for it here. Prepended to the base class's own
    pivot_offset_x readout (see CooperativeFramePrimitive.step()) rather than replacing it, so
    a slider change stays visible here too."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.3, 0.0])  # actual ry = 0.3
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    env = CalibratePrimitive(
        _ry_calibration_task_frame(), {"left": left, "right": right}, {}, driver="left", fps=30.0,
    )
    try:
        _obs, _reward, _terminated, _truncated, info = env.step({"left": {}})

        assert info["record_status"] == "ry = +0.3000 rad  |  pivot_x = +0.0000 m"
    finally:
        import matplotlib.pyplot as plt

        plt.close(env._slider_fig)


def test_ry_angle_calibration_lets_the_spacemouse_move_the_whole_pivot():
    """Unlike bimanual_pick.py's PivotOffsetXCalibrationPrimitive, this one keeps xyz/ry live
    -- the whole point is teleoperating ry (and whatever else) to the right value."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    env = CalibratePrimitive(
        _ry_calibration_task_frame(), {"left": left, "right": right}, {}, driver="left", fps=30.0,
    )
    try:
        env.step({"left": {}})  # captures

        env.step({"left": {"z.ee_pos": 0.6}})

        assert left.last_action["z.ee_pos"] != pytest.approx(0.0)
    finally:
        import matplotlib.pyplot as plt

        plt.close(env._slider_fig)
