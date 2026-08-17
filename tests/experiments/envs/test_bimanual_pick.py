import math
import os

os.environ.setdefault("PYNPUT_BACKEND", "dummy")

import pytest

from lerobot.processor import TransitionKey, create_transition

from experiments.envs.bimanual_pick import BimanualPickEnvConfig, CooperativeFramePrimitive
from share.envs.manipulation_primitive.config_manipulation_primitive import MoveDeltaPrimitiveConfig
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.robots.ur import SimUR, SimURConfig
from share.teleoperators import TeleopEvents
from share.utils.mock_utils import MockTeleoperator


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


def test_bimanual_pick_config_builds_expected_graph():
    cfg = BimanualPickEnvConfig()

    assert list(cfg.primitives) == ["teleop_left", "teleop_right", "teleop_midpoint"]
    assert isinstance(cfg.primitives["teleop_left"], MoveDeltaPrimitiveConfig)
    assert isinstance(cfg.primitives["teleop_right"], MoveDeltaPrimitiveConfig)
    assert cfg.primitives["teleop_right"].teleop_mapping == {"right": "left"}
    assert cfg.primitives["teleop_midpoint"].env_class is CooperativeFramePrimitive
    assert [(t.source, t.target) for t in cfg.transitions] == [
        ("teleop_left", "teleop_right"),
        ("teleop_right", "teleop_midpoint"),
        ("teleop_midpoint", "teleop_left"),
    ]


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


def _task_frame(policy_mode, origin=None) -> TaskFrame:
    return TaskFrame(
        target=[0.0] * 6,
        control_mode=[ControlMode.POS] * 6,
        policy_mode=policy_mode,
        origin=origin,
    )


def test_cooperative_frame_first_step_holds_current_poses():
    """With zero teleop delta, the midpoint hasn't moved -- both arms hold their entry pose."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # in right's own base frame
    task_frame = {
        "left": _task_frame([PolicyMode.RELATIVE] * 6),
        "right": _task_frame([None] * 6, origin=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    env = CooperativeFramePrimitive(task_frame, {"left": left, "right": right}, {}, driver="left", fps=30.0)

    env.step({"left": {}})

    assert left.last_action["x.ee_pos"] == pytest.approx(0.0, abs=1e-9)
    assert right.last_action["x.ee_pos"] == pytest.approx(0.0, abs=1e-9)
    assert right.last_action["z.ee_pos"] == pytest.approx(1.0, abs=1e-9)


def test_cooperative_frame_driver_translation_moves_both_arms_together():
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    task_frame = {
        "left": _task_frame([PolicyMode.RELATIVE] * 6),
        "right": _task_frame([None] * 6, origin=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    fps = 30.0
    env = CooperativeFramePrimitive(task_frame, {"left": left, "right": right}, {}, driver="left", fps=fps)
    env.step({"left": {}})  # captures the initial midpoint offsets

    dz = 0.6  # m/s along z
    env.step({"left": {"z.ee_pos": dz}})

    expected_world_dz = dz / fps
    # follower (right) reports an absolute target in its own base frame (origin shifts x by 1m)
    assert right.last_action["z.ee_pos"] == pytest.approx(1.0 + expected_world_dz, abs=1e-9)
    assert right.last_action["x.ee_pos"] == pytest.approx(0.0, abs=1e-9)
    # driver (left) reports a velocity that integrates to the same world delta over dt
    assert left.last_action["z.ee_pos"] * (1.0 / fps) == pytest.approx(expected_world_dz, abs=1e-9)


def test_cooperative_frame_keeps_fixed_grasp_offset_between_arms():
    """The arms' relative offset (grasp geometry) must stay constant as the midpoint moves."""
    left = _FakeRobot([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.2, 0.0, 1.0, 0.0, 0.0, 0.0])  # 0.2m ahead of left along x, own frame
    task_frame = {
        "left": _task_frame([PolicyMode.RELATIVE] * 6),
        "right": _task_frame([None] * 6, origin=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    fps = 30.0
    env = CooperativeFramePrimitive(task_frame, {"left": left, "right": right}, {}, driver="left", fps=fps)
    env.step({"left": {}})

    for _ in range(5):
        env.step({"left": {"x.ee_pos": 0.3}})

    world_left_dx = 0.3 * 5 / fps
    # right's own-frame x = world x - origin x = (left_world_x + 0.2m offset + 1.0m origin) - 1.0m
    assert right.last_action["x.ee_pos"] == pytest.approx(world_left_dx + 0.2, abs=1e-6)


def test_cooperative_frame_rotation_swings_arms_about_the_midpoint():
    """A yaw rate about the shared midpoint should swing both arms along a circle."""
    left = _FakeRobot([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    right = _FakeRobot([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])  # same shared frame: origin=[0]*6
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
    assert right.last_action["rz.ee_pos"] == pytest.approx(theta, abs=1e-6)
    # driver's own reported velocity, integrated over dt, reaches the same yaw
    assert left.last_action["rz.ee_pos"] / fps == pytest.approx(theta, abs=1e-6)
