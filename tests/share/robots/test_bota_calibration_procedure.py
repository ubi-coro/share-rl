"""The hand-guided payload calibration, run against a simulated arm and sensor on a virtual clock."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from share.robots.viperx.lerobot_robot_viperx import ViperX, ViperXConfig
from share.robots.viperx.lerobot_robot_viperx.bota import calibration as calibration_module
from share.robots.viperx.lerobot_robot_viperx.bota.calibration import (
    PayloadModel, ee_rotation, read_payload, read_trajectory, trajectory_path, write_trajectory,
)
from share.robots.viperx.lerobot_robot_viperx.config_viperx import BotaConfig

ARM = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
HOLD_S, MOVE_S = 2.0, 0.5  # every pose is reached in MOVE_S (fast) and then held still until HOLD_S


def _cable(joints: np.ndarray) -> np.ndarray:
    """A smooth pull that depends on the wrist only, like cables bending at the wrist."""
    return np.array([2.0 * np.sin(joints[4]), 1.5 * np.cos(joints[3]), 3.0 * np.sin(joints[4] + joints[5]), 0.0, 0.0, 0.0])


class _World:
    """A person moving the arm through random poses, and the sensor wrench that goes with them."""

    def __init__(self, truth: PayloadModel, poses: int = 40, cable: bool = False):
        self.now = 0.0
        self.truth = truth
        self.cable = cable
        self.rng = np.random.default_rng(0)
        self.poses = self.rng.uniform(-1.2, 1.2, (poses, 6))
        self.bus_calls: list[tuple] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def joints_at(self, time_s: float) -> np.ndarray:
        index = min(int(time_s // HOLD_S), len(self.poses) - 1)
        phase = time_s - index * HOLD_S
        if index == 0 or phase >= MOVE_S:
            return self.poses[index]
        blend = phase / MOVE_S
        return (1 - blend) * self.poses[index - 1] + blend * self.poses[index]

    # --- the bus ---
    @contextmanager
    def torque_disabled(self):
        self.bus_calls.append(("torque_off",))
        yield
        self.bus_calls.append(("torque_on",))

    def sync_read(self, name: str) -> dict[str, float]:
        assert name == "Present_Position"
        values = dict(zip(ARM, self.joints_at(self.now)))
        return {**values, "shoulder_shadow": values["shoulder"], "elbow_shadow": values["elbow"], "gripper": 0.5}

    def sync_write(self, name: str, goal: dict[str, float]) -> None:
        self.bus_calls.append(("write", name, goal))

    # --- the sensor ---
    def recent(self, duration_s: float) -> np.ndarray:
        times = np.arange(self.now - duration_s, self.now, 0.045)  # a thinned stream
        joints = [self.joints_at(t) for t in times]
        clean = [self.truth.predict(ee_rotation(pose)) + (_cable(pose) if self.cable else 0.0) for pose in joints]
        return np.array(clean) + self.rng.normal(0.0, 0.02, (len(times), 6))


def _truth_model() -> PayloadModel:
    return PayloadModel(
        mass=0.7, center_of_mass=np.array([0.0, 0.01, 0.06]),
        mount_rotvec=Rotation.from_euler("xyz", [0.0, 0.0, 1.57]).as_rotvec(),
        force_bias=np.array([30.0, 97.0, -176.0]), torque_bias=np.array([1.8, -0.4, 0.05]),
    )


def _simulated_robot(
    tmp_path, monkeypatch, world: _World, duration_s: float = 85.0, residual: bool = False, answers: str = ""
) -> ViperX:
    robot = ViperX(ViperXConfig(
        port="/dev/null", id="sim", calibration_dir=tmp_path,
        bota=BotaConfig(enable=True, port="/dev/null", calibration_duration_s=duration_s, residual=residual),
    ))
    robot.bus = SimpleNamespace(
        sync_read=world.sync_read, sync_write=world.sync_write, torque_disabled=world.torque_disabled,
    )
    robot.bota_driver = SimpleNamespace(recent=world.recent, latest=lambda: SimpleNamespace(temperature_c=31.0))
    monkeypatch.setattr(robot, "configure", lambda: None)
    monkeypatch.setattr("builtins.input", lambda *_: answers)
    monkeypatch.setattr(calibration_module, "time", SimpleNamespace(monotonic=world.monotonic, sleep=world.sleep))
    monkeypatch.setattr(calibration_module, "_enter_pressed", lambda: False)  # nobody at the console
    return robot


def test_hand_guided_calibration_fits_the_whole_trajectory_and_holds_the_final_pose(tmp_path, monkeypatch):
    world = _World(_truth_model())
    robot = _simulated_robot(tmp_path, monkeypatch, world)

    robot._calibrate_payload()

    assert abs(robot.payload_model.mass) == pytest.approx(0.7, rel=0.05)
    np.testing.assert_allclose(robot.payload_model.center_of_mass, world.truth.center_of_mass, atol=0.005)
    assert read_payload(robot.calibration_fpath) is not None
    trajectory = read_trajectory(trajectory_path(robot.calibration_fpath))
    assert len(trajectory) > 1000, "every sample is stored, not a dozen poses"
    assert trajectory[:, 1].tolist() == [31.0] * len(trajectory), "with the sensor temperature"
    # The torque is off while moving and the arm is told to hold its final pose before it comes back on
    assert [call[0] for call in world.bus_calls] == ["torque_off", "write", "torque_on"]
    held = world.bus_calls[1][2]
    assert not any(motor.endswith("_shadow") for motor in held)
    assert held["wrist_rotate"] == pytest.approx(world.joints_at(world.now)[5])


def test_enter_finishes_the_calibration_early_once_enough_orientations_are_covered(tmp_path, monkeypatch, capsys):
    world = _World(_truth_model())
    robot = _simulated_robot(tmp_path, monkeypatch, world)
    presses = [5.0, 60.0]  # Enter at 5 s comes too early and is refused, the one at 60 s ends the calibration

    def enter_pressed() -> bool:
        if presses and world.now > presses[0]:
            presses.pop(0)
            return True
        return False

    monkeypatch.setattr(calibration_module, "_enter_pressed", enter_pressed)

    robot._calibrate_payload()

    assert "need 12: keep going" in capsys.readouterr().out
    assert world.now < 70.0, "finished on Enter, not at the deadline"
    assert abs(robot.payload_model.mass) == pytest.approx(0.7, rel=0.05)


def test_calibration_prints_the_measured_wrench_and_the_coverage_live(tmp_path, monkeypatch, capsys):
    world = _World(_truth_model())
    robot = _simulated_robot(tmp_path, monkeypatch, world)

    robot._calibrate_payload()

    out = capsys.readouterr().out
    assert "F [" in out and "] N  T [" in out and "] Nm" in out, "the live line shows the wrench"
    assert "using" in out and "too fast" in out, "and whether the sample counts"
    assert "orientations 12/12" in out, "and the coverage"
    assert "Payload fit on" in out and "mass 0.700 kg" in out


def test_calibration_gives_up_when_the_arm_is_never_slow_enough(tmp_path, monkeypatch):
    world = _World(PayloadModel(0.7, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)))
    world.joints_at = lambda time_s: np.full(6, 0.3) * np.sin(20.0 * time_s)  # constantly shaking
    robot = _simulated_robot(tmp_path, monkeypatch, world, duration_s=20.0)

    with pytest.raises(RuntimeError, match="orientations"):
        robot._calibrate_payload()

    assert robot.payload_model is None
    assert not robot.calibration_fpath.exists()


def test_the_residual_net_is_fitted_as_part_of_the_calibration(tmp_path, monkeypatch, capsys):
    world = _World(_truth_model(), poses=80, cable=True)
    robot = _simulated_robot(tmp_path, monkeypatch, world, duration_s=165.0, residual=True)

    robot._calibrate_payload()

    assert "kept." in capsys.readouterr().out
    assert read_payload(robot.calibration_fpath).residual is not None, "the net is stored with the fit"
    joints = world.rng.uniform(-1.2, 1.2, 6)
    measured = world.truth.predict(ee_rotation(joints)) + _cable(joints)
    compensated = robot.payload_model.compensate(measured, ee_rotation(joints), joints)
    assert np.abs(compensated[:3]).max() < 1.5, "the cable pull of up to 3 N is mostly gone at an unseen pose"


def test_a_stored_trajectory_can_be_refitted_instead_of_recording_a_new_one(tmp_path, monkeypatch, capsys):
    world = _World(_truth_model())
    first = _simulated_robot(tmp_path, monkeypatch, world)
    first._calibrate_payload()
    stored = read_trajectory(trajectory_path(first.calibration_fpath))

    world = _World(_truth_model())  # fresh clock and bus: nothing may be recorded this time
    again = _simulated_robot(tmp_path, monkeypatch, world, answers="y")
    again.payload_model = None
    again._calibrate_payload()

    assert world.bus_calls == [] and world.now == 0.0, "the torque is never touched and no time passes"
    assert abs(again.payload_model.mass) == pytest.approx(0.7, rel=0.05)
    np.testing.assert_allclose(read_trajectory(trajectory_path(again.calibration_fpath)), stored, rtol=1e-8)
    assert "Payload fit on" in capsys.readouterr().out


def test_declining_the_refit_records_a_new_trajectory(tmp_path, monkeypatch):
    world = _World(_truth_model())
    write_trajectory(trajectory_path(tmp_path / "sim.json"), np.zeros((3, 14)))
    robot = _simulated_robot(tmp_path, monkeypatch, world, answers="")

    robot._calibrate_payload()

    assert world.bus_calls[0] == ("torque_off",)
    assert len(read_trajectory(trajectory_path(robot.calibration_fpath))) > 1000


def test_enter_pressed_is_false_without_a_console():
    """Under pytest stdin cannot be polled: the calibration must then just run to its end."""
    assert calibration_module._enter_pressed() is False
