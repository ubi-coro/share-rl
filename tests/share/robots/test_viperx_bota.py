"""The ViperX publishes the compensated wrench as an observation only, and keeps the payload fit in its calibration file."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from share.robots.viperx.lerobot_robot_viperx import ViperX, ViperXConfig
from share.robots.viperx.lerobot_robot_viperx import viperx as viperx_module
from share.robots.viperx.lerobot_robot_viperx.bota.calibration import (
    PAYLOAD_KEY, PayloadModel, ResidualNet, read_payload, wrench_observation,
)
from share.robots.viperx.lerobot_robot_viperx.config_viperx import BotaConfig

WRENCH_KEYS = [f"{axis}.ee_wrench" for axis in ("x", "y", "z", "rx", "ry", "rz")]
ZERO_POSE = {f"{joint}.pos": 0.0 for joint in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")}
MOTORS = {"gripper": {"id": 9, "drive_mode": 4, "homing_offset": 0, "range_min": 1920, "range_max": 2822}}
ENABLED = BotaConfig(enable=True, port="/dev/null")


def _robot(tmp_path, bota: BotaConfig) -> ViperX:
    return ViperX(ViperXConfig(port="/dev/null", id="test", calibration_dir=tmp_path, bota=bota))


def _payload() -> PayloadModel:
    return PayloadModel(
        mass=0.5, center_of_mass=np.array([0.0, 0.0, 0.05]), mount_rotvec=np.zeros(3),
        force_bias=np.array([1.0, 2.0, 3.0]), torque_bias=np.zeros(3),
    )


def _write_calibration_file(tmp_path, payload: PayloadModel | None) -> None:
    data = dict(MOTORS)
    if payload is not None:
        data[PAYLOAD_KEY] = payload.to_dict()
    (tmp_path / "test.json").write_text(json.dumps(data, indent=4))


def test_wrench_is_an_observation_but_never_an_action(tmp_path):
    robot = _robot(tmp_path, ENABLED)

    assert set(WRENCH_KEYS) <= set(robot.observation_features)
    assert not set(WRENCH_KEYS) & set(robot.action_features)


def test_robot_without_the_sensor_is_unchanged(tmp_path):
    robot = _robot(tmp_path, BotaConfig())

    assert not set(WRENCH_KEYS) & set(robot.observation_features)
    assert robot.bota_driver is None


def test_enabling_without_a_port_fails_early(tmp_path):
    with pytest.raises(ValueError, match="bota.port"):
        _robot(tmp_path, BotaConfig(enable=True))


def test_wrench_observation_is_nan_until_the_payload_is_calibrated():
    obs = wrench_observation(None, None, ZERO_POSE)

    assert list(obs) == WRENCH_KEYS
    assert all(np.isnan(value) for value in obs.values())


def test_wrench_observation_removes_the_payload_and_bias():
    model = _payload()
    # At the zero pose the EE frame is the base frame, so only the payload's weight and the bias are measured.
    measured = model.predict(np.eye(3)) + np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    driver = SimpleNamespace(latest=lambda: SimpleNamespace(wrench=measured))

    obs = wrench_observation(model, driver, ZERO_POSE)

    assert [obs[key] for key in WRENCH_KEYS] == pytest.approx([0.0, 0.0, 5.0, 0.0, 0.0, 0.0], abs=1e-9)


def test_one_calibration_file_holds_the_motors_and_the_payload(tmp_path):
    _write_calibration_file(tmp_path, _payload())

    robot = _robot(tmp_path, ENABLED)

    assert list(robot.calibration) == ["gripper"], "the payload entry is not mistaken for a motor"
    assert robot.calibration["gripper"].range_max == 2822
    assert robot.payload_model.mass == 0.5


def test_a_calibration_file_with_a_payload_still_loads_with_the_sensor_disabled(tmp_path):
    _write_calibration_file(tmp_path, _payload())

    robot = _robot(tmp_path, BotaConfig())

    assert list(robot.calibration) == ["gripper"]
    assert robot.payload_model is None


@pytest.mark.parametrize("bota", [ENABLED, BotaConfig()], ids=["sensor enabled", "sensor disabled"])
def test_saving_the_motor_calibration_keeps_the_stored_payload(tmp_path, bota):
    _write_calibration_file(tmp_path, _payload())
    robot = _robot(tmp_path, bota)
    robot.calibration["gripper"].range_max = 3000

    robot._save_calibration()

    stored = json.loads((tmp_path / "test.json").read_text())
    assert stored["gripper"]["range_max"] == 3000
    assert read_payload(tmp_path / "test.json").mass == 0.5


# --- when connecting calibrates ---


def _connectable(tmp_path, monkeypatch, residual: bool, model: PayloadModel | None):
    """A ViperX whose connect() runs against a fake bus and sensor, recording what it calibrates and tares."""
    robot = _robot(tmp_path, BotaConfig(enable=True, port="/dev/null", residual=residual))
    robot.payload_model = model
    calls = []
    robot.bus = SimpleNamespace(is_connected=False, is_calibrated=True, connect=lambda: None, enable_torque=lambda: None)
    robot.bota_driver = SimpleNamespace(is_connected=False, connect=lambda: None)
    monkeypatch.setattr(robot, "configure", lambda: None)
    monkeypatch.setattr(robot, "get_observation", lambda: {})
    monkeypatch.setattr(robot, "_calibrate_payload", lambda: calls.append("calibrate"))
    monkeypatch.setattr(viperx_module, "tare_payload", lambda *args: calls.append("tare"))
    return robot, calls


def _variant(**changes) -> PayloadModel:
    net = ResidualNet(
        layers=[(np.zeros((6, 4)), np.zeros(4)), (np.zeros((4, 6)), np.zeros(6))],
        input_mean=np.zeros(6), input_std=np.ones(6), output_scale=np.ones(6), output_limit=np.ones(6),
    )
    changes = {key: net if value == "net" else value for key, value in changes.items()}
    return PayloadModel(**{**_payload().__dict__, **changes})


@pytest.mark.parametrize(
    "residual, model, expected",
    [
        (False, None, ["calibrate"]),                  # first run: no fit yet
        (False, _variant(), ["tare"]),                 # a fit, no net asked for
        (True, _variant(), ["calibrate", "tare"]),     # a net is asked for and the stored fit has none
        (True, _variant(net_rejected=True), ["tare"]),  # it was tried and rejected: do not ask again
        (True, _variant(residual="net"), ["tare"]),    # the net is there
    ],
    ids=["no fit", "fit without net asked", "net asked, none stored", "net rejected before", "net stored"],
)
def test_connecting_calibrates_when_there_is_no_fit_or_the_asked_for_net_was_never_tried(
    tmp_path, monkeypatch, residual, model, expected
):
    robot, calls = _connectable(tmp_path, monkeypatch, residual, model)

    robot.connect(calibrate=True)

    assert calls == expected


def test_connecting_without_calibration_only_warns_about_the_missing_net(tmp_path, monkeypatch, caplog):
    robot, calls = _connectable(tmp_path, monkeypatch, residual=True, model=_variant())

    with caplog.at_level("WARNING"):
        robot.connect(calibrate=False)

    assert calls == ["tare"]
    assert "stored payload fit has no net" in caplog.text
