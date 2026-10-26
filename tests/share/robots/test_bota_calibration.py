"""The payload model recovers the gripper's mass, centre of mass and bias, whatever the mount, from a whole trajectory."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from share.robots.viperx.lerobot_robot_viperx.bota.calibration import (
    MAX_SPEED_RAD_S, TRAJECTORY_COLUMNS, PayloadModel, ResidualNet, ee_rotation, fit_payload, fit_payload_model,
    needs_calibration, read_payload, read_trajectory, trajectory_path, write_payload, write_trajectory,
)

SAMPLE_RATE_HZ = 20.0


def _truth(mass: float = 0.8) -> PayloadModel:
    return PayloadModel(
        mass=mass,
        center_of_mass=np.array([0.01, -0.02, 0.07]),
        mount_rotvec=Rotation.from_euler("xyz", [0.3, -1.2, 0.7]).as_rotvec(),
        force_bias=np.array([30.0, 97.0, -176.0]),
        torque_bias=np.array([1.8, -0.4, 0.05]),
    )


def _measure(model: PayloadModel, rotations: np.ndarray, noise: float = 0.02, seed: int = 1) -> np.ndarray:
    return model.predict(rotations) + np.random.default_rng(seed).normal(0.0, noise, (len(rotations), 6))


def _cable(joints: np.ndarray) -> np.ndarray:
    """A smooth pull that depends on the wrist only, like cables bending at the wrist."""
    return np.stack([
        2.0 * np.sin(joints[:, 4]), 1.5 * np.cos(joints[:, 3]), 3.0 * np.sin(joints[:, 4] + joints[:, 5]),
        0.10 * np.sin(joints[:, 4]), np.zeros(len(joints)), 0.05 * np.cos(joints[:, 5]),
    ], axis=1)


def _trajectory(truth: PayloadModel, waypoints: int, cable: bool = False, seed: int = 0) -> np.ndarray:
    """What a hand-guided session records: moves between random poses with a pause at each, 20 Hz.

    Samples recorded while the arm moves faster than MAX_SPEED_RAD_S carry a large inertial force the static
    model knows nothing about.
    """
    rng = np.random.default_rng(seed)
    targets = rng.uniform(-1.2, 1.2, (waypoints, 6))
    path = []
    for start, end in zip(targets[:-1], targets[1:]):
        path += [start + (end - start) * blend for blend in np.linspace(0.0, 1.0, int(3 * SAMPLE_RATE_HZ), endpoint=False)]
        path += [end] * int(1.5 * SAMPLE_RATE_HZ)
    joints = np.array(path)
    wrenches = _measure(truth, np.stack([ee_rotation(pose) for pose in joints]), seed=seed + 1)
    if cable:
        wrenches = wrenches + _cable(joints)
    speed = np.r_[np.inf, np.abs(np.diff(joints, axis=0)).max(axis=1) * SAMPLE_RATE_HZ]
    wrenches[speed > MAX_SPEED_RAD_S] += 20.0  # inertia of a fast move
    times = np.arange(len(joints)) / SAMPLE_RATE_HZ
    return np.column_stack([times, np.full(len(joints), 31.0), joints, wrenches])


def test_fit_recovers_the_payload_and_compensates_unseen_poses():
    truth = _truth()
    rotations = Rotation.random(40, random_state=0).as_matrix()

    model, force_rms, torque_rms = fit_payload_model(_measure(truth, rotations), rotations)

    assert abs(model.mass) == pytest.approx(0.8, rel=0.02)
    np.testing.assert_allclose(model.center_of_mass, truth.center_of_mass, atol=0.003)
    # The residual of a correct fit is the injected sensor noise (0.02 per channel)
    assert force_rms < 0.03 and torque_rms < 0.03
    unseen = Rotation.random(10, random_state=5).as_matrix()
    residual = model.compensate(truth.predict(unseen), unseen)
    assert np.abs(residual).max() < 0.1


def test_fit_handles_a_sensor_that_reports_the_reaction_force():
    """Opposite sign convention: the mass comes out negative, the physical mass and the residual stay right."""
    truth = _truth(mass=-0.8)
    rotations = Rotation.random(40, random_state=2).as_matrix()

    model, force_rms, _ = fit_payload_model(_measure(truth, rotations), rotations)

    assert abs(model.mass) == pytest.approx(0.8, rel=0.02)
    assert force_rms < 0.05


def test_fit_rejects_poses_that_only_pitch_about_one_axis():
    rotations = Rotation.from_euler("y", np.linspace(-0.8, 0.8, 30)[:, None]).as_matrix()

    with pytest.raises(ValueError, match="orientations"):
        fit_payload_model(_measure(_truth(), rotations), rotations)


def test_rezero_only_moves_the_bias():
    model = _truth()
    rotation = Rotation.random(1, random_state=3).as_matrix()[0]
    drifted = model.predict(rotation) + np.array([0.5, -0.2, 0.1, 0.0, 0.02, -0.01])

    model.rezero(drifted, rotation)

    np.testing.assert_allclose(model.compensate(drifted, rotation), 0.0, atol=1e-9)
    assert model.mass == 0.8


def test_model_survives_a_json_round_trip():
    model = _truth()
    restored = PayloadModel.from_dict(json.loads(json.dumps(model.to_dict())))
    rotation = Rotation.random(1, random_state=4).as_matrix()[0]

    np.testing.assert_allclose(restored.predict(rotation), model.predict(rotation))


def test_forward_kinematics_at_the_zero_pose_is_the_identity_rotation():
    np.testing.assert_allclose(ee_rotation(np.zeros(6)), np.eye(3), atol=1e-12)


# --- the whole trajectory, and the residual net ---


def test_the_fit_uses_the_whole_trajectory_but_only_what_is_slow_enough():
    table = _trajectory(_truth(), waypoints=25)

    model, summary = fit_payload(table, residual=False)

    assert abs(model.mass) == pytest.approx(0.8, rel=0.02), "the inertia of the fast moves must not leak into the fit"
    used = int(summary.split("fit on ")[1].split(" of ")[0])
    assert 0.3 * len(table) < used < len(table), "far more than a dozen poses, but not the fast samples"


def test_a_residual_net_that_beats_the_weight_model_on_held_out_orientations_is_kept():
    truth = _truth()
    model, summary = fit_payload(_trajectory(truth, waypoints=60, cable=True), residual=True)

    assert model.residual is not None and summary.endswith("kept.") and not model.net_rejected
    joints = np.random.default_rng(9).uniform(-1.2, 1.2, (60, 6))
    rotations = np.stack([ee_rotation(pose) for pose in joints])
    measured = truth.predict(rotations) + _cable(joints)
    with_net = np.abs(model.compensate(measured, rotations, joints)[:, :3]).mean()
    without_net = np.abs(measured - PayloadModel(**{**model.__dict__, "residual": None}).predict(rotations))[:, :3].mean()
    assert with_net < 0.6 * without_net


def test_a_residual_net_that_does_not_help_is_dropped():
    model, summary = fit_payload(_trajectory(_truth(), waypoints=60, cable=False), residual=True)

    assert model.residual is None and "dropped" in summary
    assert model.net_rejected, "remembered, so a later start does not ask for a net again"
    assert PayloadModel.from_dict(json.loads(json.dumps(model.to_dict()))).net_rejected


def test_the_residual_net_is_skipped_when_too_few_orientations_are_covered():
    model, summary = fit_payload(_trajectory(_truth(), waypoints=8, cable=True), residual=True)

    assert model.residual is None and "skipped" in summary
    assert not model.net_rejected, "a net that was never tried can still be asked for"


def test_calibration_is_needed_without_a_fit_or_when_a_net_was_never_tried():
    plain = _truth()
    rejected = PayloadModel(**{**plain.__dict__, "net_rejected": True})
    net = ResidualNet(
        layers=[(np.zeros((6, 4)), np.zeros(4)), (np.zeros((4, 6)), np.zeros(6))],
        input_mean=np.zeros(6), input_std=np.ones(6), output_scale=np.ones(6), output_limit=np.ones(6),
    )
    with_net = PayloadModel(**{**plain.__dict__, "residual": net})

    assert needs_calibration(None, residual=False) and needs_calibration(None, residual=True)
    assert not needs_calibration(plain, residual=False), "no net asked for"
    assert needs_calibration(plain, residual=True), "a net is asked for and was never tried"
    assert not needs_calibration(rejected, residual=True), "tried and rejected: do not ask again"
    assert not needs_calibration(with_net, residual=True) and not needs_calibration(with_net, residual=False)


def test_a_model_with_a_net_needs_the_joints_and_keeps_its_bias_apart_from_the_net(tmp_path):
    rng = np.random.default_rng(0)
    net = ResidualNet(
        layers=[(rng.normal(size=(6, 16)), rng.normal(size=16)), (rng.normal(size=(16, 16)), rng.normal(size=16)),
                (rng.normal(size=(16, 6)), rng.normal(size=6))],
        input_mean=np.zeros(6), input_std=np.ones(6), output_scale=np.full(6, 0.5), output_limit=np.full(6, 2.0),
    )
    model = PayloadModel(**{**_truth().__dict__, "residual": net})
    joints = np.array([0.1, 0.2, -0.3, 0.5, -0.4, 0.9])
    rotation = ee_rotation(joints)

    with pytest.raises(ValueError, match="joints"):
        model.predict(rotation)
    drifted = model.predict(rotation, joints) + 0.7
    model.rezero(drifted, rotation, joints)
    np.testing.assert_allclose(model.compensate(drifted, rotation, joints), 0.0, atol=1e-9)

    restored = PayloadModel.from_dict(json.loads(json.dumps(model.to_dict())))
    np.testing.assert_allclose(restored.predict(rotation, joints), model.predict(rotation, joints), rtol=1e-4, atol=1e-5)

    fpath = tmp_path / "right.json"
    write_payload(fpath, model)
    assert len(fpath.read_text().splitlines()) < 150, "weights are one line per row, not one per number"
    assert read_payload(fpath).residual is not None


# --- files ---


def test_payload_is_stored_next_to_the_motors_and_the_original_file_is_backed_up_once(tmp_path):
    fpath = tmp_path / "right.json"
    motors = {"gripper": {"id": 9, "drive_mode": 4, "homing_offset": 0, "range_min": 1920, "range_max": 2822}}
    fpath.write_text(json.dumps(motors, indent=4))

    write_payload(fpath, _truth())
    write_payload(fpath, _truth(mass=0.9))  # a refit

    assert json.loads(fpath.read_text())["gripper"] == motors["gripper"]
    assert read_payload(fpath).mass == 0.9
    assert json.loads((tmp_path / "right.json.bak").read_text()) == motors, "the file as it was before the first write"
    assert not (tmp_path / "right.json.tmp").exists()


def test_read_payload_is_none_without_a_file_or_an_entry(tmp_path):
    (tmp_path / "right.json").write_text("{}")

    assert read_payload(tmp_path / "missing.json") is None
    assert read_payload(tmp_path / "right.json") is None


def test_the_trajectory_csv_round_trips_with_a_header(tmp_path):
    table = _trajectory(_truth(), waypoints=4)
    path = trajectory_path(tmp_path / "right.json")

    write_trajectory(path, table)

    assert path.name == "right_bota_trajectory.csv"
    assert path.read_text().splitlines()[0] == ",".join(TRAJECTORY_COLUMNS)
    np.testing.assert_allclose(read_trajectory(path), table, rtol=1e-8)
