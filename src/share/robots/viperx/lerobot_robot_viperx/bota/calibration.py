"""Payload calibration of the Bota wrist sensor on the ViperX.

With the gripper mounted, the sensor reads the gripper's weight in its own frame, so the reading
depends on the arm pose. The model predicts that reading from the end-effector orientation and a
fitted payload: mass, centre of mass, EE-to-sensor mount rotation and a force/torque bias. No
inertia is needed because the model is static.

Optionally a small net on top learns what the weight model cannot know and that depends on the pose,
such as the pull of the cables. It is kept only if it beats the weight model on held-out poses.

The fit uses the whole trajectory recorded while the arm is moved by hand, not a few held poses: every
sample slow enough for the static model counts.

This module holds the model and its fit, the hand-guided procedure that records the trajectory, the
taring, the compensated wrench observation, and the files: the payload entry in the arm's calibration
file (everything needed to run) and the CSV of the trajectory (everything needed to refit).
"""

import json
import logging
import os
import re
import select
import shutil
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import modern_robotics as mr
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from share.envs.manipulation_primitive.task_frame import TASK_FRAME_AXIS_NAMES
from share.robots.viperx.lerobot_robot_viperx.bota.driver import BotaDriver
from share.utils.kinematics import MRKinematics

logger = logging.getLogger(__name__)

ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
WRIST = slice(3, 6)  # forearm_roll, wrist_angle, wrist_rotate: where the cables bend
WRENCH_FEATURES = {f"{axis}.ee_wrench": float for axis in TASK_FRAME_AXIS_NAMES}
PAYLOAD_KEY = "bota_payload"  # entry of the arm's calibration file, next to the motors

# The recorded trajectory, one row per sample, as stored in the CSV next to the calibration file.
TRAJECTORY_COLUMNS = ["time_s", "temperature_c", *ARM_JOINTS, "fx", "fy", "fz", "tx", "ty", "tz"]
JOINT_COLUMNS, WRENCH_COLUMNS = slice(2, 8), slice(8, 14)

# --- weight model ---
GRAVITY = 9.81
DOWN = np.array([0.0, 0.0, -1.0])  # gravity direction in the (level) base frame

# Torque residuals (Nm) are scaled by this (1/m) to be comparable with force residuals (N).
TORQUE_WEIGHT = 10.0

# Minimum spread (std) of the gravity direction along its least-varying axis. Below this the poses
# are nearly coplanar and the mount rotation cannot be told apart from the bias.
MIN_ORIENTATION_SPREAD = 0.15

# --- residual net ---
HIDDEN = 16
TRAIN_STEPS = 1500
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-2
CLIP_FACTOR = 1.5  # the net's output never exceeds this times the largest residual it was trained on
HELD_OUT_FRACTION = 0.2
MIN_IMPROVEMENT = 0.9  # held-out error must drop to this fraction for the net to be kept

# --- hand-guided procedure ---
POLL_S = 0.05
# Samples recorded while the arm moves faster than this are not used: the model is static, so the inertia of a
# fast move would be fitted as if it were part of the pose.
MAX_SPEED_RAD_S = 0.3
# Orientations closer than this count as the same one. They measure the coverage, and held-out data is
# whole orientations, because neighbouring samples are nearly identical.
ORIENTATION_BIN_DEG = 10.0
MIN_ORIENTATIONS = 12
MIN_ORIENTATIONS_FOR_NET = 30
STILL_FORCE_STD_N = 0.3  # taring: the wrench must not vary more than this over the averaging window
STATUS_PERIOD_S = 0.25
STATUS_WIDTH = 100


@dataclass
class ResidualNet:
    """Two tanh layers on the pose features, in numpy so running it needs no torch."""

    layers: list[tuple[np.ndarray, np.ndarray]]  # (weight (in, out), bias (out,)) per layer
    input_mean: np.ndarray
    input_std: np.ndarray
    output_scale: np.ndarray
    output_limit: np.ndarray

    def __call__(self, features: np.ndarray) -> np.ndarray:
        hidden = (features - self.input_mean) / self.input_std
        for weight, bias in self.layers[:-1]:
            hidden = np.tanh(hidden @ weight + bias)
        weight, bias = self.layers[-1]
        return np.clip((hidden @ weight + bias) * self.output_scale, -self.output_limit, self.output_limit)

    def to_dict(self) -> dict:
        return {
            "layers": [{"weight": _short(weight), "bias": _short(bias)} for weight, bias in self.layers],
            "input_mean": _short(self.input_mean),
            "input_std": _short(self.input_std),
            "output_scale": _short(self.output_scale),
            "output_limit": _short(self.output_limit),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResidualNet":
        return cls(
            layers=[(np.array(layer["weight"]), np.array(layer["bias"])) for layer in data["layers"]],
            input_mean=np.array(data["input_mean"]),
            input_std=np.array(data["input_std"]),
            output_scale=np.array(data["output_scale"]),
            output_limit=np.array(data["output_limit"]),
        )


@dataclass
class PayloadModel:
    """Wrench of the payload's weight plus a bias, as seen by the sensor, and optionally a residual net.

    ``mass`` is signed by the sensor's force convention, so use ``abs(mass)`` as the physical mass.
    """

    mass: float  # kg
    center_of_mass: np.ndarray  # (3,) m, sensor frame
    mount_rotvec: np.ndarray  # (3,) rotation vector, EE frame -> sensor frame
    force_bias: np.ndarray  # (3,) N
    torque_bias: np.ndarray  # (3,) Nm
    residual: ResidualNet | None = None  # pose-dependent remainder, needs the joints
    net_rejected: bool = False  # a net was fitted and did not beat the weight model: do not ask for one again

    def gravity_wrench(self, rotation_base_ee: np.ndarray) -> np.ndarray:
        """Wrench caused by the payload's weight in the sensor frame; (..., 3, 3) -> (..., 6)."""
        gravity_ee = np.swapaxes(rotation_base_ee, -1, -2) @ DOWN
        mount = Rotation.from_rotvec(self.mount_rotvec).as_matrix()
        force = self.mass * GRAVITY * (gravity_ee @ mount.T)
        torque = np.cross(self.center_of_mass, force)
        return np.concatenate([force, torque], axis=-1)

    def predict(self, rotation_base_ee: np.ndarray, joints: np.ndarray | None = None) -> np.ndarray:
        """Expected sensor reading with nothing but the payload attached."""
        prediction = self.gravity_wrench(rotation_base_ee) + np.concatenate([self.force_bias, self.torque_bias])
        if self.residual is None:
            return prediction
        if joints is None:
            raise ValueError("This payload model has a residual net: pass the joints too.")
        return prediction + self.residual(pose_features(rotation_base_ee, joints))

    def compensate(self, wrench: np.ndarray, rotation_base_ee: np.ndarray, joints: np.ndarray | None = None) -> np.ndarray:
        """Remove the payload's weight, the pose-dependent remainder and the bias from a measured wrench."""
        return wrench - self.predict(rotation_base_ee, joints)

    def rezero(self, wrench: np.ndarray, rotation_base_ee: np.ndarray, joints: np.ndarray | None = None) -> None:
        """Re-estimate only the bias from one reading taken with nothing touching the gripper."""
        bias = wrench - replace(self, force_bias=np.zeros(3), torque_bias=np.zeros(3)).predict(rotation_base_ee, joints)
        self.force_bias, self.torque_bias = bias[:3], bias[3:]

    def to_dict(self) -> dict:
        data = {
            "mass": float(self.mass),
            "center_of_mass": self.center_of_mass.tolist(),
            "mount_rotvec": self.mount_rotvec.tolist(),
            "force_bias": self.force_bias.tolist(),
            "torque_bias": self.torque_bias.tolist(),
        }
        if self.residual is not None:
            data["residual"] = self.residual.to_dict()
        if self.net_rejected:
            data["net_rejected"] = True
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PayloadModel":
        return cls(
            mass=float(data["mass"]),
            center_of_mass=np.asarray(data["center_of_mass"], dtype=float),
            mount_rotvec=np.asarray(data["mount_rotvec"], dtype=float),
            force_bias=np.asarray(data["force_bias"], dtype=float),
            torque_bias=np.asarray(data["torque_bias"], dtype=float),
            residual=ResidualNet.from_dict(data["residual"]) if "residual" in data else None,
            net_rejected=bool(data.get("net_rejected", False)),
        )


def fit_payload_model(
    wrenches: np.ndarray, rotations_base_ee: np.ndarray
) -> tuple[PayloadModel, float, float]:
    """Fit the weight model to static poses.

    Args:
        wrenches: (N, 6) measured [Fx, Fy, Fz, Tx, Ty, Tz], one still reading per pose.
        rotations_base_ee: (N, 3, 3) EE orientation in the base frame for each pose.

    Returns:
        The model, plus the RMS force (N) and torque (Nm) residual over the poses.
    """
    count = len(wrenches)
    gravity_ee = np.swapaxes(rotations_base_ee, -1, -2) @ DOWN

    spread = np.linalg.svd(gravity_ee - gravity_ee.mean(axis=0), compute_uv=False)[-1] / np.sqrt(count)
    if spread < MIN_ORIENTATION_SPREAD:
        raise ValueError(
            f"Poses do not cover enough orientations (gravity spread {spread:.2f} < {MIN_ORIENTATION_SPREAD}); "
            "increase the sweep."
        )

    # Linear initial guess. Force is linear in the gravity direction: F = A g + b with A = m * g0 * R_mount.
    design = np.hstack([gravity_ee, np.ones((count, 1))])
    coefficients, *_ = np.linalg.lstsq(design, wrenches[:, :3], rcond=None)
    force_matrix, force_bias = coefficients[:3].T, coefficients[3]
    left, _, right_t = np.linalg.svd(force_matrix)
    mount = left @ right_t
    if np.linalg.det(mount) < 0:  # sensor reports the reaction force, so the mass comes out negative
        mount = -mount
    mass = np.trace(mount.T @ force_matrix) / (3 * GRAVITY)

    # Torque is linear in the centre of mass given the force: tau = c x F + b = -skew(F) c + b.
    weight_force = wrenches[:, :3] - force_bias
    torque_design = np.concatenate([-_skew(weight_force), np.broadcast_to(np.eye(3), (count, 3, 3))], axis=2)
    torque_solution, *_ = np.linalg.lstsq(torque_design.reshape(3 * count, 6), wrenches[:, 3:].reshape(-1), rcond=None)

    def unpack(theta: np.ndarray) -> PayloadModel:
        return PayloadModel(theta[0], theta[1:4].copy(), theta[4:7].copy(), theta[7:10].copy(), theta[10:13].copy())

    def residual(theta: np.ndarray) -> np.ndarray:
        error = wrenches - unpack(theta).predict(rotations_base_ee)
        error[:, 3:] *= TORQUE_WEIGHT
        return error.ravel()

    theta0 = np.concatenate([
        [mass], torque_solution[:3], Rotation.from_matrix(mount).as_rotvec(), force_bias, torque_solution[3:],
    ])
    model = unpack(least_squares(residual, theta0).x)

    error = wrenches - model.predict(rotations_base_ee)
    return model, float(np.sqrt(np.mean(error[:, :3] ** 2))), float(np.sqrt(np.mean(error[:, 3:] ** 2)))


def _skew(vectors: np.ndarray) -> np.ndarray:
    """(N, 3) -> (N, 3, 3) cross-product matrices, skew(v) @ w == v x w."""
    zero = np.zeros(len(vectors))
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    return np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=1).reshape(-1, 3, 3)


def pose_features(rotation_base_ee: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """What the residual net sees: gravity direction in the EE frame and the wrist joint angles (raw, cables twist)."""
    gravity_ee = np.swapaxes(rotation_base_ee, -1, -2) @ DOWN
    return np.concatenate([gravity_ee, joints[..., WRIST]], axis=-1)


def train_residual_net(
    physical: PayloadModel, wrenches: np.ndarray, rotations: np.ndarray, joints: np.ndarray, seed: int = 0
) -> ResidualNet:
    """Train the net on what ``physical`` (a model without a net) leaves unexplained."""
    import torch

    targets = wrenches - physical.predict(rotations)
    features = pose_features(rotations, joints)
    input_mean, input_std = features.mean(axis=0), np.maximum(features.std(axis=0), 1e-3)
    output_scale = np.maximum(targets.std(axis=0), 1e-3)
    inputs = torch.tensor((features - input_mean) / input_std, dtype=torch.float32)
    outputs = torch.tensor(targets / output_scale, dtype=torch.float32)

    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], HIDDEN), torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.Tanh(),
        torch.nn.Linear(HIDDEN, 6),
    )
    torch.nn.init.zeros_(net[-1].weight)  # starts as the weight model alone
    torch.nn.init.zeros_(net[-1].bias)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    for _ in range(TRAIN_STEPS):
        optimizer.zero_grad()
        ((net(inputs) - outputs) ** 2).mean().backward()
        optimizer.step()

    layers = [
        (layer.weight.detach().numpy().T.astype(float), layer.bias.detach().numpy().astype(float))
        for layer in net if isinstance(layer, torch.nn.Linear)
    ]
    return ResidualNet(layers, input_mean, input_std, output_scale, CLIP_FACTOR * np.abs(targets).max(axis=0))


def fit_payload(table: np.ndarray, residual: bool) -> tuple[PayloadModel, str]:
    """Fit the payload model to a recorded trajectory, and with ``residual`` a net on top if it helps on held-out orientations.

    Args:
        table: (N, len(TRAJECTORY_COLUMNS)) samples in time order. Those recorded while the arm moved faster
            than MAX_SPEED_RAD_S are not used.

    Returns:
        The model and a text summary of the fit.
    """
    used = table[_slow_enough(table)]
    if len(used) == 0:
        raise ValueError(f"None of the {len(table)} samples is slower than {MAX_SPEED_RAD_S} rad/s.")
    joints, wrenches = used[:, JOINT_COLUMNS], used[:, WRENCH_COLUMNS]
    rotations = np.stack([ee_rotation(pose) for pose in joints])
    orientation = _orientation_ids(np.swapaxes(rotations, -1, -2) @ DOWN)
    count_orientations = int(orientation.max()) + 1
    model, force_rms, torque_rms = fit_payload_model(wrenches, rotations)
    summary = (
        f"Payload fit on {len(used)} of {len(table)} samples in {count_orientations} orientations: "
        f"mass {abs(model.mass):.3f} kg (compare with the gripper's weight), "
        f"center of mass {np.round(model.center_of_mass, 3).tolist()} m, residual {force_rms:.2f} N / {torque_rms:.3f} Nm."
    )
    if not residual:
        return model, summary
    if count_orientations < MIN_ORIENTATIONS_FOR_NET:
        return model, summary + (
            f"\nResidual net skipped: {count_orientations} orientations, it needs {MIN_ORIENTATIONS_FOR_NET}."
        )

    # Hold out whole orientations: neighbouring samples are nearly identical, so a random split would flatter the net.
    held_orientations = np.random.default_rng(0).permutation(count_orientations)[
        : round(HELD_OUT_FRACTION * count_orientations)
    ]
    held_out = np.isin(orientation, held_orientations)
    train = ~held_out
    plain, _, _ = fit_payload_model(wrenches[train], rotations[train])
    with_net = replace(plain, residual=train_residual_net(plain, wrenches[train], rotations[train], joints[train]))
    error_plain = wrenches[held_out] - plain.predict(rotations[held_out])
    error_net = wrenches[held_out] - with_net.predict(rotations[held_out], joints[held_out])

    summary += (
        f"\nResidual net on {len(held_orientations)} held-out orientations: force {_rms(error_plain[:, :3]):.2f} -> "
        f"{_rms(error_net[:, :3]):.2f} N, torque {_rms(error_plain[:, 3:]):.3f} -> {_rms(error_net[:, 3:]):.3f} Nm, "
    )
    if _weighted_rms(error_net) < MIN_IMPROVEMENT * _weighted_rms(error_plain):
        model.residual = train_residual_net(model, wrenches, rotations, joints)
        return model, summary + "kept."
    model.net_rejected = True
    return model, summary + f"dropped: it did not beat the weight model by {100 * (1 - MIN_IMPROVEMENT):.0f}%."


def _slow_enough(table: np.ndarray) -> np.ndarray:
    """Which samples were recorded while the arm moved slower than MAX_SPEED_RAD_S (the first one has no speed)."""
    speed = np.full(len(table), np.inf)
    steps = np.maximum(np.diff(table[:, 0]), 1e-3)
    speed[1:] = np.abs(np.diff(table[:, JOINT_COLUMNS], axis=0)).max(axis=1) / steps
    return speed <= MAX_SPEED_RAD_S


def _leader_index(direction: np.ndarray, leaders: list[np.ndarray]) -> int | None:
    """Index of the first leader within ORIENTATION_BIN_DEG of a gravity direction, or None if it is a new orientation."""
    if not leaders:
        return None
    angles = np.degrees(np.arccos(np.clip(np.asarray(leaders) @ direction, -1.0, 1.0)))
    close = np.flatnonzero(angles < ORIENTATION_BIN_DEG)
    return int(close[0]) if len(close) else None


def _orientation_ids(gravity: np.ndarray) -> np.ndarray:
    """Number the distinct orientations of (N, 3) gravity directions, in order of first appearance."""
    leaders: list[np.ndarray] = []
    ids = np.empty(len(gravity), dtype=int)
    for index, direction in enumerate(gravity):
        match = _leader_index(direction, leaders)
        if match is None:
            leaders.append(direction)
            match = len(leaders) - 1
        ids[index] = match
    return ids


def _rms(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(error ** 2)))


def _weighted_rms(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(error[:, :3] ** 2) + np.mean((TORQUE_WEIGHT * error[:, 3:]) ** 2)))


def read_arm_positions(bus) -> dict[str, float]:
    """Present position of every motor except the shadows (radians, gripper 0-1)."""
    positions = bus.sync_read("Present_Position")
    return {motor: float(value) for motor, value in positions.items() if not motor.endswith("_shadow")}


def ee_rotation(joints: np.ndarray) -> np.ndarray:
    """EE orientation in the base frame from the six arm joint angles (radians, ARM_JOINTS order)."""
    description = MRKinematics.ROBOT_DESC["vx300s-bota"]
    return mr.FKinSpace(description["M"], description["Slist"], joints)[:3, :3]


def needs_calibration(model: PayloadModel | None, residual: bool) -> bool:
    """No fit stored yet, or a net is asked for and was never tried (a rejected one was tried, it is not asked again)."""
    return model is None or (residual and model.residual is None and not model.net_rejected)


def wrench_observation(model: PayloadModel | None, driver: BotaDriver, motor_obs: dict[str, float]) -> dict[str, float]:
    """Gravity-compensated wrench as `{axis}.ee_wrench` entries, NaN while there is no payload fit."""
    if model is None:
        wrench = np.full(6, np.nan)
    else:
        joints = np.array([motor_obs[f"{joint}.pos"] for joint in ARM_JOINTS])
        wrench = model.compensate(driver.latest().wrench, ee_rotation(joints), joints)
    return {key: float(value) for key, value in zip(WRENCH_FEATURES, wrench)}


def tare_payload(model: PayloadModel, bus, driver: BotaDriver, duration_s: float) -> None:
    """Re-zero the wrench bias. The arm must be still and nothing may touch the gripper."""
    time.sleep(duration_s)
    driver.latest()  # raises if the stream stalled
    window = driver.recent(duration_s)
    if window[:, :3].std(axis=0).max() > STILL_FORCE_STD_N:
        logger.warning("The wrench was not still while taring, the zero point may be off.")
    positions = read_arm_positions(bus)
    joints = np.array([positions[joint] for joint in ARM_JOINTS])
    model.rezero(window.mean(axis=0), ee_rotation(joints), joints)


def calibrate_payload(
    bus, driver: BotaDriver, duration_s: float, residual: bool, stored: np.ndarray | None = None
) -> tuple[PayloadModel, np.ndarray]:
    """Fit the payload model to the trajectory recorded while the arm is moved by hand with the torque off.

    Args:
        stored: The trajectory of an earlier calibration. If given, the user may refit it instead of recording
            a new one, e.g. to add the residual net without moving the arm again.

    Returns:
        The model and the whole trajectory (rows of TRAJECTORY_COLUMNS) it was fitted on.
    """
    if stored is not None:
        answer = input(f"\nA trajectory of {len(stored)} samples is stored. Refit it instead of recording a new one? [y/N] ")
        if answer.strip().lower() == "y":
            model, summary = fit_payload(stored, residual)
            print(summary)
            return model, stored

    min_orientations = MIN_ORIENTATIONS_FOR_NET if residual else MIN_ORIENTATIONS
    input(
        "\nBota payload calibration\n"
        "  1. Support the arm, then press Enter: the torque switches off.\n"
        "  2. Move the arm slowly by hand through as many orientations as you can (tilt and roll the wrist). Hold "
        "it by the wrist or forearm, not the gripper. Everything is recorded, moves faster than "
        f"{np.degrees(MAX_SPEED_RAD_S):.0f} deg/s are not used. The line below shows the measured wrench, whether "
        "the current sample is used, and how many different orientations you have covered.\n"
        f"  3. Cover at least {min_orientations} orientations.\n"
        f"  4. Press Enter to finish once you have enough, otherwise it ends after {duration_s:.0f}s. The fit "
        "runs, the arm holds the pose you left it in and the torque comes back on.\n"
        "Press Enter to start. "
    )

    rows, leaders = [], []
    previous = None  # (time, joints) of the last sample
    start = time.monotonic()
    deadline = start + duration_s
    last_status = 0.0
    with bus.torque_disabled():
        while time.monotonic() < deadline:
            if _enter_pressed():
                if len(leaders) >= min_orientations:
                    break
                print(f"\r  only {len(leaders)} orientations, need {min_orientations}: keep going".ljust(STATUS_WIDTH))
            time.sleep(POLL_S)
            now = time.monotonic()
            positions = read_arm_positions(bus)
            joints = np.array([positions[joint] for joint in ARM_JOINTS])
            window = driver.recent(POLL_S)
            if len(window) == 0:
                continue
            measured = window.mean(axis=0)
            speed = np.inf if previous is None else np.abs(joints - previous[1]).max() / max(now - previous[0], 1e-3)
            previous = (now, joints)
            rows.append([now - start, driver.latest().temperature_c, *joints, *measured])

            used = speed <= MAX_SPEED_RAD_S
            if used:
                gravity = ee_rotation(joints).T @ DOWN
                if _leader_index(gravity, leaders) is None:
                    leaders.append(gravity)
            if now - last_status >= STATUS_PERIOD_S:
                last_status = now
                print(
                    f"\r  {_format_wrench(measured)}  {'using   ' if used else 'too fast'}  "
                    f"orientations {len(leaders)}/{min_orientations}".ljust(STATUS_WIDTH),
                    end="", flush=True,
                )
        print()

        # Hold the pose the arm was left in, instead of driving back to the last commanded goal.
        bus.sync_write("Goal_Position", read_arm_positions(bus))

    if len(leaders) < min_orientations:
        raise RuntimeError(
            f"Covered only {len(leaders)} orientations, need {min_orientations}: move the arm through more "
            "orientations, or raise bota.calibration_duration_s."
        )
    table = np.array(rows)
    model, summary = fit_payload(table, residual)
    print(summary)
    return model, table


def read_payload(fpath: Path) -> PayloadModel | None:
    """The payload fit stored in the arm's calibration file, or None if there is none yet."""
    if not fpath.is_file():
        return None
    entry = json.loads(fpath.read_text()).get(PAYLOAD_KEY)
    return None if entry is None else PayloadModel.from_dict(entry)


def write_payload(fpath: Path, model: PayloadModel) -> None:
    """Store the fit under PAYLOAD_KEY in the arm's calibration file, keeping everything else in it.

    The file is replaced atomically, and its content from before the first payload write is kept as `<file>.bak`.
    """
    data = {}
    if fpath.is_file():
        data = json.loads(fpath.read_text())
        backup = fpath.with_name(fpath.name + ".bak")
        if not backup.exists():
            shutil.copy2(fpath, backup)
    data[PAYLOAD_KEY] = model.to_dict()
    temporary = fpath.with_name(fpath.name + ".tmp")
    temporary.write_text(_dump_json(data))
    os.replace(temporary, fpath)


def trajectory_path(calibration_fpath: Path) -> Path:
    """The CSV of the recorded trajectory that belongs to an arm's calibration file."""
    return calibration_fpath.with_name(f"{calibration_fpath.stem}_bota_trajectory.csv")


def write_trajectory(path: Path, table: np.ndarray) -> None:
    np.savetxt(path, table, delimiter=",", header=",".join(TRAJECTORY_COLUMNS), comments="", fmt="%.9g")


def read_trajectory(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def _dump_json(data: dict) -> str:
    """JSON with every list of numbers on one line, so weight matrices stay readable."""
    text = json.dumps(data, indent=4)
    return re.sub(r"\[\s*([-+.\deE,\s]*?)\s*\]", lambda match: "[" + re.sub(r"\s+", " ", match.group(1)) + "]", text)


def _short(values: np.ndarray) -> list:
    """Six significant digits are plenty for weights and keep the lines short."""
    return np.vectorize(lambda value: float(f"{value:.6g}"), otypes=[float])(values).tolist()


def _format_wrench(wrench: np.ndarray) -> str:
    return "F [{:+8.2f} {:+8.2f} {:+8.2f}] N  T [{:+7.3f} {:+7.3f} {:+7.3f}] Nm".format(*wrench)


def _enter_pressed() -> bool:
    """True if a line was entered on the console since the last call. Always False without a console."""
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return False
    return bool(ready) and sys.stdin.readline() != ""
