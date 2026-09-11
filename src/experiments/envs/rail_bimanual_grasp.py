"""Bimanual rail grasping and cooperative learning environment.

The rail is fixed in the workspace and extends along world X. The graph first moves both
arms slowly to the rail's grasp plane, then lets the operator position each arm in X/Z and
close its gripper -- X matters there, positioning where along the rail's length to grasp.
From there, cooperative_reset (manual, Y/Z + Ry) and cooperative_insert (the learned
primitive, Y/Z translation only) loop against each other for repeated episodes -- X and Ry
are locked once the shared TCP takes over, frozen at whatever they measure when the loop
first activates (see CooperativeFramePrimitive.freeze_driver_axes_at_entry).
"""

from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from lerobot.envs import EnvConfig
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import Robot
from pynput import keyboard

from share.envs.manipulation_primitive.config_manipulation_primitive import (
    PRIMITIVE_COMPLETE_INFO_KEY,
    EventConfig,
    GripperConfig,
    ImagePreprocessingConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    MoveDeltaPrimitiveConfig,
    ObservationConfig,
    ZeroFTPrimitiveConfig,
)
from share.envs.manipulation_primitive.env_manipulation_primitive import ManipulationPrimitive
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.transitions import OnEvent, OnSuccess, OnTargetPoseReached, OnTimeLimit, RewardClassifierTransition
from share.robots.ur import SimURConfig, URConfig
from share.teleoperators import TeleopEvents
from share.teleoperators.spacemouse import SpaceMouseConfig
from share.utils.transformation_utils import (
    euler_xyz_from_rotation,
    get_robot_pose_from_observation,
    get_robot_poses_in_world,
    rotation_from_extrinsic_xyz,
    task_pose_to_world_pose,
    world_pose_to_task_pose,
)

# Units use meters and radians, rotations are rpy unless specified otherwise


# Stable names shared with runtime whose values persist between primitives
RUNTIME_KEY_VTCP_X_OFFSET_M = "cooperative_vtcp_offset_x"  # vTCP x offset along the rail, tuned once in calibrate
RUNTIME_KEY_ROBOT_POSES_FROM_VTCP = "cooperative_robot_poses_in_vtcp"  # each robot pose relative to the vTCP
RUNTIME_KEY_FROZEN_AXES = "cooperative_frozen_axes"  # per-axis driver values frozen at first activation
RECALIBRATE_RY_REQUEST_FLAG = "recalibrate_ry_request"
RECALIBRATE_RY_KEY = keyboard.Key.up

# Fixed policy/action shapes shared by multiple primitives.
# Initial teleop in the x/z grasp plane.
XZ_POLICY = [PolicyMode.RELATIVE, None, PolicyMode.RELATIVE, None, None, None]
# Calibration exposes xyz plus Ry.
CALIBRATION_POLICY = [PolicyMode.RELATIVE, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, PolicyMode.RELATIVE, None]
# cooperative_reset/cooperative_insert's shared policy_mode: x locked (was letting the
# workpiece drift out of camera frame), y/z live, rotation locked (ry frozen via
# freeze_driver_axes_at_entry, see CooperativeFramePrimitive).
YZ_POLICY = [None, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, None, None]
# Pushdown only exposes relative Ry; all other driver axes stay locked.
PUSHDOWN_POLICY = [None, None, None, None, PolicyMode.RELATIVE, None]

class CooperativeFramePrimitive(ManipulationPrimitive):
    """Drives N robots from one virtual TCP (vTCP). The driver's task frame must be
    RELATIVE POS; every other robot must be POS with policy_mode=None. Each step, the driver's
    teleop delta integrates into the vTCP; every robot's target (driver included) is then a
    fresh absolute POS reprojected from the vTCP plus that robot's own fixed offset from it --
    so all robots stay rigidly related by construction, with no per-robot drift or windup.
    Anti-windup for the vTCP itself is left to controller_overrides'
    compliance_reference_limit_enable, same as any RELATIVE axis.

    Every task-frame origin must stay [0]*6; robot_base_pose_in_world carries the base-to-base
    offset between robots instead (each robot's base pose in one shared world frame).

    ``vtcp_offset_x``: a scalar world-x nudge to the vTCP (e.g. toward the rail's true
    contact point, since the raw grasp midpoint rarely is it). Live-adjustable via
    CalibratePrimitive's slider and persisted across activations in
    ``RUNTIME_KEY_VTCP_X_OFFSET_M``, so a value measured once propagates to every later
    cooperative primitive without sharing a Python instance.

    Each robot's fixed offset from the vTCP (the grasp geometry) is captured once, on the
    loop's first activation, and persisted in ``RUNTIME_KEY_ROBOT_POSES_FROM_VTCP`` -- reusing
    it instead of re-deriving it every activation prevents small compliant deflection from
    compounding into drift over many loop iterations. Cleared on a genuine full reset.

    ``freeze_driver_axes_at_entry``: axis indices that hold the driver's measured pose on first
    capture instead of the frame's static target, persisted the same way
    (``RUNTIME_KEY_FROZEN_AXES``) so it isn't re-measured (and can't drift) every activation.
    ``driver_axis_overrides`` pins an exact value instead of measuring it, for run-to-run
    reproducibility -- a rotation axis still needs to be in freeze_driver_axes_at_entry too, or
    the per-step lock overwrites it with the static target; a translation axis doesn't.

    ``apply_task_frames()`` reports the driver's task frame to its own controller as fully
    ABSOLUTE (a per-step deep copy), leaving ``self.task_frame[driver]`` -- and so the action
    processor's live-input passthrough -- untouched.
    """

    def __init__(
        self,
        task_frame: dict[str, TaskFrame],
        robot_dict: dict[str, Robot],
        cameras: dict[str, Any],
        display_cameras: bool = False,
        driver: str = "left",
        fps: float = 30.0,
        robot_base_pose_in_world: dict[str, list[float]] | None = None,
        vtcp_offset_x: float = 0.0,
        freeze_driver_axes_at_entry: tuple[int, ...] = (),
        driver_axis_overrides: dict[int, float] | None = None,
    ):
        super().__init__(task_frame, robot_dict, cameras, display_cameras)
        self.driver = driver
        self.fps = fps
        self.freeze_driver_axes_at_entry = tuple(freeze_driver_axes_at_entry)
        self.driver_axis_overrides = dict(driver_axis_overrides or {})
        self.robot_base_pose_in_world = {name: [0.0] * 6 for name in robot_dict}
        self.robot_base_pose_in_world.update(robot_base_pose_in_world or {})
        self._vtcp_world: list[float] | None = None
        self._offset_from_vtcp: dict[str, list[float]] = {}
        # vTCP x offset is a plain attribute now -- no keyboard events drive it here.
        # CalibratePrimitive is the only place that changes it live (a matplotlib slider);
        # _sync_vtcp_offset_x below applies the current value to the already-captured vTCP,
        # regardless of which source set it -- see there.
        self.vtcp_offset_x = float(vtcp_offset_x)
        self._vtcp_offset_x_applied = 0.0

    def apply_task_frames(self) -> None:
        """Re-send task frames, always reporting the driver's whole task frame to its own
        controller as fully ABSOLUTE (see the class docstring) via a per-step deep copy --
        self.task_frame[driver] itself, and so the action processor's live-input passthrough,
        is left untouched. Followers already report their real policy_mode as-is (always
        ABSOLUTE already), so only the driver needs this override."""
        # ManipulationPrimitive.__init__() calls this (via reset_runtime_state()'s sibling
        # apply_task_frames() call) before this class's own __init__ body has set driver --
        # getattr guards that first call.
        driver = getattr(self, "driver", None)
        for name, robot in self.robot_dict.items():
            if not self._is_task_frame_robot.get(name, False):
                continue
            frame = self.task_frame[name]
            if name == driver:
                frame = copy.deepcopy(frame)
                frame.policy_mode = [None] * len(frame.policy_mode)
            robot.set_task_frame(frame)

    def reset_runtime_state(self) -> None:
        super().reset_runtime_state()
        self._vtcp_world = None
        self._offset_from_vtcp = {}
        self._vtcp_offset_x_applied = 0.0

    def _sync_vtcp_offset_x(self) -> None:
        """Relocate the vTCP's x without moving either robot: recompute each robot's
        offset-from-vTCP so its world target is unchanged at the new vTCP, then move
        vTCP world itself -- moving it alone would drag both robots along with it instead,
        since rotation happens about the vTCP's own position. Diffs against the last-applied
        value, so it's a no-op unless the vTCP x offset actually changed."""
        delta = self.vtcp_offset_x - self._vtcp_offset_x_applied
        if delta == 0.0:
            return
        if self._vtcp_world is not None:
            shifted_vtcp_world = list(self._vtcp_world)
            shifted_vtcp_world[0] += delta
            for name in self._offset_from_vtcp:
                target_before = task_pose_to_world_pose(self._offset_from_vtcp[name], self._vtcp_world)
                self._offset_from_vtcp[name] = world_pose_to_task_pose(target_before, shifted_vtcp_world)
            self._vtcp_world[0] = shifted_vtcp_world[0]
            if self._shared_runtime_values is not None:
                self.set_runtime_value(RUNTIME_KEY_ROBOT_POSES_FROM_VTCP, dict(self._offset_from_vtcp))
        self._vtcp_offset_x_applied = self.vtcp_offset_x

    def _get_reference_error_budget(
        self, controller_overrides: dict[str, Any] | None, axis: int
    ) -> float | None:
        """Return this axis's configured reference-error budget, or None if unbounded."""
        if not controller_overrides:
            return None
        enabled = controller_overrides.get("compliance_reference_limit_enable")
        wrench_limits = controller_overrides.get("wrench_limits")
        kp = controller_overrides.get("kp")
        if not enabled or not wrench_limits or not kp or not enabled[axis]:
            return None
        kp_axis = float(kp[axis])
        if kp_axis <= 0.0:
            return None
        return float(wrench_limits[axis]) / kp_axis

    def _get_measured_midpoint(self, pose_world: dict[str, list[float]]) -> list[float]:
        names = list(self.robot_dict)
        return [sum(pose_world[name][axis] for name in names) / len(names) for axis in range(3)]

    def _initialize_vtcp(self, pose_world: dict[str, list[float]]) -> None:
        names = list(self.robot_dict)
        # Adopt a calibrated/live-nudged vTCP x offset if one was published; else keep
        # this instance's own configured starting value.
        calibrated_offset_x = self.get_runtime_value(RUNTIME_KEY_VTCP_X_OFFSET_M, None)
        if calibrated_offset_x is not None:
            self.vtcp_offset_x = float(calibrated_offset_x)
        midpoint = self._get_measured_midpoint(pose_world)
        self._vtcp_offset_x_applied = self.vtcp_offset_x

        # Default: measured midpoint for translation, static target for rotation.
        # freeze_driver_axes_at_entry swaps in the driver's persisted/measured pose instead;
        # driver_axis_overrides swaps in an exact value and wins over both.
        vtcp_pose = midpoint + list(self.task_frame[self.driver].target[3:6])
        frozen_axes = self.get_runtime_value(RUNTIME_KEY_FROZEN_AXES, {})
        for axis in self.freeze_driver_axes_at_entry:
            vtcp_pose[axis] = frozen_axes.get(axis, pose_world[self.driver][axis])
        for axis, value in self.driver_axis_overrides.items():
            vtcp_pose[axis] = value
        # Publish the frozen baseline before the vTCP x offset is added below -- it must not
        # include the vTCP shift, or it would compound further on every activation.
        if self._shared_runtime_values is not None and self.freeze_driver_axes_at_entry:
            self.set_runtime_value(
                RUNTIME_KEY_FROZEN_AXES,
                {**frozen_axes, **{axis: vtcp_pose[axis] for axis in self.freeze_driver_axes_at_entry}},
            )
        # Apply the vTCP x offset after freeze/override handling so it is a pure relocation.
        vtcp_pose[0] += self.vtcp_offset_x
        self._vtcp_world = vtcp_pose

        calibrated_offset_from_vtcp = self.get_runtime_value(RUNTIME_KEY_ROBOT_POSES_FROM_VTCP, None)
        for name in names:
            if calibrated_offset_from_vtcp is not None and name in calibrated_offset_from_vtcp:
                self._offset_from_vtcp[name] = list(calibrated_offset_from_vtcp[name])
                continue
            capture_pose = list(pose_world[name])
            for axis, value in self.driver_axis_overrides.items():
                capture_pose[axis] = value
            self._offset_from_vtcp[name] = world_pose_to_task_pose(capture_pose, self._vtcp_world)
        if self._shared_runtime_values is not None:
            self.set_runtime_value(RUNTIME_KEY_ROBOT_POSES_FROM_VTCP, dict(self._offset_from_vtcp))

    def _clip_vtcp_to_midpoint(self, pose_world: dict[str, list[float]]) -> None:
        """Keep vTCP translation within the configured reference budget of the measured midpoint."""
        names = list(self.robot_dict)
        midpoint = self._get_measured_midpoint(pose_world)
        for axis in range(3):
            budgets = [
                self._get_reference_error_budget(self.task_frame[name].controller_overrides, axis)
                for name in names
            ]
            finite_budgets = [budget for budget in budgets if budget is not None]
            if not finite_budgets:
                continue
            budget = min(finite_budgets)
            self._vtcp_world[axis] = min(
                max(self._vtcp_world[axis], midpoint[axis] - budget),
                midpoint[axis] + budget,
            )

    def step(self, action: dict[str, dict[str, float]]):
        pose_world = get_robot_poses_in_world(
            observations={name: robot.get_observation() for name, robot in self.robot_dict.items()},
            task_frame_origins={name: frame.origin for name, frame in self.task_frame.items()},
            robot_base_pose_in_world=self.robot_base_pose_in_world,
        )

        if self._vtcp_world is None:
            self._initialize_vtcp(pose_world)

        self._sync_vtcp_offset_x()

        dt = 1.0 / self.fps
        driver_delta = action.get(self.driver, {})
        d = [float(driver_delta.get(f"{ax}.ee_pos", 0.0)) if self.task_frame[self.driver].policy_mode[i] is not None else 0.0 for i, ax in enumerate(("x", "y", "z", "rx", "ry", "rz"))]

        self._vtcp_world[0] += d[0] * dt
        self._vtcp_world[1] += d[1] * dt
        self._vtcp_world[2] += d[2] * dt
        new_rot = rotation_from_extrinsic_xyz(*[v * dt for v in d[3:]]) * rotation_from_extrinsic_xyz(*self._vtcp_world[3:])
        self._vtcp_world[3:] = euler_xyz_from_rotation(new_rot)

        self._clip_vtcp_to_midpoint(pose_world)

        # Every step, not just on a nudge, so the next primitive picks up the latest value.
        if self._shared_runtime_values is not None:
            self.set_runtime_value(RUNTIME_KEY_VTCP_X_OFFSET_M, self.vtcp_offset_x)

        cooperative_action: dict[str, dict[str, float]] = {}
        fixed_rotation_axes = {
            axis
            for axis in range(3, 6)
            if self.task_frame[self.driver].policy_mode[axis] is None
        }
        follower_fixed_rotation_axes = fixed_rotation_axes or set(range(3, 6))

        for name, frame in self.task_frame.items():
            # Every robot, driver included, gets a plain absolute target reprojected from the
            # one shared vTCP -- see the class docstring.
            target_world = task_pose_to_world_pose(self._offset_from_vtcp[name], self._vtcp_world)
            native_target = world_pose_to_task_pose(target_world, self.robot_base_pose_in_world[name])
            values = world_pose_to_task_pose(native_target, frame.origin)

            rotation_axes_to_lock = fixed_rotation_axes if name == self.driver else follower_fixed_rotation_axes
            for axis in rotation_axes_to_lock:
                if axis in self.freeze_driver_axes_at_entry:
                    # target_world already reflects the frozen value; overwriting with the
                    # static frame.target here would snap it back to the un-calibrated default.
                    continue
                if axis < len(frame.target):
                    values[axis] = frame.target[axis]
            cooperative_action[name] = dict(zip(("x.ee_pos", "y.ee_pos", "z.ee_pos", "rx.ee_pos", "ry.ee_pos", "rz.ee_pos"), values))
            if "gripper.pos" in action.get(name, {}):
                cooperative_action[name]["gripper.pos"] = action[name]["gripper.pos"]

        obs, reward, terminated, truncated, info = super().step(cooperative_action)
        # Live readout for record.py's status line; subclasses append rather than overwrite.
        info["record_status"] = f"vtcp_x = {self.vtcp_offset_x:+.4f} m"
        return obs, reward, terminated, truncated, info

    def get_display_points(self) -> dict[str, list[float]]:
        """Each robot's current TCP position plus "vTCP" once the vTCP has been captured
        (absent only before this activation's first step)."""
        pose_world = get_robot_poses_in_world(
            observations={name: robot.get_observation() for name, robot in self.robot_dict.items()},
            task_frame_origins={name: frame.origin for name, frame in self.task_frame.items()},
            robot_base_pose_in_world=self.robot_base_pose_in_world,
        )
        points = {name: list(pose[:3]) for name, pose in pose_world.items()}
        if self._vtcp_world is not None:
            points["vtcp"] = list(self._vtcp_world[:3])
        return points


def _processor(
    fps: float,
    gripper: GripperConfig,
    observation: ObservationConfig | None = None,
    extra_key_events: dict[str, Any] | None = None,
) -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=fps,
        observation=observation or ObservationConfig(
            add_ee_pos_to_observation=False,
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=False,
            add_joint_position_to_observation=False,
        ),
        image_preprocessing=ImagePreprocessingConfig(
            resize_size=[64, 64],
            crop_params_dict={
                "left": [174, 166, 186, 227]
            }
        ),
        gripper=gripper,
        events=EventConfig(
            key_mapping={
                TeleopEvents.SUCCESS: keyboard.Key.space,
                TeleopEvents.RERECORD_EPISODE: keyboard.Key.left,
                TeleopEvents.STOP_RECORDING: keyboard.Key.down,
                **(extra_key_events or {}),
            },
            pulse_events=(TeleopEvents.SUCCESS,),
        ),
    )


def _gripper(live: str | None, static: dict[str, float]) -> GripperConfig:
    enable = {name: name == live for name in ("left", "right")}
    static_pos: dict[str, float | None] = {name: static.get(name) for name in ("left", "right")}
    if live is not None:
        static_pos[live] = None
    return GripperConfig(enable=enable, discretize=True, min_pos=0.0, static_pos=static_pos)


def _single_arm_frames(
    active: str,
    passive: str,
    target_poses: dict[str, list[float]],
    controller_overrides: dict[str, Any] | None = None,
) -> dict[str, TaskFrame]:
    return {
        active: TaskFrame(
            target=list(target_poses[active]),
            policy_mode=list(XZ_POLICY),
            controller_overrides=controller_overrides,
        ),
        passive: TaskFrame(
            target=list(target_poses[passive]),
            policy_mode=[None] * 6,
            controller_overrides=controller_overrides,
        ),
    }


@dataclass
class RailCalibration:
    """Persisted rail calibration and its interactive calibration settings."""

    vtcp_offset_x_m: float | None = None
    driver_x_m: float | None = None
    driver_ry_rad: float | None = None
    vtcp_offset_range_m: float = 0.3
    path: str | None = field(
        default_factory=lambda: str(Path(tempfile.gettempdir()) / "rail_bimanual_grasp_calibration.json")
    )
    load_from_path: bool = False

    def resolve(self) -> None:
        if not self.load_from_path:
            return
        if self.path is None:
            raise ValueError("load_from_path=True requires a calibration path.")
        calibration_path = Path(self.path)
        if not calibration_path.exists():
            raise ValueError(
                f"load_from_path=True but calibration path {calibration_path} doesn't exist."
            )
        payload = json.loads(calibration_path.read_text())

        def read(name: str, *legacy_names: str) -> float:
            value = payload.get(name)
            for legacy_name in legacy_names:
                if value is not None:
                    break
                value = payload.get(legacy_name)
            if value is None:
                raise ValueError(f"Calibration file {calibration_path} is missing '{name}'.")
            return float(value)

        self.vtcp_offset_x_m = read("vtcp_offset_x_m", "pivot_offset_x_m", "pivot_offset_x")
        self.driver_x_m = read("driver_x_m", "x_offset")
        self.driver_ry_rad = read("driver_ry_rad", "ry_angle")

    def save(self) -> None:
        if self.path is None:
            return
        if self.vtcp_offset_x_m is None or self.driver_x_m is None or self.driver_ry_rad is None:
            raise ValueError("A complete rail calibration is required before saving.")
        Path(self.path).write_text(
            json.dumps(
                {
                    "vtcp_offset_x_m": self.vtcp_offset_x_m,
                    "driver_x_m": self.driver_x_m,
                    "driver_ry_rad": self.driver_ry_rad,
                }
            )
        )


class CalibratePrimitive(CooperativeFramePrimitive):
    """The graph's manual setup step: teleop ry to the right value (reported live via
    ``info["record_status"]``), drag the vTCP's x-offset slider toward the true contact point.
    Space ends it; ry then freezes at entry into cooperative_reset (see
    ``CooperativeFramePrimitive.freeze_driver_axes_at_entry``).

    The slider window is created lazily on first ``step()`` (no robot is connected yet at
    ``__init__``) and persists for the process, reused if this primitive is re-entered."""

    def __init__(self, *args, vtcp_offset_range_m: float = 0.3, calibration_path: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._vtcp_offset_range_m = float(vtcp_offset_range_m)
        self._calibration_path = Path(calibration_path) if calibration_path is not None else None
        self._slider_fig = None
        self._slider_x = None

    def _ensure_vtcp_offset_sliders(self) -> None:
        if self._slider_fig is not None:
            return

        initial_x = self.get_runtime_value(RUNTIME_KEY_VTCP_X_OFFSET_M, self.vtcp_offset_x)
        r = self._vtcp_offset_range_m
        fig, ax_x = plt.subplots(figsize=(5, 1.3))
        fig.canvas.manager.set_window_title("vTCP offset")
        self._slider_x = Slider(ax_x, "vTCP x offset (m)", -r, r, valinit=initial_x)
        fig.tight_layout()
        plt.show(block=False)
        self._slider_fig = fig
        self.vtcp_offset_x = initial_x

    def step(self, action: dict[str, dict[str, float]]):
        self._ensure_vtcp_offset_sliders()
        self.vtcp_offset_x = self._slider_x.val
        plt.pause(0.001)

        obs, reward, terminated, truncated, info = super().step(action)
        try:
            driver_pose = get_robot_pose_from_observation(obs, self.driver)
            ry = driver_pose[4]
            info["record_status"] = f"ry = {ry:+.4f} rad  |  {info['record_status']}"
            self._save_calibration(driver_x_m=driver_pose[0], driver_ry_rad=ry)
        except KeyError:
            pass
        return obs, reward, terminated, truncated, info

    def _save_calibration(self, driver_x_m: float, driver_ry_rad: float) -> None:
        """Written every step this primitive is active (it has no visibility into the SUCCESS
        keypress that ends it) -- the last write before moving on is what sticks."""
        if self._calibration_path is None:
            return
        RailCalibration(
            vtcp_offset_x_m=self.vtcp_offset_x,
            driver_x_m=driver_x_m,
            driver_ry_rad=driver_ry_rad,
            path=str(self._calibration_path),
        ).save()

    def close(self) -> None:
        if self._slider_fig is not None:
            plt.close(self._slider_fig)
        super().close()


@EnvConfig.register_subclass("rail_bimanual_grasp")
@dataclass
class RailBimanualGraspEnvConfig(ManipulationPrimitiveNetConfig):
    """align_to_rail -> teleop_left -> teleop_right -> zero_ft -> calibrate ->
    zero_ft_before_cooperative -> cooperative_reset ⇄ cooperative_insert, looping for repeated
    episodes. The approach/grasp/calibration chain only runs on the very first reset; none of
    these primitives is_terminal, so the cooperative loop never triggers a second full reset.
    Teleop cooperative_reset by hand (Y/Z only -- X/Ry frozen on entry, see
    CooperativeFramePrimitive.freeze_driver_axes_at_entry); space starts cooperative_insert
    (same DOF, learned); on success, back to cooperative_reset for the next episode."""

    fps: int = 10
    left_robot_ip: str = "172.22.22.5"
    right_robot_ip: str = "172.22.22.2"
    right_base_pose_in_left_base: list[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    # Nominal rail-plane grasp poses in [x, y, z, rx, ry, rz]. Ry is teleoped in calibrate,
    # then frozen once cooperative_reset/cooperative_insert take over.
    rail_target_poses: dict[str, list[float]] = field(
        default_factory=lambda: {
            "right": [0.0, -0.30186, 0.0, float(np.pi), 0.0, 0.0],
            "left": [0.0, -0.29813, 0.0, float(np.pi), 0.0, -float(np.pi)],
        }
    )
    alignment_linear_speed_mps: float = 0.05
    alignment_angular_speed_rad_s: float = 0.10
    translation_action_scale: float = 0.1
    rotation_action_scale: float = 0.5
    # Expected side length of the y/z square cooperative_insert's driver moves within from
    # episode start -- half of this is the dataset_stats bound for the relative-position state
    # delta-position channel. Not enforced, just the assumed normalization range.
    cooperative_insert_position_range_m: float = 0.05
    open_gripper_position: float = 0.0
    closed_gripper_position: float = 1.0
    mock: bool = False
    start_primitive: str = "align_to_rail"
    reset_primitive: str = "align_to_rail"
    calibration: RailCalibration = field(default_factory=RailCalibration)
    # Mid-air support: skip the initial grasp sequence and jump straight into calibration
    # (or zero_ft_before_cooperative when loading calibration). Also used as reset_primitive.
    # ry still comes out right without re-running calibrate:
    # freeze_driver_axes_at_entry reads the arm's actual physical pose.
    skip_grasp: bool = False
    # Force-mode singularities can leave the arms stuck under cooperative_reset's compliant
    # teleop. With skip_grasp=True, makes cooperative_reset rigid/servo instead (same as
    # align_to_rail/zero_ft) so the SpaceMouse can drive the arms clear. A recovery switch, not
    # a normal operating mode -- turn back off once clear.
    cooperative_reset_servo_recovery: bool = False
    # zero_ft re-zeros both F/T sensors once both grippers hold the workpiece -- whatever
    # zero_ft() set at connect() is stale by then. Runs right before the compliant hold phase
    # begins, so admittance doesn't chase a phantom residual wrench.
    zero_ft_settle_duration_s: float = 0.3
    # Optional automatic success signal for cooperative_insert -> cooperative_reset, alongside
    # the always-available manual OnSuccess (space). Additional, not a replacement -- see
    # RewardClassifierTransition.
    reward_classifier_path: str | None = None
    reward_classifier_threshold: float = 0.7
    reward_classifier_device: str = "cuda"
    # cooperative_reset/cooperative_insert are translation-only against a rigid rail -- cap
    # contact force well below the shared 30 N. Rotation and calibrate are untouched.
    cooperative_translation_wrench_limit_n: float = 10.0
    # Auto-truncate cooperative_insert if neither SUCCESS nor reward_classifier_path has fired
    # yet, to keep a stuck episode from running forever.
    cooperative_insert_episode_duration_s: float = 5.0
    # World-frame +z distance both arms lift straight up by in pull_out, on a genuine success
    # only (never on the OnTimeLimit fallback -- nothing to pull out of there).
    pull_out_height_m: float = 0.02
    # When set, a successful cooperative_insert routes to pushdown instead of pull_out.
    pushdown: bool = False
    # rz stiffness/force cap in pushdown, well below the shared 150/4.0, so the rail settles
    # into whatever yaw the slot wants instead of fighting it.
    pushdown_yaw_kp: float = 15.0
    pushdown_yaw_wrench_limit_nm: float = 1.0

    def _validate_configuration(self) -> None:
        if self.calibration.load_from_path and not self.skip_grasp:
            raise ValueError(
                "calibration.load_from_path requires skip_grasp=True because the arms must already "
                "be in the mid-air cooperative setup."
            )
        self.calibration.resolve()
        if self.cooperative_reset_servo_recovery and not self.skip_grasp:
            raise ValueError(
                "cooperative_reset_servo_recovery requires skip_grasp=True -- it only makes "
                "sense entering straight into the stuck loop, not walking the ordinary grasp "
                "chain into it."
            )
        if self.calibration.driver_ry_rad is not None and not (
            self.skip_grasp or self.calibration.load_from_path
        ):
            raise ValueError(
                "calibration.driver_ry_rad requires skip_grasp=True or "
                "calibration.load_from_path=True -- calibrate is unreachable either way."
            )

    def _build_controller_overrides(self) -> dict[str, dict[str, Any]]:
        controller = {
            "use_force_mode": True,
            "compliance_reference_limit_enable": [True] * 6,
            "kp": [3000.0, 3000.0, 3000.0, 150.0, 150.0, 150.0],
            "kd": [60.0, 60.0, 60.0, 6.0, 6.0, 6.0],
            "wrench_limits": [30.0, 30.0, 30.0, 4.0, 4.0, 4.0],
        }
        alignment = dict(controller)
        alignment.update(use_force_mode=False, simple_pose_use_servo=True)

        cooperative = dict(controller)
        cooperative["wrench_limits"] = [
            self.cooperative_translation_wrench_limit_n,
        ] * 3 + list(controller["wrench_limits"][3:])

        cooperative_reset = cooperative
        if self.cooperative_reset_servo_recovery:
            cooperative_reset = dict(controller)
            cooperative_reset.update(
                use_force_mode=False,
                simple_pose_use_servo=True,
                compliance_reference_limit_enable=[False] * 6,
            )

        calibrate = dict(controller)
        calibrate["compliance_reference_limit_enable"] = [False] * 6

        pushdown = dict(cooperative)
        pushdown["compliance_reference_limit_enable"] = [True] * 3 + [False] * 3
        pushdown["kp"] = list(cooperative["kp"])
        pushdown["kp"][5] = self.pushdown_yaw_kp
        pushdown["wrench_limits"] = list(cooperative["wrench_limits"])
        pushdown["wrench_limits"][5] = self.pushdown_yaw_wrench_limit_nm

        zero_ft = dict(controller)
        zero_ft.update(use_force_mode=False, simple_pose_use_servo=True)
        return {
            "controller": controller,
            "alignment": alignment,
            "cooperative": cooperative,
            "cooperative_reset": cooperative_reset,
            "calibrate": calibrate,
            "pushdown": pushdown,
            "zero_ft": zero_ft,
        }

    def _configure_hardware(self) -> None:
        self.cameras = {"left": OpenCVCameraConfig(index_or_path="/dev/video0")}
        if self.mock:
            self.robot = {
                name: SimURConfig(
                    use_gripper=True,
                    initial_pose=[0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                )
                for name in ("left", "right")
            }
        else:
            speed = [self.alignment_linear_speed_mps] * 3 + [self.alignment_angular_speed_rad_s] * 3
            self.robot = {
                "left": URConfig(
                    robot_ip=self.left_robot_ip,
                    frequency=125,
                    soft_real_time=True,
                    simple_pose_max_speed=speed,
                    rt_core=3,
                    use_gripper=True,
                ),
                "right": URConfig(
                    robot_ip=self.right_robot_ip,
                    frequency=125,
                    soft_real_time=True,
                    simple_pose_max_speed=speed,
                    rt_core=4,
                    use_gripper=True,
                    debug=False,
                    debug_axis=4,
                ),
            }
        self.teleop = SpaceMouseConfig(
            action_scale=[
                self.translation_action_scale,
                self.translation_action_scale,
                self.translation_action_scale,
                self.rotation_action_scale,
                self.rotation_action_scale,
                self.rotation_action_scale,
            ],
            gripper_close_button_idx=None,
            gripper_open_button_idx=None,
            button_mapping={
                0: {"event": TeleopEvents.SUCCESS, "toggle": False},
                1: {"event": TeleopEvents.IS_INTERVENTION, "toggle": False},
            },
        )

    def _build_processors(self) -> dict[str, ManipulationPrimitiveProcessorConfig]:
        closed_grippers = _gripper(
            live=None,
            static={"left": self.closed_gripper_position, "right": self.closed_gripper_position},
        )
        def cooperative_observation() -> ObservationConfig:
            return ObservationConfig(
                add_ee_pos_to_observation=False,
                add_ee_velocity_to_observation={"left": True, "right": False},
                ee_velocity_axes=["x.ee_vel", "y.ee_vel", "z.ee_vel"],
                add_ee_wrench_to_observation=False,
                add_joint_position_to_observation=False,
            )
        return {
            "alignment": _processor(
                self.fps,
                _gripper(
                    live=None,
                    static={"left": self.open_gripper_position, "right": self.open_gripper_position},
                ),
            ),
            "left_grasp": _processor(
                self.fps,
                _gripper(
                    live=None,
                    static={"left": self.open_gripper_position, "right": self.open_gripper_position},
                ),
            ),
            "right_grasp": _processor(
                self.fps,
                _gripper(
                    live=None,
                    static={"left": self.closed_gripper_position, "right": self.open_gripper_position},
                ),
            ),
            "cooperative": _processor(self.fps, closed_grippers, observation=cooperative_observation()),
            "cooperative_reset": _processor(
                self.fps,
                _gripper(
                    live=None,
                    static={"left": self.closed_gripper_position, "right": self.closed_gripper_position},
                ),
                observation=cooperative_observation(),
                extra_key_events={RECALIBRATE_RY_REQUEST_FLAG: RECALIBRATE_RY_KEY},
            ),
            "cooperative_insert": _processor(
                self.fps,
                _gripper(
                    live=None,
                    static={"left": self.closed_gripper_position, "right": self.closed_gripper_position},
                ),
                observation=ObservationConfig(
                    add_ee_pos_to_observation={"left": True, "right": False},
                    ee_pos_axes=["y.ee_pos", "z.ee_pos"],
                    relative_ee_pos={"left": True, "right": False},
                    add_ee_velocity_to_observation={"left": True, "right": False},
                    ee_velocity_axes=["y.ee_vel", "z.ee_vel"],
                    add_previous_action_to_observation=True,
                    add_ee_wrench_to_observation=False,
                    add_joint_position_to_observation=False,
                ),
            ),
        }

    def __post_init__(self) -> None:
        self._validate_configuration()
        overrides = self._build_controller_overrides()
        self._configure_hardware()
        processors = self._build_processors()

        alignment_frames = {
            name: TaskFrame(
                target=list(self.rail_target_poses[name]),
                policy_mode=[None] * 6,
                controller_overrides=overrides["alignment"],
            )
            for name in ("left", "right")
        }
        # x/z hold entry pose (locked, not in absolute_axes); y/rx/ry/rz resolve straight to
        # the configured rail_target_poses target.
        alignment = MoveDeltaPrimitiveConfig(
            notes="Move both arms slowly into the rail plane and fixed grasp orientation, holding entry x/z.",
            processor=processors["alignment"],
            task_frame=alignment_frames,
            absolute_axes={"left": ["y", "rx", "ry", "rz"], "right": ["y", "rx", "ry", "rz"]},
        )

        left_grasp = ManipulationPrimitiveConfig(
            notes="Teleop the left grasp in X/Z only, gripper held open; SUCCESS (either "
            "SpaceMouse button) moves on to right_grasp, whose static target closes it.",
            processor=processors["left_grasp"],
            task_frame=_single_arm_frames("left", "right", self.rail_target_poses, overrides["controller"]),
            teleop_mapping={"left": "main"},
        )
        right_grasp = ManipulationPrimitiveConfig(
            notes="Teleop the right grasp in X/Z only, gripper held open (left is already "
            "closed, statically, the moment this primitive activates); SUCCESS (either "
            "SpaceMouse button) moves on to zero_ft, whose static target closes right too.",
            processor=processors["right_grasp"],
            task_frame=_single_arm_frames("right", "left", self.rail_target_poses, overrides["controller"]),
            teleop_mapping={"right": "main"},
        )

        cooperative_env_kwargs = {
            "driver": "left",
            "fps": float(self.fps),
            "robot_base_pose_in_world": {
                "right": list(self.right_base_pose_in_left_base),
            },
            "vtcp_offset_x": (
                0.0 if self.calibration.vtcp_offset_x_m is None else self.calibration.vtcp_offset_x_m
            ),
        }
        # x and ry are frozen at whatever they measure on the loop's first activation;
        # explicit calibration values override either with a known-good exact value instead.
        axis_overrides = {}
        if self.calibration.driver_x_m is not None:
            axis_overrides[0] = self.calibration.driver_x_m
        if self.calibration.driver_ry_rad is not None:
            axis_overrides[4] = self.calibration.driver_ry_rad
        cooperative_insert_env_kwargs = {
            **cooperative_env_kwargs,
            "freeze_driver_axes_at_entry": (0, 4),
            "driver_axis_overrides": axis_overrides,
        }

        def zero_ft_primitive(notes: str) -> ZeroFTPrimitiveConfig:
            return ZeroFTPrimitiveConfig(
                notes=notes,
                processor=processors["cooperative"],
                task_frame={
                    name: TaskFrame(
                        target=list(self.rail_target_poses[name]),
                        policy_mode=[None] * 6,
                        controller_overrides=overrides["zero_ft"],
                    )
                    for name in ("left", "right")
                },
                settle_duration_s=self.zero_ft_settle_duration_s,
            )

        # First zero: right after grasping. Second: right before cooperative_reset, catching
        # any residual bias from teleoperating ry in calibrate -- also the skip_grasp entry point.
        zero_ft = zero_ft_primitive(
            "Re-zero both F/T sensors now that both grippers are loaded with the workpiece."
        )
        zero_ft_before_cooperative = zero_ft_primitive(
            "Re-zero both F/T sensors once more right before the compliant hold phase begins "
            "-- also the skip_grasp entry point, since teleop_left/teleop_right/calibrate "
            "are unreachable there."
        )

        calibrate = ManipulationPrimitiveConfig(
            notes="Teleop ry to the right value (printed live in the status line), and drag "
            "the vTCP x-offset slider (a separate matplotlib window) toward the actual "
            "contact point; space moves on. rz is never touched. Once you know good values, "
            "calibration values with skip_grasp=True reuse them exactly instead of measuring/"
            "dragging again -- this primitive is unreachable there anyway.",
            processor=processors["cooperative"],
            env_class=CalibratePrimitive,
            env_kwargs={
                **cooperative_env_kwargs,
                "vtcp_offset_range_m": self.calibration.vtcp_offset_range_m,
                "calibration_path": self.calibration.path,
            },
            task_frame={
                "left": TaskFrame(
                    target=list(self.rail_target_poses["left"]),
                    policy_mode=CALIBRATION_POLICY,
                    controller_overrides=overrides["calibrate"],
                ),
                "right": TaskFrame(
                    target=list(self.rail_target_poses["right"]),
                    policy_mode=[None] * 6,
                    controller_overrides=overrides["calibrate"],
                ),
            },
            teleop_mapping={"left": "main"},
        )

        cooperative_reset = ManipulationPrimitiveConfig(
            notes="Teleop the shared vTCP (Y/Z translation only -- X and Ry stay frozen at "
            "whatever they measure on entry) into the episode's initial state; space starts "
            "cooperative_insert. Up arrow jumps back to calibrate to re-teleop ry.",
            processor=processors["cooperative_reset"],
            task_frame={
                "left": TaskFrame(
                    target=list(self.rail_target_poses["left"]),
                    policy_mode=YZ_POLICY,
                    controller_overrides=overrides["cooperative_reset"],
                ),
                "right": TaskFrame(
                    target=list(self.rail_target_poses["right"]),
                    policy_mode=[None] * 6,
                    controller_overrides=overrides["cooperative_reset"],
                ),
            },
            env_class=CooperativeFramePrimitive,
            env_kwargs=cooperative_insert_env_kwargs,
            teleop_mapping={"left": "main"},
        )

        cooperative_insert = ManipulationPrimitiveConfig(
            notes="Learn cooperative Y/Z translation only -- X and Ry are frozen at whatever "
            "they measure on entry (or explicit calibration values, if set), Rx/Rz stay at their locked "
            "targets as always. Set pushdown=True for manual seating, or use a policy via actor_server.py/"
            "learner_server.py. Success via space, or reward_classifier_path if supplied.",
            processor=processors["cooperative_insert"],
            task_frame={
                "left": TaskFrame(
                    target=list(self.rail_target_poses["left"]),
                    policy_mode=YZ_POLICY,
                    controller_overrides=overrides["cooperative"],
                ),
                "right": TaskFrame(
                    target=list(self.rail_target_poses["right"]),
                    policy_mode=[None] * 6,
                    controller_overrides=overrides["cooperative"],
                ),
            },
            # Action is 2-dim (x/z translation only); an unset dataset_stats silently falls
            # back to SACConfig's placeholder shape/range, which only checks shape, not values.
            # Mirrors InsertionSACConfig, but with a real pretrained backbone (frozen) instead
            # of the small default encoder trained from scratch -- standard for vision SAC.
            policy=SACConfig(
                device="cuda",
                storage_device="cpu",
                dataset_stats={
                    "action": {
                        "min": [-self.translation_action_scale] * 2,
                        "max": [self.translation_action_scale] * 2,
                    },
                    # Order matches ObservationConfig: relative y/z position, then y/z velocity,
                    # then previous y/z action. Position bound is half of
                    # cooperative_insert_position_range_m; velocity/action share
                    # translation_action_scale.
                    "observation.state": {
                        "min": (
                            [-self.cooperative_insert_position_range_m / 2] * 2
                            + [-self.translation_action_scale] * 4
                        ),
                        "max": (
                            [self.cooperative_insert_position_range_m / 2] * 2
                            + [self.translation_action_scale] * 4
                        ),
                    },
                },
                vision_encoder_name="helper2424/resnet10",
                freeze_vision_encoder=True,
                use_amp=False,
                online_steps=int(1e8),
                async_prefetch=True,
                online_step_before_learning=300,
                online_buffer_capacity=30000,
                offline_buffer_capacity=10000,
                utd_ratio=3,
                shared_encoder=True,
                num_critics=2,
                target_entropy=-1.5,
                critic_target_update_weight=0.003,
                use_backup_entropy=False,
            ),
            env_class=CooperativeFramePrimitive,
            env_kwargs=cooperative_insert_env_kwargs,
            teleop_mapping={"left": "main"},
        )

        # Scripted: hold x/y/rotation, move only z. Stays compliant rather than rigid --
        # extracting from a still-engaged insertion should give the same way a bind would.
        pull_out = MoveDeltaPrimitiveConfig(
            notes="Lift both arms pull_out_height_m straight up (world +z) after a successful "
            "cooperative_insert, holding x/y/rotation wherever they are; then back to "
            "cooperative_reset.",
            processor=processors["cooperative"],
            delta_frame="world",
            delta=[0.0, 0.0, self.pull_out_height_m, 0.0, 0.0, 0.0],
            task_frame={
                name: TaskFrame(
                    target=list(self.rail_target_poses[name]),
                    policy_mode=[None] * 6,
                    controller_overrides=overrides["cooperative"],
                )
                for name in ("left", "right")
            },
        )

        pushdown_env_kwargs = {
            **cooperative_env_kwargs,
            # ry must stay frozen here too -- otherwise it snaps to its static 0.0 the instant
            # pushdown activates instead of continuing from wherever cooperative_insert left it.
            "freeze_driver_axes_at_entry": (0, 4),
            "driver_axis_overrides": (
                {0: self.calibration.driver_x_m}
                if self.calibration.driver_x_m is not None
                else {}
            ),
        }
        pushdown = ManipulationPrimitiveConfig(
            notes="Manual exploration of seating the rail fully after a successful "
            "cooperative_insert (pushdown=True only): x/y locked wherever cooperative_insert left "
            "them, z remains where cooperative_insert left it, rz soft (settles "
            "into the slot's actual yaw instead of fighting it), ry live -- teleop it toward "
            "0deg by hand, no movement unless actively teleoped. Space back to "
            "cooperative_reset.",
            processor=processors["cooperative"],
            task_frame={
                "left": TaskFrame(
                    target=list(self.rail_target_poses["left"]),
                    # x/y/z all locked, rx/rz at their static target (rz softened via controller
                    # overrides, not policy_mode), ry live -- teleop it down toward 0deg by hand.
                    policy_mode=PUSHDOWN_POLICY,
                    controller_overrides=overrides["pushdown"],
                ),
                "right": TaskFrame(
                    target=list(self.rail_target_poses["right"]),
                    policy_mode=[None] * 6,
                    controller_overrides=overrides["pushdown"],
                ),
            },
            env_class=CooperativeFramePrimitive,
            env_kwargs=pushdown_env_kwargs,
            teleop_mapping={"left": "main"},
        )

        self.primitives = {
            "align_to_rail": alignment,
            "teleop_left": left_grasp,
            "teleop_right": right_grasp,
            "zero_ft": zero_ft,
            "calibrate": calibrate,
            "zero_ft_before_cooperative": zero_ft_before_cooperative,
            "cooperative_reset": cooperative_reset,
            "cooperative_insert": cooperative_insert,
            "pull_out": pull_out,
            "pushdown": pushdown,
        }

        transitions = [
            OnTargetPoseReached(
                source="align_to_rail",
                target="teleop_left",
                axes=[0, 1, 2, 3, 4, 5],
                tolerance=[0.003, 0.003, 0.003, 0.03, 0.03, 0.03],
            ),
            OnSuccess(source="teleop_left", target="teleop_right"),
            OnSuccess(source="teleop_right", target="zero_ft"),
            # zero_ft primitives are scripted, so they advance on their own completion flag.
            # Loading calibration bypasses calibrate entirely, reusing the loaded values.
            OnEvent(
                source="zero_ft",
                target=(
                    "zero_ft_before_cooperative"
                    if self.calibration.load_from_path
                    else "calibrate"
                ),
                event_key=PRIMITIVE_COMPLETE_INFO_KEY,
            ),
            OnSuccess(source="calibrate", target="zero_ft_before_cooperative"),
            OnEvent(
                source="zero_ft_before_cooperative",
                target="cooperative_reset",
                event_key=PRIMITIVE_COMPLETE_INFO_KEY,
            ),
            # Escape hatch: re-teleop ry without a full reset (up arrow, not space).
            OnEvent(
                source="cooperative_reset",
                target="calibrate",
                event_key=RECALIBRATE_RY_REQUEST_FLAG,
            ),
            OnSuccess(source="cooperative_reset", target="cooperative_insert"),
        ]
        # Manual success always works; a reward classifier is an additional trigger, not a
        # replacement. The time limit targets cooperative_reset directly, not pull_out --
        # nothing was achieved on a timeout, so there's nothing to pull out of.
        transitions.append(
            OnTimeLimit(
                source="cooperative_insert",
                target="cooperative_reset",
                max_steps=int(self.fps * self.cooperative_insert_episode_duration_s),
            )
        )
        # pushdown=True routes a genuine success there instead of the normal pull_out path.
        insert_success_target = "pushdown" if self.pushdown else "pull_out"
        transitions.append(OnSuccess(source="cooperative_insert", target=insert_success_target))
        if self.reward_classifier_path is not None:
            transitions.append(
                RewardClassifierTransition(
                    source="cooperative_insert",
                    target=insert_success_target,
                    pretrained_path=self.reward_classifier_path,
                    threshold=self.reward_classifier_threshold,
                    device=self.reward_classifier_device,
                )
            )
        transitions.append(
            OnTargetPoseReached(source="pull_out", target="cooperative_reset", axes=[2], tolerance=0.003)
        )
        transitions.append(OnSuccess(source="pushdown", target="cooperative_reset"))
        self.transitions = transitions

        if self.skip_grasp:
            # Straight into calibrate, skipping zero_ft_before_cooperative's F/T re-zero for
            # faster iteration; loading calibration skips calibrate too.
            self.start_primitive = (
                "zero_ft_before_cooperative"
                if self.calibration.load_from_path
                else "calibrate"
            )
            self.reset_primitive = self.start_primitive

        super().__post_init__()


__all__ = [
    "CALIBRATION_POLICY",
    "CooperativeFramePrimitive",
    "RailCalibration",
    "RUNTIME_KEY_FROZEN_AXES",
    "RECALIBRATE_RY_KEY",
    "RECALIBRATE_RY_REQUEST_FLAG",
    "RailBimanualGraspEnvConfig",
    "CalibratePrimitive",
    "PUSHDOWN_POLICY",
    "YZ_POLICY",
]
