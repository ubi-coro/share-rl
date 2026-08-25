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
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from share.envs.manipulation_primitive.task_frame import ControlMode, ControlSpace, PolicyMode, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.transitions import OnEvent, OnSuccess, OnTargetPoseReached, OnTimeLimit, RewardClassifierTransition
from share.robots.ur import SimURConfig, URConfig
from share.teleoperators import TeleopEvents
from share.teleoperators.spacemouse import SpaceMouseConfig
from share.utils.transformation_utils import (
    euler_xyz_from_rotation,
    euler_xyz_from_rotvec,
    get_robot_pose_from_observation,
    rotation_from_extrinsic_xyz,
    task_pose_to_world_pose,
    world_pose_to_task_pose,
    wrap_to_pi,
)

# CooperativeFramePrimitive and its module-level dependencies below are a deliberate,
# self-contained copy of the class this env used to import from experiments.envs.bimanual_pick
# -- that module has been archived (see src/experiments/envs/archive/) and this env no longer
# depends on it at all. Keep this copy in sync by hand if bimanual_pick.py's own copy is ever
# revived and diverges; there is intentionally no shared import between them anymore.

# Shared-runtime-value key every CooperativeFramePrimitive activation (calibration included)
# reads/writes to hand the live pivot_offset_x scalar off across a primitive switch -- see the
# CooperativeFramePrimitive docstring below.
PIVOT_OFFSET_X_RUNTIME_KEY = "cooperative_pivot_offset_x"

# Shared-runtime-value key every CooperativeFramePrimitive activation reads/writes to hand off
# each robot's fixed offset from the vtcp -- the grasp geometry between robots -- captured once
# and reused thereafter instead of being re-derived every activation. See the
# CooperativeFramePrimitive docstring below.
OFFSET_FROM_VTCP_RUNTIME_KEY = "cooperative_offset_from_vtcp"

# Same idea, for freeze_driver_axes_at_entry: the first activation of any cooperative primitive
# in a loop captures each frozen axis's value; every activation after that reuses it instead of
# re-measuring.
FROZEN_AXES_RUNTIME_KEY = "cooperative_frozen_axes"


def _pose_from_raw_obs(obs: dict[str, float]) -> list[float]:
    rotvec = [obs["rx.ee_pos"], obs["ry.ee_pos"], obs["rz.ee_pos"]]
    return [obs["x.ee_pos"], obs["y.ee_pos"], obs["z.ee_pos"], *euler_xyz_from_rotvec(rotvec)]


def _origin_from_raw_obs(obs: dict[str, float], fallback: list[float]) -> list[float]:
    # robot.set_task_frame() only updates the robot wrapper's local attribute; a changed
    # origin only reaches the controller on the next send_action(). Until then, observations
    # are still expressed in whatever origin the controller actually has, which the controller
    # reports per-sample as task_frame_origin -- trust that over the primitive's configured
    # origin, which may not have taken effect on the controller yet. SimUR/mocks don't report
    # this field (they don't model origin transitions at all), hence the fallback. Defensive:
    # every robot here keeps origin=[0]*6 for its whole lifetime, so this should never actually
    # observe a mismatch, but there's no reason to rely on that not changing in the future.
    if "x.task_frame_origin" not in obs:
        return fallback
    return [obs[f"{ax}.task_frame_origin"] for ax in ("x", "y", "z", "rx", "ry", "rz")]


def _poses_world(
    robot_dict: dict[str, Robot],
    task_frame: dict[str, TaskFrame],
    robot_base_pose_in_world: dict[str, list[float]],
) -> dict[str, list[float]]:
    """Each robot's current TCP pose in one shared world frame (see robot_base_pose_in_world
    docs on CooperativeFramePrimitive). Shared by every primitive/net class in this file that
    needs a consistent notion of "world" -- duplicating this composition anywhere is exactly
    how the base-relation bug got introduced in the first place."""
    poses = {}
    for name, robot in robot_dict.items():
        obs = robot.get_observation()
        active_origin = _origin_from_raw_obs(obs, task_frame[name].origin)
        native_pose = task_pose_to_world_pose(_pose_from_raw_obs(obs), active_origin)
        poses[name] = task_pose_to_world_pose(native_pose, robot_base_pose_in_world.get(name, [0.0] * 6))
    return poses


def _reference_error_budget(controller_overrides: dict[str, Any] | None, axis: int) -> float | None:
    """This axis's reference-error budget (wrench_limits[axis]/kp[axis]), read straight off a
    controller_overrides dict -- mirrors controller.py's own _get_reference_error_limit
    exactly, just off config instead of live controller state. None if
    compliance_reference_limit_enable/wrench_limits/kp don't all supply a usable value for
    this axis (unbounded, i.e. do nothing), matching what an unset override already means at
    the controller."""
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


class CooperativeFramePrimitive(ManipulationPrimitive):
    """Drives N robots from one teleop input via a shared virtual midpoint frame.

    On first step, captures each robot's fixed offset from the midpoint of all robot poses --
    the grasp geometry between robots -- once, on the very first activation of any cooperative
    primitive sharing this loop, and reuses it on every activation after that (see
    OFFSET_FROM_VTCP_RUNTIME_KEY below); only the midpoint's own position/rotation is
    re-captured fresh each activation. Each step, integrates the driver's raw teleop delta
    into the midpoint, then re-projects every robot's target from it. The driver's task frame
    must be RELATIVE POS (its per-step delta is read from the incoming action); other robots
    must be POS with policy_mode=None. Anti-windup is left to controller.py's
    compliance_reference_limit_enable -- set it via controller_overrides on every robot, this
    primitive does not clamp anything itself.

    Every robot's task-frame origin should stay [0]*6 -- the controller's own "origin" always
    means "task frame relative to this robot's own native base", not relative to another
    robot's base, so it can't directly express the base-to-base offset between two arms.
    robot_base_pose_in_world carries that offset instead (each robot's base pose in one shared
    world frame, e.g. the driver's own base); this primitive composes it explicitly on top of
    each robot's own-native-frame reading/command.

    To restrict which of the driver's axes can move the pivot at all (e.g. free xyz
    translation but only y rotation, for an insertion task), set policy_mode=None on the
    disabled axes in the driver's own TaskFrame -- no support needed here. A locked axis
    falls back to the frame's static target (0.0, never touched by this primitive) instead of
    live teleop input, which _project_policy_action already does for any policy_mode=None
    axis. The vtcp's rotation always starts at identity each activation and only ever
    receives increments about whatever axes are actually live, so it stays within that
    subspace for the whole activation.

    ``pivot_offset_x`` is the pivot's one calibrated degree of freedom: a scalar world-x offset
    from the raw TCP midpoint, for sliding the rotation point toward wherever contact actually
    happens along the rail's length -- the raw grasp midpoint is rarely the true contact point.
    Added to the pivot's world-x on capture. A plain attribute, not driven by keyboard here --
    CalibratePrimitive is the only place that changes it live (a matplotlib slider; keyboard
    nudges turned out unreliable -- unpredictable delivery, and the vtcp reference-limit clamp
    silently capped them without feedback), by writing straight into pivot_offset_x and relying
    on _sync_pivot_offset_x (called every step, source-agnostic) to apply the delta. Every step
    this primitive is active, its current value is published to ``PIVOT_OFFSET_X_RUNTIME_KEY``
    in shared runtime state; every activation's capture (above) checks that key first and
    adopts it over its own configured starting value if present. That is how CalibratePrimitive
    hands a measured value to every later cooperative primitive without them sharing one Python
    instance -- and also how a slider change made there keeps propagating to whichever
    primitive is entered afterward. Pass a known-good value through the primitive config (and
    skip CalibratePrimitive in the graph entirely) once it has been measured for a given cell.

    ``midpoint_offset`` is a plain [x, y, z] constant added to the captured midpoint (not
    persisted or live-adjustable, just a fixed per-primitive nudge -- e.g. a small negative z
    to bias a locked z axis into gentle contact under compliance). Unlike pivot_offset_x, it
    also corrects every robot's offset-from-vtcp capture (see the loop below), so it survives
    as an actual commanded displacement instead of being cancelled by that recomposition and
    only ever affecting the rotation pivot.

    ``OFFSET_FROM_VTCP_RUNTIME_KEY`` persists each robot's captured offset from the vtcp the
    same way -- published once it's captured or reused, checked first on every later capture
    -- but unlike pivot_offset_x it is never live-adjustable and is only ever written once per
    loop: whichever activation captures it first (fresh, from the actual measured poses) wins,
    and every activation after that, in cooperative_reset or cooperative_insert alike, reuses
    exactly that value instead of re-deriving it. This matters because the grasp geometry
    between two rigidly-coupled robots is physically constant -- it should not appear to
    change between activations. Left free to re-derive every activation, a robot's small
    compliant give under contact gets baked in as the new "true" offset the next time around,
    and the relative pose between the two arms creeps a little further every loop iteration.
    Cleared on a genuine full reset, same as pivot_offset_x, since a new grasp needs its
    geometry captured fresh.

    ``freeze_driver_axes_at_entry`` is a third, simpler kind of lock: axis indices that should
    hold *whatever the driver's own pose actually measures* on capture, rather than the
    default for that axis kind (the static configured target for rotation, the robots'
    averaged midpoint for translation). Captured once, on the very first activation of any
    cooperative primitive in a loop, and persisted (``FROZEN_AXES_RUNTIME_KEY``) the same way
    ``OFFSET_FROM_VTCP_RUNTIME_KEY`` persists grasp geometry -- not re-measured every
    activation, so it can't drift the way an unprotected re-measurement would. Use it to demote
    an axis from policy-learnable to fixed without hardcoding an assumed value: whatever pose
    the operator left the arm in when the loop first activates is what gets held from then on.

    ``driver_axis_overrides`` supplies an exact numeric value for an axis instead of reading
    the driver's currently measured pose -- for full run-to-run reproducibility instead of
    whatever small physical variance a fresh grasp lands at. A rotation axis (3/4/5) still
    needs to be listed in ``freeze_driver_axes_at_entry`` too, or the per-step lock below
    overwrites it with the static target anyway; a translation axis (0/1/2) doesn't, since a
    locked (``policy_mode=None``) translation axis already holds whatever it's captured at, no
    separate opt-in needed. Propagates to every robot's offset capture, not just the driver's
    -- see the capture loop below for why.

    Every robot, driver included, is commanded with plain absolute POS targets -- there is no
    velocity/RELATIVE path at the controller for anyone here, even on the driver's own live
    axes. All that ever differs between robots is which axes are *live* (teleop input reaches
    them, via ``self.task_frame[driver].policy_mode`` -- still RELATIVE there, still what the
    action processor reads to pass live input through, and still what makes those axes
    learnable/recorded for policy training); the *value sent to the controller* for every
    axis, on every robot, is always a fresh absolute target reprojected from ``target_world``
    (the same computation a locked follower axis already used). ``apply_task_frames()``
    reports the driver's whole task frame to its controller as fully ABSOLUTE (policy_mode=
    None) on a per-step deep copy, leaving ``self.task_frame[driver]`` itself -- and therefore
    the action processor's live-input passthrough -- untouched.

    This is deliberate, not an approximation: since every robot's target is recomputed every
    step directly from the one shared ``_vtcp_world`` plus that robot's own fixed captured
    offset, all robots' targets stay perfectly rigidly related to each other by construction
    -- there is no per-robot open-loop integration (and so no per-robot windup) to drift them
    apart. The tradeoff is that controller.py's compliance_reference_limit_enable has no
    effect here (it only clamps DeltaMode.RELATIVE axes, and nothing here ever is one) -- each
    robot's own compliance (kp/kd/wrench_limits) still bounds how fast it can chase a target
    that outran its actual pose, so momentarily continuing to move after teleop input stops is
    still possible under sustained contact, but it now happens the *same way*, by the *same
    law*, on every robot -- never more on one arm than another.
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
        pivot_offset_x: float = 0.0,
        freeze_driver_axes_at_entry: tuple[int, ...] = (),
        driver_axis_overrides: dict[int, float] | None = None,
        midpoint_offset: list[float] | None = None,
    ):
        super().__init__(task_frame, robot_dict, cameras, display_cameras)
        self.driver = driver
        self.fps = fps
        self.freeze_driver_axes_at_entry = tuple(freeze_driver_axes_at_entry)
        self.driver_axis_overrides = dict(driver_axis_overrides or {})
        self.midpoint_offset = list(midpoint_offset) if midpoint_offset is not None else [0.0, 0.0, 0.0]
        self.robot_base_pose_in_world = {name: [0.0] * 6 for name in robot_dict}
        self.robot_base_pose_in_world.update(robot_base_pose_in_world or {})
        self._vtcp_world: list[float] | None = None
        self._offset_from_vtcp: dict[str, list[float]] = {}
        # pivot_offset_x is a plain attribute now -- no keyboard events drive it here.
        # CalibratePrimitive is the only place that changes it live (a matplotlib slider);
        # _sync_pivot_offset_x below just applies whatever the current value is, from
        # whatever source set it, to the already-captured vtcp -- see there.
        self.pivot_offset_x = float(pivot_offset_x)
        self._pivot_offset_x_applied = 0.0

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
        self._pivot_offset_x_applied = 0.0

    def _sync_pivot_offset_x(self) -> None:
        """Relocate the rotation pivot's x without moving either robot: recompute each
        robot's offset-from-vtcp so its *world* target is unchanged at the new pivot, then
        move vtcp_world itself. Source-agnostic (set by CalibratePrimitive's slider, or left
        at its constructed/persisted value) -- just diffs against the last-applied value. See
        step()'s record_status line for a live readout.

        A plain ``vtcp_world[0] += delta`` (the previous implementation) instead translated
        the whole rigid assembly by delta every time this changed after the first capture,
        since offset_from_vtcp stayed pinned to the old geometry -- rotation always happens
        about vtcp_world's position (task_pose_to_world_pose rotates the offset about it), so
        leaving the offset untouched while only vtcp_world moved just dragged both robots
        along with the pivot instead of relocating where rotation happens about. That's
        "changing pivot_x visibly moves the rail" -- the bug this was found from.
        """
        delta = self.pivot_offset_x - self._pivot_offset_x_applied
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
                self.set_runtime_value(OFFSET_FROM_VTCP_RUNTIME_KEY, dict(self._offset_from_vtcp))
        self._pivot_offset_x_applied = self.pivot_offset_x

    def _clamp_vtcp_reference_error(self, pose_world: dict[str, list[float]]) -> None:
        """Bound how far the shared vtcp can run ahead of any one robot's own actual measured
        pose -- the same protection controller.py's compliance_reference_limit_enable gives a
        RELATIVE axis (identical budget: e_max = wrench_limits/kp), but computed once against
        the single shared vtcp instead of per-robot at the controller. Clamping each robot's
        own sent target independently would pull them apart from each other again (exactly
        what sending everyone an absolute target off the one shared vtcp was meant to avoid,
        see the class docstring); clamping the vtcp itself keeps every robot's target
        consistent with the others no matter what.

        Reads wrench_limits/kp/compliance_reference_limit_enable straight off each robot's own
        frame.controller_overrides -- nothing new to configure or keep in sync. A robot/axis
        missing any of those, or with compliance_reference_limit_enable off, is left unbounded
        (current behavior). When two robots' budgets disagree, this intersects their allowed
        ranges per axis rather than clamping sequentially (which could overcorrect one robot
        while re-violating the other); an infeasible intersection is left unclamped for that
        axis rather than picking a side.

        Translation (axes 0-2) is exact for any vtcp rotation or robot base geometry --
        task_pose_to_world_pose rotates the robot's fixed *offset*, not the vtcp's own
        translation, so a given robot's world-frame target moves exactly 1:1 with vtcp[:3]
        regardless of anything else. Rotation (axes 3-5) treats vtcp[3:6] as the reference RPY
        directly, the same simplification the rest of this class already relies on -- exact
        while the vtcp's own rotation doesn't move (true for every current cooperative
        primitive, which locks or freezes every rotation axis) and approximate otherwise.
        """
        for i in range(3):
            lower, upper = -math.inf, math.inf
            for name in self.robot_dict:
                budget = _reference_error_budget(self.task_frame[name].controller_overrides, i)
                if budget is None:
                    continue
                # offset_component is how much of this robot's world-frame target comes from
                # its fixed offset (rotated by the vtcp's current orientation) rather than
                # from vtcp[i] itself -- recomputed fresh since it depends on vtcp's current
                # rotation, but treated as constant for this step's clamp (the actual
                # geometry can't change mid-step).
                target_i = task_pose_to_world_pose(self._offset_from_vtcp[name], self._vtcp_world)[i]
                offset_component = target_i - self._vtcp_world[i]
                lower = max(lower, pose_world[name][i] - budget - offset_component)
                upper = min(upper, pose_world[name][i] + budget - offset_component)
            if lower <= upper:
                self._vtcp_world[i] = min(max(self._vtcp_world[i], lower), upper)

        for axis in range(3, 6):
            lower, upper = -math.inf, math.inf
            for name in self.robot_dict:
                budget = _reference_error_budget(self.task_frame[name].controller_overrides, axis)
                if budget is None:
                    continue
                err = wrap_to_pi(self._vtcp_world[axis] - pose_world[name][axis])
                lower = max(lower, self._vtcp_world[axis] - err - budget)
                upper = min(upper, self._vtcp_world[axis] - err + budget)
            if lower <= upper:
                self._vtcp_world[axis] = min(max(self._vtcp_world[axis], lower), upper)

    def step(self, action: dict[str, dict[str, float]]):
        pose_world = _poses_world(self.robot_dict, self.task_frame, self.robot_base_pose_in_world)

        if self._vtcp_world is None:
            names = list(self.robot_dict)
            # A calibration primitive (or a live nudge made in a previously-active cooperative
            # primitive) may have published a pivot_offset_x other than this instance's own
            # configured starting value -- adopt it if present, and fall back to that starting
            # value otherwise (also what happens with no calibration step in the graph at all,
            # or in isolation in tests).
            calibrated_offset_x = self.get_runtime_value(PIVOT_OFFSET_X_RUNTIME_KEY, None)
            if calibrated_offset_x is not None:
                self.pivot_offset_x = float(calibrated_offset_x)
            midpoint = [sum(pose_world[n][i] for n in names) / len(names) for i in range(3)]
            for i in range(3):
                midpoint[i] += self.midpoint_offset[i]
            self._pivot_offset_x_applied = self.pivot_offset_x

            # Natural per-axis default: measured midpoint for translation, static target for
            # rotation. freeze_driver_axes_at_entry swaps in the driver's own measured pose --
            # captured once across the whole loop and persisted (FROZEN_AXES_RUNTIME_KEY), same
            # as OFFSET_FROM_VTCP_RUNTIME_KEY does for grasp geometry, so it isn't re-measured
            # (and doesn't drift) every activation. driver_axis_overrides swaps in an exact
            # value instead, for either axis kind, and wins over both.
            vtcp_pose = midpoint + list(self.task_frame[self.driver].target[3:6])
            frozen_axes = self.get_runtime_value(FROZEN_AXES_RUNTIME_KEY, {})
            for axis in self.freeze_driver_axes_at_entry:
                vtcp_pose[axis] = frozen_axes.get(axis, pose_world[self.driver][axis])
            for axis, value in self.driver_axis_overrides.items():
                vtcp_pose[axis] = value
            # Publish the frozen baseline *before* pivot_offset_x is added below -- it must
            # never include the pivot shift, or the next primitive's fresh capture would adopt
            # a value that already has pivot_offset_x baked in, add its own on top, and
            # compound a little further every activation (cooperative_reset: +offset,
            # cooperative_insert: +2*offset, pushdown: +3*offset, ...).
            if self._shared_runtime_values is not None and self.freeze_driver_axes_at_entry:
                self.set_runtime_value(
                    FROZEN_AXES_RUNTIME_KEY,
                    {**frozen_axes, **{axis: vtcp_pose[axis] for axis in self.freeze_driver_axes_at_entry}},
                )
            # pivot_offset_x applies last, on top of whatever determined x's baseline above
            # (averaged midpoint, freeze_driver_axes_at_entry, or an override) -- x is both the
            # locked translation axis in cooperative_reset/cooperative_insert/pushdown *and*
            # the axis pivot_offset_x relocates, and freeze_driver_axes_at_entry runs after the
            # midpoint is built, so adding pivot_offset_x to midpoint[0] earlier (the previous
            # implementation) got silently overwritten by the freeze/override branch the moment
            # x was also a frozen or overridden axis -- exactly the "pivot_offset_x doesn't
            # carry over" bug this was found from. capture_pose below stays uncorrected for it
            # (unlike driver_axis_overrides/midpoint_offset), same as it always has -- that's
            # what makes it a pure pivot relocation instead of a visible jump (see the class
            # docstring's pivot_offset_x paragraph).
            vtcp_pose[0] += self.pivot_offset_x
            self._vtcp_world = vtcp_pose
            # The grasp geometry between robots (each one's fixed offset from the vtcp) is
            # captured once, on the very first activation of any cooperative primitive in a
            # loop, and reused on every activation after that -- never re-derived from the
            # current pose again. Re-deriving it fresh each time is what caused the bug this
            # guards against: under compliant contact a robot gives slightly, the next
            # activation's capture then treats that slight deflection as the new "true" grasp
            # geometry, and the relative pose between the two arms creeps a little further
            # every loop iteration. Persisted through shared runtime state (like
            # pivot_offset_x) since cooperative_reset/cooperative_insert are separate
            # instances; cleared, and so correctly re-derived from scratch, on a genuine full
            # reset (new grasp -- the old geometry no longer applies).
            calibrated_offset_from_vtcp = self.get_runtime_value(OFFSET_FROM_VTCP_RUNTIME_KEY, None)
            for name in names:
                if calibrated_offset_from_vtcp is not None and name in calibrated_offset_from_vtcp:
                    self._offset_from_vtcp[name] = list(calibrated_offset_from_vtcp[name])
                    continue
                # An overridden axis's vtcp value deliberately differs from what a robot's own
                # pose actually measures -- capturing that robot's offset from its *raw*
                # measured pose would encode exactly that mismatch, and target_world's later
                # offset+vtcp recomposition would silently reconstruct the original measured
                # value, undoing the override entirely for that robot's own commanded action.
                # This applies to every robot, not just the driver: pose_world is one shared
                # world frame (see _poses_world), so an overridden axis means the same value
                # for everyone in it. Substituting the override into the pose used for capture
                # makes every affected robot's offset zero on that axis instead, exactly like
                # freeze_driver_axes_at_entry's un-overridden case already relies on.
                # midpoint_offset needs the same correction, for the same reason -- otherwise
                # it only ever affects the rotation pivot (like pivot_offset_x, deliberately),
                # never producing an actual commanded displacement: the offset-vtcp
                # recomposition would reconstruct the unshifted measured pose right back.
                capture_pose = list(pose_world[name])
                for axis, value in self.driver_axis_overrides.items():
                    capture_pose[axis] = value
                for axis in range(3):
                    capture_pose[axis] += self.midpoint_offset[axis]
                self._offset_from_vtcp[name] = world_pose_to_task_pose(capture_pose, self._vtcp_world)
            if self._shared_runtime_values is not None:
                self.set_runtime_value(OFFSET_FROM_VTCP_RUNTIME_KEY, dict(self._offset_from_vtcp))

        self._sync_pivot_offset_x()

        dt = 1.0 / self.fps
        driver_delta = action.get(self.driver, {})
        d = [float(driver_delta.get(f"{ax}.ee_pos", 0.0)) if self.task_frame[self.driver].policy_mode[i] is not None else 0.0 for i, ax in enumerate(("x", "y", "z", "rx", "ry", "rz"))]

        self._vtcp_world[0] += d[0] * dt
        self._vtcp_world[1] += d[1] * dt
        self._vtcp_world[2] += d[2] * dt
        new_rot = rotation_from_extrinsic_xyz(*[v * dt for v in d[3:]]) * rotation_from_extrinsic_xyz(*self._vtcp_world[3:])
        self._vtcp_world[3:] = euler_xyz_from_rotation(new_rot)

        self._clamp_vtcp_reference_error(pose_world)

        # Publish the live scalar every step (not just on a nudge) so any cooperative
        # primitive entered next picks up the most recent value the instant it captures.
        # Guarded: unit tests construct this primitive directly, without going through
        # ManipulationPrimitiveNet's attach_shared_runtime_values().
        if self._shared_runtime_values is not None:
            self.set_runtime_value(PIVOT_OFFSET_X_RUNTIME_KEY, self.pivot_offset_x)

        cooperative_action: dict[str, dict[str, float]] = {}
        fixed_rotation_axes = {
            axis
            for axis in range(3, 6)
            if self.task_frame[self.driver].policy_mode[axis] is None
        }
        follower_fixed_rotation_axes = fixed_rotation_axes or set(range(3, 6))

        for name, frame in self.task_frame.items():
            # Every robot, driver included, gets a plain absolute target reprojected from the
            # one shared vtcp -- see the class docstring for why (rigidly relates every
            # robot's target to every other's by construction, no per-robot open-loop
            # integration to drift apart).
            target_world = task_pose_to_world_pose(self._offset_from_vtcp[name], self._vtcp_world)
            native_target = world_pose_to_task_pose(target_world, self.robot_base_pose_in_world[name])
            values = world_pose_to_task_pose(native_target, frame.origin)

            rotation_axes_to_lock = fixed_rotation_axes if name == self.driver else follower_fixed_rotation_axes
            for axis in rotation_axes_to_lock:
                if axis in self.freeze_driver_axes_at_entry:
                    # Already correct: target_world above already reflects the value frozen
                    # at capture (baked into _offset_from_vtcp/_vtcp_world). Overwriting with
                    # the static frame.target here would silently discard the freeze and snap
                    # back to the un-calibrated default (e.g. ry=0.0), which is exactly the
                    # bug this guard fixes.
                    continue
                if axis < len(frame.target):
                    values[axis] = frame.target[axis]
            cooperative_action[name] = dict(zip(("x.ee_pos", "y.ee_pos", "z.ee_pos", "rx.ee_pos", "ry.ee_pos", "rz.ee_pos"), values))
            if "gripper.pos" in action.get(name, {}):
                cooperative_action[name]["gripper.pos"] = action[name]["gripper.pos"]

        obs, reward, terminated, truncated, info = super().step(cooperative_action)
        # Live readout every step (not just print-on-change) so a `,`/`.`/`[`/`]` nudge is
        # immediately visible in record.py's own status line instead of a plain print() that
        # would otherwise interleave badly with its in-place \r updates. Subclasses (e.g.
        # CalibratePrimitive) append to this rather than overwriting it.
        info["record_status"] = f"pivot_x = {self.pivot_offset_x:+.4f} m"
        return obs, reward, terminated, truncated, info

    def get_display_points(self) -> dict[str, list[float]]:
        """Live 3D points for visualization: each robot's current end-effector
        position (composed into the shared world frame via
        robot_base_pose_in_world, same as everything else in this class) plus
        "vtcp" once the shared pivot has been captured (see step() -- absent only
        before this activation's very first step). See
        ManipulationPrimitive.get_display_points.
        """
        pose_world = _poses_world(self.robot_dict, self.task_frame, self.robot_base_pose_in_world)
        points = {name: list(pose[:3]) for name, pose in pose_world.items()}
        if self._vtcp_world is not None:
            points["vtcp"] = list(self._vtcp_world[:3])
        return points


# UR task frames use metres and radians. The supplied rail-plane coordinates are millimetres.
DEFAULT_RAIL_Y_M = {"right": -0.30186, "left": -0.29813}
DEFAULT_GRASP_ORIENTATION_RPY = {
    "right": [float(np.pi), 0.0, 0.0],
    "left": [float(np.pi), 0.0, -float(np.pi)],
}

XZ_RELATIVE_POLICY = [
    PolicyMode.RELATIVE,
    None,
    PolicyMode.RELATIVE,
    None,
    None,
    None,
]
COOPERATIVE_LEFT_POLICY = [
    PolicyMode.RELATIVE,
    PolicyMode.RELATIVE,
    PolicyMode.RELATIVE,
    None,
    PolicyMode.RELATIVE,
    None,
]
# cooperative_reset/cooperative_insert's shared policy_mode. x is locked -- motion along the
# rail's own length doesn't help alignment and is what let the workpiece drift out of camera
# frame, spuriously triggering the reward classifier. y (across-rail) and z (push depth) stay
# live; all rotations are locked (ry frozen/overridable via freeze_driver_axes_at_entry, see
# CooperativeFramePrimitive; rx/rz always at their static target).
COOPERATIVE_INSERT_LEFT_POLICY = [
    None,
    PolicyMode.RELATIVE,
    PolicyMode.RELATIVE,
    None,
    None,
    None,
]
# pushdown's policy_mode: x/y/z all locked (x/y unchanged from cooperative_insert, z gently
# pushed via midpoint_offset), rx/rz at their static target (rz softened via controller
# overrides, not policy_mode), ry live -- teleop it down toward 0deg by hand.
PUSHDOWN_LEFT_POLICY = [
    None,
    None,
    None,
    None,
    PolicyMode.RELATIVE,
    None,
]

# Key the operator taps in cooperative_reset to jump back to calibrate (which
# unlocks ry for live teleop again) without a full reset back through align_to_rail/
# teleop_left/teleop_right -- for re-calibrating ry mid-session. calibrate's own
# OnSuccess edge already routes back through zero_ft_before_cooperative -> cooperative_reset
# afterward, so this is a self-contained detour, not a new loop shape.
RECALIBRATE_RY_REQUEST_FLAG = "recalibrate_ry_request"
RECALIBRATE_RY_KEY = keyboard.Key.up


def _rail_target(name: str) -> list[float]:
    """The nominal rail-plane grasp pose. rz is never touched -- the TCP never moves in rz at
    all, it's always the fixed DEFAULT_GRASP_ORIENTATION_RPY value. Only ry ever gets set, and
    that happens purely by teleoperating it to the right value in calibrate; it's
    then frozen there (see CooperativeFramePrimitive.freeze_driver_axes_at_entry) once
    cooperative_reset/cooperative_insert take over. That's the entire calibration."""
    rx, ry, rz = DEFAULT_GRASP_ORIENTATION_RPY[name]
    return [0.0, DEFAULT_RAIL_Y_M[name], 0.0, rx, ry, rz]


def _task_frame(
    name: str,
    policy_mode: list[PolicyMode | None],
    controller_overrides: dict[str, Any] | None = None,
) -> TaskFrame:
    return TaskFrame(
        target=_rail_target(name),
        control_mode=[ControlMode.POS] * 6,
        policy_mode=list(policy_mode),
        origin=[0.0] * 6,
        controller_overrides=controller_overrides,
    )


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


def _controller_overrides() -> dict[str, Any]:
    return {
        "use_force_mode": True,
        "compliance_reference_limit_enable": [True] * 6,
        "kp": [3000.0, 3000.0, 3000.0, 150.0, 150.0, 150.0],
        "kd": [60.0, 60.0, 60.0, 6.0, 6.0, 6.0],
        "wrench_limits": [30.0, 30.0, 30.0, 4.0, 4.0, 4.0],
    }


def _single_arm_frames(active: str, passive: str) -> dict[str, TaskFrame]:
    return {
        active: _task_frame(active, XZ_RELATIVE_POLICY),
        passive: _task_frame(passive, [None] * 6),
    }


class CalibratePrimitive(CooperativeFramePrimitive):
    """The graph's manual setup step -- more than just ry by now, so named plainly. Teleop ry
    to wherever it needs to be (reported live through ``info["record_status"]``); rz is never
    touched, the TCP never moves in rz. pivot_offset_x/y are set by dragging sliders in a
    matplotlib window instead of keyboard nudges -- those turned out unreliable in practice
    (unpredictable delivery, and CooperativeFramePrimitive's own reference-limit clamp used to
    cap them silently with no feedback that anything had stopped moving). Space (the ordinary
    OnSuccess advance key) just ends this primitive; ry then gets frozen at whatever it
    measures on entry into cooperative_reset (see
    ``CooperativeFramePrimitive.freeze_driver_axes_at_entry``) -- there is nothing else to
    lock in here.

    The slider window is created lazily on the first ``step()`` (not ``__init__``, which runs
    at MP-Net construction time before any robot is connected) and persists for the rest of
    the process -- re-entering this primitive (e.g. via cooperative_reset's up-arrow escape
    hatch) reuses the same window rather than recreating it. Runs in-process (``plt.pause()``
    every step) rather than the separate-process pattern
    ``share/robots/ur/lerobot_robot_ur/wrench_monitor.py`` uses for its own plot -- appropriate
    here since this is unhurried manual teleop, not a real-time control loop; switch to that
    pattern if the per-step GUI pause turns out to be noticeable.
    """

    def __init__(self, *args, pivot_offset_range_m: float = 0.3, calibration_path: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pivot_offset_range_m = float(pivot_offset_range_m)
        self._calibration_path = Path(calibration_path) if calibration_path is not None else None
        self._slider_fig = None
        self._slider_x = None

    def _ensure_pivot_offset_sliders(self) -> None:
        if self._slider_fig is not None:
            return
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        initial_x = self.get_runtime_value(PIVOT_OFFSET_X_RUNTIME_KEY, self.pivot_offset_x)
        r = self._pivot_offset_range_m
        fig, ax_x = plt.subplots(figsize=(5, 1.3))
        fig.canvas.manager.set_window_title("pivot offset")
        self._slider_x = Slider(ax_x, "pivot_offset_x (m)", -r, r, valinit=initial_x)
        fig.tight_layout()
        plt.show(block=False)
        self._slider_fig = fig
        self.pivot_offset_x = initial_x

    def step(self, action: dict[str, dict[str, float]]):
        import matplotlib.pyplot as plt

        self._ensure_pivot_offset_sliders()
        self.pivot_offset_x = self._slider_x.val
        plt.pause(0.001)

        obs, reward, terminated, truncated, info = super().step(action)
        try:
            driver_pose = get_robot_pose_from_observation(obs, self.driver)
            ry = driver_pose[4]
            info["record_status"] = f"ry = {ry:+.4f} rad  |  {info['record_status']}"
            self._save_calibration(x_offset=driver_pose[0], ry_angle=ry)
        except KeyError:
            pass
        return obs, reward, terminated, truncated, info

    def _save_calibration(self, x_offset: float, ry_angle: float) -> None:
        """Written every step this primitive is active, not on some detected "finish" -- the
        raw primitive has no visibility into the SUCCESS keypress that ends it (handled by the
        outer action processor, after this env's own step() already returned). The practical
        effect is the same either way: the last write before you move on is what sticks."""
        if self._calibration_path is None:
            return
        self._calibration_path.write_text(
            json.dumps({"pivot_offset_x": self.pivot_offset_x, "x_offset": x_offset, "ry_angle": ry_angle})
        )

    def close(self) -> None:
        if self._slider_fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._slider_fig)
        super().close()


class CooperativeInsertPrimitive(CooperativeFramePrimitive):
    """cooperative_insert's own subclass: adds two extra observation channels on top of the
    driver's y/z velocity -- delta position from wherever the driver was when this activation
    started, and the previous step's action -- so the state observation carries enough to be
    (close to) Markovian without needing full pose history. x is excluded throughout: it's
    locked (see COOPERATIVE_INSERT_LEFT_POLICY), so its delta/velocity are always ~0.

    Injected as extra keys into the raw observation (``{driver}.dy/dz.ee_pos``,
    ``{driver}.prev_y/prev_z.ee_vel``), reusing the existing ee_pos/ee_velocity observation
    machinery with custom axis names (see cooperative_insert's ObservationConfig) instead of a
    new processor step.

    Injected from ``_get_observation()``, not ``step()`` alone: ``reset()`` calls
    ``_get_observation()`` too, and record.py stores the pre-step observation alongside each
    action, so the first frame of an episode comes from reset()'s observation.
    """

    def reset_runtime_state(self) -> None:
        super().reset_runtime_state()
        self._entry_pos: list[float] | None = None
        self._prev_action = [0.0, 0.0]

    def _get_observation(self):
        obs = super()._get_observation()
        driver_pos = [obs[f"{self.driver}.{ax}.ee_pos"] for ax in ("y", "z")]
        if self._entry_pos is None:
            self._entry_pos = driver_pos
        delta_pos = [driver_pos[i] - self._entry_pos[i] for i in range(2)]
        for ax, value in zip(("dy", "dz"), delta_pos):
            obs[f"{self.driver}.{ax}.ee_pos"] = value
        for ax, value in zip(("prev_y", "prev_z"), self._prev_action):
            obs[f"{self.driver}.{ax}.ee_vel"] = value
        return obs

    def step(self, action: dict[str, dict[str, float]]):
        # Update _prev_action before stepping -- ManipulationPrimitive.step() calls
        # _get_observation() internally, right after send_action.
        driver_action = action.get(self.driver, {})
        self._prev_action = [float(driver_action.get(f"{ax}.ee_pos", 0.0)) for ax in ("y", "z")]
        return super().step(action)


@EnvConfig.register_subclass("rail_bimanual_grasp")
@dataclass
class RailBimanualGraspEnvConfig(ManipulationPrimitiveNetConfig):
    """align_to_rail -> teleop_left -> teleop_right -> zero_ft -> calibrate ->
    zero_ft_before_cooperative -> cooperative_reset ⇄ cooperative_insert, looping for repeated
    episodes -- same shape as bimanual_pick.py's teleop_left/.../reset ⇄ policy: the one-time
    approach/grasp/calibration chain is only ever walked on the very first full reset
    (start_primitive/reset_primitive stay at "align_to_rail"; none of these primitives
    is_terminal, so cooperative_insert <-> cooperative_reset is a plain graph loop that never
    triggers a second full reset). Teleop into the initial state by hand in "cooperative_reset"
    (Y/Z translation only -- X and Ry stay frozen at whatever they measure on entry, see
    CooperativeFramePrimitive.freeze_driver_axes_at_entry), space starts "cooperative_insert"
    (same Y/Z-only DOF, learned this time); on success there (space, or reward_classifier_path
    crossing threshold if supplied), teleop back up in "cooperative_reset" again for the next
    episode."""

    fps: int = 10
    left_robot_ip: str = "172.22.22.5"
    right_robot_ip: str = "172.22.22.2"
    right_base_pose_in_left_base: list[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    alignment_linear_speed_mps: float = 0.05
    alignment_angular_speed_rad_s: float = 0.10
    translation_action_scale: float = 0.1
    rotation_action_scale: float = 0.5
    # Expected side length of the y/z square cooperative_insert's driver moves within, relative
    # to wherever it was when the episode started -- half of this is the dataset_stats bound
    # for CooperativeInsertPrimitive's delta-position observation channel (see there). Not
    # enforced anywhere; just the assumed normalization range.
    cooperative_insert_position_range_m: float = 0.05
    open_gripper_position: float = 0.0
    closed_gripper_position: float = 1.0
    mock: bool = False
    start_primitive: str = "align_to_rail"
    reset_primitive: str = "align_to_rail"
    # World-x offset of the cooperative rotation pivot from the raw left/right midpoint --
    # replaces the old calibrate_origin multi-sample fit, which was hard to debug/tune for a
    # pivot that only ever needs to slide along one axis (the rail is aligned with world x).
    # None starts at 0.0; pass a known-good value to skip dragging it into place, still
    # adjustable afterward via the pivot offset slider window -- CalibratePrimitive.
    pivot_offset_x: float | None = None
    # +/- range of the pivot_offset_x slider in CalibratePrimitive's matplotlib window.
    pivot_offset_range_m: float = 0.4
    # Resume support: jump straight into "zero_ft_before_cooperative" (both grippers assumed
    # already holding the rail from a previous run) instead of align_to_rail -> teleop_left ->
    # teleop_right -> zero_ft -> calibrate. Also used as reset_primitive, since
    # resets are done by hand -- teleoping the pivot back out with the spacemouse -- rather
    # than through align_to_rail's scripted trajectory. ry still comes out right without
    # re-running calibrate: freeze_driver_axes_at_entry reads the arm's actual
    # physical pose, not any persisted config/runtime value, so as long as the arm hasn't
    # moved since it was last calibrated, it re-captures the same ry.
    skip_grasp: bool = False
    # Reproducible alternative to calibrate's live teleop-and-measure: pose ry to a
    # known-good value directly instead of freezing whatever the arm's actual physical pose
    # happens to measure on entry. calibrate prints the live measured ry in its
    # status line -- once you know a good value, set it here and skip re-measuring it (subject
    # to whatever small run-to-run variance a fresh grasp/re-home lands at) on every run.
    # Requires skip_grasp=True (enforced): calibrate is unreachable there anyway, and
    # this value is what fills in for it. x/y/z and the rest of the grasp geometry still come
    # from wherever the arms physically are, exactly as skip_grasp already works -- only ry
    # becomes an exact, reproducible number instead of a measurement.
    ry_angle: float | None = None
    # Same idea as ry_angle, but for x (locked in COOPERATIVE_INSERT_LEFT_POLICY): pin it to a
    # known-good value instead of wherever teleop last left it. No skip_grasp requirement --
    # unlike ry there's no dedicated calibration primitive this would make redundant. Usually
    # unnecessary now that freeze_driver_axes_at_entry persists the captured value across the
    # whole loop on its own (see FROZEN_AXES_RUNTIME_KEY) -- set this only when you want a
    # specific known value from the start, not just "whatever it measures on first activation".
    x_offset: float | None = None
    # Load pivot_offset_x/x_offset/ry_angle from calibration_path instead of running
    # calibrate at all -- calibrate becomes unreachable (zero_ft routes straight to
    # zero_ft_before_cooperative, and skip_grasp's entry point becomes
    # zero_ft_before_cooperative instead of calibrate) whenever this is set, independent
    # of skip_grasp: you might still want the real grasp/align chain but reuse a known-good
    # calibration rather than re-teleoping ry / re-dragging the pivot slider. The file must
    # already exist -- an explicit request to load one that isn't there is a config error, not
    # something to silently fall back past.
    load_calibration: bool = False
    # Where CalibratePrimitive saves pivot_offset_x/x_offset/ry_angle (every step it's active,
    # not just on some detected "finish" -- the raw primitive has no visibility into the
    # SUCCESS keypress that ends it, since that's handled by the outer action processor; the
    # practical effect is the same, since the last write before you move on is what sticks)
    # and where load_calibration reads them back from. Defaults to one fixed path in the
    # system temp dir -- deliberately not per-process-unique, so the next run's
    # load_calibration=True picks up the last calibrate session's values without you having
    # to thread a path through by hand. Point it elsewhere to keep more than one calibration
    # around (e.g. per rail/cell).
    calibration_path: str = str(Path(tempfile.gettempdir()) / "rail_bimanual_grasp_calibration.json")
    # Force-mode singularities can leave the arms stuck under cooperative_reset's normal
    # compliant teleop with no way to drive out of them -- force mode itself is what's stuck.
    # Set this (with skip_grasp=True, to enter straight into cooperative_reset) to make
    # cooperative_reset rigid/servo-position-controlled instead: use_force_mode=False,
    # simple_pose_use_servo=True, same as align_to_rail/zero_ft, so the SpaceMouse can drive
    # the arms clear by streaming servo position commands instead of a compliant wrench.
    # Reference-limiting is off too (it's a wrench_limits/kp budget, meaningless outside force
    # mode); servo's own rate limit is simple_pose_max_speed (alignment_linear_speed_mps/
    # alignment_angular_speed_rad_s), already configured on both robots. Turn back off once
    # clear -- this is a recovery switch, not a normal operating mode.
    cooperative_reset_servo_recovery: bool = False
    # zero_ft re-zeros both F/T sensors once both grippers are loaded with the workpiece --
    # align_to_rail/teleop_left/teleop_right all run in either non-force-mode (align_to_rail)
    # or with an as-yet ungrasped/half-grasped load, so whatever zero_ft() set at connect()
    # (before any of that) is stale by the time both arms are actually gripping. Re-zeroing
    # here, right before the compliant hold phase (calibrate/cooperative_reset/
    # cooperative_insert) begins, keeps that phase's admittance behavior from chasing a
    # phantom residual wrench left over from the ungrasped baseline.
    zero_ft_settle_duration_s: float = 0.3
    # Optional automatic success signal for cooperative_insert -> cooperative_reset, alongside
    # the always-available manual OnSuccess (space). When set, a lerobot reward classifier
    # checkpoint (trained with share/scripts/train_reward_classifier.py) is evaluated every
    # step; crossing reward_classifier_threshold fires the transition on its own, in addition
    # to -- not instead of -- the manual trigger. See RewardClassifierTransition.
    reward_classifier_path: str | None = None
    reward_classifier_threshold: float = 0.7
    reward_classifier_device: str = "cuda"
    # cooperative_reset/cooperative_insert are translation-only against a rigid rail -- cap
    # contact force there well below the 30 N shared by align_to_rail/teleop_left/teleop_right
    # (see _controller_overrides()). compliance_reference_limit_enable is already on for every
    # axis, and its budget is wrench_limits/kp, so this tightens automatically for free -- no
    # separate setting needed. Rotation axes are untouched (locked/absolute there, not the
    # live contact interface). calibrate is untouched too -- only the two primitives
    # that actually drive the insertion contact get the softer limit.
    cooperative_translation_wrench_limit_n: float = 10.0
    # Auto-truncate cooperative_insert after this many seconds if neither SUCCESS (space/
    # SpaceMouse button) nor reward_classifier_path has fired yet -- keeps a stuck/no-progress
    # episode from running forever. Additional, not instead of: whichever transition fires
    # first still wins (see the comment above the transitions list).
    cooperative_insert_episode_duration_s: float = 5.0
    # World-frame +z distance both arms lift straight up by, together, in pull_out --
    # entered only on an actual cooperative_insert success (space, or reward_classifier_path
    # crossing threshold), never on the OnTimeLimit fallback (nothing was achieved there, so
    # nothing needs pulling out of). x/y/rotation hold wherever they are; only z moves.
    pull_out_height_m: float = 0.02
    # When set, a successful cooperative_insert routes to pushdown instead of pull_out --
    # manual exploration of seating the rail fully (soft yaw + gentle push, teleop ry down to
    # 0deg) rather than the normal train/record loop.
    demo: bool = False
    # Constant [x, y, z] added to pushdown's captured midpoint (see
    # CooperativeFramePrimitive.midpoint_offset) -- z only, negative pushes the driver gently
    # into the rail under compliance (x/y are locked and otherwise unmoved from wherever
    # cooperative_insert left them). Sign/magnitude to be tuned on hardware.
    pushdown_z_offset_m: float = -0.005
    # rz (yaw) stiffness/force cap in pushdown, well below the shared 150/4.0 -- lets the rail
    # settle into whatever yaw the receiving slot actually wants instead of fighting it.
    pushdown_yaw_kp: float = 15.0
    pushdown_yaw_wrench_limit_nm: float = 1.0

    def __post_init__(self) -> None:
        if self.cooperative_reset_servo_recovery and not self.skip_grasp:
            raise ValueError(
                "cooperative_reset_servo_recovery requires skip_grasp=True -- it only makes "
                "sense entering straight into the stuck loop, not walking the ordinary grasp "
                "chain into it."
            )
        if self.ry_angle is not None and not (self.skip_grasp or self.load_calibration):
            raise ValueError(
                "ry_angle requires skip_grasp=True or load_calibration=True -- calibrate is "
                "unreachable either way, and ry_angle is what fills in for it."
            )
        if self.load_calibration:
            calibration_path = Path(self.calibration_path)
            if not calibration_path.exists():
                raise ValueError(
                    f"load_calibration=True but calibration_path {calibration_path} doesn't "
                    "exist -- run calibrate at least once (load_calibration=False) first, or "
                    "point calibration_path at a file that already has one."
                )
            calibration = json.loads(calibration_path.read_text())
            self.pivot_offset_x = calibration["pivot_offset_x"]
            self.x_offset = calibration["x_offset"]
            self.ry_angle = calibration["ry_angle"]
        controller_overrides = _controller_overrides()
        alignment_controller_overrides = dict(controller_overrides)
        alignment_controller_overrides.update(
            use_force_mode=False,
            simple_pose_use_servo=True,
        )

        self.cameras = {
            "left": OpenCVCameraConfig(index_or_path="/dev/video0"),
        }

        if self.mock:
            self.robot = {
                "left": SimURConfig(
                    use_gripper=True,
                    initial_pose=[0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                ),
                "right": SimURConfig(
                    use_gripper=True,
                    initial_pose=[0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                ),
            }
        else:
            self.robot = {
                "left": URConfig(
                    robot_ip=self.left_robot_ip,
                    frequency=125,
                    soft_real_time=True,
                    simple_pose_max_speed=[self.alignment_linear_speed_mps] * 3 + [self.alignment_angular_speed_rad_s] * 3,
                    rt_core=3,
                    use_gripper=True,
                ),
                "right": URConfig(
                    robot_ip=self.right_robot_ip,
                    frequency=125,
                    soft_real_time=True,
                    simple_pose_max_speed=[self.alignment_linear_speed_mps] * 3 + [self.alignment_angular_speed_rad_s] * 3,
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
            # The SpaceMouse never drives the gripper directly -- the next primitive's own
            # static gripper target closes it on transition instead (see left_grasp/
            # right_grasp below). Button 0 is SUCCESS; button 1 is IS_INTERVENTION, held
            # while actively steering so record.py routes to live teleop instead of the
            # policy and records the frame as an intervention (toggle=False: level-triggered
            # on the physical button state, same as the hoermann connector env's own
            # IS_INTERVENTION mapping) -- exit recording with Ctrl-C instead of a button.
            gripper_close_button_idx=None,
            gripper_open_button_idx=None,
            button_mapping={
                0: {"event": TeleopEvents.SUCCESS, "toggle": False},
                1: {"event": TeleopEvents.IS_INTERVENTION, "toggle": False},
            },
        )

        alignment_processor = _processor(
            self.fps,
            _gripper(
                live=None,
                static={
                    "left": self.open_gripper_position,
                    "right": self.open_gripper_position,
                },
            ),
        )
        # Neither grasp stage drives a gripper live (live=None, both grippers fully static)
        # -- the SpaceMouse never touches the gripper at all. left_grasp holds both open
        # while the operator positions the left arm; right_grasp's static target for "left"
        # is already closed_gripper_position, so left snaps shut on its own the instant
        # right_grasp activates -- the primitive transition closes it, nothing else has to.
        left_grasp_processor = _processor(
            self.fps,
            _gripper(
                live=None,
                static={"left": self.open_gripper_position, "right": self.open_gripper_position},
            ),
        )
        right_grasp_processor = _processor(
            self.fps,
            _gripper(
                live=None,
                static={"left": self.closed_gripper_position, "right": self.open_gripper_position},
            ),
        )
        cooperative_processor = _processor(
            self.fps,
            _gripper(
                live=None,
                static={
                    "left": self.closed_gripper_position,
                    "right": self.closed_gripper_position,
                },
            ),
            # Both arms are rigidly coupled through the workpiece -- their xyz velocity is
            # mostly the same signal up to inertia, and rotation is locked entirely in every
            # cooperative primitive. Driver-only, xyz-only keeps observation.state a clean,
            # near-Markovian signal instead of two robots' worth of largely-redundant 6-axis
            # velocity in two different native frames (see the runbook's "Worth knowing" note
            # on why that raw signal isn't expressed in the vtcp frame either).
            observation=ObservationConfig(
                add_ee_pos_to_observation=False,
                add_ee_velocity_to_observation={"left": True, "right": False},
                ee_velocity_axes=["x.ee_vel", "y.ee_vel", "z.ee_vel"],
                add_ee_wrench_to_observation=False,
                add_joint_position_to_observation=False,
            ),
        )
        # Same as cooperative_processor, plus the recalibrate-ry escape hatch (feeds
        # RECALIBRATE_RY_REQUEST_FLAG only cooperative_reset's own OnEvent edge reads).
        cooperative_reset_processor = _processor(
            self.fps,
            _gripper(
                live=None,
                static={
                    "left": self.closed_gripper_position,
                    "right": self.closed_gripper_position,
                },
            ),
            observation=ObservationConfig(
                add_ee_pos_to_observation=False,
                add_ee_velocity_to_observation={"left": True, "right": False},
                ee_velocity_axes=["x.ee_vel", "y.ee_vel", "z.ee_vel"],
                add_ee_wrench_to_observation=False,
                add_joint_position_to_observation=False,
            ),
            extra_key_events={RECALIBRATE_RY_REQUEST_FLAG: RECALIBRATE_RY_KEY},
        )
        # cooperative_insert only: adds delta-position-from-entry (position slot, custom
        # dx/dy/dz axis names) and the previous action (velocity slot, alongside the current
        # velocity) -- see CooperativeInsertPrimitive. Nobody else needs either channel.
        cooperative_insert_processor = _processor(
            self.fps,
            _gripper(
                live=None,
                static={
                    "left": self.closed_gripper_position,
                    "right": self.closed_gripper_position,
                },
            ),
            # x is dropped here too (not just from the action) -- it's locked, so its delta
            # and velocity are always ~0, dead weight in the state vector.
            observation=ObservationConfig(
                add_ee_pos_to_observation={"left": True, "right": False},
                ee_pos_axes=["dy.ee_pos", "dz.ee_pos"],
                add_ee_velocity_to_observation={"left": True, "right": False},
                ee_velocity_axes=["y.ee_vel", "z.ee_vel", "prev_y.ee_vel", "prev_z.ee_vel"],
                add_ee_wrench_to_observation=False,
                add_joint_position_to_observation=False,
            ),
        )

        alignment_frames = {
            name: _task_frame(name, [None] * 6, alignment_controller_overrides)
            for name in ("left", "right")
        }
        # x/z are fixed POS axes (policy_mode=None) not listed in absolute_axes, so
        # move_delta resolves them as entry_pose + delta(=0) -- i.e. hold wherever the
        # arm currently is. y/rx/ry/rz are absolute_axes, so they resolve straight to
        # the task frame's configured DEFAULT_RAIL_Y_M / DEFAULT_GRASP_ORIENTATION_RPY
        # target regardless of the entry pose.
        alignment = MoveDeltaPrimitiveConfig(
            notes="Move both arms slowly into the rail plane and fixed grasp orientation, holding entry x/z.",
            processor=alignment_processor,
            task_frame=alignment_frames,
            absolute_axes={"left": ["y", "rx", "ry", "rz"], "right": ["y", "rx", "ry", "rz"]},
        )

        left_grasp = ManipulationPrimitiveConfig(
            notes="Teleop the left grasp in X/Z only, gripper held open; SUCCESS (either "
            "SpaceMouse button) moves on to right_grasp, whose static target closes it.",
            processor=left_grasp_processor,
            task_frame={
                name: TaskFrame(
                    target=list(frame.target),
                    control_mode=list(frame.control_mode),
                    policy_mode=list(frame.policy_mode),
                    origin=list(frame.origin),
                    controller_overrides=controller_overrides,
                )
                for name, frame in _single_arm_frames("left", "right").items()
            },
            teleop_mapping={"left": "main"},
        )
        right_grasp = ManipulationPrimitiveConfig(
            notes="Teleop the right grasp in X/Z only, gripper held open (left is already "
            "closed, statically, the moment this primitive activates); SUCCESS (either "
            "SpaceMouse button) moves on to zero_ft, whose static target closes right too.",
            processor=right_grasp_processor,
            task_frame={
                name: TaskFrame(
                    target=list(frame.target),
                    control_mode=list(frame.control_mode),
                    policy_mode=list(frame.policy_mode),
                    origin=list(frame.origin),
                    controller_overrides=controller_overrides,
                )
                for name, frame in _single_arm_frames("right", "left").items()
            },
            teleop_mapping={"right": "main"},
        )

        def cooperative_task_frame(
            policy_mode: list[PolicyMode | None],
            frame_controller_overrides: dict[str, Any],
        ) -> dict[str, TaskFrame]:
            # Fresh TaskFrame instances every call -- shared across calibrate/
            # cooperative_reset/cooperative_insert, which must not leak target mutations into
            # each other.
            return {
                "left": TaskFrame(
                    target=_rail_target("left"),
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=list(policy_mode),
                    origin=[0.0] * 6,
                    controller_overrides=frame_controller_overrides,
                ),
                "right": TaskFrame(
                    target=_rail_target("right"),
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=[None] * 6,
                    origin=[0.0] * 6,
                    controller_overrides=frame_controller_overrides,
                ),
            }

        # cooperative_reset/cooperative_insert are translation-only against a rigid rail --
        # cap contact force well below the shared 30 N. Rotation stays at the shared limit
        # (locked/absolute there, not the live contact interface); reference-limiting is
        # already on for every axis and its budget is wrench_limits/kp, so it tightens
        # automatically alongside this, nothing else to change.
        cooperative_controller_overrides = dict(controller_overrides)
        cooperative_controller_overrides["wrench_limits"] = [
            self.cooperative_translation_wrench_limit_n,
        ] * 3 + list(controller_overrides["wrench_limits"][3:])

        # Recovery: cooperative_reset rigid/servo instead of compliant -- see the field
        # docstring above. Only meaningful entering straight into the stuck loop via
        # skip_grasp; validated in __post_init__ below.
        cooperative_reset_controller_overrides = cooperative_controller_overrides
        if self.cooperative_reset_servo_recovery:
            cooperative_reset_controller_overrides = dict(controller_overrides)
            cooperative_reset_controller_overrides.update(
                use_force_mode=False,
                simple_pose_use_servo=True,
                compliance_reference_limit_enable=[False] * 6,
            )

        # calibrate needs real range on every axis it manually adjusts -- ry (teleop,
        # potentially tens of degrees from nominal) and pivot_offset_x/y (`,`/`.`/`[`/`]`,
        # potentially several cm from the raw grasp midpoint). CooperativeFramePrimitive's vtcp
        # reference-limit clamp (_clamp_vtcp_reference_error) would otherwise cap either at
        # wrench_limits[i]/kp[i] of instantaneous lead over the arm's actual measured pose --
        # wrench_limits[4]/kp[4] = 4/150 ~= 1.5 deg for ry, wrench_limits[0]/kp[0] = 30/3000 =
        # 1cm for a pivot offset -- past which further nudges silently stop moving the arm at
        # all (the internal/printed offset keeps climbing regardless, since the clamp only
        # bounds what gets sent, not the value itself -- exactly the "works once, then nothing"
        # symptom this was found from). Disabled for every axis, here only -- wrench_limits/kp
        # themselves (the actual torque/stiffness) are untouched, and cooperative_reset/
        # cooperative_insert's own protection is unaffected.
        calibrate_ry_controller_overrides = dict(controller_overrides)
        calibrate_ry_controller_overrides["compliance_reference_limit_enable"] = [False] * 6

        # pushdown: same unrestricted rotation range as calibrate (live ry teleop),
        # plus rz softened to pushdown_yaw_kp/pushdown_yaw_wrench_limit_nm so the rail can
        # settle into whatever yaw the slot wants instead of fighting a rigid target.
        pushdown_controller_overrides = dict(cooperative_controller_overrides)
        pushdown_controller_overrides["compliance_reference_limit_enable"] = [True] * 3 + [False] * 3
        pushdown_controller_overrides["kp"] = list(cooperative_controller_overrides["kp"])
        pushdown_controller_overrides["kp"][5] = self.pushdown_yaw_kp
        pushdown_controller_overrides["wrench_limits"] = list(cooperative_controller_overrides["wrench_limits"])
        pushdown_controller_overrides["wrench_limits"][5] = self.pushdown_yaw_wrench_limit_nm

        cooperative_env_kwargs = {
            "driver": "left",
            "fps": float(self.fps),
            "robot_base_pose_in_world": {
                "right": list(self.right_base_pose_in_left_base),
            },
            "pivot_offset_x": 0.0 if self.pivot_offset_x is None else float(self.pivot_offset_x),
        }
        # x and ry are both frozen at whatever they measure on the loop's first activation
        # (persisted via FROZEN_AXES_RUNTIME_KEY, not re-measured every episode) in
        # cooperative_reset and cooperative_insert alike -- only calibrate exposes ry
        # live; x has no live-teleop primitive at all (policy_mode locks it everywhere).
        # x_offset/ry_angle override either with a known-good exact value instead -- see
        # CooperativeFramePrimitive.driver_axis_overrides.
        axis_overrides = {}
        if self.x_offset is not None:
            axis_overrides[0] = self.x_offset
        if self.ry_angle is not None:
            axis_overrides[4] = self.ry_angle
        cooperative_insert_env_kwargs = {
            **cooperative_env_kwargs,
            "freeze_driver_axes_at_entry": (0, 4),
            "driver_axis_overrides": axis_overrides,
        }

        # Re-zero, not compliant: hold rigidly in plain position mode (not force mode) while
        # zeroing, so the zero operation itself isn't happening under an active admittance
        # loop that's already reacting to the very bias being nulled out.
        zero_ft_controller_overrides = dict(controller_overrides)
        zero_ft_controller_overrides.update(use_force_mode=False, simple_pose_use_servo=True)

        def zero_ft_primitive(notes: str) -> ZeroFTPrimitiveConfig:
            return ZeroFTPrimitiveConfig(
                notes=notes,
                processor=cooperative_processor,
                task_frame={
                    name: TaskFrame(
                        target=_rail_target(name),
                        control_mode=[ControlMode.POS] * 6,
                        policy_mode=[None] * 6,
                        origin=[0.0] * 6,
                        controller_overrides=zero_ft_controller_overrides,
                    )
                    for name in ("left", "right")
                },
                settle_duration_s=self.zero_ft_settle_duration_s,
            )

        # First zero: right after grasping, before the compliant hold phase begins. Second
        # zero: right before cooperative_reset itself, catching any residual bias picked up
        # while teleoperating ry in calibrate -- and, since it sits right before
        # cooperative_reset, this is also skip_grasp's start/reset primitive (the first
        # zero_ft is unreachable there, since teleop_left/teleop_right/calibrate are
        # all skipped -- but zeroing while a settled, static grasp is already held is exactly
        # the same "not mid-motion" situation the first zero_ft relies on, so it's just as
        # safe as an entry point).
        zero_ft = zero_ft_primitive(
            "Re-zero both F/T sensors now that both grippers are loaded with the workpiece."
        )
        zero_ft_before_cooperative = zero_ft_primitive(
            "Re-zero both F/T sensors once more right before the compliant hold phase begins "
            "-- also skip_grasp's entry point, since teleop_left/teleop_right/calibrate "
            "are unreachable there."
        )

        calibrate = ManipulationPrimitiveConfig(
            notes="Teleop ry to the right value (printed live in the status line), and drag "
            "the pivot_offset_x/y sliders (a separate matplotlib window) toward the actual "
            "contact point; space moves on. rz is never touched. Once you know good values, "
            "ry_angle/x_offset+skip_grasp=True reuse them exactly instead of measuring/"
            "dragging again -- this primitive is unreachable there anyway.",
            processor=cooperative_processor,
            env_class=CalibratePrimitive,
            env_kwargs={
                **cooperative_env_kwargs,
                "pivot_offset_range_m": self.pivot_offset_range_m,
                "calibration_path": self.calibration_path,
            },
            task_frame=cooperative_task_frame(COOPERATIVE_LEFT_POLICY, calibrate_ry_controller_overrides),
            teleop_mapping={"left": "main"},
        )

        cooperative_reset = ManipulationPrimitiveConfig(
            notes="Teleop the shared pivot (Y/Z translation only -- X and Ry stay frozen at "
            "whatever they measure on entry) into the episode's initial state; space starts "
            "cooperative_insert. Up arrow jumps back to calibrate to re-teleop ry.",
            processor=cooperative_reset_processor,
            task_frame=cooperative_task_frame(COOPERATIVE_INSERT_LEFT_POLICY, cooperative_reset_controller_overrides),
            env_class=CooperativeFramePrimitive,
            env_kwargs=cooperative_insert_env_kwargs,
            teleop_mapping={"left": "main"},
        )

        cooperative_insert = ManipulationPrimitiveConfig(
            notes="Learn cooperative Y/Z translation only -- X and Ry are frozen at whatever "
            "they measure on entry (or x_offset/ry_angle, if set), Rx/Rz stay at their locked "
            "targets as always. Teleop for demos, or a policy via actor_server.py/"
            "learner_server.py. Success via space, or reward_classifier_path if supplied.",
            processor=cooperative_insert_processor,
            task_frame=cooperative_task_frame(COOPERATIVE_INSERT_LEFT_POLICY, cooperative_controller_overrides),
            # action is 2-dim here (x/z translation only). Supplying real dataset_stats matters
            # -- an unset one silently falls back to SACConfig's own placeholder shape/range
            # (see share/rl/runtime.py resolve_policy_dataset_stats), which only checks shape,
            # not whether the values make sense for this env.
            # Everything below dataset_stats but vision_encoder_name/freeze_vision_encoder is
            # copied from InsertionSACConfig (the hoermann connector env's own SAC preset).
            # That preset actually leaves vision_encoder_name unset (relies on lerobot's small
            # default DefaultImageEncoder, trained from scratch via freeze_vision_encoder=
            # False) -- we use a real pretrained backbone instead (same helper2424/resnet10
            # train_reward_classifier.py already uses), paired with freeze_vision_encoder=True:
            # the standard recipe for vision-based SAC is frozen pretrained features, not
            # fine-tuning a backbone through sparse/noisy RL gradients. modeling_sac.py's
            # PretrainedImageEncoder measures the encoder's output spatial size via a dummy
            # forward pass rather than hardcoding it, so this works fine at the 64x64
            # resize_size already configured in _processor() -- no other change needed.
            policy=SACConfig(
                device="cuda",
                storage_device="cpu",
                dataset_stats={
                    "action": {
                        "min": [-self.translation_action_scale] * 2,
                        "max": [self.translation_action_scale] * 2,
                    },
                    # Order matches ObservationConfig: dy/dz (position), then y/z velocity,
                    # then prev_y/prev_z. Position bound is half of
                    # cooperative_insert_position_range_m; velocity/prev-action share
                    # translation_action_scale (prev-action *is* that quantity, one step late).
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
            env_class=CooperativeInsertPrimitive,
            env_kwargs=cooperative_insert_env_kwargs,
            teleop_mapping={"left": "main"},
        )

        # Scripted, both arms together: hold x/y/rotation at whatever they are (delta=0, no
        # absolute_axes -- see align_to_rail's own comment above for why that means "hold"),
        # move only z by +pull_out_height_m in world frame. Stays compliant
        # (cooperative_controller_overrides, same softened wrench limit as cooperative_reset/
        # cooperative_insert) rather than a rigid move -- extracting from a still-engaged
        # insertion should give the same way a bind would, not fight it.
        pull_out = MoveDeltaPrimitiveConfig(
            notes="Lift both arms pull_out_height_m straight up (world +z) after a successful "
            "cooperative_insert, holding x/y/rotation wherever they are; then back to "
            "cooperative_reset.",
            processor=cooperative_processor,
            delta_frame="world",
            delta=[0.0, 0.0, self.pull_out_height_m, 0.0, 0.0, 0.0],
            task_frame={
                name: TaskFrame(
                    target=_rail_target(name),
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=[None] * 6,
                    origin=[0.0] * 6,
                    controller_overrides=cooperative_controller_overrides,
                )
                for name in ("left", "right")
            },
        )

        pushdown_env_kwargs = {
            **cooperative_env_kwargs,
            # ry must stay frozen here too, same as cooperative_reset/cooperative_insert
            # (cooperative_insert_env_kwargs's (0, 4)) -- otherwise CooperativeFramePrimitive's
            # capture (no per-axis freeze/override -> static DEFAULT_GRASP_ORIENTATION_RPY
            # target) snaps ry to its static 0.0 (parallel) the instant pushdown activates,
            # instead of continuing from wherever cooperative_insert left it. PolicyMode.RELATIVE
            # then teleops live from that frozen baseline, same as x does.
            "freeze_driver_axes_at_entry": (0, 4),
            "driver_axis_overrides": {0: self.x_offset} if self.x_offset is not None else {},
            "midpoint_offset": [0.0, 0.0, self.pushdown_z_offset_m],
        }
        pushdown = ManipulationPrimitiveConfig(
            notes="Manual exploration of seating the rail fully after a successful "
            "cooperative_insert (demo=True only): x/y locked wherever cooperative_insert left "
            "them, z gently pushed (pushdown_z_offset_m) under compliance, rz soft (settles "
            "into the slot's actual yaw instead of fighting it), ry live -- teleop it toward "
            "0deg by hand, no movement unless actively teleoped. Space back to "
            "cooperative_reset.",
            processor=cooperative_processor,
            task_frame=cooperative_task_frame(PUSHDOWN_LEFT_POLICY, pushdown_controller_overrides),
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
            # zero_ft/zero_ft_before_cooperative are scripted (settle -> zero_ft() -> done in
            # one on_entry call), so they advance on their own completion flag rather than an
            # operator/policy event. load_calibration bypasses calibrate entirely -- straight
            # to zero_ft_before_cooperative, reusing the loaded pivot_offset_x/x_offset/
            # ry_angle instead of measuring them again.
            OnEvent(
                source="zero_ft",
                target="zero_ft_before_cooperative" if self.load_calibration else "calibrate",
                event_key=PRIMITIVE_COMPLETE_INFO_KEY,
            ),
            OnSuccess(source="calibrate", target="zero_ft_before_cooperative"),
            OnEvent(
                source="zero_ft_before_cooperative",
                target="cooperative_reset",
                event_key=PRIMITIVE_COMPLETE_INFO_KEY,
            ),
            # Escape hatch: re-teleop ry without a full reset. Listed before the ordinary
            # advance edge below since it's keyed on a different physical input (up arrow, not
            # space) and never competes with it -- order doesn't affect correctness here.
            OnEvent(
                source="cooperative_reset",
                target="calibrate",
                event_key=RECALIBRATE_RY_REQUEST_FLAG,
            ),
            OnSuccess(source="cooperative_reset", target="cooperative_insert"),
        ]
        # Manual (space, or a mapped SpaceMouse button -- see share/teleoperators/spacemouse)
        # always works; a reward classifier is an *additional* automatic trigger, not a
        # replacement -- whichever fires first wins (transitions are evaluated in order and
        # the first that fires short-circuits the rest, see
        # ManipulationPrimitiveNet._step_env_and_check_transitions). The time limit is listed
        # first as a safety fallback, but in practice never races the other two -- it only
        # ever fires when neither has. It targets cooperative_reset directly, not pull_out --
        # a timeout means nothing was actually achieved, so there's nothing to pull out of;
        # only a genuine success (manual or classifier) routes through pull_out.
        transitions.append(
            OnTimeLimit(
                source="cooperative_insert",
                target="cooperative_reset",
                max_steps=int(self.fps * self.cooperative_insert_episode_duration_s),
            )
        )
        # demo=True routes a genuine success to pushdown (manual seating exploration) instead
        # of the normal pull_out/loop-again path -- both success triggers (manual, classifier)
        # redirect the same way.
        insert_success_target = "pushdown" if self.demo else "pull_out"
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
            # Straight into calibrate -- skips zero_ft_before_cooperative's F/T
            # re-zero entirely for now, trading correctness there for faster iteration while
            # setting ry/pivot_offset_x interactively. Revisit once skip_grasp's semantics
            # settle -- the zero step matters for real recording/training runs. load_calibration
            # skips calibrate too (see the zero_ft edge above) -- go one further and skip past
            # its now-pointless entry point straight to zero_ft_before_cooperative.
            self.start_primitive = "zero_ft_before_cooperative" if self.load_calibration else "calibrate"
            self.reset_primitive = self.start_primitive

        super().__post_init__()


__all__ = [
    "CooperativeFramePrimitive",
    "CooperativeInsertPrimitive",
    "DEFAULT_RAIL_Y_M",
    "FROZEN_AXES_RUNTIME_KEY",
    "DEFAULT_GRASP_ORIENTATION_RPY",
    "RECALIBRATE_RY_KEY",
    "RECALIBRATE_RY_REQUEST_FLAG",
    "RailBimanualGraspEnvConfig",
    "CalibratePrimitive",
]
