"""Bimanual pick: teleop+grasp left, teleop+grasp right, then teleop the shared midpoint.

ARCHIVED -- not registered (see experiments/envs/__init__.py), not actively maintained.
rail_bimanual_grasp.py carries its own self-contained copy of CooperativeFramePrimitive and no
longer imports anything from here; the two have no dependency on each other anymore. Kept for
reference/history -- import directly by path (experiments.envs.archive.bimanual_pick) to run
it again.

One SpaceMouse drives whichever arm is active via teleop_mapping. Uses env_class to plug a
custom cooperative-frame env into a plain ManipulationPrimitiveConfig -- no bespoke Config
subclass needed.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from lerobot.envs import EnvConfig
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.robots import Robot
from pynput import keyboard

from share.envs.manipulation_primitive.config_manipulation_primitive import (
    EventConfig,
    GripperConfig,
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveProcessorConfig,
    MoveDeltaPrimitiveConfig,
    ObservationConfig,
)
from share.envs.manipulation_primitive.env_manipulation_primitive import ManipulationPrimitive
from share.processor.info import AddKeyboardEventsAsInfoStep
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.envs.manipulation_primitive_net.transitions import OnSuccess
from share.robots.ur import SimURConfig, URConfig
from share.teleoperators import TeleopEvents
from share.teleoperators.spacemouse import SpaceMouseConfig
from share.utils.transformation_utils import (
    euler_xyz_from_rotation,
    euler_xyz_from_rotvec,
    rotation_from_extrinsic_xyz,
    task_pose_to_world_pose,
    world_pose_to_task_pose,
    wrap_to_pi,
)

# Shared-runtime-value key every CooperativeFramePrimitive activation (calibration included)
# reads/writes to hand the live pivot_offset_x scalar off across a primitive switch -- see the
# CooperativeFramePrimitive docstring below.
PIVOT_OFFSET_X_RUNTIME_KEY = "cooperative_pivot_offset_x"
PIVOT_OFFSET_X_INCREASE_EVENT = "pivot_offset_x_increase"
PIVOT_OFFSET_X_DECREASE_EVENT = "pivot_offset_x_decrease"

# Shared-runtime-value key every CooperativeFramePrimitive activation reads/writes to hand off
# each robot's fixed offset from the vtcp -- the grasp geometry between robots -- captured once
# and reused thereafter instead of being re-derived every activation. See the
# CooperativeFramePrimitive docstring below.
OFFSET_FROM_VTCP_RUNTIME_KEY = "cooperative_offset_from_vtcp"


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

    ``pivot_offset_x`` is the pivot's one calibrated degree of freedom: a scalar world-x
    offset from the raw TCP midpoint, for tasks (like this one) where the rotation point only
    ever needs to slide along one world axis (e.g. a rail aligned with world x). It is added
    to the pivot's world-x on capture, and can be nudged live with `,`/`.` at any time,
    including mid-activation while already teleoping, which shifts the live pivot immediately.
    Every step this primitive is active, its current value is published to
    ``PIVOT_OFFSET_X_RUNTIME_KEY`` in shared runtime state; every activation's capture (above)
    checks that key first and adopts it over its own configured starting value if present.
    That is how a dedicated calibration primitive (e.g. ``PivotOffsetXCalibrationPrimitive``
    below) hands a measured value to every later cooperative primitive without them sharing
    one Python instance -- and also how a live nudge made in one cooperative primitive (e.g.
    ``reset``) keeps propagating to the next one entered afterward. Pass a known-good value
    through the primitive config (and skip the calibration primitive in the graph entirely)
    once it has been measured for a given cell.

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

    ``freeze_driver_axes_at_entry`` is a third, simpler kind of lock: rotational axis indices
    (3/4/5 for rx/ry/rz) that should hold *whatever the driver's own pose actually measures*
    on capture, instead of the driver's own configured static target. Unlike pivot_offset_x,
    this is not calibrated/persisted across activations -- it is re-captured fresh
    every time this primitive activates, exactly like a passive (non-driver, non-teleop)
    robot already holds its own entry pose elsewhere in this codebase. Use it to demote an
    axis from policy-learnable to fixed without hardcoding an assumed value for it: whatever
    orientation the operator left the arm in (e.g. during a preceding manual primitive) is
    what gets held for the rest of this activation.

    ``driver_axis_overrides`` supplies an exact numeric value for one or more of those same
    frozen axes instead of reading the driver's currently measured pose -- for a caller that
    wants full run-to-run reproducibility (a known-good calibrated angle) rather than whatever
    small physical variance a fresh grasp happens to land at. An axis with a supplied override
    still needs to be listed in ``freeze_driver_axes_at_entry`` to be captured at all; the
    override only changes where its captured value comes from. It propagates to every robot's
    offset capture, not just the driver's: pose_world is one shared world frame, so an
    overridden rotation axis means the same declared value for everyone in it, and every
    robot needs its own offset corrected the same way or their offsets stop agreeing with
    each other -- the follower would silently keep holding its old measured value instead of
    rotating along with the driver's, exactly the way live teleop on that axis already moves
    both robots together through this same mechanism.

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
        pivot_offset_x_step: float = 0.005,
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
        self.pivot_offset_x = float(pivot_offset_x)
        self.pivot_offset_x_step = float(pivot_offset_x_step)
        self._pivot_offset_x_applied = 0.0
        self._pivot_offset_x_event = AddKeyboardEventsAsInfoStep(
            mapping={
                PIVOT_OFFSET_X_INCREASE_EVENT: ".",
                PIVOT_OFFSET_X_DECREASE_EVENT: ",",
            },
            pulse_events=(PIVOT_OFFSET_X_INCREASE_EVENT, PIVOT_OFFSET_X_DECREASE_EVENT),
        )

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
        if hasattr(self, "_pivot_offset_x_event"):
            self._pivot_offset_x_event.reset()

    def _update_pivot_offset_x(self) -> None:
        """Apply any live `,`/`.` nudges to the already-captured pivot and print on change."""
        offset_info = self._pivot_offset_x_event.info({})
        if offset_info.get(PIVOT_OFFSET_X_INCREASE_EVENT, False):
            self.pivot_offset_x += self.pivot_offset_x_step
        if offset_info.get(PIVOT_OFFSET_X_DECREASE_EVENT, False):
            self.pivot_offset_x -= self.pivot_offset_x_step

        delta = self.pivot_offset_x - self._pivot_offset_x_applied
        if delta == 0.0:
            return
        if self._vtcp_world is not None:
            self._vtcp_world[0] += delta
        self._pivot_offset_x_applied = self.pivot_offset_x
        print(f"[cooperative] pivot_offset_x = {self.pivot_offset_x:.4f} m", flush=True)

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
            midpoint[0] += self.pivot_offset_x
            self._pivot_offset_x_applied = self.pivot_offset_x
            vtcp_rotation = list(self.task_frame[self.driver].target[3:6])
            # freeze_driver_axes_at_entry axes hold whatever the driver actually measures
            # right now instead of the driver's static configured target -- re-captured fresh
            # every activation, not calibrated/persisted. driver_axis_overrides supplies an
            # exact value for one of those axes instead, for run-to-run reproducibility.
            for axis in self.freeze_driver_axes_at_entry:
                if 3 <= axis < 6:
                    if axis in self.driver_axis_overrides:
                        vtcp_rotation[axis - 3] = self.driver_axis_overrides[axis]
                    else:
                        vtcp_rotation[axis - 3] = pose_world[self.driver][axis]
            self._vtcp_world = [*midpoint, *vtcp_rotation]
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
                # world frame (see _poses_world), so an overridden rotation axis means the
                # same thing for everyone in it -- e.g. ry is nominally 0.0 for both arms here
                # (only rz differs, mirrored), and live ry teleop already rotates both arms
                # together via this exact offset+vtcp mechanism, so a *fixed* ry override must
                # propagate the same way or the two arms' offsets stop agreeing with each
                # other. Substituting the override into the pose used for capture makes every
                # affected robot's offset zero on that axis instead, exactly like
                # freeze_driver_axes_at_entry's un-overridden case already relies on.
                capture_pose = list(pose_world[name])
                for axis, value in self.driver_axis_overrides.items():
                    if 3 <= axis < 6:
                        capture_pose[axis] = value
                self._offset_from_vtcp[name] = world_pose_to_task_pose(capture_pose, self._vtcp_world)
            if self._shared_runtime_values is not None:
                self.set_runtime_value(OFFSET_FROM_VTCP_RUNTIME_KEY, dict(self._offset_from_vtcp))

        self._update_pivot_offset_x()

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

        return super().step(cooperative_action)


class PivotOffsetXCalibrationPrimitive(CooperativeFramePrimitive):
    """One-shot, keyboard-only measurement of ``pivot_offset_x`` (see the docstring on
    ``CooperativeFramePrimitive.pivot_offset_x`` above).

    Both robots' task frames are fully locked (``policy_mode=[None]*6``) in this primitive's
    config, so the SpaceMouse cannot move anything here -- only the inherited `,`/`.`
    keyboard nudge moves the pivot, one ``pivot_offset_x_step`` at a time. The live value is
    surfaced on every step through ``info["record_status"]`` (both record.py and
    actor_server.py append that to their one-line status), so the operator can watch it
    settle continuously while sliding the grasped workpiece by eye against some external
    reference, rather than only seeing it reprinted on a change.

    There is nothing extra to do on "lock": CooperativeFramePrimitive.step() already
    publishes the live value to PIVOT_OFFSET_X_RUNTIME_KEY every step, so whatever is last
    printed here is exactly what teleop_midpoint/reset/policy pick up the moment they first
    activate. Space (the ordinary OnSuccess advance key, unchanged) just ends this primitive.
    """

    def step(self, action: dict[str, dict[str, float]]):
        obs, reward, terminated, truncated, info = super().step(action)
        info["record_status"] = (
            f"pivot_offset_x = {self.pivot_offset_x:+.4f} m  (,/. to nudge, space to lock)"
        )
        return obs, reward, terminated, truncated, info


def _processor(fps: float, gripper: GripperConfig) -> ManipulationPrimitiveProcessorConfig:
    return ManipulationPrimitiveProcessorConfig(
        fps=fps,
        observation=ObservationConfig(
            add_ee_pos_to_observation=True,
            add_ee_velocity_to_observation=True,
            add_ee_wrench_to_observation=True,
            add_joint_position_to_observation=False,
        ),
        gripper=gripper,
        events=EventConfig(
            key_mapping={
                TeleopEvents.SUCCESS: keyboard.Key.space,
                TeleopEvents.RERECORD_EPISODE: keyboard.Key.left,
                TeleopEvents.STOP_RECORDING: keyboard.Key.down,
            },
            pulse_events=(TeleopEvents.SUCCESS,),
        ),
    )


def _gripper(live: str, held: dict[str, float]) -> GripperConfig:
    enable = {live: True}
    static_pos: dict[str, float | None] = {live: None}
    for name, pos in held.items():
        enable[name] = False
        static_pos[name] = pos
    return GripperConfig(enable=enable, discretize=True, min_pos=0.0, static_pos=static_pos)




def _fixed_rotation_target(robot_name: str) -> list[float]:
    """Return the fixed task-frame orientation for one bimanual arm."""
    if robot_name == "left":
        return [0.0, 0.0, 0.0, float(np.pi), 0.0, float(np.pi)]
    if robot_name == "right":
        return [0.0, 0.0, 0.0, float(np.pi), 0.0, 0.0]
    raise ValueError(f"Unknown bimanual arm {robot_name!r}")


def _lower_dimensional_policy_mode() -> list[PolicyMode | None]:
    """Allow xyz and ry motion; rx and rz remain fixed by the task-frame target."""
    return [PolicyMode.RELATIVE, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, PolicyMode.RELATIVE, None]


def _single_arm_primitive(
    active: str,
    passive: str,
    processor: ManipulationPrimitiveProcessorConfig,
    controller_overrides: dict[str, Any],
) -> MoveDeltaPrimitiveConfig:
    active_target = _fixed_rotation_target(active)
    passive_target = _fixed_rotation_target(passive)
    return MoveDeltaPrimitiveConfig(
        notes=f"Teleop {active} 6-DoF; {passive} holds its entry pose.",
        processor=processor,
        delta={active: [0.0] * 6, passive: [0.0] * 6},
        absolute_axes={active: [3, 5], passive: []},
        task_frame={
            active: TaskFrame(
                target=active_target,
                control_mode=[ControlMode.POS] * 6,
                policy_mode=_lower_dimensional_policy_mode(),
                controller_overrides=controller_overrides,
            ),
            passive: TaskFrame(
                target=passive_target,
                control_mode=[ControlMode.POS] * 6,
                policy_mode=[None] * 6,
                controller_overrides=controller_overrides,
            ),
        },
    )


@EnvConfig.register_subclass("bimanual_pick")
@dataclass
class BimanualPickEnvConfig(ManipulationPrimitiveNetConfig):
    """teleop_left (grasp) -> teleop_right (grasp) -> calibrate_pivot_offset_x ->
    teleop_midpoint (cooperative) -- manual alignment, done once per session -- then reset ->
    policy, looping for repeated episodes: teleop into the initial state in "reset", space
    starts "policy" (record.py for demos, actor_server.py/learner_server.py for HIL-RL); on
    success there, teleop back up in "reset" again for the next episode. Once
    ``pivot_offset_x`` has been measured for a given cell, pass it in and
    calibrate_pivot_offset_x drops out of the graph entirely (teleop_right -> teleop_midpoint
    directly). start_primitive/reset_primitive stay at teleop_left: only the very first reset
    of a run goes there -- reset -> policy -> reset is a plain graph loop (no primitive here
    is_terminal), so it never triggers a second full reset back to teleop_left."""

    fps: int = 30
    left_robot_ip: str = "172.22.22.5"
    right_robot_ip: str = "172.22.22.2"
    # right arm's base pose expressed in left arm's base frame -- calibrate for your cell
    right_base_pose_in_left_base: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    translation_action_scale: float = 0.1
    rotation_action_scale: float = 0.5
    open_gripper_position: float = 0.0
    closed_gripper_position: float = 1.0
    start_primitive: str = "teleop_left"
    reset_primitive: str = "teleop_left"
    # kinematic-only simulated arms instead of real UR hardware -- for previewing this graph
    # with a real SpaceMouse before running on the robots
    mock: bool = False
    # World-x offset of the cooperative pivot from the raw left/right TCP midpoint (see
    # CooperativeFramePrimitive.pivot_offset_x) -- the only degree of freedom this rig's pivot
    # ever needs, since the grasped workpiece slides along a rail aligned with world x. None
    # (default) inserts calibrate_pivot_offset_x into the graph to measure it live once, by
    # keyboard, before teleop_midpoint. A known value skips that primitive entirely and starts
    # every cooperative primitive from it directly (still further nudgeable with `,`/`.`).
    pivot_offset_x: float | None = None

    def __post_init__(self) -> None:
        controller_overrides = {
            "use_force_mode": True,
            "compliance_reference_limit_enable": [True] * 6,
            "kp": [3000.0, 3000.0, 3000.0, 150.0, 150.0, 150.0],
            "kd": [60.0, 60.0, 60.0, 6.0, 6.0, 6.0],
            "wrench_limits": [30.0, 30.0, 30.0, 4.0, 4.0, 4.0],
        }

        if self.mock:
            self.robot = {
                "left": SimURConfig(use_gripper=True, initial_pose=[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]),
                "right": SimURConfig(use_gripper=True, initial_pose=[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]),
            }
        else:
            self.robot = {
                "left": URConfig(robot_ip=self.left_robot_ip, frequency=125, soft_real_time=True, rt_core=3, use_gripper=True),
                "right": URConfig(robot_ip=self.right_robot_ip, frequency=125, soft_real_time=True, rt_core=3, use_gripper=True),
            }
        self.teleop = {
            "left": SpaceMouseConfig(
                action_scale=[self.translation_action_scale] * 3 + [self.rotation_action_scale] * 3,
            ),
        }

        left_processor = _processor(self.fps, _gripper("left", {"right": self.open_gripper_position}))
        right_processor = _processor(self.fps, _gripper("right", {"left": self.closed_gripper_position}))
        # Both arms already hold the workpiece from calibrate_pivot_offset_x onward -- static,
        # closed grippers for calibrate_pivot_offset_x/teleop_midpoint/reset/policy alike.
        held_processor = _processor(
            self.fps,
            GripperConfig(enable=False, static_pos={"left": self.closed_gripper_position, "right": self.closed_gripper_position}),
        )

        def cooperative_task_frame() -> dict[str, TaskFrame]:
            # Fresh TaskFrame instances every call -- each primitive owns its env's copy.
            # TaskFrame.target is mutated in place (set_target_pose(), on_entry()), so sharing
            # instances across two primitives would leak writes.
            return {
                "left": TaskFrame(
                    target=_fixed_rotation_target("left"),
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=_lower_dimensional_policy_mode(),
                    controller_overrides=controller_overrides,
                ),
                "right": TaskFrame(
                    target=_fixed_rotation_target("right"),
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=[None] * 6,
                    controller_overrides=controller_overrides,
                ),
            }

        def insertion_task_frame() -> dict[str, TaskFrame]:
            # xyz translation free; rotation restricted to y only (see CooperativeFramePrimitive
            # docstring) -- the driver's rx/rz stay locked to their static target regardless of
            # what teleop/policy input reports.
            return {
                "left": TaskFrame(
                    target=_fixed_rotation_target("left"),
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=_lower_dimensional_policy_mode(),
                    controller_overrides=controller_overrides,
                ),
                "right": TaskFrame(
                    target=_fixed_rotation_target("right"),
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=[None] * 6,
                    controller_overrides=controller_overrides,
                ),
            }

        def locked_task_frame() -> dict[str, TaskFrame]:
            # Every axis locked for both arms -- the SpaceMouse can't move anything here, only
            # the `,`/`.` keyboard nudge (CooperativeFramePrimitive.pivot_offset_x) can.
            return {
                name: TaskFrame(
                    target=_fixed_rotation_target(name),
                    control_mode=[ControlMode.POS] * 6,
                    policy_mode=[None] * 6,
                    controller_overrides=controller_overrides,
                )
                for name in ("left", "right")
            }

        cooperative_env_kwargs = {
            "driver": "left",
            "fps": float(self.fps),
            "robot_base_pose_in_world": {"right": list(self.right_base_pose_in_left_base)},
            "pivot_offset_x": 0.0 if self.pivot_offset_x is None else float(self.pivot_offset_x),
        }

        calibrate_pivot_offset_x_primitive = ManipulationPrimitiveConfig(
            notes="Nudge the cooperative pivot's world-x offset with `,`/`.` (printed live in "
            "the status line); space locks it in and moves on.",
            processor=held_processor,
            env_class=PivotOffsetXCalibrationPrimitive,
            env_kwargs=cooperative_env_kwargs,
            task_frame=locked_task_frame(),
        )

        midpoint_primitive = ManipulationPrimitiveConfig(
            notes="Teleop the shared pivot; both arms track it cooperatively.",
            processor=held_processor,
            env_class=CooperativeFramePrimitive,
            env_kwargs=cooperative_env_kwargs,
            task_frame=cooperative_task_frame(),
        )

        reset_primitive = ManipulationPrimitiveConfig(
            notes="Teleop the shared pivot into the episode's initial state; space starts policy.",
            processor=held_processor,
            env_class=CooperativeFramePrimitive,
            env_kwargs=cooperative_env_kwargs,
            task_frame=cooperative_task_frame(),
        )

        policy_primitive = ManipulationPrimitiveConfig(
            notes="Cooperative insertion: xyz free, rotation restricted to y. Teleop for demos, "
            "or a policy via actor_server.py/learner_server.py.",
            processor=held_processor,
            env_class=CooperativeFramePrimitive,
            env_kwargs=cooperative_env_kwargs,
            task_frame=insertion_task_frame(),
            policy=SACConfig(device="cpu", storage_device="cpu"),
        )

        self.primitives = {
            "teleop_left": _single_arm_primitive("left", "right", left_processor, controller_overrides),
            "teleop_right": _single_arm_primitive("right", "left", right_processor, controller_overrides),
        }
        if self.pivot_offset_x is None:
            self.primitives["calibrate_pivot_offset_x"] = calibrate_pivot_offset_x_primitive
        self.primitives.update({
            "teleop_midpoint": midpoint_primitive,
            "reset": reset_primitive,
            "policy": policy_primitive,
        })
        self.primitives["teleop_right"].teleop_mapping = {"right": "left"}

        transitions = [OnSuccess(source="teleop_left", target="teleop_right")]
        if self.pivot_offset_x is None:
            transitions += [
                OnSuccess(source="teleop_right", target="calibrate_pivot_offset_x"),
                OnSuccess(source="calibrate_pivot_offset_x", target="teleop_midpoint"),
            ]
        else:
            transitions.append(OnSuccess(source="teleop_right", target="teleop_midpoint"))
        transitions += [
            OnSuccess(source="teleop_midpoint", target="reset"),
            OnSuccess(source="reset", target="policy"),
            OnSuccess(source="policy", target="reset"),
        ]
        self.transitions = transitions

        super().__post_init__()



__all__ = [
    "CooperativeFramePrimitive",
    "PivotOffsetXCalibrationPrimitive",
    "BimanualPickEnvConfig",
]
