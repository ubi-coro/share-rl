"""SpaceMouse teleop for an already-commissioned FR3.

Run this only after franka_first_motion.py has passed on the same robot and
host (see wiki/franka_setup.rst, "First motion and acceptance", steps 1-4).
This script does not add anything beyond what that acceptance progression
already exercised -- it just replaces the scripted 1 cm/s reference with a
live SpaceMouse and widens the workspace slightly.

Controls:
    - Move/tilt the SpaceMouse cap: relative Cartesian velocity in the world
      frame, integrated by the 500 Hz bridge (see CartesianReferenceController)
      and tracked by Franky's native CartesianImpedanceTrackingMotion.
    - Button 0: ends the session (a few zero-velocity commands, then a clean
      disconnect). Ctrl-C works too.

Safety:
    - Keep a hand near the emergency stop and the enabling/hand button on the
      controller box for the entire session. Software cannot override either.
    - The commanded workspace is a small box around the pose the arm was in
      when this script connected -- see TRANSLATION_HALF_RANGE_M below.
    - Stiffness and force_constraints below are deliberately soft. Raise them
      only after this profile feels controllable.
    - use_gripper is off. Enable it in FrankaConfig once arm teleop is
      comfortable, not on the first run.
"""

from __future__ import annotations

import time

from lerobot.utils.errors import DeviceNotConnectedError

from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    PolicyMode,
    TaskFrame,
)
from share.robots.franka import Franka, FrankaConfig
from share.robots.franka.lerobot_robot_franka.controller import FrankaControllerError
from share.teleoperators import TeleopEvents
from share.teleoperators.spacemouse import SpaceMouse, SpaceMouseConfig


# TODO: replace with the FR3's actual IP (check Desk / your network notes).
ROBOT_IP = "172.16.0.2"

# Half-range of the allowed translation box around the connect-time pose, in
# meters. Rotation is left unconstrained (matches franka_first_motion.py).
TRANSLATION_HALF_RANGE_M = 0.10

# Max commanded velocity at full SpaceMouse deflection: [x, y, z, rx, ry, rz]
# in m/s / rad/s. Conservative on purpose -- raise once this feels tame.
ACTION_SCALE = [0.05, 0.05, 0.05, 0.25, 0.25, 0.25]

# Same conservative profile as franka_first_motion.py. force_constraints is
# not here -- Franky fixes it for the whole connection (no live setter), so
# it's a FrankaConfig field below, not a per-command controller override.
CONTROLLER_OVERRIDES = {
    "translational_stiffness": 200.0,
    "rotational_stiffness": 20.0,
    "compliance_reference_limit_enable": [True] * 6,
}
FORCE_CONSTRAINTS = [15.0, 15.0, 15.0, 2.0, 2.0, 2.0]

LOOP_HZ = 100.0


def main() -> None:
    robot = Franka(
        FrankaConfig(
            robot_ip=ROBOT_IP,
            enforce_realtime=True,
            force_constraints=FORCE_CONSTRAINTS,
        )
    )
    teleop = None

    try:
        # Opens the HID device immediately -- fails fast if the SpaceMouse
        # isn't plugged in or udev permissions are wrong.
        teleop = SpaceMouse(
            SpaceMouseConfig(
                action_scale=ACTION_SCALE,
                button_mapping={0: {"event": TeleopEvents.SUCCESS, "toggle": False}},
            )
        )

        print(f"Connecting to FR3 at {ROBOT_IP} ...")
        robot.connect(calibrate=False)
        robot.zero_ft()
        teleop.connect()

        observation = robot.get_observation()
        current_pose = [
            observation[f"{axis}.ee_pos"] for axis in ("x", "y", "z", "rx", "ry", "rz")
        ]
        half = TRANSLATION_HALF_RANGE_M
        robot.set_task_frame(
            TaskFrame(
                space=ControlSpace.TASK,
                target=[0.0] * 6,
                policy_mode=[PolicyMode.RELATIVE] * 6,
                control_mode=[ControlMode.POS] * 6,
                origin=[0.0] * 6,
                min_pose=[
                    current_pose[0] - half,
                    current_pose[1] - half,
                    current_pose[2] - half,
                    -3.14159,
                    -3.14159,
                    -3.14159,
                ],
                max_pose=[
                    current_pose[0] + half,
                    current_pose[1] + half,
                    current_pose[2] + half,
                    3.14159,
                    3.14159,
                    3.14159,
                ],
                controller_overrides=CONTROLLER_OVERRIDES,
            )
        )

        print(
            "Live. SpaceMouse drives the arm; button 0 or Ctrl-C ends the "
            "session.\nKeep a hand near the e-stop and the enabling button."
        )
        dt = 1.0 / LOOP_HZ
        while True:
            t_start = time.perf_counter()

            events = teleop.get_teleop_events()
            if events.get(TeleopEvents.SUCCESS):
                print("Button 0 pressed -- ending session.")
                break

            raw_action = teleop.get_action()
            # Keys are ".ee_pos", not ".ee_vel": the task frame below uses
            # ControlMode.POS + PolicyMode.RELATIVE, so the controller treats
            # this target as a velocity to integrate (see franka_first_motion.py).
            # ".ee_vel" would instead select true velocity control -- a
            # different control law from the one the gains below are tuned for.
            action = {
                f"{axis}.ee_pos": raw_action[f"{axis}.vel"]
                for axis in ("x", "y", "z", "rx", "ry", "rz")
            }
            robot.send_action(action)

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0.0, dt - elapsed))
    except FrankaControllerError as error:
        print(
            f"\nController fault: {error}\n"
            "If the e-stop or the enabling button was pressed, FCI drops and "
            "libfranka reports an error in Desk. Clear it there, re-activate "
            "FCI, then re-run this script."
        )
    except KeyboardInterrupt:
        print("\nCtrl-C -- ending session.")
    finally:
        if robot.is_connected:
            print("Ramping down ...")
            for _ in range(20):
                try:
                    robot.send_action({f"{axis}.ee_pos": 0.0 for axis in ("x", "y", "z", "rx", "ry", "rz")})
                except FrankaControllerError:
                    break
                time.sleep(0.01)
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()
        try:
            robot.disconnect()
        except DeviceNotConnectedError:
            pass


if __name__ == "__main__":
    main()
