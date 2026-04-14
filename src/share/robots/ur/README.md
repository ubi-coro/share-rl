# UR Integration Notes

## Installation

The UR package expects both `ur_rtde` and `ur_ikfast` to be installed.

`ur_ikfast` is pulled from GitHub via the UR package dependency metadata and builds
native extensions for the supported UR families:

- `ur3`
- `ur3e`
- `ur5`
- `ur5e`
- `ur10`
- `ur10e`

Make sure the target environment has the usual native build prerequisites available,
including a working C/C++ toolchain, LAPACK, and the Python build dependencies needed
by `ur_ikfast`.

## Command Modes

The shared pipeline keeps `ControlSpace` geometric (`TASK` or `JOINT`) and selects the
UR actuation behavior with `TaskFrame.stiffness_mode`:

- `TASK + COMPLIANT` uses `forceMode`
- `JOINT + COMPLIANT` uses `directTorque`
- `TASK + STIFF` uses `servoL`
- `JOINT + STIFF` uses `servoJ`

`URConfig.default_stiffness_mode` controls the default when a task frame leaves
`stiffness_mode=None`.
