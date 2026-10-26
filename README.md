# share-rl

`share-rl` is a LeRobot extension for structured manipulation workflows.

The repository adds four main things on top of LeRobot:

- new robot and teleoperator integrations, packaged as sibling installable Python projects
- task-frame manipulation primitives and manipulation-primitive nets (MP-Nets) for graph-structured behavior
- recording, training, and offline evaluation helpers around those MP-Nets
- a local-first workspace/runtime layer for editing and reasoning about MP-Nets programmatically

This README is written for someone who wants to use the repo, understand what is installable, and see the intended MP-Net usage pattern.

## What You Get

From the root `share-rl` package itself you get:

- `share.envs.manipulation_primitive`: a single manipulation primitive environment with task-frame constraints, processor pipelines, and intervention logic
- `share.envs.manipulation_primitive_net`: a graph of primitives plus transitions, start/reset semantics, and MP-Net validation
- `share.workspace.mpnet`: helpers to create, save, load, validate, summarize, and edit MP-Net configs
- `share.scripts.record`: record datasets from MP-Nets with per-primitive datasets and optional per-primitive policies
- `share.scripts.train`: thin adapter for training policies on recorded datasets
- `share.scripts.eval_on_dataset`: lightweight offline evaluation/summary adapter for recorded primitive datasets

From the sibling standalone packages in this repo you can also get:

- `lerobot_robot_ur`: UR e-series integration
- `lerobot_robot_viperx`: Trossen ViperX integration
- `lerobot_teleoperator_delta_keyboard`: keyboard delta-velocity teleoperation
- `lerobot_teleoperator_spacemouse`: SpaceMouse teleoperation
- `lerobot_teleoperator_widowx`: WidowX leader-arm style teleoperation

Those sibling packages are installed through the root extras when you install from this repo checkout.

## Current Status

What is already useful:

- MP-Net config objects, transitions, serialization, and validation
- manipulation-primitive processor stack
- UR, ViperX, keyboard, SpaceMouse, and WidowX integration code
- dataset recording, training adapter, and dataset-summary evaluation adapter

What is still more experimental:

- some policy code under `src/share/policies`
- the broader workspace/agent tooling under `src/share/workspace`
- packaging polish for publishing the sibling subprojects independently outside this monorepo checkout

## Installation

The repository currently targets Python 3.10+ and LeRobot `>=0.5`.

A typical local setup is:

```bash
conda create -n share python=3.12
conda activate share
pip install -e .
```

### Optional extras

These extras are designed for use from this repository checkout.

```bash
pip install -e .[ur]
pip install -e .[spacemouse]
pip install -e .[aloha]
pip install -e .[bota]
pip install -e .[all]
```

What each extra does:

- `.[ur]`: installs the root `share` package plus the sibling `lerobot_robot_ur` project from `src/share/robots/ur`
- `.[spacemouse]`: installs the root `share` package plus the sibling `lerobot_teleoperator_spacemouse` project from `src/share/teleoperators/spacemouse`
- `.[aloha]`: installs the root `share` package plus the sibling `lerobot_robot_viperx` and `lerobot_teleoperator_widowx` projects
- `.[bota]`: installs `pyserial`, the only dependency of the Bota wrist force/torque sensor reader (see below)
- `.[all]`: installs the hardware extras above plus test dependencies

If you want one sibling package directly, you can also install it by path:

```bash
pip install -e src/share/robots/ur
pip install -e src/share/teleoperators/spacemouse
```

### Bota wrist force/torque sensor (ViperX, diagnosis only)

A Bota MiniONE between the wrist and the gripper measures contact forces, e.g. to see an open-loop policy
grabbing before a human hand lets go. Enable it on the robot config with the sensor's stable port:

```bash
--robot.bota.enable=true --robot.bota.port=/dev/serial/by-id/usb-Bota_Systems_..._<serial>-if00
```

- `ViperX.calibrate()` (and the first `connect()` without a saved fit) fits the gripper's payload from the whole
  trajectory you record. Workflow: support the arm and press Enter, which switches the torque off; move the arm
  slowly by hand through as many orientations as you can, tilting and rolling the wrist and holding it by the wrist
  or forearm, not the gripper. A live line shows the measured wrench, whether the current sample is used (moves
  faster than about 17 deg/s are recorded but not used, the model is static) and how many different orientations
  you have covered, at least 12. Press Enter to finish once you have enough, or it ends after
  `bota.calibration_duration_s`. The fit then runs, the arm holds the pose you left it in and the torque comes back
  on. The fit is stored in the arm's calibration file `<id>.json` under `bota_payload`, next to the motors (the
  file as it was is kept once as `<id>.json.bak`); its mass should match the gripper's weight.
- The recorded trajectory is stored next to it as `<id>_bota_trajectory.csv` (time, sensor temperature, joints,
  wrench). If one exists, the next calibration offers to refit it instead of recording a new one.
- `--robot.bota.residual=true` adds a small net for what depends on the pose and the weight model cannot know, e.g.
  cable pull (inputs: gravity direction in the EE frame and the three wrist angles). It needs 30 orientations, is
  checked on held-out orientations during the fit, and is kept only if it clearly beats the weight model. Its
  weights are stored with the fit. Connecting with it on and a stored fit that has no net runs the calibration
  (unless a net was already tried and rejected), which offers to refit the stored trajectory, so the net can be
  turned on without moving the arm again.
- Every `connect()` re-zeroes the bias for `bota.tare_duration_s`, so keep the arm still and the gripper untouched.
- The observation then carries the gravity-compensated wrench (N, Nm, sensor frame) as `x,y,z,rx,ry,rz.ee_wrench`.
  It stays out of `observation.state` unless the env sets `add_ee_wrench_to_observation`.
- `python -m share.scripts.record_beast` with `--output_dir` writes it per episode to
  `<output_dir>/<primitive>/<episode>/wrench.csv`, next to that episode's dashboard, and logs the peak force.
- To watch it live, run `python -m share.robots.viperx.lerobot_robot_viperx.bota.dashboard --env.type=<env>` with an
  env config that has a ViperX with `bota.enable`. It opens Rerun with the raw and the calibrated wrench, and
  switches the arm torque off (after you confirm) so the arm can be moved by hand.

## Installed Commands

The root package currently exposes:

- `share-record`
- `share-design-constraints`
- `share-train`
- `share-eval`

The root `pyproject.toml` also declares `share-workspace`, but this checkout currently does not contain the `share.scripts.robot_workspace` wrapper module. The workspace functionality is present as Python modules under `share.workspace`, but that CLI wrapper is currently in flux.

## Core Concepts

### Manipulation primitive

A manipulation primitive is one locally coherent behavior with:

- one or more task frames
- processor settings
- an optional policy
- optional notes and task description
- an `is_terminal` flag

### Task frame

A `TaskFrame` defines the control contract for one robot inside a primitive:

- target pose or joint target
- command space: `TASK` or `JOINT`
- per-axis control mode: `POS`, `VEL`, or `WRENCH`
- per-axis policy mode: `ABSOLUTE`, `RELATIVE`, or `None`
- gains and min/max bounds

Only axes whose `policy_mode` is not `None` are learnable by a policy.

### MP-Net

A manipulation-primitive net is a directed graph with:

- `start_primitive`
- `reset_primitive`
- a dictionary of named primitives
- a list of transitions between primitives

The config validates several invariants for you, including:

- transition sources and targets must exist
- non-terminal primitives must have outgoing edges
- terminal primitives must be reachable from `start_primitive`

## Minimal MP-Net Example

This is a small, faithful example of the intended usage pattern: define primitives, define transitions, then save the config as JSON.

```python
from pathlib import Path

from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
from share.envs.manipulation_primitive.task_frame import ControlMode, PolicyMode, TaskFrame
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.transitions import OnSuccess
from share.utils.constants import DEFAULT_ROBOT_NAME
from share.workspace.mpnet import save_mpnet_config, summarize_mpnet

approach = ManipulationPrimitiveConfig(
    task_frame={
        DEFAULT_ROBOT_NAME: TaskFrame(
            target=[0.0, 0.0, 0.15, 0.0, 0.0, 0.0],
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[PolicyMode.RELATIVE, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, None, None],
            min_pose=[-0.10, -0.10, 0.05, -0.5, -0.5, -0.5],
            max_pose=[0.10, 0.10, 0.25, 0.5, 0.5, 0.5],
        )
    },
    notes="Approach the object from above.",
    task_description="approach object",
    is_terminal=False,
)

close_gripper = ManipulationPrimitiveConfig(
    task_frame={
        DEFAULT_ROBOT_NAME: TaskFrame(
            target=[0.0, 0.0, 0.10, 0.0, 0.0, 0.0],
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[None] * 6,
        )
    },
    notes="Close and settle.",
    task_description="close gripper",
    is_terminal=True,
)

mpnet = ManipulationPrimitiveNetConfig(
    start_primitive="approach",
    reset_primitive="approach",
    fps=10,
    robot=None,
    teleop=None,
    primitives={
        "approach": approach,
        "close_gripper": close_gripper,
    },
    transitions=[
        OnSuccess(source="approach", target="close_gripper", additional_reward=1.0),
    ],
)

summary = summarize_mpnet(mpnet)
print(summary["primitives"])

save_mpnet_config(mpnet, Path("mpnets/pick.json"))
```

The key pattern is:

1. define a `TaskFrame` for each robot involved in a primitive
2. wrap those task frames in a `ManipulationPrimitiveConfig`
3. connect primitives with explicit transition objects
4. build a `ManipulationPrimitiveNetConfig`
5. save it as JSON with `save_mpnet_config`

If you want a one-node starter template instead, use:

```python
from share.workspace.mpnet import create_template_mpnet, save_mpnet_config

config = create_template_mpnet("main")
save_mpnet_config(config, "mpnets/template.json")
```

## Using the Runtime Pieces

### Recording

`share-record` runs an MP-Net environment, routes teleop/policy actions through the processor stack, and writes one dataset per adaptive primitive.

### Constraint Design

`share-design-constraints` runs an MP-Net as a live calibration routine and lets you edit adaptive primitive task frames in place. The intended workflow is to teleoperate an adaptive primitive, press `o` to set the current pose as the task-frame origin, move through the desired workspace, and let the tool accumulate `min_pose` / `max_pose` in that frame before saving the updated MP-Net JSON.

At a high level it:

- builds a `ManipulationPrimitiveNet`
- creates one dataset per adaptive primitive
- optionally loads one policy per adaptive primitive
- steps the active primitive
- stores transitions in the corresponding primitive dataset

### Training

`share-train` is a thin adapter around LeRobot training. It builds a train config from a policy checkpoint and a dataset location, then delegates to LeRobot’s training entrypoint.

### Evaluation

`share-eval` is a lightweight dataset-summary pass. It reads recorded primitive datasets and emits a compact summary JSON with counts and available metadata.

## Typical Package Imports

Single-robot usage usually touches:

```python
from share.envs.manipulation_primitive.task_frame import TaskFrame, ControlMode, PolicyMode
from share.envs.manipulation_primitive.config_manipulation_primitive import ManipulationPrimitiveConfig
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
```

If you are using local hardware integrations from this repo checkout, common imports are:

```python
from share.robots.ur import URConfig, UR
from share.teleoperators.delta_keyboard import KeyboardVelocityTeleopConfig, KeyboardVelocityTeleop
```

And if you want the standalone LeRobot-style sibling package imports after installing the corresponding extras:

```python
from lerobot_robot_ur import URConfig, UR
from lerobot_teleoperator_spacemouse import SpacemouseConfig, SpaceMouse
```

## Repository Layout

- `src/share/envs`: manipulation primitives, MP-Nets, task frames, transitions
- `src/share/processor`: observation/action/info processing steps used by the env stack
- `src/share/workspace`: MP-Net persistence, summaries, structured editing helpers, and agent/runtime building blocks
- `src/share/scripts`: recording, training, and evaluation entrypoints
- `src/share/robots`: sibling installable robot packages plus `share` namespace facades
- `src/share/teleoperators`: sibling installable teleoperator packages plus `share` namespace facades
- `tests`: release-readiness, env, policy, and workspace tests

## Important Caveats

- The extras in the root package currently use local sibling-package references. They are ideal for working from this monorepo checkout, but they are not yet the final pattern you would want for publishing all pieces independently on PyPI.
- Some of the sibling standalone packages still import `share.*` internals. That is fine when you install from this repository, but it means the subprojects are not fully decoupled yet.
- The workspace/runtime code is present and useful as a library, but the CLI packaging around it is still being stabilized.
