Franka Research 3
=================

Overview
--------

The SHARE Franka backend is ROS-free. The application-facing Franka object
exchanges complete commands and state through shared memory with a dedicated
500 Hz Python process. That process integrates task-frame commands into a
target pose (or joint reference) -- relative axes as a velocity, including
proper SO(3) composition for rotation, absolute axes imposed directly,
workspace bounds clamped -- and streams the result into one of Franky 2's
native motions, which owns the actual impedance control law:

- Task-space (position-only) streams into ``CartesianImpedanceTrackingMotion``
  via ``set_reference(CartesianReference(...))``/``set_gains(...)`` every
  tick. Nullspace posture is Franky's own ``PostureTask``, not a hand-rolled
  Jacobian projection.
- Joint-space streams into ``JointImpedanceTrackingMotion`` via
  ``set_reference(JointReference(q=...))``, unchanged from before.

Franky and libfranka run that impedance law, Coriolis compensation,
torque-rate limiting, the torque watchdog, and soft joint-limit repulsion on
their own real-time thread -- the Python bridge never computes torque or a
wrench itself.

The optional Franka Hand runs in a different process because homing, moves,
and grasps may block. No ROS node, topic, service, or ROS environment leaks
into the LeRobot process.

What Desk and FCI are
---------------------

**Desk** is the robot's browser-based operator interface. Open the robot's IP
address from a control PC on the robot network. Desk is where an operator takes
control, unlocks the brakes, acknowledges errors, configures the end-effector
and load, and activates the Franka Control Interface (FCI). Desk is not a ROS
application and does not execute the SHARE controller.

**FCI** is Franka's low-level external control interface. Libfranka connects
to the robot controller over Ethernet and must exchange packets at 1 kHz. On
robot software 4.2 and newer, FCI must be explicitly activated in Desk before
a client connects. A released brake is not sufficient by itself.

Do not start with motion
------------------------

Commission the robot and validate the link before running SHARE:

1. Read the FR3 safety manual, establish an accessible emergency-stop path,
   clear the workspace, and keep the operator at the enabling device.
2. Connect the control PC by wired Ethernet to the robot control network. Give
   the PC a static address in the robot's subnet. Do not route the FCI stream
   over Wi-Fi, a VPN, a USB Ethernet adapter, or a busy office switch.
3. Confirm that the robot IP responds, open Desk, take control, release the
   brakes, and activate FCI.
4. Install a libfranka build compatible with the robot's server version. Run
   libfranka's communication_test ROBOT_IP until it reports a stable 1 kHz
   connection and a high communication success rate.
5. Only then run idle SHARE state streaming and the staged checks below.

The authoritative compatibility and commissioning instructions are at
https://frankarobotics.github.io/docs/.

Franky 2 installation
---------------------

The backend accepts only Franky 2.x. Franky 2 is MIT licensed; Franky 1.x is
LGPL-3.0 and is deliberately rejected rather than silently installed. Franky
is an external dependency and no Franky, RCS, or HIL-SERL controller source is
vendored into SHARE.

A Python wheel embeds or links a particular libfranka protocol implementation.
Use the robot-server-version wheel index documented by Franky, replacing the
placeholder with the server version reported for the robot:

.. code-block:: bash

   pip install "franky-control>=2,<3" \
     --extra-index-url "https://timschneider42.github.io/franky/whl/by-robot-server-version/ROBOT_SERVER_VERSION/"
   pip install -e ".[franka]"

Do not fix a protocol mismatch by trying arbitrary wheel versions on a live
robot. Match the compatibility table, then repeat communication_test.

Realtime host setup
-------------------

The reference host is Ubuntu 24.04 with PREEMPT_RT. Follow Franka's
realtime-kernel guide at
https://frankarobotics.github.io/docs/doc/libfranka/docs/real_time_kernel.html
for the exact kernel and distribution. After reboot, verify that uname -a
identifies the realtime kernel.

The user running SHARE needs permission to create realtime threads and lock
memory. A typical deployment uses a dedicated realtime group and a limits file
granting rtprio 99 and memlock unlimited. Log out and in after changing group
membership, then verify the effective limits with ulimit -r and ulimit -l.
Keep Franky's default RealtimeConfig.Enforce; set enforce_realtime=False only
for an explicit connectivity diagnostic, never for contact work.

CUDA and PREEMPT_RT are not inherently incompatible. GPU drivers, power
management, displays, storage, and network interrupt placement can introduce
latency on a particular machine. Validate the whole host with cyclic latency
tests and a sustained FCI run under the expected GPU workload. Prefer a
separate control PC if GPU load produces deadline or communication failures.

Load and end-effector data
--------------------------

External wrench estimates depend directly on the configured end-effector and
payload. Set payload_mass, payload_center_of_mass, and payload_inertia
together; inertia is a row-major 3 by 3 matrix. Set end_effector_transform when
the configured flange-to-EE transform differs from the robot setup. Keep Desk
and application configuration consistent.

zero_ft() is software bias capture, not sensor calibration. Call it only while
the arm is stationary and unloaded by contact. It subtracts the current
base-frame external-wrench estimate before SHARE transforms the wrench into
the active task frame.

Controller behavior
-------------------

The public pose convention is xyz plus extrinsic XYZ roll, pitch, yaw. The
controller converts rotations to matrices and uses an SO(3) logarithm where it
needs an orientation error (the reference-error clamp below); target poses
handed to Franky are always absolute, in the base frame.

Task-space control is position-only: Franky's native Cartesian impedance
motion owns the wrench law, so there is no SHARE-owned VEL or WRENCH axis to
send it, and ``FrankaTaskFrameCommand`` rejects anything else at construction.
Relative POS axes are a velocity integrated at 500 Hz -- translation directly,
rotation via SO(3) composition so a mix of relative axes still yields a proper
3D rotation. Absolute POS axes are then imposed directly. A reference-error
clamp limits how far the stored (virtual) target may run ahead of the measured
pose, to ``force_constraints / stiffness`` per axis, when
``compliance_reference_limit_enable`` is set for that axis -- pure anti-windup,
computed entirely on the Python side. Workspace violations clip the target
pose directly into the configured box; rotational bounds support linear and
ccw_arc intervals. There is no adaptive, contact-reactive wrench scaling
anymore -- that required owning the wrench law, which Franky does now.

``translational_stiffness``/``rotational_stiffness`` are scalar (Franky's
Cartesian impedance is isotropic per axis group, not six independent gains)
and live-updatable per command -- ``CartesianImpedanceTrackingMotion`` smooths
``set_gains`` changes itself via ``gains_time_constant``. ``force_constraints``
and the nullspace ``PostureTask`` (``nullspace_stiffness``/
``nullspace_max_torque``) are fixed for the whole connection on
``FrankaConfig``, not per command: Franky fixes both at motion-construction
time with no live setter, and the posture target is the joint configuration
captured the moment task-space control starts.

Custom Python strategies subclass and register FrankaControllerConfig, then
implement make_strategy(). The returned strategy receives a NumPy-only
FrankaState, the latest complete command snapshot (or None while stale), and
dt on every 500 Hz tick, and returns a ReferenceOutput -- the target pose
Franky should track next, plus state-publication diagnostics. It never
returns torque or a wrench; Franky's motion computes that from the pose it's
handed.

Every command is a complete atomic snapshot. If its monotonic timestamp is
older than 250 ms, the controller captures the measured pose and holds it with
conservative gains. A fresh complete command resumes control. Parent death,
Franky watchdog expiry, FCI errors, and strategy exceptions trigger a
TorqueStopMotion and are raised by the Franka wrapper -- ``CartesianImpedanceTrackingMotion``
and ``JointImpedanceTrackingMotion`` are both client-side torque motions
underneath, so the same graceful-stop path covers either.

Joint-space control is otherwise unchanged and untouched by the above: it
still streams a clamped joint reference into ``JointImpedanceTrackingMotion``
and is configured by the separate ``joint_stiffness``/``joint_damping``/
``joint_error_clip`` fields.

First motion and acceptance
---------------------------

Run examples/robots/franka_first_motion.py only after replacing the IP,
checking its three-centimeter workspace against the physical setup, and
reviewing the gains. It streams a 1 cm/s reference for half a second and then
commands zero relative velocity.

Use this progression for each new robot and host combination:

1. Pass official communication_test and stream idle SHARE state.
2. Capture wrench bias and physically verify all force and torque signs.
3. Test low-gain free-space translation, then rotation, one axis at a time.
4. Test nullspace posture (PostureTask) and soft joint-limit behavior away
   from hard limits.
5. Introduce controlled contact and validate the reference-error clamp
   (compliance_reference_limit_enable) and force_constraints.
6. Stop command publication, kill the parent process, and inject a controller
   failure to verify hold, error propagation, and torque stop.
7. Record 500 Hz loop durations and FCI communication success during a
   sustained run under representative camera and GPU load.

The impedance control law itself already runs on Franky's real-time thread,
not in this Python bridge, so there is no further promotion step for it. If
step 7 shows meaningful deadline misses, the suspect is the bridge's own
per-tick work (state read, pose integration, the ``set_reference``/
``set_gains`` calls) or host latency (see Realtime host setup), not the
control law.
