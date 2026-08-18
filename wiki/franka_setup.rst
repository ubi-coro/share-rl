Franka Research 3
=================

Overview
--------

The SHARE Franka backend is ROS-free. The application-facing Franka object
exchanges complete commands and state through shared memory with a dedicated
500 Hz Python process. That process evaluates the mixed-axis SHARE controller
and sends seven torques to Franky 2's SimpleTorqueMotion. Franky and libfranka
own the 1 kHz FCI connection, Coriolis compensation, torque-rate limiting, the
torque watchdog, and soft joint-limit repulsion.

Joint-space commands use Franky's native JointImpedanceTrackingMotion. The
optional Franka Hand runs in a different process because homing, moves, and
grasps may block. No ROS node, topic, service, or ROS environment leaks into
the LeRobot process.

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
controller converts rotations to matrices and uses an SO(3) logarithm for
impedance error. Wrenches are shifted between the stiffness point, task-frame
origin, and EE Jacobian point with the corresponding force/moment lever arm.

A task command may mix position, velocity, and wrench axes. Relative position
targets are velocities integrated at 500 Hz. Reference limiting clamps stored
position error to wrench_limits / kp. Adaptive limiting exponentially shrinks
the final wrench budget only when measured contact opposes the command.
Workspace violations suppress outward wrench and add an inward spring.
Rotational bounds support linear and ccw_arc intervals.

Custom Python controllers subclass and register FrankaControllerConfig, then
implement make_strategy(). The returned strategy receives a NumPy-only
FrankaState, the latest complete command snapshot (or None while stale), the
Franky model handle, and dt on every 500 Hz tick. Its step method may return a
finite seven-element torque array directly. The built-in adaptive strategy
returns the same torques together with state-publication diagnostics. This is
also the boundary intended for a future pybind-backed C++ strategy.


Every command is a complete atomic snapshot. If its monotonic timestamp is
older than 250 ms, the controller captures the measured pose and holds it with
conservative gains. A fresh complete command resumes control. Parent death,
Franky watchdog expiry, FCI errors, and strategy exceptions trigger a
TorqueStopMotion and are raised by the Franka wrapper.

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
4. Test nullspace posture and soft joint-limit behavior away from hard limits.
5. Introduce controlled contact and validate reference/adaptive limiting.
6. Stop command publication, kill the parent process, and inject a controller
   failure to verify hold, error propagation, and torque stop.
7. Record 500 Hz loop durations and FCI communication success during a
   sustained run under representative camera and GPU load.

Promote the controller to a native Franky C++ Motion only if these measurements
show meaningful 500 Hz deadline misses or unacceptable contact-loop stiffness
or damping.
