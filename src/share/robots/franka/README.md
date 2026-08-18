# Franka Research 3 backend

This optional backend exposes an FR3 through the same TaskFrame API as the
SHARE UR backend. It does not use ROS. A Python process evaluates the SHARE
controller at 500 Hz while Franky 2 runs the libfranka torque motion and safety
checks at 1 kHz.

Install the Franky 2 wheel matching the robot's reported server version before
installing the extra:

    pip install "franky-control>=2,<3" \
      --extra-index-url "https://timschneider42.github.io/franky/whl/by-robot-server-version/ROBOT_SERVER_VERSION/"
    pip install -e ".[franka]"

Franky 2 is MIT licensed. Franky 1 is LGPL-3.0 and is intentionally rejected at
runtime. Franky is imported only in worker processes, so importing SHARE or
using MockFranka does not require it.

See wiki/franka_setup.rst for hardware preparation, realtime configuration,
payload setup, the safety model, and staged acceptance.
