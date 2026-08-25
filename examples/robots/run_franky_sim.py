"""Standalone franky-sim launcher: run this in one terminal, leave it running,
then point any Franka-backed script at the printed hostname to test against a
local MuJoCo simulation instead of real (or "virtual" 172.16.0.2) hardware.

No real robot is touched by this script under any circumstances -- it only
ever binds a local loopback address (127.0.0.1 by default).

Usage:
    python share-rl/examples/robots/run_franky_sim.py [--visualize]

Then, in another terminal, point a Franka-backed script/env at the printed
hostname. demo_franka_movedelta.py reads FRANKA_ROBOT_IP if set:

    FRANKA_ROBOT_IP=127.0.0.1 python share-rl/src/share/scripts/record.py \\
        --env.type=demo_franka_movedelta

Ctrl+C to stop.
"""
from __future__ import annotations

import argparse
import time

from franky_sim.mujoco_simulator import MujocoSimulator
from franky_sim.simulation_server import SimulationServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize", action="store_true", help="Open the MuJoCo viewer (requires a display)."
    )
    args = parser.parse_args()

    with MujocoSimulator(enable_visualization=args.visualize) as sim:
        robot_model = sim.add_robot()
        with SimulationServer(sim) as server:
            # __enter__ already called server.init() (binds the local FCI server and
            # starts the sim) -- do not call it again here, it would register a
            # second, redundant server for the same robot.
            print(f"franky-sim running at {robot_model.hostname} (local-only, no real robot involved)")
            print(f"  export FRANKA_ROBOT_IP={robot_model.hostname}")
            print("Ctrl+C to stop.")
            try:
                server.run_forever()
            except KeyboardInterrupt:
                print("\nStopping.")


if __name__ == "__main__":
    main()
