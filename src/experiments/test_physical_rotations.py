import time
import sys
import os
import numpy as np

# Append src folder to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.envs.bimanual_cooperative import DemoURBimanualCooperativeEnvConfig
from share.envs.manipulation_primitive.task_frame import TASK_FRAME_AXIS_NAMES

def countdown(seconds, message):
    print(f"\n{message}")
    for i in range(seconds, 0, -1):
        print(f"Starting in {i} seconds... (Make sure area is clear! Press Ctrl+C to abort)")
        time.sleep(1.0)

def main():
    print("--- Cooperating physical arm rotations validation script ---")
    print("Initializing environment and connecting to the robots...")
    config = DemoURBimanualCooperativeEnvConfig()
    net = config.make()
    coop_env = net._envs["cooperative"]
    
    # Run active reset first to initialize the robots
    net.reset()
    
    print("\nRobots connected and initialized.")
    print("The robots are currently holding onto the rigid object.")
    
    # Define steps to execute: 20 degrees = 0.35 radians
    # At action scale of 0.3, a full joystick deflection of 1.0 (or command 1.0)
    # means 0.3 rad/s command.
    # So we need to command 0.3 rad/s for:
    # duration = 0.35 / 0.3 = 1.16 seconds.
    # At 30 fps, this is: 1.16 * 30 = 35 steps.
    
    tests = [
        ("ry.ee_pos", "20-degree Pitch (tilt up then down)"),
        ("rz.ee_pos", "20-degree Yaw (steer left then right)"),
        ("rx.ee_pos", "20-degree Roll (spin about rod axis)")
    ]
    
    for axis, name in tests:
        # Countdown before each test
        countdown(5, f"=== NEXT TEST: {name} ===")
        
        # 1. Rotate in positive direction
        print("Rotating positive...")
        for step in range(35):
            action = {"left": {f"{ax}.ee_pos": 0.0 for ax in TASK_FRAME_AXIS_NAMES}}
            action["left"][axis] = 1.0 # command full speed (0.3 rad/s equivalent)
            coop_env.step(action)
            time.sleep(1.0 / 30.0)
            
        # 2. Pause
        print("Pausing...")
        for step in range(15):
            action = {"left": {f"{ax}.ee_pos": 0.0 for ax in TASK_FRAME_AXIS_NAMES}}
            coop_env.step(action)
            time.sleep(1.0 / 30.0)
            
        # 3. Rotate back in negative direction
        print("Rotating negative...")
        for step in range(35):
            action = {"left": {f"{ax}.ee_pos": 0.0 for ax in TASK_FRAME_AXIS_NAMES}}
            action["left"][axis] = -1.0
            coop_env.step(action)
            time.sleep(1.0 / 30.0)
            
        # 4. Zero out commands to stop completely
        print("Stopping...")
        for step in range(10):
            action = {"left": {f"{ax}.ee_pos": 0.0 for ax in TASK_FRAME_AXIS_NAMES}}
            coop_env.step(action)
            time.sleep(1.0 / 30.0)
            
        print("Test finished.")
        time.sleep(1.5)

    print("\nAll rotation tests finished. Disconnecting from robots...")
    net.close()

if __name__ == "__main__":
    main()
