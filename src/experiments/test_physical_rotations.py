import time
import sys
import os
import numpy as np

# Append src folder to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.envs.bimanual_cooperative import DemoURBimanualCooperativeEnvConfig
from share.envs.manipulation_primitive.task_frame import TASK_FRAME_AXIS_NAMES

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
    
    while True:
        print("\nSelect a rotation to test:")
        print("1. 20-degree Pitch (tilt up then down)")
        print("2. 20-degree Yaw (steer left then right)")
        print("3. 20-degree Roll (spin about rod axis)")
        print("4. Exit")
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '4':
            break
            
        axis = None
        if choice == '1':
            axis = "ry.ee_pos"
            name = "Pitch"
        elif choice == '2':
            axis = "rz.ee_pos"
            name = "Yaw"
        elif choice == '3':
            axis = "rx.ee_pos"
            name = "Roll"
        else:
            print("Invalid choice.")
            continue
            
        print(f"\nReady to execute {name} rotation test.")
        print("The arms will tilt in one direction by 20 degrees, pause, and tilt back.")
        input("Press Enter to START (Make sure area is clear!)...")
        
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
            
        print("\nRotation step test finished.")

    print("\nDisconnecting from robots...")
    net.close()

if __name__ == "__main__":
    main()
