import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os

# Append src directory to system path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    PolicyMode,
    TaskFrame,
    TASK_FRAME_AXIS_NAMES,
)
from experiments.envs.bimanual_cooperative import SynchronousArmPrimitive
from lerobot.robots import Robot, RobotConfig

class MockRobot(Robot):
    config_class = RobotConfig
    name = "mock_robot"

    def __init__(self, current_pose):
        cfg = RobotConfig(id="mock_robot")
        super().__init__(cfg)
        self.current_pose = list(current_pose)
        self.task_frame = None
        self.last_action = None

    @property
    def observation_features(self) -> dict:
        return {f"{ax}.ee_pos": float for ax in TASK_FRAME_AXIS_NAMES}

    @property
    def action_features(self) -> dict:
        return {f"{ax}.ee_pos": float for ax in TASK_FRAME_AXIS_NAMES}

    @property
    def _motors_ft(self) -> dict:
        return self.observation_features

    @property
    def is_connected(self) -> bool: return True
    @property
    def is_calibrated(self) -> bool: return True

    def connect(self, calibrate: bool = True): pass
    def disconnect(self): pass
    def calibrate(self): pass
    def configure(self): pass

    def send_action(self, action):
        self.last_action = action
        return action

    def get_observation(self):
        return {
            f"{ax}.ee_pos": self.current_pose[i]
            for i, ax in enumerate(TASK_FRAME_AXIS_NAMES)
        }

    def set_task_frame(self, task_frame):
        self.task_frame = task_frame

    def step(self):
        pass

def print_poses(step_idx, left_robot, right_robot):
    left_z = left_robot.current_pose[2]
    left_y = left_robot.current_pose[1]
    left_rx, left_ry, left_rz = R.from_rotvec(left_robot.current_pose[3:6]).as_euler("xyz", degrees=True)

    # Right base is offset by X=1.0m, so we get right pose in left base (world) coordinate frame:
    right_z = right_robot.current_pose[2]
    right_y = right_robot.current_pose[1]
    right_rx, right_ry, right_rz = R.from_rotvec(right_robot.current_pose[3:6]).as_euler("xyz", degrees=True)

    print(
        f"Step {step_idx:02d} | "
        f"Left (Z={left_z:.4f}, Y={left_y:.4f}, Ry={left_ry:6.1f}°, Rz={left_rz:6.1f}°) | "
        f"Right (Z={right_z:.4f}, Y={right_y:.4f}, Ry={right_ry:6.1f}°, Rz={right_rz:6.1f}°)"
    )

def run_test():
    # 1. Setup mock robots
    # Both start at local [0, 0, 1.0].
    # Left base is at X=0, so left EE is at world [0.0, 0.0, 1.0].
    # Right base is at X=1.0, so right EE is at world [1.0, 0.0, 1.0].
    left_start = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    right_start = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] 
    
    left_robot = MockRobot(left_start)
    right_robot = MockRobot(right_start)
    
    robot_dict = {
        "left": left_robot,
        "right": right_robot
    }
    
    task_frame = {
        "left": TaskFrame(
            target=[0.0] * 6,
            space=ControlSpace.TASK,
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[PolicyMode.RELATIVE] * 6,
        ),
        "right": TaskFrame(
            target=[0.0] * 6,
            space=ControlSpace.TASK,
            control_mode=[ControlMode.POS] * 6,
            policy_mode=[None] * 6,
        )
    }
    
    # Instantiate SynchronousArmPrimitive
    # Offset of center of rotation: 50 cm below midpoint along Z
    env = SynchronousArmPrimitive(
        task_frame=task_frame,
        robot_dict=robot_dict,
        cameras={},
        right_arm_base_pose_in_left_base=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        v_tcp_offset_in_midpoint=[0.0, 0.0, -0.5],
        fps=30.0
    )
    
    dt = 1.0 / 30.0

    print("\n=== STEP 1: INITIALIZATION ===")
    action = {"left": {f"{ax}.ee_pos": 0.0 for ax in TASK_FRAME_AXIS_NAMES}}
    env.step(action)
    print_poses(0, left_robot, right_robot)
    
    # Update poses (should remain identical to start)
    for i, ax in enumerate(TASK_FRAME_AXIS_NAMES):
        left_robot.current_pose[i] += left_robot.last_action[f"{ax}.ee_pos"] * dt
        right_robot.current_pose[i] += right_robot.last_action[f"{ax}.ee_pos"] * dt

    print("\n=== STEP 2: MULTI-STEP PITCH TRAJECTORY (ry = 0.3 rad/s) ===")
    print("Commanding constant positive Pitch command. One arm should go UP, the other DOWN, and wrists should pitch.")
    for step in range(1, 16):
        action = {"left": {f"{ax}.ee_pos": 0.0 for ax in TASK_FRAME_AXIS_NAMES}}
        action["left"]["ry.ee_pos"] = 0.3
        env.step(action)
        
        # Integrate robot poses
        for i, ax in enumerate(TASK_FRAME_AXIS_NAMES):
            left_robot.current_pose[i] += left_robot.last_action[f"{ax}.ee_pos"] * dt
            right_robot.current_pose[i] += right_robot.last_action[f"{ax}.ee_pos"] * dt
            
        print_poses(step, left_robot, right_robot)

    # Let's reset the robots to start position before the Yaw test
    left_robot.current_pose = list(left_start)
    right_robot.current_pose = list(right_start)
    env._initialized = False # force V-TCP re-initialization

    print("\n=== STEP 3: INITIALIZATION ===")
    action = {"left": {f"{ax}.ee_pos": 0.0 for ax in TASK_FRAME_AXIS_NAMES}}
    env.step(action)
    print_poses(0, left_robot, right_robot)
    for i, ax in enumerate(TASK_FRAME_AXIS_NAMES):
        left_robot.current_pose[i] += left_robot.last_action[f"{ax}.ee_pos"] * dt
        right_robot.current_pose[i] += right_robot.last_action[f"{ax}.ee_pos"] * dt

    print("\n=== STEP 4: MULTI-STEP YAW TRAJECTORY (rz = 0.3 rad/s) ===")
    print("Commanding constant positive Yaw command. One arm should go FORWARD (+Y), the other BACKWARD (-Y), and wrists should yaw.")
    for step in range(1, 16):
        action = {"left": {f"{ax}.ee_pos": 0.0 for ax in TASK_FRAME_AXIS_NAMES}}
        action["left"]["rz.ee_pos"] = 0.3
        env.step(action)
        
        # Integrate robot poses
        for i, ax in enumerate(TASK_FRAME_AXIS_NAMES):
            left_robot.current_pose[i] += left_robot.last_action[f"{ax}.ee_pos"] * dt
            right_robot.current_pose[i] += right_robot.last_action[f"{ax}.ee_pos"] * dt
            
        print_poses(step, left_robot, right_robot)

if __name__ == "__main__":
    run_test()
