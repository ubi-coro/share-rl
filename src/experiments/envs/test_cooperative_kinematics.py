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

def run_test():
    # 1. Setup mock robots
    # Left robot starting at [0, 0, 1.0], right robot starting at [0, 0, 1.0] in right base frame
    # (Note: right base is offset by x=1.0, so right EE is at world [1.0, 0, 1.0])
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
    
    print("\n--- STEP 1: INITIALIZATION (0 INPUT) ---")
    action = {
        "left": {f"{ax}.ee_pos": 0.0 for ax in TASK_FRAME_AXIS_NAMES}
    }
    
    env.step(action)
    print("V-TCP Target position:", env._T_world_v_tcp[:3, 3])
    print("Left command:", [f"{k}: {v:.4f}" for k, v in left_robot.last_action.items() if "ee_pos" in k])
    print("Right command:", [f"{k}: {v:.4f}" for k, v in right_robot.last_action.items() if "ee_pos" in k])
    
    # Update mock robot positions to simulate movement
    dt = 1.0 / 30.0
    # Left and right both run in RELATIVE controller mode due to synchronous arm init overrides:
    for i, ax in enumerate(TASK_FRAME_AXIS_NAMES):
        left_robot.current_pose[i] += left_robot.last_action[f"{ax}.ee_pos"] * dt
        right_robot.current_pose[i] += right_robot.last_action[f"{ax}.ee_pos"] * dt

    print("\n--- STEP 2: YAW ROTATION COMMAND (rz = 0.3 rad/s) ---")
    action = {
        "left": {f"{ax}.ee_pos": 0.0 for ax in TASK_FRAME_AXIS_NAMES}
    }
    action["left"]["rz.ee_pos"] = 0.3  # Command yaw rotation
    
    env.step(action)
    print("Left command (should show non-zero y translation):")
    print("  x.ee_pos:", f"{left_robot.last_action['x.ee_pos']:.4f}")
    print("  y.ee_pos:", f"{left_robot.last_action['y.ee_pos']:.4f}")
    print("  z.ee_pos:", f"{left_robot.last_action['z.ee_pos']:.4f}")
    print("Right command (should show opposite y translation):")
    print("  x.ee_pos:", f"{right_robot.last_action['x.ee_pos']:.4f}")
    print("  y.ee_pos:", f"{right_robot.last_action['y.ee_pos']:.4f}")
    print("  z.ee_pos:", f"{right_robot.last_action['z.ee_pos']:.4f}")
    
    # Update mock robot positions
    for i, ax in enumerate(TASK_FRAME_AXIS_NAMES):
        left_robot.current_pose[i] += left_robot.last_action[f"{ax}.ee_pos"] * dt
        right_robot.current_pose[i] += right_robot.last_action[f"{ax}.ee_pos"] * dt

    print("\n--- STEP 3: PITCH ROTATION COMMAND (ry = 0.3 rad/s) ---")
    action = {
        "left": {f"{ax}.ee_pos": 0.0 for ax in TASK_FRAME_AXIS_NAMES}
    }
    action["left"]["ry.ee_pos"] = 0.3  # Command pitch rotation
    
    env.step(action)
    print("Left command (should translate down in Z):")
    print("  z.ee_pos:", f"{left_robot.last_action['z.ee_pos']:.4f}")
    print("Right command (should translate up in Z):")
    print("  z.ee_pos:", f"{right_robot.last_action['z.ee_pos']:.4f}")
    
    # Update mock robot positions
    for i, ax in enumerate(TASK_FRAME_AXIS_NAMES):
        left_robot.current_pose[i] += left_robot.last_action[f"{ax}.ee_pos"] * dt
        right_robot.current_pose[i] += right_robot.last_action[f"{ax}.ee_pos"] * dt

    print("\n--- STEP 4: RELEASE JOYSTICK (0 INPUT) ---")
    action = {
        "left": {f"{ax}.ee_pos": 0.0 for ax in TASK_FRAME_AXIS_NAMES}
    }
    env.step(action)
    print("Left command (should be exactly 0):")
    print("  x.ee_pos:", f"{left_robot.last_action['x.ee_pos']:.4f}")
    print("  y.ee_pos:", f"{left_robot.last_action['y.ee_pos']:.4f}")
    print("  z.ee_pos:", f"{left_robot.last_action['z.ee_pos']:.4f}")
    print("Right command (should be exactly 0):")
    print("  x.ee_pos:", f"{right_robot.last_action['x.ee_pos']:.4f}")
    print("  y.ee_pos:", f"{right_robot.last_action['y.ee_pos']:.4f}")
    print("  z.ee_pos:", f"{right_robot.last_action['z.ee_pos']:.4f}")

if __name__ == "__main__":
    run_test()
