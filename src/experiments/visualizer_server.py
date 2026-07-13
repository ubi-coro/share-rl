import sys
import os
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer
import numpy as np
from scipy.spatial.transform import Rotation as R

# Append src folder to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from share.envs.manipulation_primitive.task_frame import (
    ControlMode,
    ControlSpace,
    PolicyMode,
    TaskFrame,
    TASK_FRAME_AXIS_NAMES,
)
from experiments.envs.bimanual_cooperative import (
    SynchronousArmPrimitive,
    DemoURBimanualCooperativeEnvConfig,
)
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
        features = {f"{ax}.ee_pos": float for ax in TASK_FRAME_AXIS_NAMES}
        features["gripper.pos"] = float
        return features

    @property
    def action_features(self) -> dict:
        features = {f"{ax}.ee_pos": float for ax in TASK_FRAME_AXIS_NAMES}
        features["gripper.pos"] = float
        return features

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
        obs = {
            f"{ax}.ee_pos": self.current_pose[i]
            for i, ax in enumerate(TASK_FRAME_AXIS_NAMES)
        }
        obs["gripper.pos"] = 0.0
        return obs

    def set_task_frame(self, task_frame):
        self.task_frame = task_frame

    def step(self):
        pass

# Global state
state_lock = threading.Lock()
latest_data = {
    "left_pos": [0.0, 0.0, 1.0],
    "left_rot": [0.0, 0.0, 0.0],
    "right_pos": [1.0, 0.0, 1.0],
    "right_rot": [0.0, 0.0, 0.0],
    "v_tcp_pos": [0.5, 0.0, 0.5],
    "v_tcp_rot": [0.0, 0.0, 0.0],
    "inputs": [0.0] * 6,
    "active_primitive": "left_arm"
}

# User inputs from keyboard/webpage (dx, dy, dz, drx, dry, drz)
keyboard_inputs = [0.0] * 6
spacebar_pressed = False

def simulation_thread():
    global latest_data, keyboard_inputs, spacebar_pressed
    
    # 1. Setup config and mock robots
    config = DemoURBimanualCooperativeEnvConfig()
    
    left_start = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    right_start = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0] 
    
    left_robot = MockRobot(left_start)
    right_robot = MockRobot(right_start)
    robot_dict = {"left": left_robot, "right": right_robot}
    
    # 2. Build primitives dict from config
    envs = {}
    for name, prim_cfg in config.primitives.items():
        env, _, _ = prim_cfg.make(robot_dict, teleop_dict={}, cameras={}, device="cpu")
        envs[name] = env

    active_primitive = config.start_primitive
    envs[active_primitive].step({"left": {f"{ax}.ee_pos": 0.0 for ax in TASK_FRAME_AXIS_NAMES}})
    
    dt = 1.0 / 30.0
    
    while True:
        # Get latest commands
        with state_lock:
            cmds = list(keyboard_inputs)
            trigger_success = spacebar_pressed
            spacebar_pressed = False # consume the flag
        
        # Evaluate transitions
        if trigger_success:
            next_primitive = None
            for trans in config.transitions:
                if trans.source == active_primitive:
                    next_primitive = trans.target
                    break
            
            if next_primitive:
                print(f"\n[TRANSITION] Primitive switched: {active_primitive} -> {next_primitive}")
                active_primitive = next_primitive
                
                # Re-initialize the next primitive state to prevent jumps
                envs[active_primitive]._initialized = False
                init_action = {"left": {f"{ax}.ee_pos": 0.0 for ax in TASK_FRAME_AXIS_NAMES}}
                envs[active_primitive].step(init_action)
        
        # Map actions according to active mode
        action = {}
        if active_primitive == "left_arm":
            action = {
                "left": {
                    "x.ee_pos": cmds[0] * 0.05,
                    "y.ee_pos": cmds[1] * 0.05,
                    "z.ee_pos": cmds[2] * 0.2,
                    "rx.ee_pos": cmds[3] * 0.3,
                    "ry.ee_pos": cmds[4] * 0.3,
                    "rz.ee_pos": cmds[5] * 0.3
                }
            }
        elif active_primitive == "right_arm":
            action = {
                "right": {
                    "x.ee_pos": cmds[0] * 0.05,
                    "y.ee_pos": cmds[1] * 0.05,
                    "z.ee_pos": cmds[2] * 0.2,
                    "rx.ee_pos": cmds[3] * 0.3,
                    "ry.ee_pos": cmds[4] * 0.3,
                    "rz.ee_pos": cmds[5] * 0.3
                }
            }
        elif active_primitive in ["cooperative_translation", "cooperative_rotation"]:
            action = {
                "left": {
                    "x.ee_pos": cmds[0] * 0.05,
                    "y.ee_pos": cmds[1] * 0.05,
                    "z.ee_pos": cmds[2] * 0.2,
                    "rx.ee_pos": cmds[3] * 0.3,
                    "ry.ee_pos": cmds[4] * 0.3,
                    "rz.ee_pos": cmds[5] * 0.3
                }
            }
            
        # Step active environment primitive
        envs[active_primitive].step(action)
        
        # Integrate poses based on active commands received by robots
        for name, robot in robot_dict.items():
            if robot.last_action is not None:
                for i, ax in enumerate(TASK_FRAME_AXIS_NAMES):
                    cmd_val = robot.last_action.get(f"{ax}.ee_pos", 0.0)
                    robot.current_pose[i] += cmd_val * dt
                robot.last_action = None
                
        # Update shared state
        with state_lock:
            left_pos = list(left_robot.current_pose[:3])
            left_rot = list(R.from_rotvec(left_robot.current_pose[3:6]).as_euler("xyz", degrees=True))
            right_pos = [right_robot.current_pose[0] + 1.0, right_robot.current_pose[1], right_robot.current_pose[2]]
            right_rot = list(R.from_rotvec(right_robot.current_pose[3:6]).as_euler("xyz", degrees=True))
            
            active_env = envs[active_primitive]
            if hasattr(active_env, "_T_world_v_tcp") and active_env._T_world_v_tcp is not None:
                v_tcp_pos = list(active_env._T_world_v_tcp[:3, 3])
                v_tcp_rot = list(R.from_matrix(active_env._T_world_v_tcp[:3, :3]).as_euler("xyz", degrees=True))
            else:
                v_tcp_pos = [0.5 * (left_pos[0] + right_pos[0]), 0.5 * (left_pos[1] + right_pos[1]), 0.5 * (left_pos[2] + right_pos[2]) - 0.5]
                v_tcp_rot = [0.0, 0.0, 0.0]
                
            latest_data = {
                "left_pos": left_pos,
                "left_rot": left_rot,
                "right_pos": right_pos,
                "right_rot": right_rot,
                "v_tcp_pos": v_tcp_pos,
                "v_tcp_rot": v_tcp_rot,
                "inputs": cmds,
                "active_primitive": active_primitive
            }
            
        time.sleep(dt)

class VisualizerHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            html_path = os.path.join(os.path.dirname(__file__), 'index.html')
            with open(html_path, 'rb') as f:
                self.wfile.write(f.read())
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            try:
                while True:
                    with state_lock:
                        data = json.dumps(latest_data)
                    self.wfile.write(f"data: {data}\n\n".encode('utf-8'))
                    self.wfile.flush()
                    time.sleep(1/30.0)
            except (ConnectionResetError, BrokenPipeError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/control':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                cmds = json.loads(post_data.decode('utf-8'))
                global keyboard_inputs, spacebar_pressed
                with state_lock:
                    if cmds.get("success", False):
                        spacebar_pressed = True
                    else:
                        keyboard_inputs = [
                            float(cmds.get("dx", 0.0)),
                            float(cmds.get("dy", 0.0)),
                            float(cmds.get("dz", 0.0)),
                            float(cmds.get("drx", 0.0)),
                            float(cmds.get("dry", 0.0)),
                            float(cmds.get("drz", 0.0))
                        ]
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(str(e).encode())

def run_server():
    server_address = ('', 8000)
    ThreadingTCPServer.allow_reuse_address = True
    server = ThreadingTCPServer(server_address, VisualizerHTTPHandler)
    print("-----------------------------------------------------------------")
    print("3D Bimanual Cooperative Visualizer Server Running at:")
    print("       http://localhost:8000/")
    print("-----------------------------------------------------------------")
    print("Controls in browser:")
    print("  Cycle Modes (State transition): Spacebar")
    print("  Translate (active): WASD (X/Y), Space/Shift (Z)")
    print("  Rotate (active):    Q/E (Roll), ArrowUp/ArrowDown (Pitch), ArrowLeft/ArrowRight (Yaw)")
    print("-----------------------------------------------------------------")
    
    t = threading.Thread(target=simulation_thread, daemon=True)
    t.start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()

if __name__ == '__main__':
    run_server()
