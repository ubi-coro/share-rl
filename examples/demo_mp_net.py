import torch
import numpy as np
from typing import Any

from share.envs.manipulation_primitive.task_frame import TaskFrame, PolicyMode, ControlMode
from share.envs.manipulation_primitive.config_manipulation_primitive import (
    ManipulationPrimitiveConfig,
    MoveDeltaPrimitiveConfig,
    OpenLoopTrajectorySpec,
    OpenLoopTrajectoryPrimitiveConfig,
)
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import ManipulationPrimitiveNetConfig
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.envs.manipulation_primitive_net.transitions import (
    Always,
    OnObservationThreshold,
    OnSuccess,
    OnTargetPoseReached,
)
from share.utils.mock_utils import MockRobot, MockKinematicsSolver, MockTeleoperator


class RandomObservationRobot(MockRobot):
    """
    Subclass of MockRobot that provides stochastic data for transitions.
    Note: We provide '_motors_ft' as a property mapping to ensure compatibility
    with the ManipulationPrimitive initialization logic.
    """
    def __init__(self, name="random_bot"):
        super().__init__(name=name, is_task_frame=True)
        self.current_joints = np.zeros(6)

    @property
    def _motors_ft(self):
        # Maps internal LeRobot feature expectations to this mock
        return {f"joint_{i+1}.pos": float for i in range(6)}

    def get_observation(self) -> dict[str, Any]:
        # Return random poses centered around current state
        obs = {f"joint_{i+1}.pos": float(self.current_joints[i] + np.random.normal(0, 1.0)) for i in range(6)}
        # Simulate End-Effector Cartesian positions for transition thresholding
        for i, ax in enumerate(["x", "y", "z", "wx", "wy", "wz"]):
            obs[f"{ax}.ee_pos"] = float(self.current_joints[i] + np.random.normal(0, 1.0))
        return obs


# Initialize shared mock infrastructure
solver = MockKinematicsSolver()
robot = RandomObservationRobot("mock_robot")
teleop = MockTeleoperator("mock_robot", is_delta=True)


# 1. Search (Start State)
search_cfg = MoveDeltaPrimitiveConfig(
    task_frame={"random_bot": TaskFrame(
        target=[0.4, 0.0, 0.4, 0, 0, 0],
        policy_mode=[PolicyMode.RELATIVE] * 3 + [None] * 3,
        control_mode=[ControlMode.POS] * 6
    )},
    delta={"random_bot": [0.05, 0.0, 0.0, 0.0, 0.0, 0.0]},
)

# 2. Final Stage (Terminal)
final_success_cfg = ManipulationPrimitiveConfig(
    task_frame={"random_bot": TaskFrame(target=[0.5, 0.5, 0.5, 0, 0, 0])},
    is_terminal=True
)

# 3. Reset Step 1: Retract
retract_cfg = OpenLoopTrajectoryPrimitiveConfig(
    task_frame={"random_bot": TaskFrame(target=[0.4, 0.0, 0.6, 0, 0, 0])},
    trajectory=OpenLoopTrajectorySpec(
        delta={"random_bot": [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]},
        frame={"random_bot": "world"},
        duration_s={"random_bot": 0.6},
    ),
)

# 4. Reset Step 2: Home
home_cfg = ManipulationPrimitiveConfig(
    task_frame={"random_bot": TaskFrame(target=[0.0, 0.0, 0.0, 0, 0, 0])},
)

# Define the Net with Diverse Transitions
net_config = ManipulationPrimitiveNetConfig(
    start_primitive="home",
    reset_primitive="retract",
    primitives={
        "search": search_cfg,
        "final_stage": final_success_cfg,
        "retract": retract_cfg,
        "home": home_cfg,
    },
    transitions=[
        OnObservationThreshold(
            source="search",
            target="final_stage",
            obs_key="random_bot.x.ee_pos",
            threshold=0.45,
            operator="ge"
        ),
        Always(source="final_stage", target="retract"),
        OnSuccess(source="retract", target="home", success_key="primitive_complete"),
        OnTargetPoseReached(source="home", target="search", robot_name="random_bot", axes=["x"], tolerance=0.05),
    ],
)


# Demo wrapper to inject mocks
class DemoNet(ManipulationPrimitiveNet):
    def connect(self):
        bot = RandomObservationRobot("random_bot")
        return {"random_bot": bot}, {"random_bot": MockTeleoperator("random_bot")}, {}


def run_demo():
    num_episodes = 0
    net = DemoNet(net_config)
    net.reset()

    print(f"--- ROLLOUT START: {net.active_primitive} ---")

    # Run until we hit the terminal stage
    for i in range(100):
        print(f"Step {i}: Running [{net.active_primitive}]...")

        # Dummy action tensor matching the search primitive's 3 adaptive dimensions
        action = torch.randn(net.action_dim)

        net.step(action)

        if net.in_terminal:
            print(f"  -> Terminal condition met!")
            net.reset()

            num_episodes += 1
            if num_episodes > 5:
                break

    # The "Two Processor Step Reset" happens here
    print("\n--- TRIGGERING SYSTEM RESET ---")
    net.reset()


if __name__ == "__main__":
    run_demo()
