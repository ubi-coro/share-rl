from __future__ import annotations

import copy
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from share.configs.rl import MPNetTrainRLServerPipelineConfig

os.environ.setdefault("PYNPUT_BACKEND", "dummy")

from lerobot.policies.sac.configuration_sac import (  # noqa: E402
    ActorLearnerConfig,
    ConcurrencyConfig,
    SACConfig,
)
from share.envs.manipulation_primitive.config_manipulation_primitive import (  # noqa: E402
    ManipulationPrimitiveConfig,
    OpenLoopTrajectoryPrimitiveConfig,
    OpenLoopTrajectorySpec,
)
from share.envs.manipulation_primitive.task_frame import (  # noqa: E402
    ControlMode,
    PolicyMode,
    TaskFrame,
)
from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (  # noqa: E402
    ManipulationPrimitiveNetConfig,
)
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import (  # noqa: E402
    ManipulationPrimitiveNet,
)
from share.envs.manipulation_primitive_net.transitions import Always, OnTimeLimit  # noqa: E402
from share.scripts.actor_server import run_actor  # noqa: E402
from share.scripts.learner_server import run_learner  # noqa: E402  # noqa: E402
from share.utils.mock_utils import MockRobot, MockTeleoperator  # noqa: E402


class MockPoseRobot(MockRobot):
    @property
    def observation_features(self) -> dict:
        base = super().observation_features
        return {
            **base,
            "x.ee_pos": float,
            "y.ee_pos": float,
            "z.ee_pos": float,
            "wx.ee_pos": float,
            "wy.ee_pos": float,
            "wz.ee_pos": float,
        }

    def get_observation(self) -> dict:
        joints = super().get_observation()
        return {
            **joints,
            "x.ee_pos": float(self.current_joints[0]),
            "y.ee_pos": float(self.current_joints[1]),
            "z.ee_pos": float(self.current_joints[2]),
            "wx.ee_pos": float(self.current_joints[3]),
            "wy.ee_pos": float(self.current_joints[4]),
            "wz.ee_pos": float(self.current_joints[5]),
        }


def _free_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except PermissionError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Socket binding is not permitted in this test environment: {exc}")


def _adaptive_task_frame() -> TaskFrame:
    return TaskFrame(
        target=[0.0] * 6,
        origin=[0.0] * 6,
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[PolicyMode.RELATIVE, PolicyMode.RELATIVE, PolicyMode.RELATIVE, None, None, None],
    )


def _scripted_task_frame() -> TaskFrame:
    return TaskFrame(
        target=[0.0] * 6,
        origin=[0.0] * 6,
        control_mode=[ControlMode.POS] * 6,
        policy_mode=[None] * 6,
    )


def _build_env_config(port: int) -> ManipulationPrimitiveNetConfig:
    actor_learner_config = ActorLearnerConfig(
        learner_host="127.0.0.1",
        learner_port=port,
        policy_parameters_push_frequency=0.05,
        queue_get_timeout=0.1,
    )
    concurrency = ConcurrencyConfig(actor="threads", learner="threads")
    policy_cfg = SACConfig(
        device="cpu",
        storage_device="cpu",
        online_steps=2,
        online_buffer_capacity=32,
        online_step_before_learning=1,
        policy_update_freq=1,
        utd_ratio=1,
        use_torch_compile=False,
        actor_learner_config=actor_learner_config,
        concurrency=concurrency,
    )

    adaptive = ManipulationPrimitiveConfig(
        task_frame={"arm": _adaptive_task_frame()},
        policy=policy_cfg,
        is_terminal=False,
    )
    reset_scripted = OpenLoopTrajectoryPrimitiveConfig(
        task_frame={"arm": _scripted_task_frame()},
        trajectory=OpenLoopTrajectorySpec(
            delta={"arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            frame={"arm": "world"},
            duration_s={"arm": 0.1},
        ),
        is_terminal=False,
    )
    done = OpenLoopTrajectoryPrimitiveConfig(
        task_frame={"arm": _scripted_task_frame()},
        trajectory=OpenLoopTrajectorySpec(
            target={"arm": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
            frame={"arm": "task"},
            duration_s={"arm": 0.1},
        ),
        is_terminal=True,
    )

    return ManipulationPrimitiveNetConfig(
        start_primitive="approach",
        reset_primitive="reset_scripted",
        primitives={
            "approach": adaptive,
            "reset_scripted": reset_scripted,
            "done": done,
        },
        transitions=[
            Always(source="reset_scripted", target="approach"),
            OnTimeLimit(source="approach", target="done", max_steps=2, step_key="primitive_step"),
        ],
        fps=60,
        robot=None,
        teleop=None,
        cameras={},
    )


def test_mpn_rl_servers_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _mock_connect(self: ManipulationPrimitiveNet):
        return (
            {"arm": MockPoseRobot(name="arm", is_task_frame=True)},
            {"arm": MockTeleoperator(name="arm", is_delta=True)},
            {},
        )

    monkeypatch.setattr(ManipulationPrimitiveNet, "connect", _mock_connect)

    port = _free_port()
    env_cfg = _build_env_config(port)

    base_cfg = MPNetTrainRLServerPipelineConfig(
        env=env_cfg,
        policy=None,
        dataset=None,
        output_dir=tmp_path / "run",
        job_name="mpnet-e2e",
        seed=7,
        batch_size=2,
        log_freq=1,
        save_freq=1,
        save_checkpoint=True,
    )
    base_cfg.wandb.enable = False
    base_cfg.validate()

    learner_cfg = copy.deepcopy(base_cfg)
    actor_cfg = copy.deepcopy(base_cfg)

    shutdown_event = threading.Event()
    learner_result: dict[str, dict] = {}
    learner_error: list[Exception] = []

    def _run_learner():
        try:
            learner_result["result"] = run_learner(learner_cfg, shutdown_event=shutdown_event)
        except Exception as exc:  # noqa: BLE001
            learner_error.append(exc)
            shutdown_event.set()

    learner_thread = threading.Thread(target=_run_learner, daemon=True)
    learner_thread.start()
    time.sleep(0.3)

    try:
        actor_result = run_actor(actor_cfg, shutdown_event=shutdown_event)
    except Exception:
        shutdown_event.set()
        raise

    learner_thread.join(timeout=30)
    shutdown_event.set()

    if learner_error:
        raise learner_error[0]
    assert not learner_thread.is_alive(), "Learner thread did not terminate."

    assert actor_result["per_primitive_steps"]["approach"] >= 2
    assert actor_result["applied_parameter_updates"] >= 1

    assert "result" in learner_result
    assert learner_result["result"]["optimization_steps"]["approach"] >= 1

    primitive_root = tmp_path / "run" / "approach"
    assert (primitive_root / "checkpoints" / "last").exists()
    assert (primitive_root / "dataset").exists()
