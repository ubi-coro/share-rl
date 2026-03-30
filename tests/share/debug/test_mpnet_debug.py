"""Tests for the async MP-Net debug visualizer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from threading import Event

import pytest
import torch

from lerobot.processor import TransitionKey, create_transition

from share.debug.mpnet_debug import MPNetDebugConfig, MPNetDebugger
from share.workspace.mpnet import (
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveNetConfig,
    OnSuccess,
    OpenLoopTrajectorySpec,
    OpenLoopTrajectoryPrimitiveConfig,
    Always,
    TaskFrame,
)


def _make_debug_config() -> ManipulationPrimitiveNetConfig:
    home = ManipulationPrimitiveConfig(task_frame={"arm": TaskFrame(target=[0.0] * 6)})
    scripted = OpenLoopTrajectoryPrimitiveConfig(
        task_frame={"arm": TaskFrame(target=[0.0] * 6)},
        trajectory=OpenLoopTrajectorySpec(
            delta={"arm": [0.3, 0.0, 0.1, 0.0, 0.0, 0.0]},
            frame={"arm": "world"},
            duration_s={"arm": 0.5},
        ),
    )
    done = ManipulationPrimitiveConfig(task_frame={"arm": TaskFrame(target=[0.3, 0.0, 0.1, 0.0, 0.0, 0.0])}, is_terminal=True)
    return ManipulationPrimitiveNetConfig(
        start_primitive="home",
        reset_primitive="home",
        primitives={
            "home": home,
            "scripted": scripted,
            "done": done,
        },
        transitions=[
            Always(source="home", target="scripted"),
            OnSuccess(source="scripted", target="done", success_key="primitive_complete"),
        ],
    )


def _read_trace(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_mpnet_debugger_writes_trace_and_config_summary(tmp_path):
    config = _make_debug_config()
    trace_path = tmp_path / "trace.jsonl"
    debugger = MPNetDebugger.start(
        MPNetDebugConfig(
            enabled=True,
            live_rerun=False,
            trace_path=trace_path,
            flush_interval_s=0.01,
        ),
        config,
    )
    net = SimpleNamespace(active_primitive="scripted", config=config)

    reset_transition = create_transition(
        observation={
            "arm.x.ee_pos": torch.tensor([0.0]),
            "arm.y.ee_pos": torch.tensor([0.0]),
            "arm.z.ee_pos": torch.tensor([0.0]),
            "arm.rx.ee_pos": torch.tensor([0.0]),
            "arm.ry.ee_pos": torch.tensor([0.0]),
            "arm.rz.ee_pos": torch.tensor([0.0]),
        },
        info={
            "primitive_target_pose": {"arm": [0.3, 0.0, 0.1, 0.0, 0.0, 0.0]},
            "trajectory_progress": 0.0,
            "primitive_complete": False,
        },
    )
    debugger.log_reset(net, reset_transition)

    step_transition = create_transition(
        observation={
            "arm.x.ee_pos": torch.tensor([0.3]),
            "arm.y.ee_pos": torch.tensor([0.0]),
            "arm.z.ee_pos": torch.tensor([0.1]),
            "arm.rx.ee_pos": torch.tensor([0.0]),
            "arm.ry.ee_pos": torch.tensor([0.0]),
            "arm.rz.ee_pos": torch.tensor([0.0]),
        },
        info={
            "primitive_target_pose": {"arm": [0.3, 0.0, 0.1, 0.0, 0.0, 0.0]},
            "trajectory_progress": 1.0,
            "primitive_complete": True,
            "primitive_step": 2,
            "episode_step": 5,
            "transition_from": "scripted",
            "transition_to": "done",
            "transition_reason": "primitive_complete",
        },
    )
    net.active_primitive = "done"
    debugger.log_step(net, step_transition)
    debugger.close()

    events = _read_trace(trace_path)
    assert [event["kind"] for event in events] == [
        "session_start",
        "reset",
        "step",
        "transition",
        "session_end",
    ]
    step_event = events[2]
    assert step_event["trajectory_progress"] == 1.0
    assert step_event["transition_reason"] == "primitive_complete"
    assert step_event["robots"]["arm"]["target_pose"] == [0.3, 0.0, 0.1, 0.0, 0.0, 0.0]
    assert step_event["robots"]["arm"]["current_pose"] == pytest.approx([0.3, 0.0, 0.1, 0.0, 0.0, 0.0])

    summary_path = trace_path.with_name("config_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["primitive_count"] == 3
    assert summary["transitions"][0]["condition_summary"] == "always"
    assert summary["transitions"][1]["condition_summary"] == "info.primitive_complete == True"


def test_mpnet_debugger_drops_events_when_queue_is_full(tmp_path):
    config = _make_debug_config()
    trace_path = tmp_path / "trace.jsonl"
    debugger = MPNetDebugger.start(
        MPNetDebugConfig(
            enabled=True,
            live_rerun=False,
            trace_path=trace_path,
            queue_size=2,
            flush_interval_s=0.01,
        ),
        config,
    )
    session = debugger._session
    assert session is not None

    gate = Event()
    original_write_event = session._write_event

    def slow_write_event(trace_file, event):
        if event["kind"] != "session_start":
            gate.wait(timeout=0.2)
        original_write_event(trace_file, event)

    session._write_event = slow_write_event

    for index in range(8):
        session._enqueue_event({"kind": "step", "timestamp": time.time(), "index": index})

    time.sleep(0.05)
    assert session.dropped_events > 0
    gate.set()
    debugger.close()

    events = _read_trace(trace_path)
    assert events[-1]["kind"] == "session_end"
