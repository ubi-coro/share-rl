"""Tests for MP-Net debug-oriented workspace summaries."""

from __future__ import annotations

from share.workspace.mpnet import (
    ManipulationPrimitiveConfig,
    ManipulationPrimitiveNetConfig,
    MoveDeltaPrimitiveConfig,
    OnSuccess,
    OnTargetPoseReached,
    OpenLoopTrajectorySpec,
    OpenLoopTrajectoryPrimitiveConfig,
    TaskFrame,
    summarize_mpnet_debug,
)


def test_summarize_mpnet_debug_reports_roles_types_and_conditions():
    config = ManipulationPrimitiveNetConfig(
        start_primitive="home",
        reset_primitive="retreat",
        primitives={
            "home": ManipulationPrimitiveConfig(task_frame={"arm": TaskFrame(target=[0.0] * 6)}),
            "approach": MoveDeltaPrimitiveConfig(
                task_frame={"arm": TaskFrame(target=[0.0] * 6)},
                delta={"arm": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]},
            ),
            "retreat": OpenLoopTrajectoryPrimitiveConfig(
                task_frame={"arm": TaskFrame(target=[0.0] * 6)},
                trajectory=OpenLoopTrajectorySpec(
                    delta={"arm": [-0.1, 0.0, 0.1, 0.0, 0.0, 0.0]},
                    frame={"arm": "world"},
                    duration_s={"arm": 0.5},
                ),
                is_terminal=True,
            ),
        },
        transitions=[
            OnTargetPoseReached(source="home", target="approach", axes=["x"], tolerance=0.01),
            OnSuccess(source="approach", target="retreat", success_key="primitive_complete"),
        ],
    )

    summary = summarize_mpnet_debug(config)

    assert summary["start_primitive"] == "home"
    assert summary["reset_primitive"] == "retreat"
    assert summary["primitive_count"] == 3
    assert summary["transition_count"] == 2

    primitives = {primitive["name"]: primitive for primitive in summary["primitives"]}
    assert primitives["home"]["roles"]["is_start"] is True
    assert primitives["retreat"]["roles"]["is_reset"] is True
    assert primitives["retreat"]["roles"]["is_terminal"] is True
    assert primitives["approach"]["type"] == "move_delta"
    assert primitives["retreat"]["type"] == "open_loop_trajectory"

    transitions = summary["transitions"]
    assert transitions[0]["condition_summary"].startswith("target pose reached")
    assert transitions[1]["condition_summary"] == "info.primitive_complete == True"
    assert transitions[1]["parameters"]["success_key"] == "primitive_complete"
