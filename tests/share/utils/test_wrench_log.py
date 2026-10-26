"""The diagnosis wrench rides through the env processor without reaching the policy state, and is logged per episode."""

from __future__ import annotations

import csv

import pytest
import torch
from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import TransitionKey
from lerobot.utils.constants import OBS_STATE

from share.processor.observation import DefaultObservationProcessor
from share.utils.wrench_log import read_ee_wrenches, wrench_peak_summary, write_wrench_log

WRENCH = {f"viperx.{axis}.ee_wrench": float(i) for i, axis in enumerate(("x", "y", "z", "rx", "ry", "rz"), start=1)}
JOINTS = {"viperx.waist.pos": 0.1, "viperx.shoulder.pos": 0.2}


def _transition(observation: dict) -> dict:
    return {
        TransitionKey.OBSERVATION: observation, TransitionKey.ACTION: None, TransitionKey.REWARD: 0.0,
        TransitionKey.DONE: False, TransitionKey.TRUNCATED: False, TransitionKey.INFO: {},
        TransitionKey.COMPLEMENTARY_DATA: {},
    }


def test_wrench_survives_the_env_processor_but_stays_out_of_the_state_by_default():
    step = DefaultObservationProcessor(add_joint_position_to_observation={"viperx": True})

    obs = step(_transition({**JOINTS, **WRENCH}))[TransitionKey.OBSERVATION]

    assert obs[OBS_STATE].shape == (2,)
    assert {key: obs[key] for key in WRENCH} == WRENCH


def test_wrench_joins_the_state_only_when_the_processor_asks_for_it():
    step = DefaultObservationProcessor(
        add_joint_position_to_observation={"viperx": True}, add_ee_wrench_to_observation={"viperx": True}
    )

    state = step(_transition({**JOINTS, **WRENCH}))[TransitionKey.OBSERVATION][OBS_STATE]

    assert state.tolist() == pytest.approx([0.1, 0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_inferred_policy_features_ignore_the_wrench_by_default():
    step = DefaultObservationProcessor(add_joint_position_to_observation={"viperx": True})
    initial = {PipelineFeatureType.OBSERVATION: {
        key: PolicyFeature(type=FeatureType.STATE, shape=(1,)) for key in {**JOINTS, **WRENCH}
    }}

    features = step.transform_features(initial)[PipelineFeatureType.OBSERVATION]

    assert features[OBS_STATE].shape == (2,)


def test_read_ee_wrenches_picks_only_wrench_entries_from_floats_and_tensors():
    obs = {**JOINTS, "viperx.x.ee_wrench": torch.tensor([1.5]), "viperx.y.ee_wrench": 2.5, "observation.state": torch.zeros(2)}

    assert read_ee_wrenches(obs) == {"viperx.x.ee_wrench": 1.5, "viperx.y.ee_wrench": 2.5}


def test_wrench_log_is_one_csv_row_per_step_and_reports_the_peak(tmp_path):
    rows = [
        {"viperx.x.ee_wrench": 0.0, "viperx.y.ee_wrench": 0.0, "viperx.z.ee_wrench": 0.0},
        {"viperx.x.ee_wrench": 3.0, "viperx.y.ee_wrench": 4.0, "viperx.z.ee_wrench": 0.0},
        {"viperx.x.ee_wrench": 1.0, "viperx.y.ee_wrench": 0.0, "viperx.z.ee_wrench": 0.0},
    ]
    path = tmp_path / "primitive" / "episode_000000_wrench.csv"

    write_wrench_log(path, rows, fps=10.0)

    with open(path) as file:
        written = list(csv.reader(file))
    assert written[0] == ["step", "time_s", "viperx.x.ee_wrench", "viperx.y.ee_wrench", "viperx.z.ee_wrench"]
    assert written[2] == ["1", "0.1000", "3.0", "4.0", "0.0"]
    assert wrench_peak_summary(rows, fps=10.0) == "viperx peak force 5.0 N at 0.1s"
