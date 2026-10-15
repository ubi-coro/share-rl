"""record_beast: rollout recorders and the record() session around record_loop()."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from lerobot.processor import create_transition

from share.configs.record import RecordConfig
from share.scripts import record_beast
from share.scripts.record_beast import record_loop
from share.teleoperators import TeleopEvents


def _identity(x):
    return x


class _FakePolicy:
    """Stands in for a lerobot policy on the synchronous predict_action path."""

    def __init__(self):
        self.config = SimpleNamespace(device="cpu", use_amp=False, input_features={"obs": object()})
        self.resets = 0

    def reset(self):
        self.resets += 1

    def select_action(self, batch):
        return torch.zeros(1, 1)


class _FakeMPNet:
    """One primitive with a policy; every episode lasts ``episode_len`` steps, ``stop_on_reset`` ends the session."""

    def __init__(self, episode_len: int = 1, stop_on_reset: bool = False):
        self.action_dim = 1
        self.active_primitive = "pick"
        self.episode_len = episode_len
        self.stop_on_reset = stop_on_reset
        self.steps = 0
        self.closed = False
        self.config = SimpleNamespace(
            fps=10_000,
            type="mock_robot",
            primitives={"pick": SimpleNamespace(task_description="pick task")},
        )

    def reset(self):
        self.steps = 0
        info = {TeleopEvents.STOP_RECORDING: True} if self.stop_on_reset else {}
        return create_transition(observation={"obs": torch.tensor([[1.0]])}, info=info)

    def step(self, action):
        self.steps += 1
        return create_transition(
            observation={"obs": torch.tensor([[2.0]])},
            action=action,
            reward=0.0,
            done=self.steps >= self.episode_len,
            info={},
        )

    def close(self):
        self.closed = True


def test_record_loop_without_recorders_does_not_need_them():
    """Dashboards are optional: no recorder for the active primitive means nothing is recorded, nothing breaks."""
    policy = _FakePolicy()

    for recorders in (None, {}):
        record_loop(
            mp_net=_FakeMPNet(),
            datasets={},
            policies={"pick": policy},
            preprocessors={"pick": _identity},
            postprocessors={"pick": _identity},
            recorders=recorders,
        )

    assert policy.resets == 2


def _record_session(monkeypatch, cfg: RecordConfig, mp_net: _FakeMPNet, policies: dict | None = None):
    """Run ``record()`` against a fake MP-net, without datasets, robot or sound."""
    policies = policies if policies is not None else {"pick": _FakePolicy()}
    monkeypatch.setattr(record_beast, "ManipulationPrimitiveNet", lambda env_cfg: mp_net)
    monkeypatch.setattr(
        record_beast,
        "make_policies_and_datasets",
        lambda _cfg: ({}, policies, {n: _identity for n in policies}, {n: _identity for n in policies}),
    )
    monkeypatch.setattr(record_beast, "log_say", lambda *a, **k: None)
    record_beast.record(cfg)


def _record_config(**kwargs) -> RecordConfig:
    return RecordConfig(env=SimpleNamespace(fps=10_000), play_sounds=False, **kwargs)


def test_record_session_without_output_dir_builds_no_recorders(monkeypatch):
    """output_dir=None used to crash on Path(None); now it simply records no dashboards."""
    built = []
    monkeypatch.setattr(record_beast, "EpisodeRecorder", lambda **kw: built.append(kw))
    monkeypatch.setattr(record_beast, "PolicyVisualizationAdapter", lambda *a, **kw: None)

    mp_net = _FakeMPNet(stop_on_reset=True)
    _record_session(monkeypatch, _record_config(output_dir=None), mp_net)

    assert built == []
    assert mp_net.closed


def test_record_session_with_output_dir_builds_one_recorder_per_policy(monkeypatch, tmp_path):
    built = []
    monkeypatch.setattr(record_beast, "EpisodeRecorder", lambda **kw: built.append(kw) or kw)
    monkeypatch.setattr(record_beast, "PolicyVisualizationAdapter", lambda *a, **kw: None)

    _record_session(monkeypatch, _record_config(output_dir=str(tmp_path)), _FakeMPNet(stop_on_reset=True))

    assert len(built) == 1
