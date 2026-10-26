"""Async inference helpers in control_utils: predict_action_async, make_async_runners and TickStats."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from beast_refiner.async_inference import AsyncInferenceConfig
from beast_refiner.policies.beast_xvla_refiner_v2.modeling_beast_xvla_refiner_v2 import BeastXVLARefinerV2Model

from share.utils.control_utils import TickStats, make_async_runners, predict_action_async

# ---------------------------------------------------------------------------------------------------------
# predict_action_async
# ---------------------------------------------------------------------------------------------------------


class _Runner:
    def __init__(self):
        self.calls = []

    def select_action(self, batch, obs_timestep):
        self.calls.append(
            {
                "batch": batch,
                "obs_timestep": obs_timestep,
                "grad": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
            }
        )
        return torch.ones(1, 2)


def _predict(runner, observation, **kwargs):
    return predict_action_async(
        observation=observation,
        runner=runner,
        obs_timestep=kwargs.pop("obs_timestep", 0),
        device=torch.device("cpu"),
        preprocessor=kwargs.pop("preprocessor", lambda batch: batch),
        postprocessor=kwargs.pop("postprocessor", lambda action: action),
        **kwargs,
    )


def test_runner_gets_the_preprocessed_batch_and_the_timestep():
    runner = _Runner()
    observation = {"observation.state": torch.zeros(1, 3)}

    action = _predict(
        runner,
        observation,
        obs_timestep=7,
        task="fold",
        robot_type="aloha",
        preprocessor=lambda batch: {**batch, "preprocessed": True},
        postprocessor=lambda a: a * 3,
    )

    (call,) = runner.calls
    assert call["obs_timestep"] == 7
    assert call["batch"]["preprocessed"]
    assert call["batch"]["task"] == "fold" and call["batch"]["robot_type"] == "aloha"
    assert torch.equal(action, torch.full((1, 2), 3.0)), "the runner's action is postprocessed"


def test_the_callers_observation_is_not_modified():
    observation = {"observation.state": torch.zeros(1, 3)}

    _predict(_Runner(), observation)

    assert list(observation) == ["observation.state"]


def test_runs_under_no_grad_but_not_inference_mode():
    """The batch is read by the runner's System 2 thread, where inference tensors may not be updated in place."""
    runner = _Runner()

    _predict(runner, {"observation.state": torch.zeros(1, 3)})

    assert runner.calls[0]["grad"] is False
    assert runner.calls[0]["inference_mode"] is False


# ---------------------------------------------------------------------------------------------------------
# make_async_runners
# ---------------------------------------------------------------------------------------------------------


class _StubRunner:
    """AsyncPolicyRunner without a worker thread or CUDA."""

    instances: list[_StubRunner] = []
    fail_on_start = False

    def __init__(self, policy, config):
        self.policy, self.config = policy, config
        self.start_kwargs = None
        self.stopped = False
        _StubRunner.instances.append(self)

    def start(self, **kwargs):
        self.start_kwargs = kwargs
        if _StubRunner.fail_on_start:
            raise RuntimeError("warmup failed")

    def stop(self, timeout: float = 2.0):
        self.stopped = True


@pytest.fixture
def stub_runners(monkeypatch):
    """Swap the runner for a stub; process tuning (gc.freeze, torch threads) must not leak into the test run."""
    _StubRunner.instances, _StubRunner.fail_on_start = [], False
    tuned = []
    monkeypatch.setattr("beast_refiner.async_inference.AsyncPolicyRunner", _StubRunner)
    monkeypatch.setattr("beast_refiner.async_inference.apply_process_tuning", tuned.append)
    return SimpleNamespace(tuned=tuned, instances=_StubRunner.instances)


class _V2Model(BeastXVLARefinerV2Model):
    """A beast_xvla_refiner_v2 model without the weights."""

    def __init__(self, fps: float):
        torch.nn.Module.__init__(self)
        self.fps = fps


class _LadderModel(_V2Model):
    """Variants such as beast_xvla_refiner_v2_ladder subclass v2's model."""


def _policy(type: str = "beast_xvla_refiner_v2", use_amp: bool = False, fps: float = 30, model_cls=_V2Model):
    model = model_cls(fps) if issubclass(model_cls, _V2Model) else SimpleNamespace(fps=fps)
    return SimpleNamespace(config=SimpleNamespace(type=type, use_amp=use_amp), model=model)


def test_async_inference_is_off_without_a_config(stub_runners):
    assert make_async_runners({"pick": _policy()}, None, fps=30) == {}
    assert stub_runners.instances == [] and stub_runners.tuned == []


def test_runners_are_made_for_beast_policies_only(stub_runners, caplog):
    beast = _policy()
    policies = {"pick": beast, "place": _policy(type="act", model_cls=SimpleNamespace), "reset": None}

    with caplog.at_level(logging.WARNING):
        runners = make_async_runners(policies, AsyncInferenceConfig(sys2_trigger_margin_ticks=6), fps=30)

    assert list(runners) == ["pick"]
    assert runners["pick"].policy is beast
    assert runners["pick"].config.sys2_trigger_margin_ticks == 6
    assert "[place] async inference only supports beast_xvla_refiner_v2" in caplog.text


def test_runners_are_made_for_v2_variants(stub_runners):
    ladder = _policy(type="beast_xvla_refiner_v2_ladder", model_cls=_LadderModel)

    assert list(make_async_runners({"grab": ladder}, AsyncInferenceConfig(), fps=30)) == ["grab"]


def test_policies_without_a_v2_model_run_synchronously(stub_runners):
    no_model = SimpleNamespace(config=SimpleNamespace(type="custom", use_amp=False))
    policies = {"act": _policy(type="act", model_cls=SimpleNamespace), "custom": no_model}

    assert make_async_runners(policies, AsyncInferenceConfig(), fps=30) == {}


def test_process_tuning_is_applied_once_after_all_runners_exist(stub_runners):
    policies = {"pick": _policy(), "place": _policy()}
    cfg = AsyncInferenceConfig()

    runners = make_async_runners(policies, cfg, fps=30)

    assert stub_runners.tuned == [cfg]
    assert all(r.start_kwargs == {"tune_process": False} for r in runners.values())


def test_no_process_tuning_without_a_beast_policy(stub_runners):
    assert make_async_runners({"pick": _policy(type="act", model_cls=SimpleNamespace)}, AsyncInferenceConfig(), fps=30) == {}
    assert stub_runners.tuned == []


def test_runners_refuse_amp(stub_runners):
    with pytest.raises(ValueError, match="use_amp=False"):
        make_async_runners({"pick": _policy(use_amp=True)}, AsyncInferenceConfig(), fps=30)

    assert stub_runners.instances == []


def test_runners_warn_about_an_fps_mismatch(stub_runners, caplog):
    with caplog.at_level(logging.WARNING):
        make_async_runners({"pick": _policy(fps=25)}, AsyncInferenceConfig(), fps=30)

    assert "trained at 25 fps but the env runs at 30 fps" in caplog.text


def test_runners_are_stopped_when_one_fails_to_start(stub_runners):
    _StubRunner.fail_on_start = True

    with pytest.raises(RuntimeError, match="warmup failed"):
        make_async_runners({"pick": _policy()}, AsyncInferenceConfig(), fps=30)

    assert stub_runners.instances and all(r.stopped for r in stub_runners.instances)


# ---------------------------------------------------------------------------------------------------------
# TickStats
# ---------------------------------------------------------------------------------------------------------


def test_tick_stats_summarise_once_per_window():
    stats = TickStats(window=3)

    assert stats.add(0.030, 0.010) is None
    assert stats.add(0.034, 0.012) is None
    summary = stats.add(0.050, 0.020)

    assert "last 3 ticks" in summary
    assert "max 50.00ms" in summary and "20.0hz min" in summary and "dt_load max 20.00ms" in summary
    assert stats.add(0.030, 0.010) is None, "the window starts over"


def test_tick_stats_report_phases_as_mean_and_max():
    stats = TickStats(window=2)

    stats.add(0.030, 0.030, policy=0.010, env_step=0.020)
    summary = stats.add(0.034, 0.034, policy=0.020, env_step=0.010, store=0.004)

    assert "policy 15.0/20.0" in summary and "env_step 15.0/20.0" in summary
    assert "store 4.0/4.0" in summary, "a phase that occurred on some ticks is averaged over those"
    assert "policy" not in (stats.add(0.030, 0.030) or "") and not stats.phases, "phases start over with the window"
