"""record_beast: rollout recorders, async inference runners and the record() session around record_loop()."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from beast_refiner.async_inference import AsyncInferenceConfig
from lerobot.processor import create_transition

from share.configs.record_beast import RecordBeastConfig
from share.scripts import record_beast
from share.scripts.record_beast import record_loop
from share.teleoperators import TeleopEvents


def _identity(x):
    return x


class _FakePolicy:
    """Stands in for a lerobot policy on the synchronous predict_action path."""

    def __init__(self, type: str = "act", use_amp: bool = False, fps: float = 30):
        self.config = SimpleNamespace(type=type, device="cpu", use_amp=use_amp, input_features={"obs": object()})
        self.model = SimpleNamespace(fps=fps)
        self.resets = 0
        self.selected = 0

    def reset(self):
        self.resets += 1

    def select_action(self, batch):
        self.selected += 1
        return torch.zeros(1, 1)


class _FakeRunner:
    """Stands in for an AsyncPolicyRunner: records what the control loop asks of it."""

    def __init__(self):
        self.timesteps: list[int] = []
        self.batches: list[dict] = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def select_action(self, batch, obs_timestep):
        self.batches.append(batch)
        self.timesteps.append(obs_timestep)
        return torch.zeros(1, 1)


class _FakeMPNet:
    """One primitive with a policy; every episode lasts ``episode_len`` steps, ``stop_on_reset`` ends the session."""

    def __init__(self, episode_len: int = 1, stop_on_reset: bool = False, fps: int = 30, fail_on_step: int | None = None):
        self.fail_on_step = fail_on_step
        self.action_dim = 1
        self.active_primitive = "pick"
        self.episode_len = episode_len
        self.stop_on_reset = stop_on_reset
        self.steps = 0
        self.actions: list[torch.Tensor] = []
        self.closed = False
        self.config = SimpleNamespace(
            fps=fps,
            type="mock_robot",
            primitives={"pick": SimpleNamespace(task_description="pick task")},
            transitions=[],
        )

    def reset(self):
        self.steps = 0
        info = {TeleopEvents.STOP_RECORDING: True} if self.stop_on_reset else {}
        return create_transition(observation={"obs": torch.tensor([[1.0]])}, info=info)

    def step(self, action):
        self.steps += 1
        if self.steps == self.fail_on_step:
            raise RuntimeError("control board disconnected")
        self.actions.append(action)
        return create_transition(
            observation={"obs": torch.tensor([[2.0]])},
            action=action,
            reward=0.0,
            done=self.steps >= self.episode_len,
            info={},
        )

    def close(self):
        self.closed = True


def _loop(mp_net, policy, datasets=None, **kwargs):
    return record_loop(
        mp_net=mp_net,
        datasets=datasets or {},
        policies={"pick": policy},
        preprocessors={"pick": _identity},
        postprocessors={"pick": _identity},
        **kwargs,
    )


# ---------------------------------------------------------------------------------------------------------
# rollout recorders
# ---------------------------------------------------------------------------------------------------------


def test_record_loop_without_recorders_does_not_need_them():
    """Dashboards are optional: no recorder for the active primitive means nothing is recorded, nothing breaks."""
    policy = _FakePolicy()

    for recorders in (None, {}):
        _loop(_FakeMPNet(), policy, recorders=recorders)

    assert policy.resets == 2


class _WrenchMPNet(_FakeMPNet):
    """Observations carry the wrench, like a ViperX with the Bota sensor enabled."""

    def reset(self):
        transition = super().reset()
        transition[record_beast.TransitionKey.OBSERVATION]["viperx.z.ee_wrench"] = 1.0
        return transition

    def step(self, action):
        transition = super().step(action)
        transition[record_beast.TransitionKey.OBSERVATION]["viperx.z.ee_wrench"] = 1.0 + self.steps
        return transition


def test_record_loop_reads_the_wrench_off_every_observation():
    """Diagnosis only: the wrench is read from the observation the policy sees, one row per control step."""
    wrench_log = []

    _loop(_WrenchMPNet(episode_len=3), _FakePolicy(), wrench_log=wrench_log)

    assert wrench_log == [{"viperx.z.ee_wrench": 1.0}, {"viperx.z.ee_wrench": 2.0}, {"viperx.z.ee_wrench": 3.0}]


def test_record_loop_without_a_wrench_log_ignores_the_wrench():
    _loop(_WrenchMPNet(episode_len=2), _FakePolicy())


# ---------------------------------------------------------------------------------------------------------
# a failing env step
# ---------------------------------------------------------------------------------------------------------


class _FakeDataset:
    """Counts the frames added to the episode buffer and the episodes saved."""

    features = {"obs": None}
    num_episodes = 4

    def __init__(self):
        self.writer = SimpleNamespace(episode_buffer={"size": 0})
        self.saved = 0

    def add_frame(self, frame):
        self.writer.episode_buffer["size"] += 1

    def save_episode(self):
        self.saved += 1


def _answers(monkeypatch, *answers: str) -> list:
    """Answer the console prompts in order; returns the prompts that were asked."""
    prompts, remaining = [], iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt) or next(remaining))
    return prompts


def _run_failing_episode(dataset, fail_on_step: int = 3):
    """The env step number `fail_on_step` raises; the error must still reach the caller."""
    datasets = {} if dataset is None else {"pick": dataset}
    with pytest.raises(RuntimeError, match="control board disconnected"):
        _loop(_FakeMPNet(episode_len=10, fail_on_step=fail_on_step), _FakePolicy(), datasets=datasets)


def test_failed_env_step_saves_the_episode_when_confirmed(monkeypatch):
    dataset = _FakeDataset()
    prompts = _answers(monkeypatch, "maybe", "y")

    _run_failing_episode(dataset)

    assert dataset.saved == 1
    assert len(prompts) == 2, "anything but y/n is asked again"
    assert "2 frames" in prompts[0] and "episode 4" in prompts[0]


def test_failed_env_step_discards_the_episode_when_declined(monkeypatch):
    dataset = _FakeDataset()
    _answers(monkeypatch, "n")

    _run_failing_episode(dataset)

    assert dataset.saved == 0


def test_failed_env_step_does_not_ask_without_frames_or_dataset(monkeypatch):
    prompts = _answers(monkeypatch)

    _run_failing_episode(_FakeDataset(), fail_on_step=1)  # nothing recorded yet
    _run_failing_episode(None)  # nowhere to save to

    assert prompts == []


def test_failed_env_step_with_closed_stdin_counts_as_no(monkeypatch):
    dataset = _FakeDataset()

    def closed_stdin(prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", closed_stdin)

    _run_failing_episode(dataset)

    assert dataset.saved == 0


# ---------------------------------------------------------------------------------------------------------
# async inference in record_loop
# ---------------------------------------------------------------------------------------------------------


def test_record_loop_drives_the_runner_with_a_per_episode_timestep():
    """System 1 is indexed against obs_timestep: +1 per control step, back to 0 with every episode."""
    policy, runner = _FakePolicy(), _FakeRunner()

    for _ in range(2):
        _loop(_FakeMPNet(episode_len=3), policy, runners={"pick": runner})

    assert runner.timesteps == [0, 1, 2, 0, 1, 2]
    assert runner.resets == 2, "one runner reset per episode"
    assert policy.resets == 0, "the runner's reset covers the policy's"
    assert policy.selected == 0, "the policy is only ever driven through the runner"


def test_record_loop_pre_and_postprocesses_around_the_runner():
    runner = _FakeRunner()
    mp_net = _FakeMPNet()

    record_loop(
        mp_net=mp_net,
        datasets={},
        policies={"pick": _FakePolicy()},
        preprocessors={"pick": lambda batch: {**batch, "preprocessed": True}},
        postprocessors={"pick": lambda action: action + 5},
        runners={"pick": runner},
    )

    (batch,) = runner.batches
    assert batch["preprocessed"]
    assert batch["task"] == "pick task"
    assert batch["robot_type"] == "mock_robot"
    assert torch.equal(mp_net.actions[0], torch.tensor(5.0)), "the env gets the postprocessed runner action"


def test_record_loop_keeps_the_sync_path_for_primitives_without_a_runner():
    policy = _FakePolicy()

    _loop(_FakeMPNet(episode_len=2), policy, runners={})

    assert policy.selected == 2
    assert policy.resets == 1


# ---------------------------------------------------------------------------------------------------------
# record() session
# ---------------------------------------------------------------------------------------------------------


def _record_session(
    monkeypatch, cfg: RecordBeastConfig, mp_net: _FakeMPNet, policies: dict | None = None, datasets: dict | None = None
):
    """Run ``record()`` against a fake MP-net, without a robot or sound; datasets default to none."""
    policies = policies if policies is not None else {"pick": _FakePolicy()}
    monkeypatch.setattr(record_beast, "ManipulationPrimitiveNet", lambda env_cfg: mp_net)
    monkeypatch.setattr(
        record_beast,
        "make_policies_and_datasets",
        lambda _cfg: (datasets or {}, policies, {n: _identity for n in policies}, {n: _identity for n in policies}),
    )
    monkeypatch.setattr(record_beast, "log_say", lambda *a, **k: None)
    record_beast.record(cfg)


def _record_config(**kwargs) -> RecordBeastConfig:
    return RecordBeastConfig(env=SimpleNamespace(fps=30), play_sounds=False, **kwargs)


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


def test_record_session_names_dashboards_after_the_dataset_episode(monkeypatch, tmp_path):
    """A resumed dataset continues at its episode count; its dashboards must not restart at episode_000000."""
    flushed = []
    monkeypatch.setattr(record_beast, "EpisodeRecorder", lambda **kw: SimpleNamespace(flush=lambda **f: flushed.append(f)))
    monkeypatch.setattr(record_beast, "PolicyVisualizationAdapter", lambda *a, **kw: None)
    monkeypatch.setattr(record_beast, "MultiVideoEncodingManager", lambda datasets: nullcontext())
    results = iter([{}, {TeleopEvents.STOP_RECORDING: True}])
    monkeypatch.setattr(record_beast, "record_loop", lambda **kw: next(results))
    dataset = SimpleNamespace(
        num_episodes=12, writer=SimpleNamespace(episode_buffer={"size": 1}), save_episode=lambda: None
    )

    _record_session(
        monkeypatch, _record_config(output_dir=str(tmp_path)), _FakeMPNet(), datasets={"pick": dataset}
    )

    assert flushed == [{"output_root": tmp_path / "pick", "episode_name": "episode_000012"}]


def _wrench_session(monkeypatch, tmp_path, flush) -> None:
    """One episode with two wrench rows, then stop; ``flush`` stands in for EpisodeRecorder.flush."""
    monkeypatch.setattr(record_beast, "EpisodeRecorder", lambda **kw: SimpleNamespace(flush=flush))
    monkeypatch.setattr(record_beast, "PolicyVisualizationAdapter", lambda *a, **kw: None)
    monkeypatch.setattr(record_beast, "MultiVideoEncodingManager", lambda datasets: nullcontext())
    episodes = iter([[{"viperx.x.ee_wrench": 3.0, "viperx.y.ee_wrench": 4.0, "viperx.z.ee_wrench": 0.0}] * 2, None])

    def fake_loop(**kwargs):
        rows = next(episodes)
        if rows is None:
            return {TeleopEvents.STOP_RECORDING: True}
        kwargs["wrench_log"].extend(rows)
        return {}

    monkeypatch.setattr(record_beast, "record_loop", fake_loop)
    dataset = SimpleNamespace(
        num_episodes=7, writer=SimpleNamespace(episode_buffer={"size": 1}), save_episode=lambda: None
    )
    _record_session(monkeypatch, _record_config(output_dir=str(tmp_path)), _FakeMPNet(), datasets={"pick": dataset})


def test_record_session_writes_the_wrench_into_the_folder_of_the_episode_dashboard(monkeypatch, tmp_path):
    def flush(output_root, episode_name):
        (output_root / episode_name).mkdir(parents=True)
        return output_root / episode_name

    _wrench_session(monkeypatch, tmp_path, flush)

    written = (tmp_path / "pick" / "episode_000007" / "wrench.csv").read_text().splitlines()
    assert written[0] == "step,time_s,viperx.x.ee_wrench,viperx.y.ee_wrench,viperx.z.ee_wrench"
    assert len(written) == 3
    assert not list((tmp_path / "pick").glob("*.csv")), "nothing is left next to the episode folders"


def test_record_session_puts_the_wrench_where_the_recorder_put_the_dashboard(monkeypatch, tmp_path):
    """The recorder names its own folder when the dataset gives no name, the wrench follows it."""
    _wrench_session(monkeypatch, tmp_path, lambda output_root, episode_name: output_root / "recorder_choice")

    assert (tmp_path / "pick" / "recorder_choice" / "wrench.csv").is_file()


def test_record_session_gives_the_wrench_a_folder_when_the_dashboard_wrote_nothing(monkeypatch, tmp_path):
    """No policy steps means nothing to flush, yet the wrench of the episode is still kept."""
    _wrench_session(monkeypatch, tmp_path, lambda output_root, episode_name: None)

    assert (tmp_path / "pick" / "episode_000007" / "wrench.csv").is_file()


def test_record_session_keeps_no_wrench_log_without_output_dir(monkeypatch):
    seen = []
    monkeypatch.setattr(record_beast, "MultiVideoEncodingManager", lambda datasets: nullcontext())
    monkeypatch.setattr(
        record_beast, "record_loop", lambda **kw: seen.append(kw["wrench_log"]) or {TeleopEvents.STOP_RECORDING: True}
    )

    _record_session(monkeypatch, _record_config(output_dir=None), _FakeMPNet())

    assert seen == [None]


class _SessionRunner:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


def _stub_make_async_runners(monkeypatch, runners: dict) -> list:
    calls = []

    def fake(policies, async_cfg, fps):
        calls.append((policies, async_cfg, fps))
        return runners

    monkeypatch.setattr(record_beast, "make_async_runners", fake)
    return calls


def test_record_session_starts_runners_before_the_loop_and_stops_them_after(monkeypatch):
    runner = _SessionRunner()
    calls = _stub_make_async_runners(monkeypatch, {"pick": runner})
    seen_by_loop = []

    def fake_loop(**kwargs):
        seen_by_loop.append((dict(kwargs["runners"]), runner.stopped))
        return {TeleopEvents.STOP_RECORDING: True}

    monkeypatch.setattr(record_beast, "record_loop", fake_loop)

    policies = {"pick": _FakePolicy(type="beast_xvla_refiner_v2")}
    async_cfg = AsyncInferenceConfig(sys2_trigger_margin_ticks=6)
    mp_net = _FakeMPNet()
    _record_session(monkeypatch, _record_config(async_inference=async_cfg), mp_net, policies)

    assert calls == [(policies, async_cfg, 30)]
    assert seen_by_loop == [({"pick": runner}, False)], "the loop gets the runners while they are running"
    assert runner.stopped
    assert mp_net.closed


def test_record_session_stops_runners_when_the_loop_raises(monkeypatch):
    runner = _SessionRunner()
    _stub_make_async_runners(monkeypatch, {"pick": runner})

    def boom(**_kwargs):
        raise RuntimeError("robot fell over")

    monkeypatch.setattr(record_beast, "record_loop", boom)

    mp_net = _FakeMPNet()
    with pytest.raises(RuntimeError, match="robot fell over"):
        _record_session(monkeypatch, _record_config(async_inference=AsyncInferenceConfig()), mp_net)

    assert runner.stopped
    assert mp_net.closed


def test_record_session_without_async_inference_runs_the_synchronous_path(monkeypatch):
    calls = _stub_make_async_runners(monkeypatch, {})

    _record_session(monkeypatch, _record_config(), _FakeMPNet(stop_on_reset=True))

    assert calls[0][1] is None, "no AsyncInferenceConfig, so make_async_runners has nothing to start"
