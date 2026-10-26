"""episode_progress in control_utils: per-episode tqdm line for primitives with a dataset and/or an OnTimeLimit."""
import io
import logging
from types import SimpleNamespace

from share.envs.manipulation_primitive_net.transitions import Always, OnTimeLimit
from share.utils import control_utils
from share.utils.control_utils import episode_progress, mute_log_messages


def _mp_net(*transitions, fps=10):
    return SimpleNamespace(active_primitive="grab", config=SimpleNamespace(fps=fps, transitions=list(transitions)))


def _dataset(saved_episodes):
    return SimpleNamespace(num_episodes=saved_episodes)


def _capture_bars(monkeypatch) -> list:
    """Collect the bars episode_progress creates; forced on, as tqdm disables itself when stderr is no terminal."""
    bars = []
    real_tqdm = control_utils.tqdm

    def make_bar(**kwargs):
        bars.append(real_tqdm(**{**kwargs, "file": io.StringIO(), "disable": False}))
        return bars[-1]

    monkeypatch.setattr(control_utils, "tqdm", make_bar)
    return bars


def test_nothing_to_show_without_dataset_and_time_limit():
    with episode_progress(_mp_net(Always(source="grab", target="reset"))) as on_step:
        assert on_step is None


def test_time_limit_bar_counts_seconds_and_names_the_next_episode(monkeypatch):
    bars = _capture_bars(monkeypatch)
    limits = [OnTimeLimit(source="other", target="x", max_steps=5), OnTimeLimit(source="grab", target="x", max_steps=50)]

    with episode_progress(_mp_net(*limits), _dataset(6)) as on_step:
        on_step({"primitive_step": 20})  # ignored: the limit counts "step", absent here like on an intervention step
        on_step({"step": 20})
        assert bars[0].desc == "grab ep 7"
        assert bars[0].total == 5.0
        assert bars[0].n == 2.0


def test_dataset_without_time_limit_counts_elapsed_seconds(monkeypatch):
    bars = _capture_bars(monkeypatch)

    with episode_progress(_mp_net(), _dataset(0)) as on_step:
        on_step({"primitive_step": 30})
        assert bars[0].desc == "grab ep 1"
        assert bars[0].total is None
        assert bars[0].n == 3.0


def test_console_logs_keep_level_and_filters_and_are_restored(capsys):
    root = logging.getLogger()
    handler = logging.StreamHandler()  # stderr, as made by init_logging()
    handler.setLevel(logging.INFO)
    root.addHandler(handler)
    previous_level = root.level
    root.setLevel(logging.NOTSET)
    original_stream = handler.stream
    try:
        mute_log_messages("muted")
        with episode_progress(_mp_net(), _dataset(0)):
            logging.info("shown")
            logging.info("muted line")
            logging.debug("below the handler level")
        assert handler.stream is original_stream
    finally:
        root.setLevel(previous_level)
        root.removeHandler(handler)

    err = capsys.readouterr().err
    assert "shown" in err
    assert "muted line" not in err and "below the handler level" not in err
