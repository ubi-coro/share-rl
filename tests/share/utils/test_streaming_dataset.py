"""Round-trip tests for the zero-copy streaming video encoder used by ``record_beast.py``.

Small synthetic episodes go through ``LeRobotDataset.create`` / ``resume`` exactly as
``make_policies_and_datasets`` configures them for ``dataset.streaming_encoding=true``.
"""

from __future__ import annotations

import queue

import numpy as np
import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from share.utils.video_utils import ZeroCopyStreamingVideoEncoder, use_zero_copy_streaming_encoder

FPS = 30
HEIGHT, WIDTH = 48, 64
FEATURES = {
    "observation.images.cam": {"dtype": "video", "shape": (3, HEIGHT, WIDTH), "names": ["channels", "height", "width"]},
    "observation.state": {"dtype": "float32", "shape": (4,), "names": None},
    "action": {"dtype": "float32", "shape": (4,), "names": None},
}


def _frame(t: int) -> dict:
    # float32 CHW in [0, 1] with channels-last strides, like ObservationProcessor._process_image
    hwc = np.full((HEIGHT, WIDTH, 3), (t * 7) % 255, dtype=np.uint8)
    image = torch.from_numpy(hwc).permute(2, 0, 1).to(torch.float32) / 255.0
    return {
        "observation.images.cam": image,
        "observation.state": torch.full((4,), float(t)),
        "action": torch.full((4,), float(-t)),
        "task": "test task",
    }


def _record(dataset: LeRobotDataset, num_frames: int) -> None:
    for t in range(num_frames):
        dataset.add_frame(_frame(t))
    dataset.save_episode()


def _create(root, **kwargs) -> LeRobotDataset:
    dataset = LeRobotDataset.create(
        "test/streaming",
        FPS,
        features=FEATURES,
        root=root,
        streaming_encoding=True,
        encoder_queue_maxsize=64,
        encoder_threads=1,
        image_writer_threads=0,
        **kwargs,
    )
    return use_zero_copy_streaming_encoder(dataset)


def test_swap_installs_zero_copy_encoder_with_same_settings(tmp_path):
    dataset = _create(tmp_path / "ds")
    encoder = dataset.writer._streaming_encoder
    assert isinstance(encoder, ZeroCopyStreamingVideoEncoder)
    assert (encoder.fps, encoder.queue_maxsize, encoder.encoder_threads) == (FPS, 64, 1)
    dataset.finalize()


def test_swap_is_a_noop_without_streaming(tmp_path):
    dataset = LeRobotDataset.create("test/plain", FPS, features=FEATURES, root=tmp_path / "ds", image_writer_threads=1)
    assert use_zero_copy_streaming_encoder(dataset).writer._streaming_encoder is None
    dataset.finalize()


def test_episodes_round_trip_and_resume(tmp_path):
    root = tmp_path / "ds"
    dataset = _create(root)
    _record(dataset, 20)
    _record(dataset, 15)
    dataset.finalize()

    reopened = LeRobotDataset("test/streaming", root=root)
    assert reopened.num_episodes == 2
    assert reopened.num_frames == 35
    item = reopened[0]
    assert item["observation.images.cam"].shape == (3, HEIGHT, WIDTH)
    assert float(item["observation.state"][0]) == 0.0

    resumed = LeRobotDataset.resume(
        "test/streaming", root=root, streaming_encoding=True, encoder_queue_maxsize=64, encoder_threads=1
    )
    use_zero_copy_streaming_encoder(resumed)
    _record(resumed, 10)
    resumed.finalize()

    final = LeRobotDataset("test/streaming", root=root)
    assert final.num_episodes == 3
    assert final.num_frames == 45
    # last frame of the resumed episode decodes and matches its state
    assert float(final[44]["observation.state"][0]) == 9.0
    assert final[44]["observation.images.cam"].shape == (3, HEIGHT, WIDTH)


def test_clear_episode_buffer_discards_a_rerecorded_episode(tmp_path):
    root = tmp_path / "ds"
    dataset = _create(root)
    for t in range(12):
        dataset.add_frame(_frame(t))
    dataset.clear_episode_buffer()  # RERECORD_EPISODE
    _record(dataset, 18)
    dataset.finalize()

    reopened = LeRobotDataset("test/streaming", root=root)
    assert (reopened.num_episodes, reopened.num_frames) == (1, 18)
    assert not list(root.glob("tmp*"))  # no leftover temp video dirs


def test_full_queue_drops_frames_and_logs_an_error(tmp_path, caplog):
    dataset = _create(tmp_path / "ds")
    encoder = dataset.writer._streaming_encoder
    dataset.add_frame(_frame(0))  # starts the episode and its encoder threads

    key = "observation.images.cam"

    def always_full(item, timeout=None):
        raise queue.Full

    encoder._frame_queues[key].put = always_full
    with caplog.at_level("ERROR"):
        encoder.feed_frame(key, np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    assert encoder._dropped_frames[key] == 1
    assert "dropping frames" in caplog.text

    dataset.clear_episode_buffer()
    dataset.finalize()


# --- wiring in make_policies_and_datasets ---------------------------------------------------------------


def _stub_cfg(root, *, streaming: bool, resume: bool, video_files_size_in_mb: int | None = None):
    from types import SimpleNamespace

    from lerobot.configs.types import FeatureType, PolicyFeature

    from share.configs.record import DatasetRecordConfig

    features = {
        "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, HEIGHT, WIDTH)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
    }
    primitive = SimpleNamespace(is_adaptive=True, features=features, num_cameras=1, policy=None)
    env = SimpleNamespace(
        primitives={"folding": primitive}, reset_primitive="reset", fps=FPS, type="test_robot"
    )
    dataset = DatasetRecordConfig(
        repo_id="test/wiring",
        root=str(root),
        streaming_encoding=streaming,
        encoder_queue_maxsize=77,
        encoder_threads=2,
        video_files_size_in_mb=video_files_size_in_mb,
    )
    return SimpleNamespace(env=env, dataset=dataset, resume=resume, rename_map={})


def _frame_with_reward(t: int) -> dict:
    # env_to_dataset_features adds reward/done features on top of the env's own
    return {**_frame(t), "next.reward": np.array([0.0], dtype=np.float32), "next.done": np.array([False])}


@pytest.mark.parametrize("streaming", [True, False])
def test_make_policies_and_datasets_wires_streaming_options(tmp_path, streaming):
    from share.utils.control_utils import make_policies_and_datasets

    datasets, *_ = make_policies_and_datasets(_stub_cfg(tmp_path, streaming=streaming, resume=False))
    writer = datasets["folding"].writer
    if streaming:
        encoder = writer._streaming_encoder
        assert isinstance(encoder, ZeroCopyStreamingVideoEncoder)
        assert (encoder.queue_maxsize, encoder.encoder_threads) == (77, 2)
        assert writer.image_writer is None  # no PNGs are written in streaming mode
    else:
        assert writer._streaming_encoder is None
        assert writer.image_writer is not None
    datasets["folding"].finalize()


def test_make_policies_and_datasets_resume_keeps_streaming(tmp_path):
    from share.utils.control_utils import make_policies_and_datasets

    first, *_ = make_policies_and_datasets(_stub_cfg(tmp_path, streaming=True, resume=False))
    dataset = first["folding"]
    for t in range(10):
        dataset.add_frame(_frame_with_reward(t))
    dataset.save_episode()
    dataset.finalize()

    resumed, *_ = make_policies_and_datasets(_stub_cfg(tmp_path, streaming=True, resume=True))
    assert isinstance(resumed["folding"].writer._streaming_encoder, ZeroCopyStreamingVideoEncoder)
    for t in range(6):
        resumed["folding"].add_frame(_frame_with_reward(t))
    resumed["folding"].save_episode()
    resumed["folding"].finalize()

    final = LeRobotDataset("test/wiring-folding", root=tmp_path / "folding")
    assert (final.num_episodes, final.num_frames) == (2, 16)


def test_make_policies_and_datasets_applies_video_files_size(tmp_path):
    from share.utils.control_utils import make_policies_and_datasets

    datasets, *_ = make_policies_and_datasets(
        _stub_cfg(tmp_path, streaming=True, resume=False, video_files_size_in_mb=1)
    )
    assert datasets["folding"].meta.get_chunk_settings()["video_files_size_in_mb"] == 1
    datasets["folding"].finalize()


def test_make_policies_and_datasets_leaves_default_video_files_size(tmp_path):
    from lerobot.datasets.utils import DEFAULT_VIDEO_FILE_SIZE_IN_MB
    from share.utils.control_utils import make_policies_and_datasets

    datasets, *_ = make_policies_and_datasets(_stub_cfg(tmp_path, streaming=True, resume=False))
    assert datasets["folding"].meta.get_chunk_settings()["video_files_size_in_mb"] == DEFAULT_VIDEO_FILE_SIZE_IN_MB
    datasets["folding"].finalize()
