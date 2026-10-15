#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import queue
import shutil

from lerobot.datasets.video_utils import StreamingVideoEncoder

logger = logging.getLogger(__name__)


class ZeroCopyStreamingVideoEncoder(StreamingVideoEncoder):
    """`StreamingVideoEncoder` whose `feed_frame` does not copy the frame on the calling thread.

    Upstream copies every frame so camera drivers may reuse their buffers. With four 640x480 float32 cameras
    that copy costs ~10 ms per step on the control thread. Frames fed from `record_loop` are fresh tensors
    that nobody mutates afterwards (the PNG `AsyncImageWriter` this replaces also keeps unprotected references
    until a worker writes them), so the encoder thread can read them directly. Do not use this with sources
    that overwrite the array after `add_frame`.
    """

    def feed_frame(self, video_key: str, image) -> None:
        if not self._episode_active:
            raise RuntimeError("No active episode. Call start_episode() first.")

        thread = self._threads[video_key]
        if not thread.is_alive():
            try:
                status, msg = self._result_queues[video_key].get_nowait()
                if status == "error":
                    raise RuntimeError(f"Encoder thread for {video_key} crashed: {msg}")
            except queue.Empty:
                pass
            raise RuntimeError(f"Encoder thread for {video_key} is not alive")

        try:
            self._frame_queues[video_key].put(image, timeout=0.1)
        except queue.Full:
            self._dropped_frames[video_key] = self._dropped_frames.get(video_key, 0) + 1
            if self._dropped_frames[video_key] == 1:
                logger.error(
                    f"Encoder queue full for {video_key}: dropping frames, the video will be out of sync "
                    f"with the recorded data. Increase dataset.encoder_queue_maxsize or reduce encoder load."
                )

    def finish_episode(self):
        dropped = {k: n for k, n in self._dropped_frames.items() if n > 0}
        if dropped:
            logger.error(f"Episode has dropped video frames {dropped}: discard it (re-record).")
        return super().finish_episode()


def use_zero_copy_streaming_encoder(dataset):
    """Swap the dataset writer's streaming encoder for a `ZeroCopyStreamingVideoEncoder` with the same settings."""
    writer = dataset.writer
    old = getattr(writer, "_streaming_encoder", None)
    if old is None:
        return dataset
    writer._streaming_encoder = ZeroCopyStreamingVideoEncoder(
        fps=old.fps,
        vcodec=old.vcodec,
        pix_fmt=old.pix_fmt,
        g=old.g,
        crf=old.crf,
        preset=old.preset,
        queue_maxsize=old.queue_maxsize,
        encoder_threads=old.encoder_threads,
    )
    return dataset


class MultiVideoEncodingManager:
    """
    Context manager that ensures proper video encoding and data cleanup even if exceptions occur.

    This manager handles:
    - Batch encoding for any remaining episodes when recording interrupted
    - Cleaning up temporary image files from interrupted episodes
    - Removing empty image directories

    Args:
        dataset: The LeRobotDataset instance
    """

    def __init__(self, datasets: dict):
        self.datasets = datasets

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Handle any remaining episodes that haven't been batch encoded
        for dataset in self.datasets.values():

            writer = dataset.writer
            if writer is not None:
                if exc_type is not None and writer._streaming_encoder is not None:
                    writer.cancel_pending_videos()

                # finalize() handles flush_pending_videos + parquet + metadata
                dataset.finalize()

                # Clean up episode images if recording was interrupted (only for non-streaming mode)
                if exc_type is not None and writer._streaming_encoder is None:
                    writer.cleanup_interrupted_episode(dataset.num_episodes)
            else:
                dataset.finalize()

            # Clean up any remaining images directory if it's empty
            img_dir = dataset.root / "images"
            if img_dir.exists():
                png_files = list(img_dir.rglob("*.png"))
                if len(png_files) == 0:
                    shutil.rmtree(img_dir)
                    logger.debug("Cleaned up empty images directory")
                else:
                    logger.debug(f"Images directory is not empty, containing {len(png_files)} PNG files")

        return False  # Don't suppress the original exception
