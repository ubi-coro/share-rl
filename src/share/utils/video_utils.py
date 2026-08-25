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
import shutil

logger = logging.getLogger(__name__)


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

            if exc_type is not None and dataset.episode_buffer is not None:
                # Recording was interrupted mid-episode -- drop whatever partial
                # episode was in flight (streaming video state + partial image
                # files) rather than finalizing with a half-written one.
                dataset.clear_episode_buffer(delete_images=True)

            # finalize() closes the parquet writers and, if streaming video encoding
            # is active, the streaming encoder (dataset._streaming_encoder) too.
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
