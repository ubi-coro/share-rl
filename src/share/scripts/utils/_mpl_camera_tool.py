"""Shared scaffolding for the interactive camera calibration tools.

Used by find_led_params.py (pick the LED pixel) and find_crop_params.py (pick the crop
centers). Both open live camera views and need the same backend guard and luminance math.
"""

from __future__ import annotations

import matplotlib
import numpy as np


def require_interactive_matplotlib_backend() -> None:
    """Fail early and actionably rather than opening a window nobody can see."""
    if matplotlib.get_backend().lower() == "agg":
        raise RuntimeError(
            f"Matplotlib is using the non-interactive backend {matplotlib.get_backend()!r}, "
            "so it cannot open a live calibration window.\n\n"
            "Fix it without sudo in the same Python environment used to run this script. "
            "For example, install a GUI backend:\n"
            "    python -m pip install PyQt6\n\n"
            "Then rerun with:\n"
            "    MPLBACKEND=qtagg python <script>"
        )


def connect_camera(camera_configs, camera_key: str):
    """Connect one camera out of a config dict, by key."""
    if camera_key not in camera_configs:
        raise KeyError(f"Camera '{camera_key}' not found. Available: {list(camera_configs)}")

    from lerobot.cameras import make_cameras_from_configs

    cameras = make_cameras_from_configs({camera_key: camera_configs[camera_key]})
    camera = cameras[camera_key]
    camera.connect()
    return camera


def luminance(image: np.ndarray, x: int, y: int, radius: int) -> float:
    """Mean Rec. 601 luma over the (2*radius+1)^2 patch centred on (x, y)."""
    height, width = image.shape[:2]
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch = image[y0:y1, x0:x1].astype(np.float32)
    return float(
        0.299 * patch[..., 0].mean() + 0.587 * patch[..., 1].mean() + 0.114 * patch[..., 2].mean()
    )
