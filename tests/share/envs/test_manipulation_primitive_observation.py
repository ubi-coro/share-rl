"""Coverage for ManipulationPrimitive._get_observation()'s camera read behavior.

A camera that can't keep up with the control loop's fps (e.g. a webcam auto-throttling in low
light, see src/share/scripts/benchmarks/episode_save/LOG.md experiment 6) used to stall every
step waiting for a fresh frame via Camera.async_read(). _get_observation() now peeks the
camera's latest buffered frame (Camera.read_latest()) instead: a slightly stale frame is
reused rather than blocking, while a frame older than ManipulationPrimitive.CAMERA_MAX_AGE_MS
still raises, so a genuinely dead camera is not silently fed forever.
"""

from __future__ import annotations

import pytest

from lerobot.utils.constants import OBS_IMAGES
from share.envs.manipulation_primitive.env_manipulation_primitive import ManipulationPrimitive


class StubCamera:
    """Minimal stand-in for a lerobot Camera: records the max_age_ms it was asked for."""

    def __init__(self, frame="frame", *, raises: bool = False):
        self.frame = frame
        self.raises = raises
        self.last_max_age_ms: float | None = None

    def read_latest(self, max_age_ms: int = 500):
        self.last_max_age_ms = max_age_ms
        if self.raises:
            raise TimeoutError("frame too old")
        return self.frame


def _primitive(cameras: dict) -> ManipulationPrimitive:
    return ManipulationPrimitive(task_frame={}, robot_dict={}, cameras=cameras)


def test_get_observation_peeks_cameras_without_blocking():
    """A camera that is keeping up just returns its latest frame, no waiting involved."""
    cam = StubCamera(frame="fresh")
    obs = _primitive({"cam_top": cam})._get_observation()

    assert obs[f"{OBS_IMAGES}.cam_top"] == "fresh"
    # read_latest, not async_read, is the call that matters here: it never blocks for a new frame.
    assert cam.last_max_age_ms == ManipulationPrimitive.CAMERA_MAX_AGE_MS


def test_get_observation_tolerates_a_slow_camera_via_read_latest():
    """A lagging camera (e.g. cam_low under auto-exposure) still gets its buffered frame served,
    as long as it is within CAMERA_MAX_AGE_MS -- read_latest() itself enforces that bound."""
    fast_cam = StubCamera(frame="fast")
    slow_cam = StubCamera(frame="stale-but-within-bound")
    obs = _primitive({"cam_top": fast_cam, "cam_low": slow_cam})._get_observation()

    assert obs[f"{OBS_IMAGES}.cam_top"] == "fast"
    assert obs[f"{OBS_IMAGES}.cam_low"] == "stale-but-within-bound"


def test_get_observation_still_raises_for_a_genuinely_dead_camera():
    """The staleness bound is a safety net, not a silent swallow: a camera whose latest frame
    is older than CAMERA_MAX_AGE_MS must still surface as an error, not feed ancient frames."""
    dead_cam = StubCamera(raises=True)
    with pytest.raises(TimeoutError):
        _primitive({"cam_dead": dead_cam})._get_observation()
