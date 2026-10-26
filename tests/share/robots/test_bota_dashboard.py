"""The Bota dashboard picks the viper with the sensor, logs raw and calibrated wrench, and cuts the torque only after the prompt."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from share.robots.viperx.lerobot_robot_viperx import ViperXConfig
from share.robots.viperx.lerobot_robot_viperx.bota import dashboard
from share.robots.viperx.lerobot_robot_viperx.bota.calibration import WRENCH_FEATURES
from share.robots.viperx.lerobot_robot_viperx.config_viperx import BotaConfig

ENABLED = BotaConfig(enable=True, port="/dev/null")


def _env(**robots) -> SimpleNamespace:
    return SimpleNamespace(robot=robots)


def test_only_vipers_with_the_sensor_are_selected():
    with_sensor = ViperXConfig(port="/dev/null", bota=ENABLED)
    env = _env(right=with_sensor, left=ViperXConfig(port="/dev/null"), other=SimpleNamespace())

    assert dashboard.select_bota_vipers(env) == {"right": with_sensor}


def test_an_env_without_a_sensor_is_refused():
    with pytest.raises(ValueError, match="bota.enable"):
        dashboard.select_bota_vipers(_env(left=ViperXConfig(port="/dev/null")))


def test_blueprint_has_a_force_and_a_torque_plot_for_raw_and_calibrated_per_arm():
    blueprint = dashboard.make_blueprint(["right", "left"])

    assert len(blueprint.root_container.contents) == 8


def test_wrench_is_logged_per_axis_with_force_and_torque_norms(monkeypatch):
    logged = {}
    monkeypatch.setattr(dashboard.rr, "log", lambda path, scalars: logged.__setitem__(path, scalars))

    dashboard.log_wrench("right", "raw", np.array([3.0, 4.0, 0.0, 0.0, 0.0, 2.0]))

    assert sorted(logged) == sorted(
        f"bota/right/raw/{quantity}/{leaf}" for quantity in ("force", "torque") for leaf in ("x", "y", "z", "norm")
    )
    assert logged["bota/right/raw/force/norm"].scalars.as_arrow_array().to_pylist() == [5.0]
    assert logged["bota/right/raw/torque/z"].scalars.as_arrow_array().to_pylist() == [2.0]


def test_a_step_logs_the_raw_sensor_wrench_next_to_the_compensated_observation(monkeypatch):
    logged, times = {}, []
    monkeypatch.setattr(dashboard.rr, "log", lambda path, scalars: logged.__setitem__(path, scalars))
    monkeypatch.setattr(dashboard.rr, "set_time", lambda timeline, duration: times.append((timeline, duration)))
    robot = SimpleNamespace(
        get_observation=lambda: {key: float(i) for i, key in enumerate(WRENCH_FEATURES)},
        bota_driver=SimpleNamespace(latest=lambda: SimpleNamespace(wrench=np.full(6, 10.0))),
    )

    dashboard.log_step({"right": robot}, time_s=1.5)

    assert times == [("time", 1.5)]
    assert logged["bota/right/raw/force/x"].scalars.as_arrow_array().to_pylist() == [10.0]
    assert logged["bota/right/calibrated/force/y"].scalars.as_arrow_array().to_pylist() == [1.0]
    assert logged["bota/right/calibrated/torque/z"].scalars.as_arrow_array().to_pylist() == [5.0]


class _Session:
    """Fake ViperX class and console, recording what happens in which order."""

    def __init__(self, monkeypatch, fail_connect_of: str | None = None, steps: int = 2):
        self.events: list[tuple] = []
        events, fail = self.events, fail_connect_of

        class FakeViper:
            def __init__(self, config):
                self.name = config.port
                self.connected = False
                self.payload_model = object()
                self.bus = SimpleNamespace(disable_torque=lambda: events.append(("torque_off", self.name)))

            @property
            def is_connected(self):
                return self.connected

            def connect(self, calibrate):
                if self.name == fail:
                    raise RuntimeError("no arm")
                self.connected = True
                events.append(("connect", self.name, calibrate))

            def disconnect(self):
                self.connected = False
                events.append(("disconnect", self.name))

        def log_step(robots, time_s):
            events.append(("step",))
            if sum(event == ("step",) for event in events) >= steps:
                raise KeyboardInterrupt

        monkeypatch.setattr(dashboard, "ViperX", FakeViper)
        monkeypatch.setattr(dashboard, "log_step", log_step)
        monkeypatch.setattr(dashboard, "init_rerun", lambda **kw: events.append(("rerun",)))
        monkeypatch.setattr(dashboard.rr, "send_blueprint", lambda blueprint: None)
        monkeypatch.setattr(dashboard, "precise_sleep", lambda seconds: None)
        monkeypatch.setattr("builtins.input", lambda *_: events.append(("prompt",)) or "")

    @staticmethod
    def config(*names: str) -> dashboard.BotaDashboardConfig:
        env = _env(**{name: ViperXConfig(port=name, bota=ENABLED) for name in names})
        return dashboard.BotaDashboardConfig(env=env)


def test_torque_goes_off_only_after_connecting_and_the_prompt_and_arms_are_released_at_the_end(monkeypatch):
    session = _Session(monkeypatch)

    dashboard.main.__wrapped__(session.config("right", "left"))

    assert session.events == [
        ("rerun",),
        ("connect", "right", False), ("connect", "left", False),  # no calibration prompt, taring is part of connect
        ("prompt",),
        ("torque_off", "right"), ("torque_off", "left"),
        ("step",), ("step",),
        ("disconnect", "right"), ("disconnect", "left"),
    ]


def test_connected_arms_are_released_when_another_one_fails_to_connect(monkeypatch):
    session = _Session(monkeypatch, fail_connect_of="left")

    with pytest.raises(RuntimeError, match="no arm"):
        dashboard.main.__wrapped__(session.config("right", "left"))

    assert ("disconnect", "right") in session.events
    assert not any(event[0] == "torque_off" for event in session.events), "the torque is never cut"
