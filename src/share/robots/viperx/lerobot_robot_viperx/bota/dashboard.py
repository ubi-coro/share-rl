"""Live Rerun view of the raw and the calibrated Bota wrench of every ViperX in an env config.

    python -m share.robots.viperx.lerobot_robot_viperx.bota.dashboard --env.type=aloha_single

The arm torque is switched off after connecting, so the arm can be moved by hand while watching the wrench.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from lerobot.configs import parser
from lerobot.envs.configs import EnvConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun

from share.robots.viperx.lerobot_robot_viperx.bota.calibration import WRENCH_FEATURES
from share.robots.viperx.lerobot_robot_viperx.config_viperx import ViperXConfig
from share.robots.viperx.lerobot_robot_viperx.viperx import ViperX

RATE_HZ = 50.0
WINDOW_S = 10.0  # how much history the plots show


@dataclass
class BotaDashboardConfig:
    env: EnvConfig
    display_ip: str | None = None
    display_port: int | None = None


def select_bota_vipers(env: EnvConfig) -> dict[str, ViperXConfig]:
    """The ViperX configs of the env with the Bota sensor enabled, by robot name."""
    robots = {
        name: config for name, config in (getattr(env, "robot", None) or {}).items()
        if isinstance(config, ViperXConfig) and config.bota.enable
    }
    if not robots:
        raise ValueError("The env config has no ViperX with bota.enable.")
    return robots


def make_blueprint(names: list[str]) -> rrb.Blueprint:
    """Per arm a 2x2 grid: force and torque, raw on top and calibrated below."""
    trailing = rrb.VisibleTimeRange(
        "time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-WINDOW_S),
        end=rrb.TimeRangeBoundary.cursor_relative(),
    )
    views = [
        rrb.TimeSeriesView(
            origin=f"bota/{name}/{kind}/{quantity}", name=f"{name} {kind} {quantity}", time_ranges=trailing
        )
        for name in names for kind in ("raw", "calibrated") for quantity in ("force", "torque")
    ]
    return rrb.Blueprint(rrb.Grid(*views, grid_columns=2))


def log_wrench(name: str, kind: str, wrench: np.ndarray) -> None:
    """Log force and torque (N, Nm) as one scalar per axis plus the norm."""
    for quantity, vector in (("force", wrench[:3]), ("torque", wrench[3:])):
        for axis, value in zip("xyz", vector):
            rr.log(f"bota/{name}/{kind}/{quantity}/{axis}", rr.Scalars(float(value)))
        rr.log(f"bota/{name}/{kind}/{quantity}/norm", rr.Scalars(float(np.linalg.norm(vector))))


def log_step(robots: dict[str, ViperX], time_s: float) -> None:
    """One reading of every arm: the sensor's raw wrench next to the compensated one from the observation."""
    rr.set_time("time", duration=time_s)
    for name, robot in robots.items():
        observation = robot.get_observation()
        log_wrench(name, "raw", robot.bota_driver.latest().wrench)
        log_wrench(name, "calibrated", np.array([observation[key] for key in WRENCH_FEATURES]))


@parser.wrap()
def main(cfg: BotaDashboardConfig) -> None:
    robots = {name: ViperX(config) for name, config in select_bota_vipers(cfg.env).items()}
    init_rerun(session_name="bota_dashboard", ip=cfg.display_ip, port=cfg.display_port)
    rr.send_blueprint(make_blueprint(list(robots)))
    try:
        for name, robot in robots.items():
            robot.connect(calibrate=False)  # taring happens here, while the torque still holds the arm
            if robot.payload_model is None:
                logging.warning(f"{name}: no payload fit yet, the calibrated wrench stays NaN until it is calibrated.")
        input("The torque is on. Support the arm(s), then press Enter to switch it off. ")
        for robot in robots.values():
            robot.bus.disable_torque()
        print("Torque is off, showing the wrench. Ctrl-C to stop.")

        start = time.perf_counter()
        while True:
            tick = time.perf_counter()
            log_step(robots, tick - start)
            precise_sleep(1.0 / RATE_HZ - (time.perf_counter() - tick))
    except KeyboardInterrupt:
        pass
    finally:
        for robot in robots.values():
            if robot.is_connected:
                robot.disconnect()


if __name__ == "__main__":
    import experiments  # registers the env configs

    main()
