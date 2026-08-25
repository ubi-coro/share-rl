"""Drive bimanual_pick with simulated arms + a real SpaceMouse.

ARCHIVED along with experiments/envs/archive/bimanual_pick.py -- bimanual_pick is no longer
registered (see experiments/envs/__init__.py), so this needs the archive path explicitly.

Same env config and code paths as the real robots, just SimUR (kinematic-only, no hardware)
instead of URConfig. Needs a physically connected SpaceMouse. Once this looks right, drop
`mock=True` (or point the robot IPs at the real arms) and run the same graph for real.

    python -m experiments.scripts.archive.preview_bimanual_pick
"""

from __future__ import annotations

from experiments.envs.archive.bimanual_pick import BimanualPickEnvConfig
from share.scripts.record import record_loop
from share.teleoperators import TeleopEvents, has_event


def main() -> None:
    cfg = BimanualPickEnvConfig(mock=True)
    mp_net = cfg.make()
    mp_net.set_step_info({TeleopEvents.IS_INTERVENTION: True})

    try:
        while True:
            info = record_loop(
                mp_net=mp_net,
                datasets={},
                policies={},
                preprocessors={},
                postprocessors={},
                force_intervention=True,
            )
            if has_event(info, TeleopEvents.STOP_RECORDING):
                break
    finally:
        mp_net.close()


if __name__ == "__main__":
    main()
