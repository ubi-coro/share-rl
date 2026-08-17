"""Drive bimanual_pick with simulated arms + a real SpaceMouse, visualized live via rerun.

Same env config and code paths as the real robots, just SimUR (kinematic-only, no hardware)
instead of URConfig. Needs a physically connected SpaceMouse. Once this looks right, drop
`mock=True` (or point the robot IPs at the real arms) and run the same graph for real.

    python -m experiments.scripts.preview_bimanual_pick
"""

from __future__ import annotations

from experiments.envs.bimanual_pick import BimanualPickEnvConfig
from share.debug.ee_pose_rerun import EEPoseRerunVisualizer
from share.envs.manipulation_primitive_net.env_manipulation_primitive_net import ManipulationPrimitiveNet
from share.scripts.record import record_loop
from share.teleoperators import TeleopEvents, has_event


def main() -> None:
    cfg = BimanualPickEnvConfig(mock=True)
    mp_net = ManipulationPrimitiveNet(cfg)
    mp_net.set_step_info({TeleopEvents.IS_INTERVENTION: True})

    debugger = EEPoseRerunVisualizer(session_name="preview_bimanual_pick")

    try:
        while True:
            info = record_loop(
                mp_net=mp_net,
                datasets={},
                policies={},
                preprocessors={},
                postprocessors={},
                force_intervention=True,
                debugger=debugger,
            )
            if has_event(info, TeleopEvents.STOP_RECORDING):
                break
    finally:
        debugger.close()
        mp_net.close()


if __name__ == "__main__":
    main()
