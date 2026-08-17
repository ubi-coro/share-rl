"""Minimal live rerun visualization of EE position, nothing else -- no MP-net graph, no
trace file, no async session. Duck-types MPNetDebugger's log_reset/log_step/close so it
drops into record_loop's `debugger` param directly."""

from __future__ import annotations

from typing import Any

from lerobot.processor import TransitionKey

from share.utils.transformation_utils import get_robot_pose_from_observation


class EEPoseRerunVisualizer:
    def __init__(self, session_name: str = "ee_pose", spawn: bool = True) -> None:
        import rerun as rr

        self._rr = rr
        rr.init(session_name, spawn=spawn)
        self._step = 0

    def log_reset(self, mp_net: Any, transition: dict[str, Any]) -> None:
        self._log(mp_net, transition)

    def log_step(self, mp_net: Any, transition: dict[str, Any]) -> None:
        self._log(mp_net, transition)

    def _log(self, mp_net: Any, transition: dict[str, Any]) -> None:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        rr = self._rr
        rr.set_time_sequence("step", self._step)
        self._step += 1
        for robot_name in getattr(mp_net, "robot_dict", {}):
            try:
                pose = get_robot_pose_from_observation(observation, robot_name)
            except KeyError:
                continue
            rr.log(f"ee/{robot_name}", rr.Points3D([pose[:3]], radii=0.01, labels=[robot_name]))
            for axis, value in zip(("x", "y", "z", "rx", "ry", "rz"), pose):
                rr.log(f"ee/{robot_name}/{axis}", rr.Scalars(float(value)))

    def close(self) -> None:
        pass
