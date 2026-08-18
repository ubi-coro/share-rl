from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from typing import Any

import numpy as np

from .controller import load_franky


class FrankaHandWorker(mp.Process):
    """Blocking Franka Hand operations isolated from the arm controller."""

    def __init__(self, config: Any):
        super().__init__(name="FrankaHandWorker")
        self.config = config
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.unexpected_exit_event = mp.Event()
        self.command_queue: mp.Queue[float] = mp.Queue(maxsize=16)
        self.error_queue: mp.Queue[str] = mp.Queue(maxsize=4)
        self.width = mp.Value("d", 0.0)
        self.max_width = mp.Value("d", 0.0)
        self.is_grasped = mp.Value("b", False)
        self._home_on_start = mp.Value("b", False)

    @property
    def is_ready(self) -> bool:
        return (
            self.ready_event.is_set()
            and self.is_alive()
            and not self.unexpected_exit_event.is_set()
        )

    def start(self, *, home: bool = False, wait: bool = True) -> None:
        self._home_on_start.value = bool(home)
        super().start()
        if wait and not self.ready_event.wait(timeout=self.config.launch_timeout):
            self.check_health()
            raise TimeoutError("Timed out while connecting to the Franka Hand")

    def move(self, position: float) -> None:
        self.check_health()
        normalized = float(np.clip(position, 0.0, 1.0))
        try:
            self.command_queue.put_nowait(normalized)
        except queue.Full:
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                pass
            self.command_queue.put_nowait(normalized)

    def home(self) -> None:
        self.check_health()
        self.command_queue.put(float("nan"))

    def get_state(self) -> dict[str, float | bool]:
        self.check_health()
        maximum = float(self.max_width.value)
        normalized = 0.0 if maximum <= 0.0 else 1.0 - float(self.width.value) / maximum
        return {
            "position": float(np.clip(normalized, 0.0, 1.0)),
            "width": float(self.width.value),
            "max_width": maximum,
            "is_grasped": bool(self.is_grasped.value),
        }

    def check_health(self) -> None:
        try:
            message = self.error_queue.get_nowait()
        except queue.Empty:
            message = None
        if message is not None:
            raise RuntimeError(message)
        if self.unexpected_exit_event.is_set():
            raise RuntimeError("The Franka Hand worker exited unexpectedly")

    def stop(self) -> None:
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout=self.config.launch_timeout)
            if self.is_alive():
                self.terminate()
                self.join(timeout=2.0)

    def run(self) -> None:
        hand = None
        normal_stop = False
        try:
            franky = load_franky()
            hand = franky.Gripper(self.config.robot_ip)
            if self._home_on_start.value:
                hand.homing()
            self._publish(hand)
            self.ready_event.set()
            period = 1.0 / float(self.config.gripper_frequency)

            while not self.stop_event.is_set():
                parent = mp.parent_process()
                if parent is not None and not parent.is_alive():
                    normal_stop = True
                    break
                try:
                    position = self.command_queue.get(timeout=period)
                except queue.Empty:
                    self._publish(hand)
                    continue

                while True:
                    try:
                        position = self.command_queue.get_nowait()
                    except queue.Empty:
                        break

                if np.isnan(position):
                    hand.homing()
                    self._publish(hand)
                    continue
                state = hand.state
                target_width = (1.0 - position) * float(state.max_width)
                if target_width < float(state.width):
                    hand.grasp(
                        target_width,
                        float(self.config.gripper_speed),
                        float(self.config.gripper_force),
                        float(self.config.gripper_epsilon_inner),
                        float(self.config.gripper_epsilon_outer),
                    )
                else:
                    hand.move(target_width, float(self.config.gripper_speed))
                self._publish(hand)
            normal_stop = True
        except BaseException:
            self.unexpected_exit_event.set()
            try:
                self.error_queue.put_nowait(traceback.format_exc())
            except queue.Full:
                pass
        finally:
            self.ready_event.clear()
            if hand is not None:
                try:
                    hand.stop()
                except BaseException:
                    pass
            if not normal_stop and not self.unexpected_exit_event.is_set():
                self.unexpected_exit_event.set()

    def _publish(self, hand: Any) -> None:
        state = hand.state
        self.width.value = float(state.width)
        self.max_width.value = float(state.max_width)
        self.is_grasped.value = bool(state.is_grasped)
