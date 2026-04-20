import logging
import threading
from typing import Any

import evdev
import torch
from lerobot.processor.hil_processor import GRIPPER_KEY

from share.teleoperators.utils import TeleopEvents


def flatten_nested_policy_action(
    action: dict[str, dict[str, Any]],
    task_frame: dict[str, "TaskFrame"],
    gripper_enable: dict[str, bool],
    like: Any | None = None,
) -> torch.Tensor:
    """Flatten keyed per-robot learning-space actions into one policy tensor."""

    # first tensor in a dict helper
    def _first_tensor(value: Any) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, dict):
            for nested in value.values():
                tensor = _first_tensor(nested)
                if tensor is not None:
                    return tensor
        return None

    tensor = _first_tensor(action)
    if tensor is None and like is not None:
        tensor = _first_tensor(like) if isinstance(like, dict) else like if isinstance(like, torch.Tensor) else None
    dtype = tensor.dtype if isinstance(tensor, torch.Tensor) else torch.float32
    device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")

    values: list[torch.Tensor] = []
    for name, frame in task_frame.items():
        robot_action = action.get(name, {})
        for key in policy_action_keys_for_robot(frame, gripper_enable[name]):
            if key not in robot_action:
                raise ValueError(f"Missing policy action key '{name}.{key}' while flattening action dict")
            values.append(torch.as_tensor(robot_action[key], dtype=dtype, device=device).reshape(1))

    if not values:
        return torch.empty(0, dtype=dtype, device=device)
    return torch.cat(values)


def policy_action_keys_for_robot(frame: "TaskFrame", gripper_enable: bool) -> list[str]:
    """Return one robot's ordered learning-space keys, optionally with gripper."""
    keys = list(frame.policy_action_keys())
    gripper_key = f"{GRIPPER_KEY}.pos"
    if gripper_enable and gripper_key not in keys:
        keys.append(f"{GRIPPER_KEY}.pos")
    return keys


class FootSwitchHandler:
    def __init__(self, device_path="/dev/input/event0", event_names: tuple[str] = (TeleopEvents.SUCCESS, ), toggle: bool = False):
        self.device = evdev.InputDevice(device_path)
        self.events = {name: False for name in event_names}
        self.toggle = toggle
        self.event_names = event_names
        self.running = True

    def start(self):
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def _run(self):
        logging.info(f"Listening for foot switch events from {self.device.name} ({self.device.path})...")
        for event in self.device.read_loop():
            if not self.running:
                break
            if event.type == evdev.ecodes.EV_KEY:
                key_event = evdev.categorize(event)
                if key_event.keystate == 1:  # Key down
                    if self.toggle:
                        if self.events[self.event_names[0]]:
                            logging.info(f"Foot switch pressed again - {self.event_names} toggled OFF")
                            for name in self.event_names:
                                self.events[name] = False
                        else:
                            logging.info(f"Foot switch pressed - {self.event_names} toggled ON")
                            for name in self.event_names:
                                self.events[name] = True
                    else:
                        logging.info(f"Foot switch pressed - {self.event_names} ON")
                        for name in self.event_names:
                            self.events[name] = True
                elif key_event.keystate == 0 and not self.toggle:  # Key release
                    logging.info(f"Foot switch released - {self.event_names} OFF")
                    for name in self.event_names:
                        self.events[name] = False

    def stop(self):
        self.running = False

    def reset(self):
        self.events = {name: False for name in self.event_names}
