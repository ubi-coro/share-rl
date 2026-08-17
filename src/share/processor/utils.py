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


# Special keys with a direct pynput.keyboard.Key equivalent -- keep this list in sync with
# what env configs actually map (grep key_mapping= across experiments/envs).
_EVDEV_SPECIAL_KEY_NAMES: dict[int, str] = {
    evdev.ecodes.KEY_SPACE: "space",
    evdev.ecodes.KEY_LEFT: "left",
    evdev.ecodes.KEY_RIGHT: "right",
    evdev.ecodes.KEY_UP: "up",
    evdev.ecodes.KEY_DOWN: "down",
    evdev.ecodes.KEY_ENTER: "enter",
    evdev.ecodes.KEY_LEFTSHIFT: "shift",
    evdev.ecodes.KEY_RIGHTSHIFT: "shift_r",
    evdev.ecodes.KEY_LEFTCTRL: "ctrl_l",
    evdev.ecodes.KEY_RIGHTCTRL: "ctrl_r",
}


def _evdev_code_to_pynput_key(code: int) -> Any:
    """Translate one evdev keycode into the same pynput Key/KeyCode object
    EventConfig.key_mapping already uses, so callers don't care which backend fired."""
    from pynput import keyboard

    special_name = _EVDEV_SPECIAL_KEY_NAMES.get(code)
    if special_name is not None:
        return getattr(keyboard.Key, special_name)

    key_name = evdev.ecodes.KEY.get(code)
    if isinstance(key_name, list):
        key_name = key_name[0] if key_name else None
    if isinstance(key_name, str) and key_name.startswith("KEY_") and len(key_name) == 5:
        return keyboard.KeyCode.from_char(key_name[-1].lower())
    return None


def _is_keyboard_like(device: "evdev.InputDevice") -> bool:
    keys = device.capabilities().get(evdev.ecodes.EV_KEY, [])
    return evdev.ecodes.KEY_SPACE in keys and evdev.ecodes.KEY_A in keys


class EvdevKeyboardListener:
    """pynput.keyboard.Listener-compatible keyboard watcher backed by raw evdev events.

    pynput's global listener needs X11 (or a compositor that forwards it) -- on Wayland it
    silently receives nothing, no error. evdev reads /dev/input/eventN directly, below the
    display server, so it works on both -- same technique as FootSwitchHandler above. Needs
    the user in the `input` group (`sudo usermod -aG input $USER`, then re-login) to read the
    device files; falls back to a logged warning (not a crash) if none are accessible.
    """

    def __init__(self, on_press=None, on_release=None, device_paths: list[str] | None = None):
        self.on_press = on_press
        self.on_release = on_release
        self.daemon = True
        self._device_paths = device_paths
        self._threads: list[threading.Thread] = []
        self._running = True

    def _discover_devices(self) -> list[str]:
        paths = []
        for path in evdev.list_devices():
            try:
                if _is_keyboard_like(evdev.InputDevice(path)):
                    paths.append(path)
            except Exception:
                continue
        return paths

    def start(self) -> None:
        device_paths = self._device_paths if self._device_paths is not None else self._discover_devices()
        if not device_paths:
            logging.warning(
                "EvdevKeyboardListener found no keyboard-like /dev/input device -- keyboard "
                "shortcuts (space/left/down/...) won't work. Check you're in the 'input' "
                "group: sudo usermod -aG input $USER, then log out and back in."
            )
        for path in device_paths:
            thread = threading.Thread(target=self._run, args=(path,), daemon=True)
            thread.start()
            self._threads.append(thread)

    def _run(self, device_path: str) -> None:
        try:
            device = evdev.InputDevice(device_path)
        except Exception:
            logging.warning(f"EvdevKeyboardListener could not open {device_path}", exc_info=True)
            return
        self._run_with_device(device)

    def _run_with_device(self, device: Any) -> None:
        for event in device.read_loop():
            if not self._running:
                break
            if event.type != evdev.ecodes.EV_KEY or event.value not in (0, 1):
                continue
            key = _evdev_code_to_pynput_key(event.code)
            if key is None:
                continue
            callback = self.on_press if event.value == 1 else self.on_release
            if callback is not None:
                callback(key)

    def stop(self) -> None:
        self._running = False

    def join(self, timeout: float | None = None) -> None:
        for thread in self._threads:
            thread.join(timeout=timeout)


class StdinKeyboardListener:
    """pynput.keyboard.Listener-compatible keyboard watcher reading this process's own
    terminal in raw/cbreak mode -- needs no special permissions (unlike evdev's
    /dev/input access), just an interactive terminal. Only sees keys typed while that
    terminal has focus (not a true OS-global listener), and there is no separate release
    event -- on_release never fires.
    """

    _ARROW_KEY_BY_FINAL_BYTE = {"A": "up", "B": "down", "C": "right", "D": "left"}

    def __init__(self, on_press=None, on_release=None):
        self.on_press = on_press
        self.on_release = on_release
        self.daemon = True
        self._running = True
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import sys

        if not sys.stdin.isatty():
            logging.warning("StdinKeyboardListener: stdin is not a terminal, keyboard shortcuts disabled.")
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import select
        import sys
        import termios
        import tty

        from pynput import keyboard

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self._running:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not ready:
                    continue
                char = sys.stdin.read(1)
                key = self._read_key(char, keyboard)
                if key is not None and self.on_press is not None:
                    self.on_press(key)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _read_key(self, char: str, keyboard: Any) -> Any:
        import sys

        if char == "\x1b":  # possible arrow-key escape sequence: ESC [ <A|B|C|D>
            import select

            if not select.select([sys.stdin], [], [], 0.05)[0]:
                return None  # bare Escape
            if sys.stdin.read(1) != "[":
                return None
            final_byte = sys.stdin.read(1)
            name = self._ARROW_KEY_BY_FINAL_BYTE.get(final_byte)
            return getattr(keyboard.Key, name) if name is not None else None
        if char == " ":
            return keyboard.Key.space
        if char in ("\n", "\r"):
            return keyboard.Key.enter
        if char.isprintable():
            return keyboard.KeyCode.from_char(char)
        return None

    def stop(self) -> None:
        self._running = False

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)


class CompositeKeyboardListener:
    """Runs several pynput.keyboard.Listener-compatible backends in parallel, feeding the
    same callbacks -- e.g. evdev when available, stdin as a permission-free fallback."""

    def __init__(self, listeners: list[Any]) -> None:
        self._listeners = listeners
        self.daemon = True

    def start(self) -> None:
        for listener in self._listeners:
            listener.start()

    def stop(self) -> None:
        for listener in self._listeners:
            listener.stop()

    def join(self, timeout: float | None = None) -> None:
        for listener in self._listeners:
            listener.join(timeout=timeout)


class FootSwitchHandler:
    def __init__(self, device_path="/dev/input/event0", event_names: tuple[str] = (TeleopEvents.SUCCESS, ), toggle: bool = False):
        self.device = evdev.InputDevice(device_path)
        print("init FootSwitchHandler")
        # self.device.grab()
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
        self.device.ungrab()

    def reset(self):
        self.events = {name: False for name in self.event_names}
