"""Test-local shims for share env tests.

These tests exercise config and processor logic in headless CI environments, so
we replace ``pynput`` with a tiny stub before modules under ``share.envs`` are
imported. The production code only needs ``keyboard.Key.*`` constants for type
annotations and static event mappings in config objects.
"""

from __future__ import annotations

import sys
import types


def _install_pynput_keyboard_stub() -> None:
    if "pynput" in sys.modules:
        return

    keyboard_module = types.ModuleType("pynput.keyboard")

    class _KeySentinel:
        """Non-str marker so isinstance(Key.x, str) is False, like real pynput."""

        def __init__(self, name: str):
            self._name = name

        def __repr__(self) -> str:
            return f"Key.{self._name}"

    class Key:
        left = _KeySentinel("left")
        right = _KeySentinel("right")
        up = _KeySentinel("up")
        down = _KeySentinel("down")
        enter = _KeySentinel("enter")
        shift = _KeySentinel("shift")
        shift_r = _KeySentinel("shift_r")
        ctrl_l = _KeySentinel("ctrl_l")
        ctrl_r = _KeySentinel("ctrl_r")
        space = _KeySentinel("space")

    keyboard_module.Key = Key

    class KeyCode:
        """Stand-in for pynput.keyboard.KeyCode -- character keys, compared by .char."""

        def __init__(self, char: str | None = None):
            self.char = char

        @classmethod
        def from_char(cls, char: str) -> "KeyCode":
            return cls(char=char)

        def __eq__(self, other: object) -> bool:
            return isinstance(other, KeyCode) and other.char == self.char

        def __hash__(self) -> int:
            return hash(("KeyCode", self.char))

        def __repr__(self) -> str:
            return f"KeyCode.from_char({self.char!r})"

    keyboard_module.KeyCode = KeyCode

    class Listener:
        """Inert stand-in for pynput.keyboard.Listener -- no real OS hook, no thread."""

        def __init__(self, on_press=None, on_release=None):
            self.on_press = on_press
            self.on_release = on_release
            self.daemon = False

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def join(self, timeout=None) -> None:
            pass

    keyboard_module.Listener = Listener

    pynput_module = types.ModuleType("pynput")
    pynput_module.keyboard = keyboard_module

    sys.modules["pynput"] = pynput_module
    sys.modules["pynput.keyboard"] = keyboard_module


_install_pynput_keyboard_stub()
