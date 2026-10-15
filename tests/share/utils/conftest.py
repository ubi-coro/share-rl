"""Test-local pynput shim for share.utils tests in headless environments."""

from __future__ import annotations

import sys
import types


def _install_pynput_keyboard_stub() -> None:
    if "pynput" in sys.modules:
        return

    keyboard_module = types.ModuleType("pynput.keyboard")

    class Key:
        left = "left"
        right = "right"
        up = "up"
        down = "down"
        enter = "enter"
        shift = "shift"
        shift_r = "shift_r"
        ctrl_l = "ctrl_l"
        ctrl_r = "ctrl_r"

    keyboard_module.Key = Key

    pynput_module = types.ModuleType("pynput")
    pynput_module.keyboard = keyboard_module

    sys.modules["pynput"] = pynput_module
    sys.modules["pynput.keyboard"] = keyboard_module


_install_pynput_keyboard_stub()
