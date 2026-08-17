"""Tests for the evdev-backed keyboard listener (the pynput/Wayland replacement)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import evdev
from pynput import keyboard as kb

from share.processor.utils import EvdevKeyboardListener, _evdev_code_to_pynput_key, _is_keyboard_like


def test_evdev_code_translates_special_keys_to_pynput_key():
    assert _evdev_code_to_pynput_key(evdev.ecodes.KEY_SPACE) is kb.Key.space
    assert _evdev_code_to_pynput_key(evdev.ecodes.KEY_LEFT) is kb.Key.left
    assert _evdev_code_to_pynput_key(evdev.ecodes.KEY_DOWN) is kb.Key.down


def test_evdev_code_translates_letter_keys_to_keycode_char():
    key = _evdev_code_to_pynput_key(evdev.ecodes.KEY_A)
    assert key.char == "a"


def test_evdev_code_returns_none_for_unmapped_code():
    assert _evdev_code_to_pynput_key(999999) is None


class _FakeDevice:
    def __init__(self, keys: list[int]):
        self._keys = keys

    def capabilities(self):
        return {evdev.ecodes.EV_KEY: self._keys}


def test_is_keyboard_like_requires_space_and_a():
    assert _is_keyboard_like(_FakeDevice([evdev.ecodes.KEY_SPACE, evdev.ecodes.KEY_A]))
    assert not _is_keyboard_like(_FakeDevice([evdev.ecodes.KEY_SPACE]))  # footswitch-shaped device
    assert not _is_keyboard_like(_FakeDevice([]))


@dataclass
class _FakeEvent:
    type: int
    code: int
    value: int  # 1 = press, 0 = release, 2 = repeat


class _FakeInputDevice:
    """Stands in for evdev.InputDevice(path) -- read_loop() yields a fixed event script."""

    def __init__(self, events: list[_FakeEvent]):
        self._events = events

    def read_loop(self):
        yield from self._events


def test_listener_dispatches_press_and_release_through_callbacks():
    events = [
        _FakeEvent(type=evdev.ecodes.EV_KEY, code=evdev.ecodes.KEY_SPACE, value=1),
        _FakeEvent(type=evdev.ecodes.EV_KEY, code=evdev.ecodes.KEY_SPACE, value=2),  # repeat, ignored
        _FakeEvent(type=evdev.ecodes.EV_KEY, code=evdev.ecodes.KEY_SPACE, value=0),
        _FakeEvent(type=evdev.ecodes.EV_SYN, code=0, value=0),  # non-key event, ignored
    ]
    pressed, released = [], []
    listener = EvdevKeyboardListener(on_press=pressed.append, on_release=released.append)
    listener._run_with_device(_FakeInputDevice(events))

    assert pressed == [kb.Key.space]
    assert released == [kb.Key.space]


def test_listener_start_with_no_devices_does_not_raise():
    listener = EvdevKeyboardListener(on_press=lambda k: None, device_paths=[])
    listener.start()  # should log a warning, not crash
    listener.stop()
