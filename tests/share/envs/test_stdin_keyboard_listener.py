"""Tests for the stdin-backed keyboard listener (the no-permissions-needed fallback)."""

from __future__ import annotations

import select
import sys

from pynput import keyboard as kb

from share.processor.utils import StdinKeyboardListener


class _QueuedStdin:
    """Fake stdin: .read(1) pops one char at a time from a preset string."""

    def __init__(self, text: str):
        self._chars = list(text)

    def read(self, n: int) -> str:
        assert n == 1
        return self._chars.pop(0) if self._chars else ""


def _patched_listener(monkeypatch, text: str) -> StdinKeyboardListener:
    fake_stdin = _QueuedStdin(text)
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    monkeypatch.setattr(select, "select", lambda *a, **k: ([fake_stdin], [], []))
    return StdinKeyboardListener()


def test_read_key_translates_space(monkeypatch):
    listener = _patched_listener(monkeypatch, "")
    assert listener._read_key(" ", kb) is kb.Key.space


def test_read_key_translates_letter_to_keycode(monkeypatch):
    listener = _patched_listener(monkeypatch, "")
    key = listener._read_key("a", kb)
    assert key.char == "a"


def test_read_key_translates_arrow_escape_sequence(monkeypatch):
    # ESC has already been consumed by the caller; "[B" (down arrow) remains queued.
    listener = _patched_listener(monkeypatch, "[B")
    assert listener._read_key("\x1b", kb) is kb.Key.down


def test_read_key_translates_left_arrow_escape_sequence(monkeypatch):
    listener = _patched_listener(monkeypatch, "[D")
    assert listener._read_key("\x1b", kb) is kb.Key.left


def test_read_key_ignores_bare_escape(monkeypatch):
    fake_stdin = _QueuedStdin("")
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    monkeypatch.setattr(select, "select", lambda *a, **k: ([], [], []))  # nothing follows ESC
    listener = StdinKeyboardListener()
    assert listener._read_key("\x1b", kb) is None
