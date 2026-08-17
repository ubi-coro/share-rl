"""Tests for the shared process-wide keyboard listener singleton in share.processor.info."""

from __future__ import annotations

import gc

from pynput import keyboard as kb

import share.processor.info as info_mod
from share.processor.info import AddKeyboardEventsAsInfoStep
from share.teleoperators import TeleopEvents


def test_two_instances_share_one_process_wide_listener():
    step1 = AddKeyboardEventsAsInfoStep(mapping={TeleopEvents.SUCCESS: kb.Key.space})
    listener_after_first = info_mod._keyboard_listener
    assert listener_after_first is not None

    step2 = AddKeyboardEventsAsInfoStep(mapping={TeleopEvents.RERECORD_EPISODE: kb.Key.left})

    assert info_mod._keyboard_listener is listener_after_first  # no new listener spawned
    assert info_mod._keyboard_listener.daemon is True
    del step1, step2


def test_keypress_dispatches_only_to_instances_whose_mapping_contains_the_key():
    step_space = AddKeyboardEventsAsInfoStep(mapping={TeleopEvents.SUCCESS: kb.Key.space})
    step_left = AddKeyboardEventsAsInfoStep(mapping={TeleopEvents.RERECORD_EPISODE: kb.Key.left})

    info_mod._dispatch_keyboard_press(kb.Key.space)

    assert step_space.info({})[TeleopEvents.SUCCESS] is True
    assert step_left.info({})[TeleopEvents.RERECORD_EPISODE] is False
    del step_space, step_left


def test_dropped_instance_is_pruned_and_others_keep_working():
    step_a = AddKeyboardEventsAsInfoStep(mapping={TeleopEvents.SUCCESS: kb.Key.space})
    step_b = AddKeyboardEventsAsInfoStep(mapping={TeleopEvents.RERECORD_EPISODE: kb.Key.left})
    id_a = id(step_a)

    assert id_a in info_mod._KEYBOARD_LISTENERS

    del step_a
    gc.collect()

    assert id_a not in info_mod._KEYBOARD_LISTENERS

    info_mod._dispatch_keyboard_press(kb.Key.left)
    assert step_b.info({})[TeleopEvents.RERECORD_EPISODE] is True
    del step_b
