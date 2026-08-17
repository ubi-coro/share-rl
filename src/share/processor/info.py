import threading
import weakref
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import InfoProcessorStep, ProcessorStepRegistry, TransitionKey, EnvTransition, ProcessorStep
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY, _check_teleop_with_events

from share.processor.utils import CompositeKeyboardListener, EvdevKeyboardListener, FootSwitchHandler, StdinKeyboardListener
from share.teleoperators import TeleopEvents

for _registry_name in (
    "add_teleop_action_on_intervention_as_complementary_data",
    "add_teleop_events_as_info",
    "add_footswitch_events_as_info",
    "add_keyboard_events_as_info",
):
    ProcessorStepRegistry.unregister(_registry_name)


@ProcessorStepRegistry.register("add_teleop_action_on_intervention_as_complementary_data")
@dataclass
class AddTeleopActionAsComplimentaryDataStep(ProcessorStep):
    """
    Adds the raw action from a teleoperator to the transition's complementary data.

    This is useful for human-in-the-loop scenarios where the human's input needs to
    be available to downstream processors, for example, to override a policy's action
    during an intervention.

    Attributes:
        teleop_device: The teleoperator instance to get the action from.
    """

    teleoperators: dict[str, "Teleoperator"] = field(default_factory=dict)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """Applies the `complementary_data` method to the transition's data."""
        self._current_transition = transition.copy()
        new_transition = self._current_transition

        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None or not isinstance(complementary_data, dict):
            raise ValueError("ComplementaryDataProcessorStep requires complementary data in the transition.")

        processed_complementary_data = complementary_data.copy()

        # avoid unnecessary I/O by only reading when the intervention event is set
        if transition[TransitionKey.INFO].get(TeleopEvents.IS_INTERVENTION, False):
            processed_complementary_data[TELEOP_ACTION_KEY] = {}
            for name in self.teleoperators:
                processed_complementary_data[TELEOP_ACTION_KEY][name] = self.teleoperators[name].get_action()

        new_transition[TransitionKey.COMPLEMENTARY_DATA] = processed_complementary_data
        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("add_teleop_events_as_info")
@dataclass
class AddTeleopEventsAsInfoStep(InfoProcessorStep):
    """
    Adds teleoperator control events (e.g., terminate, success) to the transition's info.

    This step extracts control events from teleoperators that support event-based
    interaction, making these signals available to other parts of the system.

    Attributes:
        teleop_device: An instance of a teleoperator that implements the
                       `HasTeleopEvents` protocol.
    """

    teleoperators: dict[str, "Teleoperator"] = field(default_factory=dict)

    def __post_init__(self):
        """Validates that the provided teleoperator supports events after initialization."""
        for t in self.teleoperators.values():
            _check_teleop_with_events(t)

    def info(self, info: dict) -> dict:
        """
        Retrieves teleoperator events and updates the info dictionary.

        Args:
            info: The incoming info dictionary.

        Returns:
            A new dictionary including the teleoperator events.
        """
        new_info = dict(info)
        for t in self.teleoperators.values():
            for event_name, event_value in t.get_teleop_events().items():
                new_info[event_name] = new_info.get(event_name, False) | event_value

        return new_info

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("add_footswitch_events_as_info")
@dataclass
class AddFootswitchEventsAsInfoStep(InfoProcessorStep):
    mapping: dict[tuple[TeleopEvents], dict] = field(default_factory=dict)

    def __post_init__(self):
        self._foot_switch_threads = {}
        self._require_release = {}  # event_name -> bool

        for events, params in self.mapping.items():
            handler = FootSwitchHandler(
                device_path=f'/dev/input/event{params["device"]}',
                toggle=bool(params["toggle"]),
                event_names=events,
            )
            self._foot_switch_threads[events] = handler
            handler.start()

            # Initialize all known events as armed
            for event_name in events:
                self._require_release[event_name] = False

    def info(self, info: dict) -> dict:
        new_info = dict(info)

        for handler in self._foot_switch_threads.values():
            for event_name, event_value in handler.events.items():
                if event_name not in new_info:
                    new_info[event_name] = False

                event_value = bool(event_value)

                # After reset, require observing one False before allowing True again
                if self._require_release.get(event_name, False):
                    if not event_value:
                        # Foot was released -> re-arm this event
                        self._require_release[event_name] = False
                    # Suppress event while waiting for release
                    continue

                if event_value:
                    new_info[event_name] = new_info.get(event_name, False) or True

        return new_info

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def reset(self) -> None:
        for handler in self._foot_switch_threads.values():
            handler.reset()

        # After reset, require pedal release before any event can become True again
        for event_name in self._require_release:
            self._require_release[event_name] = True

    def __del__(self):
        for handler in self._foot_switch_threads.values():
            handler.stop()


# Every primitive's action pipeline is built eagerly at MP-Net construction, so each
# AddKeyboardEventsAsInfoStep used to spawn its own pynput.Listener, all fighting over the
# same keyboard. Share one process-wide listener instead; per-instance mapping/state unchanged.
_keyboard_listener_lock = threading.Lock()
_keyboard_listener: Any = None
_KEYBOARD_LISTENERS: "weakref.WeakValueDictionary[int, AddKeyboardEventsAsInfoStep]" = weakref.WeakValueDictionary()


def _dispatch_keyboard_press(key) -> None:
    # list() snapshot: atomic under the GIL, avoids mutation-during-iteration
    for step in list(_KEYBOARD_LISTENERS.values()):
        step._on_key_press(key)


def _dispatch_keyboard_release(key) -> None:
    for step in list(_KEYBOARD_LISTENERS.values()):
        step._on_key_release(key)


def _ensure_keyboard_listener() -> None:
    global _keyboard_listener
    if _keyboard_listener is not None:
        return
    with _keyboard_listener_lock:
        if _keyboard_listener is not None:
            return
        # evdev needs /dev/input access (input group); stdin needs none but only sees keys
        # while this process's terminal has focus. Run both -- whichever actually works wins.
        listener = CompositeKeyboardListener([
            EvdevKeyboardListener(on_press=_dispatch_keyboard_press, on_release=_dispatch_keyboard_release),
            StdinKeyboardListener(on_press=_dispatch_keyboard_press, on_release=_dispatch_keyboard_release),
        ])
        listener.start()
        _keyboard_listener = listener


@ProcessorStepRegistry.register("add_keyboard_events_as_info")
@dataclass
class AddKeyboardEventsAsInfoStep(InfoProcessorStep):
    mapping: dict[TeleopEvents, Any] = field(default_factory=dict)
    pulse_events: tuple[TeleopEvents | str, ...] = ()

    def __post_init__(self):
        self._events = {event: False for event in self.mapping}
        self._pressed = {event: False for event in self.mapping}
        self._is_string_key = {event: isinstance(mapping_key, str) for event, mapping_key in self.mapping.items()}

        _ensure_keyboard_listener()
        _KEYBOARD_LISTENERS[id(self)] = self

    def _on_key_press(self, key) -> None:
        for event, mapping_key in self.mapping.items():
            try:
                matched = key.char == mapping_key if self._is_string_key[event] else key == mapping_key
                if matched:
                    if event not in self.pulse_events or not self._pressed[event]:
                        self._events[event] = True
                    self._pressed[event] = True
            except Exception:
                ...

    def _on_key_release(self, key) -> None:
        for event, mapping_key in self.mapping.items():
            try:
                matched = key.char == mapping_key if self._is_string_key[event] else key == mapping_key
                if matched:
                    self._pressed[event] = False
                    if event not in self.pulse_events:
                        self._events[event] = False
            except Exception:
                ...

    def info(self, info: dict) -> dict:
        new_info = dict(info)
        for event_name, event_value in self._events.items():
            new_info[event_name] = new_info.get(event_name, False) | event_value
            if event_name in self.pulse_events and event_value:
                self._events[event_name] = False
        return new_info

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features

    def reset(self) -> None:
        self._events = {event: False for event in self.mapping}
        self._pressed = {event: False for event in self.mapping}
