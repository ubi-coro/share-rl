"""mute_log_messages in control_utils: drops matching root-logger messages from the existing handlers."""
import logging

from share.utils.control_utils import mute_log_messages


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_mute_log_messages_drops_only_matching_prefixes():
    handler = _ListHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    previous_level = root.level
    root.setLevel(logging.INFO)
    try:
        mute_log_messages("Relative goal position magnitude")
        logging.warning("Relative goal position magnitude had to be clamped to be safe.\n{...}")
        logging.warning("something else")
    finally:
        root.setLevel(previous_level)
        root.removeHandler(handler)

    assert handler.messages == ["something else"]
