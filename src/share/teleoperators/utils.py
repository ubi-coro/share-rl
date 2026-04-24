from enum import Enum
from lerobot.teleoperators.utils import TeleopEvents as BaseTeleopEvents


class TeleopEvents(Enum):
    SUCCESS = BaseTeleopEvents.SUCCESS.value
    FAILURE = BaseTeleopEvents.FAILURE.value
    RERECORD_EPISODE = BaseTeleopEvents.RERECORD_EPISODE.value
    IS_INTERVENTION = BaseTeleopEvents.IS_INTERVENTION.value
    TERMINATE_EPISODE = BaseTeleopEvents.TERMINATE_EPISODE.value

    INTERVENTION_COMPLETED = "intervention_completed"
    STOP_RECORDING = "stop_recording"
    PAUSE_RECORDING = "pause_recording"
    RESUME_RECORDING = "resume_recording"


def has_event(info: dict, event: TeleopEvents) -> bool:
    """Return whether an info dict contains a teleop event under enum or raw-value keys."""

    if bool(info.get(event, False)):
        return True
    value = getattr(event, "value", None)
    return bool(value is not None and info.get(value, False))


def is_intervention(info: dict) -> bool:
    """Return whether an info dict marks the current step as an intervention."""

    return has_event(info, TeleopEvents.IS_INTERVENTION)
