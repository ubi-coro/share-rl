"""Tests for the generic previous-action observation step."""

from __future__ import annotations

import pytest
import torch

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import TransitionKey, create_transition
from lerobot.utils.constants import ACTION, OBS_STATE

from share.processor.observation import PreviousActionObservationProcessor


def test_previous_action_is_zero_on_reset_and_matches_current_action():
    step = PreviousActionObservationProcessor(action_dim=2)

    reset = step(create_transition(observation={OBS_STATE: torch.tensor([1.0, 2.0])}))
    assert reset[TransitionKey.OBSERVATION][OBS_STATE].tolist() == [1.0, 2.0, 0.0, 0.0]

    current = step(create_transition(
        observation={OBS_STATE: torch.tensor([3.0, 4.0])},
        action=torch.tensor([0.11, 0.33]),
    ))
    assert current[TransitionKey.OBSERVATION][OBS_STATE].tolist() == pytest.approx(
        [3.0, 4.0, 0.11, 0.33]
    )

    step.reset()
    after_reset = step(create_transition(observation={OBS_STATE: torch.tensor([5.0, 6.0])}))
    assert after_reset[TransitionKey.OBSERVATION][OBS_STATE].tolist() == [5.0, 6.0, 0.0, 0.0]


def test_previous_action_flattens_nested_actions_and_updates_feature_shape():
    step = PreviousActionObservationProcessor(action_dim=3)
    transition = create_transition(
        observation={OBS_STATE: torch.tensor([1.0])},
        action={"right": {"z.ee_pos": 0.3}, "left": {"y.ee_pos": 0.1, "z.ee_pos": 0.2}},
    )

    out = step(transition)
    assert out[TransitionKey.OBSERVATION][OBS_STATE].tolist() == pytest.approx([1.0, 0.1, 0.2, 0.3])

    features = {
        PipelineFeatureType.ACTION: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
        },
        PipelineFeatureType.OBSERVATION: {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        },
    }
    transformed = step.transform_features(features)
    assert transformed[PipelineFeatureType.OBSERVATION][OBS_STATE].shape == (7,)


def test_previous_action_can_be_disabled_and_rejects_wrong_action_shape():
    transition = create_transition(
        observation={OBS_STATE: torch.tensor([1.0])},
        action=torch.tensor([0.1, 0.2]),
    )
    disabled = PreviousActionObservationProcessor(enable=False, action_dim=2)
    assert disabled(transition) is transition
    assert disabled.transform_features({}) == {}

    with pytest.raises(ValueError, match="expected 2"):
        PreviousActionObservationProcessor(action_dim=2)(create_transition(action=torch.tensor([0.1])))
