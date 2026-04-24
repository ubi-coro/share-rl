from __future__ import annotations

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

from share.rl.runtime import make_policy_processors, preprocess_replay_batch, resolve_policy_dataset_stats


def _policy_config() -> SACConfig:
    cfg = SACConfig(device="cpu", storage_device="cpu", use_torch_compile=False)
    cfg.input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 4)),
    }
    cfg.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    }
    cfg.dataset_stats = {
        OBS_IMAGE: {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.25, 0.25, 0.25],
        },
        OBS_STATE: {
            "min": [0.0, 0.0],
            "max": [10.0, 20.0],
        },
        ACTION: {
            "min": [-2.0, -1.0],
            "max": [2.0, 3.0],
        },
    }
    return cfg


def test_resolve_policy_dataset_stats_expands_shared_visual_stats():
    cfg = _policy_config()

    dataset_stats = resolve_policy_dataset_stats(cfg)

    assert dataset_stats is not None
    assert dataset_stats["observation.images.front"]["mean"] == [[[0.5]], [[0.5]], [[0.5]]]
    assert dataset_stats["observation.images.front"]["std"] == [[[0.25]], [[0.25]], [[0.25]]]
    assert dataset_stats[OBS_STATE]["max"] == [10.0, 20.0]
    assert dataset_stats[ACTION]["min"] == [-2.0, -1.0]


def test_preprocess_replay_batch_uses_policy_dataset_stats():
    cfg = _policy_config()
    policy = type("MockPolicy", (), {"config": cfg})()
    preprocessors, _ = make_policy_processors({"main": policy})
    preprocessor = preprocessors["main"]

    observations = {
        OBS_STATE: torch.tensor([[5.0, 10.0]], dtype=torch.float32),
        "observation.images.front": torch.ones((1, 3, 4, 4), dtype=torch.float32),
    }
    actions = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    next_observations = {
        OBS_STATE: torch.tensor([[10.0, 20.0]], dtype=torch.float32),
        "observation.images.front": torch.full((1, 3, 4, 4), 0.5, dtype=torch.float32),
    }

    normalized_obs, normalized_actions, normalized_next_obs = preprocess_replay_batch(
        preprocessor=preprocessor,
        observations=observations,
        actions=actions,
        next_observations=next_observations,
    )

    assert torch.allclose(normalized_obs[OBS_STATE], torch.zeros((1, 2), dtype=torch.float32))
    assert torch.allclose(normalized_actions, torch.zeros((1, 2), dtype=torch.float32))
    assert torch.allclose(normalized_next_obs[OBS_STATE], torch.ones((1, 2), dtype=torch.float32))
    assert torch.allclose(
        normalized_obs["observation.images.front"],
        torch.full((1, 3, 4, 4), 2.0, dtype=torch.float32),
    )
