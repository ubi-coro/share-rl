from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig


@PreTrainedConfig.register_subclass(name="state_reward_classifier")
@dataclass
class StateRewardClassifierConfig(RewardClassifierConfig):
    """Reward classifier that additionally conditions on the proprioceptive state vector."""

    name: str = "state_reward_classifier"
    state_hidden_dim: int = 64
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
        }
    )
