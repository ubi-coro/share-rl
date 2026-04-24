import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from lerobot.configs.default import DatasetConfig

from share.configs.record import TrainRLServerPipelineConfig
from share.rl.runtime import build_adaptive_registry
from share.workspace.mpnet import ManipulationPrimitiveNetConfig, PreTrainedConfig


@dataclass(kw_only=True)
class MPNetTrainRLServerPipelineConfig(TrainRLServerPipelineConfig):
    """Train config for MP-Net distributed SAC actor/learner servers."""

    env: ManipulationPrimitiveNetConfig
    dataset: DatasetConfig | None = None
    policy: PreTrainedConfig | None = None
    log_freq: int = 10
    num_workers: int = 6
    batch_size: int = 256

    def validate(self) -> None:
        if not self.job_name:
            self.job_name = f"{self.env.type}_sac"

        if self.output_dir is None:
            now = dt.datetime.now()
            self.output_dir = Path(f"outputs/train/{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}")
        else:
            self.output_dir = Path(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        _ = build_adaptive_registry(self.env, self.policy)