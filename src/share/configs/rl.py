import copy
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lerobot.configs import parser

from share.configs.mpnet import DatasetRecordConfig, TrainRLServerPipelineConfig
from share.rl.runtime import build_adaptive_registry
from share.workspace.mpnet import ManipulationPrimitiveNetConfig, PreTrainedConfig


# Fields the env computes from connector data / spec / ablation flags and bakes onto each
# primitive's default policy. Preserved when a top-level --policy.type or --policy.path selects the policy.
_ENV_DERIVED_POLICY_ATTRS = ("dataset_stats", "freeze_vision_encoder")


@dataclass(kw_only=True)
class MPNetTrainRLServerPipelineConfig(TrainRLServerPipelineConfig):
    """Train config for MP-Net distributed SAC actor/learner servers."""

    env: ManipulationPrimitiveNetConfig
    dataset: DatasetRecordConfig | None = None
    policy: PreTrainedConfig | None = None
    log_freq: int = 10
    num_workers: int = 6
    batch_size: int = 256

    def resolve_policy_overrides(self) -> None:
        """Apply one umbrella policy selection to every adaptive primitive."""
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path,
                cli_overrides=cli_overrides,
            )
            self.policy.pretrained_path = Path(policy_path)

        if self.policy is None:
            policies = [p.policy for p in self.env.primitives.values() if p.policy is not None]
            if policies:
                self.policy = policies[0]
            return

        for primitive in self.env.primitives.values():
            if not (primitive.is_adaptive and primitive.policy is not None):
                continue
            selected = copy.deepcopy(self.policy)
            for attr in _ENV_DERIVED_POLICY_ATTRS:
                if hasattr(primitive.policy, attr) and hasattr(selected, attr):
                    setattr(selected, attr, copy.deepcopy(getattr(primitive.policy, attr)))
            primitive.policy = selected

    def validate(self, output_role: Literal["actor", "learner"] = "learner") -> None:
        """Resolve runtime config, including a role-specific default output directory."""
        self.resolve_policy_overrides()
        if output_role not in ("actor", "learner"):
            raise ValueError(
                f"output_role must be 'actor' or 'learner', got {output_role!r}."
            )
        if not self.job_name:
            # Env-config ablation flags (e.g. ConnectorEnvConfig.freeze_vision_encoder /
            # wrist_camera_only) surface here so ablation runs get distinct default names.
            self.job_name = f"{self.env.type}_sac{getattr(self.env, 'ablation_tag', '')}"

        if self.output_dir is None:
            now = dt.datetime.now()
            if self.dataset is not None and self.dataset.root is not None:
                self.output_dir = (
                    Path(self.dataset.root)
                    / "run"
                    / f"{output_role}-{now:%Y-%m-%d-%H-%M-%S}"
                )
            else:
                self.output_dir = Path(
                    f"outputs/train/{now:%Y-%m-%d}/"
                    f"{now:%H-%M-%S}_{output_role}_{self.job_name}"
                )
        else:
            self.output_dir = Path(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        _ = build_adaptive_registry(self.env)
