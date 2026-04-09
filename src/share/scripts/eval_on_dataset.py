#!/usr/bin/env python
"""
Script: eval_one_episode.py

Description:
    Loads a dataset and multiple trained policies, runs inference on one episode,
    and plots ground-truth vs predicted actions for grouped action dimensions.

    Each figure contains up to MAX_DIMS_PER_FIGURE subplots. If action_dim > max,
    multiple figures are created.

    Policies are compared against each other by loading checkpoint paths from the
    hardcoded POLICY_SPECS list below.
"""

import copy
import datetime
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import get_safe_torch_device, init_logging

from beast_refiner.policies import *  # noqa: F401,F403
from beast_refiner.scripts.lerobot_train import make_dataset


# ---------------------------------------------------------------------------
# Hardcoded comparison setup
# ---------------------------------------------------------------------------

MAX_DIMS_PER_FIGURE = 4

POLICY_SPECS = [
    {
        "name": "act",
        "path": "/vol/coro/jstranghoener/lerobot_volume/models/han_insertion_fb/act_forward_040426/checkpoints/last/pretrained_model",
    },
    {
        "name": "beast_refiner",
        "path": "/vol/coro/jstranghoener/lerobot_volume/models/han_insertion_fb/beast_refiner_forward_040426/checkpoints/last/pretrained_model",
    },
    {
        "name": "naive_refiner",
        "path": "/vol/coro/jstranghoener/lerobot_volume/models/han_insertion_fb/naive_refiner_forward_040426/checkpoints/last/pretrained_model",
    },
]


@dataclass
class OpenLoopEvalConfig(TrainPipelineConfig):
    episode_index: int = 0

    def __post_init__(self):
        if not self.output_dir:
            now = datetime.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("results/eval") / eval_dir


@torch.no_grad()
def run_inference_on_episode(
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy,
    preprocessor,
    postprocessor,
    episode_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run policy inference on the specified episode and return:
        gt_actions:   [T, action_dim]
        pred_actions: [T, action_dim]
    """
    gt_actions = []
    pred_actions = []

    found_episode = False

    for frame in tqdm(dataset, desc=f"Evaluating episode {episode_index}"):
        frame_episode_index = int(frame["episode_index"])

        if frame_episode_index < episode_index:
            continue

        if frame_episode_index > episode_index:
            if found_episode:
                break
            raise ValueError(
                f"Episode index {episode_index} was not found in dataset."
            )

        found_episode = True

        obs = preprocessor(frame)
        out = policy.select_action(obs)
        out = postprocessor(out)

        pred_actions.append(out.detach().cpu().float().squeeze(0))
        gt_actions.append(frame[ACTION].detach().cpu().float().squeeze(0))

    if not found_episode or not gt_actions:
        raise ValueError(f"No frames found for episode_index={episode_index}.")

    return torch.stack(gt_actions), torch.stack(pred_actions)


def load_policy_and_processors(
    cfg: OpenLoopEvalConfig,
    dataset: LeRobotDataset,
    device: torch.device,
    pretrained_path: str,
):
    """
    Create a policy and matching pre/post processors for a given checkpoint path.
    """
    local_cfg = copy.deepcopy(cfg)
    local_cfg.policy = PreTrainedConfig.from_pretrained(pretrained_path)
    local_cfg.policy.pretrained_path = pretrained_path

    logging.info("Loading trained policy from %s", pretrained_path)
    policy = make_policy(local_cfg.policy, ds_meta=dataset.meta, rename_map=local_cfg.rename_map)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=local_cfg.policy,
        pretrained_path=local_cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, local_cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "rename_observations_processor": {"rename_map": local_cfg.rename_map},
        },
    )

    return policy, preprocessor, postprocessor


def plot_predictions_grouped(
    gt_actions: torch.Tensor,
    policy_predictions: dict[str, torch.Tensor],
    save_dir: Path,
    max_dims_per_figure: int = 3,
):
    """
    Plot grouped action dimensions with up to `max_dims_per_figure` subplots per figure.

    Args:
        gt_actions: [T, action_dim]
        policy_predictions: dict mapping policy name -> [T, action_dim]
        save_dir: output directory
        max_dims_per_figure: max subplots per figure
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    if gt_actions.ndim != 2:
        raise ValueError(f"Expected gt_actions with shape [T, D], got {tuple(gt_actions.shape)}")

    T, action_dim = gt_actions.shape
    t = np.arange(T)

    for policy_name, pred_actions in policy_predictions.items():
        if pred_actions.shape != gt_actions.shape:
            raise ValueError(
                f"Shape mismatch for policy '{policy_name}': "
                f"gt_actions={tuple(gt_actions.shape)} vs pred_actions={tuple(pred_actions.shape)}"
            )

    num_figures = math.ceil(action_dim / max_dims_per_figure)

    gt_np = gt_actions.numpy()
    pred_np = {name: pred.numpy() for name, pred in policy_predictions.items()}

    for fig_idx in range(num_figures):
        start_dim = fig_idx * max_dims_per_figure
        end_dim = min((fig_idx + 1) * max_dims_per_figure, action_dim)
        dims_in_figure = end_dim - start_dim

        fig, axes = plt.subplots(
            nrows=dims_in_figure,
            ncols=1,
            figsize=(10, 3 * dims_in_figure),
            squeeze=False,
            sharex=True,
        )
        axes = axes.flatten()

        for subplot_idx, d in enumerate(range(start_dim, end_dim)):
            ax = axes[subplot_idx]
            ax.plot(t, gt_np[:, d], label="Ground Truth", linewidth=1.5)

            for policy_name, pred in pred_np.items():
                ax.plot(
                    t,
                    pred[:, d],
                    label=policy_name,
                    linewidth=1.2,
                    linestyle=":"
                )

            ax.set_ylabel(f"Action[{d}]")
            ax.set_title(f"Action dimension {d}")
            ax.grid(True, alpha=0.3)

            if subplot_idx == 0:
                ax.legend()

        axes[-1].set_xlabel("Timestep")
        fig.tight_layout()

        base_path = save_dir / f"action_dims_{start_dim}_to_{end_dim - 1}"

        fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

    logging.info(
        "Saved %d grouped figure(s) with up to %d action dims per figure to %s",
        num_figures,
        max_dims_per_figure,
        save_dir,
    )


@parser.wrap()
def eval_one_episode(cfg: OpenLoopEvalConfig):
    """
    Evaluation entrypoint — uses same config format as train.py for convenience.
    """
    init_logging()
    logging.info(pformat(cfg.to_dict()))
    cfg.validate()

    if len(POLICY_SPECS) == 0:
        raise ValueError("POLICY_SPECS is empty. Please add at least one policy path.")

    device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = make_dataset(cfg)
    dataset.delta_indices = None

    # Run all policies on the same episode
    gt_actions = None
    policy_predictions: dict[str, torch.Tensor] = {}

    for policy_spec in POLICY_SPECS:
        policy_name = policy_spec["name"]
        policy_path = policy_spec["path"]

        policy, preprocessor, postprocessor = load_policy_and_processors(
            cfg=cfg,
            dataset=dataset,
            device=device,
            pretrained_path=policy_path,
        )

        gt_i, pred_i = run_inference_on_episode(
            dataset=dataset,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            episode_index=cfg.episode_index,
        )

        if gt_actions is None:
            gt_actions = gt_i
        else:
            if gt_i.shape != gt_actions.shape or not torch.allclose(gt_i, gt_actions):
                logging.warning(
                    "Ground-truth action tensor for policy '%s' differs from previous run. "
                    "This usually should not happen if dataset iteration is deterministic.",
                    policy_name,
                )

        policy_predictions[policy_name] = pred_i

    if gt_actions is None:
        raise RuntimeError("No ground-truth actions were collected.")

    # Plot results
    save_dir = Path(cfg.output_dir) / f"eval_episode_{cfg.episode_index}"
    plot_predictions_grouped(
        gt_actions=gt_actions,
        policy_predictions=policy_predictions,
        save_dir=save_dir,
        max_dims_per_figure=MAX_DIMS_PER_FIGURE,
    )

    # Save config + comparison metadata
    save_payload = {
        "cfg": cfg.to_dict(),
        "episode_index": cfg.episode_index,
        "max_dims_per_figure": MAX_DIMS_PER_FIGURE,
        "policies": POLICY_SPECS,
    }

    save_path = save_dir / "config.json"
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(save_payload, f, ensure_ascii=False, indent=2, sort_keys=True)


if __name__ == "__main__":
    eval_one_episode()