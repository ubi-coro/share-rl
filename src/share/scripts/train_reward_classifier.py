#!/usr/bin/env python
"""Train the stock lerobot reward classifier on a rollout LeRobotDataset.

The dataset is expected to be an on-policy rollout dataset as written by
``share/scripts/learner_server.py`` (per-frame ``next.reward`` with 1.0 on the
LED-detected success frame). Episodes are split train/test at the episode
level (stratified by success). The last ``--positive-tail`` frames of every
successful episode are relabeled as positive, and class imbalance is handled
with a WeightedRandomSampler.

Example:
    python src/share/scripts/train_reward_classifier.py \
        --dataset-root /media/internal/nvme/shared_data/hoermann/insertion/e2e/RightTTL_090726/run/learner-2026-07-09-21-11-33/insert/dataset \
        --output-dir outputs/reward_classifier/RightTTL_090726
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.feature_utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
from lerobot.utils.constants import OBS_IMAGE, OBS_STATE, REWARD
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

from share.policies.reward_classifier import StateRewardClassifier, StateRewardClassifierConfig
from share.utils.device import get_safe_torch_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root of the rollout LeRobotDataset.")
    parser.add_argument("--repo-id", type=str, default="local/reward_classifier", help="Repo id (local load only).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write checkpoints and metrics.")
    parser.add_argument("--model-name", type=str, default="helper2424/resnet10")
    parser.add_argument(
        "--use-state",
        action="store_true",
        help="Train the state-augmented StateRewardClassifier instead of the stock vision-only classifier.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Images are resized to this square size before the encoder. The stock classifier head "
        "hardcodes a 4x4 feature map, which requires 128x128 inputs for the resnet10 backbone.",
    )
    parser.add_argument("--positive-tail", type=int, default=3, help="Frames at the end of successful episodes labeled positive.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of episodes held out for testing.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args(argv)


def compute_relabeled_labels(dataset: LeRobotDataset, positive_tail: int) -> tuple[np.ndarray, dict[int, list[int]], set[int]]:
    """Return per-frame labels with the last `positive_tail` frames of successful episodes set to 1.

    Also returns the frame indices of each episode (in dataset order) and the set of
    successful episode indices.
    """
    table = dataset.hf_dataset.select_columns(["episode_index", REWARD]).with_format("numpy")
    episode_indices = np.asarray(table["episode_index"]).reshape(-1)
    rewards = np.asarray(table[REWARD]).reshape(-1)

    labels = np.zeros(len(rewards), dtype=np.float32)
    episode_frames: dict[int, list[int]] = {}
    for idx, episode_index in enumerate(episode_indices):
        episode_frames.setdefault(int(episode_index), []).append(idx)

    successful_episodes: set[int] = set()
    for episode_index, frames in episode_frames.items():
        if rewards[frames[-1]] > 0:
            successful_episodes.add(episode_index)
            labels[frames[-positive_tail:]] = 1.0

    return labels, episode_frames, successful_episodes


def split_episodes(
    episode_frames: dict[int, list[int]],
    successful_episodes: set[int],
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Episode-level train/test split, stratified by episode success."""
    rng = np.random.default_rng(seed)
    train_frames: list[int] = []
    test_frames: list[int] = []
    for group in (sorted(successful_episodes), sorted(set(episode_frames) - successful_episodes)):
        group = list(group)
        rng.shuffle(group)
        num_test = max(1, round(len(group) * test_ratio)) if group else 0
        for episode_index in group[:num_test]:
            test_frames.extend(episode_frames[episode_index])
        for episode_index in group[num_test:]:
            train_frames.extend(episode_frames[episode_index])
    return train_frames, test_frames


class RelabeledSubset(Dataset):
    """Subset of a LeRobotDataset with `next.reward` replaced by relabeled classifier targets."""

    def __init__(self, dataset: LeRobotDataset, indices: list[int], labels: np.ndarray):
        self.dataset = dataset
        self.indices = indices
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        item = self.dataset[idx]
        item[REWARD] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def make_train_loader(subset: RelabeledSubset, batch_size: int, num_workers: int, seed: int) -> DataLoader:
    subset_labels = subset.labels[subset.indices]
    num_pos = int(subset_labels.sum())
    num_neg = len(subset_labels) - num_pos
    if num_pos == 0 or num_neg == 0:
        raise ValueError(f"Train split needs both classes, got {num_pos} positive / {num_neg} negative frames.")
    weights = np.where(subset_labels > 0, 1.0 / num_pos, 1.0 / num_neg)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(subset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def predict_probs(policy, batch: dict, image_keys: list[str]) -> torch.Tensor:
    images = [batch[key] for key in image_keys]
    if isinstance(policy, StateRewardClassifier):
        return policy.predict(images, state=batch[OBS_STATE]).probabilities
    return policy.predict(images).probabilities


def resize_images(batch: dict, image_keys: list[str], image_size: int) -> dict:
    for key in image_keys:
        if batch[key].shape[-1] != image_size or batch[key].shape[-2] != image_size:
            batch[key] = torch.nn.functional.interpolate(
                batch[key], size=(image_size, image_size), mode="bilinear", align_corners=False
            )
    return batch


@torch.no_grad()
def evaluate(policy, preprocessor, loader: DataLoader, image_keys: list[str], image_size: int) -> dict:
    policy.eval()
    all_probs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for batch in loader:
        labels = batch[REWARD].float()
        batch = preprocessor(batch)
        batch = resize_images(batch, image_keys, image_size)
        probs = predict_probs(policy, batch, image_keys)
        all_probs.append(probs.cpu())
        all_labels.append(labels.reshape(-1).cpu())
    policy.train()

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    metrics = {"num_frames": int(labels.numel()), "num_positive": int(labels.sum().item())}
    for threshold in (0.3, 0.5, 0.7, 0.9):
        predictions = (probs > threshold).float()
        tp = int(((predictions == 1) & (labels == 1)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())
        tn = int(((predictions == 0) & (labels == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[f"threshold_{threshold}"] = {
            "accuracy": (tp + tn) / labels.numel(),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    return metrics


def run_training(args: argparse.Namespace) -> dict:
    """Train one classifier variant; returns the trained artifacts and split for reuse."""
    set_seed(args.seed)
    device = get_safe_torch_device(args.device, log=True)

    dataset = LeRobotDataset(repo_id=args.repo_id, root=str(args.dataset_root))
    labels, episode_frames, successful_episodes = compute_relabeled_labels(dataset, args.positive_tail)
    train_frames, test_frames = split_episodes(episode_frames, successful_episodes, args.test_ratio, args.seed)
    logging.info(
        "Dataset: %d episodes (%d successful), %d frames, %d positive after relabeling (tail=%d)",
        len(episode_frames),
        len(successful_episodes),
        len(labels),
        int(labels.sum()),
        args.positive_tail,
    )
    logging.info(
        "Split: %d train frames (%d positive) / %d test frames (%d positive)",
        len(train_frames),
        int(labels[train_frames].sum()),
        len(test_frames),
        int(labels[test_frames].sum()),
    )

    num_cameras = sum(1 for key in dataset.meta.features if key.startswith(OBS_IMAGE))
    config_cls = StateRewardClassifierConfig if args.use_state else RewardClassifierConfig
    cfg = config_cls(model_name=args.model_name, device=str(device), num_cameras=num_cameras)
    # make_policy() in lerobot 0.5.1 passes dataset_stats/dataset_meta kwargs that
    # Classifier.__init__ does not accept, so fill the features and instantiate directly.
    features = dataset_to_policy_features(dataset.meta.features)
    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    cfg.validate_features()
    policy = StateRewardClassifier(cfg) if args.use_state else Classifier(cfg)
    image_keys = [key for key in cfg.input_features if key.startswith(OBS_IMAGE)]
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset.meta.stats)

    train_loader = make_train_loader(
        RelabeledSubset(dataset, train_frames, labels), args.batch_size, args.num_workers, args.seed
    )
    test_loader = DataLoader(
        RelabeledSubset(dataset, test_frames, labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        policy.get_optim_params(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy.train()
    policy.to(device)

    step = 0
    running_loss = 0.0
    running_accuracy = 0.0
    train_iter = iter(train_loader)
    while step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = preprocessor(batch)
        batch = resize_images(batch, image_keys, args.image_size)
        batch[REWARD] = batch[REWARD].float().reshape(-1).to(device)
        loss, output_dict = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg.grad_clip_norm)
        optimizer.step()

        step += 1
        running_loss += loss.item()
        running_accuracy += output_dict["accuracy"]
        if step % args.log_freq == 0:
            logging.info(
                "step=%d loss=%.4f accuracy=%.2f%%",
                step,
                running_loss / args.log_freq,
                running_accuracy / args.log_freq,
            )
            running_loss = 0.0
            running_accuracy = 0.0

        if step % args.eval_freq == 0 or step == args.steps:
            metrics = evaluate(policy, preprocessor, test_loader, image_keys, args.image_size)
            summary = metrics["threshold_0.5"]
            logging.info(
                "eval step=%d accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f (threshold=0.5)",
                step,
                summary["accuracy"],
                summary["precision"],
                summary["recall"],
                summary["f1"],
            )
            checkpoint_dir = args.output_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
            policy.save_pretrained(checkpoint_dir)
            preprocessor.save_pretrained(checkpoint_dir)
            postprocessor.save_pretrained(checkpoint_dir)
            with open(checkpoint_dir / "eval_metrics.json", "w") as f:
                json.dump({"step": step, **metrics}, f, indent=2)

    final_metrics = evaluate(policy, preprocessor, test_loader, image_keys, args.image_size)
    with open(args.output_dir / "final_eval_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    logging.info("Final test metrics: %s", json.dumps(final_metrics, indent=2))

    return {
        "policy": policy,
        "preprocessor": preprocessor,
        "image_keys": image_keys,
        "checkpoint_dir": args.output_dir / "checkpoints" / f"{args.steps:06d}" / "pretrained_model",
        "final_metrics": final_metrics,
        "dataset": dataset,
        "labels": labels,
        "episode_frames": episode_frames,
        "successful_episodes": successful_episodes,
        "test_frames": test_frames,
    }


def main() -> None:
    args = parse_args()
    init_logging()
    run_training(args)


if __name__ == "__main__":
    main()
