#!/usr/bin/env python
"""Train both reward-classifier variants and render a one-page PDF report.

Trains the vision-only (stock lerobot) and state-augmented reward classifiers on a
rollout dataset via ``share/scripts/train_reward_classifier.py`` (fast: frozen encoder,
a few minutes per variant), evaluates both on the held-out test episodes, and writes:

    <output-dir>/models/vision[_state]/...      checkpoints + eval metrics
    <output-dir>/reward_classifier_report.pdf   the figure
    <output-dir>/summary.md                     text summary

Example:
    python src/share/scripts/plotting/plot_reward_classifier.py \
        --dataset-root /media/internal/nvme/shared_data/hoermann/insertion/e2e/RightTTL_090726/run/learner-2026-07-09-21-11-33/insert/dataset \
        --output-dir /media/internal/nvme/shared_data/hoermann/insertion/e2e/RightTTL_090726/reward_classifier_report
"""

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lerobot.utils.utils import init_logging  # noqa: E402

from share.scripts import train_reward_classifier as trainer  # noqa: E402

# Categorical slots 1 (blue) and 2 (aqua) of the validated reference palette.
COLOR_VISION = "#2a78d6"
COLOR_STATE = "#1baf7a"
COLOR_NEUTRAL = "#52514e"
COLOR_GRID = "#d8d7d2"

VARIANTS = {
    "vision": {"label": r"images only", "color": COLOR_VISION, "use_state": False},
    "vision_state": {"label": r"images $+$ state", "color": COLOR_STATE, "use_state": True},
}

TAIL = 3  # frames relabeled positive at the end of successful episodes
NEAR = 5  # "almost in tail": first fire within this many frames of the end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.9, help="Operating threshold highlighted in the report.")
    return parser.parse_args()


@torch.no_grad()
def collect_episode_probs(result: dict, episodes: list[int], image_size: int = 128) -> dict[int, np.ndarray]:
    """Per-episode success probabilities on the test episodes, in frame order."""
    policy = result["policy"]
    policy.eval()
    dataset = result["dataset"]
    probs: dict[int, np.ndarray] = {}
    for episode in episodes:
        frames = result["episode_frames"][episode]
        chunks = []
        for start in range(0, len(frames), 128):
            batch = torch.utils.data.default_collate([dataset[j] for j in frames[start:start + 128]])
            batch = result["preprocessor"](batch)
            batch = trainer.resize_images(batch, result["image_keys"], image_size)
            chunks.append(trainer.predict_probs(policy, batch, result["image_keys"]).cpu())
        probs[episode] = torch.cat(chunks).numpy()
    policy.train()
    return probs


def episode_outcomes(probs: dict[int, np.ndarray], successful: set[int], threshold: float) -> dict[str, int]:
    succ = [ep for ep in probs if ep in successful]
    fail = [ep for ep in probs if ep not in successful]
    fire = {ep: probs[ep] > threshold for ep in probs}
    return {
        "fires in tail": sum(1 for ep in succ if fire[ep][-TAIL:].any() and not fire[ep][:-TAIL].any()),
        "fires 1--2 frames early": sum(1 for ep in succ if fire[ep][:-TAIL].any() and not fire[ep][:-NEAR].any()),
        "fires earlier": sum(1 for ep in succ if fire[ep][:-NEAR].any()),
        "never fires (missed)": sum(1 for ep in succ if not fire[ep].any()),
        "false fire (failure ep.)": sum(1 for ep in fail if fire[ep].any()),
        "_n_success": len(succ),
        "_n_failure": len(fail),
    }


def frame_level_sweep(probs: dict[int, np.ndarray], labels: np.ndarray, episode_frames: dict[int, list[int]]):
    """Precision/recall/F1 over a threshold sweep on all test frames."""
    y_prob = np.concatenate([probs[ep] for ep in probs])
    y_true = np.concatenate([labels[episode_frames[ep]] for ep in probs])
    thresholds = np.linspace(0.02, 0.99, 98)
    precision, recall, f1 = [], [], []
    for threshold in thresholds:
        pred = y_prob > threshold
        tp = int((pred & (y_true > 0)).sum())
        fp = int((pred & (y_true == 0)).sum())
        fn = int((~pred & (y_true > 0)).sum())
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return thresholds, np.array(precision), np.array(recall), np.array(f1)


def style_axis(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_NEUTRAL)
    ax.grid(True, color=COLOR_GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLOR_NEUTRAL, labelsize=8)


def render_report(results: dict[str, dict], data: dict[str, dict], args: argparse.Namespace, pdf_path: Path) -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
    })

    fig = plt.figure(figsize=(11.0, 7.6), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.12, w_pad=0.10)
    axes = fig.subplots(2, 2)
    (ax_pr, ax_f1), (ax_outcomes, ax_traces) = axes

    any_data = next(iter(data.values()))
    n_test_frames = sum(len(p) for p in any_data["probs"].values())
    n_test_eps = len(any_data["probs"])
    fig.suptitle(
        rf"Insertion success classifier --- RightTTL\_090726 rollout dataset "
        rf"({n_test_eps} held-out test episodes, {n_test_frames} frames, "
        rf"tail $={TAIL}$ frames, operating threshold $\tau={args.threshold}$)",
        fontsize=11,
    )

    # --- Panel A: precision-recall curves ---------------------------------------
    for name, variant in VARIANTS.items():
        thresholds, precision, recall, _ = data[name]["sweep"]
        ax_pr.plot(recall, precision, color=variant["color"], linewidth=1.8, label=variant["label"])
        op = np.argmin(np.abs(thresholds - args.threshold))
        ax_pr.plot(recall[op], precision[op], "o", color=variant["color"], markersize=6,
                   markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax_pr.annotate(rf"$\tau={args.threshold}$",
                   xy=(0.02, 0.03), xycoords="axes fraction", color=COLOR_NEUTRAL, fontsize=8)
    ax_pr.set_xlabel("recall")
    ax_pr.set_ylabel("precision")
    ax_pr.set_xlim(0.0, 1.02)
    ax_pr.set_ylim(0.0, 1.04)
    ax_pr.set_title("frame-level precision--recall (test set)")
    ax_pr.legend(loc="lower left", frameon=False)
    style_axis(ax_pr)

    # --- Panel B: F1 vs threshold ------------------------------------------------
    for name, variant in VARIANTS.items():
        thresholds, _, _, f1 = data[name]["sweep"]
        ax_f1.plot(thresholds, f1, color=variant["color"], linewidth=1.8, label=variant["label"])
        best = int(np.argmax(f1))
        ax_f1.annotate(rf"$F_1={f1[best]:.2f}$",
                       xy=(thresholds[best], f1[best]),
                       xytext=(thresholds[best] - 0.02, f1[best] + 0.055),
                       ha="right", color=variant["color"], fontsize=8)
    ax_f1.axvline(args.threshold, color=COLOR_NEUTRAL, linewidth=0.9, linestyle="--", alpha=0.7)
    ax_f1.set_xlabel(r"decision threshold $\tau$")
    ax_f1.set_ylabel(r"$F_1$")
    ax_f1.set_xlim(0.0, 1.0)
    ax_f1.set_ylim(0.0, 1.04)
    ax_f1.set_title(r"frame-level $F_1$ vs.\ threshold")
    ax_f1.legend(loc="lower left", frameon=False)
    style_axis(ax_f1)

    # --- Panel C: episode-level outcomes -----------------------------------------
    categories = ["fires in tail", "fires 1--2 frames early", "fires earlier",
                  "never fires (missed)", "false fire (failure ep.)"]
    y = np.arange(len(categories))
    bar_height = 0.38
    for offset, (name, variant) in zip((+bar_height / 2, -bar_height / 2), VARIANTS.items()):
        outcomes = data[name]["outcomes"]
        counts = [outcomes[c] for c in categories]
        bars = ax_outcomes.barh(y + offset, counts, height=bar_height - 0.04,
                                color=variant["color"], label=variant["label"])
        for bar, count in zip(bars, counts):
            ax_outcomes.annotate(str(count),
                                 xy=(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2),
                                 va="center", ha="left", fontsize=8, color=COLOR_NEUTRAL)
    outcomes = any_data["outcomes"]
    ax_outcomes.set_yticks(y)
    ax_outcomes.set_yticklabels([rf"{c}" for c in categories])
    ax_outcomes.invert_yaxis()
    ax_outcomes.set_xlabel(
        rf"episodes (of {outcomes['_n_success']} success / {outcomes['_n_failure']} failure test episodes)"
    )
    ax_outcomes.set_xlim(0, outcomes["_n_success"] + 3)
    ax_outcomes.set_title(rf"episode-level behaviour at $\tau={args.threshold}$")
    ax_outcomes.legend(loc="lower right", frameon=False)
    ax_outcomes.grid(axis="y", visible=False)
    style_axis(ax_outcomes)
    ax_outcomes.grid(axis="y", visible=False)

    # --- Panel D: probability traces (state variant) ------------------------------
    state_data = data["vision_state"]
    successful = results["vision_state"]["successful_episodes"]
    max_frames = 40
    for episode, prob in state_data["probs"].items():
        tail_prob = prob[-max_frames:]
        x = np.arange(len(tail_prob) - 1, -1, -1)  # frames before episode end
        is_success = episode in successful
        ax_traces.plot(
            x, tail_prob,
            color=COLOR_STATE if is_success else COLOR_NEUTRAL,
            alpha=0.35 if is_success else 0.55,
            linewidth=1.0 if is_success else 1.2,
            linestyle="-" if is_success else (0, (4, 2)),
        )
    ax_traces.axhline(args.threshold, color=COLOR_NEUTRAL, linewidth=0.9, linestyle="--", alpha=0.7)
    ax_traces.annotate(rf"$\tau={args.threshold}$", xy=(max_frames - 1.2, args.threshold + 0.025),
                       ha="left", fontsize=8, color=COLOR_NEUTRAL)
    ax_traces.axvspan(-0.5, TAIL - 1, color=COLOR_STATE, alpha=0.10, linewidth=0)
    ax_traces.annotate(r"relabeled tail", xy=(TAIL - 0.6, 0.06), ha="right", fontsize=8, color=COLOR_NEUTRAL)
    handles = [
        plt.Line2D([], [], color=COLOR_STATE, linewidth=1.4, label=r"successful episode"),
        plt.Line2D([], [], color=COLOR_NEUTRAL, linewidth=1.4, linestyle=(0, (4, 2)), label=r"failure episode"),
    ]
    ax_traces.legend(handles=handles, loc="center left", frameon=False)
    ax_traces.set_xlim(max_frames - 1, -0.5)
    ax_traces.set_ylim(-0.02, 1.05)
    ax_traces.set_xlabel(r"frames before episode end")
    ax_traces.set_ylabel(r"$p(\mathrm{success})$")
    ax_traces.set_title(rf"per-frame success probability, {VARIANTS['vision_state']['label']} (test episodes)")
    style_axis(ax_traces)

    fig.savefig(pdf_path)
    plt.close(fig)


def write_summary(data: dict[str, dict], args: argparse.Namespace, path: Path) -> None:
    lines = [
        "# Reward classifier report",
        "",
        f"- dataset: `{args.dataset_root}`",
        f"- steps: {args.steps}, seed: {args.seed}, operating threshold: {args.threshold}",
        "",
        "| variant | accuracy | precision | recall | F1 | in tail | 1-2 early | earlier | missed | false fire |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, variant in VARIANTS.items():
        thresholds, precision, recall, f1 = data[name]["sweep"]
        op = np.argmin(np.abs(thresholds - args.threshold))
        outcomes = data[name]["outcomes"]
        lines.append(
            f"| {variant['label'].replace('$+$', '+')} | {data[name]['accuracy']:.4f} | {precision[op]:.3f} "
            f"| {recall[op]:.3f} | {f1[op]:.3f} | {outcomes['fires in tail']} "
            f"| {outcomes['fires 1--2 frames early']} | {outcomes['fires earlier']} "
            f"| {outcomes['never fires (missed)']} | {outcomes['false fire (failure ep.)']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    init_logging()

    results: dict[str, dict] = {}
    for name, variant in VARIANTS.items():
        logging.info("=== training variant '%s' ===", name)
        train_args = trainer.parse_args([
            "--dataset-root", str(args.dataset_root),
            "--output-dir", str(args.output_dir / "models" / name),
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--seed", str(args.seed),
            "--eval-freq", str(args.steps),
            *(["--use-state"] if variant["use_state"] else []),
        ])
        results[name] = trainer.run_training(train_args)

    data: dict[str, dict] = {}
    for name, result in results.items():
        test_set = set(result["test_frames"])
        test_episodes = [ep for ep, frames in result["episode_frames"].items() if frames[0] in test_set]
        probs = collect_episode_probs(result, test_episodes)
        y_prob = np.concatenate([probs[ep] for ep in probs])
        y_true = np.concatenate([result["labels"][result["episode_frames"][ep]] for ep in probs])
        data[name] = {
            "probs": probs,
            "outcomes": episode_outcomes(probs, result["successful_episodes"], args.threshold),
            "sweep": frame_level_sweep(probs, result["labels"], result["episode_frames"]),
            "accuracy": float(((y_prob > args.threshold) == (y_true > 0)).mean()),
        }

    pdf_path = args.output_dir / "reward_classifier_report.pdf"
    render_report(results, data, args, pdf_path)
    write_summary(data, args, args.output_dir / "summary.md")
    logging.info("Report written to %s", pdf_path)


if __name__ == "__main__":
    main()
