import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Visualize a LeRobot dataset stored locally.")
    parser.add_argument("--root", type=Path, required=True, help="Root directory of the dataset.")
    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="Episode to visualize (default: 0).",
    )
    args, extra = parser.parse_known_args()

    root = args.root.resolve()
    repo_id = f"{root.parent.name}/{root.name}"

    cmd = [
        "lerobot-dataset-viz",
        "--root", str(root),
        "--repo-id", repo_id,
        "--episode-index", str(args.episode_index),
        "--display-compressed-images",
        *extra,
    ]
    sys.exit(subprocess.call(cmd))
