"""Save and compare env config snapshots alongside datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

ENV_CONFIG_FILENAME = "env_config.json"


def save_env_config_snapshot(env_config: Any, dataset_root: str | Path) -> None:
    from share.workspace.mpnet import save_mpnet_config

    path = Path(dataset_root) / ENV_CONFIG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    save_mpnet_config(env_config, path)
    logging.info("[ENV SNAPSHOT] Saved env config snapshot to %s", path)


def load_env_config_snapshot(dataset_root: str | Path) -> Any | None:
    from share.workspace.mpnet import load_mpnet_config

    path = Path(dataset_root) / ENV_CONFIG_FILENAME
    if not path.exists():
        return None
    return load_mpnet_config(path)


def _flatten(obj: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(value, f"{prefix}.{key}" if prefix else key, out)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            _flatten(value, f"{prefix}[{idx}]", out)
    else:
        out[prefix] = obj


def diff_env_configs(saved: Any, current: Any) -> list[tuple[str, Any, Any]]:
    """Return (path, saved_val, current_val) for every leaf that differs."""
    from share.workspace.mpnet import _encode_mpnet

    saved_flat: dict[str, Any] = {}
    current_flat: dict[str, Any] = {}
    _flatten(_encode_mpnet(saved), "", saved_flat)
    _flatten(_encode_mpnet(current), "", current_flat)

    all_keys = saved_flat.keys() | current_flat.keys()
    diffs = []
    for key in sorted(all_keys):
        sv = saved_flat.get(key, "<missing>")
        cv = current_flat.get(key, "<missing>")
        if sv != cv:
            diffs.append((key, sv, cv))
    return diffs


def check_and_confirm_env_config(env_config: Any, dataset_root: str | Path) -> None:
    """Compare env config against saved snapshot; abort unless the user confirms."""
    snapshot = load_env_config_snapshot(dataset_root)
    if snapshot is None:
        logging.warning(
            "[ENV SNAPSHOT] No env config snapshot found in %s; skipping compatibility check",
            dataset_root,
        )
        return

    diffs = diff_env_configs(snapshot, env_config)
    if not diffs:
        logging.info("[ENV SNAPSHOT] Env config matches dataset snapshot — OK")
        return

    col_w = max(len(d[0]) for d in diffs)
    lines = [
        f"\n[WARNING] Current env config differs from the snapshot saved with the dataset at:",
        f"          {dataset_root}",
        "",
        f"  {'Field':<{col_w}}  {'Saved (dataset)':<30}  Current",
        f"  {'-' * col_w}  {'-' * 30}  -------",
    ]
    for path, sv, cv in diffs:
        lines.append(f"  {path:<{col_w}}  {str(sv):<30}  {cv}")
    print("\n".join(lines))

    answer = input("\nContinue with mismatched env config? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        raise SystemExit("Aborted: env config mismatch with dataset snapshot.")
