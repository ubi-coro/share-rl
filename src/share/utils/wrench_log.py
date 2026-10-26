"""Reading the end-effector wrench off an observation and logging it per episode, for diagnosis."""

import csv
import math
from pathlib import Path
from typing import Any, Sequence

import torch


def read_ee_wrenches(obs: dict[str, Any]) -> dict[str, float]:
    """Every `{robot}.{axis}.ee_wrench` entry of an observation, whether or not the policy state keeps it."""
    return {
        key: float(torch.as_tensor(value).flatten()[0])
        for key, value in obs.items()
        if key.endswith(".ee_wrench")
    }


def write_wrench_log(path: Path, rows: Sequence[dict[str, float]], fps: float) -> None:
    """Write one episode's wrench readings as CSV, one row per control step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "time_s", *rows[0]])
        for step, row in enumerate(rows):
            writer.writerow([step, f"{step / fps:.4f}", *row.values()])


def wrench_peak_summary(rows: Sequence[dict[str, float]], fps: float) -> str:
    """One line with the peak force magnitude of every robot that reports a wrench, and when it occurred."""
    parts = []
    for robot in sorted({key.rsplit(".", 2)[0] for key in rows[0]}):
        forces = [
            math.sqrt(sum(row.get(f"{robot}.{axis}.ee_wrench", 0.0) ** 2 for axis in ("x", "y", "z")))
            for row in rows
        ]
        peak = max(range(len(forces)), key=forces.__getitem__)
        parts.append(f"{robot} peak force {forces[peak]:.1f} N at {peak / fps:.1f}s")
    return ", ".join(parts)
