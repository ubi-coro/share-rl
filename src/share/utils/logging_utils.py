import logging
from typing import Any

import torch


def _to_hz(duration_s: float) -> float:
    if duration_s <= 0.0:
        return 0.0
    return 1.0 / duration_s


def log_runtime_frequency(
    *,
    prefix: str,
    primitive: str,
    loop_dt_s: float,
    work_dt_s: float,
    work_label: str,
    task: str | None = None,
) -> None:
    """Log a compact runtime timing line shared by record and actor loops."""

    task_fragment = f" task={task}" if task is not None else ""
    # DEBUG, not INFO: this fires once per control-loop step (up to fps Hz), so at INFO it
    # floods the console with a full line every step -- right below the \r-overwriting status
    # line that already shows the same Hz live. init_logging's file handler stays at DEBUG by
    # default, so the full per-step trace is still captured for post-hoc debugging; only the
    # live console (default INFO) is decluttered.
    logging.debug(
        "[%s] primitive=%s%s loop=%5.2fms (%3.1fhz) %s=%5.2fms (%3.1fhz)",
        prefix,
        primitive,
        task_fragment,
        loop_dt_s * 1000.0,
        _to_hz(loop_dt_s),
        work_label,
        work_dt_s * 1000.0,
        _to_hz(work_dt_s),
    )


def primary_loss(training_infos: dict[str, Any], *, is_dagger_bc_policy: bool) -> tuple[str, float]:
    """Pick the loss metric that should headline learner console logging."""

    loss_name = "loss_bc" if is_dagger_bc_policy else "loss_critic"
    loss_value = training_infos.get(loss_name)
    if isinstance(loss_value, torch.Tensor):
        loss_value = float(loss_value.item())
    elif isinstance(loss_value, (int, float)):
        loss_value = float(loss_value)
    else:
        loss_value = float("nan")
    return loss_name, loss_value
