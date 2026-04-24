import logging


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
    logging.info(
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
