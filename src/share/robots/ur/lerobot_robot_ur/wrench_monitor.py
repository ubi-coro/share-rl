"""Asynchronous live visualization for UR controller wrench diagnostics."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

WRENCH_MONITOR_AXES = ("x", "y", "z", "rx", "ry", "rz")
WRENCH_MONITOR_SERIES = (
    "wrench_monitor_measured",
    "wrench_monitor_filtered",
    "wrench_monitor_nominal",
    "wrench_monitor_commanded",
    "wrench_monitor_adaptive_limit",
)


class WrenchMonitorHistory:
    """Fixed-size history of controller-rate wrench monitor samples."""

    def __init__(self, max_samples: int):
        self.wrench_monitor_timestamps = deque(maxlen=max_samples)
        self.wrench_monitor_sequences = deque(maxlen=max_samples)
        self.wrench_monitor_values = {
            name: deque(maxlen=max_samples) for name in WRENCH_MONITOR_SERIES
        }
        self.wrench_monitor_dropped_samples = 0

    def append_batch(self, batch: dict[str, np.ndarray], after_sequence: int) -> int:
        """Append unseen samples and return the newest sequence number."""
        sequences = np.asarray(batch["wrench_monitor_sequence"], dtype=np.int64)
        unseen_indices = np.flatnonzero(sequences > after_sequence)
        self.wrench_monitor_dropped_samples = 0
        previous_sequence = after_sequence
        for index in unseen_indices:
            sequence = int(sequences[index])
            if previous_sequence >= 0 and sequence > previous_sequence + 1:
                self.wrench_monitor_dropped_samples += sequence - previous_sequence - 1
            previous_sequence = sequence
            self.wrench_monitor_sequences.append(int(sequences[index]))
            self.wrench_monitor_timestamps.append(
                float(batch["wrench_monitor_timestamp"][index])
            )
            for name in WRENCH_MONITOR_SERIES:
                self.wrench_monitor_values[name].append(
                    np.asarray(batch[name][index], dtype=np.float64).copy()
                )
        if unseen_indices.size == 0:
            return after_sequence
        return int(sequences[unseen_indices[-1]])

    def arrays(self) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        timestamps = np.asarray(self.wrench_monitor_timestamps, dtype=np.float64)
        values = {
            name: np.asarray(series, dtype=np.float64)
            for name, series in self.wrench_monitor_values.items()
        }
        return timestamps, values


class WrenchMonitorProcess(mp.Process):
    """Render wrench telemetry independently from robot observations and control."""

    def __init__(
        self,
        wrench_monitor_rb,
        *,
        controller_unexpected_exit_event,
        wrench_monitor_refresh_hz: float,
        wrench_monitor_history_s: float,
        wrench_monitor_sample_hz: float,
    ):
        super().__init__(name="URWrenchMonitor", daemon=True)
        self.wrench_monitor_rb = wrench_monitor_rb
        self.controller_unexpected_exit_event = controller_unexpected_exit_event
        self.wrench_monitor_refresh_hz = float(wrench_monitor_refresh_hz)
        self.wrench_monitor_history_s = float(wrench_monitor_history_s)
        self.wrench_monitor_sample_hz = float(wrench_monitor_sample_hz)
        self.wrench_monitor_stop_event = mp.Event()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Request shutdown and wait briefly for the GUI process to exit."""
        self.wrench_monitor_stop_event.set()
        if self.pid is None:
            return
        self.join(timeout=timeout_s)
        if self.is_alive():
            logger.warning("Wrench monitor did not stop within %.1fs; terminating it.", timeout_s)
            self.terminate()
            self.join(timeout=timeout_s)

    def run(self) -> None:
        """Own the plotting GUI and periodically consume controller-rate batches."""
        try:
            import matplotlib.pyplot as plt

            max_samples = max(
                2,
                int(np.ceil(self.wrench_monitor_sample_hz * self.wrench_monitor_history_s)),
            )
            history = WrenchMonitorHistory(max_samples=max_samples)
            figure, axes, lines = _create_wrench_monitor_plot(plt)
            plt.show(block=False)
            last_sequence = -1
            refresh_period = 1.0 / self.wrench_monitor_refresh_hz

            while (
                not self.wrench_monitor_stop_event.is_set()
                and plt.fignum_exists(figure.number)
            ):
                loop_start = time.monotonic()
                count = self.wrench_monitor_rb.count
                if count > 0:
                    batch = self.wrench_monitor_rb.get_last_k(
                        min(count, self.wrench_monitor_rb.get_max_k)
                    )
                    last_sequence = history.append_batch(batch, after_sequence=last_sequence)
                    if history.wrench_monitor_dropped_samples:
                        logger.warning(
                            "Wrench monitor dropped %d controller samples.",
                            history.wrench_monitor_dropped_samples,
                        )
                    _update_wrench_monitor_plot(axes, lines, history)
                    figure.canvas.draw_idle()

                if self.controller_unexpected_exit_event.is_set():
                    snapshot_path = _save_wrench_monitor_snapshot(figure)
                    print(
                        "[URWrenchMonitor] Controller exited unexpectedly; "
                        f"saved the last wrench monitor to {snapshot_path}",
                        flush=True,
                    )
                    break

                elapsed = time.monotonic() - loop_start
                plt.pause(max(0.001, refresh_period - elapsed))
        except Exception:
            logger.exception("Wrench monitor process failed.")
        finally:
            try:
                import matplotlib.pyplot as plt

                plt.close("all")
            except Exception:
                pass


def _save_wrench_monitor_snapshot(figure, output_dir: str | Path | None = None) -> Path:
    """Save a crash snapshot to the system temporary directory."""
    snapshot_dir = Path(tempfile.gettempdir() if output_dir is None else output_dir)
    snapshot_path = snapshot_dir / (
        "ur_wrench_monitor_crash_"
        f"{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}_{time.time_ns()}.png"
    )
    figure.savefig(snapshot_path, dpi=150)
    return snapshot_path.resolve()


def _create_wrench_monitor_plot(plt) -> tuple[Any, Any, dict[str, list[Any]]]:
    figure, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    figure.canvas.manager.set_window_title("UR Wrench Monitor")
    figure.suptitle("UR controller-rate wrench telemetry")
    lines: dict[str, list[Any]] = {name: [] for name in WRENCH_MONITOR_SERIES}
    lines["wrench_monitor_negative_adaptive_limit"] = []

    styles = {
        "wrench_monitor_measured": dict(label="measured", color="0.65", alpha=0.6),
        "wrench_monitor_filtered": dict(label="filtered", color="tab:blue"),
        "wrench_monitor_nominal": dict(label="nominal", color="tab:orange", linestyle="--"),
        "wrench_monitor_commanded": dict(label="commanded", color="tab:green"),
        "wrench_monitor_adaptive_limit": dict(
            label="adaptive limit", color="tab:red", linestyle=":"
        ),
    }

    for axis_index, axis_name in enumerate(WRENCH_MONITOR_AXES):
        plot_axis = axes.flat[axis_index]
        for series_name in WRENCH_MONITOR_SERIES:
            (line,) = plot_axis.plot([], [], linewidth=1.2, **styles[series_name])
            lines[series_name].append(line)
        (negative_limit,) = plot_axis.plot(
            [], [], linewidth=1.2, color="tab:red", linestyle=":"
        )
        lines["wrench_monitor_negative_adaptive_limit"].append(negative_limit)
        plot_axis.set_title(axis_name)
        plot_axis.set_ylabel("N" if axis_index < 3 else "Nm")
        plot_axis.grid(True, alpha=0.3)
        plot_axis.legend(loc="upper right", fontsize="small")

    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 1].set_xlabel("Time [s]")
    figure.tight_layout()
    return figure, axes, lines


def _update_wrench_monitor_plot(
    axes,
    lines: dict[str, list[Any]],
    history: WrenchMonitorHistory,
) -> None:
    timestamps, values = history.arrays()
    if timestamps.size == 0:
        return
    times = timestamps - timestamps[-1]

    for axis_index in range(6):
        for series_name in WRENCH_MONITOR_SERIES:
            lines[series_name][axis_index].set_data(
                times,
                values[series_name][:, axis_index],
            )
        limit_values = values["wrench_monitor_adaptive_limit"][:, axis_index]
        lines["wrench_monitor_negative_adaptive_limit"][axis_index].set_data(
            times,
            -limit_values,
        )
        plot_axis = axes.flat[axis_index]
        plot_axis.set_xlim(float(times[0]), 0.0 if times.size > 1 else 1.0)
        plot_axis.relim()
        plot_axis.autoscale_view(scalex=False, scaley=True)
