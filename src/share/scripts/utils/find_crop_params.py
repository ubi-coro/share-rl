"""Interactively choose image crops for every camera in an environment.

Usage:
    MPLBACKEND=qtagg python src/share/scripts/utils/find_crop_params.py \
        --env.type=teleop_spacemouse_6dof

Add --object_dir=/path/to/output to save connector.json; without it, pressing "w"
prints the crop parameters only.

The selected environment supplies the camera configurations through env.cameras.
Only cameras are connected; the environment's robots and teleoperators are not started.

Drag a rectangle around the relevant scene area in each view, press "w" to print
the crop parameters (and write <object_dir>/connector.json when configured), or
press "q" to quit. Crops are stored as
[top, left, height, width] -- top is the y coordinate and left is x.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from lerobot.configs import parser

from share.envs.manipulation_primitive_net.config_manipulation_primitive_net import (
    ManipulationPrimitiveNetConfig,
)
from share.scripts.utils._mpl_camera_tool import require_interactive_matplotlib_backend

# ---- state ----------------------------------------------------------------
_quit_requested = False
_write_requested = False


@dataclass(kw_only=True)
class CropParamsConfig:
    """CLI configuration for camera crop calibration."""

    env: ManipulationPrimitiveNetConfig
    object_dir: Path | None = None
    resize_size: tuple[int, int] = (64, 64)
    dry_run: bool = False


def _on_key(event) -> None:
    global _quit_requested, _write_requested
    if event.key == "q":
        _quit_requested = True
        plt.close(event.canvas.figure)
    elif event.key == "w":
        _write_requested = True
        _quit_requested = True
        plt.close(event.canvas.figure)


def _crop_box(extents: tuple[float, float, float, float], shape) -> list[int]:
    """Selector extents -> [top, left, height, width], clamped to the frame."""
    frame_height, frame_width = shape[:2]
    x_min, x_max, y_min, y_max = extents
    left = max(0, min(math.floor(min(x_min, x_max)), frame_width - 1))
    right = max(left + 1, min(math.ceil(max(x_min, x_max)), frame_width))
    top = max(0, min(math.floor(min(y_min, y_max)), frame_height - 1))
    bottom = max(top + 1, min(math.ceil(max(y_min, y_max)), frame_height))
    return [top, left, bottom - top, right - left]


def _write_connector_json(
    object_dir: Path,
    params: dict[str, list[int]],
    resize_size: tuple[int, int],
) -> Path:
    """Merge crop settings into connector.json without discarding other settings."""
    path = object_dir / "connector.json"
    payload: dict = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in {path}, got {type(payload).__name__}.")

    payload["crop"] = {"params": params, "resize_size": list(resize_size)}
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")
    return path


def _grid_shape(camera_count: int) -> tuple[int, int]:
    """Return a compact row/column layout for the camera views."""
    columns = max(1, math.ceil(math.sqrt(camera_count)))
    rows = math.ceil(camera_count / columns)
    return rows, columns


@parser.wrap()
def find_crop_params(cfg: CropParamsConfig) -> None:
    """Connect an environment's configured cameras and interactively crop them."""
    if cfg.object_dir is not None and not cfg.object_dir.is_dir():
        raise SystemExit(f"No output directory: {cfg.object_dir}")
    if not cfg.env.cameras:
        raise SystemExit(f"Environment '{cfg.env.type}' has no configured cameras.")
    if any(int(size) <= 0 for size in cfg.resize_size):
        raise SystemExit(f"resize_size must contain positive values, got {cfg.resize_size!r}.")

    require_interactive_matplotlib_backend()

    from lerobot.cameras import make_cameras_from_configs

    camera_keys = tuple(cfg.env.cameras)
    cameras = make_cameras_from_configs(cfg.env.cameras)
    connected_cameras = []

    try:
        for camera in cameras.values():
            camera.connect()
            connected_cameras.append(camera)

        plt.ion()
        rows, columns = _grid_shape(len(camera_keys))
        figure, axes_grid = plt.subplots(
            rows,
            columns,
            num="Crop calibration",
            figsize=(6 * columns, 5 * rows),
            squeeze=False,
        )
        axes_by_camera = {}
        for index, camera_key in enumerate(camera_keys):
            axes_by_camera[camera_key] = axes_grid[index // columns, index % columns]
        for index in range(len(camera_keys), rows * columns):
            axes_grid[index // columns, index % columns].set_visible(False)

        artists, selectors = {}, {}
        selected: set[str] = set()

        for camera, axes in axes_by_camera.items():
            frame = cameras[camera].async_read()
            axes.set_axis_off()
            axes.set_title(camera)
            artists[camera] = axes.imshow(frame, interpolation="nearest", vmin=0, vmax=255)

            def on_select(_click, _release, name=camera):
                selected.add(name)
                print(
                    f"  {name}: [top, left, height, width] = "
                    f"{_crop_box(selectors[name].extents, artists[name].get_array().shape)}"
                )

            selectors[camera] = RectangleSelector(
                axes,
                on_select,
                button=[1],
                minspanx=1,
                minspany=1,
                spancoords="data",
                interactive=True,
                drag_from_anywhere=True,
                use_data_coordinates=True,
                props={"facecolor": "none", "edgecolor": "lime", "linewidth": 1.5},
                handle_props={"markeredgecolor": "lime", "markerfacecolor": "lime"},
            )

        figure.suptitle(
            f"[drag] draw/move/resize each crop   [w] write   [q] quit   |   env={cfg.env.type}"
        )
        figure.canvas.mpl_connect("key_press_event", _on_key)
        figure.tight_layout()
        figure.show()

        print(__doc__)
        while not _quit_requested and plt.fignum_exists(figure.number):
            for camera, axes in axes_by_camera.items():
                frame = cameras[camera].async_read()
                artists[camera].set_data(frame)
            figure.canvas.draw()
            plt.pause(0.03)

        frames = {camera: cameras[camera].async_read() for camera in camera_keys}
    finally:
        for camera in connected_cameras:
            camera.disconnect()
        plt.close("all")

    print("\n" + "=" * 60)
    if not _write_requested:
        raise SystemExit("Quit without writing ('q'). Press 'w' to save. Nothing was changed.")

    missing = [camera for camera in camera_keys if camera not in selected]
    if missing:
        raise SystemExit(
            f"No box was placed for: {', '.join(missing)}. Every configured camera needs one. "
            "Nothing was saved."
        )

    params = {
        camera: _crop_box(selectors[camera].extents, frames[camera].shape)
        for camera in camera_keys
    }
    for camera, box in params.items():
        print(f"  {camera}: [top, left, height, width] = {box}")
    print(f"  resize_size: {list(cfg.resize_size)}")

    if cfg.dry_run or cfg.object_dir is None:
        message = "Dry run -- not writing." if cfg.dry_run else "No object_dir supplied -- not writing."
        print(f"\n{message}")
        return
    path = _write_connector_json(cfg.object_dir, params, cfg.resize_size)
    print(f"\nWrote {path}")
    print("=" * 60)


def main() -> None:
    # Like record.py, import the experiment package before draccus parses --env.type.
    import experiments  # noqa: F401

    find_crop_params()


if __name__ == "__main__":
    main()
