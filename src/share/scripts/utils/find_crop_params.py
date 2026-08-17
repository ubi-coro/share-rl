"""Interactive tool for choosing a connector's camera crops.

The policy sees only these crops, so they decide what the robot can actually learn from:
each box should be centred on the socket and tight enough that the plug fills it.

Usage:
    MPLBACKEND=qtagg python src/share/scripts/utils/find_crop_params.py \
        --object-dir /media/internal/nvme/shared_data/hoermann/plugs/NewPlug

Live views of the wrist and side cameras appear side by side.
- Drag a rectangle around the socket in each view.
- Drag inside a rectangle to move it, or drag its handles to resize it.
- Press 'w' to write the crops to <object_dir>/connector.json.
- Press 'q' to quit.

Crops are stored as [top, left, height, width] -- note top is the y coordinate and left is
x. Usually invoked via tools/setup_connector.sh.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from experiments.envs.hoermann.connector.cameras import (
    SIDE_SERIAL,
    WRIST_SERIAL,
    make_hoermann_cameras,
)
from experiments.envs.hoermann.connector.spec import update_connector_json
from share.scripts.utils._mpl_camera_tool import (
    connect_camera,
    require_interactive_matplotlib_backend,
)

CAMERA_KEYS = ("wrist", "side")

# ---- state ----------------------------------------------------------------
_quit_requested = False
_write_requested = False


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--object-dir", type=Path, required=True,
                        help="Connector dir; the crops are written to its connector.json")
    parser.add_argument("--size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--resize", type=int, nargs=2, default=[64, 64],
                        metavar=("HEIGHT", "WIDTH"),
                        help="What the crops are resized to for the policy (default: 64 64)")
    parser.add_argument("--wrist-serial", default=WRIST_SERIAL)
    parser.add_argument("--side-serial", default=SIDE_SERIAL)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the crops without writing connector.json")
    args = parser.parse_args()

    if not args.object_dir.is_dir():
        raise SystemExit(f"No such connector dir: {args.object_dir}")

    require_interactive_matplotlib_backend()
    # Straight from the rig config: this runs before connector.json exists, so an env
    # cannot be built here.
    camera_configs = make_hoermann_cameras(args.wrist_serial, args.side_serial)
    cameras = {key: connect_camera(camera_configs, key) for key in CAMERA_KEYS}

    try:
        plt.ion()
        figure, axes_list = plt.subplots(1, len(CAMERA_KEYS), num="Crop calibration",
                                         figsize=(12, 5))
        axes_by_camera = dict(zip(CAMERA_KEYS, axes_list))
        artists, selectors = {}, {}
        selected: set[str] = set()

        for camera, axes in axes_by_camera.items():
            frame = cameras[camera].async_read()
            axes.set_axis_off()
            axes.set_title(camera)
            artists[camera] = axes.imshow(frame, interpolation="nearest", vmin=0, vmax=255)
            def on_select(_click, _release, name=camera):
                selected.add(name)
                print(f"  {name}: [top, left, height, width] = "
                      f"{_crop_box(selectors[name].extents, artists[name].get_array().shape)}")

            selectors[camera] = RectangleSelector(
                axes, on_select, button=[1], minspanx=1, minspany=1,
                spancoords="data", interactive=True, drag_from_anywhere=True,
                use_data_coordinates=True,
                props={"facecolor": "none", "edgecolor": "lime", "linewidth": 1.5},
                handle_props={"markeredgecolor": "lime", "markerfacecolor": "lime"},
            )

        figure.suptitle("[drag] draw/move/resize each crop   [w] write   [q] quit")
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

        frames = {camera: cameras[camera].async_read() for camera in CAMERA_KEYS}
    finally:
        for camera in cameras.values():
            camera.disconnect()
        plt.close("all")

    print("\n" + "=" * 60)
    if not _write_requested:
        raise SystemExit("Quit without writing ('q'). Press 'w' to save. Nothing was changed.")

    missing = [camera for camera in CAMERA_KEYS if camera not in selected]
    if missing:
        raise SystemExit(
            f"No box was placed for: {', '.join(missing)}. Both cameras need one -- the "
            f"policy reads both. Nothing was saved."
        )

    params = {
        camera: _crop_box(selectors[camera].extents, frames[camera].shape)
        for camera in CAMERA_KEYS
    }
    for camera, box in params.items():
        print(f"  {camera}: [top, left, height, width] = {box}")
    print(f"  resize_size: {list(args.resize)}")

    if args.dry_run:
        print("\nDry run -- not writing.")
        return
    path = update_connector_json(
        args.object_dir, {"crop": {"params": params, "resize_size": list(args.resize)}}
    )
    print(f"\nWrote {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
