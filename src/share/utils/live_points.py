"""Live Rerun visualization of an MP-Net's tracked 3D points.

Logs whatever ``mp_net.get_display_points()`` reports for the currently active
primitive -- typically each robot's end-effector, plus a shared virtual TCP
("vtcp") for primitives that track one (see
``ManipulationPrimitive.get_display_points`` and
``rail_bimanual_grasp.CooperativeFramePrimitive.get_display_points``). MP-Net and
this module stay agnostic of what any of that means; the active primitive alone
decides what it has to show, and how many points that is. Callers (record.py,
actor_server.py) just call ``log_mpnet_live_points(mp_net)`` once per step,
after ``rerun`` has been initialized (``lerobot.utils.visualization_utils.
init_rerun``) -- same live recording ``log_rerun_data`` writes camera/action data
into, so both show up in one Rerun session.
"""

from typing import Any

import rerun as rr

# Fixed colors for the names rail_bimanual_grasp's CooperativeFramePrimitive
# reports today, so left/right/vtcp stay visually consistent across primitives and
# runs; any other name (e.g. a future single-robot env, or "hoermann") falls back
# to _DEFAULT_COLOR and still gets logged and labeled fine, just uncolored.
_COLORS: dict[str, tuple[int, int, int, int]] = {
    "left": (69, 154, 255, 255),
    "right": (255, 144, 61, 255),
    "vtcp": (255, 189, 46, 255),
}
_DEFAULT_COLOR = (143, 153, 168, 255)

_ENTITY_PREFIX = "mpnet/live_points"


def log_mpnet_live_points(mp_net: Any) -> None:
    """Log the active primitive's live 3D points to the current Rerun recording.

    Each named point gets its own entity path (``mpnet/live_points/<name>``) so it
    keeps its own trail/visibility toggle in the viewer. Clears the whole group
    first, so a point that stops being reported (e.g. "vtcp" outside a cooperative
    primitive) disappears instead of lingering at its last position -- a no-op,
    silent by design, when the active primitive reports nothing at all (see
    ManipulationPrimitive.get_display_points's default).
    """
    points = mp_net.get_display_points()
    rr.log(_ENTITY_PREFIX, rr.Clear(recursive=True))
    for name, point in points.items():
        rr.log(
            f"{_ENTITY_PREFIX}/{name}",
            rr.Points3D([point], colors=[_COLORS.get(name, _DEFAULT_COLOR)], labels=[name]),
        )
