"""Import side effect: registers every experiment env config with draccus.

``record.py`` (and any other entry point) does a bare ``import experiments``
to make ``--env.type=<name>`` resolvable. Since each env config only
registers itself with ``EnvConfig.register_subclass`` when its module is
actually imported, every new env config module added under
``experiments/envs`` must be imported here (or from a subpackage ``__init__``
that is itself imported here, as with ``foundationpose``).
"""

import logging

from experiments.envs.fiddle_out import DemoUR3eTeleopFiddleOutEnvConfig
from experiments.envs.teleop_spacemouse_6dof import TeleopSpaceMouse6DofEnvConfig

logger = logging.getLogger(__name__)

__all__ = [
    "DemoUR3eTeleopFiddleOutEnvConfig",
    "TeleopSpaceMouse6DofEnvConfig",
]

try:
    from experiments.envs import foundationpose  # noqa: F401

    __all__.append("foundationpose")
except ImportError as exc:
    # foundationpose pulls in the optional `pose_estimation` package, which
    # isn't installed in every environment (e.g. plain teleop workstations).
    # Don't take every other env config down with it -- skip and say why.
    logger.warning("Skipping experiments.envs.foundationpose registration: %s", exc)
