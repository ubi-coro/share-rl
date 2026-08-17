"""Top-level experiments package.

``import experiments`` is the hook entry points (e.g. ``share/scripts/record.py``)
use to register every env config in ``experiments.envs`` with draccus before
parsing ``--env.type=<name>``.
"""

from experiments import envs  # noqa: F401

__all__ = ["envs"]
