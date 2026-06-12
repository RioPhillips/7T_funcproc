"""funcproc.core
==============
Shared, dependency-light building blocks used across the commands. Submodules
are re-exported here so callers can do ``from funcproc.core import bids, io``
or ``from funcproc.core import verbose``.

Note: heavy/optional dependencies (nibabel, prfpy, fmriproc, ...) are imported
lazily *inside* functions, so importing ``funcproc.core`` stays cheap and the
CLI starts fast regardless of which command is run.
"""

from . import bids, config, io, pybest, unzscore, utils
from .utils import verbose

__all__ = [
    "bids",
    "config",
    "io",
    "pybest",
    "unzscore",
    "utils",
    "verbose",
]