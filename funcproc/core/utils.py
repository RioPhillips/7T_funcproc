"""funcproc.core.utils
====================
Generic, dependency-light helpers shared across the package.

The colleague's commands lean on ``lazyfmri.utils`` for a much larger set of
helpers (split_bids_components, get_file_from_substring, get_vertex_nr, ...);
we port/replace the subset we actually need here as the package grows.
"""

import sys


def verbose(msg, flag=True, **kwargs):
    """Print ``msg`` only if ``flag`` is truthy.

    Mirrors ``lazyfmri.utils.verbose`` so command code can gate informational
    output behind a ``-v/--verbose`` flag.
    """
    if flag:
        print(msg, **kwargs)


class color:
    """ANSI escape codes for the occasional highlighted log line."""
    RED = "\033[31m"
    GREEN = "\033[32m"
    BOLD = "\033[1m"
    END = "\033[0m"


# --- to implement / port as the commands are built out ---
# def split_bids_components(path): ...
# def get_file_from_substring(filters, source, ...): ...
# def get_vertex_nr(subject, ...): ...