"""funcproc subcommands.

Each command module defines two functions:
    add_parser(subparsers) -> registers the subcommand and its arguments
    run(args)              -> executes the subcommand

The COMMANDS registry below is the single place where subcommands are wired in;
``funcproc.cli`` consumes it, so adding a new command means adding its module
here (and nothing else).
"""

from . import denoise, prf

# name -> module (insertion order = order shown in --help)
COMMANDS = {
    "denoise": denoise,
    "prf": prf,
}

__all__ = ["denoise", "prf", "COMMANDS"]