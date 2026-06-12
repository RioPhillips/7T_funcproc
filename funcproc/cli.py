"""funcproc.cli
=================
Top-level command-line dispatcher.

Usage:
    funcproc <subcommand> [options]
    funcproc denoise ...
    funcproc prf ...

Each subcommand lives in funcproc/commands/<name>.py and exposes:
    add_parser(subparsers)  -> registers the subcommand and its arguments
    run(args)               -> executes the subcommand
"""

import argparse
import sys

from funcproc import __version__
from funcproc.commands import COMMANDS


def build_parser():
    parser = argparse.ArgumentParser(
        prog="funcproc",
        description="Functional (BOLD) analysis on fMRIPrep outputs (denoising, pRF fitting).",
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"funcproc {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<subcommand>")
    for module in COMMANDS.values():
        module.add_parser(subparsers)
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    return COMMANDS[args.command].run(args)


if __name__ == "__main__":
    raise SystemExit(main())