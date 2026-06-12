"""funcproc
================
Functional (BOLD) analysis for the 7T pipeline. Operates on fMRIPrep outputs
(from the 7T_anatprep stage) and provides command-line subcommands:

    funcproc denoise   denoise fMRIPrep output (via pybest) -> PSC-ready data
    funcproc prf       population receptive field fitting with prfpy

Command logic is based on the colleague's `call_pybest` / `call_prf` wrappers
(fmriproc / lazyfmri). Marcus's example-style prfpy code and Dag's analysis /
visualisation tooling (dag_prf_utils) will be integrated later, under vendor/.
"""

__version__ = "0.0.1"