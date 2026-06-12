"""funcproc.core.pybest
=========================
Thin wrapper around the ``pybest`` executable, mirroring the command that the
colleague's ``call_pybest`` builds. Surface spaces (fsnative/fsaverage) are run
per hemisphere; volumetric spaces are run once per task.
"""

import shutil
import subprocess


def is_surface_space(space):
    return isinstance(space, str) and space.startswith("fs")


def build_pybest_cmd(
    input_dir,
    output_dir,
    sub,
    task,
    space="fsnative",
    ses=None,
    hemi=None,
    n_comps=20,
    n_cpus=None,
    extra=None,
):
    """Construct the pybest command for one task (and hemisphere, if surface)."""
    cmd = [
        "pybest",
        "--subject", str(sub),
    ]
    if ses is not None:
        cmd += ["--session", str(ses)]
    cmd += [
        "--out-dir", str(output_dir),
        "--n-comps", str(n_comps),
        "--space", str(space),
        "--verbose", "ERROR",
        "--save-all",
        "--task", str(task),
    ]
    if hemi is not None:
        cmd += ["--hemi", str(hemi)]
    if n_cpus is not None:
        cmd += ["--n-cpus", str(n_cpus)]
    if extra:
        cmd += list(extra)
    cmd += [str(input_dir)]
    return cmd


def run_pybest(
    input_dir,
    output_dir,
    sub,
    tasks,
    space="fsnative",
    ses=None,
    n_comps=20,
    n_cpus=None,
    extra=None,
    dry_run=False,
    verbose=True,
):
    """Run pybest over the given task(s). Loops hemispheres for surface spaces.

    Returns the list of commands that were (or would be) executed.
    """
    if shutil.which("pybest") is None and not dry_run:
        raise RuntimeError(
            "Could not find the 'pybest' executable on PATH. Activate the "
            "conda environment that has pybest installed (see "
            "https://github.com/lukassnoek/pybest)."
        )

    hemis = ["L", "R"] if is_surface_space(space) else [None]
    commands = []
    for task in tasks:
        for hemi in hemis:
            cmd = build_pybest_cmd(
                input_dir, output_dir, sub, task, space=space, ses=ses,
                hemi=hemi, n_comps=n_comps, n_cpus=n_cpus, extra=extra,
            )
            commands.append(cmd)
            if verbose:
                print(" " + " ".join(cmd), flush=True)
            if not dry_run:
                subprocess.run(cmd, check=True)
    return commands