"""funcproc.core.prf_data
=====================
Assemble pRF-ready time series from the denoise output: find the per-run
un-z-scored files, convert each to percent signal change (Marco Aqil's method) and median-average across runs.

prfpy wants data as (units x time); our denoise npy files are (time x vertices),
so we transpose on load.
"""

import os
import numpy as np

from funcproc.core import bids

opj = os.path.join


#  PSC 
def percent_change(ts, baseline=20):
    """Convert (vertices x time) to percent signal change, baseline median -> 0.

    Replicates marcus_prf_eg.utils.percent_change:
      1. scale each timecourse so its mean is 100,
      2. subtract the median of the baseline timepoints.
    ``baseline`` is either an int N (use the first N volumes) or a list/array of
    volume indices treated as the no-stimulation baseline.
    """
    ts = np.asarray(ts, dtype=float)
    if ts.ndim == 1:
        ts = ts[np.newaxis, :]
    t_dim = 1
    psc_factor = np.nan_to_num(100.0 / np.mean(ts, axis=t_dim))
    ts_m = ts * psc_factor[..., np.newaxis]
    if isinstance(baseline, (int, np.integer)):
        median_baseline = np.median(ts_m[:, :int(baseline)], axis=t_dim)
    else:
        median_baseline = np.median(ts_m[:, list(baseline)], axis=t_dim)
    return ts_m - median_baseline[..., np.newaxis]


def raw_ts_to_average_psc(raw_ts, baseline=20):
    """PSC each run then median-average across runs. ``raw_ts`` is a list of
    (vertices x time) arrays (or a single array)."""
    if not isinstance(raw_ts, list):
        raw_ts = [raw_ts]
    psc = [percent_change(run, baseline=baseline) for run in raw_ts]
    return np.median(np.array(psc), axis=0)


def infer_baseline_from_dm(dm, max_frac=0.5):
    """Number of leading all-zero (no-stimulation) frames in the design matrix.
    ``dm`` is (n_pix, n_pix, time). Capped at ``max_frac`` of the run length so a
    degenerate dm can't swallow the whole timecourse."""
    per_frame = dm.reshape(-1, dm.shape[-1]).sum(axis=0)
    n = 0
    for v in per_frame:
        if v > 0:
            break
        n += 1
    return min(n, int(max_frac * dm.shape[-1])) if n else 20


#  finding/loading 
def find_run_files(input_dir, sub, task, space, hemi, ses=None,
                   desc="unzscored", ext="npy"):
    """Find per-run surface files for one hemisphere, sorted by run number."""
    root = opj(input_dir, f"sub-{sub}")
    inc = [f"sub-{sub}", f"desc-{desc}_bold", f"space-{space}", f"hemi-{hemi}_"]
    if task is not None:
        inc += [f"task-{task}_"]
    if ses is not None:
        inc += [f"ses-{ses}_"]
    files = bids.find_files(root, include=inc, exclude=["raw"], extension=ext)
    # keep only per-run files (skip any run-less aggregate) and sort by run
    runned = [f for f in files if "run" in bids.parse_entities(f)]
    return sorted(runned, key=lambda f: int(bids.parse_entities(f)["run"]))


def load_runs_vertices_time(files):
    """Load npy run files as (vertices x time), transposing the stored
    (time x vertices) layout."""
    runs = []
    for f in files:
        a = np.load(f)
        runs.append(a.T if a.ndim == 2 else a)  # (time, vx) -> (vx, time)
    return runs


def assemble_psc(input_dir, sub, task, space, hemi, ses=None,
                 baseline=20, desc="unzscored"):
    """End-to-end: find runs -> load -> PSC + median-average -> (vertices x time)."""
    files = find_run_files(input_dir, sub, task, space, hemi, ses=ses, desc=desc)
    if not files:
        raise ValueError(
            f"No per-run '{desc}' files found under {opj(input_dir, f'sub-{sub}')} "
            f"for task={task}, space={space}, hemi={hemi}. Run 'funcproc denoise' first."
        )
    runs = load_runs_vertices_time(files)
    return raw_ts_to_average_psc(runs, baseline=baseline), files