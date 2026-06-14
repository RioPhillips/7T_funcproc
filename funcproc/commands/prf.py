"""funcproc prf
=================
Population receptive field (pRF) fitting with prfpy, on the denoise output.

Flow (mirrors the spinoza substance, built directly on prfpy a la Marcus/Dag):
    [1] load settings (code/fit_settings_prf.yml) + design matrix
    [2] assemble data: per-run unzscored -> Marco PSC -> median across runs
    [3] PRFStimulus2D -> Gaussian grid+iterative fit (optionally Norm/DN after)
    [4] save params/grid/settings pickle per hemisphere
    [5] render a visual-field coverage heatmap
"""

import glob
import json
import os

from funcproc.core import bids, config, prf_data, prf_fit, prf_plot
from funcproc.core.utils import verbose as vrb

opj = os.path.join


def add_parser(subparsers):
    p = subparsers.add_parser(
        "prf",
        help="population receptive field fitting (prfpy)",
        description="Fit pRF models on denoised surface time series.",
    )
    p.add_argument("input_dir", nargs="?", default=None,
                   help="pybest dir (default: from config / <studydir>/derivatives/pybest)")
    p.add_argument("output_dir", nargs="?", default=None,
                   help="output dir (default: <studydir>/derivatives/prf)")
    p.add_argument("-s", "--sub", required=True, help="subject ID (without 'sub-')")
    p.add_argument("-n", "--ses", default=None, help="session ID")
    p.add_argument("-t", "--task", default=None,
                   help="task (default: config 'tasks', e.g. 8bars)")
    p.add_argument("-m", "--model", default="gauss",
                   choices=["gauss", "norm"],
                   help="pRF model (default: gauss; 'norm' also runs gauss first)")
    p.add_argument("-r", "--space", default=None, help="space (default: fsnative)")
    p.add_argument("--lh", action="store_true", help="left hemisphere only")
    p.add_argument("--rh", action="store_true", help="right hemisphere only")
    p.add_argument("--dm", default=None,
                   help="design matrix .npy/.mat (default: prf.design_matrix in "
                        "funcproc.yml, else code/design_matrices/design_task-<task>.mat)")
    p.add_argument("-x", "--settings", default=None,
                   help="override settings yaml (default: the 'prf:' section of funcproc.yml)")
    p.add_argument("--tr", type=float, default=None,
                   help="repetition time (default: read from *_bold.json, else settings)")
    p.add_argument("--baseline", type=int, default=None,
                   help="number of initial no-stim volumes (default: inferred from DM)")
    p.add_argument("--grid", action="store_true", help="grid fit only (no iterative)")
    p.add_argument("--no-hrf", action="store_true", help="do not fit the HRF")
    p.add_argument("--rsq-thresh", type=float, default=0.1,
                   help="rsq threshold for the coverage heatmap (default: 0.1)")
    p.add_argument("-j", "--jobs", type=int, default=None, help="parallel jobs")
    p.add_argument("--no-fit", action="store_true",
                   help="assemble + save data only; skip fitting")
    p.add_argument("--no-plot", action="store_true", help="skip the coverage heatmap")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _hemis(args):
    if args.lh:
        return ["L"]
    if args.rh:
        return ["R"]
    return ["L", "R"]


def _resolve_task(args, cfg):
    if args.task:
        return args.task
    tasks = cfg.get("tasks") or []
    if tasks:
        return tasks[0]
    raise ValueError("No task given. Pass -t <task> or set 'tasks' in code/funcproc.yml.")


def _read_tr(studydir, sub, task, ses, fallback):
    """Best-effort: read RepetitionTime from a fMRIPrep *_bold.json; else fallback."""
    if not studydir:
        return fallback
    func = opj(studydir, "derivatives", "fmriprep", f"sub-{sub}", "func")
    if ses:
        func = opj(studydir, "derivatives", "fmriprep", f"sub-{sub}", f"ses-{ses}", "func")
    cands = sorted(glob.glob(opj(func, f"sub-{sub}*task-{task}*_bold.json")))
    for j in cands:
        try:
            with open(j) as f:
                tr = json.load(f).get("RepetitionTime")
            if tr:
                return float(tr)
        except Exception:
            pass
    return fallback


def _resolve_dm(args, cfg, studydir, task):
    """Resolve the design-matrix path, in order of preference:
      1. --dm PATH
      2. funcproc.yml  prf.design_matrix  (a path, or a {task: path} mapping)
      3. <studydir>/code/design_matrices/design_task-<task>.{mat,npy}
      4. <studydir>/code/design_task-<task>.{mat,npy}
      5. <studydir>/code/design_matrix.{npy,mat}
    Relative config paths are resolved against the study dir."""
    candidates = []
    if args.dm:
        candidates.append(args.dm)
    cfg_dm = (cfg.get("prf") or {}).get("design_matrix")
    if isinstance(cfg_dm, dict):
        if task in cfg_dm:
            candidates.append(cfg_dm[task])
    elif isinstance(cfg_dm, str):
        candidates.append(cfg_dm)
    if studydir:
        code = opj(studydir, "code")
        for ext in (".mat", ".npy"):
            candidates.append(opj(code, "design_matrices", f"design_task-{task}{ext}"))
            candidates.append(opj(code, f"design_task-{task}{ext}"))
        candidates += [opj(code, "design_matrix.npy"), opj(code, "design_matrix.mat")]

    resolved = []
    for c in candidates:
        if not c:
            continue
        resolved.append(c if os.path.isabs(c) else (opj(studydir, c) if studydir else c))
    for p in resolved:
        if os.path.exists(p):
            return p
    raise ValueError(
        "Design matrix not found. Looked for:\n  " + "\n  ".join(resolved)
        + "\nPass --dm /path/to/design_task-<task>.mat or set prf.design_matrix in funcproc.yml."
    )


def run(args):
    verbose = args.verbose

    studydir = config.resolve_studydir(explicit=None, hint=args.input_dir)
    cfg = config.load_funcproc_config(studydir)
    # input defaults to the pybest dir; output to derivatives/prf
    pybest_dir = (args.input_dir or cfg.get("pybest_dir")
                  or (opj(studydir, "derivatives", "pybest") if studydir else None))
    output_dir = (args.output_dir or cfg.get("prf_dir")
                  or (opj(studydir, "derivatives", "prf") if studydir else None))
    space = args.space or cfg.get("space") or "fsnative"
    task = _resolve_task(args, cfg)
    hemis = _hemis(args)

    if not pybest_dir:
        raise ValueError("Could not determine the pybest input dir. Pass it explicitly "
                         "or run from inside the study tree.")
    if not output_dir:
        raise ValueError("Could not determine the output dir. Pass it explicitly.")

    # settings: prefer an explicit -x yaml, else the 'prf' section of funcproc.yml
    settings = prf_fit.load_settings(args.settings or cfg.get("prf", {}))
    if args.jobs:
        settings["n_jobs"] = args.jobs

    dm_path = _resolve_dm(args, cfg, studydir, task)
    dm = prf_fit.load_design_matrix(dm_path)

    tr = args.tr or _read_tr(studydir, args.sub, task, args.ses, settings["TR"])
    baseline = args.baseline if args.baseline is not None else prf_data.infer_baseline_from_dm(dm)
    fit_hrf = not args.no_hrf

    if verbose:
        vrb(f"studydir : {studydir}", True)
        vrb(f"pybest   : {pybest_dir}", True)
        vrb(f"output   : {output_dir}", True)
        vrb(f"task={task} space={space} model={args.model} TR={tr} baseline={baseline}", True)
        vrb(f"design   : {dm_path}  shape={dm.shape}", True)

    stim = prf_fit.build_stimulus(dm, settings, tr)
    max_ecc = stim.screen_size_degrees / 2.0
    vrb(f"screen size: {stim.screen_size_degrees:.2f} dva (max ecc {max_ecc:.2f})", True)

    out_sub = opj(output_dir, bids.sub_ses_dir({"sub": args.sub,
                                                **({"ses": args.ses} if args.ses else {})}))
    sesstr = f"_ses-{args.ses}" if args.ses else ""

    for hemi in hemis:
        vrb(f"\n=== hemi-{hemi} ===", True)
        data, used = prf_data.assemble_psc(
            pybest_dir, args.sub, task, space, hemi, ses=args.ses, baseline=baseline)
        # data is (vertices x time); prfpy wants units x time -- already correct
        vrb(f"assembled {data.shape[0]} vertices x {data.shape[1]} timepoints "
            f"from {len(used)} runs", True)
        if data.shape[1] != dm.shape[-1]:
            vrb(f"  WARNING: timepoints ({data.shape[1]}) != design-matrix frames "
                f"({dm.shape[-1]}). Check --baseline / volume cutting.", True)

        base = f"sub-{args.sub}{sesstr}_task-{task}_space-{space}_hemi-{hemi}_model-{args.model}"
        if args.no_fit:
            import numpy as np
            np.save(opj(out_sub, base + "_desc-pscavg_bold.npy"), data)
            vrb(f"  saved averaged data (no fit): {base}_desc-pscavg_bold.npy", True)
            continue

        # --- Gaussian fit (always; Norm seeds from it) ---
        vrb("  Gaussian fit ...", True)
        gf, gpars, _ = prf_fit.fit_gauss(
            data, stim, settings, fit_hrf=fit_hrf,
            grid_only=args.grid, verbose=verbose)
        prf_fit.save_params(opj(out_sub, base.replace(f"model-{args.model}", "model-gauss")
                                + "_desc-prfparams.pkl"),
                            gpars, "gauss", settings)
        plot_pars, plot_model = gpars, "gauss"

        # --- Norm/DN fit (optional), seeded by the gaussian fitter ---
        if args.model == "norm":
            if args.grid:
                vrb("  (note: --grid given; running Norm grid only)", True)
            vrb("  Norm/DN fit ...", True)
            nf, npars, _ = prf_fit.fit_norm(
                data, stim, settings, gauss_fitter=gf, fit_hrf=fit_hrf,
                grid_only=args.grid, verbose=verbose)
            prf_fit.save_params(opj(out_sub, base + "_desc-prfparams.pkl"),
                                npars, "norm", settings)
            plot_pars, plot_model = npars, "norm"

        if not args.no_plot:
            png = opj(out_sub, base + "_desc-coverage.png")
            prf_plot.coverage_heatmap(
                plot_pars, max_ecc, png, rsq_thresh=args.rsq_thresh,
                title=f"sub-{args.sub} {task} {hemi} ({plot_model})")
            vrb(f"  coverage heatmap -> {png}", True)

    vrb("\nDone.", True)
    return 0