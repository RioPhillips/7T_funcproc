"""funcproc denoise
======================
Denoise fMRIPrep output with pybest and produce both z-scored and
un-z-scored (PSC-ready) time series.

Pipeline:
    [pre]   store per-vertex/voxel mean+std of the raw fMRIPrep BOLD  -> tmp/
    [run]   run pybest (per task; per hemisphere for surface)         -> denoising/
    [post]  invert pybest's z-scoring (zscored*std + mean)            -> unzscored/

Paths and settings resolve in this precedence: explicit CLI value ->
code/funcproc.yml (auto-detected study config) -> built-in defaults. Session
handling is dynamic: paths collapse to ``sub-XX/...`` without a session and
expand to ``sub-XX/ses-YY/...`` with one.
"""

import json
import os

from funcproc.core import bids, config, io, pybest, unzscore
from funcproc.core.utils import verbose as vrb

opj = os.path.join


# --- parser ------
def add_parser(subparsers):
    p = subparsers.add_parser(
        "denoise",
        help="denoise fMRIPrep output (pybest) -> z-scored + un-z-scored data",
        description="Denoise fMRIPrep output with pybest.",
    )
    p.add_argument("input_dir", nargs="?", default=None,
                   help="fMRIPrep dir (default: from config / <studydir>/derivatives/fmriprep)")
    p.add_argument("output_dir", nargs="?", default=None,
                   help="pybest output dir (default: from config / <studydir>/derivatives/pybest)")
    p.add_argument("-s", "--sub", required=True, help="subject ID (without 'sub-')")
    p.add_argument("-n", "--ses", default=None,
                   help="session ID; omit for datasets without sessions")
    p.add_argument("-t", "--task", default=None,
                   help="task to process; default: config 'tasks', else all found")
    p.add_argument("--studydir", default=None,
                   help="study dir (default: auto-detect code/funcproc.yml from cwd / input)")
    p.add_argument("-r", "--space", default=None,
                   help="space (default: config 'space', else fsnative)")
    p.add_argument("-p", "--n-comps", type=int, default=None,
                   help="pybest PCA components (default: config, else 20)")
    p.add_argument("-c", "--n-cpus", type=int, default=None,
                   help="CPUs/slots for pybest")
    p.add_argument("--lh", action="store_true", help="left hemisphere only")
    p.add_argument("--rh", action="store_true", help="right hemisphere only")
    p.add_argument("--pre-only", action="store_true", help="only compute mean/std")
    p.add_argument("--pyb-only", action="store_true", help="only run pybest")
    p.add_argument("--post-only", action="store_true", help="only un-z-score")
    p.add_argument("--no-unzscore", action="store_true",
                   help="skip the un-z-score step (keep z-scored output only)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the pybest command(s) without running")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


# -------- helpers 
def _hemis(args):
    if args.lh:
        return ["L"]
    if args.rh:
        return ["R"]
    return ["L", "R"]


def _raw_filters(sub, space, ses, task, surface):
    inc = [f"sub-{sub}", "bold"]
    if surface:
        inc += [f"space-{space}"]
    else:
        inc += ["desc-preproc_bold"]
        if space != "func":
            inc += [f"space-{space}"]
    if ses is not None:
        inc += [f"ses-{ses}_"]
    if task is not None:
        inc += [f"task-{task}_"]
    return inc


def _stats_match_filters(entities,
                         keys=("ses", "task", "acq", "dir", "run", "hemi", "space")):
    return [f"{k}-{entities[k]}_" for k in keys if k in entities] + ["_desc-avgstd"]


def _diag_no_files(what, root, filters, ext):
    exists = os.path.isdir(root)
    lines = [
        f"No {what} found.",
        f"  searched : {root}" + ("" if exists else "   (DIRECTORY DOES NOT EXIST)"),
        f"  matching : files ending '{ext}' whose path contains all of {filters}",
        "",
        "Check that the path is correct and that fMRIPrep produced this space",
        "(e.g. for surface you need fsnative giftis: run fMRIPrep with",
        "--output-spaces fsnative). You can also pass the task explicitly with",
        "'-t <task>' or list tasks in code/funcproc.yml.",
    ]
    return "\n".join(lines)


# - stages --------
def _run_pre(input_dir, output_dir, sub, ses, task, space, hemis, surface, verbose):
    root = opj(input_dir, f"sub-{sub}")
    ext = "func.gii" if surface else "nii.gz"
    n_done = 0
    hemi_iter = hemis if surface else [None]
    for hemi in hemi_iter:
        inc = _raw_filters(sub, space, ses, task, surface)
        if hemi is not None:
            inc += [f"hemi-{hemi}_"]
        files = bids.find_files(root, include=inc, exclude=["json"], extension=ext)
        if not files:
            raise ValueError(_diag_no_files("raw fMRIPrep files", root, inc, ext))
        for func in files:
            vrb(f" {func}", verbose)
            ent = bids.parse_entities(func)
            tmp_dir = opj(output_dir, bids.sub_ses_dir(ent), "tmp")
            stats_path = opj(tmp_dir, bids.derivatives_base(ent) + "_desc-avgstd.pkl")
            if surface:
                stats = unzscore.compute_stats_surface(io.read_surface_array(func))
            else:
                stats = unzscore.compute_stats_nifti(io.read_nifti(func))
            unzscore.save_stats(stats_path, stats)
            n_done += 1
    vrb(f"[pre] wrote mean/std for {n_done} file(s)", True)
    return n_done


def _discover_tasks(input_dir, sub, space, ses, surface):
    root = opj(input_dir, f"sub-{sub}")
    ext = "func.gii" if surface else "nii.gz"
    inc = _raw_filters(sub, space, ses, None, surface)
    files = bids.find_files(root, include=inc, exclude=["json"], extension=ext)
    return bids.unique_values(files, "task"), root, inc, ext


def _resolve_tasks(cfg, args, input_dir, sub, space, ses, surface):
    if args.task:
        return [args.task]
    if cfg.get("tasks"):
        return list(cfg["tasks"])
    tasks, root, inc, ext = _discover_tasks(input_dir, sub, space, ses, surface)
    if not tasks:
        raise ValueError(_diag_no_files("fMRIPrep files (for task discovery)",
                                        root, inc, ext))
    return tasks


def _run_post(output_dir, sub, ses, task, space, hemis, surface, verbose):
    ext = "npy" if surface else "nii.gz"
    inc = [f"sub-{sub}", "desc-denoised_bold"]
    if surface:
        inc += [f"space-{space}"]
    if ses is not None:
        inc += [f"ses-{ses}_"]
    if task is not None:
        inc += [f"task-{task}_"]
    hemi_iter = hemis if surface else [None]
    n_done = 0
    for hemi in hemi_iter:
        this_inc = inc + ([f"hemi-{hemi}_"] if hemi is not None else [])
        denoised = bids.find_files(
            output_dir, include=this_inc, exclude=["unzscored", "raw"], extension=ext
        )
        for func in denoised:
            ent = bids.parse_entities(func)
            tmp_dir = opj(output_dir, bids.sub_ses_dir(ent), "tmp")
            matches = bids.find_files(tmp_dir, include=_stats_match_filters(ent),
                                      extension="pkl")
            if len(matches) == 0:
                raise ValueError(
                    f"No mean/std file for '{os.path.basename(func)}' in '{tmp_dir}'. "
                    "Did the 'pre' stage run for this subject?"
                )
            if len(matches) > 1:
                # pybest also writes a run-less aggregate (its across-runs
                # concatenation). That has no 'run-' entity, so it matches every
                # per-run mean/std file. We skip it -- it can't map to a single
                # mean/std and isn't used downstream (call_prf works from the
                # per-run files). This mirrors the colleague's call_unzscore,
                # which only un-z-scores per-run files.
                if "run" not in ent:
                    vrb(f" skipping run-less aggregate: {os.path.basename(func)}", verbose)
                    continue
                raise ValueError(
                    f"Ambiguous mean/std match for '{os.path.basename(func)}' in "
                    f"'{tmp_dir}': found {len(matches)} candidates: {matches}"
                )
            stats = unzscore.load_stats(matches[0])

            out_dir = opj(output_dir, bids.sub_ses_dir(ent), "unzscored")
            out_name = os.path.basename(func).replace("desc-denoised", "desc-unzscored")
            out_path = opj(out_dir, out_name)

            if surface:
                unz = unzscore.apply_unzscore_surface(io.read_surface_array(func), stats)
                io.save_surface_array(out_path, unz)
            else:
                unz = unzscore.apply_unzscore_nifti(io.read_nifti(func), stats)
                os.makedirs(out_dir, exist_ok=True)
                unz.to_filename(out_path)

            _update_sidecar_json(func, out_path, ext)
            vrb(f" wrote {out_path}", verbose)
            n_done += 1
    vrb(f"[post] un-z-scored {n_done} file(s)", True)
    return n_done


def _update_sidecar_json(src_func, out_path, ext):
    src_json = src_func[: -len(ext)] + "json" if src_func.endswith(ext) else None
    if not src_json or not os.path.exists(src_json):
        return
    with open(src_json) as f:
        meta = json.load(f)
    meta["Unzscored"] = True
    meta["SkullStripped"] = False
    out_json = out_path[: -len(ext)] + "json"
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=4)


# --- run 
def run(args):
    verbose = args.verbose

    # resolve study config + paths + settings (CLI > config > defaults)
    studydir = config.resolve_studydir(explicit=args.studydir, hint=args.input_dir)
    cfg = config.load_funcproc_config(studydir)
    input_dir, output_dir = config.resolve_paths(
        cfg, studydir, args.input_dir, args.output_dir
    )
    space = args.space or cfg.get("space") or "fsnative"
    n_comps = args.n_comps or (cfg.get("denoise") or {}).get("n_comps") or 20
    surface = pybest.is_surface_space(space)
    hemis = _hemis(args)

    if verbose:
        vrb(f"studydir : {studydir or '(none; using explicit paths)'}", True)
        vrb(f"fmriprep : {input_dir}", True)
        vrb(f"pybest   : {output_dir}", True)
        vrb(f"space    : {space} | n_comps: {n_comps}", True)

    # stage selection (default: all three)
    only = args.pre_only or args.pyb_only or args.post_only
    do_pre = (not only) or args.pre_only
    do_pyb = (not only) or args.pyb_only
    do_post = ((not only) or args.post_only) and not args.no_unzscore

    if (do_pre or do_pyb) and not input_dir:
        raise ValueError(
            "Could not determine the fMRIPrep directory. Pass it as the first "
            "argument, set 'fmriprep_dir' in code/funcproc.yml, or run from "
            "within the study tree."
        )
    if not output_dir:
        raise ValueError(
            "Could not determine the pybest output directory. Pass it as the "
            "second argument or set 'pybest_dir'/'derivatives' in code/funcproc.yml."
        )

    if do_pre:
        vrb("Computing mean/std (pre-pybest)", True)
        _run_pre(input_dir, output_dir, args.sub, args.ses, args.task,
                 space, hemis, surface, verbose)

    if do_pyb:
        tasks = _resolve_tasks(cfg, args, input_dir, args.sub, space, args.ses, surface)
        vrb(f"Running pybest on task(s): {tasks}", True)
        os.makedirs(output_dir, exist_ok=True)  # mirror call_pybest's mkdir -p
        pybest.run_pybest(
            input_dir, output_dir, args.sub, tasks, space=space,
            ses=args.ses, n_comps=n_comps, n_cpus=args.n_cpus,
            dry_run=args.dry_run, verbose=True,
        )

    if do_post and not args.dry_run:
        vrb("Un-z-scoring pybest output (post-pybest)", True)
        _run_post(output_dir, args.sub, args.ses, args.task, space,
                  hemis, surface, verbose)

    vrb("Done.", True)
    return 0