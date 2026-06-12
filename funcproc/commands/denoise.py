"""funcproc denoise
======================
Denoise fMRIPrep output with pybest and produce both z-scored and
un-z-scored (PSC-ready) time series.

Pipeline:
    [pre]   store per-vertex/voxel mean+std of the raw fMRIPrep BOLD  -> tmp/
    [run]   run pybest (per task; per hemisphere for surface)         -> denoising/
    [post]  invert pybest's z-scoring (zscored*std + mean)            -> unzscored/

Session handling: paths collapse to ``sub-XX/...`` when no
session is present and expand to ``sub-XX/ses-YY/...`` when it is.
"""

import json
import os

from funcproc.core import bids, io, pybest, unzscore
from funcproc.core.utils import verbose as vrb

opj = os.path.join


#  parser 
def add_parser(subparsers):
    p = subparsers.add_parser(
        "denoise",
        help="denoise fMRIPrep output (pybest) -> z-scored + un-z-scored data",
        description="Denoise fMRIPrep output with pybest.",
    )
    p.add_argument("input_dir", nargs="?", default=None,
                   help="fMRIPrep derivatives dir (contains sub-XX/...)")
    p.add_argument("output_dir", nargs="?", default=None,
                   help="output (pybest) dir; default: sibling 'pybest' of input")
    p.add_argument("-s", "--sub", required=True, help="subject ID (without 'sub-')")
    p.add_argument("-n", "--ses", default=None,
                   help="session ID; omit for datasets without sessions")
    p.add_argument("-t", "--task", default=None,
                   help="limit to a single task; default: all tasks found")
    p.add_argument("-r", "--space", default="fsnative",
                   help="space to process (default: fsnative)")
    p.add_argument("-p", "--n-comps", type=int, default=20,
                   help="pybest PCA components (default: 20)")
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


#  helpers
def _hemis(args):
    if args.lh:
        return ["L"]
    if args.rh:
        return ["R"]
    return ["L", "R"]


def _resolve_output_dir(input_dir, output_dir):
    if output_dir:
        return output_dir
    if input_dir and input_dir.rstrip("/").endswith("fmriprep"):
        return opj(os.path.dirname(input_dir.rstrip("/")), "pybest")
    raise ValueError(
        "Could not infer output_dir; pass it explicitly (or point input_dir "
        "at a '.../fmriprep' folder so a sibling '.../pybest' can be used)."
    )


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
    """Filters to locate the mean/std file for a given denoised file, tolerant
    to entities pybest may add/drop, as long as the shared ones disambiguate."""
    return [f"{k}-{entities[k]}_" for k in keys if k in entities] + ["_desc-avgstd"]


#  stages 
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


def _discover_tasks(input_dir, sub, space, ses, task, surface):
    if task is not None:
        return [task]
    root = opj(input_dir, f"sub-{sub}")
    ext = "func.gii" if surface else "nii.gz"
    inc = _raw_filters(sub, space, ses, None, surface)
    files = bids.find_files(root, include=inc, exclude=["json"], extension=ext)
    return bids.unique_values(files, "task")


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
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one mean/std file for '{os.path.basename(func)}' "
                    f"in '{tmp_dir}', found {len(matches)}: {matches}"
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


#  run 
def run(args):
    verbose = args.verbose
    input_dir = args.input_dir
    output_dir = _resolve_output_dir(input_dir, args.output_dir)
    surface = pybest.is_surface_space(args.space)
    hemis = _hemis(args)

    # stage selection (default: all three)
    only = args.pre_only or args.pyb_only or args.post_only
    do_pre = (not only) or args.pre_only
    do_pyb = (not only) or args.pyb_only
    do_post = ((not only) or args.post_only) and not args.no_unzscore

    if (do_pre or do_pyb) and not input_dir:
        raise ValueError("input_dir (fMRIPrep derivatives) is required for pre/pybest stages.")

    if do_pre:
        vrb("Computing mean/std (pre-pybest)", True)
        _run_pre(input_dir, output_dir, args.sub, args.ses, args.task,
                 args.space, hemis, surface, verbose)

    if do_pyb:
        tasks = _discover_tasks(input_dir, args.sub, args.space, args.ses,
                                args.task, surface)
        if not tasks:
            raise ValueError("No tasks found to process.")
        vrb(f"Running pybest on task(s): {tasks}", True)
        pybest.run_pybest(
            input_dir, output_dir, args.sub, tasks, space=args.space,
            ses=args.ses, n_comps=args.n_comps, n_cpus=args.n_cpus,
            dry_run=args.dry_run, verbose=True,
        )

    if do_post and not args.dry_run:
        vrb("Un-z-scoring pybest output (post-pybest)", True)
        _run_post(output_dir, args.sub, args.ses, args.task, args.space,
                  hemis, surface, verbose)

    vrb("Done.", True)
    return 0