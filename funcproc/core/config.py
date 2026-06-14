"""funcproc.core.config
====================
Project config handling, mirroring the bids7t/anatprep convention: a
``code/funcproc.yml`` file living next to ``rawdata/`` and ``derivatives/`` in
the study directory, auto-detected by walking up from the current working
directory (or derived from an explicit fMRIPrep path / ``--studydir``).

    <studydir>/
      code/funcproc.yml      <-- settings (tasks, space, paths, denoise opts)
      rawdata/
      derivatives/{fmriprep,pybest,prf}/
"""

import os

opj = os.path.join

FUNCPROC_CONFIG_NAME = "funcproc.yml"
MAX_SEARCH_DEPTH = 8


# -------- loading 
def load_config(path):
    """Read a YAML file into a dict (empty dict if missing/empty)."""
    import yaml
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---- studydir ----
def _is_studydir(d):
    return (
        os.path.exists(opj(d, "code", FUNCPROC_CONFIG_NAME))
        or os.path.isdir(opj(d, "rawdata"))
        or os.path.isdir(opj(d, "derivatives"))
    )


def find_studydir_from(start, max_depth=MAX_SEARCH_DEPTH):
    """Walk upward from ``start`` (a dir or file path) looking for a study
    directory (one containing code/funcproc.yml, rawdata/, or derivatives/)."""
    if not start:
        return None
    cur = os.path.abspath(start)
    if os.path.isfile(cur):
        cur = os.path.dirname(cur)
    for _ in range(max_depth):
        if _is_studydir(cur):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def find_studydir_from_cwd(max_depth=MAX_SEARCH_DEPTH):
    return find_studydir_from(os.getcwd(), max_depth)


def resolve_studydir(explicit=None, hint=None):
    """Resolve the study directory: explicit ``--studydir`` first, then derived
    from a hint path (e.g. the fMRIPrep dir), then auto-detected from CWD.
    Returns ``None`` if nothing is found (callers then rely on explicit paths)."""
    if explicit:
        p = os.path.abspath(explicit)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"--studydir does not exist: {p}")
        return p
    return find_studydir_from(hint) or find_studydir_from_cwd()


def load_funcproc_config(studydir):
    """Load ``<studydir>/code/funcproc.yml`` (empty dict if absent)."""
    if not studydir:
        return {}
    return load_config(opj(studydir, "code", FUNCPROC_CONFIG_NAME))


# ---- path resolution -----
def resolve_paths(cfg, studydir, input_dir=None, output_dir=None):
    """Resolve fMRIPrep (input) and pybest (output) dirs from, in order of
    precedence: explicit CLI value -> config -> ``<studydir>/derivatives/...``."""
    derivatives = cfg.get("derivatives") or (
        opj(studydir, "derivatives") if studydir else None
    )

    fmriprep = (
        input_dir
        or cfg.get("fmriprep_dir")
        or (opj(derivatives, "fmriprep") if derivatives else None)
    )
    pybest = (
        output_dir
        or cfg.get("pybest_dir")
        or (opj(derivatives, "pybest") if derivatives else None)
    )
    return fmriprep, pybest