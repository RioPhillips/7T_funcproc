"""funcproc.core.bids
======================
Lightweight, dependency-free BIDS helpers: parse entities from filenames,
build filename bases and derivative sub-directories, and find files by
entity filters. Session is treated as fully optional throughout, so paths
collapse to ``sub-XX/...`` when no session is present and expand to
``sub-XX/ses-YY/...`` when it is.
"""

import os
import re

opj = os.path.join

# Canonical BIDS-ish ordering for building filename bases. Only keys that are
# actually present are used, so this works for datasets with or without
# sessions, directions, acquisitions, etc.
ENTITY_ORDER = [
    "sub", "ses", "task", "acq", "ce", "rec", "dir", "run",
    "echo", "part", "hemi", "space", "desc",
]

_ENTITY_RE = re.compile(r"^([a-zA-Z0-9]+)-(.+)$")


def parse_entities(path):
    """Parse BIDS key-value entities from a file path.

    Returns a dict like ``{"sub": "001", "task": "pRF", "run": "1", ...}``.
    The trailing suffix (e.g. ``bold``) and extension are ignored. Works on a
    full path or a bare filename.
    """
    name = os.path.basename(path)
    # strip known compound/simple extensions
    for ext in (".nii.gz", ".func.gii", ".surf.gii", ".gii", ".nii",
                ".npy", ".pkl", ".json", ".mat"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break

    entities = {}
    for chunk in name.split("_"):
        m = _ENTITY_RE.match(chunk)
        if m:
            entities[m.group(1)] = m.group(2)
        # else: it's the suffix (e.g. "bold") -> ignore
    return entities


def build_base(entities, drop=("desc",), extra_drop=()):
    """Build a filename base (no suffix/extension) from an entities dict.

    Keys in ``drop``/``extra_drop`` are excluded. Ordering follows
    ``ENTITY_ORDER``; unknown keys are appended in their natural order.
    """
    skip = set(drop) | set(extra_drop)
    keys = [k for k in ENTITY_ORDER if k in entities and k not in skip]
    keys += [k for k in entities if k not in ENTITY_ORDER and k not in skip]
    return "_".join(f"{k}-{entities[k]}" for k in keys)


def sub_ses_dir(entities):
    """Return ``sub-XX`` or ``sub-XX/ses-YY`` depending on session presence."""
    base = f"sub-{entities['sub']}"
    if entities.get("ses") is not None:
        base = opj(base, f"ses-{entities['ses']}")
    return base


def derivatives_base(entities):
    """Filename base for derivative outputs (subject[, session], task, run,
    plus acq/dir/space/hemi if present), i.e. everything except ``desc``."""
    return build_base(entities, drop=("desc",))


def entity_signature(entities, keys=("sub", "ses", "task", "acq", "dir",
                                     "run", "hemi", "space")):
    """A hashable signature of the matching entities, used to pair a denoised
    file with its corresponding mean/std file regardless of ``desc`` or
    suffix differences. Missing keys are simply omitted."""
    return tuple((k, entities[k]) for k in keys if k in entities)


def find_files(root, include=(), exclude=(), extension=None):
    """Recursively find files under ``root`` whose path contains every string
    in ``include`` and none in ``exclude``. Optionally filter by extension
    (e.g. ``"func.gii"``, ``"nii.gz"``, ``"npy"``)."""
    include = [include] if isinstance(include, str) else list(include)
    exclude = [exclude] if isinstance(exclude, str) else list(exclude)
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = opj(dirpath, fn)
            if extension is not None and not fn.endswith(extension):
                continue
            if any(inc not in full for inc in include):
                continue
            if any(exc in full for exc in exclude):
                continue
            matches.append(full)
    return sorted(matches)


def unique_values(paths, key):
    """Unique values of a BIDS entity across a list of paths (sorted)."""
    vals = []
    for p in paths:
        ent = parse_entities(p)
        if key in ent and ent[key] not in vals:
            vals.append(ent[key])
    return sorted(vals)