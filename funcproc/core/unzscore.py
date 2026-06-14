"""funcproc.core.unzscore
==========================
Undo pybest's internal z-scoring, reproducing the colleague's ``call_unzscore``
logic natively. The idea: before pybest, store the per-vertex (or per-voxel)
mean and std of the raw fMRIPrep BOLD; after pybest, invert with
``raw = zscored * std + mean`` so the data can be percent-signal-changed later.

Surface data is treated element-wise and is orientation-robust: the vertex
axis is identified by matching its length to the stored statistics, so it
works whether the denoised array is ``(time, vertices)`` or ``(vertices, time)``
and always returns the same shape it was given.
"""

import os
import pickle
import numpy as np


# ---- surface ----
def compute_stats_surface(arr):
    """Per-vertex mean and std over time. ``arr`` is ``(time, vertices)``."""
    return {
        "avg": np.asarray(arr).mean(axis=0),
        "std": np.asarray(arr).std(axis=0),
    }


def _vertex_axis(shape, n_verts):
    axes = [i for i, s in enumerate(shape) if s == n_verts]
    if not axes:
        raise ValueError(
            f"No axis of length {n_verts} (vertices) in array of shape {shape}"
        )
    return axes[-1]  # prefer last axis; downstream treats shape[-1] as vertices


def apply_unzscore_surface(zscored, stats):
    """``zscored * std + avg`` broadcast along the vertex axis. Returns an
    array of the same shape as ``zscored``."""
    zscored = np.asarray(zscored)
    avg, std = np.asarray(stats["avg"]), np.asarray(stats["std"])
    vax = _vertex_axis(zscored.shape, avg.shape[0])
    bshape = [1] * zscored.ndim
    bshape[vax] = avg.shape[0]
    return zscored * std.reshape(bshape) + avg.reshape(bshape)


# ----- volume ----
def compute_stats_nifti(img):
    """Per-voxel mean and std over time for a 4D nibabel image."""
    data = img.get_fdata()
    return {
        "avg": data.mean(axis=-1),
        "std": data.std(axis=-1),
        "tr": float(img.header["pixdim"][4]),
        "affine": img.affine,
    }


def apply_unzscore_nifti(img, stats):
    import nibabel as nb
    data = img.get_fdata()
    unz = data * stats["std"][..., np.newaxis] + stats["avg"][..., np.newaxis]
    hdr = img.header.copy()
    if stats.get("tr"):
        hdr["pixdim"][4] = float(stats["tr"])
    return nb.Nifti1Image(unz, affine=img.affine, header=hdr)


# -------- io -----
def save_stats(path, stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(stats, f)


def load_stats(path):
    with open(path, "rb") as f:
        return pickle.load(f)