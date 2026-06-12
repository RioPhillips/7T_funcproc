"""funcproc.core.io
====================
Reading and writing functional data. Surface time series are handled as
``(timepoints, vertices)`` arrays (matching pybest/.npy and downstream
expectations); volumetric data is handled as nibabel images.
"""

import os
import numpy as np


def is_surface(path):
    return path.endswith((".func.gii", ".gii", ".npy"))


def is_nifti(path):
    return path.endswith((".nii.gz", ".nii"))


def read_gii_timeseries(path):
    """Read a functional GIFTI as a ``(timepoints, vertices)`` array.

    Each darray in a func.gii is one timepoint (length = n_vertices); stacking
    along a new leading axis yields ``(T, V)``.
    """
    import nibabel as nb
    img = nb.load(path)
    return np.stack([d.data for d in img.darrays], axis=0)


def read_surface_array(path):
    """Read surface data as ``(timepoints, vertices)``.

    ``.func.gii`` -> stacked darrays; ``.npy`` -> loaded as-is (already in
    pybest's on-disk orientation).
    """
    if path.endswith((".func.gii", ".gii")):
        return read_gii_timeseries(path)
    elif path.endswith(".npy"):
        return np.load(path)
    raise ValueError(f"Not a recognised surface file: {path}")


def save_surface_array(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def read_nifti(path):
    import nibabel as nb
    return nb.load(path)