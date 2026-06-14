"""funcproc.core.prf_fit
====================
prfpy fitting: stimulus construction, Gaussian grid+iterative fit, and the
Norm/DN extension (seeded by the Gaussian fit). Settings come from funcproc.yml.
Heavy prfpy imports are done inside
the functions so importing this module stays cheap.

Parameter columns (prfpy order; rsq is always the LAST column):
  gauss : x, y, size, amp, baseline, [hrf_1, hrf_2], rsq
  norm  : x, y, size, amp, baseline, srf_amp, srf_size,
          neural_baseline, surround_baseline, [hrf_1, hrf_2], rsq
"""

import os
import pickle
from datetime import datetime

import numpy as np

opj = os.path.join

DEFAULTS = {
    "hrf": {"pars": [1, 1, 0], "deriv_bound": [0, 10], "disp_bound": [0, 0]},
    "xtol": 1e-4, "ftol": 1e-4, "grid_nr": 20, "rsq_threshold": 0.1,
    "TR": 1.5, "filter_predictions": False, "n_jobs": 4, "n_batches": 10,
    "screen_size_cm": 39.3, "screen_distance_cm": 196,
    "normalize_RFs": False, "eps": 1e-1, "fixed_grid_baseline": 0,
    "bold_bsl": [0, 0], "prf_ampl": [0, 1000], "constraints": False,
    "save_grids": True,
}


def load_settings(source=None):
    """Build a settings dict from built-in defaults plus, in order of preference,
    a ``source`` that is either a dict (the ``prf:`` section of funcproc.yml) or
    a path to a standalone yaml. Missing keys fall back to DEFAULTS."""
    s = dict(DEFAULTS)
    data = {}
    if isinstance(source, dict):
        data = dict(source)
    elif isinstance(source, str) and os.path.exists(source):
        import yaml
        with open(source) as f:
            data = yaml.safe_load(f) or {}
    s.update(data)
    hrf = dict(DEFAULTS["hrf"]); hrf.update(s.get("hrf", {})); s["hrf"] = hrf
    return s


def load_design_matrix(path):
    """Load a (n_pix x n_pix x time) design matrix from .npy or .mat, oriented
    so the last axis is time."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        dm = np.load(path)
    elif ext == ".mat":
        from scipy.io import loadmat
        mat = loadmat(path)
        cand = {k: v for k, v in mat.items()
                if not k.startswith("__") and isinstance(v, np.ndarray)}
        # prefer common key names, else the first 3D array
        dm = None
        for key in ("design", "design_matrix", "dm", "DM", "stim", "stimulus", "M"):
            if key in cand and cand[key].ndim == 3:
                dm = cand[key]; break
        if dm is None:
            threed = [v for v in cand.values() if v.ndim == 3]
            if not threed:
                raise ValueError(f"No 3D design array found in {path}; keys: {list(cand)}")
            dm = threed[0]
    else:
        raise ValueError(f"Unsupported design matrix format '{ext}' ({path}); use .npy or .mat")

    dm = np.asarray(dm)
    if dm.ndim != 3:
        raise ValueError(f"design matrix should be 3D (n_pix,n_pix,time), got {dm.shape}")
    # orient so time is the last axis (the two equal-length axes are spatial)
    if dm.shape[0] == dm.shape[1]:
        pass                                   # (n_pix, n_pix, time)
    elif dm.shape[1] == dm.shape[2]:
        dm = np.moveaxis(dm, 0, -1)            # (time, n_pix, n_pix) -> (n_pix, n_pix, time)
    return dm


def build_stimulus(dm, settings, tr):
    from prfpy.stimulus import PRFStimulus2D
    return PRFStimulus2D(
        screen_size_cm=settings["screen_size_cm"],
        screen_distance_cm=settings["screen_distance_cm"],
        design_matrix=dm,
        TR=tr,
    )


def _filter_nans(params):
    """Replace rows containing NaN/inf with zeros (prfpy occasionally returns NaNs)."""
    p = np.array(params, dtype=float)
    bad = ~np.isfinite(p).all(axis=1)
    p[bad] = 0.0
    return p


def _grids(stim, settings):
    max_ecc = stim.screen_size_degrees / 2.0
    g = int(settings["grid_nr"])
    eccs = max_ecc * np.linspace(0.25, 1, g) ** 2
    sizes = max_ecc * np.linspace(0.1, 1, g) ** 2
    polars = np.linspace(0, 2 * np.pi, g)
    return eccs, sizes, polars, max_ecc


def _hrf_grids(settings, fit_hrf):
    if not fit_hrf:
        return None, None
    d = settings["hrf"]["deriv_bound"]
    return np.linspace(d[0], d[1], 5), np.array([0.0])


def _finalize_bounds(core, grid_params, fit_hrf, settings):
    """Pad ``core`` bounds out to the model's full parameter count. The trailing
    parameters are the two HRF terms: bounded by deriv/disp ranges when fitting
    the HRF, otherwise fixed at the (constant) values the grid produced so they
    don't move. Any other trailing params are likewise fixed at grid medians."""
    nparams = grid_params.shape[1] - 1  # last column is rsq
    bounds = list(core)
    remaining = nparams - len(bounds)
    if remaining == 2 and fit_hrf:
        bounds += [tuple(settings["hrf"]["deriv_bound"]),
                   tuple(settings["hrf"]["disp_bound"])]
    else:
        for c in range(len(bounds), nparams):
            v = float(np.median(grid_params[:, c]))
            bounds.append((v, v))
    assert len(bounds) == nparams, (len(bounds), nparams)
    return bounds


#  gauss 
def fit_gauss(data, stim, settings, fit_hrf=True, grid_only=False, verbose=False):
    from prfpy.model import Iso2DGaussianModel
    from prfpy.fit import Iso2DGaussianFitter

    gg = Iso2DGaussianModel(
        stimulus=stim, hrf=settings["hrf"]["pars"],
        filter_predictions=settings["filter_predictions"],
        normalize_RFs=settings["normalize_RFs"],
    )
    gf = Iso2DGaussianFitter(data=data, model=gg, n_jobs=int(settings["n_jobs"]))

    eccs, sizes, polars, max_ecc = _grids(stim, settings)
    h1, h2 = _hrf_grids(settings, fit_hrf)
    gf.grid_fit(
        ecc_grid=eccs, polar_grid=polars, size_grid=sizes,
        hrf_1_grid=h1, hrf_2_grid=h2, verbose=verbose,
        n_batches=int(settings["n_batches"]),
        fixed_grid_baseline=settings["fixed_grid_baseline"],
        grid_bounds=[tuple(settings["prf_ampl"])],
    )
    gf.gridsearch_params = _filter_nans(gf.gridsearch_params)
    if grid_only:
        return gf, gf.gridsearch_params, gg

    eps = float(settings["eps"])
    core = [
        (-1.5 * max_ecc, 1.5 * max_ecc),                 # x
        (-1.5 * max_ecc, 1.5 * max_ecc),                 # y
        (eps, max_ecc * 3),                              # size
        tuple(settings["prf_ampl"]),                     # amplitude
        tuple(settings["bold_bsl"]),                     # baseline (fixed)
    ]
    bounds = _finalize_bounds(core, gf.gridsearch_params, fit_hrf, settings)
    constraints = [] if settings["constraints"] else None
    gf.iterative_fit(
        rsq_threshold=settings["rsq_threshold"], verbose=verbose,
        bounds=bounds, constraints=constraints,
        tol=float(settings["ftol"]),  # prfpy forwards this to scipy.minimize
    )
    gf.iterative_search_params = _filter_nans(gf.iterative_search_params)
    return gf, gf.iterative_search_params, gg


#  norm 
def fit_norm(data, stim, settings, gauss_fitter, fit_hrf=True,
             grid_only=False, verbose=False):
    from prfpy.model import Norm_Iso2DGaussianModel
    from prfpy.fit import Norm_Iso2DGaussianFitter

    nm = Norm_Iso2DGaussianModel(
        stimulus=stim, hrf=settings["hrf"]["pars"],
        filter_predictions=settings["filter_predictions"],
        normalize_RFs=settings["normalize_RFs"],
    )
    nf = Norm_Iso2DGaussianFitter(
        data=data, model=nm, n_jobs=int(settings["n_jobs"]),
        previous_gaussian_fitter=gauss_fitter,
    )

    _, _, _, max_ecc = _grids(stim, settings)
    # Norm-specific grids (sensible spinoza-like defaults; tune via settings)
    ng = settings.get("norm_grid", {})
    srf_amp = np.array(ng.get("surround_amplitude", [0.0, 0.5, 1.0, 2.0]))
    srf_size = np.array(ng.get("surround_size",
                               list(max_ecc * np.linspace(0.5, 2.0, 4))))
    neural_bsl = np.array(ng.get("neural_baseline", [0.0, 1.0, 10.0, 100.0]))
    surround_bsl = np.array(ng.get("surround_baseline", [1.0, 10.0, 100.0, 1000.0]))
    h1, h2 = _hrf_grids(settings, fit_hrf)

    nf.grid_fit(
        srf_amp, srf_size, neural_bsl, surround_bsl,
        verbose=verbose, n_batches=int(settings["n_batches"]),
        rsq_threshold=settings["rsq_threshold"],
        fixed_grid_baseline=settings["fixed_grid_baseline"],
        grid_bounds=[tuple(settings["prf_ampl"])],
        hrf_1_grid=h1, hrf_2_grid=h2,
    )
    nf.gridsearch_params = _filter_nans(nf.gridsearch_params)
    if grid_only:
        return nf, nf.gridsearch_params, nm

    eps = float(settings["eps"])
    core = [
        (-1.5 * max_ecc, 1.5 * max_ecc),                 # x
        (-1.5 * max_ecc, 1.5 * max_ecc),                 # y
        (eps, max_ecc * 3),                              # prf size
        tuple(settings["prf_ampl"]),                     # prf amplitude (A)
        tuple(settings["bold_bsl"]),                     # bold baseline
        (0, 1000),                                       # surround amplitude (C)
        (eps, max_ecc * 6),                              # surround size
        (0, 1000),                                       # neural baseline (B)
        (1e-6, 1000),                                    # surround baseline (D)
    ]
    bounds = _finalize_bounds(core, nf.gridsearch_params, fit_hrf, settings)
    constraints = [] if settings["constraints"] else None
    nf.iterative_fit(
        rsq_threshold=settings["rsq_threshold"], verbose=verbose,
        bounds=bounds, constraints=constraints,
        tol=float(settings["ftol"]),  # prfpy forwards this to scipy.minimize
    )
    nf.iterative_search_params = _filter_nans(nf.iterative_search_params)
    return nf, nf.iterative_search_params, nm


#  save 
def save_params(path, params, model, settings, grid_params=None, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {
        "pars": params, "model": model, "settings": settings,
        "date": datetime.now().strftime("%Y-%m-%d_%H-%M"),
    }
    if grid_params is not None:
        out["grid_pars"] = grid_params
    if extra:
        out.update(extra)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return path