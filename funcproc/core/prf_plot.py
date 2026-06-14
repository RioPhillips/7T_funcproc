"""funcproc.core.prf_plot
=====================
Visual-field plots from fitted pRF parameters. The main product is a coverage
heatmap (max-of-Gaussians over well-fit vertices), the standard way to show
which part of the visual field a set of pRFs represents.
"""

import os
import numpy as np

opj = os.path.join

# param column indices (rsq is always last)
X, Y, SIZE = 0, 1, 2


def cart2pol(x, y):
    return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)


def coverage_map(params, max_ecc, rsq_thresh=0.1, res=150, cap=4000):
    """Max-of-Gaussians coverage over the visual field.

    Returns (grid, extent) where grid is (res x res) in [0,1] and extent is
    [-max_ecc, max_ecc, -max_ecc, max_ecc] for imshow.
    """
    p = np.asarray(params, dtype=float)
    rsq = p[:, -1]
    keep = np.isfinite(rsq) & (rsq > rsq_thresh) & np.isfinite(p[:, X]) & np.isfinite(p[:, Y])
    idx = np.where(keep)[0]
    if idx.size == 0:
        return np.zeros((res, res)), [-max_ecc, max_ecc, -max_ecc, max_ecc], 0
    if idx.size > cap:                      # keep the best-fitting `cap` vertices
        idx = idx[np.argsort(rsq[idx])[-cap:]]
    x, y, s = p[idx, X], p[idx, Y], np.clip(p[idx, SIZE], 1e-2, None)

    gx = np.linspace(-max_ecc, max_ecc, res)
    grid = np.zeros((res, res))
    # accumulate the max gaussian, looping over rows to bound memory
    for j, gy in enumerate(np.linspace(-max_ecc, max_ecc, res)):
        d2 = (gx[None, :] - x[:, None]) ** 2 + (gy - y[:, None]) ** 2
        g = np.exp(-d2 / (2 * s[:, None] ** 2))
        grid[j] = g.max(axis=0)
    return grid, [-max_ecc, max_ecc, -max_ecc, max_ecc], int(idx.size)


def coverage_heatmap(params, max_ecc, out_path, rsq_thresh=0.1,
                     title=None, res=150):
    """Render and save a coverage heatmap PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid, extent, n = coverage_map(params, max_ecc, rsq_thresh=rsq_thresh, res=res)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, extent=extent, origin="lower", cmap="hot",
                   vmin=0, vmax=1, aspect="equal")
    # eccentricity rings + polar spokes
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in np.linspace(max_ecc / 3, max_ecc, 3):
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="white",
                lw=0.6, alpha=0.4)
    for a in np.arange(0, np.pi, np.pi / 4):
        ax.plot([-max_ecc * np.cos(a), max_ecc * np.cos(a)],
                [-max_ecc * np.sin(a), max_ecc * np.sin(a)],
                color="white", lw=0.5, alpha=0.3)
    ax.set_xlim(-max_ecc, max_ecc); ax.set_ylim(-max_ecc, max_ecc)
    ax.set_xlabel("x (dva)"); ax.set_ylabel("y (dva)")
    ax.set_title(title or f"pRF coverage (rsq>{rsq_thresh}, n={n})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="coverage")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path