"""M5.50, a descriptor outside the plan: the radius of the line read as the place where the two largest eigenvalues
meet, continuous in place of the reader's plaquette sets (K2's radius moves in jumps of the plaquette grid).

In the plane z = 0, about the box axis, along each of 72 azimuths, the six components are sampled by trilinear
interpolation at radial steps of h/20, the gap l3 - l2 is taken at each sample, and its least value is located by a
parabola through the three samples about the smallest. The radius is the mean over the azimuths; the spread is the
largest less the smallest of the 72.

Usage:  python m5_50_diag_radius.py m5_46_B0c_end.npz m5_50_B14a_end.npz ...      (files in ../data/)
"""

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
IDX = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def line_radius(f, n_phi=72):
    z = np.load(DATA / f, allow_pickle=False)
    m6, x = z["m6"], z["x"]
    h = x[1] - x[0]
    phis = np.arange(n_phi) * 2 * np.pi / n_phi
    rho = np.arange(0.5, x[-1] - 1.0, h / 20)
    P, R = np.meshgrid(phis, rho, indexing="ij")
    X, Y = R * np.cos(P), R * np.sin(P)
    idx = [(X - x[0]) / h, (Y - x[0]) / h, (np.zeros_like(X) - x[0]) / h]
    M = np.zeros(X.shape + (3, 3))
    for (a, b), k in zip(IDX, range(6)):
        M[..., a, b] = M[..., b, a] = map_coordinates(m6[..., k], idx, order=1)
    w = np.linalg.eigvalsh(M)
    g = w[..., 2] - w[..., 1]
    out = []
    for i in range(n_phi):
        k = min(max(int(g[i].argmin()), 1), len(rho) - 2)
        y0, y1, y2 = g[i, k - 1], g[i, k], g[i, k + 1]
        den = y0 - 2 * y1 + y2
        out.append(rho[k] + (0.5 * (y0 - y2) / den * (rho[1] - rho[0]) if den > 0 else 0.0))
    out = np.array(out)
    return float(out.mean()), float(out.max() - out.min())


if __name__ == "__main__":
    for f in sys.argv[1:]:
        r, s = line_radius(f)
        print(f"{f}: line radius {r:.4f}, spread over the azimuths {s:.4f}")
