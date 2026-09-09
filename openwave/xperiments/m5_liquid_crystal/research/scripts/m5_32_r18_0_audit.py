"""M5.32 R18-0 INDEPENDENT ADVERSARIAL AUDIT (claims C1-C5 of m5_32_r18_0_closures.py, data/m5_32_r18_0.json).

Every number here is produced by this script's OWN constructions (no producer function is re-run on the producer's path;
the producer modules are imported only to (a) compare the analytic hedgehog seed and the frame's G with our own, and (b)
read the cfg constants).  Methods:
  C1  our own quartic density d(X, Y) = tr(G F G F^T), F = X G Y - Y G X (rebuild completion), G from our own spectral
      projector of N = M eta; the kinetic density kin(A_0) = 4 sum_i d(A_0, A_i) is exactly quadratic in A_0, so
      H_00[p, q] = 2 q(B_p, B_q) by the polarization identity (no finite differences); our own hedgehog M = 8 e_0 e_0^T
      (+) [delta I + (1 - delta) rhat rhat^T]; our own central-difference jets.
  C2  the producer's five readings plus: the traceless 8-space, the Lorentz-traceless 9-space, the Kramers block split
      (E_00 | 0i | spatial trace | spatial traceless), the diagonal entries in four bases (coordinate / orthonormal, lattice-
      and ray-adapted), the N-space basis E_pq eta (the 0i directions antisymmetric in M); at r 12 and r 6 on the analytic
      hedgehog and at r 12 on the relaxed R16-1 core; the match test = the 8 nonzero values proportional to
      {1,1,1,1,1,2,2,3} within 5 percent (least-squares scale).  PLUS the exact continuum H_00 (analytic jets of the
      hedgehog, d_i(rhat rhat^T) = (e_i rhat^T + rhat e_i^T - 2 rhat_i rhat rhat^T) / r, no lattice) ON the z axis, off the
      axis at the producer's cell position, and at a generic point; and the lattice convergence of reading (ii) at h 0.75.
  C3  r_0 from the field: lambda_1 = the largest eigenvalue of N = M eta per cell (np.linalg.eigvals, our own), 1.5 h shells
      on the free cells, the 0.8 crossing of the shell mean by linear interpolation; plus the raw central-line crossing.
  C4  counter A (primary, chart-free): the six Cartesian components of S = M[1:, 1:] on the shell cells are fitted with REAL
      scalar spherical harmonics (l <= L), the fitted tensor is evaluated on a fine (theta, phi) grid, the director is its
      top eigenvector oriented outward, the split section zeta = S_ee - S_ff + 2 i S_ef is formed PER FACE in a local
      polar frame whose pole is far from the face (about z on |z| < 0.7, about x elsewhere), the face winding of arg zeta
      is the index content of the face (quads plus the two cap polygons); zero faces are merged into zeros (4 degrees)
      and zeros into clusters (15 degrees).  Counter B (the task's method (a)): trilinear interpolation of S onto the
      sphere r_c, the same face-winding count; its artefact is measured on the split-free analytic hedgehog.  Counter A's
      own artefact is measured on a synthetic radial hedgehog carrying the R16-1 core's shell-mean eigenvalue profiles.
      Both counters are gated on synthetic sections (hedgehog + uniform traceless Q) built here.
  C5  our own spin-weighted harmonics (Wigner small-d) fitted to the z-frame chart section on the cells (validated: the
      constant-Q section lies in the l = 2 span to 1e-10, and the uniform-weight residual reproduces the producer's), then
      l_max in {2, 4, 6, 8} and three least-squares weightings; the zero count of each fit through counter A's machinery.

usage: cd scripts && /opt/anaconda3/envs/master312/bin/python3 m5_32_r18_0_audit.py
out:   data/m5_32_r18_0_audit.json
"""
from __future__ import annotations
import json
import math
import os
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import lpmv

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
DATA = os.path.join(RES, "data")
CK16 = os.path.join(RES, "checkpoints", "m5_32_r16")
CK17 = os.path.join(RES, "checkpoints", "m5_32_r17")
T0 = time.time()
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
AUTHOR = np.array([1, 1, 1, 1, 1, 2, 2, 3], dtype=float)
OURS = np.array([1, 1, 1, 1, 2, 2, 2, 6], dtype=float)
G_VAC, DELTA = 8.0, 0.3


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def plain(o):
    if isinstance(o, dict):
        return {str(k): plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return plain(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, complex):
        return [o.real, o.imag]
    return o


# ================================================================ our own lattice primitives
def coords(n, h):
    x = (np.arange(n) - (n - 1) / 2.0) * h
    return np.meshgrid(x, x, x, indexing="ij")


def own_hedgehog(n, h, lam1=1.0, lam23=DELTA, g=G_VAC):
    """M = g e_0 e_0^T (+) [lam23 I + (lam1 - lam23) rhat rhat^T]; lam1, lam23 may be arrays over the grid."""
    X, Y, Z = coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    rh = np.stack([X, Y, Z], -1) / r[..., None]
    M = np.zeros((n, n, n, 4, 4))
    M[..., 0, 0] = g
    l1 = np.broadcast_to(np.asarray(lam1, dtype=float), (n, n, n))
    l23 = np.broadcast_to(np.asarray(lam23, dtype=float), (n, n, n))
    M[..., 1:, 1:] = l23[..., None, None] * np.eye(3) + (l1 - l23)[..., None, None] * rh[..., :, None] * rh[..., None, :]
    return M


def central_jets(M, h):
    A = []
    for ax in range(3):
        d = np.zeros_like(M)
        sl_p = [slice(None)] * 3; sl_m = [slice(None)] * 3; sl_c = [slice(None)] * 3
        sl_p[ax] = slice(2, None); sl_m[ax] = slice(0, -2); sl_c[ax] = slice(1, -1)
        d[tuple(sl_c)] = (M[tuple(sl_p)] - M[tuple(sl_m)]) / (2.0 * h)
        A.append(d)
    return A


def own_G(M):
    """G = eta (I - 2 P_g), P_g the spectral projector of N = M eta onto its most negative eigenvalue, from our own eigen-solve."""
    N = M @ ETA
    w, V = np.linalg.eig(N)
    k = int(np.argmin(np.real(w)))
    v = np.real(V[:, k])
    # the eta-adjoint left eigenvector of an eta-self-adjoint N is eta v: P = v (eta v)^T / (v^T eta v)
    P = np.outer(v, ETA @ v) / float(v @ ETA @ v)
    assert np.allclose(P @ P, P, atol=1e-10) and np.allclose(P @ N, N @ P, atol=1e-10)
    return ETA @ (np.eye(4) - 2.0 * P), float(np.real(w[k]))


def pair_density(X, Y, G):
    F = X @ G @ Y - Y @ G @ X
    return float(np.trace(G @ F @ G @ F.T))


def kin_bilinear(X, Y, Ai, G):
    """q(X, Y) = 4 sum_i tr(G F_i(X) G F_i(Y)^T), F_i(X) = X G A_i - A_i G X (symmetric bilinear; q(X, X) = kin_h(X))."""
    tot = 0.0
    for A in Ai:
        FX = X @ G @ A - A @ G @ X
        FY = Y @ G @ A - A @ G @ Y
        tot += 4.0 * float(np.trace(G @ FX @ G @ FY.T))
    return tot


def sym_basis(pairs=None):
    """the Frobenius-orthonormal basis of Sym(4): E_aa and (E_ab + E_ba) / sqrt 2, labels 'ab'."""
    B, lab = [], []
    for a in range(4):
        for b in range(a, 4):
            E = np.zeros((4, 4))
            if a == b:
                E[a, a] = 1.0
            else:
                E[a, b] = E[b, a] = 2 ** -0.5
            B.append(E); lab.append(f"{a}{b}")
    return B, lab


def H00_exact(Ai, G, basis):
    nb = len(basis)
    H = np.zeros((nb, nb))
    for p in range(nb):
        for q in range(p, nb):
            H[p, q] = H[q, p] = 2.0 * kin_bilinear(basis[p], basis[q], Ai, G)
    return H


def pattern_match(vals, pattern, tol=0.05):
    """the sorted nonzero values proportional to the pattern within tol (least-squares scale)?  Returns (ok, maxdev, scale)."""
    v = np.sort(np.asarray(vals, dtype=float))
    if len(v) != len(pattern):
        return False, None, None
    s = float(np.dot(v, pattern) / np.dot(pattern, pattern))
    if s == 0.0:
        return False, None, None
    dev = float(np.max(np.abs(v / s - pattern) / pattern))
    return bool(dev < tol), dev, s


def nonzero(e, rel=1e-6):
    e = np.asarray(e, dtype=float)
    scale = max(float(np.max(np.abs(e))), 1e-300)
    return e[np.abs(e) > rel * scale], int(np.sum(np.abs(e) <= rel * scale))


def reading(vals, name):
    nz, n0 = nonzero(vals)
    okA, devA, sA = pattern_match(nz, AUTHOR)
    okO, devO, sO = pattern_match(nz, OURS)
    nzs = np.sort(nz)
    unit = float(np.min(np.abs(nzs))) if len(nzs) else 1.0
    return {"reading": name, "n_zero": n0, "n_nonzero": int(len(nz)), "nonzero_sorted_over_smallest": [round(float(x) / unit, 4) for x in nzs],
            "matches_author_5pct": okA, "author_maxdev": devA, "matches_ours_5pct": okO, "ours_maxdev": devO}


def compress(H, U):
    """the form restricted to the subspace spanned by the orthonormal columns of U."""
    return U.T @ H @ U


def orth_complement(vectors, dim=10):
    V = np.array(vectors).T
    Q, _ = np.linalg.qr(np.concatenate([V, np.eye(dim)], 1))
    return Q[:, V.shape[1]:dim]


def basis_from_frame(frame4):
    """the orthonormal Sym(4) basis adapted to an orthonormal 4-frame (columns): E_ab' = sym(f_a f_b^T) normalized."""
    B, lab = [], []
    names = ["0", "r", "e", "f"]
    for a in range(4):
        for b in range(a, 4):
            fa, fb = frame4[:, a], frame4[:, b]
            E = np.outer(fa, fb) + np.outer(fb, fa)
            E = E / np.linalg.norm(E)
            B.append(E); lab.append(names[a] + names[b])
    return B, lab


def all_readings(H, lab, rhat):
    """every reading of one 10 x 10 form H (Frobenius-orthonormal basis, labels 'ab')."""
    Hs = 0.5 * (H + H.T)
    out = []
    w, V = np.linalg.eigh(Hs)
    out.append(reading(w, "(i) Frobenius-orthonormal eigenvalues"))
    T = np.diag([1.0 if l[0] == l[1] else np.sqrt(2.0) for l in lab])
    out.append(reading(np.linalg.eigvalsh(T @ Hs @ T), "(ii) coordinate basis E_ab + E_ba (norm sqrt 2 off-diagonal)"))
    Ge = np.diag([ETA[int(l[0]), int(l[0])] * ETA[int(l[1]), int(l[1])] for l in lab])
    ge = np.linalg.eigvals(Ge @ Hs)
    out.append(reading(np.real(ge), "(iii) eta-weighted generalized eigenvalues of G^-1 H"))
    out.append(reading(np.abs(np.real(ge)), "(iii-abs) eta-weighted generalized eigenvalues, absolute values"))
    sp = [i for i, l in enumerate(lab) if l[0] != "0"]
    tm = [i for i, l in enumerate(lab) if l[0] == "0"]
    out.append(reading(np.linalg.eigvalsh(Hs[np.ix_(sp, sp)]), "(iv) spatial 6x6 block"))
    out.append(reading(np.linalg.eigvalsh(Hs[np.ix_(tm, tm)]), "(v) time-row 4x4 block"))
    e00 = np.zeros(10); e00[lab.index("00")] = 1.0
    tr3 = np.zeros(10)
    for k in ("11", "22", "33"):
        tr3[lab.index(k)] = 1.0 / np.sqrt(3.0)
    U8 = orth_complement([e00, tr3])
    out.append(reading(np.linalg.eigvalsh(compress(Hs, U8)), "(vi) traceless 8-space (E_00 and the spatial trace removed)"))
    # Lorentz-traceless: tr(eta X) = 0 -> the direction (E_00 + E_11 + E_22 + E_33)/2 removed?  tr(eta X) = -X_00 + X_11 + X_22 + X_33:
    # the normal vector is (-1, 1, 1, 1)/2 on the diagonal labels
    lt = np.zeros(10); lt[lab.index("00")] = -0.5
    for k in ("11", "22", "33"):
        lt[lab.index(k)] = 0.5
    U9 = orth_complement([lt])
    out.append(reading(np.linalg.eigvalsh(compress(Hs, U9)), "(vi-b) Lorentz-traceless 9-space (tr(eta X) = 0)"))
    # Kramers split: E_00 | 0i | spatial trace | spatial traceless (5)
    sp_tl = orth_complement([e00, tr3] + [np.eye(10)[i] for i in tm])
    blocks = {"E_00": np.linalg.eigvalsh(compress(Hs, e00[:, None])), "time-space 0i (3)": np.linalg.eigvalsh(Hs[np.ix_(tm, tm)]),
              "spatial trace (1)": np.linalg.eigvalsh(compress(Hs, tr3[:, None])), "spatial traceless (5)": np.linalg.eigvalsh(compress(Hs, sp_tl))}
    kram = np.concatenate([np.atleast_1d(v) for v in blocks.values()])
    rk = reading(kram, "(vii) Kramers block split: E_00 | 0i | spatial trace | spatial traceless, eigenvalues per block")
    # the coupling between the blocks (nonzero would make the block eigenvalues a different reading from (i))
    Ub = [e00[:, None], np.eye(10)[:, tm], tr3[:, None], sp_tl]
    coup = max(float(np.linalg.norm(Ub[a].T @ Hs @ Ub[b])) for a in range(4) for b in range(4) if a != b)
    rk["max_interblock_coupling_over_norm"] = coup / max(float(np.linalg.norm(Hs)), 1e-300)
    rk["blocks_over_smallest_nonzero"] = {k: [round(float(x) / max(float(np.min(np.abs(nonzero(kram)[0]))), 1e-300), 4) for x in np.atleast_1d(v)] for k, v in blocks.items()}
    out.append(rk)
    # diagonal entries in four bases
    out.append(reading(np.diag(Hs), "(viii-a) diagonal entries, lattice orthonormal basis"))
    out.append(reading(np.diag(T @ Hs @ T), "(viii-b) diagonal entries, lattice coordinate basis"))
    # the ray-adapted frame (e_0, rhat, e, f)
    e_ = np.cross(rhat, [0.0, 0.0, 1.0])
    if np.linalg.norm(e_) < 1e-6:
        e_ = np.cross(rhat, [1.0, 0.0, 0.0])
    e_ = e_ / np.linalg.norm(e_); f_ = np.cross(rhat, e_)
    F4 = np.zeros((4, 4)); F4[0, 0] = 1.0; F4[1:, 1] = rhat; F4[1:, 2] = e_; F4[1:, 3] = f_
    Bray, labray = basis_from_frame(F4)
    Bl, _ = sym_basis()
    # the change of basis: H_ray[p, q] = sum H[a, b] <B_a, Bray_p> <B_b, Bray_q>
    Cm = np.array([[float(np.sum(Bl[a] * Bray[p])) for p in range(10)] for a in range(10)])
    Hray = Cm.T @ Hs @ Cm
    Tray = np.diag([1.0 if l[0] == l[1] else np.sqrt(2.0) for l in labray])
    dray = reading(np.diag(Hray), "(viii-c) diagonal entries, ray-adapted orthonormal basis (0, r, e, f)")
    dray["entries"] = {labray[k]: round(float(Hray[k, k]), 6) for k in range(10)}
    out.append(dray)
    out.append(reading(np.diag(Tray @ Hray @ Tray), "(viii-d) diagonal entries, ray-adapted coordinate basis"))
    return out, (w, V)


# ================================================================ spherical-harmonic tools (our own)
def real_Y(lmax, th, ph):
    """real scalar spherical harmonics, all (l, m) with l <= lmax, as columns (…, (lmax + 1)^2)."""
    x = np.cos(th)
    cols = []
    for l in range(lmax + 1):
        for m in range(0, l + 1):
            N = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m))
            P = lpmv(m, l, x)
            if m == 0:
                cols.append(N * P)
            else:
                cols.append(math.sqrt(2.0) * N * P * np.cos(m * ph))
                cols.append(math.sqrt(2.0) * N * P * np.sin(m * ph))
    return np.stack(cols, -1)


def wigner_d(l, mp, m, beta):
    """d^l_{m' m}(beta), the explicit sum (Wikipedia / Sakurai)."""
    pref = math.sqrt(math.factorial(l + mp) * math.factorial(l - mp) * math.factorial(l + m) * math.factorial(l - m))
    c, s = np.cos(beta / 2.0), np.sin(beta / 2.0)
    tot = np.zeros_like(beta, dtype=float)
    for k in range(0, 2 * l + 1):
        a, b, cc, d = l + m - k, k, mp - m + k, l - mp - k
        if min(a, b, cc, d) < 0:
            continue
        tot += (-1) ** (mp - m + k) * c ** (2 * l + m - mp - 2 * k) * s ** (mp - m + 2 * k) / (math.factorial(a) * math.factorial(b) * math.factorial(cc) * math.factorial(d))
    return pref * tot


def sY(s, l, m, th, ph):
    """spin-weighted spherical harmonic (Wigner-d form): sY_lm = (-1)^s sqrt((2l+1)/4pi) d^l_{m,-s}(theta) e^{i m phi}."""
    return (-1) ** s * math.sqrt((2 * l + 1) / (4 * math.pi)) * wigner_d(l, m, -s, th) * np.exp(1j * m * ph)


def spin2_design(lmax, th, ph):
    lm = [(l, m) for l in range(2, lmax + 1) for m in range(-l, l + 1)]
    return np.stack([sY(2, l, m, th, ph) for l, m in lm], -1), lm


# ================================================================ the section and the counters (our own)
def polar_frame(rhat, axis):
    c = np.clip(np.sum(rhat * axis, -1), -1.0, 1.0)
    sn = np.maximum(np.sqrt(1.0 - c * c), 1e-300)
    e_th = (c[..., None] * rhat - axis) / sn[..., None]
    e_ph = np.cross(axis, rhat) / sn[..., None]
    return e_th, e_ph


def section(S, rhat, axis):
    """zeta = S_ee - S_ff + 2 i S_ef; e = e_theta(axis) projected transverse to the outward-oriented director, f = n x e."""
    w, V = np.linalg.eigh(S)
    nvec = V[..., :, -1]
    sg = np.sum(nvec * rhat, -1)
    sg = np.where(np.abs(sg) > 1e-12, np.sign(sg), 1.0)
    nvec = nvec * sg[..., None]
    e_th, _ = polar_frame(rhat, axis)
    e = e_th - np.sum(e_th * nvec, -1, keepdims=True) * nvec
    e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-300)
    f = np.cross(nvec, e)
    See = np.einsum("...a,...ab,...b->...", e, S, e)
    Sff = np.einsum("...a,...ab,...b->...", f, S, f)
    Sef = np.einsum("...a,...ab,...b->...", e, S, f)
    return See - Sff + 2j * Sef, w[..., -1] - w[..., -2], w[..., -2] - w[..., -3]


def wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


NTH, NPH = 180, 360
TH_G = (np.arange(NTH) + 0.5) * np.pi / NTH
PH_G = (np.arange(NPH) + 0.5) * 2 * np.pi / NPH
TG, PG = np.meshgrid(TH_G, PH_G, indexing="ij")
RH_G = np.stack([np.sin(TG) * np.cos(PG), np.sin(TG) * np.sin(PG), np.cos(TG)], -1)
EZ = np.array([0.0, 0.0, 1.0]); EX = np.array([1.0, 0.0, 0.0])


def count_on_grid(S_grid):
    """S_grid (NTH, NPH, 3, 3) on the offset (theta, phi) corner grid: face windings with per-face local frames
    (z-polar on |z| < 0.7, x-polar elsewhere), the two cap polygons, the merged zeros and clusters."""
    zz, gap, split = section(S_grid, RH_G, EZ)
    zx, _, _ = section(S_grid, RH_G, EX)
    zmax = float(np.max(np.abs(zz)))
    if zmax < 1e-10:
        return {"total_index": None, "caps": None, "n_zero_faces": 0, "zeros": [], "n_zeros": 0, "index_histogram": {}, "clusters": [], "n_clusters": 0, "cluster_indices": [],
                "split_rms_grid": float(np.sqrt(np.mean(np.abs(zz) ** 2))), "split_max_grid": zmax, "min_director_gap_grid": float(np.min(gap)), "max_abs_face_winding_nonint": 0.0,
                "note": "section zero to roundoff: nothing to count"}
    pz, px = np.angle(zz), np.angle(zx)
    kp1 = np.roll(np.arange(NPH), -1)
    # quads (i, k) -> (i+1, k) -> (i+1, k+1) -> (i, k+1): counterclockwise seen from outside
    zc = np.cos(0.5 * (TH_G[:-1] + TH_G[1:]))[:, None] * np.ones((1, NPH))
    use_z = np.abs(zc) < 0.7
    def wind(p):
        a, b, c, d = p[:-1, :], p[1:, :], p[1:, kp1], p[:-1, kp1]
        return (wrap(b - a) + wrap(c - b) + wrap(d - c) + wrap(a - d)) / (2 * np.pi)
    wq = np.where(use_z, wind(pz), wind(px))
    wq_i = np.rint(wq).astype(int)
    # the caps in the x-frame: north with increasing phi, south with decreasing phi
    north = float(np.sum(wrap(px[0, kp1] - px[0, :])) / (2 * np.pi))
    south = float(-np.sum(wrap(px[-1, kp1] - px[-1, :])) / (2 * np.pi))
    caps = [int(np.rint(north)), int(np.rint(south))]
    total = int(np.sum(wq_i)) + caps[0] + caps[1]
    # zero faces -> positions
    pts, idx = [], []
    for i, k in zip(*np.where(wq_i != 0)):
        t_, p_ = 0.5 * (TH_G[i] + TH_G[i + 1]), PH_G[k] + np.pi / NPH
        pts.append([np.sin(t_) * np.cos(p_), np.sin(t_) * np.sin(p_), np.cos(t_)]); idx.append(int(wq_i[i, k]))
    if caps[0] != 0:
        pts.append([0.0, 0.0, 1.0]); idx.append(caps[0])
    if caps[1] != 0:
        pts.append([0.0, 0.0, -1.0]); idx.append(caps[1])
    if len(pts) > 4000:
        return {"total_index": total, "caps": caps, "n_zero_faces": int(len(pts)), "zeros": [], "n_zeros": int(len(pts)), "index_histogram": {}, "clusters": [], "n_clusters": 0, "cluster_indices": [],
                "split_rms_grid": float(np.sqrt(np.mean(np.abs(zz) ** 2))), "split_max_grid": zmax, "min_director_gap_grid": float(np.min(gap)), "max_abs_face_winding_nonint": float(np.max(np.abs(wq - np.rint(wq)))),
                "note": "more than 4000 winding faces: the section is noise at the grid scale, zeros not merged"}
    zeros = merge(pts, idx, np.radians(4.0))
    clusters = merge([z["xyz"] for z in zeros], [z["index"] for z in zeros], np.radians(25.0), members=[z["index"] for z in zeros])
    for c in clusters:
        c["dot_z"] = round(float(abs(c["xyz"][2])), 4); c["dot_111"] = round(float(abs(np.dot(c["xyz"], [1, 1, 1]) / np.sqrt(3))), 4)
    return {"total_index": total, "caps": caps, "n_zero_faces": int(len(pts)), "zeros": zeros, "n_zeros": len(zeros),
            "index_histogram": {str(k): int(sum(1 for z in zeros if z["index"] == k)) for k in sorted(set(z["index"] for z in zeros))},
            "clusters": clusters, "n_clusters": len(clusters), "cluster_indices": sorted([c["index"] for c in clusters], reverse=True),
            "split_rms_grid": float(np.sqrt(np.mean(np.abs(zz) ** 2))), "split_max_grid": float(np.max(np.abs(zz))), "min_director_gap_grid": float(np.min(gap)),
            "max_abs_face_winding_nonint": float(np.max(np.abs(wq - np.rint(wq))))}


def merge(pts, idx, ang, members=None):
    """single-linkage merge of points on the sphere within the angle; the index sums."""
    pts = [np.asarray(p, dtype=float) for p in pts]
    n = len(pts)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if np.arccos(np.clip(np.dot(pts[i], pts[j]) / (np.linalg.norm(pts[i]) * np.linalg.norm(pts[j])), -1, 1)) < ang:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        c = np.mean([pts[i] for i in g], 0); c = c / max(np.linalg.norm(c), 1e-300)
        rec = {"xyz": [round(float(v), 4) for v in c], "index": int(sum(idx[i] for i in g)), "n_merged": len(g)}
        if members is not None:
            rec["member_indices"] = [int(members[i]) for i in g]
        out.append(rec)
    out.sort(key=lambda r_: (-abs(r_["index"]), -r_["xyz"][2]))
    return out


def shell_cells(M, n, h, r_c, width=None):
    X, Y, Z = coords(n, h)
    pos = np.stack([X, Y, Z], -1)
    r = np.sqrt(np.sum(pos * pos, -1))
    w = 1.5 * h if width is None else width
    m = np.abs(r - r_c) < w / 2.0
    return M[m][:, 1:, 1:], pos[m] / r[m][:, None], r[m]


def transverse_tensor(S):
    """the chart-free split object: T = P S P - (tr(P S P) / 2) P, P = I - n n^T (n the top eigenvector); symmetric, traceless,
    T n = 0, transverse eigenvalues +-(lambda_2 - lambda_3) / 2; independent of the radial eigenvalue profile."""
    w, V = np.linalg.eigh(S)
    nv = V[..., :, -1]
    P = np.eye(3) - nv[..., :, None] * nv[..., None, :]
    T = P @ S @ P
    tr = np.einsum("...aa->...", T)
    return T - 0.5 * tr[..., None, None] * P


def fit_tensor(S, rhat, lmax, weights=None):
    """least-squares fit of the six components of a symmetric tensor field with real Y_lm, l <= lmax; returns (coef, rel residual)."""
    th = np.arccos(np.clip(rhat[:, 2], -1, 1)); ph = np.arctan2(rhat[:, 1], rhat[:, 0])
    A = real_Y(lmax, th, ph)
    comp = np.stack([S[:, 0, 0], S[:, 1, 1], S[:, 2, 2], S[:, 0, 1], S[:, 0, 2], S[:, 1, 2]], -1)
    sw = np.ones(len(rhat)) if weights is None else np.sqrt(weights)
    coef = np.linalg.lstsq(A * sw[:, None], comp * sw[:, None], rcond=None)[0]
    fit = A @ coef
    resid = float(np.linalg.norm(fit - comp) / max(np.linalg.norm(comp), 1e-300))
    return coef, resid


def tensor_on_grid(coef, lmax):
    A = real_Y(lmax, TG, PG)
    c6 = A @ coef
    S = np.zeros(TG.shape + (3, 3))
    S[..., 0, 0], S[..., 1, 1], S[..., 2, 2] = c6[..., 0], c6[..., 1], c6[..., 2]
    S[..., 0, 1] = S[..., 1, 0] = c6[..., 3]; S[..., 0, 2] = S[..., 2, 0] = c6[..., 4]; S[..., 1, 2] = S[..., 2, 1] = c6[..., 5]
    return S


def pick_lmax(nc):
    return max(l for l in (2, 3, 4, 6, 8) if (l + 1) ** 2 <= nc - 6) if nc >= 15 else 2


def counter_A(M, n, h, r_c, lmax=None, width=None, weights_fn=None, mode="transverse"):
    """mode 'transverse' (primary): fit the chart-free transverse traceless tensor T on the cells, count on 2 rhat rhat^T + I + T_fit;
    mode 'components': fit the raw components of S (kept only to document its radial-profile artefact)."""
    S, rh, r = shell_cells(M, n, h, r_c, width)
    nc = len(rh)
    if lmax is None:
        lmax = pick_lmax(nc)
    wts = None if weights_fn is None else weights_fn(rh, r, r_c, h)
    if mode == "transverse":
        coef, resid = fit_tensor(transverse_tensor(S), rh, lmax, wts)
        Tg = tensor_on_grid(coef, lmax)
        rec = count_on_grid(2.0 * RH_G[..., :, None] * RH_G[..., None, :] + np.eye(3) + Tg)
    else:
        coef, resid = fit_tensor(S, rh, lmax, wts)
        rec = count_on_grid(tensor_on_grid(coef, lmax))
    zc, _, split_cells = section(S, rh, EZ)
    # the chart-free uniform quadrupole: the traceless part of the shell mean of S (the cubic-symmetric cell set averages the
    # hedgehog's rhat rhat^T to I / 3 exactly)
    Sm = np.mean(S, 0); Q = Sm - np.trace(Sm) / 3.0 * np.eye(3)
    wq, vq = np.linalg.eigh(Q)
    axis = vq[:, int(np.argmax(np.abs(wq)))]
    rec.update({"n_cells": nc, "lmax": lmax, "fit_residual_rel": resid, "split_rms_cells": float(np.sqrt(np.mean(np.abs(zc) ** 2))), "split_max_cells": float(np.max(np.abs(zc))),
                "Q_eff_eigs": [float(x) for x in wq], "Q_eff_axis": [round(float(x), 4) for x in axis], "Q_eff_axis_dot_z": round(float(abs(axis[2])), 4),
                "Q_eff_axis_dot_111": round(float(abs(np.dot(axis, [1, 1, 1]) / np.sqrt(3))), 4)})
    return rec


def counter_B(M, n, h, r_c):
    """trilinear interpolation of S onto the sphere r_c (the task's method (a)), the same face-winding count."""
    idx = (r_c * RH_G / h + (n - 1) / 2.0)
    S = np.zeros(TG.shape + (3, 3))
    for a in range(3):
        for b in range(a, 3):
            v = map_coordinates(np.ascontiguousarray(M[..., 1 + a, 1 + b]), [idx[..., 0].ravel(), idx[..., 1].ravel(), idx[..., 2].ravel()], order=1, mode="nearest").reshape(TG.shape)
            S[..., a, b] = v; S[..., b, a] = v
    rec = count_on_grid(S)
    return {k: rec[k] for k in ("total_index", "n_zeros", "index_histogram", "n_clusters", "cluster_indices", "split_rms_grid", "split_max_grid", "clusters")}


def w_triangular(rh, r, r_c, h):
    return np.maximum(1.0 - np.abs(r - r_c) / (0.75 * h), 0.05)


def w_inv_density(rh, r, r_c, h):
    c = rh @ rh.T
    cnt = np.sum(c > np.cos(np.radians(25.0)), 1)
    return 1.0 / cnt


def spin2_fit_count(M, n, h, r_c, lmax, weights_fn=None):
    """our spin-weighted fit of the z-frame chart section on the cells; the zero count through counter A's grid machinery
    (the fitted zeta is turned into the transverse tensor 2 rhat rhat^T + I + T(zeta), whose section in any frame is zeta)."""
    S, rh, r = shell_cells(M, n, h, r_c)
    zc, _, _ = section(S, rh, EZ)
    th = np.arccos(np.clip(rh[:, 2], -1, 1)); ph = np.arctan2(rh[:, 1], rh[:, 0])
    A, lm = spin2_design(lmax, th, ph)
    wts = np.ones(len(rh)) if weights_fn is None else weights_fn(rh, r, r_c, h)
    sw = np.sqrt(wts)
    coef = np.linalg.lstsq(A * sw[:, None], zc * sw, rcond=None)[0]
    fit = A @ coef
    zrms = float(np.sqrt(np.mean(np.abs(zc) ** 2)))
    resid = float(np.sqrt(np.mean(np.abs(fit - zc) ** 2)) / max(zrms, 1e-300))
    wres = float(np.sqrt(np.sum(wts * np.abs(fit - zc) ** 2) / np.sum(wts)) / max(zrms, 1e-300))
    Ag, _ = spin2_design(lmax, np.maximum(TG, 1e-9), PG)
    zg = Ag @ coef
    eth, eph = polar_frame(RH_G, EZ)
    T = 0.5 * np.real(zg)[..., None, None] * (eth[..., :, None] * eth[..., None, :] - eph[..., :, None] * eph[..., None, :]) \
        + 0.5 * np.imag(zg)[..., None, None] * (eth[..., :, None] * eph[..., None, :] + eph[..., :, None] * eth[..., None, :])
    Sg = 2.0 * RH_G[..., :, None] * RH_G[..., None, :] + np.eye(3) + T
    rec = count_on_grid(Sg)
    pw = {str(l): float(sum(abs(c) ** 2 for c, (l_, _) in zip(coef, lm) if l_ == l)) for l in range(2, lmax + 1)}
    ptot = max(sum(pw.values()), 1e-300)
    p2 = {str(m): float(abs(c) ** 2) for c, (l_, m) in zip(coef, lm) if l_ == 2}
    p2tot = max(sum(p2.values()), 1e-300)
    return {"lmax": lmax, "n_basis": len(lm), "n_cells": len(rh), "fit_residual_rel": resid, "weighted_residual_rel": wres, "total_index": rec["total_index"], "n_zeros": rec["n_zeros"],
            "index_histogram": rec["index_histogram"], "n_clusters": rec["n_clusters"], "cluster_indices": rec["cluster_indices"], "clusters": rec["clusters"],
            "l_power_fraction": {k: v / ptot for k, v in pw.items()}, "l2_m_power_fraction": {k: v / p2tot for k, v in p2.items()},
            "l2_nonaxial_amplitude_fraction": float(np.sqrt(sum(v for k, v in p2.items() if k != "0") / p2tot))}


# ================================================================ C1 + C2
def audit_C1_C2():
    log("C1/C2: our own H_00 on our own hedgehog (exact quadratic form by polarization)")
    n, L = 32, 48.0
    h = L / n
    X, Y, Z = coords(n, h)
    Mh = own_hedgehog(n, h)
    # convention cross-checks against the producer's seed and frame (comparison only)
    sys.path.insert(0, HERE)
    import m5_32_r16_common as C
    cfg = C.cfg_v4(n, L, mu=0.0, cP=0.0, cs=0.0, completion="rebuild", n_samples=1)
    Mp = C.C15.seed_uniaxial(cfg)
    seed_diff = float(np.max(np.abs(Mp - Mh)))
    j = n // 2
    cells = {}
    for rt in (12.0, 6.0):
        kz = int(np.argmin(np.abs(Z[j, j, :] - rt)))
        cells[f"axis_r{rt:g}"] = (j, j, kz)
    Ah = central_jets(Mh, h)
    out = {"seed_max_abs_diff_ours_vs_producer": seed_diff, "h": h, "cells": {}, "C1": {}, "C2": {}}
    prod = json.load(open(os.path.join(DATA, "m5_32_r18_0.json")))["a_H00_conventions"]["backgrounds"]
    Bl, lab = sym_basis()
    Mrel = np.load(os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"))
    Arel = central_jets(Mrel, h)
    frp = C.frame(Mp, C.radial_ref(cfg))
    frr = C.frame(Mrel, C.radial_ref(cfg))
    backgrounds = {"analytic_hedgehog_n32_L48": (Mh, Ah, frp), "r16_1_end_n32_L48": (Mrel, Arel, frr)}
    for bg, (M, A, frx) in backgrounds.items():
        for cname, c in cells.items():
            if bg == "r16_1_end_n32_L48" and cname != "axis_r12":
                continue
            pos = np.array([X[c], Y[c], Z[c]]); rhat = pos / np.linalg.norm(pos)
            Gm, lg = own_G(M[c])
            G_diff = float(np.max(np.abs(Gm - np.real(frx["G"][c]))))
            Ai = [a[c] for a in A]
            H = H00_exact(Ai, Gm, Bl)
            # self-gate: the quadratic form reproduces kin_h on random A_0 (0.5 x^T H x = kin(x))
            rng = np.random.default_rng(1)
            gate = 0.0
            for _ in range(5):
                x = rng.standard_normal(10)
                X0 = sum(x[p] * Bl[p] for p in range(10))
                gate = max(gate, abs(0.5 * x @ H @ x - kin_bilinear(X0, X0, Ai, Gm)) / max(abs(kin_bilinear(X0, X0, Ai, Gm)), 1e-300))
            w, V = np.linalg.eigh(0.5 * (H + H.T))
            key = f"{bg}|{cname}"
            out["cells"][key] = {"index": [int(v) for v in c], "xyz": [float(v) for v in pos], "r": float(np.linalg.norm(pos)), "G_is_identity_maxdev": float(np.max(np.abs(Gm - np.eye(4)))),
                                 "G_max_abs_diff_ours_vs_producer_frame": G_diff, "lambda_g": lg, "quadratic_form_self_gate_rel": gate}
            pe = np.array(prod[bg]["contractions"]["quartic_rebuild"][cname]["eigenvalues_raw"])
            nz, n0 = nonzero(w)
            unit = float(np.min(nz))
            ratios = np.sort(nz) / unit
            okO, devO, _ = pattern_match(nz, OURS)
            # kernel: E_00 and the spatial identity
            e00 = Bl[lab.index("00")]; id3 = sum(Bl[lab.index(k)] for k in ("11", "22", "33")) / np.sqrt(3.0)
            def Hv(Xm):
                v = np.array([float(np.sum(Xm * Bl[p])) for p in range(10)])
                return float(np.linalg.norm(H @ v) / max(np.max(np.abs(w)), 1e-300))
            # top eigenvector vs the axial quadrupole along the ray
            Qray = np.zeros((4, 4)); Qray[1:, 1:] = (3.0 * np.outer(rhat, rhat) - np.eye(3)) / np.sqrt(6.0)
            vtop = sum(V[p, -1] * Bl[p] for p in range(10))
            ov = float(abs(np.sum(vtop * Qray)))
            rec = {"eigenvalues_ours": w.tolist(), "eigenvalues_producer": pe.tolist(), "max_abs_diff_ours_vs_producer": float(np.max(np.abs(np.sort(w) - np.sort(pe)))),
                   "max_rel_diff_on_nonzero": float(np.max(np.abs(np.sort(w)[n0:] - np.sort(pe)[n0:]) / np.sort(pe)[n0:])) if n0 < 10 else None,
                   "n_zero_ours": n0, "nonzero_over_smallest": [round(float(x), 5) for x in ratios], "matches_0011112226_5pct": okO, "maxdev_from_pattern": devO,
                   "kernel_residual_E00": Hv(e00), "kernel_residual_spatial_identity": Hv(id3), "top_overlap_with_axial_quadrupole_along_ray": ov,
                   "top_eigenvalue": float(w[-1]), "trace": float(np.trace(H))}
            out["C1"][key] = rec
            rd, _ = all_readings(H, lab, rhat)
            # the N-space reading: perturbations E_pq eta (antisymmetric in M on the 0i directions; the density accepts any 4 x 4)
            BN = [B @ ETA for B in Bl]
            HN = H00_exact(Ai, Gm, BN)
            rd.append(reading(np.linalg.eigvalsh(0.5 * (HN + HN.T)), "(ix) N-space basis E_pq eta (the 0i directions antisymmetric in M)"))
            rd.append(reading(np.diag(HN), "(ix-b) N-space basis, diagonal entries"))
            cand = [r_ for r_ in rd if r_["author_maxdev"] is not None]
            closest = min(cand, key=lambda r_: r_["author_maxdev"]) if cand else None
            out["C2"][key] = {"readings": rd, "any_reading_matches_author": bool(any(r_["matches_author_5pct"] for r_ in rd)),
                              "readings_matching_author": [r_["reading"] for r_ in rd if r_["matches_author_5pct"]],
                              "closest_reading": None if closest is None else {"reading": closest["reading"], "author_maxdev": closest["author_maxdev"], "nonzero_sorted_over_smallest": closest["nonzero_sorted_over_smallest"]}}
            log(f"  {key}: r {np.linalg.norm(pos):.2f}, G-I {rec_or(out['cells'][key]['G_is_identity_maxdev'])}, gate {gate:.1e}, ours/producer maxdiff {rec['max_abs_diff_ours_vs_producer']:.2e}; "
                f"ratios {rec['nonzero_over_smallest']}; kernel res {rec['kernel_residual_E00']:.1e}/{rec['kernel_residual_spatial_identity']:.1e}; top overlap {ov:.4f}; author match in {out['C2'][key]['readings_matching_author']}")
    # the exact continuum H_00 (no lattice): on the z axis, at the producer's off-axis cell position, at a generic point
    log("C2: the exact continuum H_00 (analytic jets), on and off the z axis")
    out["C2_exact_continuum"] = {}
    for nm, x in (("on_z_axis_r12", np.array([0.0, 0.0, 12.0])), ("producer_cell_position_(0.75,0.75,11.25)", np.array([0.75, 0.75, 11.25])), ("generic_point_(3,-2,5)", np.array([3.0, -2.0, 5.0]))):
        H = H00_exact(exact_jets(x), np.eye(4), Bl)
        rd, _ = all_readings(H, lab, x / np.linalg.norm(x))
        out["C2_exact_continuum"][nm] = {"xyz": x.tolist(), "angle_from_z_axis_deg": float(np.degrees(np.arccos(x[2] / np.linalg.norm(x)))), "readings": rd,
                                         "readings_matching_author": [r_["reading"] for r_ in rd if r_["matches_author_5pct"]],
                                         "orthonormal": rd[0]["nonzero_sorted_over_smallest"], "coordinate_basis": rd[1]["nonzero_sorted_over_smallest"], "coordinate_basis_author_maxdev": rd[1]["author_maxdev"]}
        log(f"  {nm}: orthonormal {rd[0]['nonzero_sorted_over_smallest']}; coordinate {rd[1]['nonzero_sorted_over_smallest']} (author maxdev {rd[1]['author_maxdev']:.1e}); matches {out['C2_exact_continuum'][nm]['readings_matching_author']}")
    # the lattice convergence of reading (ii) at the producer's cell rule (x = y = h/2, z nearest 12) for h 1.5 and 0.75
    conv = {}
    for n_, L_ in ((32, 48.0), (64, 48.0)):
        h_ = L_ / n_
        M_ = own_hedgehog(n_, h_)
        X_, Y_, Z_ = coords(n_, h_)
        j_ = n_ // 2; kz_ = int(np.argmin(np.abs(Z_[j_, j_, :] - 12.0)))
        A_ = [a[j_, j_, kz_] for a in central_jets(M_, h_)]
        H_ = H00_exact(A_, np.eye(4), Bl)
        pos_ = np.array([X_[j_, j_, kz_], Y_[j_, j_, kz_], Z_[j_, j_, kz_]])
        rd_, _ = all_readings(H_, lab, pos_ / np.linalg.norm(pos_))
        conv[f"h_{h_}"] = {"cell_xyz": pos_.tolist(), "angle_from_z_axis_deg": float(np.degrees(np.arccos(pos_[2] / np.linalg.norm(pos_)))), "coordinate_basis": rd_[1]["nonzero_sorted_over_smallest"], "author_maxdev": rd_[1]["author_maxdev"]}
        log(f"  lattice h {h_}: cell {pos_.tolist()} ({conv[f'h_{h_}']['angle_from_z_axis_deg']:.1f} deg off axis) coordinate {rd_[1]['nonzero_sorted_over_smallest']} author maxdev {rd_[1]['author_maxdev']:.3f}")
    out["C2_lattice_convergence_of_reading_ii"] = conv
    return out


def exact_jets(x, delta=DELTA):
    """the continuum jets of the analytic hedgehog at the point x: A_i = d_i M, spatial block (1 - delta) d_i(rhat rhat^T)."""
    r = np.linalg.norm(x); nv = x / r
    A = []
    for i in range(3):
        d = np.zeros((4, 4))
        for a in range(3):
            for b in range(3):
                d[1 + a, 1 + b] = (1 - delta) * ((1.0 if i == a else 0.0) * nv[b] + nv[a] * (1.0 if i == b else 0.0) - 2 * nv[a] * nv[b] * nv[i]) / r
        A.append(d)
    return A


def rec_or(x):
    return f"{x:.1e}"


# ================================================================ C3
def audit_C3():
    log("C3: r_0 recomputed from the field")
    n, L = 32, 48.0
    h = L / n
    M = np.load(os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"))
    X, Y, Z = coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    N = M @ ETA
    w = np.linalg.eigvals(N)
    im = float(np.max(np.abs(np.imag(w))))
    l1 = np.max(np.real(w), -1)
    wc = int(math.ceil(1.6 / h))
    free = np.ones((n, n, n), dtype=bool)
    for ax in range(3):
        sl = [slice(None)] * 3
        sl[ax] = slice(0, wc); free[tuple(sl)] = False
        sl[ax] = slice(n - wc, n); free[tuple(sl)] = False
    edges = np.arange(0.0, L / 2 + h, 1.5 * h)
    rr, ll, nn = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mk = (r >= a) & (r < b) & free
        if np.sum(mk) == 0:
            continue
        rr.append(0.5 * (a + b)); ll.append(float(np.mean(l1[mk]))); nn.append(int(np.sum(mk)))
    rr, ll = np.array(rr), np.array(ll)
    cross = np.where((ll[:-1] < 0.8) & (ll[1:] >= 0.8))[0]
    k = int(cross[0])
    r0 = float(rr[k] + (0.8 - ll[k]) * (rr[k + 1] - rr[k]) / (ll[k + 1] - ll[k]))
    # the raw central-line crossing (x = y = h/2 line along +z) as a second read
    j = n // 2
    zl = Z[j, j, j:]; l1l = l1[j, j, j:]
    cl = np.where((l1l[:-1] < 0.8) & (l1l[1:] >= 0.8))[0]
    kc = int(cl[0])
    r0_line = float(zl[kc] + (0.8 - l1l[kc]) * (zl[kc + 1] - zl[kc]) / (l1l[kc + 1] - l1l[kc]))
    # the finer-shell variant (0.75 h shells) as a sensitivity read
    edges2 = np.arange(0.0, L / 2 + h, 0.75 * h)
    rr2, ll2 = [], []
    for a, b in zip(edges2[:-1], edges2[1:]):
        mk = (r >= a) & (r < b) & free
        if np.sum(mk) == 0:
            continue
        rr2.append(0.5 * (a + b)); ll2.append(float(np.mean(l1[mk])))
    rr2, ll2 = np.array(rr2), np.array(ll2)
    c2 = np.where((ll2[:-1] < 0.8) & (ll2[1:] >= 0.8))[0]; k2 = int(c2[0])
    r0_fine = float(rr2[k2] + (0.8 - ll2[k2]) * (rr2[k2 + 1] - rr2[k2]) / (ll2[k2 + 1] - ll2[k2]))
    prod = json.load(open(os.path.join(DATA, "m5_32_r17_0_record.json")))["c_core_reads_r16_1"]
    key = "r_0_profile (shell-mean lambda_1 = 0.8)"
    rows = []
    for tag, hh in (("r16_1_end_n32_L48", 1.5), ("r16_1_end_n48_L72", 1.5), ("r16_1_end_n64_L48", 0.75)):
        v = float(prod[tag][key])
        rows.append({"core": tag, "h": hh, "r_0_box_units_record": v, "cells": v / hh, "r_0_sqrt_mu": v * 0.1, "section_177_reading_cells": v * 0.1 / hh})
    out = {"max_imag_eig_N": im, "shells": [{"r_mid": float(a), "l1_mean": float(b), "n": c} for a, b, c in zip(rr, ll, nn)][:8], "r_0_profile_ours": r0, "r_0_profile_record": rows[0]["r_0_box_units_record"],
           "abs_diff": abs(r0 - rows[0]["r_0_box_units_record"]), "r_0_central_line_raw_crossing": r0_line, "r_0_profile_0.75h_shells": r0_fine, "record_rows_arithmetic": rows,
           "claimed_cells": [2.03, 2.07, 1.33], "recomputed_cells": [round(r_["cells"], 3) for r_ in rows]}
    log(f"  r_0 ours {r0:.4f} vs record {rows[0]['r_0_box_units_record']:.4f} (diff {out['abs_diff']:.1e}); central line {r0_line:.3f}; 0.75h shells {r0_fine:.3f}; cells {out['recomputed_cells']}")
    return out


# ================================================================ C4 + C5
def fmt(rec):
    return f"tot {rec['total_index']} nz {rec['n_zeros']} hist {rec['index_histogram']} clusters {rec['cluster_indices']} at {[c['xyz'] for c in rec['clusters']]}"


def audit_C4_C5():
    log("C4: counters gated on our synthetic sections")
    n, L = 32, 48.0
    h = L / n
    Mh = own_hedgehog(n, h)
    out = {"gate": {}, "artefact": {}, "fields": {}, "C5": {}}
    radii = (1.5, 2.25, 3.0, 4.5, 6.0)
    # ---- gates: hedgehog + amp Q
    def rot(th_, ph_):
        return np.array([[np.cos(th_), 0, np.sin(th_)], [0, 1, 0], [-np.sin(th_), 0, np.cos(th_)]]) @ np.array([[np.cos(ph_), -np.sin(ph_), 0], [np.sin(ph_), np.cos(ph_), 0], [0, 0, 1]])
    Qx = np.diag([1.0, -1.0, 0.0]); Qz = np.diag([-0.5, -0.5, 1.0])
    R1 = rot(0.0, 0.3); R2 = rot(0.3, 0.2); Rr = rot(0.7, 1.9)
    Rz111 = None
    tests = {"Qx (xx - yy)": (Qx, None), "Qz (zz - (xx+yy)/2)": (Qz, [0, 0, 1]), "Qx rotated 0.3 about z": (R1 @ Qx @ R1.T, None), "Qz tilted (0.3, 0.2)": (R2 @ Qz @ R2.T, R2 @ np.array([0, 0, 1.0])),
             "Qx generic rotation": (Rr @ Qx @ Rr.T, None)}
    gate_ok, gate_ind = True, {}
    for name, (Q, ax) in tests.items():
        Mq = Mh.copy(); Mq[..., 1:, 1:] += 0.02 * Q
        # the true zeros of the linear section: Q = xx - yy type -> four simple zeros where the transverse part of Q is isotropic;
        # Q = zz type -> two double zeros on +-axis.  We locate the truth from the linear section on the fine grid of the exact
        # tensor (no fit): the faces where |Q_t| is minimal, by the same winding count on the exact tensor.
        Sg = 2.0 * RH_G[..., :, None] * RH_G[..., None, :] + np.eye(3) + 0.02 * (Q - RH_G[..., :, None] * (RH_G @ Q)[..., None, :] - (RH_G @ Q)[..., :, None] * RH_G[..., None, :] + np.einsum("...a,ab,...b->...", RH_G, Q, RH_G)[..., None, None] * RH_G[..., :, None] * RH_G[..., None, :])
        truth = count_on_grid(Sg)
        res = {"truth_on_exact_tensor": {k: truth[k] for k in ("total_index", "n_zeros", "index_histogram", "cluster_indices", "clusters")}}
        for r_c in radii:
            a = counter_A(Mq, n, h, r_c)
            res[f"{r_c:g}"] = {"A": {k: a[k] for k in ("total_index", "n_zeros", "index_histogram", "n_clusters", "cluster_indices", "lmax", "n_cells", "fit_residual_rel", "clusters")}}
            # net-index gate: within 30 degrees of each true zero the indices sum to the true index; total 4
            net_ok = a["total_index"] == 4
            for tz in truth["zeros"]:
                near = sum(z["index"] for z in a["zeros"] if np.dot(z["xyz"], tz["xyz"]) > np.cos(np.radians(30)))
                net_ok = net_ok and near == tz["index"]
            ind_ok = net_ok and a["n_zeros"] == truth["n_zeros"] and a["index_histogram"] == truth["index_histogram"]
            res[f"{r_c:g}"]["net_index_ok"] = bool(net_ok); res[f"{r_c:g}"]["individual_zeros_ok"] = bool(ind_ok)
            gate_ok = gate_ok and net_ok
            gate_ind.setdefault(name, []).append(bool(ind_ok))
        out["gate"][name] = res
        log(f"  gate A {name}: truth {truth['index_histogram']}; " + "; ".join(f"r{k} net {'OK' if v['net_index_ok'] else 'FAIL'} ind {'OK' if v['individual_zeros_ok'] else 'FAIL'} {fmt(v['A'])}" for k, v in res.items() if k != "truth_on_exact_tensor"))
    out["gate"]["counter_A_net_index_all_pass"] = bool(gate_ok)
    out["gate"]["counter_A_individual_zeros_pass_by_shell"] = {k: dict(zip([f"{r_:g}" for r_ in radii], v)) for k, v in gate_ind.items()}
    out["gate"]["counter_A_all_pass"] = bool(gate_ok)
    # counter B gate at a large amplitude (0.3: well above the interpolation artefact) and at 0.02
    gB = {}
    for amp in (0.3, 0.02):
        Mq = Mh.copy(); Mq[..., 1:, 1:] += amp * Qx
        gB[f"Qx amp {amp}"] = {f"{r_c:g}": counter_B(Mq, n, h, r_c) for r_c in radii}
        log(f"  gate B Qx amp {amp}: " + "; ".join(f"r{k} tot {v['total_index']} nz {v['n_zeros']} cl {v['cluster_indices']}" for k, v in gB[f'Qx amp {amp}'].items()))
    out["gate"]["counter_B"] = gB
    # ---- artefacts
    artB = {}
    for r_c in radii:
        b = counter_B(Mh, n, h, r_c)
        artB[f"{r_c:g}"] = {"split_rms": b["split_rms_grid"], "split_max": b["split_max_grid"], "h_over_r_sq": (h / r_c) ** 2, "total_index": b["total_index"], "n_zeros": b["n_zeros"]}
    out["artefact"]["B_trilinear_on_exact_hedgehog"] = artB
    log("  artefact B (trilinear, exact hedgehog): " + "; ".join(f"r{k} rms {v['split_rms']:.1e} max {v['split_max']:.1e}" for k, v in artB.items()))
    # counter A's own artefact: the exact hedgehog (cells) and a radial hedgehog carrying the R16-1 shell-mean profiles
    Mrel = np.load(os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"))
    X, Y, Z = coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    wN = np.sort(np.real(np.linalg.eigvals(Mrel @ ETA)), -1)
    l1, l2, l3 = wN[..., -1], wN[..., -2], wN[..., -3]
    edges = np.arange(0.0, L / 2 + h, 0.75 * h)
    rm, p1, p23 = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mk = (r >= a) & (r < b)
        if np.sum(mk) == 0:
            continue
        rm.append(0.5 * (a + b)); p1.append(float(np.mean(l1[mk]))); p23.append(float(np.mean(0.5 * (l2 + l3)[mk])))
    L1 = np.interp(r, rm, p1); L23 = np.interp(r, rm, p23)
    Mprof = own_hedgehog(n, h, lam1=L1, lam23=L23)
    artA = {}
    for r_c in radii:
        a0 = counter_A(Mh, n, h, r_c)
        a1 = counter_A(Mprof, n, h, r_c)
        a2 = counter_A(Mprof, n, h, r_c, mode="components")
        artA[f"{r_c:g}"] = {"transverse_fit_exact_hedgehog_split_max_grid": a0["split_max_grid"], "transverse_fit_profiled_hedgehog_split_max_grid": a1["split_max_grid"],
                            "components_fit_profiled_hedgehog_split_max_grid (the rejected first design)": a2["split_max_grid"], "components_fit_profiled_n_zeros": a2["n_zeros"], "transverse_fit_profiled_note": a1.get("note"),
                            "profiled_lambda1_shell_mean": float(np.interp(r_c, rm, p1)), "lmax": a1["lmax"]}
    out["artefact"]["A_fit_on_split_free_hedgehogs"] = artA
    out["artefact"]["profiles_used"] = {"r_mid": rm, "lambda_1": p1, "lambda_23": p23}
    log("  artefact A (split-free hedgehogs): " + "; ".join(f"r{k} transverse exact {v['transverse_fit_exact_hedgehog_split_max_grid']:.1e} profiled {v['transverse_fit_profiled_hedgehog_split_max_grid']:.1e} (components fit {v['components_fit_profiled_hedgehog_split_max_grid (the rejected first design)']:.1e}, {v['components_fit_profiled_n_zeros']} zeros)" for k, v in artA.items()))
    # ---- the fields
    prod = json.load(open(os.path.join(DATA, "m5_32_r18_0.json")))["c_spin2_zeros"]["fields"]
    fields = {"v6_gW1.35_seeded_end": os.path.join(CK17, "r17_2_v6_gW1.35_n32_L48_r16_1_split0.05.npy"),
              "v6_gW0.5_seeded_end": os.path.join(CK17, "r17_2_v6_gW0.5_n32_L48_r16_1_split0.05.npy"),
              "v4rel_static_end (residual split 2e-4)": os.path.join(CK17, "r17_2_v4rel_n32_L48_r16_1.npy")}
    for name, p in fields.items():
        M = np.load(p)
        rec = {}
        for r_c in radii:
            a = counter_A(M, n, h, r_c)
            # robustness of counter A: the other admissible l_max values and the two weightings
            variants = {}
            for lm in (2, 3, 4, 6, 8):
                if (lm + 1) ** 2 <= a["n_cells"] - 6 and lm != a["lmax"]:
                    v = counter_A(M, n, h, r_c, lmax=lm)
                    variants[f"lmax{lm}"] = {k: v[k] for k in ("total_index", "n_zeros", "cluster_indices", "fit_residual_rel")}
            for wn, wf in (("triangular", w_triangular), ("inv_density", w_inv_density)):
                v = counter_A(M, n, h, r_c, weights_fn=wf)
                variants[wn] = {k: v[k] for k in ("total_index", "n_zeros", "cluster_indices", "fit_residual_rel")}
            v = counter_A(M, n, h, r_c, width=2.0 * h)
            variants["width_2h"] = {k: v[k] for k in ("total_index", "n_zeros", "cluster_indices", "fit_residual_rel", "n_cells")}
            b = counter_B(M, n, h, r_c)
            artefact_B = artB[f"{r_c:g}"]["split_max"]
            artefact_A = artA[f"{r_c:g}"]["transverse_fit_profiled_hedgehog_split_max_grid"]
            pz = prod[name][f"{r_c:g}"]
            # the producer's zeros against ours: the angular distance from each producer zero to our nearest zero (degrees)
            pzx = np.array([z["xyz"] for z in pz["zeros"]], dtype=float)
            ozx = np.array([z["xyz"] for z in a["zeros"]], dtype=float) if a["zeros"] else np.zeros((0, 3))
            if len(pzx) and len(ozx):
                dd = np.degrees(np.arccos(np.clip(pzx @ ozx.T / (np.linalg.norm(pzx, axis=1)[:, None] * np.linalg.norm(ozx, axis=1)[None, :]), -1, 1)))
                match_deg = [round(float(x), 1) for x in dd.min(1)]
            else:
                match_deg = None
            rec[f"{r_c:g}"] = {"A": a, "A_variants": variants, "B": b,
                               "producer": {"total_index": pz["total_index"], "n_zeros": pz["n_zeros"], "index_histogram": pz["index_histogram"], "fit_residual_rel": pz["fit_residual_rel"], "zeta_rms_cells": pz["zeta_rms_cells"],
                                            "zeros_xyz": [z["xyz"] for z in pz["zeros"]], "hull": pz["hull_count"]},
                               "signal_over_artefact_A": a["split_rms_cells"] / max(artefact_A, 1e-300), "signal_over_artefact_B": a["split_rms_cells"] / max(artefact_B, 1e-300),
                               "producer_zero_to_our_nearest_zero_deg": match_deg,
                               "A_trustworthy (signal > 3 x artefact A and fit residual < 0.3)": bool(a["split_rms_cells"] > 3 * artefact_A and a["fit_residual_rel"] < 0.3), "B_trustworthy (signal > 3 x artefact B)": bool(a["split_rms_cells"] > 3 * artefact_B),
                               "total_agrees_with_producer": bool(a["total_index"] == pz["total_index"]),
                               "variants_all_total_4": bool(all(v_["total_index"] == 4 for v_ in variants.values()) and a["total_index"] == 4),
                               "variants_cluster_indices": sorted(set(str(v_["cluster_indices"]) for v_ in variants.values()) | {str(a["cluster_indices"])})}
            log(f"  {name} r{r_c:g}: A {fmt(a)} (lmax {a['lmax']}, resid {a['fit_residual_rel']:.3f}, split rms {a['split_rms_cells']:.1e}, Qeff axis z {a['Q_eff_axis_dot_z']} 111 {a['Q_eff_axis_dot_111']}); "
                f"B tot {b['total_index']} nz {b['n_zeros']} cl {b['cluster_indices']}; producer tot {pz['total_index']} nz {pz['n_zeros']} hist {pz['index_histogram']} (their zeros to ours {match_deg} deg); sig/art A {rec[f'{r_c:g}']['signal_over_artefact_A']:.1e} B {rec[f'{r_c:g}']['signal_over_artefact_B']:.1f}; "
                f"variants totals {[v_['total_index'] for v_ in variants.values()]} clusters {rec[f'{r_c:g}']['variants_cluster_indices']}")
        out["fields"][name] = rec
    # ---- C5: our spin-weighted fit, l_max and weighting scan
    log("C5: our spin-weighted fit of the chart section (Wigner-d harmonics), l_max and weighting scan")
    # validation of our sY basis: the constant-Q section on the exact hedgehog lies in the l = 2 span
    # (a) the LINEAR section of a constant traceless Q in the rhat-frame, zeta = Q_thth - Q_phph + 2 i Q_thph, is exactly spin-2, l = 2
    Sh, rhq, rr_ = shell_cells(Mh, n, h, 4.5)
    Qg = Rr @ Qx @ Rr.T
    eth, eph = polar_frame(rhq, EZ)
    zq = np.einsum("ia,ab,ib->i", eth, Qg, eth) - np.einsum("ia,ab,ib->i", eph, Qg, eph) + 2j * np.einsum("ia,ab,ib->i", eth, Qg, eph)
    th = np.arccos(np.clip(rhq[:, 2], -1, 1)); ph = np.arctan2(rhq[:, 1], rhq[:, 0])
    A2, lm2 = spin2_design(2, th, ph)
    c2 = np.linalg.lstsq(A2, zq, rcond=None)[0]
    span_res = float(np.linalg.norm(A2 @ c2 - zq) / np.linalg.norm(zq))
    # (b) the same for a degree-4 tensor field T_ab = (rhat . v)^2 Q_ab (its transverse section has l <= 4, and an l = 3 component in general)
    v_ = np.array([0.3, -0.5, 0.8]); v_ = v_ / np.linalg.norm(v_)
    sc = (rhq @ v_) ** 2
    zq4 = sc * zq
    A4, lm4 = spin2_design(4, th, ph)
    c4 = np.linalg.lstsq(A4, zq4, rcond=None)[0]
    span_res4 = float(np.linalg.norm(A4 @ c4 - zq4) / np.linalg.norm(zq4))
    c4b = np.linalg.lstsq(A2, zq4, rcond=None)[0]
    span_res4_in_l2 = float(np.linalg.norm(A2 @ c4b - zq4) / np.linalg.norm(zq4))
    # (c) orthonormality of our sY (l <= 6) under Gauss-Legendre in cos theta x uniform phi (exact for the polynomial degree)
    xg, wg = np.polynomial.legendre.leggauss(60)
    pq = (np.arange(120) + 0.5) * 2 * np.pi / 120
    TQ, PQ = np.meshgrid(np.arccos(xg), pq, indexing="ij")
    Aq, lmq = spin2_design(6, TQ.ravel(), PQ.ravel())
    wq_ = (wg[:, None] * np.ones((1, 120)) * (2 * np.pi / 120)).ravel()
    Gram = (Aq.conj() * wq_[:, None]).T @ Aq
    orth_dev = float(np.max(np.abs(Gram - np.eye(len(lmq)))))
    out["C5"]["basis_validation"] = {"linear_constant_Q_section_in_l2_span_residual": span_res, "degree4_section_in_l_le_4_span_residual": span_res4, "degree4_section_in_l2_only_residual (must be large)": span_res4_in_l2,
                                     "gram_max_dev_from_identity_l_le_6_gauss_legendre": orth_dev}
    log(f"  sY basis: linear constant-Q in l=2 span residual {span_res:.1e}; degree-4 section in l<=4 span {span_res4:.1e} (in l=2 only {span_res4_in_l2:.2f}); Gram deviation {orth_dev:.1e}")
    c5 = {}
    for name, p in fields.items():
        M = np.load(p)
        c5[name] = {}
        for r_c in radii:
            nc = len(shell_cells(M, n, h, r_c)[1])
            rows = {}
            for lm in (2, 4, 6, 8):
                nb = sum(2 * l + 1 for l in range(2, lm + 1))
                if nb * 1.5 > nc:
                    continue
                for wn, wf in (("uniform", None), ("triangular", w_triangular), ("inv_density", w_inv_density)):
                    v = spin2_fit_count(M, n, h, r_c, lm, wf)
                    rows[f"lmax{lm}_{wn}"] = {k: v[k] for k in ("n_basis", "fit_residual_rel", "weighted_residual_rel", "total_index", "n_zeros", "index_histogram", "n_clusters", "cluster_indices", "l2_nonaxial_amplitude_fraction", "l_power_fraction")}
                    rows[f"lmax{lm}_{wn}"]["clusters"] = v["clusters"]
            pz = prod[name][f"{r_c:g}"]
            pk = f"lmax{pz['lmax']}_uniform"
            same = rows.get(pk, {}).get("fit_residual_rel")
            c5[name][f"{r_c:g}"] = {"n_cells": nc, "producer_lmax": pz["lmax"], "producer_residual": pz["fit_residual_rel"], "our_residual_same_lmax_uniform": same,
                                    "residual_reproduced": bool(same is not None and abs(same - pz["fit_residual_rel"]) < 1e-3 * max(1.0, pz["fit_residual_rel"])), "fits": rows,
                                    "totals": sorted(set(v["total_index"] for v in rows.values())), "cluster_patterns": sorted(set(str(v["cluster_indices"]) for v in rows.values())),
                                    "l2_nonaxial_amplitude_fraction_range": [min(v["l2_nonaxial_amplitude_fraction"] for v in rows.values()), max(v["l2_nonaxial_amplitude_fraction"] for v in rows.values())] if rows else None,
                                    "n_zeros_range": [min(v["n_zeros"] for v in rows.values()), max(v["n_zeros"] for v in rows.values())] if rows else None,
                                    "residual_range": [min(v["fit_residual_rel"] for v in rows.values()), max(v["fit_residual_rel"] for v in rows.values())] if rows else None}
            log(f"  {name} r{r_c:g} ({nc} cells): producer resid {pz['fit_residual_rel']:.3f} (l{pz['lmax']}) ours same {same if same is None else round(same, 4)}; " +
                "; ".join(f"{k}: res {v['fit_residual_rel']:.3f} tot {v['total_index']} nz {v['n_zeros']} cl {v['cluster_indices']} nonax {v['l2_nonaxial_amplitude_fraction']:.3f}" for k, v in rows.items()))
    out["C5"]["scan"] = c5
    return out


# ================================================================ verdicts
def verdicts(c12, c3, c45):
    V = {}
    # C1
    c1 = c12["C1"]["analytic_hedgehog_n32_L48|axis_r12"]
    ok = c1["max_rel_diff_on_nonzero"] is not None and c1["max_rel_diff_on_nonzero"] < 1e-4 and c1["matches_0011112226_5pct"] and c1["n_zero_ours"] == 2 \
        and c1["kernel_residual_E00"] < 1e-8 and c1["kernel_residual_spatial_identity"] < 1e-8 and c1["top_overlap_with_axial_quadrupole_along_ray"] > 0.99
    V["C1_H00_spectrum"] = {"verdict": "CONFIRMED" if ok else "REFUTED", "numbers": {"nonzero_over_smallest_ours": c1["nonzero_over_smallest"], "maxdev_from_{1,1,1,1,2,2,2,6}": c1["maxdev_from_pattern"],
                            "max_rel_diff_ours_vs_producer_eigenvalues": c1["max_rel_diff_on_nonzero"], "max_abs_diff": c1["max_abs_diff_ours_vs_producer"], "kernel_residuals": [c1["kernel_residual_E00"], c1["kernel_residual_spatial_identity"]],
                            "top_overlap_with_axial_quadrupole": c1["top_overlap_with_axial_quadrupole_along_ray"], "G_is_identity_maxdev": c12["cells"]["analytic_hedgehog_n32_L48|axis_r12"]["G_is_identity_maxdev"],
                            "seed_max_abs_diff_ours_vs_producer": c12["seed_max_abs_diff_ours_vs_producer"], "cell_xyz": c12["cells"]["analytic_hedgehog_n32_L48|axis_r12"]["xyz"]},
                            "reason": ("our exact polarization-identity H_00 on our own hedgehog reproduces the producer's central-difference spectrum to the stated tolerance, with the kernel {E_00, spatial identity} and the ray quadrupole on top"
                                       if ok else "our exact H_00 disagrees with the producer's spectrum, kernel or top direction (see numbers)")}
    # C2
    anym = any(v["any_reading_matches_author"] for v in c12["C2"].values())
    matched = {k: v["readings_matching_author"] for k, v in c12["C2"].items() if v["any_reading_matches_author"]}
    nread = {k: len(v["readings"]) for k, v in c12["C2"].items()}
    ex = c12["C2_exact_continuum"]
    axis_match = ex["on_z_axis_r12"]["readings_matching_author"]
    conv = c12["C2_lattice_convergence_of_reading_ii"]
    refuted = bool(axis_match) or anym
    V["C2_no_basis_matches_author"] = {"verdict": "REFUTED" if refuted else "CONFIRMED",
                                       "numbers": {"readings_tried_per_lattice_cell": nread, "lattice_cell_readings_matching_author": matched,
                                                   "closest_reading_per_lattice_cell": {k: v["closest_reading"] for k, v in c12["C2"].items()},
                                                   "exact_continuum_on_z_axis": {"coordinate_basis": ex["on_z_axis_r12"]["coordinate_basis"], "author_maxdev": ex["on_z_axis_r12"]["coordinate_basis_author_maxdev"], "orthonormal": ex["on_z_axis_r12"]["orthonormal"], "matching_readings": axis_match},
                                                   "exact_continuum_at_producer_cell_position": {"angle_off_axis_deg": ex["producer_cell_position_(0.75,0.75,11.25)"]["angle_from_z_axis_deg"], "coordinate_basis": ex["producer_cell_position_(0.75,0.75,11.25)"]["coordinate_basis"], "author_maxdev": ex["producer_cell_position_(0.75,0.75,11.25)"]["coordinate_basis_author_maxdev"]},
                                                   "exact_continuum_generic_point": {"coordinate_basis": ex["generic_point_(3,-2,5)"]["coordinate_basis"], "author_maxdev": ex["generic_point_(3,-2,5)"]["coordinate_basis_author_maxdev"]},
                                                   "lattice_convergence_of_reading_ii": conv},
                                       "reason": ("the coordinate-basis reading (ii) (the form on the unnormalized directions E_ab + E_ba) reproduces the author's 1:1:1:1:1:2:2:3 EXACTLY on the z axis of the analytic hedgehog "
                                                  "(continuum H_00 from the analytic jets, maxdev 4e-16); the producer evaluated it at the lattice cell x = y = h/2, 5.4 degrees off the axis, where this basis-DEPENDENT reading is distorted "
                                                  "by 7 percent (3.4 percent at h 0.75, converging), and a 2 percent clustering tolerance reported 'None': the two H_00 differ by a basis convention only, not by a contraction"
                                                  if refuted else "no reading reproduces the author's pattern on the lattice cells nor on the exact axis")}
    # C3
    ok3 = c3["abs_diff"] < 2e-3 and all(abs(a - b) < 0.006 for a, b in zip(c3["claimed_cells"], c3["recomputed_cells"]))
    V["C3_section_177_units"] = {"verdict": "CONFIRMED" if ok3 else "REFUTED", "numbers": {"r_0_ours": c3["r_0_profile_ours"], "r_0_record": c3["r_0_profile_record"], "abs_diff": c3["abs_diff"], "cells_claimed": c3["claimed_cells"], "cells_recomputed": c3["recomputed_cells"],
                                 "r_0_central_line_raw": c3["r_0_central_line_raw_crossing"], "r_0_0.75h_shells": c3["r_0_profile_0.75h_shells"]},
                                 "reason": ("our own lambda_1 shell profile on r16_1_rebuild_n32_L48.npy crosses 0.8 at the recorded radius, and the cell conversions of the three record values are arithmetically right; the definition-sensitivity (central line, finer shells) is reported beside it"
                                            if ok3 else "our recomputed r_0 or the cell arithmetic disagrees with the record")}
    # C4
    F = c45["fields"]
    gate_ok = c45["gate"]["counter_A_all_pass"]
    trusted = {name: [r_ for r_, v in rec.items() if v["A_trustworthy (signal > 3 x artefact A and fit residual < 0.3)"]] for name, rec in F.items()}
    totals_ok = all(v["A"]["total_index"] == 4 and v["variants_all_total_4"] for name, rec in F.items() for r_, v in rec.items() if v["A_trustworthy (signal > 3 x artefact A and fit residual < 0.3)"])
    totals_agree = all(v["total_agrees_with_producer"] for rec in F.values() for v in rec.values())
    # the content: on the seeded fields two clusters of net index 2 at the poles; on the static two clusters of net 2 on the body diagonal
    content = {}
    for name, rec in F.items():
        content[name] = {r_: {"A_cluster_indices": v["A"]["cluster_indices"], "A_cluster_xyz": [c["xyz"] for c in v["A"]["clusters"]], "A_n_zeros": v["A"]["n_zeros"], "producer_n_zeros": v["producer"]["n_zeros"],
                              "producer_hist": v["producer"]["index_histogram"], "Q_eff_axis_dot_z": v["A"]["Q_eff_axis_dot_z"], "Q_eff_axis_dot_111": v["A"]["Q_eff_axis_dot_111"], "Q_eff_eigs": v["A"]["Q_eff_eigs"],
                              "B_total": v["B"]["total_index"], "B_n_zeros": v["B"]["n_zeros"], "A_fit_residual": round(v["A"]["fit_residual_rel"], 3), "A_trustworthy": v["A_trustworthy (signal > 3 x artefact A and fit residual < 0.3)"], "signal_over_artefact_B": round(v["signal_over_artefact_B"], 2),
                              "producer_zero_to_our_nearest_zero_deg": v["producer_zero_to_our_nearest_zero_deg"], "A_variants_cluster_patterns": v["variants_cluster_indices"]} for r_, v in rec.items()}
    seeded_pairs = all(v["A"]["cluster_indices"] == [2, 2] and all(c["dot_z"] > np.cos(np.radians(25)) for c in v["A"]["clusters"]) for name, rec in F.items() if "seeded" in name for v in rec.values() if v["A_trustworthy (signal > 3 x artefact A and fit residual < 0.3)"])
    static_pairs = all(v["A"]["cluster_indices"] == [2, 2] and all(c["dot_111"] > np.cos(np.radians(25)) for c in v["A"]["clusters"]) for name, rec in F.items() if "static" in name for v in rec.values() if v["A_trustworthy (signal > 3 x artefact A and fit residual < 0.3)"])
    B_trust = {name: [r_ for r_, v in rec.items() if v["B_trustworthy (signal > 3 x artefact B)"]] for name, rec in F.items()}
    verdict4 = "CONFIRMED" if (gate_ok and totals_ok and totals_agree) else ("QUALIFIED" if gate_ok and totals_ok else "REFUTED")
    if verdict4 == "CONFIRMED":
        verdict4 = "QUALIFIED"       # the total is confirmed; the per-zero content is a resolution of degenerate index-2 zeros (see reason)
    V["C4_poincare_hopf_lattice"] = {"verdict": verdict4, "numbers": {"counter_A_gate_pass": gate_ok, "shells_where_counter_A_is_trustworthy": trusted, "shells_where_counter_B_is_trustworthy": B_trust,
                                     "artefact_B_trilinear_exact_hedgehog_split_max": {k: v["split_max"] for k, v in c45["artefact"]["B_trilinear_on_exact_hedgehog"].items()},
                                     "artefact_A_profiled_hedgehog_split_max": {k: v["transverse_fit_profiled_hedgehog_split_max_grid"] for k, v in c45["artefact"]["A_fit_on_split_free_hedgehogs"].items()},
                                     "totals_4_on_trusted_shells_all_variants": totals_ok, "totals_agree_with_producer_everywhere": totals_agree, "seeded_fields_two_clusters_of_index_2_on_pm_z": seeded_pairs, "static_field_two_clusters_of_index_2_on_pm_111": static_pairs,
                                     "hazard": "total index 4 is a THEOREM for any generic section of the spin-2 bundle over the outward-oriented director's transverse plane (Chern number 4): the check can only fail through a counter defect (as the producer's hull count does, totals 0 / 2 / 3), so SPIN2_INDEX_4_CONFIRMED validates the instrument, not the field; the trilinear counter B returns 4 with garbage zeros on every shell",
                                     "content": content},
                                     "reason": ("total index 4 is reproduced by an independent chart-free counter on every shell and every fit variant (but it is a theorem for any generic section: an instrument check, not a field property); "
                                                "the producer's per-zero content is the resolution of two DEGENERATE index-2 zeros: on the seeded fields the 4 simple zeros are two pairs straddling +-z at 8 to 10 degrees, produced by a "
                                                "3 to 5 percent (amplitude) m = +-2 admixture on the x = y diagonal that every fit agrees on (the producer's 'pure m = 0' is the 2-decimal rounding of a 1e-3 power fraction); on the split-free static the "
                                                "section is an m = 0 pattern about the body diagonal (Q_eff axis . (1,1,1) = 1.000 on every shell) whose two net-index-2 clusters resolve into 8, 4 or 2 zeros depending on the fit; the net cluster indices "
                                                "(2, 2) and their axes are the measurement, the individual zero count is not; the trilinear counter B is untrustworthy on every shell of these fields (artefact 1.1e-2 to 1.4e-1 > signal)")}
    # C5
    S5 = c45["C5"]["scan"]
    repro = all(v["residual_reproduced"] for rec in S5.values() for v in rec.values())
    stable_tot = all(v["totals"] == [4] for rec in S5.values() for v in rec.values())
    stable_cl = {name: {r_: v["cluster_patterns"] for r_, v in rec.items()} for name, rec in S5.items()}
    resid_ranges = {name: {r_: v["residual_range"] for r_, v in rec.items()} for name, rec in S5.items()}
    nz_ranges = {name: {r_: v["n_zeros_range"] for r_, v in rec.items()} for name, rec in S5.items()}
    ok5 = repro and stable_tot
    seeded_stable = all(v["n_zeros_range"] is not None and v["n_zeros_range"][0] == v["n_zeros_range"][1] for name, rec in S5.items() if "seeded" in name for v in rec.values())
    static_moves = any(v["n_zeros_range"] is not None and v["n_zeros_range"][0] != v["n_zeros_range"][1] for name, rec in S5.items() if "static" in name for v in rec.values())
    V["C5_fit_residual_flag"] = {"verdict": "QUALIFIED" if (ok5 and static_moves) else ("CONFIRMED" if ok5 else "REFUTED"), "numbers": {"producer_residual_reproduced_by_our_basis": repro, "basis_validation": c45["C5"]["basis_validation"], "total_index_always_4": stable_tot,
                                 "seeded_fields_n_zeros_stable_under_lmax_and_weighting": seeded_stable, "static_field_n_zeros_moves_under_lmax_and_weighting": static_moves,
                                 "cluster_patterns_by_variant": stable_cl, "n_zeros_range_by_shell": nz_ranges, "residual_range_by_shell": resid_ranges,
                                 "l2_nonaxial_amplitude_fraction_range_by_shell": {name: {r_: v["l2_nonaxial_amplitude_fraction_range"] for r_, v in rec.items()} for name, rec in S5.items()}},
                                 "reason": ("our Wigner-d spin-2 basis reproduces the producer's residuals to 1e-4; under l_max 2 / 4 / 6 / 8 and three weightings the seeded fields' residuals (0.01 to 0.2 on r <= 4.5) and their four zeros are stable, "
                                            "so the flag holds where the producer applied it; but a LOW residual is not sufficient: the static field at residual 0.012 to 0.15 keeps its (2, 2) clusters while its individual zero count moves between 2, 4 and 8 with the fit, "
                                            "so the residual certifies the section, not the count of simple zeros"
                                            if (ok5 and static_moves) else ("the residual flag holds: residuals reproduced, counts stable" if ok5 else "the residual or the total index is not stable under the l_max / weighting change (see numbers)"))}
    return V


if __name__ == "__main__":
    c12 = audit_C1_C2()
    c3 = audit_C3()
    c45 = audit_C4_C5()
    V = verdicts(c12, c3, c45)
    out = {"rung": "R18-0 audit", "verdicts": V, "C1_C2_detail": c12, "C3_detail": c3, "C4_C5_detail": c45, "wall_s": time.time() - T0}
    json.dump(plain(out), open(os.path.join(DATA, "m5_32_r18_0_audit.json"), "w"), indent=1)
    print("\n| claim | verdict | key numbers |")
    print("| --- | --- | --- |")
    for k, v in V.items():
        nums = v["numbers"]
        if k.startswith("C1"):
            s = f"ratios {nums['nonzero_over_smallest_ours']}, rel diff vs producer {nums['max_rel_diff_ours_vs_producer_eigenvalues']:.1e}, kernel res {nums['kernel_residuals'][0]:.1e}/{nums['kernel_residuals'][1]:.1e}, top overlap {nums['top_overlap_with_axial_quadrupole']:.4f}"
        elif k.startswith("C2"):
            s = f"exact z-axis coordinate basis {nums['exact_continuum_on_z_axis']['coordinate_basis']} (maxdev {nums['exact_continuum_on_z_axis']['author_maxdev']:.1e}); producer cell {nums['exact_continuum_at_producer_cell_position']['angle_off_axis_deg']:.1f} deg off axis: maxdev {nums['exact_continuum_at_producer_cell_position']['author_maxdev']:.3f}; lattice h1.5 / h0.75 maxdev {[round(v['author_maxdev'], 3) for v in nums['lattice_convergence_of_reading_ii'].values()]}"
        elif k.startswith("C3"):
            s = f"r_0 ours {nums['r_0_ours']:.4f} vs {nums['r_0_record']:.4f}; cells {nums['cells_recomputed']}; central line {nums['r_0_central_line_raw']:.2f}; 0.75h shells {nums['r_0_0.75h_shells']:.2f}"
        elif k.startswith("C4"):
            s = f"gate {nums['counter_A_gate_pass']}; trusted shells A {nums['shells_where_counter_A_is_trustworthy']}; B {nums['shells_where_counter_B_is_trustworthy']}; totals 4 {nums['totals_4_on_trusted_shells_all_variants']}; seeded (2,2) on z {nums['seeded_fields_two_clusters_of_index_2_on_pm_z']}; static (2,2) on 111 {nums['static_field_two_clusters_of_index_2_on_pm_111']}"
        else:
            s = f"residuals reproduced {nums['producer_residual_reproduced_by_our_basis']}; total always 4 {nums['total_index_always_4']}; n_zeros ranges {nums['n_zeros_range_by_shell']}"
        print(f"| {k} | {v['verdict']} | {s} |")
    print(f"\nwall {out['wall_s']:.0f} s; written data/m5_32_r18_0_audit.json")
