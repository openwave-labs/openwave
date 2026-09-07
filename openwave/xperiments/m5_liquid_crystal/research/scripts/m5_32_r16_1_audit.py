#!/usr/bin/env python3
"""M5.32 R16-1 ADVERSARIAL AUDIT: the statics of the object v4 (C1.1-C1.7).

Independent re-implementation from the brief's definitions only. Nothing
is imported from m5_32_r16_*.py or m5_32_r15_common.py; the certified
stencil layer (d1, d1_adj, branches, coords, pin_shell) is imported from
m5_21_3_a_4d.py, and the registry m5_32_terms_ext.py is read ONLY for the
I1_h cross-read of claim C1.6.

EQUATIONS FIRST (the brief's conventions, as implemented here)
--------------------------------------------------------------
  N = M eta, eta = diag(-1, 1, 1, 1); per cell eigh(N) (N is symmetric on
  every field here because M_0mu = 0, asserted at load).
  u = eigenvector of the most negative eigenvalue, u^T eta u = -1;
  G = eta + 2 (eta u)(eta u)^T (verified == I on every field).
  spatial triple = the other three eigenvalues, lambda_1 >= lambda_2 >=
  lambda_3; half = (lambda_2 - lambda_3)/2; rho^2 = half^2;
  beta^2 = 1 - 6 (tr Q^3)^2 / (tr Q^2)^3, Q the traceless triple.
  n = eigenvector of lambda_1, oriented outward (n . r_hat > 0);
  (e, f) = the pair-plane eigenvectors with det[u, n, e, f] > 0 enforced;
  J = f e^T - e f^T == [n]_x (the cross-product matrix, verified).
  R(beta) = I + sin(beta) J + (1 - cos beta) J^2 (Rodrigues about n);
  T_alpha M = R(alpha/2) M R(alpha/2)^T; the circle average
  E_v4 = (1/n_s) sum_k E[T_(2 pi k/n_s) M], the transformed field's own
  lattice derivatives, n_s = 8 (4 in the descent).
  stencil "sym" = the certified semantic: energies averaged over the
  fwd and bwd one-sided branches, weight 1/2 each (m5_21_3_a_4d.a_fields /
  e_parts), h^3-weighted sums over ALL cells.
  A_i = d_i M;  F_ij = A_i G A_j - A_j G A_i;
  E_h  = 4 sum_{i<j} tr(G F_ij G F_ij^T)            (I_rebuild)
  E_h' = 4 sum_{i<j} tr(eta F'_ij eta F'_ij^T), F' with eta   (I_norm)
  V4   = W1 sum_{p=1..4} (tr N^p - C_p)^2, C_p = (-8)^p + 1 + 2 (0.3)^p
  U    = mu rho^2, mu = 1e-2
  K_P  = (1/2) sum_i tr(Om_i^T eta Om_i eta), Om_i = w(N) A_i eta w(N),
         w = the plateau weight (1 on [-0.2, 0.8], cosine tapers to 0 at
         1 and at -1), c_P = 1
  reg  = c_s rho^2 E2, E2 = sum_i tr(A_i G A_i G), c_s = 0.4
  E_stat = E_h + V4 + U + K_P + reg.

THE GRADIENT (claim C1.2) is the hand-derived reverse-mode gradient of the
circle-averaged E_stat with respect to M, INCLUDING the dependence of the
rotation R on the director n(M) (eigenvector perturbation) and the
Daleckii-Krein adjoint of the spectral function w(N); it is certified
against central finite differences of THIS script's energy at random
cells and entries before it is used in the 100-iteration FIRE descent.

Usage: OMP_NUM_THREADS=2 python3 m5_32_r16_1_audit.py [--quick]
Out:   ../data/m5_32_r16_1_audit.json (relative paths only)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.normpath(os.path.join(HERE, ".."))
DATA = os.path.join(RESEARCH, "data")
OUT_JSON = os.path.join(DATA, "m5_32_r16_1_audit.json")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


STACK = _load("m5_21_3_a_4d", "m5_21_3_a_4d.py")     # certified stencil layer
d1, d1_adj, branches = STACK.d1, STACK.d1_adj, STACK.branches
coords, pin_shell = STACK.coords, STACK.pin_shell

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
ETA_D = np.diag(ETA)
G_T, DELTA = 8.0, 0.3
W1 = 0.000724023879
MU, C_P, C_S = 1e-2, 1.0, 0.4
C_POW = [(-G_T) ** p + 1.0 + 2.0 * DELTA ** p for p in range(1, 5)]
VAC_TRIPLE = np.array([1.0, DELTA, DELTA])
NS_READ, NS_DESCENT = 8, 4

FIELDS = {
    "seed_n32": ("checkpoints/m5_32_r15/m_hedgehog/relax_n32_L48_mu0.01_cP1.npy", 32, 48.0),
    "seed_n48": ("checkpoints/m5_32_r15/m_hedgehog/relax_n48_L72_mu0.01_cP1.npy", 48, 72.0),
    "end_n32": ("checkpoints/m5_32_r16/r16_1_rebuild_n32_L48.npy", 32, 48.0),
    "end_n48": ("checkpoints/m5_32_r16/r16_1_rebuild_n48_L72.npy", 48, 72.0),
    "end_n64": ("checkpoints/m5_32_r16/r16_1_rebuild_n64_L48_analytic.npy", 64, 48.0),
    "vac_n32": ("checkpoints/m5_32_r16/vac_n32_L48.npy", 32, 48.0),
}

# the producer's numbers (the brief), for the side-by-side
PROD = {
    "seed_n32": {"E_h": 5.36427, "V4": 0.15468, "U": 5.869e-5, "K_P": 14.5230,
                 "reg": 1.238e-4, "E_stat": 20.0421},
    "end_n32": {"E_h": 5.43151, "V4": 0.08726, "U": 4.566e-6, "K_P": 8.29773,
                "reg": 6.27e-6, "E_stat": 13.81651},
    "seed_n48": {"E_stat": 20.4541},
    "end_n48": {"E_h": 6.11231, "V4": 0.09698, "U": 1.098e-5, "K_P": 8.67449,
                "E_stat": 14.88381},
    "end_n64": {"E_h": 9.72347, "V4": 0.01617, "K_P": 3.91381, "E_stat": 13.65345},
}
PROD_TEX = {
    "end_n32": {"beta2_max": 4.2e-4, "half_max": 5.6e-4, "center": (0.569, 0.519, 0.518),
                "lam1_min": 0.569, "gap": 0.050, "r_d": 3.27, "n_d": 56,
                "ext_max_dev": 5.2e-4, "tail_exp": -4.03},
    "end_n48": {"beta2_max": 3.5e-3, "half_max": 1.2e-3, "center": (0.542, 0.489, 0.487),
                "lam1_min": 0.542, "gap": 0.054, "r_d": 3.27, "n_d": 56,
                "ext_max_dev": 8e-5, "tail_exp": -4.17},
    "end_n64": {"beta2_max": 2.5e-8, "half_max": 8.4e-6, "center": (0.716, 0.608, 0.608),
                "lam1_min": 0.716, "gap": 0.107, "r_d": 0.65, "n_d": 8,
                "ext_max_dev": 6e-5, "tail_exp": -4.14},
    "seed_n32": {"beta2_max": 0.171, "half_max": 6.7e-3, "center": (0.491, 0.424, 0.412)},
}
PROD_GATE = {"end_n32": (4.8e-13, 1.4e-6, 1e-15), "end_n48": (6e-15, 1.3e-6, 4e-14),
             "end_n64": (2e-14, 9.9e-6, 5e-15)}


def T(X):
    return X.swapaxes(-1, -2)


def tr(X):
    return np.einsum("...kk->...", X)


def frob(X, Y):
    return np.einsum("...ab,...ab->...", X, Y)


# ================= plateau weight =================
def w_plateau(lam):
    lam = np.asarray(lam, dtype=float)
    w = np.zeros_like(lam)
    w[np.abs(lam - 0.3) <= 0.5] = 1.0
    hi = (lam > 0.8) & (lam < 1.0)
    w[hi] = 0.5 * (1.0 + np.cos(np.pi * (lam[hi] - 0.8) / 0.2))
    lo = (lam > -1.0) & (lam < -0.2)
    w[lo] = 0.5 * (1.0 + np.cos(np.pi * (-0.2 - lam[lo]) / 0.8))
    return w


def w_plateau_prime(lam):
    lam = np.asarray(lam, dtype=float)
    d = np.zeros_like(lam)
    hi = (lam > 0.8) & (lam < 1.0)
    d[hi] = -(np.pi / 0.4) * np.sin(np.pi * (lam[hi] - 0.8) / 0.2)
    lo = (lam > -1.0) & (lam < -0.2)
    d[lo] = (np.pi / 1.6) * np.sin(np.pi * (-0.2 - lam[lo]) / 0.8)
    return d


def spectral_W(N):
    """w(N) = V w(Lambda) V^T for symmetric N; returns (W, lam, V)."""
    lam, V = np.linalg.eigh(N)
    W = np.einsum("...ak,...k,...bk->...ab", V, w_plateau(lam), V)
    return W, lam, V


def dk_gamma(lam):
    """Daleckii-Krein divided differences of w on the spectrum lam (..., 4)."""
    wl = w_plateau(lam)
    li, lj = lam[..., :, None], lam[..., None, :]
    num = wl[..., :, None] - wl[..., None, :]
    den = li - lj
    close = np.abs(den) < 1e-9 * np.maximum(np.abs(li), 1.0)
    gam = np.where(close, w_plateau_prime(0.5 * (li + lj)),
                   num / np.where(close, 1.0, den))
    return gam


# ================= per-cell reads =================
def cell_reads(M, X, Y, Z):
    """u, G, the spatial triple, the oriented director n, the pair plane
    (e, f) with det[u,n,e,f] > 0, J = f e^T - e f^T, K = [n]_x, beta^2."""
    N = M @ ETA
    asym = float(np.abs(N - T(N)).max())
    assert asym < 1e-12, f"N not symmetric ({asym:.2e}); the evaluator assumes M_0mu = 0"
    lam, V = np.linalg.eigh(N)                       # ascending
    u = V[..., :, 0]
    uu = np.einsum("...a,a,...a->...", u, ETA_D, u)
    assert np.all(uu < 0), "the most negative eigenvector is not timelike"
    u = u / np.sqrt(-uu)[..., None]
    eu = u * ETA_D
    G = ETA + 2.0 * eu[..., :, None] * eu[..., None, :]
    triple = lam[..., 1:][..., ::-1]                 # (lam1, lam2, lam3)
    n = V[..., :, 3]
    nn = np.einsum("...a,a,...a->...", n, ETA_D, n)
    n = n / np.sqrt(nn)[..., None]
    r = np.sqrt(X * X + Y * Y + Z * Z)
    ndotr = (n[..., 1] * X + n[..., 2] * Y + n[..., 3] * Z) / r
    n = n * np.sign(ndotr)[..., None]
    ndotr = np.abs(ndotr)
    e, f = V[..., :, 2], V[..., :, 1]
    det = np.linalg.det(np.stack([u, n, e, f], axis=-1))
    f = f * np.sign(det)[..., None]
    det2 = np.linalg.det(np.stack([u, n, e, f], axis=-1))
    J = f[..., :, None] * e[..., None, :] - e[..., :, None] * f[..., None, :]
    K = cross_matrix(n)
    half = 0.5 * (triple[..., 1] - triple[..., 2])
    q = triple - triple.mean(axis=-1, keepdims=True)
    q2, q3 = (q ** 2).sum(-1), (q ** 3).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta2 = np.where(q2 > 1e-24, 1.0 - 6.0 * q3 ** 2 / q2 ** 3, np.nan)
    return {"N": N, "lam": lam, "V": V, "u": u, "G": G, "triple": triple,
            "n": n, "ndotr": ndotr, "J": J, "K": K, "det": det2, "half": half,
            "beta2": beta2, "r": r, "e": e, "f": f}


def cross_matrix(n):
    """[n]_x embedded in 4x4 (spatial block), n (..., 4) with n_0 ignored."""
    K = np.zeros(n.shape + (4,))
    n1, n2, n3 = n[..., 1], n[..., 2], n[..., 3]
    K[..., 1, 2], K[..., 1, 3] = -n3, n2
    K[..., 2, 1], K[..., 2, 3] = n3, -n1
    K[..., 3, 1], K[..., 3, 2] = -n2, n1
    return K


def rodrigues(K, beta):
    I4 = np.broadcast_to(np.eye(4), K.shape)
    return I4 + np.sin(beta) * K + (1.0 - np.cos(beta)) * (K @ K)


# ================= energy =================
def v4_u_density(N, half):
    P = N.copy()
    t = [tr(P)]
    for _ in range(3):
        P = P @ N
        t.append(tr(P))
    vd = sum((t[p] - C_POW[p]) ** 2 for p in range(4))
    return W1 * vd, MU * half ** 2


def energy(M, h, X, Y, Z, ns=NS_READ, K=None, stencil="sym", registry=None,
           per_cell=False, half_override=None):
    """The circle-averaged parts (dict of h^3-weighted totals); per_cell
    returns the per-cell E_stat density (sum of all parts) too.
    K: the generator [n]_x to rotate about (default: the field's own lifted
    director). registry: the I1_h_np callable for the C1.6 cross-read."""
    rd = cell_reads(M, X, Y, Z)
    if K is None:
        K = rd["K"]
    G = rd["G"]
    h3 = h ** 3
    half = rd["half"] if half_override is None else half_override
    v4d, ud = v4_u_density(rd["N"], half)
    acc = {k: 0.0 for k in ("E_h", "E_h_norm", "K_P", "reg", "E2", "I1h_reg")}
    dens = np.zeros(M.shape[:3])
    dens_h = np.zeros(M.shape[:3])
    per_sample = []
    for k in range(ns):
        beta = np.pi * k / ns                       # alpha_k / 2
        R = rodrigues(K, beta)
        Mk = R @ M @ T(R)
        Nk = Mk @ ETA
        Wk = spectral_W(Nk)[0]
        s_eh = 0.0
        for br, wt in branches(stencil):
            A = [d1(Mk, ax, h, br) for ax in range(3)]
            eh = 0.0
            ehn = 0.0
            for i in range(3):
                for j in range(i + 1, 3):
                    F = A[i] @ G @ A[j] - A[j] @ G @ A[i]
                    eh = eh + 4.0 * frob(G @ F @ G, F)
                    Fn = A[i] @ ETA @ A[j] - A[j] @ ETA @ A[i]
                    ehn = ehn + 4.0 * frob(ETA @ Fn @ ETA, Fn)
            kp = 0.0
            e2 = 0.0
            for i in range(3):
                Om = Wk @ A[i] @ ETA @ Wk
                kp = kp + 0.5 * np.einsum("a,b,...ab,...ab->...", ETA_D, ETA_D, Om, Om)
                e2 = e2 + frob(T(A[i] @ G), A[i] @ G)   # tr(A G A G)
            rg = C_S * half ** 2 * e2
            acc["E_h"] += wt * h3 * eh.sum()
            acc["E_h_norm"] += wt * h3 * ehn.sum()
            acc["K_P"] += wt * h3 * kp.sum()
            acc["reg"] += wt * h3 * rg.sum()
            acc["E2"] += wt * h3 * e2.sum()
            dens += wt * (eh + C_P * kp + rg)
            dens_h += wt * eh
            s_eh += wt * h3 * eh.sum()
            if registry is not None:
                A4 = np.stack([np.zeros_like(A[0])] + A, axis=0)
                acc["I1h_reg"] += wt * h3 * 4.0 * registry(A4, Mk, None).sum()
        per_sample.append(float(s_eh))
    out = {k: float(v / ns) for k, v in acc.items()}
    out["V4"] = float(h3 * v4d.sum())
    out["U"] = float(h3 * ud.sum())
    out["E_stat"] = out["E_h"] + out["V4"] + out["U"] + C_P * out["K_P"] + out["reg"]
    out["E_h_per_sample"] = per_sample
    if per_cell:
        out["dens"] = h3 * (dens / ns + v4d + ud)
        out["dens_h"] = h3 * dens_h / ns
    return out


# ================= the gradient (hand-derived reverse mode) =================
def gradient(M, h, X, Y, Z, free, ns=NS_DESCENT, with_dR=True, stencil="sym",
             parts=("Eh", "KP", "reg", "U", "V4")):
    """dE_stat/dM (symmetrized, masked to the free cells) of the
    ns-sample circle average; G = I assumed (asserted in cell_reads)."""
    rd = cell_reads(M, X, Y, Z)
    assert np.abs(rd["G"] - np.eye(4)).max() < 1e-12
    n, K, half = rd["n"], rd["K"], rd["half"]
    lam, V = rd["lam"], rd["V"]
    N = rd["N"]
    h3 = h ** 3
    GM = np.zeros_like(M)
    # ---- V4 (sample invariant)
    pows = [np.broadcast_to(np.eye(4), M.shape)]
    for _ in range(3):
        pows.append(pows[-1] @ N)
    GN = np.zeros_like(M)
    for p in range(1, 5):
        tp = tr(pows[p - 1] @ N)
        if "V4" in parts:
            GN += (2.0 * W1 * (tp - C_POW[p - 1]) * p)[..., None, None] * T(pows[p - 1])
    # ---- half^2 adjoint: d half^2 = half (v2 v2^T - v3 v3^T) : dN
    v2, v3 = V[..., :, 2], V[..., :, 1]
    P23 = v2[..., :, None] * v2[..., None, :] - v3[..., :, None] * v3[..., None, :]
    g_h2 = (MU if "U" in parts else 0.0) * np.ones(M.shape[:3])   # U = mu half^2
    E2_acc = np.zeros(M.shape[:3])
    G_n = np.zeros(M.shape[:3] + (4,))
    for k in range(ns):
        beta = np.pi * k / ns
        s, c = np.sin(beta), np.cos(beta)
        R = rodrigues(K, beta)
        Mk = R @ M @ T(R)
        Nk = Mk @ ETA
        Wk, lamk, Vk = spectral_W(Nk)
        Gk = np.zeros_like(M)
        GW = np.zeros_like(M)
        for br, wt in branches(stencil):
            A = [d1(Mk, ax, h, br) for ax in range(3)]
            dA = [np.zeros_like(M) for _ in range(3)]
            for i in range(3):
                for j in range(i + 1, 3):
                    F = A[i] @ A[j] - A[j] @ A[i]
                    if "Eh" in parts:
                        dA[i] += 8.0 * (F @ T(A[j]) - T(A[j]) @ F)
                        dA[j] += 8.0 * (T(A[i]) @ F - F @ T(A[i]))
            e2 = 0.0
            for i in range(3):
                Om = Wk @ A[i] @ ETA @ Wk
                Yi = ETA @ Om @ ETA
                if "KP" in parts:
                    dA[i] += C_P * (Wk @ Yi @ Wk @ ETA)
                    GW += C_P * wt * (Yi @ Wk @ ETA @ T(A[i]) + ETA @ T(A[i]) @ Wk @ Yi)
                if "reg" in parts:
                    dA[i] += C_S * (half ** 2)[..., None, None] * 2.0 * T(A[i])
                e2 = e2 + frob(A[i], A[i])
            E2_acc += wt * e2
            for ax in range(3):
                Gk += wt * d1_adj(dA[ax], ax, h, br)
        # W adjoint (Daleckii-Krein)
        GWs = 0.5 * (GW + T(GW))
        inner = np.einsum("...ak,...ab,...bl->...kl", Vk, GWs, Vk) * dk_gamma(lamk)
        GNk = np.einsum("...ak,...kl,...bl->...ab", Vk, inner, Vk)
        Gk += GNk @ ETA
        # Mk = R M R^T
        GM += T(R) @ Gk @ R
        if with_dR:
            GR = (Gk + T(Gk)) @ R @ M
            GK = s * GR + (1.0 - c) * (GR @ T(K) + T(K) @ GR)
            gn = np.zeros(M.shape[:3] + (4,))
            gn[..., 1] = GK[..., 3, 2] - GK[..., 2, 3]
            gn[..., 2] = GK[..., 1, 3] - GK[..., 3, 1]
            gn[..., 3] = GK[..., 2, 1] - GK[..., 1, 2]
            G_n += gn
    GM /= ns
    E2_acc /= ns
    G_n /= ns
    if "reg" in parts:
        g_h2 += C_S * E2_acc                      # reg = c_s half^2 E2
    GN += (g_h2 * half)[..., None, None] * P23
    if with_dR:
        # n = top eigenvector of N: dn = sum_{k != top} v_k (v_k^T dN n)/(lam_n - lam_k)
        for kk in range(3):
            vk = V[..., :, kk]
            ck = np.einsum("...a,...a->...", G_n, vk) / (lam[..., 3] - lam[..., kk])
            GN += ck[..., None, None] * vk[..., :, None] * n[..., None, :]
    GM += GN @ ETA
    GM *= h3
    GM = 0.5 * (GM + T(GM))
    return GM * free[..., None, None]


def fd_check(M, h, X, Y, Z, free, ns, rng, n_pts=12, eps=1e-5):
    """central FD of energy() vs the analytic gradient at random free
    cells / symmetric entries (+ the M_0i null directions)."""
    Gfull = gradient(M, h, X, Y, Z, free, ns=ns, with_dR=True)
    GnoR = gradient(M, h, X, Y, Z, free, ns=ns, with_dR=False)
    E0 = energy(M, h, X, Y, Z, ns=ns)["E_stat"]
    idx = np.argwhere(free)
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    # random free cells + the innermost cell + a taper cell if any
    picks = [tuple(idx[i]) for i in rng.choice(len(idx), n_pts - 2, replace=False)]
    picks.append(tuple(np.unravel_index(np.argmin(r), r.shape)))
    trip = cell_reads(M, X, Y, Z)["triple"][..., 0]
    tap = np.argwhere((trip > 0.8) & (trip < 1.0) & free)
    if len(tap):
        picks.append(tuple(tap[rng.integers(len(tap))]))
    rows = []
    for cell in picks:
        a, b = rng.integers(1, 4), rng.integers(1, 4)
        if rng.random() < 0.25:
            a = b = 0
        D = np.zeros((4, 4))
        D[a, b] += 1.0
        if a != b:
            D[b, a] += 1.0
        ep = eps if a != 0 else 0.1 * eps      # V4 is stiff in M_00: O(eps^2) FD error
        Mp, Mm = M.copy(), M.copy()
        Mp[cell] += ep * D
        Mm[cell] -= ep * D
        fd = (energy(Mp, h, X, Y, Z, ns=ns)["E_stat"]
              - energy(Mm, h, X, Y, Z, ns=ns)["E_stat"]) / (2 * ep)
        an = float(np.sum(Gfull[cell] * D))
        an0 = float(np.sum(GnoR[cell] * D))
        rows.append({"cell": [int(c) for c in cell], "entry": [int(a), int(b)],
                     "r": float(r[cell]), "lam1": float(trip[cell]),
                     "fd": fd, "analytic": an, "analytic_no_dR": an0,
                     "rel_err": abs(fd - an) / max(abs(fd), 1e-9),
                     "rel_err_no_dR": abs(fd - an0) / max(abs(fd), 1e-9)})
    # the M_0i directions must be null (time-reflection even)
    null = []
    for cell in picks[:3]:
        for i in (1, 2, 3):
            D = np.zeros((4, 4))
            D[0, i] = D[i, 0] = 1.0
            Mp, Mm = M.copy(), M.copy()
            Mp[cell] += eps * D
            Mm[cell] -= eps * D
            # cell_reads asserts symmetric N; the M_0i perturbation breaks
            # it, so evaluate with the general-eig fallback energy instead
            null.append((energy_general(Mp, h, X, Y, Z, ns=ns)
                         - energy_general(Mm, h, X, Y, Z, ns=ns)) / (2 * eps))
    return {"E0": E0, "rows": rows, "max_rel_err": max(r_["rel_err"] for r_ in rows),
            "max_rel_err_no_dR": max(r_["rel_err_no_dR"] for r_ in rows),
            "M0i_directional_derivatives": [float(x) for x in null],
            "grad_norm_free": float(np.sqrt((Gfull ** 2).sum())),
            "grad_fmax_free": float(np.abs(Gfull).max()),
            "grad_norm_free_no_dR": float(np.sqrt((GnoR ** 2).sum())),
            "grad_fmax_free_no_dR": float(np.abs(GnoR).max())}


def energy_general(M, h, X, Y, Z, ns):
    """E_stat for a field whose N = M eta is non-symmetric at a few cells
    (used only for the M_0i null-direction FD): the symmetric cells go
    through cell_reads / spectral_W, the non-symmetric cells through
    np.linalg.eig with the left/right eigenvector projectors."""
    N = M @ ETA
    nonsym = np.abs(N - T(N)).max(axis=(-1, -2)) > 1e-14
    Msym = M.copy()
    Msym[nonsym, 0, :] = 0.0
    Msym[nonsym, :, 0] = 0.0
    Msym[nonsym, 0, 0] = M[nonsym, 0, 0]
    rd = cell_reads(Msym, X, Y, Z)
    G, n, half = rd["G"].copy(), rd["n"].copy(), rd["half"].copy()
    r = np.sqrt(X * X + Y * Y + Z * Z)

    def gen_eig(Nc):
        lam, V = np.linalg.eig(Nc)
        assert np.abs(lam.imag).max() < 1e-9
        lam, V = lam.real, V.real
        return lam, V, np.linalg.inv(V)

    for cell in map(tuple, np.argwhere(nonsym)):
        lam, V, Vi = gen_eig(N[cell])
        k0, k1 = np.argmin(lam), np.argmax(lam)
        u = V[:, k0]
        u = u / np.sqrt(-(u * ETA_D) @ u)
        eu = u * ETA_D
        G[cell] = ETA + 2.0 * np.outer(eu, eu)
        nv = V[:, k1]
        nv = nv / np.sqrt((nv * ETA_D) @ nv)
        rv = np.array([X[cell], Y[cell], Z[cell]]) / r[cell]
        nv = nv * np.sign(nv[1:] @ rv)
        n[cell] = nv
        ls = np.sort(lam)
        half[cell] = 0.5 * (ls[2] - ls[1])
    K = cross_matrix(n)
    h3 = h ** 3
    v4d, ud = v4_u_density(N, half)
    tot = 0.0
    for k in range(ns):
        R = rodrigues(K, np.pi * k / ns)
        Mk = R @ M @ T(R)
        Nk = Mk @ ETA
        Nk_sym = 0.5 * (Nk + T(Nk))
        Wk = spectral_W(Nk_sym)[0]
        for cell in map(tuple, np.argwhere(nonsym)):
            lam, V, Vi = gen_eig(Nk[cell])
            Wk[cell] = (V * w_plateau(lam)) @ Vi
        for br, wt in branches("sym"):
            A = [d1(Mk, ax, h, br) for ax in range(3)]
            eh = 0.0
            for i in range(3):
                for j in range(i + 1, 3):
                    F = A[i] @ G @ A[j] - A[j] @ G @ A[i]
                    eh = eh + 4.0 * frob(G @ F @ G, F)
            kp, e2 = 0.0, 0.0
            for i in range(3):
                Om = Wk @ A[i] @ ETA @ Wk
                kp = kp + 0.5 * np.einsum("a,b,...ab,...ab->...", ETA_D, ETA_D, Om, Om)
                e2 = e2 + frob(T(A[i] @ G), A[i] @ G)
            tot += wt * h3 * (eh + C_P * kp + C_S * half ** 2 * e2).sum()
    return float(tot / ns + h3 * (v4d + ud).sum())


def fire(M0, h, X, Y, Z, free, max_iter, ns=NS_DESCENT, dt0=0.02, dt_max=0.2,
         log_every=10, tag=""):
    """My own FIRE (same parameters as the certified stack's)."""
    M = M0.copy()
    v = np.zeros_like(M)
    dt, alpha, n_up = dt0, 0.1, 0
    F = -gradient(M, h, X, Y, Z, free, ns=ns)
    hist = [{"it": 0, "E": energy(M, h, X, Y, Z, ns=ns)["E_stat"],
             "fmax": float(np.abs(F).max()), "dt": dt}]
    t0 = time.time()
    for it in range(1, max_iter + 1):
        P = float(np.sum(F * v))
        if P > 0.0:
            n_up += 1
            vn = np.sqrt(np.sum(v * v))
            fn = np.sqrt(np.sum(F * F))
            v = (1 - alpha) * v + alpha * (F / max(fn, 1e-300)) * vn
            if n_up > 5:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
        else:
            v[:] = 0.0
            dt *= 0.5
            alpha = 0.1
            n_up = 0
        v += dt * F
        M += dt * v
        F = -gradient(M, h, X, Y, Z, free, ns=ns)
        if it % log_every == 0 or it == max_iter:
            E = energy(M, h, X, Y, Z, ns=ns)["E_stat"]
            hist.append({"it": it, "E": E, "fmax": float(np.abs(F).max()), "dt": dt})
            print(f"  {tag} it {it:4d} E4 {E:.6f} fmax {hist[-1]['fmax']:.3e} "
                  f"dt {dt:.3f} [{time.time() - t0:.0f}s]", flush=True)
    return M, hist


# ================= texture reads =================
def texture(M, h, L, X, Y, Z):
    rd = cell_reads(M, X, Y, Z)
    n = M.shape[0]
    trip, r = rd["triple"], rd["r"]
    c = slice(n // 2 - 1, n // 2 + 1)
    center = trip[c, c, c].reshape(-1, 3).mean(axis=0)
    lam1 = trip[..., 0]
    inside = lam1 < 0.8
    dev = np.abs(trip - VAC_TRIPLE)
    dev_max = dev.max(axis=-1)
    dev_l2 = np.sqrt((dev ** 2).sum(axis=-1))
    ext = r > 0.45 * L
    # shell means between 0.12 L and 0.42 L, bins of width h
    edges = np.arange(0.12 * L, 0.42 * L + 1e-9, h)
    rs, dm, dl = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.sum() >= 4:
            rs.append(r[m].mean())
            dm.append(dev_max[m].mean())
            dl.append(dev_l2[m].mean())
    rs, dm, dl = map(np.array, (rs, dm, dl))
    slope_max = float(np.polyfit(np.log(rs), np.log(dm), 1)[0])
    slope_l2 = float(np.polyfit(np.log(rs), np.log(dl), 1)[0])
    return {
        "beta2_max": float(np.nanmax(rd["beta2"])),
        "half_max": float(rd["half"].max()),
        "center_triple": [float(x) for x in center],
        "center_gap_lam1_minus_lam2": float(center[0] - center[1]),
        "center_gap_lam1_minus_pairmean": float(center[0] - 0.5 * (center[1] + center[2])),
        "lam1_min": float(lam1.min()),
        "r_d": float(r[inside].max()) if inside.any() else 0.0,
        "n_cells_lam1_below_0.8": int(inside.sum()),
        "min_n_dot_rhat": float(rd["ndotr"].min()),
        "min_n_dot_rhat_free": float(rd["ndotr"][~pin_shell(n, h, 1.6)].min()),
        "K_vs_J_max_abs": float(np.abs(rd["K"] - rd["J"]).max()),
        "det_u_n_e_f_min": float(rd["det"].min()),
        "G_minus_I_max": float(np.abs(rd["G"] - np.eye(4)).max()),
        "u_minus_e0_max": float(np.abs(np.abs(rd["u"]) - np.array([1, 0, 0, 0])).max()),
        "ext_max_dev_beyond_0.45L": float(dev_max[ext].max()),
        "ext_n_cells": int(ext.sum()),
        "tail_exp_maxdev": slope_max, "tail_exp_l2dev": slope_l2,
        "tail_shells": int(len(rs)),
        "tail_r_range": [float(rs.min()), float(rs.max())],
    }


# ================= driver =================
def verdict(ok, qual=False):
    return "CONFIRMED" if ok and not qual else ("QUALIFIED" if ok else "REFUTED")


def main(quick=False):
    t_start = time.time()
    np.set_printoptions(precision=6, suppress=True)
    rng = np.random.default_rng(20260906)
    res = {"script": "scripts/m5_32_r16_1_audit.py", "python": sys.version.split()[0],
           "numpy": np.__version__, "omp_threads": os.environ.get("OMP_NUM_THREADS"),
           "fields": {k: v[0] for k, v in FIELDS.items()},
           "stencil_semantic": "sym = energies averaged over the fwd and bwd one-sided "
                               "branches (weight 1/2 each), the certified m5_21_3_a_4d "
                               "semantic; h^3 sums over all cells",
           "n_s_reads": NS_READ, "n_s_descent": NS_DESCENT, "claims": {}, "timing_s": {}}
    fields = {}
    for key, (rel, n, L) in FIELDS.items():
        M = np.load(os.path.join(RESEARCH, rel))
        h = L / n
        X, Y, Z = coords(n, h)
        fields[key] = (M, h, L, X, Y, Z, pin_shell(n, h, 1.6))
        assert np.abs(M[..., 0, 1:]).max() == 0.0, f"{key}: M_0i != 0"

    # ---------- registry (C1.6 cross-read only)
    t0 = time.time()
    REG = _load("m5_32_terms_ext", "m5_32_terms_ext.py")
    I1h_np = REG.REGISTRY_EXT["I1_h"].density
    res["timing_s"]["registry_import"] = time.time() - t0

    # ---------- C1.1 energies (8 samples)
    t0 = time.time()
    E = {}
    for key in ("vac_n32", "seed_n32", "end_n32", "seed_n48", "end_n48", "end_n64"):
        M, h, L, X, Y, Z, pin = fields[key]
        ns = NS_READ if not (quick and key == "end_n64") else 4
        t1 = time.time()
        e = energy(M, h, X, Y, Z, ns=ns, registry=I1h_np if key != "vac_n32" else None)
        e["n_s"] = ns
        e["wall_s"] = time.time() - t1
        E[key] = e
        print(f"[C1.1] {key:9s} n_s {ns} E_h {e['E_h']:.5f} V4 {e['V4']:.5f} "
              f"U {e['U']:.3e} K_P {e['K_P']:.5f} reg {e['reg']:.3e} "
              f"E_stat {e['E_stat']:.5f}  [{e['wall_s']:.0f}s]", flush=True)
    res["timing_s"]["energies"] = time.time() - t0
    # the derivative-averaged (central) variant, for the record only
    M, h, L, X, Y, Z, pin = fields["seed_n32"]
    e_c = energy(M, h, X, Y, Z, ns=NS_READ, stencil="cen")
    c11 = {"mine": {k: {kk: E[k][kk] for kk in ("E_h", "E_h_norm", "V4", "U", "K_P",
                                                  "reg", "E_stat", "I1h_reg", "n_s")}
                    for k in E}, "producer": PROD,
           "vacuum_E_stat": E["vac_n32"]["E_stat"],
           "central_stencil_variant_seed_n32": {k: e_c[k] for k in ("E_h", "K_P", "E_stat")}}
    worst = 0.0
    rows = []
    for key, pr in PROD.items():
        for part, pv in pr.items():
            mv = E[key][part]
            rel = abs(mv - pv) / max(abs(pv), 1e-12)
            rows.append((key, part, mv, pv, rel))
            if part in ("E_h", "V4", "K_P", "E_stat"):
                worst = max(worst, rel)
            elif abs(mv - pv) > 0.05 * abs(pv) + 1e-7:
                worst = max(worst, rel)
    c11["max_rel_dev_main_parts"] = worst
    c11["rows"] = [{"field": a, "part": b, "mine": c, "producer": d, "rel": e_}
                   for a, b, c, d, e_ in rows]
    c11["verdict"] = verdict(worst < 2e-4)
    c11["method"] = ("own evaluator: eigh per cell (u, G, n, pair plane, plateau weight), "
                     "own Rodrigues circle (8 samples), certified fwd/bwd branch average")
    res["claims"]["C1.1"] = c11

    # ---------- C1.3 / C1.4 textures
    t0 = time.time()
    TX = {}
    for key in ("seed_n32", "end_n32", "end_n48", "end_n64"):
        M, h, L, X, Y, Z, pin = fields[key]
        TX[key] = texture(M, h, L, X, Y, Z)
        print(f"[C1.3] {key:9s} beta2max {TX[key]['beta2_max']:.2e} halfmax "
              f"{TX[key]['half_max']:.2e} center {np.array(TX[key]['center_triple'])} "
              f"lam1min {TX[key]['lam1_min']:.3f} r_d {TX[key]['r_d']:.2f} "
              f"({TX[key]['n_cells_lam1_below_0.8']}) K-J {TX[key]['K_vs_J_max_abs']:.1e}",
              flush=True)
        print(f"[C1.4] {key:9s} ext max dev {TX[key]['ext_max_dev_beyond_0.45L']:.2e} "
              f"tail exp {TX[key]['tail_exp_maxdev']:.2f} (l2 {TX[key]['tail_exp_l2dev']:.2f})",
              flush=True)
    res["timing_s"]["textures"] = time.time() - t0
    ok13, q13 = True, False
    for key in ("end_n32", "end_n48", "end_n64"):
        p, m = PROD_TEX[key], TX[key]
        for a, b, tol in ((p["beta2_max"], m["beta2_max"], 0.15),
                          (p["half_max"], m["half_max"], 0.15),
                          (p["lam1_min"], m["lam1_min"], 0.01),
                          (p["r_d"], m["r_d"], 0.01)):
            if abs(a - b) > tol * max(abs(a), 1e-12):
                ok13 = False
        if np.abs(np.array(p["center"]) - np.array(m["center_triple"])).max() > 2e-3:
            ok13 = False
        if p["n_d"] != m["n_cells_lam1_below_0.8"]:
            ok13 = False
    res["claims"]["C1.3"] = {"mine": TX, "producer": PROD_TEX, "verdict": verdict(ok13, q13),
                             "method": "eigh of the spatial triple per cell; center = mean "
                                       "over the 8 innermost cells; r_d = max r with lam1 < 0.8"}
    ok14 = True
    for key in ("end_n32", "end_n48", "end_n64"):
        p, m = PROD_TEX[key], TX[key]
        if abs(m["ext_max_dev_beyond_0.45L"] - p["ext_max_dev"]) > 0.2 * p["ext_max_dev"]:
            ok14 = False
        if abs(m["tail_exp_maxdev"] - p["tail_exp"]) > 0.15:
            ok14 = False
    res["claims"]["C1.4"] = {"mine": {k: {kk: TX[k][kk] for kk in (
        "ext_max_dev_beyond_0.45L", "ext_n_cells", "tail_exp_maxdev", "tail_exp_l2dev",
        "tail_shells", "tail_r_range")} for k in ("end_n32", "end_n48", "end_n64")},
        "producer": {k: {"ext_max_dev": PROD_TEX[k]["ext_max_dev"],
                         "tail_exp": PROD_TEX[k]["tail_exp"]} for k in
                     ("end_n32", "end_n48", "end_n64")},
        "verdict": verdict(ok14),
        "method": "max |triple - (1,.3,.3)| beyond 0.45 L; shell means in bins of width h "
                  "on [0.12 L, 0.42 L], least-squares log-log slope"}

    # ---------- C1.5 instrument gates
    t0 = time.time()
    G15 = {}
    for key in ("end_n32", "end_n48", "end_n64"):
        M, h, L, X, Y, Z, pin = fields[key]
        rd = cell_reads(M, X, Y, Z)
        K = rd["K"]
        base8 = E[key]["E_stat"] if E[key]["n_s"] == NS_READ else \
            energy(M, h, X, Y, Z, ns=NS_READ)["E_stat"]
        e1 = energy(M, h, X, Y, Z, ns=1)
        reg1 = e1["reg"]
        g = {"E_stat_8": base8, "reg_unaveraged": reg1, "rot": {}}
        for bet in (0.4, 1.1):
            R = rodrigues(K, 0.5 * bet)
            Mb = R @ M @ T(R)
            rdb = cell_reads(Mb, X, Y, Z)
            n_drift = float(np.abs(rdb["n"] - rd["n"]).max())
            eb = energy(Mb, h, X, Y, Z, ns=NS_READ, K=K)
            eb1 = energy(Mb, h, X, Y, Z, ns=1, K=K)
            Rm = rodrigues(K, -0.5 * bet)
            ebm1 = energy(Rm @ M @ T(Rm), h, X, Y, Z, ns=1, K=K)
            g["rot"][str(bet)] = {
                "E_stat_8_rel": abs(eb["E_stat"] - base8) / base8,
                "reg_unaveraged_rel": abs(eb1["reg"] - reg1) / reg1,
                "reg_unaveraged_rel_minus_beta": abs(ebm1["reg"] - reg1) / reg1,
                "E_h_unaveraged_rel": abs(eb1["E_h"] - e1["E_h"]) / e1["E_h"],
                "K_P_unaveraged_rel": abs(eb1["K_P"] - e1["K_P"]) / e1["K_P"],
                "director_drift_max": n_drift}
        ns_list = (4, 5, 6, 7, 16) if key == "end_n32" else (16,)
        if quick and key == "end_n64":
            ns_list = (16,)
        g["n_s_scan"] = {str(k): (energy(M, h, X, Y, Z, ns=k)["E_stat"] - base8) / base8
                         for k in ns_list}
        G15[key] = g
        print(f"[C1.5] {key:9s} T0.4 {g['rot']['0.4']['E_stat_8_rel']:.1e} "
              f"T1.1 {g['rot']['1.1']['E_stat_8_rel']:.1e} | reg unavg "
              f"{g['rot']['0.4']['reg_unaveraged_rel']:.2e}/{g['rot']['1.1']['reg_unaveraged_rel']:.2e}"
              f" | ns scan {g['n_s_scan']}", flush=True)
    # the aliased m = 4 harmonic of the n_s = 4 descent objective scales as
    # half^4: quantify it on the seed (half 6.7e-3) next to the end (5.6e-4)
    Ms, hs, Ls, Xs, Ys, Zs, _ = fields["seed_n32"]
    G15["seed_n32_n_s_scan"] = {
        str(k): (energy(Ms, hs, Xs, Ys, Zs, ns=k)["E_stat"] - E["seed_n32"]["E_stat"])
        / E["seed_n32"]["E_stat"] for k in (4, 16)}
    print(f"[C1.5] seed_n32 n_s scan {G15['seed_n32_n_s_scan']}", flush=True)
    res["timing_s"]["gates"] = time.time() - t0
    ok15 = True
    for key, (pE, preg, p16) in PROD_GATE.items():
        g = G15[key]
        if max(g["rot"]["0.4"]["E_stat_8_rel"], g["rot"]["1.1"]["E_stat_8_rel"]) > 1e-11:
            ok15 = False
        if abs(g["n_s_scan"]["16"]) > 1e-11:
            ok15 = False
        r04 = g["rot"]["0.4"]["reg_unaveraged_rel"]
        if not (1e-8 < r04 < 1e-3):
            ok15 = False
    res["claims"]["C1.5"] = {
        "mine": G15, "producer": {k: {"E_stat_8_rel": v[0], "reg_unaveraged_rel_0.4": v[1],
                                      "n16_rel": v[2]} for k, v in PROD_GATE.items()},
        "verdict": verdict(ok15, qual=True),
        "qualification": "T_alpha M = Mbar + half (cos(alpha) P + sin(alpha) Q) in the "
                         "cell's eigenframe, so E(alpha) is a trigonometric polynomial of "
                         "degree <= 4 in alpha and the n_s-point trapezoid rule is EXACT for "
                         "every n_s >= 5: the T_beta invariance and the 8->16 doubling are "
                         "roundoff identities of the quadrature, not a property of the field; "
                         "the n_s = 4 descent objective aliases only the m = 4 harmonic, whose "
                         "coefficient is O(half^4) (1e-15 relative on the end fields, see "
                         "n_s_scan['4'] and seed_n32_n_s_scan['4'])",
        "method": "pointwise T_beta with the field's own [n]_x, beta 0.4/1.1 (both signs "
                  "for the unaveraged regulator); n_s = 4,5,6,7,16 scan on n32"}

    # ---------- C1.6 completions
    ok16 = True
    c16 = {}
    for key in ("seed_n32", "end_n32", "end_n48", "end_n64"):
        e = E[key]
        c16[key] = {"E_h_rebuild": e["E_h"], "E_h_norm": e["E_h_norm"],
                    "4xI1h_registry": e["I1h_reg"],
                    "rel_norm_vs_rebuild": abs(e["E_h_norm"] - e["E_h"]) / e["E_h"],
                    "rel_registry_vs_rebuild": abs(e["I1h_reg"] - e["E_h"]) / e["E_h"]}
        if c16[key]["rel_norm_vs_rebuild"] > 1e-12 or c16[key]["rel_registry_vs_rebuild"] > 1e-10:
            ok16 = False
        print(f"[C1.6] {key:9s} rebuild {e['E_h']:.10f} norm {e['E_h_norm']:.10f} "
              f"4xI1h {e['I1h_reg']:.10f}", flush=True)
    res["claims"]["C1.6"] = {"mine": c16, "verdict": verdict(ok16),
                             "method": "own I_norm (eta in place of G) and the registry "
                                       "I1_h_np on the same circle samples and branches"}

    # ---------- C1.7 statement consistency
    e32, e48 = E["end_n32"], E["end_n48"]
    ex = {p: float(np.log(e48[p] / e32[p]) / np.log(72.0 / 48.0))
          for p in ("E_stat", "E_h", "K_P")}
    ex_prod = {p: float(np.log(b / a) / np.log(72.0 / 48.0)) for p, (a, b) in
               (("E_stat", (14.472, 14.884)), ("E_h", (5.830, 6.112)), ("K_P", (8.545, 8.674)))}
    res["claims"]["C1.7"] = {
        "end_vs_end_exponents_mine": ex,
        "producer_iteration_matched_exponents_recomputed_from_stated_numbers": ex_prod,
        "producer_stated": {"E_stat": 0.07, "E_h": 0.12, "K_P": 0.04, "end_vs_end_E_stat": 0.18},
        "verdict": "QUALIFIED",
        "method": "log(E48/E32)/log(72/48) from my own end energies; the iteration-1500 "
                  "n32 checkpoint is not stored, so the 0.07 is untestable; the stated "
                  "iteration-matched numbers reproduce 0.07/0.12/0.04 arithmetically"}
    print(f"[C1.7] end-vs-end exponents {ex}; producer's stated numbers give {ex_prod}",
          flush=True)

    # ---------- C1.2 monotone / stationarity / own descent
    t0 = time.time()
    c12 = {"E_stat_end_lt_seed": {"n32": E["end_n32"]["E_stat"] < E["seed_n32"]["E_stat"],
                                  "n48": E["end_n48"]["E_stat"] < E["seed_n48"]["E_stat"]},
           "E_stat_drop": {"n32": E["seed_n32"]["E_stat"] - E["end_n32"]["E_stat"],
                           "n48": E["seed_n48"]["E_stat"] - E["end_n48"]["E_stat"]}}
    M, h, L, X, Y, Z, pin = fields["end_n32"]
    free = ~pin
    fd4 = fd_check(M, h, X, Y, Z, free, NS_DESCENT, rng, n_pts=10 if not quick else 5)
    print(f"[C1.2] FD check (n_s=4): max rel err {fd4['max_rel_err']:.2e} "
          f"(without dR path {fd4['max_rel_err_no_dR']:.2e}); |g| {fd4['grad_norm_free']:.4f} "
          f"fmax {fd4['grad_fmax_free']:.4f}; M0i dirs {np.array(fd4['M0i_directional_derivatives'])}",
          flush=True)
    g8 = gradient(M, h, X, Y, Z, free, ns=NS_READ)
    c12["fd_check_ns4"] = fd4
    c12["grad_norm_free_ns8"] = float(np.sqrt((g8 ** 2).sum()))
    c12["grad_fmax_free_ns8"] = float(np.abs(g8).max())
    c12["grad_norm_free_ns4"] = fd4["grad_norm_free"]
    c12["grad_fmax_free_ns4"] = fd4["grad_fmax_free"]
    # seed gradient for scale
    Ms = fields["seed_n32"][0]
    gs = gradient(Ms, h, X, Y, Z, free, ns=NS_DESCENT)
    c12["seed_grad_norm_free_ns4"] = float(np.sqrt((gs ** 2).sum()))
    c12["seed_grad_fmax_free_ns4"] = float(np.abs(gs).max())
    n_it = 100 if not quick else 10
    E8_before = E["end_n32"]["E_stat"]
    E4_before = energy(M, h, X, Y, Z, ns=NS_DESCENT)["E_stat"]
    M_end, hist = fire(M, h, X, Y, Z, free, n_it, ns=NS_DESCENT, tag="[C1.2 FIRE]")
    e8_after = energy(M_end, h, X, Y, Z, ns=NS_READ)
    E4_after = hist[-1]["E"]
    c12["descent"] = {"iterations": n_it, "n_s": NS_DESCENT, "trace": hist,
                      "E4_before": E4_before, "E4_after": E4_after,
                      "E4_drop": E4_before - E4_after,
                      "E8_before": E8_before, "E8_after": e8_after["E_stat"],
                      "E8_drop": E8_before - e8_after["E_stat"],
                      "E4_minus_E8_before": E4_before - E8_before,
                      "monotone_E4_trace": bool(all(hist[i + 1]["E"] <= hist[i]["E"] + 1e-12
                                                    for i in range(len(hist) - 1))),
                      "parts_after_8": {k: e8_after[k] for k in ("E_h", "V4", "U", "K_P", "reg")},
                      "max_M0i_after": float(np.abs(M_end[..., 0, 1:]).max()),
                      "max_shell_change": float(np.abs((M_end - M)[pin]).max())}
    ok12 = (c12["E_stat_end_lt_seed"]["n32"] and c12["E_stat_end_lt_seed"]["n48"]
            and fd4["max_rel_err"] < 1e-3
            and 0.0 < c12["descent"]["E4_drop"] < 0.05
            and 0.0 < c12["descent"]["E8_drop"] < 0.05)
    q12 = not (c12["grad_norm_free_ns4"] < 0.6 and c12["grad_fmax_free_ns4"] <= 0.5 * 1.3)
    c12["producer"] = {"grad_norm_free": "< 0.6", "fmax": 0.5,
                       "drift_per_500_it": 0.05}
    c12["verdict"] = verdict(ok12, qual=q12)
    c12["method"] = ("hand-derived reverse-mode gradient of the 4-sample circle average "
                     "(incl. dR/dn and the Daleckii-Krein w(N) adjoint), certified vs central "
                     "FD of this script's energy; own FIRE, 100 iterations, pinned shell fixed")
    res["claims"]["C1.2"] = c12
    res["timing_s"]["descent_block"] = time.time() - t0
    print(f"[C1.2] E4 {E4_before:.6f} -> {E4_after:.6f} (drop {E4_before - E4_after:.5f}); "
          f"E8 {E8_before:.6f} -> {e8_after['E_stat']:.6f} (drop "
          f"{E8_before - e8_after['E_stat']:.5f}); |g|4 {c12['grad_norm_free_ns4']:.4f} "
          f"|g|8 {c12['grad_norm_free_ns8']:.4f}", flush=True)

    # ---------- tally
    tally = {}
    for k, v in res["claims"].items():
        tally[v["verdict"]] = tally.get(v["verdict"], 0) + 1
    res["tally"] = tally
    res["timing_s"]["total"] = time.time() - t_start
    with open(OUT_JSON, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\n| claim | verdict | mine vs producer |")
    print("| --- | --- | --- |")
    print(f"| C1.1 | {c11['verdict']} | max rel dev of E_h/V4/K_P/E_stat {worst:.2e} |")
    print(f"| C1.2 | {c12['verdict']} | E4 drop {c12['descent']['E4_drop']:.4f}, E8 drop "
          f"{c12['descent']['E8_drop']:.4f} in {n_it} it; |g| {c12['grad_norm_free_ns4']:.3f} "
          f"fmax {c12['grad_fmax_free_ns4']:.3f} (prod < 0.6 / 0.5) |")
    print(f"| C1.3 | {res['claims']['C1.3']['verdict']} | beta2max n32 {TX['end_n32']['beta2_max']:.1e} "
          f"(4.2e-4), half n32 {TX['end_n32']['half_max']:.1e} (5.6e-4) |")
    print(f"| C1.4 | {res['claims']['C1.4']['verdict']} | tail exps "
          f"{TX['end_n32']['tail_exp_maxdev']:.2f}/{TX['end_n48']['tail_exp_maxdev']:.2f}/"
          f"{TX['end_n64']['tail_exp_maxdev']:.2f} (-4.03/-4.17/-4.14) |")
    print(f"| C1.5 | {res['claims']['C1.5']['verdict']} | exact quadrature for n_s >= 5 |")
    print(f"| C1.6 | {res['claims']['C1.6']['verdict']} | max rel "
          f"{max(v['rel_registry_vs_rebuild'] for v in c16.values()):.1e} |")
    print(f"| C1.7 | QUALIFIED | end-vs-end E_stat exponent {ex['E_stat']:.3f} (0.18) |")
    print(f"tally {tally}; total {res['timing_s']['total']:.0f}s; -> {os.path.relpath(OUT_JSON, RESEARCH)}")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv[1:])
