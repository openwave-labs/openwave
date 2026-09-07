#!/usr/bin/env python3
"""M5.32 R16-3 adversarial audit: the fixed-K descents (claims C3.1 to C3.7).

Independent evaluator written from the brief's definitions only (no producer
code is imported or read; the certified stencil helpers d1 / branches / coords /
pin_shell / vac4 / base_cfg of m5_21_3_a_4d.py are the only imports).

Conventions (own implementation):
  - spatial triple = eigenvalues of the 3x3 spatial block of N = M eta (exact
    because every field has M_0i = 0, asserted), u = e_0, G = I (asserted);
  - director n = eigenvector of lambda_1 lifted outward (n . r_hat > 0);
    (n, e, f) right-handed (det[u, n, e, f] = det3[n, e, f] > 0);
    J = [n]_x (right-handed rotation about n), a0 = J M + M J^T;
  - the circle T_alpha M = R(alpha/2) M R(alpha/2)^T, Rodrigues, 8 samples,
    every transformed field re-diagonalized (a0, W refreshed per sample);
  - energies h^3-weighted, sym stencil = (fwd + bwd)/2 one-sided branches;
  - "unit direction" = unit Frobenius norm of the full (n,n,n,4,4) array,
    supported on the free cells (outside pin_shell depth 1.6) and on the
    7 components M_00 + spatial symmetric block (M_0i stay 0, so G = I);
  - frozen-a0 protocol (own definition): a0, the circle rotations R_k and the
    spectral weight W frozen at the reference field; everything else exact.

Outputs: data/m5_32_r16_3_audit.json (relative paths only) + the verdict
table in the terminal.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from m5_21_3_a_4d import d1, branches, coords, pin_shell, vac4, base_cfg  # noqa: E402

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
ETAD = np.array([-1.0, 1.0, 1.0, 1.0])
I4 = np.eye(4)
G_T, DELTA, W1 = 8.0, 0.3, 0.000724023879
MU, C_S, C_P, OMEGA_C = 1e-2, 0.4, 1.0, 0.05
CP = [(-G_T) ** p + 1.0 + 2.0 * DELTA ** p for p in range(1, 5)]
NS = 8
RNG = np.random.default_rng(20260906)
T0 = time.time()

FIELDS = {
    "r16_1_n32": ("checkpoints/m5_32_r16/r16_1_rebuild_n32_L48.npy", 32, 48.0, None),
    "r16_1_n48": ("checkpoints/m5_32_r16/r16_1_rebuild_n48_L72.npy", 48, 72.0, None),
    "n32_K50": ("checkpoints/m5_32_r16/r16_3_rebuild_n32_L48_K50.npy", 32, 48.0, 50.0),
    "n32_K200": ("checkpoints/m5_32_r16/r16_3_rebuild_n32_L48_K200.npy", 32, 48.0, 200.0),
    "n48_K50": ("checkpoints/m5_32_r16/r16_3_rebuild_n48_L72_K50.npy", 48, 72.0, 50.0),
    "n48_K200": ("checkpoints/m5_32_r16/r16_3_rebuild_n48_L72_K200.npy", 48, 72.0, 200.0),
}

# the producer's numbers (from the brief), never trusted, only compared
PROD = {
    "n32_K50": dict(E_stat=21.498, E_h=7.627, V4=0.077, U=0.023, K_P=12.988, reg=0.783,
                    kin_tot=90.64, kin_h=85.99, kin_KP=3.26, kin_reg=1.40, E_K=28.393,
                    omega=0.2758, bound_gap=14.58, max_half=0.508, rms=4.92,
                    quad=(-0.327, -0.116, 0.443), tilt=0.403, deriv_true=4.8,
                    deriv_frozen=2e-2),
    "n32_K200": dict(E_K=48.846, kin_tot=658.2, kin_h=638.8, kin_KP=8.2, kin_reg=11.2,
                     omega=0.1519, bound_gap=35.03, max_half=0.645, rms=4.93,
                     quad=(-0.140, 0.070, 0.070), tilt=0.716, deriv_true=4.3e-2,
                     deriv_frozen=1e-2),
    "n48_K50": dict(E_K=31.289, kin_tot=65.46, kin_h=54.0, kin_KP=10.4, kin_reg=1.07,
                    omega=0.3819, bound_gap=16.41, max_half=0.362, rms=3.27,
                    quad=(-0.315, 0.157, 0.158), tilt=0.099, deriv_true=3.4e-2,
                    deriv_frozen=5e-3),
    "n48_K200": dict(E_K=55.232, kin_tot=544.9, kin_h=522.9, kin_KP=10.9, kin_reg=11.1,
                     omega=0.1835, bound_gap=40.35, max_half=0.644, rms=3.27,
                     quad=(-0.315, 0.157, 0.158), tilt=0.064, deriv_true=3.2e-2,
                     deriv_frozen=7e-3),
    "r16_1_n32": dict(E_stat=13.81651),
    "r16_1_n48": dict(E_stat=14.88381),
}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- stencil adjoint (own)
def d1_adj(g, ax, h, st):
    """Transpose of the certified one-sided d1 (fwd / bwd), own derivation."""
    out = np.zeros_like(g)
    sl = [slice(None)] * g.ndim

    def at(i):
        s = list(sl)
        s[ax] = i
        return tuple(s)
    if st == "fwd":
        out[at(slice(1, None))] += g[at(slice(0, -1))] / h
        out[at(slice(0, -1))] -= g[at(slice(0, -1))] / h
    elif st == "bwd":
        out[at(slice(1, None))] += g[at(slice(1, None))] / h
        out[at(slice(0, -1))] -= g[at(slice(1, None))] / h
    else:
        raise ValueError(st)
    return out


# ---------------------------------------------------------------- spectral frame
def wfun(lam):
    lam = np.asarray(lam, float)
    w = np.zeros_like(lam)
    w[np.abs(lam - 0.3) <= 0.5] = 1.0
    m = (lam > 0.8) & (lam < 1.0)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * (lam[m] - 0.8) / 0.2))
    m = (lam > -1.0) & (lam < -0.2)
    w[m] = 0.5 * (1.0 + np.cos(np.pi * (-0.2 - lam[m]) / 0.8))
    return w


def cross_mat(n):
    J = np.zeros(n.shape[:-1] + (4, 4))
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    J[..., 1, 2] = -nz
    J[..., 1, 3] = ny
    J[..., 2, 1] = nz
    J[..., 2, 3] = -nx
    J[..., 3, 1] = -ny
    J[..., 3, 2] = nx
    return J


def frame(M, geo):
    """Own reads: triple, lifted director, right-handed pair frame, J, a0, W, rho2."""
    assert np.abs(M[..., 0, 1:]).max() == 0.0, "M_0i must vanish (brief)"
    S = M[..., 1:, 1:]
    lam, V = np.linalg.eigh(S)           # ascending
    l3, l2, l1 = lam[..., 0], lam[..., 1], lam[..., 2]
    n = V[..., :, 2].copy()
    e = V[..., :, 1].copy()
    f = V[..., :, 0].copy()
    s = np.sign(np.einsum("...i,...i->...", n, geo["rhat"]))
    s[s == 0] = 1.0
    n *= s[..., None]
    det = np.einsum("...i,...i->...", n, np.cross(e, f))
    sd = np.sign(det)
    sd[sd == 0] = 1.0
    f *= sd[..., None]
    J = cross_mat(n)
    a0 = J @ M + M @ J.swapaxes(-1, -2)
    wl = wfun(lam)
    W = np.zeros_like(M)
    W[..., 1:, 1:] = np.einsum("...k,...ik,...jk->...ij", wl, V, V)
    half = 0.5 * (l2 - l3)
    return dict(l1=l1, l2=l2, l3=l3, n=n, e=e, f=f, J=J, a0=a0, W=W, wl=wl,
                half=half, rho2=half ** 2, gap=l1 - l2, w0=wfun(-M[..., 0, 0]))


def rot(J, beta):
    return I4 + np.sin(beta) * J + (1.0 - np.cos(beta)) * (J @ J)


def congr(R, M):
    return R @ M @ R.swapaxes(-1, -2)


# ---------------------------------------------------------------- energies (own)
def tr_eta2(Om):
    """tr(Om^T eta Om eta) per cell = sum_ab eta_a eta_b Om_ba^2."""
    return np.einsum("...ba,...ba,a,b->...", Om, Om, ETAD, ETAD)


def terms(M, fr, h, want_grad=False):
    """All static terms and omega^2 coefficients on ONE field (no circle), h^3-weighted.
    With want_grad: also the gradient wrt M of (E_h + K_P + reg_A + kin_h_A) through
    the lattice derivatives only (a0, W, rho2 held), as (gstat, gkin)."""
    h3 = h ** 3
    a0, W, rho2 = fr["a0"], fr["W"], fr["rho2"]
    out = dict(E_h=0.0, K_P=0.0, kin_h=0.0)
    e2 = np.zeros(M.shape[:3])
    gstat = np.zeros_like(M) if want_grad else None
    gkin = np.zeros_like(M) if want_grad else None
    for br, wt in branches("sym"):
        A = [d1(M, ax, h, br) for ax in range(3)]
        gA = [np.zeros_like(M) for _ in range(3)] if want_grad else None
        gAk = [np.zeros_like(M) for _ in range(3)] if want_grad else None
        for i in range(3):
            for j in range(i + 1, 3):
                F = A[i] @ A[j] - A[j] @ A[i]
                out["E_h"] += wt * 4.0 * h3 * np.sum(F * F)
                if want_grad:
                    gA[i] += 8.0 * wt * h3 * (F @ A[j] - A[j] @ F)
                    gA[j] += 8.0 * wt * h3 * (A[i] @ F - F @ A[i])
            Om = W @ A[i] @ ETA @ W
            out["K_P"] += wt * 0.5 * h3 * np.sum(tr_eta2(Om))
            e2 += wt * np.sum(A[i] * A[i], axis=(-1, -2))
            F0 = a0 @ A[i] - A[i] @ a0
            out["kin_h"] += wt * 4.0 * h3 * np.sum(F0 * F0)
            if want_grad:
                gA[i] += wt * h3 * (W @ (ETA @ Om @ ETA) @ W @ ETA)
                gA[i] += 2.0 * wt * h3 * C_S * rho2[..., None, None] * A[i]
                gAk[i] += 8.0 * wt * h3 * (a0 @ F0 - F0 @ a0)
        if want_grad:
            for ax in range(3):
                gstat += d1_adj(gA[ax], ax, h, br)
                gkin += d1_adj(gAk[ax], ax, h, br)
    out["E2"] = e2
    out["reg"] = C_S * h3 * np.sum(rho2 * e2)
    Om0 = W @ a0 @ ETA @ W
    out["kin_KP"] = 0.5 * h3 * np.sum(tr_eta2(Om0))
    out["a0a0"] = np.sum(a0 * a0, axis=(-1, -2))
    out["kin_reg"] = C_S * h3 * np.sum(rho2 * out["a0a0"])
    N = M @ ETA
    P = N.copy()
    vd = 0.0
    for p in range(4):
        if p:
            P = P @ N
        vd = vd + (np.einsum("...kk->...", P) - CP[p]) ** 2
    out["V4"] = W1 * h3 * np.sum(vd)
    out["U"] = MU * h3 * np.sum(rho2)
    if want_grad:
        return out, gstat, gkin
    return out


def circle_eval(M, geo, h, ns=NS, frozen=None):
    """Circle-averaged action terms. frozen = None: the true evaluator (every sample
    re-diagonalized). frozen = dict(R, a0, W): the frozen-a0 protocol."""
    fr = frame(M, geo)
    keys = ["E_h", "V4", "U", "K_P", "reg", "kin_h", "kin_KP", "kin_reg"]
    acc = {k: 0.0 for k in keys}
    for k in range(ns):
        if frozen is None:
            R = rot(fr["J"], np.pi * k / ns)
            Mk = congr(R, M)
            frk = frame(Mk, geo)
        else:
            R = frozen["R"][k]
            Mk = congr(R, M)
            frk = dict(a0=frozen["a0"][k], W=frozen["W"][k], rho2=fr["rho2"])
        t = terms(Mk, frk, h)
        for kk in keys:
            acc[kk] += t[kk] / ns
    acc["E_stat"] = acc["E_h"] + acc["V4"] + acc["U"] + C_P * acc["K_P"] + acc["reg"]
    acc["kin_tot"] = acc["kin_h"] + C_P * acc["kin_KP"] + acc["kin_reg"]
    return acc


def e_k(acc, K):
    return acc["E_stat"] + K ** 2 / (4.0 * acc["kin_tot"])


def freeze(M, geo, ns=NS):
    fr = frame(M, geo)
    Rs, a0s, Ws = [], [], []
    for k in range(ns):
        R = rot(fr["J"], np.pi * k / ns)
        Rs.append(R)
        a0s.append(congr(R, fr["a0"]))
        Ws.append(congr(R, fr["W"]))
    return dict(R=Rs, a0=a0s, W=Ws)


def grad_frozen(M, geo, h, K, mask, ns=NS):
    """Own analytic gradient of E_K in the frozen-a0 protocol (frames re-frozen at M)."""
    h3 = h ** 3
    fr = frame(M, geo)
    frz = freeze(M, geo, ns)
    gstat = np.zeros_like(M)
    gkin = np.zeros_like(M)
    e2avg = np.zeros(M.shape[:3])
    acc = dict(E_stat=0.0, kin_tot=0.0)
    for k in range(ns):
        R = frz["R"][k]
        Mk = congr(R, M)
        frk = dict(a0=frz["a0"][k], W=frz["W"][k], rho2=fr["rho2"])
        t, gs, gk = terms(Mk, frk, h, want_grad=True)
        Rt = R.swapaxes(-1, -2)
        gstat += congr(Rt, gs) / ns
        gkin += congr(Rt, gk) / ns
        e2avg += t["E2"] / ns
        acc["E_stat"] += (t["E_h"] + t["V4"] + t["U"] + C_P * t["K_P"] + t["reg"]) / ns
        acc["kin_tot"] += (t["kin_h"] + C_P * t["kin_KP"] + t["kin_reg"]) / ns
    # spectral parts: rho2 = half^2, d rho2 / dS = half (e e^T - f f^T)
    dr = np.zeros_like(M)
    ee = np.einsum("...i,...j->...ij", fr["e"], fr["e"])
    ff = np.einsum("...i,...j->...ij", fr["f"], fr["f"])
    dr[..., 1:, 1:] = fr["half"][..., None, None] * (ee - ff)
    a0a0 = np.sum(fr["a0"] * fr["a0"], axis=(-1, -2))
    gstat += (MU * h3 + C_S * h3 * e2avg)[..., None, None] * dr
    gkin += (C_S * h3 * a0a0)[..., None, None] * dr
    N = M @ ETA
    Ppow = [np.broadcast_to(I4, M.shape).copy()]
    for p in range(1, 4):
        Ppow.append(Ppow[-1] @ N)
    for p in range(1, 5):
        tp = np.einsum("...kk->...", Ppow[p - 1] @ N)
        gstat += (W1 * h3 * 2.0 * (tp - CP[p - 1]) * p)[..., None, None] * (ETA @ Ppow[p - 1])
    kin = acc["kin_tot"]
    g = gstat - K ** 2 / (4.0 * kin ** 2) * gkin
    g = 0.5 * (g + g.swapaxes(-1, -2))
    g = g * mask
    return g, e_k(acc, K)


def free_mask(n, h):
    pinned = pin_shell(n, h, 1.6)
    mask = np.zeros((n, n, n, 4, 4))
    mask[~pinned] = 1.0
    mask[..., 0, 1:] = 0.0
    mask[..., 1:, 0] = 0.0
    return mask


def rand_dir(mask):
    D = RNG.standard_normal(mask.shape)
    D = 0.5 * (D + D.swapaxes(-1, -2)) * mask
    return D / np.linalg.norm(D)


# ---------------------------------------------------------------- geometry reads (own)
def geometry(M, geo, L, fr=None):
    fr = fr or frame(M, geo)
    rho2 = fr["rho2"]
    tot = rho2.sum()
    X, Y, Z, r = geo["X"], geo["Y"], geo["Z"], geo["r"]
    rms = float(np.sqrt(np.sum(rho2 * r ** 2) / tot))
    frac_beyond = float(rho2[r > 0.35 * L].sum() / tot)
    xi = np.stack([X, Y, Z], -1) / r[..., None]
    Q = np.einsum("abc,abci,abcj->ij", rho2, xi, xi) / tot - np.eye(3) / 3.0
    qe, qv = np.linalg.eigh(Q)
    ext = int(np.argmax(np.abs(qe)))
    axis = qv[:, ext]
    d111 = float(abs(axis @ np.ones(3) / np.sqrt(3.0)))
    tilt_c = 1.0 - np.einsum("...i,...i->...", fr["n"], geo["rhat"]) ** 2
    tilt = float(np.sum(rho2 * tilt_c) / tot)
    imax = np.unravel_index(int(np.argmax(fr["half"])), rho2.shape)
    # the shell: cells carrying 90 percent of rho2 (sorted, own definition)
    order = np.argsort(rho2.ravel())[::-1]
    cum = np.cumsum(rho2.ravel()[order]) / tot
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n50 = int(np.searchsorted(cum, 0.50) + 1)
    shell = np.zeros(rho2.size, bool)
    shell[order[:n90]] = True
    shell = shell.reshape(rho2.shape)
    r_shell = np.sqrt(np.sum(rho2[shell] * r[shell] ** 2) / rho2[shell].sum())
    # spin-2 mean of the pair angle zeta on the shell (own frame: t1 = n x z / |.|, t2 = n x t1)
    nn = fr["n"][shell]
    ee = fr["e"][shell]
    zhat = np.zeros_like(nn)
    zhat[:, 2] = 1.0
    t1 = np.cross(nn, zhat)
    bad = np.linalg.norm(t1, axis=-1) < 1e-8
    zhat[bad] = [1.0, 0.0, 0.0]
    t1 = np.cross(nn, zhat)
    t1 /= np.linalg.norm(t1, axis=-1)[:, None]
    t2 = np.cross(nn, t1)
    zeta = np.arctan2(np.sum(ee * t2, -1), np.sum(ee * t1, -1))
    m2 = np.sum(rho2[shell] * np.exp(2j * zeta)) / rho2[shell].sum()
    top = [dict(cell=[float(X[i]), float(Y[i]), float(Z[i])], r=float(r[i]),
                half=float(fr["half"][i]), triple=[float(fr["l1"][i]), float(fr["l2"][i]),
                                                     float(fr["l3"][i])])
           for i in [np.unravel_index(int(j), rho2.shape) for j in order[:6]]]
    pair_out = int(np.sum((fr["l2"] > 0.8) | (fr["l2"] < -0.2) | (fr["l3"] > 0.8) | (fr["l3"] < -0.2)))
    return dict(
        max_half=float(fr["half"].max()), rms_radius=rms, frac_rho2_beyond_035L=frac_beyond,
        quad_eigs=[float(v) for v in qe], quad_axis=[float(v) for v in axis],
        quad_axis_dot_111=d111, quad_shape=("prolate" if qe[ext] > 0 else "oblate"),
        tilt=tilt, max_split_cell=[float(X[imax]), float(Y[imax]), float(Z[imax])],
        max_split_r=float(r[imax]),
        max_split_triple=[float(fr["l1"][imax]), float(fr["l2"][imax]), float(fr["l3"][imax])],
        max_split_gap=float(fr["gap"][imax]), min_gap=float(fr["gap"].min()),
        min_gap_cell=[float(v) for v in
                      (X[np.unravel_index(int(np.argmin(fr["gap"])), rho2.shape)],
                       Y[np.unravel_index(int(np.argmin(fr["gap"])), rho2.shape)],
                       Z[np.unravel_index(int(np.argmin(fr["gap"])), rho2.shape)])],
        l3_min=float(fr["l3"].min()), l2_max=float(fr["l2"].max()), l1_max=float(fr["l1"].max()),
        cells_pair_outside_plateau=pair_out,
        n_cells_90pct_rho2=n90, n_cells_50pct_rho2=n50, shell_rms_radius=float(r_shell),
        spin2_mean_abs=float(abs(m2)), top_cells=top,
        escapes=dict(
            a=bool(fr["half"].max() < 1e-3), a_box=bool(rms > 0.35 * L),
            b=bool(qe.min() < -0.20 and tilt > 0.3),
            c=bool(qe.min() < -0.20 and tilt > 0.3 and abs(m2) > 0.5),
            d=bool(fr["gap"].min() <= 1e-3)),
    )


# ---------------------------------------------------------------- main
def main():
    res = dict(script="scripts/m5_32_r16_3_audit.py", ns=NS, fields={}, selftests={},
               derivatives={}, descent={}, verdicts={}, runtime_s=None)
    geos = {}
    M_all = {}
    for tag, (rel, n, L, K) in FIELDS.items():
        M = np.load(os.path.join(ROOT, rel))
        h = L / n
        if n not in geos:
            X, Y, Z = coords(n, h)
            r = np.sqrt(X * X + Y * Y + Z * Z)
            geos[n] = dict(X=X, Y=Y, Z=Z, r=r, rhat=np.stack([X, Y, Z], -1) / r[..., None])
        M_all[tag] = (M, n, L, h, K)

    # --- self-tests of the conventions
    cfg = base_cfg(n=32, L=48.0)
    st = {}
    st["vac4_certified_stack"] = [float(v) for v in np.diag(vac4(cfg))]
    st["vac_brief_N_spectrum"] = [-8.0, 1.0, 0.3, 0.3]
    st["vac4_note"] = ("the certified stack's vac4 is diag(-8,1,0.3,0) (a different era); the "
                       "brief's degenerate vacuum diag(8,1,0.3,0.3) is what these fields carry "
                       "at the corner cell (checked below), the audit follows the brief")
    M32 = M_all["n32_K50"][0]
    corner = M32[0, 0, 0]
    st["corner_cell_spatial_eigs"] = [float(v) for v in np.linalg.eigvalsh(corner[1:, 1:])]
    fr = frame(M32, geos[32])
    u = np.zeros(4)
    u[0] = 1.0
    G = ETA + 2.0 * np.outer(ETA @ u, ETA @ u)
    st["G_minus_I_max"] = float(np.abs(G - I4).max())
    st["w_of_timelike_eigenvalue_max"] = float(fr["w0"].max())
    # a0 == (l2 - l3)(f e^T + e f^T) check, and a0(T_k M) == R a0 R^T
    a0_formula = np.zeros_like(M32)
    a0_formula[..., 1:, 1:] = (fr["l2"] - fr["l3"])[..., None, None] * (
        np.einsum("...i,...j->...ij", fr["f"], fr["e"]) + np.einsum("...i,...j->...ij", fr["e"], fr["f"]))
    st["a0_formula_maxdev"] = float(np.abs(fr["a0"] - a0_formula).max())
    R = rot(fr["J"], np.pi * 3 / NS)
    frk = frame(congr(R, M32), geos[32])
    st["a0_transported_maxdev"] = float(np.abs(frk["a0"] - congr(R, fr["a0"])).max())
    st["director_fixed_under_circle_maxdev"] = float(np.abs(np.abs(np.einsum("...i,...i->...", frk["n"], fr["n"])) - 1).max())
    st["det_nef_min"] = float(np.einsum("...i,...i->...", fr["n"], np.cross(fr["e"], fr["f"])).min())
    st["n_dot_rhat_min"] = float(np.einsum("...i,...i->...", fr["n"], geos[32]["rhat"]).min())
    res["selftests"] = st
    log("selftests", {k: v for k, v in st.items() if not isinstance(v, str)})

    # --- C3.1 / C3.2: energies, inertias, E_K
    for tag, (M, n, L, h, K) in M_all.items():
        t1 = time.time()
        acc = circle_eval(M, geos[n], h)
        # sample-by-sample spread (V4, U must be circle-invariant; the rest varies)
        fr = frame(M, geos[n])
        per = []
        for k in range(NS):
            Rk = rot(fr["J"], np.pi * k / NS)
            Mk = congr(Rk, M)
            tk = terms(Mk, frame(Mk, geos[n]), h)
            per.append(dict(E_stat=tk["E_h"] + tk["V4"] + tk["U"] + C_P * tk["K_P"] + tk["reg"],
                            kin_tot=tk["kin_h"] + C_P * tk["kin_KP"] + tk["kin_reg"], V4=tk["V4"], U=tk["U"]))
        d = {k: float(v) for k, v in acc.items()}
        d["per_sample_E_stat"] = [float(p["E_stat"]) for p in per]
        d["per_sample_kin_tot"] = [float(p["kin_tot"]) for p in per]
        d["V4_circle_spread"] = float(max(p["V4"] for p in per) - min(p["V4"] for p in per))
        d["U_circle_spread"] = float(max(p["U"] for p in per) - min(p["U"] for p in per))
        d["kin_h_fraction"] = float(acc["kin_h"] / acc["kin_tot"])
        if K is not None:
            d["K"] = K
            d["E_K"] = float(e_k(acc, K))
            d["omega"] = float(K / (2.0 * acc["kin_tot"]))
        d["geometry"] = geometry(M, geos[n], L, fr)
        d["eval_time_s"] = time.time() - t1
        res["fields"][tag] = d
        log(tag, "E_stat %.5f kin_tot %.4f" % (acc["E_stat"], acc["kin_tot"]),
            ("E_K %.4f omega %.4f" % (d["E_K"], d["omega"])) if K else "",
            "eval %.1fs" % d["eval_time_s"])
        log("   geometry: max_half %.4f rms %.3f quad %s tilt %.3f n90 %d min_gap %.2e escapes %s" % (
            d["geometry"]["max_half"], d["geometry"]["rms_radius"],
            np.round(d["geometry"]["quad_eigs"], 3).tolist(), d["geometry"]["tilt"],
            d["geometry"]["n_cells_90pct_rho2"], d["geometry"]["min_gap"],
            {k: v for k, v in d["geometry"]["escapes"].items() if v}))
    for tag in ("n32_K50", "n32_K200", "n48_K50", "n48_K200"):
        ref = "r16_1_n32" if tag.startswith("n32") else "r16_1_n48"
        d = res["fields"][tag]
        d["bound_gap"] = d["E_K"] - res["fields"][ref]["E_stat"]
        d["omega_c_K"] = OMEGA_C * d["K"]
        d["bound_ratio"] = d["bound_gap"] / d["omega_c_K"]

    # --- gradient self-test (frozen protocol, analytic vs finite difference)
    tag = "n32_K200"
    M, n, L, h, K = M_all[tag]
    mask = free_mask(n, h)
    g, ek_g = grad_frozen(M, geos[n], h, K, mask)
    frz = freeze(M, geos[n])
    D = rand_dir(mask)
    eps = 1e-4
    fd = (e_k(circle_eval(M + eps * D, geos[n], h, frozen=frz), K)
          - e_k(circle_eval(M - eps * D, geos[n], h, frozen=frz), K)) / (2 * eps)
    an = float(np.sum(g * D))
    res["selftests"]["grad_frozen_fd"] = float(fd)
    res["selftests"]["grad_frozen_analytic"] = an
    res["selftests"]["grad_frozen_reldev"] = float(abs(fd - an) / max(abs(fd), 1e-300))
    res["selftests"]["grad_frozen_norm"] = float(np.linalg.norm(g))
    res["selftests"]["E_K_frozen_at_ref_minus_true"] = float(ek_g - res["fields"][tag]["E_K"])
    log("gradient selftest (frozen): fd %.6e analytic %.6e reldev %.2e |g| %.4f" % (
        fd, an, res["selftests"]["grad_frozen_reldev"], np.linalg.norm(g)))

    # --- C3.5 (i) + C3.6: true vs frozen directional derivatives, 3 random directions
    for tag in ("n32_K50", "n32_K200", "n48_K50", "n48_K200"):
        M, n, L, h, K = M_all[tag]
        mask = free_mask(n, h)
        frz = freeze(M, geos[n])
        E0 = res["fields"][tag]["E_K"]
        rows = []
        for j in range(3):
            D = rand_dir(mask)
            row = dict(direction=j)
            for eps in (1e-4, 1e-3):
                Ep = e_k(circle_eval(M + eps * D, geos[n], h), K)
                Em = e_k(circle_eval(M - eps * D, geos[n], h), K)
                row["true_%g" % eps] = float((Ep - Em) / (2 * eps))
                row["true_min_onesided_drop_%g" % eps] = float(min(Ep, Em) - E0)
                if eps == 1e-4:
                    Fp = e_k(circle_eval(M + eps * D, geos[n], h, frozen=frz), K)
                    Fm = e_k(circle_eval(M - eps * D, geos[n], h, frozen=frz), K)
                    row["frozen_%g" % eps] = float((Fp - Fm) / (2 * eps))
            rows.append(row)
        ndim = int(mask[..., 0, 0].sum() * 7)
        tmax = max(abs(r["true_0.0001"]) for r in rows)
        fmax = max(abs(r["frozen_0.0001"]) for r in rows)
        rms_t = float(np.sqrt(np.mean([r["true_0.0001"] ** 2 for r in rows])))
        rms_f = float(np.sqrt(np.mean([r["frozen_0.0001"] ** 2 for r in rows])))
        res["derivatives"][tag] = dict(
            rows=rows, max_abs_true=float(tmax), max_abs_frozen=float(fmax),
            ratio_true_over_frozen_rms=float(rms_t / rms_f) if rms_f else None,
            ratio_true_over_frozen_per_dir=[abs(r["true_0.0001"]) / abs(r["frozen_0.0001"]) for r in rows],
            grad_norm_estimate_true=float(rms_t * np.sqrt(ndim)),
            grad_norm_estimate_frozen=float(rms_f * np.sqrt(ndim)), free_dim=ndim,
            min_onesided_drop_1em4=float(min(r["true_min_onesided_drop_0.0001"] for r in rows)),
            min_onesided_drop_1em3=float(min(r["true_min_onesided_drop_0.001"] for r in rows)),
            eps_consistency=[abs(r["true_0.0001"] - r["true_0.001"]) / max(abs(r["true_0.0001"]), 1e-300) for r in rows])
        log(tag, "true derivs", ["%.3e" % r["true_0.0001"] for r in rows],
            "frozen", ["%.3e" % r["frozen_0.0001"] for r in rows],
            "ratio rms %.1f" % res["derivatives"][tag]["ratio_true_over_frozen_rms"])

    # --- C3.5 (ii): own descent on the TRUE E_K (frames re-frozen each iteration for the
    #     search direction, line search on the true E_K), n32 K200, up to 100 iterations
    for tag, max_it, wall in (("n32_K200", 100, 600.0), ("n32_K50", 40, 300.0)):
        M, n, L, h, K = M_all[tag]
        mask = free_mask(n, h)
        Mc = M.copy()
        E = e_k(circle_eval(Mc, geos[n], h), K)
        E_start = E
        hist = [float(E)]
        t = None
        n_eval = 1
        fails = 0
        dir_check = None
        t_start = time.time()
        it = 0
        for it in range(max_it):
            g, _ = grad_frozen(Mc, geos[n], h, K, mask)
            gn = np.linalg.norm(g)
            d = -g / gn
            if it == 0:
                eps = 1e-4
                dd = (e_k(circle_eval(Mc + eps * d, geos[n], h), K)
                      - e_k(circle_eval(Mc - eps * d, geos[n], h), K)) / (2 * eps)
                dir_check = dict(true_derivative_along_minus_grad_frozen=float(dd),
                                 frozen_derivative_along_minus_grad_frozen=float(-gn))
                n_eval += 2
                t = 1e-2 / gn if dd < 0 else 1e-3 / gn
            accepted = False
            for _ in range(10):
                En = e_k(circle_eval(Mc + t * d, geos[n], h), K)
                n_eval += 1
                if En < E:
                    Mc = Mc + t * d
                    E = En
                    t *= 1.5
                    accepted = True
                    break
                t *= 0.4
            if not accepted:
                fails += 1
                # fallback: projected gradient in span{d, 3 random directions} by true derivatives
                dirs = [d] + [rand_dir(mask) for _ in range(3)]
                eps = 1e-4
                coef = []
                for Dd in dirs:
                    dd = (e_k(circle_eval(Mc + eps * Dd, geos[n], h), K)
                          - e_k(circle_eval(Mc - eps * Dd, geos[n], h), K)) / (2 * eps)
                    coef.append(dd)
                    n_eval += 2
                d2 = -sum(c * Dd for c, Dd in zip(coef, dirs))
                d2n = np.linalg.norm(d2)
                if d2n == 0:
                    break
                d2 /= d2n
                tt = 1e-3
                ok = False
                for _ in range(12):
                    En = e_k(circle_eval(Mc + tt * d2, geos[n], h), K)
                    n_eval += 1
                    if En < E:
                        Mc = Mc + tt * d2
                        E = En
                        ok = True
                        break
                    tt *= 0.4
                if not ok:
                    break
                t = max(t, 1e-3 / gn)
            hist.append(float(E))
            if time.time() - t_start > wall:
                break
        fr_end = frame(Mc, geos[n])
        acc_end = circle_eval(Mc, geos[n], h)
        res["descent"][tag] = dict(
            iterations=len(hist) - 1, E_K_start=float(E_start), E_K_end=float(E),
            drop=float(E_start - E), drop_rel=float((E_start - E) / E_start), history=hist,
            n_true_evals=n_eval, line_search_failures=fails, direction_check=dir_check,
            wall_s=time.time() - t_start, displacement_frobenius=float(np.linalg.norm(Mc - M)),
            displacement_max_entry=float(np.abs(Mc - M).max()),
            end_E_stat=float(acc_end["E_stat"]), end_kin_tot=float(acc_end["kin_tot"]),
            end_omega=float(K / (2 * acc_end["kin_tot"])), end_max_half=float(fr_end["half"].max()),
            end_min_gap=float(fr_end["gap"].min()),
            method="search direction = -grad of the frozen-a0 protocol re-frozen at each iterate; "
                   "step accepted only if the TRUE E_K (all frames refreshed) decreases; fallback = "
                   "projected true gradient in span{d, 3 random dirs} by central differences")
        log(tag, "descent: %d it, E_K %.5f -> %.5f (drop %.3e), evals %d, fails %d, wall %.0fs, dir_check %s" % (
            len(hist) - 1, E_start, E, E_start - E, n_eval, fails, time.time() - t_start, dir_check))


    verdicts(res)
    res["runtime_s"] = time.time() - T0
    out = os.path.join(ROOT, "data", "m5_32_r16_3_audit.json")
    write_and_print(res, out)


def verdicts(res):
    """All verdict rules from the measured numbers in res (re-runnable with --verdicts-only)."""
    def rd(a, b):
        return abs(a - b) / max(abs(b), 1e-300)
    # ------------------------------------------------------------ verdicts
    V = {}
    F = res["fields"]


    # C3.1
    devs = {}
    for tag in ("n32_K50", "n32_K200", "n48_K50", "n48_K200"):
        for key in ("E_K", "kin_tot", "kin_h", "kin_KP", "kin_reg", "omega", "E_stat", "E_h", "V4", "U", "K_P", "reg"):
            if key in PROD[tag]:
                devs[f"{tag}.{key}"] = rd(F[tag][key], PROD[tag][key])
    worst = max(devs.values())
    # rounding-aware: the producer's numbers are printed to 2-4 significant digits; a deviation
    # inside 0.6 units of the last printed digit is rounding, not a discrepancy
    beyond_rounding = {}
    for tag in ("n32_K50", "n32_K200", "n48_K50", "n48_K200"):
        for key in ("E_K", "kin_tot", "kin_h", "kin_KP", "kin_reg", "omega", "E_stat", "E_h", "V4", "U", "K_P", "reg"):
            if key in PROD[tag]:
                pv = PROD[tag][key]
                dec = len(repr(float(pv)).split(".")[1]) if "." in repr(float(pv)) else 0
                tol = 0.6 * 10.0 ** (-dec)
                if abs(F[tag][key] - pv) > tol and rd(F[tag][key], pv) > 0.005:
                    beyond_rounding[f"{tag}.{key}"] = float(abs(F[tag][key] - pv))
    fracs = [F[t]["kin_h_fraction"] for t in ("n32_K50", "n32_K200", "n48_K50", "n48_K200")]
    frac_ok = all(0.82 <= f <= 0.97 + 0.005 for f in fracs)
    V["C3.1"] = dict(verdict="CONFIRMED" if not beyond_rounding and frac_ok else ("QUALIFIED" if worst < 0.1 else "REFUTED"),
                     worst_reldev=float(worst), worst_key=max(devs, key=devs.get), reldevs=devs,
                     deviations_beyond_rounding=beyond_rounding, kin_h_fractions=fracs,
                     method="own circle-averaged evaluator (8 samples, every sample re-diagonalized) on the four end fields")
    # C3.2
    gaps = {t: (F[t]["bound_gap"], F[t]["omega_c_K"], F[t]["bound_ratio"]) for t in ("n32_K50", "n32_K200", "n48_K50", "n48_K200")}
    gdev = max(rd(F[t]["bound_gap"], PROD[t]["bound_gap"]) for t in gaps)
    allfar = all(v[2] >= 3.5 for v in gaps.values())
    V["C3.2"] = dict(verdict="CONFIRMED" if (allfar and gdev < 0.01) else ("QUALIFIED" if allfar else "REFUTED"),
                     gaps=gaps, worst_reldev=float(gdev),
                     R16_1_E_stat=dict(n32=F["r16_1_n32"]["E_stat"], n48=F["r16_1_n48"]["E_stat"]),
                     method="own E_K minus own E_stat of the R16-1 end fields vs omega_c K")
    # C3.3
    g3 = {t: F[t]["geometry"] for t in gaps}
    c33 = []
    for t in gaps:
        c33.append(rd(g3[t]["max_half"], PROD[t]["max_half"]) < 0.02)
        c33.append(rd(g3[t]["rms_radius"], PROD[t]["rms"]) < 0.02)
        c33.append(max(abs(a - b) for a, b in zip(sorted(g3[t]["quad_eigs"]), sorted(PROD[t]["quad"]))) < 0.01)
        c33.append(abs(g3[t]["tilt"] - PROD[t]["tilt"]) < 0.01)
    lower_member_n48_K50_below = g3["n48_K50"]["l3_min"] < -0.2
    lower_member_n48_K200_below = g3["n48_K200"]["l3_min"] < -0.2
    n32_K50_triple_ok = max(abs(a - b) for a, b in zip(g3["n32_K50"]["max_split_triple"], (0.949, 0.949, -0.066))) < 0.01
    n32_K200_triple_ok = max(abs(a - b) for a, b in zip(g3["n32_K200"]["max_split_triple"], (0.923, 0.675, -0.616))) < 0.01
    all_num = all(c33) and n32_K50_triple_ok and n32_K200_triple_ok
    V["C3.3"] = dict(verdict=("CONFIRMED" if all_num and lower_member_n48_K50_below and lower_member_n48_K200_below
                              else ("QUALIFIED" if all_num else "REFUTED")),
                     numeric_checks_passed=int(sum(c33)), numeric_checks=len(c33),
                     n32_triples_match=[bool(n32_K50_triple_ok), bool(n32_K200_triple_ok)],
                     n48_K50_pair_lower_member_min=g3["n48_K50"]["l3_min"],
                     n48_K50_pair_upper_member_max=g3["n48_K50"]["l2_max"],
                     n48_K200_pair_lower_member_min=g3["n48_K200"]["l3_min"],
                     n48_K50_lower_below_minus02=bool(lower_member_n48_K50_below),
                     n48_K200_lower_below_minus02=bool(lower_member_n48_K200_below),
                     cells_pair_outside_plateau=dict(n48_K50=g3["n48_K50"]["cells_pair_outside_plateau"],
                                                     n48_K200=g3["n48_K200"]["cells_pair_outside_plateau"]),
                     quad_shapes={t: (g3[t]["quad_shape"], g3[t]["quad_axis_dot_111"]) for t in gaps},
                     method="own rho^2 map (half split squared), rho^2-weighted rms radius, quadrupole, tilt, triple at the max-split cell")
    # C3.4
    esc = {t: g3[t]["escapes"] for t in gaps}
    m2 = {t: g3[t]["spin2_mean_abs"] for t in gaps}
    exp_ok = (esc["n32_K50"]["d"] and esc["n32_K50"]["b"]
              and not any(esc[t][k] for t in ("n32_K200", "n48_K50", "n48_K200") for k in esc[t]))
    m_ok = all(v <= 0.12 + 0.02 for v in m2.values())
    V["C3.4"] = dict(verdict="CONFIRMED" if exp_ok and m_ok else ("QUALIFIED" if exp_ok else "REFUTED"),
                     escapes=esc, spin2_mean_abs=m2, min_gap={t: g3[t]["min_gap"] for t in gaps},
                     note="the iteration at which (d) fired cannot be audited from the end state; the end-state escape flags are",
                     method="own escape reads on the end fields (own zeta frame for (c): t1 = n x z_hat, t2 = n x t1)")
    # C3.5
    dv = res["derivatives"]
    i_ok = dv["n32_K200"]["max_abs_true"] > 1e-2 and dv["n48_K50"]["max_abs_true"] > 1e-2 and dv["n48_K200"]["max_abs_true"] > 1e-2
    ii_ok = all(res["descent"][t]["drop"] > 0 for t in res["descent"])
    onesided = all(dv[t]["min_onesided_drop_1em3"] < 0 for t in dv)
    mag_dev = {t: rd(dv[t]["max_abs_true"], PROD[t]["deriv_true"]) for t in dv}
    V["C3.5"] = dict(verdict="CONFIRMED" if (i_ok and ii_ok) else ("QUALIFIED" if (i_ok or ii_ok) else "REFUTED"),
                     test_i_max_abs_true={t: dv[t]["max_abs_true"] for t in dv},
                     test_ii_descent_drop={t: res["descent"][t]["drop"] for t in res["descent"]},
                     onesided_drop_exists_all_fields=bool(onesided),
                     producer_max_abs_true={t: PROD[t]["deriv_true"] for t in dv},
                     order_of_magnitude_reldev=mag_dev,
                     dE_dK_test="NOT AUDITED (needs the producer's relaxation protocol)",
                     method="central differences of own true E_K (eps 1e-4 and 1e-3, 3 unit random directions on the free 7-component block) + own line-searched descent on the true E_K")
    # C3.6
    ratios = {t: dv[t]["ratio_true_over_frozen_rms"] for t in dv}
    per = {t: dv[t]["ratio_true_over_frozen_per_dir"] for t in dv}
    big = all(r >= 3.0 for r in ratios.values())
    inrange = all(3.0 <= r <= 200.0 for r in ratios.values())
    V["C3.6"] = dict(verdict="CONFIRMED" if inrange else ("QUALIFIED" if big else "REFUTED"),
                     ratio_rms=ratios, ratio_per_direction=per,
                     frozen_protocol="own: a0, the circle rotations and W frozen at the reference; rho2, V4, U and all lattice derivatives exact",
                     method="same random directions, true vs frozen central differences (eps 1e-4)")
    # C3.7
    n90 = {t: g3[t]["n_cells_90pct_rho2"] for t in gaps}
    shell_r = {t: g3[t]["shell_rms_radius"] for t in gaps}
    top = {t: [c["cell"] for c in g3[t]["top_cells"][:3]] for t in gaps}
    r_ok = (abs(shell_r["n32_K50"] - 4.92) < 0.05 and abs(shell_r["n32_K200"] - 4.92) < 0.05
            and abs(shell_r["n48_K50"] - 3.27) < 0.05 and abs(shell_r["n48_K200"] - 3.27) < 0.05)
    V["C3.7"] = dict(verdict="CONFIRMED" if r_ok and all(v <= 30 for v in n90.values()) else ("QUALIFIED" if r_ok else "REFUTED"),
                     n_cells_90pct_rho2=n90, n_cells_50pct_rho2={t: g3[t]["n_cells_50pct_rho2"] for t in gaps},
                     shell_rms_radius=shell_r, top3_cells=top,
                     n64_statement="NOT AUDITED (outside the brief)",
                     method="own rho^2 map: cells sorted by rho^2, count to 90 percent; rms radius of that shell")
    # C3.3 addendum: which body diagonal the quadrupole axis follows (the producer says (1,1,1))
    diags = [np.array(v) / np.sqrt(3.0) for v in ((1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1))]
    for t in gaps:
        ax = np.array(g3[t]["quad_axis"])
        ov = [float(abs(ax @ dg)) for dg in diags]
        V["C3.3"].setdefault("quad_axis_body_diagonal_overlap", {})[t] = dict(
            dot_111=ov[0], best=["(1,1,1)", "(1,1,-1)", "(1,-1,1)", "(-1,1,1)"][int(np.argmax(ov))], best_overlap=max(ov))
    # C3.6 refined rule: the omitted part (true - frozen) relative to the true derivative
    rel_om = {}
    for t in dv:
        tr = np.array([r["true_0.0001"] for r in dv[t]["rows"]])
        fz = np.array([r["frozen_0.0001"] for r in dv[t]["rows"]])
        rel_om[t] = float(np.sqrt(np.mean((tr - fz) ** 2)) / np.sqrt(np.mean(tr ** 2)))
    V["C3.6"]["omitted_part_rel_rms"] = rel_om
    not_small = all(v >= 0.5 for v in rel_om.values())
    V["C3.6"]["verdict"] = "CONFIRMED" if inrange else ("QUALIFIED" if not_small else "REFUTED")
    V["C3.6"]["rule"] = ("CONFIRMED if every field's true/frozen rms ratio is in [3, 200]; QUALIFIED if the "
                         "omitted part |true - frozen| is >= 0.5 of |true| (rms over the 3 directions) on every field "
                         "but the 3x-200x range fails somewhere; REFUTED otherwise")
    res["verdicts"] = V
    tally = {k: sum(1 for v in V.values() if v["verdict"] == k) for k in ("CONFIRMED", "QUALIFIED", "REFUTED")}
    res["tally"] = tally
    return V, tally


def write_and_print(res, out):
    V, tally = res["verdicts"], res["tally"]
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("\n| Claim | Verdict | Method |")
    print("| --- | --- | --- |")
    for k, v in V.items():
        print("| %s | %s | %s |" % (k, v["verdict"], v["method"]))
    print("tally", tally, "runtime %.0fs" % res["runtime_s"], "->", os.path.relpath(out, ROOT))

if __name__ == "__main__":
    if "--verdicts-only" in sys.argv:
        out = os.path.join(ROOT, "data", "m5_32_r16_3_audit.json")
        with open(out) as fh:
            res = json.load(fh)
        res["verdicts_recomputed"] = True
        verdicts(res)
        write_and_print(res, out)
    else:
        main()
