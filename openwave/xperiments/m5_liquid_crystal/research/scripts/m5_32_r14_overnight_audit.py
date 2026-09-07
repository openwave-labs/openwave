"""M5.32 R14 overnight rungs: adversarial audit of R14-D2 (the coexistence wall on the
4x4 field under V' = V4 + mu (m2 - m3)^2) and R14-B2 (the norm-200 LP corner under the
fixed-J descent).  Independent implementation; the producer's scripts and results
(m5_32_r14_d2_wall.*, m5_32_r14_b2_vertex.*) were never opened.

EQUATIONS FIRST (own, checked against the registries in stage W5)
-----------------------------------------------------------------
Diagonal planar states M(z) = diag(g, 1, m2, m3), N = M eta, spectrum (-g, 1, m2, m3),
f(x) = (x + g)(x - 1), s = m2 - m3.
    V4      = W1 sum_{p=1..4} (tr N^p - C_p)^2,  C_p = (-g)^p + 1 + delta^p
    V'      = V4 + mu s^2
    iota    = [f(m2) f(m3)]^2 s^2                       (K_P^h inertia per volume)
    K_P^h static per volume = (1/2)[f(m2)^4 m2'^2 + f(m3)^4 m3'^2]
    I1 flank inertia        = 8 s^2 (m2' - m3')^2
    F[m2, m3] = int dz { K_P^h static + V' - omega^2 [iota + 8 s^2 (m2' - m3')^2] }
    exterior (t*, t*) = argmin_t V4(t, t);  crossing omega_*^2 = inf_m [V'(m) - V'(t*)] / iota(m)
    sigma = F[wall] - F[uniform]  (per area);  Bogomolny: sigma = min_path int sqrt(2 U) ds_g,
    U = V_eff - V_eff(ext), ds_g^2 = f2^4 dm2^2 + f3^4 dm3^2 - 16 omega^2 s^2 (dm2 - dm3)^2,
    and on the minimizer (1/2) g m' m' = U pointwise (the first integral).
Wall methods used here: (A) Jacobi geodesic in the plane + the first integral for the
profile; (B) L-BFGS on an own midpoint discretization of F on a z line with pinned ends.

B2: E = E_u + V4 + sum_k chat_k s_k + J^2 / (4 kin_tot), own densities for I6 = R^2 with
R = sum_{ij} F_ij[j, i], R_G = G_cd T^cd, K_lambda from the sorted spectrum; gradients from
the registries, gated by finite differences of the OWN energies; own FIRE and own plain
gradient descent (at most 100 steps); own localization diagnostic.

Run:  /opt/anaconda3/envs/master312/bin/python3 m5_32_r14_overnight_audit.py [--stage d2|b2|all]
Out:  ../data/m5_32_r14_overnight_audit.json
"""
import sys

ARGV = list(sys.argv)                       # captured before any import

import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
DATA = os.path.join(RES, "data")
OUT = os.path.join(DATA, "m5_32_r14_overnight_audit.json")
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CM = _load("m5_32_r13w_common", "m5_32_r13w_common.py")
TM = _load("m5_32_r14_terms", "m5_32_r14_terms.py")
INS4, LAG = CM.INS4, CM.LAG
G, DELTA, W1 = 8.0, 0.3, INS4.W1
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
ETA_D = np.diag(ETA)
RESULTS = {"argv": ARGV}


# ============================================================ own reduced densities (D2)
def f_of(x):
    return (x + G) * (x - 1.0)


def fp_of(x):
    return 2.0 * x + G - 1.0


def V4_of(m2, m3):
    m2, m3 = np.asarray(m2, float), np.asarray(m3, float)
    tot = 0.0
    for p in range(1, 5):
        tr = (-G) ** p + 1.0 + m2 ** p + m3 ** p
        Cp = (-G) ** p + 1.0 + DELTA ** p
        tot = tot + (tr - Cp) ** 2
    return W1 * tot


def dV4_of(m2, m3):
    m2, m3 = np.asarray(m2, float), np.asarray(m3, float)
    g2 = 0.0
    g3 = 0.0
    for p in range(1, 5):
        d = (m2 ** p + m3 ** p - DELTA ** p)
        g2 = g2 + 2.0 * d * p * m2 ** (p - 1)
        g3 = g3 + 2.0 * d * p * m3 ** (p - 1)
    return W1 * g2, W1 * g3


def iota_of(m2, m3):
    return (f_of(m2) * f_of(m3)) ** 2 * (m2 - m3) ** 2


def diota_of(m2, m3):
    f2, f3, s = f_of(m2), f_of(m3), m2 - m3
    c = (f2 * f3) ** 2
    return (2.0 * f2 * fp_of(m2) * f3 ** 2 * s ** 2 + 2.0 * c * s,
            2.0 * f3 * fp_of(m3) * f2 ** 2 * s ** 2 - 2.0 * c * s)


def Vp_of(m2, m3, mu):
    return V4_of(m2, m3) + mu * (m2 - m3) ** 2


def dVp_of(m2, m3, mu):
    g2, g3 = dV4_of(m2, m3)
    return g2 + 2.0 * mu * (m2 - m3), g3 - 2.0 * mu * (m2 - m3)


def Veff_of(m2, m3, mu, om2):
    return Vp_of(m2, m3, mu) - om2 * iota_of(m2, m3)


def dVeff_of(m2, m3, mu, om2):
    a2, a3 = dVp_of(m2, m3, mu)
    b2, b3 = diota_of(m2, m3)
    return a2 - om2 * b2, a3 - om2 * b3


def hess_fd(fun, x, eps=1e-4):
    x = np.asarray(x, float)
    n = len(x)
    Hm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i] = eps
            ej = np.zeros(n); ej[j] = eps
            Hm[i, j] = (fun(x + ei + ej) - fun(x + ei - ej) - fun(x - ei + ej) + fun(x - ei - ej)) / (4 * eps * eps)
    return Hm


def t_star():
    r = minimize(lambda t: V4_of(t[0], t[0]), [0.15], method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-20})
    return float(r.x[0])


def crossing(mu, tst, box=(-2.5, 2.5), npts=501):
    """omega_*^2 = inf over the plane of r = (V' - V'(ext)) / iota; grid + Nelder-Mead."""
    Vext = Vp_of(tst, tst, mu)
    x = np.linspace(box[0], box[1], npts)
    M2, M3 = np.meshgrid(x, x, indexing="ij")
    io = iota_of(M2, M3)
    dV = Vp_of(M2, M3, mu) - Vext
    ok = (io > 1e-10) & (dV > 0)
    r = np.where(ok, dV / np.where(ok, io, 1.0), np.inf)
    idx = np.argsort(r, axis=None)[:12]
    best = None
    for k in idx:
        i, j = np.unravel_index(k, r.shape)
        def fun(m):
            io_ = iota_of(m[0], m[1]); dv = Vp_of(m[0], m[1], mu) - Vext
            return dv / io_ if (io_ > 1e-12 and dv > 0) else 1e9
        res = minimize(fun, [M2[i, j], M3[i, j]], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-22, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    m2, m3 = float(best.x[0]), float(best.x[1])
    om2 = float(best.fun)
    # fold the mirror (m2 <-> m3) to the producer's orientation (m2 < m3 for the D audit states? they quote (-1.054, -0.243))
    return om2, (m2, m3), float(Vext)


def local_min_check(mu, om2, m):
    Hm = hess_fd(lambda x: Veff_of(x[0], x[1], mu, om2), m)
    ev = np.linalg.eigvalsh(Hm)
    gr = dVeff_of(m[0], m[1], mu, om2)
    return {"hess_eigs": [float(e) for e in ev], "grad": [float(gr[0]), float(gr[1])]}


def veff_global_scan(mu, om2, box=(-4.0, 5.0), npts=901):
    x = np.linspace(box[0], box[1], npts)
    M2, M3 = np.meshgrid(x, x, indexing="ij")
    V = Veff_of(M2, M3, mu, om2)
    k = np.argmin(V)
    i, j = np.unravel_index(k, V.shape)
    # strict 8-neighbor local minima list (top 6 deepest)
    inner = V[1:-1, 1:-1]
    ismin = np.ones_like(inner, dtype=bool)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ismin &= inner < V[1 + di:npts - 1 + di, 1 + dj:npts - 1 + dj]
    ii, jj = np.nonzero(ismin)
    vals = inner[ii, jj]
    order = np.argsort(vals)[:6]
    mins = [[float(x[ii[o] + 1]), float(x[jj[o] + 1]), float(vals[o])] for o in order]
    return {"grid_global_min": [float(x[i]), float(x[j]), float(V[i, j])], "local_minima_deepest": mins}


def straight_barrier(mu, om2, mext, mint, n=4001):
    t = np.linspace(0, 1, n)
    m2 = mext[0] + t * (mint[0] - mext[0])
    m3 = mext[1] + t * (mint[1] - mext[1])
    U = Veff_of(m2, m3, mu, om2) - Veff_of(mext[0], mext[1], mu, om2)
    return float(np.max(U)), float(t[np.argmax(U)])


# ------------------------------------------------ the metric and the Bogomolny integrals
def metric_g(m2, m3, om2):
    """g_ab with (1/2) g_ab m_a' m_b' = (1/2)[f2^4 m2'^2 + f3^4 m3'^2] - 8 om2 s^2 (m2' - m3')^2."""
    s = m2 - m3
    c = 16.0 * om2 * s * s
    g22 = f_of(m2) ** 4 - c
    g33 = f_of(m3) ** 4 - c
    g23 = c
    return g22, g33, g23


def subsample(P, nsub):
    """linear sub-sampling of a polyline (nsub points per segment, endpoints kept)."""
    P = np.asarray(P)
    if nsub <= 1:
        return P
    t = np.linspace(0, 1, nsub + 1)[:-1]
    a, b = P[:-1], P[1:]
    Q = a[:, None, :] * (1 - t)[None, :, None] + b[:, None, :] * t[None, :, None]
    return np.vstack([Q.reshape(-1, 2), P[-1:]])


def bogo_path_integral(path, mu, om2, Uext, variant="correct", nsub=8):
    """int sqrt(2U) ds_g along a polyline (midpoint U per sub-segment, metric at the midpoint;
    every waypoint segment is sub-sampled nsub times so a long segment cannot skip a barrier).
    variant 'brief_kappa' reproduces the R14-D brief formula sqrt(2 kappa DV) ds with
    kappa = (f2^4 + f3^4)/8 and ds the split increment (a sqrt-2 slip, flagged by the R14-D audit);
    variant 'flat' uses g = identity (the wrong-metric mutant)."""
    P = subsample(path, nsub)
    a, b = P[:-1], P[1:]
    mid = 0.5 * (a + b)
    d = b - a
    U = np.maximum(Veff_of(mid[:, 0], mid[:, 1], mu, om2) - Uext, 0.0)
    if variant == "flat":
        ds = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
        return float(np.sum(np.sqrt(2.0 * U) * ds))
    if variant == "brief_kappa":
        kappa = (f_of(mid[:, 0]) ** 4 + f_of(mid[:, 1]) ** 4) / 8.0
        dsplit = np.abs(d[:, 0] - d[:, 1])
        return float(np.sum(np.sqrt(2.0 * kappa * U) * dsplit))
    g22, g33, g23 = metric_g(mid[:, 0], mid[:, 1], om2)
    ds2 = g22 * d[:, 0] ** 2 + g33 * d[:, 1] ** 2 - 2.0 * g23 * d[:, 0] * d[:, 1]
    return float(np.sum(np.sqrt(2.0 * U) * np.sqrt(np.maximum(ds2, 0.0))))


def geodesic_path(mu, om2, mext, mint, Uext, nseg=60, init="straight"):
    """minimize the Bogomolny integral over the interior waypoints (L-BFGS-B, numerical gradient)."""
    t = np.linspace(0, 1, nseg + 1)[:, None]
    P0 = np.asarray(mext)[None, :] * (1 - t) + np.asarray(mint)[None, :] * t
    if init == "bent":
        perp = np.array([-(mint[1] - mext[1]), mint[0] - mext[0]])
        P0 = P0 + 0.3 * np.sin(np.pi * t) * perp[None, :]
    x0 = P0[1:-1].ravel()

    def fun(x):
        P = np.vstack([mext, x.reshape(-1, 2), mint])
        return bogo_path_integral(P, mu, om2, Uext)
    r = minimize(fun, x0, method="L-BFGS-B", options={"maxiter": 20000, "maxfun": 400000, "ftol": 1e-15, "gtol": 1e-12})
    P = np.vstack([mext, r.x.reshape(-1, 2), mint])
    return float(r.fun), P, r


def profile_from_path(P, mu, om2, Uext, clip=1e-9, nsub=32):
    """first integral: dz = ds_g / sqrt(2U); returns z (centered), m2, m3 along the path."""
    P = subsample(P, nsub)
    a, b = P[:-1], P[1:]
    mid = 0.5 * (a + b)
    d = b - a
    U = np.maximum(Veff_of(mid[:, 0], mid[:, 1], mu, om2) - Uext, clip)
    g22, g33, g23 = metric_g(mid[:, 0], mid[:, 1], om2)
    ds = np.sqrt(np.maximum(g22 * d[:, 0] ** 2 + g33 * d[:, 1] ** 2 - 2.0 * g23 * d[:, 0] * d[:, 1], 0.0))
    dz = ds / np.sqrt(2.0 * U)
    z = np.concatenate([[0.0], np.cumsum(dz)])
    return z, P[:, 0], P[:, 1]


# ------------------------------------------------ method B: own midpoint discretization of F on a line
def F_line(m, h, mu, om2, Uext, grad=False):
    """m: (N, 2) profile on a uniform z grid of spacing h; returns F - F[uniform] (the excess over
    the exterior plateau, both ends pinned); midpoint metric, trapezoid potential."""
    a, b = m[:-1], m[1:]
    mid = 0.5 * (a + b)
    d = (b - a) / h
    f2, f3 = f_of(mid[:, 0]), f_of(mid[:, 1])
    s = mid[:, 0] - mid[:, 1]
    kin = 0.5 * (f2 ** 4 * d[:, 0] ** 2 + f3 ** 4 * d[:, 1] ** 2) - 8.0 * om2 * s ** 2 * (d[:, 0] - d[:, 1]) ** 2
    U = Veff_of(m[:, 0], m[:, 1], mu, om2) - Uext
    w = np.ones(len(m)); w[0] = w[-1] = 0.5
    E = h * np.sum(kin) + h * np.sum(w * U)
    if not grad:
        return E
    Gm = np.zeros_like(m)
    # potential part
    u2, u3 = dVeff_of(m[:, 0], m[:, 1], mu, om2)
    Gm[:, 0] += h * w * u2
    Gm[:, 1] += h * w * u3
    # kinetic part: kin_k depends on a = m_k, b = m_{k+1}
    dk_dd2 = f2 ** 4 * d[:, 0] - 16.0 * om2 * s ** 2 * (d[:, 0] - d[:, 1])
    dk_dd3 = f3 ** 4 * d[:, 1] + 16.0 * om2 * s ** 2 * (d[:, 0] - d[:, 1])
    dk_dmid2 = 2.0 * f2 ** 3 * fp_of(mid[:, 0]) * d[:, 0] ** 2 - 16.0 * om2 * s * (d[:, 0] - d[:, 1]) ** 2
    dk_dmid3 = 2.0 * f3 ** 3 * fp_of(mid[:, 1]) * d[:, 1] ** 2 + 16.0 * om2 * s * (d[:, 0] - d[:, 1]) ** 2
    # d = (b - a)/h, mid = (a + b)/2
    Gm[:-1, 0] += h * (-dk_dd2 / h + 0.5 * dk_dmid2)
    Gm[1:, 0] += h * (dk_dd2 / h + 0.5 * dk_dmid2)
    Gm[:-1, 1] += h * (-dk_dd3 / h + 0.5 * dk_dmid3)
    Gm[1:, 1] += h * (dk_dd3 / h + 0.5 * dk_dmid3)
    return E, Gm


def relax_line(mu, om2, mext, mint, Uext, L, h, width_guess=2000.0, init=None, maxiter=30000):
    N = int(round(L / h)) + 1
    z = (np.arange(N) - (N - 1) / 2.0) * h
    if init is None:
        t = 0.5 * (1.0 + np.tanh(z / (width_guess / 2.0)))
        m = np.asarray(mext)[None, :] * (1 - t[:, None]) + np.asarray(mint)[None, :] * t[:, None]
    else:
        m = init.copy()
    m[0], m[-1] = mext, mint
    x0 = m[1:-1].ravel()

    def fun(x):
        mm = np.vstack([mext, x.reshape(-1, 2), mint])
        E, Gm = F_line(mm, h, mu, om2, Uext, grad=True)
        return E, Gm[1:-1].ravel()
    r = minimize(fun, x0, jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter, "maxfun": 2 * maxiter, "ftol": 1e-16, "gtol": 1e-11, "maxcor": 30})
    m = np.vstack([mext, r.x.reshape(-1, 2), mint])
    return z, m, float(r.fun), {"nit": int(r.nit), "nfev": int(r.nfev), "status": int(r.status), "msg": str(r.message)}


def tail_decay(z, s, s_plateau, side, lo=1e-2, hi=1e-5):
    """exponential decay length of |s - s_plateau| on one side (log-linear fit over the band where
    the residual relative to the total jump lies in [hi, lo]); side = 'left' or 'right'."""
    res = np.abs(s - s_plateau)
    jump = np.abs(s[-1] - s[0])
    band = (res > hi * jump) & (res < lo * jump)
    if side == "left":
        band &= z < 0
    else:
        band &= z > 0
    if np.sum(band) < 5:
        return float("nan"), int(np.sum(band))
    A = np.vstack([z[band], np.ones(np.sum(band))]).T
    slope, _ = np.linalg.lstsq(A, np.log(res[band]), rcond=None)[0]
    return float(1.0 / abs(slope)), int(np.sum(band))


def width_10_90(z, s, s_ext, s_int):
    lev = s_ext + (s_int - s_ext) * np.array([0.1, 0.9])
    frac = (s - s_ext) / (s_int - s_ext)
    z10 = np.interp(0.1, frac, z) if frac[0] < frac[-1] else np.interp(0.1, frac[::-1], z[::-1])
    z90 = np.interp(0.9, frac, z) if frac[0] < frac[-1] else np.interp(0.9, frac[::-1], z[::-1])
    return float(abs(z90 - z10)), [float(l) for l in lev]


def first_integral_check(z, m, mu, om2, Uext):
    """(1/2) g m'm' - U on the interior (central differences); returns max |ratio - 1| where U > 1e-3 max U."""
    dz = z[1] - z[0]
    d2 = np.gradient(m[:, 0], dz)
    d3 = np.gradient(m[:, 1], dz)
    s = m[:, 0] - m[:, 1]
    kin = 0.5 * (f_of(m[:, 0]) ** 4 * d2 ** 2 + f_of(m[:, 1]) ** 4 * d3 ** 2) - 8.0 * om2 * s ** 2 * (d2 - d3) ** 2
    U = Veff_of(m[:, 0], m[:, 1], mu, om2) - Uext
    sel = U > 1e-3 * np.max(U)
    return float(np.max(np.abs(kin[sel] / U[sel] - 1.0))), float(np.sum(kin) * dz), float(np.sum(U) * dz)


# ------------------------------------------------ slab embedding + registry read (W5)
def slab_of(m2, m3, h):
    nz = len(m2)
    M = np.zeros((4, 4, nz, 4, 4))
    M[..., 0, 0] = G
    M[..., 1, 1] = 1.0
    M[..., 2, 2] = m2[None, None, :]
    M[..., 3, 3] = m3[None, None, :]
    cfg = INS4.base_cfg(s=-1.0, g=G, n=4, L=4.0 * h, delta=DELTA)
    return M, cfg


def registry_slab_read(M, cfg):
    a0 = CM.a0_local(M)
    e_u, e_v = INS4.e_parts(M, cfg)
    area = (4.0 * cfg["h"]) ** 2
    return {"E_u": float(e_u) / area, "V4": float(e_v) / area,
            "KPh_static": TM.static_energy("K_P_h", M, cfg) / area,
            "KPh_kin": TM.kin_energy("K_P_h", M, a0, cfg) / area,
            "I1_kin": float(INS4.kin_of(M, a0, cfg)) / area}


def own_reduced_sums(m2, m3, h, stencil="sym"):
    """own reduced sums per unit area on the certified stencil conventions (fwd/bwd branches
    averaged, boundary derivative 0 on the missing side); 'central' is the mutant."""
    n = len(m2)
    out = {"V4": h * float(np.sum(V4_of(m2, m3))), "KPh_kin": h * float(np.sum(iota_of(m2, m3))), "E_u": 0.0}
    f2, f3, s = f_of(m2), f_of(m3), m2 - m3
    ks, ki = 0.0, 0.0
    if stencil == "central":
        branches = [("cen", 1.0)]
    else:
        branches = [("fwd", 0.5), ("bwd", 0.5)]
    for br, wt in branches:
        d2, d3 = np.zeros(n), np.zeros(n)
        if br == "fwd":
            d2[:-1] = (m2[1:] - m2[:-1]) / h; d3[:-1] = (m3[1:] - m3[:-1]) / h
        elif br == "bwd":
            d2[1:] = (m2[1:] - m2[:-1]) / h; d3[1:] = (m3[1:] - m3[:-1]) / h
        else:
            d2 = np.gradient(m2, h); d3 = np.gradient(m3, h)
        ks += wt * 0.5 * float(np.sum(f2 ** 4 * d2 ** 2 + f3 ** 4 * d3 ** 2))
        ki += wt * 8.0 * float(np.sum(s ** 2 * (d2 - d3) ** 2))
    out["KPh_static"] = h * ks
    out["I1_kin"] = h * ki
    return out


def rel(a, b):
    return float(abs(a - b) / max(abs(b), 1e-300))


# ============================================================ STAGE D2
def stage_d2():
    R = {}
    tst = t_star()
    R["t_star"] = tst
    R["V4_ext"] = float(V4_of(tst, tst))
    log(f"t* = {tst:.6f}  V4(t*,t*) = {R['V4_ext']:.4e}")
    # W5 part 1: reduced densities vs the registry on a random smooth diagonal profile
    rng = np.random.default_rng(7)
    h = 8.0
    nz = 64
    z = np.arange(nz) * h
    Lz = nz * h
    m2 = 0.15 - 0.6 * np.sin(2 * np.pi * z / Lz) + 0.2 * np.cos(4 * np.pi * z / Lz + 1.0) + 0.05 * rng.normal()
    m3 = 0.15 + 0.3 * np.sin(2 * np.pi * z / Lz + 0.7) - 0.15 * np.cos(6 * np.pi * z / Lz) + 0.05 * rng.normal()
    M, cfg = slab_of(m2, m3, h)
    reg = registry_slab_read(M, cfg)
    own = own_reduced_sums(m2, m3, h)
    mut = own_reduced_sums(m2, m3, h, stencil="central")
    a0_G3 = CM.G3 @ M - M @ CM.G3
    kin_G3 = float(INS4.kin_of(M, a0_G3, cfg)) / (4 * h) ** 2
    W5a = {k: {"registry": reg[k], "own": own[k], "rel": rel(own[k], reg[k]), "mutant_central_rel": rel(mut[k], reg[k])} for k in reg}
    W5a["mutant_G3_clock_I1_kin"] = kin_G3
    W5a["a0_local_equals_G1_clock"] = float(np.max(np.abs(np.abs(CM.a0_local(M)) - np.abs(CM.a0_G1(M)))))
    R["W5_random_profile"] = W5a
    log("W5 random profile: " + ", ".join(f"{k} rel {W5a[k]['rel']:.1e}" for k in reg))

    for mu in (1e-2, 1e-3):
        key = f"mu_{mu:g}"
        S = {}
        om2, mint, Vext = crossing(mu, tst)
        om = float(np.sqrt(om2))
        S["omega2_star"], S["omega_star"], S["interior"] = om2, om, list(mint)
        S["interior_split"] = mint[0] - mint[1]
        mext = (tst, tst)
        Uext = float(Veff_of(tst, tst, mu, om2))
        S["Veff_ext"] = Uext
        S["Veff_int"] = float(Veff_of(mint[0], mint[1], mu, om2))
        S["ext_check"] = local_min_check(mu, om2, mext)
        S["int_check"] = local_min_check(mu, om2, mint)
        S["global_scan_at_omega_star"] = veff_global_scan(mu, om2)
        S["barrier_straight"], S["barrier_straight_t"] = straight_barrier(mu, om2, mext, mint)
        # exterior instability (the top of the coexistence window)
        Hext = hess_fd(lambda x: Vp_of(x[0], x[1], mu), mext)
        # V_ss/2 along the split unit direction (1,-1)/sqrt2 against f(t*)^4: om_inst^2 = (V_ss/2)/f^4 with V_ss = d^2V/ds^2, s = m2 - m3
        e = np.array([1.0, -1.0]) / np.sqrt(2.0)
        Vss = float(e @ Hext @ e) / 2.0        # d^2V/ds^2 (s = sqrt2 * coordinate along e)
        S["omega2_inst_ext"] = (Vss / 2.0) / f_of(tst) ** 4
        log(f"{key}: omega_*^2 {om2:.4e} omega_* {om:.4e} interior {mint} split {S['interior_split']:.4f} "
            f"barrier(straight) {S['barrier_straight']:.3e} inst^2 {S['omega2_inst_ext']:.3e}")
        # ---- W2: Bogomolny integrals
        Pstr = np.vstack([mext, mint])
        Pstr_fine = np.array(mext)[None, :] + np.linspace(0, 1, 2001)[:, None] * (np.array(mint) - np.array(mext))[None, :]
        S["bogo_straight"] = bogo_path_integral(Pstr_fine, mu, om2, Uext)
        S["bogo_straight_brief_kappa"] = bogo_path_integral(Pstr_fine, mu, om2, Uext, variant="brief_kappa")
        S["bogo_straight_flat_metric"] = bogo_path_integral(Pstr_fine, mu, om2, Uext, variant="flat")
        S["bogo_straight_over_sqrt2"] = S["bogo_straight"] / np.sqrt(2.0)
        S["producer_straight"] = 3.13 if mu == 1e-2 else 0.49
        S["ratio_bogo_straight_to_producer"] = S["bogo_straight"] / S["producer_straight"]
        geo = {}
        best = None
        for nseg, init in ((40, "straight"), (80, "straight"), (80, "bent"), (160, "straight")):
            val, P, r = geodesic_path(mu, om2, mext, mint, Uext, nseg=nseg, init=init)
            geo[f"nseg{nseg}_{init}"] = {"sigma": val, "nit": int(r.nit), "success": bool(r.success)}
            if best is None or val < best[0]:
                best = (val, P)
        S["bogo_geodesic"] = geo
        S["sigma_A_geodesic"] = best[0]
        Pbest = best[1]
        S["geodesic_path_sample"] = Pbest[:: max(1, len(Pbest) // 16)].tolist()
        # the min-tension path's excursion: max distance from the straight segment
        v = np.array(mint) - np.array(mext)
        vn = v / np.linalg.norm(v)
        perp = np.array([-vn[1], vn[0]])
        S["geodesic_max_offset"] = float(np.max(np.abs((Pbest - np.array(mext)) @ perp)))
        log(f"{key}: Bogomolny straight {S['bogo_straight']:.4f} (brief-kappa variant {S['bogo_straight_brief_kappa']:.4f}), geodesic {best[0]:.4f}")
        # profile from the first integral (method A): the 10-90 width from the geodesic path
        zA, m2A, m3A = profile_from_path(Pbest, mu, om2, Uext, clip=1e-14)
        sA = m2A - m3A
        s_int = float(mint[0] - mint[1])
        wA, _ = width_10_90(zA, sA, 0.0, s_int)
        S["width_10_90_A_first_integral"] = wA
        # ---- method B: relaxation on the line (own discretization), two spacings and two lengths
        Lp, hp = (12000.0, 8.0) if mu == 1e-2 else (40000.0, 20.0)
        runs = {}
        prof_keep = None
        prof_long = None
        for (L, hh, init) in ((Lp, hp / 2, "tanh"), (Lp, hp, "tanh"), (2 * Lp, hp, "tanh"), (Lp / 4, hp / 2, "tanh"),
                              (Lp, hp, "A"), (2 * Lp, hp, "A")):
            init_prof = None
            if init == "A":
                N = int(round(L / hh)) + 1
                zz = (np.arange(N) - (N - 1) / 2.0) * hh
                z10 = np.interp(0.5, (sA - 0.0) / s_int, zA)     # center the A profile at half rise
                init_prof = np.stack([np.interp(zz, zA - z10, m2A, left=mext[0], right=mint[0]),
                                      np.interp(zz, zA - z10, m3A, left=mext[1], right=mint[1])], 1)
            zB, mB, sigB, info = relax_line(mu, om2, mext, mint, Uext, L, hh, width_guess=0.5 * wA, init=init_prof,
                                            maxiter=60000 if init == "A" else 30000)
            sB = mB[:, 0] - mB[:, 1]
            wB, _ = width_10_90(zB, sB, 0.0, s_int)
            fi, kin_int, U_int = first_integral_check(zB, mB, mu, om2, Uext)
            # plateau residuals over the outer 5 percent and the outer 1 percent of the line, in units of the split jump
            n5 = max(2, len(zB) // 20)
            n1 = max(2, len(zB) // 100)
            res_ext5 = float(np.max(np.abs(sB[:n5])) / abs(s_int))
            res_int5 = float(np.max(np.abs(sB[-n5:] - s_int)) / abs(s_int))
            res_ext1 = float(np.max(np.abs(sB[:n1])) / abs(s_int))
            res_int1 = float(np.max(np.abs(sB[-n1:] - s_int)) / abs(s_int))
            gmax = float(np.max(np.abs(np.gradient(sB, hh)[[1, -2]])))
            fracB = sB / s_int                      # 0 at the exterior end, 1 at the interior end (monotone up)
            z50 = float(np.interp(0.5, fracB, zB)) if fracB[0] < fracB[-1] else float(np.interp(0.5, fracB[::-1], zB[::-1]))
            dl_ext, nb_ext = tail_decay(zB - z50, sB, 0.0, "left")
            dl_int, nb_int = tail_decay(zB - z50, sB, s_int, "right")
            tag = f"L{L:g}_h{hh:g}_{init}"
            runs[tag] = {"sigma": sigB, "width_10_90": wB, "first_integral_max_dev": fi,
                         "int_kin": kin_int, "int_U": U_int, "plateau_res_ext_5pct": res_ext5, "plateau_res_int_5pct": res_int5,
                         "plateau_res_ext_1pct": res_ext1, "plateau_res_int_1pct": res_int1,
                         "end_slope": gmax, "center_z50": z50, "decay_len_ext": dl_ext, "decay_len_int": dl_int,
                         "opt": info, "npts": len(zB)}
            log(f"{key} line {tag}: sigma {sigB:.5f} width {wB:.1f} first-integral dev {fi:.2e} plateau res(5pct) {res_ext5:.1e}/{res_int5:.1e} "
                f"res(1pct) {res_ext1:.1e}/{res_int1:.1e} decay {dl_ext:.0f}/{dl_int:.0f} center {z50:.0f} nit {info['nit']}")
            if L == Lp and hh == hp and init == "A":
                prof_keep = (zB, mB)
            if L == 2 * Lp and hh == hp and init == "A":
                prof_long = (zB, mB)
        # the double-length converged profile read at the producer's end positions (+- Lp/2 from its center)
        zL, mL = prof_long
        sL = mL[:, 0] - mL[:, 1]
        zc = runs[f"L{2 * Lp:g}_h{hp:g}_A"]["center_z50"]
        S["long_profile_residual_at_producer_ends"] = {
            "ext_end": float(abs(np.interp(zc - Lp / 2, zL, sL)) / abs(s_int)),
            "int_end": float(abs(np.interp(zc + Lp / 2, zL, sL) - s_int) / abs(s_int))}
        S["line_runs"] = runs
        S["sigma_B_line"] = runs[f"L{Lp:g}_h{hp:g}_A"]["sigma"]
        S["sigma_B_line_tanh_start"] = runs[f"L{Lp:g}_h{hp:g}_tanh"]["sigma"]
        S["width_10_90_B_line"] = runs[f"L{Lp:g}_h{hp:g}_A"]["width_10_90"]
        log(f"{key}: long profile residual at the producer's ends ext {S['long_profile_residual_at_producer_ends']['ext_end']:.2e} int {S['long_profile_residual_at_producer_ends']['int_end']:.2e}")
        # ---- W5 part 2: the relaxed profile on the lattice slab (spacing = the profile's)
        zB, mB = prof_keep
        M, cfg = slab_of(mB[:, 0], mB[:, 1], hp)
        reg = registry_slab_read(M, cfg)
        own = own_reduced_sums(mB[:, 0], mB[:, 1], hp)
        midE = F_line(mB, hp, mu, om2, Uext)
        # the producer-like comparison: midpoint discretization vs the registry's cell-stencil static
        # (registry static - own static on the registry stencil) should be 1e-16; midpoint vs registry O(h^2)
        mid_kin = 0.0
        a, b = mB[:-1], mB[1:]
        midp = 0.5 * (a + b); d = (b - a) / hp
        mid_kin = hp * float(np.sum(0.5 * (f_of(midp[:, 0]) ** 4 * d[:, 0] ** 2 + f_of(midp[:, 1]) ** 4 * d[:, 1] ** 2)))
        mid_i1 = hp * float(np.sum(8.0 * (midp[:, 0] - midp[:, 1]) ** 2 * (d[:, 0] - d[:, 1]) ** 2))
        S["W5_wall_slab"] = {k: {"registry": reg[k], "own_same_stencil": own[k], "rel": rel(own[k], reg[k])} for k in reg}
        S["W5_wall_slab"]["midpoint_vs_registry"] = {"KPh_static_rel": rel(mid_kin, reg["KPh_static"]), "I1_kin_rel": rel(mid_i1, reg["I1_kin"])}
        S["W5_wall_slab"]["E_u_registry_exact"] = reg["E_u"]
        # sigma from the registry lattice functional on the slab (per area, minus the uniform plateau)
        Nz = len(zB)
        sig_reg = reg["KPh_static"] + reg["V4"] + mu * hp * float(np.sum((mB[:, 0] - mB[:, 1]) ** 2)) \
            - om2 * (reg["KPh_kin"] + reg["I1_kin"]) - hp * Nz * Uext
        S["sigma_registry_slab"] = float(sig_reg)
        log(f"{key} W5 wall slab: " + ", ".join(f"{k} rel {S['W5_wall_slab'][k]['rel']:.1e}" for k in reg)
            + f"; midpoint vs registry KPh {S['W5_wall_slab']['midpoint_vs_registry']['KPh_static_rel']:.1e} I1 {S['W5_wall_slab']['midpoint_vs_registry']['I1_kin_rel']:.1e}; sigma(registry slab) {sig_reg:.5f}")
        # ---- W4: bag law at 1.03 omega_*
        om2b = (1.03 * om) ** 2
        rint = minimize(lambda x: Veff_of(x[0], x[1], mu, om2b), list(mint), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-22})
        rext = minimize(lambda x: Veff_of(x[0], x[1], mu, om2b), list(mext), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-22})
        p = float(rext.fun - rint.fun)
        p_frozen = float(Veff_of(tst, tst, mu, om2b) - Veff_of(mint[0], mint[1], mu, om2b))
        S["bag_1p03"] = {"omega": 1.03 * om, "interior": [float(x) for x in rint.x], "exterior": [float(x) for x in rext.x],
                         "p": p, "p_frozen_states": p_frozen, "R_2sigma_over_p_A": 2 * S["sigma_A_geodesic"] / p,
                         "R_2sigma_over_p_B": 2 * S["sigma_B_line"] / p,
                         "ext_hess_at_1p03": [float(e) for e in np.linalg.eigvalsh(hess_fd(lambda x: Veff_of(x[0], x[1], mu, om2b), mext))]}
        # window check: omega_* below the exterior instability
        S["bag_1p03"]["omega2_1p03_over_inst"] = om2b / S["omega2_inst_ext"]
        log(f"{key} bag at 1.03 omega_*: p {p:.4e} (frozen {p_frozen:.4e}) R {2 * S['sigma_A_geodesic'] / p:.4e}")
        R[key] = S
    # W1 gate mutation: mu = 1e-4 (second order in the plane: inf r sits at the exterior)
    om2m, mintm, _ = crossing(1e-4, tst)
    R["mutation_mu_1e-4"] = {"omega2_inf_r": om2m, "argmin": list(mintm), "dist_to_exterior": float(np.hypot(mintm[0] - tst, mintm[1] - tst))}
    log(f"mutation mu 1e-4: inf r {om2m:.3e} at {mintm} (distance to exterior {R['mutation_mu_1e-4']['dist_to_exterior']:.3f})")
    # W1 gate mutation 2: c = -1 (wrong-sign inertia): no crossing (r < 0 everywhere off the diagonal)
    RESULTS["D2"] = R


# ============================================================ B2: own densities on the lattice
def own_jets(M, h, br):
    return [INS4.d1(M, ax, h, br) for ax in range(3)]


def own_I6_cells(A):
    """R = sum_{ij} F_ij[j, i] with F_ij = A_i eta A_j - A_j eta A_i (spatial, static); I6 = R^2."""
    R = np.zeros(A[0].shape[:-2])
    for i in range(3):
        for j in range(3):
            F = A[i] @ ETA @ A[j] - A[j] @ ETA @ A[i]
            R += F[..., 1 + j, 1 + i]
    return R * R


def own_RG_cells(A, Gten):
    """T^{cd} = sum_{ij} [(A_i)^{jc} (A_j)^{id} - (A_i)^{ic} (A_j)^{jd}], R_G = G_cd T^{cd}."""
    T = np.zeros(A[0].shape)
    div = np.zeros(A[0].shape[:-2] + (4,))
    for i in range(3):
        div += A[i][..., 1 + i, :]
    for i in range(3):
        for j in range(3):
            T += A[i][..., 1 + j, :, None] * A[j][..., 1 + i, None, :]
    T -= div[..., :, None] * div[..., None, :]
    return np.einsum("...cd,...cd->...", Gten, T)


def own_I1_cells(A):
    e = np.zeros(A[0].shape[:-2])
    for i in range(3):
        for j in range(i + 1, 3):
            F = A[i] @ ETA @ A[j] - A[j] @ ETA @ A[i]
            e += np.einsum("...ab,...ab,a,b->...", F, F, ETA_D, ETA_D)
    return e


def own_V4_cells(M):
    N = M @ ETA
    P = np.broadcast_to(np.eye(4), M.shape)
    tot = 0.0
    for p in range(1, 5):
        P = P @ N
        Cp = (-G) ** p + 1.0 + DELTA ** p
        tot = tot + (np.einsum("...kk->...", P) - Cp) ** 2
    return W1 * tot


def own_energies(M, cfg, chat):
    """own static energies (h^3 sum, sym stencil) of every piece; R_hcov by its own formula
    G = eta + 2 (eta u)(eta u)^T with u the timelike eigenvector."""
    h = cfg["h"]
    h3 = h ** 3
    out = {"E_u": 0.0, "V4": h3 * float(np.sum(own_V4_cells(M))), "I6": 0.0, "R_etaMeta": 0.0, "R_hcov": 0.0, "K_lambda": 0.0}
    lam = np.linalg.eigvals(M @ ETA).real
    lam_s = -np.sort(-lam, axis=-1)
    # own h_cov: timelike eigenvector u of N (u^T eta u = -1)
    lamc, V = np.linalg.eig(M @ ETA)
    V = V.real
    n2 = np.einsum("...ak,a,...ak->...k", V, ETA_D, V)
    kt = np.argmin(n2, axis=-1)
    u = np.take_along_axis(V, kt[..., None, None], axis=-1)[..., 0]
    u = u / np.sqrt(np.abs(np.einsum("...a,a,...a->...", u, ETA_D, u)))[..., None]
    w = u @ ETA
    Hcov = ETA + 2.0 * w[..., :, None] * w[..., None, :]
    GM = ETA @ M @ ETA
    for br, wt in INS4.branches(cfg["stencil"]):
        A = own_jets(M, h, br)
        out["E_u"] += wt * 4.0 * h3 * float(np.sum(own_I1_cells(A)))
        out["I6"] += wt * h3 * float(np.sum(own_I6_cells(A)))
        out["R_etaMeta"] += wt * h3 * float(np.sum(own_RG_cells(A, GM)))
        out["R_hcov"] += wt * h3 * float(np.sum(own_RG_cells(A, Hcov)))
        for ax in range(3):
            d = INS4.d1(lam_s, ax, h, br)
            out["K_lambda"] += wt * h3 * 0.5 * float(np.sum(d * d))
    out["E_total_static"] = out["E_u"] + out["V4"] + sum(chat[k] * out[k] for k in chat)
    return out


def registry_energies(M, cfg, chat, p):
    e_u, e_v = INS4.e_parts(M, cfg)
    out = {"E_u": float(e_u), "V4": float(e_v),
           "I6": float(LAG.term_lagrangian(LAG.REGISTRY["I6"], M, cfg, p)),
           "R_etaMeta": TM.static_energy("R_etaMeta", M, cfg),
           "R_hcov": TM.static_energy("R_hcov", M, cfg),
           "K_lambda": TM.static_energy("K_lambda", M, cfg)}
    out["E_total_static"] = out["E_u"] + out["V4"] + sum(chat[k] * out[k] for k in chat)
    return out


def total_grad(M, cfg, chat, p, J=None):
    Gt = INS4.grad(M, cfg)
    if chat.get("I6", 0.0):
        Gt = Gt + chat["I6"] * LAG.term_grad_lagrangian(LAG.REGISTRY["I6"], M, cfg, p)
    if chat.get("R_etaMeta", 0.0):
        Gt = Gt + chat["R_etaMeta"] * TM.rg_grad(M, cfg, "etaMeta")
    if chat.get("R_hcov", 0.0):
        Gt = Gt + chat["R_hcov"] * TM.rg_hcov_energy_grad(M, cfg)[1]
    if chat.get("K_lambda", 0.0):
        Gt = Gt + chat["K_lambda"] * TM.klam_energy_grad(M, cfg)[1]
    kin = None
    if J is not None and J != 0.0:
        a0 = CM.a0_local(M)
        kin = float(INS4.kin_of(M, a0, cfg))
        Gt = Gt - (J * J / (4.0 * kin * kin)) * INS4.kin_grad(M, a0, cfg)
    return Gt, kin


def total_energy_own(M, cfg, chat, J=None):
    e = own_energies(M, cfg, chat)
    E = e["E_total_static"]
    if J is not None and J != 0.0:
        kin = float(INS4.kin_of(M, CM.a0_local(M), cfg))
        E += J * J / (4.0 * kin)
        e["kin_I1"] = kin
        e["omega"] = J / (2.0 * kin)
    e["E"] = E
    return e


def cell_radius(cfg):
    X, Y, Z = INS4.coords(cfg["n"], cfg["h"])
    return np.sqrt(X * X + Y * Y + Z * Z)


def localization(M, M0, cfg, chat):
    """where the I6 energy sits: per-cell chat_I6 * I6 density (h^3, sym stencil) of the current
    field minus the seed's; top cell, its radius, the share of the total change in the top
    10 cells and within 1 / 2 lattice steps of the top cell."""
    h = cfg["h"]
    h3 = h ** 3

    def dens(Mx):
        d = np.zeros(Mx.shape[:-2])
        du = np.zeros(Mx.shape[:-2])
        for br, wt in INS4.branches(cfg["stencil"]):
            A = own_jets(Mx, h, br)
            d += wt * h3 * own_I6_cells(A)
            du += wt * 4.0 * h3 * own_I1_cells(A)
        return d, du
    d1, du1 = dens(M)
    d0, du0 = dens(M0)
    dd = d1 - d0
    r = cell_radius(cfg)
    k = np.argmax(np.abs(dd))
    ijk = np.unravel_index(k, dd.shape)
    tot = float(np.sum(dd))
    flat = np.sort(np.abs(dd).ravel())[::-1]
    share10 = float(np.sum(flat[:10]) / max(np.sum(flat), 1e-300))
    I, Jj, K = np.indices(dd.shape)
    dist = np.sqrt((I - ijk[0]) ** 2 + (Jj - ijk[1]) ** 2 + (K - ijk[2]) ** 2)
    share_r1 = float(np.sum(np.abs(dd)[dist <= 1.0]) / max(np.sum(np.abs(dd)), 1e-300))
    share_r2 = float(np.sum(np.abs(dd)[dist <= 2.0]) / max(np.sum(np.abs(dd)), 1e-300))
    share_r3 = float(np.sum(np.abs(dd)[dist <= 3.0]) / max(np.sum(np.abs(dd)), 1e-300))
    n_half = int(np.searchsorted(np.cumsum(flat), 0.5 * np.sum(flat)) + 1)
    # E_u concentration at the same spot
    ddu = du1 - du0
    share_u_r2 = float(np.sum(ddu[dist <= 2.0]) / max(abs(np.sum(ddu)), 1e-300))
    gap = CM.gap23(M)
    dM = M - M0
    dMn = np.sqrt(np.sum(dM * dM, axis=(-1, -2)))
    kd = np.argmax(dMn)
    ijkd = np.unravel_index(kd, dMn.shape)
    return {"top_cell": [int(x) for x in ijk], "top_cell_radius": float(r[ijk]), "top_cell_dI6": float(dd[ijk]),
            "total_dI6": tot, "share_top10": share10, "share_within_1": share_r1, "share_within_2": share_r2,
            "share_within_3": share_r3, "cells_for_half": n_half, "dE_u_share_within_2": share_u_r2,
            "gap_at_top": float(gap[ijk]), "gap_min_field": float(gap.min()),
            "max_dM_cell": [int(x) for x in ijkd], "max_dM_cell_radius": float(r[ijkd]), "max_dM": float(dMn[ijkd]),
            "dM_share_within_2_of_top": float(np.sum(dMn[dist <= 2.0] ** 2) / np.sum(dMn ** 2))}


def fire_own(M0, cfg, chat, p, free, J=None, max_iter=100, log_every=5, dt0=1e-3, dt_max=0.1,
             tag="", plain_gd=False, dt_fixed=None):
    """own FIRE (Bitzek 2006: N_min 5, f_inc 1.1, f_dec 0.5, alpha0 0.1, f_alpha 0.99, semi-implicit
    Euler, velocity mixed before the update); plain_gd = fixed-step steepest descent."""
    M = M0.copy()
    fr = free[..., None, None].astype(float)
    v = np.zeros_like(M)
    dt, alpha, n_up = dt0, 0.1, 0
    hist = []
    Gt, kin = total_grad(M, cfg, chat, p, J)
    F = -Gt * fr
    stop = "max_iter"
    it = 0
    n0 = np.sqrt(np.sum(M0 * M0))
    for it in range(1, max_iter + 1):
        if plain_gd:
            M = M + dt_fixed * F
        else:
            P = float(np.sum(F * v))
            if P > 0.0:
                n_up += 1
                vn = np.sqrt(np.sum(v * v)); fn = np.sqrt(np.sum(F * F))
                v = (1.0 - alpha) * v + alpha * (F / max(fn, 1e-300)) * vn
                if n_up > 5:
                    dt = min(dt * 1.1, dt_max); alpha *= 0.99
            else:
                v[:] = 0.0; dt *= 0.5; alpha = 0.1; n_up = 0
            v = v + dt * F
            M = M + dt * v
        if not np.all(np.isfinite(M)):
            stop = "non-finite"; break
        Gt, kin = total_grad(M, cfg, chat, p, J)
        F = -Gt * fr
        fmax = float(np.max(np.abs(F)))
        if not np.isfinite(fmax):
            stop = "non-finite"; break
        if it % log_every == 0 or it == max_iter:
            e = total_energy_own(M, cfg, chat, J)
            row = {"it": it, "E": e["E"], "E_u": e["E_u"], "V4": e["V4"], "I6": e["I6"], "R_etaMeta": e["R_etaMeta"],
                   "R_hcov": e["R_hcov"], "K_lambda": e["K_lambda"], "fmax": fmax, "dt": dt,
                   "dist_from_seed": float(np.sqrt(np.sum((M - M0) ** 2)) / n0),
                   "max_abs_M0i": float(np.max(np.abs(M[..., 0, 1:]))), "gap_min": float(CM.gap23(M).min())}
            if kin is not None:
                row["kin_I1"] = kin; row["omega"] = J / (2.0 * kin)
            hist.append(row)
            print(f"  {tag} it {it:4d} E {e['E']:14.4f} E_u {e['E_u']:10.4f} V4 {e['V4']:.4e} I6 {e['I6']:10.4f} "
                  f"RM {e['R_etaMeta']:10.4f} KL {e['K_lambda']:8.4f} fmax {fmax:.3e} dt {dt:.2e} dist {row['dist_from_seed']:.2e}", flush=True)
        if fmax > 1e8:
            stop = "fmax > 1e8"; break
    last_finite = M if np.all(np.isfinite(M)) else None
    return M, {"stop": stop, "iters": it, "trace": hist}, last_finite


def stage_b2():
    R = {}
    chat = {"I6": -176.997, "R_etaMeta": 10.363, "R_hcov": -10.802, "K_lambda": 2.132}
    M0, cfg, rec = CM.seed_hedgehog(32, 48.0)
    p = LAG.default_params(s=-1.0, g=G, delta=DELTA)
    free = ~INS4.pin_shell(32, cfg["h"])
    R["seed"] = rec
    # V1: own energies vs registry
    own = own_energies(M0, cfg, chat)
    reg = registry_energies(M0, cfg, chat, p)
    R["V1"] = {"own": own, "registry": reg, "rel": {k: rel(own[k], reg[k]) for k in reg}}
    a0 = CM.a0_local(M0)
    kin = float(INS4.kin_of(M0, a0, cfg))
    A6, B6, C6 = LAG.omega_decompose(LAG.REGISTRY["I6"], M0, cfg, p, a0)
    kl_kin = TM.kin_energy("K_lambda", M0, a0, cfg)
    R["V1"]["kin_I1"] = kin
    R["V1"]["q_I6"] = float(-C6)
    R["V1"]["q_K_lambda"] = kl_kin
    R["V1"]["kin_tot_J200"] = kin + chat["I6"] * (-C6) + chat["K_lambda"] * kl_kin
    R["V1"]["E_J200"] = own["E_total_static"] + 200.0 ** 2 / (4.0 * R["V1"]["kin_tot_J200"])
    R["V1"]["R_eta_static"] = TM.static_energy("R_eta", M0, cfg)
    R["V1"]["chat_I6_over_4"] = chat["I6"] / 4.0
    R["V1"]["I6_over_I1"] = own["I6"] / (own["E_u"] / 4.0)
    log("V1 own: " + ", ".join(f"{k} {own[k]:.4f}" for k in own) + f"; rel to registry max {max(R['V1']['rel'].values()):.1e}")
    log(f"V1 kin_I1 {kin:.4f} q_I6 {-C6:.2e} q_Klam {kl_kin:.2e} E(J=200) {R['V1']['E_J200']:.4f}")
    # gradient gate: FD of the OWN energy along a random free direction vs the registry-assembled gradient
    rng = np.random.default_rng(3)
    d = rng.normal(size=M0.shape); d = 0.5 * (d + d.swapaxes(-1, -2)) * free[..., None, None]
    d /= np.sqrt(np.sum(d * d))
    gates = {}
    for J in (None, 200.0):
        Gt, _ = total_grad(M0, cfg, chat, p, J)
        eps = 1e-4
        Ep = total_energy_own(M0 + eps * d, cfg, chat, J)["E"]
        Em = total_energy_own(M0 - eps * d, cfg, chat, J)["E"]
        fd = (Ep - Em) / (2 * eps)
        an = float(np.sum(Gt * d))
        gates[f"J{J}"] = {"fd": fd, "analytic": an, "rel": rel(fd, an)}
    # per-term gradient gates (own energy of each term against its registry gradient)
    for name, gfun, efun in (("I6", lambda: LAG.term_grad_lagrangian(LAG.REGISTRY["I6"], M0, cfg, p), lambda Mx: own_energies(Mx, cfg, chat)["I6"]),
                             ("R_etaMeta", lambda: TM.rg_grad(M0, cfg, "etaMeta"), lambda Mx: own_energies(Mx, cfg, chat)["R_etaMeta"]),
                             ("R_hcov", lambda: TM.rg_hcov_energy_grad(M0, cfg)[1], lambda Mx: own_energies(Mx, cfg, chat)["R_hcov"]),
                             ("K_lambda", lambda: TM.klam_energy_grad(M0, cfg)[1], lambda Mx: own_energies(Mx, cfg, chat)["K_lambda"])):
        Gk = gfun()
        eps = 1e-4
        fd = (efun(M0 + eps * d) - efun(M0 - eps * d)) / (2 * eps)
        an = float(np.sum(Gk * d))
        gates[name] = {"fd": fd, "analytic": an, "rel": rel(fd, an) if abs(an) > 1e-12 else float(abs(fd - an)),
                       "free_grad_max": float(np.max(np.abs(Gk[free]))), "pinned_grad_max": float(np.max(np.abs(Gk[~free])))}
    R["gradient_gates"] = gates
    log("gradient gates: " + ", ".join(f"{k} rel {v['rel']:.1e}" for k, v in gates.items()))
    log("R_hcov free-cell gradient max " + f"{gates['R_hcov']['free_grad_max']:.2e} (pinned {gates['R_hcov']['pinned_grad_max']:.2e})")
    # V2: own FIRE at J = 0 and J = 200 (at most 100 steps)
    runs = {}
    for J, nmax in ((None, 100), (200.0, 60)):
        tag = f"J{0 if J is None else int(J)}"
        log(f"own FIRE {tag} (max {nmax} steps)")
        Mend, info, last = fire_own(M0, cfg, chat, p, free, J=J, max_iter=nmax, tag=tag)
        runs[tag] = info
        if last is not None:
            runs[tag]["localization_last_finite"] = localization(last, M0, cfg, chat)
        # the last finite logged state is reconstructed from the trace rows only; localization uses the end field
        log(f"{tag}: stop {info['stop']} at it {info['iters']}")
        if J is None:
            # keep the it-45-like state for the ray scan: rerun to 45 steps is costly; use the end field if finite
            pass
    R["V2"] = runs
    # localization at a controlled distance: rerun J0 to the step whose distance ~ 4e-3 (the producer's it 45)
    # (the own FIRE trace identifies the step; the run is short)
    tr = runs["J0"]["trace"]
    it45 = None
    for row in tr:
        if row["it"] == 45:
            it45 = row
    R["V2"]["J0_row45"] = it45
    M45, info45, _ = fire_own(M0, cfg, chat, p, free, J=None, max_iter=45, log_every=45, tag="J0_to45")
    R["V2"]["localization_it45"] = localization(M45, M0, cfg, chat)
    e45 = total_energy_own(M45, cfg, chat)
    R["V2"]["own_energies_it45"] = e45
    log(f"it45: E {e45['E']:.2f} E_u {e45['E_u']:.2f} V4 {e45['V4']:.2f} I6 {e45['I6']:.2f}; localization {R['V2']['localization_it45']}")
    # the UV nature: E along the ray M0 + t (M45 - M0) with the direction normalized to the it-45 move
    dvec = M45 - M0
    ray = []
    for t in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        e = total_energy_own(M0 + t * dvec, cfg, chat)
        ray.append({"t": t, "E": e["E"], "E_u": e["E_u"], "I6": e["I6"], "V4": e["V4"], "R_etaMeta": e["R_etaMeta"], "K_lambda": e["K_lambda"]})
    R["V2"]["ray_scan"] = ray
    # quartic fit of E(t) - E(0) on the ray: leading t^4 coefficient from the two largest t
    E0 = own["E_total_static"]
    t1, t2 = 2.0, 3.0
    e1 = [r for r in ray if r["t"] == t1][0]["E"] - E0
    e2 = [r for r in ray if r["t"] == t2][0]["E"] - E0
    R["V2"]["ray_exponent_2_to_3"] = float(np.log(abs(e2) / abs(e1)) / np.log(t2 / t1))
    # I6 <= 6 I1 bound on M0i = 0 fields: check on the seed and the it-45 field (cellwise ratio max)
    def ratio_max(Mx):
        rmax = 0.0
        for br, wt in INS4.branches(cfg["stencil"]):
            A = own_jets(Mx, cfg["h"], br)
            i6 = own_I6_cells(A); i1 = own_I1_cells(A)
            sel = i1 > 1e-12 * i1.max()
            rmax = max(rmax, float(np.max(i6[sel] / i1[sel])))
        return rmax
    R["V2"]["I6_over_I1_cellwise_max"] = {"seed": ratio_max(M0), "it45": ratio_max(M45)}
    # plain gradient descent (fixed dt), a different integrator
    log("plain gradient descent dt 0.01, 100 steps")
    Mg, infog, lastg = fire_own(M0, cfg, chat, p, free, J=None, max_iter=100, log_every=10, tag="GD", plain_gd=True, dt_fixed=0.01)
    R["V2"]["plain_gd_dt0.01"] = infog
    # V3: chat_I6 = 0
    chat0 = dict(chat); chat0["I6"] = 0.0
    log("V3: chat_I6 = 0, 50 steps")
    M3, info3, _ = fire_own(M0, cfg, chat0, p, free, J=None, max_iter=50, tag="I6=0")
    R["V3"] = {"chat_I6_0": info3}
    # mutations: I6 alone at -176.997 (the others 0) must blow up; chat_I6 = -0.6 (inside the I6 <= 6 I1 bound) must not
    chatA = {"I6": -176.997, "R_etaMeta": 0.0, "R_hcov": 0.0, "K_lambda": 0.0}
    log("mutation: I6 alone at -176.997, 60 steps")
    MA, infoA, _ = fire_own(M0, cfg, chatA, p, free, J=None, max_iter=60, tag="I6only")
    R["V3"]["I6_alone_-176.997"] = infoA
    chatB = dict(chat); chatB["I6"] = -0.6
    log("mutation: chat_I6 = -0.6 with the other three kept, 50 steps")
    MB, infoB, _ = fire_own(M0, cfg, chatB, p, free, J=None, max_iter=50, tag="I6=-0.6")
    R["V3"]["chat_I6_-0.6"] = infoB
    # the LP blind spot: a single plane wave has F = 0 (every curvature term vanishes), a crossed pair does not
    nb = 16
    cfgb = INS4.base_cfg(s=-1.0, g=G, n=nb, L=24.0, delta=DELTA)
    X, Y, Z = INS4.coords(nb, cfgb["h"])
    Xs = rng.normal(size=(4, 4)); Xs = 0.5 * (Xs + Xs.T)
    Ys = rng.normal(size=(4, 4)); Ys = 0.5 * (Ys + Ys.T)
    Mv = np.zeros((nb, nb, nb, 4, 4)); Mv[:] = np.diag([G, 1.0, DELTA, 0.0])
    kz = 2 * np.pi * 2 / 24.0
    Mpw = Mv + 0.3 * np.cos(kz * Z)[..., None, None] * Xs
    Mcr = Mv + 0.3 * np.cos(kz * X)[..., None, None] * Xs + 0.3 * np.cos(kz * Y)[..., None, None] * Ys
    epw = own_energies(Mpw, cfgb, chat); ecr = own_energies(Mcr, cfgb, chat)
    R["blind_spot"] = {"plane_wave": {"E_u": epw["E_u"], "I6": epw["I6"], "4I1_minus_177I6": epw["E_u"] + chat["I6"] * epw["I6"]},
                       "crossed_waves": {"E_u": ecr["E_u"], "I6": ecr["I6"], "4I1_minus_177I6": ecr["E_u"] + chat["I6"] * ecr["I6"]}}
    log(f"blind spot: plane wave E_u {epw['E_u']:.2e} I6 {epw['I6']:.2e}; crossed E_u {ecr['E_u']:.3f} I6 {ecr['I6']:.3f} quartic sum {R['blind_spot']['crossed_waves']['4I1_minus_177I6']:.2f}")
    RESULTS["B2"] = R


def stage_b2gd():
    """the second integrator: plain steepest descent at a fixed dt below the stiff terms' stability
    limit (dt 0.01 diverges at step 3 from the seed: an integrator artifact, not evidence)."""
    chat = {"I6": -176.997, "R_etaMeta": 10.363, "R_hcov": -10.802, "K_lambda": 2.132}
    M0, cfg, rec = CM.seed_hedgehog(32, 48.0)
    p = LAG.default_params(s=-1.0, g=G, delta=DELTA)
    free = ~INS4.pin_shell(32, cfg["h"])
    R = {}
    for dtf in (4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4):
        log(f"plain gradient descent dt {dtf}, 20 steps (stability ladder)")
        Mg, infog, lastg = fire_own(M0, cfg, chat, p, free, J=None, max_iter=20, log_every=5, tag=f"GD{dtf:g}", plain_gd=True, dt_fixed=dtf)
        R[f"plain_gd_dt{dtf:g}"] = infog
    # the same ladder without I6 (is the stiff mode I6's?)
    chat0 = dict(chat); chat0["I6"] = 0.0
    for dtf in (1e-3, 2.5e-4):
        Mg, infog, lastg = fire_own(M0, cfg, chat0, p, free, J=None, max_iter=20, log_every=5, tag=f"GD{dtf:g}_I6=0", plain_gd=True, dt_fixed=dtf)
        R[f"plain_gd_dt{dtf:g}_I6=0"] = infog
    # the certified functional alone
    chatc = {"I6": 0.0, "R_etaMeta": 0.0, "R_hcov": 0.0, "K_lambda": 0.0}
    for dtf in (1e-2, 2.5e-4):
        Mg, infog, lastg = fire_own(M0, cfg, chatc, p, free, J=None, max_iter=20, log_every=5, tag=f"GD{dtf:g}_cert", plain_gd=True, dt_fixed=dtf)
        R[f"plain_gd_dt{dtf:g}_cert"] = infog
    # integrator-free: the fine ray from the seed toward the it-45 state (own FIRE to 45 steps), E(t) monotone?
    M45, _, _ = fire_own(M0, cfg, chat, p, free, J=None, max_iter=45, log_every=45, tag="J0_to45")
    dvec = M45 - M0
    E0 = own_energies(M0, cfg, chat)["E_total_static"]
    ray = [{"t": 0.0, "E": E0}]
    for t in (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0):
        e = own_energies(M0 + t * dvec, cfg, chat)
        ray.append({"t": t, "E": e["E_total_static"], "E_u": e["E_u"], "I6": e["I6"], "V4": e["V4"], "R_etaMeta": e["R_etaMeta"], "K_lambda": e["K_lambda"]})
    R["fine_ray"] = ray
    R["fine_ray_monotone"] = bool(all(ray[i + 1]["E"] < ray[i]["E"] for i in range(len(ray) - 1)))
    log("fine ray: " + ", ".join(f"t {r['t']:g} E {r['E']:.2f}" for r in ray) + f"; monotone {R['fine_ray_monotone']}")
    RESULTS["B2_gd"] = R


def main():
    stage = "all"
    if "--stage" in ARGV:
        stage = ARGV[ARGV.index("--stage") + 1]
    if stage in ("d2", "all"):
        stage_d2()
    if stage in ("b2", "all"):
        stage_b2()
    if stage in ("b2gd", "all"):
        stage_b2gd()
    RESULTS["wall_s"] = round(time.time() - T0, 1)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    if stage != "all" and os.path.exists(OUT):
        old = json.load(open(OUT))
        old.update(clean(RESULTS))
        json.dump(old, open(OUT, "w"), indent=1)
    else:
        json.dump(clean(RESULTS), open(OUT, "w"), indent=1)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
