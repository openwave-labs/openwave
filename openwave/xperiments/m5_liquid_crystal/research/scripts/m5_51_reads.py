"""M5.51: the readers of section 4 of the plan, their control K4, the words of section 5, the summary and the plots.

With l1 >= l2 >= l3 the eigenvalues of M and g12 = l1 - l2 the top gap:

  R_A    the archive's ring reader (a descriptor and a first guess): nodes with g12 < 0.15 and r < 0.93 L, weights
         0.15 - g12, the weighted centroid, the normal as the least eigenvector of the weighted covariance, R_A the
         weighted mean in-plane distance
  R_L    the line radius: from a first guess (R_A with 4 nodes or more, else the fallback over the 13 lattice
         directions), two passes; in each of 96 meridian half-planes about the current axis g12 (trilinear) is
         sampled at steps of h/10 for rho within 1.5 of the current radius and zeta within +-1.5, the least sample
         refined by Nelder-Mead to a crossing; the crossings fitted by a plane and a circle in it; R_L the mean
         in-plane distance of the crossings from the circle's center
  P_m    the co-rotating azimuthal power about the snapped axis on twelve circles (rho 1..4, zeta -1, 0, 1), 96
         azimuths: P_m = sum(|M_m|^2 + |M_-m|^2) / sum |M_0|^2 of the co-rotated matrices
  N/U    the Laue read: V_a = (chi+ D+ M + chi- D- M) / 2, N_a = 2 h^3 sum_free [ L_V s |V_a|^2
         + kappa sum_b sum_t 1/2 chi |[V_a, D_b^t M]|^2 ], N/U = (N_x + N_y + N_z) / (3 U)
  degree |q| of the long axis on centered cube surfaces (m5_46_degree.py, unchanged), half-widths the multiples of h
         nearest to 3.5, 4, ..., L - h

Usage:
  python m5_51_reads.py --k4                       the control K4 and the reads of the author's stored arrays
  python m5_51_reads.py --summary                  the words of section 5, the summary JSON and the plots
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import m5_51_action as A  # noqa: E402
import m5_46_degree as DG  # noqa: E402

CK = HERE.parent / "checkpoints" / "m5_51"
DATA = HERE.parent / "data"
PLOTS = HERE.parent / "plots"
NPHI = 96

GRIDS = dict(G1=(6.0, 0.5, 25), G2=(6.0, 0.375, 33), G3=(6.0, 0.25, 49), G4=(9.0, 0.5, 37), G5=(9.0, 0.375, 49), G6=(12.0, 0.5, 49))


def lattice_dirs():
    D = [("axis", np.eye(3)[i]) for i in range(3)]
    D += [("face", np.array(v, float) / np.sqrt(2)) for v in ([1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1])]
    D += [("diag", np.array(v, float) / np.sqrt(3)) for v in ([1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1])]
    return D


def angle(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.degrees(np.arccos(min(1.0, abs(float(a @ b)) / np.linalg.norm(a) / np.linalg.norm(b)))))


def frame(nu, e1=None):
    nu = np.asarray(nu, float)
    nu = nu / np.linalg.norm(nu)
    if e1 is None:
        t = np.array([1.0, 0, 0]) if abs(nu[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = t
    e1 = np.asarray(e1, float)
    e1 = e1 - (e1 @ nu) * nu
    e1 /= np.linalg.norm(e1)
    return nu, e1, np.cross(nu, e1)


def interp(m6, x, P):
    """M at the points P (..., 3), trilinear on the six components."""
    h = x[1] - x[0]
    idx = [(P[..., a] - x[0]) / h for a in range(3)]
    out = np.empty(P.shape[:-1] + (6,))
    for k in range(6):
        out[..., k] = map_coordinates(m6[..., k], idx, order=1, mode="nearest")
    return out[..., A.IDX]


def gap12(M):
    w = np.linalg.eigvalsh(M)
    return w[..., 2] - w[..., 1]


# ----------------------------------------------------------------------------- the constructed rings

def ring_field(x, R0, nu, e1, c=(0.0, 0.0, 0.0), eps=0.0):
    nu, e1, e2 = frame(nu, e1)
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1) - np.asarray(c, float)
    zeta = X @ nu
    u = X @ e1
    v = X @ e2
    rho = np.sqrt(u * u + v * v)
    phi = np.arctan2(v, u)
    Rp = R0 * (1 + eps * np.cos(2 * phi))
    rhat = (u[..., None] * e1 + v[..., None] * e2) / np.maximum(rho, 1e-300)[..., None]
    d = np.sqrt((rho - Rp) ** 2 + zeta ** 2)
    th = np.arctan2(zeta, rho - Rp)
    f = np.tanh(d)
    n = np.cos(th / 2)[..., None] * rhat + np.sin(th / 2)[..., None] * nu
    m = -np.sin(th / 2)[..., None] * rhat + np.cos(th / 2)[..., None] * nu
    M = 0.5 * (1 + f)[..., None, None] * np.einsum("...i,...j->...ij", n, n) + 0.5 * (1 - f)[..., None, None] * np.einsum("...i,...j->...ij", m, m)
    return A.to6(M)


ORIENT = dict(axis=([1.0, 0, 0], [0, 1.0, 0]), diag=([1.0, 1, -1], [1.0, -1, 0]),
              tilt16=([np.sin(np.radians(16)), 0, np.cos(np.radians(16))], [0, 1.0, 0]))


# ----------------------------------------------------------------------------- the ring readers

def R_A(m6, x, L, thr=0.15):
    M = A.from6(m6)
    g = gap12(M)
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
    r = np.linalg.norm(X, axis=-1)
    msk = (g < thr) & (r < 0.93 * L)
    n = int(msk.sum())
    if n < 4:
        return dict(nodes=n)
    pts = X[msk]
    wt = thr - g[msk]
    c = (pts * wt[:, None]).sum(0) / wt.sum()
    q = pts - c
    C = (q[:, :, None] * q[:, None, :] * wt[:, None, None]).sum(0) / wt.sum()
    _, ev = np.linalg.eigh(C)
    nu = ev[:, 0]
    rad = np.sqrt(np.maximum((q ** 2).sum(1) - (q @ nu) ** 2, 0.0))
    return dict(nodes=n, c=c, nu=nu, R=float((rad * wt).sum() / wt.sum()))


def _refine(m6, x, c, rh, nu, r0, z0, h):
    def fun(p):
        return float(gap12(interp(m6, x, (c + p[0] * rh + p[1] * nu)[None]))[0])
    res = minimize(fun, [r0, z0], method="Nelder-Mead",
                   options=dict(xatol=1e-5, fatol=1e-12, maxiter=2000, initial_simplex=[[r0, z0], [r0 + h / 10, z0], [r0, z0 + h / 10]]))
    return res.x, float(res.fun)


def _halfturn(m6, x, p0, rh, nu):
    ang = np.linspace(0, 2 * np.pi, 129)
    Q = p0 + 0.5 * (np.cos(ang)[:, None] * rh + np.sin(ang)[:, None] * nu)
    _, V = np.linalg.eigh(interp(m6, x, Q))
    n = V[..., 2].copy()
    for t in range(1, len(n)):
        if n[t] @ n[t - 1] < 0:
            n[t] = -n[t]
    return bool(n[-1] @ n[0] < 0)


def meridian_pass(m6, x, c, nu, Rg, nphi=NPHI, step_div=10, rho_lo=None, rho_hi=None, window=1.5, halfturn=True):
    h = x[1] - x[0]
    nu, e1, e2 = frame(nu)
    lo = max(0.0, Rg - window) if rho_lo is None else rho_lo
    hi = Rg + window if rho_hi is None else rho_hi
    rho = np.arange(lo, hi + 1e-9, h / step_div)
    zet = np.arange(-window, window + 1e-9, h / step_div)
    RR, ZZ = np.meshgrid(rho, zet, indexing="ij")
    out = []
    for k in range(nphi):
        ph = 2 * np.pi * k / nphi
        rh = np.cos(ph) * e1 + np.sin(ph) * e2
        P = c + RR[..., None] * rh + ZZ[..., None] * nu
        g = gap12(interp(m6, x, P))
        i, j = np.unravel_index(np.argmin(g), g.shape)
        (rs, zs), gv = _refine(m6, x, c, rh, nu, rho[i], zet[j], h)
        p0 = c + rs * rh + zs * nu
        out.append(dict(phi=ph, p=p0, rho=float(rs), zeta=float(zs), g=gv, halfturn=_halfturn(m6, x, p0, rh, nu) if halfturn else None))
    return out


def fit_circle(points):
    c0 = points.mean(0)
    q = points - c0
    _, ev = np.linalg.eigh(q.T @ q)
    nu = ev[:, 0]
    nu_, e1, e2 = frame(nu)
    u = q @ e1
    v = q @ e2
    Am = np.c_[u, v, np.ones_like(u)]
    s = np.linalg.lstsq(Am, u * u + v * v, rcond=None)[0]
    cc = c0 + s[0] / 2 * e1 + s[1] / 2 * e2
    qq = points - cc
    rad = np.sqrt(np.maximum((qq ** 2).sum(1) - (qq @ nu) ** 2, 0.0))
    return cc, nu, float(rad.mean())


def fallback_guess(m6, x, L):
    """The frozen fallback start: 13 lattice directions about the box center, 24 half-planes each."""
    h = x[1] - x[0]
    best = None
    rows = []
    for kind, d in lattice_dirs():
        out = meridian_pass(m6, x, np.zeros(3), d, None, nphi=24, step_div=4, rho_lo=0.5, rho_hi=L - 1.5, halfturn=False)
        cnt = sum(1 for o in out if o["g"] <= 0.02)
        rms = float(np.sqrt(np.mean([o["zeta"] ** 2 for o in out if o["g"] <= 0.02]))) if cnt else np.inf
        rows.append(dict(direction=d.tolist(), crossings=cnt, rms_zeta=rms))
        key = (-cnt, rms)
        if best is None or key < best[0]:
            best = (key, d, out)
    pts = np.array([o["p"] for o in best[2] if o["g"] <= 0.02])
    if len(pts) < 3:
        return None, rows
    c, nu, R = fit_circle(pts)
    return (c, nu, R), rows


def line_reader(m6, x, L, withhold_first_guess=False):
    """R_L with its regime, the crossings and the descriptors of the line."""
    h = x[1] - x[0]
    ra = R_A(m6, x, L)
    used = "R_A"
    if ra["nodes"] >= 4 and not withhold_first_guess:
        c, nu, R = ra["c"], ra["nu"], ra["R"]
    else:
        g0, _ = fallback_guess(m6, x, L)
        used = "fallback"
        if g0 is None:
            return dict(ok=False, reason="no first guess", RA=ra.get("R"), RA_nodes=ra["nodes"], first_guess=used)
        c, nu, R = g0
    for _ in range(2):
        out = meridian_pass(m6, x, c, nu, R)
        pts = np.array([o["p"] for o in out])
        c, nu, R = fit_circle(pts)
    nu_, e1, e2 = frame(nu)
    q = pts - c
    zeta = q @ nu_
    rho = np.sqrt(np.maximum((q ** 2).sum(1) - zeta ** 2, 0.0))
    phis = np.arctan2(q @ e2, q @ e1)
    c2 = np.mean(rho * np.exp(-2j * phis))
    gs = np.array([o["g"] for o in out])
    ht = [o["halfturn"] for o in out]
    regime = bool((gs <= 0.02).all() and all(ht))
    dcons = np.linalg.norm(np.roll(pts, -1, 0) - pts, axis=1)
    return dict(ok=True, RL=R, c=c, nu=nu_, crossings=pts, rho_star=rho, phi_star=phis, g_star=gs, halfturn=ht, regime=regime,
                gmax=float(gs.max()), halfturn_all=bool(all(ht)), out_of_plane_spread=float(zeta.max() - zeta.min()),
                m2_harmonic=float(2 * abs(c2) / R), max_consecutive=float(dcons.max()),
                RA=ra.get("R"), RA_nodes=ra["nodes"], RA_c=None if ra["nodes"] < 4 else ra["c"], RA_nu=None if ra["nodes"] < 4 else ra["nu"], first_guess=used)


def one_ring(m6, x, line):
    """The crossing on all 96 half-planes, consecutive crossings within 2h, no node with g12 < 0.15 farther than 1.5
    from the fitted circle."""
    if not line.get("ok"):
        return dict(one_ring=False, reason="no line")
    h = x[1] - x[0]
    M = A.from6(m6)
    g = gap12(M)
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
    P = X[g < 0.15] - line["c"]
    nu = line["nu"]
    z = P @ nu
    rho = np.sqrt(np.maximum((P ** 2).sum(1) - z ** 2, 0.0))
    dist = np.sqrt((rho - line["RL"]) ** 2 + z ** 2)
    far = int((dist > 1.5).sum())
    all_cross = bool((line["g_star"] <= 0.02).all())
    cons = line["max_consecutive"] <= 2 * h
    return dict(one_ring=bool(all_cross and cons and far == 0), all_crossings=all_cross, consecutive_ok=bool(cons), max_consecutive=line["max_consecutive"],
                far_nodes=far, far_max=float(dist.max()) if len(dist) else 0.0)


def orientation(nu):
    dirs = lattice_dirs()
    a_ax = min(angle(nu, d) for k, d in dirs if k == "axis")
    a_dg = min(angle(nu, d) for k, d in dirs if k == "diag")
    cls = "axis" if a_ax <= 1.0 else ("diag" if a_dg <= 1.0 else "neither")
    return dict(cls=cls, angle_axis=a_ax, angle_diag=a_dg, angles_to_box_axes=[angle(nu, np.eye(3)[i]) for i in range(3)])


def _rot(nu, ang):
    K = np.array([[0, -nu[2], nu[1]], [nu[2], 0, -nu[0]], [-nu[1], nu[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def Pm(m6, x, c, nu, ms=(1, 2, 3, 4), corotate=True):
    nu, e1, _ = frame(nu)
    ph = 2 * np.pi * np.arange(NPHI) / NPHI
    Rk = np.array([_rot(nu, p) for p in ph])
    num = {m: 0.0 for m in ms}
    den = 0.0
    for rho in (1.0, 2.0, 3.0, 4.0):
        for z in (-1.0, 0.0, 1.0):
            y = rho * e1 + z * nu
            P = c + np.einsum("kab,b->ka", Rk, y)
            M = interp(m6, x, P)
            Mt = np.einsum("kba,kbc,kcd->kad", Rk, M, Rk) if corotate else M
            F = np.fft.fft(Mt, axis=0) / NPHI
            den += float((np.abs(F[0]) ** 2).sum())
            for m in ms:
                num[m] += float((np.abs(F[m]) ** 2).sum() + (np.abs(F[-m]) ** 2).sum())
    return {m: num[m] / den for m in ms}


def snapped_axis(line, h):
    """The axis of P_m: snapped to the box center and the lattice direction within h/4 and 1 degree."""
    c = line["c"]
    nu = line["nu"]
    c_ok = float(np.linalg.norm(c)) <= h / 4
    best = min(((angle(nu, d), k, d) for k, d in lattice_dirs()), key=lambda t: t[0])
    d_ok = best[0] <= 1.0
    cs = np.zeros(3) if c_ok else c
    ns = best[2] if d_ok else nu
    regime = bool(c_ok and d_ok and best[1] in ("axis", "diag"))
    return cs, ns, dict(center_snapped=bool(c_ok), center_offset=float(np.linalg.norm(c)), direction_snapped=bool(d_ok),
                        direction_kind=best[1] if d_ok else None, direction_angle=best[0], regime=regime)


# ----------------------------------------------------------------------------- N/U, the degree, the labels, the loophole

def laue(M, h, mask, kap=1.0, LV=1.0, c=(1.0, 1.0, 1.0)):
    N = M.shape[0]
    s = A.sfun(M, c)
    D = {(a, sg): (A._shift(M, a, sg) - M) / (sg * h) for a in range(3) for sg in (1, -1)}
    chi = {(a, sg): A._chi(N, a, sg) for a in range(3) for sg in (1, -1)}
    Ns = []
    for a in range(3):
        V = 0.5 * (chi[(a, 1)][..., None, None] * D[(a, 1)] + chi[(a, -1)][..., None, None] * D[(a, -1)])
        dens = LV * s * A._nrm(V)
        for b in range(3):
            for t in (1, -1):
                dens = dens + kap * 0.5 * chi[(b, t)] * A._nrm(A._cm(V, D[(b, t)]))
        Ns.append(float(2 * h ** 3 * (dens * mask).sum()))
    return Ns


_SURF = {}


def surfaces(x):
    key = (len(x), float(x[-1]))
    if key not in _SURF:
        N, h, L = len(x), x[1] - x[0], x[-1]
        c = (N - 1) // 2
        ws = [float(w) for w in np.arange(3.5, L - h + 1e-9, 0.5)]
        ms = sorted({int(round(w / h)) for w in ws} | {int(round((L - h) / h))})
        _SURF[key] = [DG.Surface(N, [c - m] * 3, [c + m] * 3, half_width=m * h) for m in ms]
    return _SURF[key]


def degree(m6, x, ring_extent=None):
    n, g = DG.director(m6)
    rows = []
    for S in surfaces(x):
        r = S.read(n, g)
        surrounds = ring_extent is None or ring_extent < S.W - 1e-9
        rows.append(dict(W=S.W, liftable=r["liftable"], abs_q=r["abs_q"], q=r["q"], reason=r["reason"], min_gap=r["min_gap"], surrounds_ring=bool(surrounds)))
    used = [q["abs_q"] for q in rows if q["liftable"] and q["surrounds_ring"]]
    return dict(surfaces=rows, values=used, reads_one=bool(len(used) > 0 and all(v == 1 for v in used)), n_used=len(used))


def labels(mod, m6):
    I = mod.inertia(m6)
    idn = I["density"]
    h3 = mod.h ** 3
    share = h3 * idn / I["I"]
    Nn = idn.shape[0]
    pad = np.pad(idn, 1)
    cnt = np.pad(np.ones_like(idn), 1)
    S = np.zeros_like(idn)
    Cn = np.zeros_like(idn)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                S += pad[a:a + Nn, b:b + Nn, c:c + Nn]
                Cn += cnt[a:a + Nn, b:b + Nn, c:c + Nn]
    nb_mean = (S - idn) / (Cn - 1)
    big = share > 1e-4
    ratio = float((idn[big] / np.maximum(nb_mean[big], 1e-300)).max()) if big.any() else 0.0
    smax = float(share.max())
    label = "SMOOTH" if (smax <= 0.01 and ratio <= 50.0) else ("ONE-CELL" if smax > 0.5 else "MIXED")
    # the author's readout rule, printed beside
    padE = np.pad(idn, 1, mode="edge")
    nbE = sum(padE[1 + a:Nn + 1 + a, 1 + b:Nn + 1 + b, 1 + c:Nn + 1 + c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)) - idn
    spike = float((idn / (nbE / 26 + 1e-30))[1:-1, 1:-1, 1:-1].max())
    shareA = float(idn.max() / idn.sum())
    labA = "ONE-CELL" if shareA > 0.5 else ("SMOOTH" if (shareA < 0.01 and spike < 50) else "MIXED")
    return dict(label=label, largest_share=smax, largest_over_neighbors=ratio, author_label=labA, author_share=shareA, author_spike=spike,
                I=I["I"], I_LV=I["I_LV"], I_4=I["I_4"])


def loophole(M, c=(1.0, 1.0, 1.0)):
    s = A.sfun(M, c)
    _, V = np.linalg.eigh(M)
    n = V[..., :, 2]
    cnt = 0
    tot = 0
    for a in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[a] = slice(0, -1)
        s1[a] = slice(1, None)
        both = (s[tuple(s0)] < 1e-4) & (s[tuple(s1)] < 1e-4)
        cs = np.abs((n[tuple(s0)] * n[tuple(s1)]).sum(-1))
        cnt += int((both & (cs < np.cos(np.radians(45)))).sum())
        tot += int(both.sum())
    return dict(jump_links=cnt, vacuum_links=tot)


def read_state(m6, x, half_box, weight, withhold_first_guess=False, with_grad=True):
    """Every read of section 4 on one state (faces as given)."""
    N = len(x)
    h = x[1] - x[0]
    m6 = np.asarray(m6, float).reshape(N, N, N, 6)
    mod = A.Model(N, half_box, weight=weight)
    p = mod.evaluate(m6, need_grad=with_grad)
    out = dict(U=p["U"], U4=p["U4"], UV=p["UV"], ULV=p["ULV"])
    if with_grad:
        out["gate"] = float(np.abs(p["grad"]).max() / h ** 3)
    other = A.Model(N, half_box, weight="end" if weight == "mid" else "mid").evaluate(m6, need_grad=False)
    out["U_other_weight"] = other["U"]
    M = A.from6(m6)
    w = np.linalg.eigvalsh(M)
    g = w[..., 2] - w[..., 1]
    out.update(eigenvalue_min=float(w.min()), eigenvalue_max=float(w.max()), least_gap_node=float(g.min()))
    out.update(labels(mod, m6))
    line = line_reader(m6, x, half_box, withhold_first_guess)
    if line["ok"]:
        out.update(RL=line["RL"], center=line["c"].tolist(), normal=line["nu"].tolist(), line_regime=line["regime"], line_gmax=line["gmax"],
                   halfturn_all=line["halfturn_all"], out_of_plane_spread=line["out_of_plane_spread"], m2_harmonic_rho=line["m2_harmonic"],
                   RA=line["RA"], RA_nodes=line["RA_nodes"], first_guess=line["first_guess"],
                   rho_star=line["rho_star"].tolist(), phi_star=line["phi_star"].tolist(), zeta_star=((line["crossings"] - line["c"]) @ line["nu"]).tolist())
        out.update(orientation(line["nu"]))
        out.update(one_ring(m6, x, line))
        cs, ns, snap = snapped_axis(line, h)
        out["snap"] = snap
        out["P"] = Pm(m6, x, cs, ns)
        ext = float(np.abs(line["crossings"]).max()) + 0.0
        out["ring_max_norm"] = ext
    else:
        out.update(RL=None, line_regime=False, one_ring=False, cls=None, RA=line.get("RA"), RA_nodes=line.get("RA_nodes"), first_guess=line.get("first_guess"))
        ext = None
    deg = degree(m6, x, ext)
    out["degree"] = deg
    mask = np.zeros((N, N, N), bool)
    mask[1:-1, 1:-1, 1:-1] = True
    Ns = laue(M, h, mask)
    out.update(N_axes=Ns, N_over_U=float(np.mean(Ns) / p["U"]), Nvir_over_U=float((2 / 3 * p["ULV"] + 4 / 3 * p["U4"]) / p["U"]))
    out["loophole"] = loophole(M)
    return out


# ----------------------------------------------------------------------------- K4

def k4(out_file):
    res = dict()
    allok = True
    # (a) the line reader on the constructed rings
    a_rows = []
    jit = {}
    L = 6.0
    ok_a = True
    for h in (0.5, 0.375):
        N = int(round(2 * L / h)) + 1
        x = np.linspace(-L, L, N)
        for on, (nu, e1) in ORIENT.items():
            nu_u = np.asarray(nu) / np.linalg.norm(nu)
            errs = []
            erA = []
            for off in (0.0, 1 / 3, 2 / 3):
                R0 = 2.75 + off * h
                m6 = ring_field(x, R0, nu, e1)
                ln = line_reader(m6, x, L)
                orr = one_ring(m6, x, ln)
                row = dict(h=h, orientation=on, R0=R0, first_guess=ln["first_guess"], RA_err=(ln["RA"] - R0) if ln["RA"] is not None else None,
                           nuA_err=angle(ln["RA_nu"], nu_u) if ln["RA_nu"] is not None else None, RL_err=ln["RL"] - R0, nuL_err=angle(ln["nu"], nu_u),
                           center_offset=float(np.linalg.norm(ln["c"])), gmax=ln["gmax"], halfturn_all=ln["halfturn_all"], regime=ln["regime"],
                           one_ring=orr["one_ring"], cls=orientation(ln["nu"])["cls"])
                row["passed"] = bool(abs(row["RL_err"]) <= 0.01 and row["nuL_err"] <= 0.5 and row["regime"] and row["one_ring"])
                ok_a &= row["passed"]
                errs.append(row["RL_err"])
                if row["RA_err"] is not None:
                    erA.append(row["RA_err"])
                a_rows.append(row)
                print("K4a", json.dumps(row, default=float), flush=True)
            jL = max(errs) - min(errs)
            jA = (max(erA) - min(erA)) if erA else None
            jit[f"{h}_{on}"] = dict(jL=jL, jA=jA, passed=bool(jL < 0.005))
            ok_a &= jL < 0.005
    # the fallback start with the first guess withheld, h = 0.5
    fb_rows = []
    h = 0.5
    N = 25
    x = np.linspace(-L, L, N)
    for on, (nu, e1) in ORIENT.items():
        nu_u = np.asarray(nu) / np.linalg.norm(nu)
        errs = []
        for off in (0.0, 1 / 3, 2 / 3):
            R0 = 2.75 + off * h
            m6 = ring_field(x, R0, nu, e1)
            ln = line_reader(m6, x, L, withhold_first_guess=True)
            row = dict(h=h, orientation=on, R0=R0, RL_err=ln["RL"] - R0, nuL_err=angle(ln["nu"], nu_u), regime=ln["regime"], first_guess=ln["first_guess"])
            row["passed"] = bool(abs(row["RL_err"]) <= 0.01 and row["nuL_err"] <= 0.5 and row["regime"])
            ok_a &= row["passed"]
            errs.append(row["RL_err"])
            fb_rows.append(row)
            print("K4a fallback", json.dumps(row, default=float), flush=True)
        jf = max(errs) - min(errs)
        ok_a &= jf < 0.005
        jit[f"fallback_{h}_{on}"] = dict(jL=jf, passed=bool(jf < 0.005))
    # one ring: two constructed rings must fail
    two = []
    for zc in (3.0, -3.0):
        two.append(ring_field(x, 1.5, [0, 0, 1.0], [1.0, 0, 0], c=(0, 0, zc)))
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1)
    m6two = np.where((X[..., 2] >= 0)[..., None], two[0], two[1])
    ln2 = line_reader(m6two, x, L)
    or2 = one_ring(m6two, x, ln2)
    ok_two = not or2["one_ring"]
    ok_a &= ok_two
    res["K4a"] = dict(passed=bool(ok_a), rows=a_rows, fallback_rows=fb_rows, jitter=jit, two_rings=dict(one_ring=or2["one_ring"], detail=or2, passed=bool(ok_two)))
    print("K4a", "PASS" if ok_a else "FAIL", json.dumps(jit), "two rings one_ring =", or2["one_ring"], flush=True)
    allok &= ok_a
    # (b) P_2 on the axisymmetric and elliptic rings, both classes, the three spacings
    b_rows = []
    bounds = {}
    ok_b = True
    for h in (0.5, 0.375, 0.25):
        N = int(round(2 * L / h)) + 1
        x = np.linspace(-L, L, N)
        for on in ("axis", "diag"):
            nu, e1 = ORIENT[on]
            nu_u = np.asarray(nu) / np.linalg.norm(nu)
            P0 = Pm(ring_field(x, 2.75, nu, e1), x, np.zeros(3), nu_u)
            P2 = Pm(ring_field(x, 2.75, nu, e1, eps=1e-2), x, np.zeros(3), nu_u)
            P3 = Pm(ring_field(x, 2.75, nu, e1, eps=1e-3), x, np.zeros(3), nu_u)
            ratio = P2[2] / P3[2]
            ok = P0[2] <= 1e-20 and 90.0 <= ratio <= 110.0
            ok_b &= ok
            bounds[f"{h}_{on}"] = dict(B_abs=P3[2] / 100, B_pres=P2[2])
            nc = Pm(ring_field(x, 2.75, nu, e1), x, np.zeros(3), nu_u, corotate=False)
            row = dict(h=h, cls=on, P2_axisym=P0[2], P_axisym=P0, P2_eps1e2=P2[2], P2_eps1e3=P3[2], ratio=ratio, B_abs=P3[2] / 100, B_pres=P2[2],
                       failing_input_no_corotation_P2=nc[2], passed=bool(ok))
            b_rows.append(row)
            print("K4b", json.dumps(row, default=float), flush=True)
    res["K4b"] = dict(passed=bool(ok_b), rows=b_rows, bounds=bounds)
    allok &= ok_b
    # (c) N/U
    c_rows = []
    ok_c = True
    posted = dict(G1=0.947, G4=0.968, G6=1.005)
    printed = dict(G1=("18.478", "19.573", "19.573"), G4=("20.248", "20.79", "20.79"))
    for tag in ("G1", "G4", "G6"):
        m6, x, hb = A.load_stored(tag)
        N = len(x)
        mod = A.Model(N, hb, weight="end")
        m6 = mod.hold_faces(m6)
        U = mod.evaluate(m6, False)["U"]
        mask = np.zeros((N, N, N), bool)
        mask[1:-1, 1:-1, 1:-1] = True
        Ns = laue(A.from6(m6), mod.h, mask)
        nu = float(np.mean(Ns) / U)
        ok = abs(nu - posted[tag]) <= 0.001
        row = dict(grid=tag, N_axes=Ns, N_over_U=nu, posted=posted[tag], diff=nu - posted[tag])
        if tag in printed:
            pk = [A._printed_ok(v, t) for v, t in zip(Ns, printed[tag])]
            row.update(printed=list(printed[tag]), printed_match=pk)
            ok &= all(pk)
        row["passed"] = bool(ok)
        ok_c &= ok
        c_rows.append(row)
        print("K4c", json.dumps(row), flush=True)
    k = 0.7
    Lam = np.diag([0.8, 0.3, 0.0])
    Gx = np.array([[0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    tw = {}
    for h in (0.5, 0.25):
        N = int(round(6.0 / h)) + 1
        x = np.linspace(-3.0, 3.0, N)
        Rz = np.zeros((N, 3, 3))
        Rz[:, 0, 0] = 1
        Rz[:, 1, 1] = np.cos(k * x)
        Rz[:, 1, 2] = -np.sin(k * x)
        Rz[:, 2, 1] = np.sin(k * x)
        Rz[:, 2, 2] = np.cos(k * x)
        Mz = Rz @ Lam @ np.transpose(Rz, (0, 2, 1))
        M = np.broadcast_to(Mz[None, None], (N, N, N, 3, 3)).copy()
        mask = np.zeros((N, N, N), bool)
        mask[1:-1, 1:-1, 1:-1] = True
        Ns = laue(M, h, mask)
        dMz = k * (Gx @ Mz - Mz @ Gx)
        sv = A.sfun(Mz)
        ex = float((2 * sv * A._nrm(dMz))[1:-1].sum() * (N - 2) ** 2 * h ** 3)
        tw[h] = dict(Nx=Ns[0], Ny=Ns[1], Nz=Ns[2], exact=ex, err=Ns[2] - ex)
    ratio = tw[0.5]["err"] / tw[0.25]["err"]
    ok_t = all(tw[h]["Nx"] == 0.0 and tw[h]["Ny"] == 0.0 for h in tw) and 3.0 <= ratio <= 5.0
    ok_c &= ok_t
    res["K4c"] = dict(passed=bool(ok_c), rows=c_rows, twisted=dict(rows={str(k_): v for k_, v in tw.items()}, ratio=ratio, passed=bool(ok_t)))
    print("K4c", "PASS" if ok_c else "FAIL", json.dumps(res["K4c"]["twisted"]), flush=True)
    allok &= ok_c
    # (d) the degree: its own selftest and the seed of every grid
    sf = CK / "k4d_m5_46_selftest.json"
    rc = subprocess.run([sys.executable, "-I", str(HERE / "m5_46_degree.py"), "--selftest", "--out", str(sf)], capture_output=True, text=True)
    (CK / "k4d_m5_46_selftest.log").write_text(rc.stdout + rc.stderr)
    seeds = {}
    ok_s = True
    for tag, (hb, h, N) in GRIDS.items():
        mod = A.Model(N, hb)
        d = degree(mod.seed6(), mod.x)
        seeds[tag] = dict(values=d["values"], all_liftable=all(q["liftable"] for q in d["surfaces"]), reads_one=d["reads_one"])
        ok_s &= d["reads_one"]
    ok_d = rc.returncode == 0 and ok_s
    res["K4d"] = dict(passed=bool(ok_d), m5_46_selftest_exit=rc.returncode, m5_46_last_line=rc.stdout.strip().splitlines()[-1] if rc.stdout.strip() else "", seeds=seeds)
    print("K4d", "PASS" if ok_d else "FAIL", rc.stdout.strip().splitlines()[-1] if rc.stdout.strip() else rc.stderr[-300:], json.dumps(seeds), flush=True)
    allok &= ok_d
    res["passed"] = bool(allok)
    Path(out_file).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("K4:", "ALL PASS" if allok else "FAIL", flush=True)
    return allok


def stored_reads(out_file):
    """Every read of section 4 on the author's four stored arrays (faces reset to M_0), both weights for U."""
    res = {}
    for tag in ("G1", "G2", "G4", "G6"):
        m6, x, hb = A.load_stored(tag)
        mod = A.Model(len(x), hb, weight="end")
        m6 = mod.hold_faces(m6)
        r = read_state(m6, x, hb, "end")
        res[tag] = r
        print("stored", tag, json.dumps({k: r.get(k) for k in ("U", "U_other_weight", "RL", "RA", "cls", "angle_axis", "angle_diag", "one_ring", "far_nodes", "label", "N_over_U", "P", "snap", "line_regime")}, default=float), flush=True)
    Path(out_file).write_text(json.dumps(res, indent=1, default=float) + "\n")


POSTED_U = dict(G1=20.281, G2=20.423, G4=21.287, G6=21.805)
POSTED_NU = dict(G1=0.947, G4=0.968, G6=1.005)
RELEASE_R = dict(G1=2.73, G2=2.74, G4=2.83, G6=2.82)
FIXED_RISE = dict(G1=("posted", 1.07), G2=("posted", 0.54), G4=("posted", 1.01), G6=("this plan", 1.69))
COLORS = dict(end="#2a78d6", mid="#eb6834", author="#1baf7a", ref="#8a8a85")
MARK = dict(a="o", b="s")


def _rec(name):
    f = CK / "chains" / f"{name}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d if d.get("finished") else None


def tau_L(h, k4):
    """max(0.01, 2 j_L), j_L the largest jitter of K4 (a) over the three orientations at that spacing; at a spacing
    that K4 (a) does not cover (0.25) the floor 0.01 (the strictest tolerance)."""
    js = [v["jL"] for k, v in k4["K4a"]["jitter"].items() if k.startswith(f"{h}_")]
    return max(0.01, 2 * max(js)) if js else 0.01


def _side(v, B):
    return 0 if v <= B["B_abs"] else (2 if v >= B["B_pres"] else 1)


def p2_bounds(r, h, k4):
    sn = r.get("snap") or {}
    if not sn.get("regime"):
        return None
    return k4["K4b"]["bounds"][f"{h}_{sn['direction_kind']}"]


def second_step(end, er, h, k4):
    tau = tau_L(h, k4)
    out = {}
    out["U"] = dict(end=end["U"], earlier=er["U"], ok=bool(abs(end["U"] / er["U"] - 1) <= 1e-4))
    if end.get("RL") is not None and er.get("RL") is not None:
        out["RL"] = dict(end=end["RL"], earlier=er["RL"], ok=bool(abs(end["RL"] - er["RL"]) <= tau / 10))
    else:
        out["RL"] = dict(end=end.get("RL"), earlier=er.get("RL"), ok=False)
    out["N_over_U"] = dict(end=end["N_over_U"], earlier=er["N_over_U"], ok=bool(abs(end["N_over_U"] - er["N_over_U"]) <= 0.0005))
    B = p2_bounds(end, h, k4)
    if B is not None and er.get("P") is not None:
        out["P2"] = dict(end=end["P"]["2"] if "2" in end["P"] else end["P"][2], earlier=er["P"]["2"] if "2" in er["P"] else er["P"][2])
        out["P2"]["ok"] = bool(_side(out["P2"]["end"], B) == _side(out["P2"]["earlier"], B))
    else:
        out["P2"] = dict(end=None, earlier=None, ok=False)
    out["degree"] = dict(end=end["degree"]["values"], earlier=er["degree"]["values"], ok=bool(end["degree"]["values"] == er["degree"]["values"]))
    out["cls"] = dict(end=end.get("cls"), earlier=er.get("cls"), ok=bool(end.get("cls") == er.get("cls")))
    out["one_ring"] = dict(end=end.get("one_ring"), earlier=er.get("one_ring"), ok=bool(end.get("one_ring") == er.get("one_ring")))
    # (after the audit) a state outside the regime of the line reader (no crossing at g12 <= 0.02 on every half-plane,
    # or no half-turn about every crossing) has no ring to read: its R_L, class and one ring carry nothing, so the
    # second step withholds them
    in_regime = bool(end.get("line_regime") and end.get("halfturn_all") and er.get("line_regime") and er.get("halfturn_all"))
    if not in_regime:
        for k in ("RL", "cls", "one_ring"):
            out[k]["ok"] = False
            out[k]["outside_regime"] = True
    return out


def rotation(end, er):
    """The angle between the ring normals of the earlier state and the end (degrees, the sign of a normal free)."""
    if end.get("normal") is None or er.get("normal") is None:
        return None
    return angle(end["normal"], er["normal"])


def _P(r, m):
    P = r.get("P")
    if P is None:
        return None
    return P.get(str(m), P.get(m))


def chain_row(name, k4):
    r = _rec(name)
    if r is None:
        return None
    g, w, s = name.split("_")
    e = r["end"]
    k5 = _rec(f"k5_{name}")
    k5s = "not run" if k5 is None else ("met" if k5["k5"]["passed"] else "not met")
    stalled = k5s == "not met"
    ss = second_step(e, r["earlier"], r["h"], k4)
    row = dict(chain=name, grid=g, weight=w, start=s, start_state=r.get("start_state"), end_state=r["end_state"], stalled=stalled,
               only_first_clause=r.get("only_first_clause"), iterations=r["iterations"], evaluations=r["evaluations"], message=r["message"],
               restarts=len(r["restarts"]), resumes=len(r["resumes"]), U=e["U"], U4=e["U4"], UV=e["UV"], ULV=e["ULV"], U_other_weight=e["U_other_weight"],
               gate=e["gate"], RL=e.get("RL"), RA=e.get("RA"), RA_nodes=e.get("RA_nodes"), first_guess=e.get("first_guess"), cls=e.get("cls"),
               angle_axis=e.get("angle_axis"), angle_diag=e.get("angle_diag"), normal=e.get("normal"), center=e.get("center"),
               one_ring=e.get("one_ring"), far_nodes=e.get("far_nodes"), line_regime=e.get("line_regime"), line_gmax=e.get("line_gmax"),
               out_of_plane_spread=e.get("out_of_plane_spread"), m2_harmonic_rho=e.get("m2_harmonic_rho"),
               label=e["label"], largest_share=e["largest_share"], largest_over_neighbors=e["largest_over_neighbors"], author_label=e["author_label"],
               eigenvalue_min=e["eigenvalue_min"], eigenvalue_max=e["eigenvalue_max"], least_gap_node=e["least_gap_node"],
               I=e["I"], P={m: _P(e, m) for m in (1, 2, 3, 4)}, snap=e.get("snap"), N_over_U=e["N_over_U"], Nvir_over_U=e["Nvir_over_U"], N_axes=e["N_axes"],
               degree=e["degree"]["values"], degree_reads_one=e["degree"]["reads_one"], loophole=e["loophole"],
               earlier=dict(iteration=r["earlier"]["iteration"], kind=r["earlier"]["kind"]), second_step=ss, k5=k5s,
               halfturn_all=e.get("halfturn_all"), rotation_deg_earlier_to_end=rotation(e, r["earlier"]),
               earlier_in_regime=bool(r["earlier"].get("line_regime") and r["earlier"].get("halfturn_all")),
               earlier_angles=[r["earlier"].get("angle_axis"), r["earlier"].get("angle_diag")],
               k5_detail=(k5["k5"] if k5 else None), k5_run=(dict(iterations=k5["iterations"], end_state=k5["end_state"], message=k5["message"]) if k5 else None),
               U_start=r.get("U_start"), pair=r.get("pair"),
               stops=[dict(iteration=q["iteration"], kind=q["kind"], U=q["U"], gate=q["gate"], RL=q.get("RL"), cls=q.get("cls"), label=q["label"]) for q in r["stops"]])
    row["qualifies"] = bool(row["end_state"] == "CONVERGED" and not stalled and k5s == "met" and row["label"] == "SMOOTH" and row["one_ring"])
    why = []
    if row["end_state"] != "CONVERGED":
        why.append("CAPPED")
    if stalled:
        why.append("STALLED (K5 not met)")
    if k5s == "not run" and row["end_state"] == "CONVERGED" and s == "a":
        why.append("K5 not run")
    if row["label"] != "SMOOTH":
        why.append(row["label"])
    if not row["one_ring"]:
        why.append("not one ring")
    row["why_no_word"] = why
    return row


def same_orientation(a, st):
    if a.get("cls") != st.get("cls"):
        return False
    if a["cls"] == "neither":
        return bool(abs(a["angle_axis"] - st["angle_axis"]) <= 1.0 and abs(a["angle_diag"] - st["angle_diag"]) <= 1.0)
    return True


def words_end(rows, stored, k4):
    out = {}
    for g in ("G1", "G2", "G3", "G4", "G5", "G6"):
        a = rows.get(f"{g}_end_a")
        st = stored.get(g)
        hb, h, N = GRIDS[g]
        tau = tau_L(h, k4)
        ent = []
        if a is None:
            out[g] = dict(entries=[], note="chain missing")
            continue
        base = None
        if not a["qualifies"]:
            base = "no label (" + ", ".join(a["why_no_word"]) + ")"
        elif st is None:
            base = "no label (no posted number on this grid: a table)"
        elif not same_orientation(a, st):
            base = "no label (other orientation)"
        ss = a["second_step"]

        def give(q, val, ref, cond_ok, cond_not, ss_key, extra=None):
            e = dict(quantity=q, value=val, reference=ref, difference=(val - ref if isinstance(val, float) and isinstance(ref, float) else None))
            if extra:
                e.update(extra)
            if base is not None:
                e["word"] = base
            elif not ss[ss_key]["ok"]:
                e["word"] = "no label (second step: end {} against earlier {})".format(ss[ss_key]["end"], ss[ss_key]["earlier"])
            elif cond_ok:
                e["word"] = "REPRODUCED"
            elif cond_not:
                e["word"] = "NOT REPRODUCED"
            else:
                e["word"] = "no label (between the bounds)"
            ent.append(e)

        if g in POSTED_U:
            Up = POSTED_U[g]
            give("U", a["U"], Up, abs(a["U"] - Up) <= 1e-3 * Up, abs(a["U"] - Up) > 1e-3 * Up, "U", dict(tolerance=1e-3 * Up))
        if st is not None and st.get("RL") is not None and a["RL"] is not None:
            give("R_L", a["RL"], st["RL"], abs(a["RL"] - st["RL"]) <= tau, abs(a["RL"] - st["RL"]) > tau, "RL", dict(tolerance=tau))
        if st is not None:
            B = p2_bounds(a, h, k4)
            p2 = a["P"][2]
            if B is None:
                ent.append(dict(quantity="m = 2", value=p2, reference="1e-11 (release)", word=base or "no label (the axis is not a box axis or a body diagonal)"))
            else:
                give("m = 2", p2, "1e-11 (release)", p2 <= B["B_abs"], p2 >= B["B_pres"], "P2", dict(B_abs=B["B_abs"], B_pres=B["B_pres"]))
        if g in POSTED_NU:
            give("N/U", a["N_over_U"], POSTED_NU[g], abs(a["N_over_U"] - POSTED_NU[g]) <= 0.005, abs(a["N_over_U"] - POSTED_NU[g]) > 0.005, "N_over_U", dict(tolerance=0.005))
        beside = dict(RA=a["RA"], release_radius=RELEASE_R.get(g), posted_range=[2.73, 2.83], release_m2=1e-11, P1=a["P"][1], P3=a["P"][3], P4=a["P"][4],
                      degree=a["degree"], posted_Q_over_4pi=0.999, Nvir_over_U=a["Nvir_over_U"])
        if g == "G3":
            beside["posted_running_value"] = "20.509 so far"
        out[g] = dict(chain=a["chain"], cls=a["cls"], angles=[a["angle_axis"], a["angle_diag"]], stored_cls=(st or {}).get("cls"),
                      stored_angles=([st["angle_axis"], st["angle_diag"]] if st else None), base=base, tau_L=tau, entries=ent, beside=beside)
    return out


def stationary(rows, stored, k4):
    out = {}
    for g in ("G1", "G2", "G4", "G6"):
        b = rows.get(f"{g}_end_b")
        st = stored[g]
        if b is None:
            out[g] = dict(word="not run")
            continue
        h = GRIDS[g][1]
        tau = tau_L(h, k4)
        drop = (st["U"] - b["U"]) / st["U"]
        dR = abs(b["RL"] - st["RL"]) if b["RL"] is not None else None
        fails = []
        if not drop < 1e-4:
            fails.append("energy")
        if dR is None or dR > tau:
            fails.append("radius")
        if b["cls"] != st["cls"]:
            fails.append("orientation")
        out[g] = dict(U_stored=st["U"], U_end=b["U"], fall_rel=drop, RL_stored=st["RL"], RL_end=b["RL"], dRL=dR, tau_L=tau, cls_stored=st["cls"], cls_end=b["cls"],
                      end_state=b["end_state"], label=b["label"], iterations=b["iterations"],
                      word="STATIONARY" if not fails else "NOT STATIONARY (" + ", ".join(fails) + ")")
    return out


def s1(rows):
    grids = []
    for g in ("G1", "G2", "G3", "G4", "G5", "G6"):
        e, m = rows.get(f"{g}_end_a"), rows.get(f"{g}_mid_a")
        if e is None or m is None:
            grids.append(dict(grid=g, qualifies=False, why="chain missing"))
            continue
        q = e["qualifies"] and m["qualifies"]
        d = dict(grid=g, qualifies=bool(q), why=None if q else dict(end=e["why_no_word"], mid=m["why_no_word"]))
        dU = m["U"] / e["U"] - 1
        dR = (m["RL"] / e["RL"] - 1) if (m["RL"] and e["RL"]) else None
        shape = bool(m["one_ring"] and m["label"] == "SMOOTH" and m["degree_reads_one"])
        stable = dict(energy=bool(e["second_step"]["U"]["ok"] and m["second_step"]["U"]["ok"]),
                      size=bool(e["second_step"]["RL"]["ok"] and m["second_step"]["RL"]["ok"]),
                      shape=bool(m["second_step"]["one_ring"]["ok"] and m["second_step"]["degree"]["ok"]))
        d.update(energy=dU, energy_holds=bool(abs(dU) < 0.03), size=dR, size_holds=bool(dR is not None and abs(dR) < 0.03), shape_holds=shape, stable=stable)
        grids.append(d)
    qual = [d for d in grids if d["qualifies"]]
    fails_stable = [(d["grid"], c) for d in qual for c in ("energy", "size", "shape") if not d[f"{c}_holds"] and d["stable"][c]]
    fails_unstable = [(d["grid"], c) for d in qual for c in ("energy", "size", "shape") if not d[f"{c}_holds"] and not d["stable"][c]]
    unstable_holds = [(d["grid"], c) for d in qual for c in ("energy", "size", "shape") if d[f"{c}_holds"] and not d["stable"][c]]
    if fails_stable:
        word = "NOT HELD (" + "; ".join(f"{g} {c}" for g, c in fails_stable) + ")"
    elif fails_unstable:
        word = "no label (a clause fails on a read that the second step withholds: " + "; ".join(f"{g} {c}" for g, c in fails_unstable) + ")"
    elif unstable_holds:
        word = "no label (a clause rests on a read that the second step withholds: " + "; ".join(f"{g} {c}" for g, c in unstable_holds) + ")"
    elif len(qual) >= 2:
        word = "HELD"
    else:
        word = f"no label (fewer than two qualifying grids: {len(qual)})"
    return dict(grids=grids, qualifying=[d["grid"] for d in qual], word=word)


def two_states(rows):
    out = {}
    for g in ("G1", "G2", "G3", "G4", "G5", "G6"):
        for w in ("end", "mid"):
            a, b = rows.get(f"{g}_{w}_a"), rows.get(f"{g}_{w}_b")
            if a is None or b is None:
                continue
            same = bool(a["RL"] is not None and b["RL"] is not None and abs(a["U"] / b["U"] - 1) <= 1e-5 and abs(a["RL"] - b["RL"]) <= 0.01
                        and a["cls"] == b["cls"] and a["degree"] == b["degree"])
            d = dict(same_state=same, dU_rel=a["U"] / b["U"] - 1, dRL=(a["RL"] - b["RL"]) if (a["RL"] is not None and b["RL"] is not None) else None,
                     cls_a=a["cls"], cls_b=b["cls"])
            if a["cls"] != b["cls"]:
                d["lower_U"] = "a" if a["U"] < b["U"] else "b"
            out[f"{g}_{w}"] = d
    return out


def midpoint_table(rows):
    out = []
    for g in ("G1", "G2", "G3", "G4", "G5", "G6"):
        for s in ("a", "b"):
            e, m = rows.get(f"{g}_end_{s}"), rows.get(f"{g}_mid_{s}")
            if e is None or m is None:
                continue
            d = dict(grid=g, start=s, U_end=e["U"], U_mid=m["U"], shift_pct=100 * (m["U"] / e["U"] - 1), RL_end=e["RL"], RL_mid=m["RL"],
                     radius_shift_pct=(100 * (m["RL"] / e["RL"] - 1) if (m["RL"] and e["RL"]) else None), cls_end=e["cls"], cls_mid=m["cls"],
                     fixed_field_shift_of_mid_state_pct=100 * (m["U"] / m["U_other_weight"] - 1), fixed_field_shift_of_end_state_pct=100 * (e["U_other_weight"] / e["U"] - 1))
            if s == "b" and g in FIXED_RISE:
                d["fixed_field_rise_of_author_state_pct"] = FIXED_RISE[g]
                d["three_clauses"] = dict(energy=bool(abs(m["U"] / e["U"] - 1) < 0.03), size=bool(d["radius_shift_pct"] is not None and abs(d["radius_shift_pct"]) < 3),
                                          shape=bool(m["one_ring"] and m["label"] == "SMOOTH" and m["degree_reads_one"]))
            out.append(d)
    return out


def plots(rows, stored, k4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6})
    files = {}
    # 1. energies against the spacing (half-box 6) and the box (spacing 0.5)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, gs, xk, xl in ((axs[0], ("G1", "G2", "G3"), 1, "spacing h (half-box 6)"), (axs[1], ("G1", "G4", "G6"), 0, "half-box L (spacing 0.5)")):
        for w in ("end", "mid"):
            for s in ("a", "b"):
                xs, ys = [], []
                for g in gs:
                    r = rows.get(f"{g}_{w}_{s}")
                    if r is not None:
                        xs.append(GRIDS[g][xk])
                        ys.append(r["U"])
                ax.plot(xs, ys, marker=MARK[s], ms=6, lw=1.5, color=COLORS[w], ls="-" if s == "a" else "--", mfc=COLORS[w] if s == "a" else "white",
                        label=f"{w} weight, chain {s}")
        px = [GRIDS[g][xk] for g in gs if g in POSTED_U]
        py = [POSTED_U[g] for g in gs if g in POSTED_U]
        ax.plot(px, py, ls="none", marker="x", ms=11, mew=1.5, color=COLORS["author"], zorder=1, label="the author's posted U (end weight)")
        ax.set_xlabel(xl)
        ax.set_ylabel("U")
    axs[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    f = PLOTS / "m5_51_energies.png"
    fig.savefig(f, dpi=110)
    plt.close(fig)
    files["energies"] = f.name
    # 2. the gap on a meridian plane of each state
    names = [f"{g}_{w}_{s}" for g in ("G1", "G2", "G3", "G4", "G5", "G6") for w in ("end", "mid") for s in ("a", "b")]
    fig, axs = plt.subplots(6, 4, figsize=(9, 13.5))
    for k, n in enumerate(names):
        ax = axs[k // 4, k % 4]
        r = rows.get(n)
        rec = _rec(n)
        if r is None or r["RL"] is None:
            ax.set_axis_off()
            continue
        z = np.load(CK / "chains" / n / rec["end_file"], allow_pickle=False)
        m6, x = z["m6"], z["x"]
        c = np.array(r["center"])
        nu, e1, _ = frame(r["normal"])
        Lp = min(x[-1] - 0.5, 5.0)
        t = np.linspace(-Lp, Lp, 161)
        S, Z = np.meshgrid(t, t, indexing="ij")
        P = c + S[..., None] * e1 + Z[..., None] * nu
        g = gap12(interp(m6, x, P))
        im = ax.imshow(g.T, origin="lower", extent=(-Lp, Lp, -Lp, Lp), cmap="Blues_r", vmin=0, vmax=1)
        ax.set_title(f"{n}  R_L {r['RL']:.3f}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(False)
    fig.colorbar(im, ax=axs, shrink=0.3, label="top gap g12 (meridian plane through the ring axis)")
    f = PLOTS / "m5_51_gap_meridian.png"
    fig.savefig(f, dpi=90)
    plt.close(fig)
    files["gap_meridian"] = f.name
    # 3. the line radius against the azimuth
    fig, axs = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for k, g in enumerate(("G1", "G2", "G3", "G4", "G5", "G6")):
        ax = axs[k // 3, k % 3]
        for w in ("end", "mid"):
            for s in ("a", "b"):
                rec = _rec(f"{g}_{w}_{s}")
                if rec is None or rec["end"].get("rho_star") is None:
                    continue
                ph = np.degrees(np.mod(rec["end"]["phi_star"], 2 * np.pi))
                o = np.argsort(ph)
                ax.plot(ph[o], np.array(rec["end"]["rho_star"])[o], color=COLORS[w], ls="-" if s == "a" else "--", lw=1.2, label=f"{w}, {s}")
        if g in stored and stored[g].get("rho_star") is not None:
            ph = np.degrees(np.mod(stored[g]["phi_star"], 2 * np.pi))
            o = np.argsort(ph)
            ax.plot(ph[o], np.array(stored[g]["rho_star"])[o], color=COLORS["author"], lw=1.2, ls=":", label="author's stored array")
        ax.set_title(f"{g} (half-box {GRIDS[g][0]:g}, h {GRIDS[g][1]:g})", fontsize=8)
        ax.set_xlabel("azimuth about the fitted axis (deg)")
        if k % 3 == 0:
            ax.set_ylabel("crossing radius rho*")
    axs[0, 0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    f = PLOTS / "m5_51_line_radius.png"
    fig.savefig(f, dpi=100)
    plt.close(fig)
    files["line_radius"] = f.name
    # 4. the spectrum P_m
    fig, axs = plt.subplots(2, 3, figsize=(11, 6), sharey=True)
    for k, g in enumerate(("G1", "G2", "G3", "G4", "G5", "G6")):
        ax = axs[k // 3, k % 3]
        off = {("end", "a"): -0.21, ("end", "b"): -0.07, ("mid", "a"): 0.07, ("mid", "b"): 0.21}
        for (w, s), dx in off.items():
            r = rows.get(f"{g}_{w}_{s}")
            if r is None or r["P"][1] is None:
                continue
            ax.semilogy([m + dx for m in (1, 2, 3, 4)], [max(r["P"][m], 1e-36) for m in (1, 2, 3, 4)], ls="none", marker=MARK[s], ms=6, color=COLORS[w],
                        mfc=COLORS[w] if s == "a" else "white", label=f"{w}, {s} ({r['cls']})")
        if g in stored and stored[g].get("P"):
            P = stored[g]["P"]
            ax.semilogy([1, 2, 3, 4], [max(P[str(m)] if str(m) in P else P[m], 1e-36) for m in (1, 2, 3, 4)], ls="none", marker="x", ms=7, mew=1.5,
                        color=COLORS["author"], label=f"author ({stored[g]['cls']})")
        h = GRIDS[g][1]
        for cl, lsx in (("axis", "--"), ("diag", ":")):
            B = k4["K4b"]["bounds"][f"{h}_{cl}"]
            ax.hlines([B["B_abs"], B["B_pres"]], 1.6, 2.4, colors=COLORS["ref"], linestyles=lsx, lw=1)
        ax.axhline(1e-11, color=COLORS["ref"], lw=0.8, alpha=0.6)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("m")
        ax.set_title(f"{g}", fontsize=8)
        ax.legend(fontsize=5.5, frameon=False)
        if k % 3 == 0:
            ax.set_ylabel("P_m")
    fig.tight_layout()
    f = PLOTS / "m5_51_spectrum.png"
    fig.savefig(f, dpi=100)
    plt.close(fig)
    files["spectrum"] = f.name
    return files


def expectations(rows, words, stat, mid, S1, k4):
    ex = []
    st = [stat[g]["word"] for g in ("G1", "G2", "G4")]
    ex.append(dict(expectation="STATIONARY on G1, G2 and G4 (end weight)", outcome=dict(zip(("G1", "G2", "G4"), st)), met=all(q == "STATIONARY" for q in st)))
    ex.append(dict(expectation="G6: STATIONARY, or NOT STATIONARY by the energy clause", outcome=stat["G6"]["word"],
                   met=stat["G6"]["word"] in ("STATIONARY", "NOT STATIONARY (energy)")))
    a_rows = [rows.get(f"{g}_{w}_a") for g in ("G1", "G2", "G3", "G4", "G5", "G6") for w in ("end", "mid")]
    ok = [r is not None and r["one_ring"] and r["degree_reads_one"] and r["label"] == "SMOOTH" for r in a_rows]
    ex.append(dict(expectation="every chain a forms one ring with degree 1, SMOOTH", outcome=[(r["chain"], r["one_ring"], r["degree"], r["label"]) for r in a_rows if r], met=all(ok)))
    rep = {g: [(e["quantity"], e["word"]) for e in words[g]["entries"]] for g in ("G1", "G2", "G4", "G6")}
    matched = {g: v for g, v in rep.items() if not any("other orientation" in w for _, w in v)}
    ex.append(dict(expectation="where the class matches, U, R_L and N/U REPRODUCED", outcome=matched,
                   met=all(w == "REPRODUCED" for v in matched.values() for q, w in v if q in ("U", "R_L", "N/U")) if matched else None))
    ex.append(dict(expectation="m = 2: a matching chain a on G1 or G4 under B_abs; on G2 a matching state carries no m = 2 word",
                   outcome={g: [w for q, w in rep[g] if q == "m = 2"] for g in ("G1", "G2", "G4", "G6")},
                   met=bool(all(w == "REPRODUCED" for q, w in rep["G1"] + rep["G4"] if q == "m = 2") and all(w.startswith("no label") for q, w in rep["G2"] if q == "m = 2"))))
    ex.append(dict(expectation="some entries may read no label (other orientation)", outcome=[g for g in rep if any("other orientation" in w for _, w in rep[g])], met=None))
    sh = {d["grid"]: d["shift_pct"] for d in mid if d["start"] == "a"}
    ex.append(dict(expectation="midpoint U above the end weight by 0.3 to 1.7 percent on G1, G2, G4, G6, by less on G3", outcome=sh,
                   met=all(0.3 <= sh.get(g, -1) <= 1.7 for g in ("G1", "G2", "G4", "G6")) and sh.get("G3", 99) < min(sh.get(g, 99) for g in ("G1", "G2", "G4", "G6"))))
    rs = {d["grid"]: d["radius_shift_pct"] for d in mid if d["start"] == "a"}
    ex.append(dict(expectation="the line radius within 1.5 percent (midpoint against end, chains a)", outcome=rs,
                   met=all(v is not None and abs(v) <= 1.5 for v in rs.values())))
    ex.append(dict(expectation="S1 HELD", outcome=S1["word"], met=S1["word"] == "HELD"))
    lp = {n: r["loophole"]["jump_links"] for n, r in rows.items() if r}
    ex.append(dict(expectation="the loophole count near zero on every relaxed state", outcome=lp, met=bool(all(v == 0 for v in lp.values()))))
    return ex


def _trim_stored(st):
    return {k: v for k, v in st.items() if k not in ("rho_star", "phi_star", "zeta_star")}


def summary():
    k4 = json.loads((CK / "controls_k4_try1.json").read_text())
    k13 = json.loads((CK / "controls_k1_k3_try2.json").read_text())
    stored = json.loads((CK / "stored_reads.json").read_text())
    names = [f"{g}_{w}_{s}" for s in ("a", "b") for g in ("G1", "G2", "G3", "G4", "G5", "G6") for w in ("end", "mid")]
    rows = {n: chain_row(n, k4) for n in names}
    words = words_end(rows, stored, k4)
    stat = stationary(rows, stored, k4)
    mid = midpoint_table(rows)
    S1 = s1(rows)
    two = two_states(rows)
    ex = expectations(rows, words, stat, mid, S1, k4)
    files = plots(rows, stored, k4)
    hashes = {f: __import__("hashlib").sha256((A.AUTHOR / f).read_bytes()).hexdigest() for f, _ in A.STORED.values()}
    k5 = {n: dict(run=r["k5_run"], detail=r["k5_detail"]) for n, r in rows.items() if r and r["k5"] != "not run"}
    out = dict(task="M5.51", plan_commit="e6da01ff",
               author_arrays_sha256=hashes,
               author_archives_sha256={"LdGS_milestone_arrays_20nov.zip": "9d1bc04ef8a2a3eba47501538c4fde4430e3ae4fc43f9a0c5d7c4d99a1d6ae9c",
                                       "LdGS_v42_engine_midpoint.zip": "87620fa4ef1729a484f15df758abf730566d4b4bbcd7e964f1b3f66f0b1a594b"},
               controls=dict(K1_K3=k13, K4=dict(passed=k4["passed"], jitter=k4["K4a"]["jitter"], two_rings=k4["K4a"]["two_rings"], P2_bounds=k4["K4b"]["bounds"],
                                                  K4b_rows=k4["K4b"]["rows"], K4c=k4["K4c"], K4d=k4["K4d"]), K5=k5),
               stored=_trim_stored_all(stored), chains={n: r for n, r in rows.items() if r}, words_end_weight=words, stationary=stat,
               midpoint_table=mid, S1=S1, two_states=two, expectations=ex, plots=files)
    post = {}
    for n in ("G2_end_a", "G2_end_b", "G2_mid_a", "G2_mid_b"):
        f = CK / "chains" / f"post_{n}.json"
        if f.exists():
            q = json.loads(f.read_text())
            post[n] = {k: q[k] for k in ("iterations", "message", "ftol", "gtol", "maxcor", "cap", "reads")}
    if post:
        out["after_audit_G2_continuations"] = dict(note="added after the audit: a descriptor with no word; ftol 0, otherwise the chain's options", chains=post)
    DATA.mkdir(exist_ok=True)
    f = DATA / "m5_51_summary.json"
    f.write_text(json.dumps(out, indent=1, default=_jsonable) + "\n")
    print("summary written", f, f.stat().st_size, "bytes")
    print("S1:", S1["word"])
    for g, v in words.items():
        print(g, v.get("base"), [(e["quantity"], e["word"]) for e in v["entries"]])
    for g, v in stat.items():
        print("STATIONARY", g, v["word"])


def _trim_stored_all(stored):
    return {g: _trim_stored(v) for g, v in stored.items()}


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k4", action="store_true")
    ap.add_argument("--stored", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--try-tag", default="try1")
    a = ap.parse_args()
    if a.k4:
        ok = k4(CK / f"controls_k4_{a.try_tag}.json")
        sys.exit(0 if ok else 1)
    if a.stored:
        stored_reads(CK / "stored_reads.json")
        return
    if a.summary:
        summary()
        return
    ap.print_help()


if __name__ == "__main__":
    main()
