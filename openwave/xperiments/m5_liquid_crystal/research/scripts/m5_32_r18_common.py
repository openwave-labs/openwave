"""M5.32 R18 shared instrument (ledger 6.7): the R17 instrument (m5_32_r17_common, consumed read-only; the R16 stack
under it) extended by (1) the per-cell-weighted static action, (2) the TUBE radial instrument (R18-1), (3) the
pinning penalty (R18-3), (4) the static twist (R18-2) and (5) the spin-2 zero counter on shells (R18-0c).  Each
extension carries its gate in the selftest; the tube's three run gates (against the 3D code) live in
m5_32_r18_1_radial.py.

EQUATIONS FIRST
---------------
1. The weighted static action.  E_w = sum_c w_c e_c(M), e_c the SAME per-cell static density as m5_32_r16_common.action
   (E_h + V4 + U + c_P K_P^proj + c_s rho^2 E2, plus U_v6 for object C through m5_32_r17_common.u_v6_cells, the plateau
   weight per cfg['weight']), w_c an arbitrary per-cell weight (h^3 everywhere reproduces action EXACTLY: the gate).
   The gradient is the R16 adjoint with the weight applied at the cell whose density is weighted (before the stencil
   adjoint, on every cotangent).
2. The tube.  The spherically symmetric sector of the 4x4 field: four radial profiles (m_g, lambda_1, lambda_23, beta)
   on the cell-centered grid r_j = (j + 1/2) h, j < N_g = L / 2h, linearly interpolated to any radius,
       M(x) = Lambda(beta(r), rhat) [ m_g e_0 e_0^T + lambda_1 rhat rhat^T + lambda_23 (I_3 - rhat rhat^T) ] Lambda^T,
       Lambda = I + (cosh beta - 1)(e_0 e_0^T + rhat rhat^T) + sinh beta (e_0 rhat^T + rhat e_0^T)   (the boost along rhat),
   so the N = M eta spectrum is (-m_g, lambda_1, lambda_23, lambda_23) with the director radial and the pair degenerate
   (the circle acts trivially: T_alpha M = M, the average equals the plain action, gated).  The tube lattice is the
   line x_i = (i + 1/2 - N_g) h, i < 2 N_g (both halves of one diameter) with its two transverse neighbors at +-h on
   each of the other two axes: an (2 N_g, 3, 3) lattice of cells carrying the ansatz's exact values, on which the 3D
   stencil (INS4.d1, the sym branches) gives the center line its correct derivatives.  The energy
       E_tube = sum_i 2 pi x_i^2 h e(x_i)   (the center line; the eight other lines weight 0)
   is the spherical quadrature of E = 4 pi int r^2 e dr.  With the center line placed at (y, z) = (h/2, h/2) the tube
   reproduces the 3D lattice's own ray cells and their neighbors exactly (gate 1 of R18-1: the densities agree to
   roundoff); the run uses (0, 0) (the true axis, r = |x_i|).  The profile gradient is the chain rule
       dE / dP_k[j] = sum_c (interpolation weight of node j at cell c) <grad_M(c), dM / dp_k(c)>,
   grad_M from the weighted action, dM / dp_k by complex step through the ansatz (exact to roundoff; gated against
   the complex step of the whole tube energy on the profiles at 1e-10 and against Richardson central differences).
   Descent: L-BFGS-B on (m_g, lambda_23, Delta = lambda_1 - lambda_23, beta) with the box bound Delta >= GAP_MIN
   (the admissible domain: escape (d) is the bound becoming active), the outer nodes r > L/2 - 1.6 pinned to the vacuum.
3. The pinning penalty.  E_pin = (k / 2) sum_{c in pinned} |M_c - M_c^ref|_F^2, gradient k (M - M^ref) on the pinned
   cells; k = 0 reproduces the free problem exactly (the gate).
4. The static twist.  M_q(x) = T_{theta(x)} M(x) with theta = q . x through the local generator (R(theta / 2) M R(theta / 2)^T,
   R = R(J) of the field's own frame); E(q) - E(0) = S q^2 / 2 + O(q^4) (the circle-averaged action is invariant under
   a uniform theta, so only grad theta costs); gate: even in q and quadratic to 1e-3 at q <= 0.1 on a split-carrying field.
5. The spin-2 zero counter.  On a lattice shell |r - r_c| < 0.75 h the split section zeta = S_ee - S_ff + 2 i S_ef (S = M's
   spatial block at the CELL: interpolating a hedgehog's projector between cells creates an artefact split of order (h / r)^2,
   4.5e-2 at r 3 on the exact hedgehog, so the samples are used as they are), (e, f) the transverse frame of the
   outward-oriented director in the polar frame about z, is a spin-weight-2 function on the sphere: it is fitted by least
   squares with the spin-weighted harmonics 2Y_lm (l <= 4 on shells under 150 cells, else 6; the residual reported), the
   fit is evaluated on a (theta, phi) grid and its zeros are the faces where arg zeta winds; the polar frame's own defect at
   the poles is +-2 (spin 2), so the pole caps carry the index ring_N + 2 and 2 - ring_S.  Poincare-Hopf: the indices sum
   to 4 on every shell (Euler number 4); the content is the number, index and position of the zeros.  A hull-triangle count
   on the raw samples (chart-free, resolution-limited: a zero straddling a symmetric sample pair is lost) is reported beside
   it.  Gate: the hedgehog plus a uniform traceless Q = xx - yy gives four simple zeros, Q = zz - (xx + yy) / 2 two index-2
   zeros at the poles, the tilted and the rotated Q total 4, the fit residual under 1e-3 on all of them.

Selftest: python3 m5_32_r18_common.py  ->  data/m5_32_r18_common_selftest.json
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

ARGV = list(sys.argv)
sys.argv = [sys.argv[0]]
import m5_32_r17_common as R                              # noqa: E402
import m5_32_r16_common as C                              # noqa: E402
sys.argv = ARGV
C15, INS4 = C.C15, C.INS4
ETA, EYE = C.ETA, C.EYE
RES, DATA, PLOTS = C.RES, C.DATA, C.PLOTS
CK = os.path.join(RES, "checkpoints", "m5_32_r18")
os.makedirs(CK, exist_ok=True)
T0 = time.time()
sym, W_to_gradM = C.sym, C.W_to_gradM
GAP_MIN = C.GAP_MIN


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


# ================================================================ 1. the weighted static action
def action_weighted(M, cfg, wc, fr=None, n_ref=None, need_grad=True):
    """E_w = sum_c wc_c e_c (the R16 static density per cell, plus U_v6 for object v6), its M-gradient, the per-cell
    density e_c (unweighted) and the parts (weighted sums)."""
    h = cfg["h"]
    comp = cfg["completion"]
    mu, cP, cs = cfg["mu"], cfg["cP"], cfg["cs"]
    with R.weight_mode(cfg):
        if fr is None:
            fr = C.frame(M, n_ref)
        Gm = fr["G"]
        cplx = np.iscomplexobj(M)
        num = (lambda x: complex(x)) if cplx else (lambda x: float(x))
        w = wc[..., None, None]
        spl, dspl = C15.split_cells(M, need_grad)
        rho2 = spl / 4.0
        v4, W_v4 = C.v4_cells(fr, cfg, need_grad)
        Eh = np.zeros(M.shape[:-2], dtype=M.dtype)
        e2, kp = np.zeros_like(Eh), np.zeros_like(Eh)
        adj = np.zeros_like(M) if need_grad else None
        Wst = np.zeros_like(M) if need_grad else None
        LamG = np.zeros_like(M) if need_grad else None
        for br, wt in INS4.branches(cfg["stencil"]):
            A = [INS4.d1(M, ax, h, br) for ax in range(3)]
            gA = [np.zeros_like(M) for _ in range(3)] if need_grad else None
            for i in range(3):
                for j in range(i + 1, 3):
                    d, dX, dY, dG = C.quartic_pair(A[i], A[j], Gm, comp, need_grad)
                    Eh = Eh + wt * 4.0 * d
                    if need_grad:
                        gA[i] += 4.0 * dX
                        gA[j] += 4.0 * dY
                        LamG += wt * 4.0 * w * dG
            for i in range(3):
                d, dA, dG = C.e2_cells(A[i], Gm, need_grad)
                e2 = e2 + wt * d
                if need_grad:
                    gA[i] += cs * rho2[..., None, None] * dA
                    LamG += wt * cs * (w * rho2[..., None, None]) * dG
            Ek, Wk, dAk = C.kp_cells(A, fr, need_grad)
            kp = kp + wt * Ek
            if need_grad:
                Wst += wt * cP * w * Wk
                for i in range(3):
                    gA[i] += cP * dAk[i]
                    adj += wt * INS4.d1_adj(sym(w * gA[i]), i, h, br)
        e_cells = Eh + v4 + mu * rho2 + cP * kp + cs * rho2 * e2
        parts = {"E_h": num(np.sum(wc * Eh)), "V4": num(np.sum(wc * v4)), "U": num(mu * np.sum(wc * rho2)), "KP": num(cP * np.sum(wc * kp)),
                 "reg": num(cs * np.sum(wc * rho2 * e2)), "rho2_sum": num(np.sum(wc * rho2))}
        if need_grad:
            Gst = W_to_gradM(Wst + w * W_v4 + C.W_through_G(LamG, fr))
            Gst += (wc * (mu + cs * e2))[..., None, None] * dspl / 4.0
            grad = Gst + adj
        else:
            grad = None
        if cfg.get("object") == "v6":
            U, gU, Wc, _ = R.u_v6_cells(M, cfg, fr, need_grad)
            e_cells = e_cells + U
            parts["U_v6"] = num(np.sum(wc * U))
            parts["W_max"] = float(np.max(np.real(Wc)))
            parts["mu_eff_min"] = float(np.min(np.real(cfg["mu_v6"] - cfg["gW"] * Wc)))
            if need_grad:
                grad = grad + w * gU
    parts["E_stat"] = num(np.sum(wc * e_cells))
    return parts["E_stat"], grad, parts, e_cells, fr


# ================================================================ 2. the tube
def tube_setup(h, L, y0=0.0, z0=0.0):
    """the tube lattice for spacing h and box L: the profile grid r_j, the line x_i, the (2 N_g, 3, 3, 3) cell positions,
    the interpolation indices / weights per cell, the spherical quadrature weights on the center line."""
    Ng = int(round(L / (2.0 * h)))
    rg = (np.arange(Ng) + 0.5) * h
    x = (np.arange(2 * Ng) + 0.5 - Ng) * h
    pos = np.zeros((2 * Ng, 3, 3, 3))
    pos[..., 0] = x[:, None, None]
    pos[..., 1] = y0 + (np.arange(3) - 1)[None, :, None] * h
    pos[..., 2] = z0 + (np.arange(3) - 1)[None, None, :] * h
    r = np.sqrt(np.sum(pos * pos, axis=-1))
    rhat = pos / np.maximum(r, 1e-300)[..., None]
    u = r / h - 0.5
    j = np.clip(np.floor(u).astype(int), 0, Ng - 2)
    t = np.clip(u - j, 0.0, 1.0)
    wc = np.zeros((2 * Ng, 3, 3))
    wc[:, 1, 1] = 2.0 * np.pi * r[:, 1, 1] ** 2 * h
    return {"h": h, "L": L, "Ng": Ng, "rg": rg, "x": x, "pos": pos, "r": r, "rhat": rhat, "j": j, "t": t, "wc": wc, "y0": y0, "z0": z0}


def interp_profiles(P, tube):
    """P (4, N_g) -> the four values per cell (4, 2 N_g, 3, 3), linear interpolation on the profile grid (complex-safe)."""
    j, t = tube["j"], tube["t"]
    return (1.0 - t)[None] * P[:, j] + t[None] * P[:, j + 1]


def ansatz_cells(vals, rhat):
    """M per cell from the interpolated (m_g, lambda_1, lambda_23, beta) and the unit radial vectors (complex-safe)."""
    mg, l1, l23, be = vals
    shape = mg.shape
    dt = np.result_type(mg, rhat)
    M0 = np.zeros(shape + (4, 4), dtype=dt)
    M0[..., 0, 0] = mg
    I3 = np.broadcast_to(np.eye(3), shape + (3, 3))
    rr = rhat[..., :, None] * rhat[..., None, :]
    M0[..., 1:, 1:] = l23[..., None, None] * I3 + (l1 - l23)[..., None, None] * rr
    ch, sh = np.cosh(be), np.sinh(be)
    Lam = np.zeros(shape + (4, 4), dtype=dt)
    Lam[..., 0, 0] = ch
    Lam[..., 0, 1:] = sh[..., None] * rhat
    Lam[..., 1:, 0] = sh[..., None] * rhat
    Lam[..., 1:, 1:] = I3 + (ch - 1.0)[..., None, None] * rr
    return Lam @ M0 @ np.swapaxes(Lam, -1, -2)


def tube_field(P, tube):
    return ansatz_cells(interp_profiles(P, tube), tube["rhat"])


def field_3d_from_profiles(P, tube, cfg):
    """the same ansatz on the 3D lattice of cfg (the same h): the gate-1 / gate-2 field."""
    X, Y, Z = INS4.coords(cfg["n"], cfg["h"])
    pos = np.stack([X, Y, Z], -1)
    r = np.sqrt(np.sum(pos * pos, -1))
    rhat = pos / r[..., None]
    u = r / tube["h"] - 0.5
    j = np.clip(np.floor(u).astype(int), 0, tube["Ng"] - 2)
    t = np.clip(u - j, 0.0, 1.0)
    vals = (1.0 - t)[None] * P[:, j] + t[None] * P[:, j + 1]
    return ansatz_cells(vals, rhat)


def tube_energy(P, tube, cfg, need_grad=True, n_ref=None):
    """E_tube(P) and dE / dP (4, N_g); parts, the center-line density e(x_i), the frame."""
    vals = interp_profiles(P, tube)
    M = ansatz_cells(vals, tube["rhat"])
    E, gM, parts, e_cells, fr = action_weighted(M, cfg, tube["wc"], n_ref=n_ref, need_grad=need_grad)
    e_line = e_cells[:, 1, 1]
    if not need_grad:
        return E, None, parts, e_line, fr
    # dM / d(cell value of type k) by complex step through the ansatz, contracted with grad_M, scattered to the nodes
    gP = np.zeros_like(P)
    eps = 1e-20
    j, t = tube["j"], tube["t"]
    for k in range(4):
        vk = vals.astype(complex).copy()
        vk[k] = vk[k] + 1j * eps
        dM = np.imag(ansatz_cells(vk, tube["rhat"])) / eps
        dc = np.sum(gM * dM, axis=(-1, -2))
        np.add.at(gP[k], j.reshape(-1), ((1.0 - t) * dc).reshape(-1))
        np.add.at(gP[k], (j + 1).reshape(-1), (t * dc).reshape(-1))
    return E, gP, parts, e_line, fr


def vacuum_profiles(cfg, Ng):
    vac = INS4.vac4(cfg)
    P = np.zeros((4, Ng))
    P[0], P[1], P[2], P[3] = vac[0, 0], vac[1, 1], vac[2, 2], 0.0
    return P


def profiles_from_field(M, cfg, tube):
    """the shell means (1.5 h shells) of (-lambda_g, lambda_1, (lambda_2 + lambda_3) / 2) of a 3D field on the tube's grid,
    beta = 0 (every field of this program has M_0i = 0); the vacuum beyond the last shell."""
    n, h = cfg["n"], cfg["h"]
    X, Y, Z = INS4.coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    fr = C.frame(M, C.radial_ref(cfg))
    lg, l1, s = (np.real(fr[k]) for k in ("lg", "l1", "s"))
    edges = np.arange(0.0, cfg["L"] / 2 + h, 1.5 * h)
    rs, vg, v1, v23 = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if np.sum(m) < 4:
            continue
        rs.append(np.mean(r[m])); vg.append(-np.mean(lg[m])); v1.append(np.mean(l1[m])); v23.append(np.mean(s[m]) / 2.0)
    rs = np.array(rs)
    P = vacuum_profiles(cfg, tube["Ng"])
    for k, v in ((0, vg), (1, v1), (2, v23)):
        P[k] = np.interp(tube["rg"], rs, np.array(v), right=P[k][-1])
    return P


def melted_seed(cfg, tube, r_m=2.0):
    """the melted-core seed: lambda_1 = lambda_23 + max(0.7 (1 - exp(-(r / r_m)^2)), GAP_MIN), m_g and lambda_23 at the vacuum."""
    P = vacuum_profiles(cfg, tube["Ng"])
    P[1] = P[2] + np.maximum((P[1] - P[2]) * (1.0 - np.exp(-(tube["rg"] / r_m) ** 2)), GAP_MIN)
    return P


def pack(P):
    """(m_g, lambda_1, lambda_23, beta) -> the descent variables (m_g, lambda_23, Delta, beta)."""
    return np.stack([P[0], P[2], P[1] - P[2], P[3]])


def unpack(Q):
    return np.stack([Q[0], Q[1] + Q[2], Q[1], Q[3]])


def profile_reads(P, tube, cfg, e_line=None, parts=None):
    """r_0 under the three R17-0 definitions, Delta_min, the tail amplitude of the 1 / r^4 density, the parts."""
    rg, h, L = tube["rg"], tube["h"], tube["L"]
    l1, l23 = P[1], P[2]
    gap = l1 - l23
    out = {"Delta_min": float(np.min(gap)), "r_at_Delta_min": float(rg[int(np.argmin(gap))]), "lambda_1_center": float(l1[0]), "lambda_23_center": float(l23[0]), "m_g_center": float(P[0][0]), "beta_max_abs": float(np.max(np.abs(P[3])))}
    below = np.where(l1 < C.PL_HI)[0]
    out["r_0_taper (max r with lambda_1 < 0.8)"] = float(rg[below[-1]]) if len(below) else 0.0
    half = np.where(l1 < 0.5 * (1.0 + C.DELTA))[0]
    out["r_0_half (max r with lambda_1 < (1+delta)/2)"] = float(rg[half[-1]]) if len(half) else 0.0
    cross = np.where((l1[:-1] < 0.8) & (l1[1:] >= 0.8))[0]
    if len(cross):
        c = cross[-1]
        out["r_0_profile (lambda_1 = 0.8 crossing)"] = float(rg[c] + (0.8 - l1[c]) * (rg[c + 1] - rg[c]) / (l1[c + 1] - l1[c]))
    else:
        out["r_0_profile (lambda_1 = 0.8 crossing)"] = 0.0
    mu = cfg.get("mu_v6", cfg["mu"]) if cfg.get("object") == "v6" else cfg["mu"]
    mu = 1e-2 if mu == 0.0 else mu
    for k in list(out):
        if k.startswith("r_0"):
            out[k + " x sqrt(mu)"] = out[k] * np.sqrt(mu)
    out["r_0_profile_over_h"] = out["r_0_profile (lambda_1 = 0.8 crossing)"] / h
    if e_line is not None:
        x = np.abs(tube["x"])
        e = np.real(e_line)
        sel = (x >= 0.15 * L) & (x < 0.42 * L)
        out["tail_A_median_e_r4"] = float(np.median(e[sel] * x[sel] ** 4)) if np.any(sel) else None
        ok = sel & (e > 0)
        if np.sum(ok) > 3:
            sl, ic = np.polyfit(np.log(x[ok]), np.log(e[ok]), 1)
            out["tail_loglog_slope"] = float(sl)
    if parts is not None:
        out["parts"] = {k: (float(np.real(v)) if not isinstance(v, (int, bool)) else v) for k, v in parts.items()}
    return out


def _grad_packed(Q, tube, cfg, free, n_ref=None):
    P = unpack(Q)
    E, gP, parts, e_line, fr = tube_energy(P, tube, cfg, n_ref=n_ref)
    gQ = np.stack([gP[0], gP[1] + gP[2], gP[1], gP[3]])
    gQ[:, ~free] = 0.0
    return E, gQ, parts, e_line, P


def hessian_diag(Q, tube, cfg, free, n_ref=None, colors=7):
    """the diagonal of the Hessian of E_tube in the packed variables by the COLORED complex step of the gradient: the nodes of one
    profile type at j = c (mod colors) are perturbed together by i eps (the stencil couples a node to its nearest few neighbors only,
    so the imaginary part of the gradient at a perturbed node is its own diagonal entry); 4 x colors gradient evaluations."""
    eps = 1e-20
    Ng = Q.shape[1]
    D = np.zeros_like(Q)
    for k in range(4):
        for c in range(colors):
            sel = (np.arange(Ng) % colors) == c
            Qc = Q.astype(complex).copy()
            Qc[k, sel] += 1j * eps
            _, gQ, _, _, _ = _grad_packed(Qc, tube, cfg, free, n_ref)
            D[k, sel] = np.imag(gQ[k, sel]) / eps
    return D


def solve_tube(P0, tube, cfg, pin_depth=1.6, maxiter=5000, gtol=1e-9, log_every=200, tag="", n_ref=None, callback=None, precondition=True):
    """L-BFGS-B on the packed profiles with the bound Delta >= GAP_MIN, the outer nodes pinned to the vacuum; with precondition=True
    the variables are scaled by 1 / sqrt(diag Hessian) at the seed (the m_g direction is 1e5 to 1e7 stiffer than the eigenvalue
    directions and the stiffness grows as r^2 with the quadrature weight: unpreconditioned L-BFGS-B crawled at gradients of 1 to
    40 after 4000 iterations on the L 96 / 144 boxes)."""
    from scipy.optimize import minimize
    Ng = tube["Ng"]
    Q0 = pack(P0)
    free = tube["rg"] < tube["L"] / 2.0 - pin_depth
    Qvac = pack(vacuum_profiles(cfg, Ng))
    Q0 = np.where(free[None, :], Q0, Qvac)
    if precondition:
        Hd = hessian_diag(Q0, tube, cfg, free, n_ref)
        floor = 1e-6 * np.max(np.abs(Hd))
        Dsc = 1.0 / np.sqrt(np.maximum(np.abs(Hd), floor))
        Dsc[:, ~free] = 1.0
    else:
        Hd = None
        Dsc = np.ones_like(Q0)
    lo = np.full(Q0.shape, -np.inf); hi = np.full(Q0.shape, np.inf)
    lo[2] = GAP_MIN
    lo[:, ~free] = Qvac[:, ~free]; hi[:, ~free] = Qvac[:, ~free]
    lo[2, ~free] = Qvac[2, ~free]
    # q = (Q - Q0) / D  ->  Q = Q0 + D q; the bounds transform node by node (D > 0)
    qlo = (lo - Q0) / Dsc; qhi = (hi - Q0) / Dsc
    state = {"n": 0, "t0": time.time(), "trace": []}

    def f(q):
        Q = Q0 + Dsc * q.reshape(4, Ng)
        E, gQ, parts, e_line, P = _grad_packed(Q, tube, cfg, free, n_ref)
        state["n"] += 1
        state["last"] = (E, parts, e_line, P, gQ)
        return float(E), (Dsc * gQ).reshape(-1)

    def cb(q):
        Q = Q0 + Dsc * q.reshape(4, Ng)
        E, parts, e_line, P, gQ = state["last"]
        it = len(state["trace"]) + 1
        row = {"it": it, "E": float(E), "Delta_min": float(np.min(Q[2])), "n_eval": state["n"], "grad_max": float(np.max(np.abs(gQ)))}
        state["trace"].append(row)
        if it % log_every == 0:
            print(f"  {tag} it {it:5d} E {E:14.8f} E_h {np.real(parts['E_h']):10.5f} KP {np.real(parts['KP']):.4e} V4 {np.real(parts['V4']):.4e} Delta_min {row['Delta_min']:.4f} l1(0) {P[1][0]:.4f} |g| {row['grad_max']:.1e} [{time.time() - state['t0']:.0f}s]", flush=True)
        if callback is not None:
            callback(P, state)

    res = minimize(f, np.zeros(Q0.size), jac=True, method="L-BFGS-B", bounds=list(zip(qlo.reshape(-1), qhi.reshape(-1))), callback=cb,
                   options={"maxiter": maxiter, "maxfun": 4 * maxiter, "gtol": gtol, "ftol": 1e-15, "maxcor": 30})
    Q = Q0 + Dsc * res.x.reshape(4, Ng)
    P = unpack(Q)
    E, gQ, parts, e_line, P = _grad_packed(Q, tube, cfg, free, n_ref)
    at_bound = (Q[2] <= GAP_MIN + 1e-12) & free
    info = {"success": bool(res.success), "message": str(res.message), "nit": int(res.nit), "n_eval": state["n"], "wall_s": round(time.time() - state["t0"], 1),
            "grad_max_free": float(np.max(np.abs(gQ))), "grad_max_by_type": [float(np.max(np.abs(gQ[k]))) for k in range(4)], "preconditioned_grad_max": float(np.max(np.abs(Dsc * gQ))),
            "preconditioner": {"hessian_diag_range_by_type": [[float(np.min(np.abs(Hd[k][free]))), float(np.max(np.abs(Hd[k][free])))] for k in range(4)] if Hd is not None else None, "negative_diag_entries": int(np.sum(Hd[:, free] < 0)) if Hd is not None else None},
            "cells_at_gap_bound": int(np.sum(at_bound)), "escape_d_bound_active": bool(np.any(at_bound)),
            "r_at_gap_bound_max": float(tube["rg"][at_bound].max()) if np.any(at_bound) else 0.0, "trace": state["trace"][::max(1, len(state["trace"]) // 200)]}
    return P, E, parts, e_line, info


# ================================================================ 3. the pinning penalty
def pin_penalty(M, M_ref, mask, k):
    """E_pin = (k / 2) sum_mask |M - M_ref|^2 and its gradient (zero off the mask)."""
    if k == 0.0:
        return 0.0, np.zeros_like(M)
    D = (M - M_ref) * mask[..., None, None]
    return 0.5 * k * float(np.sum(np.real(D * D))) if not np.iscomplexobj(M) else 0.5 * k * np.sum(D * D), k * D


def energy_object_pinned(M, cfg, K, n_ref=None, need_grad=True, pin=None):
    """m5_32_r17_common.energy_object plus the pinning penalty pin = (M_ref, mask, k)."""
    E, g, pp, dom, fr = R.energy_object(M, cfg, K, n_ref, need_grad)
    if pin is not None:
        Ep, gp = pin_penalty(M, pin[0], pin[1], pin[2])
        pp["E_pin"] = float(np.real(Ep))
        pp["E_stat_free"] = pp["E_stat"]
        pp["E_stat"] = pp["E_stat"] + pp["E_pin"]
        E = E + Ep
        if K is not None:
            pp["E_K"] = E
        if need_grad:
            g = g + gp
    return E, g, pp, dom, fr


def fire_pinned(M0, cfg, free_mask, max_iter, pin, K=None, n_ref=None, **kw):
    eg = lambda M, cfg_, K_, nref, need_grad=True: energy_object_pinned(M, cfg_, K_, nref, need_grad, pin=pin)
    return R._fire_generic(eg, M0, cfg, free_mask, max_iter, K, n_ref, **kw)


# ================================================================ 4. the static twist
def twist_field(M, theta, fr):
    """T_theta M per cell: R(theta / 2) M R(theta / 2)^T through the field's own frame."""
    Rm = C.rot_R(fr["J"], 0.5 * theta[..., None, None] * np.ones(M.shape[:-2] + (1, 1)))
    return Rm @ M @ np.swapaxes(Rm, -1, -2)


def twist_energy(M, cfg, q_vec, n_ref=None, n_samples=8):
    """E(q) of the field twisted by theta = q . x (the coordinates of cfg); returns (E, parts)."""
    n, h = cfg["n"], cfg["h"]
    X, Y, Z = INS4.coords(n, h)
    theta = q_vec[0] * X + q_vec[1] * Y + q_vec[2] * Z
    fr = C.frame(M, n_ref)
    Mq = twist_field(M, theta, fr)
    cf = dict(cfg); cf["n_samples"] = n_samples
    E, _, pp, dom, _ = R.energy_object(Mq, cf, None, fr["n"], need_grad=False)
    return E, pp, dom


def twist_stiffness(M, cfg, n_ref=None, qs=(0.05, 0.1, 0.2), axis=0, n_samples=8):
    """S from E(q) - E(0) = S q^2 / 2 at the qs (each with its sign partner), the evenness and quadraticity checks."""
    E0, p0, _ = twist_energy(M, cfg, np.zeros(3), n_ref, n_samples)
    rows = []
    for q in qs:
        v = np.zeros(3); v[axis] = q
        Ep, pp, dp = twist_energy(M, cfg, v, n_ref, n_samples)
        Em, pm, dm = twist_energy(M, cfg, -v, n_ref, n_samples)
        rows.append({"q": q, "dE_plus": Ep - E0, "dE_minus": Em - E0, "S_plus": 2.0 * (Ep - E0) / q ** 2, "S_minus": 2.0 * (Em - E0) / q ** 2,
                     "odd_part_rel": abs(Ep - Em) / max(abs(Ep - E0) + abs(Em - E0), 1e-300), "parts_plus": {k: pp[k] - p0[k] for k in ("E_h", "KP", "reg", "V4", "U") if k in pp and k in p0},
                     "escape_d": bool(dp["escape_d"] or dm["escape_d"])})
    for r_ in rows:
        r_["S_even"] = (r_["dE_plus"] + r_["dE_minus"]) / r_["q"] ** 2          # 2 x the even part / q^2 = S
        r_["torque_odd"] = (r_["dE_plus"] - r_["dE_minus"]) / (2.0 * r_["q"])   # the first-order term: zero on a stationary field
    S = [r_["S_even"] for r_ in rows]
    out = {"E0": E0, "rows": rows, "S_at_smallest_q": S[0], "quadratic_rel_dev_q1_q2": abs(S[1] - S[0]) / max(abs(S[0]), 1e-300) if len(S) > 1 else None,
           "torque_odd_at_smallest_q": rows[0]["torque_odd"]}
    return out


# ================================================================ 5. the spin-2 zero counter
def _local_frame(rhat, axis_vec):
    """the polar frame about axis_vec (per point, (..., 3)) at the unit vectors rhat: e_theta (away from the pole), e_phi."""
    c = np.clip(np.sum(rhat * axis_vec, -1), -1.0, 1.0)
    sn = np.maximum(np.sqrt(1.0 - c * c), 1e-300)
    e_th = (c[..., None] * rhat - axis_vec) / sn[..., None]
    e_ph = np.cross(axis_vec, rhat) / sn[..., None]
    return e_th, e_ph


def zeta_at(S_pts, rhat, axis_vec):
    """zeta = S_ee - S_ff + 2 i S_ef: e the frame's e_theta projected transverse to the outward-oriented director n, f = n x e."""
    w, V = np.linalg.eigh(S_pts)
    nvec = V[..., :, -1]
    gap = w[..., -1] - w[..., -2]
    sg = np.sum(nvec * rhat, -1)
    sg = np.where(np.abs(sg) > 1e-9, np.sign(sg), 1.0)
    nvec = nvec * sg[..., None]
    e_th, _ = _local_frame(rhat, axis_vec)
    e = e_th - np.sum(e_th * nvec, -1, keepdims=True) * nvec
    e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-300)
    f = np.cross(nvec, e)
    See = np.einsum("...a,...ab,...b->...", e, S_pts, e)
    Sff = np.einsum("...a,...ab,...b->...", f, S_pts, f)
    Sef = np.einsum("...a,...ab,...b->...", e, S_pts, f)
    return See - Sff + 2j * Sef, gap, np.abs(np.sum(nvec * rhat, -1))


def _perp_axis(c):
    """a unit vector perpendicular to each c (..., 3): the poles of the per-triangle frame sit 90 degrees from the triangle."""
    u = np.zeros_like(c); u[..., 2] = 1.0
    swap = np.abs(c[..., 2]) > 0.9
    u[swap] = np.array([1.0, 0.0, 0.0])
    a = np.cross(c, u)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def _wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def spin2_zeros_on_cells(S_cells, rhat):
    """the zeros of the spin-2 section sampled on shell CELLS (S_cells (n, 3, 3), rhat (n, 3)): the convex-hull triangulation of
    the directions, each triangle oriented outward, its winding computed in the polar frame about an axis perpendicular to its
    own centroid (smooth on the triangle by construction, no chart partition).  Returns the record."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(rhat)
    tri = hull.simplices.copy()
    a, b, c = rhat[tri[:, 0]], rhat[tri[:, 1]], rhat[tri[:, 2]]
    cen = (a + b + c) / 3.0
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), cen) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    a, b, c = rhat[tri[:, 0]], rhat[tri[:, 1]], rhat[tri[:, 2]]
    cen = (a + b + c) / 3.0
    cen = cen / np.linalg.norm(cen, axis=-1, keepdims=True)
    ax = _perp_axis(cen)
    zs, gaps = [], []
    for corner in (tri[:, 0], tri[:, 1], tri[:, 2]):
        z, gap, _ = zeta_at(S_cells[corner], rhat[corner], ax)
        zs.append(z); gaps.append(gap)
    p0, p1, p2 = (np.angle(z) for z in zs)
    d01, d12, d20 = _wrap(p1 - p0), _wrap(p2 - p1), _wrap(p0 - p2)
    wnd = np.rint((d01 + d12 + d20) / (2 * np.pi)).astype(int)
    amp = np.minimum.reduce([np.abs(z) for z in zs])
    zmax = float(np.max(np.abs(zs[0])))
    # the two ways a sample can defeat the count: a zero AT a sample point (the phase undefined) and an edge whose phase
    # difference is +-pi to roundoff (the wrap ambiguous); both are reported, neither is silently rounded
    at_sample = amp < 1e-9 * max(zmax, 1e-300)
    amb = (np.abs(np.abs(d01) - np.pi) < 1e-3) | (np.abs(np.abs(d12) - np.pi) < 1e-3) | (np.abs(np.abs(d20) - np.pi) < 1e-3)   # a zero on an edge (a symmetric sample pair): the wrap ambiguous
    out = {"n_cells": int(len(rhat)), "n_triangles": int(len(tri)), "zeros": [], "total_index": int(np.sum(wnd)), "n_zeros": int(np.sum(wnd != 0)),
           "index_histogram": {str(k): int(np.sum(wnd == k)) for k in sorted(set(wnd.tolist())) if k != 0},
           "zeta_rms_cells": float(np.sqrt(np.mean(np.abs(zs[0]) ** 2))), "zeta_max_cells": zmax,
           "min_director_gap": float(min(np.min(g) for g in gaps)),
           "n_triangles_zero_at_sample": int(np.sum(at_sample)), "n_triangles_ambiguous_pi_edge": int(np.sum(amb)),
           "count_reliable": bool(not np.any(at_sample) and not np.any(amb))}
    for i in np.where(wnd != 0)[0]:
        out["zeros"].append({"index": int(wnd[i]), "xyz": [float(v) for v in cen[i]], "abs_zeta_min_corner": float(amp[i])})
    return out


def _F0():
    import importlib.util
    spec = importlib.util.spec_from_file_location("m5_32_r16_0_fields_r18", os.path.join(C.HERE, "m5_32_r16_0_fields.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


F0 = _F0()


def spin2_fit_zeros(S_cells, rhat, lmax=6, nth=90, nph=180):
    """the PRIMARY counter: zeta on the shell cells in the polar frame about z is a spin-weight-2 function on the sphere,
    fitted by least squares with the spin-weighted harmonics 2Y_lm, l <= lmax (the residual reported); the fit is evaluated
    on a (theta, phi) grid, the zeros are the faces where arg zeta winds (loop oriented outward, calibrated on Q = xx - yy),
    and the two pole caps carry the index ring_N + 2 and 2 - ring_S (ring = the phase winding of the boundary ring with
    increasing phi; the polar frame's own spin-2 defect is -2 / +2 in this orientation, calibrated on Q = xx - yy (no pole
    zeros) and on Q = zz - (xx + yy) / 2 (index-2 zeros at both poles), checked on a rotated Q)."""
    ez = np.zeros_like(rhat); ez[:, 2] = 1.0
    zc, gap, ndr = zeta_at(S_cells, rhat, ez)
    th = np.arccos(np.clip(rhat[:, 2], -1.0, 1.0)); ph = np.arctan2(rhat[:, 1], rhat[:, 0])
    lm = [(l, mm) for l in range(2, lmax + 1) for mm in range(-l, l + 1)]
    A = np.stack([F0.sY2(2, l, mm, th, ph) for l, mm in lm], -1)
    coef = np.linalg.lstsq(A, zc, rcond=None)[0]
    fit = A @ coef
    zrms = float(np.sqrt(np.mean(np.abs(zc) ** 2)))
    resid = float(np.sqrt(np.mean(np.abs(fit - zc) ** 2)) / max(zrms, 1e-300))
    TH = (np.arange(nth) + 0.5) * np.pi / nth; PH = np.arange(nph) * 2 * np.pi / nph
    T, P = np.meshgrid(TH, PH, indexing="ij")
    G = np.zeros(T.shape, dtype=complex)
    for c, (l, mm) in zip(coef, lm):
        G += c * F0.sY2(2, l, mm, T, P)
    psi = np.angle(G)
    a = psi[:-1, :]; b = psi[:-1, np.roll(np.arange(nph), -1)]; c_ = psi[1:, np.roll(np.arange(nph), -1)]; d = psi[1:, :]
    wn = np.rint(-(_wrap(b - a) + _wrap(c_ - b) + _wrap(d - c_) + _wrap(a - d)) / (2 * np.pi)).astype(int)
    north = float(np.sum(_wrap(np.roll(psi[0], -1) - psi[0])) / (2 * np.pi))
    south = float(np.sum(_wrap(np.roll(psi[-1], -1) - psi[-1])) / (2 * np.pi))
    iN, iS = int(np.rint(north + 2.0)), int(np.rint(2.0 - south))
    zeros = []
    for i, k in zip(*np.where(wn != 0)):
        t_, p_ = 0.5 * (TH[i] + TH[i + 1]), PH[k] + np.pi / nph
        zeros.append({"index": int(wn[i, k]), "xyz": [float(np.sin(t_) * np.cos(p_)), float(np.sin(t_) * np.sin(p_)), float(np.cos(t_))], "abs_fit_at_face": float(min(abs(G[i, k]), abs(G[i + 1, k])))})
    if iN != 0:
        zeros.append({"index": iN, "xyz": [0.0, 0.0, 1.0], "pole": "north"})
    if iS != 0:
        zeros.append({"index": iS, "xyz": [0.0, 0.0, -1.0], "pole": "south"})
    power = {str(l): float(sum(abs(c) ** 2 for c, (l_, _) in zip(coef, lm) if l_ == l)) for l in range(2, lmax + 1)}
    ptot = max(sum(power.values()), 1e-300)
    P2 = {str(mm): float(abs(c) ** 2) for c, (l_, mm) in zip(coef, lm) if l_ == 2}
    return {"n_cells": int(len(rhat)), "lmax": lmax, "fit_residual_rel": resid, "zeta_rms_cells": zrms, "zeta_max_cells": float(np.max(np.abs(zc))),
            "min_director_gap": float(np.min(gap)), "min_director_dot_rhat": float(np.min(ndr)),
            "zeros": zeros, "n_zeros": len(zeros), "total_index": int(sum(z["index"] for z in zeros)), "faces_total": int(np.sum(wn)), "ring_N": north, "ring_S": south,
            "index_histogram": {str(k): int(sum(1 for z in zeros if z["index"] == k)) for k in sorted(set(z["index"] for z in zeros))},
            "l_power_fraction": {k: v / ptot for k, v in power.items()}, "l2_m_power": P2}


def spin2_zero_count(M, cfg, radii=(1.5, 2.25, 3.0, 4.5, 6.0), width=None, lmax=None, hull_too=True):
    """per shell |r - r_c| < width / 2 (default 1.5 h): the harmonic-fit count (primary) and the hull-triangle count (resolution-limited,
    reported) on the lattice cells of the shell (no interpolation)."""
    n, h = cfg["n"], cfg["h"]
    X, Y, Z = INS4.coords(n, h)
    pos = np.stack([X, Y, Z], -1)
    r = np.sqrt(np.sum(pos * pos, -1))
    S = M[..., 1:, 1:]
    w = 1.5 * h if width is None else width
    out = {}
    for r_c in radii:
        m = np.abs(r - r_c) < w / 2.0
        nc = int(np.sum(m))
        if nc < 30:
            out[f"{r_c:g}"] = {"n_cells": nc, "note": "too few cells"}
            continue
        rh = pos[m] / r[m][:, None]
        lm = lmax if lmax is not None else (4 if nc < 150 else 6)
        rec = spin2_fit_zeros(S[m], rh, lmax=lm)
        rec["r"] = float(r_c); rec["r_range"] = [float(r[m].min()), float(r[m].max())]
        if hull_too:
            hh = spin2_zeros_on_cells(S[m], rh)
            rec["hull_count"] = {k: hh[k] for k in ("total_index", "n_zeros", "index_histogram", "count_reliable", "n_triangles")}
        out[f"{r_c:g}"] = rec
    return out


# ================================================================ selftest
def _richardson(fn, P, D, eps=1e-5):
    f = lambda e: (float(fn(P + e * D)) - float(fn(P - e * D))) / (2 * e)
    d1, d2 = f(eps), f(eps / 2)
    return (4 * d2 - d1) / 3.0, d1, d2


def selftest(write=True):
    res, lines = {}, []
    rng = np.random.default_rng(1809)

    def check(name, ok, val):
        res[name] = {"ok": bool(ok), "value": val}
        lines.append(f"{'PASS' if ok else 'FAIL'} {name}: {val}")
        log(lines[-1])

    # ---------------- 1. the weighted action == action at wc = h^3 (v4 absolute, v4 relative, v6)
    n, L = 6, 9.0
    for obj in ("v4abs", "v4rel", "v6"):
        if obj == "v6":
            cfg = R.cfg_v6(n, L, gW=1.1, n_samples=1)
        else:
            cfg = C.cfg_v4(n, L, completion="rebuild", n_samples=1); cfg["object"] = "v4"; cfg["weight"] = "absolute" if obj == "v4abs" else "relative"
        M = C.random_spectral_field(rng, n, cfg, dir_noise=(-0.15, 0.05), pair_noise=0.12)
        nref = C.frame(M)["n"]
        wc = np.full((n, n, n), cfg["h"] ** 3)
        E, g, parts, e_cells, fr = action_weighted(M, cfg, wc, n_ref=nref)
        E2, g2, pp2, dom2, fr2 = R.energy_object(M, cfg, None, nref)
        check(f"weighted action at wc = h^3 == energy_object ({obj}): energy 1e-13 and gradient 1e-12 relative", abs(E - E2) < 1e-13 * abs(E2) and np.max(np.abs(g - g2)) < 1e-12 * np.max(np.abs(g2)),
              {"E": E, "E_ref": E2, "grad_max_dev": float(np.max(np.abs(g - g2))), "parts": {k: [parts[k], pp2.get(k)] for k in ("E_h", "V4", "KP", "reg") if k in pp2}})
        wr = rng.uniform(0.5, 2.0, size=(n, n, n))
        E, g, parts, e_cells, fr = action_weighted(M, cfg, wr, n_ref=nref)
        D = sym(rng.normal(size=M.shape)); D /= np.sqrt(np.sum(D * D))
        an = float(np.sum(g * D))
        cs = float(np.imag(action_weighted(M + 1e-20j * D, cfg, wr, n_ref=nref, need_grad=False)[0]) / 1e-20)
        check(f"weighted action gradient with random weights vs complex step (1e-10, {obj})", abs(an - cs) < 1e-10 * abs(cs), {"analytic": an, "cs": cs, "E_sum_wc_e": E, "check_sum": float(np.sum(wr * np.real(e_cells)))})
    # ---------------- 2. the tube
    cfg = C.cfg_v4(32, 48.0, completion="rebuild", n_samples=1); cfg["object"] = "v4"; cfg["weight"] = "absolute"
    h = cfg["h"]
    tube = tube_setup(h, 48.0)
    tube_g = tube_setup(h, 48.0, y0=h / 2, z0=h / 2)
    P = melted_seed(cfg, tube)
    P = P + 0.02 * rng.normal(size=P.shape) * (tube["rg"] < 12.0)[None, :]      # a rough profile: the gates must hold on it
    P[3] = 0.1 * np.exp(-(tube["rg"] / 4.0) ** 2)                                 # a boost bump
    P[1] = np.maximum(P[1], P[2] + 0.05)
    M3 = field_3d_from_profiles(P, tube_g, cfg)
    frN = C.frame(M3, C.radial_ref(cfg))
    spec = np.sort(np.linalg.eigvals(M3 @ ETA).real, axis=-1)
    j = 16
    check("the ansatz on the 3D lattice: the N spectrum is (-m_g, lambda_23, lambda_23, lambda_1) at the profile values (1e-12), the director radial (|n . rhat| = 1 at 1e-10)",
          np.max(np.abs(spec[..., 1] - spec[..., 2])) < 1e-12 and float(np.min(np.abs(np.einsum("...a,...a->...", np.real(frN["n"])[..., 1:], np.stack(INS4.coords(32, h), -1) / np.sqrt(np.sum(np.stack(INS4.coords(32, h), -1) ** 2, -1))[..., None])))) > 1.0 - 1e-10,
          {"pair_split_max": float(np.max(np.abs(spec[..., 1] - spec[..., 2]))), "spectrum_at_center_cell": spec[j, j, j].tolist()})
    E3, g3, p3, e3, _ = action_weighted(M3, cfg, np.ones((32, 32, 32)), n_ref=C.radial_ref(cfg))
    Et, gt, pt, et, frt = tube_energy(P, tube_g, cfg)
    line3 = np.real(e3[:, j, j])
    check("GATE 1: the tube center line at (h/2, h/2) reproduces the 3D lattice's own ray densities (1e-10 relative, every cell)", np.max(np.abs(np.real(et) - line3)) < 1e-10 * np.max(np.abs(line3)), {"max_rel_dev": float(np.max(np.abs(np.real(et) - line3)) / np.max(np.abs(line3))), "n_cells": int(len(line3))})
    Eh3 = float(np.sum(np.real(e3)) * h ** 3)
    Et0, _, pt0, et0, _ = tube_energy(P, tube, cfg)
    check("GATE 2 (reported): the tube quadrature 2 pi sum x^2 h e(x) on the axis vs the 3D lattice sum h^3 sum e (the angular-quadrature defect at h 1.5)", True, {"E_tube_axis": float(Et0), "E_3D": Eh3, "rel": float(abs(Et0 - Eh3) / abs(Eh3))})
    # the circle acts trivially
    Mt = tube_field(P, tube)
    frT = C.frame(Mt, None)
    a_plain = C.action(Mt, cfg, need_grad=False, fr=frT)["parts"]
    worst = 0.0
    for beta in (0.4, 1.1, np.pi / 2):
        Rb = C.rot_R(frT["J"], beta)
        Mb = Rb @ Mt @ np.swapaxes(Rb, -1, -2)
        worst = max(worst, float(np.max(np.abs(Mb - Mt))))
    check("the circle acts trivially on the ansatz (T_beta M == M to 1e-12: the average equals the plain action)", worst < 1e-12, {"max_|T_beta M - M|": worst, "E_plain": a_plain["E_stat"]})
    # the profile gradient vs complex step and Richardson
    D = rng.normal(size=P.shape) * (tube["rg"] < 20.0)[None, :]; D /= np.sqrt(np.sum(D * D))
    an = float(np.sum(gt * D)) if False else None
    Et, gt, pt, et, _ = tube_energy(P, tube, cfg)
    an = float(np.sum(gt * D))
    cs = float(np.imag(tube_energy(P + 1e-20j * D, tube, cfg, need_grad=False)[0]) / 1e-20)
    rich, d1, d2 = _richardson(lambda Q: tube_energy(Q, tube, cfg, need_grad=False)[0], P, D, eps=1e-4)
    check("the profile gradient (weighted adjoint + the ansatz chain) vs complex step (1e-10) and Richardson central differences (1e-6)", abs(an - cs) < 1e-10 * abs(cs) and abs(an - rich) < 1e-6 * abs(rich), {"analytic": an, "cs": cs, "richardson": rich, "rel_cs": abs(an - cs) / abs(cs), "rel_fd": abs(an - rich) / abs(rich)})
    # evenness in beta at beta = 0
    P0 = P.copy(); P0[3] = 0.0
    E0, g0, _, _, _ = tube_energy(P0, tube, cfg)
    check("beta = 0 is stationary (the static energy is even in the radial boost: dE / d beta = 0 at 1e-10 of |grad|)", np.max(np.abs(g0[3])) < 1e-10 * np.max(np.abs(g0)), {"grad_beta_max": float(np.max(np.abs(g0[3]))), "grad_max": float(np.max(np.abs(g0)))})
    Pb = P0.copy(); Pb[3] = 0.05 * np.exp(-((tube["rg"] - 3.0) / 2.0) ** 2)
    Eb = tube_energy(Pb, tube, cfg, need_grad=False)[0]
    res["beta_stability_probe"] = {"E(beta bump 0.05 at r 3) - E(0)": float(Eb - E0), "stable_if_positive": bool(Eb > E0)}
    log(f"  beta probe: {res['beta_stability_probe']}")
    # a short solve on the vacuum-plus-seed converges to a lower energy and keeps the bound
    t = time.time()
    Ps, Es, ps, es, info = solve_tube(melted_seed(cfg, tube), tube, cfg, maxiter=60, log_every=1000, tag="selftest")
    check("solve_tube: 60 L-BFGS-B iterations lower the energy of the melted seed, the gap bound respected", Es < tube_energy(melted_seed(cfg, tube), tube, cfg, need_grad=False)[0] and np.min(Ps[1] - Ps[2]) >= GAP_MIN - 1e-12,
          {"E_seed": float(tube_energy(melted_seed(cfg, tube), tube, cfg, need_grad=False)[0]), "E_60": float(Es), "nit": info["nit"], "wall_s": info["wall_s"], "Delta_min": float(np.min(Ps[1] - Ps[2]))})
    # ---------------- 3. the pinning penalty
    cfg6 = C.cfg_v4(6, 9.0, completion="rebuild", n_samples=1); cfg6["object"] = "v4"; cfg6["weight"] = "absolute"
    M = C.random_spectral_field(rng, 6, cfg6); nref = C.frame(M)["n"]
    Mref = C.random_spectral_field(rng, 6, cfg6)
    mask = np.zeros((6, 6, 6), dtype=bool); mask[2:4, 2:4, 2:4] = True
    E0, g0, _, _, _ = energy_object_pinned(M, cfg6, None, nref, pin=(Mref, mask, 0.0))
    E1, g1, _, _, _ = R.energy_object(M, cfg6, None, nref)
    Ek, gk, ppk, _, _ = energy_object_pinned(M, cfg6, None, nref, pin=(Mref, mask, 3.0))
    D = sym(rng.normal(size=M.shape)); D /= np.sqrt(np.sum(D * D))
    an = float(np.sum(gk * D)); cs = float(np.imag(energy_object_pinned(M + 1e-20j * D, cfg6, None, nref, need_grad=False, pin=(Mref, mask, 3.0))[0]) / 1e-20)
    check("pinning: k = 0 reproduces the free energy and gradient exactly; k = 3 gradient vs complex step (1e-10); E_pin = (k/2) sum |dM|^2 on the mask", E0 == E1 and np.array_equal(g0, g1) and abs(an - cs) < 1e-10 * abs(cs) and abs(ppk["E_pin"] - 1.5 * np.sum(((M - Mref)[mask]) ** 2)) < 1e-12,
          {"E_pin": ppk["E_pin"], "analytic": an, "cs": cs})
    # ---------------- 4. the static twist
    cfgt = C.cfg_v4(8, 12.0, completion="rebuild", n_samples=4); cfgt["object"] = "v4"; cfgt["weight"] = "relative"
    Mt = C.random_spectral_field(rng, 8, cfgt, dir_noise=(0.0, 0.05), pair_noise=0.05, tilt=0.1, boost=0.0, smooth=3)
    nreft = C.frame(Mt)["n"]
    tw = twist_stiffness(Mt, cfgt, nreft, qs=(0.02, 0.04, 0.08), n_samples=4)
    check("static twist: S from the even part of E(q) quadratic (q 0.02 vs 0.04 within 2e-2) on a random split-carrying field; the odd part is the first-order torque (nonzero on a random field, reported; zero on a stationary one)", tw["quadratic_rel_dev_q1_q2"] < 2e-2,
          {"S_even": [r_["S_even"] for r_ in tw["rows"]], "torque_odd": [r_["torque_odd"] for r_ in tw["rows"]], "quad_dev": tw["quadratic_rel_dev_q1_q2"]})
    # the uniform twist is a symmetry (theta constant): E unchanged to 1e-12
    frq = C.frame(Mt, nreft)
    Mc = twist_field(Mt, 0.7 * np.ones((8, 8, 8)), frq)
    cf8 = dict(cfgt); cf8["n_samples"] = 8
    Ea = R.energy_object(Mt, cf8, None, nreft, need_grad=False)[0]; Eb = R.energy_object(Mc, cf8, None, frq["n"], need_grad=False)[0]
    check("a uniform twist (theta = 0.7 everywhere) leaves the 8-sample averaged energy unchanged (1e-10)", abs(Ea - Eb) < 1e-10 * abs(Ea), [Ea, Eb])
    # ---------------- 5. the spin-2 zero counter on synthetic sections
    cfgz = C.cfg_v4(32, 48.0, completion="rebuild")
    Mh = C15.seed_uniaxial(cfgz)

    def with_Q(Q, amp=0.02):
        Mq = Mh.copy()
        Mq[..., 1:, 1:] = Mq[..., 1:, 1:] + amp * Q
        return Mq
    def rot(th_, ph_):
        return np.array([[np.cos(th_), 0, np.sin(th_)], [0, 1, 0], [-np.sin(th_), 0, np.cos(th_)]]) @ np.array([[np.cos(ph_), -np.sin(ph_), 0], [np.sin(ph_), np.cos(ph_), 0], [0, 0, 1]])
    Qx = np.diag([1.0, -1.0, 0.0])
    Qz = np.diag([-0.5, -0.5, 1.0])
    R1 = rot(0.0, 0.3)                              # xx - yy turned by 0.3 rad about z: the four zeros off the lattice axes
    R2 = rot(0.3, 0.2)                              # the m = 0 axis tilted by 0.3 rad: the double zeros off the lattice axes
    Rr = rot(0.7, 1.9)
    Qx1, Qz2, Qr = R1 @ Qx @ R1.T, R2 @ Qz @ R2.T, Rr @ Qx @ Rr.T
    ax2 = R2 @ np.array([0.0, 0.0, 1.0])
    radii = (3.0, 4.5, 6.0, 9.0)
    zx = spin2_zero_count(with_Q(Qx), cfgz, radii=radii)
    zz = spin2_zero_count(with_Q(Qz), cfgz, radii=radii)
    zr = spin2_zero_count(with_Q(Qr), cfgz, radii=radii)
    zt = spin2_zero_count(with_Q(Qz2), cfgz, radii=radii)
    check("spin-2 zeros (harmonic fit), Q = xx - yy on the hedgehog: four simple zeros of total index 4 on every shell (Poincare-Hopf, Euler number 4), the fit residual under 1e-3", all(v["total_index"] == 4 and v["n_zeros"] == 4 and v["index_histogram"] == {"1": 4} and v["fit_residual_rel"] < 1e-3 for v in zx.values()),
          {k: {"total": v["total_index"], "n": v["n_zeros"], "hist": v["index_histogram"], "resid": v["fit_residual_rel"], "cells": v["n_cells"]} for k, v in zx.items()})
    check("spin-2 zeros (harmonic fit), Q = zz - (xx + yy)/2 (the m = 0 pattern): two index-2 zeros at the poles, total 4, on every shell", all(v["total_index"] == 4 and v["n_zeros"] == 2 and v["index_histogram"] == {"2": 2} for v in zz.values()),
          {k: {"total": v["total_index"], "n": v["n_zeros"], "hist": v["index_histogram"], "rings": [v["ring_N"], v["ring_S"]]} for k, v in zz.items()})
    check("spin-2 zeros (harmonic fit), the m = 0 pattern with its axis tilted: total 4, every zero within 20 degrees of the axis", all(v["total_index"] == 4 and all(abs(np.dot(z_["xyz"], ax2)) > np.cos(np.radians(20)) for z_ in v["zeros"]) for v in zt.values()),
          {k: {"total": v["total_index"], "n": v["n_zeros"], "hist": v["index_histogram"]} for k, v in zt.items()})
    check("spin-2 zeros (harmonic fit), a rotated Q (generic orientation): total 4, four simple zeros on every shell; the hull count agrees where it is reliable", all(v["total_index"] == 4 and v["n_zeros"] == 4 for v in zr.values()) and all(v["hull_count"]["total_index"] == 4 for v in zr.values() if v["hull_count"]["count_reliable"]),
          {k: {"total": v["total_index"], "n": v["n_zeros"], "hull": v["hull_count"]} for k, v in zr.items()})
    z0 = spin2_zero_count(Mh, cfgz, radii=(3.0, 6.0))
    res["spin2_on_the_exact_hedgehog"] = {k: {"zeta_max_cells": v["zeta_max_cells"], "fit_residual_rel": v["fit_residual_rel"], "note": "zeta == 0 to roundoff: nothing to count; the amplitude floor is the number to quote"} for k, v in z0.items()}
    log(f"  exact hedgehog: {res['spin2_on_the_exact_hedgehog']}")
    res["n_pass"] = sum(1 for v in res.values() if isinstance(v, dict) and v.get("ok"))
    res["n_total"] = sum(1 for v in res.values() if isinstance(v, dict) and "ok" in v)
    log(f"selftest {res['n_pass']}/{res['n_total']}")
    if write:
        json.dump(res, open(os.path.join(DATA, "m5_32_r18_common_selftest.json"), "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    selftest()
