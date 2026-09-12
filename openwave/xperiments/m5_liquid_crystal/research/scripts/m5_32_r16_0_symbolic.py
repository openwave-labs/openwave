"""M5.32 R16-0 (symbolic and point-jet): claims C1, C4, C5, C6 of the author's 2026-09-06 comments
(ledger 6.5, the R16-0 row).  Every claim is the author's text; every number here is ours.

EQUATIONS FIRST
---------------
eta = diag(-1, 1, 1, 1); M 4x4 symmetric contravariant; A_mu = d_mu M.  The two H-adjoint completions
(the author's names, 09-06 appendix):
    F^eta_mn = A_m eta A_n - A_n eta A_m,       I_norm    = sum_{m<n} eta^m eta^n tr(G F^eta G F^eta^T)
    F^G_mn   = A_m G A_n - A_n G A_m,           I_rebuild = sum_{m<n} eta^m eta^n tr(G F^G G F^G^T)
    G = eta + 2 (eta u)(eta u)^T,  u the timelike unit eigenvector of N = M eta (u^T eta u = -1).
    Our registry's I1_h (m5_32_terms_ext.py) is I_norm.  Static E-orientation: E = +4 x the read at omega = 0.
C1  the counterexample at G = I: A1 = I, A2 = E01 + E10 gives |F^G|^2 = 0 and |F^eta|^2_G = 8 (and the reverse
    at A1 = eta); the two forms coincide when u is fixed AND the jets are block-diagonal with respect to u.
C4  the local circle T_alpha M = R_n(alpha/2) M R_n(alpha/2)^T, R_n the rotation about the local director n
    (the eigenvector of the isolated eigenvalue 1 of N).  On the sheet M(z) = R12(psi) R23(phi) D_s R23^T R12^T,
    D_s = diag(g, 1, delta + s, delta - s): T_alpha shifts phi by alpha/2 (weight two on B), fixes s = 0,
    T_{2 pi} = id.  Point jets: the field M(x) = D_s + x1 A1 + x2 A2 (spatial-block jets, u = e0 fixed),
    n(x) the leading eigenvector of the spatial block, M'(x) = R_{n(x)}(alpha/2) M(x) R^T, A_i' = d_i M'(0)
    (central differences, Richardson).  Densities as functions of alpha: V4^dd, the split, K_P^23 (expected
    invariant), I1, I1_h = I_norm, I_rebuild, E2 = sum_i tr(A_i' G A_i' G) (expected NOT invariant).
    The circle average by equispaced samples: exact iff the density is a trigonometric polynomial of degree
    below the sample count (the doubling test).
C5  sheet inertias under the LOCAL generator G_loc = R12(psi) G23 R12(psi)^T on the twist sheet
    M(z) = R12(psi(z)) D_s R12^T:  A_0 = omega [G_loc, M], A_z = psi' d_psi M,  <F_0z, F_0z>_eta = tr(eta F eta F^T)
    (the author: 8 omega^2 psi'^2 s^2 (delta + s - 1)^2, zero at s = 0; the rigid G23 nonzero at s = 0);
    K_P^23 static = 0 on the sheet, its inertia (1/2) tr(Om_0^T eta Om_0 eta);  E2 static = tr(A_z G A_z G);
    the tilt condition from R15-H (data/m5_32_r15_h_tilt.json) with w -> c_s s^2: hyperbolic iff c_s > 16 omega^2;
    boost sheets A_z = b (K M + M K), K = E_0i + E_i0: <F_0z, F_0z>_eta (the author: -8 b^2 omega^2 s^2 (g - delta -+ s)^2
    for i in the pair planes, zero along the director) against the G-norm.
C6  the eigenvalue metric of B = [[a, b], [b, -a]]: (d lambda_+)^2 + (d lambda_-)^2 = 2 (a da + b db)^2 / (a^2 + b^2)
    (direction-dependent at a = b = 0) against rho^2 = (1/2) tr B^2 = a^2 + b^2 (smooth).

usage: python3 m5_32_r16_0_symbolic.py
out:   data/m5_32_r16_0_symbolic.json (relative paths only), checkpoints/m5_32_r16/symbolic.log
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np
import sympy as sp

sys.argv = [sys.argv[0]]                                  # the R15 common module reads argv
import m5_32_r15_common as C15                            # noqa: E402

INS4, C13, B8, EXT = C15.INS4, C15.C13, C15.B8, C15.EXT
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
DATA = os.path.join(RES, "data")
CK = os.path.join(RES, "checkpoints", "m5_32_r16")
os.makedirs(CK, exist_ok=True)
T0 = time.time()
LOG = open(os.path.join(CK, "symbolic.log"), "a")


def log(m):
    line = f"[{time.time() - T0:8.1f}s] {m}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


G, DELTA = 8.0, 0.3
ETA = sp.diag(-1, 1, 1, 1)
ETAn = np.diag([-1.0, 1.0, 1.0, 1.0])
E0 = sp.Matrix([1, 0, 0, 0])
OUT = {"rung": "R16-0 symbolic", "claims": {}}


def rot(p, q, a):
    R = sp.eye(4)
    c, s = sp.cos(a), sp.sin(a)
    R[p, p] = c; R[q, q] = c; R[p, q] = -s; R[q, p] = s
    return R


def comm_eta(A, B):
    return A * ETA * B - B * ETA * A


def comm_G(A, B, Gm):
    return A * Gm * B - B * Gm * A


def inner(F, Gm):
    return (Gm * F * Gm * F.T).trace()


def hcov(u):
    v = ETA * u
    return ETA + 2 * v * v.T


# ============================================================== C1
def c1():
    log("C1: the two H-adjoint completions")
    I4 = sp.eye(4)
    E01 = sp.zeros(4); E01[0, 1] = 1; E01[1, 0] = 1
    rec = {}
    for lab, A1 in (("A1 = I", I4), ("A1 = eta", ETA)):
        A2 = E01
        fG = comm_G(A1, A2, I4)
        fE = comm_eta(A1, A2)
        rec[lab] = {"|F^G|^2_G (G = I)": int(inner(fG, I4)), "|F^eta|^2_G (G = I)": int(inner(fE, I4))}
        log(f"  {lab}, A2 = E01 + E10, G = I: |F^G|^2 = {rec[lab]['|F^G|^2_G (G = I)']}, |F^eta|^2_G = {rec[lab]['|F^eta|^2_G (G = I)']}")
    # block-diagonal jets with u = e0 fixed: the two completions coincide (G = I in that frame)
    a, b = sp.symbols("a b", real=True)
    S1 = sp.Matrix(3, 3, sp.symbols("s1_0:9", real=True)); S1 = (S1 + S1.T) / 2
    S2 = sp.Matrix(3, 3, sp.symbols("s2_0:9", real=True)); S2 = (S2 + S2.T) / 2
    B1 = sp.diag(a, S1); B2 = sp.diag(b, S2)
    diff = sp.simplify(comm_G(B1, B2, I4) - comm_eta(B1, B2))
    rec["block_diagonal_jets_u_fixed"] = {"F^I - F^eta": str(diff), "coincide": diff == sp.zeros(4)}
    log(f"  block-diagonal jets (u = e0 fixed): F^I - F^eta = {diff.tolist()} -> coincide {diff == sp.zeros(4)}")
    # a general spatial-off-block jet: they differ (the boost sector)
    K2 = sp.zeros(4); K2[0, 2] = 1; K2[2, 0] = 1
    diff2 = sp.simplify(comm_G(K2, B2, I4) - comm_eta(K2, B2))
    rec["boost_jet_vs_block_jet"] = {"F^I - F^eta nonzero": diff2 != sp.zeros(4)}
    # the R15-V-a jet: twist inside a boost dressing, both completions with G = h(u)
    g, d, d4, bb, k = sp.symbols("g delta delta4 b k", positive=True)
    chi, psi = sp.symbols("chi psi", real=True)
    D = sp.diag(g, 1, d, d4)
    La = sp.eye(4); c_, s_ = sp.cosh(chi), sp.sinh(chi)
    La[0, 0] = c_; La[1, 1] = c_; La[0, 1] = s_; La[1, 0] = s_          # boost along axis 1, gradient along axis 1... the R15-V-a pair (a=1, s=2 twist along 3)
    Q = La * rot(1, 2, psi)
    M = Q * D * Q.T
    u = Q * E0
    H = hcov(u)
    As = bb * sp.diff(M, chi)
    At = k * sp.diff(M, psi)
    Fn = comm_eta(As, At)
    Fr = comm_G(As, At, H)
    Un = sp.expand(4 * inner(Fn, H))
    Ur = sp.expand(4 * inner(Fr, H))
    jets = {}
    for chiv in (0, sp.Rational(1, 2)):
        for psiv in (0, sp.pi / 4):
            sub = {g: 8, d: sp.Rational(3, 10), d4: sp.Rational(3, 10), chi: chiv, psi: psiv, bb: 1, k: 1}
            un, ur = float(Un.subs(sub)), float(Ur.subs(sub))
            jets[f"chi{float(chiv):g}_psi{float(psiv):.3f}"] = {"U_norm": un, "U_rebuild": ur, "rel_diff": (ur - un) / max(abs(un), 1e-300)}
    rec["twist_inside_boost_jet_g8_d0.3_degenerate"] = jets
    rec["twist_inside_boost_jet_coefficient_at_chi0_psi0"] = {"U_norm/(b^2 k^2)": str(sp.factor((Un / (bb * bb * k * k)).subs({chi: 0, psi: 0}))),
                                                             "U_rebuild/(b^2 k^2)": str(sp.factor((Ur / (bb * bb * k * k)).subs({chi: 0, psi: 0})))}
    log(f"  twist-inside jet, chi0 psi0: U_norm = {rec['twist_inside_boost_jet_coefficient_at_chi0_psi0']['U_norm/(b^2 k^2)']}, U_rebuild = {rec['twist_inside_boost_jet_coefficient_at_chi0_psi0']['U_rebuild/(b^2 k^2)']}")
    for kk, v in jets.items():
        log(f"  jet {kk}: U_norm {v['U_norm']:+.4f} U_rebuild {v['U_rebuild']:+.4f} rel {v['rel_diff']:+.3e}")
    OUT["claims"]["C1"] = rec
    return rec


# ============================================================== C4
def c4():
    log("C4: the local circle")
    rec = {}
    g, d, s, psi, phi, al = sp.symbols("g delta s psi phi alpha", real=True)
    Ds = sp.diag(g, 1, d + s, d - s)
    Q = rot(1, 2, psi) * rot(2, 3, phi)
    M = Q * Ds * Q.T
    Rn = rot(1, 2, psi) * rot(2, 3, al / 2) * rot(1, 2, psi).T               # rotation about n = R12(psi) e1 by alpha/2
    TM = Rn * M * Rn.T
    Mshift = M.subs(phi, phi + al / 2)
    def numzero(expr, syms, n=6, seed=3):
        """max |expr| over random rational substitutions (trig identities that simplify() misses)."""
        rng = np.random.default_rng(seed)
        f = sp.lambdify(syms, expr, "numpy")
        return float(max(np.max(np.abs(np.asarray(f(*rng.uniform(-2, 2, size=len(syms))), dtype=float))) for _ in range(n)))
    syms = (g, d, s, psi, phi, al)
    rec["T_alpha == phi -> phi + alpha/2 on the sheet (max abs, random points)"] = numzero(TM - Mshift, syms)
    rec["T_2pi == id (max abs)"] = numzero(TM.subs(al, 2 * sp.pi) - M, syms)
    rec["s = 0 fixed pointwise (max abs)"] = numzero((TM - M).subs(s, 0), syms)
    rec["T_alpha == phi -> phi + alpha/2 on the sheet"] = rec["T_alpha == phi -> phi + alpha/2 on the sheet (max abs, random points)"] < 1e-12
    rec["T_2pi == id"] = rec["T_2pi == id (max abs)"] < 1e-12
    rec["s = 0 fixed pointwise"] = rec["s = 0 fixed pointwise (max abs)"] < 1e-12
    # weight two on B: the traceless (2,3) block in the frame R12(psi): angle 2 phi
    Bloc = (rot(1, 2, psi).T * M * rot(1, 2, psi))[2:4, 2:4]
    Bt = sp.simplify(Bloc - (Bloc.trace() / 2) * sp.eye(2))
    rec["B in the local frame"] = {"B_22": str(sp.simplify(Bt[0, 0])), "B_23": str(sp.simplify(Bt[0, 1]))}
    log(f"  sheet: T_alpha = phi shift {rec['T_alpha == phi -> phi + alpha/2 on the sheet']}, T_2pi = id {rec['T_2pi == id']}, s=0 fixed {rec['s = 0 fixed pointwise']}, B = ({rec['B in the local frame']['B_22']}, {rec['B in the local frame']['B_23']})")
    # invariances on the sheet with z-dependence: psi(z), phi(z)
    psz, phz = sp.symbols("psi_z phi_z", real=True)
    Az = psz * sp.diff(M, psi) + phz * sp.diff(M, phi)
    P23 = rot(1, 2, psi) * sp.diag(0, 0, 1, 1) * rot(1, 2, psi).T
    Om = P23 * Az * ETA * P23
    kp = (Om.T * ETA * Om * ETA).trace() / 2
    e2 = (Az * Az).trace()                                                    # G = I on this sheet (u = e0)
    N = M * ETA
    traces = [(N ** p).trace() for p in range(1, 5)]
    syms2 = (g, d, s, psi, phi, psz, phz)
    kp_phi = numzero(sp.diff(kp, phi), syms2)
    e2_phi = numzero(sp.diff(e2, phi), syms2)
    tr_phi = max(numzero(sp.diff(t, phi), syms2) for t in traces)
    # the closed forms at psi = 0 (the local frame): K_P sees the split rotation phi_z, E2 sees both gradients
    kp0 = sp.factor(sp.simplify(kp.subs(psi, 0)))
    e20 = sp.factor(sp.simplify(e2.subs(psi, 0)))
    rec["sheet densities"] = {"K_P^23 static at psi = 0": str(kp0), "K_P^23 max |d/dphi| (random points)": kp_phi, "K_P^23 phi-independent": kp_phi < 1e-10,
                              "E2 static at psi = 0": str(e20), "E2 max |d/dphi|": e2_phi, "E2 phi-independent": e2_phi < 1e-10,
                              "traces max |d/dphi|": tr_phi, "traces phi-independent": tr_phi < 1e-10}
    log(f"  sheet K_P^23 static (psi 0) = {kp0}, max|d/dphi| {kp_phi:.1e}; E2 (psi 0) = {e20}, max|d/dphi| {e2_phi:.1e}; traces max|d/dphi| {tr_phi:.1e}")
    # ---- point jets, numeric
    rng = np.random.default_rng(1606)
    sv = 0.15
    M0 = np.diag([G, 1.0, DELTA + sv, DELTA - sv])
    Gs = [B8.G1, B8.G2, B8.G3]

    def rand_spatial_jet():
        X = rng.normal(size=(3, 3)); X = 0.5 * (X + X.T)
        A = np.zeros((4, 4)); A[1:, 1:] = X
        return A
    A1, A2 = rand_spatial_jet(), rand_spatial_jet()
    cfg = C15.cfg_dd(4, 4.0, mu=1e-2, cP=1.0)

    def field(x1, x2):
        return M0 + x1 * A1 + x2 * A2

    def director(Mx):
        w, V = np.linalg.eigh(Mx[1:, 1:])
        n = V[:, -1]
        return n if n[0] > 0 else -n

    def Rn(n, beta):
        Gn = sum(n[a] * Gs[a] for a in range(3))
        return np.eye(4) + np.sin(beta) * Gn + (1 - np.cos(beta)) * (Gn @ Gn)

    def transformed(x1, x2, alpha):
        Mx = field(x1, x2)
        R = Rn(director(Mx), alpha / 2)
        return R @ Mx @ R.T

    def jets(alpha, eps=1e-3):
        def d(i, e):
            xp = [0.0, 0.0]; xm = [0.0, 0.0]
            xp[i] += e; xm[i] -= e
            return (transformed(*xp, alpha) - transformed(*xm, alpha)) / (2 * e)
        out = []
        for i in range(2):
            d1, d2 = d(i, eps), d(i, eps / 2)
            out.append((4 * d2 - d1) / 3)                                     # Richardson
        return transformed(0, 0, alpha), out

    def dens(alpha):
        Mp, (B1, B2) = jets(alpha)
        Fe = B1 @ ETAn @ B2 - B2 @ ETAn @ B1
        u0, *_ = EXT.timelike_eig_np(Mp[None])
        hu = u0[0] @ ETAn
        Gm = ETAn + 2.0 * np.outer(hu, hu)
        Fg = B1 @ Gm @ B2 - B2 @ Gm @ B1
        i1 = np.trace(ETAn @ Fe @ ETAn @ Fe.T)
        i1h = np.trace(Gm @ Fe @ Gm @ Fe.T)
        i1r = np.trace(Gm @ Fg @ Gm @ Fg.T)
        e2 = sum(np.trace(B @ Gm @ B @ Gm) for B in (B1, B2))
        _, _, _, P, *_ = C15.projectors(Mp[None])
        P = P[0]
        kp = 0.0
        for B in (B1, B2):
            Om = P @ B @ ETAn @ P
            kp += 0.5 * np.trace(Om.T @ ETAn @ Om @ ETAn)
        e_u, e_v = INS4.e_parts(np.broadcast_to(Mp, (1, 1, 1, 4, 4)).copy(), cfg)
        spl, _ = C15.split_cells(Mp[None], need_grad=False)
        return {"I1": i1, "I1_h (I_norm)": i1h, "I_rebuild": i1r, "E2": e2, "K_P^23": kp, "V4^dd": float(e_v), "split": float(spl[0])}
    alphas = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    rows = [dens(a) for a in alphas]
    keys = list(rows[0].keys())
    per = {}
    for kkey in keys:
        v = np.array([r[kkey] for r in rows])
        ref = max(np.max(np.abs(v)), 1e-300)
        spec = np.abs(np.fft.rfft(v)) / len(v)
        deg = int(np.max(np.nonzero(spec > 1e-9 * max(spec[0], 1e-300))[0])) if np.any(spec[1:] > 1e-9 * max(spec[0], 1e-300)) else 0
        # doubling test: equispaced averages with 2, 4, 8, 16, 32 samples against the 64-sample mean
        avg64 = float(np.mean(v))
        dbl = {}
        for nsmp in (2, 4, 8, 16, 32):
            idx = np.arange(0, 64, 64 // nsmp)
            dbl[str(nsmp)] = float(abs(np.mean(v[idx]) - avg64) / max(abs(avg64), 1e-300))
        v2pi = dens(2 * np.pi)[kkey]
        per[kkey] = {"alpha_variation_rel": float((np.max(v) - np.min(v)) / ref), "invariant_1e-9": bool((np.max(v) - np.min(v)) / ref < 1e-9),
                     "trig_degree_in_alpha": deg, "value_alpha0": float(v[0]), "circle_average": avg64,
                     "doubling_test_rel_err": dbl, "periodic_2pi_rel": float(abs(v2pi - v[0]) / ref)}
        log(f"  point jet {kkey:14s}: variation {per[kkey]['alpha_variation_rel']:.2e} invariant {per[kkey]['invariant_1e-9']} trig degree {deg} "
            f"periodic {per[kkey]['periodic_2pi_rel']:.1e} doubling {dbl}")
    rec["point_jets_s0.15"] = per
    # ---- lattice: K_P^23 invariance with a varying director, O(h^2) convergence of the defect
    lat = {}
    for n in (16, 32):
        cfgl = C15.cfg_dd(n, 8.0, mu=1e-2, cP=1.0)
        rngl = np.random.default_rng(7)
        X = rngl.normal(size=(6, 6, 6, 3, 3)); X = 0.5 * (X + np.swapaxes(X, -1, -2))
        # a smooth field: trilinear interpolation of a coarse random field (the same coarse field for both n)
        from scipy.ndimage import zoom
        Xf = np.stack([np.stack([zoom(X[..., a, b], n / 6, order=3) for b in range(3)], -1) for a in range(3)], -2)
        Ml = np.broadcast_to(np.diag([G, 1.0, DELTA + sv, DELTA - sv]), (n, n, n, 4, 4)).copy()
        Ml[..., 1:, 1:] += 0.12 * Xf
        w, V = np.linalg.eigh(Ml[..., 1:, 1:])
        nn = V[..., :, -1]
        sgn = np.where(nn[..., 0] > 0, 1.0, -1.0)
        nn = nn * sgn[..., None]
        gap = float(np.min(w[..., -1] - w[..., -2]))

        def T(alpha):
            Gn = nn[..., 0, None, None] * B8.G1 + nn[..., 1, None, None] * B8.G2 + nn[..., 2, None, None] * B8.G3
            R = np.eye(4) + np.sin(alpha / 2) * Gn + (1 - np.cos(alpha / 2)) * (Gn @ Gn)
            return R @ Ml @ np.swapaxes(R, -1, -2)

        def reads(Mx):
            e_u, e_v = INS4.e_parts(Mx, cfgl)
            kp = C15.kp23_energy_grad(Mx, cfgl, need_grad=False)[0]
            spl = C15.split_energy_grad(Mx, cfgl, need_grad=False)[0]
            ih = C15.i1h_static(Mx, cfgl)
            h3 = cfgl["h"] ** 3
            e2 = 0.0
            Gm = EXT.h_cov_np(Mx)
            for br, wt in INS4.branches(cfgl["stencil"]):
                for ax in range(3):
                    A = INS4.d1(Mx, ax, cfgl["h"], br)
                    e2 += wt * h3 * float(np.sum(np.einsum("...ab,...bc,...cd,...da->...", A, Gm, A, Gm)))
            return {"E_u (4 I1)": float(e_u), "V4^dd": float(e_v), "K_P^23": float(kp), "split": float(spl), "4 I1_h": float(ih), "E2 (G = h)": e2}
        r0 = reads(Ml)
        row = {"h": cfgl["h"], "min_director_gap": gap, "base": r0, "defect_rel": {}}
        for alpha in (0.7, 2.1, np.pi, 2 * np.pi):
            ra = reads(T(alpha))
            row["defect_rel"][f"{alpha:.4f}"] = {kk: float(abs(ra[kk] - r0[kk]) / max(abs(r0[kk]), 1e-300)) for kk in r0}
        lat[str(n)] = row
        log(f"  lattice n{n} h{cfgl['h']}: base {r0}; defects {row['defect_rel']}")
    if "16" in lat and "32" in lat:
        ratio = {kk: (lat["16"]["defect_rel"]["0.7000"][kk] / max(lat["32"]["defect_rel"]["0.7000"][kk], 1e-300)) for kk in lat["16"]["base"]}
        rec["lattice_defect_ratio_h_over_h/2_at_alpha0.7"] = ratio
        log(f"  defect ratio h/(h/2) at alpha 0.7 (4 = O(h^2) continuum invariance, ~1 = a real non-invariance): {ratio}")
    rec["lattice"] = lat
    OUT["claims"]["C4"] = rec
    return rec


# ============================================================== C5
def c5():
    log("C5: sheet inertias, the tilt condition, the boost sheets")
    rec = {}
    g, d, s, om, psz, b, cs = sp.symbols("g delta s omega psi_z b c_s", positive=True)
    psi = sp.symbols("psi", real=True)
    Ds = sp.diag(g, 1, d + s, d - s)
    G23 = sp.zeros(4); G23[2, 3] = -1; G23[3, 2] = 1
    R = rot(1, 2, psi)
    M = R * Ds * R.T
    Gloc = R * G23 * R.T
    Az = psz * sp.diff(M, psi)
    out = {}
    for lab, Gen in (("local G_loc = R G23 R^T", Gloc), ("rigid G23", G23)):
        A0 = om * (Gen * M - M * Gen)
        F = comm_eta(A0, Az)
        ff = sp.factor(sp.simplify(inner(F, ETA)))
        out[lab] = {"<F_0z,F_0z>_eta": str(ff), "at s = 0": str(sp.factor(ff.subs(s, 0)))}
        log(f"  twist sheet, {lab}: <F0z,F0z>_eta = {ff}; at s = 0: {sp.factor(ff.subs(s, 0))}")
    rec["twist_sheet_inertia"] = out
    rec["author"] = "8 omega^2 psi'^2 s^2 (delta + s - 1)^2 (local), nonzero at s = 0 (rigid)"
    # K_P^23 on the sheet: static zero, inertia under the local generator
    P23 = R * sp.diag(0, 0, 1, 1) * R.T
    A0 = om * (Gloc * M - M * Gloc)
    Om0 = P23 * A0 * ETA * P23
    Omz = P23 * Az * ETA * P23
    kp_kin = sp.factor(sp.simplify((Om0.T * ETA * Om0 * ETA).trace() / 2))
    kp_stat = sp.simplify((Omz.T * ETA * Omz * ETA).trace() / 2)
    e2_stat = sp.factor(sp.simplify((Az * Az).trace()))
    rec["K_P^23 on the twist sheet"] = {"inertia density (1/2) tr(Om_0^T eta Om_0 eta)": str(kp_kin), "static density": str(kp_stat)}
    rec["E2 static on the twist sheet (G = 1)"] = str(e2_stat)
    rec["rho^2 E2 static"] = str(sp.factor(s * s * e2_stat))
    log(f"  K_P^23 inertia density on the sheet = {kp_kin} (the author: inertia 8 omega^2 s^2, i.e. K = dL/domega with L = 4 c_P omega^2 s^2); static = {kp_stat}")
    log(f"  E2 static = {e2_stat}; rho^2 E2 = {sp.factor(s * s * e2_stat)} (the author: stiffness prop s^2 psi'^2)")
    # the tilt condition: R15-H stored forms with w -> c_s s^2
    H = json.load(open(os.path.join(DATA, "m5_32_r15_h_tilt.json")))
    w = sp.symbols("w", positive=True)
    loc = {"delta": d, "s": s, "omega": om, "w": w, "c_P": sp.symbols("c_P", positive=True)}
    gam = sp.sympify(H["hyperbolicity"]["gamma_total"], locals=loc)
    alp = sp.sympify(H["hyperbolicity"]["alpha_total"], locals=loc)
    gam_cs = sp.factor(gam.subs(w, cs * s * s))
    alp_cs = sp.factor(alp.subs(w, cs * s * s))
    cond = sp.solve(sp.simplify(-gam_cs / (2 * (d + s - 1) ** 2 * s * s)) > 0, cs)
    rec["tilt_condition"] = {"R15-H gamma_total": H["hyperbolicity"]["gamma_total"], "R15-H alpha_total": H["hyperbolicity"]["alpha_total"],
                             "with w = c_s s^2: gamma": str(gam_cs), "alpha": str(alp_cs), "hyperbolic iff": str(cond),
                             "author": "c_s > 16 omega^2, independent of s and delta"}
    log(f"  tilt: gamma(w = c_s s^2) = {gam_cs}, alpha = {alp_cs} -> hyperbolic iff {cond}")
    # boost sheets on the diagonal split sheet (psi = 0)
    Md = Ds
    A0d = om * (G23 * Md - Md * G23)
    u = E0
    Hm = hcov(u)
    bs = {}
    for i in (1, 2, 3):
        K = sp.zeros(4); K[0, i] = 1; K[i, 0] = 1
        Azb = b * (K * Md + Md * K)
        Fe = comm_eta(A0d, Azb)
        Fg = comm_G(A0d, Azb, Hm)
        bs[f"K = E_0{i} + E_{i}0"] = {"<F,F>_eta": str(sp.factor(sp.simplify(inner(Fe, ETA)))),
                                       "<F^eta,F^eta>_G (I_norm)": str(sp.factor(sp.simplify(inner(Fe, Hm)))),
                                       "<F^G,F^G>_G (I_rebuild)": str(sp.factor(sp.simplify(inner(Fg, Hm))))}
        log(f"  boost sheet i={i}: eta {bs[f'K = E_0{i} + E_{i}0']['<F,F>_eta']}; norm {bs[f'K = E_0{i} + E_{i}0']['<F^eta,F^eta>_G (I_norm)']}; rebuild {bs[f'K = E_0{i} + E_{i}0']['<F^G,F^G>_G (I_rebuild)']}")
    rec["boost_sheets"] = bs
    rec["boost_author"] = "-8 b^2 omega^2 s^2 (g - delta -+ s)^2 in the pair planes (all-eta), zero along the director"
    OUT["claims"]["C5"] = rec
    return rec


# ============================================================== C6
def c6():
    log("C6: the eigenvalue metric at the degenerate pair")
    a, b, da, db = sp.symbols("a b da db", real=True)
    Bm = sp.Matrix([[a, b], [b, -a]])
    lam = sp.sqrt(a * a + b * b)
    dl = (sp.diff(lam, a) * da + sp.diff(lam, b) * db)
    metric = sp.simplify(dl ** 2 + dl ** 2)                        # (d lambda_+)^2 + (d lambda_-)^2, lambda_- = -lambda_+
    rho2 = sp.simplify((Bm * Bm).trace() / 2)
    drho2 = sp.diff(rho2, a) * da + sp.diff(rho2, b) * db
    t = sp.symbols("t", positive=True)
    # direction dependence at the origin: along (da, db) = t (cos c, sin c)
    c = sp.symbols("c", real=True)
    lim = sp.simplify(metric.subs({a: t * sp.cos(c), b: t * sp.sin(c), da: sp.cos(c), db: sp.sin(c)}))
    lim2 = sp.simplify(metric.subs({a: t * sp.cos(c), b: t * sp.sin(c), da: -sp.sin(c), db: sp.cos(c)}))
    rec = {"(d lambda_+)^2 + (d lambda_-)^2": str(metric), "author": "2 (a da + b db)^2 / (a^2 + b^2)",
           "matches_author": sp.simplify(metric - 2 * (a * da + b * db) ** 2 / (a * a + b * b)) == 0,
           "radial direction at the origin": str(lim), "tangential direction at the origin": str(lim2),
           "rho^2 = (1/2) tr B^2": str(rho2), "d rho^2": str(sp.expand(drho2)), "rho^2 smooth (polynomial)": True}
    log(f"  metric = {metric} (author match {rec['matches_author']}); radial limit {lim}, tangential {lim2}; rho^2 = {rho2}")
    OUT["claims"]["C6"] = rec
    return rec


if __name__ == "__main__":
    for fn in (c1, c6, c5, c4):
        fn()
        json.dump(OUT, open(os.path.join(CK, "r16_0_symbolic.json"), "w"), indent=1, default=str)
    OUT["wall_s"] = round(time.time() - T0, 1)
    json.dump(OUT, open(os.path.join(DATA, "m5_32_r16_0_symbolic.json"), "w"), indent=1, default=str)
    log("wrote data/m5_32_r16_0_symbolic.json")
