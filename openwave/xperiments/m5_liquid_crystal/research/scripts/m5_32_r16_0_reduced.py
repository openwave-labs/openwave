"""M5.32 R16-0 (the reduced line): claim C2 of the author's 2026-09-06 comments, Coleman's condition
(ledger 6.5, the R16-0 row).  The author's reduced model, our numbers.

EQUATIONS FIRST
---------------
The diagonal split sheet N = diag(-g, 1, delta + s, delta - s) (code branch s = -1, g = 8, delta = 0.3):
    V4^dd(s) = W1 sum_p (tr N^p - C_p)^2,  C_p = (-g)^p + 1 + 2 delta^p
             = W1 [(2 s^2)^2 + (6 delta s^2)^2 + (12 delta^2 s^2 + 2 s^4)^2]  =  a s^4 + O(s^6),  quartic-flat at the pair.
Two normalizations of the split potential, both on record:  U_R15 = mu (lambda_2 - lambda_3)^2 = 4 mu s^2 (the R15 object),
    U_v4 = mu rho^2 = mu s^2 (the author's v4, rho^2 = (1/2) tr B^2 = s^2 on the sheet).
The K_P^23 inertia of the uniform split under the (2,3) clock (our instrument, per unit volume):
    kin = (1/2) tr(Om_0^T eta Om_0 eta), Om_0 = P23 [G23, M] eta P23  ->  c s^2 with c measured (expected 4 c_P).
The author's reduced functional  F = int [(c/2) s'^2 + V(s) - omega^2 c s^2]  (one c for gradient and inertia).
Coleman's condition (the author): a Maxwell crossing needs an interior minimum of V(s)/s^2; the onset is
omega_c^2 = min_s V/s^2 / c; the crossing omega*^2 c = min V/s^2 attained at s* > 0 is first order.
    V = mu s^2 + a s^4:  V/s^2 = mu + a s^2, minimized at s -> 0: no crossing, omega_c^2 = mu/c (the R15 (iii) theorem).
    V = mu s^2 - nu s^4 + kappa s^6 (object v2):  min V/s^2 = mu - nu^2/(4 kappa) at s*^2 = nu/(2 kappa)
        (the author: 9e-3 and s* = 0.2236 at (1e-2, 4e-2, 0.4)).
Fixed J on the reduced line:  E_J = int 4 pi r^2 [(c/2) s'^2 + V(s)] dr + J^2 / (4 int 4 pi r^2 c s^2 dr)
    (the uniform limit E = V(s) Vol + J^2 / (4 c s^2 Vol) with x = Vol s^2: infimum omega_c J as Vol -> infinity,
     never attained, for V/s^2 minimized at 0); a localized profile with E_J < omega_c J exists iff Coleman's condition holds.
The weighted Coleman condition (the author's correction 4): Om = w(N) dN w(N) gives C(s) = c s^2 W(s),
    W(s) = [w(delta + s) w(delta - s)]^2;  the crossing needs min_s U/(s^2 W) < mu... (the author: with the rational
    weight w = f(lambda)/f(delta), f = (lambda - g)(lambda - 1)/[(lambda - g)^2 + (lambda - 1)^2], min U/(s^2 W) = 0.01117 > mu,
    no crossing; the plateau weight W = 1 restores 0.0090 at s* = 0.224).
NOT CHECKABLE from the thread (definitions absent): the Gaussian control 13/12 - sqrt(3)/18 = 0.987, the v3 Q-ball
branch (J >~ 2000), "dE/dJ = omega to 1 percent, stable branch 2 percent below the gap".

usage: python3 m5_32_r16_0_reduced.py
out:   data/m5_32_r16_0_reduced.json, plots/m5_32_r16_0_c2_coleman.png, checkpoints/m5_32_r16/reduced.log
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize, minimize_scalar

sys.argv = [sys.argv[0]]
import m5_32_r15_common as C15                            # noqa: E402

INS4, B8 = C15.INS4, C15.B8
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
DATA, PLOTS = os.path.join(RES, "data"), os.path.join(RES, "plots")
CK = os.path.join(RES, "checkpoints", "m5_32_r16")
os.makedirs(CK, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
T0 = time.time()
LOG = open(os.path.join(CK, "reduced.log"), "a")


def log(m):
    line = f"[{time.time() - T0:8.1f}s] {m}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


G, DELTA, W1 = C15.G, C15.DELTA, C15.W1
MU, NU, KAP, CP = 1e-2, 4e-2, 0.4, 1.0
OUT = {"rung": "R16-0 reduced line (C2)", "params": {"g": G, "delta": DELTA, "W1": W1, "mu": MU, "nu": NU, "kappa": KAP, "c_P": CP}}


def v4dd(s):
    """V4^dd on the diagonal split sheet, the exact trace form."""
    d = DELTA
    return W1 * ((2 * s ** 2) ** 2 + (6 * d * s ** 2) ** 2 + (12 * d * d * s ** 2 + 2 * s ** 4) ** 2)


def a_coef():
    d = DELTA
    return W1 * (4 + 36 * d * d + 144 * d ** 4)


def main():
    # ---- the sheet potential against the lattice instrument (gate)
    cfg = C15.cfg_dd(4, 4.0, mu=MU, cP=CP)
    gate = []
    for s in (0.05, 0.15, 0.25):
        M = np.broadcast_to(np.diag([G, 1.0, DELTA + s, DELTA - s]), (4, 4, 4, 4, 4)).copy()
        e_u, e_v = INS4.e_parts(M, cfg)
        per_cell = float(e_v) / (4 ** 3 * cfg["h"] ** 3)
        gate.append({"s": s, "V4dd_formula": v4dd(s), "V4dd_lattice_per_volume": per_cell, "rel": abs(per_cell - v4dd(s)) / v4dd(s)})
    OUT["V4dd_sheet_gate"] = gate
    log(f"V4^dd sheet formula vs lattice: {gate}")
    OUT["a_quartic_coefficient"] = {"a": a_coef(), "formula": "W1 (4 + 36 delta^2 + 144 delta^4)", "check_s6_term_small_at_s0.3": (v4dd(0.3) - a_coef() * 0.3 ** 4) / v4dd(0.3)}
    # ---- the K_P^23 inertia coefficient c of the uniform split (per unit volume, per s^2)
    cs = []
    for s in (0.05, 0.15, 0.25):
        M = np.broadcast_to(np.diag([G, 1.0, DELTA + s, DELTA - s]), (2, 2, 2, 4, 4)).copy()
        a0 = B8.G1 @ M - M @ B8.G1
        E, _, _ = C15.kp23_cells([a0], M, need_grad=False)
        cs.append({"s": s, "kin_per_volume": float(E[0, 0, 0]), "kin/s^2": float(E[0, 0, 0]) / s ** 2})
    c_meas = cs[0]["kin/s^2"]
    OUT["KP23_inertia_uniform_split"] = {"rows": cs, "c = kin/(s^2) per c_P": c_meas, "expected": "4 (Om_0 = [[0, 2s], [2s, 0]] on the pair block, (1/2)|Om|^2 = 4 s^2)"}
    log(f"K_P^23 inertia of the uniform split: c = {c_meas:.6f} per c_P (expected 4)")
    c = CP * c_meas
    # ---- Coleman: V/s^2 for the three potentials
    s_grid = np.linspace(1e-4, 0.34, 4000)
    pots = {"V4dd + U_R15 (4 mu s^2)": lambda s: v4dd(s) + 4 * MU * s ** 2,
            "V4dd + U_v4 (mu s^2)": lambda s: v4dd(s) + MU * s ** 2,
            "sextic v2 (mu s^2 - nu s^4 + kappa s^6)": lambda s: MU * s ** 2 - NU * s ** 4 + KAP * s ** 6,
            "sextic v2 + V4dd": lambda s: MU * s ** 2 - NU * s ** 4 + KAP * s ** 6 + v4dd(s)}
    col = {}
    for lab, V in pots.items():
        r = V(s_grid) / s_grid ** 2
        i = int(np.argmin(r))
        interior = 0 < i < len(s_grid) - 1
        col[lab] = {"min V/s^2": float(r[i]), "argmin s": float(s_grid[i]), "interior_minimum": bool(interior), "V/s^2 at s->0": float(r[0]),
                    "omega_onset^2 = min(V/s^2)/c": float(r[i] / c), "omega_c^2 = (V/s^2)(0)/c": float(r[0] / c)}
        log(f"Coleman {lab}: min V/s^2 = {r[i]:.6f} at s = {s_grid[i]:.4f} (interior {interior}); (V/s^2)(0) = {r[0]:.6f}; omega_onset^2 = {r[i] / c:.5f}")
    col["sextic analytic"] = {"mu - nu^2/(4 kappa)": MU - NU ** 2 / (4 * KAP), "s* = sqrt(nu/(2 kappa))": float(np.sqrt(NU / (2 * KAP))),
                              "author": "9e-3 and 0.2236", "note": "the author's omega*^2 c = mu - nu^2/(4 kappa) holds with c the inertia coefficient of s^2; with the K_P^23 inertia c = 4 c_P the onset for U_v4 = mu s^2 is omega_c^2 = mu/(4 c_P), for U_R15 = 4 mu s^2 it is mu/c_P"}
    OUT["coleman"] = col
    # ---- the fixed-J uniform-limit ladder (the delocalized infimum)
    a = a_coef()
    J = 200.0
    lad = []
    for lab, mu_eff in (("U_R15", 4 * MU), ("U_v4", MU)):
        om_c = np.sqrt(mu_eff / c)
        rows = []
        for Vol in (1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
            f = lambda x: mu_eff * x + a * x * x / Vol + J * J / (4 * c * x)
            r = minimize_scalar(f, bounds=(1e-6, 1e9), method="bounded", options={"xatol": 1e-10})
            rows.append({"Vol": Vol, "E_min": float(r.fun), "x = Vol s^2": float(r.x), "s": float(np.sqrt(r.x / Vol)), "E_min/(omega_c J)": float(r.fun / (om_c * J))})
        lad.append({"potential": lab, "omega_c": float(om_c), "omega_c J": float(om_c * J), "rows": rows})
        log(f"fixed-J uniform ladder {lab}: omega_c = {om_c:.5f}, E_min/(omega_c J) = {[round(x['E_min/(omega_c J)'], 6) for x in rows]}")
    OUT["fixed_J_uniform_ladder"] = lad
    # ---- the 1D spherical fixed-J profile problem (our script for the author's claim: a Q-ball needs Coleman)
    # thin-wall estimate for the sextic: omega*^2 = (mu - nu^2/4kappa)/c, s*^2 = nu/2kappa, wall tension
    # sigma = int_0^{s*} sqrt(2 c V_eff) ds with V_eff = kappa s^2 (s^2 - s*^2)^2 -> sigma = sqrt(2 c kappa) s*^4 / 4;
    # J = 2 c s*^2 omega* Vol; the Q-ball undercuts omega_c J once 4 pi R^2 sigma < (omega_c - omega*) J.
    def EJ_grad(sv, V, dV, Jv, r, dr):
        w = 4 * np.pi * r * r * dr
        rm = 0.5 * (r[1:] + r[:-1])
        wm = 4 * np.pi * rm * rm * dr
        ds = (sv[1:] - sv[:-1]) / dr
        Es = np.sum(wm * 0.5 * c * ds * ds) + np.sum(w * V(sv))
        kin = np.sum(w * c * sv * sv)
        E = Es + Jv * Jv / (4 * kin)
        g = w * dV(sv) - (Jv * Jv / (4 * kin * kin)) * (2 * w * c * sv)
        gd = wm * c * ds / dr
        g[1:] += gd; g[:-1] -= gd
        return E, g, Es, kin

    dpots = {"V4dd + U_v4 (mu s^2)": lambda s: 2 * MU * s + W1 * (2 * (2 * s ** 2) * 4 * s + 2 * (6 * DELTA * s ** 2) * 12 * DELTA * s + 2 * (12 * DELTA ** 2 * s ** 2 + 2 * s ** 4) * (24 * DELTA ** 2 * s + 8 * s ** 3)),
             "sextic v2 + V4dd": lambda s: 2 * MU * s - 4 * NU * s ** 3 + 6 * KAP * s ** 5 + W1 * (2 * (2 * s ** 2) * 4 * s + 2 * (6 * DELTA * s ** 2) * 12 * DELTA * s + 2 * (12 * DELTA ** 2 * s ** 2 + 2 * s ** 4) * (24 * DELTA ** 2 * s + 8 * s ** 3))}
    # gradient gate
    rg = np.random.default_rng(0)
    rt = (np.arange(200) + 0.5) * 0.5
    st = 0.1 + 0.05 * rg.uniform(size=200)
    e0, g0, _, _ = EJ_grad(st, pots["sextic v2 + V4dd"], dpots["sextic v2 + V4dd"], 300.0, rt, 0.5)
    D = rg.normal(size=200); D /= np.linalg.norm(D)
    fd = (EJ_grad(st + 1e-6 * D, pots["sextic v2 + V4dd"], dpots["sextic v2 + V4dd"], 300.0, rt, 0.5)[0] - EJ_grad(st - 1e-6 * D, pots["sextic v2 + V4dd"], dpots["sextic v2 + V4dd"], 300.0, rt, 0.5)[0]) / 2e-6
    OUT["EJ_gradient_gate"] = {"analytic": float(np.dot(g0, D)), "fd": float(fd), "rel": float(abs(np.dot(g0, D) - fd) / abs(fd))}
    log(f"1D E_J gradient gate rel {OUT['EJ_gradient_gate']['rel']:.2e}")
    om_star2 = (MU - NU ** 2 / (4 * KAP)) / c
    s_star = np.sqrt(NU / (2 * KAP))
    sigma = np.sqrt(2 * c * KAP) * s_star ** 4 / 4
    om_c_v4 = np.sqrt(MU / c)
    # J_cross: 4 pi R^2 sigma = (omega_c - omega*) J with J = 2 c s*^2 omega* (4/3) pi R^3
    kJ = 2 * c * s_star ** 2 * np.sqrt(om_star2) * 4 * np.pi / 3
    R_cross = 4 * np.pi * sigma / ((om_c_v4 - np.sqrt(om_star2)) * kJ)
    OUT["thin_wall_estimate_sextic"] = {"omega*": float(np.sqrt(om_star2)), "omega_c": float(om_c_v4), "s*": float(s_star), "sigma": float(sigma), "R_cross": float(R_cross), "J_cross": float(kJ * R_cross ** 3),
                                        "J(R) = 2 c s*^2 omega* (4/3) pi R^3": float(kJ)}
    log(f"thin wall (sextic, c = {c:g}): omega* {np.sqrt(om_star2):.4f} omega_c {om_c_v4:.4f} sigma {sigma:.2e} -> the Q-ball undercuts omega_c J for R > {R_cross:.0f}, J > {kJ * R_cross ** 3:.0f}")
    R, N = 240.0, 2400
    r = (np.arange(N) + 0.5) * (R / N)
    dr = R / N
    w = 4 * np.pi * r * r * dr
    prof = {}
    for lab in ("V4dd + U_v4 (mu s^2)", "sextic v2 + V4dd"):
        V, dV = pots[lab], dpots[lab]
        om_c = np.sqrt(MU / c)
        rows = []
        for Jv in (200.0, 5000.0, 3e4, 1e5, 3e5):
            best = None
            for R0 in (10.0, 30.0, 60.0, 100.0, 150.0):
                s0 = s_star * 0.5 * (1 - np.tanh((r - R0) / 4.0)) + 1e-3
                res = minimize(lambda x: EJ_grad(x, V, dV, Jv, r, dr)[:2], s0, jac=True, method="L-BFGS-B", bounds=[(1e-6, 0.34)] * N, options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-11})
                if best is None or res.fun < best[0]:
                    best = (res.fun, res.x, R0)
            e, sv, R0 = best
            _, _, Es, kin = EJ_grad(sv, V, dV, Jv, r, dr)
            cum = np.cumsum(w * sv ** 2) / np.sum(w * sv ** 2)
            r90 = float(r[np.searchsorted(cum, 0.9)])
            rows.append({"J": Jv, "E_J": float(e), "E_stat": float(Es), "kin": float(kin), "omega = J/(2 kin)": float(Jv / (2 * kin)), "E_J/(omega_c J)": float(e / (om_c * Jv)),
                         "s_max": float(np.max(sv)), "r90": r90, "localized (r90 < R/2)": bool(r90 < R / 2), "seed_R0": R0})
            log(f"1D fixed-J {lab} J {Jv:g}: E_J {e:.4f} = {e / (om_c * Jv):.4f} omega_c J, s_max {np.max(sv):.4f}, r90 {r90:.1f}, omega {Jv / (2 * kin):.5f}")
        prof[lab] = {"omega_c": float(om_c), "rows": rows, "box_R": R, "N": N}
    OUT["fixed_J_1D_profiles"] = prof
    # ---- the weighted Coleman condition
    def w_rational(lam, gz):
        f = lambda x: (x - gz) * (x - 1.0) / ((x - gz) ** 2 + (x - 1.0) ** 2)
        return f(lam) / f(DELTA)
    wc = {}
    lam_grid = np.linspace(-9, 9, 40001)
    for gz, lab in ((8.0, "author literal: zeros at +g and 1"), (-8.0, "our branch: zeros at -g (the timelike eigenvalue) and 1")):
        wr = w_rational(lam_grid, gz)
        sup = float(np.max(np.abs(wr)))
        rows = {}
        for plab, V in (("sextic v2", pots["sextic v2 (mu s^2 - nu s^4 + kappa s^6)"]), ("sextic v2 + V4dd", pots["sextic v2 + V4dd"]), ("V4dd + U_v4", pots["V4dd + U_v4 (mu s^2)"])):
            Wr = (w_rational(DELTA + s_grid, gz) * w_rational(DELTA - s_grid, gz)) ** 2
            ratio = V(s_grid) / (s_grid ** 2 * Wr)
            i = int(np.argmin(ratio))
            j_ = int(np.argmin(np.abs(s_grid - np.sqrt(NU / (2 * KAP)))))
            rows[plab] = {"min U/(s^2 W)": float(ratio[i]), "argmin s": float(s_grid[i]), "> mu": bool(ratio[i] > MU), "U/(s^2 W) at s* = 0.2236": float(ratio[j_]), "W(s*)": float(Wr[j_])}
        wc[lab] = {"sup|w| on [-9, 9]": sup, "w at the timelike eigenvalue -g": float(w_rational(-8.0, gz)), "w at 1": float(w_rational(1.0, gz)), "w at -1": float(w_rational(-1.0, gz)), "rows": rows}
        log(f"weighted Coleman, {lab}: sup|w| {sup:.3f}, w(-g) {w_rational(-8.0, gz):.3f}; {rows}")
    # the plateau weight: W = 1 on the whole sheet for |s| <= 0.5 (both pair members inside the plateau)
    rows = {}
    for plab, V in (("sextic v2", pots["sextic v2 (mu s^2 - nu s^4 + kappa s^6)"]), ("sextic v2 + V4dd", pots["sextic v2 + V4dd"])):
        ratio = V(s_grid) / s_grid ** 2
        i = int(np.argmin(ratio))
        rows[plab] = {"min U/(s^2 W), W = 1": float(ratio[i]), "argmin s": float(s_grid[i])}
    wc["plateau weight (W = 1 on |s| <= 0.5)"] = rows
    wc["author"] = "rational: min U/(s^2 W) = 0.01117 > mu (no crossing); plateau: 0.0090 at s* = 0.224"
    OUT["weighted_coleman"] = wc
    OUT["not_checkable_from_the_thread"] = ["Gaussian control 13/12 - sqrt(3)/18 = 0.987 < 1 (definition absent)", "reduced v3 Q-ball: bound branch only for J >~ 2000, dE/dJ = omega to 1 percent, stable branch 2 percent below the gap (v3 not defined in the thread)"]
    # ---- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for lab, V in pots.items():
        ax[0].plot(s_grid, V(s_grid) / s_grid ** 2, label=lab)
    ax[0].axhline(MU, color="k", lw=0.6, ls="--"); ax[0].set_xlabel("s"); ax[0].set_ylabel("V/s^2"); ax[0].set_ylim(0, 0.05); ax[0].legend(fontsize=7); ax[0].set_title("Coleman: V/s^2 (interior minimum = crossing)")
    for lab, P in prof.items():
        ax[1].plot([x["J"] for x in P["rows"]], [x["E_J/(omega_c J)"] for x in P["rows"]], "o-", label=lab)
    ax[1].axhline(1.0, color="k", lw=0.6, ls="--"); ax[1].set_xscale("log"); ax[1].set_xlabel("J"); ax[1].set_ylabel("E_J / (omega_c J)"); ax[1].legend(fontsize=7); ax[1].set_title("1D fixed-J profiles (c = K_P inertia)")
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "m5_32_r16_0_c2_coleman.png"), dpi=110)
    OUT["wall_s"] = round(time.time() - T0, 1)
    json.dump(OUT, open(os.path.join(DATA, "m5_32_r16_0_reduced.json"), "w"), indent=1, default=float)
    log("wrote data/m5_32_r16_0_reduced.json, plots/m5_32_r16_0_c2_coleman.png")


if __name__ == "__main__":
    main()
