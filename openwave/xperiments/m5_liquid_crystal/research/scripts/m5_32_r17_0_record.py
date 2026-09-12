"""M5.32 R17-0 (the record arm, ledger 6.6): the dual-reading falsifier read on our own fields
(a), request 3 from the record (b), the second agent's reconstruction and the section-28
definitions on the full R16 fields (c), the spin-weight-2 shell content on the R16-1 cores and the
R16-3 end states (d), and the collision capacity on the saved R16-2 mode (h).  Every author number
is a claim to check; every number here is ours.

EQUATIONS FIRST
---------------
(a) The tail.  On a uniaxial texture M_sp = delta I + (1 - delta) n n^T the static quartic density of
    the certified stack is E_h = 4 I1 = 8 (1 - delta)^4 Omega^2, Omega^2 = sum_{i<j} [n . (d_i n x d_j n)]^2
    (the report's 39.1 identity; derived by sympy in m5_32_r17_0_symbolic.py), and on the unit hedgehog
    Omega^2 = 1 / r^4, so the far-field density is A / r^4 with A = 8 (1 - delta)^4 = 1.92080 at delta 0.3.
    MEASURED here: the shell mean of the per-cell E_h density times r^4 on the three R16-1 end fields
    and on the analytic seed (the exact control), shells inside the inscribed sphere and outside the
    core (0.15 L to 0.42 L); the log-log slope and the amplitude; the same read on the certified-stack
    single hedgehog of the R3 record (vacuum (1, delta, 0), g 32) and on the M5.21.4 3x3 single
    (c2_selfcal = A / 8 by that script's definition).
    The pair.  The record's pair energies E_int(d) = E(pair, d) - 2 E(single) in the SAME unit system as
    the tail read on the same stack (R3 undressed rows on the certified 4x4 stack at n32 L48, which
    reproduce the M5.21.4 3x3 ladder), fitted as A_fit + B / d and as a log-log slope; the
    superposition prediction 8 pi A / d (the M5.21.4 form 64 pi c2 / d with c2 = A / 8) placed beside
    them; the report's pair 4 (1 - delta)^4 / (pi d) placed in the same units.  The normalization
    verdict itself is decided in the symbolic arm (the derivation); this arm decides
    PAIR_LAW_CERTIFIED / PAIR_LAW_NOT_CERTIFIED on the record (a certified 1/d coefficient needs a
    like-charge E_int(d) that FALLS as 1/d; the record's like-charge E_int RISES with d).
(b) Request 3 (relax a dressed pair and read the force): the record rows, both protocols, from the
    stored JSONs: R3 (ii) relaxed dressed pairs (amplitudes held), R14-C relaxed R_G pairs, R11 the
    imposed same-sign notebook pair, R3 (i) the imposed ansatz; sign per row.
(c) Per cell the spectrum (lambda_g, lambda_1, lambda_2, lambda_3) of N = M eta (the R15 projectors:
    s = l2 + l3, p = l2 l3, half split hs = sqrt(s^2 - 4 p) / 2, pair mean m = s / 2).  The split
    curvature of the potential at the cell's own spectrum, the pair varied as (m + x, m - x):
        hV(x) = d^2 / dx^2 [ W1 sum_p (lg^p + l1^p + (m + x)^p + (m - x)^p - C_p)^2 ]  at x = hs
    (sympy-derived closed form; also reported divided by W1 since the second agent's normalization
    is not stated).  The director gap Delta = l1 - l2, Delta_min = min over the free cells (the
    domain's gap_1_2_min) and over the central line; the core radius r_0 as the taper-core radius
    r_d (the largest r with l1 < 0.8, the plateau edge) and as r_half (l1 < (1 + delta) / 2); r_0 sqrt(mu).
(d) zeta = S_ee - S_ff + 2 i S_ef in the frame transverse to the outward-lifted director (R16-0 C8),
    decomposed on shells in the spin-weight-2 harmonics 2Y_2m: P_m = |c_m|^2, <m>, the l = 2 fraction.
(h) K_coll = 2 omega Delta_min^2 int f^2 d^3x with f = |zeta| / max |zeta| of the R16-2 lowest doublet
    mode on the R16-1 n32 core, omega = sqrt(Omega^2) / 2 the mode's clock rate, Delta_min from (c);
    reported for the mode as saved (a box mode) and restricted to r < r_0; the unit of K is the
    instrument's (K = 2 kin_tot omega); the author's "one unit" (hbar in program units) is not
    translated (author-gated, Q on the tracker).

usage: python3 m5_32_r17_0_record.py
out:   data/m5_32_r17_0_record.json, plots/m5_32_r17_0_tail.png, plots/m5_32_r17_0_core.png,
       checkpoints/m5_32_r17/r17_0_record.log
"""
from __future__ import annotations
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np

ARGS = list(sys.argv[1:])
sys.argv = [sys.argv[0]]
import m5_32_r16_common as C                              # noqa: E402
import m5_32_r16_0_fields as F0                           # noqa: E402

C15, INS4 = C.C15, C.INS4
ETA = C.ETA
RES, DATA, PLOTS = C.RES, C.DATA, C.PLOTS
CK16 = C.CK
CK = os.path.join(RES, "checkpoints", "m5_32_r17")
os.makedirs(CK, exist_ok=True)
T0 = time.time()
LOG = open(os.path.join(CK, "r17_0_record.log"), "a")
MU = 1e-2
DELTA = C.DELTA
A_REPORT_TAIL = 8.0 * (1.0 - DELTA) ** 4
U_REPORT_PAIR = 4.0 * (1.0 - DELTA) ** 4 / np.pi


def log(m):
    line = f"[{time.time() - T0:8.1f}s] {m}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


def rel(p):
    return os.path.relpath(p, RES)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(C.HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------ per-cell static densities (plain, the field's own frame)
def densities(M, cfg):
    """per-cell static densities of the v4 object (no h^3, no circle average): E_h, V4, U, KP, reg."""
    h = cfg["h"]
    fr = C.frame(M, C.radial_ref(cfg))
    Gm = fr["G"]
    comp = cfg["completion"]
    spl, _ = C15.split_cells(M, need_grad=False)
    rho2 = spl / 4.0
    v4, _ = C.v4_cells(fr, cfg, need_grad=False)
    Eh = np.zeros(M.shape[:-2])
    e2 = np.zeros_like(Eh)
    kp = np.zeros_like(Eh)
    for br, wt in INS4.branches(cfg["stencil"]):
        A = [INS4.d1(M, ax, h, br) for ax in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                d, _, _, _ = C.quartic_pair(A[i], A[j], Gm, comp, need_grad=False)
                Eh += wt * 4.0 * np.real(d)
            d2, _, _ = C.e2_cells(A[i], Gm, need_grad=False)
            e2 += wt * np.real(d2)
        Ek, _, _ = C.kp_cells(A, fr, need_grad=False)
        kp += wt * np.real(Ek)
    return {"E_h": Eh, "V4": np.real(v4), "U": cfg["mu"] * rho2, "KP": cfg["cP"] * kp, "reg": cfg["cs"] * rho2 * e2}, fr


def shell_stats(dens, r, h, L, lo=0.15, hi=0.42, width=None):
    """shell means of dens on [lo L, hi L) in shells of width (default 1.5 h); returns (r_mid, mean)."""
    w = width or 1.5 * h
    edges = np.arange(lo * L, hi * L + 1e-9, w)
    rs, ds, ns = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if np.sum(m) >= 8:
            rs.append(0.5 * (a + b)); ds.append(float(np.mean(dens[m]))); ns.append(int(np.sum(m)))
    return np.array(rs), np.array(ds), ns


def tail_fit(dens, r, h, L, lo=0.15, hi=0.42):
    rs, ds, ns = shell_stats(dens, r, h, L, lo, hi)
    ok = ds > 0
    out = {"shells_r": rs.tolist(), "shells_mean": ds.tolist(), "shells_ncells": ns}
    if np.sum(ok) >= 3:
        sl, ic = np.polyfit(np.log(rs[ok]), np.log(ds[ok]), 1)
        out["loglog_slope"] = float(sl)
        out["loglog_amplitude_at_r1"] = float(np.exp(ic))
    sel = (r >= max(9.0, lo * L)) & (r < hi * L)
    out["median_dens_r4"] = float(np.median(dens[sel] * r[sel] ** 4))
    out["mean_dens_r4_shells"] = float(np.mean(ds[ok] * rs[ok] ** 4)) if np.any(ok) else None
    out["A_over_report_tail"] = out["median_dens_r4"] / A_REPORT_TAIL
    return out


# ------------------------------------------------ (c) the split curvature of V4
def hV_closed_form():
    import sympy as sp
    lg, l1, m, x, W = sp.symbols("lg l1 m x W", real=True)
    cp = sp.symbols("C1:5", real=True)
    V = W * sum((lg ** p + l1 ** p + (m + x) ** p + (m - x) ** p - cp[p - 1]) ** 2 for p in range(1, 5))
    d2 = sp.diff(V, x, 2)
    f = sp.lambdify((lg, l1, m, x, W) + tuple(cp), d2, "numpy")
    return f, sp.srepr(sp.simplify(d2))[:0] or str(sp.expand(d2))[:400]


def core_reads(M, cfg, fr=None):
    n, h, L = cfg["n"], cfg["h"], cfg["L"]
    X, Y, Z = INS4.coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    free = ~INS4.pin_shell(n, h, 1.6)
    if fr is None:
        fr = C.frame(M, C.radial_ref(cfg))
    lg, l1, s, p = (np.real(fr[k]) for k in ("lg", "l1", "s", "p"))
    disc = np.sqrt(np.maximum(s * s - 4.0 * p, 0.0))
    hs, m = disc / 2.0, s / 2.0
    l2, l3 = m + hs, m - hs
    gap = l1 - l2
    cp = C15.cp_dd(cfg["g"], cfg["delta"])
    f, _ = hV_closed_form()
    hV = f(lg, l1, m, hs, C.W1, *cp)
    hV0 = f(lg, l1, m, 0.0 * hs, C.W1, *cp)
    j = n // 2
    line = slice(j, n)
    out = {"central_line": {"r": X[line, j, j].tolist(), "lambda_g": lg[line, j, j].tolist(), "lambda_1": l1[line, j, j].tolist(),
                            "lambda_2": l2[line, j, j].tolist(), "lambda_3": l3[line, j, j].tolist(), "gap": gap[line, j, j].tolist(),
                            "hV": hV[line, j, j].tolist(), "hV_over_W1": (hV[line, j, j] / C.W1).tolist()}}
    edges = np.arange(0.0, L / 2 + h, 1.5 * h)
    sh = []
    for a, b in zip(edges[:-1], edges[1:]):
        mk = (r >= a) & (r < b) & free
        if np.sum(mk) == 0:
            continue
        sh.append({"r": [float(a), float(b)], "n": int(np.sum(mk)), "hV_mean": float(np.mean(hV[mk])), "hV_min": float(np.min(hV[mk])), "hV_at_zero_split_mean": float(np.mean(hV0[mk])),
                   "gap_min": float(np.min(gap[mk])), "gap_mean": float(np.mean(gap[mk])), "l1_mean": float(np.mean(l1[mk])), "half_split_max": float(np.max(hs[mk]))})
    out["shells"] = sh
    out["hV_min_free"] = float(np.min(hV[free])); out["hV_min_free_over_W1"] = float(np.min(hV[free]) / C.W1)
    out["hV_min_central_line"] = float(np.min(hV[line, j, j])); out["hV_min_central_line_over_W1"] = float(np.min(hV[line, j, j]) / C.W1)
    out["hV_at_zero_split_min_free"] = float(np.min(hV0[free]))
    out["r_at_hV_min"] = float(r[free].reshape(-1)[int(np.argmin(hV[free]))])
    out["Delta_min_free"] = float(np.min(gap[free]))
    out["r_at_Delta_min"] = float(r[free].reshape(-1)[int(np.argmin(gap[free]))])
    out["Delta_min_central_line"] = float(np.min(gap[line, j, j]))
    dom = C.domain(fr, cfg)
    out["domain"] = dom
    inpl = (l1 < C.PL_HI) & free
    out["r_0_taper (max r with lambda_1 < 0.8)"] = float(np.max(r[inpl])) if np.any(inpl) else 0.0
    inh = (l1 < 0.5 * (1.0 + DELTA)) & free
    out["r_0_half (max r with lambda_1 < (1+delta)/2)"] = float(np.max(r[inh])) if np.any(inh) else 0.0
    # the profile-based r_0: where the shell-mean lambda_1 crosses 0.8 by linear interpolation
    rr = np.array([0.5 * (s_["r"][0] + s_["r"][1]) for s_ in sh]); ll = np.array([s_["l1_mean"] for s_ in sh])
    cross = np.where((ll[:-1] < 0.8) & (ll[1:] >= 0.8))[0]
    out["r_0_profile (shell-mean lambda_1 = 0.8)"] = float(rr[cross[0]] + (0.8 - ll[cross[0]]) * (rr[cross[0] + 1] - rr[cross[0]]) / (ll[cross[0] + 1] - ll[cross[0]])) if len(cross) else 0.0
    for k in ("r_0_taper (max r with lambda_1 < 0.8)", "r_0_half (max r with lambda_1 < (1+delta)/2)", "r_0_profile (shell-mean lambda_1 = 0.8)"):
        out[k + " x sqrt(mu)"] = out[k] * np.sqrt(MU)
    out["Delta_min_free_ge_report_gate_note"] = "the report's 27.3 gate: nu / kappa < 2 Delta_min^2; v6 (nu, kappa) = (1e-2, 0.4): nu / kappa = 0.025"
    out["v6_gate_nu_over_kappa_0.025_lt_2_Delta_min2"] = bool(0.025 < 2.0 * out["Delta_min_free"] ** 2)
    out["s_star_v6_0.112_lt_Delta_min"] = bool(0.112 < out["Delta_min_free"])
    return out, hV, gap, l1, hs


# ------------------------------------------------ (d) spin-2 shells
def spin2_shells(M, cfg, shells=((0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 12.0), (12.0, 15.0))):
    n, h = cfg["n"], cfg["h"]
    X, Y, Z = INS4.coords(n, h)
    zeta, gap, th, ph, r = F0.frame_zeta(M, X, Y, Z)
    out = {}
    for a, b in shells:
        m_ = (r >= a) & (r < b)
        if np.sum(m_) < 8:
            continue
        c = F0.shell_decomp(zeta[m_], th[m_], ph[m_])
        P = {mm: abs(c[mm]) ** 2 for mm in c}
        tot = max(sum(P.values()), 1e-300)
        ztot = float(4 * np.pi / zeta[m_].size * np.sum(np.abs(zeta[m_]) ** 2))
        out[f"[{a:g},{b:g})"] = {"n_cells": int(np.sum(m_)), "P_m": {str(mm): float(P[mm]) for mm in sorted(P)}, "mean_m": float(sum(mm * P[mm] for mm in P) / tot),
                                 "l2_fraction": float(tot / max(ztot, 1e-300)), "chirality": float((P[2] - P[-2]) / max(P[2] + P[-2], 1e-300)),
                                 "mean_abs_zeta": float(np.mean(np.abs(zeta[m_]))), "min_director_gap": float(np.min(gap[m_]))}
    return out


# ------------------------------------------------ main
def main():
    out = {"rung": "R17-0 record", "report_numbers_as_claims": {"tail_amplitude_8(1-delta)^4": A_REPORT_TAIL, "pair_coefficient_4(1-delta)^4/pi": U_REPORT_PAIR,
                                                                    "ratio_pair_over_tail": U_REPORT_PAIR / A_REPORT_TAIL, "superposition_8piA": 8.0 * np.pi * A_REPORT_TAIL,
                                                                    "factor_superposition_over_report": 8.0 * np.pi * A_REPORT_TAIL / U_REPORT_PAIR, "16pi2": 16.0 * np.pi ** 2}}
    # ---------------- (a) the tail on our fields
    log("(a) the tail amplitude on the R16-1 end fields and the analytic seed")
    fields = {"analytic_seed_n32_L48": (None, 32, 48.0), "analytic_seed_n64_L48": (None, 64, 48.0),
              "r16_1_end_n32_L48": (os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"), 32, 48.0),
              "r16_1_end_n48_L72": (os.path.join(CK16, "r16_1_rebuild_n48_L72.npy"), 48, 72.0),
              "r16_1_end_n64_L48": (os.path.join(CK16, "r16_1_rebuild_n64_L48_analytic.npy"), 64, 48.0)}
    tails, cores, spin2 = {}, {}, {}
    dens_keep = {}
    for lab, (p, n, L) in fields.items():
        cfg = C.cfg_v4(n, L, n_samples=8)
        M = C15.seed_uniaxial(cfg) if p is None else np.load(p)
        X, Y, Z = INS4.coords(n, cfg["h"])
        r = np.sqrt(X * X + Y * Y + Z * Z)
        dn, fr = densities(M, cfg)
        t = tail_fit(dn["E_h"], r, cfg["h"], L)
        t["source"] = rel(p) if p else "C15.seed_uniaxial (the analytic radial hedgehog of the degenerate vacuum)"
        t["other_terms_median_r4_on_tail"] = {k: float(np.median((dn[k] * r ** 4)[(r >= 9.0) & (r < 0.42 * L)])) for k in ("V4", "U", "KP", "reg")}
        t["h"] = cfg["h"]
        tails[lab] = t
        dens_keep[lab] = (r, dn["E_h"], L)
        log(f"  {lab}: E_h tail slope {t.get('loglog_slope', float('nan')):.3f}, median E_h r^4 {t['median_dens_r4']:.5f} (report 8(1-delta)^4 = {A_REPORT_TAIL:.5f}, ratio {t['A_over_report_tail']:.4f}); K_P r^4 {t['other_terms_median_r4_on_tail']['KP']:.4f}")
        if p is not None:
            cr, hV, gap, l1, hs = core_reads(M, cfg, fr)
            cores[lab] = cr
            log(f"  {lab}: hV min (free) {cr['hV_min_free']:.3e} (/W1 {cr['hV_min_free_over_W1']:.4f}) at r {cr['r_at_hV_min']:.2f}; central-line min {cr['hV_min_central_line']:.3e} (/W1 {cr['hV_min_central_line_over_W1']:.4f}); "
                f"Delta_min {cr['Delta_min_free']:.4f} (line {cr['Delta_min_central_line']:.4f}) at r {cr['r_at_Delta_min']:.2f}; r_0 taper {cr['r_0_taper (max r with lambda_1 < 0.8)']:.2f} half {cr['r_0_half (max r with lambda_1 < (1+delta)/2)']:.2f} profile {cr['r_0_profile (shell-mean lambda_1 = 0.8)']:.2f}; r_0 sqrt(mu) {cr['r_0_profile (shell-mean lambda_1 = 0.8) x sqrt(mu)']:.3f}")
            spin2[lab] = spin2_shells(M, cfg)
            log(f"  {lab}: spin-2 shells " + "; ".join(f"{k}: <m> {v['mean_m']:+.2f} l2frac {v['l2_fraction']:.2f} |zeta| {v['mean_abs_zeta']:.1e}" for k, v in spin2[lab].items()))
        json.dump({"tails": tails, "cores": cores, "spin2": spin2}, open(os.path.join(CK, "r17_0_record_partial.json"), "w"), indent=1, default=float)
    out["a_tail_on_our_fields"] = tails
    out["c_core_reads_r16_1"] = cores
    # the R16-3 end states: core reads + spin-2 (d, and Delta_min at convergence for the rotating cores)
    r16_3 = {}
    for p in sorted(glob.glob(os.path.join(CK16, "r16_3_rebuild_*.npy"))):
        tag = os.path.basename(p)[:-4]
        n = int(tag.split("_n")[1].split("_")[0]); L = float(tag.split("_L")[1].split("_")[0])
        cfg = C.cfg_v4(n, L, n_samples=8)
        M = np.load(p)
        cr, hV, gap, l1, hs = core_reads(M, cfg)
        cr.pop("central_line", None)
        r16_3[tag] = {"core": cr, "spin2": spin2_shells(M, cfg), "field": rel(p)}
        log(f"  {tag}: Delta_min {cr['Delta_min_free']:.4f} at r {cr['r_at_Delta_min']:.2f}, half split max {max(s_['half_split_max'] for s_ in cr['shells']):.4f}, r_0 profile {cr['r_0_profile (shell-mean lambda_1 = 0.8)']:.2f}, hV min/W1 {cr['hV_min_free_over_W1']:.4f}; "
            + "; ".join(f"{k}: <m> {v['mean_m']:+.2f}" for k, v in r16_3[tag]["spin2"].items()))
    out["d_r16_3_end_states"] = r16_3
    out["d_r16_1_spin2"] = spin2
    json.dump(out, open(os.path.join(CK, "r17_0_record_partial.json"), "w"), indent=1, default=float)
    # ---------------- (a) the record's own stack: the R3 single and pairs on the certified 4x4 stack (g 32, vacuum (1, delta, 0))
    log("(a) the record in one unit system: the certified stack's single-hedgehog tail and its pair energies (R3 undressed rows)")
    r3 = json.load(open(os.path.join(DATA, "m5_32_r3_pair.json")))
    rows = {row["tag"]: row for row in r3["rows"]}
    B3 = INS4
    single = np.load(os.path.join(DATA, "m5_32_r3_ii", "lam0_un0_single_d0_n32.npz"))
    Ms = single[single.files[0]].astype(float)
    cfg3 = B3.base_cfg(n=32, L=48.0, s=-1.0, g=32.0)
    # the certified curvature density 4 sum_{i<j} |[A_i, A_j]_eta|^2 per cell (the record's E_u), sym stencil
    h = cfg3["h"]
    X, Y, Z = INS4.coords(32, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    eu = np.zeros((32, 32, 32))
    for br, wt in INS4.branches("sym"):
        A = [INS4.d1(Ms, ax, h, br) for ax in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                Cm = A[i] @ ETA @ A[j] - A[j] @ ETA @ A[i]
                eu += wt * 4.0 * np.einsum("a,b,...ab,...ab->...", np.diag(ETA), np.diag(ETA), Cm, Cm)
    # E_u per the certified stack (B3.e_parts) as the cross-check of the density's normalization
    e_parts = B3.e_parts(Ms, cfg3)
    t3 = tail_fit(eu, r, h, 48.0)
    t3["E_u_from_density_h3"] = float(h ** 3 * np.sum(eu)); t3["E_u_certified_e_parts"] = float(e_parts[0]); t3["E_total_certified"] = float(sum(e_parts))
    t3["source"] = "data/m5_32_r3_ii/lam0_un0_single_d0_n32.npz (the R3 undressed single, certified 4x4 stack, g 32, vacuum (1, delta, 0), 1500 accepted FIRE steps)"
    t3["vacuum_spatial_spectrum"] = [1.0, DELTA, 0.0]
    log(f"  R3 single: E_u density h^3 sum {t3['E_u_from_density_h3']:.4f} vs certified e_parts {t3['E_u_certified_e_parts']:.4f}; tail slope {t3.get('loglog_slope', float('nan')):.3f}, median E_u r^4 = A_R3 {t3['median_dens_r4']:.4f}")
    A_R3 = t3["median_dens_r4"]
    pairs = {}
    Es = rows["lam0_un0_single_d0_n32"]["E_end"] if "E_end" in rows["lam0_un0_single_d0_n32"] else None
    tab = r3["results"]["tables"]["n32"]
    for kind in ("same", "anti"):
        ds, Ei = [], []
        for d in (10.0, 14.0, 18.0, 24.0):
            k = f"lam0_{kind}_d{d:g}_n32"
            if k in tab:
                ds.append(d); Ei.append(tab[k]["Eint_static_undressed"])
        ds, Ei = np.array(ds), np.array(Ei)
        fit = {}
        if len(ds) >= 3:
            Am = np.stack([np.ones_like(ds), 1.0 / ds], -1)
            c, *_ = np.linalg.lstsq(Am, Ei, rcond=None)
            fit["A_plus_B_over_d"] = {"A": float(c[0]), "B": float(c[1]), "max_rel_resid": float(np.max(np.abs(Am @ c - Ei)) / np.max(np.abs(Ei)))}
            if np.all(Ei > 0) or np.all(Ei < 0):
                fit["loglog_slope_abs_Eint"] = float(np.polyfit(np.log(ds), np.log(np.abs(Ei)), 1)[0])
            fit["dEint_dd_outer"] = float((Ei[-1] - Ei[-2]) / (ds[-1] - ds[-2]))
            fit["monotone"] = "increasing" if np.all(np.diff(Ei) > 0) else ("decreasing" if np.all(np.diff(Ei) < 0) else "non-monotone")
        pairs[kind] = {"d": ds.tolist(), "E_int": Ei.tolist(), "fit": fit, "superposition_8piA_over_d": (8.0 * np.pi * A_R3 / ds).tolist(),
                       "report_pair_over_d_in_these_units_if_A_were_the_report_tail": (U_REPORT_PAIR * (A_R3 / A_REPORT_TAIL) / ds).tolist()}
        log(f"  R3 {kind}: E_int {np.round(Ei, 3).tolist()} at d {ds.tolist()}; fit {fit}; superposition 8 pi A_R3 / d = {np.round(8 * np.pi * A_R3 / ds, 2).tolist()}")
    # like-charge: a certified 1/d coefficient requires a positive E_int FALLING as 1/d; the record's rises (the string form)
    same = pairs["same"]
    certified = bool(same["fit"].get("monotone") == "decreasing" and same["fit"].get("loglog_slope_abs_Eint", 0) < -0.5)
    pairs["verdict"] = "PAIR_LAW_CERTIFIED" if certified else "PAIR_LAW_NOT_CERTIFIED"
    pairs["reason"] = ("the like-charge E_int(d) on the record rises with d (a string form, M5.21.4's verdict), so no 1/d coefficient is certified for like charges; "
                       "the anti-pair E_int is negative and steeper than 1/d at reachable d (near-field)")
    # M5.21.4 3x3 rows in their own units
    m214 = {}
    for it, f in ((120, "m5_21_4_ladder_it120.json"), (400, "m5_21_4_ladder_it400.json"), (1500, "m5_21_4_ladder.json")):
        d = json.load(open(os.path.join(DATA, f)))
        rr = {(row["kind"], row["d"]): row["E"] for row in d["rows"]}
        m214[f"it{it}"] = {"E_single": d["E_single"], "c2_selfcal": d["c2_selfcal"], "A_tail = 8 c2": 8.0 * d["c2_selfcal"], "64pi_c2 = 8 pi A": d["coulomb_pred_coeff_64pi_c2"],
                           "E_int_same": {str(k[1]): rr[k] - 2 * d["E_single"] for k in rr if k[0] == "same"}, "E_int_anti": {str(k[1]): rr[k] - 2 * d["E_single"] for k in rr if k[0] == "anti"}}
    out["a_record_pairs"] = {"R3_certified_stack": {"single_tail": t3, "pairs": pairs}, "M5_21_4_3x3": m214,
                             "unit_note": ("the record's tail and pairs live on the (1, delta, 0) vacuum at g 32 (the certified stack); the report's 1.92080 / 0.30570 are for the degenerate (1, delta, delta) vacuum of R16, "
                                           "on which no pair has been relaxed; the numbers are compared through the ratio pair / tail, which the derivation fixes")}
    json.dump(out, open(os.path.join(CK, "r17_0_record_partial.json"), "w"), indent=1, default=float)
    # ---------------- (b) request 3 from the record
    log("(b) request 3: the dressed-pair record (relaxed and imposed)")
    b = {"R3_ii_relaxed_dressed_pairs": {}, "R14_C_relaxed_RG_pairs": {}, "R11_imposed_notebook_pair": {}, "R3_i_imposed_ansatz": {}}
    for lam in ("0", "0.75", "1"):
        rowsl = {}
        for kind in ("same", "anti"):
            for d in (10.0, 14.0, 18.0, 24.0):
                k = f"lam{lam}_{kind}_d{d:g}_n32"
                if k in tab:
                    tt = tab[k]
                    rowsl[k] = {"E_int_dressed_total": tt.get("Eint_dressed_total"), "E_int_static_undressed": tt.get("Eint_static_undressed"), "dressed_verdict": tt.get("dressed_verdict"),
                                "amp_trend": tt.get("amp_trend"), "amp_end_over_seed_grid": (tt["amp_end"]["grid_max_norm_M0i"] / tt["amp_seed"]["grid_max_norm_M0i"]) if tt.get("amp_end") and tt.get("amp_seed") else None}
        # the sign: dE_int/dd on the outer window of the dressed total
        for kind in ("same", "anti"):
            ks = [f"lam{lam}_{kind}_d{d:g}_n32" for d in (18.0, 24.0)]
            if all(k in tab for k in ks):
                dE = tab[ks[1]]["Eint_dressed_total"] - tab[ks[0]]["Eint_dressed_total"]
                rowsl[f"sign_{kind}_dressed_outer"] = "REPULSIVE (E_int falls with d)" if dE < 0 else "ATTRACTIVE (E_int rises with d)"
        b["R3_ii_relaxed_dressed_pairs"][f"lambda_{lam}"] = rowsl
    b["R3_ii_relaxed_dressed_pairs"]["fits_recorded"] = r3["results"].get("fits", {}).get("n32", {})
    b["R3_ii_relaxed_dressed_pairs"]["protocol"] = "the relaxed heal with the boost amplitudes held (amp_trend 'held' = the seed amplitude within 0.03 percent), 1500 accepted FIRE steps, verdict FALLING at the budget on every dressed row"
    r14 = json.load(open(os.path.join(DATA, "m5_32_r14_c_newton.json")))
    b["R14_C_relaxed_RG_pairs"] = r14["arms"]["R_G_relaxed_pairs"]["reads"]
    r11 = json.load(open(os.path.join(DATA, "m5_32_r11_samesign.json")))
    b["R11_imposed_notebook_pair"] = {k: r11["arm_b"][k] for k in r11["arm_b"] if k in ("certified_sign", "flip_sign", "pair_reads", "pairs", "boxes_summary")}
    if "boxes" in r11["arm_b"]:
        b["R11_imposed_notebook_pair"]["boxes_n"] = [bx.get("n") for bx in r11["arm_b"]["boxes"]]
    ans = json.load(open(os.path.join(DATA, "m5_32_r3_ansatz.json")))
    ctrl = ans.get("controls", {}).get("a_calibration", {}).get("rows", {})
    b["R3_i_imposed_ansatz"] = {k: {kk: v[kk] for kk in v if kk in ("dEdd_sign_outer", "force_read", "B_outer_2term")} for k, v in ctrl.items()}
    b["statement"] = ("request 3 (relax a dressed pair and read the force from the relaxed energy) was run BOTH ways on the record: the ansatz imposed (R3 i, R11, the R14-C ansatz arm) and the relaxed heal with the "
                      "amplitudes held (R3 ii, R14-C); the relaxed heals never converged at the budget (FALLING), the sign of dE_int/dd is the same in both protocols on the certified sector (repulsive for like charges)")
    out["b_request_3_record"] = b
    log("  " + b["statement"])
    # ---------------- (h) K_coll on the saved R16-2 mode
    log("(h) K_coll on the R16-2 lowest doublet mode (n32 core)")
    r162 = json.load(open(os.path.join(DATA, "m5_32_r16_2.json")))
    mode = np.load(os.path.join(CK16, "r16_2_r16_1_end_n32_mode0.npy"))
    cfg = C.cfg_v4(32, 48.0)
    X, Y, Z = INS4.coords(32, cfg["h"])
    r = np.sqrt(X * X + Y * Y + Z * Z)
    amp = np.sqrt(mode[..., 0] ** 2 + mode[..., 1] ** 2)
    f = amp / np.max(amp)
    Om2 = r162["runs"]["r16_1_end_n32"]["modes"][0]["Omega2"]
    om = np.sqrt(Om2) / 2.0
    Dm = cores["r16_1_end_n32_L48"]["Delta_min_free"]
    r0 = cores["r16_1_end_n32_L48"]["r_0_profile (shell-mean lambda_1 = 0.8)"]
    h3 = cfg["h"] ** 3
    intf2 = float(h3 * np.sum(f ** 2))
    core_m = r < max(r0, 3.0)
    fcore = amp * core_m / max(np.max(amp[core_m]), 1e-300)
    intf2_core = float(h3 * np.sum(fcore ** 2))
    hh = {"mode": rel(os.path.join(CK16, "r16_2_r16_1_end_n32_mode0.npy")), "Omega2": Om2, "omega_clock": om, "Delta_min": Dm, "r_0_profile": r0, "int_f2_box_mode": intf2, "K_coll_box_mode": 2 * om * Dm ** 2 * intf2,
          "f_peak_radius": float(r.reshape(-1)[int(np.argmax(amp))]), "weight_fraction_r_lt_r0": float(np.sum(amp[core_m] ** 2) / np.sum(amp ** 2)),
          "int_f2_core_restricted": intf2_core, "K_coll_core_restricted": 2 * om * Dm ** 2 * intf2_core, "r_0_sqrt_mu": r0 * np.sqrt(MU), "r_0_sqrt_mu_ge_0.5": bool(r0 * np.sqrt(MU) >= 0.5),
          "unit_note": "K in the instrument's units (K = 2 kin_tot omega, the fixed-K functional E_K = E_stat + K^2 / (4 kin_tot)); the author's 'one unit' (hbar in program units) is author-gated and not translated here; the saved lowest mode is a BOX mode (R16-2: T-weighted rms radius 16.6, 2.7 percent of the weight inside r < 8), so the box-mode K_coll measures the box, not the core"}
    out["h_K_coll"] = hh
    log(f"  K_coll (box mode as saved) {hh['K_coll_box_mode']:.4f} (int f^2 {intf2:.1f}); core-restricted {hh['K_coll_core_restricted']:.5f} (int f^2 {intf2_core:.2f}, weight fraction inside r_0 {hh['weight_fraction_r_lt_r0']:.4f}); omega {om:.4f}, Delta_min {Dm:.4f}, r_0 sqrt(mu) {hh['r_0_sqrt_mu']:.3f}")
    # ---------------- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for lab, (r_, eh, L) in dens_keep.items():
        rs, ds, _ = shell_stats(eh, r_, tails[lab]["h"], L, 0.08, 0.45)
        ax[0].loglog(rs, ds, "o-", ms=3, label=f"{lab} (slope {tails[lab].get('loglog_slope', float('nan')):.2f})")
    rr = np.linspace(4, 30, 50)
    ax[0].loglog(rr, A_REPORT_TAIL / rr ** 4, "k--", lw=0.8, label="8(1-delta)^4 / r^4 = 1.9208 / r^4")
    ax[0].set_xlabel("r"); ax[0].set_ylabel("shell-mean E_h density"); ax[0].legend(fontsize=5); ax[0].set_title("(a) the r^-4 tail of the certified quartic on our fields", fontsize=8)
    for kind, mk in (("same", "s"), ("anti", "^")):
        pr = pairs[kind]
        ax[1].plot(pr["d"], pr["E_int"], mk + "-", label=f"R3 record {kind} E_int (undressed, certified stack)")
    ds_ = np.linspace(8, 26, 40)
    ax[1].plot(ds_, 8 * np.pi * A_R3 / ds_, "k--", lw=0.8, label=f"superposition 8 pi A_R3 / d, A_R3 {A_R3:.2f}")
    ax[1].plot(ds_, U_REPORT_PAIR * (A_R3 / A_REPORT_TAIL) / ds_, "r:", lw=0.8, label="the report's pair / tail ratio applied to A_R3")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].set_xlabel("d"); ax[1].set_ylabel("E_int"); ax[1].legend(fontsize=5); ax[1].set_title(f"(a) the pair record: {pairs['verdict']}", fontsize=8)
    for lab, cr in cores.items():
        cl = cr["central_line"]
        ax[2].plot(cl["r"], np.array(cl["hV_over_W1"]), "o-", ms=2, label=f"{lab}: hV / W1 on the central line")
    ax[2].axhline(0, color="k", lw=0.5); ax[2].set_xlabel("r"); ax[2].set_ylabel("d^2 V4 / ds^2 / W1"); ax[2].legend(fontsize=5); ax[2].set_xlim(0, 12); ax[2].set_title("(c) the split curvature of V4 on the full fields", fontsize=8)
    fig.savefig(os.path.join(PLOTS, "m5_32_r17_0_tail.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for lab, cr in cores.items():
        sh = cr["shells"]
        rm = [0.5 * (s_["r"][0] + s_["r"][1]) for s_ in sh]
        ax[0].plot(rm, [s_["gap_min"] for s_ in sh], "o-", ms=2, label=f"{lab} shell-min gap (Delta_min {cr['Delta_min_free']:.3f})")
        ax[1].plot(rm, [s_["l1_mean"] for s_ in sh], "o-", ms=2, label=f"{lab} (r_0 {cr['r_0_profile (shell-mean lambda_1 = 0.8)']:.2f})")
    ax[0].axhline(0.112, color="r", ls=":", lw=0.8, label="v6 s* = 0.112"); ax[0].set_xlim(0, 12); ax[0].set_xlabel("r"); ax[0].set_ylabel("Delta = lambda_1 - lambda_2"); ax[0].legend(fontsize=5); ax[0].set_title("(c) the director gap", fontsize=8)
    ax[1].axhline(0.8, color="r", ls=":", lw=0.8, label="plateau edge 0.8"); ax[1].set_xlim(0, 12); ax[1].set_xlabel("r"); ax[1].set_ylabel("shell-mean lambda_1"); ax[1].legend(fontsize=5); ax[1].set_title("(c) the core radius r_0", fontsize=8)
    for lab in list(spin2.keys())[:3]:
        keys = list(spin2[lab].keys())
        ax[2].plot(range(len(keys)), [spin2[lab][k]["mean_m"] for k in keys], "o-", ms=3, label=lab)
    for tag in list(r16_3.keys()):
        keys = list(r16_3[tag]["spin2"].keys())
        ax[2].plot(range(len(keys)), [r16_3[tag]["spin2"][k]["mean_m"] for k in keys], "x--", ms=3, label=tag)
    ax[2].set_xticks(range(5)); ax[2].set_xticklabels(["[0,3)", "[3,6)", "[6,9)", "[9,12)", "[12,15)"], fontsize=7); ax[2].set_ylabel("<m> of the spin-2 shell content"); ax[2].legend(fontsize=5); ax[2].set_title("(d) request (ii): the shell angular index", fontsize=8)
    fig.savefig(os.path.join(PLOTS, "m5_32_r17_0_core.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    out["plots"] = ["plots/m5_32_r17_0_tail.png", "plots/m5_32_r17_0_core.png"]
    out["wall_s"] = time.time() - T0
    json.dump(out, open(os.path.join(DATA, "m5_32_r17_0_record.json"), "w"), indent=1, default=float)
    log(f"written data/m5_32_r17_0_record.json ({out['wall_s']:.0f} s)")


if __name__ == "__main__":
    main()
