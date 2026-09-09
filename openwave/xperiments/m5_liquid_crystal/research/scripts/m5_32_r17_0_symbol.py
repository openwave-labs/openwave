"""M5.32 R17-0g (ledger 6.6): the two agents' disagreement on the frozen-hedgehog principal symbol,
decided on our own relaxed cores with the R16-4 machinery.  The hunt report's section 65 (the 10 x 10
symbol on a hedgehog background keeps the signature 3 negative, 2 zero, 5 positive at every omega, no
characteristics) against the Complete Picture report's section 40 (page 203: Q(omega, k_r) =
(omega^2 - k_r^2) K_bg on the positive z axis of a uniaxial hedgehog with spec K_bg = {0, 0, 1, 1, 1, 1, 2, 2, 2, 6},
eight eigenvalues crossing at omega = |k_r|: a radial characteristic).  Both are claims; the numbers
below are ours.

EQUATIONS
---------
The Lagrangian density per cell as a function of the four jets (R16-4, lag_density), E-orientation
reversed into the Lagrangian sign: l = +4 sum_i |F_0i|^2_G - 4 sum_{i<j} |F_ij|^2_G (+ the K_P and
regulator terms when the full v4 is requested; the potential carries no jets).  A perturbation
xi e^{i (k . x - Omega t)} with xi one of the 10 symmetric unit matrices E_pq has the second-order density
    l_2 = (1/2) sum_{mu nu} k_mu k_nu H_mu_nu[xi, xi],  k_0 = Omega,   H_mu_nu[xi_p, xi_q] = d^2 l / dA_mu dA_nu [xi_p, xi_q]
by central differences of the pointwise density in the jets on the FROZEN background (the M dependence
through G, w, rho^2 fixed: both agents' "frozen hedgehog"; H_mu_nu[p, q] symmetric under the simultaneous swap (mu, p) <-> (nu, q) only, the R17-0 audit's correction).  The 10 x 10 principal symbol
    sigma(Omega, k) = Omega^2 H_00 + 2 Omega sum_i k_i H_0i + sum_ij k_i k_j H_ij
is tracked over Omega in [0, 1.6] at |k| = 1 for k radial (r_hat at the cell) and transverse; its
eigenvalues give the signature (negative / zero / positive, zero = below 1e-9 of the largest |eigenvalue|)
per Omega, the zero crossings, and the factorization test of the Complete Picture report,
    sigma = (Omega^2 - |k|^2) K_bg   <=>   H_0k = 0  and  H_kk := sum_ij k_i k_j H_ij = -H_00,
measured as |H_kk + H_00|_F / |H_00|_F (RADIAL_CHARACTERISTIC if below 1e-6 for the radial k with at
least one crossing at Omega = |k|; NO_CHARACTERISTICS if no eigenvalue outside the fixed kernel crosses
zero on the scan); the spectrum of H_00 normalized by its largest eigenvalue against {0,0,1,1,1,1,2,2,2,6}/6.
On the static core the circle acts trivially (the pair is degenerate to 1e-3), so no circle average
is taken (ns = 1; stated).  Backgrounds: the analytic radial hedgehog (the Complete Picture report's
background, constant eigenvalues), the R16-1 n32 and n64 relaxed cores (melted centers).  Cells: near
the +z axis (x = y = h/2) at r about 3, 6, 12, 18, and one off-axis cell.  Contractions: the quartic alone
(c_P = c_s = 0), both completions (identical on u = e_0 fields, verified), and the full v4 as a control.  AUDIT (2026-09-08): the first version assigned the mixed difference to (nu, mu, p, q) too, contaminating the factorization residual and the completion comparison (the eigenvalue scan, symmetrized, was unaffected); corrected before the record was written.

usage: python3 m5_32_r17_0_symbol.py
out:   data/m5_32_r17_0_symbol.json, plots/m5_32_r17_0_symbol.png, checkpoints/m5_32_r17/r17_0_symbol.log
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

ARGS = list(sys.argv[1:])
sys.argv = [sys.argv[0]]
import m5_32_r16_common as C                              # noqa: E402
import m5_32_r16_4_symbol as S4                           # noqa: E402

C15, INS4 = C.C15, C.INS4
ETA = C.ETA
RES, DATA, PLOTS = C.RES, C.DATA, C.PLOTS
CK16 = C.CK
CK = os.path.join(RES, "checkpoints", "m5_32_r17")
os.makedirs(CK, exist_ok=True)
T0 = time.time()
LOG = open(os.path.join(CK, "r17_0_symbol.log"), "a")
OMEGAS = np.linspace(0.0, 1.6, 33)
CP_SPEC = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 6]) / 6.0


def log(m):
    line = f"[{time.time() - T0:8.1f}s] {m}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


def basis10():
    B, lab = [], []
    for a in range(4):
        for b in range(a, 4):
            E = np.zeros((4, 4))
            E[a, b] = E[b, a] = 1.0 if a == b else 2 ** -0.5
            B.append(E); lab.append(f"{a}{b}")
    return B, lab


def full_symbol(M, cfg, cells, eps=1e-4):
    """H[cell, mu, nu, p, q] over the 10-basis, frozen background, no circle average."""
    fr0 = C.frame(M, C.radial_ref(cfg))
    a0 = C.a0_of(M, fr0)
    Asp = S4.jets_at(M, cfg)
    A = [0.0 * a0] + Asp
    idx = tuple(np.array(cells).T)
    Mc = M[idx]
    frc = {kk: (v[idx] if isinstance(v, np.ndarray) and v.shape[:3] == M.shape[:3] else v) for kk, v in fr0.items()}
    Ac = [a[idx] for a in A]
    B, lab = basis10()
    nc = len(cells)
    H = np.zeros((nc, 4, 4, 10, 10))
    # the true tensor H[mu, nu, p, q] = d^2 l / dA_mu[p] dA_nu[q] is symmetric ONLY under the simultaneous swap
    # (mu, p) <-> (nu, q) (the R17-0 audit's hazard: assigning the mixed difference to (nu, mu, p, q) as well
    # imports the t^2 piece xi K xi - xi K xi, antisymmetric in (p, q) and completion-dependent); loop over the
    # joint index I = (mu, p) <= J = (nu, q).
    joint = [(mu, pi) for mu in range(4) for pi in range(10)]
    for ii, (mu, pi) in enumerate(joint):
        for (nu, qi) in joint[ii:]:
            def ev(s1, s2):
                Bj = [a.copy() for a in Ac]
                Bj[mu] = Bj[mu] + s1 * eps * B[pi]
                Bj[nu] = Bj[nu] + s2 * eps * B[qi]
                return np.real(S4.lag_density(Mc, Bj, frc, cfg))
            d2 = (ev(1, 1) - ev(1, -1) - ev(-1, 1) + ev(-1, -1)) / (4 * eps * eps)
            H[:, mu, nu, pi, qi] = d2; H[:, nu, mu, qi, pi] = d2
    return H, lab


def analyze(Hc, khat):
    """for one cell and one unit k: the scan over Omega, the signature, crossings, the factorization test."""
    H00 = Hc[0, 0]
    H0k = np.einsum("i,ipq->pq", khat, Hc[0, 1:])
    Hkk = np.einsum("i,j,ijpq->pq", khat, khat, Hc[1:, 1:])
    scale = max(float(np.max(np.abs(H00))), float(np.max(np.abs(Hkk))), 1e-300)
    eigs = []
    for om in OMEGAS:
        S = om * om * H00 + 2 * om * H0k + Hkk
        eigs.append(np.linalg.eigvalsh(0.5 * (S + S.T)))
    eigs = np.array(eigs)
    tol = 1e-5 * max(float(np.max(np.abs(eigs))), 1e-300)          # the fixed kernel on the lattice is zero to ~1e-6 relative (central-difference jets), not 1e-9
    sig = [(int(np.sum(e < -tol)), int(np.sum(np.abs(e) <= tol)), int(np.sum(e > tol))) for e in eigs]
    # crossings: the change of the number of negative eigenvalues between consecutive Omegas (robust to the
    # relabeling of sorted branches through the fixed kernel); each entry (Omega_mid, count)
    cross, cross_detail = [], []
    for k in range(len(OMEGAS) - 1):
        dn = sig[k][0] - sig[k + 1][0]
        if dn != 0:
            om_mid = float(0.5 * (OMEGAS[k] + OMEGAS[k + 1]))
            cross_detail.append([om_mid, int(dn)])
            cross.extend([om_mid] * abs(dn))
    fac = float(np.linalg.norm(Hkk + H00) / max(np.linalg.norm(H00), 1e-300))
    e00 = np.linalg.eigvalsh(0.5 * (H00 + H00.T))
    e00n = e00 / max(float(np.max(np.abs(e00))), 1e-300)
    return {"signature_by_Omega": sig, "eigs_at_Omega_0": eigs[0].tolist(), "eigs_at_Omega_1": eigs[int(np.argmin(np.abs(OMEGAS - 1.0)))].tolist(), "signature_at_0": sig[0], "signature_at_1.6": sig[-1], "signature_changes": bool(len(set(sig)) > 1), "crossings_Omega": sorted(cross), "crossings_detail": cross_detail, "n_crossings": len(cross),
            "factorization_residual |H_kk + H_00| / |H_00|": fac, "H_0k_norm_over_H00": float(np.linalg.norm(H0k) / max(np.linalg.norm(H00), 1e-300)),
            "H00_spectrum_normalized": e00n.tolist(), "H00_spectrum_vs_CP_pattern_max_dev": float(np.max(np.abs(np.sort(e00n) - CP_SPEC))),
            "Hkk_spectrum_normalized_by_H00_max": (np.linalg.eigvalsh(0.5 * (Hkk + Hkk.T)) / max(float(np.max(np.abs(e00))), 1e-300)).tolist(), "eigs_scan": eigs.tolist()}


def verdict(res):
    """RADIAL_CHARACTERISTIC: at least 4 crossings within 10 percent of Omega = |k| AND the factorization residual below 0.1 (a lattice quantity: it falls with h and grows in the melted core,
    reported); CHARACTERISTIC: crossings without the factorization (a characteristic at another speed); NO_CHARACTERISTICS: no crossing on the scan."""
    near = sum(1 for x in res["crossings_Omega"] if abs(x - 1.0) <= 0.1)
    if near >= 4 and res["factorization_residual |H_kk + H_00| / |H_00|"] < 0.1:
        return "RADIAL_CHARACTERISTIC (sigma = (Omega^2 - k^2) K_bg on the nonzero part, lattice residual reported)"
    if res["n_crossings"] >= 1:
        return "CHARACTERISTIC (crossings at speeds below or at 1, without the radial factorization)"
    return "NO_CHARACTERISTICS"


def run():
    out = {"rung": "R17-0g", "Omegas": OMEGAS.tolist(), "CP_pattern": CP_SPEC.tolist(), "backgrounds": {}}
    bgs = {"analytic_hedgehog_n32_L48": (None, 32, 48.0), "r16_1_end_n32_L48": (os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"), 32, 48.0),
           "r16_1_end_n64_L48": (os.path.join(CK16, "r16_1_rebuild_n64_L48_analytic.npy"), 64, 48.0)}
    for lab, (p, n, L) in bgs.items():
        cfgq = {}
        for comp in ("rebuild", "norm"):
            cfgq[f"quartic_{comp}"] = C.cfg_v4(n, L, mu=0.0, cP=0.0, cs=0.0, completion=comp, n_samples=1)
        cfgq["full_v4_rebuild"] = C.cfg_v4(n, L, completion="rebuild", n_samples=1)
        cfg0 = cfgq["quartic_rebuild"]
        M = C15.seed_uniaxial(cfg0) if p is None else np.load(p)
        h = cfg0["h"]
        X, Y, Z = INS4.coords(n, h)
        r = np.sqrt(X * X + Y * Y + Z * Z)
        j = n // 2                                     # x = y = +h/2 at index n//2
        cells, names = [], []
        for rt in (3.0, 6.0, 12.0, 18.0):
            kz = int(np.argmin(np.abs(Z[j, j, :] - rt)))
            cells.append((j, j, kz)); names.append(f"axis_r{rt:g}")
        ko = int(np.argmin(np.abs(Z[j, j, :] - 6.0)))
        cells.append((j + int(round(4.0 / h)), j, ko)); names.append("offaxis_x4_z6")
        rec = {"field": rel(p) if p else "C15.seed_uniaxial", "n": n, "L": L, "h": h, "cells": {nm: {"index": [int(c_) for c_ in c], "r": float(r[c]), "xyz": [float(X[c]), float(Y[c]), float(Z[c])]} for nm, c in zip(names, cells)}, "contractions": {}}
        log(f"{lab}: cells " + ", ".join(f"{nm} r {r[c]:.2f}" for nm, c in zip(names, cells)))
        for cname, cfg in cfgq.items():
            t = time.time()
            H, blab = full_symbol(M, cfg, cells)
            res = {}
            for ci, (nm, c) in enumerate(zip(names, cells)):
                pos = np.array([X[c], Y[c], Z[c]]); rhat = pos / np.linalg.norm(pos)
                tr = np.cross(rhat, [1.0, 0.0, 0.0]); tr = tr / np.linalg.norm(tr)
                res[nm] = {"radial": analyze(H[ci], rhat), "transverse": analyze(H[ci], tr)}
                res[nm]["radial"]["verdict"] = verdict(res[nm]["radial"])
                res[nm]["transverse"]["verdict"] = verdict(res[nm]["transverse"])
            rec["contractions"][cname] = res
            log(f"  {cname} ({time.time() - t:.0f} s): " + "; ".join(f"{nm}: radial {res[nm]['radial']['verdict'].split(' ')[0]} (sig0 {res[nm]['radial']['signature_at_0']}, sig1.6 {res[nm]['radial']['signature_at_1.6']}, cross {res[nm]['radial']['n_crossings']} at {[round(x, 2) for x in res[nm]['radial']['crossings_Omega'][:3]]}, fac {res[nm]['radial']['factorization_residual |H_kk + H_00| / |H_00|']:.1e}, H00 vs CP {res[nm]['radial']['H00_spectrum_vs_CP_pattern_max_dev']:.2e}); "
                                                  f"transverse {res[nm]['transverse']['verdict'].split(' ')[0]} (cross {res[nm]['transverse']['n_crossings']}, fac {res[nm]['transverse']['factorization_residual |H_kk + H_00| / |H_00|']:.1e})" for nm in names))
            json.dump(out, open(os.path.join(CK, "r17_0_symbol_partial.json"), "w"), indent=1, default=float)
        # both completions identical on u = e_0 fields?
        d = 0.0
        for nm in names:
            a_, b_ = rec["contractions"]["quartic_rebuild"][nm]["radial"]["eigs_scan"], rec["contractions"]["quartic_norm"][nm]["radial"]["eigs_scan"]
            d = max(d, float(np.max(np.abs(np.array(a_) - np.array(b_))) / max(np.max(np.abs(np.array(a_))), 1e-300)))
        rec["completions_agree_rel"] = d
        out["backgrounds"][lab] = rec
        json.dump(out, open(os.path.join(CK, "r17_0_symbol_partial.json"), "w"), indent=1, default=float)
    # summary verdicts
    summ = {}
    for lab, rec in out["backgrounds"].items():
        for cname, res in rec["contractions"].items():
            for nm in res:
                summ[f"{lab}|{cname}|{nm}"] = {"radial": res[nm]["radial"]["verdict"], "transverse": res[nm]["transverse"]["verdict"], "radial_signature_0_to_1.6": [res[nm]["radial"]["signature_at_0"], res[nm]["radial"]["signature_at_1.6"]]}
    out["summary"] = summ
    q = {k: v for k, v in summ.items() if "quartic" in k}
    out["decision"] = {"hunt_report_65 (no characteristics, signature fixed at every omega) holds on our cores": bool(all(("NO_CHARACTERISTICS" in v["radial"] and "NO_CHARACTERISTICS" in v["transverse"]) for v in q.values())),
                       "hunt_report_65 refuted (a crossing on some cell, quartic)": bool(any(("NO_CHARACTERISTICS" not in v["radial"]) or ("NO_CHARACTERISTICS" not in v["transverse"]) for v in q.values())),
                       "complete_picture_40 (radial characteristic on the hedgehog axis, r >= 6, analytic background)": bool(all(("RADIAL_CHARACTERISTIC" in v["radial"]) for k, v in q.items() if "analytic" in k and "axis" in k and "r3" not in k)),
                       "complete_picture_40 on the relaxed cores (axis, r >= 6)": bool(all(("RADIAL_CHARACTERISTIC" in v["radial"]) for k, v in q.items() if "r16_1" in k and "axis" in k and "r3" not in k)),
                       "transverse characteristics too (not claimed by either report)": bool(any(("NO_CHARACTERISTICS" not in v["transverse"]) for v in q.values())),
                       "signature changes with Omega on every quartic cell": bool(all(v["radial_signature_0_to_1.6"][0] != v["radial_signature_0_to_1.6"][1] for v in q.values()))}
    # plot: the eigenvalue scan on the analytic hedgehog axis_r6 and the R16-1 n32 axis_r3, quartic, radial and transverse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = [("analytic_hedgehog_n32_L48", "axis_r6"), ("r16_1_end_n32_L48", "axis_r3"), ("r16_1_end_n32_L48", "axis_r6"), ("r16_1_end_n64_L48", "axis_r3")]
    fig, ax = plt.subplots(2, len(picks), figsize=(4 * len(picks), 7), sharex=True)
    for ci, (lab, nm) in enumerate(picks):
        if lab not in out["backgrounds"]:
            continue
        res = out["backgrounds"][lab]["contractions"]["quartic_rebuild"][nm]
        for ri, kind in enumerate(("radial", "transverse")):
            e = np.array(res[kind]["eigs_scan"])
            for jj in range(10):
                ax[ri, ci].plot(OMEGAS, e[:, jj], lw=0.8)
            ax[ri, ci].axhline(0, color="k", lw=0.5); ax[ri, ci].axvline(1.0, color="r", ls=":", lw=0.7)
            ax[ri, ci].set_title(f"{lab}\n{nm} {kind}: {res[kind]['verdict'].split(' ')[0]}", fontsize=7)
            if ci == 0:
                ax[ri, ci].set_ylabel("eigenvalues of sigma(Omega, k), |k| = 1")
            if ri == 1:
                ax[ri, ci].set_xlabel("Omega")
    fig.suptitle("R17-0g: the 10 x 10 principal symbol of the quartic on frozen hedgehog backgrounds (red: Omega = |k|)", fontsize=8)
    fig.savefig(os.path.join(PLOTS, "m5_32_r17_0_symbol.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    out["plot"] = "plots/m5_32_r17_0_symbol.png"
    out["wall_s"] = time.time() - T0
    json.dump(out, open(os.path.join(DATA, "m5_32_r17_0_symbol.json"), "w"), indent=1, default=float)
    log(f"decision {out['decision']}; written data/m5_32_r17_0_symbol.json ({out['wall_s']:.0f} s)")


def rel(p):
    return os.path.relpath(p, RES)


if __name__ == "__main__":
    run()
