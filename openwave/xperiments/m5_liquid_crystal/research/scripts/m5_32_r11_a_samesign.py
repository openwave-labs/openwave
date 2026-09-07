"""M5.32 R11 (producer): the author's 2026-08-29 same-sign notebook, the cost
of the global sign flip it uses, and the two readings of (F_abcd F^abcd)^2.

Source: theory/duda_2026-08-29_newton_same_sign_boost_hedgehogs.pdf (local
corpus). Transcribed difference from the 2026-08-17 notebook (R0): the two
centers carry the SAME sign of m (both PDFs; the R0 audit already had
o_k = 1 + m X_k for k = 1, 2) and the spatial density is NEGATED:

    Hs_29 = - sum_coms (com[2,3]^2 + com[3,4]^2 + com[4,2]^2)  =  - Hs_17

    en(d) = int 4 pi x Hs_29 [x^2 + (z - d)^2 > 1e-3] dx dz / (m^4 g^4)
    fit   en(d) = A + B/d  ->  PDF: A = -863.733, B = -167.668

PRE-REGISTERED HYPOTHESES (before any number here)
--------------------------------------------------
H-a  The PDF's fit is the R0 fit with both coefficients negated (the R0 audit
     reproduced A = +863.733, B = +167.668 to six digits from the 08-17 PDF).
     Gate: |A_29 + A_17| / |A_17| < 1e-3 and the same for B, with A_17, B_17
     recomputed here by the R0 audit's own integrator.
H-b  The flip that buys the attraction (negating the static curvature sector,
     which is what "reversing sign of boost curvature Lagrangian
     contributions" does on this ansatz: R3 arm i measured E_int =
     4 [S_int - (1 - 2 lambda) T_int] with the sign living in S) has no
     vacuum: under E_flip = -4 h^3 sum I1 + V4 the lattice energy of a
     perturbed vacuum and of the single hedgehog falls WITHOUT BOUND, at a
     rate that grows as the lattice is refined, while the certified sign
     relaxes the same perturbation back to the vacuum.
     Gate (revised before the first number of arm b was read: the first
     launch's random perturbation left the Lorentz orbit and V4, stiff at
     g^4, stalled the descent, so the probe is now the exact-orbit families):
     along the m ladder (0.05 to 1.6) and the squeeze ladder (s = 1 to 4) of
     the single hedgehog, both exact Lorentz orbits (V4 = 0 to roundoff per
     cell), E_flip decreases monotonically with NO floor at both resolutions
     (h = 1, 2/3); the squeeze exponent of -E_flip is near +1 (density ~ s^4,
     volume ~ s^-3). The pair read under the flip is the R3 read negated
     (E_int(d) attractive by construction, V4_int = 0).
H-c  (F_abcd F^abcd)^2 = R8's Q_I1sq (class C5 in the plan's vocabulary; NOT
     C6): in the ENERGY reading it drives the clock (C2_q < 0 on the certified
     stack) but the certified omega^2 inertia is extensive, so the coefficient
     that opens a well grows with L (R8: 312.6 / 476.5 / 637.2) and at any
     fixed coefficient omega* drifts with L beyond the 10 % bar; the statics
     deformation c5 A_static at the L = 48 threshold exceeds the hedgehog's
     static energy many times over (the parallel report 008 works at a 5 %
     budget on a single 32^3 box, so it cannot see the drift). In the
     FUNDAMENTAL (Legendre) reading, with L_extra = sigma c5 (s + kappa
     omega^2)^2, sigma = +1 gives H_extra = c5 [-s^2 + 2 s k + 3 k^2] (static
     square negative: unbounded below in statics, the parallel report's
     measured runaway) and sigma = -1 gives H_extra = c5 [s^2 - 2 s k - 3 k^2]
     (omega^4 coefficient negative: unbounded below in omega). Neither
     fundamental sign yields a bounded clock. Gate: the Legendre identity
     checked symbolically; omega* drift at fixed c5 > 10 % across the R8
     box ladder; deformation ratio > 1 at the threshold.

Outputs: ../data/m5_32_r11_samesign.json, ../plots/m5_32_r11_samesign.png
Run: nice -n 10 /opt/anaconda3/envs/master312/bin/python3 m5_32_r11_a_samesign.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
DATA, PLOTS = os.path.join(RES, "data"), os.path.join(RES, "plots")
OUT = os.path.join(DATA, "m5_32_r11_samesign.json")
PNG = os.path.join(PLOTS, "m5_32_r11_samesign.png")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R0A = _load("m5_32_r0_audit_notebook", "m5_32_r0_audit_notebook.py")
R3I = _load("m5_32_r3_i_ansatz", "m5_32_r3_i_ansatz.py")
RB = _load("m5_32_r2_b_bounded", "m5_32_r2_b_bounded.py")
LAG, B3 = R3I.LAG, R3I.B3
S_TOY, G_TOY, DELTA_TOY = R3I.S_TOY, R3I.G_TOY, R3I.DELTA_TOY


# ============================ arm a: the notebook =============================
def arm_a():
    ds = [round(0.1 * k, 10) for k in range(1, 31)]
    es17 = [R0A.energy(dv, 1e-3) for dv in ds]
    A17, B17 = R0A.fit_lin(ds, es17, [lambda v: 1.0, lambda v: 1.0 / v])
    es29 = [-e for e in es17]                       # Hs_29 = -Hs_17, transcribed
    A29, B29 = R0A.fit_lin(ds, es29, [lambda v: 1.0, lambda v: 1.0 / v])
    pdf = {"A": -863.733, "B": -167.668}
    out = {
        "transcription": {
            "centers": "o1 = exp(m f({x,y,z+d}.Gb)), o2 = exp(m f({x,y,z-d}.Gb)), SAME sign of m (as in the 08-17 PDF)",
            "density": "Hs = -Sum(com[2,3]^2 + com[3,4]^2 + com[4,2]^2) (the 08-17 sum NEGATED)",
            "fit_line": "-863.733 - 167.668/d (PDF Out[15])",
        },
        "fit_17_recomputed": {"A": A17, "B": B17},
        "fit_29": {"A": A29, "B": B29},
        "pdf_29": pdf,
        "rel_err_A": abs(A29 - pdf["A"]) / abs(pdf["A"]),
        "rel_err_B": abs(B29 - pdf["B"]) / abs(pdf["B"]),
        "en_table": dict(zip([str(v) for v in ds], es29)),
    }
    out["gate_Ha"] = bool(out["rel_err_A"] < 1e-3 and out["rel_err_B"] < 1e-3)
    log(f"arm a: 17 fit A={A17:.3f} B={B17:.3f}; 29 fit A={A29:.3f} B={B29:.3f} "
        f"(PDF -863.733, -167.668); gate {out['gate_Ha']}")
    return out


# ===================== arm b: the flipped static sector =======================
def _squeezed(M0, centers, m, rc, n, L, s):
    X, Y, Z = B3.coords(n, L / n)
    o = None
    for c in centers:
        ok = R3I.boost_field(s * X, s * Y, s * Z, (s * c[0], s * c[1], s * c[2]), m, rc)
        o = ok if o is None else o @ ok
    return o @ M0 @ o.swapaxes(-1, -2)


def arm_b():
    """E_flip = -4 h^3 sum I1 + V4 along EXACT Lorentz-orbit families (V4 = 0
    pointwise by F1 of the R3 audit): the m ladder and the squeeze ladder of
    the single hedgehog. Unbounded below <=> E_flip decreasing without a floor
    along the family while V4 stays at roundoff."""
    out = {"boxes": []}
    p = LAG.default_params(s=S_TOY, g=G_TOY, delta=DELTA_TOY)
    M0 = R3I.M0_SET["toy"]
    for n, L in ((32, 32.0), (48, 32.0)):
        cfg = B3.base_cfg(s=S_TOY, g=G_TOY, n=n, L=L, delta=DELTA_TOY)
        box = {"n": n, "L": L, "h": L / n, "m_ladder": [], "squeeze_ladder": []}
        for m in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6):
            M = R3I.ansatz(M0, [(0.0, 0.0, 0.0)], m, 0.5, n, L)
            tot = R3I.evaluate(M, cfg, p)
            box["m_ladder"].append({"m": m, "E_curv": 4.0 * tot["I1"], "E_flip": -4.0 * tot["I1"] + tot["V4"],
                                    "V4": tot["V4"], "V4_cell_max_abs": tot["V4_cell_max_abs"], "S": tot["S"], "T": tot["T"]})
            log(f"arm b n={n} m={m}: E_curv {4*tot['I1']:.5g} E_flip {-4*tot['I1']+tot['V4']:.5g} V4cell {tot['V4_cell_max_abs']:.2e}")
        for sq in (1.0, 1.5, 2.0, 3.0, 4.0):
            M = _squeezed(M0, [(0.0, 0.0, 0.0)], 0.2, 0.5, n, L, sq)
            tot = R3I.evaluate(M, cfg, p)
            box["squeeze_ladder"].append({"s": sq, "E_curv": 4.0 * tot["I1"], "E_flip": -4.0 * tot["I1"] + tot["V4"],
                                          "V4": tot["V4"], "V4_cell_max_abs": tot["V4_cell_max_abs"]})
            log(f"arm b n={n} squeeze {sq}: E_curv {4*tot['I1']:.5g} E_flip {-4*tot['I1']+tot['V4']:.5g} V4cell {tot['V4_cell_max_abs']:.2e}")
        em = [r["E_flip"] for r in box["m_ladder"]]
        es = [r["E_flip"] for r in box["squeeze_ladder"]]
        box["m_monotone_down"] = bool(all(em[k + 1] < em[k] for k in range(len(em) - 1)))
        box["squeeze_monotone_down"] = bool(all(es[k + 1] < es[k] for k in range(len(es) - 1)))
        box["squeeze_exponent"] = float(np.polyfit(np.log([r["s"] for r in box["squeeze_ladder"]]), np.log([-v for v in es]), 1)[0])
        box["V4_cell_max_abs_over_ladders"] = max(r["V4_cell_max_abs"] for r in box["m_ladder"] + box["squeeze_ladder"])
        out["boxes"].append(box)
    # (iii) the pair read under both signs (R3 arm i construction, unrelaxed)
    n, L, rc, m = 48, 48.0, 0.5, 0.1
    tot = {}
    for kind, d in (("vac", 0), ("single", 0), ("pair", 8), ("pair", 10), ("pair", 12)):
        r = R3I.job(("toy", n, L, rc, m, kind, d))
        tot[f"{kind}_{d}"] = r
        log(f"arm b pair read {kind} d={d}: I1 {r['I1']:.5g} V4 {r['V4']:.3g} ({r['runtime_s']:.0f}s)")
    ev = R3I.e_lambda(tot["vac_0"], 0.0)
    es1 = R3I.e_lambda(tot["single_0"], 0.0) - ev
    eint = {}
    for d in (8, 10, 12):
        ep = R3I.e_lambda(tot[f"pair_{d}"], 0.0) - ev
        curv = 4.0 * (tot[f"pair_{d}"]["I1"] - tot["vac_0"]["I1"]) - 2 * 4.0 * (tot["single_0"]["I1"] - tot["vac_0"]["I1"])
        v4i = (tot[f"pair_{d}"]["V4"] - tot["vac_0"]["V4"]) - 2 * (tot["single_0"]["V4"] - tot["vac_0"]["V4"])
        eint[str(d)] = {"E_int_certified": ep - 2 * es1, "E_int_flip": -curv + v4i, "V4_int": v4i}
    ds = [8, 10, 12]
    cert = [eint[str(d)]["E_int_certified"] for d in ds]
    flip = [eint[str(d)]["E_int_flip"] for d in ds]
    out["pair"] = {
        "construction": "R3 arm i (unrelaxed product ansatz, toy point, lambda = 0), n 48, L 48, r_c 0.5, m 0.1",
        "E_int": eint,
        "certified_sign": "REPULSIVE" if cert[0] > cert[-1] else "ATTRACTIVE",
        "flip_sign": "REPULSIVE" if flip[0] > flip[-1] else "ATTRACTIVE",
        "V4_int_max_abs": max(abs(eint[str(d)]["V4_int"]) for d in ds),
    }
    g = all(b["m_monotone_down"] and b["squeeze_monotone_down"] and b["V4_cell_max_abs_over_ladders"] < 1e-8 for b in out["boxes"])
    out["gate_Hb"] = bool(g and out["pair"]["certified_sign"] == "REPULSIVE" and out["pair"]["flip_sign"] == "ATTRACTIVE")
    log(f"arm b gate {out['gate_Hb']} (pair {out['pair']['certified_sign']} / flip {out['pair']['flip_sign']}; squeeze exponents {[round(b['squeeze_exponent'], 3) for b in out['boxes']]})")
    return out


# ================= arm c: (I1)^2 in the two readings (R8 data) ================
def arm_c():
    import sympy as sp
    r8 = json.load(open(os.path.join(DATA, "m5_32_r8_quartics.json")))
    q = r8["scaling"]["Q_I1sq"]
    i1 = r8["scaling"]["I1"]
    boxes = [b["L"] for b in r8["boxes"]]
    H2_cert = [-4.0 * c for c in i1["C2"]]                # the certified inertia (R8 note)
    E_stat_cert = [4.0 * a for a in i1["A_static"]]        # 4 A_I1 (V4 = 0 at the toy vacuum)
    c5_thr = [h / (-c) for h, c in zip(H2_cert, q["C2"])]
    # Legendre check
    s, k, c5, w, kap = sp.symbols("s k c5 omega kappa", real=True)
    res = {}
    for sigma in (+1, -1):
        Lx = sigma * c5 * (s + kap * w**2) ** 2
        Hx = sp.expand(w * sp.diff(Lx, w) - Lx)            # H = q' dL/dq' - L on the clock
        Hx = Hx.subs(kap * w**2, k)
        res[f"sigma_{sigma:+d}"] = str(sp.expand(Hx.subs(w, sp.sqrt(k / kap))))
    # per-box readings at the fixed coefficient c5 = c5_thr[0] * 1.5 (opens a well at L = 48)
    rows = []
    for c5_fix, tag in ((1.5 * c5_thr[0], "1.5 x threshold(L=48)"), (1.5 * c5_thr[-1], "1.5 x threshold(L=96)")):
      for j, L in enumerate(boxes):
          H2 = H2_cert[j] + c5_fix * q["C2"][j]
          H4 = c5_fix * q["C4"][j]
          w2 = -H2 / (2 * H4) if H2 < 0 else 0.0
          rows.append({
              "c5_tag": tag, "L": L, "H2_certified": H2_cert[j], "C2_q": q["C2"][j], "C4_q": q["C4"][j],
              "c5_threshold": c5_thr[j], "deformation_ratio_at_threshold": c5_thr[j] * q["A_static"][j] / E_stat_cert[j],
              "energy_reading": {"c5": c5_fix, "H2": H2, "H4": H4, "omega_star": math.sqrt(w2)},
              "fundamental_sigma_plus": {"static_square": -c5_fix * q["A_static"][j], "H2": H2_cert[j] - c5_fix * q["C2"][j] * (-1),
                                         "H4": 3 * c5_fix * q["C4"][j], "note": "H2 keeps the drive, static square NEGATIVE: unbounded below in statics"},
              "fundamental_sigma_minus": {"static_square": +c5_fix * q["A_static"][j], "H2": H2_cert[j] - c5_fix * q["C2"][j],
                                          "H4": -3 * c5_fix * q["C4"][j], "note": "H4 NEGATIVE: unbounded below in omega"},
          })
    ws = [r["energy_reading"]["omega_star"] for r in rows if r["c5_tag"].endswith("(L=96)")]
    drift = (max(ws) - min(ws)) / max(ws) if max(ws) > 0 else float("nan")
    out = {
        "term": "Q_I1sq = (F_abcd F^abcd)^2, R8 class C5 (the note's convo/tracker entries said C6: corrected here)",
        "legendre": {"L_extra": "sigma c5 (s + kappa omega^2)^2, k = kappa omega^2",
                     "H_extra": res,
                     "expected": {"sigma_+1": "c5 (-s^2 + 2 s k + 3 k^2)", "sigma_-1": "c5 (s^2 - 2 s k - 3 k^2)"}},
        "rows": rows,
        "omega_star_drift_energy_reading": drift,
        "parallel_report_008": "single 32^3 box, gamma fixed by a 5 % statics deformation, energy reading only; cannot see the L drift",
    }
    out["gate_Hc"] = bool(drift > 0.10 and all(r["deformation_ratio_at_threshold"] > 1 for r in rows)
                          and "3*c5*k**2" in res["sigma_+1"].replace(" ", "") and "-3*c5*k**2" in res["sigma_-1"].replace(" ", ""))
    log(f"arm c: thresholds {c5_thr}; omega* at c5={c5_fix:.1f}: {ws}; drift {drift:.3f}; "
        f"deformation ratios {[round(r['deformation_ratio_at_threshold'], 1) for r in rows]}; gate {out['gate_Hc']}")
    log(f"arm c Legendre: {res}")
    return out


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ds = [float(k) for k in res["arm_a"]["en_table"]]
    ax[0].plot(ds, list(res["arm_a"]["en_table"].values()), "o", ms=3, label="en(d), Hs negated")
    A, B = res["arm_a"]["fit_29"]["A"], res["arm_a"]["fit_29"]["B"]
    dd = np.linspace(0.2, 3, 200)
    ax[0].plot(dd, A + B / dd, "-", label=f"fit {A:.1f} + {B:.1f}/d")
    ax[0].set_xlabel("d"); ax[0].set_title("arm a: the 08-29 notebook (R0 integrator)"); ax[0].legend()
    for box in res["arm_b"]["boxes"]:
        ax[1].loglog([r["m"] for r in box["m_ladder"]], [-r["E_flip"] for r in box["m_ladder"]], "o-", label=f"m ladder, h={box['h']:.2f}")
        ax[1].loglog([r["s"] for r in box["squeeze_ladder"]], [-r["E_flip"] for r in box["squeeze_ladder"]], "s--", label=f"squeeze ladder (m 0.2), h={box['h']:.2f}")
    ax[1].set_xlabel("m  or  squeeze s"); ax[1].set_ylabel("-E_flip (V4 = 0 exactly)"); ax[1].set_title("arm b: the flip has no floor on the Lorentz orbit"); ax[1].legend(fontsize=7)
    rows = res["arm_c"]["rows"]
    for tag in sorted(set(r["c5_tag"] for r in rows)):
        rr = [r for r in rows if r["c5_tag"] == tag]
        ax[2].plot([r["L"] for r in rr], [r["energy_reading"]["omega_star"] for r in rr], "o-", label=f"omega* at c5 = {tag}")
    rr = [r for r in rows if r["c5_tag"].endswith("(L=96)")]
    ax[2].plot([r["L"] for r in rr], [r["c5_threshold"] / 1000 for r in rr], "s--", label="c5 threshold / 1000")
    ax[2].set_xlabel("L"); ax[2].set_title("arm c: (I1)^2 energy reading vs box"); ax[2].legend()
    fig.tight_layout(); fig.savefig(PNG, dpi=110)


def main():
    res = {"task": "M5.32 R11", "source": "theory/duda_2026-08-29_newton_same_sign_boost_hedgehogs.pdf"}
    res["arm_a"] = arm_a()
    json.dump(res, open(OUT, "w"), indent=1)
    res["arm_c"] = arm_c()
    json.dump(res, open(OUT, "w"), indent=1)
    res["arm_b"] = arm_b()
    res["gates"] = {"Ha": res["arm_a"]["gate_Ha"], "Hb": res["arm_b"]["gate_Hb"], "Hc": res["arm_c"]["gate_Hc"]}
    res["runtime_s"] = time.time() - T0
    json.dump(res, open(OUT, "w"), indent=1)
    plot(res)
    log(f"DONE gates {res['gates']} -> {OUT}")


if __name__ == "__main__":
    main()
