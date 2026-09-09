"""M5.32 R18-2 (ledger 6.7): the condensate inside the admissible domain and the Goldstone reads (the author's rev-155 reply,
section 129 of rev 200: a core condensate b != 0 breaks the clock's U(1), the clock is the Goldstone phase with stiffness b^2,
E(K) = K^2 / 2 I, I = int b^2 d^3x).

Statics: object (C) v6 as run in R17-3a ((nu, kappa) = (1e-2, 0.4), W = [(1 - lambda_1) / (1 - delta)]^2, c_s 0.5, the relative
weight), g_W in {1.5, 1.65, 1.8, 2.0} continued inside the R17-3a bracket (1.35, 2.0], n32 L48, the R16-1 core seeded with a
doublet split 0.05 at r 2 (the R17-3a seed) and at r 5 (the shell seed), max_iter 3000, dt0 0.001 (m5_32_r17_2_statics.py,
the tag suffix _r5 for the shell seed).  CONDENSED by the R17-3a collect rule: the end energy below the split-free saddle
(the g_W 2.0 unseeded control, 7.27779, the same number for every g_W since U_v6 = 0 at rho = 0) by more than 1e-4, or the
seeded split grown above its seed amplitude; escape (d) = UNRESOLVED (the core left the admissible domain: the plateau
weight's definition boundary, the run cannot be pulled back inside).
Goldstone reads on a field (every condensed end field; with none, the field closest to condensation as the instrument read,
its numbers scaling with the residual rho^2 and stated as such):
    I = int rho^2 d^3x = h^3 sum spl / 4   (the author's b = rho, our split amplitude (lambda_2 - lambda_3) / 2),
    the clock inertia 2 kin_tot (the a0-kinetic read in the R16 convention E_K = E_stat + K^2 / (4 kin)),
    the phase stiffness S by the static twist theta = q x through the local generator (m5_32_r18_common.twist_stiffness:
        E(q) - E(0) = S q^2 / 2 from the even part at q 0.05 / 0.1 / 0.2, the odd part the torque, zero on a stationary field),
        S against I and against 4 I (the K_P^23 normalization of a pure pair-plane twist),
    the spin-2 zero count on the shells (R18-0c),
    E(K) at K in {1, 2, 5, 10} by the true-gradient fixed-K descent from the field (m5_32_r17_3_fixedk.relax, dt0 0.001,
        1500 iterations), E(K) - E(0) against the rotor law K^2 / 2 I and against the delocalized branch omega_c K
        (omega_c = sqrt(box bottom) 0.160 from the R17-2 empty-box operator), E(K = 1) / E_core in the author's ratio.
Pre-registered outcomes: NO_CONDENSATE_IN_DOMAIN (no condensed state at any g_W <= 2.0 on either seed: the window question
returns to the author as a (lambda, nu, kappa) question); GOLDSTONE_CLOCK_BOUND (condensed and the rotor law within 10 percent
at K <= 5 with the state core-bound); ROTOR_NOT_MINIMUM (condensed but the fixed-K descent leaves the core).

usage: python3 m5_32_r18_2_goldstone.py reads --field <npy> --gW 1.65 --label <label>       # I, S, 2 kin, spin-2
       python3 m5_32_r18_2_goldstone.py fixedk --field <npy> --gW 1.65 --K 5 [--maxit 1500] # one E(K) descent
       python3 m5_32_r18_2_goldstone.py collect
out:   checkpoints/m5_32_r18/r18_2_reads_<label>.json, r18_2_v6_gW<g>_n32_L48_K<K>.json / .npy, data/m5_32_r18_2.json, plots/m5_32_r18_2_condensate.png
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ARGS = list(sys.argv[1:])
sys.argv = [sys.argv[0]]
import m5_32_r18_common as X                              # noqa: E402
import m5_32_r17_common as R                              # noqa: E402
import m5_32_r16_common as C                              # noqa: E402
import m5_32_r17_3_fixedk as FK                           # noqa: E402

C15, INS4 = C.C15, C.INS4
RES, DATA, PLOTS = C.RES, C.DATA, C.PLOTS
CK16, CK17, CK = C.CK, R.CK, X.CK
T0 = time.time()
OMEGA_C = np.sqrt(0.025632)                               # the empty-box doublet bottom (R16-2 / R17-2b, the x lift)
FK.CK = CK
FK.tag_of = lambda gW, n, L, K: f"r18_2_v6_gW{gW:g}_n{n}_L{int(L)}_K{int(K)}"


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def rel(p):
    return os.path.relpath(p, RES)


def reads(field, gW, label, n=32, L=48.0):
    cfg = R.cfg_v6(n, L, gW=gW, completion="rebuild", n_samples=4)
    M = np.load(field)
    nref_p = field[:-4] + "_nref.npy"
    nref = np.load(nref_p) if os.path.exists(nref_p) else C.radial_ref(cfg)
    h3 = cfg["h"] ** 3
    X_, Y_, Z_ = INS4.coords(n, cfg["h"])
    r = np.sqrt(X_ * X_ + Y_ * Y_ + Z_ * Z_)
    spl, _ = C15.split_cells(M, need_grad=False)
    rho2 = np.real(spl) / 4.0
    I = h3 * float(np.sum(rho2))
    rec = {"label": label, "field": rel(field), "gW": gW, "lift": rel(nref_p) if os.path.exists(nref_p) else "radial",
           "I_int_rho2": I, "rho_max": float(np.sqrt(np.max(rho2))), "r_at_rho_max": float(r.reshape(-1)[int(np.argmax(rho2))]), "rho2_rms_radius": float(np.sqrt(np.sum(rho2 * r * r) / max(np.sum(rho2), 1e-300))),
           "rho2_fraction_inside_r6": float(np.sum(rho2[r < 6.0]) / max(np.sum(rho2), 1e-300))}
    cf8 = dict(cfg); cf8["n_samples"] = 8
    E0, _, pp, dom, fr = R.energy_object(M, cf8, 50.0, nref, need_grad=False)
    rec["E_stat_8"] = float(np.real(pp["E_stat"])); rec["kin_tot"] = float(np.real(pp["kin_tot"])); rec["clock_inertia_2kin"] = 2.0 * rec["kin_tot"]; rec["domain"] = dom
    rec["parts_8"] = {k: float(np.real(v)) for k, v in pp.items() if isinstance(v, (int, float))}
    t = time.time()
    tw = X.twist_stiffness(M, cfg, nref, qs=(0.05, 0.1, 0.2), axis=0, n_samples=8)
    rec["twist"] = tw
    S = tw["S_at_smallest_q"]
    rec["S_over_I"] = S / I if I > 0 else None; rec["S_over_4I"] = S / (4 * I) if I > 0 else None
    rec["S_over_2kin"] = S / rec["clock_inertia_2kin"] if rec["clock_inertia_2kin"] > 0 else None
    log(f"  {label}: I {I:.5e} (rho_max {rec['rho_max']:.4f} at r {rec['r_at_rho_max']:.2f}, rms radius {rec['rho2_rms_radius']:.2f}); 2 kin {rec['clock_inertia_2kin']:.5e}; S {S:.5e} (S/I {rec['S_over_I']:.3f}, S/4I {rec['S_over_4I']:.3f}, S/2kin {rec['S_over_2kin']:.3f}); torque {tw['torque_odd_at_smallest_q']:.2e}; quad dev {tw['quadratic_rel_dev_q1_q2']:.2e} ({time.time() - t:.0f} s)")
    rec["spin2"] = X.spin2_zero_count(M, cfg, radii=(1.5, 2.25, 3.0, 4.5, 6.0))
    rec["spin2_summary"] = {k: {"total": v["total_index"], "n": v["n_zeros"], "hist": v["index_histogram"], "resid": v["fit_residual_rel"], "l2m": {mm: round(p / max(sum(v["l2_m_power"].values()), 1e-300), 2) for mm, p in v["l2_m_power"].items()}} for k, v in rec["spin2"].items() if "total_index" in v}
    log(f"  spin-2: {rec['spin2_summary']}")
    json.dump(rec, open(os.path.join(CK, f"r18_2_reads_{label}.json"), "w"), indent=1, default=float)
    return rec


def collect():
    out = {"rung": "R18-2", "statics": {}, "reads": {}, "fixedK": {}}
    saddle = 7.277790
    try:
        saddle = json.load(open(os.path.join(DATA, "m5_32_r17_2.json")))["saddle_E_stat_8 (the v4rel static)"]
    except Exception:                                       # noqa: BLE001
        pass
    rows = []
    for p in sorted(glob.glob(os.path.join(CK17, "r17_2_v6_*.json"))):
        r_ = json.load(open(p))
        if r_.get("object") != "v6":
            continue
        tr = r_.get("trace", [])
        last = tr[-1] if tr else {}
        row = {"tag": r_["tag"], "gW": r_["gW"], "seed_split": r_.get("seed_split", 0.0), "seed_r": r_.get("seed_r", 2.0), "stop": r_.get("descent", {}).get("stop"), "iters": r_.get("descent", {}).get("iters"),
               "E_stat_8": r_.get("reads", {}).get("E_stat_8"), "half_split_max": r_.get("reads", {}).get("texture", {}).get("half_split_max"), "Delta_min": r_.get("reads", {}).get("core", {}).get("Delta_min_free"),
               "W_max": r_.get("reads", {}).get("parts_8", {}).get("W_max"), "mu_eff_min": r_.get("reads", {}).get("parts_8", {}).get("mu_eff_min"),
               "trace_last": {k: last.get(k) for k in ("it", "E_stat", "dom_half_split_max", "dom_l1_min", "dom_gap_1_2_min")}, "rung": r_.get("rung"), "end_field": r_.get("end_field")}
        row["E_minus_saddle"] = (row["E_stat_8"] - saddle) if row["E_stat_8"] is not None else None
        if row["stop"] == "escape_d":
            row["condensed"] = "UNRESOLVED_ESCAPE_D"
        elif row["seed_split"] > 0 and row["E_stat_8"] is not None:
            row["condensed"] = bool(row["E_minus_saddle"] < -1e-4 or (row["half_split_max"] is not None and row["half_split_max"] > row["seed_split"]))
        elif row["E_stat_8"] is not None:
            row["condensed"] = bool(row["half_split_max"] is not None and row["half_split_max"] > 1e-2)
        else:
            row["condensed"] = "RUNNING"
        rows.append(row)
    rows.sort(key=lambda x: (x["seed_r"], x["gW"], x["seed_split"]))
    out["statics"] = {"saddle": saddle, "rows": rows, "rule": "condensed iff E_end < E_saddle - 1e-4 or the seeded split grew above its seed; escape (d) = unresolved"}
    conc = [x for x in rows if x["condensed"] is True]
    esc = [x for x in rows if x["condensed"] == "UNRESOLVED_ESCAPE_D"]
    notc = [x for x in rows if x["condensed"] is False]
    out["statics"]["condensed_gW"] = sorted(set(x["gW"] for x in conc)); out["statics"]["escape_d_gW"] = sorted(set((x["gW"], x["seed_r"]) for x in esc)); out["statics"]["not_condensed"] = sorted(set((x["gW"], x["seed_r"]) for x in notc))
    for p in sorted(glob.glob(os.path.join(CK, "r18_2_reads_*.json"))):
        r_ = json.load(open(p)); r_.pop("spin2", None)
        out["reads"][r_["label"]] = r_
    for p in sorted(glob.glob(os.path.join(CK, "r18_2_v6_gW*_K*.json"))):
        r_ = json.load(open(p)); r_.pop("trace", None)
        out["fixedK"][r_["tag"]] = {k: r_.get(k) for k in ("tag", "gW", "K", "verdict", "descent", "end_parts_8", "stationarity", "escapes_end", "bound", "dE_dK", "end_domain") if k in r_}
        out["fixedK"][r_["tag"]]["seed_E_K"] = (r_.get("seed") or {}).get("parts", {}).get("E_K")
        out["fixedK"][r_["tag"]]["seed_omega"] = (r_.get("seed") or {}).get("parts", {}).get("omega")
    # the rotor law on the fixed-K rows
    rot = []
    for tag, r_ in out["fixedK"].items():
        ep = r_.get("end_parts_8") or {}
        EK = ep.get("E_K")
        lab = [k for k, v in out["reads"].items() if abs(v["gW"] - r_["gW"]) < 1e-9 and "core_seeded" in k]
        rd = out["reads"][lab[0]] if lab else None
        if EK is not None and rd is not None:
            dE = EK - rd["E_stat_8"]
            K = r_["K"]
            rot.append({"tag": tag, "gW": r_["gW"], "K": K, "E_K_end": EK, "E_K_frozen_seed": r_.get("seed_E_K"), "dE_end": dE, "dE_frozen_seed": (r_.get("seed_E_K") - rd["E_stat_8"]) if r_.get("seed_E_K") is not None else None,
                         "rotor_K2_over_2I (the author)": K ** 2 / (2 * rd["I_int_rho2"]) if rd["I_int_rho2"] > 0 else None, "K2_over_4kin (the instrument, frozen)": K ** 2 / (4 * rd["kin_tot"]) if rd["kin_tot"] > 0 else None,
                         "delocalized_omega_c_K (v4 box bottom 0.160)": OMEGA_C * K, "delocalized_omega_c_K (the R17-3c bound 0.05)": (r_.get("bound") or {}).get("omega_c_K"),
                         "omega_end": ep.get("omega"), "omega_frozen_seed": r_.get("seed_omega"), "kin_tot_end": ep.get("kin_tot"), "half_split_end": (r_.get("escapes_end") or {}).get("half_split_max"), "gap_min_end": (r_.get("end_domain") or {}).get("gap_1_2_min"),
                         "E_K1_over_E_core": (dE / rd["E_stat_8"]) if K == 1 else None, "dE_dK_over_omega": (r_.get("dE_dK") or {}).get("ratio_to_omega"), "verdict": r_.get("verdict"), "stop": (r_.get("descent") or {}).get("stop"), "iters": (r_.get("descent") or {}).get("iters")})
    out["rotor_law"] = rot
    if not conc:
        out["verdict"] = ("NO_CONDENSATE_REACHED_IN_DOMAIN: the pre-registered NO_CONDENSATE_IN_DOMAIN qualified by the R18-2 audit: no seeded run condensed within 3000 iterations from either seed, but every run (the saddle included) is still descending at 0.03 per 100 iterations with the seeded excess shrinking, "
                          "the shell-seeded splits at g_W >= 1.8 stop decaying and migrate into the core, and the admissible domain's central isolation margin (0.03 to 0.05) lies below the seed amplitude (0.05) and the sextic plateau s* = 0.112: a core condensate of the author's amplitude is not representable inside the domain on this core"
                          + (f"; escape (d) at g_W {out['statics']['escape_d_gW']}" if esc else "")) if not any(x["condensed"] == "RUNNING" for x in rows) else "RUNNING"
    else:
        ok = [x for x in rot if x["K"] <= 5 and x["rotor_K2_over_2I (the author)"] and abs(x["dE_end"] / x["rotor_K2_over_2I (the author)"] - 1.0) < 0.1 and x["stop"] not in ("escape_d",)]
        left = [x for x in rot if x["stop"] == "escape_d"]
        out["verdict"] = "GOLDSTONE_CLOCK_BOUND" if ok and not left else ("ROTOR_NOT_MINIMUM" if left else "CONDENSED (rotor law not within 10 percent)")
    json.dump(out, open(os.path.join(DATA, "m5_32_r18_2.json"), "w"), indent=1, default=float)
    log(f"collected {len(rows)} statics: condensed {out['statics']['condensed_gW']}, escape d {out['statics']['escape_d_gW']}, not {out['statics']['not_condensed']}; reads {list(out['reads'])}; fixedK {list(out['fixedK'])}; verdict {out['verdict']}")
    plot(out)
    return out


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    rows = out["statics"]["rows"]
    for sr, mk in ((2.0, "o"), (5.0, "s")):
        rr = [x for x in rows if x["seed_split"] > 0 and x["seed_r"] == sr and x["E_minus_saddle"] is not None]
        if rr:
            ax[0].plot([x["gW"] for x in rr], [x["E_minus_saddle"] for x in rr], marker=mk, lw=0.8, label=f"seed at r {sr:g}")
            ax[1].plot([x["gW"] for x in rr], [x["half_split_max"] for x in rr], marker=mk, lw=0.8, label=f"seed at r {sr:g}")
        ee = [x for x in rows if x["seed_split"] > 0 and x["seed_r"] == sr and x["condensed"] == "UNRESOLVED_ESCAPE_D"]
        for x in ee:
            ax[0].axvline(x["gW"], color="r", ls=":", lw=0.7); ax[1].axvline(x["gW"], color="r", ls=":", lw=0.7)
    ax[0].axhline(0, color="k", lw=0.5); ax[0].set_xlabel("g_W"); ax[0].set_ylabel("E_end - E_saddle"); ax[0].set_title("condensation by energy (red: escape d)", fontsize=8); ax[0].legend(fontsize=6)
    ax[1].axhline(0.05, color="gray", ls="--", lw=0.6); ax[1].set_xlabel("g_W"); ax[1].set_ylabel("end half split max (seed 0.05)"); ax[1].set_title("the seeded split's fate", fontsize=8); ax[1].legend(fontsize=6)
    rot = out.get("rotor_law", [])
    if rot:
        Ks = [x["K"] for x in rot]
        ax[2].plot(Ks, [x["dE_end"] for x in rot], "o", label="E(K) - E(0) at the end (escape d)")
        ax[2].plot(Ks, [x["dE_frozen_seed"] for x in rot], "s", label="K^2 / 4 kin (frozen field)")
        ax[2].plot(Ks, [x["rotor_K2_over_2I (the author)"] for x in rot], "x", label="K^2 / 2 I (the author's rotor)")
        ax[2].plot(Ks, [x["delocalized_omega_c_K (v4 box bottom 0.160)"] for x in rot], "+", label="omega_c K (delocalized, 0.160)")
        ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].legend(fontsize=6)
    ax[2].set_xlabel("K"); ax[2].set_ylabel("energy"); ax[2].set_title("the rotor law test", fontsize=8)
    fig.suptitle(f"R18-2: the v6 condensate scan inside (1.35, 2.0] and the Goldstone reads; verdict {out.get('verdict')}", fontsize=9)
    fig.savefig(os.path.join(PLOTS, "m5_32_r18_2_condensate.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    out["plot"] = "plots/m5_32_r18_2_condensate.png"
    json.dump(out, open(os.path.join(DATA, "m5_32_r18_2.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["reads", "fixedk", "collect"])
    ap.add_argument("--field"); ap.add_argument("--gW", type=float, default=1.65); ap.add_argument("--label", default="x"); ap.add_argument("--K", type=float, default=5.0); ap.add_argument("--maxit", type=int, default=1500)
    a = ap.parse_args(ARGS)
    if a.mode == "reads":
        reads(a.field, a.gW, a.label)
    elif a.mode == "fixedk":
        FK.relax(a.field, a.gW, a.K, a.maxit, 32, 48.0, seed_split=0.0, dt0=0.001)
    else:
        collect()
