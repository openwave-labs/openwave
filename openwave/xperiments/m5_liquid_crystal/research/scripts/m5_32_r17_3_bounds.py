"""M5.32 R17-3b (the bounded read): variational upper bounds on the lowest doublet frequency of the v6 end fields
and the radial effective potential by term, without waiting for the Lanczos solve (the seeded v6 fields' residual split
makes the lowest pair nearly degenerate and ARPACK did not converge within the wall clock; ledger 6.6 fallback: report
the profile and the decomposition instead of the spectrum).

EQUATIONS (m5_32_r17_2_operator.py): for a trial doublet zeta (a shell bump times 2Y_lm in the pair frame, the measured
handedness), the Rayleigh quotient <zeta, H zeta> / <zeta, 2 T zeta> from second differences of the object's static
energy (8 circle samples, eps 1e-3) and the three kinetic reads is an UPPER BOUND on the lowest Omega^2 of the doublet
operator (Rayleigh-Ritz).  A trial below the empty box's bottom 0.025632 PROVES a mode below the box bottom
(BOUND_BELOW_BOX); no trial below it leaves NO_BOUND_MODE unproven for the full spectrum but bounds the core-localized
sector: every core-localized trial's quotient is reported.  The radial effective potential of the (2, m) patterns per
shell, split into E_h (the connection floor), V4 + U_v6 (the well, now with W), K_P and reg: prediction (3') (the floor
outside the core, its absence inside) and the well's sign inside the melted core are read from it.

usage: python3 m5_32_r17_3_bounds.py run --field <end.npy> --object v6 --gW 1.35 --label <lab>
out:   checkpoints/m5_32_r17/r17_3b_<label>.json, data/m5_32_r17_3b.json (collect), plots/m5_32_r17_3b_<label>.png
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
import m5_32_r17_common as R                              # noqa: E402
import m5_32_r16_common as C                              # noqa: E402
import m5_32_r17_2_operator as OPR                        # noqa: E402
import m5_32_r17_0_record as REC                          # noqa: E402

C15, INS4 = C.C15, C.INS4
RES, DATA, PLOTS = C.RES, C.DATA, C.PLOTS
CK = R.CK
T0 = time.time()
BOX_BOTTOM = 0.025632


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def rel(p):
    return os.path.relpath(p, RES)


def run(field, obj, gW, label, n=32, L=48.0):
    cfg = OPR.make_cfg(obj, n, L, gW, ns=4)
    M, nref, free, Ea, Eb, fr, r, th, ph = OPR.setup(field, cfg)
    Tm = OPR.inertia(M, cfg, Ea, Eb, nref, free)
    cf8 = dict(cfg); cf8["n_samples"] = 8
    E0, _, pp0, _, _ = R.energy_object(M, cf8, None, nref, need_grad=False)
    rec = {"label": label, "field": rel(field), "object": obj, "gW": gW, "handedness": OPR.HAND["sign"], "box_bottom": BOX_BOTTOM, "E_stat_8": E0}
    with R.weight_mode(cfg):
        core, _, _, _, _ = REC.core_reads(M, cfg, C.frame(M, nref)); core.pop("central_line", None)
    rec["core"] = {k: core[k] for k in core if k.startswith(("Delta_min", "r_0", "hV_min"))}
    log(f"{label}: g_W {gW}, E_stat {E0:.6f}, Delta_min {core['Delta_min_free']:.4f}, r_0 {core['r_0_profile (shell-mean lambda_1 = 0.8)']:.2f}")
    eps = 1e-3
    trials = []
    best = None
    for l, mm in ((2, 0), (2, 2), (2, -2)):
        for r_s in (1.5, 2.25, 3.0, 4.0, 6.0, 9.0):
            for wfac in (1.0, 2.0):
                w = wfac * 1.5 * cfg["h"]
                V, ab = OPR.pattern_field(l, mm, r_s, w, Ea, Eb, r, th, ph, free)
                Ep = R.energy_object(M + eps * V, cf8, None, nref, need_grad=False)[2]
                Em = R.energy_object(M - eps * V, cf8, None, nref, need_grad=False)[2]
                terms = ["E_h", "V4", "U_v6" if obj == "v6" else "U", "KP", "reg"]
                q = {t: (Ep[t] - 2 * pp0[t] + Em[t]) / (eps * eps) for t in terms}
                tt = 2.0 * float(np.sum(np.einsum("xyzi,xyzij,xyzj->xyz", ab, Tm, ab)))
                Om2 = sum(q.values()) / max(tt, 1e-300)
                row = {"l": l, "m": mm, "r_s": r_s, "width": w, "Omega2_upper_bound": Om2, "by_term": {t: q[t] / max(tt, 1e-300) for t in terms}}
                trials.append(row)
                if best is None or Om2 < best["Omega2_upper_bound"]:
                    best = row
        log(f"  pattern ({l},{mm}): Omega2 bounds " + " ".join(f"{t_['Omega2_upper_bound']:.4f}" for t_ in trials if t_['l'] == l and t_['m'] == mm))
    rec["trials"] = trials
    rec["best_trial"] = best
    rec["verdict"] = "BOUND_BELOW_BOX (a trial doublet below the empty box's bottom)" if best["Omega2_upper_bound"] < BOX_BOTTOM else "NO_TRIAL_BELOW_BOX (the core-localized sector bounded above the box bottom; the full spectrum's lowest mode not converged by Lanczos within the wall clock)"
    # the radial effective potential of (2, 0) per shell (the floor / well / K_P split with W in the well)
    rec["effective_potential"] = OPR.effective_potential(M, cfg, Ea, Eb, Tm, r, th, ph, nref, free, ls=(2, 3, 4))
    fl = rec["effective_potential"].get("connection_floor_by_shell", [])
    rec["well_negative_shells"] = [q["r_s"] for q in fl if q["well_Omega2 (V4 + U)"] < 0]
    log(f"  VERDICT {rec['verdict']}; best trial ({best['l']},{best['m']}) r_s {best['r_s']} width {best['width']:.2f}: {best['Omega2_upper_bound']:.4f} vs box {BOX_BOTTOM}; well negative on shells {rec['well_negative_shells']}; floor by shell " + " ".join(f"{q['floor_Omega2']:.4f}" for q in fl[:6]))
    rec["wall_s"] = time.time() - T0
    json.dump(rec, open(os.path.join(CK, f"r17_3b_{label}.json"), "w"), indent=1, default=float)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for (l, mm), mk in (((2, 0), "o"), ((2, 2), "s"), ((2, -2), "^")):
        for wfac, ls in ((1.0, "-"), (2.0, "--")):
            rows = [t_ for t_ in trials if t_["l"] == l and t_["m"] == mm and abs(t_["width"] - wfac * 1.5 * cfg["h"]) < 1e-9]
            ax[0].plot([t_["r_s"] for t_ in rows], [t_["Omega2_upper_bound"] for t_ in rows], mk + ls, ms=4, label=f"({l},{mm}) width {wfac:g}x")
    ax[0].axhline(BOX_BOTTOM, color="r", ls=":", lw=0.8, label="box bottom"); ax[0].set_xlabel("trial shell radius"); ax[0].set_ylabel("Rayleigh quotient (upper bound on Omega^2)"); ax[0].legend(fontsize=6); ax[0].set_title(f"{label}: trial doublets", fontsize=8)
    ax[1].plot([q["r_s"] for q in fl], [q["floor_Omega2"] for q in fl], "o-", ms=3, label="E_h connection floor"); ax[1].plot([q["r_s"] for q in fl], [q["well_Omega2 (V4 + U)"] for q in fl], "s-", ms=3, label="V4 + U_v6 well"); ax[1].plot([q["r_s"] for q in fl], [q["KP_Omega2"] for q in fl], "^-", ms=3, label="K_P")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].axvline(core["r_0_profile (shell-mean lambda_1 = 0.8)"], color="r", ls=":", lw=0.7, label="r_0"); ax[1].set_xlabel("shell radius"); ax[1].legend(fontsize=6); ax[1].set_title("(2,0) effective potential by term", fontsize=8)
    p = os.path.join(PLOTS, f"m5_32_r17_3b_{label}.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    rec["plot"] = rel(p)
    json.dump(rec, open(os.path.join(CK, f"r17_3b_{label}.json"), "w"), indent=1, default=float)
    log(f"  written checkpoints/m5_32_r17/r17_3b_{label}.json")
    return rec


def collect():
    out = {"rung": "R17-3b (bounds)", "runs": {}}
    for p in sorted(glob.glob(os.path.join(CK, "r17_3b_*.json"))):
        rr = json.load(open(p)); out["runs"][rr["label"]] = rr
    out["verdicts"] = {k: v["verdict"] for k, v in out["runs"].items()}
    out["best_bounds"] = {k: v["best_trial"]["Omega2_upper_bound"] for k, v in out["runs"].items()}
    out["well_negative_shells"] = {k: v["well_negative_shells"] for k, v in out["runs"].items()}
    json.dump(out, open(os.path.join(DATA, "m5_32_r17_3b.json"), "w"), indent=1, default=float)
    log(f"collected {len(out['runs'])}: {out['verdicts']}; best bounds {out['best_bounds']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "collect"])
    ap.add_argument("--field"); ap.add_argument("--object", default="v6"); ap.add_argument("--gW", type=float, default=1.35); ap.add_argument("--label", default="x")
    a = ap.parse_args(ARGS)
    if a.mode == "run":
        run(a.field, a.object, a.gW, a.label)
    else:
        collect()
