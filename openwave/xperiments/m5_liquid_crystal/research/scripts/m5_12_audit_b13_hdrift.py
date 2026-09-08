"""M5.12 block-13 ADVERSARIAL AUDIT, script 3: the H-drift metric (L5 + the
H-drift half of L3).

Attack: drift_rel = (max-min)/|mean| of H(t). At omega_bal the period-mean
of H is ZERO BY CONSTRUCTION: with T := (1/ns) SUM_j SUM w 4 SUM_i
ieta(F_0i-hat), Shat = S0 - w^2 T and Q2 = T, while
mean_t H = S0 + w^2 Q2, so at w = omega_bal = sqrt(S0/-Q2) the mean
vanishes IDENTICALLY (up to the ns=10 vs ns=20 V-term quadrature aliasing,
which grows with amplitude). drift_rel is therefore 90/(quadrature noise):
its "10x/rung improvement" can only track the aliasing in the denominator.

Checks, with MY OWN H (block-11 audit lib, ALL pairs +):
  1. per rung: H_mean, H_swing = max-min, drift_rel, |H_mean|/H_swing
  2. the identity: H_mean(ns) == S0(ns) + w^2 Q2(ns) to fp
  3. H_mean at ns matched to the solver quadrature (ns=10) vs BG5's ns=20:
     the denominator is pure quadrature choice
  4. the honest metric: absolute swing per rung: does 10x/rung survive?

Run: /opt/anaconda3/envs/master312/bin/python m5_12_audit_b13_hdrift.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATA = os.path.join(HERE, "..", "data")

ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import m5_12_audit_b11_lib as lib                                  # noqa: E402
from m5_12_d3b_newton import wscale_at                             # noqa: E402

NR, NZ, H = 32, 64, 1.0
WSCALE = wscale_at(NR, NZ, H, 8.0 * NR / 96)
TAGS = ["r1", "r2", "r4", "r8", "x2", "x4"]
CLAIM_DRIFT = {"r1": 3.89e11, "r2": 3.61e10, "r4": 3.28e9, "r8": 3.01e8,
               "x2": 4.21e7, "x4": 1.57e7}


def main():
    out = {"rungs": {}}
    print(f"{'rung':>4} {'w_bal':>8} {'H_mean(20)':>12} {'H_swing':>9} "
          f"{'drift_rel':>11} {'claim':>9} {'|mean|/swing':>12} "
          f"{'identity_err':>12} {'H_mean(10)':>12}")
    for tag in TAGS:
        X, _ = lib.load_state(os.path.join(DATA, f"m5_12_b12_hard_{tag}_state.npz"))
        # omega_bal from the solver's own quadrature convention (ns=10)
        S0_10, Q2_10, _ = lib.my_s0_q2(X, H, WSCALE, ns=10)
        w = float(np.sqrt(S0_10 / -Q2_10))
        Hs20 = lib.my_H_samples(X, w, H, WSCALE, ns=20)
        Hs10 = lib.my_H_samples(X, w, H, WSCALE, ns=10)
        S0_20, Q2_20, _ = lib.my_s0_q2(X, H, WSCALE, ns=20)
        mean20, swing = float(np.mean(Hs20)), float(np.max(Hs20) - np.min(Hs20))
        ident = S0_20 + w ** 2 * Q2_20
        rec = {"omega_bal": w,
               "H_mean_ns20": mean20, "H_min": float(np.min(Hs20)),
               "H_max": float(np.max(Hs20)), "H_swing_abs": swing,
               "drift_rel_mine": swing / abs(mean20),
               "drift_rel_claim": CLAIM_DRIFT[tag],
               "mean_over_swing": abs(mean20) / swing,
               "identity_S0_p_w2Q2_ns20": ident,
               "identity_minus_Hmean": ident - mean20,
               "H_mean_ns10_solver_quadrature": float(np.mean(Hs10)),
               "S0_ns10": S0_10, "S0_ns20": S0_20,
               "Q2_ns10": Q2_10, "Q2_ns20": Q2_20}
        out["rungs"][tag] = rec
        print(f"{tag:>4} {w:8.4f} {mean20:12.4e} {swing:9.3f} "
              f"{rec['drift_rel_mine']:11.3e} {CLAIM_DRIFT[tag]:9.2e} "
              f"{rec['mean_over_swing']:12.3e} "
              f"{rec['identity_minus_Hmean']:12.3e} "
              f"{rec['H_mean_ns10_solver_quadrature']:12.4e}")
    sw = [out["rungs"][t]["H_swing_abs"] for t in TAGS]
    out["swing_sequence"] = sw
    out["swing_ratios"] = [sw[i + 1] / sw[i] for i in range(len(sw) - 1)]
    print("absolute swing sequence:", " ".join(f"{s:.1f}" for s in sw))
    print("swing ratios (10x claim needs 0.1):",
          " ".join(f"{r:.3f}" for r in out["swing_ratios"]))
    with open(os.path.join(DATA, "m5_12_audit_b13_hdrift.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("-> m5_12_audit_b13_hdrift.json")


if __name__ == "__main__":
    main()
