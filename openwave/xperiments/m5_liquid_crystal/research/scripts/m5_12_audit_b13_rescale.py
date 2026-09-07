"""M5.12 block-13 ADVERSARIAL AUDIT, script 2: pure-rescaling null tests.

Attacks:
  L1  Is the 1/a "scaling law" a solved-branch property or a near-trivial
      property of first-harmonic rescaling? From ONE state (the mixfix chain
      seed), retract to every rung a2 with NO solving and compare
      omega_bal(a2) against the ladder endpoints.
  L2  Channel anatomy of the "Q2 saturation": along a fine rescaling ladder
      from the x2 and x4 endpoints, split Q2 into its eta-negative
      (time-mixing) and eta-positive parts; local exponent
      d ln|Q2| / d ln a2. Does the mixing channel saturate, or does the
      positive channel grow and cancel?
  L2c The "omega_bal floors near 6 / M5.8 band unreachable" corollary:
      extend the rescaling from the deepest states to a2 up to 64x the last
      rung; track min sqrt(S0/-Q2) and whether Q2 stays negative.

All functionals are the INDEPENDENT block-11 audit implementations.

Run: /opt/anaconda3/envs/master312/bin/python m5_12_audit_b13_rescale.py
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
RUNG_A2 = {"r1": 0.07593116171219504, "r2": 0.15186232342439007,
           "r4": 0.30372464684878014, "r8": 0.6074492936975603,
           "x2": 1.2148985955129903, "x4": 2.4297971910259806}
END_W = {"r1": 32.8045, "r2": 22.6952, "r4": 15.5075, "r8": 10.4021,
         "x2": 7.3337, "x4": 6.3455}


def load(path):
    X, w = lib.load_state(path)
    f, pin = lib.free_mask(NR, NZ, X["M0"].shape)
    for k in range(len(X["A"])):        # zero harmonics on pins (solver rule)
        X["A"][k][pin[..., ]] = 0.0
        X["B"][k][pin[..., ]] = 0.0
    return X, f


def a2_of(X, f):
    return float(np.sum(X["A"][0][f] ** 2) + np.sum(X["B"][0][f] ** 2))


def retract(X, f, a2_star):
    """first-harmonic-only rescale (the claimant's retraction)."""
    a2 = a2_of(X, f)
    s = np.sqrt(a2_star / a2)
    Y = {"M0": X["M0"].copy(),
         "A": [X["A"][0] * s] + [a.copy() for a in X["A"][1:]],
         "B": [X["B"][0] * s] + [b.copy() for b in X["B"][1:]]}
    return Y


def probe(X):
    S0, Q2, Q2mix = lib.my_s0_q2(X, H, WSCALE)
    w = float(np.sqrt(S0 / -Q2)) if Q2 < 0 else None
    return {"S0": S0, "Q2": Q2, "Q2_mix": Q2mix, "Q2_pos": Q2 - Q2mix,
            "omega_bal": w}


def main():
    out = {}

    # ---- L1 null: rescaling ladder from the chain SEED, no solving ----
    Xs, fs = load(os.path.join(DATA, "m5_12_d3b_breather_n32_mixfix_state.npz"))
    rows = {}
    print("== L1 null: mixfix seed retracted to each rung a2 (NO solving) ==")
    for tag, a2 in RUNG_A2.items():
        p = probe(retract(Xs, fs, a2))
        p["omega_bal_ladder_endpoint"] = END_W[tag]
        p["ratio_null_over_endpoint"] = (p["omega_bal"] / END_W[tag]
                                         if p["omega_bal"] else None)
        rows[tag] = p
        wtxt = (f"{p['omega_bal']:7.3f}" if p["omega_bal"]
                else "NONE(Q2>=0)")
        rtxt = (f"{p['ratio_null_over_endpoint']:.3f}"
                if p["ratio_null_over_endpoint"] else "n/a")
        print(f"  [{tag}] a2={a2:.3f} S0={p['S0']:8.3f} Q2={p['Q2']:+.5f} "
              f"w_null={wtxt}  vs endpoint {END_W[tag]:7.3f}  ratio {rtxt}")
    out["L1_null_from_seed"] = rows

    # ---- L2 anatomy: fine rescaling from x2 and x4 endpoints ----
    for src in ("x2", "x4"):
        Xe, fe = load(os.path.join(DATA, f"m5_12_b12_hard_{src}_state.npz"))
        base = a2_of(Xe, fe)
        grid = base * np.array([0.5, 0.7071, 1.0, 1.4142, 2.0, 2.83, 4.0,
                                5.66, 8.0, 11.3, 16.0, 32.0, 64.0])
        seq = []
        print(f"== L2 rescaling ladder from the {src} endpoint "
              f"(base a2={base:.3f}) ==")
        prev = None
        for a2 in grid:
            p = probe(retract(Xe, fe, float(a2)))
            p["a2"] = float(a2)
            if prev is not None and p["Q2"] < 0 and prev["Q2"] < 0:
                p["exp_lnQ2_lna2"] = (np.log(abs(p["Q2"]) / abs(prev["Q2"]))
                                      / np.log(p["a2"] / prev["a2"]))
                p["exp_mix"] = (np.log(abs(p["Q2_mix"]) / abs(prev["Q2_mix"]))
                                / np.log(p["a2"] / prev["a2"]))
            prev = p
            seq.append(p)
            wtxt = f"{p['omega_bal']:.3f}" if p["omega_bal"] else "NONE(Q2>=0)"
            print(f"  a2={a2:8.3f} S0={p['S0']:10.3f} Q2={p['Q2']:+10.5f} "
                  f"mix={p['Q2_mix']:+10.5f} pos={p['Q2_pos']:+9.5f} "
                  f"w_bal={wtxt}"
                  + (f" expQ2={p.get('exp_lnQ2_lna2', float('nan')):.2f}"
                     f" expMix={p.get('exp_mix', float('nan')):.2f}"
                     if "exp_lnQ2_lna2" in p else ""))
        ws = [p["omega_bal"] for p in seq if p["omega_bal"]]
        out[f"L2_rescale_from_{src}"] = {
            "rows": seq, "min_omega_bal": min(ws) if ws else None}
        print(f"  -> min omega_bal along the {src} rescaling: "
              f"{min(ws) if ws else None}")

    with open(os.path.join(DATA, "m5_12_audit_b13_rescale.json"), "w") as fjs:
        json.dump(out, fjs, indent=1)
    print("-> m5_12_audit_b13_rescale.json")


if __name__ == "__main__":
    main()
