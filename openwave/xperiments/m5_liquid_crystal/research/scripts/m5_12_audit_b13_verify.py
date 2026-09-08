"""M5.12 block-13 ADVERSARIAL AUDIT, script 1: independent endpoint verification.

For every block-12 ladder endpoint state (r1,r2,r4,r8,x2,x4):
  - recompute a^2 (first-harmonic norm^2 on the free mask) vs the rung target
  - recompute S0, Q2 with the INDEPENDENT block-11 audit functional
    (m5_12_audit_b11_lib.my_s0_q2: own stencil, own inner_eta, own V) and the
    closed form omega_bal = sqrt(S0/-Q2); compare against the ladder JSON
  - verify the free-period condition c_omega = dShat/dw - Shat/w = 0 at
    omega_bal by FD of MY OWN Shat (L4: omega elimination exactness)
  - verify the claimant's domega_bal/dX rank-one formula by FD (L4 solver)
  - the L1 arithmetic: per-doubling omega ratios, Q2/a^2 exponents
  - the L2 mechanism probe: the mixing-channel share of Q2 per rung
  - recompute the endpoint |F| (residual on free DOF + phase row) for the
    L3 absolute-floor table (claimant instrument residual, audited BG-gated)

Run: /opt/anaconda3/envs/master312/bin/python m5_12_audit_b13_verify.py
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
from m5_12_d3a_bvp import residual                                 # noqa: E402

NR, NZ, H = 32, 64, 1.0
WSCALE = wscale_at(NR, NZ, H, 8.0 * NR / 96)
TAGS = ["r1", "r2", "r4", "r8", "x2", "x4"]

# claimant endpoint values (m5_12_b12_hard_ladder{,_x}.json)
CLAIM = {
    "r1": dict(a2=0.07593116171219504, w=32.804483995433884,
               S0=44.4452328399412, Q2=-0.04130082853089334, F=432.9786278097177),
    "r2": dict(a2=0.15186232342439007, w=22.6951943790761,
               S0=44.40521072223861, Q2=-0.08621168270588697, F=338.0374405544578),
    "r4": dict(a2=0.30372464684878014, w=15.507528337510594,
               S0=44.34207040782907, Q2=-0.18438721312449502, F=264.65716656203705),
    "r8": dict(a2=0.6074492936975603, w=10.40209640087479,
               S0=44.23212429617879, Q2=-0.4087860329895108, F=185.32373882820926),
    "x2": dict(a2=1.2148985955129903, w=7.333718160723068,
               S0=44.15428872976518, Q2=-0.8209646585848063, F=203.1331438842955),
    "x4": dict(a2=2.4297971910259806, w=6.3454550444148845,
               S0=45.41527488033425, Q2=-1.1279150820411843, F=948.4270062455894),
}


def free4(shape):
    f, pin = lib.free_mask(NR, NZ, shape)
    return f, pin


def a2_of(X, f):
    return float(np.sum(X["A"][0][f] ** 2) + np.sum(X["B"][0][f] ** 2))


def my_c_omega(X, w, eps=1e-5):
    """c_omega = dShat/dw - Shat/w with MY OWN Shat (FD derivative)."""
    sp = lib.my_shat(X, w + eps, H, WSCALE)
    sm = lib.my_shat(X, w - eps, H, WSCALE)
    s = lib.my_shat(X, w, H, WSCALE)
    return (sp - sm) / (2 * eps) - s / w


def endpoint_Fnorm(X, w, f):
    """|F| as the solver defines it: residual rows on free DOF + phase row
    (phase reference = the state's own A1 direction; the phase row value at
    the endpoint is what it is; report both with and without it)."""
    Rd, _ = residual(X, w, H, WSCALE)
    v = lib.rd_flat({"M0": np.where(f, Rd["M0"], 0.0),
                     "A": [np.where(f, a, 0.0) for a in Rd["A"]],
                     "B": [np.where(f, b, 0.0) for b in Rd["B"]]}, f)
    return float(np.linalg.norm(v))


def main():
    out = {"wscale": WSCALE, "rungs": {}}
    prev_w, prev_q2, prev_a2 = None, None, None
    for tag in TAGS:
        X, w_saved = lib.load_state(
            os.path.join(DATA, f"m5_12_b12_hard_{tag}_state.npz"))
        f, pin = free4(X["M0"].shape)
        # pins should carry zero harmonics (the block-11 rule)
        pin_h = max(float(np.max(np.abs(X["A"][k][pin]))) for k in range(2))
        a2 = a2_of(X, f)
        S0, Q2, Q2mix = lib.my_s0_q2(X, H, WSCALE)
        w_bal = float(np.sqrt(S0 / -Q2)) if Q2 < 0 else None
        c_om = my_c_omega(X, w_bal) if w_bal else None
        Fn = endpoint_Fnorm(X, w_bal, f)
        c = CLAIM[tag]
        rec = {
            "a2_target": c["a2"], "a2_measured": a2,
            "a2_rel_err": abs(a2 - c["a2"]) / c["a2"],
            "S0_mine": S0, "S0_claim": c["S0"],
            "S0_rel_err": abs(S0 - c["S0"]) / abs(c["S0"]),
            "Q2_mine": Q2, "Q2_claim": c["Q2"],
            "Q2_rel_err": abs(Q2 - c["Q2"]) / abs(c["Q2"]),
            "omega_bal_mine": w_bal, "omega_bal_claim": c["w"],
            "omega_saved_in_npz": w_saved,
            "c_omega_at_bal_mine": c_om,
            "c_omega_rel": abs(c_om) / (S0 / w_bal) if w_bal else None,
            "Q2_mix_share": Q2mix / Q2,
            "Q2_over_a2": Q2 / a2,
            "Fnorm_recomputed_f32": Fn, "Fnorm_claim_f64": c["F"],
        }
        if prev_w is not None:
            rec["omega_ratio_vs_prev"] = w_bal / prev_w
            rec["Q2_ratio_vs_prev"] = Q2 / prev_q2
            rec["a2_ratio_vs_prev"] = a2 / prev_a2
        prev_w, prev_q2, prev_a2 = w_bal, Q2, a2
        out["rungs"][tag] = rec
        print(f"[{tag}] a2={a2:.4f} (rel err {rec['a2_rel_err']:.1e}) "
              f"S0={S0:.4f} Q2={Q2:+.6f} w_bal={w_bal:.4f} "
              f"(claim {c['w']:.4f}) c_om_rel={rec['c_omega_rel']:.2e} "
              f"mixshare={rec['Q2_mix_share']:.3f} "
              f"|F|={Fn:.1f} (claim {c['F']:.1f}) pin_h={pin_h:.1e}")

    # L4: the rank-one domega_bal/dX formula vs FD of omega_bal (r4 state)
    X, _ = lib.load_state(os.path.join(DATA, "m5_12_b12_hard_r4_state.npz"))
    f, _ = free4(X["M0"].shape)
    rng = np.random.default_rng(13)
    D = {"M0": lib.np.zeros_like(X["M0"]),
         "A": [np.zeros_like(a) for a in X["A"]],
         "B": [np.zeros_like(b) for b in X["B"]]}
    for blk in ("M0",):
        d = rng.standard_normal(X["M0"].shape)
        D["M0"] = np.where(f, 0.5 * (d + np.swapaxes(d, -1, -2)), 0.0)
    for k in range(2):
        d = rng.standard_normal(X["M0"].shape)
        D["A"][k] = np.where(f, 0.5 * (d + np.swapaxes(d, -1, -2)), 0.0)
        d = rng.standard_normal(X["M0"].shape)
        D["B"][k] = np.where(f, 0.5 * (d + np.swapaxes(d, -1, -2)), 0.0)

    def wbal_of(Xd):
        S0, Q2, _ = lib.my_s0_q2(Xd, H, WSCALE)
        return float(np.sqrt(S0 / -Q2))

    def perturbed(sgn, eps):
        Xd = {"M0": X["M0"] + sgn * eps * D["M0"],
              "A": [X["A"][k] + sgn * eps * D["A"][k] for k in range(2)],
              "B": [X["B"][k] + sgn * eps * D["B"][k] for k in range(2)]}
        return Xd

    eps = 1e-6
    dw_fd = (wbal_of(perturbed(+1, eps)) - wbal_of(perturbed(-1, eps))) / (2 * eps)
    # claimant formula: dw.D = [ (R0.D)/P + (S0/P^2)(RQ.D) ] / (2w)
    S0, Q2, _ = lib.my_s0_q2(X, H, WSCALE)
    P, w = -Q2, wbal_of(X)
    R0d, _ = residual(X, 0.0, H, WSCALE)
    R1d, _ = residual(X, 1.0, H, WSCALE)

    def dot(Rd):
        s = float(np.sum(np.where(f, Rd["M0"], 0.0) * D["M0"]))
        for k in range(2):
            s += float(np.sum(np.where(f, Rd["A"][k], 0.0) * D["A"][k]))
            s += float(np.sum(np.where(f, Rd["B"][k], 0.0) * D["B"][k]))
        return s

    r0dot = dot(R0d)
    rqdot = r0dot - dot(R1d)
    dw_an = (r0dot / P + (S0 / P ** 2) * rqdot) / (2 * w)
    out["L4_rank_one_grad"] = {
        "dw_fd_of_my_omega_bal": dw_fd, "dw_claimant_formula": dw_an,
        "rel_err": abs(dw_fd - dw_an) / (abs(dw_fd) + 1e-300)}
    print(f"[L4 grad] FD={dw_fd:+.6e} formula={dw_an:+.6e} "
          f"rel={out['L4_rank_one_grad']['rel_err']:.2e}")

    with open(os.path.join(DATA, "m5_12_audit_b13_verify.json"), "w") as fjs:
        json.dump(out, fjs, indent=1)
    print("-> m5_12_audit_b13_verify.json")


if __name__ == "__main__":
    main()
