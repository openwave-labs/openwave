"""M5.12 block-13 ADVERSARIAL AUDIT: verdict assembly.

Reads the three audit result JSONs (verify / rescale / hdrift / seed) and the
claimant ladder JSONs, and writes data/m5_12_audit_b13.json with per-claim
verdicts {L1..L5}. All numbers cited in the verdicts are produced by the
b13 audit scripts (independent block-11-lib functionals) or are arithmetic
on the claimant's own published records (marked as such).

Run: /opt/anaconda3/envs/master312/bin/python m5_12_audit_b13_verdicts.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def jload(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def main():
    ver = jload("m5_12_audit_b13_verify.json")
    res = jload("m5_12_audit_b13_rescale.json")
    hdr = jload("m5_12_audit_b13_hdrift.json")
    seed = jload("m5_12_audit_b13_seed.json")
    tags = ["r1", "r2", "r4", "r8", "x2", "x4"]
    R = ver["rungs"]

    q2 = {t: R[t]["Q2_mine"] for t in tags}
    mix = {t: R[t]["Q2_mix_share"] * q2[t] for t in tags}
    pos = {t: q2[t] - mix[t] for t in tags}

    # L3 arithmetic on the claimant's own ladder records
    rel = [0.7413687821824508, 0.32040595966905305, 0.1295218355586663,
           0.045668274800260755, 0.025342711357990412, 0.05957803365681135]
    Fend = [432.9786278097177, 338.0374405544578, 264.65716656203705,
            185.32373882820926, 203.1331438842955, 948.4270062455894]
    F0 = [f / r for f, r in zip(Fend, rel)]

    sw = hdr["swing_sequence"]
    relax = seed["L4_fresh_seed_relax"]
    sl = seed["L4_fresh_seed_level"]
    adj = max(a["rel"] for a in seed["L4_adjoint_r4"])

    out = {
        "task": "M5.12 block 12",
        "role": "INDEPENDENT ADVERSARIAL AUDIT (block 13) of the block-12 "
                "omega-eliminated hard-amplitude ladder",
        "date": "2026-07-08",
        "scripts": ["m5_12_audit_b13_verify.py", "m5_12_audit_b13_rescale.py",
                    "m5_12_audit_b13_hdrift.py", "m5_12_audit_b13_seed.py",
                    "m5_12_audit_b13_verdicts.py"],
        "audit_instruments": "independent block-11 audit functionals "
                             "(m5_12_audit_b11_lib: own stencil, own "
                             "inner_eta, own V, own H); the claimant d3a "
                             "residual is used only where the audited "
                             "BG1-gated instrument is itself the object "
                             "(|F| recomputation, relax driver), stated "
                             "per check",
        "claims": {},
    }

    out["claims"]["L1_scaling_law"] = {
        "verdict": "REFUTED as a seed-independent amplitude law (the "
                   "pre-registered chain-dependence criterion fired); "
                   "survives only as a single-chain regularity plus a "
                   "seed-INDEPENDENT S0 ~ 44.5 pinning",
        "claim": "omega_bal falls as pure 1/a across the ladder (Q2 "
                 "prop a^2, S0 pinned ~45, retraction shock relaxes away "
                 "every rung)",
        "evidence": [
            "DATA CONFIRMED: every endpoint a2/S0/Q2/omega_bal reproduced "
            "by the independent functional to <=1e-6 rel (a2 to 4e-8); "
            "per-doubling omega ratios 0.692/0.683/0.671/0.705/0.865 as "
            "published.",
            "NOT retraction-trivial (the null test PASSES in the "
            "claimant's favor): pure first-harmonic rescaling of the chain "
            "seed does NOT reproduce the law: omega_null/omega_endpoint = "
            "1.20/1.32/1.73/4.45 at r1-r8 and the balance root is "
            "DESTROYED (Q2 >= 0) at the x2/x4 amplitudes. The Newton "
            "relaxation is load-bearing; the law is a property of the "
            "relaxed endpoints.",
            "BUT 'pure 1/a / Q2 prop a^2' overstates: endpoint Q2 "
            "per-doubling ratios run 2.09/2.14/2.22/2.01: systematically "
            "SUPER-quadratic through x16 (Q2/a2 drifts -0.544 to -0.676, "
            "+24%), which is why omega ratios run below 1/sqrt(2); and "
            "'S0 relaxes away every rung' FAILS at x4 (endpoint S0 45.42 "
            "vs the chain's 44.15, never relaxed).",
            "ALL SIX endpoints are unconverged (best rel 2.5e-2, worst "
            "0.741 at r1: the first point of the 'law' is a barely-relaxed "
            "state) and the ladder is ONE warm-start chain.",
            "THE KILL (fresh-seed check, r4 amplitude): the canonical bmix "
            f"seed at a2=0.3037 (eps*={sl['eps_star']:.4f}) relaxed with MY "
            "independent solver reaches a BETTER floor than the chain's r4 "
            "endpoint (|F| 229 vs 265; rel 2.0e-2 vs 1.3e-1) with "
            "omega_bal SETTLED at "
            + " -> ".join(f"{r['omega_bal']:.2f}" for r in relax)
            + " vs the chain's 15.51 at the SAME amplitude: 1.76x apart. "
            "S0 relaxes to ~44.6 on BOTH branches (the pinning is "
            "seed-independent), but Q2 at fixed a2 is state-dependent by "
            "3.1x (-0.572 vs -0.184). omega_bal(a2) is not a function of "
            "amplitude; the published curve is a property of one "
            "warm-start chain.",
        ],
        "surviving_form": "Within the one chain, the relaxed endpoints "
            "show |Q2| growing ~quadratically and omega ratios 0.67-0.71 "
            "per doubling: a real single-chain regularity. Seed-"
            "independent survivors: S0 relaxes to ~44.5 regardless of "
            "seed; Q2 < 0 with |Q2| growing with amplitude (qualitative); "
            "omega_bal DECREASES with amplitude on both branches. The "
            "quantitative law (the coefficient, hence any predicted "
            "omega at a given a2) does not transfer across seeds.",
        "numbers": {
            "omega_ratios": [0.6919, 0.6833, 0.6708, 0.7050, 0.8652],
            "Q2_ratios": [2.0874, 2.1388, 2.2170, 2.0083, 1.3739],
            "Q2_over_a2": [R[t]["Q2_over_a2"] for t in tags],
            "null_rescale_omega_over_endpoint": {
                t: res["L1_null_from_seed"][t]["ratio_null_over_endpoint"]
                for t in tags},
            "endpoint_F_rel": rel,
        },
    }

    out["claims"]["L2_Q2_saturation"] = {
        "verdict": "WEAKENED (bend datum real but from a transient state; "
                   "claimed mechanism REFUTED in its channel attribution; "
                   "floor corollary stays a hypothesis, with new "
                   "independent structural support)",
        "claim": "the x32 softening (ratio 0.87, Q2 ratio 1.37) = "
                 "mixing-channel saturation, NOT S0 growth; omega_bal may "
                 "floor near 6: M5.8 band unreachable in this seed class",
        "evidence": [
            "Bend arithmetic CONFIRMED: omega x4/x2 = 0.8652 = "
            "sqrt((S0 ratio)/(Q2 ratio)); S0 growth contributes only a "
            "1.4% factor (so 'NOT S0 growth' holds arithmetically), the "
            "Q2 shortfall contributes 1.21x.",
            "MECHANISM REFUTED: the mixing (eta-negative, 0i) channel "
            "does NOT saturate. Endpoint Q2_mix per-doubling ratios "
            "2.12/2.19/2.27/2.14/1.92: still ~quadratic at the last "
            "doubling; under pure rescaling its exponent in a2 is 1.00 "
            "EXACTLY at every tested amplitude. The net-Q2 bend comes "
            "from the eta-POSITIVE time-pair channel growing "
            "super-quadratically (endpoint ratios 3.4/3.5/3.0/3.7/5.4; "
            "pos/|mix| 0.022 -> 0.380) and cancelling the mixing "
            "channel. 'Mixing-channel saturation' names the wrong "
            "channel: it is positive-channel CANCELLATION growth.",
            "The claimant's own counter-hypothesis (LSMR stall) is "
            "SUPPORTED: the x4 endpoint Q2 (-1.128) is SHALLOWER than "
            "its own rung-open retracted value (-1.282); the last "
            "accepted step moved Q2 by 72% (-0.656 -> -1.128); S0 never "
            "relaxed (45.4 vs 44.15). The x4 point is a solver "
            "transient: the 1.37 'saturation ratio' is not endpoint-"
            "grade data.",
            "FLOOR COROLLARY: independent structural support at the "
            "functional level: because pos grows with exponent ~2 in a2 "
            "vs mix exponent 1.00, the balance root is DESTROYED "
            "(Q2 >= 0) at a2 ~ 5-7 under rescaling from BOTH the x2 and "
            "x4 endpoints; min omega_bal along every tested rescaling "
            "path = the endpoint value itself (6.35). Reaching the M5.8 "
            "band (1.07-1.15) needs |Q2| ~ 37 at S0 ~ 45: two orders "
            "beyond anything observed with the net sign already dying. "
            "BUT relaxation reshapes states (it deepened Q2 at every "
            "rung), so the corollary remains an extrapolation: correctly "
            "kept audit-gated by the claimant.",
        ],
        "surviving_form": "Net Q2 does go sub-quadratic at x32 and an "
            "omega_bal floor well above the M5.8 band is plausible in "
            "this seed class, but via eta-positive cancellation growth, "
            "not mixing saturation, and the quantitative bend (1.37, "
            "floor ~6) cannot be trusted from a 2-step transient.",
        "numbers": {
            "Q2_mix_endpoints": [mix[t] for t in tags],
            "Q2_pos_endpoints": [pos[t] for t in tags],
            "pos_over_mix": [pos[t] / abs(mix[t]) for t in tags],
            "rescale_root_death_a2": "Q2 >= 0 at a2 = 6.88 (from x2) and "
                                     "6.88 (from x4); still negative at "
                                     "4.86",
            "x4_open_vs_end_Q2": [-1.282235, -1.127915],
        },
    }

    out["claims"]["L3_more_solvable"] = {
        "verdict": "WEAKENED (rel-floor framing dishonest as published; "
                   "absolute improvement survives only to r8; the H-drift "
                   "half is REFUTED, see L5)",
        "claim": "larger amplitude = more solvable: REL floor "
                 "0.741 -> 0.025 and H-drift improves ~10x/rung",
        "evidence": [
            "The rel denominators |F|0 = rung-open norms grow ~2.0x per "
            "rung (584/1055/2043/4058/8015/15919: ratios 1.81/1.94/1.99/"
            "1.98/1.99), tracking the retraction shock, i.e. the a2 "
            "target itself. Half of every 'rel improvement' decade is "
            "denominator inflation.",
            "ABSOLUTE endpoint floors (claimant's own records, confirmed "
            "by my float32-state recomputation 433.6/338.8/265.7/186.9/"
            "205.3/948.4): 433/338/265/185/203/948: improve 0.78/0.78/"
            "0.70 per rung through r8, then REVERSE (+10% at x2, x4.7 at "
            "x4). 'Larger amplitude = more solvable' is false beyond "
            "a2 = 0.607 in absolute terms.",
            "H-drift 10x/rung: REFUTED: pure denominator artifact of a "
            "mean that is zero by construction (L5).",
            "The claimant pre-flagged this exact angle at close: honest "
            "flag, dishonest headline table.",
        ],
        "surviving_form": "A modest genuine absolute-floor improvement "
            "(~25%/rung) through the first four doublings, plus larger "
            "opening Newton steps per rung. Nothing survives past r8.",
        "numbers": {"F0_sequence": F0, "Fend_sequence": Fend,
                    "rel_sequence": rel,
                    "abs_ratios": [Fend[i + 1] / Fend[i] for i in range(5)]},
    }

    out["claims"]["L4_solver_and_ladder_status"] = {
        "verdict": "solver machinery CONFIRMED; one-chain ladder "
                   "meaningfulness WEAKENED (fresh-seed check: see "
                   "numbers; the block-11 C1 critique still binds)",
        "claim": "omega elimination exact (HG2); ladder meaningful "
                 "despite one warm-start chain",
        "evidence": [
            "Omega elimination CONFIRMED independently: with MY OWN Shat "
            "(independent stencil/inner_eta/V), c_omega(X, omega_bal) / "
            "(S0/omega) = 8.0e-10 .. 6.7e-11 at ALL six endpoints: every "
            "iterate satisfies the free-period condition to fp, as "
            "claimed.",
            "The rank-one domega_bal/dX formula CONFIRMED: FD of my own "
            "omega_bal vs the claimant formula converges O(eps^2) to the "
            "formula (rel 1.1e-4 / 2.8e-5 / 7.1e-6 at eps 1e-4 / 5e-5 / "
            "2.5e-5): pure FD truncation, no formula error.",
            f"Adjoint consistency of the claimant operator (matvec = FD "
            f"of composed map vs rmatvec = Hessian FD + rank-one + "
            f"phase) at the r4 endpoint: worst rel {adj:.2e} over 3 "
            "random pairs: the LSMR pair is consistent (no repeat of the "
            "block-11 w_amp adjoint bug).",
            "ONE-CHAIN CRITIQUE UPHELD: the fresh bmix seed at the r4 "
            f"amplitude (a2=0.3037, eps*={sl['eps_star']:.4f}) sits at "
            f"seed level at S0={sl['S0']:.2f}, Q2={sl['Q2']:.4f}, "
            f"omega_bal={sl['omega_bal']:.2f}; after 4 Newton steps of MY "
            "independent solver (same math, fresh code, LSMR 60) it "
            "settles at omega_bal "
            + " -> ".join(f"{r['omega_bal']:.2f}" for r in relax)
            + f" (endpoint rel {relax[-1].get('rel', 1.0):.3f}, |F| 229 "
            "vs the chain r4's 265) vs the chain's 15.51: the ladder's "
            "quantitative content is chain-specific (see L1). Seed-level "
            "scan (r1/r4/x2 amplitudes): fresh-seed omega_bal 19.23 / "
            "11.03 / 10.61 vs chain endpoints 32.80 / 15.51 / 7.33.",
        ],
        "surviving_form": "The solver is sound and does what is claimed "
            "(exact omega elimination, exact retraction, consistent "
            "adjoint pair). The LADDER remains one chain of unconverged "
            "stall points; its numbers do not transfer.",
        "numbers": {
            "c_omega_rel_all_endpoints": [R[t]["c_omega_rel"] for t in tags],
            "rank_one_grad_rel": ver["L4_rank_one_grad"]["rel_err"],
            "adjoint_worst_rel": adj,
            "fresh_seed_level": sl,
            "fresh_seed_relax_traj": relax,
        },
    }

    out["claims"]["L5_H_drift_metric"] = {
        "verdict": "REFUTED",
        "claim": "drift_rel = (max-min)/|mean| of H(t) is a solution-"
                 "quality metric whose ~10x/rung improvement means the "
                 "ladder improves",
        "evidence": [
            "At omega = omega_bal the period-mean of H is ZERO BY "
            "CONSTRUCTION: mean_t H = S0 + omega^2 Q2 (same quadrature), "
            "and omega_bal^2 = S0/(-Q2) makes it vanish identically. "
            "Verified: with the solver's own ns=10 quadrature my H_mean "
            "= O(1e-14) at every endpoint; the identity H_mean = S0 + "
            "w^2 Q2 holds to 3e-14 at ns=20.",
            "The reported nonzero means (-2.3e-10 -> -8.9e-6) are pure "
            "quadrature aliasing: BG5 samples at ns=20 while omega_bal "
            "is defined by the ns=10 functional, and the degree-16 V "
            "harmonics alias differently; the aliasing GROWS ~10x/rung "
            "with amplitude. drift_rel therefore divides a ~90-unit "
            "physical swing by an amplitude-growing rounding error: its "
            "'10x/rung improvement' is exactly the 10x/rung GROWTH of "
            "the denominator noise, and carries zero information about "
            "solution quality.",
            "My independent H(t) reproduces the claimant's drift_rel "
            "values (3.93e11/3.61e10/3.28e9/3.01e8/4.21e7/1.57e7 vs "
            "published 3.9e11/3.6e10/3.3e9/3.0e8/4.2e7/1.6e7): the DATA "
            "is honest; the METRIC is degenerate.",
            "The honest absolute metric, H_swing = max-min: "
            + "/".join(f"{s:.1f}" for s in sw)
            + " across the six rungs: FLAT for four rungs then WORSE "
            "(+4.4%, +48%). No improvement at any rung. The swing is "
            "~2x S0 at every rung: none of these states is anywhere "
            "near conserving H.",
        ],
        "surviving_form": "H_min/H_max themselves are honest numbers; "
            "any future use must report absolute swing (or swing/S0), "
            "never drift_rel, at omega_bal.",
        "numbers": {
            "H_mean_ns10": [hdr["rungs"][t]["H_mean_ns10_solver_quadrature"]
                            for t in tags],
            "identity_err_ns20": [hdr["rungs"][t]["identity_minus_Hmean"]
                                  for t in tags],
            "mean_over_swing": [hdr["rungs"][t]["mean_over_swing"]
                                for t in tags],
            "H_swing_abs": sw,
            "swing_ratios": hdr["swing_ratios"],
        },
    }

    with open(os.path.join(DATA, "m5_12_audit_b13.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("verdicts -> m5_12_audit_b13.json")
    for k, v in out["claims"].items():
        print(f"  {k}: {v['verdict']}")


if __name__ == "__main__":
    main()
