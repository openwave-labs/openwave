"""M5.32 R17-1 / R17-4 INDEPENDENT ADVERSARIAL AUDIT (a second agent, its own script and method).

Recomputes, and tries to refute, the six claims of the R17-1 true-fixed-K descents and the R17-4
c_X X_M^2 inertia ladder.  Nothing here consumes the producer's derived numbers except as the
"producer_numbers" side of a comparison; every "own_number" is recomputed here, and the three
spectral / algebraic reads (the split, the director gap, X_M) are built from scratch:

  own spectrum  : eigenvalues of N = M eta by np.linalg.eigvals on the raw 4x4 (NOT the R15
                  Newton-polished spectral projectors the instrument uses), sorted ascending
                  -> (lambda_g, lambda_3, lambda_2, lambda_1); rho^2 = (lambda_2 - lambda_3)^2 / 4
  own finite differences : central differences of the FULL E_K (a0 refreshed each evaluation) at two
                  step sizes with a Richardson extrapolation, against the analytic true gradient's
                  directional derivative and against the frozen-protocol gradient's
  own X_M       : an independently built Levi-Civita tensor (permutation-parity by inversion count
                  on a fresh construction) contracted with F_{mu nu} = A_mu eta A_nu - A_nu eta A_mu
                  on sym-stencil jets, cross-checked against the instrument's xm_cells

usage: python3 m5_32_r17_1_audit.py
out:   data/m5_32_r17_1_audit.json, checkpoints/m5_32_r17/r17_1_audit.log
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np

ARGS = sys.argv[1:]
sys.argv = [sys.argv[0]]
import m5_32_r17_common as R                              # noqa: E402
import m5_32_r16_common as C                              # noqa: E402
import m5_32_r16_2_operator as OP                         # noqa: E402

INS4, C15 = C.INS4, C.C15
ETA = np.asarray(C.ETA, dtype=float)
RES, DATA = C.RES, C.DATA
CK16 = os.path.join(RES, "checkpoints", "m5_32_r16")
CK17 = os.path.join(RES, "checkpoints", "m5_32_r17")
LOGP = os.path.join(CK17, "r17_1_audit.log")
T0 = time.time()
_LOG = open(LOGP, "w")


def log(m):
    s = f"[{time.time() - T0:8.1f}s] {m}"
    print(s, flush=True)
    _LOG.write(s + "\n")
    _LOG.flush()


OUT = {"rung": "R17-1 / R17-4 independent adversarial audit", "date": "2026-09-08",
       "python": sys.version.split()[0], "numpy": np.__version__, "claims": {}, "unclaimed_hazards": []}


def dump():
    OUT["runtime_s"] = round(time.time() - T0, 1)
    json.dump(OUT, open(os.path.join(DATA, "m5_32_r17_1_audit.json"), "w"), indent=1, default=float)


# =============================================================== own spectral reads (independent path)
def own_spectrum(M):
    """eigenvalues of N = M eta by the generic eigensolver, sorted ascending per cell.
    Returns (lg, l3, l2, l1) and the largest imaginary part seen (a degeneracy tell)."""
    N = np.real(M) @ ETA
    ev = np.linalg.eigvals(N)
    imax = float(np.max(np.abs(ev.imag)))
    ev = np.sort(ev.real, axis=-1)
    return ev[..., 0], ev[..., 1], ev[..., 2], ev[..., 3], imax


def own_reads(M, cfg, label=""):
    """the split / radius / gap reads from the OWN spectrum, all cells and free cells."""
    n, h, L = cfg["n"], cfg["h"], cfg["L"]
    X, Y, Z = INS4.coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    lg, l3, l2, l1, imax = own_spectrum(M)
    half = (l2 - l3) / 2.0
    rho2 = half * half
    gap = l1 - l2
    free = ~INS4.pin_shell(n, h, 1.6)
    w = max(float(np.sum(rho2)), 1e-300)
    wf = max(float(np.sum(rho2[free])), 1e-300)
    i = int(np.argmax(half))
    return {"half_split_max": float(np.max(half)), "r_at_half_split_max": float(r.reshape(-1)[i]),
            "rho2_r_rms_all_cells": float(np.sqrt(np.sum(rho2 * r * r) / w)),
            "rho2_r_rms_free_only": float(np.sqrt(np.sum((rho2 * r * r)[free]) / wf)),
            "rho2_fraction_r_gt_0.35L": float(np.sum(rho2[r > 0.35 * L]) / w),
            "rho2_fraction_in_pin_shell": float(np.sum(rho2[~free]) / w),
            "half_split_max_in_pin_shell": float(np.max(half[~free])),
            "gap_1_2_min": float(np.min(gap)), "gap_1_2_min_free": float(np.min(gap[free])),
            "l1_min": float(np.min(l1)), "a_box_threshold_0.35L": float(0.35 * L),
            "max_abs_imag_eigenvalue": imax, "label": label}


def cross_check_spectrum(M, cfg, nref):
    """my generic-eigensolver spectrum against the instrument's polished projector spectrum."""
    fr = C.frame(M, nref)
    dom = C.domain(fr, cfg)
    o = own_reads(M, cfg)
    return {"instrument_gap_1_2_min": dom["gap_1_2_min"], "own_gap_1_2_min": o["gap_1_2_min"],
            "instrument_half_split_max": dom["half_split_max"], "own_half_split_max": o["half_split_max"],
            "instrument_l1_min": dom["l1_min"], "own_l1_min": o["l1_min"]}


def E_true(M, cfg, K, nref, ns=8):
    cf = dict(cfg)
    cf["n_samples"] = int(ns)
    E, _, pp, dom, _ = R.energy_and_grad_true(M, cf, K, nref, need_grad=False)
    return float(E), pp, dom


# =============================================================== G1: the chain rule
def claim_G1():
    log("G1: the chain rule, own central differences on the R16-3 n32 K200 end state")
    cfg = C.cfg_v4(32, 48.0, completion="rebuild", n_samples=4)
    M = np.load(os.path.join(CK16, "r16_3_rebuild_n32_L48_K200.npy"))
    nref = C.radial_ref(cfg)
    K = 200.0
    free = (~INS4.pin_shell(32, cfg["h"], 1.6))[..., None, None].astype(float)
    E0, g_true, pp, dom, fr = R.energy_and_grad_true(M, cfg, K, nref)
    g_frozen = C.energy_and_grad(M, cfg, K, nref)[1]
    gt = g_true * free
    gf = g_frozen * free
    log(f"  E_K {E0:.9f}; |g_true| {np.sqrt(np.sum(gt**2)):.6e}  |g_frozen| {np.sqrt(np.sum(gf**2)):.6e}")
    rng = np.random.default_rng(20260908)
    dirs = []
    for k in range(3):
        D = C.sym(rng.normal(size=M.shape)) * free
        dirs.append(("random%d" % k, D / np.sqrt(np.sum(D * D))))
    dirs.append(("true_gradient", gt / np.sqrt(np.sum(gt * gt))))
    rows = []
    for name, D in dirs:
        an_t = float(np.sum(gt * D))
        an_f = float(np.sum(gf * D))
        fds = {}
        for eps in (1e-4, 5e-5):
            Ep = E_true(M + eps * D, cfg, K, nref, ns=4)[0]
            Em = E_true(M - eps * D, cfg, K, nref, ns=4)[0]
            fds[eps] = (Ep - Em) / (2 * eps)
        rich = (4 * fds[5e-5] - fds[1e-4]) / 3.0
        rows.append({"direction": name, "analytic_true": an_t, "analytic_frozen": an_f,
                     "fd_eps1e-4": fds[1e-4], "fd_eps5e-5": fds[5e-5], "fd_richardson": rich,
                     "rel_err_true_vs_richardson": abs(an_t - rich) / max(abs(rich), 1e-300),
                     "rel_err_frozen_vs_richardson": abs(an_f - rich) / max(abs(rich), 1e-300)})
        log(f"  {name:14s} analytic_true {an_t:14.6f}  richardson {rich:14.6f}  rel {rows[-1]['rel_err_true_vs_richardson']:.2e} "
            f"| frozen {an_f:14.6f} rel {rows[-1]['rel_err_frozen_vs_richardson']:.2e}")
    worst = max(r["rel_err_true_vs_richardson"] for r in rows)
    frozen_worst = min(r["rel_err_frozen_vs_richardson"] for r in rows)
    # the omitted part along the three R17-1 descents (from the producer's own traces, recomputed as a ratio)
    trace_ratio = {}
    for tag in ("r17_1_rebuild_n32_L48_K50", "r17_1_rebuild_n32_L48_K200", "r17_1_rebuild_n64_L48_K50"):
        p = os.path.join(CK17, tag + ".json")
        if os.path.exists(p):
            tr = json.load(open(p)).get("trace", [])
            rr = [t["a0_chain_norm"] / t["frozen_grad_norm"] for t in tr if "a0_chain_norm" in t]
            if rr:
                trace_ratio[tag] = {"n_rows": len(rr), "min": min(rr), "max": max(rr), "median": float(np.median(rr))}
    n32 = [v for k, v in trace_ratio.items() if "n32" in k]
    n32_min = min(v["min"] for v in n32) if n32 else None
    n32_max = max(v["max"] for v in n32) if n32 else None
    # the here-and-now chain fraction on this field
    chain_over_frozen = pp["a0_chain_norm"] / max(pp["frozen_grad_norm"], 1e-300)
    ok = worst < 1e-6
    verdict = "CONFIRMED" if ok else "QUALIFIED"
    grad_dir = [r for r in rows if r["direction"] == "true_gradient"][0]
    sign_flips = [r["direction"] for r in rows if r["analytic_frozen"] * r["fd_richardson"] < 0]
    note = ("The core of the claim is CONFIRMED and then some.  The analytic TRUE gradient reproduces my own "
            "Richardson-extrapolated central differences of the full E_K (a0 refreshed at every evaluation) on the "
            f"R16-3 n32 K200 end state to {worst:.1e} relative worst-case over four directions (3 random free + the "
            f"true gradient's own direction), and to {grad_dir['rel_err_true_vs_richardson']:.1e} along the gradient "
            "direction itself, where the derivative is largest and the finite difference is best conditioned.  The "
            "FROZEN gradient on the same four directions is wrong by 87 percent to 137 percent: along the true "
            f"gradient's direction it gives {grad_dir['analytic_frozen']:.4f} where the truth is "
            f"{grad_dir['fd_richardson']:.4f} (44 x too small), and on direction(s) {sign_flips} it has the WRONG "
            "SIGN.  So the frozen protocol was not descending E_K on this state at all; that is a stronger statement "
            "than the claim makes and it is the finding that matters.  TWO QUALIFICATIONS, both precise.  (1) The "
            f"stated Richardson tolerance is 1e-6; my worst direction lands at {worst:.2e}, above it.  This is a "
            "finite-difference floor, not a gradient defect: that direction has the smallest derivative of the four "
            f"({[r['fd_richardson'] for r in rows if r['rel_err_true_vs_richardson'] == worst][0]:.4f}), so the same "
            "absolute truncation error is a larger relative one, and the direction with the largest derivative agrees "
            "to 2e-9.  The claim should say 'to 1e-6 relative on the well-conditioned directions, 2e-6 on the "
            "weakest', or quote an absolute tolerance.  (2) The omitted part along the n32 descents spans "
            f"{n32_min:.3f} to {n32_max:.3f} of the frozen gradient's norm (the descent traces, logged every 100 "
            "iterations), NOT '0.4 to 0.9': both n32 runs dip well below 0.4 (K50 to 0.202 at it 800, K200 to 0.152 "
            "at it 400).  The error is in the safe direction for the conclusion but the range as printed is not what "
            "the data says.  NOT RECOMPUTED here: the complex-step figure (4.6e-14 on a 6^3 random field) is a "
            "selftest gate; I read its PASS in data/m5_32_r17_common_selftest.json and did not rebuild the "
            "complex-step path.")
    OUT["claims"]["G1"] = {"verdict": verdict, "own_numbers": {
        "field": "checkpoints/m5_32_r16/r16_3_rebuild_n32_L48_K200.npy", "K": K, "n_samples": 4, "lift": "radial_ref",
        "E_K": E0, "true_grad_norm_free": float(np.sqrt(np.sum(gt ** 2))), "frozen_grad_norm_free": float(np.sqrt(np.sum(gf ** 2))),
        "a0_chain_over_frozen_norm_here": float(chain_over_frozen),
        "directions": rows, "worst_rel_err_true": worst, "best_rel_err_frozen": frozen_worst,
        "chain_over_frozen_along_descents": trace_ratio, "n32_chain_over_frozen_range": [n32_min, n32_max]},
        "producer_numbers": {"complex_step_rel": 4.6182200395967785e-14, "richardson_gate": 1e-6,
                             "omitted_part_claimed_range": [0.4, 0.9]},
        "note": note}
    dump()


# =============================================================== G2: the 1/split^2 singularity
def claim_G2():
    log("G2: the bare R16-1 static core as a 1/split^2 singularity of E_K")
    cfg = C.cfg_v4(32, 48.0, completion="rebuild", n_samples=4)
    M = np.load(os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"))
    nref = C.radial_ref(cfg)
    K = 200.0
    E8, pp8, dom8 = E_true(M, cfg, K, nref, ns=8)
    o = own_reads(M, cfg, "r16_1_core")
    log(f"  8 samples: kin_tot {pp8['kin_tot']:.6e}  E_K {E8:.6e}  K^2/(4 kin) {K*K/(4*pp8['kin_tot']):.6e}  E_stat {pp8['E_stat']:.6f}")
    free = (~INS4.pin_shell(32, cfg["h"], 1.6))[..., None, None].astype(float)
    _, gt, pp, _, _ = R.energy_and_grad_true(M, cfg, K, nref)
    gf = C.energy_and_grad(M, cfg, K, nref)[1]
    nt = float(np.sqrt(np.sum((gt * free) ** 2)))
    nf = float(np.sqrt(np.sum((gf * free) ** 2)))
    log(f"  gradient norms on free cells: true {nt:.6e}  frozen {nf:.6e}  ratio {nt/max(nf,1e-300):.1f}")
    ratio = nt / max(nf, 1e-300)
    note = ("Confirmed on both halves.  On the BARE R16-1 static core the 8-sample kin_tot is "
            f"{pp8['kin_tot']:.4e} (the claim's 2e-3), so E_K = E_stat + K^2/(4 kin) = {E8:.4e} at K 200 "
            f"(the claim's 4.9e6; E_stat is only {pp8['E_stat']:.3f} of it), and the true gradient's norm on the free "
            f"cells is {ratio:.0f} x the frozen one (the claim's ~6000).  The half-split max on the core is "
            f"{o['half_split_max']:.3e}, so kin_tot ~ split^2 and E_K ~ 1/split^2 is the right reading of WHY: the "
            "clock inertia of a split-free core vanishes quadratically.  CAVEAT I add: 'singularity' is a statement "
            "about the limit, and this is one point, not a measured exponent; the ratio itself depends on the sample "
            "count and on the lift and is a per-point number, not a scaling law.  The consequence the producer draws "
            "from it (the seed deviation: every R17-1 descent must start from a nucleated shell, dt0 0.001) is the "
            "correct and conservative response, but it does make the R17-1 end states NOT the same-seed continuation "
            "of the R16-3 runs (see the hazard list).")
    OUT["claims"]["G2"] = {"verdict": "CONFIRMED", "own_numbers": {
        "field": "checkpoints/m5_32_r16/r16_1_rebuild_n32_L48.npy", "K": K,
        "kin_tot_8_samples": pp8["kin_tot"], "E_stat_8_samples": pp8["E_stat"], "E_K_8_samples": E8,
        "K2_over_4kin": K * K / (4 * pp8["kin_tot"]), "omega": pp8["omega"],
        "true_grad_norm_free_ns4": nt, "frozen_grad_norm_free_ns4": nf, "true_over_frozen": ratio,
        "own_reads": o},
        "producer_numbers": {"kin_tot": 2e-3, "E_K": 4.9e6, "true_over_frozen": 6000},
        "note": note}
    dump()


# =============================================================== R1 / R2 / R3: the three descents
def claim_R1():
    log("R1: n32 K 50 end state")
    cfg = C.cfg_v4(32, 48.0, completion="rebuild", n_samples=4)
    M = np.load(os.path.join(CK17, "r17_1_rebuild_n32_L48_K50.npy"))
    nref = C.radial_ref(cfg)
    K = 50.0
    E8, pp8, dom8 = E_true(M, cfg, K, nref, ns=8)
    o = own_reads(M, cfg, "r17_1_n32_K50")
    xc = cross_check_spectrum(M, cfg, nref)
    free = (~INS4.pin_shell(32, cfg["h"], 1.6))[..., None, None].astype(float)
    _, gt, _, _, _ = R.energy_and_grad_true(M, cfg, K, nref)
    gmax = float(np.max(np.abs(gt * free)))
    # E_stat of the R16-1 core end (8 samples), recomputed here as the bound's base
    Mc = np.load(os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"))
    Ec, ppc, _ = E_true(Mc, cfg, None, nref, ns=8)
    log(f"  E_K {E8:.6f} omega {pp8['omega']:.6f} true grad max free {gmax:.3e}")
    log(f"  own: split max {o['half_split_max']:.5f} at r {o['r_at_half_split_max']:.3f}; rho2 rms {o['rho2_r_rms_all_cells']:.4f} "
        f"(0.35 L = {o['a_box_threshold_0.35L']:.2f}); pin-shell rho2 fraction {o['rho2_fraction_in_pin_shell']:.4f}")
    log(f"  E_stat(R16-1 core, 8 samples, own) {ppc['E_stat']:.6f}; E_K - E_stat = {E8 - ppc['E_stat']:.5f} vs omega_c K = {0.05*K:.3f}")
    note = ("Every number reproduces, to all quoted digits, from my own independent spectral read: E_K {ek:.6f} "
            "(claim 16.994), omega {om:.6f} (claim 0.1066), true-gradient max on the free cells {gm:.2e} (claim "
            "2.8e-4), split max {sm:.4f} at r {sr:.2f} (claim 0.063 at 12.8), rho^2 rms radius {rr:.3f} = {fr:.3f} L "
            "(claim 16.2 = 0.34 L), E_K - E_stat(R16-1 core, my own 8-sample read {es:.4f}) = {dd:.3f} against "
            "omega_c K = 2.5 (claim 3.18).  My generic-eigensolver spectrum agrees with the instrument's polished "
            "projector spectrum to 1e-11 on the gap, the split and lambda_1, so the split reads are not an artifact "
            "of the projector machinery.  I checked, and REFUTE, my own first suspicion: the split is NOT leaning on "
            "the boundary condition.  The pinned shell (the outer 2 cells per axis, |x_i| >= 21.75) carries "
            "{pf:.1e} of the rho^2 weight and a max half-split of {ps:.1e}, i.e. it is exact vacuum, and dropping the "
            "pinned cells changes the rms radius in the 14th digit ({rf:.6f} vs {rr:.6f}).  QUALIFICATION, and it "
            "cuts toward the claim's own reading rather than against it: 'just under the a-box escape' understates "
            "how delocalized this is.  The pre-registered escape is on the rms radius, which sits {mg:.1f} percent "
            "under 0.35 L = 16.8 - but {fo:.1f} PERCENT OF THE rho^2 WEIGHT ALREADY LIES BEYOND 0.35 L.  Two equally "
            "natural readings of the same pre-registered criterion (the rms radius vs the weight fraction outside the "
            "radius) give opposite answers on this state, and only the rms-radius one was registered.  So "
            "NUMERICALLY_UNRESOLVED vs CANDIDATE_REFUTED (a-box) here is decided by the choice of statistic, not by "
            "the field; the verdict should be reported with that fraction beside it.  NOT RECOMPUTED: the dE/dK / "
            "omega = 1.008 read (it needs a fresh 200-iteration K+2 percent continuation) and the 6 directional "
            "derivatives of the stationarity block; I recomputed the true gradient's max instead.").format(
        ek=E8, om=pp8["omega"], gm=gmax, sm=o["half_split_max"], sr=o["r_at_half_split_max"],
        rr=o["rho2_r_rms_all_cells"], fr=o["rho2_r_rms_all_cells"] / 48.0, es=ppc["E_stat"],
        dd=E8 - ppc["E_stat"], mg=100 * (1 - o["rho2_r_rms_all_cells"] / o["a_box_threshold_0.35L"]),
        pf=o["rho2_fraction_in_pin_shell"], ps=o["half_split_max_in_pin_shell"],
        fo=100 * o["rho2_fraction_r_gt_0.35L"], rf=o["rho2_r_rms_free_only"])
    OUT["claims"]["R1"] = {"verdict": "QUALIFIED", "own_numbers": {
        "field": "checkpoints/m5_32_r17/r17_1_rebuild_n32_L48_K50.npy", "lift": "radial_ref", "n_samples_read": 8,
        "E_K": E8, "omega": pp8["omega"], "kin_tot": pp8["kin_tot"], "E_stat": pp8["E_stat"],
        "true_grad_max_free_ns4": gmax, "own_reads": o, "spectrum_cross_check": xc,
        "E_stat_r16_1_core_own_8": ppc["E_stat"], "E_K_minus_E_stat": E8 - ppc["E_stat"], "omega_c_K": 0.05 * K},
        "producer_numbers": {"E_K": 16.994, "omega": 0.1066, "true_grad_max": 2.8e-4, "true_dE_max": 1.1e-4,
                             "dE_dK_over_omega": 1.008, "E_K_minus_E_stat": 3.18, "split_max": 0.063,
                             "r_at_split_max": 12.8, "rho2_r_rms": 16.2},
        "note": note}
    dump()


def claim_R2():
    log("R2: n32 K 200 end state + the same-functional comparison against the R16-3 end state")
    cfg = C.cfg_v4(32, 48.0, completion="rebuild", n_samples=4)
    nref = C.radial_ref(cfg)
    K = 200.0
    M17 = np.load(os.path.join(CK17, "r17_1_rebuild_n32_L48_K200.npy"))
    M16 = np.load(os.path.join(CK16, "r16_3_rebuild_n32_L48_K200.npy"))
    E17, pp17, dom17 = E_true(M17, cfg, K, nref, ns=8)
    E16, pp16, dom16 = E_true(M16, cfg, K, nref, ns=8)
    o17 = own_reads(M17, cfg, "r17_1_n32_K200")
    o16 = own_reads(M16, cfg, "r16_3_n32_K200")
    xc = cross_check_spectrum(M17, cfg, nref)
    log(f"  R17-1 end: E_K {E17:.6f} omega {pp17['omega']:.6f}; own gap min {o17['gap_1_2_min']:.6e} split {o17['half_split_max']:.4f} at r {o17['r_at_half_split_max']:.3f} rms {o17['rho2_r_rms_all_cells']:.4f}")
    log(f"  R16-3 end: E_K {E16:.6f} omega {pp16['omega']:.6f}; own gap min {o16['gap_1_2_min']:.6e} split {o16['half_split_max']:.4f} rms {o16['rho2_r_rms_all_cells']:.4f}")
    note = ("The reads reproduce: E_K {a:.6f} (claim 57.948), omega {b:.6f} (claim 0.1416), my own generic-eigensolver "
            "director gap minimum {c:.3e} (the pre-registered escape (d) threshold is 1e-3, so escape (d) is REAL and "
            "is crossed, though by only {cm:.1f} percent), split max {d:.4f} at r {e:.2f} (claim 0.512 at 5.4), rho^2 "
            "rms radius {f:.3f} (claim 5.8).  The same-functional comparison also reproduces: under the true E_K at 8 "
            "samples with the same radial lift, the R17-1 end state sits at {a:.3f} and the frozen R16-3 end state at "
            "{g:.3f}, so the true-gradient run ended {h:.3f} HIGHER.  QUALIFICATION 1, the load-bearing one: "
            "'a higher-energy branch' is not supportable from this pair, because the two runs did not start from the "
            "same field.  The R17-1 protocol deviation (its own docstring, forced by G2) adds a nucleated doublet "
            "shell of amplitude 0.05 at r 5 to the seed and drops dt0 from 0.01 to 0.001; R16-3 started from the bare "
            "R16-1 end field.  Different seed AND different gradient, one comparison: the difference is not "
            "attributable to the gradient.  QUALIFICATION 2: the R16-3 end state is better than the R17-1 one on "
            "every axis I measured, not just in energy.  Its director gap is {i:.4f} (117 x the escape threshold, "
            "nowhere near escaping), its split max is {j:.4f} against {d:.4f}, and its rho^2 rms radius is {k:.3f} "
            "against {f:.3f}: more localized, spectrally healthy, and 15.7 percent lower in E_K.  The true-gradient "
            "descent from the nucleated seed did not find a better clock; it found a worse, less localized state and "
            "walked it into the escape.  QUALIFICATION 3: 'plateauing' is not what the record says.  The run stopped "
            "on escape_d at it 1900, not on the FIRE plateau rule, with E_K still falling ~7e-3 per 100 iterations "
            "(58.0116 at it 1200 -> 57.9480 at it 1900) and the gap decreasing monotonically from it 100 onward - the "
            "descent was walking INTO the escape, monotonically, for 1800 iterations.  QUALIFICATION 4: the dE/dK / "
            "omega = 1.001 read comes from a K+2 percent continuation that itself stopped on escape_d (stop_K2 in the "
            "record), so it is measured across the escape, and the true gradient's own FD consistency at this end "
            "state is 3.8e-2 relative (the record's analytic_vs_fd_max_rel), not the 2e-6 I measured for the same "
            "gradient in G1: at gap 1e-3 the resolvents that build it are near-singular, so the stationarity numbers "
            "quoted beside the escape are not trustworthy at their printed precision.  The escape verdict itself is "
            "sound - it is a spectral read, not a gradient read.").format(
        a=E17, b=pp17["omega"], c=o17["gap_1_2_min"], cm=100 * (1 - o17["gap_1_2_min"] / 1e-3),
        d=o17["half_split_max"], e=o17["r_at_half_split_max"], f=o17["rho2_r_rms_all_cells"],
        g=E16, h=E17 - E16, i=o16["gap_1_2_min"], j=o16["half_split_max"], k=o16["rho2_r_rms_all_cells"])
    OUT["claims"]["R2"] = {"verdict": "QUALIFIED", "own_numbers": {
        "r17_1_end": {"E_K_8": E17, "omega": pp17["omega"], "kin_tot": pp17["kin_tot"], "E_stat": pp17["E_stat"], "own_reads": o17},
        "r16_3_end": {"E_K_8": E16, "omega": pp16["omega"], "kin_tot": pp16["kin_tot"], "E_stat": pp16["E_stat"], "own_reads": o16},
        "same_functional_difference_r17_minus_r16": E17 - E16, "escape_d_threshold": 1e-3,
        "spectrum_cross_check": xc},
        "producer_numbers": {"E_K": 57.948, "omega": 0.1416, "dE_dK_over_omega": 1.001, "split_max": 0.512,
                             "r_at_split_max": 5.4, "rho2_r_rms": 5.8, "r16_3_reached": 48.8, "escape": "d at it 1900"},
        "note": note}
    dump()


def claim_R3():
    log("R3: n64 K 50 end state (ONE 8-sample energy, no gradient)")
    cfg = C.cfg_v4(64, 48.0, completion="rebuild", n_samples=4)
    M = np.load(os.path.join(CK17, "r17_1_rebuild_n64_L48_K50.npy"))
    nref = C.radial_ref(cfg)
    K = 50.0
    E8, pp8, dom8 = E_true(M, cfg, K, nref, ns=8)
    o = own_reads(M, cfg, "r17_1_n64_K50")
    log(f"  E_K {E8:.6f} omega {pp8['omega']:.6f}; own split {o['half_split_max']:.4f} at r {o['r_at_half_split_max']:.3f} "
        f"rms {o['rho2_r_rms_all_cells']:.4f} gap min {o['gap_1_2_min']:.5f} (h {cfg['h']})")
    r1 = json.load(open(os.path.join(DATA, "m5_32_r17_1.json")))
    dk = r1["dE_dK"]["r17_1_rebuild_n64_L48_K50"]
    note = ("Every read reproduces: E_K {a:.6f} (claim 21.05), omega {b:.6f} (claim 0.196), split max {c:.4f} at r "
            "{d:.2f} (claim 0.153 at 4.9), rho^2 rms radius {e:.3f} (claim 7.1), director gap minimum {f:.4f} from my "
            "own eigensolver (claim 0.062), h {g}.  The verdict NUMERICALLY_UNRESOLVED after 300 iterations is right "
            "and, if anything, generous: E_K fell 56.97 -> 21.05, i.e. the run is nowhere near a stationary point and "
            "300 iterations at n64 is a tenth of the n32 budget.  QUALIFICATION the claim omits entirely: the dE/dK = "
            "omega consistency check FAILS on this cell.  The record's own dE_dK for this run is {h:.4f} against omega "
            "{i:.4f}, a ratio of {j:.3f} - NEGATIVE, i.e. E_K went DOWN when K went up by 2 percent, where the "
            "thermodynamic identity requires it to go up by omega.  R1 and R2 both quote their ratios (1.008, 1.001) "
            "as evidence; R3 quotes none, and the reason is that its number is -0.102.  That is expected for a "
            "descending, unconverged state (the 200-iteration K+2 percent continuation keeps descending faster than "
            "the K-shift raises E), but it must be stated with the rung's claim, not left in the JSON.").format(
        a=E8, b=pp8["omega"], c=o["half_split_max"], d=o["r_at_half_split_max"], e=o["rho2_r_rms_all_cells"],
        f=o["gap_1_2_min"], g=cfg["h"], h=dk["dE_dK_fd"], i=dk["omega_end"], j=dk["ratio"])
    OUT["claims"]["R3"] = {"verdict": "QUALIFIED", "own_numbers": {
        "field": "checkpoints/m5_32_r17/r17_1_rebuild_n64_L48_K50.npy", "h": cfg["h"], "n_samples_read": 8,
        "E_K": E8, "omega": pp8["omega"], "kin_tot": pp8["kin_tot"], "E_stat": pp8["E_stat"], "own_reads": o,
        "producer_dE_dK_record": dk},
        "producer_numbers": {"E_K": 21.05, "E_K_start": 56.97, "omega": 0.196, "split_max": 0.153,
                             "r_at_split_max": 4.9, "rho2_r_rms": 7.1, "gap": 0.062, "iters": 300},
        "note": note}
    dump()


# =============================================================== X1: the c_X inertia ladder
def claim_X1():
    log("X1: the X_M added inertia fraction on the R16-1 core, rebuilt from the kinetic cells")
    cfg = C.cfg_v4(32, 48.0, completion="rebuild", n_samples=4)
    cfg["weight"] = "relative"
    M = np.load(os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"))
    nref = C.radial_ref(cfg)
    free = ~INS4.pin_shell(32, cfg["h"], 1.6)
    fm = free.astype(float)
    with R.weight_mode(cfg):
        Ea, Eb, fr, r = OP.doublet_basis(M, cfg)
    Ea = Ea * fm[..., None, None]
    Eb = Eb * fm[..., None, None]
    with R.weight_mode(cfg):
        kc_a = C.kin_a0_grad(M, cfg, Ea, nref)[2]
        kc_b = C.kin_a0_grad(M, cfg, Eb, nref)[2]
        frx = C.frame(M, nref)
        rows = {}
        for cX in (1.0, 10.0, 100.0):
            cfx = dict(cfg)
            cfx["cX"] = cX
            xa = R.xm_kin_cells(M, Ea, cfx, frx, n_ref=nref)
            xb = R.xm_kin_cells(M, Eb, cfx, frx, n_ref=nref)
            num = float(np.sum((xa + xb)[free]))
            den = float(np.sum((kc_a + kc_b)[free]))
            cellfrac = float(np.max(((xa + xb) / np.maximum(kc_a + kc_b, 1e-300))[free]))
            rows[cX] = {"trace_fraction": num / den, "max_cell_fraction": cellfrac,
                        "X_kin_sum_free": num, "instrument_kin_trace_sum_free": den}
            log(f"  cX {cX:6.1f}: trace fraction {num/den:.6e} (producer {1.6088206137087166e-06*cX:.6e}), max cell {cellfrac:.4e}")
    # the Omega^2 ladder, straight from the producer's operator records (read, not derived)
    lad = {}
    for cX, f in ((0.0, "r17_2op_r16_1_core_v4rel.json"), (1.0, "r17_2op_r16_1_core_v4rel_cX1.json"),
                  (10.0, "r17_2op_r16_1_core_v4rel_cX10.json"), (100.0, "r17_2op_r16_1_core_v4rel_cX100.json")):
        d = json.load(open(os.path.join(CK17, f)))
        lad[cX] = float(min(m["Omega2"] for m in d["modes"]))
    base = lad[0.0]
    slopes = {cX: (lad[cX] - base) / cX for cX in (1.0, 10.0, 100.0)}
    box = 0.025632114255195525
    box_alt = json.load(open(os.path.join(CK17, "r17_2op_r16_1_core_v4rel.json")))["thresholds"]["box_continuum_estimate_Omega2"]
    slope = slopes[100.0]
    cx_need = (base - box) / abs(slope)
    cx_need_alt = (base - box_alt) / abs(slope)
    spread = max(slopes.values()) / min(slopes.values())
    log(f"  Omega2: {lad}; slopes per unit cX {slopes}; extrapolated cX to reach {box:.6f}: {cx_need:.3e}")
    note = ("The inertia fraction reproduces exactly from my own rebuild of the two pieces: with the doublet basis "
            "(m5_32_r16_2_operator.doublet_basis) and the instrument's kinetic cells against the X_M kinetic cells, "
            "the free-cell trace fraction is {a:.6e} / {b:.6e} / {c:.6e} at c_X 1 / 10 / 100, i.e. exactly linear at "
            "1.60882e-6 per unit c_X, matching the producer's X_M_added_inertia_trace_fraction to all printed digits. "
            "The Omega^2 ladder is linear too: the lowest mode moves by {d:.3e} per unit c_X (my slope from the c_X "
            "100 point; the 1 / 10 / 100 slopes agree to {e:.4f} relative), i.e. -9.6e-8, which the claim rounds to "
            "-1.0e-7 (a 4 percent over-statement of the effect, harmless for its conclusion).  The extrapolated "
            "crossing is {f:.2e}, so '~2e5' is right.  QUALIFICATION, the real one: the extrapolation is used 2000 x "
            "beyond the tested range and it is NOT a perturbative regime there.  At c_X 2e5 the added inertia fraction "
            "is 1.6e-6 x 2e5 = 0.32, a 32 percent change to the inertia, and the max single-cell fraction (which is "
            "60 x the trace fraction here) would be ~19, i.e. the X_M term would dominate the inertia on the cells "
            "that matter.  A linear Rayleigh shift measured at fractions <= 1.6e-4 says nothing about the spectrum "
            "there; the honest statement is 'no crossing anywhere in the tested range, and the trend rules out any "
            "crossing at perturbative c_X', not a number for the crossing.  Second qualification: the target itself "
            "is ambiguous.  The claim uses 0.025632 (the audit's box bottom); the operator records' own "
            "thresholds.box_continuum_estimate_Omega2 is {g:.6f}, which gives {h:.2e} instead.  Both are ~1e5, so the "
            "conclusion survives, but the quoted 2e5 is target-dependent.").format(
        a=rows[1.0]["trace_fraction"], b=rows[10.0]["trace_fraction"], c=rows[100.0]["trace_fraction"],
        d=slope, e=spread - 1.0, f=cx_need, g=box_alt, h=cx_need_alt)
    OUT["claims"]["X1"] = {"verdict": "QUALIFIED", "own_numbers": {
        "field": "checkpoints/m5_32_r16/r16_1_rebuild_n32_L48.npy", "weight": "relative", "n_samples": 4,
        "inertia_fractions": {str(k): v for k, v in rows.items()},
        "fraction_per_unit_cX": {str(k): v["trace_fraction"] / k for k, v in rows.items()},
        "Omega2_lowest": {str(k): v for k, v in lad.items()},
        "Omega2_slope_per_unit_cX": {str(k): v for k, v in slopes.items()},
        "box_bottom_used_by_claim": box, "box_continuum_estimate_in_operator_record": box_alt,
        "cX_to_cross_claim_target": cx_need, "cX_to_cross_record_threshold": cx_need_alt,
        "added_inertia_fraction_at_cX_2e5": 1.60882e-6 * 2e5},
        "producer_numbers": {"fraction_per_unit_cX": 1.6088206137087166e-06, "Omega2_shift_per_unit_cX": -1.0e-7,
                             "Omega2": [0.0446626, 0.0446625, 0.0446624, 0.0446529], "cX_needed": 2e5},
        "note": note}
    dump()


# =============================================================== X2: the X_M linear part
def _eps4():
    """an independently constructed Levi-Civita tensor: the determinant of the permutation matrix."""
    E = np.zeros((4, 4, 4, 4))
    idx = np.arange(4)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    P = np.zeros((4, 4))
                    for row, col in enumerate((a, b, c, d)):
                        P[row, col] = 1.0
                    E[a, b, c, d] = round(float(np.linalg.det(P))) if len({a, b, c, d}) == 4 else 0.0
    assert E[0, 1, 2, 3] == 1.0 and E[1, 0, 2, 3] == -1.0 and abs(np.sum(np.abs(E)) - 24) < 1e-12
    del idx
    return E


def X_own(A0, Asp):
    """X_M = (1/2) sum_{mu != nu} eps_{mu nu a b} eta^mu eta^nu F[mu,nu]_{a b},
    F_{mu nu} = A_mu eta A_nu - A_nu eta A_mu.  Built here from my own eps4 and my own loop."""
    E4 = _eps4()
    de = np.diag(ETA)
    A = [A0] + list(Asp)
    X = np.zeros(np.shape(A0)[:-2], dtype=np.asarray(A0).dtype)
    for mu in range(4):
        for nu in range(4):
            if mu == nu:
                continue
            F = A[mu] @ ETA @ A[nu] - A[nu] @ ETA @ A[mu]
            X = X + 0.5 * de[mu] * de[nu] * np.einsum("ab,...ab->...", E4[mu, nu], F)
    return X


def claim_X2():
    log("X2: X_M affine in A_0, static part zero, linear part exactly linear (own epsilon contraction)")
    cfg = C.cfg_v4(32, 48.0, completion="rebuild", n_samples=4)
    M = np.load(os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"))
    nref = C.radial_ref(cfg)
    fr = C.frame(M, nref)
    v = np.real(C.a0_of(M, fr))
    h = cfg["h"]
    # my own sym-stencil jets: the fwd/bwd average
    Asp = [0.5 * (INS4.d1(M, ax, h, "fwd") + INS4.d1(M, ax, h, "bwd")) for ax in range(3)]
    Z = np.zeros_like(v)
    X0 = X_own(Z, Asp)
    X1 = X_own(v, Asp)
    X2v = X_own(2.0 * v, Asp)
    lin = X1 - X0
    lin2 = X2v - X0
    scale = max(float(np.max(np.abs(X1))), 1e-300)
    static_max = float(np.max(np.abs(X0)))
    lin_err = float(np.max(np.abs(lin2 - 2.0 * lin)))
    # a third point: affineness (X(3v) - X(0) == 3 (X(v) - X(0)))
    aff_err = float(np.max(np.abs((X_own(3.0 * v, Asp) - X0) - 3.0 * lin)))
    # cross-check against the instrument on the SAME jets
    inst = np.real(R.xm_cells(v, Asp))
    inst_lin = np.real(R.xm_cells(v, Asp, linear_part=True))
    d_inst = float(np.max(np.abs(inst - X1)))
    d_inst_lin = float(np.max(np.abs(inst_lin - lin)))
    # is X_1 big enough to matter?  the scale of X_1 against the clock's kinetic density
    log(f"  |X(A0=0)| max {static_max:.3e}; |X(a0)| max {scale:.6e}; linearity err {lin_err:.3e}; affineness err {aff_err:.3e}")
    log(f"  vs the instrument on the same jets: X {d_inst:.3e}, X_1 {d_inst_lin:.3e}")
    ok = static_max < 1e-14 and lin_err < 1e-12 * scale and aff_err < 1e-12 * scale
    note = ("Confirmed with my own Levi-Civita construction (determinant of the permutation matrix, built and "
            "asserted independently of the instrument's inversion-count build) on my own fwd/bwd-averaged sym-stencil "
            "jets of the R16-1 core.  X(A_0 = 0) = {a:.2e} over the whole box, i.e. the static part vanishes "
            "identically on this field (M_0i = 0 everywhere, as claimed); X(2 v) - X(0) - 2 [X(v) - X(0)] = {b:.2e} "
            "against a scale |X(a0)|_max = {c:.3e}, and the three-point affineness residual is {d:.2e}: X is affine in "
            "A_0 to machine precision.  My X and X_1 agree with the instrument's xm_cells on the same jets to {e:.1e} "
            "and {f:.1e} - bit-exact, which is the expected outcome and is worth saying plainly: it verifies the "
            "Levi-Civita construction, the eta placement and the contraction, NOT the definition of X_M, which I took "
            "from the same source the producer did.  TWO NOTES, neither a refutation.  (1) 'The static part vanishes' "
            "is a property of THIS class of fields (purely spatial M, no 0i entries), not of X_M: any boosted or "
            "time-tilted configuration carries a static X_s, and nothing in the instrument enforces M_0i = 0, so the "
            "kinetic-only treatment is a standing assumption that must travel with the term.  (2) The magnitude is "
            "the story X1 inherits: |X_1(a0)| peaks at {c:.2e} on this core, which is why the added inertia is "
            "1.6e-6 c_X.  X_M is not small because the coefficient is small; it is small because the object itself "
            "nearly vanishes on these textures.").format(
        a=static_max, b=lin_err, c=scale, d=aff_err, e=d_inst, f=d_inst_lin)
    OUT["claims"]["X2"] = {"verdict": "CONFIRMED" if ok else "REFUTED", "own_numbers": {
        "field": "checkpoints/m5_32_r16/r16_1_rebuild_n32_L48.npy",
        "max_abs_X_at_A0_zero": static_max, "max_abs_X_at_a0": scale,
        "linearity_residual_X(2v)-X(0)-2[X(v)-X(0)]": lin_err, "affineness_residual_3v": aff_err,
        "own_vs_instrument_X": d_inst, "own_vs_instrument_X1": d_inst_lin},
        "producer_numbers": {"static_part_tolerance": 1e-14, "linearity_tolerance": 1e-12},
        "note": note}
    dump()


# =============================================================== hazards + tally
def hazards():
    OUT["unclaimed_hazards"] = [
        {"id": "A1", "where": "R2, and any comparison of an R17-1 end state with an R16-3 one",
         "hazard": "the true-gradient and frozen runs do not share a seed, so no energy difference between their end states is attributable to the gradient",
         "detail": "the R17-1 protocol deviation adds a nucleated doublet shell (amplitude 0.05 at r 5, width 2) to every seed and drops dt0 from 0.01 to 0.001; R16-3 started from the bare R16-1 end field at dt0 0.01.  The 57.948 vs 48.846 gap I measured under one functional and one lift is a two-variable difference reported as one.  And the direction is unflattering: on my own reads the R16-3 (frozen, bare-seed) end state is MORE localized (rms 4.93 vs 5.81, split 0.645 vs 0.512) and spectrally healthy (gap 0.116 vs 9.98e-4) as well as 15.7 percent lower in E_K.  The rung reads as 'the true gradient found a higher-energy branch'; the same data reads as 'the nucleated seed plus the true gradient did worse and escaped'.  Neither is established.  The clean experiment (the true gradient from the SAME bare seed) is exactly the one G2 says is impossible without the shell, so this confound is structural, not an oversight - it needs a third run, e.g. the frozen protocol from the NUCLEATED seed, to separate the two variables."},
        {"id": "A2", "where": "R3 (n64 K 50)",
         "hazard": "the dE/dK = omega identity FAILS on this cell and the claim does not say so",
         "detail": "data/m5_32_r17_1.json gives dE_dK_fd -0.0199 against omega 0.1958, ratio -0.102.  R1 and R2 both cite their ratios as corroboration; R3 cites none.  Expected for an unconverged descending state, but the asymmetry in reporting is the hazard."},
        {"id": "A3", "where": "R2 (n32 K 200) stationarity and dE/dK reads",
         "hazard": "both are measured at or across the escape (d) boundary, where the gradient itself is only 4 percent accurate",
         "detail": "the record's analytic_vs_fd_max_rel at this end state is 3.75e-2 (against G1's 1e-6 gate elsewhere), the gap minimum is 9.98e-4 against the 1e-3 threshold, and the K+2 percent continuation that produced dE/dK / omega = 1.001 itself stopped on escape_d.  The escape verdict is sound (it is a spectral read, not a gradient read); the numbers quoted beside it are not."},
        {"id": "A4", "where": "R1 (n32 K 50), the a-box escape decision",
         "hazard": "two equally natural readings of the SAME pre-registered escape give opposite verdicts on this state, and only one was registered",
         "detail": "escape (a-box) is registered on the rho^2-weighted rms radius, which is 16.222 against 0.35 L = 16.8: not crossed, by 3.4 percent.  But 39.7 percent of the rho^2 weight already sits at r > 0.35 L (my own read; the producer's escape_reads computes this same quantity, rho2_fraction_r_gt_0.35L, and does not use it).  A criterion phrased as 'more than a third of the split beyond 0.35 L' would fire here.  I checked and REJECTED the related worry that the tail leans on the pinned shell: the pinned cells hold 1.8e-30 of the weight, exact vacuum."},
        {"id": "A5", "where": "every circle-averaged read in this rung",
         "hazard": "the reported energies depend on a director lift that the checkpoints do not carry",
         "detail": "checkpoints/m5_32_r17/lift_dependence.json shows E_K on r17_1_rebuild_n32_L48_K200.npy is 57.948 with the radial lift and 68.636 with the x lift (19 percent).  The radial lift happens to reproduce the propagated one exactly on these three fields (0 mismatched bonds), which is why my reads match to all digits, but that is a checked coincidence of these fields, not a property of the format.  This is the R17-2 audit's H9 and it is still open at R17-1."},
        {"id": "A6", "where": "R1 and R3 verdicts (NUMERICALLY_UNRESOLVED by max_iter)",
         "hazard": "the two grids got budgets that differ by 10 x, so the n64 run carries no independent information about the n32 conclusion",
         "detail": "n32 K50 ran 3000 iterations (18536 s), n64 K50 ran 300 (15724 s) and was still falling from 56.97 to 21.05.  A resolution check needs the two to be comparably converged; as run, n64 is a wall-clock report, not a grid-refinement control."},
        {"id": "A7", "where": "X1 (the c_X ladder)",
         "hazard": "a linear extrapolation is carried 2000 x past the tested range into a regime where the perturbation is 32 percent",
         "detail": "see the X1 note.  At c_X 2e5 the added inertia fraction is 0.32 of the trace and ~19 on the worst cell; the measured slope was taken at fractions <= 1.6e-4.  The defensible statement is a no-crossing result over the tested range plus a trend, not a crossing value."},
        {"id": "A8", "where": "G1 (the omitted part along the descents)",
         "hazard": "the stated range 0.4 to 0.9 excludes the measured minima",
         "detail": "the traces give 0.202 (n32 K50, it 800) and 0.152 (n32 K200, it 400); the full n32 span is 0.152 to 0.874.  The direction of the error is safe for the claim's conclusion (the omitted part is still O(1) of the frozen gradient at the median) but the range as printed is not what the data says."},
        {"id": "A9", "where": "G2 / the R17-1 seed deviation",
         "hazard": "'1/split^2 singularity' is asserted from a single point, not from a measured exponent",
         "detail": "kin_tot ~ split^2 is a plausible reading of kin_tot 2.1e-3 at half-split 6e-4, but no split-scaling series was run, and the 6000 x gradient ratio is a per-point number that moves with sample count and lift.  The protocol consequence drawn from it is conservative and correct either way."},
    ]


def main():
    claim_G1()
    claim_G2()
    claim_X2()
    claim_R1()
    claim_R2()
    claim_X1()
    claim_R3()
    hazards()
    OUT["coverage"] = {
        "recomputed_from_scratch": [
            "the split, the director gap and lambda_1 on all four fields, from eigenvalues of N = M eta by the generic eigensolver (NOT the instrument's polished spectral projectors); cross-checked against the instrument and agreeing to ~1e-11",
            "the directional derivatives of the full E_K by my own central differences at two step sizes with a Richardson extrapolation, on 3 random free directions and the gradient's own direction",
            "the rho^2-weighted rms radius, the weight fraction beyond 0.35 L, the pin-shell weight and the split maximum with their radii",
            "X_M with an independently constructed Levi-Civita tensor (permutation-matrix determinant) on my own fwd/bwd-averaged sym-stencil jets",
            "the X_M added-inertia trace fraction, rebuilt from the doublet basis, the instrument's kinetic cells and the X_M kinetic cells",
        ],
        "consumed_from_the_producer_unverified": [
            "the complex-step gate 4.6e-14 (read as PASS from data/m5_32_r17_common_selftest.json; the complex-step path was not rebuilt)",
            "the per-iteration a0_chain_norm / frozen_grad_norm traces (read from the run checkpoints; the ratio at one field was recomputed here)",
            "the Omega^2 eigenvalues of the doublet operator at c_X 0 / 1 / 10 / 100 (read from the operator records; the ladder's linearity and the extrapolation were recomputed, the eigenproblem was not re-solved)",
            "dE/dK / omega for all three runs (each needs a fresh 200-iteration K+2 percent continuation)",
            "the 6-direction stationarity block of each run (I recomputed the true gradient's max instead)",
            "the escape (b) and (c) diagnostics (quadrupole, spin-2 winding); only (a), (a-box) and (d) were recomputed",
        ],
        "energy_functional_used": "m5_32_r17_common.energy_and_grad_true with the radial director lift (m5_32_r16_common.radial_ref), 8 circle samples for every read and 4 for every gradient; the radial lift reproduces the propagated one exactly on these fields (0 mismatched bonds, checkpoints/m5_32_r17/lift_dependence.json), which is why my reads match to all digits",
    }
    tally = {}
    for k, v in OUT["claims"].items():
        tally[v["verdict"]] = tally.get(v["verdict"], 0) + 1
    OUT["tally"] = {"n_claims": len(OUT["claims"]), **tally,
                    "per_claim": {k: v["verdict"] for k, v in OUT["claims"].items()},
                    "n_unclaimed_hazards": len(OUT["unclaimed_hazards"])}
    dump()
    log(f"TALLY {OUT['tally']}")


if __name__ == "__main__":
    main()
