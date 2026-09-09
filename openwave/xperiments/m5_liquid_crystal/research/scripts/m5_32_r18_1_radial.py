"""M5.32 R18-1 (ledger 6.7): the radial solve for r_0 sqrt(mu), the author's top ask (rev-200 section 180.6 item 1).
The tube instrument of m5_32_r18_common (equations there): the spherically symmetric sector of the 4x4 field, four radial
profiles on the cell-centered grid, evaluated by the SAME 3D static energy code on a one-ray (2 N_g, 3, 3) tube (gate 1 in the
selftest: the tube center line at (h/2, h/2) reproduces the 3D lattice's own ray densities to roundoff), minimized by
L-BFGS-B with the box bound Delta = lambda_1 - lambda_23 >= GAP_MIN (the admissible domain).

Objects: (A) v4 with the absolute plateau weight (R16-1's, mu 1e-2, c_P 1, c_s 0.4, I_rebuild) and (B) v4 with the
director-relative weight (R17-2's).  Object (C) v6 is NOT a separate solve: on a split-free profile rho^2 = 0 exactly, so
U_v6 = mu rho^2 - nu rho^4 + kappa rho^6 = 0 and the regulator c_s rho^2 E2 = 0 at every g_W: the v6 energy of the (B)
profile equals the (B) energy (the identity R17-3a found on the lattice at 1e-5, checked here at 1e-12 on every (B) end
profile at g_W 2.0).  The split-free radial sector is the same problem for (B) and (C).

Scan: h in {1.5, 0.75, 0.375, 0.15, 0.075, 0.03} at L 48 and 96, L 144 at h >= 0.15, run as six chains (object x box): the
coarsest rung from the R16-1 n32 core's shell means, each finer rung seeded from the previous rung's end profile; the melted
core lambda_1 = lambda_23 + 0.7 (1 - exp(-(r/2)^2)) as the INDEPENDENT seed at every rung h >= 0.075 of L 48 and L 96 (the two
seeds must reach the same minimum: the path-independence check); reads per end profile: r_0 under the three R17-0
definitions (profile = the lambda_1 = 0.8 crossing, taper, half), Delta_min, E_stat and its parts, the tail amplitude of the
1 / r^4 density; Richardson extrapolation in h at fixed L, the L-exponent at fixed h.
Gates: (1) the selftest identity; (2) the tube quadrature against the 3D energy of the same ansatz field (reported, O(h^2));
(3) the relaxed tube at h 1.5 L 48 against the R16-1 n32 core (r_0 within 10 percent, E_stat within 5 percent) and (B)
against the R17-2 v4rel static; at h 0.75 against the n64 L48 core.
Pre-registered outcomes: RADIAL_CONVERGED (the two finest h within 5 percent at L 96 and the L 96 -> 144 change under 5
percent at h 0.15): the number with its uncertainty; CORE_GRID_SET (r_0 falls with h without a floor: r_0 / h roughly
constant on the finest rungs); the tube minimum is not the 3D minimum (gate 3 fails at the 5 percent level in E_stat):
reported with the 3D comparison, the radial number qualified.

usage: python3 m5_32_r18_1_radial.py scan [--workers 4]     # every (object, h, L, seed) job, JSON + profile per job
       python3 m5_32_r18_1_radial.py collect                 # the ladder, the outcomes, the plot
out:   checkpoints/m5_32_r18/r18_1_<tag>.json / _profile.npy, data/m5_32_r18_1.json, plots/m5_32_r18_1_radial.png
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

C15, INS4 = C.C15, C.INS4
RES, DATA, PLOTS = C.RES, C.DATA, C.PLOTS
CK16, CK17, CK = C.CK, R.CK, X.CK
T0 = time.time()
HS = [1.5, 0.75, 0.375, 0.15, 0.075, 0.03]
LS = [48.0, 96.0]
OBJECTS = ("v4abs", "v4rel")
SEEDS = ("r16_1", "melted")


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def rel(p):
    return os.path.relpath(p, RES)


def make_cfg(obj, h, L):
    n = int(round(L / h))
    cfg = C.cfg_v4(n, L, completion="rebuild", n_samples=1)
    cfg["object"] = "v4"
    cfg["weight"] = "absolute" if obj == "v4abs" else "relative"
    cfg["h"] = float(h)
    return cfg


def tag_of(obj, h, L, seed):
    return f"r18_1_{obj}_h{h:g}_L{int(L)}_{seed}"


_SEED_CACHE = {}


def seed_profiles(seed, cfg, tube):
    if seed == "melted":
        return X.melted_seed(cfg, tube), "the melted core: lambda_1 = lambda_23 + max(0.7 (1 - exp(-(r/2)^2)), GAP_MIN), m_g and lambda_23 at the vacuum"
    if "r16_1" not in _SEED_CACHE:
        cfg32 = C.cfg_v4(32, 48.0, completion="rebuild")
        M32 = np.load(os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"))
        t32 = X.tube_setup(1.5, 48.0)
        _SEED_CACHE["r16_1"] = (X.profiles_from_field(M32, cfg32, t32), t32)
    P32, t32 = _SEED_CACHE["r16_1"]
    P = X.vacuum_profiles(cfg, tube["Ng"])
    for k in range(3):
        P[k] = np.interp(tube["rg"], t32["rg"], P32[k], right=P[k][-1])
    P[1] = np.maximum(P[1], P[2] + X.GAP_MIN)
    return P, "the R16-1 n32 L48 core's shell means (-lambda_g, lambda_1, (lambda_2 + lambda_3)/2 in 1.5 h shells) interpolated to the grid, beta 0, the vacuum beyond r 24"


def job(args):
    obj, h, L, seed = args[:4]
    from_profile = args[4] if len(args) > 4 else None
    tag = tag_of(obj, h, L, seed)
    outp = os.path.join(CK, tag + ".json")
    if os.path.exists(outp):
        return tag, "exists"
    cfg = make_cfg(obj, h, L)
    tube = X.tube_setup(h, L)
    if from_profile is not None:
        Pc = np.load(from_profile["file"])
        tc = X.tube_setup(from_profile["h"], from_profile["L"])
        P0 = X.vacuum_profiles(cfg, tube["Ng"])
        for k in range(4):
            P0[k] = np.interp(tube["rg"], tc["rg"], Pc[k], right=P0[k][-1])
        P0[1] = np.maximum(P0[1], P0[2] + X.GAP_MIN)
        how = f"the end profile of {os.path.basename(from_profile['file'])} (h {from_profile['h']:g}, L {from_profile['L']:g}) interpolated to this grid"
    else:
        P0, how = seed_profiles(seed, cfg, tube)
    E0, _, p0, e0, _ = X.tube_energy(P0, tube, cfg, need_grad=False)
    rec = {"tag": tag, "rung": "R18-1", "object": obj, "weight": cfg["weight"], "h": h, "L": L, "Ng": tube["Ng"], "n_cells_tube": int(2 * tube["Ng"] * 9), "seed": {"kind": seed, "how": how, "E": float(E0), "reads": X.profile_reads(P0, tube, cfg, e0, p0)},
           "cfg": {k: cfg[k] for k in ("mu", "cP", "cs", "completion", "weight", "stencil")}}
    t = time.time()
    maxiter = 4000
    P, E, parts, e_line, info = X.solve_tube(P0, tube, cfg, maxiter=maxiter, gtol=1e-8, log_every=500, tag=tag)
    rec["descent"] = info
    rec["E"] = float(E)
    rec["reads"] = X.profile_reads(P, tube, cfg, e_line, parts)
    # the v6 identity on the (B) end profile at g_W 2.0 (rho^2 = 0 exactly: U_v6 = 0, reg = 0)
    if obj == "v4rel":
        cfg6 = R.cfg_v6(cfg["n"], L, gW=2.0, completion="rebuild", n_samples=1); cfg6["h"] = float(h)
        E6, _, p6, _, _ = X.tube_energy(P, tube, cfg6, need_grad=False)
        rec["v6_identity"] = {"E_v6_gW2.0": float(E6), "E_v4rel": float(E), "rel_diff": float(abs(E6 - E) / abs(E)), "U_v6": float(np.real(p6.get("U_v6", 0.0))), "identity_holds_1e-12": bool(abs(E6 - E) < 1e-12 * abs(E))}
    # the profile on a coarse grid for the record (the full profile in the npy)
    step = max(1, tube["Ng"] // 160)
    rec["profile_coarse"] = {"r": tube["rg"][::step].tolist(), "m_g": P[0][::step].tolist(), "lambda_1": P[1][::step].tolist(), "lambda_23": P[2][::step].tolist(), "beta": P[3][::step].tolist(), "e_line_positive_half": np.real(e_line[tube["Ng"]:][::step]).tolist()}
    np.save(os.path.join(CK, tag + "_profile.npy"), P)
    rec["profile_file"] = rel(os.path.join(CK, tag + "_profile.npy"))
    rec["wall_s"] = time.time() - t
    json.dump(rec, open(outp, "w"), indent=1, default=float)
    rd = rec["reads"]
    return tag, f"E {E:.6f} (seed {E0:.4f}) r_0 profile {rd['r_0_profile (lambda_1 = 0.8 crossing)']:.4f} taper {rd['r_0_taper (max r with lambda_1 < 0.8)']:.3f} Delta_min {rd['Delta_min']:.4f} l1(0) {rd['lambda_1_center']:.4f} bound {info['cells_at_gap_bound']} nit {info['nit']} {info['message'][:40]} {rec['wall_s']:.0f}s"


def chain(args):
    """one (object, L) ladder in h: the coarsest rung from the r16_1 seed, each finer rung seeded from the previous rung's end profile
    (the 'r16_1' seed label carries the chain); returns the per-rung messages."""
    obj, L = args
    msgs = []
    prev = None
    for h in HS:
        if L == 144.0 and h < 0.15:
            continue
        tag, msg = job((obj, h, L, "r16_1", prev))
        msgs.append(f"{tag}: {msg}")
        prev = {"file": os.path.join(CK, tag + "_profile.npy"), "h": h, "L": L}
    return msgs


def scan(workers):
    """the six chains (2 objects x 3 boxes) in a pool, then the independent melted-seed checks at L 48 and L 96 (h >= 0.075) as
    separate jobs: the two seeds must reach the same minimum (the check that the chain is not path-dependent)."""
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    chains = [(obj, L) for obj in OBJECTS for L in LS + [144.0]]
    log(f"{len(chains)} chains, {workers} workers")
    with ctx.Pool(workers) as pool:
        for msgs in pool.imap_unordered(chain, chains):
            for m in msgs:
                log(f"  {m}")
    checks = [(obj, h, L, "melted") for obj in OBJECTS for L in LS for h in HS if h >= 0.075]
    log(f"{len(checks)} melted-seed checks")
    with ctx.Pool(workers) as pool:
        for tag, msg in pool.imap_unordered(job, checks):
            log(f"  {tag}: {msg}")
    log("scan done")


def gate3(runs):
    """the relaxed tube against the 3D cores (R16-1 n32 / n64, the R17-2 v4rel static)."""
    d17 = json.load(open(os.path.join(DATA, "m5_32_r17_0_record.json")))["c_core_reads_r16_1"]
    ref = {("v4abs", 1.5): {"r_0_profile": d17["r16_1_end_n32_L48"]["r_0_profile (shell-mean lambda_1 = 0.8)"], "Delta_min": d17["r16_1_end_n32_L48"]["Delta_min_free"], "E_stat": None, "core": "R16-1 n32 L48"},
           ("v4abs", 0.75): {"r_0_profile": d17["r16_1_end_n64_L48"]["r_0_profile (shell-mean lambda_1 = 0.8)"], "Delta_min": d17["r16_1_end_n64_L48"]["Delta_min_free"], "E_stat": None, "core": "R16-1 n64 L48"}}
    try:
        d16 = json.load(open(os.path.join(DATA, "m5_32_r16_1.json")))["runs"]
        for k, tg in ((("v4abs", 1.5), "r16_1_rebuild_n32_L48"), (("v4abs", 0.75), "r16_1_rebuild_n64_L48_analytic")):
            g = d16[tg].get("gates", {})
            ref[k]["E_stat"] = g.get("reads", {}).get("rebuild", {}).get("parts_8", {}).get("E_stat")
    except Exception as e:                                  # noqa: BLE001
        log(f"  R16-1 energies not found: {e!r}")
    try:
        d172 = json.load(open(os.path.join(DATA, "m5_32_r17_2.json")))["runs"]["r17_2_v4rel_n32_L48_r16_1"]
        ref[("v4rel", 1.5)] = {"r_0_profile": d172["reads"]["core"]["r_0_profile (shell-mean lambda_1 = 0.8)"], "Delta_min": d172["reads"]["core"]["Delta_min_free"], "E_stat": d172["reads"]["E_stat_8"], "core": "R17-2 v4rel static n32 L48"}
    except Exception as e:                                  # noqa: BLE001
        log(f"  R17-2 v4rel record not found: {e!r}")
    out = {}
    for (obj, h), rf in ref.items():
        for seed in SEEDS:
            tag = tag_of(obj, h, 48.0, seed)
            if tag not in runs:
                continue
            rd = runs[tag]["reads"]
            row = {"core": rf["core"], "tube_r_0_profile": rd["r_0_profile (lambda_1 = 0.8 crossing)"], "core_r_0_profile": rf["r_0_profile"],
                   "r_0_rel_dev": abs(rd["r_0_profile (lambda_1 = 0.8 crossing)"] - rf["r_0_profile"]) / max(abs(rf["r_0_profile"]), 1e-300),
                   "tube_Delta_min": rd["Delta_min"], "core_Delta_min": rf["Delta_min"], "tube_E": runs[tag]["E"], "core_E_stat": rf["E_stat"]}
            if rf["E_stat"] is not None:
                row["E_rel_dev"] = abs(runs[tag]["E"] - rf["E_stat"]) / abs(rf["E_stat"])
            row["r_0_within_10pct"] = bool(row["r_0_rel_dev"] < 0.10)
            row["E_within_5pct"] = bool(row.get("E_rel_dev", 1.0) < 0.05) if rf["E_stat"] is not None else None
            out[tag] = row
    return out


def gap_reads(P, tube, cfg):
    """the core radius from the director GAP Delta = lambda_1 - lambda_23 (the read that survives lambda_1 never crossing 0.8 on the fine
    rungs): r_gap = the outermost radius where Delta crosses half the vacuum gap (0.35) by interpolation; r_gap_0.1 likewise at 0.1;
    the spectrum at the center; the radius of the lambda_23 maximum (the pair eigenvalue rises in the melted core)."""
    rg = tube["rg"]
    gap = P[1] - P[2]
    vac = INS4.vac4(cfg)
    Dv = vac[1, 1] - vac[2, 2]
    out = {"gap_center": float(gap[0]), "gap_min": float(np.min(gap)), "r_at_gap_min": float(rg[int(np.argmin(gap))]), "lambda_23_max": float(np.max(P[2])), "r_at_lambda_23_max": float(rg[int(np.argmax(P[2]))]),
           "spatial_trace_center": float(P[1][0] + 2 * P[2][0]), "m_g_center": float(P[0][0])}
    for lab, lev in (("r_gap_half (Delta = 0.35 crossing)", 0.5 * Dv), ("r_gap_0.1 (Delta = 0.1 crossing)", 0.1), ("r_gap_0.2 (Delta = 0.2 crossing)", 0.2)):
        cross = np.where((gap[:-1] < lev) & (gap[1:] >= lev))[0]
        out[lab] = float(rg[cross[-1]] + (lev - gap[cross[-1]]) * (rg[cross[-1] + 1] - rg[cross[-1]]) / (gap[cross[-1] + 1] - gap[cross[-1]])) if len(cross) else 0.0
    out["r_gap_half_over_h"] = out["r_gap_half (Delta = 0.35 crossing)"] / tube["h"]
    return out


def gate3_3d(runs):
    """the apples-to-apples comparison: the tube's end profile placed on the 3D lattice (the same ansatz, the same h, the 3D lattice
    sum) against the 3D relaxed core's own E_stat on that lattice (the quadrature defect of gate 2 drops out: both are lattice sums).
    A tube profile BELOW the 3D core on the 3D lattice means the 3D descent did not reach the radial sector's minimum."""
    out = {}
    items = [("v4abs", 1.5, 48.0, 32, "r16_1_rebuild_n32_L48.npy", "R16-1 n32 L48 (absolute weight)"), ("v4rel", 1.5, 48.0, 32, None, "R17-2 v4rel static n32 L48"),
             ("v4abs", 0.75, 48.0, 64, "r16_1_rebuild_n64_L48_analytic.npy", "R16-1 n64 L48 (absolute weight)")]
    for obj, h, L, n, core_file, core_name in items:
        cfg = make_cfg(obj, h, L)
        cf8 = dict(cfg); cf8["n_samples"] = 8
        row = {"core": core_name, "h": h, "n": n}
        if core_file is not None:
            Mc = np.load(os.path.join(CK16, core_file))
        else:
            Mc = np.load(os.path.join(CK17, "r17_2_v4rel_n32_L48_r16_1.npy"))
        Ec = R.energy_object(Mc, cf8, None, C.radial_ref(cfg), need_grad=False)
        row["E_3D_core"] = float(np.real(Ec[2]["E_stat"])); row["core_domain"] = Ec[3]
        tube = X.tube_setup(h, L)
        for seed in SEEDS:
            tag = tag_of(obj, h, L, seed)
            pf = os.path.join(CK, tag + "_profile.npy")
            if not os.path.exists(pf):
                continue
            P = np.load(pf)
            M3 = X.field_3d_from_profiles(P, tube, cfg)
            E3 = R.energy_object(M3, cf8, None, C.radial_ref(cfg), need_grad=False)
            row[f"E_3D_tube_profile_{seed}"] = float(np.real(E3[2]["E_stat"])); row[f"tube_profile_domain_{seed}"] = E3[3]; row[f"E_tube_{seed}"] = runs[tag]["E"] if tag in runs else None
            row[f"tube_profile_below_core_{seed}"] = bool(row[f"E_3D_tube_profile_{seed}"] < row["E_3D_core"])
            row[f"parts_3D_tube_profile_{seed}"] = {k: float(np.real(v)) for k, v in E3[2].items() if k in ("E_h", "V4", "U", "KP", "reg")}
        row["parts_3D_core"] = {k: float(np.real(v)) for k, v in Ec[2].items() if k in ("E_h", "V4", "U", "KP", "reg")}
        out[f"{obj}_h{h:g}_n{n}"] = row
        log(f"  gate3 3D: {obj} h {h:g} n{n}: core E_3D {row['E_3D_core']:.4f} vs the tube profiles on the 3D lattice " + ", ".join(f"{sd} {row.get(f'E_3D_tube_profile_{sd}', float('nan')):.4f} (tube quadrature {row.get(f'E_tube_{sd}')})" for sd in SEEDS if f"E_3D_tube_profile_{sd}" in row))
    return out


def collect():
    runs = {}
    for p in sorted(glob.glob(os.path.join(CK, "r18_1_*.json"))):
        r_ = json.load(open(p))
        pf = os.path.join(CK, r_["tag"] + "_profile.npy")
        if os.path.exists(pf):
            tube = X.tube_setup(r_["h"], r_["L"])
            r_["reads"].update(gap_reads(np.load(pf), tube, make_cfg(r_["object"], r_["h"], r_["L"])))
        runs[r_["tag"]] = r_
    out = {"rung": "R18-1", "runs": {k: {kk: vv for kk, vv in v.items() if kk not in ("profile_coarse",)} for k, v in runs.items()}, "ladder": {}, "gate3": gate3(runs), "gate3_3D_energy": gate3_3d(runs)}
    for obj in OBJECTS:
        for seed in SEEDS:
            for L in LS + [144.0]:
                rows = []
                for h in HS:
                    tag = tag_of(obj, h, L, seed)
                    if tag in runs:
                        rd = runs[tag]["reads"]
                        rows.append({"h": h, "r_0_profile": rd["r_0_profile (lambda_1 = 0.8 crossing)"], "r_0_taper": rd["r_0_taper (max r with lambda_1 < 0.8)"], "r_0_half": rd["r_0_half (max r with lambda_1 < (1+delta)/2)"],
                                     "r_gap_half": rd.get("r_gap_half (Delta = 0.35 crossing)"), "r_gap_0.1": rd.get("r_gap_0.1 (Delta = 0.1 crossing)"), "r_gap_half_over_h": rd.get("r_gap_half_over_h"), "gap_center": rd.get("gap_center"), "lambda_23_max": rd.get("lambda_23_max"), "trace_center": rd.get("spatial_trace_center"),
                                     "r_0_over_h": rd["r_0_profile_over_h"], "Delta_min": rd["Delta_min"], "lambda_1_center": rd["lambda_1_center"], "E": runs[tag]["E"], "E_h": rd["parts"]["E_h"], "KP": rd["parts"]["KP"], "V4": rd["parts"]["V4"],
                                     "tail_A": rd.get("tail_A_median_e_r4"), "tail_slope": rd.get("tail_loglog_slope"), "bound_cells": runs[tag]["descent"]["cells_at_gap_bound"], "nit": runs[tag]["descent"]["nit"], "grad_max": runs[tag]["descent"]["grad_max_free"]})
                if not rows:
                    continue
                key = f"{obj}|{seed}|L{int(L)}"
                lad = {"rows": rows}
                # Richardson in h: r_0(h) = r_0(0) + c h^p from the three finest
                if len(rows) >= 3 and all(r_["r_gap_half"] is not None for r_ in rows[-3:]):
                    rgh = np.array([r_["r_gap_half"] for r_ in rows[-3:]]); hh3 = np.array([r_["h"] for r_ in rows[-3:]])
                    lad["finest_pair_rel_change_r_gap_half"] = float(abs(rgh[2] - rgh[1]) / max(abs(rgh[2]), 1e-300))
                    if (rgh[0] - rgh[1]) * (rgh[1] - rgh[2]) > 0 and rgh[1] != rgh[2]:
                        pg = np.log((rgh[0] - rgh[1]) / (rgh[1] - rgh[2])) / np.log(hh3[0] / hh3[1])
                        lad["richardson_r_gap_half"] = {"order_p": float(pg), "extrapolated": float(rgh[2] + (rgh[2] - rgh[1]) / ((hh3[1] / hh3[2]) ** pg - 1.0)) if pg > 0 else None}
                if len(rows) >= 3:
                    r0 = np.array([r_["r_0_profile"] for r_ in rows[-3:]]); hh = np.array([r_["h"] for r_ in rows[-3:]])
                    E = np.array([r_["E"] for r_ in rows[-3:]])
                    if r0[0] != r0[1] and r0[1] != r0[2] and (r0[0] - r0[1]) * (r0[1] - r0[2]) > 0:
                        p = np.log((r0[0] - r0[1]) / (r0[1] - r0[2])) / np.log(hh[0] / hh[1])
                        lad["richardson_r0"] = {"order_p": float(p), "r_0_extrapolated": float(r0[2] + (r0[2] - r0[1]) / ((hh[1] / hh[2]) ** p - 1.0)) if p > 0 else None}
                    if (E[0] - E[1]) * (E[1] - E[2]) > 0 and E[1] != E[2]:
                        pE = np.log((E[0] - E[1]) / (E[1] - E[2])) / np.log(hh[0] / hh[1])
                        lad["richardson_E"] = {"order_p": float(pE), "E_extrapolated": float(E[2] + (E[2] - E[1]) / ((hh[1] / hh[2]) ** pE - 1.0)) if pE > 0 else None}
                    lad["finest_pair_rel_change_r0"] = float(abs(r0[2] - r0[1]) / max(abs(r0[2]), 1e-300))
                    lad["finest_pair_rel_change_E"] = float(abs(E[2] - E[1]) / max(abs(E[2]), 1e-300))
                    lad["r0_over_h_finest_three"] = [r_["r_0_over_h"] for r_ in rows[-3:]]
                out["ladder"][key] = lad
    # the LOWEST basin per rung (the two seeds can land in different local minima of the radial sector: object A at h 0.75 did)
    for obj in OBJECTS:
        for L in LS + [144.0]:
            rows, basins = [], []
            for h in HS:
                cands = [(runs[tag_of(obj, h, L, sd)], sd) for sd in SEEDS if tag_of(obj, h, L, sd) in runs]
                if not cands:
                    continue
                best, sd = min(cands, key=lambda c: c[0]["E"])
                rd = best["reads"]
                rows.append({"h": h, "seed": sd, "r_0_profile": rd["r_0_profile (lambda_1 = 0.8 crossing)"], "r_0_taper": rd["r_0_taper (max r with lambda_1 < 0.8)"], "r_gap_half": rd.get("r_gap_half (Delta = 0.35 crossing)"), "r_gap_0.1": rd.get("r_gap_0.1 (Delta = 0.1 crossing)"),
                             "r_gap_half_over_h": rd.get("r_gap_half_over_h"), "gap_center": rd.get("gap_center"), "lambda_1_center": rd["lambda_1_center"], "lambda_23_max": rd.get("lambda_23_max"), "Delta_min": rd["Delta_min"], "E": best["E"], "E_h": rd["parts"]["E_h"], "KP": rd["parts"]["KP"],
                             "tail_A": rd.get("tail_A_median_e_r4"), "bound_cells": best["descent"]["cells_at_gap_bound"], "grad_max": best["descent"]["grad_max_free"], "nit": best["descent"]["nit"], "r_0_over_h": rd["r_0_profile_over_h"]})
                if len(cands) == 2 and abs(cands[0][0]["E"] - cands[1][0]["E"]) > 1e-6 * abs(best["E"]):
                    basins.append({"h": h, "E_by_seed": {c[1]: c[0]["E"] for c in cands}, "r_0_by_seed": {c[1]: c[0]["reads"]["r_0_profile (lambda_1 = 0.8 crossing)"] for c in cands}, "lambda_1_center_by_seed": {c[1]: c[0]["reads"]["lambda_1_center"] for c in cands}, "lowest": sd})
            if rows:
                lad = {"rows": rows, "two_basins_at": basins}
                for key, kk in (("r_gap_half", "richardson_r_gap_half"), ("r_0_profile", "richardson_r0"), ("E", "richardson_E")):
                    if len(rows) >= 3 and all(r_[key] is not None for r_ in rows[-3:]):
                        v = np.array([r_[key] for r_ in rows[-3:]]); hh = np.array([r_["h"] for r_ in rows[-3:]])
                        lad[f"finest_pair_rel_change_{key}"] = float(abs(v[2] - v[1]) / max(abs(v[2]), 1e-300))
                        if (v[0] - v[1]) * (v[1] - v[2]) > 0 and v[1] != v[2]:
                            pp = np.log((v[0] - v[1]) / (v[1] - v[2])) / np.log(hh[0] / hh[1])
                            lad[kk] = {"order_p": float(pp), "extrapolated": float(v[2] + (v[2] - v[1]) / ((hh[1] / hh[2]) ** pp - 1.0)) if pp > 0 else None}
                out["ladder"][f"{obj}|lowest|L{int(L)}"] = lad
    # the L-exponent at fixed h
    Lexp = {}
    for obj in OBJECTS:
        for seed in SEEDS:
            for h in HS:
                vals = {}
                for L in LS + [144.0]:
                    lad = out["ladder"].get(f"{obj}|lowest|L{int(L)}", {})
                    row = [r_ for r_ in lad.get("rows", []) if r_["h"] == h]
                    if row and seed == "r16_1":                    # the lowest-basin ladder carries the L dependence
                        vals[L] = (row[0]["r_0_profile"], row[0]["E"], row[0]["r_gap_half"])
                if len(vals) >= 2 and seed == "r16_1":
                    Ls = sorted(vals)
                    row = {"L": Ls, "r_0": [vals[l_][0] for l_ in Ls], "E": [vals[l_][1] for l_ in Ls], "r_gap_half": [vals[l_][2] for l_ in Ls]}
                    row["r_0_rel_change_48_96"] = abs(vals[Ls[1]][0] - vals[Ls[0]][0]) / max(abs(vals[Ls[0]][0]), 1e-300)
                    row["r_gap_half_rel_change_48_96"] = abs(vals[Ls[1]][2] - vals[Ls[0]][2]) / max(abs(vals[Ls[0]][2]), 1e-300) if None not in (vals[Ls[1]][2], vals[Ls[0]][2]) else None
                    if len(Ls) == 3:
                        row["r_0_rel_change_96_144"] = abs(vals[Ls[2]][0] - vals[Ls[1]][0]) / max(abs(vals[Ls[1]][0]), 1e-300)
                        row["r_gap_half_rel_change_96_144"] = abs(vals[Ls[2]][2] - vals[Ls[1]][2]) / max(abs(vals[Ls[1]][2]), 1e-300) if None not in (vals[Ls[2]][2], vals[Ls[1]][2]) else None
                        row["E_rel_change_96_144"] = abs(vals[Ls[2]][1] - vals[Ls[1]][1]) / max(abs(vals[Ls[1]][1]), 1e-300)
                    Lexp[f"{obj}|lowest|h{h:g}"] = row
                    continue
                for L in LS + [144.0]:
                    tag = tag_of(obj, h, L, seed)
                    if tag in runs:
                        vals[L] = (runs[tag]["reads"]["r_0_profile (lambda_1 = 0.8 crossing)"], runs[tag]["E"])
                if len(vals) >= 2:
                    Ls = sorted(vals)
                    row = {"L": Ls, "r_0": [vals[l_][0] for l_ in Ls], "E": [vals[l_][1] for l_ in Ls]}
                    row["r_0_rel_change_48_96"] = abs(vals[Ls[1]][0] - vals[Ls[0]][0]) / max(abs(vals[Ls[0]][0]), 1e-300)
                    if len(Ls) == 3:
                        row["r_0_rel_change_96_144"] = abs(vals[Ls[2]][0] - vals[Ls[1]][0]) / max(abs(vals[Ls[1]][0]), 1e-300)
                        row["E_rel_change_96_144"] = abs(vals[Ls[2]][1] - vals[Ls[1]][1]) / max(abs(vals[Ls[1]][1]), 1e-300)
                    Lexp[f"{obj}|{seed}|h{h:g}"] = row
    out["L_dependence"] = Lexp
    # outcomes per object (the melted seed as the primary, the r16_1 seed as the check that the two converge to the same minimum)
    outcomes = {}
    for obj in OBJECTS:
        lad = out["ladder"].get(f"{obj}|lowest|L96", {})
        rows = lad.get("rows", [])
        o = {"n_rungs_L96": len(rows), "two_basins_at_L96": lad.get("two_basins_at"), "two_basins_at_L48": out["ladder"].get(f"{obj}|lowest|L48", {}).get("two_basins_at")}
        if len(rows) >= 2:
            fin = lad.get("finest_pair_rel_change_r_0_profile")
            fing = lad.get("finest_pair_rel_change_r_gap_half")
            r0h = [r_["r_0_over_h"] for r_ in rows]
            o["finest_pair_rel_change_r0"] = fin; o["finest_pair_rel_change_r_gap_half"] = fing
            o["r_0_over_h_ladder"] = r0h
            o["r_0_ladder"] = [r_["r_0_profile"] for r_ in rows]
            o["r_gap_half_ladder"] = [r_["r_gap_half"] for r_ in rows]
            o["r_gap_half_over_h_ladder"] = [r_["r_gap_half_over_h"] for r_ in rows]
            o["gap_center_ladder"] = [r_["gap_center"] for r_ in rows]
            o["lambda_1_center_ladder"] = [r_["lambda_1_center"] for r_ in rows]
            o["E_ladder"] = [r_["E"] for r_ in rows]
            o["h_ladder"] = [r_["h"] for r_ in rows]
            o["finest_pair_rel_change_E"] = lad.get("finest_pair_rel_change_E")
            o["richardson"] = {k: lad.get(k) for k in ("richardson_r_gap_half", "richardson_r0", "richardson_E")}
            o["bound_cells_ladder"] = [r_["bound_cells"] for r_ in rows]
            o["lambda_23_max_ladder"] = [r_["lambda_23_max"] for r_ in rows]
            o["seed_of_lowest_ladder"] = [r_["seed"] for r_ in rows]
            L144 = out["L_dependence"].get(f"{obj}|lowest|h0.15", {}).get("r_gap_half_rel_change_96_144")
            o["L_dependence_h0.15"] = out["L_dependence"].get(f"{obj}|lowest|h0.15")
            o["L_96_to_144_rel_change_at_h0.15"] = L144
            rgh = [r_ for r_ in o["r_gap_half_ladder"] if r_ is not None]
            gc = [g_ for g_ in o["gap_center_ladder"] if g_ is not None]
            # the r_0 (lambda_1 = 0.8) read is lost on the fine rungs (lambda_1 stays above 0.8); the gap read carries the ladder
            grid_set = len(rgh) >= 3 and rgh[-1] < 0.5 * rgh[0] and all(r_["r_gap_half_over_h"] is not None for r_ in rows[-3:]) and max(r_["r_gap_half_over_h"] for r_ in rows[-3:]) / max(min(r_["r_gap_half_over_h"] for r_ in rows[-3:]), 1e-300) < 1.5
            gap_closing = len(gc) >= 3 and gc[-1] < 0.5 * gc[-2] < 0.25 * gc[-3]
            if fing is not None and fing < 0.05 and (L144 is None or L144 < 0.05):
                o["outcome"] = "RADIAL_CONVERGED (on the gap read r_gap_half)"
                o["r_0_sqrt_mu"] = {"r_gap_half": rgh[-1], "value_r_gap_half": rgh[-1] * 0.1, "r_0_profile_finest_nonzero": ([r_ for r_ in o["r_0_ladder"] if r_ > 0] or [None])[-1], "h_uncertainty_rel": fing, "L_uncertainty_rel": L144, "richardson": lad.get("richardson_r_gap_half"), "gap_center_finest": gc[-1] if gc else None}
            elif grid_set:
                o["outcome"] = "CORE_GRID_SET (r_gap_half falls with h, r_gap_half / h roughly constant on the finest rungs)"
            elif gap_closing:
                o["outcome"] = "CORE_GAP_CLOSING (the center's director gap falls with h without a floor: the continuum core is the isotropic point, escape (d) in the limit)"
            else:
                o["outcome"] = "UNRESOLVED_ON_THE_LADDER (neither converged at 5 percent on the gap read nor grid-set nor gap-closing by the rules)"
        # the two seeds agree?
        agree = []
        for L in LS:
            for h in HS:
                a, b = tag_of(obj, h, L, "r16_1"), tag_of(obj, h, L, "melted")
                if a in runs and b in runs:
                    agree.append({"h": h, "L": L, "E_r16_1": runs[a]["E"], "E_melted": runs[b]["E"], "rel_E": abs(runs[a]["E"] - runs[b]["E"]) / max(abs(runs[b]["E"]), 1e-300),
                                  "r0_r16_1": runs[a]["reads"]["r_0_profile (lambda_1 = 0.8 crossing)"], "r0_melted": runs[b]["reads"]["r_0_profile (lambda_1 = 0.8 crossing)"]})
        o["two_seeds"] = agree
        o["seeds_agree_1e-6_in_E"] = bool(agree and all(a_["rel_E"] < 1e-6 for a_ in agree))
        outcomes[obj] = o
    out["outcomes"] = outcomes
    g3 = out["gate3"]
    out["gate3_summary"] = {k: {"r_0_within_10pct": v["r_0_within_10pct"], "E_within_5pct": v["E_within_5pct"], "r_0_rel_dev": v["r_0_rel_dev"], "E_rel_dev": v.get("E_rel_dev")} for k, v in g3.items()}
    json.dump(out, open(os.path.join(DATA, "m5_32_r18_1.json"), "w"), indent=1, default=float)
    log(f"collected {len(runs)} runs; outcomes {json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ('outcome', 'r_0_ladder', 'r_gap_half_ladder', 'gap_center_ladder', 'lambda_1_center_ladder', 'E_ladder', 'h_ladder', 'finest_pair_rel_change_r_gap_half', 'finest_pair_rel_change_E', 'L_96_to_144_rel_change_at_h0.15', 'seeds_agree_1e-6_in_E', 'r_0_sqrt_mu')} for k, v in outcomes.items()}, default=float)}; gate3 {out['gate3_summary']}")
    plot(runs, out)
    return out


def plot(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for oi, obj in enumerate(OBJECTS):
        a = ax[oi, 0]
        for h in HS:
            tag = tag_of(obj, h, 96.0, "melted")
            if tag in runs:
                pc = runs[tag]["profile_coarse"]
                a.plot(pc["r"], pc["lambda_1"], lw=0.9, label=f"h {h:g}")
                a.plot(pc["r"], pc["lambda_23"], lw=0.6, ls="--", color=a.lines[-1].get_color())
        a.axhline(0.8, color="r", ls=":", lw=0.7); a.set_xlim(0, 12); a.set_xlabel("r (box units)"); a.set_ylabel("lambda_1 (solid), lambda_23 (dashed)")
        a.set_title(f"{obj}: the relaxed profiles, L 96, melted seed", fontsize=8); a.legend(fontsize=6)
        a = ax[oi, 1]
        for L in LS + [144.0]:
            for seed, mk in (("melted", "o"), ("r16_1", "x")):
                key = f"{obj}|{seed}|L{int(L)}"
                lad = out["ladder"].get(key)
                if lad:
                    a.plot([r_["h"] for r_ in lad["rows"]], [r_["r_0_profile"] for r_ in lad["rows"]], marker=mk, ms=4, lw=0.8, label=f"L {int(L)} {seed}: r_0 (lambda_1 = 0.8)")
                    a.plot([r_["h"] for r_ in lad["rows"]], [r_["r_gap_half"] for r_ in lad["rows"]], marker=mk, ms=4, lw=0.8, ls="--", color=a.lines[-1].get_color(), label=f"L {int(L)} {seed}: r_gap_half")
        a.set_xscale("log"); a.set_xlabel("h"); a.set_ylabel("r_0 (lambda_1 = 0.8 crossing)"); a.set_title(f"{obj}: r_0 vs h ({out['outcomes'].get(obj, {}).get('outcome', '')})", fontsize=8); a.legend(fontsize=6)
        for r0, lab in ((3.043, "n32 L48 core"), (3.111, "n48 L72"), (0.999, "n64 L48")):
            a.axhline(r0, color="gray", lw=0.5, ls=":")
        a = ax[oi, 2]
        for L in LS + [144.0]:
            key = f"{obj}|melted|L{int(L)}"
            lad = out["ladder"].get(key)
            if lad:
                a.plot([r_["h"] for r_ in lad["rows"]], [r_["E"] for r_ in lad["rows"]], "o-", ms=4, lw=0.8, label=f"L {int(L)}")
        a.set_xscale("log"); a.set_xlabel("h"); a.set_ylabel("E_stat (tube)"); a.set_title(f"{obj}: E vs h", fontsize=8); a.legend(fontsize=6)
    fig.suptitle("R18-1: the radial (tube) solve of the spherically symmetric sector; the dotted lines mark the 3D cores' r_0", fontsize=9)
    fig.savefig(os.path.join(PLOTS, "m5_32_r18_1_radial.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    out["plot"] = "plots/m5_32_r18_1_radial.png"
    json.dump(out, open(os.path.join(DATA, "m5_32_r18_1.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scan", "collect", "one"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--object", default="v4abs"); ap.add_argument("--h", type=float, default=1.5); ap.add_argument("--L", type=float, default=48.0); ap.add_argument("--seed", default="melted")
    a = ap.parse_args(ARGS)
    if a.mode == "scan":
        scan(a.workers)
    elif a.mode == "one":
        log(job((a.object, a.h, a.L, a.seed)))
    else:
        collect()
