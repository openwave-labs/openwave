"""M5.32 R13-W, step W3: the hedgehog core inside a FREE spherical degenerate shell
at fixed J (the frozen packet, ledger 6.2, verbatim), the decisive 3D read.

OBJECT       the R10 relaxed hedgehog (n32 L48: R10's 12000-iteration end state;
             other boxes: B8.dressed(cfg, 0) relaxed 3000 iterations under the R10
             protocol), with a ONE-CELL degenerate shell at radius R_s imposed once
             (cells with |r - R_s| < h/2 projected: the two smallest spatial
             eigenvalues replaced by their mean), then FREE.
CONVENTION   the author's two-region clock: the interior (r < R_s) rotates at omega,
             the exterior is at rest; a0 = a0_local(M) 1[r < R_s], a0_local = the
             rotation about the local leading eigenvector (R12; = B8.a0_unit on the
             hedgehog).  The indicator's edge sits on the surface where a0 = 0
             identically while the shell holds (S1), so it is the wall, not a taper;
             if the shell melts the edge becomes a hard mask on a nonzero a0, which
             is the failure the W2 gates describe (logged as a deviation from the
             packet's "no mask" line, with this reason).  CONTROL: the rigid
             (unmasked) generator read on the same end states (the closed R7-R12
             extensive case) and the g = 32 shell numbers (static).
FUNCTIONAL   E_J[M] = E_stat[M] + J^2 / (4 kin[M]), kin = INS4.kin_of(M, a0(M)),
             a0 refreshed every step, frozen in the gradient (m5_32_r13w_common.fire_proj).
OBSERVABLES  R*: argmin over R_s of the relaxed E_J at fixed J (tracks the box = fail);
             omega* = J / (2 kin*); dE/dJ = omega closure (central difference over
             J = 180 / 200 / 220 at R_s = 9, n32); shell survival: the radial gap
             profile d2 - d3 (min over the shell bin) at the end vs the start, and
             the kin radial profile (where the inertia sits).
LADDERS      n32 L48 (h 1.5): R_s in {6, 9, 12, 15}, J in {50, 200, 800}, 3000 it,
             plus R_s = 9 J = 200 to 12000 it; n48 L48 (h 1.0): R_s = 9, J = 200;
             n48 L72 (h 1.5): R_s in {9, 15, 21}, J = 200.
VERDICTS     pass: PERIODIC_ORBIT_EXISTS at the relaxed-field level (shell survives,
             R* interior to the box, dE/dJ = omega).  fail (shell melts, or R* tracks
             L): ESTABLISHED_KINEMATIC at best.
W0 PREDICTION (S5, S6): the shell SURVIVES (kinetic reward >> V4 cost) and the
             relaxation does not converge to a finite omega: kin grows monotonically
             (free inertia, lattice-limited), omega falls, E_J(R_s) decreases with R_s
             (R* beyond the box), omega* ~ h.

Modes:  run n L Rs J maxit      one relaxation (cached in ../checkpoints/m5_32_r13w/)
        collect                  assemble ../data/m5_32_r13w_w3.json + ../plots/m5_32_r13w_w3.png
Run:    /opt/anaconda3/envs/master312/bin/python3 m5_32_r13w_w3.py run 32 48 9 200 3000
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m5_32_r13w_common", os.path.join(HERE, "m5_32_r13w_common.py"))
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
INS4 = C.INS4
OUT = os.path.join(C.DATA, "m5_32_r13w_w3.json")
PNG = os.path.join(C.PLOTS, "m5_32_r13w_w3.png")


def radial(cfg):
    X, Y, Z = INS4.coords(cfg["n"], cfg["h"])
    return np.sqrt(X * X + Y * Y + Z * Z)


def shell_mask(cfg, Rs):
    r = radial(cfg)
    return np.abs(r - Rs) < 0.5 * cfg["h"]


def interior_mask(cfg, Rs):
    return radial(cfg) < Rs


def a0_two_region(M, cfg, Rs):
    return C.a0_local(M) * interior_mask(cfg, Rs)[..., None, None]


def radial_profiles(M, cfg, a0, dr=None):
    """per radial bin (width h): min and mean of the gap d2 - d3, and the kin density sum."""
    r = radial(cfg)
    dr = dr or cfg["h"]
    gap = C.gap23(M)
    kd = C.kin_density(M, a0, cfg)
    edges = np.arange(0.0, cfg["L"] / 2 * np.sqrt(3.0) + dr, dr)      # to the box corner (audit J3)
    rows = []
    for i in range(len(edges) - 1):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if not m.any():
            continue
        rows.append({"r_lo": float(edges[i]), "r_hi": float(edges[i + 1]), "cells": int(m.sum()),
                     "gap_min": float(gap[m].min()), "gap_mean": float(gap[m].mean()),
                     "kin_bin": float(kd[m].sum())})
    return rows


def run(n, L, Rs, J, maxit, frozen=False):
    key = f"w3_n{n}_L{L:g}_R{Rs:g}_J{J:g}_it{maxit}" + ("_frz" if frozen else "")
    js, npy = os.path.join(C.CK, key + ".json"), os.path.join(C.CK, key + ".npy")
    if os.path.exists(js) and os.path.exists(npy):
        C.log(f"{key}: restored")
        return json.load(open(js))
    seed, cfg, srec = C.seed_hedgehog(n, L)
    r = radial(cfg)
    sh = shell_mask(cfg, Rs)
    M0 = C.degenerate_project(seed, sh)
    free = ~INS4.pin_shell(n, cfg["h"])
    if frozen:
        # audit J3 control: the generator FROZEN at the shelled seed (the R10/R12 convention), so the
        # descended functional is exactly E_stat + J^2/(4 kin[M; a0 fixed])
        a0_fixed = a0_two_region(M0, cfg, Rs)
        a0_of = lambda M: a0_fixed
    else:
        a0_of = lambda M: a0_two_region(M, cfg, Rs)
    # start numbers
    e_u0, e_v0 = INS4.e_parts(seed, cfg)
    e_u1, e_v1 = INS4.e_parts(M0, cfg)
    a00 = a0_of(M0)
    k_seed_masked = float(INS4.kin_of(seed, a0_two_region(seed, cfg, Rs), cfg))
    k_seed_rigid = float(INS4.kin_of(seed, C.a0_local(seed), cfg))
    k0 = float(INS4.kin_of(M0, a00, cfg))
    kd0 = C.kin_density(M0, a00, cfg)
    start = {"E_u_seed": float(e_u0), "V4_seed": float(e_v0), "E_u_shell": float(e_u1), "V4_shell": float(e_v1),
             "shell_cells": int(sh.sum()), "V4_cost_of_shell": float(e_v1 - e_v0), "E_u_cost_of_shell": float(e_u1 - e_u0),
             "kin_seed_two_region": k_seed_masked, "kin_seed_rigid": k_seed_rigid,
             "kin_shell_two_region": k0, "kin_shell_rigid": float(INS4.kin_of(M0, C.a0_local(M0), cfg)),
             "kin_in_flank_bin": float(kd0[np.abs(r - (Rs - cfg["h"])) < 0.5 * cfg["h"]].sum()),
             "E_J_seed_two_region": float(e_u0 + e_v0 + J * J / (4 * k_seed_masked)),
             "E_J_start": float(e_u1 + e_v1 + J * J / (4 * k0)), "omega_start": J / (2 * k0),
             "profiles": radial_profiles(M0, cfg, a00)}
    C.log(f"{key}: seed E_u {e_u0:.4f} V4 {e_v0:.4f}; shell adds E_u {start['E_u_cost_of_shell']:+.4e} V4 {start['V4_cost_of_shell']:+.4e}; "
          f"kin two-region {k_seed_masked:.3f} -> {k0:.3f} (rigid {k_seed_rigid:.3f}); E_J {start['E_J_seed_two_region']:.4f} -> {start['E_J_start']:.4f}")

    def diag(M):
        g = C.gap23(M)
        return {"gap_shell_min": float(g[sh].min()), "gap_shell_mean": float(g[sh].mean()),
                "gap_global_min": float(g[free].min()), "n_cells_gap_lt_0.03": int((g[free] < 0.03).sum())}
    M, info = C.fire_proj(M0, cfg, free, maxit, project=None, J=J, a0_of=a0_of, tag=key, log_every=250, diag=diag)
    a0 = a0_of(M)
    e_u, e_v = INS4.e_parts(M, cfg)
    k = float(INS4.kin_of(M, a0, cfg))
    k_frozen_seed_a0 = float(INS4.kin_of(M, a0_two_region(M0, cfg, Rs), cfg))
    k_refreshed = float(INS4.kin_of(M, a0_two_region(M, cfg, Rs), cfg))
    end = {"E_u": float(e_u), "V4": float(e_v), "kin": k, "E_stat": float(e_u + e_v), "E_J": float(e_u + e_v + J * J / (4 * k)),
           "omega": J / (2 * k), "kin_rigid_control": float(INS4.kin_of(M, C.a0_local(M), cfg)),
           "frozen": frozen, "kin_with_seed_a0": k_frozen_seed_a0, "kin_with_refreshed_a0": k_refreshed,
           "E_J_with_seed_a0": float(e_u + e_v + J * J / (4 * k_frozen_seed_a0)), "E_J_with_refreshed_a0": float(e_u + e_v + J * J / (4 * k_refreshed)),
           "profiles": radial_profiles(M, cfg, a0), **diag(M)}
    tr = info["trace"]
    kins = [row["kin"] for row in tr]
    rec = {"n": n, "L": L, "h": cfg["h"], "Rs": Rs, "J": J, "maxit": maxit, "frozen": frozen, "seed": srec, "stop": info["stop"],
           "iters": info["iters"], "wall_s": info["wall_s"], "start": start, "end": end, "trace": tr,
           "kin_monotone_increasing": bool(all(b >= a for a, b in zip(kins, kins[1:]))),
           "kin_last_quarter_growth": float(kins[-1] / kins[max(0, 3 * len(kins) // 4 - 1)] - 1.0) if len(kins) > 3 else None}
    np.save(npy, M)
    json.dump(rec, open(js, "w"), indent=1)
    C.log(f"{key}: stop {info['stop']} @ {info['iters']} in {info['wall_s']}s | E_J {end['E_J']:.4f} E_stat {end['E_stat']:.4f} "
          f"kin {k:.3f} omega {end['omega']:.5f} | shell gap min {end['gap_shell_min']:.4f} mean {end['gap_shell_mean']:.4f} "
          f"| kin monotone {rec['kin_monotone_increasing']}")
    return rec


def collect():
    recs_all = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(C.CK, "w3_*.json")))]
    recs = [r for r in recs_all if not r.get("frozen")]
    frz = [r for r in recs_all if r.get("frozen")]
    res = {"runs": [{k: v for k, v in r.items() if k != "trace"} | {"trace_tail": r["trace"][-4:]} for r in recs]}
    summ = {}
    # R* ladders at fixed J
    for (n, L, J), grp in _group(recs, lambda r: (r["n"], r["L"], r["J"])):
        grp = [r for r in grp if r["maxit"] == 3000]
        if len(grp) < 2:
            continue
        grp.sort(key=lambda r: r["Rs"])
        EJ = [r["end"]["E_J"] for r in grp]
        Rs = [r["Rs"] for r in grp]
        i = int(np.argmin(EJ))
        summ[f"Rstar_n{n}_L{L:g}_J{J:g}"] = {
            "Rs": Rs, "E_J": EJ, "E_stat": [r["end"]["E_stat"] for r in grp], "kin": [r["end"]["kin"] for r in grp],
            "omega": [r["end"]["omega"] for r in grp], "R_star": Rs[i],
            "at_box_edge": bool(i == len(grp) - 1), "monotone_decreasing": bool(all(b < a for a, b in zip(EJ, EJ[1:]))),
            "shell_gap_min_end": [r["end"]["gap_shell_min"] for r in grp],
            "shell_gap_mean_end": [r["end"]["gap_shell_mean"] for r in grp]}
    # dE/dJ closure at R_s = 9, n32
    dj = {r["J"]: r for r in recs if r["n"] == 32 and r["Rs"] == 9 and r["maxit"] == 3000 and r["J"] in (180.0, 200.0, 220.0)}
    if len(dj) == 3:
        dEdJ = (dj[220.0]["end"]["E_J"] - dj[180.0]["end"]["E_J"]) / 40.0
        summ["dEdJ_closure_n32_R9"] = {"dE_dJ_central": dEdJ, "omega_at_200": dj[200.0]["end"]["omega"],
                                       "rel_dev": dEdJ / dj[200.0]["end"]["omega"] - 1.0,
                                       "E_J": {str(k): v["end"]["E_J"] for k, v in dj.items()},
                                       "kin": {str(k): v["end"]["kin"] for k, v in dj.items()}}
    # h and L dependence at R_s = 9, J = 200
    pts = {(r["n"], r["L"]): r for r in recs if r["Rs"] == 9 and r["J"] == 200.0 and r["maxit"] == 3000}
    summ["omega_star_vs_box"] = {f"n{n}_L{L:g}_h{r['h']:.3g}": {"omega": r["end"]["omega"], "kin": r["end"]["kin"],
                                                             "E_J": r["end"]["E_J"], "gap_shell_min": r["end"]["gap_shell_min"]}
                                 for (n, L), r in pts.items()}
    # long run
    lng = [r for r in recs if r["maxit"] > 3000]
    if lng:
        r = lng[0]
        summ["long_run"] = {"key": f"n{r['n']}_L{r['L']:g}_R{r['Rs']:g}_J{r['J']:g}_it{r['maxit']}", "stop": r["stop"],
                            "kin_trace": [(row["it"], row["kin"], row["omega"], row["E"]) for row in r["trace"][::4]],
                            "kin_monotone": r["kin_monotone_increasing"], "end": {k: v for k, v in r["end"].items() if k != "profiles"}}
    # end-field cross-checks recomputed from the saved fields (audit A2 / J1: 19 of 20 records predate the schema)
    xchk = {}
    for r in recs_all:
        key = f"n{r['n']}_L{r['L']:g}_R{r['Rs']:g}_J{r['J']:g}_it{r['maxit']}" + ("_frz" if r.get("frozen") else "")
        npy = os.path.join(C.CK, "w3_" + key + ".npy")
        if not os.path.exists(npy):
            continue
        M = np.load(npy); seed, cfg, _ = C.seed_hedgehog(r["n"], r["L"])
        M0 = C.degenerate_project(seed, shell_mask(cfg, r["Rs"]))
        a_seed, a_ref = a0_two_region(M0, cfg, r["Rs"]), a0_two_region(M, cfg, r["Rs"])
        e_u, e_v = INS4.e_parts(M, cfg)
        ks, kr = float(INS4.kin_of(M, a_seed, cfg)), float(INS4.kin_of(M, a_ref, cfg))
        J = r["J"]
        # where the refreshed-generator inertia sits: two layers inside the mask edge vs the rest
        rr = radial(cfg); kd = C.kin_density(M, a_ref, cfg)
        inner2 = (rr < r["Rs"]) & (rr >= r["Rs"] - 2 * cfg["h"])
        g = C.gap23(M)
        xchk[key] = {"kin_seed_a0": ks, "kin_refreshed_a0": kr, "E_J_seed_a0": float(e_u + e_v + J * J / (4 * ks)),
                     "E_J_refreshed_a0": float(e_u + e_v + J * J / (4 * kr)),
                     "kin_fraction_in_two_layers_inside_edge": float(kd[inner2].sum() / max(kd.sum(), 1e-300)),
                     "gap_max_in_those_layers": float(g[inner2].max()), "d3_min_in_those_layers": float(C.spatial_eigs(M)[inner2][..., 2].min()),
                     "shell_gap_mean_end": float(g[shell_mask(cfg, r["Rs"])].mean()), "seed_kin_two_region": float(INS4.kin_of(seed, a0_two_region(seed, cfg, r["Rs"]), cfg))}
    summ["end_field_cross_checks"] = xchk
    # frozen-a0 control (audit J3)
    if frz:
        summ["frozen_a0_control"] = [{"key": f"n{r['n']}_L{r['L']:g}_R{r['Rs']:g}_J{r['J']:g}_it{r['maxit']}", "stop": r["stop"],
                                      "kin_start": r["start"]["kin_shell_two_region"], "kin_end": r["end"]["kin"], "omega_end": r["end"]["omega"],
                                      "E_J_end": r["end"]["E_J"], "kin_monotone": r["kin_monotone_increasing"],
                                      "gap_shell_min_end": r["end"]["gap_shell_min"]} for r in frz]
    # refreshed vs seed-a0 E_J on the same end fields (audit J3)
    summ["E_J_refreshed_vs_seed_a0_on_end_fields"] = "see end_field_cross_checks (recomputed from the saved fields)"
    # bag closure from MEASURED inputs (audit F4 to F6): the seed's two-region interior law and the 3D shell cost per area
    seeds32 = sorted([r for r in recs if r["n"] == 32 and r["J"] == 200.0 and r["maxit"] == 3000], key=lambda r: r["Rs"])
    if len(seeds32) >= 3:
        Rs_ = np.array([r["Rs"] for r in seeds32]); kin_ = np.array([r["start"]["kin_seed_two_region"] for r in seeds32])
        sig_ = np.array([(r["start"]["E_u_cost_of_shell"] + r["start"]["V4_cost_of_shell"]) / (4 * np.pi * r["Rs"] ** 2) for r in seeds32])
        p_, la_ = np.polyfit(np.log(Rs_), np.log(kin_), 1)
        a_ = float(np.exp(la_)); sig_mean = float(sig_.mean())
        Rg = np.linspace(2.0, 400.0, 200000)
        bag = {"kin_seed_two_region": dict(zip([str(v) for v in Rs_], kin_.tolist())), "power_law_exponent": float(p_), "power_law_prefactor": a_,
               "shell_cost_per_area_3D": dict(zip([str(v) for v in Rs_], sig_.tolist())), "sigma_3D_mean": sig_mean,
               "note": "E_J(R) = 4 pi R^2 sigma_3D + J^2 / (4 a R^p) on the SEED family (before the fixed-J descent inflates kin)"}
        for Jv in (50.0, 200.0, 800.0):
            E = 4 * np.pi * Rg ** 2 * sig_mean + Jv ** 2 / (4 * a_ * Rg ** p_)
            i = int(np.argmin(E))
            bag[f"J{Jv:g}"] = {"R_star_seed_family": float(Rg[i]), "omega_star_seed_family": float(Jv / (2 * a_ * Rg[i] ** p_)),
                               "inside_free_region_n32": bool(Rg[i] < 24.0 - 2 * 1.5)}
        summ["bag_closure_measured_inputs"] = bag
    # survival + monotone kin over all runs
    def melt(r):
        tr = r["trace"] if "trace" in r else None
        first, last = (tr[0], tr[-1]) if tr else (None, None)
        return {"gap_shell_mean_first_log": first["gap_shell_mean"] if first else None, "gap_shell_mean_end": r["end"]["gap_shell_mean"],
                "cells_gap_lt_0.03_first_log": first["n_cells_gap_lt_0.03"] if first else None, "cells_gap_lt_0.03_end": r["end"]["n_cells_gap_lt_0.03"],
                "melting": bool(first is not None and r["end"]["gap_shell_mean"] > first["gap_shell_mean"])}
    full = {f"n{r['n']}_L{r['L']:g}_R{r['Rs']:g}_J{r['J']:g}_it{r['maxit']}": json.load(open(os.path.join(C.CK, f"w3_n{r['n']}_L{r['L']:g}_R{r['Rs']:g}_J{r['J']:g}_it{r['maxit']}.json"))) for r in recs}
    summ["shell_melting"] = {k: melt(v) for k, v in full.items()}
    summ["shell_melting_all_runs"] = bool(all(v["melting"] for v in summ["shell_melting"].values()))
    summ["shell_gap_mean_end_range"] = [min(r["end"]["gap_shell_mean"] for r in recs), max(r["end"]["gap_shell_mean"] for r in recs)]
    def dips(v):
        ks = [row["kin"] for row in v["trace"]]
        return {"net_growth": ks[-1] / ks[0] - 1.0, "largest_dip": min([b / a - 1.0 for a, b in zip(ks, ks[1:])] + [0.0]), "monotone": all(b >= a for a, b in zip(ks, ks[1:]))}
    summ["kin_growth_with_dips"] = {k: dips(v) for k, v in full.items()}
    summ["kin_monotone_all_runs"] = bool(all(v["monotone"] for v in summ["kin_growth_with_dips"].values()))
    summ["kin_net_growth_all_runs"] = bool(all(v["net_growth"] > 0 for v in summ["kin_growth_with_dips"].values()))
    summ["inflation_factor_end_over_seed"] = {k: v["kin_refreshed_a0"] / v["seed_kin_two_region"] for k, v in xchk.items()}
    for k, v in summ.items():
        if k.startswith("Rstar_"):
            i = v["Rs"].index(v["R_star"])
            v["argmin_position"] = "lower grid edge" if i == 0 else ("upper grid edge" if i == len(v["Rs"]) - 1 else "interior")
            v["kin_decreasing_with_Rs"] = bool(all(b < a for a, b in zip(v["kin"], v["kin"][1:])))
    summ["kin_growth_last_quarter"] = {f"n{r['n']}_L{r['L']:g}_R{r['Rs']:g}_J{r['J']:g}_it{r['maxit']}": r["kin_last_quarter_growth"] for r in recs}
    # g = 32 static control on the shell numbers
    seed, cfg8, _ = C.seed_hedgehog(32, 48.0)
    cfg32 = C.cfg_of(32, 48.0, g=32.0)
    sh = shell_mask(cfg8, 9.0)
    M8 = C.degenerate_project(B8_dressed(cfg8), sh); M32 = C.degenerate_project(B8_dressed(cfg32), sh)
    v8 = INS4.e_parts(M8, cfg8)[1] - INS4.e_parts(B8_dressed(cfg8), cfg8)[1]
    v32 = INS4.e_parts(M32, cfg32)[1] - INS4.e_parts(B8_dressed(cfg32), cfg32)[1]
    k8 = INS4.kin_of(M8, a0_two_region(M8, cfg8, 9.0), cfg8); k32 = INS4.kin_of(M32, a0_two_region(M32, cfg32, 9.0), cfg32)
    summ["g32_control_on_the_ansatz"] = {"V4_cost_of_shell_g8": float(v8), "V4_cost_of_shell_g32": float(v32),
                                          "kin_two_region_g8": float(k8), "kin_two_region_g32": float(k32),
                                          "note": "an IDENTITY on the ansatz (the spatial blocks are bit-identical and M00 = g exactly, so both numbers cannot depend on g; audit I2); a g control with content needs the relaxed g = 32 field, which is owed (PAUSE RECORD 3)"}
    # verdict built from the computed fields (audit H2 to H4): the frozen packet's pass needs a relaxed state
    # (a stop on f_tol or plateau with kin no longer growing); its fail modes are "shell melts" and "R* tracks L";
    # a third mode observed here is added explicitly: the descent reaches no stationary point
    stationary = [r for r in recs if r["stop"] in ("f_tol", "plateau")]
    growing = summ["kin_net_growth_all_runs"]
    ladders = {k: v for k, v in summ.items() if k.startswith("Rstar_")}
    edge_pos = {k: v["argmin_position"] for k, v in ladders.items()}
    interior_R = [k for k, v in ladders.items() if v["argmin_position"] == "interior"]
    closure = summ.get("dEdJ_closure_n32_R9", {}).get("rel_dev")
    summ["stationary_runs"] = len(stationary)
    summ["min_fmax_at_stop"] = min(r["trace_tail"][-1]["fmax"] for r in res["runs"] if r["trace_tail"]) if res["runs"] else None
    facts = {"stationary_runs": len(stationary), "kin_net_growth_all_runs": growing, "shell_melting_all_runs": summ["shell_melting_all_runs"],
             "argmin_positions": edge_pos, "E_J_monotone_increasing_all_ladders": summ["E_J_increasing_with_Rs_all_ladders"] if "E_J_increasing_with_Rs_all_ladders" in summ else None,
             "dEdJ_rel_dev": closure}
    summ["E_J_increasing_with_Rs_all_ladders"] = bool(all(all(b > a for a, b in zip(v["E_J"], v["E_J"][1:])) for v in ladders.values()))
    facts["E_J_monotone_increasing_all_ladders"] = summ["E_J_increasing_with_Rs_all_ladders"]
    summ["verdict_facts"] = facts
    if len(stationary) == 0:
        summ["verdict"] = ("W3 FAIL: PERIODIC_ORBIT_EXISTS not licensed. Stationary runs: 0 of %d (every run stopped at max_iter; min fmax at stop %.3g against the packet's 1e-6); "
                           "kin net growth in every run: %s (with dips in %d runs); shell melting in every run: %s (mean gap at the end %.3f to %.3f); "
                           "argmin of E_J(R_s): %s; dE/dJ relative deviation from omega: %s. The wall convention on L_cert is ESTABLISHED_KINEMATIC at best (W1, W2); "
                           "the fail mode observed is a third one beside the two pre-registered: the fixed-J descent reaches no minimizer, the flank inertia inflates without bound."
                           % (len(recs), summ["min_fmax_at_stop"] or float("nan"), growing, sum(1 for v in summ["kin_growth_with_dips"].values() if not v["monotone"]),
                              summ["shell_melting_all_runs"], summ["shell_gap_mean_end_range"][0], summ["shell_gap_mean_end_range"][1],
                              ", ".join(f"{k[6:]}: {v}" for k, v in edge_pos.items()), "n/a" if closure is None else f"{closure:+.2f}"))
    elif not summ["shell_melting_all_runs"] and interior_R and closure is not None and abs(closure) < 0.05 and not growing:
        summ["verdict"] = "W3 PASS: PERIODIC_ORBIT_EXISTS at the relaxed-field level (stationary runs, interior R*, closure within 5 percent, shell not melting)"
    else:
        summ["verdict"] = ("W3 FAIL (pre-registered modes): stationary runs %d, interior argmin in %s, shell melting %s, closure %s; ESTABLISHED_KINEMATIC at best"
                           % (len(stationary), interior_R or "no ladder", summ["shell_melting_all_runs"], "n/a" if closure is None else f"{closure:+.2f}"))
    res["summary"] = summ
    json.dump(res, open(OUT, "w"), indent=1)
    C.log("SUMMARY " + json.dumps(summ, indent=None)[:3000])
    # plots
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for k, v in summ.items():
        if k.startswith("Rstar_"):
            ax[0].plot(v["Rs"], v["E_J"], "o-", label=k[6:])
    ax[0].set_xlabel("shell radius R_s"); ax[0].set_ylabel("relaxed E_J"); ax[0].legend(fontsize=7); ax[0].set_title("W3: E_J(R_s) at fixed J")
    for r in recs:
        if r["maxit"] == 3000 and r["J"] == 200.0 and r["n"] == 32:
            ax[1].plot([row["it"] for row in r["trace"]], [row["kin"] for row in r["trace"]], label=f"R_s {r['Rs']:g}")
    if lng:
        ax[1].plot([row["it"] for row in lng[0]["trace"]], [row["kin"] for row in lng[0]["trace"]], "k--", label="R_s 9, 12000 it")
    ax[1].set_xlabel("iteration"); ax[1].set_ylabel("kin (two-region)"); ax[1].legend(fontsize=7); ax[1].set_title("inertia under fixed-J descent, J = 200")
    r9 = [r for r in recs if r["n"] == 32 and r["Rs"] == 9 and r["J"] == 200.0 and r["maxit"] == 3000]
    if r9:
        p0, p1 = r9[0]["start"]["profiles"], r9[0]["end"]["profiles"]
        ax[2].plot([0.5 * (p["r_lo"] + p["r_hi"]) for p in p0], [p["gap_min"] for p in p0], "o-", label="gap min, start")
        ax[2].plot([0.5 * (p["r_lo"] + p["r_hi"]) for p in p1], [p["gap_min"] for p in p1], "s-", label="gap min, end")
        ax[2].plot([0.5 * (p["r_lo"] + p["r_hi"]) for p in p1], [p["kin_bin"] / max(1e-12, max(q["kin_bin"] for q in p1)) * 0.3 for p in p1], "k:", label="kin per bin (scaled)")
        ax[2].set_xlabel("r"); ax[2].set_ylabel("d2 - d3"); ax[2].legend(fontsize=7); ax[2].set_title("shell survival, n32 R_s 9 J 200")
    fig.tight_layout(); fig.savefig(PNG, dpi=110)
    C.log(f"written {OUT} and {PNG}")


def B8_dressed(cfg):
    return C.B8.dressed(cfg, 0.0)


def _group(items, keyf):
    d = {}
    for it in items:
        d.setdefault(keyf(it), []).append(it)
    return d.items()


if __name__ == "__main__":
    if sys.argv[1] == "run":
        n, L, Rs, J, it = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
        run(n, L, Rs, J, it, frozen=(len(sys.argv) > 7 and sys.argv[7] == "frozen"))
    elif sys.argv[1] == "collect":
        collect()
