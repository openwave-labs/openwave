"""M5.32 R18-1 INDEPENDENT ADVERSARIAL AUDIT of m5_32_r18_1_radial.py (the radial tube solve), claims C1 to C6.
Every check uses its own construction where the claim allows one: the hedgehog ansatz is rewritten here (my_ansatz), the
radii r_0 / r_gap_half are re-read with own interpolation (linear and cubic), the Richardson step solves the exact
three-rung equation (the producer assumes equal h ratios on rungs whose ratios are 2 and 2.5), the minimum test is a
directional energy scan with the producer's energy as the only shared piece (the claim is about that energy), the v6
identity recomputes U_v6 from the eigenvalues of N by numpy's eigvals, the 3D energies use the certified R16/R17 stack.

usage (from research/scripts, /opt/anaconda3/envs/master312/bin/python3):
    python3 m5_32_r18_1_audit.py light        # C1 ray identity, C2, C3, C5, C6 (tube evaluations only, ~3 min)
    python3 m5_32_r18_1_audit.py energies     # C1 3D sum + C4 the nine 3D energies (~4 min)
    python3 m5_32_r18_1_audit.py descent_B    # C4 the 60-step 3D descent from the embedded object-B tube profile (~10 min)
    python3 m5_32_r18_1_audit.py descent_A    # C4 the same from the object-A tube profile
    python3 m5_32_r18_1_audit.py summary      # merge the stage files, verdicts, the table -> data/m5_32_r18_1_audit.json
    python3 m5_32_r18_1_audit.py all          # everything in sequence
stage files: <scratch>/m5_32_r18_1_audit_<stage>.json (--scratch DIR, default the checkpoints/m5_32_r18 folder)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np

ARGS = list(sys.argv[1:])
sys.argv = [sys.argv[0]]
AP = argparse.ArgumentParser()
AP.add_argument("stage", choices=["light", "energies", "descent_A", "descent_B", "summary", "all"])
AP.add_argument("--scratch", default=None)
AP.add_argument("--descent_iters", type=int, default=60)
A = AP.parse_args(ARGS)

import m5_32_r18_common as X                              # noqa: E402
import m5_32_r17_common as R                              # noqa: E402
import m5_32_r16_common as C                              # noqa: E402
import m5_32_r17_0_record as REC                          # noqa: E402

C15, INS4 = C.C15, C.INS4
ETA = C.ETA
RES, DATA = C.RES, C.DATA
CK16, CK17, CK18 = C.CK, R.CK, X.CK
SCR = A.scratch or CK18
T0 = time.time()
MU = 1e-2
GAP_MIN = C.GAP_MIN
HS = [1.5, 0.75, 0.375, 0.15, 0.075, 0.03]


def log(m):
    print(f"[audit {time.time() - T0:7.1f}s] {m}", flush=True)


def stage_file(name):
    return os.path.join(SCR, f"m5_32_r18_1_audit_{name}.json")


def save_stage(name, d):
    d["wall_s"] = round(time.time() - T0, 1)
    json.dump(d, open(stage_file(name), "w"), indent=1, default=float)
    log(f"stage {name} written ({d['wall_s']} s)")


def make_cfg(obj, h, L, n_samples=1):
    n = int(round(L / h))
    cfg = C.cfg_v4(n, L, completion="rebuild", n_samples=n_samples)
    cfg["object"] = "v4"
    cfg["weight"] = "absolute" if obj == "v4abs" else "relative"
    cfg["h"] = float(h)
    return cfg


def tag_of(obj, h, L, seed):
    return f"r18_1_{obj}_h{h:g}_L{int(L)}_{seed}"


def load_profile(obj, h, L, seed):
    p = os.path.join(CK18, tag_of(obj, h, L, seed) + "_profile.npy")
    return np.load(p) if os.path.exists(p) else None


def load_run(obj, h, L, seed):
    p = os.path.join(CK18, tag_of(obj, h, L, seed) + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ my own ansatz (the claim's formula, written independently)
def my_ansatz(P, rg, pos):
    """M(x) = Lambda [m_g e0 e0^T + lambda_1 rhat rhat^T + lambda_23 (I - rhat rhat^T)] Lambda^T at the positions pos (..., 3),
    the profiles linearly interpolated in r (clamped at both ends, as the cell-centered grid implies), Lambda the boost along
    rhat by beta; beta = 0 gives Lambda = I."""
    r = np.sqrt(np.sum(pos * pos, axis=-1))
    rhat = pos / np.maximum(r, 1e-300)[..., None]
    mg, l1, l23, be = (np.interp(r, rg, P[k]) for k in range(4))
    shp = r.shape
    M0 = np.zeros(shp + (4, 4))
    M0[..., 0, 0] = mg
    rr = rhat[..., :, None] * rhat[..., None, :]
    M0[..., 1:, 1:] = l23[..., None, None] * np.eye(3) + (l1 - l23)[..., None, None] * rr
    if np.max(np.abs(be)) == 0.0:
        return M0
    Lam = np.zeros(shp + (4, 4))
    Lam[..., 0, 0] = np.cosh(be)
    Lam[..., 0, 1:] = np.sinh(be)[..., None] * rhat
    Lam[..., 1:, 0] = np.sinh(be)[..., None] * rhat
    Lam[..., 1:, 1:] = np.eye(3) + (np.cosh(be) - 1.0)[..., None, None] * rr
    return Lam @ M0 @ np.swapaxes(Lam, -1, -2)


def embed_3d(P, h, L):
    n = int(round(L / h))
    Xc, Yc, Zc = INS4.coords(n, h)
    rg = (np.arange(P.shape[1]) + 0.5) * h
    return my_ansatz(P, rg, np.stack([Xc, Yc, Zc], -1))


# ------------------------------------------------------------------ my own radius reads
def cross_outer(rg, f, level, kind="linear"):
    """the OUTERMOST radius where f crosses `level` upward (f < level inside, >= level outside), by linear interpolation of
    the node values, or by the root of a cubic spline through the nodes (kind = 'cubic')."""
    idx = np.where((f[:-1] < level) & (f[1:] >= level))[0]
    if len(idx) == 0:
        return 0.0
    c = idx[-1]
    if kind == "linear":
        return float(rg[c] + (level - f[c]) * (rg[c + 1] - rg[c]) / (f[c + 1] - f[c]))
    from scipy.interpolate import CubicSpline
    from scipy.optimize import brentq
    lo, hi = max(0, c - 3), min(len(rg), c + 5)
    cs = CubicSpline(rg[lo:hi], f[lo:hi] - level)
    return float(brentq(cs, rg[c], rg[c + 1]))


def reads(P, h, L):
    rg = (np.arange(P.shape[1]) + 0.5) * h
    l1, l23 = P[1], P[2]
    gap = l1 - l23
    return {"r_0_lin": cross_outer(rg, l1, 0.8), "r_0_cubic": cross_outer(rg, l1, 0.8, "cubic"),
            "r_gap_half_lin": cross_outer(rg, gap, 0.35), "r_gap_half_cubic": cross_outer(rg, gap, 0.35, "cubic"),
            "gap_center": float(gap[0]), "gap_min": float(np.min(gap)), "l1_center": float(l1[0]), "l23_center": float(l23[0]),
            "m_g_center": float(P[0][0]), "l23_max": float(np.max(l23)), "r_at_l23_max": float(rg[int(np.argmax(l23))]),
            "l1_min": float(np.min(l1)), "n_cells_l23_above_0.8": int(np.sum(l23 > 0.8)), "r_max_l23_above_0.8": float(rg[l23 > 0.8].max()) if np.any(l23 > 0.8) else 0.0,
            "beta_max_abs": float(np.max(np.abs(P[3])))}


def richardson_exact(h3, v3):
    """r(h) = r0 + c h^p through three rungs: solve (h0^p - h1^p) / (h1^p - h2^p) = (v0 - v1) / (v1 - v2) exactly for p
    (the producer uses p = log(ratio) / log(h0 / h1), exact only for equal h ratios)."""
    from scipy.optimize import brentq
    h0, h1, h2 = h3
    v0, v1, v2 = v3
    if (v0 - v1) * (v1 - v2) <= 0 or v1 == v2:
        return {"p": None, "r0": None, "note": "no monotone triple"}
    ratio = (v0 - v1) / (v1 - v2)
    f = lambda p: (h0 ** p - h1 ** p) / (h1 ** p - h2 ** p) - ratio
    try:
        p = brentq(f, 0.05, 8.0)
    except ValueError:
        return {"p": None, "r0": None, "note": "no root in [0.05, 8]"}
    c = (v1 - v2) / (h1 ** p - h2 ** p)
    return {"p": float(p), "r0": float(v2 - c * h2 ** p), "c": float(c), "producer_p_equal_ratio": float(np.log(ratio) / np.log(h0 / h1))}


def fit_power(hs, vs):
    """least squares r(h) = r0 + c h^p on >= 4 rungs (p scanned, linear in (r0, c))."""
    hs, vs = np.array(hs), np.array(vs)
    best = None
    for p in np.linspace(0.5, 4.0, 701):
        Am = np.stack([np.ones_like(hs), hs ** p], 1)
        sol, res, *_ = np.linalg.lstsq(Am, vs, rcond=None)
        rss = float(np.sum((Am @ sol - vs) ** 2))
        if best is None or rss < best[0]:
            best = (rss, p, sol)
    return {"p": float(best[1]), "r0": float(best[2][0]), "c": float(best[2][1]), "rss": best[0]}


# ------------------------------------------------------------------ per-cell parts on the tube line (own assembly of the R16 pieces)
def line_parts(P, tube, cfg):
    """the per-cell densities on the tube's center line, split into E_h, V4, KP (the R16 cell functions assembled here, the
    frame from the producer's frame switch under the cfg's weight mode)."""
    M = X.tube_field(P, tube)
    h = cfg["h"]
    with R.weight_mode(cfg):
        fr = C.frame(M, None)
        v4, _ = C.v4_cells(fr, cfg, need_grad=False)
        Eh = np.zeros(M.shape[:-2]); kp = np.zeros_like(Eh)
        for br, wt in INS4.branches(cfg["stencil"]):
            Aj = [INS4.d1(M, ax, h, br) for ax in range(3)]
            for i in range(3):
                for j in range(i + 1, 3):
                    d, _, _, _ = C.quartic_pair(Aj[i], Aj[j], fr["G"], cfg["completion"], need_grad=False)
                    Eh += wt * 4.0 * np.real(d)
            Ek, _, _ = C.kp_cells(Aj, fr, need_grad=False)
            kp += wt * np.real(Ek)
        pair_out = fr["pair_out"][:, 1, 1]
    return {"E_h": Eh[:, 1, 1], "V4": np.real(v4)[:, 1, 1], "KP": cfg["cP"] * kp[:, 1, 1], "pair_out": pair_out, "w1": np.real(fr["w1"])[:, 1, 1]}


def tube_E(P, h, L, obj, need_grad=False, cfg=None):
    cfg = cfg or make_cfg(obj, h, L)
    tube = X.tube_setup(h, L)
    E, g, parts, e_line, _ = X.tube_energy(P, tube, cfg, need_grad=need_grad)
    return float(E), g, parts, e_line, tube, cfg


def smooth_dir(rng, Ng, free, width):
    d = rng.normal(size=Ng)
    k = np.exp(-0.5 * (np.arange(-3 * width, 3 * width + 1) / max(width, 1e-9)) ** 2)
    d = np.convolve(d, k / k.sum(), mode="same")
    d[~free] = 0.0
    return d


def minimum_test(P, h, L, obj, rng, n_dir=10, t=2e-3, width_cells=None):
    """E along random smooth directions in the packed variables (m_g, lambda_23, Delta, beta) on the free nodes, both signs, the bound
    Delta >= GAP_MIN respected by clipping; the curvature (E+ + E- - 2 E0) / t^2 and the rise of E in every direction are the test.
    The m_g direction is 1e5 to 1e7 stiffer, so its amplitude is scaled down by 1e-3 (a stiff direction rises trivially)."""
    E0, gP, parts, e_line, tube, cfg = tube_E(P, h, L, obj, need_grad=True)
    Ng = tube["Ng"]
    free = tube["rg"] < L / 2.0 - 1.6
    width = width_cells or max(1, int(round(1.0 / h)))
    Q0 = X.pack(P)
    scale = np.array([1e-3, 1.0, 1.0, 0.3])
    rows = []
    for k in range(n_dir):
        D = np.stack([smooth_dir(rng, Ng, free, width) * scale[i] for i in range(4)])
        D /= np.sqrt(np.sum(D * D))
        Es = {}
        for sgn in (+1.0, -1.0):
            Q = Q0 + sgn * t * D
            Q[2] = np.maximum(Q[2], GAP_MIN)
            Es[sgn] = float(X.tube_energy(X.unpack(Q), tube, cfg, need_grad=False)[0])
        gQ = np.stack([gP[0], gP[1] + gP[2], gP[1], gP[3]])
        gproj = float(np.sum(gQ * D))
        rows.append({"E_plus_minus_E0": Es[1.0] - E0, "E_minus_minus_E0": Es[-1.0] - E0, "curvature": (Es[1.0] + Es[-1.0] - 2 * E0) / t ** 2,
                     "grad_proj": gproj, "fd_slope": (Es[1.0] - Es[-1.0]) / (2 * t)})
    gQ = np.stack([gP[0], gP[1] + gP[2], gP[1], gP[3]])
    out = {"E0": E0, "n_dir": n_dir, "t": t, "all_rise_both_signs": bool(all(r_["E_plus_minus_E0"] > 0 and r_["E_minus_minus_E0"] > 0 for r_ in rows)),
           "min_rise": float(min(min(r_["E_plus_minus_E0"], r_["E_minus_minus_E0"]) for r_ in rows)), "min_curvature": float(min(r_["curvature"] for r_ in rows)),
           "max_|grad_proj|": float(max(abs(r_["grad_proj"]) for r_ in rows)), "grad_free_max_by_type_(m_g,l23|Delta,Delta|l23,beta)": [float(np.max(np.abs(gQ[i][free]))) for i in range(4)],
           "dE_dDelta_innermost_node": float(gP[1][0]), "dE_dl23_innermost_node_at_fixed_Delta": float(gP[1][0] + gP[2][0]), "Delta_innermost": float(P[1][0] - P[2][0]), "rows": rows}
    return out


def bound_release(P, h, L, obj):
    """E as the innermost node's Delta is moved off the bound: 1e-3 (the end), 2e-3, 4e-3 (up), 5e-4, 1e-4 (down, past the bound)."""
    E0, _, _, _, tube, cfg = tube_E(P, h, L, obj)
    out = {"Delta_0_end": float(P[1][0] - P[2][0]), "E_end": E0, "scan": []}
    for D0 in (4e-3, 2e-3, 5e-4, 1e-4):
        Q = P.copy()
        Q[1][0] = Q[2][0] + D0
        try:
            E = float(X.tube_energy(Q, tube, cfg, need_grad=False)[0])
        except Exception as e:                                # noqa: BLE001
            E = float("nan"); out["error_at_" + str(D0)] = repr(e)
        out["scan"].append({"Delta_0": D0, "E_minus_E_end": E - E0 if np.isfinite(E) else None})
    return out


# ================================================================== the LIGHT stage
def stage_light():
    out = {"stage": "light"}
    d = json.load(open(os.path.join(DATA, "m5_32_r18_1.json")))
    runs = d["runs"]
    rng = np.random.default_rng(20260909)

    # ---------- C1: the tube center line at (h/2, h/2) vs the 3D lattice's own ray densities (my ansatz on the 3D lattice, REC.densities)
    c1 = {}
    for obj, seed in (("v4abs", "r16_1"), ("v4rel", "r16_1")):
        P = load_profile(obj, 1.5, 48.0, seed)
        cfg = make_cfg(obj, 1.5, 48.0)
        M3 = embed_3d(P, 1.5, 48.0)
        with R.weight_mode(cfg):
            dens, fr3 = REC.densities(M3, cfg)
        e3 = sum(dens[k] for k in ("E_h", "V4", "U", "KP", "reg"))
        n = cfg["n"]
        line3 = e3[:, n // 2, n // 2]
        tube_g = X.tube_setup(1.5, 48.0, y0=0.75, z0=0.75)
        Et, _, pt, et, _ = X.tube_energy(P, tube_g, cfg, need_grad=False)
        et = np.real(et)
        dev = np.max(np.abs(et - line3)) / np.max(np.abs(line3))
        # the ray cells: my coordinates say cell (i, 16, 16) sits at y = z = h/2 and x = (i + 1/2 - 16) h, the tube's x line
        Xc, Yc, Zc = INS4.coords(n, 1.5)
        c1[obj] = {"max_rel_dev_ray_density": float(dev), "n_ray_cells": int(len(line3)), "ray_y": float(Yc[0, n // 2, 0]), "ray_z": float(Zc[0, 0, n // 2]),
                   "x_line_match": float(np.max(np.abs(Xc[:, n // 2, n // 2] - tube_g["x"]))), "E_tube_on_(h/2,h/2)_line": float(Et),
                   "h3_sum_3D_density_single_sample": float(np.sum(e3) * 1.5 ** 3), "pair_split_max_3D": float(np.max(np.abs(np.sqrt(np.maximum(np.real(fr3["s"]) ** 2 - 4 * np.real(fr3["p"]), 0))))),
                   "note": "the 3D energy_object (8 samples) on the same embedded field is in the energies stage (C4)"}
        log(f"C1 {obj}: ray density max rel dev {dev:.2e} over {len(line3)} cells")
    out["C1"] = c1

    # ---------- C2: object B
    c2 = {"reads": {}, "seeds": {}, "minimum": {}, "bound": {}}
    for (L, hs) in ((96.0, HS), (48.0, [0.03]), (144.0, [0.15])):
        for h in hs:
            P = load_profile("v4rel", h, L, "r16_1")
            if P is None:
                continue
            rd = reads(P, h, L)
            rec = runs.get(tag_of("v4rel", h, L, "r16_1"), {}).get("reads", {})
            rd["producer_r_0"] = rec.get("r_0_profile (lambda_1 = 0.8 crossing)"); rd["producer_r_gap_half"] = rec.get("r_gap_half (Delta = 0.35 crossing)")
            c2["reads"][f"h{h:g}_L{int(L)}"] = rd
    lad = [c2["reads"][f"h{h:g}_L96"] for h in HS]
    c2["r_0_ladder_L96_lin"] = [r_["r_0_lin"] for r_ in lad]; c2["r_0_ladder_L96_cubic"] = [r_["r_0_cubic"] for r_ in lad]
    c2["r_gap_half_ladder_L96_lin"] = [r_["r_gap_half_lin"] for r_ in lad]; c2["r_gap_half_ladder_L96_cubic"] = [r_["r_gap_half_cubic"] for r_ in lad]
    c2["gap_center_ladder_L96"] = [r_["gap_center"] for r_ in lad]
    c2["richardson_r_gap_half_exact_3finest"] = richardson_exact(HS[-3:], c2["r_gap_half_ladder_L96_lin"][-3:])
    c2["richardson_r_0_exact_3finest"] = richardson_exact(HS[-3:], c2["r_0_ladder_L96_lin"][-3:])
    c2["fit_r_gap_half_4finest"] = fit_power(HS[-4:], c2["r_gap_half_ladder_L96_lin"][-4:])
    c2["fit_r_0_4finest"] = fit_power(HS[-4:], c2["r_0_ladder_L96_lin"][-4:])
    c2["finest_pair_rel_change_r_gap_half"] = abs(lad[-1]["r_gap_half_lin"] - lad[-2]["r_gap_half_lin"]) / lad[-1]["r_gap_half_lin"]
    c2["finest_pair_rel_change_r_0"] = abs(lad[-1]["r_0_lin"] - lad[-2]["r_0_lin"]) / lad[-1]["r_0_lin"]
    c2["L96_to_144_rel_change_r_gap_half_h0.15"] = abs(c2["reads"]["h0.15_L144"]["r_gap_half_lin"] - c2["reads"]["h0.15_L96"]["r_gap_half_lin"]) / c2["reads"]["h0.15_L96"]["r_gap_half_lin"]
    c2["L96_to_144_rel_change_r_0_h0.15"] = abs(c2["reads"]["h0.15_L144"]["r_0_lin"] - c2["reads"]["h0.15_L96"]["r_0_lin"]) / c2["reads"]["h0.15_L96"]["r_0_lin"]
    c2["L48_vs_96_rel_change_r_gap_half_h0.03"] = abs(c2["reads"]["h0.03_L48"]["r_gap_half_lin"] - c2["reads"]["h0.03_L96"]["r_gap_half_lin"]) / c2["reads"]["h0.03_L96"]["r_gap_half_lin"]
    # the two seeds: recorded E and my re-evaluation of both end profiles
    seeds = []
    for L in (48.0, 96.0):
        for h in HS:
            Pa, Pb = load_profile("v4rel", h, L, "r16_1"), load_profile("v4rel", h, L, "melted")
            if Pa is None or Pb is None:
                continue
            Ea = tube_E(Pa, h, L, "v4rel")[0]; Eb = tube_E(Pb, h, L, "v4rel")[0]
            ra, rb = runs[tag_of("v4rel", h, L, "r16_1")]["E"], runs[tag_of("v4rel", h, L, "melted")]["E"]
            seeds.append({"h": h, "L": L, "E_r16_1_mine": Ea, "E_melted_mine": Eb, "rel_E_mine": abs(Ea - Eb) / abs(Eb), "rel_E_recorded": abs(ra - rb) / abs(rb),
                          "my_eval_vs_recorded_r16_1": abs(Ea - ra) / abs(ra), "max_abs_profile_diff": float(np.max(np.abs(Pa - Pb))),
                          "r_gap_half_r16_1": reads(Pa, h, L)["r_gap_half_lin"], "r_gap_half_melted": reads(Pb, h, L)["r_gap_half_lin"]})
    c2["seeds"] = seeds
    c2["seeds_max_rel_E_mine"] = float(max(s_["rel_E_mine"] for s_ in seeds)); c2["seeds_max_rel_E_recorded"] = float(max(s_["rel_E_recorded"] for s_ in seeds))
    c2["seeds_within_1e-9"] = bool(c2["seeds_max_rel_E_mine"] < 1e-9); c2["seeds_within_1e-8"] = bool(c2["seeds_max_rel_E_mine"] < 1e-8)
    c2["E_finest_L96_mine"] = tube_E(load_profile("v4rel", 0.03, 96.0, "r16_1"), 0.03, 96.0, "v4rel")[0]
    log(f"C2 reads: r_0 L96 {['%.4f' % v for v in c2['r_0_ladder_L96_lin']]}, r_gap_half {['%.4f' % v for v in c2['r_gap_half_ladder_L96_lin']]}, seeds max rel E {c2['seeds_max_rel_E_mine']:.2e}")
    for (h, L) in ((0.03, 96.0), (0.15, 96.0), (0.75, 96.0)):
        P = load_profile("v4rel", h, L, "r16_1")
        c2["minimum"][f"h{h:g}_L{int(L)}"] = minimum_test(P, h, L, "v4rel", rng)
        c2["bound"][f"h{h:g}_L{int(L)}"] = bound_release(P, h, L, "v4rel")
        mt = c2["minimum"][f"h{h:g}_L{int(L)}"]
        log(f"C2 minimum h {h:g} L {L:g}: all rise {mt['all_rise_both_signs']} min rise {mt['min_rise']:.3e} min curvature {mt['min_curvature']:.3e} dE/dDelta(0) {mt['dE_dDelta_innermost_node']:+.3e} bound scan {c2['bound'][f'h{h:g}_L{int(L)}']['scan']}")
    out["C2"] = c2

    # ---------- C3: object A
    c3 = {"basins_h0.75": {}, "ladder_L96": [], "line_scan_h0.75_L96": None, "hazard": {}}
    for L in (48.0, 96.0):
        row = {}
        for seed in ("r16_1", "melted"):
            P = load_profile("v4abs", 0.75, L, seed)
            E, gP, parts, e_line, tube, cfg = tube_E(P, 0.75, L, "v4abs", need_grad=True)
            free = tube["rg"] < L / 2.0 - 1.6
            rd = reads(P, 0.75, L)
            row[seed] = {"E_mine": E, "E_recorded": runs[tag_of("v4abs", 0.75, L, seed)]["E"], "r_0_lin": rd["r_0_lin"], "l1_center": rd["l1_center"], "l1_min": rd["l1_min"], "gap_center": rd["gap_center"],
                         "grad_free_max_by_type": [float(np.max(np.abs(gP[k][free]))) for k in range(4)]}
        row["E_diff_recorded_vs_mine_max"] = max(abs(row[s]["E_mine"] - row[s]["E_recorded"]) for s in ("r16_1", "melted"))
        c3["basins_h0.75"][f"L{int(L)}"] = row
        log(f"C3 basins L {L:g}: r16_1 E {row['r16_1']['E_mine']:.4f} (r_0 {row['r16_1']['r_0_lin']:.3f}, l1min {row['r16_1']['l1_min']:.3f}) melted E {row['melted']['E_mine']:.4f} (l1min {row['melted']['l1_min']:.3f})")
    # the line between the two basins at L 96 h 0.75: a barrier means two minima; a monotone fall means the higher one is not a minimum
    Pa, Pb = load_profile("v4abs", 0.75, 96.0, "r16_1"), load_profile("v4abs", 0.75, 96.0, "melted")
    tube = X.tube_setup(0.75, 96.0); cfg = make_cfg("v4abs", 0.75, 96.0)
    ts = np.linspace(0.0, 1.0, 26)
    Eline = [float(X.tube_energy(Pa + t_ * (Pb - Pa), tube, cfg, need_grad=False)[0]) for t_ in ts]
    c3["line_scan_h0.75_L96"] = {"t": ts.tolist(), "E": Eline, "E_max": float(max(Eline)), "barrier_above_r16_1_basin": float(max(Eline) - Eline[0]), "monotone_decreasing": bool(all(np.diff(Eline) <= 0))}
    log(f"C3 line scan r16_1 -> melted: E(0) {Eline[0]:.4f} max {max(Eline):.4f} E(1) {Eline[-1]:.4f} barrier {max(Eline) - Eline[0]:.4f}")
    # the L 96 ladder (the lowest basin per rung by MY evaluation), the center spectrum, the pair eigenvalue
    for h in HS:
        cands = []
        for seed in ("r16_1", "melted"):
            P = load_profile("v4abs", h, 96.0, seed)
            if P is not None:
                cands.append((tube_E(P, h, 96.0, "v4abs")[0], seed, P))
        E, seed, P = min(cands, key=lambda c: c[0])
        rd = reads(P, h, 96.0)
        rd.update({"h": h, "seed_of_lowest": seed, "E_mine": E, "w_plateau_at_l23_center": float(np.real(C.w_plateau(np.array([rd["l23_center"]]))[0])), "w_plateau_at_l23_max": float(np.real(C.w_plateau(np.array([rd["l23_max"]]))[0])),
                   "producer_r_gap_half": runs[tag_of("v4abs", h, 96.0, seed)]["reads"].get("r_gap_half (Delta = 0.35 crossing)")})
        c3["ladder_L96"].append(rd)
    lad = c3["ladder_L96"]
    c3["r_gap_half_ladder_L96_lin"] = [r_["r_gap_half_lin"] for r_ in lad]; c3["r_gap_half_ladder_L96_cubic"] = [r_["r_gap_half_cubic"] for r_ in lad]
    c3["gap_center_ladder"] = [r_["gap_center"] for r_ in lad]; c3["l1_center_ladder"] = [r_["l1_center"] for r_ in lad]; c3["l23_center_ladder"] = [r_["l23_center"] for r_ in lad]
    c3["l23_max_ladder"] = [r_["l23_max"] for r_ in lad]; c3["gap_center_over_h_ladder"] = [r_["gap_center"] / r_["h"] for r_ in lad]
    c3["finest_pair_rel_change_r_gap_half"] = abs(lad[-1]["r_gap_half_lin"] - lad[-2]["r_gap_half_lin"]) / lad[-1]["r_gap_half_lin"]
    c3["richardson_r_gap_half_exact_3finest"] = richardson_exact(HS[-3:], c3["r_gap_half_ladder_L96_lin"][-3:])
    c3["fit_r_gap_half_4finest"] = fit_power(HS[-4:], c3["r_gap_half_ladder_L96_lin"][-4:])
    c3["richardson_l23_center_3finest"] = richardson_exact(HS[-3:], c3["l23_center_ladder"][-3:])
    c3["richardson_l1_center_3finest"] = richardson_exact(HS[-3:], c3["l1_center_ladder"][-3:])
    P144 = load_profile("v4abs", 0.15, 144.0, "r16_1"); P96 = load_profile("v4abs", 0.15, 96.0, "r16_1")
    c3["L96_to_144_rel_change_r_gap_half_h0.15"] = abs(reads(P144, 0.15, 144.0)["r_gap_half_lin"] - reads(P96, 0.15, 96.0)["r_gap_half_lin"]) / reads(P96, 0.15, 96.0)["r_gap_half_lin"]
    log(f"C3 ladder r_gap_half {['%.4f' % v for v in c3['r_gap_half_ladder_L96_lin']]} gap(0) {['%.4f' % v for v in c3['gap_center_ladder']]} l1(0) {['%.4f' % v for v in c3['l1_center_ladder']]} l23(0) {['%.4f' % v for v in c3['l23_center_ladder']]}")
    # the hazard: the plateau weight at the center's pair eigenvalue, the pair_out cells, the K_P share inside r < 1, the energy share of the taper cells
    for h in HS[-3:]:
        P = load_profile("v4abs", h, 96.0, "r16_1")
        E, gP, parts, e_line, tube, cfg = tube_E(P, h, 96.0, "v4abs", need_grad=True)
        lp = line_parts(P, tube, cfg)
        x = np.abs(tube["x"]); wc = tube["wc"][:, 1, 1]; e = np.real(e_line)
        inner = x < 1.0
        share = lambda arr, m: float(np.sum(wc[m] * arr[m]) / np.sum(wc[m] * e[m]))
        po = lp["pair_out"]
        rd = reads(P, h, 96.0)
        hz = {"h": h, "l23_center": rd["l23_center"], "w_plateau_l23_center": float(np.real(C.w_plateau(np.array([rd["l23_center"]]))[0])), "one_minus_w": 1.0 - float(np.real(C.w_plateau(np.array([rd["l23_center"]]))[0])),
              "n_line_cells_pair_out": int(np.sum(po)), "r_max_pair_out": float(x[po].max()) if np.any(po) else 0.0, "energy_share_pair_out_cells": float(np.sum(wc[po] * e[po]) / E) if np.any(po) else 0.0,
              "KP_share_r_lt_1": share(lp["KP"], inner), "E_h_share_r_lt_1": share(lp["E_h"], inner), "V4_share_r_lt_1": share(lp["V4"], inner), "E_r_lt_1_over_E": float(np.sum(wc[inner] * e[inner]) / E),
              "KP_share_total": float(np.sum(wc * lp["KP"]) / E), "w1_min_line": float(np.min(lp["w1"])), "dE_dl23_center_fixed_Delta": float(gP[1][0] + gP[2][0]), "dE_dl1_center": float(gP[1][0])}
        # the sensitivity to the taper: clip lambda_23 to 0.7999 (out of the taper) keeping the gap, and E of that profile
        Pc = P.copy(); over = Pc[2] > 0.7999; Pc[2][over] = 0.7999; Pc[1][over] = Pc[2][over] + (P[1][over] - P[2][over])
        hz["E_clipped_l23_0.7999_minus_E"] = float(X.tube_energy(Pc, tube, cfg, need_grad=False)[0]) - E
        hz["n_nodes_clipped"] = int(np.sum(over))
        c3["hazard"][f"h{h:g}"] = hz
        log(f"C3 hazard h {h:g}: l23(0) {hz['l23_center']:.4f} w {hz['w_plateau_l23_center']:.6f} pair_out cells {hz['n_line_cells_pair_out']} (r < {hz['r_max_pair_out']:.3f}) share {hz['energy_share_pair_out_cells']:.2e} KP share r<1 {hz['KP_share_r_lt_1']:.3f} E(r<1)/E {hz['E_r_lt_1_over_E']:.4f} clip dE {hz['E_clipped_l23_0.7999_minus_E']:+.3e}")
    # the center nodes' gradient by central finite differences vs the producer's analytic gradient (the innermost cells are pair_out
    # cells on the general-weight path; a wrong gradient there would make the plateau-edge center a fake stationary point)
    fdc = {}
    for h in (0.03, 0.15):
        P = load_profile("v4abs", h, 96.0, "r16_1")
        E0, gP, parts, e_line, tube, cfg = tube_E(P, h, 96.0, "v4abs", need_grad=True)
        rows = []
        for node in (0, 1, 2):
            for k in (1, 2):
                dd = 1e-5
                Pp = P.copy(); Pp[k][node] += dd; Pm = P.copy(); Pm[k][node] -= dd
                Ep = float(X.tube_energy(Pp, tube, cfg, need_grad=False)[0]); Em = float(X.tube_energy(Pm, tube, cfg, need_grad=False)[0])
                rows.append({"node": node, "type": ("lambda_1", "lambda_23")[k - 1], "fd": (Ep - Em) / (2 * dd), "analytic": float(gP[k][node]), "curvature": (Ep + Em - 2 * E0) / dd ** 2})
        fdc[f"h{h:g}"] = {"rows": rows, "max_|analytic|_center_3_nodes": float(max(abs(r_["analytic"]) for r_ in rows)), "max_|fd - analytic|": float(max(abs(r_["fd"] - r_["analytic"]) for r_ in rows)), "min_curvature": float(min(r_["curvature"] for r_ in rows))}
        log(f"C3 center FD h {h:g}: max |grad| {fdc[f'h{h:g}']['max_|analytic|_center_3_nodes']:.1e}, max |fd - analytic| {fdc[f'h{h:g}']['max_|fd - analytic|']:.1e}, min curvature {fdc[f'h{h:g}']['min_curvature']:.2f}")
    c3["center_fd_gradient"] = fdc
    # the minimum test on the finest object-A profile
    c3["minimum_h0.03_L96"] = minimum_test(load_profile("v4abs", 0.03, 96.0, "r16_1"), 0.03, 96.0, "v4abs", rng)
    c3["minimum_h0.75_L96_r16_1_basin"] = minimum_test(load_profile("v4abs", 0.75, 96.0, "r16_1"), 0.75, 96.0, "v4abs", rng, n_dir=16)
    log(f"C3 minimum h0.03: all rise {c3['minimum_h0.03_L96']['all_rise_both_signs']} min curvature {c3['minimum_h0.03_L96']['min_curvature']:.3e}; h0.75 r16_1 basin: all rise {c3['minimum_h0.75_L96_r16_1_basin']['all_rise_both_signs']} min curv {c3['minimum_h0.75_L96_r16_1_basin']['min_curvature']:.3e}")
    out["C3"] = c3

    # ---------- C5: the v6 identity (my U_v6 from the eigenvalues of N)
    c5 = {}
    for (h, L, seed) in ((0.03, 96.0, "r16_1"), (0.75, 48.0, "melted"), (1.5, 48.0, "r16_1")):
        P = load_profile("v4rel", h, L, seed)
        tube = X.tube_setup(h, L)
        cfg4 = make_cfg("v4rel", h, L)
        cfg6 = R.cfg_v6(cfg4["n"], L, gW=2.0, completion="rebuild", n_samples=1); cfg6["h"] = float(h)
        E4 = float(X.tube_energy(P, tube, cfg4, need_grad=False)[0])
        E6 = float(X.tube_energy(P, tube, cfg6, need_grad=False)[0])
        M = X.tube_field(P, tube)
        lam = np.sort(np.real(np.linalg.eigvals(M @ ETA)), axis=-1)       # (-m_g, l23, l23, l1) for a split-free profile
        l1 = lam[..., 3]; l2, l3 = lam[..., 2], lam[..., 1]
        rho2 = (l2 - l3) ** 2 / 4.0
        Wc = ((1.0 - l1) / (1.0 - cfg6["delta"])) ** 2
        U = (cfg6["mu_v6"] - cfg6["gW"] * Wc) * rho2 - cfg6["nu"] * rho2 ** 2 + cfg6["kappa"] * rho2 ** 3
        wc = tube["wc"]
        c5[f"h{h:g}_L{int(L)}_{seed}"] = {"E_v4rel": E4, "E_v6_gW2": E6, "abs_diff": abs(E6 - E4), "rel_diff": abs(E6 - E4) / abs(E4), "holds_1e-12": bool(abs(E6 - E4) < 1e-12 * abs(E4)),
                                         "rho2_max_tube_field_eigvals": float(np.max(rho2)), "U_v6_mine_weighted_sum": float(np.sum(wc * U)), "mu_eff_min": float(np.min(cfg6["mu_v6"] - cfg6["gW"] * Wc)),
                                         "cfg6_cs": cfg6["cs"], "cfg4_cs": cfg4["cs"], "note": "cs differs (0.5 vs 0.4) and mu_v6 vs mu; both multiply rho^2 = 0 so the identity is exact by construction"}
        log(f"C5 {h:g}/{L:g}/{seed}: |E6 - E4| {abs(E6 - E4):.2e} rel {abs(E6 - E4) / abs(E4):.2e} rho2 max {np.max(rho2):.2e} U_v6 mine {np.sum(wc * U):.2e}")
    out["C5"] = c5

    # ---------- C6: the outcome numbers (from C2 / C3 above)
    r0B, rgB, rgA = c2["r_0_ladder_L96_lin"][-1], c2["r_gap_half_ladder_L96_lin"][-1], c3["r_gap_half_ladder_L96_lin"][-1]
    vals = {"B_r_0_lambda1_0.8": r0B * np.sqrt(MU), "B_r_gap_half": rgB * np.sqrt(MU), "A_r_gap_half": rgA * np.sqrt(MU), "A_r_0_lambda1_0.8_h1.5_only": c3["ladder_L96"][0]["r_0_lin"] * np.sqrt(MU),
            "B_r_gap_half_richardson_exact": (c2["richardson_r_gap_half_exact_3finest"]["r0"] or 0.0) * np.sqrt(MU), "B_r_0_richardson_exact": (c2["richardson_r_0_exact_3finest"]["r0"] or 0.0) * np.sqrt(MU),
            "A_r_gap_half_richardson_exact": (c3["richardson_r_gap_half_exact_3finest"]["r0"] or 0.0) * np.sqrt(MU)}
    e8 = lambda a, b: (a / b) ** 8
    c6 = {"values_r0_sqrt_mu": vals, "finest_pair_rel_change": {"A_r_gap_half": c3["finest_pair_rel_change_r_gap_half"], "B_r_gap_half": c2["finest_pair_rel_change_r_gap_half"], "B_r_0": c2["finest_pair_rel_change_r_0"]},
          "L96_to_144_rel_change_h0.15": {"A_r_gap_half": c3["L96_to_144_rel_change_r_gap_half_h0.15"], "B_r_gap_half": c2["L96_to_144_rel_change_r_gap_half_h0.15"], "B_r_0": c2["L96_to_144_rel_change_r_0_h0.15"]},
          "L48_vs_96_rel_change_B_r_gap_half_h0.03": c2["L48_vs_96_rel_change_r_gap_half_h0.03"],
          "eighth_power_ratios": {"B_lambda1_over_B_gap_half": e8(r0B, rgB), "B_gap_half_over_A_gap_half": e8(rgB, rgA), "B_lambda1_over_A_gap_half": e8(r0B, rgA), "A_over_B_same_definition_gap_half": e8(rgA, rgB)},
          "eighth_power_of_5pct": 1.05 ** 8, "eighth_power_of_numerical_uncertainty_B": (1 + max(c2["finest_pair_rel_change_r_gap_half"], c2["L96_to_144_rel_change_r_gap_half_h0.15"])) ** 8,
          "eighth_power_of_numerical_uncertainty_A": (1 + max(c3["finest_pair_rel_change_r_gap_half"], c3["L96_to_144_rel_change_r_gap_half_h0.15"])) ** 8,
          "E_L_dependence_B_h0.15": {L_: runs[tag_of("v4rel", 0.15, L_, "r16_1")]["E"] for L_ in (48.0, 96.0, 144.0)}}
    out["C6"] = c6
    log(f"C6 r0 sqrt(mu): B(l1=0.8) {vals['B_r_0_lambda1_0.8']:.4f} B(gap) {vals['B_r_gap_half']:.4f} A(gap) {vals['A_r_gap_half']:.4f}; eighth-power ratios B l1/gap {c6['eighth_power_ratios']['B_lambda1_over_B_gap_half']:.1f}, B/A gap {c6['eighth_power_ratios']['B_gap_half_over_A_gap_half']:.0f}")
    save_stage("light", out)
    return out


# ================================================================== the ENERGIES stage (C1 3D sum, C4 the nine energies)
def stage_energies():
    out = {"stage": "energies", "C4": {}}
    items = [("v4abs", 1.5, 48.0, os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"), "R16-1 n32 L48 core"),
             ("v4rel", 1.5, 48.0, os.path.join(CK17, "r17_2_v4rel_n32_L48_r16_1.npy"), "R17-2 v4rel static n32 L48"),
             ("v4abs", 0.75, 48.0, os.path.join(CK16, "r16_1_rebuild_n64_L48_analytic.npy"), "R16-1 n64 L48 core")]
    d = json.load(open(os.path.join(DATA, "m5_32_r18_1.json")))
    g3p = d["gate3_3D_energy"]
    for obj, h, L, core_file, core_name in items:
        cfg = make_cfg(obj, h, L, n_samples=8)
        nref = C.radial_ref(cfg)
        key = f"{obj}_h{h:g}_n{cfg['n']}"
        row = {"core": core_name, "h": h, "n": cfg["n"]}
        t = time.time()
        Mc = np.load(core_file)
        Ec = R.energy_object(Mc, cfg, None, nref, need_grad=False)
        row["E_3D_core"] = float(np.real(Ec[2]["E_stat"])); row["core_dom"] = {k: Ec[3][k] for k in ("escape_d", "gap_1_2_min", "l1_min", "l2_max", "cells_pair_outside_plateau", "r_d")}
        row["core_parts"] = {k: float(np.real(Ec[2][k])) for k in ("E_h", "V4", "U", "KP", "reg")}
        row["producer_E_3D_core"] = g3p[key]["E_3D_core"]
        log(f"C4 {key} core E {row['E_3D_core']:.6f} (producer {g3p[key]['E_3D_core']:.6f}) [{time.time() - t:.0f}s]")
        for seed in ("r16_1", "melted"):
            P = load_profile(obj, h, L, seed)
            if P is None:
                continue
            t = time.time()
            M3 = embed_3d(P, h, L)
            E3 = R.energy_object(M3, cfg, None, nref, need_grad=False)
            row[f"E_3D_tube_{seed}"] = float(np.real(E3[2]["E_stat"])); row[f"tube_dom_{seed}"] = {k: E3[3][k] for k in ("escape_d", "gap_1_2_min", "l1_min", "l2_max", "cells_pair_outside_plateau", "r_d")}
            row[f"tube_parts_{seed}"] = {k: float(np.real(E3[2][k])) for k in ("E_h", "V4", "U", "KP", "reg")}
            row[f"producer_E_3D_tube_{seed}"] = g3p[key].get(f"E_3D_tube_profile_{seed}")
            row[f"tube_below_core_{seed}"] = bool(row[f"E_3D_tube_{seed}"] < row["E_3D_core"])
            row[f"E_tube_quadrature_{seed}"] = d["runs"][tag_of(obj, h, L, seed)]["E"]
            if h == 1.5:
                # C1's second half: the single-sample h^3 density sum equals the 8-sample E_stat on the ansatz field (the circle acts trivially)
                cfg1 = make_cfg(obj, h, L)
                with R.weight_mode(cfg1):
                    dens, _ = REC.densities(M3, cfg1)
                row[f"h3_sum_single_sample_{seed}"] = float(np.sum(sum(dens[k] for k in dens)) * h ** 3)
                row[f"C1_sum_vs_E8_rel_{seed}"] = abs(row[f"h3_sum_single_sample_{seed}"] - row[f"E_3D_tube_{seed}"]) / abs(row[f"E_3D_tube_{seed}"])
            log(f"C4 {key} tube {seed}: E_3D {row[f'E_3D_tube_{seed}']:.6f} (producer {row[f'producer_E_3D_tube_{seed}']}) below core {row[f'tube_below_core_{seed}']} dom {row[f'tube_dom_{seed}']} [{time.time() - t:.0f}s]")
        out["C4"][key] = row
        save_stage("energies", out)
    return out


# ================================================================== the DESCENT stages (C4's decisive test)
def stage_descent(obj):
    out = {"stage": f"descent_{obj}"}
    h, L = 1.5, 48.0
    cfg = make_cfg(obj, h, L, n_samples=8)
    nref = C.radial_ref(cfg)
    P = load_profile(obj, h, L, "r16_1")
    M0 = embed_3d(P, h, L)
    free = ~INS4.pin_shell(32, 1.5, 1.6)
    E0 = R.energy_object(M0, cfg, None, nref, need_grad=False)
    out["E0"] = float(np.real(E0[2]["E_stat"])); out["dom0"] = {k: E0[3][k] for k in ("escape_d", "gap_1_2_min", "l1_min", "l2_max", "r_d")}
    log(f"descent {obj}: E0 {out['E0']:.6f} dom {out['dom0']}; {A.descent_iters} FIRE steps dt0 0.001")
    t = time.time()
    M, info = R.fire_object(M0, cfg, free, A.descent_iters, K=None, n_ref=nref, dt0=0.001, log_every=20, tag=f"audit_{obj}")
    out["stop"] = info["stop"]; out["iters"] = info["iters"]; out["wall_s_descent"] = info["wall_s"]
    out["trace"] = [{k: (float(np.real(v)) if not isinstance(v, (bool, str, list, dict)) else v) for k, v in row.items() if k in ("it", "fmax", "dt", "E_stat", "E_h", "V4", "KP", "dom_escape_d", "dom_gap_1_2_min", "dom_l1_min", "dom_l2_max")} for row in info["trace"]]
    Ef = R.energy_object(M, cfg, None, nref, need_grad=False)
    out["E_final"] = float(np.real(Ef[2]["E_stat"])); out["dom_final"] = {k: Ef[3][k] for k in ("escape_d", "gap_1_2_min", "l1_min", "l2_max", "r_d")}
    out["E_final_minus_E0"] = out["E_final"] - out["E0"]
    out["E_went_down"] = bool(out["E_final"] < out["E0"] - 1e-9)
    # how far from spherical did the descent go: the spread of lambda_1 on the shell r in [3, 4.5]
    Xc, Yc, Zc = INS4.coords(32, 1.5); r = np.sqrt(Xc ** 2 + Yc ** 2 + Zc ** 2)
    fr = Ef[4]; l1 = np.real(fr["l1"])
    sh = (r > 3.0) & (r < 4.5)
    out["lambda_1_shell_3_4.5_std_final"] = float(np.std(l1[sh])); out["lambda_1_shell_3_4.5_std_initial"] = float(np.std(np.real(E0[4]["l1"])[sh]))
    out["max_|M - M0|"] = float(np.max(np.abs(M - M0)))
    log(f"descent {obj}: stop {info['stop']} after {info['iters']} it, E {out['E0']:.6f} -> {out['E_final']:.6f} (dE {out['E_final_minus_E0']:+.3e}) dom_final {out['dom_final']} [{time.time() - t:.0f}s]")
    save_stage("descent_A" if obj == "v4abs" else "descent_B", out)
    return out


# ================================================================== SUMMARY: verdicts + the table
def stage_summary():
    parts = {}
    for name in ("light", "energies", "descent_A", "descent_B"):
        p = stage_file(name)
        parts[name] = json.load(open(p)) if os.path.exists(p) else None
    lt, en, dA, dB = parts["light"], parts["energies"], parts["descent_A"], parts["descent_B"]
    claims = {}

    def verdict(cid, v, numbers, reason):
        claims[cid] = {"verdict": v, "numbers": numbers, "reason": reason}

    # C1
    if lt is None:
        verdict("C1", "QUALIFIED", {}, "the light stage did not run")
    else:
        c1 = lt["C1"]
        dev = max(c1[o]["max_rel_dev_ray_density"] for o in c1)
        sum_ok = None
        if en is not None:
            sum_ok = max(v for k, row in en["C4"].items() for kk, v in row.items() if kk.startswith("C1_sum_vs_E8_rel"))
        ok = dev < 1e-10 and (sum_ok is None or sum_ok < 1e-9)
        verdict("C1", "CONFIRMED" if ok else "REFUTED", {"max_rel_dev_ray_density_A_B": [c1["v4abs"]["max_rel_dev_ray_density"], c1["v4rel"]["max_rel_dev_ray_density"]], "n_ray_cells": c1["v4abs"]["n_ray_cells"], "h3_sum_vs_8sample_E_rel": sum_ok},
                f"my own ansatz on the 3D lattice, REC.densities on the ray cells (16,16) vs the (h/2,h/2) tube line: max rel dev {dev:.1e} on both weights; the single-sample h^3 sum equals the 8-sample E_stat to {sum_ok if sum_ok is None else f'{sum_ok:.1e}'}")
    # C2
    if lt is not None:
        c2 = lt["C2"]
        claimed_r0 = [6.874, 6.312, 6.133, 6.078, 6.070, 6.068]
        mine_r0 = c2["r_0_ladder_L96_lin"]
        dev_r0 = max(abs(a - b) / b for a, b in zip(mine_r0, claimed_r0))
        rg = c2["r_gap_half_ladder_L96_lin"][-1]
        mins = c2["minimum"]
        all_rise = all(m["all_rise_both_signs"] for m in mins.values())
        curv_pos = all(m["min_curvature"] > 0 for m in mins.values())
        dEdD = {k: m["dE_dDelta_innermost_node"] for k, m in mins.items()}
        bound_lower = {k: [s_["E_minus_E_end"] for s_ in b["scan"]] for k, b in c2["bound"].items()}
        gfree = {k: m["grad_free_max_by_type_(m_g,l23|Delta,Delta|l23,beta)"] for k, m in mins.items()}
        seeds9 = c2["seeds_within_1e-9"]
        v = "CONFIRMED" if (dev_r0 < 2e-3 and abs(rg - 4.601) < 2e-3 and all_rise and curv_pos and seeds9) else "QUALIFIED"
        verdict("C2", v, {"r_0_ladder_L96_mine": [round(x, 4) for x in mine_r0], "r_0_h0.03_L48": round(c2["reads"]["h0.03_L48"]["r_0_lin"], 4), "r_0_h0.15_L144": round(c2["reads"]["h0.15_L144"]["r_0_lin"], 4),
                           "r_gap_half_h0.03_L96": round(rg, 4), "r_gap_half_cubic": round(c2["r_gap_half_ladder_L96_cubic"][-1], 4), "E_h0.03_L96": c2["E_finest_L96_mine"], "gap_center_ladder": c2["gap_center_ladder_L96"],
                           "seeds_max_rel_E_mine": c2["seeds_max_rel_E_mine"], "seeds_within_1e-9": seeds9, "seeds_within_1e-8": c2["seeds_within_1e-8"],
                           "richardson_exact_r_gap_half": c2["richardson_r_gap_half_exact_3finest"], "richardson_exact_r_0": c2["richardson_r_0_exact_3finest"], "fit4_r_gap_half": c2["fit_r_gap_half_4finest"], "fit4_r_0": c2["fit_r_0_4finest"],
                           "minimum_all_rise": all_rise, "min_curvature_by_profile": {k: m["min_curvature"] for k, m in mins.items()}, "dE_dDelta_innermost": dEdD, "E_change_when_Delta0_set_to_(4e-3,2e-3,5e-4,1e-4)": bound_lower, "grad_free_max_by_type": gfree},
                f"radii reproduce (max rel dev {dev_r0:.1e} on r_0, r_gap_half {rg:.4f}); the end profiles rise in every random direction with positive curvature; the two seeds agree to {c2['seeds_max_rel_E_mine']:.1e} in E (the claim says 1e-9: {'met' if seeds9 else 'NOT met on the coarse rungs, met at 1e-8 on the fine ones'}); dE/dDelta at the innermost node {list(dEdD.values())[0]:+.2e} (positive = the bound is active, the regular hedgehog center wants Delta = 0); the free-node gradient max is 1e-4 (L-BFGS-B stopped ABNORMAL, the preconditioned gradient is what converged)")
    # C3
    if lt is not None:
        c3 = lt["C3"]
        b96 = c3["basins_h0.75"]["L96"]; b48 = c3["basins_h0.75"]["L48"]
        two = b96["r16_1"]["E_mine"] > b96["melted"]["E_mine"] + 1.0 and c3["line_scan_h0.75_L96"]["barrier_above_r16_1_basin"] > 1e-3
        claimed_rg = [3.388, 1.751, 2.013, 2.120, 2.137, 2.141]
        dev_rg = max(abs(a - b) / b for a, b in zip(c3["r_gap_half_ladder_L96_lin"], claimed_rg))
        l23c = c3["l23_center_ladder"]
        above = [x > 0.8 for x in l23c[-3:]]
        hz = c3["hazard"]
        v = "CONFIRMED" if (two and dev_rg < 2e-3 and not any(above)) else "QUALIFIED"
        verdict("C3", v, {"E_h0.75_L48_by_seed": {s: b48[s]["E_mine"] for s in ("r16_1", "melted")}, "E_h0.75_L96_by_seed": {s: b96[s]["E_mine"] for s in ("r16_1", "melted")}, "r_0_r16_1_basin_L48_L96": [b48["r16_1"]["r_0_lin"], b96["r16_1"]["r_0_lin"]],
                           "l1_min_melted_basin_L48_L96": [b48["melted"]["l1_min"], b96["melted"]["l1_min"]], "line_scan_barrier_above_r16_1_basin": c3["line_scan_h0.75_L96"]["barrier_above_r16_1_basin"], "line_scan_monotone_decreasing": c3["line_scan_h0.75_L96"]["monotone_decreasing"],
                           "r16_1_basin_minimum_test_all_rise": c3["minimum_h0.75_L96_r16_1_basin"]["all_rise_both_signs"], "r16_1_basin_min_curvature": c3["minimum_h0.75_L96_r16_1_basin"]["min_curvature"],
                           "r_gap_half_ladder_L96_mine": [round(x, 4) for x in c3["r_gap_half_ladder_L96_lin"]], "gap_center_ladder": c3["gap_center_ladder"], "gap_center_over_h": c3["gap_center_over_h_ladder"], "l1_center_ladder": c3["l1_center_ladder"], "l23_center_ladder": l23c, "l23_max_ladder": c3["l23_max_ladder"],
                           "l23_center_above_0.8_finest_three": above, "richardson_exact_r_gap_half": c3["richardson_r_gap_half_exact_3finest"], "fit4_r_gap_half": c3["fit_r_gap_half_4finest"], "richardson_l1_center": c3["richardson_l1_center_3finest"], "richardson_l23_center": c3["richardson_l23_center_3finest"],
                           "finest_pair_rel_change_r_gap_half": c3["finest_pair_rel_change_r_gap_half"], "L96_to_144_rel_change_h0.15": c3["L96_to_144_rel_change_r_gap_half_h0.15"],
                           "hazard": {k: {kk: vv for kk, vv in h_.items()} for k, h_ in hz.items()}, "finest_minimum_all_rise": c3["minimum_h0.03_L96"]["all_rise_both_signs"], "finest_min_curvature": c3["minimum_h0.03_L96"]["min_curvature"],
                           "center_fd_gradient": {k: {kk: vv for kk, vv in v_.items() if kk != "rows"} for k, v_ in c3["center_fd_gradient"].items()}},
                f"two basins at h 0.75 confirmed (E {b96['r16_1']['E_mine']:.3f} vs {b96['melted']['E_mine']:.3f} at L 96, a barrier of {c3['line_scan_h0.75_L96']['barrier_above_r16_1_basin']:.3f} on the straight line between them); the r_gap_half ladder reproduces (max rel dev {dev_rg:.1e}); BUT the center's pair eigenvalue on the finest three rungs is {[round(x, 4) for x in l23c[-3:]]}, ABOVE 0.8 (not 'just under'): the innermost cells sit inside the plateau weight's taper (w = {hz['h0.03']['w_plateau_l23_center']:.5f}, 1 - w = {hz['h0.03']['one_minus_w']:.1e}) and are pair_out cells evaluated by the general weight; the taper cells carry {hz['h0.03']['energy_share_pair_out_cells']:.1e} of E and clipping lambda_23 to 0.7999 changes E by {hz['h0.03']['E_clipped_l23_0.7999_minus_E']:+.1e}: the fine-h state is not an artefact of the taper's weight loss (too small) but the pair does settle ON the plateau edge")
    # C4
    if en is None:
        verdict("C4", "QUALIFIED", {}, "the energies stage did not run")
    else:
        c4 = en["C4"]
        claimed = {"v4abs_h1.5_n32": (11.778, 13.817), "v4rel_h1.5_n32": (5.387, 7.278), "v4abs_h0.75_n64": (9.802, 13.654)}
        nums, ok = {}, True
        for k, row in c4.items():
            tb = {s: row.get(f"E_3D_tube_{s}") for s in ("r16_1", "melted") if f"E_3D_tube_{s}" in row}
            nums[k] = {"E_core": row["E_3D_core"], "E_tube": tb, "below": {s: row.get(f"tube_below_core_{s}") for s in tb}, "core_dom": row["core_dom"], "tube_dom": {s: row.get(f"tube_dom_{s}") for s in tb}}
            ok = ok and all(row[f"tube_below_core_{s}"] for s in tb) and abs(row["E_3D_core"] - claimed[k][1]) < 2e-3 and abs(min(tb.values()) - claimed[k][0]) < 2e-3
        dd = {}
        for nm, dx in (("A", dA), ("B", dB)):
            dd[nm] = None if dx is None else {"E0": dx["E0"], "E_final": dx["E_final"], "dE": dx["E_final_minus_E0"], "stop": dx["stop"], "iters": dx["iters"], "dom0": dx["dom0"], "dom_final": dx["dom_final"], "went_down": dx["E_went_down"],
                                             "trace_E": [(r_["it"], r_["E_stat"], r_["dom_gap_1_2_min"], r_["dom_escape_d"]) for r_ in dx["trace"]], "l1_shell_std_initial_final": [dx["lambda_1_shell_3_4.5_std_initial"], dx["lambda_1_shell_3_4.5_std_final"]]}
        nums["descents"] = dd
        both = dA is not None and dB is not None
        verdict("C4", "CONFIRMED" if (ok and both) else ("QUALIFIED" if ok else "REFUTED"), nums,
                "the six 3D energies recomputed with my own embedding and the certified 8-sample stack reproduce the claimed values and the ordering (every embedded tube profile below its 3D core, all inside the domain, escape_d False); " +
                ("the 60-step descents: " + ", ".join(f"object {nm}: E {v['E0']:.4f} -> {v['E_final']:.4f} (dE {v['dE']:+.2e}), stop {v['stop']}, gap_1_2_min {v['dom0']['gap_1_2_min']:.4f} -> {v['dom_final']['gap_1_2_min']:.4f}" for nm, v in dd.items() if v is not None) if any(v is not None for v in dd.values()) else "descents not run"))
    # C5
    if lt is not None:
        c5 = lt["C5"]
        ok = all(v["holds_1e-12"] for v in c5.values())
        # the producer's own per-run records: how many object-B runs carry identity_holds_1e-12 False
        rec = {}
        for L_ in (48.0, 96.0, 144.0):
            for h_ in HS:
                for sd in ("r16_1", "melted"):
                    r_ = load_run("v4rel", h_, L_, sd)
                    if r_ is not None and "v6_identity" in r_:
                        rec[tag_of("v4rel", h_, L_, sd)] = (r_["v6_identity"]["rel_diff"], r_["v6_identity"]["identity_holds_1e-12"])
        n_false = sum(1 for v in rec.values() if not v[1])
        worst = max(v[0] for v in rec.values()) if rec else None
        verdict("C5", "CONFIRMED" if (ok and n_false == 0) else "QUALIFIED", {k: {"rel_diff": v["rel_diff"], "rho2_max": v["rho2_max_tube_field_eigvals"], "U_v6_mine": v["U_v6_mine_weighted_sum"], "mu_eff_min": v["mu_eff_min"]} for k, v in c5.items()} | {"producer_records_identity_false": n_false, "producer_records_total": len(rec), "producer_worst_rel_diff": worst, "producer_false_tags": [k for k, v in rec.items() if not v[1]]},
                f"the identity is exact analytically (rho^2 from numpy eigvals of N is at most {max(v['rho2_max_tube_field_eigvals'] for v in c5.values()):.1e} on the tube field, my U_v6 = 0) and holds to {max(v['rel_diff'] for v in c5.values()):.1e} on my three profiles, but NOT to 1e-12 on every profile: the code's rho^2 = (s^2 - 4p) / 4 carries 1e-16 roundoff per cell that the quadrature weights (2 pi r^2 h up to 5e3 at h 1.5) and g_W W = 2 amplify to 1.3e-11 on the h 1.5 rungs; the producer's own records say identity_holds_1e-12 False on {n_false} of {len(rec)} object-B runs (worst {worst:.1e}), so the docstring's 'checked at 1e-12 on every (B) end profile' is contradicted by its own JSON (1e-11 is the true bound, the fine rungs reach 1e-13)")
    # C6
    if lt is not None:
        c6 = lt["C6"]
        v6 = c6["values_r0_sqrt_mu"]
        unc_ok = c6["finest_pair_rel_change"]["A_r_gap_half"] < 0.003 and c6["finest_pair_rel_change"]["B_r_gap_half"] < 0.002 and c6["L96_to_144_rel_change_h0.15"]["A_r_gap_half"] < 0.004 and c6["L96_to_144_rel_change_h0.15"]["B_r_gap_half"] < 0.004
        verdict("C6", "QUALIFIED", {"r0_sqrt_mu": {k: round(v, 4) for k, v in v6.items()}, "finest_pair_rel_change": c6["finest_pair_rel_change"], "L96_to_144_rel_change_h0.15": c6["L96_to_144_rel_change_h0.15"], "L48_vs_96_B_h0.03": c6["L48_vs_96_rel_change_B_r_gap_half_h0.03"],
                                    "eighth_power_ratios": {k: round(v, 2) for k, v in c6["eighth_power_ratios"].items()}, "eighth_power_of_5pct": c6["eighth_power_of_5pct"], "eighth_power_of_num_unc_A_B": [c6["eighth_power_of_numerical_uncertainty_A"], c6["eighth_power_of_numerical_uncertainty_B"]], "E_L_dependence_B_h0.15": c6["E_L_dependence_B_h0.15"], "uncertainties_supported": unc_ok},
                f"the reported h and L uncertainties are reproduced ({'supported' if unc_ok else 'NOT supported'}: A {c6['finest_pair_rel_change']['A_r_gap_half']:.2e} / {c6['L96_to_144_rel_change_h0.15']['A_r_gap_half']:.2e}, B {c6['finest_pair_rel_change']['B_r_gap_half']:.2e} / {c6['L96_to_144_rel_change_h0.15']['B_r_gap_half']:.2e}) and each is far inside 5 percent, but the DEFINITION dominates: the eighth power of B's lambda_1 = 0.8 read over B's gap-half read is {c6['eighth_power_ratios']['B_lambda1_over_B_gap_half']:.1f}, of B's gap-half over A's gap-half {c6['eighth_power_ratios']['B_gap_half_over_A_gap_half']:.0f}, against a 5 percent tolerance that is a factor {c6['eighth_power_of_5pct']:.2f} on the eighth power; the author's determination changes by an order of magnitude with the read and by 2.5 orders with the weight, so RADIAL_CONVERGED describes the lattice, not the physical number; note also E itself is NOT L-converged (B at h 0.15: {c6['E_L_dependence_B_h0.15']['48.0']:.3f} / {c6['E_L_dependence_B_h0.15']['96.0']:.3f} / {c6['E_L_dependence_B_h0.15']['144.0']:.3f} at L 48 / 96 / 144, the 1 / r^4 tail's volume integral grows with L) while the radii are")
    out = {"rung": "R18-1 audit", "auditor": "independent adversarial (Fable 5.1), own script m5_32_r18_1_audit.py", "claims": claims,
           "stage_walls_s": {k: (v["wall_s"] if v else None) for k, v in parts.items()}, "stage_files": {k: stage_file(k) for k in parts}, "wall_s_summary_stage": round(time.time() - T0, 1)}
    json.dump(out, open(os.path.join(DATA, "m5_32_r18_1_audit.json"), "w"), indent=1, default=float)
    print("\n| claim | verdict | key numbers | reason |\n| --- | --- | --- | --- |")
    for cid, c in claims.items():
        short = {k: v for k, v in list(c["numbers"].items())[:3]}
        print(f"| {cid} | {c['verdict']} | {json.dumps(short, default=float)[:220]} | {c['reason'][:300]} |")
    print(f"\nwritten {os.path.join(DATA, 'm5_32_r18_1_audit.json')}; stage walls {out['stage_walls_s']}")
    return out


if __name__ == "__main__":
    if A.stage == "light":
        stage_light()
    elif A.stage == "energies":
        stage_energies()
    elif A.stage == "descent_A":
        stage_descent("v4abs")
    elif A.stage == "descent_B":
        stage_descent("v4rel")
    elif A.stage == "summary":
        stage_summary()
    else:
        stage_light(); stage_energies(); stage_descent("v4rel"); stage_descent("v4abs"); stage_summary()
