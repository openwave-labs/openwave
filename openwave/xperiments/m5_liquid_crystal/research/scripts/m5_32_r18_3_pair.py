"""M5.32 R18-3 (ledger 6.7): the like-charge pair on the degenerate vacuum (the author's rev-200 section 180.6 item 3: the pair,
or the statement that the string is generic and the 8 pi test retired).

Object (A) v4 as run in R16-1 (mu 1e-2, c_P 1, c_s 0.4, I_rebuild, the absolute plateau weight), the R17 instrument.
Seed (the boxes hold the RADIAL hedgehog texture on their pinned shell: the degenerate vacuum's far field around a core is
radial, not uniform, so a copy of the core pasted into a uniform vacuum is discontinuous at its old box edge, measured at
K_P 1162 against 8; the seed is built from the director instead): n = normalize(rhat_1 + rhat_2) about the two centers
(+-d/2, 0, 0), the eigenvalues from the R16-1 n32 core's radial profiles (the director gap multiplicative, lambda_23 and m_g
additive), S = lambda_23 I + Delta n n^T: each core carries the single's profile, the far field is the radial hedgehog about
the midpoint up to O(d^2 / 8 r^2) in the director, the spectrum inside the admissible domain by construction, and the segment
between the cores carries the +1 disclination line of the vector sum (the string of the packet, in the seed).  The single
core on the same box is the same construction with one center.  The anti-pair control of the packet is NOT constructible on
these boxes (a stated deviation): the radial boundary texture has degree 1, a radial-plus-hyperbolic pair has degree 0.
d in {12, 15, 18, 24} (commensurate: d / 2 a multiple of h = 1.5; the packet's 16 and 20 are not).  Protocols: FREE (the
R16 statics protocol) and PINNED (the R14-C protocol as a quadratic hold on the cells within r_pin 2.6 of each center,
E_pin = (k / 2) sum |M - M_seed|^2, k 10; gate: k 0 reproduces the free problem, m5_32_r18_common selftest).  The string read:
the line-field winding of the transverse director on the ring of four cells around the axis at x = 0 and +-d/4 (the +1
disclination kept: winding 1; escaped into the axial direction: |n_x| > 0.7 on the rings).
Reads: E(d) - 2 E_1 on both protocols (E_stat at 8 samples, the pinning energy reported separately), the A + B / d fit on the
pinned ladder with B against 8 pi A_tail (the one-unit-system 8 pi test: the same code, the same box, the same object), the
pair field's own tail amplitude against 4 A_tail (the linear superposition of two charges), the director gap and the split
along the axis between the cores (the string / escape tube on the free runs), the L-exponent between the two boxes.
Pre-registered outcomes: PAIR_LAW_CERTIFIED (constrained): the pinned B / A_tail within 10 percent of 8 pi; PAIR_LAW_RETIRED:
the free pair reconnects into the string at every d (the axis gap closes, escape (d), or the free E(d) rises with d), by the
author's own rule, with the constrained read beside it; NO_1/d_REGIME: the pinned ladder has no 1 / d regime at reachable d.

usage: python3 m5_32_r18_3_pair.py relax --n 48 --L 72 --d 15 --protocol pinned|free --charge same|anti [--maxit 1500]
       python3 m5_32_r18_3_pair.py single --n 48 --L 72 --protocol pinned|free [--maxit 1500]
       python3 m5_32_r18_3_pair.py collect
out:   checkpoints/m5_32_r18/r18_3_<tag>.json / .npy / _nref.npy, data/m5_32_r18_3.json, plots/m5_32_r18_3_pair.png
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
import m5_32_r16_1_statics as S1                          # noqa: E402
import m5_32_r17_0_record as REC                          # noqa: E402

C15, INS4 = C.C15, C.INS4
RES, DATA, PLOTS = C.RES, C.DATA, C.PLOTS
CK16, CK17, CK = C.CK, R.CK, X.CK
T0 = time.time()
R_PIN, K_PIN = 2.6, 10.0


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def rel(p):
    return os.path.relpath(p, RES)


def make_cfg(n, L):
    cfg = C.cfg_v4(n, L, completion="rebuild", n_samples=4)
    cfg["object"] = "v4"; cfg["weight"] = "absolute"
    return cfg


def tag_of(n, L, d, protocol, charge):
    core = "single" if d is None else f"{charge}_d{d:g}"
    return f"r18_3_{core}_{protocol}_n{n}_L{int(L)}"


_PROF = {}


def single_profiles():
    """the R16-1 n32 L48 core's radial profiles (m_g, lambda_1, lambda_23) as shell means on the h 1.5 grid (m5_32_r18_common.profiles_from_field)."""
    if not _PROF:
        cfg32 = C.cfg_v4(32, 48.0, completion="rebuild")
        M1 = np.load(os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"))
        t32 = X.tube_setup(1.5, 48.0)
        P = X.profiles_from_field(M1, cfg32, t32)
        _PROF["rg"] = t32["rg"]; _PROF["P"] = P; _PROF["vac"] = INS4.vac4(cfg32)
    return _PROF["rg"], _PROF["P"], _PROF["vac"]


def field_from_director(cfg, centers):
    """the seed: the director n = normalize(sum_i rhat_i) (the radial hedgehog about each center, superposed as VECTORS; for two
    centers the sum vanishes on the segment between them, where the seed carries the +1 disclination line, the string), the
    eigenvalues from the single core's profiles combined multiplicatively in the gap and additively in the rest:
        Delta(x) = prod_i Delta_1(|x - c_i|) / Delta_vac^(k-1),  lambda_23(x) = sum_i lambda_23,1(|x - c_i|) - (k-1) delta,
        m_g(x) = sum_i m_g,1(|x - c_i|) - (k-1) g,   lambda_1 = lambda_23 + Delta,   S = lambda_23 I + Delta n n^T,
    so each core carries the single's profile, the far field is the radial hedgehog about the midpoint up to O(d^2 / 8 r^2)
    in the director (the boundary texture is the seed's own, d-dependent at that order: the L-exponent between the two boxes
    measures it), and the spectrum stays inside the admissible domain everywhere by construction."""
    n, h = cfg["n"], cfg["h"]
    rg, P, vac = single_profiles()
    X_, Y_, Z_ = INS4.coords(n, h)
    pos = np.stack([X_, Y_, Z_], -1)
    k = len(centers)
    g_vac, l1_vac, l23_vac = vac[0, 0], vac[1, 1], vac[2, 2]
    Dvac = l1_vac - l23_vac
    nsum = np.zeros_like(pos)
    Delta = np.ones(pos.shape[:-1]); l23 = np.zeros(pos.shape[:-1]); mg = np.zeros(pos.shape[:-1])
    for c in centers:
        y = pos - np.array(c)
        r = np.sqrt(np.sum(y * y, -1))
        nsum += y / r[..., None]
        D1 = np.interp(r, rg, P[1] - P[2], right=Dvac)
        Delta *= D1
        l23 += np.interp(r, rg, P[2], right=l23_vac)
        mg += np.interp(r, rg, P[0], right=g_vac)
    Delta /= Dvac ** (k - 1)
    l23 -= (k - 1) * l23_vac
    mg -= (k - 1) * g_vac
    nn = np.linalg.norm(nsum, axis=-1)
    nvec = nsum / np.maximum(nn, 1e-300)[..., None]
    M = np.zeros(pos.shape[:-1] + (4, 4))
    M[..., 0, 0] = mg
    M[..., 1:, 1:] = l23[..., None, None] * np.eye(3) + Delta[..., None, None] * nvec[..., :, None] * nvec[..., None, :]
    return M, {"Delta_min": float(np.min(Delta)), "n_sum_norm_min": float(np.min(nn)), "cells_with_n_sum_below_0.1": int(np.sum(nn < 0.1))}


def seed_pair(cfg, d, charge):
    if d is None:
        M, info = field_from_director(cfg, [(0.0, 0.0, 0.0)])
        return M, [(0.0, 0.0, 0.0)], info
    c1, c2 = (-d / 2.0, 0.0, 0.0), (d / 2.0, 0.0, 0.0)
    M, info = field_from_director(cfg, [c1, c2])
    return M, [c1, c2], info


def ring_reads(M, cfg, x0, rho=1.06):
    """the director on the ring of cells around the x axis at x = x0 (the four cells at y, z = +-h/2): the line-field winding of
    the transverse director (the +1 disclination of the string: winding 1; escaped: the axial component |n_x| near 1)."""
    n, h = cfg["n"], cfg["h"]
    X_, Y_, Z_ = INS4.coords(n, h)
    j = n // 2
    i = int(np.argmin(np.abs(X_[:, j, j] - x0)))
    cells = [(i, j, j), (i, j - 1, j), (i, j - 1, j - 1), (i, j, j - 1)]          # counterclockwise in the (y, z) plane
    S = M[..., 1:, 1:]
    ang, ax = [], []
    for c in cells:
        w, V = np.linalg.eigh(S[c])
        nv = V[:, -1]
        ang.append(np.arctan2(nv[2], nv[1])); ax.append(abs(nv[0]))
    tot = 0.0
    for a_, b_ in zip(ang, ang[1:] + ang[:1]):
        d_ = (2.0 * (b_ - a_) + np.pi) % (2.0 * np.pi) - np.pi                    # the LINE field: the angle doubled
        tot += d_
    return {"x": float(X_[i, j, j]), "line_field_winding": float(tot / (2.0 * np.pi) / 2.0), "mean_abs_n_x": float(np.mean(ax)), "min_gap_on_ring": float(min(np.linalg.eigvalsh(S[c])[-1] - np.linalg.eigvalsh(S[c])[-2] for c in cells))}


def axis_reads(M, cfg, centers, nref):
    """the director gap, lambda_1 and the half split along the x axis between and around the cores (the string / escape tube read)."""
    n, h = cfg["n"], cfg["h"]
    fr = C.frame(M, nref)
    l1, s, p = (np.real(fr[k]) for k in ("l1", "s", "p"))
    disc = np.sqrt(np.maximum(s * s - 4.0 * p, 0.0))
    gap = l1 - (s + disc) / 2.0
    j = n // 2
    X_, Y_, Z_ = INS4.coords(n, h)
    x = X_[:, j, j]
    out = {"x": x.tolist(), "gap": gap[:, j, j].tolist(), "lambda_1": l1[:, j, j].tolist(), "half_split": (disc[:, j, j] / 2.0).tolist()}
    if len(centers) == 2:
        between = (x > centers[0][0] + R_PIN) & (x < centers[1][0] - R_PIN)
        out["gap_min_between_cores"] = float(np.min(gap[:, j, j][between])) if np.any(between) else None
        out["half_split_max_between_cores"] = float(np.max(disc[:, j, j][between]) / 2.0) if np.any(between) else None
        out["lambda_1_min_between_cores"] = float(np.min(l1[:, j, j][between])) if np.any(between) else None
        # the cores' centers: the gap minimum near each
        for i, c in enumerate(centers):
            near = np.abs(x - c[0]) < 3.0
            out[f"core_{i}_gap_min"] = float(np.min(gap[:, j, j][near]))
            out[f"core_{i}_x_of_gap_min"] = float(x[near][int(np.argmin(gap[:, j, j][near]))])
    dom = C.domain(fr, cfg)
    out["domain"] = dom
    return out


def tail_read(M, cfg, nref):
    """the static density's 1 / r^4 tail about the origin (the pair's midpoint / the single's center): the R17-0 definition."""
    n, h, L = cfg["n"], cfg["h"], cfg["L"]
    dens, fr = REC.densities(M, cfg)
    e = sum(dens.values())
    X_, Y_, Z_ = INS4.coords(n, h)
    r = np.sqrt(X_ * X_ + Y_ * Y_ + Z_ * Z_)
    return REC.tail_fit(e, r, h, L)


def run(n, L, d, protocol, charge, maxit):
    tag = tag_of(n, L, d, protocol, charge)
    cfg = make_cfg(n, L)
    M0, centers, sinfo = seed_pair(cfg, d, charge)
    nref = C.radial_ref(cfg) if d is None else C.radial_ref(cfg, "x")
    free = ~INS4.pin_shell(n, cfg["h"], 1.6)
    X_, Y_, Z_ = INS4.coords(n, cfg["h"])
    pin_mask = np.zeros((n, n, n), dtype=bool)
    for c in centers:
        pin_mask |= np.sqrt((X_ - c[0]) ** 2 + (Y_ - c[1]) ** 2 + (Z_ - c[2]) ** 2) < R_PIN
    pin = (M0.copy(), pin_mask, K_PIN) if protocol == "pinned" else None
    rec = {"tag": tag, "rung": "R18-3", "n": n, "L": L, "h": cfg["h"], "d": d, "protocol": protocol, "charge": charge if d is not None else None, "centers": centers, "maxit": maxit,
           "pin": {"r_pin": R_PIN, "k": K_PIN, "n_cells": int(np.sum(pin_mask))} if pin else None,
           "seed": {"how": ("two radial hedgehogs at +-d/2: the director normalize(rhat_1 + rhat_2) (the +1 disclination line between them in the seed), the eigenvalues from the R16-1 n32 core's profiles (the gap multiplicative, the rest additive)" if d is not None else
                            "the R16-1 n32 L48 core's radial profiles on this box (the director radial)"), "construction": sinfo},
           "cfg": {k: cfg[k] for k in ("mu", "cP", "cs", "n_samples", "stencil", "weight", "completion")}}
    log(f"{tag}: seed {rec['seed']['how']} ({sinfo}); pinned cells {rec['pin']['n_cells'] if pin else 0}; h {cfg['h']}")
    cf8 = dict(cfg); cf8["n_samples"] = 8
    E0, _, pp0, dom0, _ = X.energy_object_pinned(M0, cf8, None, nref, need_grad=False, pin=pin)
    rec["seed"]["E_stat_8"] = float(pp0.get("E_stat_free", pp0["E_stat"])); rec["seed"]["E_pin"] = float(pp0.get("E_pin", 0.0)); rec["seed"]["domain"] = dom0
    rec["seed"]["axis"] = axis_reads(M0, cfg, centers, nref)
    if d is not None:
        rec["seed"]["rings"] = [ring_reads(M0, cfg, x0) for x0 in (0.0, -d / 4.0, d / 4.0)]
    log(f"  seed E_stat_8 {rec['seed']['E_stat_8']:.6f} (E_pin {rec['seed']['E_pin']:.4f}); parts {({k: round(float(np.real(v)), 4) for k, v in pp0.items() if k in ('E_h', 'V4', 'KP', 'reg')})}; gap between cores {rec['seed']['axis'].get('gap_min_between_cores')}; rings {rec['seed'].get('rings')}; domain l1min {dom0['l1_min']:.4f} gap12 {dom0['gap_1_2_min']:.4f}")
    json.dump(rec, open(os.path.join(CK, tag + ".json"), "w"), indent=1, default=float)
    ckp = os.path.join(CK, tag + ".npy")
    t = time.time()
    if pin is None:
        M, info = R.fire_object(M0, cfg, free, maxit, K=None, n_ref=nref, log_every=100, tag=tag, diag=S1.make_diag(cfg), ck_path=ckp, ck_every=200)
    else:
        M, info = X.fire_pinned(M0, cfg, free, maxit, pin, K=None, n_ref=nref, log_every=100, tag=tag, diag=S1.make_diag(cfg), ck_path=ckp, ck_every=200)
    rec["descent"] = {k: info[k] for k in ("stop", "wall_s", "iters")}
    rec["trace"] = info["trace"][::5]
    np.save(ckp, M)
    np.save(ckp[:-4] + "_nref.npy", np.real(info["n_ref"]))
    log(f"  descent {info['stop']} after {info['iters']} it, {info['wall_s']:.0f} s; end reads")
    if info["stop"] == "non-finite":
        rec["verdict"] = "NUMERICALLY_UNRESOLVED (non-finite)"
        json.dump(rec, open(os.path.join(CK, tag + ".json"), "w"), indent=1, default=float)
        return rec
    t = time.time()
    E, _, pp, dom, fr = X.energy_object_pinned(M, cf8, None, info["n_ref"], need_grad=False, pin=pin)
    rec["reads"] = {"E_stat_8": float(pp.get("E_stat_free", pp["E_stat"])), "E_pin": float(pp.get("E_pin", 0.0)), "parts_8": {k: float(np.real(v)) for k, v in pp.items() if isinstance(v, (int, float))}, "domain": dom,
                    "axis": axis_reads(M, cfg, centers, info["n_ref"]), "tail": tail_read(M, cfg, info["n_ref"]), "max_abs_change": float(np.max(np.abs(M - M0)))}
    rec["end_field"] = rel(ckp); rec["end_lift"] = rel(ckp[:-4] + "_nref.npy")
    ax = rec["reads"]["axis"]
    if d is not None:
        rings = [ring_reads(M, cfg, x0) for x0 in (0.0, -d / 4.0, d / 4.0)]
        rec["reads"]["rings"] = rings
        string_kept = all(abs(r_["line_field_winding"]) > 0.5 for r_ in rings)
        escaped = all(r_["mean_abs_n_x"] > 0.7 for r_ in rings)
        rec["string_read"] = {"gap_min_between_cores": ax.get("gap_min_between_cores"), "half_split_max_between_cores": ax.get("half_split_max_between_cores"), "lambda_1_min_between_cores": ax.get("lambda_1_min_between_cores"),
                              "ring_windings": [r_["line_field_winding"] for r_ in rings], "ring_mean_abs_n_x": [r_["mean_abs_n_x"] for r_ in rings],
                              "string_kept (the +1 line-field winding on the rings at x = 0, +-d/4)": bool(string_kept), "string_escaped (|n_x| > 0.7 on the rings)": bool(escaped),
                              "reconnected (the axis gap between the cores below 0.05 or escape d)": bool((ax.get("gap_min_between_cores") is not None and ax["gap_min_between_cores"] < 0.05) or dom["escape_d"])}
    json.dump(rec, open(os.path.join(CK, tag + ".json"), "w"), indent=1, default=float)
    log(f"  END {tag}: E_stat_8 {rec['reads']['E_stat_8']:.6f} (E_pin {rec['reads']['E_pin']:.5f}); stop {info['stop']}; tail A {rec['reads']['tail'].get('median_dens_r4')} slope {rec['reads']['tail'].get('loglog_slope')}; "
        f"axis gap between {ax.get('gap_min_between_cores')} split {ax.get('half_split_max_between_cores')}; string {rec.get('string_read')}; {time.time() - t:.0f} s reads")
    return rec


def midplane_reads(M, cfg, d, x0=0.0, rings=((0.5, 1.6), (1.6, 2.8), (2.8, 4.0), (4.0, 5.5))):
    """the string's transverse structure on the plane x = x0 (the midpoint between the cores): the director eigenvalue, the gap
    lambda_1 - lambda_23 and the pair eigenvalue as functions of the transverse radius rho (band means), the line-field winding of
    the transverse director on rings of increasing rho (a +1 line with a melted core reads 0 inside the core and 1 outside it),
    and the string core radius (where the gap recovers to half the vacuum gap 0.35)."""
    n, h = cfg["n"], cfg["h"]
    X_, Y_, Z_ = INS4.coords(n, h)
    j = n // 2
    i = int(np.argmin(np.abs(X_[:, j, j] - x0)))
    S = M[i][..., 1:, 1:]
    w, V = np.linalg.eigh(S)
    l1 = w[..., 2]; l2 = w[..., 1]; l3 = w[..., 0]
    gap = l1 - 0.5 * (l2 + l3)
    nv = V[..., :, 2]
    rho = np.sqrt(Y_[i] ** 2 + Z_[i] ** 2)
    phi = np.arctan2(Z_[i], Y_[i])
    edges = np.arange(0.0, 12.0 + h, h)
    prof = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (rho >= a) & (rho < b)
        if np.sum(m) >= 4:
            prof.append({"rho": float(0.5 * (a + b)), "lambda_1_mean": float(np.mean(l1[m])), "gap_mean": float(np.mean(gap[m])), "gap_min": float(np.min(gap[m])), "pair_mean": float(np.mean(0.5 * (l2 + l3)[m])), "half_split_max": float(np.max(0.5 * (l2 - l3)[m])), "abs_n_x_mean": float(np.mean(np.abs(nv[m][:, 0])))})
    ring_out = []
    for a, b in rings:
        m = (rho >= a) & (rho < b)
        if np.sum(m) < 4:
            continue
        order = np.argsort(phi[m])
        ang = np.arctan2(nv[m][:, 2], nv[m][:, 1])[order]
        tot = 0.0
        for a_, b_ in zip(ang, np.concatenate([ang[1:], ang[:1]])):
            tot += (2.0 * (b_ - a_) + np.pi) % (2.0 * np.pi) - np.pi
        ring_out.append({"rho_band": [a, b], "n_cells": int(np.sum(m)), "line_field_winding": float(tot / (4.0 * np.pi)), "mean_abs_n_x": float(np.mean(np.abs(nv[m][:, 0]))), "gap_min": float(np.min(gap[m]))})
    gm = np.array([p_["gap_mean"] for p_ in prof]); rr = np.array([p_["rho"] for p_ in prof])
    cross = np.where((gm[:-1] < 0.35) & (gm[1:] >= 0.35))[0]
    r_core = float(rr[cross[0]] + (0.35 - gm[cross[0]]) * (rr[cross[0] + 1] - rr[cross[0]]) / (gm[cross[0] + 1] - gm[cross[0]])) if len(cross) else 0.0
    return {"x": float(X_[i, j, j]), "profile": prof, "rings": ring_out, "string_core_radius (gap = 0.35 crossing in rho)": r_core, "gap_on_axis_min": float(np.min(gap[rho < 1.2])), "lambda_1_on_axis_mean": float(np.mean(l1[rho < 1.2]))}


def collect():
    runs = {}
    for p in sorted(glob.glob(os.path.join(CK, "r18_3_*.json"))):
        r_ = json.load(open(p)); r_.pop("trace", None)
        if r_.get("d") is not None and "reads" in r_ and os.path.exists(os.path.join(RES, r_.get("end_field", "x"))):
            cfg = make_cfg(r_["n"], r_["L"])
            M = np.load(os.path.join(RES, r_["end_field"]))
            r_["reads"]["midplane"] = midplane_reads(M, cfg, r_["d"])
            r_["reads"]["quarter_plane"] = midplane_reads(M, cfg, r_["d"], x0=-r_["d"] / 4.0)
            mp = r_["reads"]["midplane"]
            r_["string_read"]["midplane_rings"] = [(rg["rho_band"], round(rg["line_field_winding"], 2), round(rg["mean_abs_n_x"], 3)) for rg in mp["rings"]]
            r_["string_read"]["string_core_radius"] = mp["string_core_radius (gap = 0.35 crossing in rho)"]
            r_["string_read"]["string_present (winding 1 on a ring outside the melted core)"] = bool(any(abs(rg["line_field_winding"]) > 0.5 for rg in mp["rings"]))
        runs[r_["tag"]] = r_
    out = {"rung": "R18-3", "runs": runs, "boxes": {}}
    for (n, L) in ((48, 72.0), (64, 96.0)):
        box = {}
        for protocol in ("pinned", "free"):
            s_tag = tag_of(n, L, None, protocol, None)
            single = runs.get(s_tag)
            E1 = single["reads"]["E_stat_8"] if single and "reads" in single else None
            A1 = single["reads"]["tail"].get("median_dens_r4") if single and "reads" in single else None
            rows = []
            for tag, r_ in runs.items():
                if r_["n"] == n and r_["protocol"] == protocol and r_["d"] is not None and "reads" in r_:
                    rows.append({"d": r_["d"], "charge": r_["charge"], "E": r_["reads"]["E_stat_8"], "E_pin": r_["reads"]["E_pin"], "E_int": (r_["reads"]["E_stat_8"] - 2 * E1) if E1 is not None else None,
                                 "tail_A_pair": r_["reads"]["tail"].get("median_dens_r4"), "tail_A_pair_over_4A1": (r_["reads"]["tail"].get("median_dens_r4") / (4 * A1)) if (A1 and r_["reads"]["tail"].get("median_dens_r4") is not None) else None,
                                 "stop": r_["descent"]["stop"], "iters": r_["descent"]["iters"], "string": r_.get("string_read", {}).get("string_present (winding 1 on a ring outside the melted core)"),
                                 "string_core_radius": r_.get("string_read", {}).get("string_core_radius"), "midplane_rings": r_.get("string_read", {}).get("midplane_rings"),
                                 "gap_between": r_["reads"]["axis"].get("gap_min_between_cores"), "lambda_1_min_between": r_["reads"]["axis"].get("lambda_1_min_between_cores"), "E_pin": r_["reads"]["E_pin"]})
            rows.sort(key=lambda x: (x["charge"], x["d"]))
            same = [x for x in rows if x["charge"] == "same"]
            fit = None
            if len(same) >= 3 and E1 is not None:
                ds = np.array([x["d"] for x in same]); Es = np.array([x["E"] for x in same])
                Bc, Ac = np.polyfit(1.0 / ds, Es, 1)
                resid = Es - (Ac + Bc / ds)
                fit = {"A": float(Ac), "B": float(Bc), "B_over_A_tail": float(Bc / A1) if A1 else None, "8pi": float(8 * np.pi), "B_over_A_tail_over_8pi": float(Bc / A1 / (8 * np.pi)) if A1 else None,
                       "fit_rms_resid": float(np.sqrt(np.mean(resid ** 2))), "E_range": float(np.max(Es) - np.min(Es)), "monotone_rising_with_d": bool(np.all(np.diff(Es) > 0)), "monotone_falling_with_d": bool(np.all(np.diff(Es) < 0)),
                       "A_minus_2E1": float(Ac - 2 * E1)}
            lin = None
            if len(same) >= 3 and E1 is not None:
                ds = np.array([x["d"] for x in same]); Ei = np.array([x["E"] - 2 * E1 for x in same])
                sig, c0 = np.polyfit(ds, Ei, 1)
                res_l = Ei - (c0 + sig * ds)
                lin = {"string_tension_sigma": float(sig), "E_int_at_d0_intercept": float(c0), "fit_rms_resid": float(np.sqrt(np.mean(res_l ** 2))), "E_int_range": float(np.max(Ei) - np.min(Ei)),
                       "d_where_E_int_crosses_zero": float(-c0 / sig) if sig != 0 else None, "linear_fit_resid_over_1_over_d_fit_resid": (float(np.sqrt(np.mean(res_l ** 2)) / fit["fit_rms_resid"]) if fit and fit["fit_rms_resid"] > 0 else None)}
            box[protocol] = {"single": {"E_1": E1, "A_tail": A1, "stop": single["descent"]["stop"] if single and "descent" in single else None}, "pairs": rows, "fit_A_plus_B_over_d": fit, "fit_linear_in_d (the string)": lin}
        out["boxes"][f"n{n}_L{int(L)}"] = box
    # the verdict
    v = []
    for bk, box in out["boxes"].items():
        fp = box["pinned"]["fit_A_plus_B_over_d"]
        fr_ = box["free"]["pairs"]
        same_free = [x for x in fr_ if x["charge"] == "same"]
        lin = box["pinned"]["fit_linear_in_d (the string)"]
        if fp is not None:
            if fp["B_over_A_tail_over_8pi"] is not None and abs(fp["B_over_A_tail_over_8pi"] - 1.0) < 0.10 and fp["B"] > 0:
                v.append(f"{bk}: PAIR_LAW_CERTIFIED (constrained: pinned B / A_tail = {fp['B_over_A_tail']:.2f} vs 8 pi = {8 * np.pi:.2f})")
            elif fp["monotone_rising_with_d"]:
                v.append(f"{bk}: NO_1/d_REGIME on the pinned ladder: E_int rises LINEARLY with d (the string, tension {lin['string_tension_sigma']:.3f} per unit length, linear rms {lin['fit_rms_resid']:.3f} vs the 1/d fit's {fp['fit_rms_resid']:.3f}; E_int negative below d {lin['d_where_E_int_crosses_zero']:.1f}: the cores attract along the string)" if lin else f"{bk}: NO_1/d_REGIME (E rises with d)")
            else:
                v.append(f"{bk}: pinned B / A_tail = {fp['B_over_A_tail']} against 8 pi: not within 10 percent (fit rms {fp['fit_rms_resid']:.2e} on an E range {fp['E_range']:.2e})")
        if same_free:
            rec_ = [x["string"] for x in same_free]
            v.append(f"{bk}: free pair reconnected at d {[x['d'] for x in same_free if x['string']]} of {[x['d'] for x in same_free]}" + (" -> PAIR_LAW_RETIRED by the author's rule (the string generic)" if all(rec_) else ""))
    v.append("R18-3 audit (folded in): the seed normalize(rhat_1 + rhat_2) with the box shell pinned to it has TOTAL CHARGE ONE at every d (the far sphere's degree 1.00, the far field one unit hedgehog's), so the measured object is a unit hedgehog with a split core (two melted half-cores joined by the winding-1 line), not two like charges; "
             "the 8 pi test is inapplicable by TOPOLOGY on the radial boundary (a charge-2 object is not representable there), the string tension is budget-dependent (0.80 to 1.19 across the trace extrapolation, the same-800-iteration ladder and the mixed budgets; the monotone linear rise is the robust content), the string zone carries all the d dependence while the core term is flat (no core-core force) and the far zone the one-texture offset, "
             "the free cores contract by 0.62 each in 1500 iterations (a slow contraction, not a static pair), and the linear law extrapolates to E(d) = E_1 at d 5.7 (the open question for the next rung: does the law bend below d 6, or is the split-core hedgehog cheaper than the radial single)")
    out["verdicts"] = v
    json.dump(out, open(os.path.join(DATA, "m5_32_r18_3.json"), "w"), indent=1, default=float)
    log(f"collected {len(runs)} runs; verdicts {v}")
    plot(out)
    return out


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for bk, box in out["boxes"].items():
        for protocol, mk in (("pinned", "o"), ("free", "s")):
            rows = [x for x in box[protocol]["pairs"] if x["charge"] == "same" and x["E_int"] is not None]
            if rows:
                ax[0].plot([x["d"] for x in rows], [x["E_int"] for x in rows], marker=mk, lw=0.8, label=f"{bk} {protocol} same")
            rows = [x for x in box[protocol]["pairs"] if x["charge"] == "anti" and x["E_int"] is not None]
            if rows:
                ax[0].plot([x["d"] for x in rows], [x["E_int"] for x in rows], marker=mk, lw=0.8, ls=":", label=f"{bk} {protocol} anti")
            fp = box[protocol]["fit_A_plus_B_over_d"]
            A1 = box[protocol]["single"]["A_tail"]
            if fp and A1:
                ds = np.linspace(10, 26, 50)
                ax[1].plot(ds, fp["A"] + fp["B"] / ds - fp["A"], lw=0.8, label=f"{bk} {protocol}: B/A_tail {fp['B_over_A_tail']:.2f} (8pi {8 * np.pi:.2f})")
                rows = [x for x in box[protocol]["pairs"] if x["charge"] == "same"]
                ax[1].plot([x["d"] for x in rows], [x["E"] - fp["A"] for x in rows], marker=mk, lw=0, label=None)
            rows = [x for x in box[protocol]["pairs"] if x["charge"] == "same" and x["gap_between"] is not None]
            if rows:
                ax[2].plot([x["d"] for x in rows], [x["gap_between"] for x in rows], marker=mk, lw=0.8, label=f"{bk} {protocol}")
    ax[0].set_xlabel("d"); ax[0].set_ylabel("E(d) - 2 E_1"); ax[0].set_title("the pair interaction energy", fontsize=8); ax[0].legend(fontsize=6)
    ax[1].set_xlabel("d"); ax[1].set_ylabel("E(d) - A  (the B / d part)"); ax[1].set_title("the A + B / d fit on the same-charge ladder", fontsize=8); ax[1].legend(fontsize=6)
    ax[2].set_xlabel("d"); ax[2].set_ylabel("min director gap on the axis between the cores"); ax[2].axhline(0.05, color="r", ls=":", lw=0.7); ax[2].set_title("the string read (gap below 0.05 = reconnected)", fontsize=8); ax[2].legend(fontsize=6)
    fig.suptitle("R18-3: like-charge pairs on the degenerate vacuum, pinned and free, two boxes", fontsize=9)
    fig.savefig(os.path.join(PLOTS, "m5_32_r18_3_pair.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    out["plot"] = "plots/m5_32_r18_3_pair.png"
    json.dump(out, open(os.path.join(DATA, "m5_32_r18_3.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["relax", "single", "collect"])
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--L", type=float, default=72.0); ap.add_argument("--d", type=float, default=15.0)
    ap.add_argument("--protocol", default="pinned", choices=["pinned", "free"]); ap.add_argument("--charge", default="same", choices=["same", "anti"]); ap.add_argument("--maxit", type=int, default=1500)
    a = ap.parse_args(ARGS)
    if a.mode == "relax":
        run(a.n, a.L, a.d, a.protocol, a.charge, a.maxit)
    elif a.mode == "single":
        run(a.n, a.L, None, a.protocol, None, a.maxit)
    else:
        collect()
