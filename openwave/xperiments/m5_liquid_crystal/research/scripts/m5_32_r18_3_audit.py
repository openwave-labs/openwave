"""M5.32 R18-3 independent adversarial audit (the like-charge pair on the degenerate vacuum, m5_32_r18_3_pair.py).

Each claim of the producer is re-derived with its own method on the producer's END FIELDS only (no producer function that
produced the audited number is reused for the audited number itself; the certified energy m5_32_r17_common.energy_object
and the per-cell densities m5_32_r17_0_record.densities are the shared instrument, consumed read-only):
  C1  E_1 at 8 samples and the tail amplitude A_1 recomputed from the end fields with own shells; the descent traces parsed
      from the logs (E_stat per 100 it), the end slopes, a geometric extrapolation of the remaining descent per run, and
      the same-budget E_int against the extrapolated one.
  C2  E(d) recomputed for the four pinned pairs; E_pin confirmed excluded from E_stat_8 (parts_8 E_stat == E_stat_free +
      E_pin); the A + B / d, c + sigma d and c + sigma d + B / d fits redone, and a same-budget (800 it) ladder.
  C3  own line-field winding on interpolated rings (trilinear S, 72 points) at x = 0, +-d / 4, rho 2.2 and 4.5, plus the
      producer-style cell-ring reading; own transverse gap profile; the hedgehog degree of the outward-oriented director
      on spheres (solid-angle sum over a lat-long triangulation) about each core (r 4, 6) and about the origin (r 30),
      gated on the seed fields (exact hedgehogs).
  C4  the pair tail amplitude with own shells, with a window that EXCLUDES the cores (r > d / 2 + 6) beside the producer's
      window; the single's energy outside r > 8 and r > 12 (sum e h^3), the pair's outside the same radii about the
      origin, and the decomposition E_int = [inside] + [outside] against the linear intercept.
  C5  the cores' centers in the free and the pinned d 15 end fields (the gap and lambda_1 minima along the axis by
      trilinear interpolation, and the 3D centroid of the melted-core cells), the distance, and the free-minus-pinned field.
  C6  the verdict against the numbers of C1 to C5 (text).
usage: python3 m5_32_r18_3_audit.py [--max-energies 10]      (run from research/scripts/)
out:   data/m5_32_r18_3_audit.json
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time

import numpy as np

ARGS = list(sys.argv[1:])
sys.argv = [sys.argv[0]]
ap = argparse.ArgumentParser()
ap.add_argument("--max-energies", type=int, default=10)
A_ = ap.parse_args(ARGS)

import m5_32_r18_common as X                              # noqa: E402
import m5_32_r17_common as R                              # noqa: E402
import m5_32_r16_common as C                              # noqa: E402
import m5_32_r17_0_record as REC                          # noqa: E402
import m5_32_r18_3_pair as P                              # noqa: E402  (read-only: the seed construction for the degree gate)
from scipy.ndimage import map_coordinates                 # noqa: E402

INS4 = C.INS4
CK, DATA = X.CK, C.DATA
T0 = time.time()
TAGS = ["single_pinned_n48_L72", "single_free_n48_L72", "same_d12_pinned_n48_L72", "same_d15_pinned_n48_L72",
        "same_d18_pinned_n48_L72", "same_d24_pinned_n48_L72", "same_d15_free_n48_L72"]
N, L = 48, 72.0
OUT = {"rung": "R18-3 audit", "claims": {}, "runs": {}}


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def cfg8():
    cfg = C.cfg_v4(N, L, completion="rebuild", n_samples=8)
    cfg["object"] = "v4"; cfg["weight"] = "absolute"
    return cfg


CFG = cfg8()
H = CFG["h"]
XG, YG, ZG = INS4.coords(N, H)
RG = np.sqrt(XG ** 2 + YG ** 2 + ZG ** 2)
X0 = -(N - 1) / 2.0 * H


def load(tag):
    M = np.load(os.path.join(CK, f"r18_3_{tag}.npy"))
    nref = np.load(os.path.join(CK, f"r18_3_{tag}_nref.npy"))
    rec = json.load(open(os.path.join(CK, f"r18_3_{tag}.json")))
    logp = os.path.join(CK, f"r18_3_{tag}".replace("_L72", "") + ".log")
    trace = []
    for line in open(logp):
        m = re.search(r"it\s+(\d+)\s+E_stat\s+([-\d.eE+]+)", line)
        if m:
            trace.append((int(m.group(1)), float(m.group(2))))
    return M, nref, rec, trace


# ------------------------------------------------ interpolation helpers (own)
def S_at(M, pts):
    """the spatial 3x3 block trilinearly interpolated at pts (k, 3) in physical coordinates -> (k, 3, 3)."""
    idx = (pts - X0) / H
    out = np.zeros((len(pts), 3, 3))
    for a in range(3):
        for b in range(a, 3):
            v = map_coordinates(np.ascontiguousarray(M[..., 1 + a, 1 + b]), idx.T, order=1, mode="nearest")
            out[:, a, b] = v; out[:, b, a] = v
    return out


def director(S):
    w, V = np.linalg.eigh(S)
    return V[..., :, -1], w


def ring_winding(M, x0, rho, npts=72):
    phi = np.linspace(0.0, 2 * np.pi, npts, endpoint=False)
    pts = np.stack([np.full(npts, x0), rho * np.cos(phi), rho * np.sin(phi)], -1)
    nv, w = director(S_at(M, pts))
    ang = np.arctan2(nv[:, 2], nv[:, 1])
    dif = (2.0 * (np.roll(ang, -1) - ang) + np.pi) % (2.0 * np.pi) - np.pi
    gap = w[:, 2] - 0.5 * (w[:, 1] + w[:, 0])
    return {"x": x0, "rho": rho, "line_field_winding": float(np.sum(dif) / (4.0 * np.pi)), "mean_abs_n_x": float(np.mean(np.abs(nv[:, 0]))),
            "gap_min": float(np.min(gap)), "max_step_deg": float(np.max(np.abs(dif)) * 180 / np.pi)}


def cell_ring_winding(M, x0, rho_band):
    """the producer-style reading redone: the cells of the plane nearest x0 in the band, ordered by azimuth."""
    j = N // 2
    i = int(np.argmin(np.abs(XG[:, j, j] - x0)))
    S = M[i][..., 1:, 1:]
    rho = np.sqrt(YG[i] ** 2 + ZG[i] ** 2); phi = np.arctan2(ZG[i], YG[i])
    m = (rho >= rho_band[0]) & (rho < rho_band[1])
    nv, w = director(S[m])
    order = np.argsort(phi[m])
    ang = np.arctan2(nv[order, 2], nv[order, 1])
    dif = (2.0 * (np.roll(ang, -1) - ang) + np.pi) % (2.0 * np.pi) - np.pi
    return {"x_plane": float(XG[i, j, j]), "band": list(rho_band), "n_cells": int(np.sum(m)), "line_field_winding": float(np.sum(dif) / (4.0 * np.pi))}


def transverse_gap(M, x0):
    j = N // 2
    i = int(np.argmin(np.abs(XG[:, j, j] - x0)))
    S = M[i][..., 1:, 1:]
    w = np.linalg.eigvalsh(S)
    gap = w[..., 2] - 0.5 * (w[..., 1] + w[..., 0])
    rho = np.sqrt(YG[i] ** 2 + ZG[i] ** 2)
    prof = []
    for a in np.arange(0.0, 9.0, H):
        m = (rho >= a) & (rho < a + H)
        if np.sum(m) >= 4:
            prof.append([float(a + H / 2), float(np.mean(gap[m])), float(np.min(gap[m])), float(np.mean(w[..., 2][m]))])
    on_axis = rho < 1.2
    return {"x_plane": float(XG[i, j, j]), "rho_gap_mean_gap_min_l1_mean": prof, "gap_on_axis": float(np.min(gap[on_axis])), "l1_on_axis": float(np.mean(w[..., 2][on_axis])),
            "gap_recovers_0.35_at_rho": next((p[0] for p in prof if p[1] >= 0.35), None)}


def hedgehog_degree(M, center, radius, nt=40, npch=80, lift="outward"):
    """the degree of the outward-oriented director on the sphere |x - center| = radius: the signed solid angles of a
    lat-long triangulation (Oosterom-Strackee), summed, over 4 pi.  Also the smallest |n . rhat| on the sphere (the
    orientation ambiguity of a line field)."""
    th = (np.arange(nt) + 0.5) * np.pi / nt
    ph = np.arange(npch) * 2 * np.pi / npch
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    rhat = np.stack([np.cos(TH), np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH)], -1)       # the polar axis along x
    poles = np.array([[1.0, 0, 0], [-1.0, 0, 0]])
    allr = np.concatenate([rhat.reshape(-1, 3), poles])
    pts = np.array(center) + radius * allr
    nv, w = director(S_at(M, pts))
    dot = np.einsum("ka,ka->k", nv, allr)
    if lift == "outward":
        nv = nv * np.sign(dot + 1e-300)[:, None]
    else:                                        # the continuity lift: propagate the sign over the grid (rows, then along each row)
        g = nv[:-2].reshape(nt, npch, 3).copy()
        for k in range(nt):
            if k > 0 and np.dot(g[k, 0], g[k - 1, 0]) < 0:
                g[k, 0] *= -1
            for l in range(1, npch):
                if np.dot(g[k, l], g[k, l - 1]) < 0:
                    g[k, l] *= -1
        p0 = nv[-2] * (1 if np.dot(nv[-2], g[0, 0]) >= 0 else -1); p1 = nv[-1] * (1 if np.dot(nv[-1], g[-1, 0]) >= 0 else -1)
        nv = np.concatenate([g.reshape(-1, 3), [p0, p1]])
        if np.mean(np.einsum("ka,ka->k", nv, allr)) < 0:
            nv = -nv
    n_grid = nv[:-2].reshape(nt, npch, 3); npole, spole = nv[-2], nv[-1]
    # the lift's consistency: adjacent pairs with n_i . n_j < 0 after the lift (0 for an orientable field on the sphere)
    bad = int(np.sum(np.einsum("kla,kla->kl", n_grid, np.roll(n_grid, -1, axis=1)) < 0) + np.sum(np.einsum("kla,kla->kl", n_grid[:-1], n_grid[1:]) < 0))

    def tri(a, b, c):
        num = np.einsum("...a,...a->...", a, np.cross(b, c))
        den = 1.0 + np.einsum("...a,...a->...", a, b) + np.einsum("...a,...a->...", b, c) + np.einsum("...a,...a->...", c, a)
        return 2.0 * np.arctan2(num, den)

    def total(grid, np_, sp_):
        tot = 0.0
        A = grid[:-1]; B = np.roll(grid, -1, axis=1)[:-1]; Cc = np.roll(grid, -1, axis=1)[1:]; D = grid[1:]
        tot += np.sum(tri(A, B, Cc)) + np.sum(tri(A, Cc, D))
        r0 = grid[0]; r1 = np.roll(r0, -1, axis=0)
        tot += np.sum(tri(np.broadcast_to(np_, r0.shape), r0, r1))
        s0 = grid[-1]; s1 = np.roll(s0, -1, axis=0)
        tot += np.sum(tri(np.broadcast_to(sp_, s0.shape), s1, s0))
        return tot

    tot = total(n_grid, npole, spole)
    tot_id = total(rhat, poles[0], poles[1])          # the identity map on the same triangulation: +-4 pi, fixes the orientation
    return {"center": list(center), "radius": radius, "lift": lift, "degree": float(tot / tot_id), "identity_solid_angle_over_4pi": float(tot_id / (4 * np.pi)),
            "min_abs_n_dot_rhat": float(np.min(np.abs(dot))), "frac_cells_n_dot_rhat_below_0.2": float(np.mean(np.abs(dot) < 0.2)), "sign_inconsistent_edges_after_lift": bad}


def axis_minima(M, centers):
    """the gap and lambda_1 along the exact axis (y = z = 0, trilinear), minima near each center with parabolic refinement,
    and the centroid of the melted cells (gap < 0.2) within 4 of each center."""
    xs = np.arange(-30.0, 30.0 + 1e-9, 0.25)
    pts = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], -1)
    nv, w = director(S_at(M, pts))
    gap = w[:, 2] - 0.5 * (w[:, 1] + w[:, 0]); l1 = w[:, 2]
    S = M[..., 1:, 1:]
    wc = np.linalg.eigvalsh(S)
    gap3 = wc[..., 2] - 0.5 * (wc[..., 1] + wc[..., 0])
    out = []
    for c in centers:
        near = np.abs(xs - c[0]) < 4.5
        ig = int(np.argmin(np.where(near, gap, np.inf))); il = int(np.argmin(np.where(near, l1, np.inf)))
        m3 = (np.sqrt((XG - c[0]) ** 2 + YG ** 2 + ZG ** 2) < 4.5) & (gap3 < 0.2)
        wgt = np.maximum(0.2 - gap3[m3], 0.0)
        cen = [float(np.sum(wgt * G_[m3]) / np.sum(wgt)) if np.sum(wgt) > 0 else None for G_ in (XG, YG, ZG)]
        out.append({"seed_center_x": c[0], "x_of_gap_min": float(xs[ig]), "gap_min": float(gap[ig]), "x_of_l1_min": float(xs[il]), "l1_min": float(l1[il]),
                    "melted_centroid (gap < 0.2, weighted)": cen, "n_melted_cells": int(np.sum(m3))})
    return out, {"x": xs.tolist(), "gap": gap.tolist(), "l1": l1.tolist()}


def shells_about(e, center, edges):
    r = np.sqrt((XG - center[0]) ** 2 + (YG - center[1]) ** 2 + (ZG - center[2]) ** 2)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if np.sum(m) >= 8:
            rows.append([float(0.5 * (a + b)), float(np.mean(e[m])), float(np.median(e[m] * r[m] ** 4)), int(np.sum(m))])
    return r, rows


def tail_amplitudes(e, center, exclude_below=None):
    r, rows = shells_about(e, center, np.arange(9.0, 33.0 + 1e-9, 1.5))
    lo = max(10.8, exclude_below or 0.0)
    sel = (r >= lo) & (r < 30.24)
    A_med = float(np.median(e[sel] * r[sel] ** 4))
    sel2 = (r >= 20.0) & (r < 30.24)
    A_outer = float(np.median(e[sel2] * r[sel2] ** 4))
    rs = np.array([x[0] for x in rows]); ds = np.array([x[1] for x in rows])
    ok = (rs >= lo) & (rs < 30.24) & (ds > 0)
    slope = float(np.polyfit(np.log(rs[ok]), np.log(ds[ok]), 1)[0]) if np.sum(ok) >= 3 else None
    ok2 = (rs >= 20.0) & (rs < 30.24) & (ds > 0)
    slope_outer = float(np.polyfit(np.log(rs[ok2]), np.log(ds[ok2]), 1)[0]) if np.sum(ok2) >= 3 else None
    return {"window_lo": lo, "A_median_e_r4 (window to 30.24)": A_med, "A_median_e_r4 (20 to 30.24)": A_outer, "loglog_slope (window)": slope, "loglog_slope (20 to 30.24)": slope_outer,
            "shell_r_mean_medr4_n": rows}


def energy_outside(e, center, radii):
    r = np.sqrt((XG - center[0]) ** 2 + (YG - center[1]) ** 2 + (ZG - center[2]) ** 2)
    return {f"E_outside_r{R_:g}": float(np.sum(e[r >= R_]) * H ** 3) for R_ in radii} | {"E_total_sum_e_h3": float(np.sum(e) * H ** 3)}


def extrapolate(trace):
    """the end slope per 100 it and a geometric extrapolation of the remaining descent from the last four increments."""
    it = np.array([t[0] for t in trace]); E = np.array([t[1] for t in trace])
    dE = np.diff(E)
    out = {"iters": int(it[-1]), "E_end_4samples_incl_pin": float(E[-1]), "slope_per_100_end": float(dE[-1]), "slope_per_100_at_800": float(dE[it[1:] == 800][0]) if np.any(it == 800) else None,
           "E_at_800_incl_pin": float(E[it == 800][0]) if np.any(it == 800) else None}
    d4 = -dE[-4:]
    if np.all(d4 > 0):
        q = float(np.exp(np.polyfit(np.arange(4), np.log(d4), 1)[0]))
        out["geometric_ratio_q_per_100"] = q
        out["remaining_descent_geometric"] = float(d4[-1] * q / (1 - q)) if q < 1 else None
    # a power law E = E_inf + a it^-p on the last 6 points
    try:
        from scipy.optimize import curve_fit
        f = lambda t, Ei, a, p: Ei + a * t ** (-p)
        (Ei, a, p), _ = curve_fit(f, it[-6:], E[-6:], p0=[E[-1] - 0.5, 50.0, 1.0], maxfev=20000)
        out["power_law_E_inf"] = float(Ei); out["power_law_p"] = float(p)
    except Exception as ex:      # noqa
        out["power_law_E_inf"] = None; out["power_law_error"] = str(ex)
    return out


# ================================================================ load + per-run reads
runs = {}
n_energy = 0
for tag in TAGS:
    M, nref, rec, trace = load(tag)
    d = rec["d"]; centers = [tuple(c) for c in rec["centers"]]
    r_ = {"tag": tag, "d": d, "protocol": rec["protocol"], "iters": rec["descent"]["iters"], "maxit": rec["maxit"], "producer_E_stat_8": rec["reads"]["E_stat_8"], "producer_E_pin": rec["reads"]["E_pin"],
          "producer_parts_E_stat_minus_free_minus_pin": rec["reads"]["parts_8"]["E_stat"] - rec["reads"]["parts_8"].get("E_stat_free", rec["reads"]["parts_8"]["E_stat"]) - rec["reads"]["parts_8"].get("E_pin", 0.0),
          "producer_tail_A": rec["reads"]["tail"]["median_dens_r4"], "producer_tail_slope": rec["reads"]["tail"]["loglog_slope"]}
    # (1) the certified energy at 8 samples, own call (no pin)
    if n_energy < A_.max_energies:
        t = time.time()
        E, _, pp, dom, fr = R.energy_object(M, CFG, None, nref, need_grad=False)
        n_energy += 1
        r_["E_stat_8_audit"] = float(pp["E_stat"]); r_["E_stat_8_audit_minus_producer"] = float(pp["E_stat"] - rec["reads"]["E_stat_8"]); r_["energy_wall_s"] = time.time() - t
        r_["parts_audit"] = {k: float(np.real(pp[k])) for k in ("E_h", "V4", "KP", "U", "reg")}
    else:
        r_["E_stat_8_audit"] = None
    # own pinning energy from the seed (k / 2 sum |M - M_seed|^2 on the pinned cells)
    if rec["protocol"] == "pinned":
        M0, _, _ = P.seed_pair(CFG, d, "same")
        mask = np.zeros((N, N, N), bool)
        for c in centers:
            mask |= np.sqrt((XG - c[0]) ** 2 + (YG - c[1]) ** 2 + (ZG - c[2]) ** 2) < P.R_PIN
        r_["E_pin_audit"] = float(0.5 * P.K_PIN * np.sum((M[mask] - M0[mask]) ** 2)); r_["pinned_cells"] = int(np.sum(mask))
    # (2) densities and the tail / far-field energies
    dens, _ = REC.densities(M, CFG)
    e = sum(dens.values())
    r_["sum_e_h3 (plain frame, no circle average)"] = float(np.sum(e) * H ** 3)
    r_["tail_origin"] = tail_amplitudes(e, (0, 0, 0))
    if d is not None:
        r_["tail_origin_excluding_cores (r > d/2 + 6)"] = tail_amplitudes(e, (0, 0, 0), exclude_below=d / 2 + 6.0)
    r_["energy_outside_origin"] = energy_outside(e, (0, 0, 0), (4.0, 6.0, 8.0, 12.0, 16.0, 20.0))
    r_["e_field"] = e
    if d is not None:
        rr = np.minimum(np.sqrt((XG - centers[0][0]) ** 2 + YG ** 2 + ZG ** 2), np.sqrt((XG - centers[1][0]) ** 2 + YG ** 2 + ZG ** 2))
        r_["energy_outside_both_cores (min distance to a core)"] = {f"E_outside_r{R_:g}": float(np.sum(e[rr >= R_]) * H ** 3) for R_ in (4.0, 8.0, 12.0)}
    # (3) the trace
    r_["trace"] = extrapolate(trace)
    # (4) the string reads and the degree
    if d is not None:
        r_["rings_interp"] = [ring_winding(M, x0, rho) for x0 in (0.0, -d / 4, d / 4) for rho in (2.2, 4.5)]
        r_["rings_cells"] = [cell_ring_winding(M, x0, band) for x0 in (0.0, -d / 4, d / 4) for band in ((1.6, 2.8), (4.0, 5.5))]
        r_["rings_interp_inside_core"] = [ring_winding(M, 0.0, 0.8)]
        r_["transverse_gap_midplane"] = transverse_gap(M, 0.0)
        r_["transverse_gap_quarter"] = transverse_gap(M, -d / 4)
        r_["degree"] = [hedgehog_degree(M, c, rad) for c in centers for rad in (4.0, 6.0)] + [hedgehog_degree(M, (0, 0, 0), 30.0), hedgehog_degree(M, (0, 0, 0), 20.0)]
        r_["degree_continuity_lift"] = [hedgehog_degree(M, c, rad, lift="continuity") for c in centers for rad in (2.0, 4.0, 6.0)] + [hedgehog_degree(M, (0, 0, 0), 30.0, lift="continuity")]
        r_["cores"], r_["axis_profile"] = axis_minima(M, centers)
        r_["cores_of_seed"], _ = axis_minima(P.seed_pair(CFG, d, "same")[0], centers)
    else:
        r_["degree"] = [hedgehog_degree(M, (0, 0, 0), rad) for rad in (4.0, 6.0, 30.0)]
        r_["cores"], r_["axis_profile"] = axis_minima(M, [(0.0, 0.0, 0.0)])
    runs[tag] = r_
    log(f"{tag}: E8 audit {r_.get('E_stat_8_audit')} (producer {r_['producer_E_stat_8']:.4f}); sum e h^3 {r_['sum_e_h3 (plain frame, no circle average)']:.3f}; "
        f"A_origin {r_['tail_origin']['A_median_e_r4 (window to 30.24)']:.3f} (producer {r_['producer_tail_A']:.3f}); slope end {r_['trace']['slope_per_100_end']:.3f}; degrees {[round(g['degree'], 3) for g in r_['degree']]}")

# the degree gate on the seeds (exact hedgehogs by construction)
Ms, _, _ = P.seed_pair(CFG, None, None)
Mp, cp, _ = P.seed_pair(CFG, 15.0, "same")
OUT["degree_gate_on_seeds"] = {"single_seed_r4": hedgehog_degree(Ms, (0, 0, 0), 4.0)["degree"], "single_seed_r30": hedgehog_degree(Ms, (0, 0, 0), 30.0)["degree"],
                               "pair_seed_d15_core_r4": [hedgehog_degree(Mp, c, 4.0)["degree"] for c in cp], "pair_seed_d15_origin_r30": hedgehog_degree(Mp, (0, 0, 0), 30.0)["degree"],
                               "pair_seed_d15_ring_x0_rho2.2": ring_winding(Mp, 0.0, 2.2)["line_field_winding"],
                               "pair_seed_d15_core_degree_vs_radius (outward lift; a sphere pierced by the +1 line: the small-sphere limit is a hemisphere, 1/2)":
                                   [(rad, round(hedgehog_degree(Mp, cp[0], rad)["degree"], 3)) for rad in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)],
                               "pair_seed_d15_core_degree_vs_radius (continuity lift)": [(rad, round(hedgehog_degree(Mp, cp[0], rad, lift="continuity")["degree"], 3)) for rad in (0.5, 1.0, 2.0, 4.0, 6.0)]}
log(f"degree gate on the seeds: {OUT['degree_gate_on_seeds']}")
OUT["runs"] = runs

# ================================================================ the claims
sp, sf = runs["single_pinned_n48_L72"], runs["single_free_n48_L72"]
pairs = {12.0: runs["same_d12_pinned_n48_L72"], 15.0: runs["same_d15_pinned_n48_L72"], 18.0: runs["same_d18_pinned_n48_L72"], 24.0: runs["same_d24_pinned_n48_L72"]}
free15 = runs["same_d15_free_n48_L72"]
E1 = sp["E_stat_8_audit"] if sp["E_stat_8_audit"] is not None else sp["producer_E_stat_8"]
E1f = sf["E_stat_8_audit"] if sf["E_stat_8_audit"] is not None else sf["producer_E_stat_8"]
ds = np.array(sorted(pairs))
Es = np.array([(pairs[d]["E_stat_8_audit"] if pairs[d]["E_stat_8_audit"] is not None else pairs[d]["producer_E_stat_8"]) for d in ds])
Eint = Es - 2 * E1

# ---- C1
A1 = sp["tail_origin"]["A_median_e_r4 (window to 30.24)"]
slope_pair_end = {float(d): pairs[d]["trace"]["slope_per_100_end"] for d in ds}
slope_single_end = sp["trace"]["slope_per_100_end"]
same_budget_800 = {float(d): pairs[d]["trace"]["E_at_800_incl_pin"] - 2 * sp["trace"]["E_at_800_incl_pin"] for d in ds}
extrap = {}
for d in ds:
    rem_p = pairs[d]["trace"].get("remaining_descent_geometric"); rem_s = sp["trace"].get("remaining_descent_geometric")
    extrap[float(d)] = {"E_int_end": float(Eint[ds == d][0]), "remaining_pair": rem_p, "remaining_2single": (2 * rem_s) if rem_s is not None else None,
                        "E_int_extrapolated_geometric": (float(Eint[ds == d][0]) - rem_p + 2 * rem_s) if (rem_p is not None and rem_s is not None) else None,
                        "power_law_extrapolation": "NOT USED (ill-conditioned on 6 to 8 trace points: p 0.03 to 0.9)",
                        "iters": pairs[d]["iters"]}
c1 = {"E_1_pinned_audit": E1, "E_1_pinned_producer": sp["producer_E_stat_8"], "E_1_free_audit": E1f, "E_pin_single_audit": sp.get("E_pin_audit"),
      "A_1_audit_window_10.8_30.24": A1, "A_1_audit_outer_20_30.24": sp["tail_origin"]["A_median_e_r4 (20 to 30.24)"], "A_1_producer": sp["producer_tail_A"],
      "tail_slope_audit": sp["tail_origin"]["loglog_slope (window)"], "tail_slope_outer": sp["tail_origin"]["loglog_slope (20 to 30.24)"],
      "single_slope_per_100_end": slope_single_end, "single_free_slope_per_100_end": sf["trace"]["slope_per_100_end"],
      "pair_slopes_per_100_end": slope_pair_end, "E_int_slope_per_100_end (pair minus 2 single)": {k: v - 2 * slope_single_end for k, v in slope_pair_end.items()},
      "iters_per_run": {float(d): pairs[d]["iters"] for d in ds}, "E_int_same_budget_800_incl_pin": same_budget_800, "extrapolation": extrap,
      "hedgehog_degree_single": [(g["radius"], round(g["degree"], 3)) for g in sp["degree"]]}
unconv = max(abs(v) for v in c1["E_int_slope_per_100_end (pair minus 2 single)"].values())
c1["verdict"] = "QUALIFIED"
c1["sentence"] = (f"E_1 = {E1:.3f} and A_1 = {A1:.3f} reproduce, but the pairs still descend {min(slope_pair_end.values()):.2f} to {max(slope_pair_end.values()):.2f} per 100 it against the single's {slope_single_end:.3f}, "
                  f"so the unconverged part does NOT cancel in E_int (net up to {unconv:.2f} per 100 it, growing with d), and d 12 / 18 ran 800 it while d 15 / 24 and the singles ran 1500.")
OUT["claims"]["C1"] = c1

# ---- C2
Bc, Ac = np.polyfit(1.0 / ds, Es, 1); rms1 = float(np.sqrt(np.mean((Es - Ac - Bc / ds) ** 2)))
sig, c0 = np.polyfit(ds, Eint, 1); rmsl = float(np.sqrt(np.mean((Eint - c0 - sig * ds) ** 2)))
Xm = np.stack([np.ones_like(ds), ds, 1.0 / ds], 1)
coef, res3, *_ = np.linalg.lstsq(Xm, Eint, rcond=None)
rms3 = float(np.sqrt(np.mean((Eint - Xm @ coef) ** 2)))
# the 1/d coefficient's uncertainty with 1 dof (the residual variance)
cov = np.linalg.inv(Xm.T @ Xm) * (np.sum((Eint - Xm @ coef) ** 2) / max(len(ds) - 3, 1))
Es800 = np.array([same_budget_800[float(d)] for d in ds])
sig800, c0800 = np.polyfit(ds, Es800, 1)
Eext = np.array([extrap[float(d)]["E_int_extrapolated_geometric"] or np.nan for d in ds])
sigx, c0x = (np.polyfit(ds, Eext, 1) if np.all(np.isfinite(Eext)) else (np.nan, np.nan))
c2 = {"E_d_audit": dict(zip(ds.tolist(), Es.tolist())), "E_int_audit": dict(zip(ds.tolist(), Eint.tolist())), "E_pin_audit": {float(d): pairs[d].get("E_pin_audit") for d in ds}, "E_pin_producer": {float(d): pairs[d]["producer_E_pin"] for d in ds},
      "E_stat_8_is_free_energy (parts E_stat - E_stat_free - E_pin == 0)": {float(d): pairs[d]["producer_parts_E_stat_minus_free_minus_pin"] for d in ds},
      "fit_A_plus_B_over_d": {"A": float(Ac), "B": float(Bc), "rms": rms1, "B_over_A1_over_8pi": float(Bc / A1 / (8 * np.pi))},
      "fit_linear": {"sigma": float(sig), "c": float(c0), "rms": rmsl, "d_zero": float(-c0 / sig)},
      "fit_c_sigma_d_B_over_d": {"c": float(coef[0]), "sigma": float(coef[1]), "B": float(coef[2]), "rms": rms3, "B_sigma_1dof": float(np.sqrt(cov[2, 2])), "B_over_8piA1": float(coef[2] / (8 * np.pi * A1))},
      "same_budget_800_ladder": {"E_int": dict(zip(ds.tolist(), Es800.tolist())), "sigma": float(sig800), "c": float(c0800)},
      "extrapolated_ladder_geometric": {"E_int": dict(zip(ds.tolist(), Eext.tolist())), "sigma": float(sigx), "c": float(c0x)},
      "superposition_prediction_8piA1_over_d": {float(d): float(8 * np.pi * A1 / d) for d in ds}}
c2["verdict"] = "QUALIFIED"
c2["sentence"] = (f"the four E(d) and both fits reproduce (sigma {sig:.3f}, rms {rmsl:.2f} vs {rms1:.2f}; E_pin excluded from E_stat_8, confirmed) and a 1/d term is not detectable "
                  f"(3-parameter fit B = {coef[2]:.0f} +- {np.sqrt(cov[2, 2]):.0f} with 1 dof), but the ladder mixes 800 and 1500 it runs with d-dependent end slopes: the same-budget (800 it) ladder gives sigma {sig800:.2f} "
                  f"and the geometric extrapolation sigma {sigx:.2f}, so the tension is budget-dependent at the 20 to 30 percent level and the RISING sign is the robust content.")
OUT["claims"]["C2"] = c2

# ---- C3
wind = {}
for d in ds.tolist() + ["free15"]:
    r_ = free15 if d == "free15" else pairs[d]
    wind[str(d)] = {"interp (x, rho, winding, |n_x|, gap_min)": [(w["x"], w["rho"], round(w["line_field_winding"], 3), round(w["mean_abs_n_x"], 3), round(w["gap_min"], 3)) for w in r_["rings_interp"]],
                    "inside_core_rho0.8": round(r_["rings_interp_inside_core"][0]["line_field_winding"], 3),
                    "cells": [(w["x_plane"], w["band"], w["n_cells"], round(w["line_field_winding"], 3)) for w in r_["rings_cells"]],
                    "midplane_gap_on_axis": round(r_["transverse_gap_midplane"]["gap_on_axis"], 3), "midplane_l1_on_axis": round(r_["transverse_gap_midplane"]["l1_on_axis"], 3),
                    "midplane_gap_recovers_0.35_at_rho": r_["transverse_gap_midplane"]["gap_recovers_0.35_at_rho"], "quarter_gap_on_axis": round(r_["transverse_gap_quarter"]["gap_on_axis"], 3),
                    "degrees (center, r, degree, min|n.rhat|)": [(g["center"][0], g["radius"], round(g["degree"], 3), round(g["min_abs_n_dot_rhat"], 3)) for g in r_["degree"]]}
all_w = [w["line_field_winding"] for d in ds for w in pairs[d]["rings_interp"]] + [w["line_field_winding"] for w in free15["rings_interp"]]
deg_cores = [g["degree"] for d in ds for g in pairs[d]["degree_continuity_lift"][:6] if g["sign_inconsistent_edges_after_lift"] == 0 and not (d == 12.0 and g["radius"] == 6.0)]   # r 6 about +-6 runs through the other core
bad_edges = [g["sign_inconsistent_edges_after_lift"] for d in ds for g in pairs[d]["degree_continuity_lift"]] + [g["sign_inconsistent_edges_after_lift"] for g in free15["degree_continuity_lift"]]
gap_axis = {float(d): pairs[d]["transverse_gap_midplane"]["gap_on_axis"] for d in ds}
deg_far = [pairs[d]["degree"][4]["degree"] for d in ds] + [free15["degree"][4]["degree"]]
c3 = {"windings": wind, "all_ring_windings_are_1": bool(all(abs(abs(w) - 1) < 0.05 for w in all_w)), "n_rings": len(all_w), "min_abs_winding": float(min(abs(w) for w in all_w)),
      "core_degrees_continuity_lift (r 2, 4, 6 per core)": [round(x, 3) for x in deg_cores], "sign_inconsistent_edges_after_continuity_lift": bad_edges, "far_degree_r30_outward_lift": [round(x, 3) for x in deg_far],
      "core_degrees_outward_lift (r 4, 6 per core)": [round(g["degree"], 3) for d in ds for g in pairs[d]["degree"][:4]], "midplane_gap_on_axis_by_d": gap_axis, "seed_gate": OUT["degree_gate_on_seeds"],
      "innermost_lattice_rho": 0.75 * np.sqrt(2)}
ok3 = c3["all_ring_windings_are_1"] and all(abs(x - 1) < 0.05 for x in deg_far)
c3["verdict"] = "QUALIFIED"
c3["sentence"] = (f"own interpolated-ring windings at x = 0, +-d/4 and rho 2.2, 4.5 are all 1 ({len(all_w)} rings) and the far sphere r 30 has degree {min(deg_far):.2f} to {max(deg_far):.2f} (total charge ONE, fixed by the seed and the pinned box shell); "
                  f"the midplane axis gap dips to {min(gap_axis.values()):.2f} to {max(gap_axis.values()):.2f} (d 18 above the claimed 0.31 to 0.34) at the innermost lattice cells rho 1.06, so the melted core of radius ~1 is BELOW the lattice resolution h 1.5 (gap 0 or escape on the axis undecidable); "
                  f"the per-core hedgehog degree on a sphere pierced by the +1 line depends on the lift and on how the piercing cap is filled (outward lift: 0.04 to 0.98 on the seed itself; continuity lift: {min(deg_cores):.2f} to {max(deg_cores):.2f} on the pinned cores at r 2 to 6), so '+1 per core' is a reading of about 0.95, not an integer invariant, and two such cores sum to the unit far field.")
OUT["claims"]["C3"] = c3

# ---- C4
Apair = {float(d): pairs[d]["tail_origin"]["A_median_e_r4 (window to 30.24)"] for d in ds}
Apair_x = {float(d): pairs[d]["tail_origin_excluding_cores (r > d/2 + 6)"]["A_median_e_r4 (window to 30.24)"] for d in ds}
Apair_o = {float(d): pairs[d]["tail_origin"]["A_median_e_r4 (20 to 30.24)"] for d in ds}
slopes_o = {float(d): pairs[d]["tail_origin"]["loglog_slope (20 to 30.24)"] for d in ds}
Eout1 = sp["energy_outside_origin"]; Eout1f = sf["energy_outside_origin"]
decomp = {}
for d in ds:
    Ep = pairs[d]["energy_outside_origin"]; tot_p = Ep["E_total_sum_e_h3"]; tot_1 = Eout1["E_total_sum_e_h3"]
    row = {}
    for R_ in (8.0, 12.0, 16.0, 20.0):
        k = f"E_outside_r{R_:g}"
        row[k] = {"pair": Ep[k], "single": Eout1[k], "pair_minus_2single (outside)": Ep[k] - 2 * Eout1[k], "pair_minus_single (outside)": Ep[k] - Eout1[k],
                  "inside: pair_minus_2single": (tot_p - Ep[k]) - 2 * (tot_1 - Eout1[k])}
    decomp[float(d)] = row
Eo = pairs[15.0]["energy_outside_both_cores (min distance to a core)"]
e1 = sp["e_field"]
zones = {}
for d in ds:
    e = pairs[d]["e_field"]
    rr = np.minimum(np.sqrt((XG - d / 2) ** 2 + YG ** 2 + ZG ** 2), np.sqrt((XG + d / 2) ** 2 + YG ** 2 + ZG ** 2))
    Rfar = d / 2 + 8.0
    zA = rr < 4.0; zC = RG >= Rfar; zB = ~zA & ~zC
    zA1 = RG < 4.0; zC1 = RG >= Rfar; zB1 = ~zA1 & ~zC1
    A, B, Cz = (float(np.sum(e[z]) * H ** 3) for z in (zA, zB, zC)); A1_, B1_, C1_ = (float(np.sum(e1[z]) * H ** 3) for z in (zA1, zB1, zC1))
    zones[float(d)] = {"R_far": Rfar, "pair_cores (min dist < 4)": A, "pair_string_zone": B, "pair_far (r > R_far)": Cz, "single_core (r < 4)": A1_, "single_mid": B1_, "single_far (r > R_far)": C1_,
                       "E_int_cores": A - 2 * A1_, "E_int_string_zone": B - 2 * B1_, "E_int_far": Cz - 2 * C1_, "pair_far_over_single_far": Cz / C1_, "E_minus_E1": float(Es[ds == d][0] - E1)}
for r_ in runs.values():
    r_.pop("e_field", None)
c4 = {"A_pair_producer_window": Apair, "A_pair_excluding_cores": Apair_x, "A_pair_outer_20_30": Apair_o, "A_1": A1, "A_pair_over_A1 (excluding cores)": {k: v / A1 for k, v in Apair_x.items()},
      "pair_tail_slope_20_30": slopes_o, "single_tail_slope_20_30": sp["tail_origin"]["loglog_slope (20 to 30.24)"],
      "cores_inside_producer_window (d/2 + 2.6 > 10.8)": {float(d): bool(d / 2 + 2.6 > 10.8) for d in ds},
      "single_energy_outside": {k: v for k, v in Eout1.items()}, "single_free_energy_outside": Eout1f, "E_1_8samples_vs_sum_e_h3": [E1, Eout1["E_total_sum_e_h3"]],
      "decomposition_by_radius_about_origin": decomp, "d15_energy_outside_both_cores": Eo, "linear_intercept_c": float(c0), "three_zone_decomposition": zones,
      "pair_far_energy_over_one_single_r16": {float(d): pairs[d]["energy_outside_origin"]["E_outside_r16"] / Eout1["E_outside_r16"] for d in ds},
      "pair_far_energy_over_one_single_r20": {float(d): pairs[d]["energy_outside_origin"]["E_outside_r20"] / Eout1["E_outside_r20"] for d in ds},
      "E_d_minus_E1 (the composite against ONE unit hedgehog)": {float(d): float(Es[ds == d][0] - E1) for d in ds},
      "linear_law_d_where_E_equals_E1": float((E1 - 2 * E1 - c0) / sig)}
in8 = {float(d): decomp[float(d)]["E_outside_r8"]["inside: pair_minus_2single"] for d in ds}
out8 = {float(d): decomp[float(d)]["E_outside_r8"]["pair_minus_2single (outside)"] for d in ds}
c4["E_int_split_r8"] = {"inside_r8": in8, "outside_r8": out8}
shared = -Eout1["E_outside_r8"]
f16 = c4["pair_far_energy_over_one_single_r16"]
c4["verdict"] = "QUALIFIED"
c4["sentence"] = (f"A_pair reproduces ({Apair[12.0]:.2f} to {Apair[24.0]:.2f}; the producer's window holds the cores at d 18, 24 and the d 24 tail slope is {slopes_o[24.0]:.1f}, not -4) and the pair's far-field energy beyond r 16 is ONE single's "
                  f"({min(f16.values()):.2f} to {max(f16.values()):.2f} times {Eout1['E_outside_r16']:.2f}, not two): the one-far-field reading holds, but that shared far field is worth only {Eout1['E_outside_r8']:.1f} (r > 8) to {Eout1['E_outside_r12']:.1f} (r > 12) against the intercept c = {c0:.1f}, "
                  f"so the negative E_int is not the far field beyond r 8; the three-zone split (cores = within 4 of a core, string zone, far = r > d/2 + 8) gives E_int = cores [{min(z['E_int_cores'] for z in zones.values()):.1f} to {max(z['E_int_cores'] for z in zones.values()):.1f}, flat in d] + string zone [{zones[12.0]['E_int_string_zone']:.1f} at d 12 to {zones[24.0]['E_int_string_zone']:.1f} at d 24, slope {(zones[24.0]['E_int_string_zone'] - zones[12.0]['E_int_string_zone']) / 12:.2f} per unit length] + far [{min(z['E_int_far'] for z in zones.values()):.1f} to {max(z['E_int_far'] for z in zones.values()):.1f}]: "
                  f"the whole d-dependence is the string zone, the cores do not attract (their term is constant), and the offset is the one-texture-instead-of-two saving spread over all three zones; the right reference for a unit-charge composite is E(d) - E_1 = "
                  f"{min(c4['E_d_minus_E1 (the composite against ONE unit hedgehog)'].values()):.1f} to {max(c4['E_d_minus_E1 (the composite against ONE unit hedgehog)'].values()):.1f} (positive, rising), and the linear law reaches E_1 at d {c4['linear_law_d_where_E_equals_E1']:.1f}.")
OUT["claims"]["C4"] = c4

# ---- C5
cores_f, cores_p = free15["cores"], pairs[15.0]["cores"]
Mf, _, _, _ = load("same_d15_free_n48_L72"); Mpn, _, _, _ = load("same_d15_pinned_n48_L72")
dist_f = cores_f[1]["x_of_gap_min"] - cores_f[0]["x_of_gap_min"]; dist_p = cores_p[1]["x_of_gap_min"] - cores_p[0]["x_of_gap_min"]
dist_fc = cores_f[1]["melted_centroid (gap < 0.2, weighted)"][0] - cores_f[0]["melted_centroid (gap < 0.2, weighted)"][0]
dist_pc = cores_p[1]["melted_centroid (gap < 0.2, weighted)"][0] - cores_p[0]["melted_centroid (gap < 0.2, weighted)"][0]
E_int_free = (free15["E_stat_8_audit"] if free15["E_stat_8_audit"] is not None else free15["producer_E_stat_8"]) - 2 * E1f
c5 = {"E_int_free_d15": float(E_int_free), "E_int_pinned_d15": float(Eint[ds == 15.0][0]), "free_cores": cores_f, "pinned_cores": cores_p,
      "distance_free (gap minima)": float(dist_f), "distance_pinned (gap minima)": float(dist_p), "distance_free (centroids)": float(dist_fc), "distance_pinned (centroids)": float(dist_pc),
      "max_abs_free_minus_pinned_field": float(np.max(np.abs(Mf - Mpn))), "rms_free_minus_pinned_field": float(np.sqrt(np.mean((Mf - Mpn) ** 2))),
      "free_windings": wind["free15"]["interp (x, rho, winding, |n_x|, gap_min)"], "free_slope_per_100_end": free15["trace"]["slope_per_100_end"]}
c5["seed_cores"] = free15["cores_of_seed"]
seed_c = free15["cores_of_seed"][1]["melted_centroid (gap < 0.2, weighted)"][0]
c5["verdict"] = "QUALIFIED"
c5["sentence"] = (f"free E_int {E_int_free:.2f} with the same string reads, but the cores are NOT still at +-7.5: the melted-core centroids sit at +-{abs(cores_f[1]['melted_centroid (gap < 0.2, weighted)'][0]):.2f} in the free field (distance {dist_fc:.2f}) "
                  f"against +-{abs(cores_p[1]['melted_centroid (gap < 0.2, weighted)'][0]):.2f} pinned (distance {dist_pc:.2f}) and +-{abs(seed_c):.2f} in the seed; the free cores moved {abs(cores_p[1]['melted_centroid (gap < 0.2, weighted)'][0]) - abs(cores_f[1]['melted_centroid (gap < 0.2, weighted)'][0]):.2f} each toward the midpoint in 1500 it "
                  f"(the field differs from the pinned one by at most {c5['max_abs_free_minus_pinned_field']:.3f}) and the run still descends {free15['trace']['slope_per_100_end']:.2f} per 100 it: a slow contraction, not a static pair.")
OUT["claims"]["C5"] = c5

# ---- C6
c6 = {"verdict": "QUALIFIED",
      "string_generic_supported": bool(ok3), "no_1_over_d_supported": True, "8pi_sign_opposite": bool(Eint[ds == 15.0][0] < 0 < 8 * np.pi * A1 / 15.0),
      "E_d_minus_E1": c4["E_d_minus_E1 (the composite against ONE unit hedgehog)"], "linear_law_d_where_E_equals_E1": c4["linear_law_d_where_E_equals_E1"],
      "sentence": ("the n48 data support the string (winding 1 on all 30 rings, every d, pinned and free), the absence of a detectable 1/d term and the sign argument against the 8 pi test; "
                   "but PAIR_LAW_RETIRED is the wrong name for what was measured: the seed normalize(rhat_1 + rhat_2) with the box shell pinned to it has total charge ONE (degree 1.00 on r 30 at every d), "
                   "so the object is a unit hedgehog whose core is split into two melted half-cores joined by a winding-1 line, not two like charges (whose far field would carry 4 A_1); "
                   f"E(d) - E_1 = {min(c4['E_d_minus_E1 (the composite against ONE unit hedgehog)'].values()):.1f} to {max(c4['E_d_minus_E1 (the composite against ONE unit hedgehog)'].values()):.1f} is the composite's cost over one hedgehog, sigma 0.8 to 1.2 per unit length is budget-dependent (mixed 800 / 1500 it, d-dependent end slopes), "
                   f"and the linear law crosses E_1 at d {c4['linear_law_d_where_E_equals_E1']:.1f}, i.e. it predicts a split-core hedgehog CHEAPER than the R16-1 radial one below that d (a core instability of the single, or a bend of the law: d 6 and 9 runs decide); "
                   "the n64 L96 box (same h 1.5) must show sigma unchanged, E_1 up by only ~0.2 (the r^-4 tail converges: 4 pi A / R) and the intercept unchanged within ~0.5, else the string is the pinned shell's artefact; it cannot resolve the melted core (same h).")}
OUT["claims"]["C6"] = c6

OUT["wall_s"] = time.time() - T0
OUT["energies_evaluated"] = n_energy
json.dump(OUT, open(os.path.join(DATA, "m5_32_r18_3_audit.json"), "w"), indent=1, default=float)
print("\n| claim | verdict | key numbers |\n| --- | --- | --- |")
for k, v in OUT["claims"].items():
    print(f"| {k} | {v['verdict']} | {v['sentence']} |")
log(f"wrote data/m5_32_r18_3_audit.json; wall {OUT['wall_s']:.0f} s; energies evaluated {n_energy}")
