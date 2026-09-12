"""M5.32 R13-W, step W2: the slab with the wall at z = 0 and a rigid generator on
each side (the frozen packet, ledger 6.2, verbatim), the decisive slab-level read.

GEOMETRY     4 x 4 x n slab, vacuum d4 = diag(8, 1, 0.3, 0) pinned at the z ends,
             the ONE-CELL degenerate layer at z = 0 (W1's constrained wall,
             re-relaxed here under the projection), the z > 0 half rotated by the
             relative angle Delta q in the (2,3) plane (the clock plane).
FUNCTIONALS  (i)  E_stat(Delta q) over one full period (the orbit period is pi:
                  R(pi) d4 R(pi)^T = d4), scanned over [0, 2 pi];
             (ii) kin_wall = the omega^2 coefficient of the slab with A_0 = omega_- a0
                  for z < 0 and omega_+ a0 for z > 0, a0 = [G1, M] rigid on each side
                  (INS4.kin_of, cross-checked by the registry omega_decompose per
                  half-space); per unit area; L-exponent over L in {48, 72, 96} at
                  h = 1.5 and h-exponent over h in {1.5, 1.0, 0.75} at L = 48.
VERDICTS     pass: E_stat(Delta q) periodic (no secular growth) AND kin_wall/area
                   L-independent -> ESTABLISHED_KINEMATIC for the decoupling, W3 runs.
             fail: secular twist energy, or kin_wall/area growing with L ->
                   CANDIDATE_REFUTED for the wall convention on L_cert, W3 does not run.
W0 PREDICTIONS (m5_32_r13w_w0.py S2 to S5): (i) E_stat(Delta q) is CONSTANT (planar
             flatness: E_u = 0; V4 rotation-invariant), so the periodicity gate passes
             vacuously; (ii) kin_wall/area = 4 delta^4 / h per rotating flank (0.0216
             at h = 1.5, one flank), exactly L-independent (the bulk is exactly zero),
             and ~ 1/h (NOT a continuum quantity).
CONTROLS (post-W0, logged as deviations from the packet, not pre-registered):
             (iii) the RELEASED wall (a pure orientation jump, no degenerate cell):
                   kin_wall(Delta q) is a lattice discretization term (the continuum
                   smooth (2,3)-twist has zero inertia, S4); shown to vanish as the
                   jump is spread over w cells, against the NON-commuting (1,2)-plane
                   twist whose inertia ~ 1/w survives at zero static cost (S5).
Numerical representation: FIRE dt0 0.01, dt_max 0.1, 12000 iterations or fmax < 1e-6.

Out: ../data/m5_32_r13w_w2.json, ../plots/m5_32_r13w_w2.png
Run: /opt/anaconda3/envs/master312/bin/python3 m5_32_r13w_w2.py
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("m5_32_r13w_common", os.path.join(HERE, "m5_32_r13w_common.py"))
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
INS4 = C.INS4
OUT = os.path.join(C.DATA, "m5_32_r13w_w2.json")
PNG = os.path.join(C.PLOTS, "m5_32_r13w_w2.png")
NXY = 4
MAXIT, FTOL = 12000, 1e-6
DQS = np.linspace(0.0, 2 * np.pi, 25)


def make_slab(n, L):
    """W1's constrained wall re-relaxed under the projection (12000 it or f_tol)."""
    cfg = C.cfg_of(n, L)
    h = cfg["h"]
    vac = INS4.vac4(cfg)
    M = np.broadcast_to(vac, (NXY, NXY, n, 4, 4)).copy()
    kz = n // 2
    wall = np.zeros((NXY, NXY, n), bool); wall[:, :, kz] = True
    wc = max(1, int(np.ceil(1.6 / h)))
    free = np.ones((NXY, NXY, n), bool); free[:, :, :wc] = False; free[:, :, n - wc:] = False
    Mc, info = C.fire_proj(M, cfg, free, MAXIT, project=lambda X: C.degenerate_project(X, wall),
                           tag=f"w2_n{n}_L{L:g}", log_every=2000, f_tol=FTOL)
    return Mc, cfg, kz, info


def rotate_half(M, kz, dq):
    Rq = C.rot(C.G1, dq)
    out = M.copy()
    out[:, :, kz + 1:] = Rq @ M[:, :, kz + 1:] @ Rq.T
    return out


def half_masks(shape, kz):
    lo = np.zeros(shape, bool); lo[:, :, :kz] = True        # z < 0 (the wall cell excluded)
    hi = np.zeros(shape, bool); hi[:, :, kz + 1:] = True    # z > 0
    return lo, hi


def two_freq_kin(M, cfg, kz, om_lo, om_hi):
    """omega^2 coefficient read: kin_of with a0 = om_lo a0 on z<0, om_hi a0 on z>0
    (the wall cell's a0 is zero by construction); returns the E_kin at unit scale,
    i.e. sum_i <F_0i>^2 with A_0 = omega(z) a0."""
    lo, hi = half_masks(M.shape[:3], kz)
    a0 = C.a0_G1(M)
    a0w = a0 * (om_lo * lo + om_hi * hi)[..., None, None]
    return float(INS4.kin_of(M, a0w, cfg)), a0w


def wall_read(n, L):
    Mc, cfg, kz, info = make_slab(n, L)
    h = cfg["h"]; area = NXY * NXY * h * h
    lo, hi = half_masks(Mc.shape[:3], kz)
    # (i) E_stat(Delta q)
    scan = []
    for dq in DQS:
        Mq = rotate_half(Mc, kz, dq)
        e_u, e_v = INS4.e_parts(Mq, cfg)
        k_co, _ = two_freq_kin(Mq, cfg, kz, 1.0, 1.0)
        k_lo, _ = two_freq_kin(Mq, cfg, kz, 1.0, 0.0)
        k_hi, _ = two_freq_kin(Mq, cfg, kz, 0.0, 1.0)
        scan.append({"dq": float(dq), "E_u": float(e_u), "V4": float(e_v), "E_stat": float(e_u + e_v),
                     "kin_co_per_area": k_co / area, "kin_lo_per_area": k_lo / area, "kin_hi_per_area": k_hi / area})
    Es = np.array([r["E_stat"] for r in scan])
    # (ii) kin_wall at Delta q = 0: per half-space via the registry omega_decompose (cross-check)
    a0 = C.a0_G1(Mc)
    k_lo_reg = C.kin_registry(Mc, cfg, a0 * lo[..., None, None]) / area
    k_hi_reg = C.kin_registry(Mc, cfg, a0 * hi[..., None, None]) / area
    k_dens = C.kin_density(Mc, a0, cfg)[0, 0]          # one column, per cell
    rec = {"n": n, "L": L, "h": h, "area": area, "stop": info["stop"], "iters": info["iters"],
           "gap_at_wall": float(C.gap23(Mc)[0, 0, kz]), "V4_wall_cell_density": float(INS4.e_parts(Mc, cfg)[1] / (NXY * NXY * h ** 3)),
           "scan": scan,
           "E_stat_range_over_period": float(Es.max() - Es.min()), "E_stat_mean": float(Es.mean()),
           "secular_slope_E_vs_dq": float(np.polyfit(DQS, Es, 1)[0]),
           "kin_wall_per_area": {"lo_rotating": scan[0]["kin_lo_per_area"], "hi_rotating": scan[0]["kin_hi_per_area"],
                                 "co_rotating": scan[0]["kin_co_per_area"],
                                 "lo_registry": k_lo_reg, "hi_registry": k_hi_reg},
           "kin_column_profile": (k_dens / (h * h)).tolist(),
           "W0_prediction_4delta4_over_h": 4 * C.DELTA ** 4 / h}
    C.log(f"W2 n{n} L{L:g} h{h:.3g}: gap {rec['gap_at_wall']:.2e}; E_stat range over 2pi {rec['E_stat_range_over_period']:.3e} "
          f"(mean {rec['E_stat_mean']:.6e}); kin_wall/area lo {rec['kin_wall_per_area']['lo_rotating']:.6f} "
          f"hi {rec['kin_wall_per_area']['hi_rotating']:.6f} co {rec['kin_wall_per_area']['co_rotating']:.6f} "
          f"| registry lo {k_lo_reg:.6f} | W0 4d^4/h {rec['W0_prediction_4delta4_over_h']:.6f}")
    return rec


def released_controls(n=32, L=48.0):
    """(iii) orientation jumps/twists in the vacuum slab, no degenerate cell."""
    cfg = C.cfg_of(n, L)
    h = cfg["h"]; area = NXY * NXY * h * h
    vac = INS4.vac4(cfg)
    kz = n // 2
    out = {"jump_23_vs_dq": [], "twist_23_vs_w": [], "twist_12_vs_w": []}
    for dq in DQS:
        M = np.broadcast_to(vac, (NXY, NXY, n, 4, 4)).copy()
        M = rotate_half(M, kz - 1, dq)             # jump between cells kz-1 and kz
        k = float(INS4.kin_of(M, C.a0_G1(M), cfg))   # a0 on EVERY cell (audit H6: the half-mask read dropped cell kz-1)
        e_u, e_v = INS4.e_parts(M, cfg)
        out["jump_23_vs_dq"].append({"dq": float(dq), "E_stat": float(e_u + e_v), "kin_co_per_area": k / area})
    for Gm, key in ((C.G1, "twist_23_vs_w"), (C.G3, "twist_12_vs_w")):
        for wc in (1, 2, 4, 8):
            ps = np.zeros(n); k0 = kz - wc // 2
            ps[k0:k0 + wc] = np.linspace(0, 1.0, wc + 1)[1:]; ps[k0 + wc:] = 1.0
            Rn = C.rot(Gm, np.broadcast_to(ps[None, None, :], (NXY, NXY, n)))
            M = np.einsum("...ab,bc,...dc->...ad", Rn, vac, Rn)
            a0 = C.a0_G1(M)
            e_u, e_v = INS4.e_parts(M, cfg)
            k = float(INS4.kin_of(M, a0, cfg)) / area
            out[key].append({"w_cells": wc, "w": wc * h, "E_u": float(e_u), "V4": float(e_v), "kin_per_area": k,
                             "kin_per_area_times_w": k * wc * h})
    C.log("controls: (2,3) jump kin/area vs dq max %.5f; (2,3) twist kin*w %s; (1,2) twist kin*w %s"
          % (max(r["kin_co_per_area"] for r in out["jump_23_vs_dq"]),
             [round(r["kin_per_area_times_w"], 5) for r in out["twist_23_vs_w"]],
             [round(r["kin_per_area_times_w"], 5) for r in out["twist_12_vs_w"]]))
    return out


if __name__ == "__main__":
    res = {"h_ladder": [], "L_ladder": []}
    for n, L in ((32, 48.0), (48, 48.0), (64, 48.0)):
        res["h_ladder"].append(wall_read(n, L)); json.dump(res, open(OUT, "w"), indent=1)
    for n, L in ((48, 72.0), (64, 96.0)):
        res["L_ladder"].append(wall_read(n, L)); json.dump(res, open(OUT, "w"), indent=1)
    res["controls"] = released_controls()
    Ls = np.array([48.0] + [r["L"] for r in res["L_ladder"]])
    kL = np.array([res["h_ladder"][0]["kin_wall_per_area"]["lo_rotating"]] + [r["kin_wall_per_area"]["lo_rotating"] for r in res["L_ladder"]])
    hs = np.array([r["h"] for r in res["h_ladder"]])
    kh = np.array([r["kin_wall_per_area"]["lo_rotating"] for r in res["h_ladder"]])
    L_exp = float(np.polyfit(np.log(Ls), np.log(kL), 1)[0])
    h_exp = float(np.polyfit(np.log(hs), np.log(kh), 1)[0])
    periodic = all(r["E_stat_range_over_period"] < 1e-9 * max(1.0, abs(r["E_stat_mean"])) for r in res["h_ladder"] + res["L_ladder"])
    L_flat = abs(L_exp) < 0.1
    res["summary"] = {
        "E_stat_periodic_no_secular": periodic,
        "E_stat_range_max": max(r["E_stat_range_over_period"] for r in res["h_ladder"] + res["L_ladder"]),
        "kin_wall_per_area_vs_L_at_h1.5": dict(zip([str(v) for v in Ls], kL.tolist())),
        "kin_wall_per_area_vs_h_at_L48": dict(zip([str(v) for v in hs], kh.tolist())),
        "L_exponent": L_exp, "h_exponent": h_exp,
        "W0_prediction": {str(hh): 4 * C.DELTA ** 4 / hh for hh in hs},
        "registry_cross_check_rel_dev": max(abs(r["kin_wall_per_area"]["lo_registry"] / r["kin_wall_per_area"]["lo_rotating"] - 1.0)
                                            for r in res["h_ladder"] + res["L_ladder"]),
    }
    s = res["summary"]
    s["verdict"] = ("W2 PASS on the pre-registered gates (E_stat(Delta q) constant, kin_wall/area L-exponent %.3f); "
                    "W3 runs. Qualification (W0 S3/S4 + audit I6): BOTH gates are identities of any planar slab with a diagonal wall "
                    "(E_stat range 0 by planar flatness and V4 invariance; L-flatness because the bulk is exactly zero), so the pass is "
                    "unfalsifiable on this geometry and carries no evidential weight; kin_wall/area ~ h^%.2f is not a continuum quantity; "
                    "the decoupling is only testable where the wall meets a non-planar texture (W3)" % (L_exp, h_exp)) if (periodic and L_flat) else \
        "CANDIDATE_REFUTED on L_cert (secular twist or L-growing kin_wall); W3 does not run"
    C.log("SUMMARY " + json.dumps(s, indent=None))
    json.dump(res, open(OUT, "w"), indent=1)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    r0 = res["h_ladder"][0]
    ax[0].plot([r["dq"] for r in r0["scan"]], [r["E_stat"] - r0["E_stat_mean"] for r in r0["scan"]], "o-", label="E_stat - mean")
    ax[0].plot([r["dq"] for r in r0["scan"]], [r["kin_co_per_area"] for r in r0["scan"]], "s-", label="kin_wall/area (co-rotating)")
    ax[0].set_xlabel("Delta q"); ax[0].legend(); ax[0].set_title("n32 L48: interface energy and inertia vs angle")
    ax[1].loglog(hs, kh, "o-", label="kin_wall/area, one flank"); ax[1].loglog(hs, 4 * C.DELTA ** 4 / hs, "k--", label="W0: 4 delta^4/h")
    ax[1].set_xlabel("h"); ax[1].legend(); ax[1].set_title(f"h-exponent {h_exp:.2f}; L-exponent {L_exp:.3f}")
    c = res["controls"]
    ax[2].loglog([r["w"] for r in c["twist_23_vs_w"]], [max(r["kin_per_area"], 1e-18) for r in c["twist_23_vs_w"]], "o-", label="(2,3) twist (clock plane)")
    ax[2].loglog([r["w"] for r in c["twist_12_vs_w"]], [r["kin_per_area"] for r in c["twist_12_vs_w"]], "s-", label="(1,2) twist (non-commuting)")
    ax[2].set_xlabel("twist width w"); ax[2].set_ylabel("kin/area"); ax[2].legend(); ax[2].set_title("free twists: E_stat = 0 for both")
    fig.tight_layout(); fig.savefig(PNG, dpi=110)
    C.log(f"written {OUT} and {PNG}")
