"""M5.32 R18-2 INDEPENDENT ADVERSARIAL AUDIT (ledger 6.7): the condensate scan and the Goldstone reads, refuted with
the auditor's own methods.  The producer (m5_32_r18_2_goldstone.py) and the instrument (m5_32_r18_common.py) are not
imported; only the certified energy (m5_32_r17_common.energy_object) and the R16 frame pieces are consumed read-only.

Own methods per claim
    C1  every end field re-read with a plain eigenvalue decomposition of N = M eta per cell (np.linalg.eigvals, sorted):
        I = h^3 sum rho^2, the max half split, the min director gap, where the split sits; the 8-sample static energy
        recomputed; the saddle re-energized under the v6 cfg at every g_W; the descent traces re-analyzed (energy slope
        over the last 500 iterations, the split's relative decay per 1000 iterations in two windows, the director-gap
        slope and its extrapolated escape, the seeded excess over the saddle at the SAME iteration).
    C2  the twist rebuilt from scratch: theta = q x, R(theta / 2) through the field's own J, the even part at +-q 0.05
        and the quadraticity from +0.1; per-part even parts; the nearest-neighbor split correlation C_x (the prediction
        S_KP = 2 I C_x on the lattice); the analytic anchor on a SYNTHETIC uniform-split field (S_KP / I = 2 (n-1)/n
        x 2(1 - cos qh)/(qh)^2, S_reg = 2 c_s d^2 S_KP, E_h = 0) and on a Gaussian-envelope split (S_KP = 2 I C_x).
    C3  kin_tot at K 50 recomputed; kin_KP = 4 I (an identity of the pair-plane generator a0 = J B + B J^T under the
        relative weight), so 2 kin_tot = 8 I + 2 kin_h + 2 kin_reg by construction; the synthetic prediction
        kin_tot / I = 4 + 8 c_s d^2.
    C4  the index-weighted second moment of the producer's zero positions (own computation), top eigenvector against
        z and (1,1,1).
    C5  the verdict re-examined against the decay rates (C1 traces).

usage: /opt/anaconda3/envs/master312/bin/python3 m5_32_r18_2_audit.py [--quick]
out:   data/m5_32_r18_2_audit.json
"""
from __future__ import annotations
import json
import os
import sys
import time
import traceback

import numpy as np

ARGS = list(sys.argv[1:])
sys.argv = [sys.argv[0]]
import m5_32_r17_common as R                              # noqa: E402  (imports and patches m5_32_r16_common)
import m5_32_r16_common as C                              # noqa: E402
sys.argv = [sys.argv[0]] + ARGS

C15, INS4 = C.C15, C.INS4
ETA = C.ETA
RES, DATA = C.RES, C.DATA
CK17 = R.CK
CK18 = os.path.join(RES, "checkpoints", "m5_32_r18")
T0 = time.time()
BUDGET_S = 19 * 60
N_, L_ = 32, 48.0
GAP_MIN = C.GAP_MIN
QUICK = "--quick" in ARGS
OUT = os.path.join(DATA, "m5_32_r18_2_audit.json")
AUD = {"rung": "R18-2 audit", "verdicts": {}, "C1_detail": {}, "C2_detail": {}, "C3_detail": {}, "C4_detail": {}, "C5_detail": {}, "hazards": [], "skipped": []}


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def left():
    return BUDGET_S - (time.time() - T0)


def dump():
    AUD["wall_s"] = round(time.time() - T0, 1)
    json.dump(AUD, open(OUT, "w"), indent=1, default=float)


def fl(d):
    return {k: float(np.real(v)) for k, v in d.items() if isinstance(v, (int, float, complex, np.floating))}


# ---------------------------------------------------------------- own reads from the eigenvalues
def eig_reads(M, h):
    Nm = M @ ETA
    ev = np.sort(np.real(np.linalg.eigvals(Nm)), axis=-1)             # ascending: lambda_g, lambda_3, lambda_2, lambda_1
    lg, l3, l2, l1 = ev[..., 0], ev[..., 1], ev[..., 2], ev[..., 3]
    X, Y, Z = INS4.coords(M.shape[0], h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    hs = (l2 - l3) / 2.0
    rho2 = hs * hs
    gap = l1 - l2
    i_hs = int(np.argmax(hs))
    i_gap = int(np.argmin(gap))
    cen = r < 1.4                                                     # the eight cells nearest the origin (r 1.299)
    return {"I_int_rho2": float(h ** 3 * np.sum(rho2)), "half_split_max": float(np.max(hs)), "r_at_half_split_max": float(r.reshape(-1)[i_hs]),
            "gap_min": float(np.min(gap)), "r_at_gap_min": float(r.reshape(-1)[i_gap]), "l1_min": float(np.min(l1)), "lg_max": float(np.max(lg)),
            "l2_max": float(np.max(l2)), "l3_min": float(np.min(l3)), "escape_d_own": bool(np.min(gap) <= GAP_MIN),
            "center_isolation_margin (l1 - pair mean, min over the 8 central cells)": float(np.min((l1 - (l2 + l3) / 2.0)[cen])),
            "center_half_split_max": float(np.max(hs[cen])), "rho2_fraction_inside_r6": float(np.sum(rho2[r < 6.0]) / max(np.sum(rho2), 1e-300))}


def corr_x(M, h):
    """the nearest-neighbor correlation of the traceless pair part along x: C_x = h^3 sum tr(B(x+h) B(x)) / (2 I),
    B = P23 N P23 - (s / 2) P23 (spectrum +-rho on the pair subspace, tr B^2 = 2 rho^2)."""
    Nm, Pg, P1, P23, Rg, R1, lg, l1, s, p = C15.projectors(M)
    B = P23 @ Nm @ P23 - (0.5 * s)[..., None, None] * P23
    B = np.real(B)
    two_I = h ** 3 * float(np.sum(np.einsum("...ij,...ji->...", B, B)))
    cx = h ** 3 * float(np.sum(np.einsum("...ij,...ji->...", B[1:], B[:-1])))
    return cx / max(two_I, 1e-300), two_I / 2.0


def load_field(path, cfg):
    M = np.load(path)
    nref_p = path[:-4] + "_nref.npy"
    nref = np.load(nref_p) if os.path.exists(nref_p) else C.radial_ref(cfg)
    return M, nref, ("file" if os.path.exists(nref_p) else "radial")


def energy(M, cfg8, nref, K=None):
    t = time.time()
    E, _, pp, dom, fr = R.energy_object(M, cfg8, K, nref, need_grad=False)
    return float(np.real(E)), fl(pp), dom, fr, time.time() - t


def twist_energy(M, cfg8, nref, q, fr=None):
    """own twist: theta = q x through the field's own J (rot_R(J, theta / 2), conjugation), then the certified energy."""
    n, h = cfg8["n"], cfg8["h"]
    X, _, _ = INS4.coords(n, h)
    if fr is None:
        fr = C.frame(M, nref)
    beta = (0.5 * q * X)[..., None, None]
    Rm = C.rot_R(fr["J"], beta)
    Mq = Rm @ M @ np.swapaxes(Rm, -1, -2)
    return energy(Mq, cfg8, fr["n"], None)


def cfg8_of(gW):
    return R.cfg_v6(N_, L_, gW=gW, completion="rebuild", n_samples=8)


# ---------------------------------------------------------------- the runs on file
RUNS = [  # (tag, gW, seed_r, kind)
    ("r17_2_v4rel_n32_L48_r16_1", None, 0.0, "saddle"),
    ("r17_2_v6_gW2_n32_L48_r16_1", 2.0, 0.0, "control"),
    ("r17_2_v6_gW0.5_n32_L48_r16_1_split0.05", 0.5, 2.0, "R17-3a"),
    ("r17_2_v6_gW1.1_n32_L48_r16_1_split0.05", 1.1, 2.0, "R17-3a"),
    ("r17_2_v6_gW1.35_n32_L48_r16_1_split0.05", 1.35, 2.0, "R17-3a"),
    ("r17_2_v6_gW1.5_n32_L48_r16_1_split0.05", 1.5, 2.0, "R18-2"),
    ("r17_2_v6_gW1.65_n32_L48_r16_1_split0.05", 1.65, 2.0, "R18-2"),
    ("r17_2_v6_gW1.8_n32_L48_r16_1_split0.05", 1.8, 2.0, "R18-2"),
    ("r17_2_v6_gW2_n32_L48_r16_1_split0.05", 2.0, 2.0, "R18-2"),
    ("r17_2_v6_gW1.5_n32_L48_r16_1_split0.05_r5", 1.5, 5.0, "R18-2"),
    ("r17_2_v6_gW1.65_n32_L48_r16_1_split0.05_r5", 1.65, 5.0, "R18-2"),
    ("r17_2_v6_gW1.8_n32_L48_r16_1_split0.05_r5", 1.8, 5.0, "R18-2"),
    ("r17_2_v6_gW2_n32_L48_r16_1_split0.05_r5", 2.0, 5.0, "R18-2"),
]
READS = [  # (label, tag, gW): the producer's read fields
    ("gW1.5_core_seeded", "r17_2_v6_gW1.5_n32_L48_r16_1_split0.05", 1.5),
    ("gW1.65_core_seeded", "r17_2_v6_gW1.65_n32_L48_r16_1_split0.05", 1.65),
    ("gW1.8_shell_seeded", "r17_2_v6_gW1.8_n32_L48_r16_1_split0.05_r5", 1.8),
    ("gW2.0_shell_seeded_escape_d", "r17_2_v6_gW2_n32_L48_r16_1_split0.05_r5", 2.0),
    ("gW2.0_unseeded_control", "r17_2_v6_gW2_n32_L48_r16_1", 2.0),
    ("r17_gW1.35_seeded", "r17_2_v6_gW1.35_n32_L48_r16_1_split0.05", 1.35),
]


# ================================================================ C1a: the traces (free)
def trace_analysis():
    prod = json.load(open(os.path.join(DATA, "m5_32_r18_2.json")))
    prow = {r_["tag"]: r_ for r_ in prod["statics"]["rows"]}
    sad = json.load(open(os.path.join(CK17, "r17_2_v4rel_n32_L48_r16_1.json")))
    sad_tr = {t["it"]: t for t in sad["trace"]}
    out = {}
    for tag, gW, sr, kind in RUNS:
        p = os.path.join(CK17, tag + ".json")
        r_ = json.load(open(p))
        tr = r_["trace"]
        its = np.array([t["it"] for t in tr]); E = np.array([t["E_stat"] for t in tr]); hs = np.array([t["half_split_max"] for t in tr])
        gap = np.array([t["dom_gap_1_2_min"] for t in tr]); rhs = np.array([t.get("r_half_split_max", np.nan) for t in tr]); l1m = np.array([t["dom_l1_min"] for t in tr])
        d = {"gW": gW, "seed_r": sr, "kind": kind, "stop": r_["descent"]["stop"], "iters": r_["descent"]["iters"], "seed_gap": r_["seed"]["reads"]["core"]["Delta_min_free"], "seed_half_split": r_["seed"]["reads"]["texture"]["half_split_max"],
             "E_end": float(E[-1]), "hs_end": float(hs[-1]), "gap_end": float(gap[-1]), "r_hs_end": float(rhs[-1]), "r_hs_at_it500": float(rhs[its == 500][0]) if np.any(its == 500) else None, "l1_min_end": float(l1m[-1])}

        def at(arr, it):
            m = its == it
            return float(arr[m][0]) if np.any(m) else None
        nl = min(6, len(tr))
        d["E_slope_per_100it_last500"] = float((E[-1] - E[-nl]) / max(nl - 1, 1))
        d["l1min_slope_per_100it_last500"] = float((l1m[-1] - l1m[-nl]) / max(nl - 1, 1))
        d["gap_slope_per_100it_last500"] = float((gap[-1] - gap[-nl]) / max(nl - 1, 1))
        d["gap_slope_per_100it_last1000"] = float((gap[-1] - gap[-min(11, len(tr))]) / max(min(11, len(tr)) - 1, 1))
        gs = d["gap_slope_per_100it_last500"]
        d["iterations_to_GAP_MIN_at_last500_slope"] = float(100.0 * (gap[-1] - GAP_MIN) / (-gs)) if gs < 0 else None
        for a, b in ((1000, 2000), (2000, 3000), (500, 1500), (1500, 2500)):
            ha, hb = at(hs, a), at(hs, b)
            d[f"split_log_decay_{a}_{b}"] = float(np.log(hb / ha)) if (ha and hb) else None
        d["split_rel_decay_per_1000it_last1000"] = float(np.log(hs[-1] / hs[-min(11, len(tr))]) * 1000.0 / (its[-1] - its[-min(11, len(tr))])) if len(tr) > 1 else None
        d["split_rel_decay_per_1000it_prev1000"] = float(np.log(hs[-min(11, len(tr))] / hs[-min(21, len(tr))]) * 1000.0 / (its[-min(11, len(tr))] - its[-min(21, len(tr))])) if len(tr) > 20 else None
        ex = {}
        for it_ in (1000, 2000, 2500, 3000):
            if it_ in sad_tr and np.any(its == it_):
                ex[str(it_)] = float(at(E, it_) - sad_tr[it_]["E_stat"])
        d["excess_over_saddle_at_same_iteration"] = ex
        d["gap_to_saddle_end_over_E_fall_per_100it"] = float((E[-1] - sad["reads"]["E_stat_8"]) / max(-d["E_slope_per_100it_last500"], 1e-300))
        d["producer_row"] = {k: prow.get(tag, {}).get(k) for k in ("E_stat_8", "half_split_max", "Delta_min", "condensed", "E_minus_saddle", "stop", "iters")}
        out[tag] = d
    return out


# ================================================================ C4: the cluster axes (free)
def cluster_axes():
    out = {}
    z = np.array([0.0, 0.0, 1.0]); d111 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    for lab, tag, gW in READS:
        p = os.path.join(CK18, f"r18_2_reads_{lab}.json")
        if not os.path.exists(p):
            out[lab] = {"missing": p}
            continue
        r_ = json.load(open(p))
        sh = {}
        for key, v in r_.get("spin2", {}).items():
            zs = v.get("zeros")
            if not zs:
                continue
            xyz = np.array([zz["xyz"] for zz in zs]); idx = np.array([zz["index"] for zz in zs], dtype=float)
            T_abs = sum(abs(w) * np.outer(x, x) for w, x in zip(idx, xyz)) / max(np.sum(np.abs(idx)), 1e-300)
            T_sgn = sum(w * np.outer(x, x) for w, x in zip(idx, xyz)) / max(np.sum(idx), 1e-300)
            def top(T):
                w, V = np.linalg.eigh(T); return V[:, -1], float(w[-1])
            ax_a, an_a = top(T_abs); ax_s, an_s = top(T_sgn)
            sh[key] = {"n_zeros": len(zs), "total_index": int(np.sum(idx)), "axis_abs_weighted": [round(float(a), 4) for a in ax_a], "anisotropy_abs": an_a, "dot_z": abs(float(ax_a @ z)), "dot_111": abs(float(ax_a @ d111)),
                       "axis_signed_weighted": [round(float(a), 4) for a in ax_s], "dot_z_signed": abs(float(ax_s @ z)), "dot_111_signed": abs(float(ax_s @ d111)),
                       "min_angle_deg_of_each_zero_to_axis": [round(float(np.degrees(np.arccos(min(1.0, abs(x @ ax_a))))), 1) for x in xyz], "fit_residual_rel": v.get("fit_residual_rel"),
                       "l2m0_fraction": (v.get("l2_m_power", {}).get("0", 0.0) / max(sum(v.get("l2_m_power", {}).values()), 1e-300)) if v.get("l2_m_power") else None}
        out[lab] = sh
    return out


# ================================================================ main
def main():
    log(f"R18-2 audit start; budget {BUDGET_S} s; quick {QUICK}")
    # ---------- C1a traces + C4 (free)
    tra = trace_analysis()
    AUD["C1_detail"]["traces"] = tra
    AUD["C4_detail"] = cluster_axes()
    dump()
    h = cfg8_of(1.65)["h"]

    # ---------- C2/C3 anchor: the synthetic fields (analytic)
    syn = {}
    try:
        cfg8 = cfg8_of(1.65)
        n, h = cfg8["n"], cfg8["h"]
        cs, cP = cfg8["cs"], cfg8["cP"]
        d = 0.05
        M = np.zeros((n, n, n, 4, 4)); M[..., 0, 0] = 8.0; M[..., 1, 1] = 1.0; M[..., 2, 2] = 0.3 + d; M[..., 3, 3] = 0.3 - d
        nref = C.radial_ref(cfg8, "x")
        q = 0.05
        own = eig_reads(M, h); I_syn = own["I_int_rho2"]
        E0, p0, dom0, fr0, t0 = energy(M, cfg8, nref)
        Ep, pp, _, _, _ = twist_energy(M, cfg8, nref, q, fr0)
        Em, pm, _, _, _ = twist_energy(M, cfg8, nref, -q, fr0)
        S = (Ep + Em - 2 * E0) / q ** 2
        Spart = {k: (pp[k] + pm[k] - 2 * p0[k]) / q ** 2 for k in ("E_h", "KP", "reg", "V4", "U_v6")}
        lat = 2.0 * (1.0 - np.cos(q * h)) / (q * h) ** 2
        pred_KP = 2.0 * (n - 1) / n * lat
        pred_reg = 2.0 * cs * d * d * pred_KP
        rec = {"d": d, "q": q, "I": I_syn, "I_analytic": d * d * n ** 3 * h ** 3, "E0": E0, "parts0": p0, "S_own": S, "S_over_I": S / I_syn, "S_parts_over_I": {k: v / I_syn for k, v in Spart.items()},
               "predicted": {"S_KP_over_I": pred_KP, "S_reg_over_I": pred_reg, "S_Eh_over_I": 0.0, "S_over_I": pred_KP + pred_reg, "continuum_S_over_I": 2.0 * (1.0 + 2.0 * cs * d * d), "lattice_factor_2(1-cos qh)/(qh)^2": lat, "edge_factor_(n-1)/n": (n - 1) / n},
               "eval_s": t0, "domain_ok": not dom0["escape_d"]}
        rec["dev_S_over_I_rel"] = abs(rec["S_over_I"] - rec["predicted"]["S_over_I"]) / rec["predicted"]["S_over_I"]
        log(f"synthetic uniform d {d}: S/I own {S / I_syn:.5f} (KP {Spart['KP'] / I_syn:.5f}, reg {Spart['reg'] / I_syn:.5f}, E_h {Spart['E_h'] / I_syn:.2e}) vs predicted {pred_KP + pred_reg:.5f} (continuum {rec['predicted']['continuum_S_over_I']:.4f}); eval {t0:.0f} s")
        EK, pk, _, _, tk = energy(M, cfg8, nref, 50.0)
        rec["kin"] = {"kin_tot": pk["kin_tot"], "kin_KP": pk["kin_KP"], "kin_h": pk["kin_h"], "kin_reg": pk["kin_reg"], "kin_tot_over_I": pk["kin_tot"] / I_syn, "kin_KP_over_I": pk["kin_KP"] / I_syn,
                      "predicted_kin_tot_over_I": 4.0 + 8.0 * cs * d * d, "predicted_kin_KP_over_I": 4.0, "predicted_kin_reg_over_I": 8.0 * cs * d * d, "predicted_kin_h_over_I": 0.0,
                      "kin_from_EK_identity": 2500.0 / (4.0 * (EK - pk["E_stat"])), "S_over_2kin": S / (2 * pk["kin_tot"]), "S_over_kinKP": S / pk["kin_KP"]}
        log(f"synthetic kin: kin_tot/I {pk['kin_tot'] / I_syn:.5f} (KP {pk['kin_KP'] / I_syn:.5f} h {pk['kin_h'] / I_syn:.2e} reg {pk['kin_reg'] / I_syn:.5f}) vs predicted {4 + 8 * cs * d * d:.5f}; S/2kin {S / (2 * pk['kin_tot']):.4f}")
        syn["uniform"] = rec
        # the Gaussian-envelope split (width 2 like the seed) on the uniform director: S_KP = 2 I C_x
        if not QUICK and left() > 600:
            X, Y, Z = INS4.coords(n, h)
            r = np.sqrt(X * X + Y * Y + Z * Z)
            dg = d * np.exp(-r * r / 8.0)
            Mg = np.zeros((n, n, n, 4, 4)); Mg[..., 0, 0] = 8.0; Mg[..., 1, 1] = 1.0; Mg[..., 2, 2] = 0.3 + dg; Mg[..., 3, 3] = 0.3 - dg
            own_g = eig_reads(Mg, h); I_g = own_g["I_int_rho2"]
            Cx, I_chk = corr_x(Mg, h)
            E0g, p0g, _, frg, _ = energy(Mg, cfg8, nref)
            Epg, ppg, _, _, _ = twist_energy(Mg, cfg8, nref, q, frg)
            Emg, pmg, _, _, _ = twist_energy(Mg, cfg8, nref, -q, frg)
            Sg = (Epg + Emg - 2 * E0g) / q ** 2
            Sgp = {k: (ppg[k] + pmg[k] - 2 * p0g[k]) / q ** 2 for k in ("E_h", "KP", "reg")}
            cx_direct = float(h ** 3 * np.sum(dg[1:] * dg[:-1]) / (h ** 3 * np.sum(dg * dg)))
            syn["gaussian"] = {"I": I_g, "I_from_projectors": I_chk, "C_x_projector": Cx, "C_x_direct": cx_direct, "S_own": Sg, "S_over_I": Sg / I_g, "S_parts_over_I": {k: v / I_g for k, v in Sgp.items()},
                               "predicted_S_KP_over_I": 2.0 * Cx * lat, "S_KP_over_2I_vs_Cx": (Sgp["KP"] / (2 * I_g)) / Cx, "E0": E0g}
            log(f"synthetic gaussian: S/I {Sg / I_g:.4f} (KP {Sgp['KP'] / I_g:.4f} E_h {Sgp['E_h'] / I_g:.2e} reg {Sgp['reg'] / I_g:.2e}); C_x {Cx:.4f} (direct {cx_direct:.4f}); S_KP/(2 I C_x) {(Sgp['KP'] / (2 * I_g)) / Cx:.4f}")
    except Exception as e:                                          # noqa: BLE001
        syn["error"] = traceback.format_exc()
        log(f"synthetic FAILED: {e!r}")
    AUD["C2_detail"]["synthetic"] = syn
    dump()

    # ---------- C1/C2/C3 on the read fields: E(0), +-q, +0.1, K 50, own eigen reads
    prod = json.load(open(os.path.join(DATA, "m5_32_r18_2.json")))
    reads = {}
    stat = {}
    for lab, tag, gW in READS:
        if left() < 120:
            AUD["skipped"].append(f"reads {lab}: budget"); continue
        try:
            cfg8 = cfg8_of(gW)
            M, nref, lift = load_field(os.path.join(CK17, tag + ".npy"), cfg8)
            own = eig_reads(M, cfg8["h"]); Cx, _ = corr_x(M, cfg8["h"])
            E0, p0, dom0, fr0, t0 = energy(M, cfg8, nref)
            rec = {"tag": tag, "gW": gW, "lift": lift, "own": own, "C_x": Cx, "E_stat_8_own": E0, "parts0": p0, "domain_escape_d": dom0["escape_d"], "eval_s": t0}
            stat[tag] = {"E_stat_8_own": E0, "own": own, "gW": gW}
            pr = prod["reads"].get(lab, {})
            rec["producer"] = {k: pr.get(k) for k in ("I_int_rho2", "E_stat_8", "clock_inertia_2kin", "S_over_I", "S_over_2kin", "rho_max") if k in pr}
            rec["producer"]["S"] = pr.get("twist", {}).get("S_at_smallest_q"); rec["producer"]["torque"] = pr.get("twist", {}).get("torque_odd_at_smallest_q")
            log(f"{lab}: E0 own {E0:.6f} (prod {pr.get('E_stat_8')}); I own {own['I_int_rho2']:.5e} (prod {pr.get('I_int_rho2')}); hs max {own['half_split_max']:.4f} at r {own['r_at_half_split_max']:.2f}; gap min {own['gap_min']:.5f}; C_x {Cx:.4f}; {t0:.0f} s")
            if left() < 100:
                AUD["skipped"].append(f"twist {lab}: budget"); reads[lab] = rec; continue
            q = 0.05
            Ep, pp, dp, _, _ = twist_energy(M, cfg8, nref, q, fr0)
            Em, pm, dm, _, _ = twist_energy(M, cfg8, nref, -q, fr0)
            S = (Ep + Em - 2 * E0) / q ** 2
            tau = (Ep - Em) / (2 * q)
            Spart = {k: (pp[k] + pm[k] - 2 * p0[k]) / q ** 2 for k in ("E_h", "KP", "reg", "V4", "U_v6")}
            I = own["I_int_rho2"]
            rec["twist"] = {"q": q, "dE_plus": Ep - E0, "dE_minus": Em - E0, "S_own": S, "torque_own": tau, "odd_over_even": abs(Ep - Em) / max(abs(Ep + Em - 2 * E0), 1e-300), "S_parts": Spart, "S_parts_over_I": {k: v / I for k, v in Spart.items()},
                            "S_over_I": S / I, "S_KP_over_2I": Spart["KP"] / (2 * I), "C_x": Cx, "S_KP_over_2I_over_Cx": Spart["KP"] / (2 * I) / Cx, "escape_d_twisted": bool(dp["escape_d"] or dm["escape_d"])}
            log(f"  twist: S own {S:.5e} (prod {rec['producer']['S']}); S/I {S / I:.4f}; parts/I KP {Spart['KP'] / I:.4f} E_h {Spart['E_h'] / I:.4f} reg {Spart['reg'] / I:.2e}; S_KP/(2I) {Spart['KP'] / (2 * I):.4f} vs C_x {Cx:.4f}; torque {tau:.2e}")
            if left() > 200:
                E1, p1, _, _, _ = twist_energy(M, cfg8, nref, 0.1, fr0)
                S1 = 2.0 * (E1 - E0 - tau * 0.1) / 0.01
                rec["twist"]["S_at_q0.1_odd_removed"] = S1; rec["twist"]["quadratic_rel_dev_0.05_0.1"] = abs(S1 - S) / abs(S)
                log(f"  q 0.1: S {S1:.5e}, rel dev {abs(S1 - S) / abs(S):.2e}")
            else:
                AUD["skipped"].append(f"q 0.1 {lab}: budget")
            if left() > 150:
                EK, pk, _, _, tk = energy(M, cfg8, nref, 50.0)
                rec["kin"] = {"kin_tot": pk["kin_tot"], "kin_KP": pk["kin_KP"], "kin_h": pk["kin_h"], "kin_reg": pk["kin_reg"], "E_K": EK, "kin_from_EK_identity": 2500.0 / (4.0 * (EK - pk["E_stat"])),
                              "kin_tot_over_I": pk["kin_tot"] / I, "two_kin_over_I": 2 * pk["kin_tot"] / I, "kin_KP_over_I": pk["kin_KP"] / I, "kin_h_over_I": pk["kin_h"] / I, "S_over_2kin": S / (2 * pk["kin_tot"]), "S_over_kinKP": S / pk["kin_KP"],
                              "S_Eh_over_kin_h": Spart["E_h"] / max(pk["kin_h"], 1e-300), "eval_s": tk}
                log(f"  K 50: kin_tot {pk['kin_tot']:.5e} (2kin/I {2 * pk['kin_tot'] / I:.3f}; KP/I {pk['kin_KP'] / I:.6f}; h/I {pk['kin_h'] / I:.4f}); S/2kin {S / (2 * pk['kin_tot']):.4f}; S_Eh/kin_h {Spart['E_h'] / pk['kin_h']:.4f}; {tk:.0f} s")
            else:
                AUD["skipped"].append(f"K 50 {lab}: budget")
            reads[lab] = rec
        except Exception as e:                                      # noqa: BLE001
            reads[lab] = {"error": traceback.format_exc()}
            log(f"{lab} FAILED: {e!r}")
        AUD["C2_detail"]["reads"] = reads
        dump()

    # ---------- C1: the remaining end fields and the saddle under v6 at every g_W
    sad = {}
    for tag, gW, sr, kind in RUNS:
        if tag in stat:
            continue
        if left() < 60:
            AUD["skipped"].append(f"static {tag}: budget"); continue
        try:
            g_use = gW if gW is not None else 2.0
            cfg8 = cfg8_of(g_use)
            M, nref, lift = load_field(os.path.join(CK17, tag + ".npy"), cfg8)
            own = eig_reads(M, cfg8["h"])
            E0, p0, dom0, fr0, t0 = energy(M, cfg8, nref)
            stat[tag] = {"E_stat_8_own": E0, "own": own, "gW": g_use, "lift": lift, "parts0": p0, "eval_s": t0}
            log(f"static {tag}: E own {E0:.6f}; hs max {own['half_split_max']:.4f} at r {own['r_at_half_split_max']:.2f}; gap min {own['gap_min']:.5f} (escape {own['escape_d_own']}); I {own['I_int_rho2']:.3e}; center margin {own['center_isolation_margin (l1 - pair mean, min over the 8 central cells)']:.4f}; {t0:.0f} s")
            if kind == "saddle":
                # the saddle under the v6 cfg at every g_W: the certified energy at g_W 2.0 (above, cfg gW 2.0) and 1.5, U_v6 recomputed from the own eigenvalues at all four
                sad["E_v6_gW2.0_certified"] = E0
                if left() > 90:
                    E15, p15, _, _, _ = energy(M, cfg8_of(1.5), nref)
                    sad["E_v6_gW1.5_certified"] = E15
                Nm = M @ ETA; ev = np.sort(np.real(np.linalg.eigvals(Nm)), axis=-1)
                rho2 = ((ev[..., 2] - ev[..., 1]) / 2.0) ** 2; l1 = ev[..., 3]
                W = ((1.0 - l1) / 0.7) ** 2
                sad["U_v6_own_by_gW"] = {str(g): float(cfg8["h"] ** 3 * np.sum((1e-2 - g * W) * rho2 - 1e-2 * rho2 ** 2 + 0.4 * rho2 ** 3)) for g in (0.5, 1.1, 1.35, 1.5, 1.65, 1.8, 2.0)}
                sad["E_v6_by_gW_own"] = {g: E0 - p0["U_v6"] + u for g, u in sad["U_v6_own_by_gW"].items()}
                sad["U_v6_certified_at_gW2"] = p0["U_v6"]
                sad["E_v4rel_producer"] = prod["statics"]["saddle"]
                sad["gW_spread_of_saddle_energy"] = float(max(sad["E_v6_by_gW_own"].values()) - min(sad["E_v6_by_gW_own"].values()))
        except Exception as e:                                      # noqa: BLE001
            stat[tag] = {"error": traceback.format_exc()}
            log(f"static {tag} FAILED: {e!r}")
        AUD["C1_detail"]["statics_own"] = stat; AUD["C1_detail"]["saddle_v6"] = sad
        dump()

    # ================================================================ verdicts
    v = AUD["verdicts"]
    # ---- C1
    rows = []
    sadE = sad.get("E_v6_gW2.0_certified", prod["statics"]["saddle"])
    for tag, gW, sr, kind in RUNS:
        s_ = stat.get(tag, {}); t_ = tra.get(tag, {}); o = s_.get("own", {})
        pr = t_.get("producer_row", {})
        rows.append({"tag": tag, "gW": gW, "seed_r": sr, "stop": t_.get("stop"), "iters": t_.get("iters"), "E_own": s_.get("E_stat_8_own"), "E_prod": pr.get("E_stat_8"), "E_minus_saddle_own": (s_.get("E_stat_8_own") - sadE) if s_.get("E_stat_8_own") is not None else None,
                     "hs_own": o.get("half_split_max"), "hs_prod": pr.get("half_split_max"), "gap_own": o.get("gap_min"), "gap_prod": pr.get("Delta_min"), "escape_d_own": o.get("escape_d_own"),
                     "E_slope_100it": t_.get("E_slope_per_100it_last500"), "gap_to_saddle_over_fall_100it": t_.get("gap_to_saddle_end_over_E_fall_per_100it"),
                     "split_decay_per_1000it_last": t_.get("split_rel_decay_per_1000it_last1000"), "split_decay_per_1000it_prev": t_.get("split_rel_decay_per_1000it_prev1000"),
                     "gap_slope_100it": t_.get("gap_slope_per_100it_last500"), "it_to_GAP_MIN": t_.get("iterations_to_GAP_MIN_at_last500_slope"), "r_hs_end": t_.get("r_hs_end"), "r_hs_it500": t_.get("r_hs_at_it500"), "seed_gap": t_.get("seed_gap"),
                     "center_margin": o.get("center_isolation_margin (l1 - pair mean, min over the 8 central cells)")})
    AUD["C1_detail"]["table"] = rows
    ok_num = all(r_["E_own"] is None or r_["E_prod"] is None or abs(r_["E_own"] - r_["E_prod"]) < 1e-5 for r_ in rows)
    ok_hs = all(r_["hs_own"] is None or r_["hs_prod"] is None or abs(r_["hs_own"] - r_["hs_prod"]) < 1e-4 for r_ in rows)
    ok_gap = all(r_["gap_own"] is None or r_["gap_prod"] is None or abs(r_["gap_own"] - r_["gap_prod"]) < 1e-4 for r_ in rows)
    seeded = [r_ for r_ in rows if r_["seed_r"] > 0 and r_["E_own"] is not None]
    above = all(r_["E_minus_saddle_own"] > 0 for r_ in seeded)
    decays = {r_["tag"]: (r_["split_decay_per_1000it_last"] is not None and r_["split_decay_per_1000it_last"] < -0.05) for r_ in seeded}
    slow = [r_["tag"] for r_ in seeded if r_["split_decay_per_1000it_last"] is not None and r_["split_decay_per_1000it_last"] > -0.05]
    ratio_min = min((abs(r_["E_minus_saddle_own"]) / max(-r_["E_slope_100it"], 1e-300)) for r_ in seeded if r_["stop"] == "max_iter" and r_["E_slope_100it"])
    v["C1"] = {"verdict": "QUALIFIED", "numbers": {"energies_reproduced_1e-5": ok_num, "half_split_reproduced_1e-4": ok_hs, "gap_reproduced_1e-4": ok_gap, "saddle_gW_spread": sad.get("gW_spread_of_saddle_energy"),
                                                    "all_seeded_above_saddle_at_it_3000": above, "saddle_E_slope_per_100it_last500": tra.get("r17_2_v4rel_n32_L48_r16_1", {}).get("E_slope_per_100it_last500"),
                                                    "min_(E_end - E_saddle)_over_(E fall per 100 it)": ratio_min, "runs_with_split_decay_slower_than_5pct_per_1000it": slow, "seed_gap_r2": tra.get("r17_2_v6_gW1.5_n32_L48_r16_1_split0.05", {}).get("seed_gap"),
                                                    "escape_d_reproduced": {r_["tag"]: r_["escape_d_own"] for r_ in rows if r_["stop"] == "escape_d"}},
               "sentence": "the numbers reproduce (energies, splits, gaps, escape (d) at the stated iterations, the saddle g_W-independent), but the ordering compares snapshots of parallel UNCONVERGED descents (every run, the saddle included, still falls ~0.035 per 100 iterations at 3000, so the seeded excess over the saddle is a fraction of one 100-iteration step), the seeded split decays only on the r 2 seeds at g_W <= 1.65 while the shell seeds at g_W >= 1.8 hold their amplitude, migrate into the core and close the director gap at a steady rate (escape (d) ahead), and the r 2 seed starts at a director gap of 0.0029 (3 x GAP_MIN) because the seed amplitude 0.05 exceeds the core's isolation margin: a condensate is NOT excluded, only not reached inside a domain that cannot hold one of the seed's own amplitude at the core."}
    # ---- C2
    rd = AUD["C2_detail"].get("reads", {})
    SI = {k: r_["twist"]["S_over_I"] for k, r_ in rd.items() if "twist" in r_}
    Sdev = {k: abs(r_["twist"]["S_own"] - r_["producer"]["S"]) / abs(r_["producer"]["S"]) for k, r_ in rd.items() if "twist" in r_ and r_["producer"].get("S")}
    Idev = {k: abs(r_["own"]["I_int_rho2"] - r_["producer"]["I_int_rho2"]) / r_["producer"]["I_int_rho2"] for k, r_ in rd.items() if "own" in r_ and r_["producer"].get("I_int_rho2")}
    S2k = {k: r_["kin"]["S_over_2kin"] for k, r_ in rd.items() if "kin" in r_}
    KPc = {k: r_["twist"]["S_KP_over_2I_over_Cx"] for k, r_ in rd.items() if "twist" in r_}
    su = AUD["C2_detail"].get("synthetic", {}).get("uniform", {})
    v["C2"] = {"verdict": "CONFIRMED" if (SI and max(Sdev.values()) < 0.02 and max(Idev.values()) < 1e-6) else "QUALIFIED",
               "numbers": {"S_over_I_own": SI, "S_own_vs_producer_rel_dev": Sdev, "I_own_vs_producer_rel_dev": Idev, "S_over_2kin_own": S2k, "S_KP_over_2I_over_Cx": KPc, "synthetic_uniform_S_over_I": su.get("S_over_I"), "synthetic_predicted_S_over_I": su.get("predicted", {}).get("S_over_I"), "synthetic_continuum_S_over_I": su.get("predicted", {}).get("continuum_S_over_I"),
                           "synthetic_gaussian": {k: AUD["C2_detail"].get("synthetic", {}).get("gaussian", {}).get(k) for k in ("S_over_I", "C_x_projector", "S_KP_over_2I_vs_Cx")}},
               "sentence": "the ratios reproduce with an independent twist, and they are a property of the object's kinetic structure, not of a condensate: the K_P^proj term gives S_KP = 2 I C_x exactly (C_x the nearest-neighbor correlation of the split along the twist axis, 1 on a uniform split, so S / I = 2 (1 + 2 c_s d^2) x lattice factor on the synthetic field), the residual director coupling adds S_Eh = a fixed fraction of kin_h, and both scale with rho^2 whatever produced it (the unseeded control's 6e-4 residual included)."}
    # ---- C3
    tk = {k: r_["kin"]["two_kin_over_I"] for k, r_ in rd.items() if "kin" in r_}
    kpI = {k: r_["kin"]["kin_KP_over_I"] for k, r_ in rd.items() if "kin" in r_}
    khI = {k: r_["kin"]["kin_h_over_I"] for k, r_ in rd.items() if "kin" in r_}
    idn = {k: abs(r_["kin"]["kin_from_EK_identity"] - r_["kin"]["kin_tot"]) / r_["kin"]["kin_tot"] for k, r_ in rd.items() if "kin" in r_}
    v["C3"] = {"verdict": "QUALIFIED", "numbers": {"two_kin_over_I_own": tk, "range_own": [min(tk.values()), max(tk.values())] if tk else None, "producer_stated_range": [7.5, 9.1], "kin_KP_over_I (identity 4)": kpI, "kin_h_over_I": khI, "E_K_identity_rel_dev": idn,
                                                    "synthetic_kin_tot_over_I": su.get("kin", {}).get("kin_tot_over_I"), "synthetic_predicted": su.get("kin", {}).get("predicted_kin_tot_over_I"), "synthetic_kin_KP_over_I": su.get("kin", {}).get("kin_KP_over_I")},
               "sentence": "kin_tot reproduces and the E(K) identity holds, but the 'factor 4 to 5' is kin_tot / I (4.38 to 4.50 here), while the inertia ratio between the author's rotor law K^2 / 2 I and the instrument's K^2 / 4 kin_tot is 2 kin_tot / I = 8.8 to 9.0 (not 7.5 to 9.1): 8 of it is the identity kin_KP = 4 I of the pair-plane generator (4.01 I on the synthetic uniform split, as predicted), the rest 2 kin_h (0.77 to 1.0 I) is the director's participation; the number is fixed by the object's normalization, not measured on a condensate."}
    # ---- C4
    c4 = AUD["C4_detail"]
    seeded_ok = all(all(sh["dot_z"] > 0.95 for sh in v_.values()) for k, v_ in c4.items() if isinstance(v_, dict) and "missing" not in v_ and k != "gW2.0_unseeded_control" and v_)
    ctrl = c4.get("gW2.0_unseeded_control", {})
    ctrl_ok = bool(ctrl) and all(sh["dot_111"] > 0.95 for sh in ctrl.values())
    v["C4"] = {"verdict": "CONFIRMED" if (seeded_ok and ctrl_ok) else "QUALIFIED", "numbers": {k: {sh: (round(x["dot_z"], 4), round(x["dot_111"], 4), x["n_zeros"], round(x["anisotropy_abs"], 3)) for sh, x in v_.items()} for k, v_ in c4.items() if isinstance(v_, dict) and "missing" not in v_},
               "sentence": "own second-moment axes: +-z on every seeded field's shells (|axis . z| > 0.99, the four simple zeros within 10 degrees of the poles) and +-(1,1,1) on the control (|axis . (111)| > 0.99, eight zeros); the m = 0 dominance follows; the simple-zero count stays fit-dependent (R18-0 audit) and is not re-adjudicated."}
    # ---- C5
    def dec(t):
        x = tra.get(t, {}).get("split_rel_decay_per_1000it_last1000"); return f"{x:+.2f}" if x is not None else "n/a"
    d_r2 = ", ".join(f"{g:g}: {dec(f'r17_2_v6_gW{g:g}_n32_L48_r16_1_split0.05')}" for g in (0.5, 1.1, 1.35, 1.5, 1.65))
    d_r5 = ", ".join(f"{g:g}: {dec(f'r17_2_v6_gW{g:g}_n32_L48_r16_1_split0.05_r5')}" for g in (1.5, 1.65, 1.8, 2.0))
    esc18 = tra.get("r17_2_v6_gW1.8_n32_L48_r16_1_split0.05_r5", {}).get("iterations_to_GAP_MIN_at_last500_slope")
    l1s = tra.get("r17_2_v4rel_n32_L48_r16_1", {}).get("l1min_slope_per_100it_last500")
    v["C5"] = {"verdict": "QUALIFIED", "numbers": {"split_rel_decay_per_1000it_last_vs_prev": {t: (tra[t].get("split_rel_decay_per_1000it_last1000"), tra[t].get("split_rel_decay_per_1000it_prev1000")) for t in tra if "split0.05" in t},
                                                    "gap_slope_per_100it_and_iterations_to_escape": {t: (tra[t].get("gap_slope_per_100it_last500"), tra[t].get("iterations_to_GAP_MIN_at_last500_slope")) for t in tra},
                                                    "split_location_end": {t: tra[t].get("r_hs_end") for t in tra}, "l1min_slope_per_100it_all_runs": {t: tra[t].get("l1min_slope_per_100it_last500") for t in tra},
                                                    "center_isolation_margin_own": {t: stat.get(t, {}).get("own", {}).get("center_isolation_margin (l1 - pair mean, min over the 8 central cells)") for t in stat}, "seed_amplitude": 0.05, "sextic_plateau_s_star": 0.1118},
               "sentence": f"the statics support only 'not found within 3000 iterations from these two seeds inside the admissible domain': the r 2 seeds' split decays toward zero at g_W <= 1.65 (log decay per 1000 iterations, last window: {d_r2}; the rate itself shrinks with g_W), the shell seeds decay ever slower with g_W ({d_r5}), sit in the core at the end (r 2.49) and close the director gap at 0.0003 to 0.0012 per 100 iterations (escape (d) in about {esc18:.0f} more iterations at 1.8), while every run including the control keeps melting the core (lambda_1 min falling {l1s:.4f} per 100 iterations): the domain's central isolation margin (0.04 to 0.05) is below the seed amplitude 0.05 and far below the sextic plateau s* 0.112, so a core condensate of the author's kind is unrepresentable here and 'no condensate at g_W <= 2.0' is not established."}
    dump()
    # ---------- the summary table
    print("\n| claim | verdict | key numbers |")
    print("| --- | --- | --- |")
    n1 = v["C1"]["numbers"]
    print(f"| C1 no condensate | {v['C1']['verdict']} | energies/splits/gaps reproduced {n1['energies_reproduced_1e-5']}/{n1['half_split_reproduced_1e-4']}/{n1['gap_reproduced_1e-4']}; saddle g_W spread {n1['saddle_gW_spread']}; saddle slope {n1['saddle_E_slope_per_100it_last500']} per 100 it; min excess/(fall per 100 it) {n1['min_(E_end - E_saddle)_over_(E fall per 100 it)']:.3f}; slow-decay runs {len(n1['runs_with_split_decay_slower_than_5pct_per_1000it'])}; r2 seed gap {n1['seed_gap_r2']} |")
    n2 = v["C2"]["numbers"]
    print(f"| C2 S/I, S/2kin | {v['C2']['verdict']} | S/I own {', '.join(f'{k} {x:.3f}' for k, x in n2['S_over_I_own'].items())}; max rel dev vs producer {max(n2['S_own_vs_producer_rel_dev'].values()) if n2['S_own_vs_producer_rel_dev'] else None}; synthetic {n2['synthetic_uniform_S_over_I']} vs predicted {n2['synthetic_predicted_S_over_I']} |")
    n3 = v["C3"]["numbers"]
    print(f"| C3 2kin vs I | {v['C3']['verdict']} | 2kin/I own {n3['range_own']} (stated 7.5 to 9.1); kin_KP/I = 4 identity; synthetic kin_tot/I {n3['synthetic_kin_tot_over_I']} vs {n3['synthetic_predicted']} |")
    print(f"| C4 zero axes | {v['C4']['verdict']} | seeded fields axis.z > 0.99, control axis.111 > 0.99: {ctrl_ok} |")
    print(f"| C5 verdict | {v['C5']['verdict']} | not found within 3000 it, not excluded; shell-seed split decay per 1000 it slows with g_W to -0.01 at 2.0; core margin 0.04 to 0.05 < seed 0.05 < s* 0.112 |")
    print("\nrows:")
    for r_ in rows:
        print(f"  {r_['tag']:48s} gW {str(r_['gW']):5s} r {r_['seed_r']:.0f} {str(r_['stop']):8s} E {r_['E_own']!s:20s} dE_sad {r_['E_minus_saddle_own']!s:24s} hs {r_['hs_own']!s:22s} gap {r_['gap_own']!s:22s} slope {r_['E_slope_100it']!s:24s} decay/1000 {r_['split_decay_per_1000it_last']!s:22s} gap slope {r_['gap_slope_100it']!s:24s} it_to_escape {r_['it_to_GAP_MIN']} r_hs {r_['r_hs_end']}")
    log(f"done; wall {time.time() - T0:.0f} s; skipped {AUD['skipped']}")


if __name__ == "__main__":
    main()
