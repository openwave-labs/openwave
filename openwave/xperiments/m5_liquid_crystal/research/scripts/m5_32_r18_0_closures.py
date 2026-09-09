"""M5.32 R18-0 (ledger 6.7): the record and the cheap closures.
(a) The H_00 multiplicity reconciliation: our {0, 0, 1, 1, 1, 1, 2, 2, 2, 6} (R17-0g, the Frobenius-orthonormal basis of Sym(4):
    E_aa and (E_ab + E_ba) / sqrt 2) against the author's {1, 1, 1, 1, 1, 2, 2, 3} (rev-155 reply), recomputed here as the
    10 x 10 form H_00[xi, xi'] = d^2 l / dA_0 dA_0 (m5_32_r16_4_symbol.lag_density on the frozen background, central
    differences at eps 1e-4, the R17-0g method) and read in four conventions: (i) the Frobenius-orthonormal basis (R17-0g),
    (ii) the coordinate basis (E_ab + E_ba, norm sqrt 2 on the six off-diagonal directions: the form doubles there),
    (iii) the eta-weighted inner product <X, Y> = tr(eta X eta Y) (the Gram matrix diag(eta_aa eta_bb) on the orthonormal
    basis: the generalized eigenvalues of (G^-1 H), real since H is positive semidefinite), (iv) the spatial 6 x 6 block and
    the time-row 4 x 4 block separately.  Pre-registered: a basis convention; reported as which convention gives which
    multiplicity pattern, with the author's pattern matched or not.
(b) The section-177 unit correction, from the R17-0 record (no run): r_0 in box units and in cells on the three cores.
(c) The spin-2 zero count (m5_32_r18_common.spin2_zero_count, the harmonic-fit counter gated in its selftest) on the R17-3
    seeded end fields, the seed itself, the g_W 2.0 control, the v4rel static, and the R17-1 n32 K50 delocalized state:
    the prediction is total index 4 on every shell (Poincare-Hopf on the spin-2 bundle, Euler number 4); the content is the
    number and index of the zeros per shell and the l = 2 m-power of the section.
(d) The mode-9 speed: closed by the author's section 138 (rev 200), recorded only.

usage: python3 m5_32_r18_0_closures.py
out:   data/m5_32_r18_0.json, plots/m5_32_r18_0_spin2.png, plots/m5_32_r18_0_h00.png, checkpoints/m5_32_r18/r18_0.log
"""
from __future__ import annotations
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
import m5_32_r16_4_symbol as S4                           # noqa: E402
import m5_32_r17_0_symbol as SYM                          # noqa: E402
import m5_32_r16_2_operator as OP                         # noqa: E402

C15, INS4 = C.C15, C.INS4
ETA = C.ETA
RES, DATA, PLOTS = C.RES, C.DATA, C.PLOTS
CK16, CK17, CK = C.CK, R.CK, X.CK
T0 = time.time()
LOG = open(os.path.join(CK, "r18_0.log"), "a")
AUTHOR_PATTERN = [1, 1, 1, 1, 1, 2, 2, 3]


def log(m):
    line = f"[{time.time() - T0:8.1f}s] {m}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


def rel(p):
    return os.path.relpath(p, RES)


# ------------------------------------------------ (a) H_00 in four conventions
def H00_at(M, cfg, cells, eps=1e-4):
    fr0 = C.frame(M, C.radial_ref(cfg))
    a0 = C.a0_of(M, fr0)
    Asp = S4.jets_at(M, cfg)
    A = [0.0 * a0] + Asp
    idx = tuple(np.array(cells).T)
    Mc = M[idx]
    frc = {kk: (v[idx] if isinstance(v, np.ndarray) and v.shape[:3] == M.shape[:3] else v) for kk, v in fr0.items()}
    Ac = [a[idx] for a in A]
    B, lab = SYM.basis10()
    nc = len(cells)
    H = np.zeros((nc, 10, 10))
    for p in range(10):
        for q in range(p, 10):
            def ev(s1, s2):
                Bj = [a.copy() for a in Ac]
                Bj[0] = Bj[0] + s1 * eps * B[p] + s2 * eps * B[q]
                return np.real(S4.lag_density(Mc, Bj, frc, cfg))
            d2 = (ev(1, 1) - ev(1, -1) - ev(-1, 1) + ev(-1, -1)) / (4 * eps * eps)
            H[:, p, q] = d2; H[:, q, p] = d2
    return H, lab


def H00_continuum_axis(cfg, z=12.0):
    """H_00 of the quartic on the analytic hedgehog EXACTLY on the z axis, from the closed-form jets (no lattice cell sits on the
    axis: the lattice cells at x = y = h/2 are 5.4 degrees off it at r 12, which the R18-0 audit found distorts the basis-dependent
    coordinate reading by 7 percent; the audit's correction).  M = diag(g, S), S = delta I + (1 - delta) rhat rhat^T,
    d_i S = (1 - delta)(d_i rhat rhat^T + rhat d_i rhat^T), d_i rhat = (e_i - rhat rhat_i) / r; the density is exactly quadratic
    in A_0, so H_00 follows by the polarization identity of the kinetic quartic (no finite differences)."""
    g, delta = cfg["g"], cfg["delta"]
    r = z
    rhat = np.array([0.0, 0.0, 1.0])
    M = np.zeros((1, 4, 4)); M[0, 0, 0] = -cfg["sg"]; M[0, 1:, 1:] = delta * np.eye(3) + (1.0 - delta) * np.outer(rhat, rhat)
    Asp = []
    for i in range(3):
        e = np.zeros(3); e[i] = 1.0
        dr = (e - rhat * rhat[i]) / r
        A = np.zeros((1, 4, 4)); A[0, 1:, 1:] = (1.0 - delta) * (np.outer(dr, rhat) + np.outer(rhat, dr))
        Asp.append(A)
    fr = C.frame(M)
    B, lab = SYM.basis10()
    def kin(v):
        Bj = [v] + Asp
        return float(np.real(S4.lag_density(M, Bj, fr, cfg))[0])
    l0 = kin(np.zeros((1, 4, 4)))
    H = np.zeros((10, 10))
    for p in range(10):
        for q in range(p, 10):
            vp = B[p][None]; vq = B[q][None]
            d2 = (kin(vp + vq) - kin(vp) - kin(vq) + l0)
            H[p, q] = d2; H[q, p] = d2
    return H, lab


def multiplicities(e, tol=2e-2):
    """cluster the sorted eigenvalues normalized by the largest; returns [(value, count)] and the integer pattern if one fits."""
    e = np.sort(np.asarray(e))
    scale = max(float(np.max(np.abs(e))), 1e-300)
    en = e / scale
    groups = []
    for v in en:
        if groups and abs(v - groups[-1][0]) < tol:
            groups[-1][1] += 1
            groups[-1][0] = (groups[-1][0] * (groups[-1][1] - 1) + v) / groups[-1][1]
        else:
            groups.append([float(v), 1])
    nz = [g for g in groups if abs(g[0]) > tol]
    smallest = min(abs(g[0]) for g in nz) if nz else 1.0
    ratios = [g[0] / smallest for g in nz]
    integer = all(abs(r_ - round(r_)) < 0.08 for r_ in ratios)
    pattern = sorted(sum([[int(round(r_))] * g[1] for r_, g in zip(ratios, nz)], [])) if integer else None
    return {"normalized_groups": [[round(g[0], 4), g[1]] for g in groups], "n_zero": int(sum(g[1] for g in groups if abs(g[0]) <= tol)),
            "integer_pattern_of_nonzero": pattern, "matches_author": bool(pattern == AUTHOR_PATTERN), "matches_ours": bool(pattern == [1, 1, 1, 1, 2, 2, 2, 6])}


def conventions(H, lab):
    """the four readings of one 10 x 10 form."""
    Hs = 0.5 * (H + H.T)
    out = {}
    e = np.linalg.eigvalsh(Hs)
    out["frobenius_orthonormal (R17-0g)"] = multiplicities(e)
    T = np.diag([1.0 if l[0] == l[1] else np.sqrt(2.0) for l in lab])
    out["coordinate basis (E_ab + E_ba, the off-diagonal directions of norm sqrt 2)"] = multiplicities(np.linalg.eigvalsh(T @ Hs @ T))
    G = np.diag([ETA[int(l[0]), int(l[0])] * ETA[int(l[1]), int(l[1])] for l in lab])
    ge = np.linalg.eigvals(G @ Hs)
    out["eta-weighted inner product (generalized eigenvalues of G^-1 H, G = diag(eta_aa eta_bb))"] = multiplicities(np.real(ge))
    out["eta-weighted: max imaginary part (must be 0: H PSD)"] = float(np.max(np.abs(np.imag(ge))))
    sp = [i for i, l in enumerate(lab) if l[0] != "0"]
    tm = [i for i, l in enumerate(lab) if l[0] == "0"]
    out["spatial 6x6 block"] = multiplicities(np.linalg.eigvalsh(Hs[np.ix_(sp, sp)]))
    out["time-row 4x4 block (0a directions)"] = multiplicities(np.linalg.eigvalsh(Hs[np.ix_(tm, tm)]))
    out["spatial-time coupling norm over H norm"] = float(np.linalg.norm(Hs[np.ix_(sp, tm)]) / max(np.linalg.norm(Hs), 1e-300))
    # the kernel directions
    w, V = np.linalg.eigh(Hs)
    ker = [i for i in range(10) if abs(w[i]) < 1e-5 * max(abs(w[-1]), 1e-300)]
    out["kernel"] = [{"eigenvalue": float(w[i]), "components": {lab[k]: round(float(V[k, i]), 3) for k in range(10) if abs(V[k, i]) > 0.05}} for i in ker]
    # the top direction
    out["top_direction_components"] = {lab[k]: round(float(V[k, -1]), 3) for k in range(10) if abs(V[k, -1]) > 0.05}
    out["eigenvalues_raw"] = w.tolist()
    return out


def part_a():
    log("(a) H_00 in four conventions")
    rec = {"author_pattern": AUTHOR_PATTERN, "our_pattern_R17_0g": [0, 0, 1, 1, 1, 1, 2, 2, 2, 6], "backgrounds": {}}
    items = {"analytic_hedgehog_n32_L48": (None, 32, 48.0), "r16_1_end_n32_L48": (os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"), 32, 48.0)}
    for lab_bg, (p, n, L) in items.items():
        cfgs = {"quartic_rebuild": C.cfg_v4(n, L, mu=0.0, cP=0.0, cs=0.0, completion="rebuild", n_samples=1), "full_v4_rebuild": C.cfg_v4(n, L, completion="rebuild", n_samples=1)}
        cfg0 = cfgs["quartic_rebuild"]
        M = C15.seed_uniaxial(cfg0) if p is None else np.load(p)
        h = cfg0["h"]
        X_, Y_, Z_ = INS4.coords(n, h)
        j = n // 2
        cells, names = [], []
        for rt in (6.0, 12.0):
            kz = int(np.argmin(np.abs(Z_[j, j, :] - rt)))
            cells.append((j, j, kz)); names.append(f"axis_r{rt:g}")
        rec["backgrounds"][lab_bg] = {"field": rel(p) if p else "C15.seed_uniaxial", "contractions": {}}
        for cname, cfg in cfgs.items():
            H, lab = H00_at(M, cfg, cells)
            res = {nm: conventions(H[ci], lab) for ci, nm in enumerate(names)}
            rec["backgrounds"][lab_bg]["contractions"][cname] = res
            for nm in names:
                r_ = res[nm]
                log(f"  {lab_bg} {cname} {nm}: orthonormal {r_['frobenius_orthonormal (R17-0g)']['integer_pattern_of_nonzero']} (zeros {r_['frobenius_orthonormal (R17-0g)']['n_zero']}); coordinate {r_['coordinate basis (E_ab + E_ba, the off-diagonal directions of norm sqrt 2)']['integer_pattern_of_nonzero']}; "
                    f"eta-weighted {r_['eta-weighted inner product (generalized eigenvalues of G^-1 H, G = diag(eta_aa eta_bb))']['integer_pattern_of_nonzero']} groups {r_['eta-weighted inner product (generalized eigenvalues of G^-1 H, G = diag(eta_aa eta_bb))']['normalized_groups']}; "
                    f"spatial {r_['spatial 6x6 block']['integer_pattern_of_nonzero']}; time-row {r_['time-row 4x4 block (0a directions)']['integer_pattern_of_nonzero']}; coupling {r_['spatial-time coupling norm over H norm']:.2e}; kernel {[k['components'] for k in r_['kernel']]}; top {r_['top_direction_components']}")
    # the exact on-axis reading (the audit's correction): the closed-form jets of the analytic hedgehog on the z axis at r 6 and 12
    cfgq = C.cfg_v4(32, 48.0, mu=0.0, cP=0.0, cs=0.0, completion="rebuild", n_samples=1)
    rec["continuum_axis"] = {}
    for z in (6.0, 12.0):
        H, lab = H00_continuum_axis(cfgq, z)
        res = conventions(H, lab)
        rec["continuum_axis"][f"z{z:g}"] = res
        log(f"  continuum on the z axis at r {z:g}: orthonormal {res['frobenius_orthonormal (R17-0g)']['integer_pattern_of_nonzero']}; COORDINATE {res['coordinate basis (E_ab + E_ba, the off-diagonal directions of norm sqrt 2)']['integer_pattern_of_nonzero']} groups {res['coordinate basis (E_ab + E_ba, the off-diagonal directions of norm sqrt 2)']['normalized_groups']}; eta-weighted {res['eta-weighted inner product (generalized eigenvalues of G^-1 H, G = diag(eta_aa eta_bb))']['integer_pattern_of_nonzero']}")
    # the lattice cell's off-axis distortion of the coordinate reading
    lat = rec["backgrounds"]["analytic_hedgehog_n32_L48"]["contractions"]["quartic_rebuild"]["axis_r12"]["coordinate basis (E_ab + E_ba, the off-diagonal directions of norm sqrt 2)"]
    rec["lattice_cell_off_axis_note"] = {"cell": "x = y = h/2 = 0.75, z 11.25: 5.4 degrees off the axis", "coordinate_reading_groups_on_the_cell": lat["normalized_groups"],
                                         "statement": "the coordinate-basis reading is basis-dependent and distorted off the axis (the audit: 7 percent at h 1.5, 3 percent at h 0.75, converging); the Frobenius-orthonormal reading is not"}
    clean = rec["continuum_axis"]["z12"]
    match = [k for k, v in clean.items() if isinstance(v, dict) and v.get("matches_author")]
    rec["decision"] = {"conventions_matching_author_on_the_exact_axis": match,
                       "statement": ("A BASIS CONVENTION, as pre-registered: the author's {1,1,1,1,1,2,2,3} is our H_00 read in the coordinate basis (E_ab + E_ba, the six off-diagonal directions of norm sqrt 2) on the axis; our {1,1,1,1,2,2,2,6} is the same form in the Frobenius-orthonormal basis; "
                                     "the characteristic speeds are basis-independent and both sides agree on them; the first version of this script decided the opposite from a lattice cell 5.4 degrees off the axis with a 2 percent clustering tolerance (the R18-0 audit's finding, corrected here)" if match else
                                     "no reading of our H_00 reproduces the author's {1,1,1,1,1,2,2,3} even on the exact axis: the two H_00 differ by more than a basis convention (author-gated)")}
    log(f"  decision: {rec['decision']}")
    return rec


# ------------------------------------------------ (b) section 177
def part_b():
    log("(b) the section-177 unit correction from the R17-0 record")
    d = json.load(open(os.path.join(DATA, "m5_32_r17_0_record.json")))["c_core_reads_r16_1"]
    rows = []
    for tag, hh, n, L in (("r16_1_end_n32_L48", 1.5, 32, 48), ("r16_1_end_n48_L72", 1.5, 48, 72), ("r16_1_end_n64_L48", 0.75, 64, 48)):
        c = d[tag]
        r0p = c["r_0_profile (shell-mean lambda_1 = 0.8)"]; r0t = c["r_0_taper (max r with lambda_1 < 0.8)"]; r0h = c["r_0_half (max r with lambda_1 < (1+delta)/2)"]
        rows.append({"core": tag, "n": n, "L": L, "h": hh, "r_0_profile_box_units": r0p, "r_0_taper": r0t, "r_0_half": r0h, "r_0_profile_sqrt_mu (mu 1e-2)": r0p * 0.1,
                     "r_0_profile_in_cells": r0p / hh, "section_177_reading_in_cells (r_0 sqrt mu read as a width in box units)": r0p * 0.1 / hh, "Delta_min": c["Delta_min_free"]})
    rec = {"rows": rows, "statement": ("section 177 reads the 0.304 as a core WIDTH of 0.304 box units and concludes the h 0.75 run resolves 0.41 cells; on the record r_0 is the radius where the shell-mean director "
                                       "eigenvalue crosses 0.8, 3.04 / 3.11 / 1.00 box units on the n32 L48 / n48 L72 / n64 L48 cores (mu = 1e-2, so r_0 sqrt mu = r_0 / 10), i.e. 2.0 / 2.1 / 1.3 cells: none resolved, "
                                       "and the fine grid shows the SMALLER core (the spread is real and points the other way); the tube solve (R18-1) is the instrument that decides it")}
    for r_ in rows:
        log(f"  {r_['core']}: r_0 profile {r_['r_0_profile_box_units']:.3f} box units = {r_['r_0_profile_in_cells']:.2f} cells at h {r_['h']} (section 177 would read {r_['section_177_reading_in_cells (r_0 sqrt mu read as a width in box units)']:.2f} cells); taper {r_['r_0_taper']:.2f}, half {r_['r_0_half']:.2f}")
    return rec


# ------------------------------------------------ (c) the spin-2 zero count
def part_c():
    log("(c) the spin-2 zero count on the R17 fields")
    cfg32 = C.cfg_v4(32, 48.0, completion="rebuild")
    fields = {"v4rel_static_end (residual split 2e-4)": os.path.join(CK17, "r17_2_v4rel_n32_L48_r16_1.npy"),
              "v6_gW0.5_seeded_end": os.path.join(CK17, "r17_2_v6_gW0.5_n32_L48_r16_1_split0.05.npy"),
              "v6_gW1.1_seeded_end": os.path.join(CK17, "r17_2_v6_gW1.1_n32_L48_r16_1_split0.05.npy"),
              "v6_gW1.35_seeded_end": os.path.join(CK17, "r17_2_v6_gW1.35_n32_L48_r16_1_split0.05.npy"),
              "v6_gW2.0_seeded_escape_d": os.path.join(CK17, "r17_2_v6_gW2_n32_L48_r16_1_split0.05.npy"),
              "v6_gW2.0_unseeded_control": os.path.join(CK17, "r17_2_v6_gW2_n32_L48_r16_1.npy"),
              "r17_1_n32_K50_delocalized (split at r 12.8)": os.path.join(CK17, "r17_1_rebuild_n32_L48_K50.npy"),
              "r17_3c_gW1.35_K200_escape_d (split 0.32 at r 2.5)": os.path.join(CK17, "r17_3c_v6_gW1.35_n32_L48_K200.npy")}
    # the seed itself (R16-1 + 0.05 Ea at r 2)
    M0 = np.load(os.path.join(CK16, "r16_1_rebuild_n32_L48.npy"))
    cfg6 = R.cfg_v6(32, 48.0, gW=1.35, completion="rebuild", n_samples=4)
    with R.weight_mode(cfg6):
        Ea, Eb, fr0, r0 = OP.doublet_basis(M0, cfg6)
    free = ~INS4.pin_shell(32, cfg32["h"], 1.6)
    env = 0.05 * np.exp(-(r0 - 2.0) ** 2 / 8.0) * free
    Mseed = M0 + env[..., None, None] * Ea
    rec = {"fields": {}, "prediction": "total index 4 on every shell (Poincare-Hopf on the spin-2 bundle over the sphere, Euler number 4)"}
    radii_core = (1.5, 2.25, 3.0, 4.5, 6.0)
    radii_far = (1.5, 2.25, 3.0, 4.5, 6.0, 9.0, 12.0, 15.0)
    allM = [("the_seed (R16-1 core + 0.05 Ea at r 2)", Mseed, radii_core)] + [(k, np.load(p), radii_far if "delocalized" in k else radii_core) for k, p in fields.items() if os.path.exists(p)]
    for name, M, radii in allM:
        t = time.time()
        z = X.spin2_zero_count(M, cfg32, radii=radii)
        rec["fields"][name] = z
        log(f"  {name} ({time.time() - t:.0f} s): " + "; ".join(f"r {k}: total {v['total_index']} n {v['n_zeros']} hist {v['index_histogram']} resid {v['fit_residual_rel']:.2f} zeta_rms {v['zeta_rms_cells']:.1e} gap {v['min_director_gap']:.3f} l2m {({kk: round(vv / max(sum(v['l2_m_power'].values()), 1e-300), 4) for kk, vv in v['l2_m_power'].items()})} hull {v['hull_count']['total_index']}/{v['hull_count']['count_reliable']}" for k, v in z.items() if "total_index" in v))
    # summary
    summ = {}
    for name, z in rec["fields"].items():
        tots = [v["total_index"] for v in z.values() if "total_index" in v]
        rels = [v["fit_residual_rel"] for v in z.values() if "total_index" in v]
        summ[name] = {"totals": tots, "all_4": bool(all(t == 4 for t in tots)), "n_zeros": [v["n_zeros"] for v in z.values() if "total_index" in v], "fit_residuals": [round(r_, 3) for r_ in rels], "shells_with_residual_below_0.3": int(sum(1 for r_ in rels if r_ < 0.3))}
    rec["summary"] = summ
    # the structure of the zeros (the audit's reading: the number of SIMPLE zeros is fit-dependent, the index-2 CLUSTERS and their axes are the observable)
    for name, z in rec["fields"].items():
        for k, v in z.items():
            if "zeros" not in v:
                continue
            pts = np.array([z_["xyz"] for z_ in v["zeros"]]); idx = np.array([z_["index"] for z_ in v["zeros"]])
            if len(pts) == 0:
                continue
            Q = np.einsum("i,ia,ib->ab", np.abs(idx).astype(float), pts, pts) / max(np.sum(np.abs(idx)), 1e-300)
            w, V = np.linalg.eigh(Q)
            ax = V[:, -1]
            v["zero_cluster_axis"] = [float(x) for x in ax]
            v["zero_cluster_axis_anisotropy (top eigenvalue of the index-weighted second moment; 1 = all zeros on one axis)"] = float(w[-1])
            v["index_within_20deg_of_axis"] = int(sum(i_ for i_, p_ in zip(idx, pts) if abs(np.dot(p_, ax)) > np.cos(np.radians(20))))
            v["axis_dot_z"] = float(abs(ax[2])); v["axis_dot_111"] = float(abs(np.dot(ax, np.ones(3) / np.sqrt(3))))
    rec["verdict"] = ("the total index 4 is a theorem for a generic section (unfalsifiable: it tests the counter, not the field; the producer's hull count fails it where the samples straddle a zero); the CONTENT: on the seeded fields the zeros sit as two index-2 clusters on the +-z axis (the m = 0 seed pattern, split into simple zeros 8 to 10 degrees off the poles by a 1e-3-power m = +-2 admixture), "
                      "on the split-free statics as two index-2 clusters on +-(1,1,1) (the cubic residual of the lattice hedgehog); the number of SIMPLE zeros is fit-dependent (the R18-0 audit: 2 to 8 on the statics across l_max), the clusters and their axes are stable"
                      if all(s_["all_4"] for s_ in summ.values()) else "the fit's total index differs from 4 on some shell: a counter defect on that shell, reported")
    log(f"  verdict: {rec['verdict']}; summary {summ}")
    # plot: the zeros on a (phi, cos theta) map for the seeded fields at r 2.25 and 4.5
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = [k for k in rec["fields"] if any(s in k for s in ("seed", "gW0.5", "gW1.35", "gW2.0_seeded", "K200"))][:5]
    fig, ax = plt.subplots(2, len(picks), figsize=(3.6 * len(picks), 6.4), squeeze=False)
    for ci, k in enumerate(picks):
        for ri, rr in enumerate(("2.25", "4.5")):
            v = rec["fields"][k].get(rr, {})
            a = ax[ri, ci]
            for z_ in v.get("zeros", []):
                x_, y_, zz_ = z_["xyz"]
                a.scatter(np.degrees(np.arctan2(y_, x_)), zz_, s=40 * abs(z_["index"]), c="r" if z_["index"] > 0 else "b", marker="o" if z_["index"] > 0 else "x")
            a.set_xlim(-180, 180); a.set_ylim(-1, 1)
            a.set_title(f"{k[:28]}\nr {rr}: total {v.get('total_index')} n {v.get('n_zeros')} resid {v.get('fit_residual_rel', 0):.2f}", fontsize=7)
            if ci == 0:
                a.set_ylabel("cos theta")
            if ri == 1:
                a.set_xlabel("phi (deg)")
    fig.suptitle("R18-0c: the zeros of the spin-2 split section on shells (red: positive index, size = |index|); Poincare-Hopf total 4", fontsize=8)
    fig.savefig(os.path.join(PLOTS, "m5_32_r18_0_spin2.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    rec["plot"] = "plots/m5_32_r18_0_spin2.png"
    return rec


def plot_h00(reca):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bg = reca["backgrounds"]["analytic_hedgehog_n32_L48"]["contractions"]["quartic_rebuild"]["axis_r12"]
    keys = [k for k in bg if isinstance(bg[k], dict) and "eigenvalues_raw" not in bg[k] and "normalized_groups" in bg[k]]
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    for i, k in enumerate(keys):
        g = bg[k]["normalized_groups"]
        vals = sum([[v] * c for v, c in g], [])
        ax.plot([i] * len(vals), vals, "o", ms=5, alpha=0.6)
        ax.text(i, 1.05, str(bg[k]["integer_pattern_of_nonzero"]), ha="center", fontsize=6)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([k.split(" (")[0] for k in keys], rotation=15, fontsize=7)
    ax.set_ylabel("eigenvalues of H_00 / max"); ax.set_title("R18-0a: H_00 on the analytic hedgehog axis (r 12), the quartic alone, in four conventions; the author's pattern {1,1,1,1,1,2,2,3}, ours {1,1,1,1,2,2,2,6}", fontsize=8)
    fig.savefig(os.path.join(PLOTS, "m5_32_r18_0_h00.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
    return "plots/m5_32_r18_0_h00.png"


if __name__ == "__main__":
    out = {"rung": "R18-0", "a_H00_conventions": part_a(), "b_section_177": part_b()}
    out["a_H00_conventions"]["plot"] = plot_h00(out["a_H00_conventions"])
    json.dump(out, open(os.path.join(CK, "r18_0_partial.json"), "w"), indent=1, default=float)
    out["c_spin2_zeros"] = part_c()
    out["d_mode_9"] = {"status": "closed by the author's section 138 (rev 200): the Fresnel surface v_1 = cos theta, v_2 = sqrt((1 + cos^2 theta) / 2), v_3 = 1; the 2 / sqrt 5 an artefact of the singular theta = 90 degree point; our R17 transverse speeds {1/sqrt2, 1/sqrt2, 1, 1, 1} stand; nothing run"}
    out["wall_s"] = time.time() - T0
    json.dump(out, open(os.path.join(DATA, "m5_32_r18_0.json"), "w"), indent=1, default=float)
    log(f"written data/m5_32_r18_0.json ({out['wall_s']:.0f} s)")
