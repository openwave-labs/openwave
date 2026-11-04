"""M5.32 R32-1: the strand read on the stored R31 fields (the author's question 4 of 2026-11-02): on the merged
J = 0 state at the sextic flip, does the minor gap stay open along the former strands, or does the lift remove the
strands themselves?

EQUATIONS FIRST
---------------
S(x) the 3x3 spatial block of the stored field M(x) (M_0i = 0 on every field of the R31 instrument), its sorted
eigenvalues lambda_0 <= lambda_1 <= lambda_2 (numpy eigvalsh).  On the pinned uniaxial exterior the spectrum is
(delta, delta, 1): the degenerate pair is the two LOWER eigenvalues.
    minor gap   m(x) = lambda_1 - lambda_0        (zero on the exterior by the pin; the biaxiality, twice the R31
                                                   read `biax_eps`; the author's strand diagnostic, Release 2.0
                                                   Figs. 8 and 9: strands dark)
    long gap    l(x) = lambda_2 - lambda_1
    biaxial region   B_th = {x : m(x) > m_th},  m_th in {0.01, 0.03, 0.1}  (0.03 = a tenth of delta, the plan's)
    strand cells     Z_th = {x : m(x) < m_th, r(x) < R_in, at least half of the 26 neighbors in B_th}
                     (closed-gap cells embedded in biaxial surroundings; R_in = 9, inside the pin at r 12 and its
                     transition shell, so the biaxial region's own outer surface is not read as a strand)
    components       26-connected components of Z_th (scipy.ndimage.label), their cell counts and extents
    along the loci   on a lift field of the same seed, m read on the center field's Z_th: median, max, the
                     fraction above m_th
    jump links       R31's definition (the article, Sec. 10.2): nearest-neighbor links with both ends at r < 14,
                     |dS|_F > 0.3 and |d sorted lambda|_2 < 0.5 |dS|_F; each link's midpoint cell is located by
                     its distance (in cells, Euclidean distance transform) to the nearest center strand cell of
                     the same seed
Pre-registered words (the plan): GAP OPEN ALONG THE STRANDS (cured): the biaxial region kept (fraction within x2 of
the center's) and m along the former loci above m_th.  HOST REMOVED: the biaxial fraction below a tenth of the
center's and m below m_th on every cell inside r 12.  STRANDS MOVED: the region kept, enclosed closed-gap
components elsewhere.  A word that changes across the three thresholds is reported as threshold-dependent.

Known-answer control (`control`): the exact uniaxial exterior everywhere, a biaxial ball of radius 11 (the pair
split by 0.1 (e1 e1^T - e2 e2^T) in the local transverse frame) with one planted line of re-degeneracy along z through the center: the reader must
return that line as ONE component inside r < 9 and nothing else; the pure vacuum must return no strand cell and a
biaxial fraction of zero.

Modes: control | reads | plot.  Output: data/m5_32_r32_strands.json, plots/m5_32_r32_strands.png.
Seconds per field, one process (a desk read; the parallel shape of the plan: sequential by choice).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
PLOTS = os.path.join(HERE, "..", "plots")
NPZ = os.path.join(DATA, "m5_32_r31_s1")
OUT = os.path.join(DATA, "m5_32_r32_strands.json")
R31 = os.path.join(DATA, "m5_32_r31_s1.json")

THRESHOLDS = (0.01, 0.03, 0.1)
R_IN = 9.0
JUMP_TENSOR, JUMP_SPEC_FRAC, JUMP_R = 0.3, 0.5, 14.0
SEEDS = ("rad", "obl", "bia", "perm", "node")
LIFTS = ("ax_sx_0.5", "ax_sx_2", "ax_sx_8.2", "ax_sel_1", "ax_sel_5", "ax_sel_19.3")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    argv = sys.argv
    sys.argv = [argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


core = _load("m5_32_r31_s1_core", "m5_32_r31_s1_core.py")
_T = {}


def terms(n):
    if n not in _T:
        _T[n] = core.Terms(n, n * 1.5, 8.0, 0.3, 25.0)
    return _T[n]


def gaps(S):
    lam = np.sort(np.linalg.eigvalsh(S), axis=-1)
    return lam[..., 1] - lam[..., 0], lam[..., 2] - lam[..., 1], lam


_K26 = np.ones((3, 3, 3), dtype=int)
_K26[1, 1, 1] = 0


def strand_cells(m, r, m_th):
    biax = m > m_th
    nb = ndimage.convolve(biax.astype(int), _K26, mode="constant", cval=0)
    return (m < m_th) & (r < R_IN) & (nb >= 13), biax


def components(mask, h):
    lab, k = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    out = []
    for i in range(1, k + 1):
        idx = np.argwhere(lab == i)
        ext = float(np.linalg.norm(idx.max(0) - idx.min(0)) * h) if len(idx) > 1 else 0.0
        out.append({"cells": int(len(idx)), "extent": ext, "centroid_cell": [float(c) for c in idx.mean(0)]})
    out.sort(key=lambda c: -c["cells"])
    return out


def jump_links(S, lam, r):
    """the R31 definition; returns the midpoint cell indices of the jump links and the link counts."""
    n = S.shape[0]
    inside = r < JUMP_R
    n_links = n_jump = 0
    mids = []
    for ax in range(3):
        a = [slice(None)] * 3
        b = [slice(None)] * 3
        a[ax] = slice(0, n - 1)
        b[ax] = slice(1, n)
        both = inside[tuple(a)] & inside[tuple(b)]
        dS = np.sqrt(np.sum((S[tuple(b)] - S[tuple(a)]) ** 2, axis=(-1, -2)))
        dl = np.sqrt(np.sum((lam[tuple(b)] - lam[tuple(a)]) ** 2, axis=-1))
        jump = both & (dS > JUMP_TENSOR) & (dl < JUMP_SPEC_FRAC * dS)
        n_links += int(both.sum())
        n_jump += int(jump.sum())
        for idx in np.argwhere(jump):
            mids.append(idx)  # the lower-index end of the link (the link is between idx and idx + e_ax)
    return np.array(mids, dtype=int).reshape(-1, 3), n_links, n_jump


def read_field(M, ref=None):
    """every read of one field; ref = the same seed's center read (its strand masks) for the along-the-loci reads."""
    n = M.shape[0]
    T = terms(n)
    r, h = T.r, T.h
    S = M[..., 1:, 1:]
    m, lgap, lam = gaps(S)
    in12 = r < 12.0
    rec = {"n": int(n), "m_max": float(m.max()), "m_max_r": float(r.ravel()[int(np.argmax(m))]),
           "m_mean_r_lt_12": float(m[in12].mean()), "long_gap_min_r_lt_12": float(lgap[in12].min()),
           "profile": {"r": [], "m_mean": [], "m_max": []}, "thresholds": {}, "masks": {}}
    for R in np.arange(1.5, 0.5 * T.L - 1.0, 1.5):
        sh = np.abs(r - R) < 0.75 * h
        if sh.any():
            rec["profile"]["r"].append(float(R))
            rec["profile"]["m_mean"].append(float(m[sh].mean()))
            rec["profile"]["m_max"].append(float(m[sh].max()))
    mids, n_links, n_jump = jump_links(S, lam, r)
    rec["jumps"] = {"n_links_r_lt_14": n_links, "n_jump_links": n_jump, "fraction": (n_jump / n_links if n_links else None)}
    if len(mids):
        rj = r[tuple(mids.T)]
        mj = m[tuple(mids.T)]
        edges = np.arange(0.0, 15.1, 1.5)
        rec["jumps"]["per_shell_r_edges"] = [float(e) for e in edges]
        rec["jumps"]["per_shell_counts"] = [int(c) for c in np.histogram(rj, bins=edges)[0]]
        rec["jumps"]["fraction_with_m_above"] = {str(th): float((mj > th).mean()) for th in THRESHOLDS}
        rec["jumps"]["m_median_at_jump_end"] = float(np.median(mj))
    for th in THRESHOLDS:
        Z, biax = strand_cells(m, r, th)
        comps = components(Z, h)
        d = {"m_th": th, "biax_fraction_r_lt_12": float(biax[in12].mean()), "biax_fraction_r_lt_9": float(biax[r < R_IN].mean()),
             "biax_cells_r_lt_12": int(biax[in12].sum()), "strand_cells": int(Z.sum()), "n_components": len(comps),
             "components": comps[:12], "all_cells_r_lt_12_below_th": bool((m[in12] < th).all())}
        # the jump links against this field's own strand cells
        if Z.any() and len(mids):
            dist = ndimage.distance_transform_edt(~Z)
            dj = dist[tuple(mids.T)]
            d["jumps_within_1_cell_of_own_strands"] = float((dj <= 1.0).mean())
            d["jumps_within_2_cells_of_own_strands"] = float((dj <= 2.0).mean())
        if ref is not None:
            Zc = ref["masks"][th]
            if Zc.any():
                mc = m[Zc]
                d["along_center_loci"] = {"cells": int(Zc.sum()), "m_median": float(np.median(mc)), "m_max": float(mc.max()),
                                          "m_min": float(mc.min()), "fraction_above_th": float((mc > th).mean()),
                                          "center_m_median": float(ref["m"][Zc].mean())}
                if len(mids):
                    distc = ndimage.distance_transform_edt(~Zc)
                    dj = distc[tuple(mids.T)]
                    d["jumps_within_1_cell_of_center_strands"] = float((dj <= 1.0).mean())
                    d["jumps_within_2_cells_of_center_strands"] = float((dj <= 2.0).mean())
            d["biax_fraction_over_center"] = (d["biax_fraction_r_lt_12"] / ref["thresholds"][th]["biax_fraction_r_lt_12"]
                                              if ref["thresholds"][th]["biax_fraction_r_lt_12"] > 0 else None)
        rec["thresholds"][th] = d
        rec["masks"][th] = Z
    rec["m"] = m
    return rec


def word(d, ref_d, all_below):
    """the pre-registered words for one lift read at one threshold."""
    frac, cfrac = d["biax_fraction_r_lt_12"], ref_d["biax_fraction_r_lt_12"]
    kept = cfrac > 0 and 0.5 <= frac / cfrac <= 2.0
    al = d.get("along_center_loci")
    if cfrac > 0 and frac < 0.1 * cfrac and all_below:
        return "HOST REMOVED"
    if kept and al is not None and al["fraction_above_th"] >= 0.5 and d["strand_cells"] == 0:
        return "GAP OPEN ALONG THE STRANDS"
    if kept and d["strand_cells"] > 0:
        return "STRANDS MOVED" if (al is None or al["fraction_above_th"] >= 0.5) else "STRANDS KEPT"
    if cfrac > 0 and frac < 0.1 * cfrac:
        return "HOST MOSTLY REMOVED (m above m_th on some cells inside r 12)"
    if frac / cfrac < 0.5 if cfrac > 0 else False:
        return "HOST SHRUNK (biaxial fraction between a tenth and a half of the center's)"
    return "NO WORD (outside the pre-registered cases)"


def strip(rec):
    return {k: v for k, v in rec.items() if k not in ("masks", "m")}


def control():
    T = terms(24)
    n, h, r = T.n, T.h, T.r
    X = (np.arange(n) - (n - 1) / 2) * h
    x, y, z = np.meshgrid(X, X, X, indexing="ij")
    S = np.array(T.S_ext, dtype=float)
    # the pair split in the LOCAL transverse frame (e1, e2 orthonormal to r-hat), so the planted minor gap is 0.2 on
    # every ball cell off the line (a perturbation in a fixed frame has no transverse part along some directions and
    # would plant degeneracy lines of its own: the first control build did, and the reader found them)
    rho = np.sqrt(x * x + y * y)
    rr = np.maximum(r, 1e-9)
    rhat = np.stack([x, y, z], -1) / rr[..., None]
    e1 = np.stack([-y, x, np.zeros_like(x)], -1) / np.maximum(rho, 1e-9)[..., None]
    e2 = np.cross(rhat, e1)
    pert = 0.1 * (e1[..., :, None] * e1[..., None, :] - e2[..., :, None] * e2[..., None, :])
    ball = r < 11.0
    line = (np.abs(x) < 0.6 * h) & (np.abs(y) < 0.6 * h)
    S2 = S + np.where((ball & ~line & (rho > 0.5 * h))[..., None, None], pert, 0.0)
    M = np.zeros((n, n, n, 4, 4))
    M[..., 1:, 1:] = S2
    rec = read_field(M)
    vac = np.zeros((n, n, n, 4, 4))
    vac[..., 1:, 1:] = S
    rv = read_field(vac)
    out = {"planted": strip(rec), "vacuum": strip(rv), "expected": {"planted": "one component on the z axis inside r < 9, about 2 R_in / h cells",
                                                                     "vacuum": "no strand cell, biaxial fraction 0"}}
    ok = True
    for th in THRESHOLDS:
        p, v = rec["thresholds"][th], rv["thresholds"][th]
        line_cells = int((line & (r < R_IN)).sum())
        good = (p["n_components"] == 1 and abs(p["strand_cells"] - line_cells) <= 2 and v["strand_cells"] == 0 and v["biax_fraction_r_lt_12"] == 0.0)
        out[f"pass_{th}"] = bool(good)
        out[f"line_cells_expected_{th}"] = line_cells
        ok &= good
        print(f"control m_th {th}: planted components {p['n_components']}, strand cells {p['strand_cells']} (line cells inside r 9: {line_cells}), "
              f"biax fraction {p['biax_fraction_r_lt_12']:.3f}; vacuum strand cells {v['strand_cells']}, biax {v['biax_fraction_r_lt_12']:.3f} -> {'PASS' if good else 'FAIL'}")
    out["pass"] = bool(ok)
    return out


def reads():
    pts = json.load(open(R31))["points"]
    tags = {pid: pts[pid]["tags"] for pid in LIFTS}
    out = {"center": {}, "lifts": {}, "secondary_J1_sx_8.2": {}, "definition": __doc__.split("Pre-registered")[0]}
    center_recs = {}
    for seed in SEEDS:
        tag = f"c0_{seed}_J0_n24_L36_g8_d0.3_w25_c60_ls0_lsel0"
        M = np.load(os.path.join(NPZ, tag + ".npz"))["M"]
        t0 = time.time()
        rec = read_field(M)
        center_recs[seed] = rec
        out["center"][seed] = dict(strip(rec), tag=tag, seconds=time.time() - t0)
        d = rec["thresholds"][0.03]
        print(f"center {seed:4s}: m_max {rec['m_max']:.3f} biax frac(r<12) {d['biax_fraction_r_lt_12']:.3f} strand cells {d['strand_cells']} comps {d['n_components']} "
              f"jumps {rec['jumps']['fraction']:.4f} within 1 cell of strands {d.get('jumps_within_1_cell_of_own_strands')}")
    for pid in LIFTS:
        out["lifts"][pid] = {}
        for seed in SEEDS:
            tag = [t for t in tags[pid] if f"_{seed}_J0_" in t][0]
            M = np.load(os.path.join(NPZ, tag + ".npz"))["M"]
            rec = read_field(M, ref=center_recs[seed])
            ws = {}
            for th in THRESHOLDS:
                d = rec["thresholds"][th]
                ws[th] = word(d, center_recs[seed]["thresholds"][th], d["all_cells_r_lt_12_below_th"])
            out["lifts"][pid][seed] = dict(strip(rec), tag=tag, words=ws)
            d = rec["thresholds"][0.03]
            al = d.get("along_center_loci", {})
            print(f"{pid:12s} {seed:4s}: m_max {rec['m_max']:.3f} biax frac {d['biax_fraction_r_lt_12']:.3f} (x{d['biax_fraction_over_center'] or 0:.2f} center) "
                  f"strand cells {d['strand_cells']} along center loci m_med {al.get('m_median', float('nan')):.3f} above th {al.get('fraction_above_th', float('nan')):.2f} "
                  f"jumps {rec['jumps']['fraction']:.4f} -> {ws[0.03]}")
    for seed in SEEDS:
        tag = [t for t in tags["ax_sx_8.2"] if f"_{seed}_J1241.91_" in t][0]
        M = np.load(os.path.join(NPZ, tag + ".npz"))["M"]
        rec = read_field(M, ref=center_recs[seed])
        out["secondary_J1_sx_8.2"][seed] = dict(strip(rec), tag=tag)
        d = rec["thresholds"][0.03]
        print(f"J1 sx 8.2 {seed:4s}: m_max {rec['m_max']:.3f} biax frac {d['biax_fraction_r_lt_12']:.3f} strand cells {d['strand_cells']} comps {d['n_components']} jumps {rec['jumps']['fraction']:.4f}")
    return out


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["c0"] + list(LIFTS)
    fig, axes = plt.subplots(len(SEEDS), len(cols), figsize=(2.4 * len(cols), 2.3 * len(SEEDS)))
    n = 24
    T = terms(n)
    for i, seed in enumerate(SEEDS):
        for j, pid in enumerate(cols):
            tag = res["center"][seed]["tag"] if pid == "c0" else res["lifts"][pid][seed]["tag"]
            M = np.load(os.path.join(NPZ, tag + ".npz"))["M"]
            m, _, _ = gaps(M[..., 1:, 1:])
            Z, _ = strand_cells(m, T.r, 0.03)
            ax = axes[i, j]
            k = n // 2
            im = ax.imshow(m[:, :, k].T, origin="lower", cmap="viridis", vmin=0, vmax=0.5)
            ys, xs = np.nonzero(Z[:, :, k].T)
            ax.scatter(xs, ys, s=6, c="white", marker="s", linewidths=0)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(pid.replace("ax_", "").replace("_", " ") if pid != "c0" else "center", fontsize=9)
            if j == 0:
                ax.set_ylabel(seed, fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5, label="minor gap m = lambda_1 - lambda_0 (z = 0 slice; white: strand cells, m_th 0.03)")
    fig.suptitle("R32-1: the minor gap on the J = 0 states, center and lifts (n 24, L 36)", fontsize=10)
    os.makedirs(PLOTS, exist_ok=True)
    fig.savefig(os.path.join(PLOTS, "m5_32_r32_strands.png"), dpi=110)
    print("plot written")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "reads"
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if mode == "control":
        res["control"] = control()
    elif mode == "reads":
        res.update(reads())
    elif mode == "plot":
        plot(res)
    json.dump(res, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)
