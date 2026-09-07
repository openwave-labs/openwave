"""M5.32 R16-1: the statics of the author's object v4 (ledger 6.5 as amended): the R15-M relaxed
hedgehog (mu 1e-2, c_P 1) relaxed under the STATIC circle-averaged L_v4 (both quartic
completions), n32 L48 and n48 L72 (h 1.5: the L-exponents) plus n64 L48 (h 0.75: the
h-refinement of the core texture), then the end texture read: the biaxiality map and its
ring quadrupole (the R16-0 C7 instrument), the spin-weight-2 shell content of the split (C8),
the split profile, the director-in-plateau radius, the exterior on the vacuum, and the
instrument gates on the end field (the 8 -> 16 doubling test, the symmetry-defect gate with
the unaveraged regulator as the control).  The Morse index of the radial hedgehog in the
split sector is R16-2's operator applied to the seed (m5_32_r16_2_operator.py).

EQUATIONS: m5_32_r16_common.py (the object, every adjoint, the gates).  Descents with 4 circle
samples (an O(h^2)-level defect on the lattice, stated), every read with 8 (exact).
Verdict rule (pre-registered here before the first run): UNIAXIAL_RADIAL if max beta^2 < 0.1
on the end field; BIAXIAL_TORUS if max beta^2 >= 0.5 and the beta^2-weighted quadrupole of the
maximal shell is oblate (one negative eigenvalue below -0.08, two positive: a ring in a plane);
SPLIT_CORE if max beta^2 >= 0.5 and the quadrupole is prolate (one positive eigenvalue above
0.16, two negative: two lumps on an axis); BIAXIAL_OTHER otherwise (0.1 <= max beta^2 < 0.5, or
no clean quadrupole signature).  The Morse index qualifies the verdict from R16-2.

usage: python3 m5_32_r16_1_statics.py relax --n 32 --L 48 --comp rebuild --maxit 3000
       python3 m5_32_r16_1_statics.py collect
out:   checkpoints/m5_32_r16/r16_1_<comp>_n<n>_L<L>.npy (the end field, local), .json (the record),
       data/m5_32_r16_1.json (collect), plots/m5_32_r16_1_<comp>_n<n>_L<L>.png
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
import m5_32_r16_common as C                              # noqa: E402
import m5_32_r16_0_fields as F0                           # noqa: E402
C15, INS4 = C.C15, C.INS4
RES, DATA, PLOTS, CK = C.RES, C.DATA, C.PLOTS, C.CK
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def rel(p):
    return os.path.relpath(p, RES)


def tag_of(comp, n, L, kind="r15"):
    return f"r16_1_{comp}_n{n}_L{int(L)}" + ("" if kind == "r15" else "_" + kind)


def seed_for(n, L, cfg, kind="r15"):
    """the R15-M relaxed hedgehog on (n, L); n64 L48 = the n32 L48 field refined by cell-centered
    linear interpolation, the vacuum re-pinned on the shell; kind = 'analytic': the radial hedgehog
    of the degenerate vacuum (C15.seed_uniaxial, the R15-M seed itself) on any box."""
    if kind == "analytic":
        return C15.seed_uniaxial(cfg), "C15.seed_uniaxial (B8.dressed at zero rapidity on the degenerate vacuum)", "the analytic radial hedgehog, the R15-M seed family, unrelaxed"
    if (n, int(L)) in ((32, 48), (48, 72)):
        M, p = C.load_r15_seed(n, L)
        return M, rel(p), "R15-M relaxed hedgehog (mu 1e-2, c_P 1), the same box"
    from scipy.ndimage import zoom
    M32, p = C.load_r15_seed(32, 48)
    f = n / 32
    M = np.stack([np.stack([zoom(M32[..., a, b], f, order=1, mode="nearest", grid_mode=True) for b in range(4)], -1) for a in range(4)], -2)
    M = C.sym(M)
    pin = INS4.pin_shell(n, cfg["h"], 1.6)
    M[pin] = INS4.vac4(cfg)
    return M, rel(p), f"the R15-M n32 L48 field refined x{f:g} (cell-centered linear interpolation), the vacuum re-pinned on the shell"


# ------------------------------------------------ reads
def texture_reads(M, cfg):
    """the biaxiality map and its ring quadrupole (R16-0 C7), the split profile, the spin-2 shell content (C8)."""
    n, h, L = cfg["n"], cfg["h"], cfg["L"]
    X, Y, Z = INS4.coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    trip, lg, disc = F0.spatial_triple(M)
    b2, q3 = F0.biaxiality(trip)
    half = np.sqrt(np.maximum(disc, 0.0)) / 2.0
    edges = np.arange(0.0, L / 2 + h, 1.5 * h)
    shells = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if not np.any(m):
            continue
        shells.append({"r": [float(a), float(b)], "beta2_mean": float(np.mean(b2[m])), "beta2_max": float(np.max(b2[m])), "trQ3_mean": float(np.mean(q3[m])),
                       "half_split_mean": float(np.mean(half[m])), "half_split_max": float(np.max(half[m])), "triple_mean": [float(x) for x in np.mean(trip[m], axis=0)], "n_cells": int(np.sum(m))})
    idx = np.argsort(r, axis=None)[:8]
    center = {"r_max": float(np.sort(r, axis=None)[7]), "triple": [[float(x) for x in trip.reshape(-1, 3)[i]] for i in idx[:2]],
              "beta2": [float(b2.reshape(-1)[i]) for i in idx], "trQ3_sign": [float(np.sign(q3.reshape(-1)[i])) for i in idx]}
    imax = int(np.argmax([s_["beta2_mean"] for s_ in shells]))
    a, b = shells[imax]["r"]
    m = (r >= a) & (r < b)
    w = b2[m]
    xs = np.stack([X[m], Y[m], Z[m]], -1) / np.maximum(r[m], 1e-300)[:, None]
    Qd = np.einsum("c,ci,cj->ij", w, xs, xs) / max(np.sum(w), 1e-300) - np.eye(3) / 3.0
    ev, evec = np.linalg.eigh(Qd)
    m2 = b2 > 0.5 * np.max(b2)
    w2 = b2[m2]
    xs2 = np.stack([X[m2], Y[m2], Z[m2]], -1) / np.maximum(r[m2], 1e-300)[:, None]
    Qd2 = np.einsum("c,ci,cj->ij", w2, xs2, xs2) / max(np.sum(w2), 1e-300) - np.eye(3) / 3.0
    ev2, evec2 = np.linalg.eigh(Qd2)
    j = n // 2
    prof = {"r": [float(x) for x in X[j:, j, j]], "triple_x_axis": [[float(v) for v in trip[i, j, j]] for i in range(j, n)],
            "beta2_x_axis": [float(b2[i, j, j]) for i in range(j, n)], "beta2_diag": [float(b2[i, i, i]) for i in range(j, n)],
            "half_split_x_axis": [float(half[i, j, j]) for i in range(j, n)], "lambda_g_x_axis": [float(lg[i, j, j]) for i in range(j, n)]}
    # the spin-2 content on three shells around the beta^2 maximum
    zeta, gap, th, ph, rr = F0.frame_zeta(M, X, Y, Z)
    spin2 = {}
    rmax = float(np.mean(shells[imax]["r"]))
    for rc in (max(rmax - 1.5 * h, 1.0), rmax, rmax + 1.5 * h, 2.0 * rmax + 1.5 * h):
        m3 = (r >= rc - 0.75 * h) & (r < rc + 0.75 * h)
        if np.sum(m3) < 8:
            continue
        dec = F0.shell_decomp(zeta[m3], th[m3], ph[m3])
        P = {mm: abs(dec[mm]) ** 2 for mm in dec}
        tot = max(sum(P.values()), 1e-300)
        power = float(np.sum(np.abs(zeta[m3]) ** 2) * 4 * np.pi / np.sum(m3))
        spin2[f"{rc:.2f}"] = {"P_m": {str(mm): P[mm] for mm in sorted(P)}, "mean_m": float(sum(mm * P[mm] for mm in P) / tot), "l2_fraction_of_power": float(tot / max(power, 1e-300)),
                              "n_cells": int(np.sum(m3)), "zeta_rms": float(np.sqrt(np.mean(np.abs(zeta[m3]) ** 2)))}
    out = {"center": center, "beta2_global_max": float(np.max(b2)), "r_at_beta2_max": float(r.reshape(-1)[int(np.argmax(b2))]),
           "n_complex_pair_cells": int(np.sum(disc < -1e-12)), "half_split_max": float(np.max(half)), "r_at_half_split_max": float(r.reshape(-1)[int(np.argmax(half))]),
           "shells": shells, "max_shell": {"r": shells[imax]["r"], "beta2_mean": shells[imax]["beta2_mean"], "quadrupole_eigenvalues": [float(x) for x in ev],
                                           "ring_axis": [float(x) for x in evec[:, 0]], "prolate_axis": [float(x) for x in evec[:, 2]]},
           "half_max_region": {"n_cells": int(np.sum(m2)), "r_mean": float(np.mean(r[m2])), "r_std": float(np.std(r[m2])), "quadrupole_eigenvalues": [float(x) for x in ev2]},
           "profiles": prof, "spin2_shells": spin2, "director_gap_min": float(np.min(gap))}
    # the verdict rule (pre-registered in the docstring)
    bmax = out["beta2_global_max"]
    if bmax < 0.1:
        v = "UNIAXIAL_RADIAL"
    elif bmax >= 0.5 and ev[0] < -0.08 and ev[1] > 0 and ev[2] > 0:
        v = "BIAXIAL_TORUS"
    elif bmax >= 0.5 and ev[2] > 0.16 and ev[0] < 0 and ev[1] < 0:
        v = "SPLIT_CORE"
    else:
        v = "BIAXIAL_OTHER"
    out["texture_verdict"] = v
    return out, b2, trip, half


def exterior_read(M, cfg):
    n, h, L = cfg["n"], cfg["h"], cfg["L"]
    X, Y, Z = INS4.coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    dev = np.max(np.abs(M - INS4.vac4(cfg)), axis=(-1, -2))
    out = {}
    for f in (0.25, 0.35, 0.45):
        m = r > f * L
        out[f"max_dev_from_vacuum_r_gt_{f:g}L"] = float(np.max(dev[m])) if np.any(m) else None
    # r^-2 tail exponent of the deviation on shells between 0.15 L and 0.4 L
    rs, ds = [], []
    for a in np.arange(0.15 * L, 0.4 * L, 2 * h):
        m = (r >= a) & (r < a + 2 * h)
        if np.any(m):
            rs.append(a + h); ds.append(np.mean(dev[m]))
    rs, ds = np.array(rs), np.array(ds)
    if len(rs) > 3 and np.all(ds > 0):
        sl = np.polyfit(np.log(rs), np.log(ds), 1)[0]
        out["tail_exponent_of_mean_deviation"] = float(sl)
    return out


def instrument_gates(M, cfg, nref):
    """on the end field: the 8 -> 16 doubling test, the symmetry-defect gate at 8 samples with
    the unaveraged regulator as the control, both completions' static reads."""
    fr = C.frame(M, nref)
    a0 = C.a0_of(M, fr)
    reads = {}
    for comp in ("rebuild", "norm"):
        cf = dict(cfg); cf["completion"] = comp
        p8 = C.averaged(M, cf, a0, need_grad=False, n_ref=nref, n_samples=8)["parts"]
        p16 = C.averaged(M, cf, a0, need_grad=False, n_ref=nref, n_samples=16)["parts"]
        p4 = C.averaged(M, cf, a0, need_grad=False, n_ref=nref, n_samples=4)["parts"]
        reads[comp] = {"parts_8": p8, "doubling_8_16_rel": {k: abs(p8[k] - p16[k]) / max(abs(p16[k]), 1e-300) for k in p8},
                       "descent_4_vs_8_rel": {k: abs(p4[k] - p8[k]) / max(abs(p8[k]), 1e-300) for k in p8}}
    cf = dict(cfg); cf["n_samples"] = 8
    defects = {}
    for beta in (0.4, 1.1):
        Rb = C.rot_R(fr["J"], beta)
        Mb = Rb @ M @ np.swapaxes(Rb, -1, -2)
        pa = C.averaged(M, cf, a0, need_grad=False, n_ref=nref)["parts"]
        pb = C.averaged(Mb, cf, C.a0_of(Mb, n_ref=nref), need_grad=False, n_ref=nref)["parts"]
        ra = C.action(M, cf, a0, need_grad=False, n_ref=nref)["parts"]
        rb = C.action(Mb, cf, C.a0_of(Mb, n_ref=nref), need_grad=False, n_ref=nref)["parts"]
        defects[str(beta)] = {"averaged": {k: abs(pa[k] - pb[k]) / max(abs(pa[k]), 1e-300) for k in pa}, "unaveraged": {k: abs(ra[k] - rb[k]) / max(abs(ra[k]), 1e-300) for k in ra}}
    worst = max(v for b in defects for k, v in defects[b]["averaged"].items() if k in ("E_stat", "kin_tot", "E_h", "KP", "reg", "V4", "U"))
    ctrl = min(defects[b]["unaveraged"]["reg"] for b in defects)
    worst_dbl = max(reads[c]["doubling_8_16_rel"][k] for c in reads for k in ("E_stat", "kin_tot", "E_h", "KP", "reg"))
    return {"reads": reads, "symmetry_defects": defects, "symmetry_gate_worst_rel": worst, "symmetry_gate_pass_1e-10": bool(worst < 1e-10),
            "control_unaveraged_reg_defect": ctrl, "control_fails_as_required": bool(ctrl > 1e-6),
            "doubling_8_16_worst_rel": worst_dbl, "doubling_gate_pass_1e-12": bool(worst_dbl < 1e-12), "domain": C.domain(fr, cfg)}


def make_diag(cfg):
    n, h, L = cfg["n"], cfg["h"], cfg["L"]
    X, Y, Z = INS4.coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    outer = r > 0.4 * L

    def diag(M, fr):
        trip, lg, disc = F0.spatial_triple(M)
        b2, _ = F0.biaxiality(trip)
        half = np.sqrt(np.maximum(disc, 0.0)) / 2.0
        dev = np.max(np.abs(M - INS4.vac4(cfg)), axis=(-1, -2))
        return {"beta2_max": float(np.max(b2)), "r_beta2_max": float(r.reshape(-1)[int(np.argmax(b2))]), "half_split_max": float(np.max(half)),
                "r_half_split_max": float(r.reshape(-1)[int(np.argmax(half))]), "exterior_dev_max": float(np.max(dev[outer]))}
    return diag


def plot_run(tag, b2, trip, half, cfg, info, rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n, h = cfg["n"], cfg["h"]
    X, Y, Z = INS4.coords(n, h)
    j = n // 2
    ext = [-n * h / 2, n * h / 2, -n * h / 2, n * h / 2]
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    im = ax[0, 0].imshow(b2[:, :, j].T, origin="lower", extent=ext, vmin=0, vmax=1, cmap="viridis"); ax[0, 0].set_title(f"{tag}: beta^2, plane z = {Z[0, 0, j]:.2f}", fontsize=8); ax[0, 0].set_xlim(-12, 12); ax[0, 0].set_ylim(-12, 12)
    ax[0, 1].imshow(b2[j, :, :].T, origin="lower", extent=ext, vmin=0, vmax=1, cmap="viridis"); ax[0, 1].set_title("beta^2, plane x (y horizontal, z vertical)", fontsize=8); ax[0, 1].set_xlim(-12, 12); ax[0, 1].set_ylim(-12, 12)
    fig.colorbar(im, ax=ax[0, :2].ravel().tolist(), shrink=0.7)
    im2 = ax[0, 2].imshow(half[:, :, j].T, origin="lower", extent=ext, cmap="magma"); ax[0, 2].set_title("half split (lambda_2 - lambda_3)/2, plane z", fontsize=8); ax[0, 2].set_xlim(-12, 12); ax[0, 2].set_ylim(-12, 12); fig.colorbar(im2, ax=ax[0, 2], shrink=0.7)
    pr = rec["texture"]["profiles"]
    ax[1, 0].plot(pr["r"], np.array(pr["triple_x_axis"]), label=["l1", "l2", "l3"]); ax[1, 0].axhline(C.DELTA, color="k", lw=0.5, ls="--"); ax[1, 0].axhline(1.0, color="k", lw=0.5, ls="--"); ax[1, 0].axhline(C.PL_HI, color="r", lw=0.5, ls=":"); ax[1, 0].set_title("spatial triple on the x axis (red: the plateau edge 0.8)", fontsize=8); ax[1, 0].set_xlabel("r"); ax[1, 0].legend(fontsize=6)
    sh = rec["texture"]["shells"]
    ax[1, 1].plot([np.mean(s_["r"]) for s_ in sh], [s_["beta2_mean"] for s_ in sh], "o-", ms=3, label="beta^2 shell mean"); ax[1, 1].plot([np.mean(s_["r"]) for s_ in sh], [s_["half_split_max"] for s_ in sh], "x--", ms=3, label="half split shell max"); ax[1, 1].set_xlabel("r"); ax[1, 1].legend(fontsize=6); ax[1, 1].set_xlim(0, 24); ax[1, 1].set_title(f"verdict {rec['texture']['texture_verdict']}", fontsize=8)
    tr = info["trace"]
    ax[1, 2].plot([t["it"] for t in tr], [t["E_stat"] for t in tr], label="E_stat"); ax[1, 2].set_xlabel("it"); ax[1, 2].set_ylabel("E_stat"); ax2 = ax[1, 2].twinx(); ax2.semilogy([t["it"] for t in tr], [t["fmax"] for t in tr], "r-", lw=0.7, label="fmax"); ax[1, 2].set_title(f"descent (stop {info['stop']}, {info['wall_s']:.0f} s)", fontsize=8); ax[1, 2].legend(fontsize=6, loc="upper right"); ax2.legend(fontsize=6, loc="center right")
    p = os.path.join(PLOTS, f"m5_32_{tag}.png")
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return rel(p)


# ------------------------------------------------ relax
def relax(n, L, comp, maxit, kind="r15"):
    tag = tag_of(comp, n, L, kind)
    cfg = C.cfg_v4(n, L, completion=comp, n_samples=4)
    M0, src, how = seed_for(n, L, cfg, kind)
    nref = C.radial_ref(cfg)
    free = ~INS4.pin_shell(n, cfg["h"], 1.6)
    log(f"{tag}: seed {src} ({how}); h {cfg['h']}, cells {n ** 3}, free {int(np.sum(free))}")
    rec = {"tag": tag, "n": n, "L": L, "h": cfg["h"], "completion": comp, "cfg": {k: cfg[k] for k in ("mu", "cP", "cs", "n_samples", "stencil")}, "seed": {"source": src, "how": how}}
    t = time.time()
    rec["seed"]["gates"] = instrument_gates(M0, cfg, nref)
    rec["seed"]["texture"], _, _, _ = texture_reads(M0, cfg)
    rec["seed"]["exterior"] = exterior_read(M0, cfg)
    log(f"  seed reads {time.time() - t:.0f} s: parts_8 {rec['seed']['gates']['reads'][comp]['parts_8']}; domain {rec['seed']['gates']['domain']}; texture {rec['seed']['texture']['texture_verdict']} beta2 max {rec['seed']['texture']['beta2_global_max']:.3f}")
    json.dump(rec, open(os.path.join(CK, tag + ".json"), "w"), indent=1, default=float)
    ckp = os.path.join(CK, tag + ".npy")
    M, info = C.fire_v4(M0, cfg, free, maxit, K=None, n_ref=nref, log_every=100, tag=tag, diag=make_diag(cfg), ck_path=ckp, ck_every=200)
    rec["descent"] = {k: info[k] for k in ("stop", "wall_s", "iters")}
    rec["trace"] = info["trace"]
    np.save(ckp, M)
    log(f"  descent {info['stop']} after {info['iters']} it, {info['wall_s']:.0f} s; end-field reads")
    t = time.time()
    rec["gates"] = instrument_gates(M, cfg, info["n_ref"])
    rec["texture"], b2, trip, half = texture_reads(M, cfg)
    rec["exterior"] = exterior_read(M, cfg)
    rec["end_field"] = rel(ckp)
    rec["plot"] = plot_run(tag, b2, trip, half, cfg, info, rec)
    rec["seed_to_end"] = {"E_stat_seed_8": rec["seed"]["gates"]["reads"][comp]["parts_8"]["E_stat"], "E_stat_end_8": rec["gates"]["reads"][comp]["parts_8"]["E_stat"],
                          "max_abs_change": float(np.max(np.abs(M - M0)))}
    log(f"  end reads {time.time() - t:.0f} s: parts_8 {rec['gates']['reads'][comp]['parts_8']}; domain {rec['gates']['domain']}; texture {rec['texture']['texture_verdict']} beta2 max {rec['texture']['beta2_global_max']:.3f} at r {rec['texture']['r_at_beta2_max']:.2f}; "
        f"quadrupole {rec['texture']['max_shell']['quadrupole_eigenvalues']}; half split max {rec['texture']['half_split_max']:.4f}; gates: symmetry {rec['gates']['symmetry_gate_pass_1e-10']} ({rec['gates']['symmetry_gate_worst_rel']:.1e}), doubling {rec['gates']['doubling_gate_pass_1e-12']} ({rec['gates']['doubling_8_16_worst_rel']:.1e}), control fails {rec['gates']['control_fails_as_required']}")
    json.dump(rec, open(os.path.join(CK, tag + ".json"), "w"), indent=1, default=float)
    log(f"  written {rel(os.path.join(CK, tag + '.json'))}")
    return rec


def spectral_exterior(M, cfg):
    """the exterior read that is blind to the hedgehog's frame: the spatial triple against (1, delta, delta)
    on shells (the R15 hedgehog texture is a frame rotation, its M deviates from diag(g, 1, delta, delta)
    everywhere; only the SPECTRUM must return to the vacuum)."""
    n, h, L = cfg["n"], cfg["h"], cfg["L"]
    X, Y, Z = INS4.coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    trip, lg, disc = F0.spatial_triple(M)
    dev = np.max(np.abs(trip - np.array([1.0, C.DELTA, C.DELTA])), axis=-1)
    dev = np.maximum(dev, np.abs(lg + C.G))
    out = {}
    for f in (0.25, 0.35, 0.45):
        m = r > f * L
        out[f"max_spectral_dev_r_gt_{f:g}L"] = float(np.max(dev[m]))
    rs, ds = [], []
    for a in np.arange(0.12 * L, 0.42 * L, 2 * h):
        m = (r >= a) & (r < a + 2 * h)
        if np.any(m):
            rs.append(a + h); ds.append(np.mean(dev[m]))
    rs, ds = np.array(rs), np.array(ds)
    if len(rs) > 3 and np.all(ds > 0):
        out["tail_exponent_of_mean_spectral_dev"] = float(np.polyfit(np.log(rs), np.log(ds), 1)[0])
    out["shells_mean_spectral_dev"] = [[float(a), float(b)] for a, b in zip(rs, ds)]
    return out


def collect():
    out = {"rung": "R16-1", "runs": {}, "L_exponents": {}, "h_refinement": {}}
    for comp in ("rebuild", "norm"):
        for n, L in ((32, 48), (48, 72), (64, 48)):
            for kind in ("r15", "analytic"):
                p = os.path.join(CK, tag_of(comp, n, L, kind) + ".json")
                if os.path.exists(p):
                    r = json.load(open(p))
                    r.pop("trace", None)
                    fp = os.path.join(CK, tag_of(comp, n, L, kind) + ".npy")
                    if "gates" in r and os.path.exists(fp):
                        r["exterior_spectral"] = spectral_exterior(np.load(fp), C.cfg_v4(n, L, completion=comp))
                    out["runs"][tag_of(comp, n, L, kind)] = r
        a, b = out["runs"].get(tag_of(comp, 32, 48)), out["runs"].get(tag_of(comp, 48, 72))
        if a and b and "gates" in a and "gates" in b:
            pa, pb = a["gates"]["reads"][comp]["parts_8"], b["gates"]["reads"][comp]["parts_8"]
            out["L_exponents"][comp] = {k: float(np.log(pb[k] / pa[k]) / np.log(72.0 / 48.0)) if pa[k] > 0 and pb[k] > 0 else None for k in ("E_h", "KP", "E_stat", "reg", "U", "V4")}
        c = out["runs"].get(tag_of(comp, 64, 48))
        if not (c and "texture" in c):
            c = out["runs"].get(tag_of(comp, 64, 48, "analytic"))
        if a and c and "texture" in a and "texture" in c:
            out["h_refinement"][comp] = {"n32": {k: a["texture"][k] for k in ("beta2_global_max", "r_at_beta2_max", "texture_verdict", "half_split_max")} | {"quadrupole": a["texture"]["max_shell"]["quadrupole_eigenvalues"], "ring_axis": a["texture"]["max_shell"]["ring_axis"], "r_shell": a["texture"]["max_shell"]["r"]},
                                         "n64": {k: c["texture"][k] for k in ("beta2_global_max", "r_at_beta2_max", "texture_verdict", "half_split_max")} | {"quadrupole": c["texture"]["max_shell"]["quadrupole_eigenvalues"], "ring_axis": c["texture"]["max_shell"]["ring_axis"], "r_shell": c["texture"]["max_shell"]["r"]}}
    out["verdicts"] = {t: r.get("texture", {}).get("texture_verdict") for t, r in out["runs"].items()}
    out["descents"] = {t: r.get("descent") for t, r in out["runs"].items()}
    out["gates"] = {t: {k: r["gates"][k] for k in ("symmetry_gate_pass_1e-10", "symmetry_gate_worst_rel", "doubling_gate_pass_1e-12", "doubling_8_16_worst_rel", "control_fails_as_required")} for t, r in out["runs"].items() if "gates" in r}
    json.dump(out, open(os.path.join(DATA, "m5_32_r16_1.json"), "w"), indent=1, default=float)
    log(f"collected {len(out['runs'])} runs: verdicts {out['verdicts']}; L-exponents {out['L_exponents']}; gates {out['gates']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["relax", "collect"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--L", type=float, default=48.0)
    ap.add_argument("--comp", default="rebuild")
    ap.add_argument("--maxit", type=int, default=3000)
    ap.add_argument("--seed", default="r15", choices=["r15", "analytic"])
    a = ap.parse_args(ARGS)
    if a.mode == "relax":
        relax(a.n, a.L, a.comp, a.maxit, a.seed)
    else:
        collect()
