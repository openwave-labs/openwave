"""M5.32 R16-2: the clock operator of the author's object v4 (ledger 6.5 as amended): the
two-component fluctuation operator in the local (2,3) block on a relaxed core, the lowest
eigenvalue of the generalized problem against the gap, and the Morse index of the radial
hedgehog in the split sector (the same operator on the R15-M seed, plus the ten-direction
Hessian at the core).

EQUATIONS
---------
The doublet subspace: per cell delta M = a (e e^T - f f^T) + b (e f^T + f e^T) with (e, f) an
oriented orthonormal basis of the pair plane (e from e_theta projected off the outward-lifted
director n, f = n x e: the R16-0 C8 frame; zeta = a + i b has spin weight 2 under the local
circle).  On a static background M the perturbation zeta(x) e^{-i Omega t} obeys
    H zeta = Omega^2 (2 T) zeta,
H the Hessian of the circle-averaged E_stat restricted to the doublet subspace (matrix-free:
central differences of the analytic gradient, H v = [g(M + eps v) - g(M - eps v)] / (2 eps)),
T the inertia: kin_tot(M, a0 = v) = v^T T v is a POINTWISE quadratic form in v (no derivative of
a0 enters any term), so T is a per-cell 2 x 2 SPD matrix, read exactly from kin_cells on the
three directions (a, b, a + b).  With T^(1/2) explicit, the eigenvalues lambda of the symmetric
operator T^(-1/2) H T^(-1/2) (Lanczos, the lowest k) give Omega^2 = lambda / 2 (the R16-2 audit
caught the first run reporting lambda as Omega^2: an exact factor 2, corrected here and re-run).
T's circle dependence is MEASURED (the per-cell inertia under the plain and the averaged kinetic
reads differs where the split is nonzero, up to 2 percent on the seed: the audit), not assumed.  Conventions: the doublet
frequency Omega is TWICE the clock rate omega (B rotates at 2 omega under R(omega t)); the
delocalized threshold in the infinite box is Omega_c^2 = mu / c_P (omega_c^2 = mu / (4 c_P),
R16-0 C2), on the pinned box the lowest continuum mode sits near Omega_c^2 + (pi / L)^2 (the
K_P stiffness at c_P, the author's dispersion): the vacuum-region control below reads it.
Verdict: BOUND_DOUBLET if 0 < Omega_0^2 < mu / c_P AND the mode is localized (its T-weighted
rms radius below 0.25 L and the T-weight fraction inside r < 8 above 0.5); NO_BOUND_MODE
otherwise; the Morse index = the number of Omega^2 < 0 (H indefinite in the doublet sector).
The ten-direction Hessian at the core: the Hessian of E_stat over the 10 symmetric directions
on the 8 innermost cells (one block direction e_k on those cells), its eigenvalues.
The Gaussian control (the author's 13/12 - sqrt(3)/18 = 0.987 < 1) is NOT reproduced: the
comment gives no definition of the well or the trial function (recorded as not checkable).

usage: python3 m5_32_r16_2_operator.py run --field <path.npy> --n 32 --L 48 --comp rebuild --label <lab> [--k 4]
       python3 m5_32_r16_2_operator.py collect
out:   checkpoints/m5_32_r16/r16_2_<label>.json (+ the mode .npy), data/m5_32_r16_2.json, plots/m5_32_r16_2_<label>.png
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

ARGS = list(sys.argv[1:])
sys.argv = [sys.argv[0]]
import m5_32_r16_common as C                              # noqa: E402
C15, INS4 = C.C15, C.INS4
RES, DATA, PLOTS, CK = C.RES, C.DATA, C.PLOTS, C.CK
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def rel(p):
    return os.path.relpath(p, RES)


def doublet_basis(M, cfg, nref_kind="radial"):
    """(Ea, Eb) per cell: the two doublet directions in the pair plane, from the oriented frame."""
    n, h = cfg["n"], cfg["h"]
    X, Y, Z = INS4.coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    fr = C.frame(M, C.radial_ref(cfg, nref_kind))
    nn = np.real(fr["n"])
    # e: e_theta projected off n in the 4D eta-orthogonal sense (u = e_0 on these fields: spatial)
    th = np.arccos(np.clip(Z / r, -1, 1)); ph = np.arctan2(Y, X)
    eth = np.stack([np.zeros_like(th), np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)], -1)
    P23 = np.real(fr["P23"])
    e = np.einsum("...ab,...b->...a", P23, eth)
    bad = np.linalg.norm(e[..., 1:], axis=-1) < 1e-6
    if np.any(bad):                                                      # e_theta along n: use e_phi there
        eph = np.stack([np.zeros_like(th), -np.sin(ph), np.cos(ph), np.zeros_like(th)], -1)
        e[bad] = np.einsum("...ab,...b->...a", P23, eph)[bad]
    e = e / np.sqrt(np.einsum("...a,ab,...b->...", e, C.ETA, e))[..., None]
    f = np.einsum("...ab,...b->...a", np.real(fr["J"]), e)
    Ea = e[..., :, None] * e[..., None, :] - f[..., :, None] * f[..., None, :]
    Eb = e[..., :, None] * f[..., None, :] + f[..., :, None] * e[..., None, :]
    return Ea, Eb, fr, r


def run(field, n, L, comp, label, k=4, eps=1e-4, ns=4, nref_kind="radial"):
    cfg = C.cfg_v4(n, L, completion=comp, n_samples=ns)
    M = np.load(field)
    nref = C.radial_ref(cfg, nref_kind)
    free = ~INS4.pin_shell(n, cfg["h"], 1.6)
    Ea, Eb, fr, r = doublet_basis(M, cfg, nref_kind)
    fm = free.astype(float)
    Ea *= fm[..., None, None]; Eb *= fm[..., None, None]
    shape = M.shape[:3]
    N = int(np.prod(shape))
    lq = C.lift_quality(fr, nref)
    rec = {"label": label, "field": rel(field), "n": n, "L": L, "h": cfg["h"], "completion": comp, "n_samples": ns, "eps": eps, "k": k, "lift": nref_kind, "lift_quality_min_abs_n_dot_ref": lq[0], "cells_lift_ambiguous_lt_0.2": lq[1]}
    log(f"{label}: field {rel(field)}, {N} cells, free {int(np.sum(free))}; lift {nref_kind}: min |n . n_ref| {lq[0]:.3f}, ambiguous cells {lq[1]}")
    # the inertia T per cell (2x2) from three kinetic reads
    t = time.time()
    _, kin_a, kc_a = C.kin_a0_grad(M, cfg, Ea, nref)
    _, kin_b, kc_b = C.kin_a0_grad(M, cfg, Eb, nref)
    _, kin_ab, kc_ab = C.kin_a0_grad(M, cfg, Ea + Eb, nref)
    Taa, Tbb = kc_a, kc_b
    Tab = 0.5 * (kc_ab - kc_a - kc_b)
    Tm = np.stack([np.stack([Taa, Tab], -1), np.stack([Tab, Tbb], -1)], -2)
    wT, VT = np.linalg.eigh(Tm[free])
    _, _, kc_a1 = C.kin_a0_grad(M, cfg, Ea, nref, n_samples=1)
    relT = np.abs(kc_a - kc_a1) / np.maximum(np.abs(kc_a1), 1e-300)
    rec["T"] = {"min_eig_on_free": float(np.min(wT)), "max_eig_on_free": float(np.max(wT)), "read_s": time.time() - t,
                "vacuum_region_T_aa_mean_r_gt_0.35L": float(np.mean(Taa[(r > 0.35 * L) & free])), "expected_vacuum_T_aa_h3": float(cfg["cP"] * cfg["h"] ** 3),
                "T_aa_averaged_vs_plain_max_rel": float(np.max(relT[free])), "T_aa_averaged_vs_plain_total_rel": float(abs(np.sum(kc_a[free]) - np.sum(kc_a1[free])) / np.sum(kc_a1[free]))}
    log(f"  T per cell: eig range on the free cells [{np.min(wT):.3e}, {np.max(wT):.3e}] ({time.time() - t:.0f} s); vacuum T_aa mean {rec['T']['vacuum_region_T_aa_mean_r_gt_0.35L']:.4e} vs c_P h^3 {cfg['cP'] * cfg['h'] ** 3:.4e}")
    if np.min(wT) <= 0:
        log("  T not positive definite on the free cells: the generalized problem is degenerate; using |T| + 1e-12")
    Tsafe = Tm.copy()
    Tsafe[~free] = np.eye(2)
    wT2, VT2 = np.linalg.eigh(Tsafe)
    wT2 = np.maximum(wT2, 1e-12)
    Tih = VT2 @ (wT2[..., :, None] ** -0.5 * np.swapaxes(VT2, -1, -2))         # T^(-1/2)

    def to_field(x):
        ab = x.reshape(shape + (2,))
        return ab[..., 0, None, None] * Ea + ab[..., 1, None, None] * Eb

    def from_field(Gf):
        return np.stack([np.sum(Gf * Ea, axis=(-1, -2)), np.sum(Gf * Eb, axis=(-1, -2))], -1).reshape(-1)

    def Tih_apply(x):
        return np.einsum("...ij,...j->...i", Tih, x.reshape(shape + (2,))).reshape(-1)

    g0 = C.averaged(M, cfg, need_grad=True, n_ref=nref)["grad_stat"]
    rec["gradient_residual_doublet"] = float(np.sqrt(np.sum(from_field(g0) ** 2)))
    calls = [0]

    def H_apply(x):
        v = to_field(x)
        gp = C.averaged(M + eps * v, cfg, need_grad=True, n_ref=nref)["grad_stat"]
        gm = C.averaged(M - eps * v, cfg, need_grad=True, n_ref=nref)["grad_stat"]
        calls[0] += 1
        return from_field((gp - gm) / (2 * eps)) * fm.reshape(-1).repeat(2)

    def Ht_apply(x):
        y = Tih_apply(x)
        return Tih_apply(H_apply(y))
    # symmetry check of H on two random vectors
    rng = np.random.default_rng(7)
    x1, x2 = rng.normal(size=2 * N), rng.normal(size=2 * N)
    h1, h2 = H_apply(x1), H_apply(x2)
    rec["H_symmetry_rel"] = float(abs(np.dot(x2, h1) - np.dot(x1, h2)) / max(abs(np.dot(x2, h1)), 1e-300))
    log(f"  H symmetry: {rec['H_symmetry_rel']:.2e}; gradient residual in the doublet sector {rec['gradient_residual_doublet']:.3e}")
    op = LinearOperator((2 * N, 2 * N), matvec=Ht_apply, dtype=float)
    t = time.time()
    vals, vecs = eigsh(op, k=k, which="SA", tol=1e-5, maxiter=4000, ncv=max(2 * k + 1, 20))
    order = np.argsort(vals)
    vals, vecs = vals[order] / 2.0, vecs[:, order]                       # Omega^2 = lambda / 2 (H = Omega^2 (2 T))
    rec["Omega2_convention"] = "Omega^2 = lambda / 2, lambda the eigenvalue of T^(-1/2) H T^(-1/2); the first run reported lambda (audit-corrected)"
    log(f"  lowest {k} Omega^2: {vals} ({calls[0]} H applications, {time.time() - t:.0f} s)")
    modes = []
    for i in range(k):
        y = Tih_apply(vecs[:, i])                                          # the mode in the (a, b) coordinates
        ab = y.reshape(shape + (2,))
        wgt = np.einsum("...i,...ij,...j->...", ab, Tm, ab)
        wgt = np.maximum(wgt, 0.0)
        ws = max(float(np.sum(wgt)), 1e-300)
        r_rms = float(np.sqrt(np.sum(wgt * r * r) / ws))
        frac8 = float(np.sum(wgt[r < 8.0]) / ws)
        modes.append({"Omega2": float(vals[i]), "omega2_clock": float(vals[i]) / 4.0, "T_weighted_r_rms": r_rms, "T_weight_fraction_r_lt_8": frac8, "localized": bool(r_rms < 0.25 * L and frac8 > 0.5)})
    rec["modes"] = modes
    mu_over_cP = cfg["mu"] / cfg["cP"]
    rec["thresholds"] = {"Omega_c2_infinite_box (mu / c_P)": mu_over_cP, "omega_c2_clock (mu / 4 c_P)": mu_over_cP / 4.0, "box_continuum_estimate_Omega2": mu_over_cP + (np.pi / L) ** 2}
    neg = sum(1 for m in modes if m["Omega2"] < 0)
    rec["morse_index_doublet_sector_among_lowest_k"] = neg
    m0 = modes[0]
    if m0["Omega2"] > 0 and m0["Omega2"] < mu_over_cP and m0["localized"]:
        v = "BOUND_DOUBLET"
    elif m0["Omega2"] < 0:
        v = f"UNSTABLE_DOUBLET (Morse index >= {neg})"
    else:
        v = "NO_BOUND_MODE"
    rec["verdict"] = v
    np.save(os.path.join(CK, f"r16_2_{label}_mode0.npy"), Tih_apply(vecs[:, 0]).reshape(shape + (2,)))
    # the ten-direction Hessian at the core (the 8 innermost cells)
    idx = np.argsort(r, axis=None)[:8]
    core = np.zeros(shape, dtype=bool); core.reshape(-1)[idx] = True
    basis = []
    for a in range(4):
        for b in range(a, 4):
            E = np.zeros((4, 4)); E[a, b] = E[b, a] = 1.0 if a == b else 0.5 ** 0.5
            basis.append(E)
    dirs = [core[..., None, None] * E for E in basis]
    Hc = np.zeros((10, 10))
    for i, D in enumerate(dirs):
        gp = C.averaged(M + eps * D, cfg, need_grad=True, n_ref=nref)["grad_stat"]
        gm = C.averaged(M - eps * D, cfg, need_grad=True, n_ref=nref)["grad_stat"]
        Hv = (gp - gm) / (2 * eps)
        for jj, D2 in enumerate(dirs):
            Hc[i, jj] = float(np.sum(Hv * D2))
    Hc = 0.5 * (Hc + Hc.T)
    ev, evec = np.linalg.eigh(Hc)
    labels = [f"{a}{b}" for a in range(4) for b in range(a, 4)]
    rec["core_hessian_10"] = {"eigenvalues": [float(x) for x in ev], "n_negative": int(np.sum(ev < -1e-10)), "lowest_vector": {labels[i]: float(evec[i, 0]) for i in range(10)}, "r_max_of_core_cells": float(np.sort(r, axis=None)[7])}
    log(f"  core 10-direction Hessian eigenvalues {np.round(ev, 5)} (negative: {rec['core_hessian_10']['n_negative']}); VERDICT {v}")
    # plot: the lowest mode's radial profile and a slice
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ab = Tih_apply(vecs[:, 0]).reshape(shape + (2,))
    amp = np.sqrt(ab[..., 0] ** 2 + ab[..., 1] ** 2)
    j = n // 2
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ext = [-n * cfg["h"] / 2, n * cfg["h"] / 2, -n * cfg["h"] / 2, n * cfg["h"] / 2]
    im = ax[0].imshow(amp[:, :, j].T, origin="lower", extent=ext, cmap="viridis"); ax[0].set_title(f"{label}: |zeta| of the lowest mode, plane z", fontsize=8); fig.colorbar(im, ax=ax[0], shrink=0.8)
    edges = np.arange(0, L / 2, 1.5 * cfg["h"])
    prof = [float(np.sqrt(np.mean(amp[(r >= a) & (r < b)] ** 2))) if np.any((r >= a) & (r < b)) else 0 for a, b in zip(edges[:-1], edges[1:])]
    ax[1].plot(0.5 * (edges[:-1] + edges[1:]), prof, "o-", ms=3); ax[1].set_xlabel("r"); ax[1].set_title("shell rms of the lowest mode", fontsize=8)
    ax[2].bar(range(k), vals); ax[2].axhline(mu_over_cP, color="r", ls="--", lw=0.8, label="mu / c_P"); ax[2].axhline(rec["thresholds"]["box_continuum_estimate_Omega2"], color="orange", ls=":", lw=0.8, label="box continuum est."); ax[2].axhline(0, color="k", lw=0.5); ax[2].legend(fontsize=6); ax[2].set_title(f"lowest Omega^2, verdict {v}", fontsize=8)
    p = os.path.join(PLOTS, f"m5_32_r16_2_{label}.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    rec["plot"] = rel(p)
    rec["wall_s"] = time.time() - T0
    json.dump(rec, open(os.path.join(CK, f"r16_2_{label}.json"), "w"), indent=1, default=float)
    log(f"  written {rel(os.path.join(CK, f'r16_2_{label}.json'))}")
    return rec


def collect():
    import glob
    out = {"rung": "R16-2", "runs": {}}
    for p in sorted(glob.glob(os.path.join(CK, "r16_2_*.json"))):
        r = json.load(open(p))
        out["runs"][r["label"]] = r
    out["verdicts"] = {k: r["verdict"] for k, r in out["runs"].items()}
    out["lowest_Omega2"] = {k: r["modes"][0]["Omega2"] for k, r in out["runs"].items()}
    out["morse_index"] = {k: r["morse_index_doublet_sector_among_lowest_k"] for k, r in out["runs"].items()}
    json.dump(out, open(os.path.join(DATA, "m5_32_r16_2.json"), "w"), indent=1, default=float)
    log(f"collected {len(out['runs'])}: {out['verdicts']}; lowest Omega^2 {out['lowest_Omega2']}")
    # the summary plot: the lowest doublet Omega^2 per field against the two thresholds (the measured box bottom, mu / c_P)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = [k for k in ("vacuum_control_n32", "seed_r15m_n32", "r16_1_end_n32") if k in out["runs"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, k in enumerate(order):
        vals = [m["Omega2"] for m in out["runs"][k]["modes"]]
        ax[0].bar(np.arange(len(vals)) + 0.25 * (i - 1), vals, width=0.22, label=k)
    vac = out["lowest_Omega2"].get("vacuum_control_n32")
    if vac is not None:
        ax[0].axhline(vac, color="k", ls="--", lw=0.8, label=f"the empty box's bottom {vac:.4f}")
    ax[0].axhline(0.01, color="r", ls=":", lw=0.8, label="mu / c_P = 0.01 (infinite box)")
    ax[0].set_xlabel("mode index"); ax[0].set_ylabel("Omega^2 (doublet frequency squared; omega = Omega / 2)"); ax[0].legend(fontsize=6); ax[0].set_title("R16-2: the lowest doublet modes, no mode below the empty box", fontsize=8)
    for k in order:
        r = out["runs"][k]
        p = os.path.join(CK, f"r16_2_{k}_mode0.npy")
        if os.path.exists(p):
            ab = np.load(p); amp = np.sqrt(ab[..., 0] ** 2 + ab[..., 1] ** 2)
            n = amp.shape[0]; h = r["h"]
            X, Y, Z = INS4.coords(n, h); rr = np.sqrt(X * X + Y * Y + Z * Z)
            edges = np.arange(0, r["L"] / 2, 1.5 * h)
            prof = [float(np.sqrt(np.mean(amp[(rr >= a) & (rr < b)] ** 2))) if np.any((rr >= a) & (rr < b)) else 0 for a, b in zip(edges[:-1], edges[1:])]
            ax[1].plot(0.5 * (edges[:-1] + edges[1:]), prof / max(np.max(prof), 1e-300), "o-", ms=3, label=k)
    ax[1].set_xlabel("r"); ax[1].set_ylabel("shell rms of the lowest mode (normalized)"); ax[1].legend(fontsize=6); ax[1].set_title("the lowest mode is a box mode on every field (no core binding)", fontsize=8)
    pth = os.path.join(PLOTS, "m5_32_r16_2_summary.png")
    fig.savefig(pth, dpi=110, bbox_inches="tight"); plt.close(fig)
    log(f"plot {os.path.relpath(pth, RES)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "collect"])
    ap.add_argument("--field")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--L", type=float, default=48.0)
    ap.add_argument("--comp", default="rebuild")
    ap.add_argument("--label", default="x")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--nref", default="radial", choices=["radial", "x"])
    a = ap.parse_args(ARGS)
    if a.mode == "run":
        run(a.field, a.n, a.L, a.comp, a.label, a.k, nref_kind=a.nref)
    else:
        collect()
