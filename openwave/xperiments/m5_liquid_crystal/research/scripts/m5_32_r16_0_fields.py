"""M5.32 R16-0 (lattice reads): claims C1 (lattice part), C3, C7, C8, C9 of the author's 2026-09-06 comments
(ledger 6.5, the R16-0 row).  Reads on existing R15 fields and on constructed textures; nothing is relaxed here.

EQUATIONS FIRST
---------------
The two H-adjoint completions per cell (m5_32_r16_0_symbolic.py for the definitions; the sym stencil, h^3-weighted):
    E_norm    = 4 sum_{i<j} tr(G F^eta_ij G F^eta_ij^T),   E_rebuild = 4 sum_{i<j} tr(G F^G_ij G F^G_ij^T),
    G = eta + 2 (eta u)(eta u)^T positive definite, so both densities are sums of squares: >= 0 pointwise (C3).
    Gate: sum E_norm == C15.i1h_static (our registry's I1_h is I_norm).
C1  the floor witness (the R15-V-b protocol): the degenerate vacuum dressed by the radial boost chi(r) = 0.5 exp(-r^2/8)
    with the (1,2) twist psi = k z INSIDE, DeltaE(k) = E(k) - E(0), both completions, on 64^3 L24 (the author's box)
    and 64^3 L48 (ours); the author: the two forms differ 2 to 6 percent on this profile.
C3  E_h(seed) and the pointwise floor on the relaxed L_cert hedgehog (R10 n32 L48) dressed the same way; the R15-V audit's
    cross terms  cross_h(k) = [E_h(inside, k) - E_h(dressed, 0)] - [E_h(twist alone, k) - E_h(hedgehog)]
    (the posted -183 / -611 / -1696), and the author's bound DeltaE_h >= -E_h(seed).
C7  the biaxial-ring reading of the R15-P-iv end fields: per cell the spatial triple (lambda_1, lambda_2, lambda_3) of N
    (the isolated root lambda_1 and the pair from (s, p): lambda_{2,3} = (s -+ sqrt(s^2 - 4p))/2), the biaxiality
    beta^2 = 1 - 6 (tr Q^3)^2 / (tr Q^2)^3 with Q the traceless triple (0 uniaxial, 1 maximal biaxiality), the sign of tr Q^3
    (prolate +, oblate -), radial shells of beta^2, and the quadrupole of the beta^2 weight on its maximal shell
    (a ring in a plane is oblate-uniaxial, a spherical shell is zero).
C8  (i) the rotation tangent d_z M = [G_z, M] - (x d_y - y d_x) M on axisymmetric fields (an analytic one at two
    resolutions, the R15-M hedgehog): zero pointwise up to O(h^2), hence J_z = 0 for any velocity field (the author's item 11).
    (ii) the spin-weight-2 shell decomposition of the transverse split: director n = the leading eigenvector of the
    spatial block, frame e = normalize(e_theta - (e_theta . n) n), f = n x e, zeta = S_ee - S_ff + 2 i S_ef
    (spin weight 2 under frame rotations), c_m = (4 pi / N_shell) sum_cells zeta conj(2Y_2m), the 2Y_2m from the
    Goldberg formula (orthonormality and the lattice recovery of each 2Y_2m are gates); P_m = |c_m|^2, <m> = sum m P_m / sum P_m.
    The chirality SIGN is convention-bound (conjugating zeta swaps m -> -m); the m-imbalance magnitude is not.
C9  the chiral pseudo-scalar tau = eps_ijk S_il d_j S_kl on the spatial block S, T2 = sum tau^2 h^3, on constructed uniaxial
    textures S = delta I + (1 - delta) n n^T (hedgehog, two-hedgehog pair, twist sheet, bend sheet; 40^3 L24) and on the
    R15 fields; the identity tau = (1 - delta)^2 n . (curl n) checked per cell (the author: 2e-29 / 1e-2 / 91 / 0, their box).

usage: python3 m5_32_r16_0_fields.py [c1|c3|c7|c8|c9|all]
out:   data/m5_32_r16_0_fields.json, plots/m5_32_r16_0_c7_biaxial.png, plots/m5_32_r16_0_c8_spin2.png,
       checkpoints/m5_32_r16/fields_<mode>.json (per mode, on arrival), checkpoints/m5_32_r16/fields.log
"""
from __future__ import annotations
import json
import math
import os
import sys
import time

import numpy as np

ARGS = list(sys.argv[1:])
sys.argv = [sys.argv[0]]
import m5_32_r15_common as C15                            # noqa: E402
import m5_32_r15_vb_lattice as VB                         # noqa: E402

INS4, C13, B8, EXT = C15.INS4, C15.C13, C15.B8, C15.EXT
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
DATA, PLOTS = os.path.join(RES, "data"), os.path.join(RES, "plots")
CK = os.path.join(RES, "checkpoints", "m5_32_r16")
CK15 = os.path.join(RES, "checkpoints", "m5_32_r15")
os.makedirs(CK, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
T0 = time.time()
LOG = open(os.path.join(CK, "fields.log"), "a")
ETA = C15.ETA
G, DELTA = C15.G, C15.DELTA


def log(m):
    line = f"[{time.time() - T0:8.1f}s] {m}"
    print(line, flush=True)
    LOG.write(line + "\n"); LOG.flush()


def rel(p):
    return os.path.relpath(p, RES)


def ck(mode, rec):
    json.dump(rec, open(os.path.join(CK, f"fields_{mode}.json"), "w"), indent=1, default=float)


# ------------------------------------------------ the two completions per cell
def completions_density(M, cfg):
    h = cfg["h"]; h3 = h ** 3
    Gm = EXT.h_cov_np(M)
    dn = np.zeros(M.shape[:3]); dr = np.zeros(M.shape[:3])
    for br, wt in INS4.branches(cfg["stencil"]):
        A = [INS4.d1(M, ax, h, br) for ax in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                Fe = A[i] @ ETA @ A[j] - A[j] @ ETA @ A[i]
                Fg = A[i] @ Gm @ A[j] - A[j] @ Gm @ A[i]
                dn += wt * 4.0 * np.einsum("...ab,...bc,...cd,...ad->...", Gm, Fe, Gm, Fe, optimize=True)
                dr += wt * 4.0 * np.einsum("...ab,...bc,...cd,...ad->...", Gm, Fg, Gm, Fg, optimize=True)
    return h3 * dn, h3 * dr


def both(M, cfg):
    dn, dr = completions_density(M, cfg)
    return {"E_norm": float(np.sum(dn)), "E_rebuild": float(np.sum(dr)), "min_density_norm": float(np.min(dn)), "min_density_rebuild": float(np.min(dr)),
            "E_eta (4 I1)": float(INS4.e_parts(M, cfg)[0])}


# ============================================================== C1 lattice
def c1():
    log("C1 lattice: I_norm vs I_rebuild on the floor-witness profile")
    rec = {"protocol": VB.__doc__.split("Reads")[0].strip()[:600], "runs": []}
    for n, L in ((64, 24.0), (64, 48.0)):
        cfg = C15.cfg_dd(n, L)
        D = np.broadcast_to(INS4.vac4(cfg), (n, n, n, 4, 4)).copy()
        Lb, Z = VB.boost_field(cfg)
        base = both(VB.conj(Lb, D), cfg)
        gate = C15.i1h_static(VB.conj(Lb, D), cfg)
        row = {"n": n, "L": L, "h": cfg["h"], "dressed_k0": base, "gate_E_norm_vs_registry_rel": abs(base["E_norm"] - gate) / gate, "k": {}}
        for k in (0.5, 1.0, 2.0):
            Rt = VB.twist_field(Z, k)
            e = both(VB.conj(Lb, VB.conj(Rt, D)), cfg)
            dn, dr, de = e["E_norm"] - base["E_norm"], e["E_rebuild"] - base["E_rebuild"], e["E_eta (4 I1)"] - base["E_eta (4 I1)"]
            row["k"][str(k)] = {"E": e, "DeltaE_norm": dn, "DeltaE_rebuild": dr, "DeltaE_eta": de, "rebuild_vs_norm_rel": (dr - dn) / max(abs(dn), 1e-300),
                                "min_density_norm": e["min_density_norm"], "min_density_rebuild": e["min_density_rebuild"]}
            log(f"  n{n} L{L:g} k {k}: DeltaE_eta {de:+.2f}  DeltaE_norm {dn:+.2f}  DeltaE_rebuild {dr:+.2f}  (rebuild-norm)/norm {(dr - dn) / dn:+.4f}; min densities {e['min_density_norm']:.2e} / {e['min_density_rebuild']:.2e}")
        rec["runs"].append(row)
        ck("c1", rec)
    rec["author"] = "the two forms differ 2 to 6 percent on this profile; the h-column 3.0-3.1 vs 4 mismatch is elsewhere"
    return rec


# ============================================================== C3
def c3():
    log("C3: the pointwise floor of E_h on the dressed relaxed hedgehog")
    rec = {"fields": {}}
    Mhh, cfg, src = C13.seed_hedgehog(32, 48)
    Lb, Z = VB.boost_field(cfg)
    e_seed = both(Mhh, cfg)
    e_dr = both(VB.conj(Lb, Mhh), cfg)
    row = {"source": src, "hedgehog": e_seed, "dressed_k0": e_dr, "k": {}}
    for k in (0.5, 1.0, 2.0):
        Rt = VB.twist_field(Z, k)
        e_in = both(VB.conj(Lb, VB.conj(Rt, Mhh)), cfg)
        e_tw = both(VB.conj(Rt, Mhh), cfg)
        r = {"inside": e_in, "twist_alone": e_tw}
        for lab in ("E_norm", "E_rebuild", "E_eta (4 I1)"):
            dE = e_in[lab] - e_dr[lab]
            cross = dE - (e_tw[lab] - e_seed[lab])
            r[f"DeltaE {lab}"] = dE
            r[f"cross {lab}"] = cross
            r[f"bound DeltaE >= -E(seed) {lab}"] = bool(dE >= -e_dr[lab])
        row["k"][str(k)] = r
        log(f"  L_cert hedgehog k {k}: DeltaE eta {r['DeltaE E_eta (4 I1)']:+.1f} norm {r['DeltaE E_norm']:+.1f} rebuild {r['DeltaE E_rebuild']:+.1f}; cross eta {r['cross E_eta (4 I1)']:+.1f} norm {r['cross E_norm']:+.1f} rebuild {r['cross E_rebuild']:+.1f}; "
            f"min density norm {e_in['min_density_norm']:.2e} rebuild {e_in['min_density_rebuild']:.2e}")
    rec["fields"]["L_cert hedgehog R10 n32 L48"] = row
    log(f"  E(seed) hedgehog: eta {e_seed['E_eta (4 I1)']:.2f} norm {e_seed['E_norm']:.2f} rebuild {e_seed['E_rebuild']:.2f}; dressed k0: eta {e_dr['E_eta (4 I1)']:.1f} norm {e_dr['E_norm']:.1f} rebuild {e_dr['E_rebuild']:.1f}")
    ck("c3", rec)
    # the R15-M degenerate hedgehog (mu 1e-2, c_P 1), the same protocol
    p = os.path.join(CK15, "m_hedgehog", "relax_n32_L48_mu0.01_cP1.npy")
    if os.path.exists(p):
        cfgd = C15.cfg_dd(32, 48.0, mu=1e-2, cP=1.0)
        Md = np.load(p)
        e_seed = both(Md, cfgd); e_dr = both(VB.conj(Lb, Md), cfgd)
        row = {"source": rel(p), "hedgehog": e_seed, "dressed_k0": e_dr, "k": {}}
        for k in (0.5, 1.0, 2.0):
            Rt = VB.twist_field(Z, k)
            e_in = both(VB.conj(Lb, VB.conj(Rt, Md)), cfgd); e_tw = both(VB.conj(Rt, Md), cfgd)
            r = {"inside": e_in, "twist_alone": e_tw}
            for lab in ("E_norm", "E_rebuild", "E_eta (4 I1)"):
                r[f"DeltaE {lab}"] = e_in[lab] - e_dr[lab]
                r[f"cross {lab}"] = e_in[lab] - e_dr[lab] - (e_tw[lab] - e_seed[lab])
            row["k"][str(k)] = r
            log(f"  R15-M degenerate hedgehog k {k}: DeltaE eta {r['DeltaE E_eta (4 I1)']:+.1f} norm {r['DeltaE E_norm']:+.1f} rebuild {r['DeltaE E_rebuild']:+.1f}; min density norm {e_in['min_density_norm']:.2e}")
        rec["fields"]["R15-M degenerate hedgehog n32 L48 mu1e-2 cP1"] = row
    rec["posted_R15_numbers"] = {"cross_h": [-183, -611, -1696], "cross_eta": [-36, -109, -188], "sentence_to_retract": "-4 I1^h is not bounded below on that background either"}
    rec["author"] = "E_h >= 0 pointwise, DeltaE_h >= -E_h(seed); the cross terms say the seed is not stationary along twist-inside-dressing"
    ck("c3", rec)
    return rec


# ============================================================== C7
def spatial_triple(M):
    N, lg, l1, s, p = C15.spectrum_parts(M)
    disc = s * s - 4.0 * p
    root = np.sqrt(np.maximum(disc, 0.0))
    l2, l3 = (s + root) / 2.0, (s - root) / 2.0
    trip = np.stack([l1, l2, l3], -1)
    trip = -np.sort(-trip, axis=-1)                                  # descending
    return trip, lg, disc


def biaxiality(trip):
    Q = trip - np.mean(trip, axis=-1, keepdims=True)
    q2 = np.sum(Q * Q, axis=-1)
    q3 = np.sum(Q ** 3, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        b2 = np.where(q2 > 1e-14, 1.0 - 6.0 * q3 * q3 / np.maximum(q2, 1e-300) ** 3, 0.0)
    return np.clip(b2, 0.0, 1.0), q3


def c7():
    log("C7: the biaxial-ring reading of the P-iv end fields")
    rec = {"fields": {}}
    items = [("P-iv end n32 L48 J200", os.path.join(CK15, "p4_fixedj", "fixedJ_n32_L48_J200.npy"), 32, 48.0),
             ("P-iv end n48 L72 J200", os.path.join(CK15, "p4_fixedj", "fixedJ_n48_L72_J200.npy"), 48, 72.0),
             ("R15-M seed n32 L48 (uniaxial control)", os.path.join(CK15, "m_hedgehog", "relax_n32_L48_mu0.01_cP1.npy"), 32, 48.0),
             ("R15-M seed n48 L72 (uniaxial control)", os.path.join(CK15, "m_hedgehog", "relax_n48_L72_mu0.01_cP1.npy"), 48, 72.0)]
    plots = {}
    for lab, p, n, L in items:
        if not os.path.exists(p):
            log(f"  missing {p}"); continue
        cfg = C15.cfg_dd(n, L, mu=1e-2, cP=1.0)
        M = np.load(p)
        h = cfg["h"]
        X, Y, Z = INS4.coords(n, h)
        r = np.sqrt(X * X + Y * Y + Z * Z)
        trip, lg, disc = spatial_triple(M)
        b2, q3 = biaxiality(trip)
        gap12 = trip[..., 0] - trip[..., 1]                            # the isolated root meeting the pair's upper member: the R15 crossing
        cross = gap12 < 1e-3
        # radial shells
        edges = np.arange(0.0, L / 2 + h, 1.5 * h)
        shells = []
        for a, b in zip(edges[:-1], edges[1:]):
            m = (r >= a) & (r < b)
            if not np.any(m):
                continue
            shells.append({"r": [float(a), float(b)], "beta2_mean": float(np.mean(b2[m])), "beta2_max": float(np.max(b2[m])), "trQ3_mean": float(np.mean(q3[m])),
                           "triple_mean": [float(x) for x in np.mean(trip[m], axis=0)], "n_cells": int(np.sum(m))})
        # the center: the 8 innermost cells
        idx = np.argsort(r, axis=None)[:8]
        center = {"r_max": float(np.sort(r, axis=None)[7]), "triple": [[float(x) for x in trip.reshape(-1, 3)[i]] for i in idx[:2]],
                  "lambda_g": [float(lg.reshape(-1)[i]) for i in idx[:2]], "beta2": [float(b2.reshape(-1)[i]) for i in idx], "trQ3_sign": [float(np.sign(q3.reshape(-1)[i])) for i in idx]}
        # the maximal shell and the quadrupole of the beta^2 weight there
        imax = int(np.argmax([s_["beta2_mean"] for s_ in shells]))
        a, b = shells[imax]["r"]
        m = (r >= a) & (r < b)
        w = b2[m]
        xs = np.stack([X[m], Y[m], Z[m]], -1) / np.maximum(r[m], 1e-300)[:, None]
        Qd = np.einsum("c,ci,cj->ij", w, xs, xs) / max(np.sum(w), 1e-300) - np.eye(3) / 3.0
        ev, evec = np.linalg.eigh(Qd)
        ring_axis = [float(x) for x in evec[:, 0]]                    # the eigenvector of the most negative eigenvalue = the ring's axis
        # the same over all cells with beta^2 above half its maximum
        m2 = b2 > 0.5 * np.max(b2)
        w2 = b2[m2]
        xs2 = np.stack([X[m2], Y[m2], Z[m2]], -1) / np.maximum(r[m2], 1e-300)[:, None]
        Qd2 = np.einsum("c,ci,cj->ij", w2, xs2, xs2) / max(np.sum(w2), 1e-300) - np.eye(3) / 3.0
        ev2 = np.linalg.eigvalsh(Qd2)
        # axis profiles (the line nearest the x axis)
        j = n // 2
        prof = {"r": [float(x) for x in X[j:, j, j]], "triple_x_axis": [[float(v) for v in trip[i, j, j]] for i in range(j, n)], "beta2_x_axis": [float(b2[i, j, j]) for i in range(j, n)],
                "beta2_z_axis": [float(b2[j, j, i]) for i in range(j, n)], "beta2_diag": [float(b2[i, i, i]) for i in range(j, n)]}
        row = {"source": rel(p), "n": n, "L": L, "h": h, "center": center, "beta2_global_max": float(np.max(b2)), "r_at_beta2_max": float(r.reshape(-1)[int(np.argmax(b2))]),
               "n_crossing_cells (top two of the triple within 1e-3)": int(np.sum(cross)), "crossing_radii": [float(x) for x in np.sort(r[cross])[:12]],
               "n_complex_pair_cells (s^2 - 4p < -1e-12)": int(np.sum(disc < -1e-12)),
               "shells": shells, "max_shell": {"r": shells[imax]["r"], "beta2_mean": shells[imax]["beta2_mean"], "quadrupole_eigenvalues_of_beta2_weight": [float(x) for x in ev], "ring_axis (eigenvector of the negative eigenvalue)": ring_axis, "perfect_great_circle_ring": [-1 / 6, 1 / 12, 1 / 12],
                                                "ring_signature": "oblate uniaxial quadrupole (two equal positive, one negative -1/3 scale) = a ring in a plane; ~0 = a spherical shell"},
               "half_max_region": {"n_cells": int(np.sum(m2)), "r_mean": float(np.mean(r[m2])), "r_std": float(np.std(r[m2])), "quadrupole_eigenvalues": [float(x) for x in ev2]},
               "profiles": prof}
        rec["fields"][lab] = row
        plots[lab] = (b2, trip, X, Y, Z, n, h)
        log(f"  {lab}: center triple {center['triple'][0]} (lambda_g {center['lambda_g'][0]:.3f}) beta2 {center['beta2'][0]:.3f} sign trQ3 {center['trQ3_sign'][0]:+.0f}; "
            f"beta2 max {row['beta2_global_max']:.3f} at r {row['r_at_beta2_max']:.2f}; max shell {shells[imax]['r']} mean {shells[imax]['beta2_mean']:.3f} quadrupole {ev} axis {np.round(ring_axis, 3)}; "
            f"half-max region r {row['half_max_region']['r_mean']:.2f} +- {row['half_max_region']['r_std']:.2f} quadrupole {ev2}; crossing cells {row['n_crossing_cells (top two of the triple within 1e-3)']} at r {row['crossing_radii'][:8]}")
        log(f"    shells beta2 mean: {[round(s_['beta2_mean'], 3) for s_ in shells[:8]]}")
        ck("c7", rec)
    rec["author"] = "the P-iv end state = the Landau-de Gennes biaxial-ring core (oblate uniaxial center, biaxial shell/ring; Penzenstadler-Trebin 1989, Majumdar 2012, McLauchlan et al. 2024)"
    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = [k for k in plots if k.startswith("P-iv")]
    fig, ax = plt.subplots(2, max(len(labs), 1) + 1, figsize=(4.2 * (len(labs) + 1), 7.5))
    for c_, lab in enumerate(labs):
        b2, trip, X, Y, Z, n, h = plots[lab]
        j = n // 2
        ext = [-n * h / 2, n * h / 2, -n * h / 2, n * h / 2]
        im = ax[0, c_].imshow(b2[:, :, j].T, origin="lower", extent=ext, vmin=0, vmax=1, cmap="viridis"); ax[0, c_].set_title(f"{lab}\nbeta^2, plane z = {Z[0, 0, j]:.2f}", fontsize=8); ax[0, c_].set_xlim(-12, 12); ax[0, c_].set_ylim(-12, 12)
        im = ax[1, c_].imshow(b2[j, :, :].T, origin="lower", extent=ext, vmin=0, vmax=1, cmap="viridis"); ax[1, c_].set_title(f"beta^2, plane x = {X[j, 0, 0]:.2f} (y horizontal, z vertical)", fontsize=8); ax[1, c_].set_xlim(-12, 12); ax[1, c_].set_ylim(-12, 12)
    fig.colorbar(im, ax=ax[:, :len(labs)].ravel().tolist(), shrink=0.6)
    for lab in labs[:1]:
        pr = rec["fields"][lab]["profiles"]
        ax[0, -1].plot(pr["r"], np.array(pr["triple_x_axis"])); ax[0, -1].set_title(f"{lab}: spatial triple on the x axis", fontsize=8); ax[0, -1].set_xlabel("r"); ax[0, -1].axhline(DELTA, color="k", lw=0.5, ls="--"); ax[0, -1].axhline(1.0, color="k", lw=0.5, ls="--")
    for lab in labs:
        sh = rec["fields"][lab]["shells"]
        ax[1, -1].plot([np.mean(s_["r"]) for s_ in sh], [s_["beta2_mean"] for s_ in sh], "o-", label=lab, ms=3)
    for lab in [k for k in plots if "control" in k]:
        sh = rec["fields"][lab]["shells"]
        ax[1, -1].plot([np.mean(s_["r"]) for s_ in sh], [s_["beta2_mean"] for s_ in sh], "x--", label=lab, ms=3)
    ax[1, -1].set_xlabel("r"); ax[1, -1].set_ylabel("shell-mean beta^2"); ax[1, -1].legend(fontsize=6); ax[1, -1].set_xlim(0, 24)
    fig.savefig(os.path.join(PLOTS, "m5_32_r16_0_c7_biaxial.png"), dpi=110, bbox_inches="tight")
    rec["plot"] = "plots/m5_32_r16_0_c7_biaxial.png"
    return rec


# ============================================================== C8
def sY2(s, l, m, th, ph):
    """spin-weighted spherical harmonic, the Goldberg et al. formula."""
    pref = (-1) ** m * math.sqrt(math.factorial(l + m) * math.factorial(l - m) * (2 * l + 1) / (4 * math.pi * math.factorial(l + s) * math.factorial(l - s)))
    half = th / 2.0
    sn, cs = np.sin(half), np.cos(half)
    tot = np.zeros_like(th, dtype=complex)
    for r_ in range(0, l - s + 1):
        k2 = r_ + s - m
        if k2 < 0 or k2 > l + s:
            continue
        e = 2 * r_ + s - m
        tot += math.comb(l - s, r_) * math.comb(l + s, k2) * (-1) ** (l - r_ - s) * cs ** (2 * l - e) * sn ** e if False else \
            math.comb(l - s, r_) * math.comb(l + s, k2) * (-1) ** (l - r_ - s) * (cs ** e) * (sn ** (2 * l - e))
    return pref * tot * np.exp(1j * m * ph)


def shell_decomp(zeta, th, ph):
    out = {}
    for m in range(-2, 3):
        Y = sY2(2, 2, m, th, ph)
        out[m] = complex(4 * np.pi / zeta.size * np.sum(zeta * np.conj(Y)))
    return out


def frame_zeta(M, X, Y, Z):
    """zeta = S_ee - S_ff + 2 i S_ef in the (e, f) frame transverse to the local director n."""
    S = M[..., 1:, 1:]
    w, V = np.linalg.eigh(S)
    nvec = V[..., :, -1]
    gap = w[..., -1] - w[..., -2]
    r = np.sqrt(X * X + Y * Y + Z * Z)
    # the director is a LINE field: orient it outward (n . r_hat >= 0) so the frame (e, f = n x e) has one handedness
    rhat = np.stack([X, Y, Z], -1) / r[..., None]
    sgn = np.sum(nvec * rhat, -1)
    sgn = np.where(np.abs(sgn) > 1e-6, np.sign(sgn), np.sign(nvec[..., 2] + 1e-300))
    nvec = nvec * sgn[..., None]
    th = np.arccos(np.clip(Z / r, -1, 1)); ph = np.arctan2(Y, X)
    eth = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)], -1)
    e = eth - np.sum(eth * nvec, -1, keepdims=True) * nvec
    e = e / np.maximum(np.linalg.norm(e, axis=-1, keepdims=True), 1e-300)
    f = np.cross(nvec, e)
    See = np.einsum("...a,...ab,...b->...", e, S, e); Sff = np.einsum("...a,...ab,...b->...", f, S, f); Sef = np.einsum("...a,...ab,...b->...", e, S, f)
    return See - Sff + 2j * Sef, gap, th, ph, r


def c8():
    log("C8: the rotation tangent on axisymmetric fields and the spin-weight-2 shell content")
    rec = {}
    # (i) the rotation tangent
    def tangent(M, cfg):
        n, h = cfg["n"], cfg["h"]
        X, Y, Z = INS4.coords(n, h)
        dy = 0.5 * (INS4.d1(M, 1, h, "fwd") + INS4.d1(M, 1, h, "bwd"))
        dx = 0.5 * (INS4.d1(M, 0, h, "fwd") + INS4.d1(M, 0, h, "bwd"))
        rotM = B8.G3 @ M - M @ B8.G3
        t = rotM - (X[..., None, None] * dy - Y[..., None, None] * dx)
        sl = (slice(1, -1),) * 3
        return float(np.sqrt(np.sum(t[sl] ** 2)) / np.sqrt(np.sum(rotM[sl] ** 2))), float(np.max(np.abs(t[sl])))

    def axisym(cfg):
        """an axisymmetric field with a split: M = R12(phi) [R13(theta) D R13^T] R12^T, D = diag(g, delta + s, delta - s, 1) (the director along z on the axis), theta ~ rho and s ~ rho^2 near the axis (both smooth in x, y, z)."""
        n, h = cfg["n"], cfg["h"]
        X, Y, Z = INS4.coords(n, h)
        rho = np.sqrt(X * X + Y * Y); ph = np.arctan2(Y, X)
        th = 0.7 * rho * np.exp(-(rho * rho + Z * Z) / 30.0)          # smooth: the tilt vanishes on the axis (the tensor is continuous there)
        s = 0.15 * (rho * rho / 4.0) * np.exp(-(rho * rho + Z * Z) / 20.0)   # weight two: an axisymmetric transverse split must vanish on the axis (the author's double zero at the poles)
        D = np.zeros(X.shape + (4, 4)); D[..., 0, 0] = G; D[..., 3, 3] = 1.0; D[..., 1, 1] = DELTA + s; D[..., 2, 2] = DELTA - s   # the director along z ON the axis (continuous there)
        R13 = B8.rot_field(B8.G2, th)                     # rotation in the (1,3) plane
        R12 = B8.rot_field(B8.G3, ph)                     # rotation about z
        M0 = np.einsum("...ab,...bc,...dc->...ad", R13, D, R13)
        return np.einsum("...ab,...bc,...dc->...ad", R12, M0, R12)
    rows = {}
    for n in (32, 64):
        cfg = C15.cfg_dd(n, 24.0)
        rel_, mx = tangent(axisym(cfg), cfg)
        rows[f"analytic axisymmetric split field n{n} L24 h{cfg['h']}"] = {"rel_norm": rel_, "max_abs": mx}
        log(f"  tangent, analytic axisymmetric field n{n}: |d_z M| / |[G_z, M]| = {rel_:.3e}")
    rows["ratio n32/n64 (4 = O(h^2))"] = rows["analytic axisymmetric split field n32 L24 h0.75"]["rel_norm"] / rows["analytic axisymmetric split field n64 L24 h0.375"]["rel_norm"]
    for lab, p, n, L in (("R15-M hedgehog n32 L48", os.path.join(CK15, "m_hedgehog", "relax_n32_L48_mu0.01_cP1.npy"), 32, 48.0),
                         ("P-iv end n32 L48", os.path.join(CK15, "p4_fixedj", "fixedJ_n32_L48_J200.npy"), 32, 48.0)):
        cfg = C15.cfg_dd(n, L, mu=1e-2, cP=1.0)
        rel_, mx = tangent(np.load(p), cfg)
        rows[lab] = {"rel_norm": rel_, "max_abs": mx}
        log(f"  tangent, {lab}: {rel_:.3e}")
    # a control that is NOT axisymmetric about z: the same analytic field rotated by 0.5 rad about x
    cfg = C15.cfg_dd(32, 24.0)
    Rx = B8.rot_field(B8.G1, 0.5 * np.ones((32, 32, 32)))
    Mc = axisym(cfg)
    # rotate the tensor AND the point: build on rotated coordinates instead (cheap: rotate the tensor only, which breaks the symmetry as a control)
    Mc = np.einsum("...ab,...bc,...dc->...ad", Rx, Mc, Rx)
    rel_, mx = tangent(Mc, cfg)
    rows["control: tensor rotated 0.5 rad about x without rotating the point (not axisymmetric about z)"] = {"rel_norm": rel_, "max_abs": mx}
    rec["rotation_tangent"] = rows
    rec["rotation_tangent"]["author"] = "on an axisymmetric configuration [G_z, M] - (x d_y - y d_x) M vanishes pointwise, so J_z = 0 for any velocity field"
    ck("c8", rec)
    # (ii) the 2Y_2m gates
    th = np.linspace(0, np.pi, 801)[1:-1]; ph = np.linspace(0, 2 * np.pi, 800, endpoint=False)
    TH, PH = np.meshgrid(th, ph, indexing="ij")
    dO = np.sin(TH) * (th[1] - th[0]) * (ph[1] - ph[0])
    gram = np.zeros((5, 5), dtype=complex)
    for i, m1 in enumerate(range(-2, 3)):
        for j_, m2 in enumerate(range(-2, 3)):
            gram[i, j_] = np.sum(sY2(2, 2, m1, TH, PH) * np.conj(sY2(2, 2, m2, TH, PH)) * dO)
    orth = float(np.max(np.abs(gram - np.eye(5))))
    # zeros: 2Y_20 at both poles, 2Y_22 at theta = pi, 2Y_2-2 at theta = 0 (values at theta = 0.01 and pi - 0.01)
    zeros = {str(m): {"theta_0.01": float(abs(sY2(2, 2, m, np.array(0.01), np.array(0.3)))), "theta_pi-0.01": float(abs(sY2(2, 2, m, np.array(np.pi - 0.01), np.array(0.3)))),
                      "equator": float(abs(sY2(2, 2, m, np.array(np.pi / 2), np.array(0.3))))} for m in range(-2, 3)}
    rec["spin2_harmonics"] = {"orthonormality_max_err": orth, "values_near_poles_and_equator": zeros}
    log(f"  2Y_2m orthonormality max err {orth:.2e}; near-pole values {zeros}")
    # lattice recovery on a shell of the n48 box
    cfg = C15.cfg_dd(48, 72.0)
    X, Y, Z = INS4.coords(48, cfg["h"])
    r = np.sqrt(X * X + Y * Y + Z * Z)
    thl = np.arccos(np.clip(Z / r, -1, 1)); phl = np.arctan2(Y, X)
    recov = {}
    for a, b in ((3.0, 6.0), (6.0, 9.0), (9.0, 12.0)):
        m_ = (r >= a) & (r < b)
        mat = np.zeros((5, 5), dtype=complex)
        for i, m1 in enumerate(range(-2, 3)):
            zeta = sY2(2, 2, m1, thl[m_], phl[m_])
            c = shell_decomp(zeta, thl[m_], phl[m_])
            mat[i] = [c[m2] for m2 in range(-2, 3)]
        recov[f"[{a:g},{b:g})"] = {"n_cells": int(np.sum(m_)), "max_err_vs_identity": float(np.max(np.abs(mat - np.eye(5))))}
    rec["lattice_recovery"] = recov
    log(f"  lattice recovery of each 2Y_2m on shells (n48 h1.5): {recov}")
    # constructed real patterns through the frame: zeta = 2Y_20 (axisymmetric), the real m = +-2 mix, the chiral m = +2
    # (i.e. the transverse split built from the pattern in the (e_theta, e_phi) frame with n = r_hat)
    def field_from_zeta(zeta, X, Y, Z):
        r = np.sqrt(X * X + Y * Y + Z * Z)
        th = np.arccos(np.clip(Z / r, -1, 1)); ph = np.arctan2(Y, X)
        nvec = np.stack([X, Y, Z], -1) / r[..., None]
        e = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)], -1)
        f = np.cross(nvec, e)
        a, b = zeta.real / 2.0, zeta.imag / 2.0
        S = np.zeros(X.shape + (3, 3))
        S += DELTA * np.eye(3) + (1 - DELTA) * nvec[..., :, None] * nvec[..., None, :]
        S += a[..., None, None] * (e[..., :, None] * e[..., None, :] - f[..., :, None] * f[..., None, :])
        S += b[..., None, None] * (e[..., :, None] * f[..., None, :] + f[..., :, None] * e[..., None, :])
        M = np.zeros(X.shape + (4, 4)); M[..., 0, 0] = G; M[..., 1:, 1:] = S
        return M
    pats = {"2Y_20": lambda th, ph: sY2(2, 2, 0, th, ph), "real m=+-2 (2Y_22 + 2Y_2-2)": lambda th, ph: sY2(2, 2, 2, th, ph) + sY2(2, 2, -2, th, ph), "chiral m=+2 (2Y_22)": lambda th, ph: sY2(2, 2, 2, th, ph)}
    pres = {}
    amp = 0.1
    for lab, fn in pats.items():
        M = field_from_zeta(amp * fn(thl, phl), X, Y, Z)
        zeta, gap, th_, ph_, r_ = frame_zeta(M, X, Y, Z)
        m_ = (r_ >= 6.0) & (r_ < 9.0)
        c = shell_decomp(zeta[m_] / amp, th_[m_], ph_[m_])
        P = {m: abs(c[m]) ** 2 for m in c}
        tot = sum(P.values())
        pres[lab] = {"P_m": {str(m): P[m] for m in P}, "<m>": sum(m * P[m] for m in P) / tot, "chirality (P2 - P-2)/(P2 + P-2)": (P[2] - P[-2]) / max(P[2] + P[-2], 1e-300)}
        log(f"  pattern {lab}: P_m {[round(P[m], 4) for m in range(-2, 3)]} <m> {pres[lab]['<m>']:+.3f}")
    rec["constructed_patterns_through_the_frame"] = pres
    ck("c8", rec)
    # the reads: P-iv end fields and the R15-M seeds
    reads = {}
    for lab, p, n, L in (("P-iv end n32 L48 J200", os.path.join(CK15, "p4_fixedj", "fixedJ_n32_L48_J200.npy"), 32, 48.0),
                         ("P-iv end n48 L72 J200", os.path.join(CK15, "p4_fixedj", "fixedJ_n48_L72_J200.npy"), 48, 72.0),
                         ("R15-M seed n32 L48", os.path.join(CK15, "m_hedgehog", "relax_n32_L48_mu0.01_cP1.npy"), 32, 48.0),
                         ("R15-M seed n48 L72", os.path.join(CK15, "m_hedgehog", "relax_n48_L72_mu0.01_cP1.npy"), 48, 72.0)):
        cfg = C15.cfg_dd(n, L, mu=1e-2, cP=1.0)
        M = np.load(p)
        X, Y, Z = INS4.coords(n, cfg["h"])
        zeta, gap, th_, ph_, r_ = frame_zeta(M, X, Y, Z)
        sh = {}
        for a, b in ((0.0, 3.0), (3.0, 6.0), (6.0, 9.0), (9.0, 12.0), (12.0, 15.0)):
            m_ = (r_ >= a) & (r_ < b)
            c = shell_decomp(zeta[m_], th_[m_], ph_[m_])
            P = {m: abs(c[m]) ** 2 for m in c}
            tot = sum(P.values())
            ztot = float(4 * np.pi / zeta[m_].size * np.sum(np.abs(zeta[m_]) ** 2))
            sh[f"[{a:g},{b:g})"] = {"n_cells": int(np.sum(m_)), "P_m": {str(m): P[m] for m in P}, "l=2 power / total |zeta|^2": tot / max(ztot, 1e-300), "total |zeta|^2 (4pi mean)": ztot,
                                     "<m>": sum(m * P[m] for m in P) / max(tot, 1e-300), "chirality": (P[2] - P[-2]) / max(P[2] + P[-2], 1e-300),
                                     "min director gap": float(np.min(gap[m_])), "mean |zeta|": float(np.mean(np.abs(zeta[m_])))}
        reads[lab] = {"source": rel(p), "shells": sh}
        log(f"  {lab}: " + "; ".join(f"{k}: |zeta| {v['mean |zeta|']:.4f} P_m {[f'{v['P_m'][str(m)]:.2e}' for m in range(-2, 3)]} l2frac {v['l=2 power / total |zeta|^2']:.2f} <m> {v['<m>']:+.2f} gap {v['min director gap']:.3f}" for k, v in sh.items()))
    rec["reads"] = reads
    rec["author"] = "stage 1 should report the split's 2Y_2m content on shells; 2Y_20 and the real m = +-2 mix are the named failure modes, a chiral m = +-2 dominance is the electron"
    ck("c8", rec)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(reads), figsize=(4 * len(reads), 3.6), sharey=False)
    for k_, (lab, rr) in enumerate(reads.items()):
        keys = list(rr["shells"].keys())
        for m in range(-2, 3):
            ax[k_].plot(range(len(keys)), [rr["shells"][k]["P_m"][str(m)] for k in keys], "o-", label=f"m = {m}", ms=3)
        ax[k_].set_xticks(range(len(keys))); ax[k_].set_xticklabels(keys, fontsize=7); ax[k_].set_yscale("log"); ax[k_].set_title(lab, fontsize=8); ax[k_].set_xlabel("shell r")
        if k_ == 0:
            ax[k_].set_ylabel("P_m = |c_m|^2"); ax[k_].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "m5_32_r16_0_c8_spin2.png"), dpi=110)
    rec["plot"] = "plots/m5_32_r16_0_c8_spin2.png"
    return rec


# ============================================================== C9
def c9():
    log("C9: the chiral pseudo-scalar tau on constructed textures and the R15 fields")
    rec = {"textures": {}}
    eps = np.zeros((3, 3, 3))
    for (i, j, k), v in {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1, (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}.items():
        eps[i, j, k] = v

    def tau_of(M, h):
        S = M[..., 1:, 1:]
        dS = np.stack([0.5 * (INS4.d1(S, ax, h, "fwd") + INS4.d1(S, ax, h, "bwd")) for ax in range(3)], 0)     # (j, ..., k, l)
        return np.einsum("ijk,...il,j...kl->...", eps, S, dS, optimize=True)

    def curl_n(nv, h):
        d = [0.5 * (INS4.d1(nv, ax, h, "fwd") + INS4.d1(nv, ax, h, "bwd")) for ax in range(3)]      # d[j][..., i] = d_j n_i
        c = np.stack([d[1][..., 2] - d[2][..., 1], d[2][..., 0] - d[0][..., 2], d[0][..., 1] - d[1][..., 0]], -1)
        return np.sum(nv * c, -1)
    n, L = 40, 24.0
    cfg = C15.cfg_dd(n, L)
    h = cfg["h"]
    X, Y, Z = INS4.coords(n, h)
    r = np.sqrt(X * X + Y * Y + Z * Z)
    k = 0.5
    r1 = np.sqrt(X * X + Y * Y + (Z - 4.0) ** 2); r2 = np.sqrt(X * X + Y * Y + (Z + 4.0) ** 2)
    pair = np.stack([X / r1 + X / r2, Y / r1 + Y / r2, (Z - 4.0) / r1 + (Z + 4.0) / r2], -1)
    pair = pair / np.maximum(np.linalg.norm(pair, axis=-1, keepdims=True), 1e-300)
    tex = {"hedgehog n = r_hat": np.stack([X, Y, Z], -1) / r[..., None],
           "two-hedgehog pair (centers +-4 z, meridional)": pair,
           "twist sheet n = (cos kz, sin kz, 0), k 0.5": np.stack([np.cos(k * Z), np.sin(k * Z), 0 * Z], -1),
           "bend sheet n = (sin kz, 0, cos kz), k 0.5": np.stack([np.sin(k * Z), 0 * Z, np.cos(k * Z)], -1)}
    sl = (slice(1, -1),) * 3
    for lab, nv in tex.items():
        S = DELTA * np.eye(3) + (1 - DELTA) * nv[..., :, None] * nv[..., None, :]
        M = np.zeros(X.shape + (4, 4)); M[..., 0, 0] = G; M[..., 1:, 1:] = S
        tau = tau_of(M, h)[sl]
        ncn = curl_n(nv, h)[sl]
        T2 = float(np.sum(tau * tau) * h ** 3)
        m_ = np.abs(ncn) > 1e-6
        ratio = (tau[m_] / ncn[m_]) if np.any(m_) else np.array([])
        rec["textures"][lab] = {"T2 = sum tau^2 h^3 (interior)": T2, "max |tau|": float(np.max(np.abs(tau))), "max |n.curl n|": float(np.max(np.abs(ncn))),
                                "identity tau/(n.curl n) vs (1-delta)^2": {"(1-delta)^2": (1 - DELTA) ** 2, "ratio_mean": float(np.mean(ratio)) if ratio.size else None, "ratio_max_dev": float(np.max(np.abs(ratio - (1 - DELTA) ** 2))) if ratio.size else None, "n_cells": int(m_.sum())}}
        log(f"  {lab}: T2 {T2:.3e}, max|tau| {np.max(np.abs(tau)):.3e}, max|n.curl n| {np.max(np.abs(ncn)):.3e}, ratio {rec['textures'][lab]['identity tau/(n.curl n) vs (1-delta)^2']}")
    rec["author"] = "T2 = 2e-29 (hedgehog) / 1e-2 (pair) / 91 (twist sheet) / 0 (bend sheet) on a 40^3 box; tau = (1-delta)^2 n.(curl n) on a uniaxial texture"
    for lab, p, n, L in (("R15-M hedgehog n32 L48", os.path.join(CK15, "m_hedgehog", "relax_n32_L48_mu0.01_cP1.npy"), 32, 48.0),
                         ("P-iv end n32 L48", os.path.join(CK15, "p4_fixedj", "fixedJ_n32_L48_J200.npy"), 32, 48.0)):
        cfg = C15.cfg_dd(n, L, mu=1e-2, cP=1.0)
        M = np.load(p)
        tau = tau_of(M, cfg["h"])[sl]
        rec["textures"][lab] = {"source": rel(p), "T2 = sum tau^2 h^3 (interior)": float(np.sum(tau * tau) * cfg["h"] ** 3), "max |tau|": float(np.max(np.abs(tau)))}
        log(f"  {lab}: T2 {rec['textures'][lab]['T2 = sum tau^2 h^3 (interior)']:.3e} max|tau| {np.max(np.abs(tau)):.3e}")
    ck("c9", rec)
    return rec


if __name__ == "__main__":
    mode = ARGS[0] if ARGS else "all"
    fns = {"c1": c1, "c3": c3, "c7": c7, "c8": c8, "c9": c9}
    todo = list(fns) if mode == "all" else [mode]
    outp = os.path.join(DATA, "m5_32_r16_0_fields.json")
    OUT = json.load(open(outp)) if os.path.exists(outp) else {"rung": "R16-0 lattice reads", "claims": {}}
    for m in todo:
        OUT["claims"][m.upper()] = fns[m]()
        json.dump(OUT, open(outp, "w"), indent=1, default=float)
        log(f"wrote {rel(outp)} ({m})")
