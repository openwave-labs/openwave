"""M5.32 R16-4: the principal symbol of the circle-averaged L_v4 on a rotating core, channel by
channel (ledger 6.5 as amended).

EQUATIONS
---------
The Lagrangian density per cell as a function of the four jets A_mu (A_0 = omega a0 on the
rotating background), E-orientation reversed into the Lagrangian sign:
    l(M; A_0, A_i) = +4 sum_i tr(G F_0i G F_0i^T) - 4 sum_{i<j} tr(G F_ij G F_ij^T)
                     + (c_P / 2) tr(Om_0^T eta Om_0 eta) - (c_P / 2) sum_i tr(Om_i^T eta Om_i eta)
                     + c_s rho^2 tr(A_0 G A_0 G) - c_s rho^2 sum_i tr(A_i G A_i G)  - V4 - mu rho^2
(the potential terms carry no jets and drop out of the symbol).  A perturbation xi(x) e^{i(k.x - Om t)}
in a channel direction xi (a symmetric 4 x 4 direction at the cell) has the second-order density
    l_2 = (1/2) sum_{mu nu} k_mu k_nu H_mu_nu,   H_mu_nu = d^2 l / dA_mu dA_nu [xi, xi]  (k_0 = Om),
the PRINCIPAL SYMBOL sigma(Om, k) = H_00 Om^2 + 2 Om sum_i H_0i k_i + sum_ij H_ij k_i k_j.  The channel is
hyperbolic iff sigma(Om, k) = 0 has real roots Om for every real k: H_00 > 0 and
Q(k) = (sum_i H_0i k_i)^2 - H_00 sum_ij H_ij k_i k_j >= 0 for every k (Q's 3 x 3 matrix PSD); the static
stiffness is -H_ij (positive definite for a well-posed static problem).  For a two-component channel
(the split doublet) the symbol is the 2 x 2 matrix pencil sigma(Om, k) = Om^2 H_00 + Om (2 H_0k) + H_kk
and hyperbolicity = four real roots of det sigma for every unit k (the quadratic eigenvalue problem by
linearization, 40 random directions).  H_mu_nu by central differences of the pointwise density in the
jets (the M dependence through G, w, rho^2 is lower order and fixed).  The circle average: the symbol
of the averaged action is the average over the samples of the symbol at (M_k, A_k) with xi -> R xi R^T
(the transformed perturbation; the connection terms of the circle carry no derivative of xi).
Channels at a cell (u = e_0, the director n, the pair frame (e, f)):
    tilt (director) n -> e, n -> f:  xi = [T, M] with T the (n, e) or (n, f) rotation generator
    split doublet:                    xi_a = e e^T - f f^T,  xi_b = e f^T + f e^T
    boost along n, along e:           xi = [B, M]_sym with B the boost generator (B = eta-symmetric)
The author's prediction: the tilt channel hyperbolic iff c_s > 16 omega^2 (R16-0 C5 on the sheet);
here the actual core at the R16-3 omega and on an omega scan (the crossing expected at
omega = sqrt(c_s / 16) = 0.158 for c_s 0.4), the exterior degenerate in the director channel
(H_00 = 0 where B = 0: a constraint, no wave).

usage: python3 m5_32_r16_4_symbol.py run --field <path.npy> --n 32 --L 48 --comp rebuild --label <lab> [--omega 0.05] [--scan]
out:   checkpoints/m5_32_r16/r16_4_<label>.json, data/m5_32_r16_4.json (collect)
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
C15, INS4 = C.C15, C.INS4
ETA = C.ETA
RES, DATA, PLOTS, CK = C.RES, C.DATA, C.PLOTS, C.CK
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def lag_density(M, A, fr, cfg):
    """the Lagrangian density per cell from the four jets A = [A_0, A_1, A_2, A_3] (potential terms omitted)."""
    comp, cP, cs = cfg["completion"], cfg["cP"], cfg["cs"]
    Gm = fr["G"]
    spl, _ = C15.split_cells(M, need_grad=False)
    rho2 = spl / 4.0
    ell = np.zeros(M.shape[:-2], dtype=M.dtype)
    for mu in range(4):
        for nu in range(mu + 1, 4):
            d, _, _, _ = C.quartic_pair(A[mu], A[nu], Gm, comp, need_grad=False)
            ell = ell - 4.0 * ETA[mu, mu] * ETA[nu, nu] * d
    for mu in range(4):
        Ek, _, _ = C.kp_cells([A[mu]], fr, need_grad=False)
        d2, _, _ = C.e2_cells(A[mu], Gm, need_grad=False)
        ell = ell - ETA[mu, mu] * (cP * Ek + cs * rho2 * d2)
    return ell


def jets_at(M, cfg):
    """central-difference spatial jets per cell (the symbol's background) on the sym stencil average."""
    h = cfg["h"]
    A = [np.zeros_like(M) for _ in range(3)]
    for br, wt in INS4.branches(cfg["stencil"]):
        for ax in range(3):
            A[ax] += wt * INS4.d1(M, ax, h, br)
    return A


def channels(fr, M):
    """the channel directions per cell."""
    u, n, J = np.real(fr["u"]), np.real(fr["n"]), np.real(fr["J"])
    P23 = np.real(fr["P23"])
    # e: a fixed reference projected onto the pair plane (x, then y where degenerate), f = J e
    ref = np.zeros(M.shape[:-1]); ref[..., 1] = 1.0
    e = np.einsum("...ab,...b->...a", P23, ref)
    bad = np.linalg.norm(e, axis=-1) < 1e-6
    ref2 = np.zeros(M.shape[:-1]); ref2[..., 2] = 1.0
    e[bad] = np.einsum("...ab,...b->...a", P23, ref2)[bad]
    e = e / np.sqrt(np.einsum("...a,ab,...b->...", e, ETA, e))[..., None]
    f = np.einsum("...ab,...b->...a", J, e)

    def gen_rot(a, b):                                                    # rotation generator in the (a, b) plane: a b^T eta - b a^T eta
        return a[..., :, None] * (b @ ETA)[..., None, :] - b[..., :, None] * (a @ ETA)[..., None, :]

    def gen_boost(a):                                                     # boost generator u <-> a
        return u[..., :, None] * (a @ ETA)[..., None, :] + a[..., :, None] * (u @ ETA)[..., None, :]

    def tangent(T):
        return T @ M + M @ np.swapaxes(T, -1, -2)
    ch = {"tilt_n_e": [tangent(gen_rot(n, e))], "tilt_n_f": [tangent(gen_rot(n, f))],
          "split_doublet": [e[..., :, None] * e[..., None, :] - f[..., :, None] * f[..., None, :], e[..., :, None] * f[..., None, :] + f[..., :, None] * e[..., None, :]],
          "boost_n": [tangent(gen_boost(n))], "boost_e": [tangent(gen_boost(e))]}
    for k in ch:
        for i, X in enumerate(ch[k]):
            nrm = np.sqrt(np.sum(X * X, axis=(-1, -2)))
            ch[k][i] = X / np.maximum(nrm, 1e-300)[..., None, None]
    return ch


def symbol_at_cells(M, cfg, cells, omega, ns=8, eps=1e-4):
    """H_mu_nu[xi_p, xi_q] per channel at the chosen cells (a list of index triples), circle-averaged."""
    nref = C.radial_ref(cfg)
    fr0 = C.frame(M, nref)
    a0 = C.a0_of(M, fr0)
    A_sp = jets_at(M, cfg)
    ch0 = channels(fr0, M)
    out = {k: None for k in ch0}
    idx = tuple(np.array(cells).T)
    for k in range(ns):
        beta = np.pi * k / ns
        R = C.rot_R(fr0["J"], beta)
        RT = np.swapaxes(R, -1, -2)
        Mk = R @ M @ RT
        Ak = [omega * (R @ a0 @ RT)] + jets_at(Mk, cfg)
        frk = C.frame(Mk, fr0["n"])
        # restrict to the chosen cells
        Mc = Mk[idx]; frc = {kk: (v[idx] if isinstance(v, np.ndarray) and v.shape[:3] == M.shape[:3] else v) for kk, v in frk.items()}
        Ac = [a[idx] for a in Ak]
        for name, xis in ch0.items():
            xic = [(R @ xi @ RT)[idx] for xi in xis]
            p = len(xic)
            H = np.zeros((len(cells), 4, 4, p, p))
            for pi in range(p):
                for qi in range(pi, p):
                    for mu in range(4):
                        for nu in range(mu, 4):
                            def ev(s1, s2):
                                B = [a.copy() for a in Ac]
                                B[mu] = B[mu] + s1 * eps * xic[pi]
                                B[nu] = B[nu] + s2 * eps * xic[qi]
                                return np.real(lag_density(Mc, B, frc, cfg))
                            d2 = (ev(1, 1) - ev(1, -1) - ev(-1, 1) + ev(-1, -1)) / (4 * eps * eps)
                            H[:, mu, nu, pi, qi] = d2; H[:, nu, mu, pi, qi] = d2; H[:, mu, nu, qi, pi] = d2; H[:, nu, mu, qi, pi] = d2
            out[name] = H / ns if out[name] is None else out[name] + H / ns
    return out


def hyperbolicity(H):
    """per cell: H_00 (p x p), the stiffness -H_ij, Q's PSD test (p = 1) or the QEP roots (p = 2)."""
    rng = np.random.default_rng(11)
    rows = []
    for c in range(H.shape[0]):
        Hc = H[c]
        p = Hc.shape[-1]
        H00 = Hc[0, 0]
        H0 = Hc[0, 1:]                                                    # (3, p, p)
        Hs = Hc[1:, 1:]                                                   # (3, 3, p, p)
        row = {"H00": Hc[0, 0].tolist(), "stiffness_eigs": None, "hyperbolic": None}
        if p == 1:
            h00 = float(H00[0, 0])
            S = -Hs[..., 0, 0]
            Q = np.outer(H0[:, 0, 0], H0[:, 0, 0]) - h00 * Hs[..., 0, 0]
            row["stiffness_eigs"] = [float(x) for x in np.linalg.eigvalsh(0.5 * (S + S.T))]
            row["Q_eigs"] = [float(x) for x in np.linalg.eigvalsh(0.5 * (Q + Q.T))]
            row["H00"] = h00
            row["degenerate"] = bool(abs(h00) < 1e-10 * max(1.0, float(np.max(np.abs(S)))))
            row["hyperbolic"] = bool(h00 > 0 and min(row["Q_eigs"]) >= -1e-9 * max(1.0, abs(max(row["Q_eigs"]))))
        else:
            imag_max, roots_all = 0.0, []
            for _ in range(40):
                kv = rng.normal(size=3); kv /= np.linalg.norm(kv)
                A2 = H00
                A1 = 2.0 * np.einsum("i,ipq->pq", kv, H0)
                A0 = np.einsum("i,j,ijpq->pq", kv, kv, Hs)
                try:
                    A2i = np.linalg.inv(A2)
                except np.linalg.LinAlgError:
                    imag_max = np.inf; break
                Cm = np.block([[np.zeros((p, p)), np.eye(p)], [-A2i @ A0, -A2i @ A1]])
                roots = np.linalg.eigvals(Cm)
                imag_max = max(imag_max, float(np.max(np.abs(roots.imag)) / max(np.max(np.abs(roots)), 1e-300)))
                roots_all.append(np.sort(roots.real).tolist())
            row["H00_eigs"] = [float(x) for x in np.linalg.eigvalsh(0.5 * (H00 + H00.T))]
            S = -np.einsum("ijpq->ipjq", Hs).reshape(3 * p, 3 * p)
            row["stiffness_eigs"] = [float(x) for x in np.linalg.eigvalsh(0.5 * (S + S.T))]
            row["max_rel_imag_root"] = float(imag_max)
            row["hyperbolic"] = bool(min(row["H00_eigs"]) > 0 and imag_max < 1e-6)
            row["degenerate"] = bool(min(np.abs(row["H00_eigs"])) < 1e-10)
            row["sample_roots"] = roots_all[:2]
        rows.append(row)
    return rows


def run(field, n, L, comp, label, omega, scan):
    cfg = C.cfg_v4(n, L, completion=comp, n_samples=8)
    M = np.load(field)
    X, Y, Z = INS4.coords(n, cfg["h"])
    r = np.sqrt(X * X + Y * Y + Z * Z)
    order = np.argsort(r, axis=None)
    picks = {"core_1": order[0], "core_2": order[3], "r_3": int(np.argmin(np.abs(r - 3.0) + 10.0 * (np.abs(Z) > 0.8))), "r_6": int(np.argmin(np.abs(r - 6.0) + 10.0 * (np.abs(Z) > 0.8))),
             "r_12": int(np.argmin(np.abs(r - 12.0) + 10.0 * (np.abs(Z) > 0.8))), "exterior_r_18": int(np.argmin(np.abs(r - 18.0) + 10.0 * (np.abs(Z) > 0.8)))}
    # the cell of maximal split and its radial neighbor (the fixed-K spike lives off the axes)
    import m5_32_r16_0_fields as F0
    trip0, lg0, disc0 = F0.spatial_triple(M)
    half0 = np.sqrt(np.maximum(disc0, 0.0)) / 2.0
    imax = int(np.argmax(half0))
    picks["max_split"] = imax
    picks["second_split"] = int(np.argsort(half0, axis=None)[-2])
    cells = [np.unravel_index(int(i), r.shape) for i in picks.values()]
    names = list(picks.keys())
    rec = {"label": label, "field": os.path.relpath(field, RES), "n": n, "L": L, "completion": comp, "omega": omega, "cells": {nm: {"index": [int(x) for x in c], "r": float(r[c])} for nm, c in zip(names, cells)}, "c_s": cfg["cs"], "omega_threshold_author": float(np.sqrt(cfg["cs"] / 16.0))}
    fr = C.frame(M, C.radial_ref(cfg))
    dom = C.domain(fr, cfg)
    rec["domain"] = dom
    trip = np.sort(np.stack([np.real(fr["l1"])] + [np.real((fr["s"] + sg * np.sqrt(np.maximum(fr["s"] ** 2 - 4 * fr["p"], 0))) / 2) for sg in (1, -1)], -1), axis=-1)
    for nm, c in zip(names, cells):
        rec["cells"][nm]["spatial_triple"] = [float(x) for x in trip[c][::-1]]
        rec["cells"][nm]["half_split"] = float((trip[c][2] - trip[c][1]) / 2.0) if False else float(np.sqrt(max(float(fr["s"][c]) ** 2 - 4 * float(fr["p"][c]), 0.0)) / 2.0)
    omegas = [omega] + ([0.0, 0.05, 0.1, 0.15, 0.2, 0.25] if scan else [])
    rec["results"] = {}
    for om in omegas:
        t = time.time()
        H = symbol_at_cells(M, cfg, cells, om)
        res = {}
        for name, Hc in H.items():
            rows = hyperbolicity(Hc)
            res[name] = {nm: rows[i] for i, nm in enumerate(names)}
        rec["results"][f"{om:.4f}"] = res
        summ = {name: {nm: (res[name][nm]["hyperbolic"], "deg" if res[name][nm].get("degenerate") else "") for nm in names} for name in res}
        log(f"omega {om:.4f} ({time.time() - t:.0f} s): " + "; ".join(f"{name}: " + ",".join(f"{nm}={'H' if v[0] else 'N'}{v[1]}" for nm, v in d.items()) for name, d in summ.items()))
    json.dump(rec, open(os.path.join(CK, f"r16_4_{label}.json"), "w"), indent=1, default=float)
    log(f"written checkpoints/m5_32_r16/r16_4_{label}.json")
    return rec


def collect():
    import glob
    out = {"rung": "R16-4", "runs": {}}
    for p in sorted(glob.glob(os.path.join(CK, "r16_4_*.json"))):
        r = json.load(open(p))
        out["runs"][r["label"]] = r
    json.dump(out, open(os.path.join(DATA, "m5_32_r16_4.json"), "w"), indent=1, default=float)
    log(f"collected {len(out['runs'])}")
    # the summary plot: per background and channel, the minimal static stiffness eigenvalue (log-signed) at the sampled cells
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labs = [k for k in ("seed_r15m_n32_scan", "r16_1_end_n32_scan", "r16_3_K50_end_n32", "r16_3_K200_end_n32", "r16_3_K50_end_n48", "r16_3_K200_end_n48") if k in out["runs"]]
    chans = ["tilt_n_e", "tilt_n_f", "split_doublet", "boost_n", "boost_e"]
    fig, axes = plt.subplots(1, len(labs), figsize=(3.2 * len(labs), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, lab in zip(axes, labs):
        r = out["runs"][lab]
        om = f"{r['omega']:.4f}"
        res = r["results"][om]
        cells = list(r["cells"].keys())
        for ci, ch in enumerate(chans):
            vals = [min(res[ch][c]["stiffness_eigs"]) for c in cells]
            hyp = [res[ch][c]["hyperbolic"] for c in cells]
            xs = np.arange(len(cells)) + 0.15 * (ci - 2)
            ax.bar(xs, [np.sign(v) * np.log10(1 + abs(v) * 1e3) for v in vals], width=0.14, color=["tab:green" if h else "tab:red" for h in hyp], label=ch if ax is axes[0] else None)
        ax.set_xticks(range(len(cells))); ax.set_xticklabels(cells, rotation=60, fontsize=6); ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{lab}\nomega {om}", fontsize=7)
    axes[0].set_ylabel("sign(s) log10(1 + 1000 |s|), s = min static stiffness eigenvalue\n(green hyperbolic, red not)", fontsize=7)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color="gray")] * 0 + [], [], fontsize=6)
    fig.suptitle("R16-4: the principal symbol per channel (bars left to right: tilt n-e, tilt n-f, split doublet, boost n, boost e)", fontsize=8)
    p = os.path.join(PLOTS, "m5_32_r16_4_symbol.png")
    fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    log(f"plot {os.path.relpath(p, RES)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "collect"])
    ap.add_argument("--field")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--L", type=float, default=48.0)
    ap.add_argument("--comp", default="rebuild")
    ap.add_argument("--label", default="x")
    ap.add_argument("--omega", type=float, default=0.05)
    ap.add_argument("--scan", action="store_true")
    a = ap.parse_args(ARGS)
    if a.mode == "run":
        run(a.field, a.n, a.L, a.comp, a.label, a.omega, a.scan)
    else:
        collect()
