"""M5.32 R3 construction (i): the author's two-boost-hedgehog ansatz at FINITE m
on the 3D lattice, under the covariant lambda-family, read for the sign of the
pair interaction vs separation.

Source construction: theory/duda_2026-08-17_newton_for_boost_hedgehogs.pdf,
reproduced at O(m) in m5_32_r0_c_notebook.py (fit A + B/d, B = +167.7 > 0)
and audited in m5_32_r0_audit_notebook.py (far field (32 pi/3)(ln d)/d + beta/d;
NO time-row curvature at O(m): first at m^3 in F). Here nothing is expanded.

EQUATIONS FIRST
---------------
Conventions (registries m5_32_lagrangian.py + m5_32_terms_ext.py, imported,
never modified): index 0 = time, eta = diag(-1, 1, 1, 1), M(x) real
symmetric 4x4, jets A_i = d_i M on the certified sym stencil (fwd / bwd
branches, the density per branch, averaged), A_0 = 0 (static read),
    F_ij = A_i eta A_j - A_j eta A_i                       (i, j = 1..3)
    I1   = sum_{i<j} <F_ij, F_ij>_eta,  <F,G>_eta = tr(eta F eta G^T)
    I1_h = sum_{i<j} tr(h F_ij h F_ij^T),  h = eta + 2 w w^T, w = eta u,
           u the timelike unit eigenvector of M eta (u^T eta u = -1)
    V4   = w4 sum_{p=1..4} (tr((M eta)^p) - C_p)^2, C_p = (s g)^p + 1 + delta^p
The lambda-family (the R2 candidate; static Hamiltonian = -Lagrangian):
    L_lambda = -4 [(1 - lambda) I1 + lambda I1_h] - V4
    E_lambda = 4 h^3 sum_cells [(1 - lambda) I1 + lambda I1_h] + h^3 sum V4

The S / T decomposition (frame-free; derived, then checked to roundoff):
    I1_h - I1 = 4 (F w)^T eta (F w)  (the w^T F w term drops: F antisymmetric)
    T := 2 sum_{i<j} (F_ij w)^T eta (F_ij w)  >= 0    ("time-row" content, the
         content of F along the local timelike eigenvector; in the u = e0
         frame T = 2 sum_b F_0b^2)
    S := I1 + T                                          (spatial-spatial content)
    I1 = S - T,  I1_h = S + T,  E_lambda = 4 [S - (1 - 2 lambda) T] + V4

The ansatz (the author's, EXACT in m):
    G_b: (G_b)_{0b} = (G_b)_{b0} = 1, b = 1..3 (boost generators)
    o_k = exp( m f(r_k) (x - x_k) . G_b ),   r_k = |x - x_k|,
    f(r) = 1 / sqrt(r^2 + r_c^2)              (core regularization r_c; the
          notebook's f = 1/r; the boost rapidity is m r f(r) -> m: a
          unit-vector hedgehog of CONSTANT rapidity m at large r)
    closed form (exact for a boost, K = n . G_b, K^2 = diag(1, n n^T),
    K^3 = K):  o = 1 + sinh(theta) K + (cosh(theta) - 1) K^2,
               theta = m r f(r), n = (x - x_k)/r
    o = o_1 o_2  (product ansatz),  M = o M0 o^T
    M0 = diag(-s g, 1, delta, 0) at the toy point (s = -1, g = 32,
         delta = 0.3): the certified vacuum;  M0 = diag(g, 0, 0, 0) = the
         author's notebook control (NOT a V4 vacuum: V4 is then a nonzero
         CONSTANT per cell since o is a Lorentz orbit pointwise)
    Because o^T eta = eta o^{-1} (o a Lorentz boost), (M eta) = o (M0 eta) o^{-1}
    keeps the spectrum of M0 eta pointwise, so V4 is constant over the box
    (zero at the toy point) and u = o e0 exactly (both checked numerically).

Energies (same box, same r_c, same stencil; vacuum-subtracted):
    E_int(d; lambda) = [E(pair, d) - E(vac)] - 2 [E(single) - E(vac)]
    pair: centers (0, 0, +-d); single: center at the origin; vac: m = 0
    (the pair is NOT relaxed: this is the ansatz read, construction (i))
Reads:
    sign of dE_int/dd over the outer window d >= 8 (consecutive differences);
    FORCE = -dE_int/dd; ATTRACTION <=> E_int increases with d;
    fits A + B/d (outer window) and A + B/d + C ln(d)/d (all d and outer);
    E_int = 4 [S_int - (1 - 2 lambda) T_int] + V4_int per lambda.
Calibration (pre-registered control (a)): lambda = 0, M0 = diag(g, 0, 0, 0),
small m: at O(m^4) the lattice energy is 8 h^3 sum Hs with Hs the notebook's
density (I1 = 2 Hs there; the eta in the bracket only flips the sign of the
spatial block), so the audit's far field predicts
    E_int ~ 8 m^4 g^4 [ (32 pi/3) (ln d)/d + beta/d ],  C/(8 m^4 g^4) -> 33.51
and the sign must be REPULSIVE (E_int decreasing with d, B > 0 in the 2-term
fit) before the instrument is trusted.
Controls (b) m = 0 pair: E_int = 0 to roundoff; (c) the box ladder n = L = 48
vs 64 (h = 1) must agree on the sign at d = 10; (d) the mutation lambda = 1 vs
lambda = 0 (the coefficient of T flips from -1 to +1).

Grid: M0 in {toy, author} x box in {(48, 48), (64, 64)} x r_c in {0.25, 0.5, 1}
x m in {0.02, 0.05, 0.1, 0.2} x {single, d = 4, 6, 8, 10, 12, 14}, plus the
m = 0 vacuum and null pair per (M0, box). Every field evaluation returns the
h^3 sums of I1, I1_h, S, T, V4, so all lambda are read from one evaluation.

Out: ../data/m5_32_r3_ansatz.json, ../plots/m5_32_r3_ansatz.png
Run: /opt/anaconda3/envs/master312/bin/python3 m5_32_r3_i_ansatz.py [--smoke] [--workers N]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
PLOTS = os.path.join(HERE, "..", "plots")
OUT_JSON = os.path.join(DATA, "m5_32_r3_ansatz.json")
OUT_PNG = os.path.join(PLOTS, "m5_32_r3_ansatz.png")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LAG = _load("m5_32_lagrangian", "m5_32_lagrangian.py")
EXT = _load("m5_32_terms_ext", "m5_32_terms_ext.py")
B3 = LAG.B3
ETA = LAG.ETA
ETA_D = np.diag(ETA)

# ---------------- the pre-registered grid ----------------
S_TOY, G_TOY, DELTA_TOY = -1.0, 32.0, 0.3
M0_SET = {"toy": np.diag([-S_TOY * G_TOY, 1.0, DELTA_TOY, 0.0]),
          "author": np.diag([G_TOY, 0.0, 0.0, 0.0])}
BOXES = [(48, 48.0), (64, 64.0)]
RCS = [0.25, 0.5, 1.0]
MS = [0.02, 0.05, 0.1, 0.2]
DS = [4, 6, 8, 10, 12, 14]
LAMBDAS = [0.0, 0.5, 0.75, 1.0]
OUTER = [d for d in DS if d >= 8]
ALPHA_PRED = 32.0 * np.pi / 3.0        # the R0 audit's (ln d)/d coefficient
T_START = time.time()


def log(msg):
    print(f"[{time.time() - T_START:8.1f}s] {msg}", flush=True)


# ---------------- the ansatz ----------------
def boost_field(X, Y, Z, center, m, rc):
    """o(x) = exp(m f(r) (x - c) . G_b), closed form, shape (..., 4, 4)."""
    vx, vy, vz = X - center[0], Y - center[1], Z - center[2]
    r = np.sqrt(vx * vx + vy * vy + vz * vz)
    theta = m * r / np.sqrt(r * r + rc * rc)             # rapidity
    safe = np.where(r > 0, r, 1.0)
    n = np.stack([vx, vy, vz], axis=-1) / safe[..., None]
    n = np.where((r > 0)[..., None], n, 0.0)
    sh = np.sinh(theta)[..., None, None]
    ch1 = (np.cosh(theta) - 1.0)[..., None, None]
    K = np.zeros(r.shape + (4, 4))
    K[..., 0, 1:] = n
    K[..., 1:, 0] = n
    K2 = np.zeros_like(K)
    K2[..., 0, 0] = 1.0
    K2[..., 1:, 1:] = n[..., :, None] * n[..., None, :]
    o = np.broadcast_to(np.eye(4), K.shape) + sh * K + ch1 * K2
    return o


def ansatz(M0, centers, m, rc, n, L):
    X, Y, Z = B3.coords(n, L / n)
    o = None
    for c in centers:
        ok = boost_field(X, Y, Z, c, m, rc)
        o = ok if o is None else o @ ok
    if o is None:
        return np.broadcast_to(M0, (n, n, n, 4, 4)).copy()
    return o @ M0 @ o.swapaxes(-1, -2)


# ---------------- densities ----------------
def t_density(F, M):
    """T = 2 sum_{i<j} (F_ij w)^T eta (F_ij w), w = eta u (u from the registry)."""
    u = EXT.timelike_eig_np(M)[0]
    w = u @ ETA
    tot = 0.0
    for i in range(1, 4):
        for j in range(i + 1, 4):
            Fw = np.einsum("...ab,...b->...a", F[..., i, j, :, :], w)
            tot = tot + np.einsum("a,...a,...a->...", ETA_D, Fw, Fw)
    return 2.0 * tot


def evaluate(M, cfg, p):
    """h^3 sums of I1, I1_h, S, T, V4 over the certified stencil branches."""
    K1 = LAG.REGISTRY["I1"]._K()
    h3 = cfg["h"] ** 3
    acc = {"I1": 0.0, "I1_h": 0.0, "T": 0.0, "split_check": 0.0}
    for A, wt in LAG.lattice_jets(M, cfg):
        F = LAG.F_of_A(A)
        i1 = LAG.density_from_K(F, K1)
        i1h = EXT.I1_h_np(A, M, p)
        t = t_density(F, M)
        acc["I1"] += wt * float(np.sum(i1))
        acc["I1_h"] += wt * float(np.sum(i1h))
        acc["T"] += wt * float(np.sum(t))
        acc["split_check"] = max(acc["split_check"], float(
            np.max(np.abs(i1h - i1 - 2.0 * t)) / max(np.max(np.abs(i1h)), 1e-300)))
        del F, i1, i1h, t
    v4 = LAG.v4_density_np(None, M, p)
    out = {k: h3 * v for k, v in acc.items() if k != "split_check"}
    out["S"] = out["I1"] + out["T"]
    out["V4"] = h3 * float(np.sum(v4))
    out["V4_cell_max_abs"] = float(np.max(np.abs(v4)))
    out["V4_cell_spread"] = float(np.max(v4) - np.min(v4))
    out["split_check_rel"] = acc["split_check"]
    return out


def e_lambda(tot, lam):
    return 4.0 * ((1.0 - lam) * tot["I1"] + lam * tot["I1_h"]) + tot["V4"]


def job(args):
    """one field evaluation: (m0_name, n, L, rc, m, kind, d) -> totals."""
    m0_name, n, L, rc, m, kind, d = args
    t0 = time.time()
    cfg = B3.base_cfg(s=S_TOY, g=G_TOY, n=n, L=L, delta=DELTA_TOY)
    p = LAG.default_params(s=S_TOY, g=G_TOY, delta=DELTA_TOY)
    if kind == "vac":
        centers = []
    elif kind == "single":
        centers = [(0.0, 0.0, 0.0)]
    else:
        centers = [(0.0, 0.0, +float(d)), (0.0, 0.0, -float(d))]
    M = ansatz(M0_SET[m0_name], centers, m, rc, n, L)
    tot = evaluate(M, cfg, p)
    tot["runtime_s"] = time.time() - t0
    tot["key"] = list(args)
    return tot


# ---------------- selftests (the instrument, before any reading) ----------------
def selftests():
    from scipy.linalg import expm
    res = {}
    rng = np.random.default_rng(7)
    P = rng.uniform(-6, 6, size=(200, 3))
    m, rc = 0.2, 0.5
    o = boost_field(P[:, 0], P[:, 1], P[:, 2], (0.3, -0.2, 1.0), m, rc)
    worst = 0.0
    for k in range(200):
        v = P[k] - np.array([0.3, -0.2, 1.0])
        r = np.linalg.norm(v)
        X = np.zeros((4, 4))
        X[0, 1:] = m * v / np.sqrt(r * r + rc * rc)
        X[1:, 0] = X[0, 1:]
        worst = max(worst, np.max(np.abs(o[k] - expm(X))))
    res["closed_form_vs_expm_max_abs"] = float(worst)
    # Lorentz: o^T eta o = eta
    res["lorentz_max_abs"] = float(np.max(np.abs(
        o.swapaxes(-1, -2) @ ETA @ o - ETA)))
    # the product ansatz on a small box: V4 constant, u = o e0, split identity
    n, L = 12, 12.0
    cfg = B3.base_cfg(s=S_TOY, g=G_TOY, n=n, L=L, delta=DELTA_TOY)
    p = LAG.default_params(s=S_TOY, g=G_TOY, delta=DELTA_TOY)
    X, Y, Z = B3.coords(n, L / n)
    for name in ("toy", "author"):
        o = boost_field(X, Y, Z, (0, 0, 2.0), 0.2, 0.5) @ boost_field(
            X, Y, Z, (0, 0, -2.0), 0.2, 0.5)
        M = o @ M0_SET[name] @ o.swapaxes(-1, -2)
        tot = evaluate(M, cfg, p)
        u = EXT.timelike_eig_np(M)[0]
        oe0 = o[..., :, 0]
        du = np.minimum(np.max(np.abs(u - oe0), axis=-1),
                        np.max(np.abs(u + oe0), axis=-1))
        res[name] = {"V4_cell_spread": tot["V4_cell_spread"],
                     "V4_cell_max_abs": tot["V4_cell_max_abs"],
                     "u_vs_o_e0_max_abs": float(np.max(du)),
                     "split_check_rel": tot["split_check_rel"],
                     "T_over_S": tot["T"] / tot["S"]}
    # spectrum of M eta pointwise vs M0 eta (the Lorentz-orbit statement)
    lam = np.sort(np.linalg.eigvals(M @ ETA).real, axis=-1)
    lam0 = np.sort(np.linalg.eigvals(M0_SET["author"] @ ETA).real)
    res["spectrum_drift_max_abs"] = float(np.max(np.abs(lam - lam0)))
    ok = (res["closed_form_vs_expm_max_abs"] < 1e-12 and res["lorentz_max_abs"] < 1e-12
          and res["toy"]["V4_cell_max_abs"] < 1e-8
          and res["author"]["V4_cell_spread"] < 1e-8 * max(res["author"]["V4_cell_max_abs"], 1)
          and res["toy"]["u_vs_o_e0_max_abs"] < 1e-8
          and res["toy"]["split_check_rel"] < 1e-10
          and res["author"]["split_check_rel"] < 1e-10)
    res["all_pass"] = bool(ok)
    for k, v in res.items():
        log(f"selftest {k}: {v}")
    return res


# ---------------- fits and reads ----------------
def fit(ds, es, logterm):
    ds = np.asarray(ds, float)
    es = np.asarray(es, float)
    cols = [np.ones_like(ds), 1.0 / ds]
    if logterm:
        cols.append(np.log(ds) / ds)
    Amat = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(Amat, es, rcond=None)
    res = es - Amat @ coef
    ss = float(np.sum((es - es.mean()) ** 2))
    r2 = 1.0 - float(np.sum(res ** 2)) / ss if ss > 0 else float("nan")
    out = {"A": float(coef[0]), "B": float(coef[1]),
           "rms_residual": float(np.sqrt(np.mean(res ** 2))), "R2": r2}
    out["C"] = float(coef[2]) if logterm else 0.0
    return out


def slope_sign(ds, es):
    diffs = np.diff(np.asarray(es, float))
    if np.all(diffs > 0):
        return "+", "ATTRACTIVE (E_int increases with d; force toward each other)"
    if np.all(diffs < 0):
        return "-", "REPULSIVE (E_int decreases with d)"
    return "mixed", "MIXED (non-monotonic over the window)"


def analyze(results, selft):
    tot = {tuple(r["key"]): r for r in results}
    out = {"script": os.path.basename(__file__), "selftests": selft,
           "grid": {"M0": {k: np.diag(v).tolist() for k, v in M0_SET.items()},
                    "boxes_n_L_h": [[n, L, L / n] for n, L in BOXES], "r_c": RCS,
                    "m": MS, "d_half_separation": DS, "lambda": LAMBDAS,
                    "outer_window": OUTER, "s": S_TOY, "g": G_TOY,
                    "delta": DELTA_TOY, "w4": LAG.W1},
           "reads": {}, "controls": {}}
    reads = out["reads"]
    for m0 in M0_SET:
        for n, L in BOXES:
            vac = tot[(m0, n, L, RCS[0], 0.0, "vac", 0)]
            for rc in RCS:
                for m in MS:
                    sg = tot[(m0, n, L, rc, m, "single", 0)]
                    key = f"{m0}|n{n}|L{L:g}|rc{rc:g}|m{m:g}"
                    row = {"M0": m0, "n": n, "L": L, "h": L / n, "r_c": rc, "m": m,
                           "E_single_minus_vac": {f"lambda_{lam:g}":
                                                  e_lambda(sg, lam) - e_lambda(vac, lam)
                                                  for lam in LAMBDAS},
                           "split_check_rel_max": max(
                               tot[(m0, n, L, rc, m, "pair", d)]["split_check_rel"]
                               for d in DS),
                           "V4_int": [], "S_int": [], "T_int": [], "per_lambda": {}}
                    for d in DS:
                        pr = tot[(m0, n, L, rc, m, "pair", d)]
                        for k in ("S", "T", "V4"):
                            row[f"{k}_int"].append(pr[k] - 2 * sg[k] + vac[k])
                    for lam in LAMBDAS:
                        es = [e_lambda(tot[(m0, n, L, rc, m, "pair", d)], lam)
                              - 2 * e_lambda(sg, lam) + e_lambda(vac, lam) for d in DS]
                        eo = [e for d, e in zip(DS, es) if d in OUTER]
                        sgn, verdict = slope_sign(OUTER, eo)
                        row["per_lambda"][f"lambda_{lam:g}"] = {
                            "E_int": es, "dEdd_sign_outer": sgn, "force_read": verdict,
                            "fit_outer_A_B": fit(OUTER, eo, False),
                            "fit_outer_A_B_C": fit(OUTER, eo, True),
                            "fit_all_A_B_C": fit(DS, es, True),
                            "E_int_S_part": [4 * s for s in row["S_int"]],
                            "E_int_T_part": [-4 * (1 - 2 * lam) * t for t in row["T_int"]],
                            "E_int_V4_part": list(row["V4_int"])}
                    reads[key] = row
    # ---- controls ----
    ctl = out["controls"]
    # (a) calibration: author M0, lambda = 0, smallest m, both boxes, r_c = 0.5
    cal = {}
    for n, L in BOXES:
        for rc in RCS:
            for m in MS:
                r = reads[f"author|n{n}|L{L:g}|rc{rc:g}|m{m:g}"]["per_lambda"]["lambda_0"]
                norm = 8.0 * m ** 4 * G_TOY ** 4
                cal[f"n{n}|rc{rc:g}|m{m:g}"] = {
                    "dEdd_sign_outer": r["dEdd_sign_outer"], "force_read": r["force_read"],
                    "B_outer_2term": r["fit_outer_A_B"]["B"],
                    "C_all_3term_over_8m4g4": r["fit_all_A_B_C"]["C"] / norm,
                    "B_all_3term_over_8m4g4": r["fit_all_A_B_C"]["B"] / norm,
                    "E_int_over_8m4g4": [e / norm for e in r["E_int"]],
                    "alpha_pred_32pi_over_3": ALPHA_PRED}
    ref = cal[f"n64|rc0.5|m{MS[0]:g}"]
    ctl["a_calibration"] = {
        "rows": cal,
        "pass": bool(ref["dEdd_sign_outer"] == "-" and ref["B_outer_2term"] > 0
                     and ref["C_all_3term_over_8m4g4"] > 0),
        "statement": "author M0, lambda = 0, m = 0.02, r_c = 0.5, n = L = 64: the sign "
                     "must be REPULSIVE and the 3-term fit must carry a positive "
                     "(ln d)/d coefficient (pred 32 pi/3 = 33.51 in units 8 m^4 g^4)"}
    # (b) vacuum null
    nulls = {}
    for m0 in M0_SET:
        for n, L in BOXES:
            vac = tot[(m0, n, L, RCS[0], 0.0, "vac", 0)]
            pr = tot[(m0, n, L, RCS[0], 0.0, "pair", 10)]
            sg = tot[(m0, n, L, RCS[0], 0.0, "single", 0)]
            nulls[f"{m0}|n{n}"] = {f"lambda_{lam:g}": e_lambda(pr, lam) - 2 * e_lambda(sg, lam)
                                   + e_lambda(vac, lam) for lam in LAMBDAS}
    ctl["b_vacuum_null"] = {"E_int_m0_d10": nulls,
                            "pass": bool(max(abs(v) for r in nulls.values()
                                             for v in r.values()) < 1e-9)}
    # (c) box ladder at d = 10 (sign of the local slope d=8->12 and the outer sign)
    ladder = {}
    allok = True
    for m0 in M0_SET:
        for rc in RCS:
            for m in MS:
                for lam in LAMBDAS:
                    ent = {}
                    for n, L in BOXES:
                        r = reads[f"{m0}|n{n}|L{L:g}|rc{rc:g}|m{m:g}"]["per_lambda"][f"lambda_{lam:g}"]
                        i8, i12 = DS.index(8), DS.index(12)
                        sl = (r["E_int"][i12] - r["E_int"][i8]) / 4.0
                        ent[f"n{n}"] = {"E_int_d10": r["E_int"][DS.index(10)],
                                        "slope_d8_to_12": sl,
                                        "slope_sign": "+" if sl > 0 else "-",
                                        "outer_sign": r["dEdd_sign_outer"]}
                    agree = ent["n48"]["slope_sign"] == ent["n64"]["slope_sign"]
                    ent["agree"] = bool(agree)
                    allok = allok and agree
                    ladder[f"{m0}|rc{rc:g}|m{m:g}|lambda_{lam:g}"] = ent
    ctl["c_box_ladder"] = {"rows": ladder, "all_agree": bool(allok)}
    # (d) mutation lambda = 1 vs lambda = 0
    mut = {}
    for m0 in M0_SET:
        for n, L in BOXES:
            for rc in RCS:
                for m in MS:
                    pl = reads[f"{m0}|n{n}|L{L:g}|rc{rc:g}|m{m:g}"]["per_lambda"]
                    b0 = pl["lambda_0"]["fit_outer_A_B"]["B"]
                    b1 = pl["lambda_1"]["fit_outer_A_B"]["B"]
                    tpart = pl["lambda_1"]["E_int_T_part"]
                    spart = pl["lambda_1"]["E_int_S_part"]
                    mut[f"{m0}|n{n}|rc{rc:g}|m{m:g}"] = {
                        "B_lambda0": b0, "B_lambda1": b1,
                        "sign_flips": bool(np.sign(b0) != np.sign(b1)),
                        "outer_sign_lambda0": pl["lambda_0"]["dEdd_sign_outer"],
                        "outer_sign_lambda1": pl["lambda_1"]["dEdd_sign_outer"],
                        "T_part_over_S_part_at_d10": tpart[DS.index(10)] / spart[DS.index(10)]
                        if spart[DS.index(10)] != 0 else float("nan")}
    ctl["d_mutation"] = mut
    return out


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=130)
    cols = {0.0: "#2563EB", 0.5: "#059669", 0.75: "#D97706", 1.0: "#DC2626"}
    # (1) headline: toy, n = 64, rc = 0.5, m = 0.05, E_int(d) per lambda
    for ax, m0, title in ((axes[0, 0], "toy", "toy M0 = diag(32, 1, 0.3, 0)"),
                          (axes[0, 1], "author", "author M0 = diag(32, 0, 0, 0)")):
        r = out["reads"][f"{m0}|n64|L64|rc0.5|m0.05"]
        for lam in LAMBDAS:
            pl = r["per_lambda"][f"lambda_{lam:g}"]
            ax.plot(DS, pl["E_int"], "o-", color=cols[lam],
                    label=f"lambda = {lam:g} ({pl['dEdd_sign_outer']})")
        ax.set_title(f"E_int(d), {title}\nm = 0.05, r_c = 0.5, n = L = 64, h = 1", fontsize=9)
        ax.set_xlabel("d (half separation)")
        ax.set_ylabel("E_int = E(pair) - 2 E(single)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(color="#E5E7EB")
    # (2) calibration: author, lambda 0, m = 0.02, E_int / 8 m^4 g^4 vs alpha ln d / d + beta / d
    ax = axes[0, 2]
    for n, L in BOXES:
        c = out["controls"]["a_calibration"]["rows"][f"n{n}|rc0.5|m0.02"]
        ax.plot(DS, c["E_int_over_8m4g4"], "o-", label=f"n = L = {n}")
    r = out["reads"]["author|n64|L64|rc0.5|m0.02"]["per_lambda"]["lambda_0"]["fit_all_A_B_C"]
    norm = 8 * 0.02 ** 4 * G_TOY ** 4
    dd = np.linspace(4, 14, 100)
    ax.plot(dd, (r["A"] + r["B"] / dd + r["C"] * np.log(dd) / dd) / norm, "k--",
            label=f"fit: C/(8m^4g^4) = {r['C'] / norm:.1f} (pred 33.5)")
    ax.set_title("calibration (a): author M0, lambda = 0, m = 0.02\nE_int / (8 m^4 g^4)",
                 fontsize=9)
    ax.set_xlabel("d")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#E5E7EB")
    # (3) S / T decomposition, toy n64 rc0.5, per m
    ax = axes[1, 0]
    for m in MS:
        r = out["reads"][f"toy|n64|L64|rc0.5|m{m:g}"]
        ax.plot(DS, np.abs(np.array(r["T_int"]) / np.array(r["S_int"])), "o-",
                label=f"m = {m:g}")
    ax.set_yscale("log")
    ax.set_title("|T_int / S_int| (toy, n = L = 64, r_c = 0.5)\nE_int = 4[S - (1-2lambda)T] + V4",
                 fontsize=9)
    ax.set_xlabel("d")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#E5E7EB")
    # (4) B (outer 2-term) vs m per lambda, toy, both boxes
    ax = axes[1, 1]
    for lam in LAMBDAS:
        for (n, L), ls in zip(BOXES, ("--", "-")):
            bs = [out["reads"][f"toy|n{n}|L{L:g}|rc0.5|m{m:g}"]["per_lambda"]
                  [f"lambda_{lam:g}"]["fit_outer_A_B"]["B"] / m ** 4 for m in MS]
            ax.plot(MS, bs, "o", ls=ls, color=cols[lam],
                    label=f"lambda {lam:g}, n {n}")
    ax.set_xscale("log")
    ax.set_title("B / m^4 (outer fit A + B/d), toy, r_c = 0.5\nB > 0 = repulsive", fontsize=9)
    ax.set_xlabel("m")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(color="#E5E7EB")
    # (5) r_c sensitivity, toy n64 m0.05 lambda 0 and 1
    ax = axes[1, 2]
    for rc, mk in zip(RCS, ("s", "o", "^")):
        for lam in (0.0, 1.0):
            r = out["reads"][f"toy|n64|L64|rc{rc:g}|m0.05"]["per_lambda"][f"lambda_{lam:g}"]
            ax.plot(DS, r["E_int"], marker=mk, ls="-", color=cols[lam],
                    label=f"r_c {rc:g}, lambda {lam:g}")
    ax.set_title("r_c sensitivity (toy, n = L = 64, m = 0.05)", fontsize=9)
    ax.set_xlabel("d")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(color="#E5E7EB")
    fig.suptitle("M5.32 R3 (i): the author's two-boost-hedgehog ansatz at finite m under "
                 "L_lambda", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    log(f"plot {OUT_PNG}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    selft = selftests()
    if a.smoke:
        r = job(("toy", 64, 64.0, 0.5, 0.05, "pair", 10))
        log(f"smoke n64 pair: {r}")
        return 0
    jobs = []
    for m0 in M0_SET:
        for n, L in BOXES:
            jobs.append((m0, n, L, RCS[0], 0.0, "vac", 0))
            jobs.append((m0, n, L, RCS[0], 0.0, "single", 0))
            jobs.append((m0, n, L, RCS[0], 0.0, "pair", 10))
            for rc in RCS:
                for m in MS:
                    jobs.append((m0, n, L, rc, m, "single", 0))
                    for d in DS:
                        jobs.append((m0, n, L, rc, m, "pair", d))
    # largest boxes first for load balance
    jobs.sort(key=lambda j: -j[1])
    log(f"{len(jobs)} field evaluations, {a.workers} workers")
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            results.append(r)
            if k % 20 == 0 or k == len(jobs) - 1:
                log(f"{k + 1}/{len(jobs)} done, last {r['key']} in {r['runtime_s']:.1f}s")
    out = analyze(results, selft)
    out["raw"] = results
    out["total_runtime_s"] = time.time() - T_START
    out["workers"] = a.workers
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(PLOTS, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    log(f"json {OUT_JSON}")
    plot(out)
    log(f"done {out['total_runtime_s']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
