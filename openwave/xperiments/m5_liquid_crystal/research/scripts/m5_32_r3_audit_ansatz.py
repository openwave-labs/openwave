"""M5.32 R3 arm (i): INDEPENDENT ADVERSARIAL AUDIT of the two-hedgehog product ansatz.

The producer's script (m5_32_r3_i_ansatz.py) and JSON were NOT read. Only the registry
densities are shared (m5_32_lagrangian.py I1, V4; m5_32_terms_ext.py I1_h): the field
builder, the lattice sum, the S/T split and every number below are this script's own.

EQUATIONS FIRST
---------------
Boost generators (4x4, index 0 = time)::

    (G_b)_{0b} = (G_b)_{b0} = 1,  b = 1..3

Closed-form boost with rapidity vector v (|v| = phi, n = v/phi)::

    exp(v . G) = 1 + sinh(phi) (n . G) + (cosh(phi) - 1) (n . G)^2,
    (n . G)^2 = diag(1, n n^T)          [(n.G)^3 = n.G, so the series closes]

Two-center product ansatz (centers z = -d, +d; d = the half separation)::

    r_k = x - x_k,   v_k = m f(|r_k|) r_k,   f(r) = 1/sqrt(r^2 + r_c^2)
    o = o1 o2 = exp(v_1 . G) exp(v_2 . G),   M = o M0 o^T
    M0 = diag(32, 1, 0.3, 0)  (toy point, s = -1 branch: M_vac = diag(-s g, 1, delta, 0))

Energy on the certified sym stencil (h = L/n, no pin)::

    E_lambda = 4 h^3 sum_cells [(1 - lambda) I1 + lambda I1_h] + h^3 sum_cells V4
    S = (I1_h + I1)/2,  T = (I1_h - I1)/2   =>  (1 - lambda) I1 + lambda I1_h = S - (1 - 2 lambda) T
    E_lambda = 4 [S_tot - (1 - 2 lambda) T_tot] + V4_tot
    E_int(d) = E(pair, d) - 2 E(single),  single = one hedgehog at the origin
    REPULSIVE := E_int decreasing in d

Two exact structural facts (checked numerically below):

    (F1) tr((M eta)^p) = tr((o M0 eta o^{-1})^p) = tr((M0 eta)^p) for any Lorentz o
         (o^T eta o = eta  =>  o^T = eta o^{-1} eta), so V4 is CONSTANT on any Lorentz
         orbit field and V4_int = 0 exactly (up to roundoff).
    (F2) In the frame of the local timelike eigenvector u of M eta, h_cov = 1, so
         I1_h = sum_{i<j} tr(F F^T) and I1 = sum_{i<j} tr(eta F eta F^T); with F
         antisymmetric, T = 2 sum_{i<j} sum_b F_ij[0,b]^2 >= 0 (the time row) and
         S = sum_{i<j} sum_{b,c>0} F_ij[b,c]^2 >= 0 (the spatial block), per cell.
    (F3) A boost of FIXED direction n with any scalar profile phi(x) has
         d_i M = phi_i X(phi), so F_ij = phi_i phi_j (X eta X - X eta X) = 0: the
         "uniform boost direction z with radial amplitude" family carries no
         curvature at all (S = T = 0).

Orders in m (single center, from R0 K3): M1 = m g a.G is pure time row, so the spatial
block of F is O(m^2) and its time row O(m^3): S ~ m^4, T ~ m^6, T/S ~ m^2 in the small-m
regime; at m ~ 1 the expansion fails and T/S is measured, not predicted.

Outputs: ../data/m5_32_r3_audit_ansatz.json
Run: nice -n 10 /opt/anaconda3/envs/master312/bin/python3 m5_32_r3_audit_ansatz.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(DATA, "m5_32_r3_audit_ansatz.json")


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L0 = _load("m5_32_lagrangian", "m5_32_lagrangian.py")
LX = _load("m5_32_terms_ext", "m5_32_terms_ext.py")
B3 = L0.B3
I1 = L0.REGISTRY["I1"]
V4 = L0.REGISTRY["V4"]
I1H = LX.REGISTRY_EXT["I1_h"]

G_B = np.zeros((3, 4, 4))
for b in range(3):
    G_B[b, 0, b + 1] = 1.0
    G_B[b, b + 1, 0] = 1.0


# ================= field builders (own) =================
def boost_closed(v):
    """v: (..., 3) rapidity vectors -> (..., 4, 4) exp(v . G), closed form."""
    phi = np.sqrt(np.sum(v * v, axis=-1))
    safe = np.where(phi > 0, phi, 1.0)
    n = v / safe[..., None]
    nG = np.einsum("...b,bij->...ij", n, G_B)
    nG2 = np.zeros(v.shape[:-1] + (4, 4))
    nG2[..., 0, 0] = 1.0
    nG2[..., 1:, 1:] = n[..., :, None] * n[..., None, :]
    I = np.broadcast_to(np.eye(4), nG2.shape)
    return I + np.sinh(phi)[..., None, None] * nG + (np.cosh(phi) - 1.0)[..., None, None] * nG2


def rapidity(X, center, m, r_c, profile="hedgehog", w=2.0):
    """rapidity vector field v(x) for one center."""
    r = X - np.asarray(center)
    R = np.sqrt(np.sum(r * r, axis=-1))
    if profile == "hedgehog":                       # m r / sqrt(R^2 + r_c^2)
        return m * r / np.sqrt(R * R + r_c * r_c)[..., None]
    if profile == "hedgehog_gauss":                 # m exp(-R^2/w^2) r / R
        Rs = np.sqrt(R * R + r_c * r_c)
        return m * np.exp(-R * R / w**2)[..., None] * r / Rs[..., None]
    if profile == "hedgehog_tanh":                  # m tanh(R/2) r / R (M5.21.14 wide family)
        Rs = np.sqrt(R * R + r_c * r_c)
        return m * np.tanh(R / 2.0)[..., None] * r / Rs[..., None]
    if profile == "uniform_z_gauss":                # m exp(-R^2/w^2) e_z
        v = np.zeros(X.shape)
        v[..., 2] = m * np.exp(-R * R / w**2)
        return v
    if profile == "vortex_gauss":                   # m exp(-R^2/w^2) (-y, x, 0)/rho
        rho = np.sqrt(r[..., 0] ** 2 + r[..., 1] ** 2 + r_c * r_c)
        v = np.zeros(X.shape)
        v[..., 0] = -r[..., 1] / rho
        v[..., 1] = r[..., 0] / rho
        return m * np.exp(-R * R / w**2)[..., None] * v
    raise ValueError(profile)


def build_M(cfg, M0, centers, m, r_c, profile="hedgehog", w=2.0):
    x, y, z = B3.coords(cfg["n"], cfg["h"])
    X = np.stack([x, y, z], axis=-1)
    o = np.broadcast_to(np.eye(4), X.shape[:-1] + (4, 4)).copy()
    for c in centers:
        o = o @ boost_closed(rapidity(X, c, m, r_c, profile, w))
    return o @ M0 @ o.swapaxes(-1, -2)


# ================= own lattice sum through the registry densities =================
def split_sums(M, cfg, p, slab=4):
    """(S_tot, T_tot, V4_tot, min S, min T) with S = (I1_h + I1)/2, T = (I1_h - I1)/2,
    summed with h^3 over the certified sym stencil (density per branch, averaged)."""
    n = cfg["n"]
    S_tot = T_tot = V_tot = 0.0
    S_min = T_min = np.inf
    for A, wt in L0.lattice_jets(M, cfg):
        for i0 in range(0, n, slab):
            As, Ms = A[:, i0:i0 + slab], M[i0:i0 + slab]
            d1 = I1.density(As, Ms, p)
            dh = I1H.density(As, Ms, p)
            S = 0.5 * (dh + d1)
            T = 0.5 * (dh - d1)
            S_tot += wt * S.sum()
            T_tot += wt * T.sum()
            S_min = min(S_min, S.min())
            T_min = min(T_min, T.min())
        V_tot += wt * V4.density(A, M, p).sum()
    h3 = cfg["h"] ** 3
    return h3 * S_tot, h3 * T_tot, h3 * V_tot, float(S_min), float(T_min)


def energy_lambda(S, T, V, lam):
    return 4.0 * (S - (1.0 - 2.0 * lam) * T) + V


def run_point(n, L, m, r_c, M0, p, ds, profile="hedgehog", w=2.0, lams=(0.0, 1.0), log=print):
    cfg = B3.base_cfg(n=n, L=float(L), s=p["s"], g=p["g"], delta=p["delta"])
    t0 = time.time()
    S1, T1, V1, _, _ = split_sums(build_M(cfg, M0, [(0.0, 0.0, 0.0)], m, r_c, profile, w), cfg, p)
    single = {"S": S1, "T": T1, "V4": V1, "E": {str(l): energy_lambda(S1, T1, V1, l) for l in lams}}
    rows = []
    for d in ds:
        Sp, Tp, Vp, Smin, Tmin = split_sums(
            build_M(cfg, M0, [(0.0, 0.0, -d), (0.0, 0.0, d)], m, r_c, profile, w), cfg, p)
        S_int, T_int, V_int = Sp - 2 * S1, Tp - 2 * T1, Vp - 2 * V1
        row = {"d": d, "S_pair": Sp, "T_pair": Tp, "V4_pair": Vp, "S_int": S_int, "T_int": T_int,
               "V4_int": V_int, "T_over_S_pair": Tp / Sp if Sp else None,
               "T_over_S_int": T_int / S_int if S_int else None,
               "min_S_cell": Smin, "min_T_cell": Tmin,
               "E_int": {str(l): energy_lambda(Sp, Tp, Vp, l) - 2 * single["E"][str(l)] for l in lams}}
        rows.append(row)
        log(f"  n={n} L={L} m={m} r_c={r_c} {profile}: d={d:5.1f} S_int={S_int:+.6g} T_int={T_int:+.6g} "
            f"T/S(pair)={row['T_over_S_pair']:.3e} V4_int={V_int:+.2e} "
            + " ".join(f"E_int(l={l})={row['E_int'][str(l)]:+.6g}" for l in lams))
    return {"n": n, "L": L, "h": cfg["h"], "m": m, "r_c": r_c, "profile": profile, "w": w,
            "M0": np.diag(M0).tolist(), "single": single, "pair": rows,
            "seconds": time.time() - t0}


def sign_word(vals):
    """monotonic verdict of E_int over increasing d."""
    diffs = np.diff(vals)
    if np.all(diffs < 0):
        return "REPULSIVE (E_int decreasing in d)"
    if np.all(diffs > 0):
        return "ATTRACTIVE (E_int increasing in d)"
    return "NON-MONOTONIC"


def main():
    t_all = time.time()
    res = {"conventions": {
        "M0_toy": [32, 1, 0.3, 0], "s": -1, "g": 32, "delta": 0.3, "w_V4": L0.W1,
        "stencil": "sym (certified), h = L/n, no pin", "single": "one hedgehog at the origin",
        "E_lambda": "4 h^3 sum[(1-lambda) I1 + lambda I1_h] + h^3 sum V4 = 4[S - (1-2 lambda) T] + V4",
        "S": "(I1_h + I1)/2 per cell", "T": "(I1_h - I1)/2 per cell",
        "field": "o = exp(v1.G) exp(v2.G), v_k = m (x - x_k)/sqrt(|x - x_k|^2 + r_c^2), M = o M0 o^T",
        "repulsive": "E_int(d) decreasing in d", "threads": os.environ.get("OMP_NUM_THREADS")}}
    p = L0.default_params(s=-1.0, g=32.0, delta=0.3)
    M0 = np.diag([32.0, 1.0, 0.3, 0.0])

    # ---- C0: builder checks (closed form vs expm; Lorentz property; F1/F3 facts) ----
    rng = np.random.default_rng(3200)
    vs = rng.normal(size=(5, 3)) * np.array([0.03, 0.5, 2.0, 1.0, 0.1])[:, None]
    err = max(np.max(np.abs(boost_closed(v) - expm(np.einsum("b,bij->ij", v, G_B)))) for v in vs)
    o = boost_closed(vs[2])
    lor = np.max(np.abs(o.T @ L0.ETA @ o - L0.ETA))
    cfg8 = B3.base_cfg(n=16, L=16.0, s=-1.0, g=32.0, delta=0.3)
    Mt = build_M(cfg8, M0, [(0, 0, -3.0), (0, 0, 3.0)], 0.7, 0.5)
    tr_dev = max(np.max(np.abs(t - t.flat[0])) for t in L0.v4_traces_np(Mt))
    Mz = build_M(cfg8, M0, [(0, 0, -3.0), (0, 0, 3.0)], 0.7, 0.5, "uniform_z_gauss", 3.0)
    Sz, Tz, Vz, _, _ = split_sums(Mz, cfg8, p)
    Mvac = np.broadcast_to(M0, (16, 16, 16, 4, 4)).copy()
    Sv, Tv, Vv, _, _ = split_sums(Mvac, cfg8, p)
    res["C0_builder"] = {"closed_vs_expm_max_abs": float(err), "lorentz_max_dev": float(lor),
                         "F1_trace_conservation_max_dev_m0.7": float(tr_dev),
                         "F3_uniform_z_gauss_S_T_V4_n16": [Sz, Tz, Vz],
                         "vacuum_null_n16_S_T_V4": [Sv, Tv, Vv]}
    print("C0:", json.dumps(res["C0_builder"]))

    # ---- P1 / P2: the producer's point, n = 64, L = 64, h = 1, m = 0.05, r_c = 0.5 ----
    ds = [4, 6, 8, 10, 12, 14]
    P1 = run_point(64, 64, 0.05, 0.5, M0, p, ds)
    prod = {"0.0": [2483.95, 1713.10, 1296.45, 1033.87, 852.51, 719.17],
            "1.0": [2493.70, 1718.82, 1300.47, 1036.96, 855.01, 721.28]}
    cmp = {}
    for l in ("0.0", "1.0"):
        mine = [r["E_int"][l] for r in P1["pair"]]
        cmp[l] = {"audit": mine, "producer": prod[l],
                  "max_rel_diff": float(max(abs(a - b) / abs(b) for a, b in zip(mine, prod[l]))),
                  "monotone": sign_word(mine)}
    res["P1"] = {"run": P1, "compare": cmp}
    print("P1 compare:", json.dumps({k: (v["max_rel_diff"], v["monotone"]) for k, v in cmp.items()}))

    # P2: T/S scaling with m at n = 48 (cheaper) plus the P1 point
    P2 = {"n64_m0.05": {"d": ds, "T_over_S_pair": [r["T_over_S_pair"] for r in P1["pair"]],
                        "T_over_S_int": [r["T_over_S_int"] for r in P1["pair"]],
                        "S_int_positive_decreasing": bool(np.all(np.diff([r["S_int"] for r in P1["pair"]]) < 0)
                                                          and all(r["S_int"] > 0 for r in P1["pair"])),
                        "T_int_positive_decreasing": bool(np.all(np.diff([r["T_int"] for r in P1["pair"]]) < 0)
                                                          and all(r["T_int"] > 0 for r in P1["pair"]))}}
    scan = {}
    for m in (0.02, 0.05, 0.1, 0.2):
        r = run_point(48, 48, m, 0.5, M0, p, [6, 10, 14])
        scan[str(m)] = r
    P2["m_scan_n48"] = scan
    ms = [0.02, 0.05, 0.1, 0.2]
    ratios = [scan[str(m)]["pair"][1]["T_over_S_pair"] for m in ms]
    slope = float(np.polyfit(np.log(ms), np.log(ratios), 1)[0])
    P2["T_over_S_pair_d10_vs_m"] = dict(zip([str(m) for m in ms], ratios))
    P2["loglog_slope_T_over_S_vs_m"] = slope
    res["P2"] = P2
    print("P2 T/S(d=10) vs m:", P2["T_over_S_pair_d10_vs_m"], "slope:", slope)

    # ---- P3(a): large m, n = 48 ----
    P3a = {}
    for m in (0.5, 1.0, 2.0):
        P3a[str(m)] = run_point(48, 48, m, 0.5, M0, p, [6, 10, 14])
        for l in ("0.0", "1.0"):
            P3a[str(m)]["monotone_" + l] = sign_word([r["E_int"][l] for r in P3a[str(m)]["pair"]])
    res["P3a"] = P3a

    # ---- P3(b): other profiles, n = 48 ----
    P3b = {}
    for prof, m, w in (("uniform_z_gauss", 1.0, 3.0), ("hedgehog_gauss", 1.0, 3.0),
                       ("hedgehog_gauss", 2.0, 3.0), ("vortex_gauss", 1.0, 3.0),
                       ("vortex_gauss", 2.0, 3.0), ("hedgehog_tanh", 0.5, 2.0),
                       ("hedgehog_tanh", 1.0, 2.0)):
        r = run_point(48, 48, m, 0.5, M0, p, [6, 10, 14], profile=prof, w=w)
        for l in ("0.0", "1.0"):
            r["monotone_" + l] = sign_word([x["E_int"][l] for x in r["pair"]])
        P3b[f"{prof}_m{m}"] = r
    res["P3b"] = P3b

    # ---- P3(c): the gaussian families in their overlap range (w = 3: no overlap at d >= 10) ----
    P3c = {}
    for prof, m, w in (("hedgehog_gauss", 0.5, 3.0), ("hedgehog_gauss", 1.0, 3.0),
                       ("hedgehog_gauss", 2.0, 3.0), ("vortex_gauss", 1.0, 3.0)):
        r = run_point(48, 48, m, 0.5, M0, p, [2, 3, 4, 5, 6], profile=prof, w=w)
        for l in ("0.0", "1.0"):
            r["monotone_" + l] = sign_word([x["E_int"][l] for x in r["pair"]])
        P3c[f"{prof}_m{m}"] = r
    res["P3c"] = P3c

    # ---- P3(d): resolution check of the large-m sign flip (h = 0.5: n = 96, L = 48) ----
    P3d = {}
    for m in (1.0, 2.0):
        r = run_point(96, 48, m, 0.5, M0, p, [6, 10, 14])
        for l in ("0.0", "1.0"):
            r["monotone_" + l] = sign_word([x["E_int"][l] for x in r["pair"]])
        P3d[str(m)] = r
    res["P3d"] = P3d

    # ---- P5: the author's vacuum M0 = diag(32, 0, 0, 0) (delta = 0; V4 constant, cancels) ----
    pa = L0.default_params(s=-1.0, g=32.0, delta=0.0)
    M0a = np.diag([32.0, 0.0, 0.0, 0.0])
    P5 = {}
    for m in (0.05, 2.0):
        r = run_point(48, 48, m, 0.5, M0a, pa, [6, 10, 14])
        for l in ("0.0", "1.0"):
            r["monotone_" + l] = sign_word([x["E_int"][l] for x in r["pair"]])
        P5[str(m)] = r
    res["P5_author_M0"] = P5

    # best T/S over everything (S_pair must be a real curvature sum, not roundoff:
    # the uniform-direction family has S = 0 exactly and only the O(h) stencil residual in T)
    best = None
    for key, blk in [("P1", {"P1": P1}), ("P3a", P3a), ("P3b", P3b), ("P3c", P3c), ("P3d", P3d), ("P2", scan), ("P5", P5)]:
        for k, r in blk.items():
            for row in r["pair"]:
                if row["S_pair"] > 1e-6 * abs(row["T_pair"]) and row["S_pair"] > 0 and (best is None or row["T_over_S_pair"] > best["T_over_S_pair"]):
                    best = {"block": key, "case": k, "n": r["n"], "L": r["L"], "h": r["h"], "m": r["m"],
                            "r_c": r["r_c"], "profile": r["profile"], **row}
    res["best_T_over_S"] = best
    # any T_int > S_int ?
    viol = []
    for key, blk in [("P1", {"P1": P1}), ("P3a", P3a), ("P3b", P3b), ("P3c", P3c), ("P3d", P3d), ("P2", scan), ("P5", P5)]:
        for k, r in blk.items():
            for row in r["pair"]:
                if abs(row["T_int"]) > abs(row["S_int"]) and abs(row["T_int"]) > 1e-3:
                    viol.append({"block": key, "case": k, "m": r["m"], "d": row["d"],
                                 "S_int": row["S_int"], "T_int": row["T_int"]})
    res["T_int_exceeds_S_int_cases"] = viol

    # ---- P4: box independence n = 48 vs 64 at m = 0.05, d = 10 ----
    r48 = scan["0.05"]["pair"][1]
    r64 = P1["pair"][3]
    res["P4"] = {"vacuum_null_n16": res["C0_builder"]["vacuum_null_n16_S_T_V4"],
                 "d10_m0.05": {"n48_L48": r48["E_int"], "n64_L64": r64["E_int"],
                               "rel_diff_l0": abs(r48["E_int"]["0.0"] - r64["E_int"]["0.0"]) / abs(r64["E_int"]["0.0"]),
                               "rel_diff_l1": abs(r48["E_int"]["1.0"] - r64["E_int"]["1.0"]) / abs(r64["E_int"]["1.0"])},
                 "sign_n48": sign_word([x["E_int"]["0.0"] for x in scan["0.05"]["pair"]]),
                 "sign_n64": sign_word([x["E_int"]["0.0"] for x in P1["pair"]])}
    print("P4:", json.dumps(res["P4"]))

    res["runtime_seconds"] = time.time() - t_all
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print("wrote", OUT, f"in {res['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
