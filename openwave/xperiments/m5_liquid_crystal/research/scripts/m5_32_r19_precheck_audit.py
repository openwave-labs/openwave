"""ADVERSARIAL AUDIT of m5_32_r19_precheck.py (R19 mixed-bracket algebra).

Independent code: explicit matrix-product loops (no producer helpers), own
seeds, own jet distributions, own transforms. The registry module is
imported ONLY for the I1..I6 columns of the span test (C3), as allowed.
Out: m5_32_r19_precheck_audit.json beside this file.
"""
from __future__ import annotations
import importlib.util
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "m5_32_r19_precheck_audit.json")
SCRIPTS = HERE
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])
ED = np.array([-1.0, 1.0, 1.0, 1.0])
E0 = np.zeros((4, 4)); E0[0, 0] = 1.0
PS = np.eye(4) - E0


def load_registry():
    spec = importlib.util.spec_from_file_location(
        "m5_32_lagrangian", os.path.join(SCRIPTS, "m5_32_lagrangian.py"))
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv; sys.argv = [argv[0]]
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


# ---------------- own primitives (loops, one jet at a time) ----------------
def prod(A):
    """P[m][n] = A_m eta A_n for one jet A = list of 4 symmetric 4x4."""
    return [[A[m] @ ETA @ A[n] for n in range(4)] for m in range(4)]


def comm(A):
    P = prod(A)
    return [[P[m][n] - P[n][m] for n in range(4)] for m in range(4)]


def acomm(A):
    P = prod(A)
    return [[P[m][n] + P[n][m] for n in range(4)] for m in range(4)]


def ip_eta(X, Y):
    """<X, Y>_eta = sum_ab eta_a eta_b X_ab Y_ab."""
    s = 0.0
    for a in range(4):
        for b in range(4):
            s += ED[a] * ED[b] * X[a, b] * Y[a, b]
    return s


def full(X):
    """(1/2) sum_{mu nu} eta^mu eta^nu <X_mn, X_mn>_eta."""
    s = 0.0
    for m in range(4):
        for n in range(4):
            s += ED[m] * ED[n] * ip_eta(X[m][n], X[m][n])
    return 0.5 * s


def half(X):
    """sum_{mu<nu} eta^mu eta^nu <X_mn, X_mn>_eta (the notebook-style sum)."""
    s = 0.0
    for m in range(4):
        for n in range(m + 1, 4):
            s += ED[m] * ED[n] * ip_eta(X[m][n], X[m][n])
    return s


def split_T(A):
    F, G = comm(A), acomm(A)
    return [[G[m][n] if (m == 0 or n == 0) else F[m][n] for n in range(4)] for m in range(4)]


def split_B(A):
    F, G = comm(A), acomm(A)
    return [[PS @ F[m][n] @ PS + E0 @ G[m][n] @ PS + PS @ G[m][n] @ E0 + E0 @ G[m][n] @ E0
             for n in range(4)] for m in range(4)]


def dens(A):
    return {"I1": full(comm(A)), "GG": full(acomm(A)), "PP": full(prod(A)),
            "T": full(split_T(A)), "B": full(split_B(A))}


def rand_sym(rng, dist="normal"):
    X = rng.normal(size=(4, 4)) if dist == "normal" else rng.uniform(-1, 1, size=(4, 4))
    return X + X.T


def rand_jet(rng, dist="normal", time_row_scale=1.0, static=False):
    A = [rand_sym(rng, dist) for _ in range(4)]
    for m in range(4):
        A[m] = A[m].copy()
        A[m][0, 1:] *= time_row_scale; A[m][1:, 0] *= time_row_scale
    if static:
        A[0] = np.zeros((4, 4))
    return A


# ---------------- Lorentz transforms (own) ----------------
def lorentz_boost(vvec):
    v = np.asarray(vvec, float); b = np.linalg.norm(v); g = 1.0 / np.sqrt(1.0 - b * b)
    n = v / b
    L = np.eye(4); L[0, 0] = g; L[0, 1:] = -g * v; L[1:, 0] = -g * v
    L[1:, 1:] = np.eye(3) + (g - 1.0) * np.outer(n, n)
    return L


def lorentz_rot(axis, th):
    n = np.asarray(axis, float); n /= np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    R3 = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    L = np.eye(4); L[1:, 1:] = R3
    return L


def transform(L, A):
    """M -> L M L^T, x -> L x, so A'_mu = sum_nu (L^-1)[nu, mu] L A_nu L^T."""
    Li = np.linalg.inv(L)
    return [sum(Li[n, m] * (L @ A[n] @ L.T) for n in range(4)) for m in range(4)]


def rel(a, b):
    return float(abs(a - b) / max(abs(b), 1e-300))


def main():
    t0 = time.time()
    rng = np.random.default_rng(20260910)
    out = {}
    N = 600
    jets = [rand_jet(rng) for _ in range(N)]

    # ---------------- C1 ----------------
    c1 = {"F_asym_munu": 0.0, "F_asym_ab": 0.0, "G_sym_munu": 0.0, "G_sym_ab": 0.0,
          "G_diag_max": 0.0, "Q_index_placement_asym_rel": 0.0, "Q_asym_zero_count": 0}
    for A in jets[:100]:
        F, G = comm(A), acomm(A)
        for m in range(4):
            for n in range(4):
                c1["F_asym_munu"] = max(c1["F_asym_munu"], np.abs(F[m][n] + F[n][m]).max())
                c1["F_asym_ab"] = max(c1["F_asym_ab"], np.abs(F[m][n] + F[m][n].T).max())
                c1["G_sym_munu"] = max(c1["G_sym_munu"], np.abs(G[m][n] - G[n][m]).max())
                c1["G_sym_ab"] = max(c1["G_sym_ab"], np.abs(G[m][n] - G[m][n].T).max())
                if m == n:
                    c1["G_diag_max"] = max(c1["G_diag_max"], np.abs(G[m][m]).max())
                Q = (A[m] @ ETA) @ (A[n] @ ETA) + (A[n] @ ETA) @ (A[m] @ ETA)
                r = np.abs(Q - Q.T).max() / np.abs(Q).max()
                c1["Q_index_placement_asym_rel"] = max(c1["Q_index_placement_asym_rel"], r)
                # (Aeta)(Beta)+(Beta)(Aeta) = (A eta B + B eta A) eta = G eta: symmetric iff G commutes with eta
                c1["Q_asym_zero_count"] += int(r < 1e-12)
    c1["Q_equals_G_eta"] = float(max(np.abs(((A[1] @ ETA) @ (A[2] @ ETA) + (A[2] @ ETA) @ (A[1] @ ETA)) - acomm(A)[1][2] @ ETA).max() for A in jets[:20]))
    out["C1"] = c1

    # ---------------- C2 ----------------
    LAG = load_registry()
    Astack = np.stack([np.stack([A[m] for A in jets]) for m in range(4)])   # (4, N, 4, 4)
    Freg = LAG.F_of_A(Astack)
    reg = {k: LAG.density_from_K(Freg, LAG.REGISTRY[k]._K()) for k in ["I1", "I2", "I3", "I4", "I5", "I6"]}
    D = [dens(A) for A in jets]
    I1 = np.array([d["I1"] for d in D]); GG = np.array([d["GG"] for d in D]); PP = np.array([d["PP"] for d in D])
    TT = np.array([d["T"] for d in D]); BB = np.array([d["B"] for d in D])
    ent = 0.0
    for A in jets[:50]:
        P, F, G = prod(A), comm(A), acomm(A)
        for m in range(4):
            for n in range(4):
                ent = max(ent, np.abs(G[m][n] ** 2 + F[m][n] ** 2 - 2 * (P[m][n] ** 2 + P[n][m] ** 2)).max())
    # half-sum (mu<nu) of GG under a boost
    Lb = lorentz_boost([0.2, -0.15, 0.1]); Lr = lorentz_rot([1, 2, -1], 0.9)
    hb = [rel(half(acomm(transform(Lb, A))), half(acomm(A))) for A in jets[:50]]
    hf = [rel(full(acomm(transform(Lb, A))), full(acomm(A))) for A in jets[:50]]
    hr = [rel(half(acomm(transform(Lr, A))), half(acomm(A))) for A in jets[:50]]
    hI = [rel(half(comm(transform(Lb, A))), half(comm(A))) for A in jets[:50]]
    out["C2"] = {"registry_I1_vs_own_full_I1_rel": float(np.abs(reg["I1"] - I1).max() / np.abs(I1).max()),
                 "entrywise_G2+F2-2(P2+PT2)_max": float(ent),
                 "GG+I1-4PP_rel": float(np.abs(GG + I1 - 4 * PP).max() / np.abs(4 * PP).max()),
                 "halfsum_GG_boost_rel_max": float(max(hb)), "halfsum_GG_boost_rel_median": float(np.median(hb)),
                 "fullsum_GG_boost_rel_max": float(max(hf)),
                 "halfsum_GG_rotation_rel_max": float(max(hr)),
                 "halfsum_I1_boost_rel_max(control)": float(max(hI))}

    # ---------------- C3 ----------------
    def tr2(A, m, n):
        return np.trace(A[m] @ ETA @ A[n] @ ETA)
    C6a = np.array([sum(ED[m] * tr2(A, m, m) for m in range(4)) ** 2 for A in jets])
    C6b = np.array([sum(ED[m] * ED[n] * tr2(A, m, n) ** 2 for m in range(4) for n in range(4)) for A in jets])
    J = np.array([sum(ED[m] * ED[n] * np.trace(ETA @ A[m] @ ETA @ A[n] @ ETA @ A[n] @ ETA @ A[m])
                      for m in range(4) for n in range(4)) for A in jets])
    K = np.array([sum(ED[m] * ED[n] * np.trace((ETA @ A[m] @ ETA @ A[n]) @ (ETA @ A[m] @ ETA @ A[n]))
                      for m in range(4) for n in range(4)) for A in jets])
    names = ["I1", "I2", "I3", "I4", "I5", "I6", "C6a", "C6b"]
    cols = [reg[k] for k in names[:6]] + [C6a, C6b]

    def normed(cs):
        X = np.stack(cs, 1); return X / np.linalg.norm(X, axis=0)

    def svals(cs):
        return [float(s) for s in np.linalg.svd(normed(cs), compute_uv=False)]

    def resid(t, cs):
        X = np.stack(cs, 1); c = np.linalg.lstsq(X, t, rcond=None)[0]
        return float(np.linalg.norm(t - X @ c) / np.linalg.norm(t)), [float(x) for x in c]
    rGG, cGG = resid(GG, cols)
    rJ, _ = resid(J, cols)
    rK, _ = resid(K, cols)
    rJ9, _ = resid(J, cols + [GG])
    rPP, _ = resid(PP, cols)
    # cross-validation: fit on first half, test on second half
    Xh = np.stack(cols, 1); c_half = np.linalg.lstsq(Xh[:N // 2], GG[:N // 2], rcond=None)[0]
    r_test = float(np.linalg.norm(GG[N // 2:] - Xh[N // 2:] @ c_half) / np.linalg.norm(GG[N // 2:]))
    out["C3"] = {"sv_basis8": svals(cols), "sv_basis8+GG": svals(cols + [GG]),
                 "sv_basis8+J": svals(cols + [J]), "sv_basis8+GG+J": svals(cols + [GG, J]),
                 "GG_resid": rGG, "GG_coef": dict(zip(names, cGG)), "GG_resid_holdout": r_test,
                 "J_resid": rJ, "K_resid": rK, "J_resid_with_GG": rJ9, "PP_resid": rPP,
                 "GG-(2J-I1)_rel": float(np.abs(GG - (2 * J - I1)).max() / np.abs(GG).max()),
                 "I1-(J-K)_rel": float(np.abs(I1 - (J - K)).max() / np.abs(I1).max())}

    # ---------------- C4 ----------------
    Ls = {"rot_axis(1,2,-1)_0.9": Lr, "boost_v(0.2,-0.15,0.1)": Lb,
          "boost_x_0.3": lorentz_boost([0.3, 0, 0]),
          "composite_R.B.R": lorentz_rot([0, 1, 1], 0.4) @ lorentz_boost([0, 0.5, 0]) @ lorentz_rot([1, 0, 0], 1.1)}
    c4 = {"eta_preserved_max": float(max(np.abs(L.T @ ETA @ L - ETA).max() for L in Ls.values()))}
    for nm, L in Ls.items():
        r = {"I1": 0.0, "GG": 0.0, "T": 0.0, "B": 0.0}
        for A in jets[:60]:
            d0, d1 = dens(A), dens(transform(L, A))
            for k in r:
                r[k] = max(r[k], rel(d1[k], d0[k]))
        c4[nm] = r
    out["C4"] = c4

    # ---------------- C5, C6 ----------------
    sjets = [rand_jet(rng, static=True) for _ in range(N)]
    c5 = {"G_0mu_max": 0.0, "T_minus_I1_max": 0.0, "I1_scale": 0.0}
    for A in sjets:
        G = acomm(A); d = dens(A)
        c5["G_0mu_max"] = max(c5["G_0mu_max"], max(np.abs(G[0][n]).max() for n in range(4)))
        c5["T_minus_I1_max"] = max(c5["T_minus_I1_max"], abs(d["T"] - d["I1"]))
        c5["I1_scale"] = max(c5["I1_scale"], abs(d["I1"]))
    out["C5"] = c5
    cjets = []
    for A in sjets:
        Ac = [X.copy() for X in A]
        for X in Ac:
            X[0, :] = 0.0; X[:, 0] = 0.0
        cjets.append(Ac)
    c6 = {"B_minus_I1_max": 0.0, "I1_scale": 0.0, "GG_vs_I1_rel_max": 0.0}
    for A in cjets:
        d = dens(A)
        c6["B_minus_I1_max"] = max(c6["B_minus_I1_max"], abs(d["B"] - d["I1"]))
        c6["I1_scale"] = max(c6["I1_scale"], abs(d["I1"]))
        c6["GG_vs_I1_rel_max"] = max(c6["GG_vs_I1_rel_max"], rel(d["GG"], d["I1"]))
    out["C6"] = c6

    # ---------------- C7 ----------------
    def blocks(A):
        P, F = prod(A), comm(A)
        Ft = [[E0 @ F[m][n] @ PS + PS @ F[m][n] @ E0 for n in range(4)] for m in range(4)]
        Pt = [[E0 @ P[m][n] @ PS + PS @ P[m][n] @ E0 + E0 @ P[m][n] @ E0 for n in range(4)] for m in range(4)]
        return Ft, Pt

    def c7_stats(js):
        idr, ft, fr = 0.0, [], {"I1": 0, "B": 0, "GG": 0, "T": 0}
        for A in js:
            d = dens(A); Ft, Pt = blocks(A)
            idr = max(idr, rel(d["B"], d["I1"] - 2 * full(Ft) + 4 * full(Pt)))
            ft.append(full(Ft))
            for k in fr:
                fr[k] += int(d[k] < 0)
        n = len(js)
        return {"identity_rel_max": idr, "I(Ft)_max": float(max(ft)), "I(Ft)_min": float(min(ft)),
                "frac_neg": {k: v / n for k, v in fr.items()}}
    c7 = {"normal_static(producer-like)": c7_stats(sjets)}
    for nm, kw in [("uniform_static", dict(dist="uniform", static=True)),
                   ("normal_static_timerow_x0.3", dict(time_row_scale=0.3, static=True)),
                   ("normal_static_timerow_x3", dict(time_row_scale=3.0, static=True)),
                   ("normal_nonstatic", dict())]:
        c7[nm] = c7_stats([rand_jet(rng, **kw) for _ in range(400)])
    # structural sign of I(Ft): <X,X>_eta on a time-row block = -2 sum_k X_0k^2
    A = sjets[0]; Ft, _ = blocks(A)
    c7["I(Ft)_equals_-sum_ij_sum_k_Ft_ij[0,k]^2_rel"] = rel(
        full(Ft), -sum(sum(Ft[i][j][0, k] ** 2 for k in range(1, 4)) for i in range(1, 4) for j in range(1, 4)))

    # most negative normalized direction (inf of density / |A|^4 on the unit sphere), static jets
    def unpack(x):
        A = [np.zeros((4, 4))]
        for i in range(3):
            S = np.zeros((4, 4)); iu = np.triu_indices(4)
            S[iu] = x[10 * i:10 * (i + 1)]; S = S + S.T - np.diag(np.diag(S))
            A.append(S)
        return A

    def normalized(x, key):
        A = unpack(x); n4 = sum(np.sum(X * X) for X in A) ** 2
        return dens(A)[key] / n4
    infs = {}
    for key in ["I1", "B", "GG", "T"]:
        xs = rng.normal(size=(3000, 30)); vals = np.array([normalized(x, key) for x in xs])
        best = [xs[i] for i in np.argsort(vals)[:4]]
        mins = [minimize(lambda x: normalized(x, key), b, method="Nelder-Mead",
                         options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10}).fun for b in best]
        infs[key] = {"sample_min": float(vals.min()), "sample_max": float(vals.max()),
                     "refined_min": float(min(mins))}
    c7["normalized_density_extrema_static"] = infs
    out["C7"] = c7

    # ---------------- C8 ----------------
    ws = np.array([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
    V = np.vander(ws, 7, increasing=True)          # fit degree 6, expect c5 = c6 = 0
    c8 = {k: {f"c{i}_max": 0.0 for i in range(7)} for k in ["I1", "GG", "T", "B", "GG_halfsum"]}
    c4_check = {"GG": 0.0, "T": 0.0, "B": 0.0}
    for A in jets[:80]:
        a0 = rand_sym(rng)
        rows = {k: [] for k in c8}
        for w in ws:
            Aw = [w * a0] + [A[m] for m in range(1, 4)]
            d = dens(Aw)
            for k in ["I1", "GG", "T", "B"]:
                rows[k].append(d[k])
            rows["GG_halfsum"].append(half(acomm(Aw)))
        for k in c8:
            coef = np.linalg.lstsq(V, np.array(rows[k]), rcond=None)[0]
            for i in range(7):
                c8[k][f"c{i}_max"] = max(c8[k][f"c{i}_max"], abs(float(coef[i])))
            if k in c4_check:
                # symbolic omega^4 coefficient: only the (0,0) block, X = a0 eta a0 (or its time block for B)
                X = a0 @ ETA @ a0
                if k == "B":
                    X = E0 @ X @ PS + PS @ X @ E0 + E0 @ X @ E0
                c4_check[k] = max(c4_check[k], rel(coef[4], 2.0 * ip_eta(X, X)))
    c8["c4_vs_symbolic_2<X,X>_rel_max"] = c4_check
    out["C8"] = c8

    # ---------------- Q1 ----------------
    q1 = {"F_shift_by_c_eta_rel": 0.0, "G_shift_by_c_eta_rel": 0.0, "F_pure_eta_jet_max": 0.0,
          "G_pure_eta_jet_max": 0.0, "F_traceless_jets_max": 0.0}
    for A in jets[:40]:
        c = rng.normal(size=4)
        As = [A[m] + c[m] * ETA for m in range(4)]
        F0, G0, F1, G1 = comm(A), acomm(A), comm(As), acomm(As)
        for m in range(4):
            for n in range(4):
                q1["F_shift_by_c_eta_rel"] = max(q1["F_shift_by_c_eta_rel"], np.abs(F1[m][n] - F0[m][n]).max() / max(np.abs(F0[m][n]).max(), 1e-300))
                q1["G_shift_by_c_eta_rel"] = max(q1["G_shift_by_c_eta_rel"], np.abs(G1[m][n] - G0[m][n]).max() / max(np.abs(G0[m][n]).max(), 1e-300))
        Ae = [c[m] * ETA for m in range(4)]
        q1["F_pure_eta_jet_max"] = max(q1["F_pure_eta_jet_max"], max(np.abs(comm(Ae)[m][n]).max() for m in range(4) for n in range(4)))
        q1["G_pure_eta_jet_max"] = max(q1["G_pure_eta_jet_max"], max(np.abs(acomm(Ae)[m][n]).max() for m in range(4) for n in range(4)))
        At = [A[m] - np.trace(A[m] @ ETA) / 4.0 * ETA for m in range(4)]   # eta-traceless: tr(A eta) = 0
        assert all(abs(np.trace(X @ ETA)) < 1e-10 for X in At)
        q1["F_traceless_jets_max"] = max(q1["F_traceless_jets_max"], max(np.abs(comm(At)[m][n]).max() for m in range(4) for n in range(4)))
    out["Q1"] = q1

    # ---------------- Q2: equivariant antisymmetric bilinears Sym x Sym -> Sym ----------------
    iu = np.triu_indices(4)
    def to_sym(v):
        S = np.zeros((4, 4)); S[iu] = v; return S + S.T - np.diag(np.diag(S))
    basis = [to_sym(np.eye(10)[i]) for i in range(10)]
    # dual: coordinates of a symmetric matrix S in the basis = S[iu]
    gens = []
    for a in range(4):
        for b in range(a + 1, 4):
            w = np.zeros((4, 4)); w[a, b] = 1.0; w[b, a] = -1.0
            gens.append(w @ ETA)                     # omega with omega^T eta + eta omega = 0
    assert all(np.abs(g.T @ ETA + ETA @ g).max() < 1e-12 for g in gens)
    # unknown T[i, j, :] (10 x 10 x 10): T(e_i, e_j) = sum_k T[i,j,k] e_k, with antisymmetry T[i,j] = -T[j,i]
    # equivariance: T(dX, Y) + T(X, dY) = d T(X, Y), dX = w X + X w^T
    rows = []
    idx = lambda i, j, k: (i * 10 + j) * 10 + k
    for w in gens:
        for i in range(10):
            dXi = w @ basis[i] + basis[i] @ w.T
            ci = dXi[iu]
            for j in range(10):
                dXj = w @ basis[j] + basis[j] @ w.T
                cj = dXj[iu]
                for k in range(10):
                    row = np.zeros(1000)
                    # LHS: sum_p ci[p] T[p,j,k] + sum_q cj[q] T[i,q,k]
                    for p in range(10):
                        row[idx(p, j, k)] += ci[p]
                        row[idx(i, p, k)] += cj[p]
                    # RHS: (d e_l)[k] T[i,j,l]  with d e_l = w e_l + e_l w^T
                    for l in range(10):
                        dl = (w @ basis[l] + basis[l] @ w.T)[iu]
                        row[idx(i, j, l)] -= dl[k]
                    rows.append(row)
    for i in range(10):
        for j in range(10):
            for k in range(10):
                row = np.zeros(1000); row[idx(i, j, k)] += 1; row[idx(j, i, k)] += 1; rows.append(row)
    Mq = np.stack(rows)
    s = np.linalg.svd(Mq, compute_uv=False)
    null_dim = int(np.sum(s < 1e-9 * s[0]))
    out["Q2"] = {"dim_equivariant_antisym_bilinear_Sym2xSym2->Sym2": null_dim,
                 "smallest_svals": [float(x) for x in s[-4:]]}
    # identify: H = tr(X eta) Y - tr(Y eta) X and the traceless-part antisymmetrizer? check H is in the nullspace
    Th = np.zeros(1000)
    for i in range(10):
        for j in range(10):
            H = np.trace(basis[i] @ ETA) * basis[j] - np.trace(basis[j] @ ETA) * basis[i]
            Th[idx(i, j, 0):idx(i, j, 0) + 10] = H[iu]
    out["Q2"]["H_trace_bilinear_in_nullspace_resid"] = float(np.linalg.norm(Mq @ Th) / np.linalg.norm(Th))
    # the commutator F eta^{-1}? F = X eta Y - Y eta X is antisymmetric, NOT in Sym2; the eta-symmetrized
    # candidate S = X eta Y + Y eta X is symmetric but symmetric in (X,Y). Any symmetric-output one besides H?
    # nullspace vectors:
    _, _, Vt = np.linalg.svd(Mq)
    NS = Vt[-null_dim:] if null_dim else np.zeros((0, 1000))
    out["Q2"]["nullspace_H_projection_rel"] = float(np.linalg.norm(NS @ Th) / np.linalg.norm(Th)) if null_dim else 0.0

    out["runtime_s"] = time.time() - t0
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
