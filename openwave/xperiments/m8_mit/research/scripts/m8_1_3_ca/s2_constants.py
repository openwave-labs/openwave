"""Slot constants for h and for the level decomposition of N(psi) = |psi|^2 psi; items C.1, C.2.

Derivation used (checked numerically below against direct cubature on SU(2)):
  psi_w(g) = s^{-1/2} w^T D(g) eta,  conj(w^T D eta_b) = (Theta w)^T D (U conj eta_b),
  (x^T D y)(x'^T D y') = sum_J (W_J(x(x)x'))^T D^J (W_J(y(x)y')),   W_J = CG matrix,
  so |psi_w|^2 = s^{-1} sum_J [w (x) Theta w]_J^T D^J b_J,  b_J = W_J z,  z = vec(P U^T),  P = eta eta^dagger,
  h(w) = int |psi_w|^4 = sum_J c_J ||[w (x) Theta w]_J||^2,   c_J = |b_J|^2 / ((2J+1) s^2),
  Pi_K N(psi_w) = s^{-3/2} sum_J (W_{K;J,j}([w(x)Theta w]_J (x) w))^T D^K (W_{K;J,j}(b_J (x) eta_a)),
  ||Pi_K N||^2 = sum_{J,J'} conj(G^K_{J'J}) <alpha_{J'}, alpha_J> / (2K+1),  G^K_{J'J} = s^{-3} tr(W_{KJ'}^T W_{KJ} (b_J b_J'^dagger (x) P)).
Measure: normalized Haar on SU(2) = the measure on X rescaled to volume 1 (V = 1), R = 1.
"""
import os, sys, json, pickle, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from common import *
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
mp.mp.dps = 60
slots = pickle.load(open(os.path.join(OUT, "slots.pkl"), "rb"))
grp = json.load(open(os.path.join(OUT, "group.json")))
irr_of = grp["slot_irreducible"]

def sym_simplify(x):
    return sp.nsimplify(0) if x == 0 else sp.radsimp(sp.expand(x))

res = {"C.1": {}, "C.2": {}, "_constants": {}}
consts = {}
for s, d in slots.items():
    j = sp.Rational(d["j"]); n = d["n"]; sN = sp.Rational(d["sN"])
    P = sp.Matrix(n, n, lambda a, b: sp.sympify(d["Pon"][a][b]))
    U = theta_U(j)
    Z = P * U.T
    z = sp.Matrix([Z[a, b] for a in range(n) for b in range(n)])
    print(f"===== slot {s}: j = {j}, s_N = {sN}")
    b = {}; cJ = {}
    for J in range(0, int(2 * j) + 1):
        WJ = W_matrix(J, j, j)
        bJ = (WJ * z).applyfunc(sym_simplify)
        nb = sym_simplify(sum((x * sp.conjugate(x) for x in bJ), sp.Integer(0)))
        b[J] = bJ
        cJ[J] = sym_simplify(nb / ((2 * J + 1) * sN ** 2))
    nonzero = [J for J in cJ if cJ[J] != 0]
    print("  c_J (h = sum_J c_J ||[w (x) Theta w]_J||^2):", {J: str(cJ[J]) for J in nonzero})
    check(f"{s}: c_J nonzero only for J in {{0, 6}} (Gamma_0-invariants of V_J, J <= 2j)", set(nonzero) == {0, 6}, str(nonzero))
    check(f"{s}: c_0 ||[w(x)Theta w]_0||^2 = ||w||^4, i.e. c_0 = 2j+1 (mean of |psi|^2 is ||w||^2)", cJ[0] == 2 * j + 1, str(cJ[0]))
    check(f"{s}: c_6 is rational", cJ[6].is_Rational, str(cJ[6]))
    # constants g_K for J = J' = 6, all K
    b6 = b[6]
    gK = {}
    K = abs(6 - j)
    while K <= 6 + j:
        WK = W_matrix(K, 6, j)
        # tr(W^T W (b b^dagger (x) P)) = sum_M sum_{r,t} W[M,r] X[r,t] W[M,t]
        tot = sp.Integer(0)
        for M in range(WK.rows):
            nz = [(r, WK[M, r]) for r in range(WK.cols) if WK[M, r] != 0]
            for r, wr in nz:
                m1, m2 = divmod(r, n)
                for t, wt in nz:
                    n1, n2 = divmod(t, n)
                    tot += wr * wt * b6[m1] * sp.conjugate(b6[n1]) * P[m2, n2]
        g = sym_simplify(tot / ((2 * K + 1) * sN ** 3))
        gK[K] = g
        K += 1
    print("  g_K (||Pi_K N||^2 = g_K ||[[w(x)Theta w]_6 (x) w]_K||^2 for K != j):", {str(k): str(v) for k, v in gK.items()})
    check(f"{s}: every g_K is rational", all(v.is_Rational for v in gK.values()))
    lv = grp["levels"][irr_of[s]]
    occurs = {int(l) for l in lv}
    zero_levels = {int(2 * k) for k, v in gK.items() if v == 0}
    nz_levels = {int(2 * k) for k, v in gK.items() if v != 0}
    check(f"{s}: g_K = 0 at every level where the slot does not occur", all(L in occurs for L in nz_levels),
          f"nonzero at {sorted(nz_levels)}, slot occurs at {sorted(occurs)}")
    consts[s] = dict(j=str(j), sN=str(sN), cJ={J: str(cJ[J]) for J in nonzero}, gK={str(k): str(v) for k, v in gK.items()},
                     b={J: [str(x) for x in b[J]] for J in nonzero})
    nlow = int(2 * j)
    above = sorted(L for L in occurs if L > nlow)
    L1 = above[0]
    res["C.2"][s] = {"lowest level": nlow, "next level": L1, "lambda_1 - lambda_0 (units 1/R^2)": str(L1 * (L1 + 2) - nlow * (nlow + 2))}
    print(f"  C.2 next level {L1}, lambda_1 - lambda_0 = {L1*(L1+2) - nlow*(nlow+2)}")

pickle.dump(consts, open(os.path.join(OUT, "consts.pkl"), "wb"))

# ---------------------------------------------------------------- level norms at a generic y (C.1), exact
def tensor_couple(J, j1, j2, x1, x2):
    """[x1 (x) x2]_J components (sympy), rows M = J..-J."""
    W = W_matrix(J, j1, j2)
    v = sp.Matrix([x1[a] * x2[c] for a in range(len(x1)) for c in range(len(x2))])
    return W * v

def theta(j, w):
    U = theta_U(j)
    return U * sp.Matrix([sp.conjugate(x) for x in w])

def level_norms_exact(s, y):
    """||Pi_K N(psi_y)||^2 for K != j, exact, y a sympy vector (any norm; result for y/||y||)."""
    c = consts[s]; j = sp.Rational(c["j"])
    ny2 = sp.expand(sum((x * sp.conjugate(x) for x in y), sp.Integer(0)))
    B6 = tensor_couple(6, j, j, y, theta(j, y))
    out = {}
    for kk, g in c["gK"].items():
        K = sp.Rational(kk); g = sp.Rational(g)
        if K == j:
            continue
        A = tensor_couple(K, 6, j, B6, y)
        val = sym_simplify(g * sum((x * sp.conjugate(x) for x in A), sp.Integer(0)) / ny2 ** 3)
        out[int(2 * K)] = val
    return out

# generic point: no symmetry (all components nonzero, distinct moduli and phases)
def generic_y(n):
    return sp.Matrix([sp.Integer(k + 1) + sp.I * sp.Integer((k * k) % 7 - 3) for k in range(n)])

for s in slots:
    j = sp.Rational(consts[s]["j"]); n = int(2 * j + 1)
    y = generic_y(n)
    ln = level_norms_exact(s, y)
    nlow = int(2 * j)
    occurs = sorted(int(l) for l in grp["levels"][irr_of[s]])
    print(f"C.1 {s}: y = {list(y)} / ||y||,  ||y||^2 = {sum(abs(x)**2 for x in y)}")
    rows = {}
    for L in occurs:
        if nlow < L <= 20:
            v = ln.get(L, sp.Integer(0))
            rows[str(L)] = str(v)
            print(f"   level {L:2d}: ||Pi_L N||^2 = {v} = {sp.N(v, 30)}")
    nzl = [int(L) for L, v in rows.items() if sp.sympify(v) != 0]
    print("   levels with nonzero norm:", nzl)
    res["C.1"][s] = {"s_N": consts[s]["sN"], "y (unnormalized; y/||y|| used)": [str(x) for x in y],
                     "level norms": rows, "levels where nonzero": nzl}

# ---------------------------------------------------------------- independent numerical check by cubature (double precision)
def D_np(j, a, b, c, d):
    from math import comb, sqrt
    M = D_generic(j, a, b, c, d, lambda nn, k: comb(nn, k), lambda p, q: sqrt(int(p) / int(q)), 0 * a, 1 + 0 * a)
    return M

def cubature_check(s, y):
    dd = slots[s]; j = sp.Rational(dd["j"]); n = dd["n"]; sN = float(sp.Rational(dd["sN"]))
    eta = np.array([[complex(float(mp.mpf(re)), float(mp.mpf(im))) for re, im in row] for row in dd["eta"]])
    yv = np.array([complex(sp.N(x)) for x in y]); yv = yv / np.linalg.norm(yv)
    deg_t = int(2 * j) * 3 // 1
    Nxi = 2 * int(6 * j) + 4; Nt = int(6 * j) + 2
    xg, wg = np.polynomial.legendre.leggauss(Nt)
    t = (xg + 1) / 2; wt = wg / 2
    T, X1, X2 = np.meshgrid(t, 2 * np.pi * np.arange(Nxi) / Nxi, 2 * np.pi * np.arange(Nxi) / Nxi, indexing="ij")
    W = np.broadcast_to(wt[:, None, None], T.shape) / Nxi ** 2
    a = np.sqrt(1 - T) * np.exp(1j * X1); bq = np.sqrt(T) * np.exp(1j * X2)
    ga, gb, gc, gd = a, bq, -np.conj(bq), np.conj(a)
    Dj = D_np(j, ga, gb, gc, gd)
    # psi_a(g) = s^{-1/2} sum_{m,m'} y_m D_{mm'} eta_{m'a}
    rowv = sum(yv[m] * np.array(Dj[m]) for m in range(n))        # shape (n, grid): (y^T D)_{m'}
    psi = np.einsum("mxyz,ma->axyz", rowv, eta) / np.sqrt(sN)
    dens = np.sum(np.abs(psi) ** 2, axis=0)
    h_num = np.sum(W * dens ** 2)
    N = dens[None] * psi
    out = {}
    K = sp.Rational(1, 2) if (2 * j) % 2 else sp.Integer(0)
    while K <= 3 * j:
        DK = np.array(D_np(K, ga, gb, gc, gd))           # (2K+1, 2K+1, grid)
        tot = 0.0
        for comp in range(N.shape[0]):
            coef = np.einsum("xyz,mnxyz->mn", W * N[comp], np.conj(DK)) * (2 * K + 1)
            tot += np.sum(np.abs(coef) ** 2) / (2 * K + 1)
        out[int(2 * K)] = tot
        K += 1
    return h_num, out, float(np.sum(W * dens ** 3))

for s in slots:
    j = sp.Rational(consts[s]["j"]); n = int(2 * j + 1)
    y = generic_y(n)
    hN, lnum, six = cubature_check(s, y)
    ny = sp.sqrt(sum(abs(x) ** 2 for x in y))
    yu = y / ny
    hform = sum(sp.Rational(c) * sum(abs(x) ** 2 for x in tensor_couple(J, j, j, yu, theta(j, yu))) for J, c in consts[s]["cJ"].items())
    hform = float(sp.N(hform, 30))
    print(f"cubature {s}: h = {hN:.15g}, CG formula h = {hform:.15g}")
    check(f"{s}: h by cubature == h by the CG formula (rel 1e-12)", abs(hN - hform) < 1e-12 * abs(hform))
    lex = level_norms_exact(s, y)
    worst = 0.0
    for L, v in sorted(lnum.items()):
        if L == int(2 * j):
            continue
        e = float(sp.N(lex.get(L, 0), 30))
        worst = max(worst, abs(v - e))
        print(f"   level {L:2d}: cubature {v:.15g}   CG exact {e:.15g}")
    check(f"{s}: level norms by cubature == CG formula at every level != lowest (abs 1e-12)", worst < 1e-12, f"{worst:.2e}")
    tot = sum(lnum.values())
    check(f"{s}: cubature level norms sum to int |psi|^6 (Parseval, rel 1e-12)", abs(tot - six) < 1e-12 * six)

res["_constants"] = {s: {"c_J": c["cJ"], "g_K": c["gK"]} for s, c in consts.items()}
json.dump(res, open(os.path.join(OUT, "C12.json"), "w"), indent=1, default=str)
print("failures:", n_fails())
