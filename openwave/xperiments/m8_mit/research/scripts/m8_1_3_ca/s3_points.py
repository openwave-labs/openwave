"""C.4 at the census points (R4, R5) and R2's eight points: criticality (exact) and R0~(u), exact,
with a 60-digit numerical evaluation and a direct double-precision cubature from the definition."""
import os, sys, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from common import *
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
mp.mp.dps = 60
consts = pickle.load(open(os.path.join(OUT, "consts.pkl"), "rb"))
slots = pickle.load(open(os.path.join(OUT, "slots.pkl"), "rb"))
inp = json.load(open(os.path.join(HERE, "inputs.json")))

def parse_point(j, comp):
    ms = mlist(j)
    v = [sp.Integer(0)] * len(ms)
    for k, val in comp.items():
        m = sp.Rational(k)
        v[ms.index(m)] = sp.sympify(val, locals={"I": sp.I, "sqrt": sp.sqrt})
    return sp.Matrix(v)

def couple(J, j1, j2, x1, x2):
    W = W_matrix(J, j1, j2)
    return W * sp.Matrix([x1[a] * x2[c] for a in range(len(x1)) for c in range(len(x2))])

def theta(j, w):
    return theta_U(j) * sp.Matrix([sp.conjugate(x) for x in w])

def simp(x):
    return sp.radsimp(sp.expand(x))

from crit import gradF, critical_exact   # lib/crit.py

def R0_exact(s, u):
    c = consts[s]; j = sp.Rational(c["j"]); n = int(2 * j)
    lam0 = n * (n + 2)
    B6 = couple(6, j, j, u, theta(j, u))
    tot = sp.Integer(0); terms = {}
    for kk, g in c["gK"].items():
        K = sp.Rational(kk); g = sp.Rational(g)
        if K == j or g == 0:
            continue
        L = int(2 * K)
        A = couple(K, 6, j, B6, u)
        nrm = simp(g * sum((x * sp.conjugate(x) for x in A), sp.Integer(0)))
        terms[L] = nrm
        tot += nrm / (L * (L + 2) - lam0)
    return simp(-tot), terms

# --------------- direct cubature of R0~ from its definition (double precision), independent of the CG formula
def D_np(j, a, b, c, d):
    from math import comb, sqrt
    return D_generic(j, a, b, c, d, lambda nn, k: comb(nn, k), lambda p, q: sqrt(int(p) / int(q)), 0 * a, 1 + 0 * a)

_grid = {}
def R0_cubature(s, u):
    dd = slots[s]; j = sp.Rational(dd["j"]); n = dd["n"]; sN = float(sp.Rational(dd["sN"]))
    eta = np.array([[complex(float(mp.mpf(re)), float(mp.mpf(im))) for re, im in row] for row in dd["eta"]])
    if j not in _grid:
        Nxi = 2 * int(6 * j) + 4; Nt = int(6 * j) + 2
        xg, wg = np.polynomial.legendre.leggauss(Nt)
        t = (xg + 1) / 2; wt = wg / 2
        T, X1, X2 = np.meshgrid(t, 2 * np.pi * np.arange(Nxi) / Nxi, 2 * np.pi * np.arange(Nxi) / Nxi, indexing="ij")
        W = np.broadcast_to(wt[:, None, None], T.shape) / Nxi ** 2
        a = np.sqrt(1 - T) * np.exp(1j * X1); b = np.sqrt(T) * np.exp(1j * X2)
        g4 = (a, b, -np.conj(b), np.conj(a))
        DK = {}
        K = sp.Rational(1, 2) if (2 * j) % 2 else sp.Integer(0)
        while K <= 3 * j:
            DK[K] = np.array(D_np(K, *g4)); K += 1
        _grid[j] = (W, np.array(D_np(j, *g4)), DK)
    W, Dj, DK = _grid[j]
    uv = np.array([complex(sp.N(x, 20)) for x in u])
    rowv = np.einsum("m,mnxyz->nxyz", uv, Dj)
    psi = np.einsum("mxyz,ma->axyz", rowv, eta) / np.sqrt(sN)
    dens = np.sum(np.abs(psi) ** 2, axis=0)
    N = dens[None] * psi
    lam0 = int(2 * j) * (int(2 * j) + 2)
    tot = 0.0
    for K, D in DK.items():
        L = int(2 * K)
        if L <= int(2 * j):
            continue
        nrm = 0.0
        for comp in range(N.shape[0]):
            coef = np.einsum("xyz,mnxyz->mn", W * N[comp], np.conj(D)) * (2 * K + 1)
            nrm += np.sum(np.abs(coef) ** 2) / (2 * K + 1)
        tot += nrm / (L * (L + 2) - lam0)
    return -tot

res = {}
pointsets = [("R4", sp.Integer(3), inp["census"]["points"]), ("R5", sp.Integer(3), inp["census"]["points"]),
             ("R2", sp.Rational(7, 2), inp["R2 points"]["points"])]
for s, j, pts in pointsets:
    res[s] = {}
    for name, comp in pts.items():
        w = parse_point(j, comp)
        nw2 = simp(sum((x * sp.conjugate(x) for x in w), sp.Integer(0)))
        u = (w / sp.sqrt(nw2)).applyfunc(simp)
        crit, resid, Fv = critical_exact(j, u)
        entry = {"critical": bool(crit)}
        line = f"{s:2s} {name:20s} critical={crit!s:5s}"
        if not crit:
            # numerical size of the residual, to show it is not a simplification failure
            rn = max(abs(complex(sp.N(r, 30))) for r in resid)
            line += f" (max |residual| = {rn:.3e})"
            entry["max |dF/dw* - 2F u| (F = ||[u (x) Theta u]_6||^2)"] = f"{rn:.6e}"
        else:
            R0, terms = R0_exact(s, u)
            R0n = sp.N(R0, 50)
            cub = R0_cubature(s, u)
            sign = "-" if R0 < 0 else ("+" if R0 > 0 else "0")
            ok = abs(float(R0n) - cub) < 1e-12 * max(1.0, abs(cub))
            check(f"C.4 {s} {name}: exact R0~ agrees with direct cubature from the definition", ok, f"{float(R0n):.15g} vs {cub:.15g}")
            line += f" R0~ = {R0}  = {mp.nstr(mp.mpf(str(R0n)), 25)}  sign {sign}"
            entry.update({"R0~ exact (V = R = 1)": str(R0), "R0~ 40 digits": str(sp.N(R0, 40)), "sign": sign,
                          "level norms ||Pi_L N||^2": {str(L): str(v) for L, v in terms.items()}})
        print(line)
        res[s][name] = entry
json.dump(res, open(os.path.join(OUT, "C4_points.json"), "w"), indent=1)
print("failures:", n_fails())
