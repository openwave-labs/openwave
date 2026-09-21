"""Items 5 and 5b: Hessian of r-hat_6 on N_u at every orbit of item 4.

Exact route: M_u = Hess N(u) - 4 N(u) I (exact sympy), N_u basis by exact nullspace,
characteristic polynomial det(lam G - H)/det G, signature by Sturm counting.
Numerical route: the same restriction orthonormalised and diagonalised with mpmath
at 50 digits (independent of the characteristic-polynomial route).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
import mpmath
import core, hess
from core import report, IDX, MS
from hess import simp, LAM

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True
mpmath.mp.dps = 50


def vec(d):
    c = [sp.Integer(0)] * 7
    for m, val in d.items():
        c[IDX[m]] += sp.sympify(val)
    return c


R = sp.Rational
ORBITS = {   # name: (representative, class it is interior to (5b) or None, chart data (w0, w1, z0))
    "v3": vec({3: 1}),
    "v2": vec({2: 1}),
    "Wmin": vec({0: 1, 3: sp.sqrt(R(10, 23)), -3: sp.sqrt(R(10, 23))}),
    "v1": vec({1: 1}),
    "Umin": vec({0: 1, 2: sp.I * sp.sqrt(R(5, 6)), -2: sp.I * sp.sqrt(R(5, 6))}),
    "L12circle": vec({1: 1, -2: R(1, 2)}),
    "L23circle": vec({2: 1, -3: sp.sqrt(R(12, 13))}),
    "oct": vec({2: 1, -2: -1}),
    "v0": vec({0: 1}),
    "hex": vec({3: 1, -3: 1}),
}
out = {}
for name, c in ORBITS.items():
    cu, r = hess.unit_real(c)
    Mu = hess.M_u(r)
    og = hess.O_gens(cu)
    Om = sp.Matrix.hstack(*og.values())
    dimO = Om.rank(simplify=True)
    B = hess.N_basis(r, og)
    dimN = len(B)
    ok &= report("%s: dim O_u + dim N_u = 13" % name, dimO + dimN == 13, "dim O_u = %d, dim N_u = %d" % (dimO, dimN))
    # O_u annihilation: M_u d for each generator d, exact
    res = {}
    for k, d in og.items():
        v = (Mu * d).applyfunc(simp)
        res[k] = sp.sqrt(simp(sum(t ** 2 for t in v)))
    ok &= report("%s: M_u annihilates the four generators of O_u exactly" % name,
                 all(v == 0 for v in res.values()), "residual norms %s" % [str(v) for v in res.values()])
    # restriction
    G, H, cp = hess.restricted(Mu, B)
    nneg, n0, npos = hess.signature_from_charpoly(cp, dimN)
    # numerical route
    Gm = mpmath.matrix([[mpmath.mpf(sp.N(G[i, j], 60)) for j in range(dimN)] for i in range(dimN)])
    Hm = mpmath.matrix([[mpmath.mpf(sp.N(H[i, j], 60)) for j in range(dimN)] for i in range(dimN)])
    ev, Q = mpmath.eigsy(Gm)
    Gih = Q * mpmath.diag([1 / mpmath.sqrt(e) for e in ev]) * Q.T
    E, _ = mpmath.eigsy(Gih * Hm * Gih)
    E = sorted([E[i] for i in range(dimN)])
    exact_roots = sp.Poly(cp, LAM).all_roots()          # exact algebraic roots, with multiplicity
    roots = sorted([mpmath.mpf(sp.N(z, 60)) for z in exact_roots])
    worst = max(abs(a - b) for a, b in zip(E, roots))
    ok &= report("%s: eigenvalues (mpmath eigsy, 50 digits) match exact roots of the charpoly" % name,
                 worst < mpmath.mpf(10) ** -30, "max diff %s" % mpmath.nstr(worst, 3))
    tol = mpmath.mpf(10) ** -40
    sig_num = (sum(1 for e in E if e < -tol), sum(1 for e in E if abs(e) <= tol), sum(1 for e in E if e > tol))
    ok &= report("%s: exact signature equals numerical signature" % name, sig_num == (nneg, n0, npos),
                 "exact %s, numerical %s" % ((nneg, n0, npos), sig_num))
    val = core.rhat6(c)
    print("  %s: r6 = %s  dim N_u = %d  signature (n-, n0, n+) = %s" % (name, val, dimN, (nneg, n0, npos)))
    print("     charpoly:", cp)
    print("     eigenvalues:", [mpmath.nstr(e, 12) for e in E])
    out[name] = {"value": str(val), "dim_O": int(dimO), "dim_N": int(dimN), "signature": [int(nneg), int(n0), int(npos)],
                 "charpoly": str(cp), "annihilation_residuals": {k: str(v) for k, v in res.items()},
                 "eigenvalues_50digits": [mpmath.nstr(e, 30) for e in E],
                 "eig_vs_charpoly_maxdiff": mpmath.nstr(worst, 3)}

json.dump(out, open(os.path.join(HERE, "out", "item5.json"), "w"), indent=1)
print("ITEM5", "ALLPASS" if ok else "SOMEFAIL")
