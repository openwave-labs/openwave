"""Item 0(c), exact form of eta.  The eta of s02 (Gram-Schmidt of P v_m) is real to 1e-55; each
entry is identified as sign*sqrt(p/q), q <= 10^6, at 50 digits and again at 30 digits, and the
identified matrix is then verified EXACTLY (sympy) against the exact projector of s02:
eta^T eta = I and eta eta^T = P (orthonormal basis), which makes eta an orthonormal basis of the
rho-isotypic subspace, i.e. an intertwiner with eta^dag eta = I."""
import os, sys, json, pickle
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp
import sympy as sp
from lib_check import check, summary
from lib_cg import mrange, w as wfac

mp.mp.dps = 60
eta_all = json.load(open(os.path.join(HERE, "out", "eta.json")))
res = {}


def ident(x, dps):
    """sign*sqrt(p/q) with q <= 10^6 matching x^2 to 10^-(dps-8)."""
    with mp.workdps(dps):
        x = mp.mpf(x)
        if abs(x) < mp.mpf(10) ** (-dps + 10):
            return Fr(0), 0
        y = x * x
        f = Fr(str(mp.nstr(y, dps))).limit_denominator(10 ** 6)
        if abs(mp.mpf(f.numerator) / f.denominator - y) > mp.mpf(10) ** (-dps + 8):
            return None, None
        return f, (1 if x > 0 else -1)


for slot, rec in eta_all.items():
    j = Fr(rec["j"])
    d = rec["dim"]
    ms = mrange(j)
    n = len(ms)
    imag = max(abs(mp.mpf(e[1])) for row in rec["eta"] for e in row)
    check(f"{slot}: eta real (max |Im| < 1e-50)", imag < mp.mpf(10) ** -50)
    E = sp.zeros(n, d)
    ok = True
    for r in range(n):
        for a in range(d):
            x = rec["eta"][r][a][0]
            f1, s1 = ident(x, 50)
            f2, s2 = ident(mp.nstr(mp.mpf(x), 30), 30)
            ok &= f1 is not None and f1 == f2 and s1 == s2
            E[r, a] = s1 * sp.sqrt(sp.Rational(f1.numerator, f1.denominator)) if f1 else 0
    check(f"{slot}: every eta entry identified as sign*sqrt(rational), same at 50 and 30 digits", ok)
    # exact checks against the exact projector
    Pm = pickle.load(open(os.path.join(HERE, "out", f"Pmono_{slot}.pkl"), "rb"))["Pm"]
    def toexpr(t):
        (ra, rb), (ia, ib) = t
        R = lambda f: sp.Rational(f.numerator, f.denominator)
        return R(ra) + R(rb) * sp.sqrt(5) + sp.I * (R(ia) + R(ib) * sp.sqrt(5))
    Po = sp.Matrix(n, n, lambda r, c: toexpr(Pm[r][c]) * sp.sqrt(sp.Rational(wfac(j, ms[r]), wfac(j, ms[c]))))
    I1 = (E.T * E - sp.eye(d)).applyfunc(sp.nsimplify).applyfunc(sp.simplify)
    check(f"{slot}: eta^T eta = I exactly", I1 == sp.zeros(d, d))
    I2 = (E * E.T - Po).applyfunc(sp.radsimp).applyfunc(sp.simplify)
    check(f"{slot}: eta eta^T = P exactly (P the exact isotypic projector)", I2 == sp.zeros(n, n))
    print(f"0(c) {slot}: eta exact (rows m = {', '.join(str(m) for m in ms)}; columns a = 1..{d}):")
    for r in range(n):
        print(f"   m={str(ms[r]):>5}: " + ", ".join(str(E[r, a]) for a in range(d)))
    print(f"   P (orthonormal basis) has entries in: {sorted({str(sp.nsimplify(x).func.__name__) for x in Po})}")
    res[slot] = [[str(E[r, a]) for a in range(d)] for r in range(n)]
json.dump(res, open(os.path.join(HERE, "out", "eta_exact.json"), "w"), indent=1)
summary()
