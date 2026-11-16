"""Points of inputs.json as unit mp vectors (CS components, index order m = j..-j), and
identification helpers for exact values from two precisions."""
import json, os
from fractions import Fraction as Fr
import mpmath as mp
import sympy as sp
from lib_cg import mrange

HERE = os.path.dirname(os.path.abspath(__file__))
INP = json.load(open(os.path.join(HERE, "inputs.json")))


def point(name, j):
    if name in INP["census"]["points"]:
        comp = INP["census"]["points"][name]
    else:
        comp = INP["R2 points"]["points"][name]
    ms = mrange(j)
    v = [mp.mpc(0)] * len(ms)
    for k, s in comp.items():
        e = sp.sympify(s, locals={"I": sp.I})
        v[ms.index(Fr(k))] = mp.mpc(sp.N(sp.re(e), mp.mp.dps + 10), sp.N(sp.im(e), mp.mp.dps + 10))
    nrm = mp.sqrt(mp.fsum(abs(c) ** 2 for c in v))
    return [c / nrm for c in v]


def ident_rational(x, dps_used, qmax=10 ** 15):
    """p/q with q <= qmax and |x - p/q| < 10^-(dps_used - 12); None if none."""
    x = mp.mpf(x)
    f = Fr(mp.nstr(x, dps_used)).limit_denominator(qmax)
    if abs(x - mp.mpf(f.numerator) / f.denominator) < mp.mpf(10) ** (-(dps_used - 12)):
        return f
    return None


def ident_q5(x, dps_used, maxc=10 ** 12):
    """a + b sqrt5 with rational a, b by PSLQ on (x, 1, sqrt5); returns (a, b) or None."""
    with mp.workdps(dps_used):
        rel = mp.pslq([mp.mpf(x), mp.mpf(1), mp.sqrt(5)], maxcoeff=maxc, maxsteps=10 ** 6)
    if rel is None or rel[0] == 0:
        return None
    return (Fr(-rel[1], rel[0]), Fr(-rel[2], rel[0]))
