"""The reduced quartic h = sum_J q_J ||[w (x) w]_J||^2 as an exact polynomial with rational coefficients
in the 14 real coordinates x of worklist Section 3 (Block D chart): x = (Re p_3..Re p_-3, Im p_3..Im p_-3),
p_m = w_m / sqrt(f_m).  Metric G = diag(f_m) twice, |x|_G = ||w||."""
import sympy as sp
from sympy import QQ
from sympy.polys.rings import ring
from fractions import Fraction as Fr
from common import cg, fm

MS = list(range(3, -4, -1))
FM = [fm(6, Fr(m)) for m in MS]
GDIAG = FM + FM
R, *X = ring(",".join(f"x{a}" for a in range(14)), QQ)

def _cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])

def channel_norm2(J):
    """||[w (x) w]_J||^2 as an element of R (rational coefficients), verified."""
    P = [(X[a], X[a + 7]) for a in range(7)]
    tot = R(0)
    for M in range(-J, J + 1):
        terms = []
        for i1, m1 in enumerate(MS):
            m2 = M - m1
            if abs(m2) > 3: continue
            i2 = MS.index(m2)
            K = cg(3, m1, 3, m2, J, M) * sp.sqrt(FM[i1] * FM[i2])
            if K != 0: terms.append((K, i1, i2))
        if not terms: continue
        c0 = terms[0][0]
        acc = (R(0), R(0))
        for K, i1, i2 in terms:
            r = sp.radsimp(K / c0)
            assert r.is_Rational, r
            pp = _cmul(P[i1], P[i2])
            acc = (acc[0] + QQ(int(r.p), int(r.q)) * pp[0], acc[1] + QQ(int(r.p), int(r.q)) * pp[1])
        c02 = sp.radsimp(c0 ** 2)
        assert c02.is_Rational
        tot += QQ(int(c02.p), int(c02.q)) * (acc[0] ** 2 + acc[1] ** 2)
    return tot

_CH = {}
def channels():
    if not _CH:
        for J in (6, 4, 2, 0):
            _CH[J] = channel_norm2(J)
    return _CH

def quartic(qJ):
    ch = channels()
    h = R(0)
    for J, q in qJ.items():
        q = sp.Rational(q)
        h += QQ(int(q.p), int(q.q)) * ch[J]
    return h

def norm2_G():
    return sum(QQ(GDIAG[a]) * X[a] ** 2 for a in range(14))

def grad(h):
    return [h.diff(X[a]) for a in range(14)]

def hess(h):
    g = grad(h)
    return [[g[a].diff(X[b]) for b in range(14)] for a in range(14)]

def to_sym(poly_elem, xs):
    """evaluate ring element at sympy values xs (exact)."""
    tot = sp.Integer(0)
    for mon, c in poly_elem.terms():
        t = sp.Rational(int(c.numerator), int(c.denominator))
        for a, e in enumerate(mon):
            if e: t *= xs[a] ** e
        tot += t
    return tot

def x_of_w(w):
    """w: list of 7 sympy CS components (m = 3..-3) -> chart x (exact)."""
    re = [sp.re(sp.sympify(t)) / sp.sqrt(FM[a]) for a, t in enumerate(w)]
    im = [sp.im(sp.sympify(t)) / sp.sqrt(FM[a]) for a, t in enumerate(w)]
    return [sp.radsimp(t) for t in re + im]
