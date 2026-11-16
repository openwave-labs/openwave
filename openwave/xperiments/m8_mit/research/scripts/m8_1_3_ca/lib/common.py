"""Shared tools for this room.

Conventions (stated once, used everywhere):
  * quaternion w + x i + y j + z k  ->  SU(2) matrix [[w + x I, y + z I], [-y + z I, w - x I]]
    (a homomorphism: i -> diag(I,-I), j -> [[0,1],[-1,0]], k -> [[0,I],[I,0]], i*j = k).
  * V_j = Sym^{2j} C^2 = binary forms of degree 2j in (x, y); orthonormal Condon-Shortley basis
    e_m = x^{j+m} y^{j-m} / sqrt((j+m)!(j-m)!), so the weighted coordinate p_m is the coefficient
    of the monomial x^{j+m} y^{j-m}.  g = [[a,b],[c,d]] acts by x -> a x + c y, y -> b x + d y
    (the Sym^{2j} of g acting on column vectors), and D^j(g)_{m m'} = <e_m, g . e_m'>.
    J_- e_m = sqrt((j+m)(j-m+1)) e_{m-1}: the Condon-Shortley phase.
  * Indices m are listed in the order m = j, j-1, ..., -j (row 0 is m = j).
"""
import os, sys, itertools
from fractions import Fraction as Fr
import sympy as sp
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ---------------------------------------------------------------- checks
_FAILS = []
def check(name, cond, detail=""):
    """Print PASS/FAIL. Every PASS is a comparison that is evaluated, so it can fail."""
    tag = "PASS" if cond else "FAIL"
    if not cond:
        _FAILS.append(name)
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    return cond

def n_fails():
    return len(_FAILS)

# ---------------------------------------------------------------- Q(sqrt5) and Q(sqrt5)(i), exact
class Q5:
    """a + b sqrt5, a, b rational."""
    __slots__ = ("a", "b")
    def __init__(self, a=0, b=0):
        self.a = Fr(a); self.b = Fr(b)
    def __add__(s, o):
        o = _q5(o); return Q5(s.a + o.a, s.b + o.b)
    __radd__ = __add__
    def __neg__(s): return Q5(-s.a, -s.b)
    def __sub__(s, o): return s + (-_q5(o))
    def __rsub__(s, o): return _q5(o) - s
    def __mul__(s, o):
        o = _q5(o); return Q5(s.a * o.a + 5 * s.b * o.b, s.a * o.b + s.b * o.a)
    __rmul__ = __mul__
    def conj5(s): return Q5(s.a, -s.b)          # Galois conjugation sqrt5 -> -sqrt5
    def inv(s):
        n = s.a * s.a - 5 * s.b * s.b
        return Q5(s.a / n, -s.b / n)
    def __truediv__(s, o): return s * _q5(o).inv()
    def __eq__(s, o):
        o = _q5(o); return s.a == o.a and s.b == o.b
    def __hash__(s): return hash((s.a, s.b))
    def is0(s): return s.a == 0 and s.b == 0
    def sym(s): return sp.Rational(s.a.numerator, s.a.denominator) + sp.Rational(s.b.numerator, s.b.denominator) * sp.sqrt(5)
    def mpf(s): return mp.mpf(s.a.numerator) / s.a.denominator + mp.mpf(s.b.numerator) / s.b.denominator * mp.sqrt(5)
    def __repr__(s): return str(s.sym())

def _q5(o):
    return o if isinstance(o, Q5) else Q5(o)

PHI = Q5(Fr(1, 2), Fr(1, 2))     # (1+sqrt5)/2
PHIINV = PHI - 1                 # 1/phi = phi - 1

class C5:
    """re + I im with re, im in Q(sqrt5)."""
    __slots__ = ("r", "i")
    def __init__(self, r=0, i=0):
        self.r = _q5(r); self.i = _q5(i)
    def __add__(s, o):
        o = _c5(o); return C5(s.r + o.r, s.i + o.i)
    __radd__ = __add__
    def __neg__(s): return C5(-s.r, -s.i)
    def __sub__(s, o): return s + (-_c5(o))
    def __mul__(s, o):
        o = _c5(o); return C5(s.r * o.r - s.i * o.i, s.r * o.i + s.i * o.r)
    __rmul__ = __mul__
    def conj(s): return C5(s.r, -s.i)
    def conj5(s): return C5(s.r.conj5(), s.i.conj5())
    def __eq__(s, o):
        o = _c5(o); return s.r == o.r and s.i == o.i
    def __hash__(s): return hash((s.r, s.i))
    def is0(s): return s.r.is0() and s.i.is0()
    def sym(s): return s.r.sym() + sp.I * s.i.sym()
    def mpc(s): return mp.mpc(s.r.mpf(), s.i.mpf())
    def __repr__(s): return str(s.sym())

def _c5(o):
    return o if isinstance(o, C5) else C5(o)

# ---------------------------------------------------------------- quaternions over Q(sqrt5)
class Quat:
    __slots__ = ("c",)
    def __init__(self, w, x, y, z):
        self.c = (_q5(w), _q5(x), _q5(y), _q5(z))
    def __mul__(s, o):
        a1, b1, c1, d1 = s.c; a2, b2, c2, d2 = o.c
        return Quat(a1*a2 - b1*b2 - c1*c2 - d1*d2,
                    a1*b2 + b1*a2 + c1*d2 - d1*c2,
                    a1*c2 - b1*d2 + c1*a2 + d1*b2,
                    a1*d2 + b1*c2 - c1*b2 + d1*a2)
    def conj(s):
        w, x, y, z = s.c; return Quat(w, -x, -y, -z)
    def norm2(s):
        return sum((t * t for t in s.c), Q5(0))
    def __eq__(s, o): return all(a == b for a, b in zip(s.c, o.c))
    def __hash__(s): return hash(s.c)
    def matrix(s):
        """SU(2) matrix [[a,b],[c,d]] with entries in C5."""
        w, x, y, z = s.c
        return [[C5(w, x), C5(y, z)], [C5(-y, z), C5(w, -x)]]
    def __repr__(s): return "(" + ", ".join(str(t) for t in s.c) + ")"

def generators():
    half = Fr(1, 2)
    q1 = Quat(half, half, half, half)
    q2 = Quat(PHI * half, PHIINV * half, half, 0)
    return q1, q2

def closure(gens):
    one = Quat(1, 0, 0, 0)
    elems = {one}
    frontier = [one]
    while frontier:
        new = []
        for g in frontier:
            for h in gens:
                p = g * h
                if p not in elems:
                    elems.add(p); new.append(p)
        frontier = new
    return list(elems)

# ---------------------------------------------------------------- half-integers
def mlist(j):
    """m = j, j-1, ..., -j as sympy Rationals."""
    j = sp.Rational(j)
    return [j - k for k in range(int(2 * j) + 1)]

# ---------------------------------------------------------------- Clebsch-Gordan (Racah formula), exact
_cg_cache = {}
def cg(j1, m1, j2, m2, J, M):
    """<j1 m1; j2 m2 | J M> in the Condon-Shortley convention, exact (sympy), from Racah's formula."""
    j1, m1, j2, m2, J, M = (sp.Rational(t) for t in (j1, m1, j2, m2, J, M))
    key = (j1, m1, j2, m2, J, M)
    if key in _cg_cache:
        return _cg_cache[key]
    val = sp.Integer(0)
    ok = (m1 + m2 == M and abs(m1) <= j1 and abs(m2) <= j2 and abs(M) <= J
          and abs(j1 - j2) <= J <= j1 + j2 and (j1 + j2 + J).is_integer
          and (j1 - m1).is_integer and (j2 - m2).is_integer and (J - M).is_integer)
    if ok:
        f = sp.factorial
        pre = (2 * J + 1) * f(J + j1 - j2) * f(J - j1 + j2) * f(j1 + j2 - J) / f(j1 + j2 + J + 1)
        pre *= f(J + M) * f(J - M) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2)
        s = sp.Integer(0)
        for k in range(0, int(j1 + j2 + J) + 2):
            args = [j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k]
            if k < 0 or any(a < 0 for a in args):
                continue
            den = f(k)
            for a in args:
                den *= f(a)
            s += sp.Integer(-1) ** k / den
        pre = sp.Rational(pre); s = sp.Rational(s)       # both exact rationals
        val = (1 if s > 0 else -1) * sp.sqrt(pre * s * s) if s != 0 else sp.Integer(0)
    _cg_cache[key] = val
    return val

def cg_rational_square(j1, m1, j2, m2, J, M):
    """(sign, square) exactly: CG = sign * sqrt(square)."""
    v = cg(j1, m1, j2, m2, J, M)
    sq = sp.Rational(v ** 2)
    return (0 if sq == 0 else (1 if v > 0 else -1)), sq

def W_matrix(J, j1, j2):
    """CG matrix W[M, (m1,m2)] = <j1 m1; j2 m2 | J M>, rows M=J..-J, columns m1 (outer) x m2 (inner). sympy."""
    Ms, m1s, m2s = mlist(J), mlist(j1), mlist(j2)
    W = sp.zeros(len(Ms), len(m1s) * len(m2s))
    for a, M in enumerate(Ms):
        for b, m1 in enumerate(m1s):
            m2 = M - m1
            if abs(m2) <= sp.Rational(j2):
                c = m2s.index(m2)
                W[a, b * len(m2s) + c] = cg(j1, m1, j2, m2, J, M)
    return W

# ---------------------------------------------------------------- D^j
def D_generic(j, a, b, c, d, binom, sqrt, zero, one):
    """D^j for g = [[a,b],[c,d]] in the orthonormal CS basis. Works for any ring given helpers."""
    j2 = int(2 * sp.Rational(j))
    ms = [sp.Rational(j2, 2) - k for k in range(j2 + 1)]
    n = j2 + 1
    D = [[zero for _ in range(n)] for _ in range(n)]
    def pw(x, e):
        r = one
        for _ in range(e):
            r = r * x
        return r
    for col, mp_ in enumerate(ms):
        jp = int(sp.Rational(j2, 2) + mp_)   # j + m'
        jm = j2 - jp                          # j - m'
        for row, m in enumerate(ms):
            tgt = int(sp.Rational(j2, 2) + m)  # exponent of x wanted: j + m
            s = zero
            for k in range(0, jp + 1):
                l = tgt - k
                if l < 0 or l > jm:
                    continue
                coef = binom(jp, k) * binom(jm, l)
                s = s + coef * pw(a, k) * pw(c, jp - k) * pw(b, l) * pw(d, jm - l)
            # orthonormal: e_m = x^{j+m}y^{j-m}/sqrt(N_m); D_{m m'} = s * sqrt(N_m / N_m')
            Nm = sp.factorial(tgt) * sp.factorial(j2 - tgt)
            Nmp = sp.factorial(jp) * sp.factorial(jm)
            D[row][col] = s * sqrt(Nm, Nmp)
    return D

def D_mp(j, g):
    """mpmath matrix D^j(g) for g a 2x2 list of mpc."""
    (a, b), (c, d) = g
    M = D_generic(j, a, b, c, d, lambda n, k: mp.binomial(n, k),
                  lambda p, q: mp.sqrt(mp.mpf(int(p)) / int(q)), mp.mpc(0), mp.mpc(1))
    return mp.matrix(M)

def D_weighted_exact(j, q):
    """D^j(q) on WEIGHTED coordinates p (coefficients of monomials), exact in C5:
    returns Dw with  p'_m = sum_m' Dw[m][m'] p_m'."""
    (a, b), (c, d) = q.matrix()
    from math import comb
    return D_generic(j, a, b, c, d, lambda n, k: comb(n, k), lambda p, q_: 1, C5(0), C5(1))

def theta_U(j):
    """U with (Theta u) = U conj(u): (Theta u)_m = (-1)^(j-m) conj(u_{-m})."""
    ms = mlist(j); n = len(ms)
    U = sp.zeros(n, n)
    for r, m in enumerate(ms):
        U[r, ms.index(-m)] = sp.Integer(-1) ** (sp.Rational(j) - m)
    return U

def quat_mp(q):
    return [[q.matrix()[r][c].mpc() for c in range(2)] for r in range(2)]

def rel(p):
    """path relative to the room's top level, for printing."""
    return os.path.relpath(p, ROOT)
