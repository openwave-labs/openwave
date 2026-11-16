"""Shared exact and high-precision machinery for this room.

Conventions (stated once, used everywhere):
  * quaternion q = w + x i + y j + z k  ->  U(q) = w I - i (x sx + y sy + z sz)
        = [[w - i z, -y - i x], [y - i x, w + i z]]   (a homomorphism: U(pq) = U(p) U(q)).
  * V_j = homogeneous polynomials of degree 2j in (x, y); e_m = x^(j+m) y^(j-m) / N_m,
    N_m = sqrt((j+m)! (j-m)!), orthonormal (Fischer product), Condon-Shortley ladder
    J+ = x d/dy, J- = y d/dx, Jz = (x d/dx - y d/dy)/2.
  * D^j(U): f(x, y) -> f((x, y) U), a homomorphism; on weighted coordinates p_m = w_m / N_m
    its matrix M^j(U) has polynomial entries in U; on w it is D = N M N^-1.
  * CG coefficients: Racah's closed formula, exact in sympy (Condon-Shortley phase).
"""
from fractions import Fraction as Fr
import itertools, math, os
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))


def rel(path):
    return os.path.join(HERE, path)


# ---------------------------------------------------------------- Q(sqrt5) and Q(sqrt5, i)
class Q5:
    __slots__ = ("a", "b")

    def __init__(self, a, b=0):
        self.a = Fr(a); self.b = Fr(b)

    def __add__(s, o):
        o = q5(o); return Q5(s.a + o.a, s.b + o.b)
    __radd__ = __add__

    def __sub__(s, o):
        o = q5(o); return Q5(s.a - o.a, s.b - o.b)

    def __rsub__(s, o):
        return q5(o) - s

    def __neg__(s):
        return Q5(-s.a, -s.b)

    def __mul__(s, o):
        o = q5(o); return Q5(s.a * o.a + 5 * s.b * o.b, s.a * o.b + s.b * o.a)
    __rmul__ = __mul__

    def inv(s):
        n = s.a * s.a - 5 * s.b * s.b
        return Q5(s.a / n, -s.b / n)

    def __truediv__(s, o):
        return s * q5(o).inv()

    def galois(s):
        return Q5(s.a, -s.b)

    def __eq__(s, o):
        o = q5(o); return s.a == o.a and s.b == o.b

    def __hash__(s):
        return hash((s.a, s.b))

    def iszero(s):
        return s.a == 0 and s.b == 0

    def sym(s):
        return sp.Rational(s.a.numerator, s.a.denominator) + sp.Rational(s.b.numerator, s.b.denominator) * sp.sqrt(5)

    def mp(s, mpm):
        return mpm.mpf(s.a.numerator) / s.a.denominator + mpm.mpf(s.b.numerator) / s.b.denominator * mpm.sqrt(5)

    def __repr__(s):
        return f"({s.a}+{s.b}r5)"


def q5(x):
    return x if isinstance(x, Q5) else Q5(x)


class K:
    """Q(sqrt5)(i): re + i im."""
    __slots__ = ("re", "im")

    def __init__(self, re, im=0):
        self.re = q5(re); self.im = q5(im)

    def __add__(s, o):
        o = kk(o); return K(s.re + o.re, s.im + o.im)
    __radd__ = __add__

    def __sub__(s, o):
        o = kk(o); return K(s.re - o.re, s.im - o.im)

    def __neg__(s):
        return K(-s.re, -s.im)

    def __mul__(s, o):
        o = kk(o); return K(s.re * o.re - s.im * o.im, s.re * o.im + s.im * o.re)
    __rmul__ = __mul__

    def conj(s):
        return K(s.re, -s.im)

    def __eq__(s, o):
        o = kk(o); return s.re == o.re and s.im == o.im

    def __hash__(s):
        return hash((s.re, s.im))

    def iszero(s):
        return s.re.iszero() and s.im.iszero()

    def sym(s):
        return s.re.sym() + sp.I * s.im.sym()

    def mp(s, mpm):
        return mpm.mpc(s.re.mp(mpm), s.im.mp(mpm))


def kk(x):
    return x if isinstance(x, K) else K(x)


# ---------------------------------------------------------------- quaternions
def qmul(p, q):
    a1, b1, c1, d1 = p; a2, b2, c2, d2 = q
    return (a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2)


def qconj(q):
    return (q[0], -q[1], -q[2], -q[3])


def qnorm2(q):
    return sum((c * c for c in q), Q5(0))


PHI = Q5(Fr(1, 2), Fr(1, 2))
PHI_INV = PHI - 1
GEN_Q1 = (Q5(Fr(1, 2)), Q5(Fr(1, 2)), Q5(Fr(1, 2)), Q5(Fr(1, 2)))
GEN_Q2 = (PHI * Fr(1, 2), PHI_INV * Fr(1, 2), Q5(Fr(1, 2)), Q5(0))


def generate_group(gens):
    one = (Q5(1), Q5(0), Q5(0), Q5(0))
    seen = {one}; frontier = [one]
    while frontier:
        nxt = []
        for g in frontier:
            for s in gens:
                h = qmul(g, s)
                if h not in seen:
                    seen.add(h); nxt.append(h)
        frontier = nxt
    return sorted(seen, key=lambda q: tuple((c.a, c.b) for c in q))


def conj_classes(G):
    remaining = set(G); classes = []
    for g in G:
        if g not in remaining:
            continue
        cl = {qmul(qmul(h, g), qconj(h)) for h in G}
        remaining -= cl
        classes.append(sorted(cl, key=lambda q: tuple((c.a, c.b) for c in q)))
    return classes


def U_of_q(q):
    w, x, y, z = q
    a = K(w, -z); b = K(-y, -x); c = K(y, -x); d = K(w, z)
    return a, b, c, d


def chebU(n, w):
    """U_n(w) = sin((n+1)t)/sin t at w = cos t; character of V_{n/2}."""
    u0, u1 = Q5(1), 2 * w
    if n == 0:
        return u0
    for _ in range(n - 1):
        u0, u1 = u1, 2 * w * u1 - u0
    return u1


# ---------------------------------------------------------------- D^j on weighted coordinates
def poly_mul(A, B, zero):
    out = [zero] * (len(A) + len(B) - 1)
    for i, a in enumerate(A):
        for k, b in enumerate(B):
            out[i + k] = out[i + k] + a * b
    return out


def Mp(n, U, zero, one):
    """Matrix of D^{n/2}(U) on weighted coordinates, index k = j + m (0..n).
    column k: image of x^k y^(n-k) is (a x + c y)^k (b x + d y)^(n-k); coefficient lists by x-power."""
    a, b, c, d = U
    lin1 = [c, a]  # c y + a x : coefficient of x^0 is c (y), x^1 is a
    lin2 = [d, b]
    M = [[zero] * (n + 1) for _ in range(n + 1)]
    for k in range(n + 1):
        P = [one]
        for _ in range(k):
            P = poly_mul(P, lin1, zero)
        for _ in range(n - k):
            P = poly_mul(P, lin2, zero)
        for kp in range(n + 1):
            M[kp][k] = P[kp]
    return M


def matmul(A, B, zero):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum((A[i][t] * B[t][k] for t in range(m)), zero) for k in range(p)] for i in range(n)]


# ---------------------------------------------------------------- Clebsch-Gordan (Racah), exact
def cg(j1, m1, j2, m2, J, M):
    j1, m1, j2, m2, J, M = map(sp.nsimplify, (j1, m1, j2, m2, J, M))
    if m1 + m2 != M or abs(m1) > j1 or abs(m2) > j2 or abs(M) > J:
        return sp.Integer(0)
    if J < abs(j1 - j2) or J > j1 + j2 or (j1 + j2 + J) % 1 != 0:
        return sp.Integer(0)
    f = sp.factorial
    pref = sp.sqrt((2 * J + 1) * f(J + j1 - j2) * f(J - j1 + j2) * f(j1 + j2 - J) / f(j1 + j2 + J + 1))
    pref *= sp.sqrt(f(J + M) * f(J - M) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2))
    s = sp.Integer(0)
    for kk_ in range(0, int(j1 + j2 + J) + 2):
        args = [kk_, j1 + j2 - J - kk_, j1 - m1 - kk_, j2 + m2 - kk_, J - j2 + m1 + kk_, J - j1 - m2 + kk_]
        if any(a < 0 for a in args):
            continue
        den = sp.Integer(1)
        for a in args:
            den *= f(a)
        s += sp.Integer(-1) ** kk_ / den
    return sp.radsimp(pref * s)


def mvals(j):
    j = sp.nsimplify(j)
    return [j - t for t in range(int(2 * j) + 1)]  # m = j, j-1, ..., -j


def Nm(j, m):
    return sp.sqrt(sp.factorial(j + m) * sp.factorial(j - m))


def sym_to_q5(e):
    e = sp.nsimplify(sp.sympify(e))
    b = sp.Rational(sp.expand(e).coeff(sp.sqrt(5)))
    a = sp.Rational(sp.expand(e - b * sp.sqrt(5)))
    return Q5(Fr(int(a.p), int(a.q)), Fr(int(b.p), int(b.q)))


def load_group_data():
    """Group elements, plus the slot characters as functions of the real part (classes are
    determined by the real part: checked in s0_group.py)."""
    import json
    d = json.load(open(rel("out_group.json")))
    G = generate_group([GEN_Q1, GEN_Q2])
    res = [sym_to_q5(c["re"]) for c in d["classes"]]
    chars = {k: [sym_to_q5(v) for v in vals] for k, vals in d["slots"].items()}

    def chi(slot, g):
        return chars[slot][res.index(g[0])]
    return G, chi


class Checker:
    def __init__(self):
        self.n_pass = 0; self.n_fail = 0

    def __call__(self, name, cond, detail=""):
        if cond:
            self.n_pass += 1; print(f"PASS  {name}" + (f"  [{detail}]" if detail else ""))
        else:
            self.n_fail += 1; print(f"FAIL  {name}" + (f"  [{detail}]" if detail else ""))
        return bool(cond)

    def summary(self):
        print(f"checks: {self.n_pass} pass, {self.n_fail} fail")
