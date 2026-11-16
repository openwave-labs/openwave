"""Exact algebra for the room: Q(sqrt5), Q(sqrt5,i), the group Gamma_0 from its two
generators, conjugacy classes, characters derived from V_j restrictions, Clebsch-Gordan
coefficients (Racah formula, Condon-Shortley), and D^j on the weighted (monomial) basis.

Conventions (stated in RETURN.md):
  quaternion q = w + x i + y j + z k  ->  U = w I - i (x sx + y sy + z sz)
      = [[w - i z, -y - i x], [y - i x, w + i z]]
  D on polynomials f(X,Y) of degree 2j: (g.f)(v) = f(U^T v), i.e. X -> aX + cY, Y -> bX + dY.
  monomial basis X^{j+m} Y^{j-m} carries the weighted coordinates p_m; the orthonormal
  Condon-Shortley basis is u_m = X^{j+m}Y^{j-m}/sqrt((j+m)!(j-m)!), w_m = p_m sqrt((j+m)!(j-m)!).
"""
from fractions import Fraction as Fr
from math import factorial, comb


# ---------------------------------------------------------------- Q(sqrt5)
class Q5:
    __slots__ = ("a", "b")  # a + b sqrt5

    def __init__(self, a=0, b=0):
        self.a = Fr(a)
        self.b = Fr(b)

    def __add__(s, o):
        o = q5(o)
        return Q5(s.a + o.a, s.b + o.b)

    __radd__ = __add__

    def __neg__(s):
        return Q5(-s.a, -s.b)

    def __sub__(s, o):
        return s + (-q5(o))

    def __rsub__(s, o):
        return q5(o) - s

    def __mul__(s, o):
        o = q5(o)
        return Q5(s.a * o.a + 5 * s.b * o.b, s.a * o.b + s.b * o.a)

    __rmul__ = __mul__

    def inv(s):
        n = s.a * s.a - 5 * s.b * s.b
        return Q5(s.a / n, -s.b / n)

    def __truediv__(s, o):
        return s * q5(o).inv()

    def __eq__(s, o):
        o = q5(o)
        return s.a == o.a and s.b == o.b

    def __hash__(s):
        return hash((s.a, s.b))

    def galois(s):
        return Q5(s.a, -s.b)

    def is_zero(s):
        return s.a == 0 and s.b == 0

    def __float__(s):
        return float(s.a) + float(s.b) * 5 ** 0.5

    def mp(s, mpm):
        return mpm.mpf(s.a.numerator) / s.a.denominator + mpm.mpf(s.b.numerator) / s.b.denominator * mpm.sqrt(5)

    def __repr__(s):
        if s.b == 0:
            return str(s.a)
        if s.a == 0:
            return f"({s.b})*sqrt(5)"
        return f"{s.a} + ({s.b})*sqrt(5)"

    def sympy_str(s):
        return f"({s.a}) + ({s.b})*sqrt(5)" if s.b != 0 else f"{s.a}"


def q5(x):
    return x if isinstance(x, Q5) else Q5(x)


# ---------------------------------------------------------------- Q(sqrt5, i)
class K:
    __slots__ = ("re", "im")

    def __init__(self, re=0, im=0):
        self.re = q5(re)
        self.im = q5(im)

    def __add__(s, o):
        o = kk(o)
        return K(s.re + o.re, s.im + o.im)

    __radd__ = __add__

    def __neg__(s):
        return K(-s.re, -s.im)

    def __sub__(s, o):
        return s + (-kk(o))

    def __mul__(s, o):
        o = kk(o)
        return K(s.re * o.re - s.im * o.im, s.re * o.im + s.im * o.re)

    __rmul__ = __mul__

    def conj(s):
        return K(s.re, -s.im)

    def __eq__(s, o):
        o = kk(o)
        return s.re == o.re and s.im == o.im

    def __hash__(s):
        return hash((s.re, s.im))

    def is_zero(s):
        return s.re.is_zero() and s.im.is_zero()

    def mpc(s, mpm):
        return mpm.mpc(s.re.mp(mpm), s.im.mp(mpm))

    def __repr__(s):
        return f"[{s.re!r}] + i[{s.im!r}]"


def kk(x):
    if isinstance(x, K):
        return x
    return K(x, 0)


# ---------------------------------------------------------------- quaternions
def qmul(p, q):
    a1, b1, c1, d1 = p
    a2, b2, c2, d2 = q
    return (a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2)


def qconj(p):
    return (p[0], -p[1], -p[2], -p[3])


def qnorm2(p):
    return p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3]


HALF = Fr(1, 2)
PHI = Q5(HALF, HALF)          # (1+sqrt5)/2
PHI_INV = Q5(-HALF, HALF)     # (sqrt5-1)/2 = 1/phi


def generators():
    q1 = (Q5(HALF), Q5(HALF), Q5(HALF), Q5(HALF))
    q2 = (PHI * HALF, PHI_INV * HALF, Q5(HALF), Q5(0))
    return q1, q2


def closure(gens, cap=100000):
    one = (Q5(1), Q5(0), Q5(0), Q5(0))
    elems = [one]
    seen = {one}
    frontier = [one]
    while frontier:
        new = []
        for g in frontier:
            for s in gens:
                h = qmul(g, s)
                if h not in seen:
                    seen.add(h)
                    elems.append(h)
                    new.append(h)
                    if len(elems) > cap:
                        raise RuntimeError("group too large")
        frontier = new
    return elems


def element_order(g):
    one = (Q5(1), Q5(0), Q5(0), Q5(0))
    h = g
    n = 1
    while h != one:
        h = qmul(h, g)
        n += 1
        if n > 10000:
            raise RuntimeError
    return n


def conjugacy_classes(G):
    Gset = set(G)
    remaining = list(G)
    done = set()
    classes = []
    for g in G:
        if g in done:
            continue
        cl = []
        cs = set()
        for x in G:
            c = qmul(qmul(x, g), qconj(x))
            assert c in Gset
            if c not in cs:
                cs.add(c)
                cl.append(c)
        for c in cl:
            done.add(c)
        classes.append(cl)
    return classes


# ---------------------------------------------------------------- characters
def cheb_U(n, x):
    """U_n(x) for x in Q5: character of V_{n/2} at an element with real part x."""
    u0, u1 = Q5(1), x * 2
    if n == 0:
        return u0
    for _ in range(n - 1):
        u0, u1 = u1, x * 2 * u1 - u0
    return u1


def inner(ch1, ch2, sizes, order):
    """<ch1, ch2> = (1/|G|) sum size * ch1 * conj(ch2); characters here are real (checked)."""
    tot = Q5(0)
    for a, b, s in zip(ch1, ch2, sizes):
        tot = tot + a * b * s
    return tot / order


# ---------------------------------------------------------------- SU(2) and D^j
def su2_of_quat(q):
    w, x, y, z = q
    a = K(w, -z)
    b = K(-y, -x)
    c = K(y, -x)
    d = K(w, z)
    return a, b, c, d


def _polymul(P, Qp):
    R = [K(0)] * (len(P) + len(Qp) - 1)
    for i, p in enumerate(P):
        if p.is_zero():
            continue
        for k, q in enumerate(Qp):
            R[i + k] = R[i + k] + p * q
    return R


def _polypow(P, n):
    R = [K(1)]
    for _ in range(n):
        R = _polymul(R, P)
    return R


def Dp_exact(q, twoj):
    """Matrix of g on weighted coordinates p (index i = j - m, i.e. m = j, j-1, ..., -j).
    p'_n = sum_m M[n][m] p_m, from f(X,Y) -> f(aX+cY, bX+dY); dehomogenised with t = X/Y."""
    a, b, c, d = su2_of_quat(q)
    n2 = twoj
    M = [[K(0)] * (n2 + 1) for _ in range(n2 + 1)]
    for col in range(n2 + 1):
        # m = j - col: exponent of X is j+m = n2 - col, of Y is col
        ex, ey = n2 - col, col
        P = _polymul(_polypow([c, a], ex), _polypow([d, b], ey))  # coefficients in t^k
        for k, coef in enumerate(P):
            # t^k  <->  X^k Y^{n2-k}, i.e. j+m' = k, row index = n2 - k
            M[n2 - k][col] = coef
    return M


def matmulK(A, B):
    n, m, p = len(A), len(B), len(B[0])
    R = [[K(0)] * p for _ in range(n)]
    for i in range(n):
        for k in range(m):
            a = A[i][k]
            if a.is_zero():
                continue
            Bk = B[k]
            Ri = R[i]
            for j in range(p):
                if not Bk[j].is_zero():
                    Ri[j] = Ri[j] + a * Bk[j]
    return R


def weights(twoj):
    """F_m = (j+m)!(j-m)! in row order m = j..-j (twoj/2 may be half-integer)."""
    return [factorial(twoj - i) * factorial(i) for i in range(twoj + 1)]


# ---------------------------------------------------------------- Clebsch-Gordan (Racah)
def cg_exact(j1, m1, j2, m2, J, M):
    """<j1 m1; j2 m2 | J M> in Condon-Shortley convention as (r, R): value = r*sqrt(R),
    r, R Fractions, R >= 0.  Arguments are Fractions (half-integers allowed)."""
    j1, m1, j2, m2, J, M = map(Fr, (j1, m1, j2, m2, J, M))
    if m1 + m2 != M or abs(m1) > j1 or abs(m2) > j2 or abs(M) > J:
        return Fr(0), Fr(0)
    if J > j1 + j2 or J < abs(j1 - j2):
        return Fr(0), Fr(0)

    def f(x):
        assert x.denominator == 1 and x >= 0, x
        return factorial(int(x))

    R = Fr((2 * J + 1) * f(J + j1 - j2) * f(J - j1 + j2) * f(j1 + j2 - J), f(j1 + j2 + J + 1))
    R *= f(J + M) * f(J - M) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2)
    tot = Fr(0)
    kmax = int(j1 + j2 + J) + 2
    for k in range(kmax + 1):
        args = [Fr(k), j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k]
        if all(a >= 0 for a in args):
            den = 1
            for a in args:
                den *= f(a)
            tot += Fr((-1) ** k, den)
    return tot, R


def cg_value_sq_sign(j1, m1, j2, m2, J, M):
    r, R = cg_exact(j1, m1, j2, m2, J, M)
    return r * r * R, (0 if r == 0 or R == 0 else (1 if r > 0 else -1))
