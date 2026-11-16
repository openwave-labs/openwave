"""Exact arithmetic in Q(sqrt5) and quaternions over it (own code, no lookup)."""
from fractions import Fraction as F
import mpmath
import sympy


def _c(o):
    return o if isinstance(o, Q5) else Q5(o, 0)


class Q5:
    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = F(a)
        self.b = F(b)

    def __add__(s, o):
        o = _c(o); return Q5(s.a + o.a, s.b + o.b)
    __radd__ = __add__

    def __sub__(s, o):
        o = _c(o); return Q5(s.a - o.a, s.b - o.b)

    def __rsub__(s, o):
        return _c(o) - s

    def __neg__(s):
        return Q5(-s.a, -s.b)

    def __mul__(s, o):
        o = _c(o); return Q5(s.a * o.a + 5 * s.b * o.b, s.a * o.b + s.b * o.a)
    __rmul__ = __mul__

    def galois(s):
        return Q5(s.a, -s.b)

    def fnorm(s):
        return s.a * s.a - 5 * s.b * s.b

    def inv(s):
        n = s.fnorm(); return Q5(s.a / n, -s.b / n)

    def __truediv__(s, o):
        return s * _c(o).inv()

    def __eq__(s, o):
        o = _c(o); return s.a == o.a and s.b == o.b

    def __hash__(s):
        return hash((s.a, s.b))

    def is_rational_integer(s):
        return s.b == 0 and s.a.denominator == 1

    def sign(s):
        a, b = s.a, s.b
        if b == 0:
            return (a > 0) - (a < 0)
        if a == 0:
            return (b > 0) - (b < 0)
        if (a > 0) == (b > 0):
            return 1 if a > 0 else -1
        return ((a > 0) - (a < 0)) if a * a > 5 * b * b else ((b > 0) - (b < 0))

    def mp(s):
        return mpmath.mpf(s.a.numerator) / s.a.denominator + mpmath.mpf(s.b.numerator) / s.b.denominator * mpmath.sqrt(5)

    def __float__(s):
        return float(s.mp())

    def sym(s):
        return sympy.Rational(s.a.numerator, s.a.denominator) + sympy.Rational(s.b.numerator, s.b.denominator) * sympy.sqrt(5)

    def __str__(s):
        return str(s.sym())
    __repr__ = __str__


def qmul(p, q):
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def qconj(q):
    return (q[0], -q[1], -q[2], -q[3])


def qnorm2(q):
    return q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]


ONE = (Q5(1), Q5(0), Q5(0), Q5(0))
PHI = Q5(F(1, 2), F(1, 2))
PHI_INV = Q5(F(-1, 2), F(1, 2))
H = F(1, 2)
q1 = (Q5(H), Q5(H), Q5(H), Q5(H))
q2 = (PHI * H, PHI_INV * H, Q5(H), Q5(0))
