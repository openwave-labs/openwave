"""Exact arithmetic used by the room's scripts.

Q5  : elements a + b*sqrt5 of Q(sqrt5), a, b Fractions.
Q5i : elements x + i*y with x, y in Q5 (the field Q(sqrt5, i)).
Quaternions with components in Q5.
"""
from fractions import Fraction as Fr


class Q5:
    __slots__ = ("a", "b")

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
        o = q5(o)
        return Q5(s.a - o.a, s.b - o.b)

    def __rsub__(s, o):
        return q5(o) - s

    def __mul__(s, o):
        o = q5(o)
        return Q5(s.a * o.a + 5 * s.b * o.b, s.a * o.b + s.b * o.a)

    __rmul__ = __mul__

    def conj(s):  # Galois conjugate sqrt5 -> -sqrt5
        return Q5(s.a, -s.b)

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

    def iszero(s):
        return s.a == 0 and s.b == 0

    def __float__(s):
        return float(s.a) + float(s.b) * 5 ** 0.5

    def mp(s, mpmath):
        return mpmath.mpf(s.a.numerator) / s.a.denominator + mpmath.mpf(s.b.numerator) / s.b.denominator * mpmath.sqrt(5)

    def sympy(s):
        import sympy as sp
        return sp.Rational(s.a.numerator, s.a.denominator) + sp.Rational(s.b.numerator, s.b.denominator) * sp.sqrt(5)

    def __repr__(s):
        return f"({s.a}+{s.b}*sqrt(5))"


def q5(x):
    return x if isinstance(x, Q5) else Q5(x, 0)


SQ5 = Q5(0, 1)
PHI = Q5(Fr(1, 2), Fr(1, 2))


class Q5i:
    __slots__ = ("re", "im")

    def __init__(self, re=0, im=0):
        self.re = q5(re)
        self.im = q5(im)

    def __add__(s, o):
        o = q5i(o)
        return Q5i(s.re + o.re, s.im + o.im)

    __radd__ = __add__

    def __neg__(s):
        return Q5i(-s.re, -s.im)

    def __sub__(s, o):
        o = q5i(o)
        return Q5i(s.re - o.re, s.im - o.im)

    def __mul__(s, o):
        o = q5i(o)
        return Q5i(s.re * o.re - s.im * o.im, s.re * o.im + s.im * o.re)

    __rmul__ = __mul__

    def cc(s):  # complex conjugate
        return Q5i(s.re, -s.im)

    def iszero(s):
        return s.re.iszero() and s.im.iszero()

    def __eq__(s, o):
        o = q5i(o)
        return s.re == o.re and s.im == o.im

    def __hash__(s):
        return hash((s.re, s.im))

    def mp(s, mpmath):
        return mpmath.mpc(s.re.mp(mpmath), s.im.mp(mpmath))

    def __repr__(s):
        return f"[{s.re} + i{s.im}]"


def q5i(x):
    if isinstance(x, Q5i):
        return x
    return Q5i(q5(x), 0)


I5 = Q5i(0, 1)


# ---------- quaternions over Q5: tuple (w, x, y, z) ----------
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


def qkey(p):
    return tuple((c.a, c.b) for c in p)


def quat_to_ab(p):
    """Quaternion-to-matrix convention of this room:
    i -> -i sigma_x, j -> -i sigma_y, k -> -i sigma_z, so
    q = w + x i + y j + z k  ->  U = [[a, -conj(b)], [b, conj(a)]],
    a = w - i z, b = y - i x."""
    w, x, y, z = p
    a = Q5i(w, -z)
    b = Q5i(y, -x)
    return a, b
