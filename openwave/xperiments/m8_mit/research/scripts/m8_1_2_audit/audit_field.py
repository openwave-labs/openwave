"""Exact arithmetic in K = Q(sqrt5, i) and exact quaternions over Q(sqrt5).

Element of K: (a + b*s) + (c + d*s)*i, s = sqrt(5), a..d Fractions.
No floating point anywhere in this module.
"""
from fractions import Fraction as Fr
import itertools


class R5:
    """a + b*sqrt5, a, b rational (a real subfield of R)."""
    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a = Fr(a)
        self.b = Fr(b)

    def __add__(self, o):
        o = R5.c(o)
        return R5(self.a + o.a, self.b + o.b)
    __radd__ = __add__

    def __sub__(self, o):
        o = R5.c(o)
        return R5(self.a - o.a, self.b - o.b)

    def __rsub__(self, o):
        return R5.c(o) - self

    def __neg__(self):
        return R5(-self.a, -self.b)

    def __mul__(self, o):
        o = R5.c(o)
        return R5(self.a * o.a + 5 * self.b * o.b, self.a * o.b + self.b * o.a)
    __rmul__ = __mul__

    def inv(self):
        n = self.a * self.a - 5 * self.b * self.b
        if n == 0:
            raise ZeroDivisionError("R5 zero")
        return R5(self.a / n, -self.b / n)

    def __truediv__(self, o):
        return self * R5.c(o).inv()

    def __eq__(self, o):
        o = R5.c(o)
        return self.a == o.a and self.b == o.b

    def __hash__(self):
        return hash((self.a, self.b))

    def iszero(self):
        return self.a == 0 and self.b == 0

    def sign(self):
        """exact sign of a + b sqrt5 as a real number"""
        a, b = self.a, self.b
        if b == 0:
            return (a > 0) - (a < 0)
        if a == 0:
            return (b > 0) - (b < 0)
        if (a > 0) == (b > 0):
            return 1 if a > 0 else -1
        # opposite signs: compare a^2 with 5 b^2
        if a * a > 5 * b * b:
            return 1 if a > 0 else -1
        return 1 if b > 0 else -1

    @staticmethod
    def c(x):
        return x if isinstance(x, R5) else R5(x)

    def __repr__(self):
        return f"({self.a}+{self.b}*s5)"

    def tofloat(self):
        return float(self.a) + float(self.b) * 5 ** 0.5


class K:
    """p + q*i with p, q in R5."""
    __slots__ = ("p", "q")

    def __init__(self, p=0, q=0):
        self.p = R5.c(p)
        self.q = R5.c(q)

    @staticmethod
    def c(x):
        if isinstance(x, K):
            return x
        return K(x, 0)

    def __add__(self, o):
        o = K.c(o)
        return K(self.p + o.p, self.q + o.q)
    __radd__ = __add__

    def __sub__(self, o):
        o = K.c(o)
        return K(self.p - o.p, self.q - o.q)

    def __rsub__(self, o):
        return K.c(o) - self

    def __neg__(self):
        return K(-self.p, -self.q)

    def __mul__(self, o):
        o = K.c(o)
        return K(self.p * o.p - self.q * o.q, self.p * o.q + self.q * o.p)
    __rmul__ = __mul__

    def conj(self):
        return K(self.p, -self.q)

    def abs2(self):
        return self.p * self.p + self.q * self.q  # R5

    def inv(self):
        n = self.abs2()
        if n.iszero():
            raise ZeroDivisionError("K zero")
        ni = n.inv()
        return K(self.p * ni, -self.q * ni)

    def __truediv__(self, o):
        return self * K.c(o).inv()

    def __eq__(self, o):
        o = K.c(o)
        return self.p == o.p and self.q == o.q

    def __hash__(self):
        return hash((self.p, self.q))

    def iszero(self):
        return self.p.iszero() and self.q.iszero()

    def __repr__(self):
        return f"[{self.p} + i{self.q}]"

    def tocomplex(self):
        return complex(self.p.tofloat(), self.q.tofloat())

    def tosympy(self):
        import sympy as sp
        s5 = sp.sqrt(5)
        return (sp.Rational(self.p.a.numerator, self.p.a.denominator)
                + sp.Rational(self.p.b.numerator, self.p.b.denominator) * s5
                + sp.I * (sp.Rational(self.q.a.numerator, self.q.a.denominator)
                          + sp.Rational(self.q.b.numerator, self.q.b.denominator) * s5))


ZERO = K(0)
ONE = K(1)
IU = K(0, 1)


class Quat:
    """w + x i + y j + z k, entries R5."""
    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w, x, y, z):
        self.w, self.x, self.y, self.z = R5.c(w), R5.c(x), R5.c(y), R5.c(z)

    def __mul__(self, o):
        a1, b1, c1, d1 = self.w, self.x, self.y, self.z
        a2, b2, c2, d2 = o.w, o.x, o.y, o.z
        return Quat(a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
                    a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                    a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                    a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2)

    def conj(self):
        return Quat(self.w, -self.x, -self.y, -self.z)

    def norm2(self):
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z

    def key(self):
        return (self.w.a, self.w.b, self.x.a, self.x.b, self.y.a, self.y.b, self.z.a, self.z.b)

    def __eq__(self, o):
        return self.key() == o.key()

    def __hash__(self):
        return hash(self.key())

    def __repr__(self):
        return f"Q({self.w},{self.x},{self.y},{self.z})"

    def su2(self):
        """2x2 SU(2) matrix: q -> w I - i (x sx + y sy + z sz) (a homomorphism).
        Returns [[a, b], [c, d]] entries in K."""
        w, x, y, z = self.w, self.x, self.y, self.z
        a = K(w, -z)
        b = K(-y, -x)       # -i*x - y
        c = K(y, -x)        # -i*x + y
        d = K(w, z)
        return [[a, b], [c, d]]


PHI = R5(Fr(1, 2), Fr(1, 2))          # (1+sqrt5)/2
PHI_INV = R5(Fr(-1, 2), Fr(1, 2))     # (sqrt5-1)/2

Q1 = Quat(Fr(1, 2), Fr(1, 2), Fr(1, 2), Fr(1, 2))
Q2 = Quat(PHI * Fr(1, 2), PHI_INV * Fr(1, 2), Fr(1, 2), 0)
QONE = Quat(1, 0, 0, 0)


def closure(gens):
    elems = {QONE}
    frontier = [QONE]
    while frontier:
        new = []
        for g in frontier:
            for h in gens:
                p = g * h
                if p not in elems:
                    elems.add(p)
                    new.append(p)
        frontier = new
    return list(elems)


def mat_mul(A, B):
    n, m, p = len(A), len(B), len(B[0])
    return [[sum((A[i][k] * B[k][j] for k in range(m)), ZERO) for j in range(p)] for i in range(n)]


def rref(M):
    """Row reduce a list-of-lists over K. Returns (R, pivots)."""
    M = [row[:] for row in M]
    rows, cols = len(M), len(M[0])
    piv = []
    r = 0
    for c in range(cols):
        pr = None
        for i in range(r, rows):
            if not M[i][c].iszero():
                pr = i
                break
        if pr is None:
            continue
        M[r], M[pr] = M[pr], M[r]
        inv = M[r][c].inv()
        M[r] = [x * inv for x in M[r]]
        for i in range(rows):
            if i != r and not M[i][c].iszero():
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == rows:
            break
    return M, piv


def nullspace(M):
    cols = len(M[0])
    R, piv = rref(M)
    free = [c for c in range(cols) if c not in piv]
    basis = []
    for f in free:
        v = [ZERO] * cols
        v[f] = ONE
        for i, pc in enumerate(piv):
            v[pc] = -R[i][f]
        basis.append(v)
    return basis


def rank(M):
    return len(rref(M)[1])
