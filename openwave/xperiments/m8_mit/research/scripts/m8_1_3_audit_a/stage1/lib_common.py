"""Shared exact arithmetic and helpers for the audit scripts.

Field K = Q(sqrt5)(i): element (a, b, c, d) = a + b*sqrt5 + i*(c + d*sqrt5), Fractions.
Everything here is derived in this room from definitions; nothing is looked up.
"""
import os, sys, json
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))


class Log:
    """print to stdout and to <script>.log beside the script."""
    def __init__(self, script_file, suffix=''):
        base = os.path.splitext(os.path.basename(script_file))[0]
        self.path = os.path.join(os.path.dirname(os.path.abspath(script_file)), base + suffix + '.log')
        self.f = open(self.path, 'w')
        self.fails = 0

    def __call__(self, *a):
        s = ' '.join(str(x) for x in a)
        print(s)
        self.f.write(s + '\n')
        self.f.flush()

    def check(self, name, ok, detail=''):
        tag = 'PASS' if ok else 'FAIL'
        if not ok:
            self.fails += 1
        self(f'{tag}: {name}' + (f'  [{detail}]' if detail else ''))
        return ok

    def close(self):
        self(f'-- {self.fails} FAIL line(s)')
        self.f.close()


def planted():
    return len(sys.argv) > 1 and sys.argv[1] == 'plant'


class K:
    __slots__ = ('a', 'b', 'c', 'd')

    def __init__(self, a=0, b=0, c=0, d=0):
        self.a = Fr(a); self.b = Fr(b); self.c = Fr(c); self.d = Fr(d)

    @staticmethod
    def real(x, y=0):  # x + y sqrt5
        return K(x, y, 0, 0)

    def __add__(self, o):
        o = tok(o); return K(self.a + o.a, self.b + o.b, self.c + o.c, self.d + o.d)
    __radd__ = __add__

    def __neg__(self):
        return K(-self.a, -self.b, -self.c, -self.d)

    def __sub__(self, o):
        return self + (-tok(o))

    def __rsub__(self, o):
        return tok(o) - self

    def __mul__(self, o):
        o = tok(o)
        # (x + i y)(u + i v), x = a + b r, y = c + d r, r^2 = 5
        def m(p, q, r, s):  # (p + q r)(r_ + s r)
            return (p * r + 5 * q * s, p * s + q * r)
        xu = m(self.a, self.b, o.a, o.b); yv = m(self.c, self.d, o.c, o.d)
        xv = m(self.a, self.b, o.c, o.d); yu = m(self.c, self.d, o.a, o.b)
        return K(xu[0] - yv[0], xu[1] - yv[1], xv[0] + yu[0], xv[1] + yu[1])
    __rmul__ = __mul__

    def conj(self):
        return K(self.a, self.b, -self.c, -self.d)

    def galois(self):  # sqrt5 -> -sqrt5
        return K(self.a, -self.b, self.c, -self.d)

    def inv(self):
        # 1/z = conj(z)/|z|^2, |z|^2 = n0 + n1 r ; 1/(n0+n1 r) = (n0 - n1 r)/(n0^2 - 5 n1^2)
        n = self * self.conj()
        n0, n1 = n.a, n.b
        den = n0 * n0 - 5 * n1 * n1
        return self.conj() * K(n0 / den, -n1 / den)

    def __truediv__(self, o):
        return self * tok(o).inv()

    def __eq__(self, o):
        o = tok(o); return (self.a, self.b, self.c, self.d) == (o.a, o.b, o.c, o.d)

    def __hash__(self):
        return hash((self.a, self.b, self.c, self.d))

    def iszero(self):
        return self.a == 0 and self.b == 0 and self.c == 0 and self.d == 0

    def isreal(self):
        return self.c == 0 and self.d == 0

    def israt(self):
        return self.b == 0 and self.c == 0 and self.d == 0

    def tofloat(self):
        import math
        r = math.sqrt(5)
        return complex(float(self.a) + float(self.b) * r, float(self.c) + float(self.d) * r)

    def tomp(self):
        import mpmath
        r = mpmath.sqrt(5)
        return mpmath.mpc(mpmath.mpf(self.a.numerator) / self.a.denominator + mpmath.mpf(self.b.numerator) / self.b.denominator * r,
                          mpmath.mpf(self.c.numerator) / self.c.denominator + mpmath.mpf(self.d.numerator) / self.d.denominator * r)

    def tosympy(self):
        import sympy as sp
        return sp.Rational(self.a) + sp.Rational(self.b) * sp.sqrt(5) + sp.I * (sp.Rational(self.c) + sp.Rational(self.d) * sp.sqrt(5))

    def __repr__(self):
        def r(x, y):
            if y == 0: return str(x)
            if x == 0: return f'{y}*sqrt5'
            return f'({x}+{y}*sqrt5)'
        re = r(self.a, self.b); im = r(self.c, self.d)
        if self.c == 0 and self.d == 0: return re
        return f'{re} + I*{im}'


def tok(x):
    return x if isinstance(x, K) else K(x)


ZERO = K(0); ONE = K(1)

# ---------------- quaternions over Q(sqrt5) ------------------
PHI = K.real(Fr(1, 2), Fr(1, 2))
PHIINV = K.real(Fr(-1, 2), Fr(1, 2))


def qmul(p, q):
    a1, b1, c1, d1 = p; a2, b2, c2, d2 = q
    return (a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
            a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
            a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
            a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2)


def qinv(p):  # unit quaternion
    return (p[0], -p[1], -p[2], -p[3])


def qkey(p):
    return tuple((x.a, x.b) for x in p)


GEN1 = (K(Fr(1, 2)), K(Fr(1, 2)), K(Fr(1, 2)), K(Fr(1, 2)))
GEN2 = (PHI * Fr(1, 2), PHIINV * Fr(1, 2), K(Fr(1, 2)), K(0))


def build_group():
    E = (K(1), K(0), K(0), K(0))
    els = {qkey(E): E}
    frontier = [E]
    while frontier:
        nf = []
        for g in frontier:
            for s in (GEN1, GEN2):
                h = qmul(g, s)
                kk = qkey(h)
                if kk not in els:
                    els[kk] = h; nf.append(h)
        frontier = nf
    return list(els.values())


def quat_to_U(q):
    """q = a + b i + c j + d k  ->  [[a + b I, c + d I], [-c + d I, a - b I]] (a homomorphism, checked)."""
    a, b, c, d = q
    I = K(0, 0, 1, 0)
    return [[a + I * b, c + I * d], [-c + I * d, a - I * b]]


def mm2(A, B):
    return [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
            [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]]


def ms(j2):
    """list of 2m for m = j..-j (descending), j2 = 2j"""
    return list(range(j2, -j2 - 1, -2))


def poly_pow_lin(cx, cy, n):
    """(cx x + cy y)^n as dict {power of x: coeff}"""
    from math import comb
    res = {}
    px = [ONE]
    for _ in range(n): px.append(px[-1] * cx)
    py = [ONE]
    for _ in range(n): py.append(py[-1] * cy)
    for k in range(n + 1):
        res[k] = K(comb(n, k)) * px[k] * py[n - k]
    return res


def Dw(U, j2):
    """matrix of D^j(U) in the monomial (weighted) basis e'_m = x^{j+m} y^{j-m};
    (D(U) f)(v) = f(U^T v); columns indexed by m' (descending m order)."""
    mlist = ms(j2)
    n = j2 + 1
    M = [[ZERO] * n for _ in range(n)]
    # U^T v = (U00 x + U10 y, U01 x + U11 y)
    for col, m2 in enumerate(mlist):
        A = (j2 + m2) // 2; B = (j2 - m2) // 2
        p1 = poly_pow_lin(U[0][0], U[1][0], A)
        p2 = poly_pow_lin(U[0][1], U[1][1], B)
        for k1, c1 in p1.items():
            for k2, c2 in p2.items():
                xp = k1 + k2  # power of x = j + m
                row = mlist.index(2 * xp - j2)
                M[row][col] = M[row][col] + c1 * c2
    return M


def matmul(A, B):
    n = len(A); m = len(B[0]); l = len(B)
    return [[sum((A[i][k] * B[k][j] for k in range(l)), ZERO) for j in range(m)] for i in range(n)]


def fact(n):
    from math import factorial
    return factorial(n)


def s2(j2, m2):
    """s_m^2 = (j+m)!(j-m)!"""
    return fact((j2 + m2) // 2) * fact((j2 - m2) // 2)


def load_inputs():
    with open(os.path.join(HERE, 'inputs.json')) as f:
        return json.load(f)


def save_json(name, obj):
    with open(os.path.join(HERE, name), 'w') as f:
        json.dump(obj, f, indent=1, default=str)


def load_json(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)
