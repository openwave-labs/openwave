"""Shared exact and numeric machinery.

Conventions (stated in RETURN.md):
  * quaternion q = w + x i + y j + z k  ->  U(q) = w*I - i(x sx + y sy + z sz),
    i.e. U = [[w - i z, -y - i x], [y - i x, w + i z]]; this is a homomorphism
    (i -> -i sx, j -> -i sy, k -> -i sz, and (-i sx)(-i sy) = -i sz).
  * V_j = Sym^{2j} C^2 realised as homogeneous polynomials in (x, y), U acting by
    substitution of the column action U e1 = U11 e1 + U21 e2, U e2 = U12 e1 + U22 e2.
    Monomial basis mono_m = x^{j+m} y^{j-m}; Fischer inner product <x^a y^b, x^a y^b> = a! b!,
    so v_m = mono_m / sqrt(f_m), f_m = (j+m)! (j-m)!, is the Condon-Shortley basis
    (J_+ has positive matrix elements, J_z = m).  w = sum w_m v_m = sum p_m mono_m with
    p_m = w_m / sqrt(f_m)  (the weighted coordinates of worklist 1.2).
  * M(U): matrix on weighted coordinates p (p -> M p).  D^j_CS[m', m] = M[m', m] sqrt(f_m'/f_m).
"""
from fractions import Fraction as Fr
from math import factorial
import sympy as sp

# ---------------------------------------------------------------- Q(sqrt5)
class Q5:
    """a + b sqrt5, a, b rational (exact)."""
    __slots__ = ("a", "b")
    def __init__(self, a=0, b=0):
        self.a = Fr(a); self.b = Fr(b)
    def __add__(s, o):
        o = _q5(o); return Q5(s.a + o.a, s.b + o.b)
    __radd__ = __add__
    def __neg__(s): return Q5(-s.a, -s.b)
    def __sub__(s, o):
        o = _q5(o); return Q5(s.a - o.a, s.b - o.b)
    def __rsub__(s, o): return _q5(o) - s
    def __mul__(s, o):
        o = _q5(o); return Q5(s.a * o.a + 5 * s.b * o.b, s.a * o.b + s.b * o.a)
    __rmul__ = __mul__
    def conj5(s): return Q5(s.a, -s.b)
    def inv(s):
        n = s.a * s.a - 5 * s.b * s.b
        return Q5(s.a / n, -s.b / n)
    def __truediv__(s, o): return s * _q5(o).inv()
    def __eq__(s, o):
        o = _q5(o); return s.a == o.a and s.b == o.b
    def __hash__(s): return hash((s.a, s.b))
    def iszero(s): return s.a == 0 and s.b == 0
    def sym(s): return sp.Rational(s.a.numerator, s.a.denominator) + sp.Rational(s.b.numerator, s.b.denominator) * sp.sqrt(5)
    def __float__(s): return float(s.a) + float(s.b) * 5 ** 0.5
    def mpf(s):
        import mpmath
        return mpmath.mpf(s.a.numerator) / s.a.denominator + mpmath.mpf(s.b.numerator) / s.b.denominator * mpmath.sqrt(5)
    def __repr__(s): return str(sp.nsimplify(s.sym()))

def _q5(o):
    return o if isinstance(o, Q5) else Q5(o, 0)

class C5:
    """re + i im with re, im in Q(sqrt5)."""
    __slots__ = ("re", "im")
    def __init__(self, re=0, im=0):
        self.re = _q5(re); self.im = _q5(im)
    def __add__(s, o):
        o = _c5(o); return C5(s.re + o.re, s.im + o.im)
    __radd__ = __add__
    def __neg__(s): return C5(-s.re, -s.im)
    def __sub__(s, o):
        o = _c5(o); return C5(s.re - o.re, s.im - o.im)
    def __mul__(s, o):
        o = _c5(o); return C5(s.re * o.re - s.im * o.im, s.re * o.im + s.im * o.re)
    __rmul__ = __mul__
    def conj(s): return C5(s.re, -s.im)
    def __eq__(s, o):
        o = _c5(o); return s.re == o.re and s.im == o.im
    def iszero(s): return s.re.iszero() and s.im.iszero()
    def sym(s): return s.re.sym() + sp.I * s.im.sym()
    def mpc(s):
        import mpmath
        return mpmath.mpc(s.re.mpf(), s.im.mpf())

def _c5(o):
    return o if isinstance(o, C5) else C5(o, 0)

# ---------------------------------------------------------------- quaternions
def qmul(p, q):
    a1, b1, c1, d1 = p; a2, b2, c2, d2 = q
    return (a1*a2 - b1*b2 - c1*c2 - d1*d2,
            a1*b2 + b1*a2 + c1*d2 - d1*c2,
            a1*c2 - b1*d2 + c1*a2 + d1*b2,
            a1*d2 + b1*c2 - c1*b2 + d1*a2)

def qconj(q):
    return (q[0], -q[1], -q[2], -q[3])

def qnorm2(q):
    return q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]

def qkey(q):
    return tuple((c.a, c.b) for c in q)

PHI = Q5(Fr(1, 2), Fr(1, 2))
PHI_INV = Q5(Fr(-1, 2), Fr(1, 2))   # 1/phi = phi - 1
HALF = Fr(1, 2)
q1 = (Q5(HALF), Q5(HALF), Q5(HALF), Q5(HALF))
q2 = (PHI * HALF, PHI_INV * HALF, Q5(HALF), Q5(0))

def su2(q, one=None):
    """U(q) = [[w - i z, -y - i x], [y - i x, w + i z]] with entries C5."""
    w, x, y, z = q
    return [[C5(w, -z), C5(-y, -x)], [C5(y, -x), C5(w, z)]]

def generate_group(gens):
    one = (Q5(1), Q5(0), Q5(0), Q5(0))
    elems = {qkey(one): one}
    frontier = [one]
    while frontier:
        new = []
        for g in frontier:
            for h in gens:
                k = qmul(g, h)
                kk = qkey(k)
                if kk not in elems:
                    elems[kk] = k; new.append(k)
        frontier = new
    return list(elems.values())

def conj_classes(G):
    keys = {qkey(g): i for i, g in enumerate(G)}
    seen = set(); classes = []
    for g in G:
        if qkey(g) in seen: continue
        cl = set()
        for h in G:
            c = qmul(qmul(h, g), qconj(h))
            cl.add(qkey(c))
        seen |= cl
        classes.append(sorted(cl))
    return classes

def cheb_U(n, c):
    """chi_{V_{n/2}}(g) = U_n(c), c = Re q (exact Q5)."""
    if n == 0: return Q5(1)
    u0, u1 = Q5(1), 2 * c
    for _ in range(n - 1):
        u0, u1 = u1, 2 * c * u1 - u0
    return u1

# ---------------------------------------------------------------- spin-j matrices
def mvals(j2):
    """m from +j down to -j, j2 = 2j; returns list of Fractions."""
    return [Fr(j2 - 2 * k, 2) for k in range(j2 + 1)]

def fm(j2, m):
    j = Fr(j2, 2)
    return factorial(int(j + m)) * factorial(int(j - m))

def poly_mul(a, b, zero):
    r = [zero for _ in range(len(a) + len(b) - 1)]
    for i, x in enumerate(a):
        for k, y in enumerate(b):
            r[i + k] = r[i + k] + x * y
    return r

def mono_matrix(U, j2, zero, one):
    """Matrix of U on weighted coordinates of V_{j2/2}; index 0 <-> m=+j (x^{2j}).
    U e1 = U11 e1 + U21 e2 ; x -> U11 x + U21 y ; y -> U12 x + U22 y.
    Polynomials in x,y of degree n represented by coefficient lists indexed by power of y."""
    n = j2
    X = [U[0][0], U[1][0]]   # x -> U11 x + U21 y : coeff of y^0, y^1
    Y = [U[0][1], U[1][1]]
    cols = []
    for k in range(n + 1):           # basis mono with y-power k  (m = j - k)
        a, b = n - k, k
        poly = [one]
        for _ in range(a): poly = poly_mul(poly, X, zero)
        for _ in range(b): poly = poly_mul(poly, Y, zero)
        cols.append(poly)
    # M[k', k] = coefficient of y^{k'} in image of basis k
    return [[cols[k][kp] for k in range(n + 1)] for kp in range(n + 1)]

# ---------------------------------------------------------------- Clebsch-Gordan (Racah formula, exact)
def cg(j1, m1, j2, m2, J, M):
    """<j1 m1; j2 m2 | J M>, Condon-Shortley, sympy exact. Arguments are sympy Rationals or ints."""
    j1, m1, j2, m2, J, M = [_rat(t) for t in (j1, m1, j2, m2, J, M)]
    if m1 + m2 != M: return sp.Integer(0)
    if abs(m1) > j1 or abs(m2) > j2 or abs(M) > J: return sp.Integer(0)
    if J < abs(j1 - j2) or J > j1 + j2: return sp.Integer(0)
    F = sp.factorial
    pre = sp.sqrt((2 * J + 1) * F(J + j1 - j2) * F(J - j1 + j2) * F(j1 + j2 - J) / F(j1 + j2 + J + 1))
    pre *= sp.sqrt(F(J + M) * F(J - M) * F(j1 - m1) * F(j1 + m1) * F(j2 - m2) * F(j2 + m2))
    s = sp.Integer(0)
    for k in range(0, int(j1 + j2 + J) + 2):
        d = [k, j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k]
        if any(t < 0 for t in d): continue
        den = sp.Integer(1)
        for t in d: den *= F(t)
        s += sp.Integer(-1) ** k / den
    return pre * s

def _rat(t):
    if isinstance(t, Fr): return sp.Rational(t.numerator, t.denominator)
    return sp.sympify(t)

# ---------------------------------------------------------------- PASS/FAIL checks
class Checker:
    def __init__(self):
        self.n_pass = 0; self.n_fail = 0
    def __call__(self, label, cond, quiet=False):
        if cond:
            self.n_pass += 1
            if not quiet: print("PASS", label)
        else:
            self.n_fail += 1
            print("FAIL", label)
        return bool(cond)
    def summary(self):
        print(f"checks: {self.n_pass} PASS, {self.n_fail} FAIL")

def planted(label, check_fn):
    """Run a check on deliberately corrupted data; it must come out False."""
    ok = check_fn()
    print(("PLANTED-DEFECT-FIRED " if not ok else "PLANTED-DEFECT-MISSED ") + label)
    return not ok
