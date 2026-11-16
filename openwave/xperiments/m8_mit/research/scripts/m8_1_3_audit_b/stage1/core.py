"""Core exact machinery, derived in this room from the two generators.

Conventions (stated once, used everywhere):
  * quaternion q = w + x i + y j + z k  ->  U(q) = [[w - i z, -y - i x], [y - i x, w + i z]]
    (i -> -i sigma_x, j -> -i sigma_y, k -> -i sigma_z; a homomorphism).  U = [[a, -conj b], [b, conj a]]
    with a = w - i z, b = y - i x.
  * V_j = homogeneous polynomials of degree 2j in (X, Y); D(U) f (X,Y) = f(a X + b Y, -conj(b) X + conj(a) Y)
    (= Sym^{2j} U, a homomorphism).  Monomial basis e_m = X^{j+m} Y^{j-m}; orthonormal (Bargmann) basis
    v_m = e_m / sqrt(F_m), F_m = (j+m)!(j-m)!.  This basis is Condon-Shortley (J+ = X d/dY has positive
    matrix elements) and R_z(theta) = U(exp(theta k/2)) acts by e^{-i m theta}.
  * Weighted coordinates p_m = w_m / sqrt(F_m) are exactly the monomial coefficients.
Fields: Q5 = Q(sqrt5) as pairs of Fractions; K = Q(sqrt5, i) as 4-tuples (re, re*sqrt5, im, im*sqrt5).
"""
from fractions import Fraction as Fr
from math import factorial
import itertools


class Q5:
    __slots__ = ('a', 'b')

    def __init__(s, a=0, b=0):
        s.a = Fr(a); s.b = Fr(b)

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

    def __eq__(s, o):
        o = q5(o); return s.a == o.a and s.b == o.b

    def __hash__(s):
        return hash((s.a, s.b))

    def iszero(s):
        return s.a == 0 and s.b == 0

    def __float__(s):
        return float(s.a) + float(s.b) * 5 ** 0.5

    def mp(s, ctx):
        return ctx.mpf(s.a.numerator) / s.a.denominator + ctx.mpf(s.b.numerator) / s.b.denominator * ctx.sqrt(5)

    def __repr__(s):
        if s.b == 0:
            return str(s.a)
        return f"({s.a}+{s.b}*sqrt5)"


def q5(x):
    return x if isinstance(x, Q5) else Q5(x, 0)


PHI = Q5(Fr(1, 2), Fr(1, 2))
PHI_INV = Q5(Fr(-1, 2), Fr(1, 2))
HALF = Q5(Fr(1, 2))


def qmul(p, q):
    w1, x1, y1, z1 = p; w2, x2, y2, z2 = q
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def qconj(q):
    return (q[0], -q[1], -q[2], -q[3])


def qkey(q):
    return tuple((c.a, c.b) for c in q)


Q1 = (HALF, HALF, HALF, HALF)
Q2 = (PHI * HALF, PHI_INV * HALF, HALF, Q5(0))
ONE = (Q5(1), Q5(0), Q5(0), Q5(0))


def generate_group(gens=(Q1, Q2)):
    elems = {qkey(ONE): ONE}
    frontier = [ONE]
    while frontier:
        new = []
        for g in frontier:
            for h in gens:
                p = qmul(g, h)
                k = qkey(p)
                if k not in elems:
                    elems[k] = p; new.append(p)
        frontier = new
    return list(elems.values())


def conj_classes(G):
    idx = {qkey(g): n for n, g in enumerate(G)}
    seen = set(); classes = []
    for g in G:
        if qkey(g) in seen:
            continue
        cl = {}
        for x in G:
            c = qmul(qmul(x, g), qconj(x))
            cl[qkey(c)] = c
        for k in cl:
            seen.add(k)
        classes.append(list(cl.values()))
    return classes, idx


def element_order(g):
    p = g; n = 1
    while qkey(p) != qkey(ONE):
        p = qmul(p, g); n += 1
    return n


def chebU(n, c):
    """U_n(c) exactly (character of V_{n/2} at an element with real part c)."""
    u0, u1 = Q5(1), 2 * c
    if n == 0:
        return u0
    for _ in range(n - 1):
        u0, u1 = u1, 2 * c * u1 - u0
    return u1


# ---------------------------------------------------------------- K = Q(sqrt5, i), as tuples of Fractions
def K(re=Q5(0), im=Q5(0)):
    re = q5(re); im = q5(im)
    return (re.a, re.b, im.a, im.b)


def kmul(x, y):
    a, b, c, d = x; e, f, g, h = y
    return (a * e + 5 * b * f - c * g - 5 * d * h, a * f + b * e - c * h - d * g,
            a * g + 5 * b * h + c * e + 5 * d * f, a * h + b * g + c * f + d * e)


def kadd(x, y):
    return (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3])


def kconj(x):
    return (x[0], x[1], -x[2], -x[3])


def kscale(x, r):
    return (x[0] * r, x[1] * r, x[2] * r, x[3] * r)


KZERO = (Fr(0), Fr(0), Fr(0), Fr(0))
KONE = (Fr(1), Fr(0), Fr(0), Fr(0))


def kiszero(x):
    return all(v == 0 for v in x)


def k_to_complex(x, ctx):
    s5 = ctx.sqrt(5)
    re = ctx.mpf(x[0].numerator) / x[0].denominator + ctx.mpf(x[1].numerator) / x[1].denominator * s5
    im = ctx.mpf(x[2].numerator) / x[2].denominator + ctx.mpf(x[3].numerator) / x[3].denominator * s5
    return re, im


def U_of(q):
    """(a, b) of U(q) = [[a, -conj b], [b, conj a]], as K numbers."""
    w, x, y, z = q
    a = K(w, -z)
    b = K(y, -x)
    return a, b


def mlist(j2):
    """weights m (as Fractions) from +j down to -j, for 2j = j2."""
    j = Fr(j2, 2)
    return [j - k for k in range(j2 + 1)]


def Fm(j2, m):
    j = Fr(j2, 2)
    return factorial(int(j + m)) * factorial(int(j - m))


def kpow(x, n):
    r = KONE
    for _ in range(n):
        r = kmul(r, x)
    return r


def binom_poly(c1, c2, n):
    """coefficients of (c1 X + c2 Y)^n: list index t -> coefficient of X^t Y^(n-t)."""
    from math import comb
    return [kscale(kmul(kpow(c1, t), kpow(c2, n - t)), comb(n, t)) for t in range(n + 1)]


def D_mono(q, j2):
    """matrix D[m_index, m'_index] in the monomial basis e_m = X^{j+m} Y^{j-m}, index 0 <-> m = +j."""
    a, b = U_of(q)
    nb = tuple(-v for v in kconj(b))  # -conj(b)
    ca = kconj(a)
    n = j2 + 1
    M = [[KZERO] * n for _ in range(n)]
    for ip in range(n):          # m' index; e_{m'} = X^{j2-ip} Y^{ip}
        alpha = j2 - ip; beta = ip
        A = binom_poly(a, b, alpha)    # (aX + bY)^alpha
        B = binom_poly(nb, ca, beta)   # (-b* X + a* Y)^beta
        for t1, c1 in enumerate(A):
            for t2, c2 in enumerate(B):
                deg = t1 + t2           # power of X
                i = j2 - deg            # row index: X^{j2-i}
                M[i][ip] = kadd(M[i][ip], kmul(c1, c2))
    return M


def kmatmul(A, B):
    n = len(A); m = len(B[0]); l = len(B)
    C = [[KZERO] * m for _ in range(n)]
    for i in range(n):
        for k in range(l):
            if kiszero(A[i][k]):
                continue
            for jj in range(m):
                if not kiszero(B[k][jj]):
                    C[i][jj] = kadd(C[i][jj], kmul(A[i][k], B[k][jj]))
    return C


# ---------------------------------------------------------------- character table by Burnside-Dixon (numerical) + exact identification
def character_table(G, classes, idx):
    import numpy as np
    nC = len(classes)
    cls_of = {}
    for r, cl in enumerate(classes):
        for g in cl:
            cls_of[qkey(g)] = r
    reps = [cl[0] for cl in classes]
    # a[r][s][t] = #{x in C_r : x^{-1} g_t in C_s}
    a = np.zeros((nC, nC, nC))
    for r, cl in enumerate(classes):
        for t, gt in enumerate(reps):
            for x in cl:
                s = cls_of[qkey(qmul(qconj(x), gt))]
                a[r, s, t] += 1
    rng = np.random.default_rng(12345)
    coeff = rng.normal(size=nC)
    M = sum(coeff[r] * a[r] for r in range(nC))   # M[s,t]
    # omega vector (indexed by class) satisfies sum_t a[r,s,t] omega_t = omega_r omega_s
    vals, vecs = np.linalg.eig(M)
    sizes = np.array([len(cl) for cl in classes], float)
    chars = []
    id_cls = cls_of[qkey(ONE)]
    for k in range(nC):
        om = vecs[:, k] / vecs[id_cls, k]
        deg2 = len(G) / np.sum(np.abs(om) ** 2 / sizes)
        deg = np.sqrt(deg2.real)
        chi = om * deg / sizes
        chars.append(chi)
    return chars, cls_of, a


def identify_zphi(x, bound=12, tol=1e-8):
    phi = (1 + 5 ** 0.5) / 2
    best = None
    for A in range(-bound, bound + 1):
        for B in range(-bound, bound + 1):
            v = A + B * phi
            if abs(x.real - v) < tol and abs(x.imag) < tol:
                if best is not None:
                    raise ValueError('ambiguous')
                best = Q5(A, 0) + Q5(B, 0) * PHI
    if best is None:
        raise ValueError(f'no Z[phi] identification for {x}')
    return best
