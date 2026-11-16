"""Exact route on S^3 = SU(2): functions are polynomials in (a, conj a, b, conj b) with U = [[a, -conj b],[b, conj a]];
coefficients in K = Q(sqrt5, i) held as integer 4-tuples over one common denominator per polynomial.
Haar probability moments: E[a^P conj(a)^P b^R conj(b)^R] = P! R! / (P+R+1)!, all other monomial moments vanish.
Level-L parts are the restrictions of harmonic polynomials of degree L (eigenvalue L(L+2) of -Laplacian on S^3);
a homogeneous f of degree d is split as f = sum_k |x|^{2k} H_{d-2k} through the exact triangular system
<Lap^i f, Lap^i g>_{S^3} = sum_{k>=i} c_{k,i}^2 <H^f_{d-2k}, H^g_{d-2k}>,  Lap(|x|^{2k} H_m) = 2k(2k+2m+2)|x|^{2k-2} H_m.
"""
from fractions import Fraction as Fr
from math import factorial, gcd, comb
from core import *
from reps import *

FACT = [factorial(n) for n in range(80)]


def _kmul_i(x, y):
    a, b, c, d = x; e, f, g, h = y
    return (a * e + 5 * b * f - c * g - 5 * d * h, a * f + b * e - c * h - d * g,
            a * g + 5 * b * h + c * e + 5 * d * f, a * h + b * g + c * f + d * e)


class Poly:
    __slots__ = ('c', 'den', 'deg')

    def __init__(s, c, den, deg):
        s.c = c; s.den = den; s.deg = deg

    @staticmethod
    def from_frac(cf, deg):
        """cf: dict mono -> K tuple of Fractions"""
        den = 1
        for v in cf.values():
            for x in v:
                den = den * x.denominator // gcd(den, x.denominator)
        c = {}
        for m, v in cf.items():
            t = tuple(int(x * den) for x in v)
            if any(t):
                c[m] = t
        return Poly(c, den, deg)

    def __mul__(s, o):
        c = {}
        for m1, v1 in s.c.items():
            for m2, v2 in o.c.items():
                m = (m1[0] + m2[0], m1[1] + m2[1], m1[2] + m2[2], m1[3] + m2[3])
                p = _kmul_i(v1, v2)
                if m in c:
                    q = c[m]; c[m] = (q[0] + p[0], q[1] + p[1], q[2] + p[2], q[3] + p[3])
                else:
                    c[m] = p
        return Poly(c, s.den * o.den, s.deg + o.deg)

    def add(s, o, so=1, oo=1):
        """so*s + oo*o with Fractions so, oo"""
        so = Fr(so); oo = Fr(oo)
        assert s.deg == o.deg or not s.c or not o.c
        den = s.den * so.denominator * o.den * oo.denominator
        f1 = so.numerator * o.den * oo.denominator
        f2 = oo.numerator * s.den * so.denominator
        c = {}
        for m, v in s.c.items():
            c[m] = tuple(x * f1 for x in v)
        for m, v in o.c.items():
            w = tuple(x * f2 for x in v)
            if m in c:
                q = c[m]; c[m] = tuple(q[i] + w[i] for i in range(4))
            else:
                c[m] = w
        c = {m: v for m, v in c.items() if any(v)}
        return Poly(c, den, max(s.deg, o.deg)).reduce()

    def kscale(s, k):
        """multiply by a K number given as Fraction tuple"""
        P = Poly.from_frac({(0, 0, 0, 0): k}, 0)
        return (s * P).reduce()

    def conj(s):
        return Poly({(m[1], m[0], m[3], m[2]): (v[0], v[1], -v[2], -v[3]) for m, v in s.c.items()}, s.den, s.deg)

    def reduce(s):
        g = s.den
        for v in s.c.values():
            for x in v:
                g = gcd(g, x)
                if g == 1:
                    return s
        if g > 1:
            return Poly({m: tuple(x // g for x in v) for m, v in s.c.items()}, s.den // g, s.deg)
        return s

    def lap(s):
        c = {}
        for (p, q, r, t), v in s.c.items():
            if p and q:
                m = (p - 1, q - 1, r, t); f = 4 * p * q
                w = tuple(x * f for x in v)
                if m in c:
                    u = c[m]; c[m] = tuple(u[i] + w[i] for i in range(4))
                else:
                    c[m] = w
            if r and t:
                m = (p, q, r - 1, t - 1); f = 4 * r * t
                w = tuple(x * f for x in v)
                if m in c:
                    u = c[m]; c[m] = tuple(u[i] + w[i] for i in range(4))
                else:
                    c[m] = w
        c = {m: v for m, v in c.items() if any(v)}
        return Poly(c, s.den, s.deg - 2).reduce()


def sphere_inner(f, g):
    """E_{S^3}[f conj g], exact: returns K tuple of Fractions."""
    bych = {}
    for m, v in g.c.items():
        bych.setdefault((m[0] - m[1], m[2] - m[3]), []).append((m, v))
    acc = [0, 0, 0, 0]
    for m1, v1 in f.c.items():
        lst = bych.get((m1[0] - m1[1], m1[2] - m1[3]))
        if not lst:
            continue
        for m2, v2 in lst:
            P = m1[0] + m2[1]; R = m1[2] + m2[3]
            w = FACT[P] * FACT[R]
            # v1 * conj(v2)
            a, b, c, d = v1; e, ff, gg, h = v2[0], v2[1], -v2[2], -v2[3]
            acc[0] += w * (a * e + 5 * b * ff - c * gg - 5 * d * h)
            acc[1] += w * (a * ff + b * e - c * h - d * gg)
            acc[2] += w * (a * gg + 5 * b * h + c * e + 5 * d * ff)
            acc[3] += w * (a * h + b * gg + c * ff + d * e)
    # the total degree of every product monomial pair is f.deg + g.deg = 2 (P+R) when nonzero; P+R = (f.deg+g.deg)/2
    if f.deg != g.deg:
        # different homogeneous degrees: still fine, P+R computed per pair; recompute exactly
        return _sphere_inner_general(f, g)
    den = FACT[f.deg + 1] * f.den * g.den
    return tuple(Fr(x, den) for x in acc)


def _sphere_inner_general(f, g):
    acc = [Fr(0)] * 4
    for m1, v1 in f.c.items():
        for m2, v2 in g.c.items():
            if m1[0] - m1[1] != m2[0] - m2[1] or m1[2] - m1[3] != m2[2] - m2[3]:
                continue
            P = m1[0] + m2[1]; R = m1[2] + m2[3]
            w = Fr(FACT[P] * FACT[R], FACT[P + R + 1])
            p = _kmul_i(v1, (v2[0], v2[1], -v2[2], -v2[3]))
            for i in range(4):
                acc[i] += w * p[i]
    den = f.den * g.den
    return tuple(x / den for x in acc)


def level_inners(f, g):
    """dict L -> <Pi_L f, Pi_L g> (exact K tuples), f and g homogeneous of the same degree d."""
    d = f.deg
    assert g.deg == d
    Kmax = d // 2
    Fs = [f]; Gs = [g]
    for i in range(Kmax):
        Fs.append(Fs[-1].lap()); Gs.append(Gs[-1].lap())
    I = [sphere_inner(Fs[i], Gs[i]) for i in range(Kmax + 1)]

    def c(k, i):
        r = 1
        for t in range(i):
            kk = k - t
            r *= 2 * kk * (2 * kk + 2 * (d - 2 * k) + 2)
        return r
    X = {}
    for i in range(Kmax, -1, -1):
        acc = list(I[i])
        for k in range(i + 1, Kmax + 1):
            cc = c(k, i) ** 2
            acc = [acc[t] - cc * X[k][t] for t in range(4)]
        cii = c(i, i) ** 2
        X[i] = tuple(Fr(x) / cii for x in acc)
    return {d - 2 * k: X[k] for k in range(Kmax + 1)}


# ---------------------------------------------------------------- the slot sections
_DPOLY = {}


def D_poly(j2):
    """D_mono entries as polynomials in (a, conj a, b, conj b): returns [[Poly]] (integer coefficients)."""
    if j2 in _DPOLY:
        return _DPOLY[j2]
    n = j2 + 1
    # (a X + b Y)^alpha (-conj(b) X + conj(a) Y)^beta ; monomial key (p,q,r,s) for a^p abar^q b^r bbar^s
    M = [[dict() for _ in range(n)] for _ in range(n)]
    for ip in range(n):
        alpha = j2 - ip; beta = ip
        for t1 in range(alpha + 1):          # X^t1 from first factor: a^t1 b^(alpha-t1)
            c1 = comb(alpha, t1)
            for t2 in range(beta + 1):       # X^t2 from second: (-bbar)^t2 abar^(beta-t2)
                c2 = comb(beta, t2) * (-1) ** t2
                i = j2 - (t1 + t2)
                mono = (t1, beta - t2, alpha - t1, t2)
                M[i][ip][mono] = M[i][ip].get(mono, 0) + c1 * c2
    res = [[Poly({m: (v, 0, 0, 0) for m, v in M[i][k].items() if v}, 1, j2) for k in range(n)] for i in range(n)]
    _DPOLY[j2] = res
    return res


def Phi_polys(name, p):
    """Phi'_k(g) = sum_{m,m''} p_m F_m D[m,m''](g) P[m'',k], k = row index; p: list of K tuples (Fractions), index 0 <-> m=+j.
    Then sum_k |Phi_k|^2 = sum_k |Phi'_k|^2 / F_k with Phi the orthonormal-basis row (= phi eta^dagger)."""
    d, n = SLOTS[name]
    j2 = n
    N = n + 1
    Dp = D_poly(j2)
    ms = mlist(j2)
    P = projector_mono(name)
    # c_{m''} = sum_m p_m F_m D[m,m'']
    cs = []
    for mm in range(N):
        acc = None
        for mi in range(N):
            if kiszero(p[mi]):
                continue
            term = Dp[mi][mm].kscale(kscale(p[mi], Fm(j2, ms[mi])))
            acc = term if acc is None else acc.add(term)
        cs.append(acc)
    out = []
    for k in range(N):
        acc = None
        for mm in range(N):
            if cs[mm] is None or kiszero(P[mm][k]):
                continue
            term = cs[mm].kscale(P[mm][k])
            acc = term if acc is None else acc.add(term)
        out.append(acc if acc is not None else Poly({}, 1, j2))
    return out


def modsq(name, A, B):
    """sum_k A_k conj(B_k) / F_k"""
    d, n = SLOTS[name]
    ms = mlist(n)
    acc = None
    for k in range(n + 1):
        if not A[k].c or not B[k].c:
            continue
        t = (A[k] * B[k].conj())
        t = t.add(Poly({}, 1, t.deg), Fr(1, Fm(n, ms[k])), 0)
        acc = t if acc is None else acc.add(t)
    return acc


def norm_inner_vec(name, A, B, levels=True):
    """sum_k <A_k, B_k>/F_k, split by level if levels else total."""
    d, n = SLOTS[name]
    ms = mlist(n)
    tot = {}
    for k in range(n + 1):
        if not A[k].c or not B[k].c:
            continue
        li = level_inners(A[k], B[k]) if levels else {None: sphere_inner(A[k], B[k])}
        w = Fr(1, Fm(n, ms[k]))
        for L, v in li.items():
            cur = tot.get(L, KZERO)
            tot[L] = kadd(cur, kscale(v, w))
    return tot


def wnorm2(name, p):
    d, n = SLOTS[name]
    ms = mlist(n)
    s = Fr(0)
    for i, m in enumerate(ms):
        v = p[i]
        # |p|^2 with p in K: (re)^2 + (im)^2 must be rational for our inputs; compute exactly in Q5
        re = Q5(v[0], v[1]); im = Q5(v[2], v[3])
        q = re * re + im * im
        assert q.b == 0
        s += q.a * Fm(n, m)
    return s


def h_exact(name, p):
    """h(p) = s^-2 E|Phi|^4 (Haar probability measure, V = 1), exact (Fraction, after checking the sqrt5 and i parts vanish)."""
    Ph = Phi_polys(name, p)
    m2 = modsq(name, Ph, Ph)
    v = sphere_inner(m2, m2)
    assert v[1] == 0 and v[2] == 0 and v[3] == 0, v
    return v[0] / s_N(name) ** 2


def N_vec(name, A, B, C):
    """T(A,B,C)_k = (sum_l A_l conj B_l / F_l) C_k  (unnormalized: multiply norms by s^-3)."""
    m = modsq(name, A, B)
    return [m * C[k] if C[k].c else Poly({}, 1, m.deg + C[k].deg) for k in range(len(C))]


def lam(L):
    return L * (L + 2)


def R0_levels(name, p):
    """level norms ||Pi_L N(psi_y)||^2 (V=1, psi = s^-1/2 phi) for y with weighted coords p (not normalized)."""
    Ph = Phi_polys(name, p)
    Nv = N_vec(name, Ph, Ph, Ph)
    tot = norm_inner_vec(name, Nv, Nv)
    s = s_N(name)
    return {L: kscale(v, 1 / s ** 3) for L, v in tot.items()}


def R0_from_levels(name, lv):
    d, n = SLOTS[name]
    acc = KZERO
    for L, v in lv.items():
        if L > n:
            acc = kadd(acc, kscale(v, Fr(-1, lam(L) - lam(n))))
    return acc


def vecadd(A, B):
    return [A[k].add(B[k]) if (A[k].c and B[k].c) else (A[k] if A[k].c else B[k]) for k in range(len(A))]


def R0_hessian_entry(name, p_u, p_v1, p_v2):
    """exact Hess R0(u)[v1, v2] for directions given in weighted coordinates (complex vectors; real directions of E0).
    Uses N(u + s v1 + t v2): coefficient of s t of -sum_L w_L ||Pi_L N||^2 (L above lowest level)."""
    d, n = SLOTS[name]
    U = Phi_polys(name, p_u); A = Phi_polys(name, p_v1); B = Phi_polys(name, p_v2)
    T = lambda x, y, z: N_vec(name, x, y, z)
    N0 = T(U, U, U)
    Ns = vecadd(vecadd(T(A, U, U), T(U, A, U)), T(U, U, A))
    Nt = vecadd(vecadd(T(B, U, U), T(U, B, U)), T(U, U, B))
    Nst = T(A, B, U)
    for x in (T(B, A, U), T(A, U, B), T(B, U, A), T(U, A, B), T(U, B, A)):
        Nst = vecadd(Nst, x)
    I1 = norm_inner_vec(name, Ns, Nt)
    I2 = norm_inner_vec(name, N0, Nst)
    s = s_N(name)
    acc = Fr(0); acc5 = Fr(0)
    for L in set(I1) | set(I2):
        if L <= n:
            continue
        v1 = I1.get(L, KZERO); v2 = I2.get(L, KZERO)
        # 2 Re <Ns,Nt> + 2 Re <N0,Nst>
        acc += -Fr(1, lam(L) - lam(n)) * 2 * (v1[0] + v2[0])
        acc5 += -Fr(1, lam(L) - lam(n)) * 2 * (v1[1] + v2[1])
    return acc / s ** 3, acc5 / s ** 3


def kfrac(re, im=0):
    return (Fr(re), Fr(0), Fr(im), Fr(0))
