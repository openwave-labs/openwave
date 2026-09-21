"""Shared exact and high-precision machinery for the spin-3 worklist.

Basis order everywhere: index i = 3 - m, i.e. (v3, v2, v1, v0, v-1, v-2, v-3).
Real coordinates of u in C^7: x = (Re c_3..Re c_-3, Im c_3..Im c_-3), so that
Re<u,w> is the Euclidean dot product of the real coordinate vectors.
"""
import os, json, itertools
import sympy as sp
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)

J = 3
MS = list(range(J, -J - 1, -1))          # 3,2,...,-3
def idx(m): return J - m
I = sp.I

# ---------------------------------------------------------------- operators
def jp_elem(m):   # <m+1|J+|m>
    return sp.sqrt(12 - m * (m + 1))
def jm_elem(m):   # <m-1|J-|m>
    return sp.sqrt(12 - m * (m - 1))

def jmats():
    Jz = sp.zeros(7); Jp = sp.zeros(7); Jm = sp.zeros(7)
    for m in MS:
        Jz[idx(m), idx(m)] = m
        if m + 1 <= J: Jp[idx(m + 1), idx(m)] = jp_elem(m)
        if m - 1 >= -J: Jm[idx(m - 1), idx(m)] = jm_elem(m)
    Jx = (Jp + Jm) / 2
    Jy = (Jp - Jm) / (2 * I)
    return Jx, Jy, Jz, Jp, Jm

JX, JY, JZ, JP, JM = jmats()

def theta(c):
    """(Theta u)_m = (-1)^(3-m) conj(c_{-m}); c is a list indexed by idx."""
    return [(-1) ** (J - m) * sp.conjugate(c[idx(-m)]) for m in MS]

# ---------------------------------------------------------------- Clebsch-Gordan
def cg_lowering(j1, j2):
    """All <j1 m1; j2 m2 | L M> built from the Condon-Shortley convention only:
    |L L> is the unit vector in the M=L weight space orthogonal to all |L' L>, L'>L,
    with <j1 j1; j2 L-j1 | L L> > 0; lower states come from J- = J-(1) + J-(2).
    Returns dict (m1, m2, L, M) -> exact value."""
    # Exact rational arithmetic: work in the rescaled basis f_m = v_m / s_m with
    # s_m = sqrt((j+m)!/(j-m)!), in which J- f_m = f_{m-1} exactly and the metric is
    # diagonal and rational, <f_m, f_m> = (j-m)!/(j+m)!.  Gram-Schmidt is then rational;
    # only the final normalization introduces a square root.
    from fractions import Fraction as Fr
    from math import factorial as fa
    basis = [(a, b) for a in range(j1, -j1 - 1, -1) for b in range(j2, -j2 - 1, -1)]
    def w(j, m): return Fr(fa(j - m), fa(j + m))          # <f_m, f_m>
    def metric(a, b): return w(j1, a) * w(j2, b)
    def ip(u, v):  # real rational vectors (dict)
        return sum(u[k] * v.get(k, 0) * metric(*k) for k in u)
    def jminus(vec):   # total J- in the f-basis: f_a (x) f_b -> f_{a-1} f_b + f_a f_{b-1}
        out = {}
        for (a, b), x in vec.items():
            if a - 1 >= -j1: out[(a - 1, b)] = out.get((a - 1, b), 0) + x
            if b - 1 >= -j2: out[(a, b - 1)] = out.get((a, b - 1), 0) + x
        return out
    states = {}   # (L, M) -> (unnormalized f-basis rational vector, its squared norm)
    for L in range(j1 + j2, abs(j1 - j2) - 1, -1):
        ws = [(a, b) for (a, b) in basis if a + b == L]
        # start from f_{j1} (x) f_{L-j1} and remove components along higher-L states
        v = {(j1, L - j1): Fr(1)}
        for Lp in range(j1 + j2, L, -1):
            h, hn = states[(Lp, L)]
            c = ip(v, h) / hn
            for k in h: v[k] = v.get(k, 0) - c * h[k]
        v = {k: x for k, x in v.items() if x != 0}
        # the M=L weight space has dim (#L'>=L), so v spans the complement; fix the
        # Condon-Shortley sign <j1 j1; j2 L-j1 | L L> > 0
        if v[(j1, L - j1)] < 0: v = {k: -x for k, x in v.items()}
        states[(L, L)] = (v, ip(v, v))
        cur = v
        for M in range(L, -L, -1):
            cur = jminus(cur)
            states[(L, M - 1)] = (cur, ip(cur, cur))
    out = {}
    for (L, M), (v, nn) in states.items():
        for (a, b) in basis:
            if a + b != M: continue
            x = v.get((a, b), Fr(0))
            # coefficient on v_a (x) v_b of the unit vector = x / (s_a s_b sqrt(nn));
            # square it rationally and restore the sign.
            sq = x * x * Fr(fa(j1 - a) * fa(j2 - b), fa(j1 + a) * fa(j2 + b)) / nn
            val = sp.sqrt(sp.Rational(sq.numerator, sq.denominator))
            out[(a, b, L, M)] = -val if x < 0 else val
    for L in range(j1 + j2, abs(j1 - j2) - 1, -1):
        assert out[(j1, L - j1, L, L)] > 0, L
    return out

def cg_racah(j1, m1, j2, m2, L, M):
    """Racah's closed formula, typed in from memory (independent route)."""
    if m1 + m2 != M or abs(m1) > j1 or abs(m2) > j2 or abs(M) > L: return sp.Integer(0)
    if L < abs(j1 - j2) or L > j1 + j2: return sp.Integer(0)
    f = sp.factorial
    pre = sp.sqrt((2 * L + 1) * f(j1 + j2 - L) * f(j1 - j2 + L) * f(-j1 + j2 + L) / f(j1 + j2 + L + 1))
    pre *= sp.sqrt(f(L + M) * f(L - M) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2))
    s = 0
    for k in range(0, 40):
        d = [k, j1 + j2 - L - k, j1 - m1 - k, j2 + m2 - k, L - j2 + m1 + k, L - j1 - m2 + k]
        if min(d) < 0: continue
        s += sp.Integer(-1) ** k / sp.prod([f(x) for x in d])
    return sp.radsimp(pre * s)

_CG = None
def CG():
    global _CG
    if _CG is None:
        _CG = cg_lowering(3, 3)
    return _CG

def cg6(m1, m2, Q):
    if m1 + m2 != Q or abs(m1) > 3 or abs(m2) > 3: return sp.Integer(0)
    return CG()[(m1, m2, 6, Q)]

def cgL(m1, m2, L, Q):
    if m1 + m2 != Q or abs(m1) > 3 or abs(m2) > 3 or abs(Q) > L: return sp.Integer(0)
    return CG()[(m1, m2, L, Q)]

# ---------------------------------------------------------------- the quartic
def rhoL(c, L=6):
    th = theta(c)
    out = []
    for Q in range(L, -L - 1, -1):
        s = 0
        for m1 in MS:
            m2 = Q - m1
            if abs(m2) <= 3:
                s += cgL(m1, m2, L, Q) * c[idx(m1)] * th[idx(m2)]
        out.append(s)
    return out

def N_exact(c, L=6):
    return sp.nsimplify(sp.radsimp(sp.expand(sum(sp.expand(r * sp.conjugate(r)) for r in rhoL(c, L)))))

def norm2(c):
    return sp.expand(sum(sp.expand(x * sp.conjugate(x)) for x in c))

def rhat_exact(c, L=6):
    return sp.radsimp(sp.nsimplify(N_exact(c, L) / norm2(c) ** 2))

# ---------------------------------------------------------------- real polynomial N(x)
XS = sp.symbols('x0:14', real=True)
def cvec_from_x(xs=XS):
    return [xs[k] + I * xs[k + 7] for k in range(7)]

_NPOLY = {}
def N_poly(L=6):
    if L not in _NPOLY:
        c = cvec_from_x()
        r = rhoL(c, L)
        tot = 0
        for z in r:
            re, im = sp.expand(z).as_real_imag()
            tot += sp.expand(re ** 2 + im ** 2)
        _NPOLY[L] = sp.expand(tot)
    return _NPOLY[L]

def realify(A):
    """complex 7x7 -> real 14x14 acting on (Re, Im)."""
    R = sp.zeros(14)
    for a in range(7):
        for b in range(7):
            z = sp.expand(A[a, b]); re, im = z.as_real_imag()
            R[a, b] = re; R[a, b + 7] = -im
            R[a + 7, b] = im; R[a + 7, b + 7] = re
    return R

def c_to_x(c):
    return sp.Matrix([sp.re(z) for z in c] + [sp.im(z) for z in c])

def x_to_c(x):
    return [x[k] + I * x[k + 7] for k in range(7)]

def apply(A, c):
    v = A * sp.Matrix(c)
    return [sp.expand(v[k]) for k in range(7)]

def orbit_gens(c):
    """The four generators of O_u exactly as 1.5 writes them: i u, -i Jx u, -i Jy u, -i Jz u."""
    return [[I * z for z in c], [sp.expand(-I * z) for z in apply(JX, c)],
            [sp.expand(-I * z) for z in apply(JY, c)], [sp.expand(-I * z) for z in apply(JZ, c)]]

def exact_rot(n, cos, sin):
    """Exact D^3(n, theta) = exp(-i theta n.J) through the spectral projectors of n.J
    (eigenvalues -3..3 for a unit axis n): D = sum_k e^{-i k theta} P_k,
    P_k = prod_{l != k} (n.J - l)/(k - l).  n must be an exact unit vector."""
    assert sp.simplify(n[0] ** 2 + n[1] ** 2 + n[2] ** 2 - 1) == 0
    nJ = n[0] * JX + n[1] * JY + n[2] * JZ
    D = sp.zeros(7)
    for k in range(-3, 4):
        Pk = sp.eye(7)
        for l in range(-3, 4):
            if l != k: Pk = Pk * (nJ - l * sp.eye(7)) / (k - l)
        ph = (cos - I * sin) ** k if k >= 0 else (cos + I * sin) ** (-k)
        D += sp.expand(ph) * Pk
    return D.applyfunc(lambda z: sp.nsimplify(sp.radsimp(sp.expand(z))))

def rot_axis_angle(axis, frac):
    """exact rotation by 2*pi*frac about the (unnormalized, exact) axis."""
    nn = sp.sqrt(sum(a ** 2 for a in axis))
    n = [sp.radsimp(a / nn) for a in axis]
    th = 2 * sp.pi * sp.Rational(frac)
    return exact_rot(n, sp.cos(th), sp.sin(th))

def multipole_parts(c):
    """rho = |u><u| / |u|^2 split into its rank-L parts rho_L (L = 0..6) with the adjoint
    Casimir  Cad(X) = sum_k [J_k, [J_k, X]]  (eigenvalue L(L+1) on rank-L operators):
    rho_L = prod_{K != L} (Cad - K(K+1)) / (L(L+1) - K(K+1)) applied to rho."""
    u = sp.Matrix(c)
    rho = (u * u.H) / norm2(c)
    def cad(X):
        out = sp.zeros(7)
        for Jk in (JX, JY, JZ):
            A = Jk * X - X * Jk
            out += Jk * A - A * Jk
        return out.applyfunc(lambda z: sp.radsimp(sp.expand(z)))
    parts = {}
    for L in range(7):
        X = rho
        for K_ in range(7):
            if K_ == L: continue
            X = ((cad(X) - K_ * (K_ + 1) * X) / (L * (L + 1) - K_ * (K_ + 1))).applyfunc(lambda z: sp.radsimp(sp.expand(z)))
        parts[L] = X
    return parts

def sextic_invariants(c, triples):
    """tr(rho_L1 rho_L2 rho_L3): rotation- and phase-invariant, bidegree (3,3)."""
    P = multipole_parts(c)
    return {tr: sp.nsimplify(sp.radsimp(sp.expand((P[tr[0]] * P[tr[1]] * P[tr[2]]).trace()))) for tr in triples}

def same_span(A, B):
    """exact test that the column spaces of A and B coincide."""
    ra = A.rank(simplify=True); rb = B.rank(simplify=True)
    return ra == rb == A.row_join(B).rank(simplify=True)

# ---------------------------------------------------------------- mpmath route
def mp_cg6(m1, m2, Q):
    # stretched coupling: all L=6 coefficients from the lowering construction are
    # sqrt(C(6,3+m1) C(6,3+m2) / C(12,6+Q)); checked against the exact table in item0.
    return mp.sqrt(mp.binomial(6, 3 + m1) * mp.binomial(6, 3 + m2) / mp.binomial(12, 6 + Q))

def mp_rhat(c):
    th = [(-1) ** (J - m) * mp.conj(c[idx(-m)]) for m in MS]
    tot = mp.mpf(0)
    for Q in range(-6, 7):
        s = mp.mpc(0)
        for m1 in MS:
            m2 = Q - m1
            if abs(m2) <= 3:
                s += mp_cg6(m1, m2, Q) * c[idx(m1)] * th[idx(m2)]
        tot += abs(s) ** 2
    n2 = sum(abs(z) ** 2 for z in c)
    return tot / n2 ** 2

def mp_rot(n, ang):
    """D^3(n, ang) = exp(-i ang (n.J)) numerically."""
    A = mp.matrix(7, 7)
    for a in range(7):
        for b in range(7):
            A[a, b] = mp.mpc(0, -1) * ang * (n[0] * to_mp(JX[a, b]) + n[1] * to_mp(JY[a, b]) + n[2] * to_mp(JZ[a, b]))
    return mp.expm(A)

def to_mp(z):
    z = sp.nsimplify(z) if not isinstance(z, sp.Basic) else z
    return mp.mpc(mp.mpf(sp.re(z).evalf(mp.mp.dps + 10)), mp.mpf(sp.im(z).evalf(mp.mp.dps + 10)))

def mp_vec(c):
    return [to_mp(z) for z in c]

_CAS = {}
def mp_casimir_projector(L):
    """P_L on V3 (x) V3 from the total Casimir, no Clebsch-Gordan coefficients used:
    P_L = prod_{K != L} (C - K(K+1)) / (L(L+1) - K(K+1)), C = (J(1)+J(2))^2."""
    key = (L, mp.mp.dps)
    if key in _CAS: return _CAS[key]
    def m(A): return mp.matrix([[to_mp(A[a, b]) for b in range(7)] for a in range(7)])
    jx, jy, jz = m(JX), m(JY), m(JZ)
    one = mp.eye(7)
    def kron(A, B):
        K = mp.matrix(49, 49)
        for a in range(7):
            for b in range(7):
                if A[a, b] == 0: continue
                for c in range(7):
                    for d in range(7):
                        K[7 * a + c, 7 * b + d] = A[a, b] * B[c, d]
        return K
    Cs = mp.matrix(49, 49)
    for A in (jx, jy, jz):
        T = kron(A, one) + kron(one, A)
        Cs += T * T
    P = mp.eye(49)
    for K in range(0, 7):
        if K == L: continue
        P = P * (Cs - K * (K + 1) * mp.eye(49)) / (L * (L + 1) - K * (K + 1))
    _CAS[key] = P
    return P

def mp_rL_casimir(c, L):
    th = [(-1) ** (J - m) * mp.conj(c[idx(-m)]) for m in MS]
    x = mp.matrix([c[a] * th[b] for a in range(7) for b in range(7)])
    P = mp_casimir_projector(L)
    y = P * x
    n2 = sum(abs(z) ** 2 for z in c)
    return sum(abs(y[k]) ** 2 for k in range(49)) / n2 ** 2

def save(name, obj):
    with open(os.path.join(OUT, name + '.json'), 'w') as f:
        json.dump(obj, f, indent=1, default=str)

class Checker:
    def __init__(self, tag):
        self.tag = tag; self.fails = 0
    def check(self, name, ok, detail=''):
        print(('PASS' if ok else 'FAIL') + ' [%s] %s %s' % (self.tag, name, detail))
        if not ok: self.fails += 1
        return ok
