"""Exact polynomials in the real chart x = (Re p_j..Re p_-j, Im p_j..Im p_-j):
  h (the reduced quartic) built from the channel coefficients q_J, which are obtained EXACTLY from the S^3 route at
  the weight vectors (triangular system), and r6hat numerator ||[w (x) Theta w]_6||^2 for j = 3.
CG coefficients: own Racah-formula implementation (exact: sign and rational square), cross-checked against sympy."""
from fractions import Fraction as Fr
from math import factorial
import functools, itertools
import sympy as sp
from core import mlist, Fm, KZERO
from reps import SLOTS, s_N


def cg_exact(j1, m1, j2, m2, J, M):
    """Condon-Shortley CG <j1 m1; j2 m2 | J M> as (sign, square) with square a Fraction (Racah formula)."""
    j1, m1, j2, m2, J, M = map(Fr, (j1, m1, j2, m2, J, M))
    if m1 + m2 != M or abs(m1) > j1 or abs(m2) > j2 or abs(M) > J or J > j1 + j2 or J < abs(j1 - j2):
        return 0, Fr(0)
    f = lambda x: factorial(int(x))
    pre = Fr((2 * J + 1) * f(J + j1 - j2) * f(J - j1 + j2) * f(j1 + j2 - J), f(j1 + j2 + J + 1))
    pre *= f(J + M) * f(J - M) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2)
    s = Fr(0)
    for k in range(0, 200):
        args = [k, j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k]
        if any(a < 0 for a in args):
            continue
        den = 1
        for a in args:
            den *= f(a)
        s += Fr((-1) ** k, den)
    if s == 0:
        return 0, Fr(0)
    return (1 if s > 0 else -1), pre * s * s


def cg_sympy(j1, m1, j2, m2, J, M):
    from sympy.physics.quantum.cg import CG
    return CG(sp.Rational(j1), sp.Rational(m1), sp.Rational(j2), sp.Rational(m2), sp.Rational(J), sp.Rational(M)).doit()


def cg_sym(j1, m1, j2, m2, J, M):
    sg, sq = cg_exact(j1, m1, j2, m2, J, M)
    return sg * sp.sqrt(sp.Rational(sq.numerator, sq.denominator))


@functools.lru_cache(None)
def q_channels(name):
    """exact q_J from Q at the weight vectors v_m (S^3 route), J = 2j, 2j-2, ..."""
    from s3poly import h_exact, kfrac, wnorm2
    d, n = SLOTS[name]
    j = Fr(n, 2)
    ms = mlist(n)
    Js = [2 * j - 2 * t for t in range(int(j + Fr(1, 2)) + (0 if n % 2 else 1)) if 2 * j - 2 * t >= 0]
    Js = [J for J in Js if J >= 0]
    pos = [m for m in ms if m >= 0][:len(Js)]   # m = j, j-1, ...
    Qv = {}
    for m in pos:
        p = [KZERO] * (n + 1); p[ms.index(m)] = kfrac(1)
        Qv[m] = h_exact(name, p) / wnorm2(name, p) ** 2
    # Q(v_m) = sum_J q_J CG(j m j m | J 2m)^2 ; triangular in (m, J)
    q = {}
    for m, J in zip(pos, Js):
        acc = Qv[m]
        for J2 in Js:
            if J2 in q:
                sg, sq = cg_exact(j, m, j, m, J2, 2 * m)
                acc -= q[J2] * sq
        sg, sq = cg_exact(j, m, j, m, J, 2 * m)
        q[J] = acc / sq
    return q, Qv


def symbols_chart(n):
    N = n + 1
    xs = sp.symbols(f'x0:{2 * N}', real=True)
    P = [xs[i] + sp.I * xs[N + i] for i in range(N)]
    return xs, P


def poly_to_dict(expr, xs):
    pol = sp.Poly(sp.expand(expr), *xs)
    out = {}
    for mon, c in pol.terms():
        c = sp.nsimplify(c)
        if not c.is_Rational:
            raise ValueError(f'non-rational coefficient {c}')
        out[tuple(mon)] = Fr(int(c.p), int(c.q))
    return out


def _exact_sqrt(r):
    """sqrt of a nonnegative Fraction that must be a perfect square (asserted)."""
    from math import isqrt
    a, b = r.numerator, r.denominator
    sa, sb = isqrt(a), isqrt(b)
    assert sa * sa == a and sb * sb == b, f'radicand {r} is not a perfect square'
    return Fr(sa, sb)


def _cpoly_from_terms(terms, N):
    """terms: list of (coef (re,im) Fractions, [(index, conj?)...]) -> real polynomial dict in x (len 2N);
    p_i = x_i + i x_{N+i}, conj(p_i) = x_i - i x_{N+i}.  Returns dict exps -> (re, im)."""
    out = {}
    for (cr, ci), facs in terms:
        cur = {tuple([0] * (2 * N)): (cr, ci)}
        for (i, cj) in facs:
            nxt = {}
            for e, (a, b) in cur.items():
                e1 = list(e); e1[i] += 1
                k = tuple(e1); v = nxt.get(k, (0, 0)); nxt[k] = (v[0] + a, v[1] + b)
                e2 = list(e); e2[N + i] += 1
                s = -1 if cj else 1        # times (+/- i)
                k = tuple(e2); v = nxt.get(k, (0, 0)); nxt[k] = (v[0] - s * b, v[1] + s * a)
            cur = nxt
        for e, (a, b) in cur.items():
            v = out.get(e, (0, 0)); out[e] = (v[0] + a, v[1] + b)
    return out


def _channel_norm_terms(n, J, second_theta=False):
    """terms of ||[w (x) w']_J||^2 with w' = w (or Theta w), in weighted coordinates: list of
    (coefficient, factor list) with exactly rational coefficients (products of CG square roots made exact)."""
    j = Fr(n, 2)
    ms = mlist(n)
    terms = []
    for M in [J - t for t in range(int(2 * J) + 1)]:
        comp = []   # (sign, radicand, i1, i2, conj2, extra sign)
        for i1, m1 in enumerate(ms):
            m2 = M - m1
            if m2 not in ms:
                continue
            sg, sq = cg_exact(j, m1, j, m2, J, M)
            if sg == 0:
                continue
            if not second_theta:
                comp.append((sg, sq * Fm(n, m1) * Fm(n, m2), i1, ms.index(m2), False))
            else:
                # (Theta w)_{m2} = (-1)^{j-m2} conj(w_{-m2})
                s2 = (-1) ** int(j - m2)
                comp.append((sg * s2, sq * Fm(n, m1) * Fm(n, -m2), i1, ms.index(-m2), True))
        for (s1, r1, a1, b1, c1) in comp:
            for (s2, r2, a2, b2, c2) in comp:
                coef = s1 * s2 * _exact_sqrt(r1 * r2)
                # comp * conj(comp): factor1 = p_a1 * (p_b1 or conj p_b1); conj factor2 = conj p_a2 * conj(...)
                terms.append(((coef, Fr(0)), [(a1, False), (b1, c1), (a2, True), (b2, not c2)]))
    return terms


def _real_dict(cp):
    out = {}
    for e, (a, b) in cp.items():
        assert b == 0, 'imaginary part in a real form'
        if a != 0:
            out[e] = Fr(a)
    return out


@functools.lru_cache(None)
def h_poly(name):
    d, n = SLOTS[name]
    N = n + 1
    q, _ = q_channels(name)
    terms = []
    for J, qJ in q.items():
        for (c, f) in _channel_norm_terms(n, J):
            terms.append(((c[0] * qJ, Fr(0)), f))
    return _real_dict(_cpoly_from_terms(terms, N))


@functools.lru_cache(None)
def r6_poly():
    """||[w (x) Theta w]_6||^2 on V_3, as a polynomial in the chart (rational coefficients checked)."""
    return _real_dict(_cpoly_from_terms(_channel_norm_terms(6, Fr(6), second_theta=True), 7))


def norm2_poly(n):
    ms = mlist(n)
    N = n + 1
    out = {}
    for i, m in enumerate(ms):
        e = [0] * (2 * N); e[i] = 2; out[tuple(e)] = Fr(Fm(n, m))
        e = [0] * (2 * N); e[N + i] = 2; out[tuple(e)] = Fr(Fm(n, m))
    return out


# ---------------------------------------------------------------- generic polynomial evaluation (any ring: mpf, iv, Fraction)
def pderiv(pol, i):
    out = {}
    for e, c in pol.items():
        if e[i]:
            e2 = list(e); e2[i] -= 1
            out[tuple(e2)] = out.get(tuple(e2), 0) + c * e[i]
    return {e: c for e, c in out.items() if c != 0}


class PolyEval:
    def __init__(self, pol, nvar):
        self.pol = pol; self.nvar = nvar
        self.g = [pderiv(pol, i) for i in range(nvar)]
        self.H = [[pderiv(self.g[i], k) for k in range(nvar)] for i in range(nvar)]
        self._T = None

    def T(self):
        if self._T is None:
            self._T = [[[pderiv(self.H[i][k], l) for l in range(self.nvar)] for k in range(self.nvar)] for i in range(self.nvar)]
        return self._T

    @staticmethod
    def ev(pol, x, conv):
        tot = conv(0)
        for e, c in pol.items():
            t = conv(c)
            for i, k in enumerate(e):
                if k:
                    t = t * x[i] ** k
            tot = tot + t
        return tot

    def value(self, x, conv):
        return self.ev(self.pol, x, conv)

    def grad(self, x, conv):
        return [self.ev(g, x, conv) for g in self.g]

    def hess(self, x, conv):
        n = self.nvar
        H = [[None] * n for _ in range(n)]
        for i in range(n):
            for k in range(i, n):
                H[i][k] = H[k][i] = self.ev(self.H[i][k], x, conv)
        return H

    def third(self, x, conv, z):
        """D^3 f(x)[z] as a matrix: sum_l d^3 f/dx_i dx_k dx_l z_l"""
        n = self.nvar
        T = self.T()
        M = [[conv(0)] * n for _ in range(n)]
        for i in range(n):
            for k in range(i, n):
                acc = conv(0)
                for l in range(n):
                    if T[i][k][l]:
                        acc = acc + self.ev(T[i][k][l], x, conv) * z[l]
                M[i][k] = M[k][i] = acc
        return M
