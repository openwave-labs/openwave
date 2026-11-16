"""Exact critical-point analysis of h = Q |w|^4 = sum_J q_J |Pi_J (w(x)w)|^2.

Coordinates: weighted p_m = a_m + i b_m (m = j..-j); h is a polynomial with rational coefficients in (a, b).
CS real coordinates x = (Re w, Im w) with w = S p, S = diag(s_m), s_m^2 = (j+m)!(j-m)!.
  grad_w h = S^-1 grad_p h ;  Hess_w h = S^-1 Hess_p h S^-1.
M_hat = Hess_w h - 4 h(u) (at unit u). M' = S M_hat S^-1 = Hess_p S^-2 - 4h has the same characteristic
polynomial; any real-linear tau commuting with S (i, Theta, k) gives charpoly(tau M_hat) = charpoly(tau M').
Signature on N_u: M_hat u = 8 h u and M_hat K_u = 0 (both checked), so
charpoly_N(x) = charpoly(M_hat)(x) / ((x - 8h) x^k); counts of signs by Descartes' rule (exact for real-rooted
polynomials).
"""
from lib_quartic import *
from lib_sym import Theta_matrix, spin_ops_cs
import sympy as sp
import mpmath

X = sp.Symbol('x')
NU = sp.Symbol('nu')


def h_poly(slot):
    name, j2, d = slot_info(slot)
    n = j2 + 1
    q, proj = channel_coeffs(slot)
    a = sp.symbols(f'a0:{n}', real=True); b = sp.symbols(f'b0:{n}', real=True)
    gens = list(a) + list(b)
    p = [sp.Poly(a[k] + sp.I * b[k], *gens, domain='QQ_I') for k in range(n)]
    pc = [sp.Poly(a[k] - sp.I * b[k], *gens, domain='QQ_I') for k in range(n)]
    sv = svec(j2)
    pp = {}; ppc = {}
    for x in range(n):
        for y in range(n):
            pp[x * n + y] = p[x] * p[y]; ppc[x * n + y] = pc[x] * pc[y]
    h = sp.Poly(0, *gens, domain='QQ_I')
    for J2 in sym_J2s(j2):
        qJ = sp.Rational(q[J2].a)
        rowsd = {}
        for (r, c), v in proj[J2].items():
            rowsd.setdefault(r, []).append((c, sp.Rational(v)))
        for r, lst in rowsd.items():
            y = sum((pp[c] * v for c, v in lst), sp.Poly(0, *gens, domain='QQ_I'))
            yc = sum((ppc[c] * v for c, v in lst), sp.Poly(0, *gens, domain='QQ_I'))
            h += y * yc * (qJ * sv[r // n] * sv[r % n])
    h = sp.Poly(h.as_expr(), *gens, domain='QQ')
    return h, gens, j2


class Point:
    def __init__(self, slot, comps, hdata=None):
        """comps: dict m-string -> sympy-parsable CS component (unnormalised)"""
        self.slot = slot
        self.h, self.gens, self.j2 = hdata if hdata else h_poly(slot)
        j2 = self.j2; n = j2 + 1; ml = ms(j2)
        w = [sp.Integer(0)] * n
        for k, v in comps.items():
            m2 = int(2 * sp.Rational(k))
            w[ml.index(m2)] = sp.sympify(v, locals={'I': sp.I, 'sqrt': sp.sqrt})
        nrm = sp.sqrt(sp.simplify(sum(sp.expand(x * sp.conjugate(x)) for x in w)))
        self.w = [sp.radsimp(x / nrm) for x in w]
        self.s = [sp.sqrt(x) for x in svec(j2)]
        self.p = [sp.radsimp(x / s) for x, s in zip(self.w, self.s)]
        self.sub = {}
        for k in range(n):
            self.sub[self.gens[k]] = sp.re(self.p[k]); self.sub[self.gens[n + k]] = sp.im(self.p[k])
        self.n = n

    def ev(self, poly):
        return sp.radsimp(sp.expand(poly.as_expr().subs(self.sub)))

    def compute(self):
        n = self.n; N = 2 * n
        h = self.h
        self.hu = self.ev(h)
        gp = [self.ev(h.diff(v)) for v in self.gens]
        Sd = self.s + self.s
        self.grad_w = [sp.radsimp(gp[i] / Sd[i]) for i in range(N)]
        self.u_real = [sp.re(x) for x in self.w] + [sp.im(x) for x in self.w]
        self.critical = all(sp.simplify(self.grad_w[i] - 4 * self.hu * self.u_real[i]) == 0 for i in range(N))
        Hp = sp.zeros(N, N)
        for i in range(N):
            di = h.diff(self.gens[i])
            for k in range(i, N):
                Hp[i, k] = Hp[k, i] = self.ev(di.diff(self.gens[k]))
        self.Hp = Hp
        S2inv = sp.diag(*[sp.Rational(1, x) for x in (svec(self.j2) * 2)])
        self.Mp = Hp * S2inv - 4 * self.hu * sp.eye(N)  # similar to M_hat
        self.Mw = sp.Matrix(N, N, lambda i, k: sp.radsimp(Hp[i, k] / (Sd[i] * Sd[k]))) - 4 * self.hu * sp.eye(N)
        return self

    # real-linear structures in (Re, Im) coordinates
    def Jc(self):
        n = self.n
        return sp.Matrix(sp.BlockMatrix([[sp.zeros(n), -sp.eye(n)], [sp.eye(n), sp.zeros(n)]]))

    def Th(self):
        E = Theta_matrix(self.j2)
        n = self.n
        return sp.Matrix(sp.BlockMatrix([[E, sp.zeros(n)], [sp.zeros(n), -E]]))

    def tangent(self, with_sp1=False):
        """K_u spanning vectors (real coords, CS): phase i u, rotations i J_a u, and in R2 Theta u, k u = Theta(i u)"""
        n = self.n
        u = sp.Matrix(self.u_real)
        Jp, Jm, Jz = spin_ops_cs(self.j2)
        Jx = (Jp + Jm) / 2; Jy = (Jp - Jm) / (2 * sp.I)
        def cplx_to_real(Mc):  # complex-linear map -> real 2n x 2n
            R = sp.re(Mc).applyfunc(sp.simplify); Im = sp.im(Mc).applyfunc(sp.simplify)
            return sp.Matrix(sp.BlockMatrix([[R, -Im], [Im, R]]))
        vecs = [self.Jc() * u]
        for A in (Jx, Jy, Jz):
            vecs.append(cplx_to_real(sp.I * A) * u)
        if with_sp1:
            vecs.append(self.Th() * u)
            vecs.append(self.Th() * self.Jc() * u)
        return [v.applyfunc(sp.radsimp) for v in vecs]


def gram_rank(vecs):
    Gm = sp.Matrix(len(vecs), len(vecs), lambda i, k: sp.simplify(sp.radsimp(sp.expand((vecs[i].T * vecs[k])[0]))))
    return Gm.rank(), Gm


def charpoly(M):
    """exact characteristic polynomial; every coefficient simplified and required to be rational
    (raises otherwise, so an irrational or unsimplified coefficient cannot pass silently)"""
    cp = M.charpoly(X)
    cs = [sp.radsimp(sp.expand(c)) for c in cp.all_coeffs()]
    cs = [c if c.is_Rational else sp.simplify(c) for c in cs]
    bad = [c for c in cs if not c.is_Rational]
    if bad:
        raise ValueError(f'non-rational charpoly coefficient: {bad[0]}')
    return sp.Poly(sum(c * X ** (len(cs) - 1 - i) for i, c in enumerate(cs)), X)


def descartes(poly):
    """for a real-rooted polynomial: (n_neg, n_zero, n_pos)"""
    P = sp.Poly(poly, X)
    z = 0
    while P.eval(0) == 0:
        P = sp.Poly(sp.quo(P.as_expr(), X), X); z += 1
    def var(cs):
        cs = [c for c in cs if c != 0]
        return sum(1 for a, b in zip(cs, cs[1:]) if a * b < 0)
    pos = var(P.all_coeffs())
    neg = var(sp.Poly(P.as_expr().subs(X, -X), X).all_coeffs())
    assert pos + neg == P.degree(), 'not real-rooted / counts inconsistent'
    return neg, z, pos


def slow_split(cp):
    """cp(x) in xi=x: returns zero multiplicity and slow polynomial in nu (primitive integer, factored)"""
    P = sp.Poly(cp, X)
    z = 0
    while P.eval(0) == 0:
        P = sp.Poly(sp.quo(P.as_expr(), X), X); z += 1
    coeffs = P.all_coeffs()
    deg = P.degree()
    assert all(coeffs[i] == 0 for i in range(1, deg + 1, 2)), 'slow polynomial not even'
    Q = sp.Poly(sum(c * NU ** ((deg - i) // 2) for i, c in enumerate(coeffs) if (deg - i) % 2 == 0), NU)
    Q = sp.Poly(Q.as_expr() * sp.lcm([sp.Rational(c).q for c in Q.all_coeffs()]), NU)
    g = sp.gcd([int(c) for c in Q.all_coeffs()])
    Q = sp.Poly(Q.as_expr() / g, NU)
    if Q.LC() < 0: Q = sp.Poly(-Q.as_expr(), NU)
    return z, Q


def root_data(Q):
    """list of (root (sympy exact or CRootOf), multiplicity, kind) ; kind in neg/pos/zero/nonreal"""
    out = []
    for fac, mult in sp.factor_list(Q.as_expr())[1]:
        F = sp.Poly(fac, NU)
        for r in F.all_roots():
            if r.is_real:
                kind = 'neg' if r < 0 else ('pos' if r > 0 else 'zero')
            else:
                kind = 'nonreal'
            out.append((r, mult, kind))
    return out


def words(rd):
    kinds = [k for r, m, k in rd]
    if all(k == 'neg' for k in kinds):
        mults = [m for r, m, k in rd]
        # each distinct root counted once
        distinct = {}
        for r, m, k in rd: distinct[sp.nsimplify(r) if r.is_Rational else r] = m
        ms_ = list(distinct.values())
        if all(m == 1 for m in ms_): return 'elliptic, simple'
        if sum(1 for m in ms_ if m == 2) == 1 and all(m in (1, 2) for m in ms_): return 'elliptic, one double frequency'
        return 'elliptic (multiplicity pattern not named by the worklist)'
    nonreal = [x for x in rd if x[2] == 'nonreal']
    pos = [x for x in rd if x[2] == 'pos']
    if not nonreal:
        tot = sum(m for r, m, k in pos)
        if tot == 1: return 'hyperbolic, one real pair'
        if tot >= 2: return 'hyperbolic, real pairs'
    if not pos and len(nonreal) == 2 and all(m == 1 for r, m, k in nonreal):
        return 'hyperbolic, a complex quartet'
    return 'hyperbolic (pattern not named by the worklist)'


def krein(Mw_num, tau_num, rd, dps=50):
    """Krein sign of each negative root nu = -t^2, from M_hat on ker(S^2 + t^2), S = tau M_hat (numerical)."""
    mpmath.mp.dps = dps
    S = tau_num * Mw_num
    N = S.rows
    res = []
    for r, mult, kind in rd:
        if kind != 'neg': continue
        t2 = -mpmath.mpf(sp.N(r, dps + 20))
        A = S * S + t2 * mpmath.eye(N)
        U, sv, V = mpmath.svd_r(A)
        svl = [sv[i] for i in range(N)]
        idx = [i for i in range(N) if svl[i] < mpmath.mpf(10) ** (-dps // 2)]
        gap = min([svl[i] for i in range(N) if i not in idx] or [mpmath.mpf(1)])
        B = mpmath.matrix(N, len(idx))
        for c, i in enumerate(idx):
            for k in range(N): B[k, c] = V[i, k]
        R = B.T * Mw_num * B
        ev = mpmath.eigsy(R)[0]
        evl = [ev[i] for i in range(len(idx))]
        if all(e > 0 for e in evl): sgn = '+'
        elif all(e < 0 for e in evl): sgn = '-'
        else: sgn = 'indefinite'
        res.append(dict(nu=r, mult=mult, dim=len(idx), sign=sgn, eig=[mpmath.nstr(e, 15) for e in evl],
                        maxsmall=mpmath.nstr(max([svl[i] for i in idx] or [0]), 3), gap=mpmath.nstr(gap, 3)))
    return res


def tonum(M, dps=60):
    mpmath.mp.dps = dps
    return mpmath.matrix([[mpmath.mpf(sp.N(M[i, k], dps + 20)) for k in range(M.cols)] for i in range(M.rows)])


def plus_xi(nu_num):
    """clause 3: +xi = sqrt(nu) with positive real part, or (real part 0) positive imaginary part"""
    z = mpmath.sqrt(mpmath.mpc(nu_num))
    if abs(z.real) < mpmath.mpf(10) ** -40:
        z = mpmath.mpc(0, abs(z.imag))
    elif z.real < 0:
        z = -z
    return z
