"""Exact analysis at a point given by a rational chart vector x0 (u = x0/sqrt(r), r = |x0|_G^2), sympy rationals.
Chart x = (Re p_j..Re p_-j, Im p_j..Im p_-j); metric G = diag(F_m, F_m); operators are G-selfadjoint:
  Mhat = G^{-1} Hess_x h(x0)/r - 4 h(x0)/r^2,  Shat = J0 Mhat,  sigma(a, b) = a^T G J0 b."""
from fractions import Fraction as Fr
import functools
import sympy as sp
from core import mlist, Fm
from reps import SLOTS
from hpoly import h_poly, PolyEval, norm2_poly


def R(x):
    if isinstance(x, Fr):
        return sp.Rational(x.numerator, x.denominator)
    if isinstance(x, int):
        return sp.Integer(x)
    x = sp.sympify(x)
    assert x.is_Rational, f'non-rational entry {x}'
    return x


@functools.lru_cache(None)
def h_eval(name):
    d, n = SLOTS[name]
    return PolyEval(h_poly(name), 2 * (n + 1))


def Gdiag(n):
    ms = mlist(n)
    return [Fm(n, m) for m in ms] * 2


def J0(n):
    N = n + 1
    M = sp.zeros(2 * N, 2 * N)
    for i in range(N):
        M[N + i, i] = 1; M[i, N + i] = -1
    return M


def real_of_complex_linear(Mc, n):
    """complex-linear map on p with complex matrix Mc (sympy) -> real 2N x 2N on (Re p, Im p)"""
    N = n + 1
    A = Mc.applyfunc(sp.re); B = Mc.applyfunc(sp.im)
    out = sp.zeros(2 * N, 2 * N)
    out[:N, :N] = A; out[:N, N:] = -B; out[N:, :N] = B; out[N:, N:] = A
    return out


def rotation_generators(n):
    """i Jz, (J+ - J-), i (J+ + J-) on weighted coordinates p (complex-linear), as real matrices."""
    j = Fr(n, 2); ms = mlist(n); N = n + 1
    Jz = sp.zeros(N, N); Jp = sp.zeros(N, N); Jm = sp.zeros(N, N)
    for a, m in enumerate(ms):
        Jz[a, a] = R(m)
        if m < j:
            Jp[ms.index(m + 1), a] = R(j - m)       # X d/dY: e_m -> (j-m) e_{m+1}
        if m > -j:
            Jm[ms.index(m - 1), a] = R(j + m)
    I = sp.I
    return [real_of_complex_linear(I * Jz, n), real_of_complex_linear(Jp - Jm, n), real_of_complex_linear(I * (Jp + Jm), n)]


def theta_real(n):
    """Theta on p: (Theta p)_m = (-1)^{j-m} conj(p_{-m})  (F_m = F_{-m})"""
    j = Fr(n, 2); ms = mlist(n); N = n + 1
    T = sp.zeros(2 * N, 2 * N)
    for a, m in enumerate(ms):
        b = ms.index(-m); s = (-1) ** int(j - m)
        T[a, b] = s; T[N + a, N + b] = -s
    return T


def group_generators(name):
    d, n = SLOTS[name]
    gens = [J0(n)] + rotation_generators(n)
    if name == 'R2':
        Th = theta_real(n)
        gens += [Th, Th * J0(n)]
    return gens


class ExactPoint:
    def __init__(self, name, x0):
        d, n = SLOTS[name]
        self.name = name; self.n = n; self.N = n + 1; self.dim = 2 * (n + 1)
        self.x0 = sp.Matrix([R(v) for v in x0])
        self.Gd = Gdiag(n)
        self.G = sp.diag(*self.Gd)
        self.Gi = sp.diag(*[sp.Rational(1, g) for g in self.Gd])
        self.r = (self.x0.T * self.G * self.x0)[0]
        he = h_eval(name)
        xf = [Fr(int(sp.fraction(v)[0]), int(sp.fraction(v)[1])) for v in self.x0]
        self.h0 = R(he.value(xf, Fr))
        self.grad0 = sp.Matrix([R(v) for v in he.grad(xf, Fr)])
        self.H0 = sp.Matrix([[R(v) for v in row] for row in he.hess(xf, Fr)])
        self.hu = self.h0 / self.r ** 2
        self.critical = (self.grad0 - 4 * self.h0 / self.r * self.G * self.x0) == sp.zeros(self.dim, 1)
        self.M = self.Gi * self.H0 / self.r - 4 * self.hu * sp.eye(self.dim)
        self.J0 = J0(n)
        self.S = self.J0 * self.M
        self.gens = group_generators(name)
        Kv = sp.Matrix.hstack(*[g * self.x0 for g in self.gens])
        self.Kspan = Kv.columnspace()
        self.k = len(self.Kspan)
        A = sp.Matrix.hstack(self.x0, *self.Kspan) if self.Kspan else self.x0
        self.Nbasis = (A.T * self.G).nullspace()
        self.Nb = sp.Matrix.hstack(*self.Nbasis)
        # sigma on K_u
        if self.Kspan:
            Kb = sp.Matrix.hstack(*self.Kspan)
            self.sigmaK = Kb.T * self.G * self.J0 * Kb
            self.r_u = self.k - self.sigmaK.rank()
        else:
            self.r_u = 0
        self.Qf = self.Nb.T * self.G * self.M * self.Nb      # quadratic form of Mhat on N_u

    def signature(self):
        lam = sp.Symbol('lam')
        cp = sp.Poly(self.Qf.charpoly(lam).as_expr(), lam)
        # all roots real (symmetric): count by Descartes after removing the zero roots
        coeffs = cp.all_coeffs()
        z = 0
        while coeffs and coeffs[-1] == 0:
            coeffs.pop(); z += 1
        def changes(cs):
            s = [c for c in cs if c != 0]
            return sum(1 for a, b in zip(s, s[1:]) if (a > 0) != (b > 0))
        npos = changes(coeffs)
        neg = [c * (-1) ** (len(coeffs) - 1 - i) for i, c in enumerate(coeffs)]
        nneg = changes(neg)
        assert npos + nneg + z == self.Qf.shape[0], 'Descartes count does not close'
        return [nneg, z, npos]

    def kernel_in_N(self):
        ks = self.Qf.nullspace()
        return [self.Nb * v for v in ks]

    def slow(self, S=None):
        """charpoly of S (default Shat) in xi -> (zero multiplicity, slow polynomial in nu, factored primitive)."""
        S = self.S if S is None else S
        xi, nu = sp.symbols('xi nu')
        cp = sp.Poly(S.charpoly(xi).as_expr(), xi)
        z = 0
        cs = cp.all_coeffs()
        while cs and cs[-1] == 0:
            cs.pop(); z += 1
        q = sp.Poly(cs, xi)
        # even polynomial in xi
        assert all(c == 0 for i, c in enumerate(reversed(q.all_coeffs())) if i % 2 == 1), 'not even'
        qn = sp.Poly(sum(c * nu ** (i // 2) for i, c in enumerate(reversed(q.all_coeffs())) if i % 2 == 0), nu)
        if qn.degree() <= 0:
            return z, sp.Integer(1)
        prim = qn.primitive()[1]
        if prim.LC() < 0:
            prim = -prim
        return z, sp.factor(prim.as_expr())

    def krein_and_roots(self):
        pass
