"""Shared definitions for the spin-3 problem (worklist section 1).

Everything here is built from the conventions of worklist.md section 1:
basis order v3, v2, v1, v0, v-1, v-2, v-3; Condon-Shortley Clebsch-Gordan
coefficients built from the Racah closed formula (no library lookup).
"""
import os
import sympy as sp
from sympy import Rational as R, sqrt, factorial as fac, I

J = 3
MS = [3, 2, 1, 0, -1, -2, -3]           # basis order of section 1.1
IDX = {m: k for k, m in enumerate(MS)}

# Planted defects, switched on by the environment variable PLANT.
PLANT = os.environ.get("PLANT", "")


_X = sp.Symbol("_X")


def exact(t):
    """Exact simplification only (no numerical identification): expand + radsimp, and simplify for
    small closed-form numbers.  (sympy's nsimplify is deliberately not used: on closed-form numbers
    it identifies values numerically, which is not an exact route.)"""
    t = sp.sympify(t)
    r = sp.radsimp(sp.expand(t))
    if r.free_symbols or r.is_Rational:
        return r
    if sp.count_ops(r) <= 40:
        return sp.simplify(r)
    return r


def is_zero_exact(t):
    t = sp.sympify(t)
    if t.free_symbols:
        return sp.simplify(t) == 0
    r = sp.radsimp(sp.expand(t))
    if r == 0:
        return True
    return sp.Poly(sp.minimal_polynomial(r, _X), _X) == sp.Poly(_X, _X)


def cg(j1, m1, j2, m2, JJ, M):
    """<j1 m1; j2 m2 | J M>, Racah closed formula, exact (Condon-Shortley)."""
    if m1 + m2 != M or abs(m1) > j1 or abs(m2) > j2 or abs(M) > JJ:
        return sp.Integer(0)
    if JJ > j1 + j2 or JJ < abs(j1 - j2):
        return sp.Integer(0)
    pref = sqrt(R((2 * JJ + 1) * fac(JJ + j1 - j2) * fac(JJ - j1 + j2) * fac(j1 + j2 - JJ),
                  fac(j1 + j2 + JJ + 1)))
    pref *= sqrt(fac(JJ + M) * fac(JJ - M) * fac(j1 - m1) * fac(j1 + m1) * fac(j2 - m2) * fac(j2 + m2))
    s = sp.Integer(0)
    for k in range(0, j1 + j2 + JJ + 2):
        dens = [k, j1 + j2 - JJ - k, j1 - m1 - k, j2 + m2 - k, JJ - j2 + m1 + k, JJ - j1 - m2 + k]
        if any(d < 0 for d in dens):
            continue
        term = sp.Integer((-1) ** k)
        for d in dens:
            term /= fac(d)
        s += term
    val = exact(pref * s)
    if PLANT == "cg_sign" and m1 < 0:
        val = -val
    return val


CG6 = {(m1, m2): cg(3, m1, 3, m2, 6, m1 + m2) for m1 in MS for m2 in MS}


def theta(c):
    """Time reversal of section 1.3 on a coefficient list c (indexed like MS)."""
    out = [None] * 7
    for m in MS:
        out[IDX[m]] = (-1) ** (3 - m) * sp.conjugate(c[IDX[-m]])
    if PLANT == "theta_sign":
        out[IDX[1]] = -out[IDX[1]]
    return out


def rho6(c, conj=sp.conjugate):
    """rho_6(u)_Q for Q = -6..6 (dict), exact."""
    tc = [(-1) ** (3 - m) * conj(c[IDX[-m]]) for m in MS]
    if PLANT == "theta_sign":
        tc[IDX[1]] = -tc[IDX[1]]
    out = {}
    for Q in range(-6, 7):
        s = 0
        for m1 in MS:
            m2 = Q - m1
            if m2 in IDX:
                s += CG6[(m1, m2)] * c[IDX[m1]] * tc[IDX[m2]]
        out[Q] = s
    return out


def rhat_k(c, k):
    """||rank-k part of u (x) Theta u||^2 / ||u||^4, exact (same construction as r-hat_6)."""
    tc = [(-1) ** (3 - m) * sp.conjugate(c[IDX[-m]]) for m in MS]
    num = 0
    for Q in range(-k, k + 1):
        s = 0
        for m1 in MS:
            m2 = Q - m1
            if m2 in IDX:
                s += cg(3, m1, 3, m2, k, Q) * c[IDX[m1]] * tc[IDX[m2]]
        num += sp.expand(s * sp.conjugate(s))
    nrm = sum(sp.expand(x * sp.conjugate(x)) for x in c)
    return exact(num / nrm ** 2)


def rhat6(c):
    """Exact r-hat_6 of a coefficient list (sympy numbers)."""
    rh = rho6(c)
    num = sum(sp.expand(v * sp.conjugate(v)) for v in rh.values())
    nrm = sum(sp.expand(x * sp.conjugate(x)) for x in c)
    return exact(num / nrm ** 2)


# ---------------------------------------------------------------- real form
# u = sum_m (x_m + i y_m) v_m ; real coordinates ordered x3..x-3, y3..y-3
XS = sp.symbols("x3 x2 x1 x0 xm1 xm2 xm3", real=True)
YS = sp.symbols("y3 y2 y1 y0 ym1 ym2 ym3", real=True)
RV = list(XS) + list(YS)


def build_N():
    c = [XS[k] + I * YS[k] for k in range(7)]
    rh = rho6(c)
    N = 0
    for v in rh.values():
        re, im = sp.expand(v).as_real_imag()
        N += sp.expand(re ** 2 + im ** 2)
    return sp.expand(N)


_N_CACHE = {}


def N_poly():
    if "N" not in _N_CACHE:
        _N_CACHE["N"] = build_N()
    return _N_CACHE["N"]


def grad_N_sym():
    if "g" not in _N_CACHE:
        N = N_poly()
        _N_CACHE["g"] = [sp.diff(N, v) for v in RV]
    return _N_CACHE["g"]


def hess_N_sym():
    if "h" not in _N_CACHE:
        g = grad_N_sym()
        _N_CACHE["h"] = [[sp.diff(g[i], RV[j]) for j in range(14)] for i in range(14)]
    return _N_CACHE["h"]


def at(expr_list, r):
    sub = dict(zip(RV, r))
    return [exact(e.xreplace(sub)) for e in expr_list]


def sphere_grad(c):
    """tangential gradient of r-hat_6 at u/||u||, for the unit vector u/||u|| (exact, real 14-vector)."""
    nrm2 = sum(sp.expand(z * sp.conjugate(z)) for z in c)
    r = [exact(v / sp.sqrt(nrm2)) for v in to_real(c)]
    g = sp.Matrix(at(grad_N_sym(), r))
    Nv = exact(N_poly().xreplace(dict(zip(RV, r))))
    return (g - 4 * Nv * sp.Matrix(r)).applyfunc(exact), r, Nv


def to_real(c):
    """complex coefficient list -> 14 real coordinates (sympy)."""
    return [sp.re(x) for x in c] + [sp.im(x) for x in c]


def from_real(r):
    return [r[k] + I * r[7 + k] for k in range(7)]


# ---------------------------------------------------------------- operators
def Jmats():
    """J_z, J_+, J_-, J_x, J_y as exact 7x7 sympy matrices in the MS basis."""
    Jz = sp.zeros(7)
    Jp = sp.zeros(7)
    Jm = sp.zeros(7)
    for m in MS:
        Jz[IDX[m], IDX[m]] = m
        if m + 1 <= 3:
            Jp[IDX[m + 1], IDX[m]] = sqrt(12 - m * (m + 1))
        if m - 1 >= -3:
            Jm[IDX[m - 1], IDX[m]] = sqrt(12 - m * (m - 1))
    Jx = (Jp + Jm) / 2
    Jy = (Jp - Jm) / (2 * I)
    return Jz, Jp, Jm, Jx, Jy


def realify(A):
    """complex-linear 7x7 map -> real 14x14 map on (x, y)."""
    Ar = A.applyfunc(sp.re)
    Ai = A.applyfunc(sp.im)
    return sp.Matrix(sp.BlockMatrix([[Ar, -Ai], [Ai, Ar]]))


def re_inner(a, b):
    """Re<a,b> for complex coefficient lists."""
    return sp.re(sum(sp.conjugate(x) * y for x, y in zip(a, b)))


# ---------------------------------------------------------------- mpmath route
def rhat6_mp(c, mp):
    """High precision r-hat_6 from mpmath complex coefficients, using float CG
    evaluated at the working precision."""
    cgm = {k: mp.mpf(sp.N(v, mp.mp.dps + 10)) for k, v in CG6.items()}
    tc = [(-1) ** (3 - m) * mp.conj(c[IDX[-m]]) for m in MS]
    tot = mp.mpf(0)
    for Q in range(-6, 7):
        s = mp.mpc(0)
        for m1 in MS:
            m2 = Q - m1
            if m2 in IDX:
                s += cgm[(m1, m2)] * c[IDX[m1]] * tc[IDX[m2]]
        tot += abs(s) ** 2
    nrm = sum(abs(x) ** 2 for x in c)
    return tot / nrm ** 2


def report(label, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + label + ((" : " + detail) if detail else ""))
    return ok
