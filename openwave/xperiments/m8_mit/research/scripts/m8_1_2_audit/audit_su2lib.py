"""Exact SU(2) tools (sympy) in the Condon-Shortley convention.

Spins are handled as sympy Rationals so half-integers work. Vectors in V_j are
dicts or lists indexed by m = -j..j (list index k <-> m = -j + k).
"""
import sympy as sp
from functools import lru_cache

R = sp.Rational


def ms(j):
    j = sp.nsimplify(j)
    return [-j + k for k in range(int(2 * j) + 1)]  # -j..j ascending


@lru_cache(maxsize=None)
def cg(j1, m1, j2, m2, J, M):
    """Racah closed formula, Condon-Shortley phase. Arguments are sympy Rationals."""
    j1, m1, j2, m2, J, M = [sp.nsimplify(x) for x in (j1, m1, j2, m2, J, M)]
    if m1 + m2 != M:
        return sp.Integer(0)
    if J < abs(j1 - j2) or J > j1 + j2 or abs(m1) > j1 or abs(m2) > j2 or abs(M) > J:
        return sp.Integer(0)
    f = sp.factorial
    pre = sp.sqrt((2 * J + 1) * f(J + j1 - j2) * f(J - j1 + j2) * f(j1 + j2 - J) / f(j1 + j2 + J + 1))
    pre *= sp.sqrt(f(J + M) * f(J - M) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2))
    s = sp.Integer(0)
    for k in range(0, int(j1 + j2 + J) + 2):
        args = [k, j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k]
        if any(a < 0 for a in args):
            continue
        s += R((-1) ** k) / sp.prod([f(a) for a in args])
    return sp.nsimplify(sp.radsimp(pre * s))


def theta(u, j=3, eps=None):
    """Theta_j v_m = eps (-1)^(j-m) v_{-m}, antilinear. u = list over m=-j..j.
    Default eps chosen so that at j=3 Theta v_m = (-1)^m v_{-m} (eps_3 = -1)."""
    mm = ms(j)
    n = len(mm)
    if eps is None:
        eps = -1 if j == 3 else 1
    out = [sp.Integer(0)] * n
    for k, m in enumerate(mm):
        # coefficient of v_{-m} gets eps (-1)^(j-m) conj(u_m)
        kk = n - 1 - k
        out[kk] += eps * (-1) ** int(j - m) * sp.conjugate(u[k])
    return out


def couple(x, y, j1, j2, J):
    """[x (x) y]_J components N = -J..J."""
    m1s, m2s = ms(j1), ms(j2)
    out = []
    for N in ms(J):
        s = sp.Integer(0)
        for a, m1 in enumerate(m1s):
            m2 = N - m1
            if abs(m2) > j2:
                continue
            b = m2s.index(m2)
            c = cg(j1, m1, j2, m2, J, N)
            if c != 0:
                s += c * x[a] * y[b]
        out.append(s)
    return out


def MK_matrix(P, K, j=3, eps=None):
    """M^{(j)}_K(P)_N = sum_{n+n'=N} <j n; j n'|K N> eps (-1)^(j-n') P_{n,-n'}.
    P is an sp.Matrix indexed by (k, l) <-> (m=-j+k, m'=-j+l)."""
    if eps is None:
        eps = -1 if j == 3 else 1
    mm = ms(j)
    n = len(mm)
    out = []
    for N in ms(K):
        s = sp.Integer(0)
        for a, m1 in enumerate(mm):
            m2 = N - m1
            if abs(m2) > j:
                continue
            b = mm.index(-m2)  # column index of -n'
            c = cg(j, m1, j, m2, K, N)
            if c != 0:
                s += c * eps * (-1) ** int(j - m2) * P[a, b]
        out.append(s)
    return out


def norm2(v):
    return sp.nsimplify(sp.radsimp(sp.expand(sum(sp.expand(x * sp.conjugate(x)) for x in v))))


def rho(u, K, j=3):
    return couple(u, theta(u, j), j, j, K)


def M_map(u, K):
    """M_K(u) = [rho_K(u) (x) u]_3."""
    return couple(rho(u, K), u, K, 3, 3)


def basis(m, j=3):
    mm = ms(j)
    v = [sp.Integer(0)] * len(mm)
    v[mm.index(m)] = sp.Integer(1)
    return v


def simp(x):
    return sp.nsimplify(sp.radsimp(sp.expand(x)))
