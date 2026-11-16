"""sympy-exact CS-basis objects."""
from lib_rep import *
import sympy as sp


def sqrtS(j2):
    return [sp.sqrt(x) for x in svec(j2)]


def P_cs(slot):
    name, j2, d = slot_info(slot)
    P = projector_w(name, j2)
    s = sqrtS(j2); n = j2 + 1
    return sp.Matrix(n, n, lambda a, b: sp.radsimp(P[a][b].tosympy() * s[a] / s[b]))


def eta_exact(slot):
    """orthonormal basis of the range of P_cs, Gram-Schmidt on its columns in order"""
    name, j2, d = slot_info(slot)
    Pc = P_cs(slot); n = j2 + 1
    basis = []
    for k in range(n):
        v = Pc[:, k]
        for b in basis:
            v = v - (b.H * v)[0] * b
        v = v.applyfunc(sp.radsimp)
        nn2 = sp.nsimplify(sp.radsimp((v.H * v)[0]))
        if nn2 != 0:
            basis.append((v / sp.sqrt(nn2)).applyfunc(sp.radsimp))
        if len(basis) == d: break
    return sp.Matrix.hstack(*basis)


def D_cs_exact(q, j2):
    """exact CS-basis D^j(q) for a group element q (tuple of K)"""
    Dm = Dw(quat_to_U(q), j2)
    s = sqrtS(j2); n = j2 + 1
    return sp.Matrix(n, n, lambda a, b: sp.radsimp(Dm[a][b].tosympy() * s[a] / s[b]))


def Theta_matrix(j2):
    """(Theta u)_m = (-1)^{j-m} conj(u_{-m}) = E conj(u); returns E"""
    n = j2 + 1; ml = ms(j2)
    E = sp.zeros(n, n)
    for r, m2 in enumerate(ml):
        c = ml.index(-m2)
        E[r, c] = (-1) ** ((j2 - m2) // 2)
    return E


def spin_ops_cs(j2):
    n = j2 + 1; ml = ms(j2)
    Jp = sp.zeros(n, n); Jz = sp.zeros(n, n)
    for c, m2 in enumerate(ml):
        m = sp.Rational(m2, 2); j = sp.Rational(j2, 2)
        Jz[c, c] = m
        if c > 0:
            Jp[c - 1, c] = sp.sqrt((j - m) * (j + m + 1))
    Jm = Jp.T
    return Jp, Jm, Jz
