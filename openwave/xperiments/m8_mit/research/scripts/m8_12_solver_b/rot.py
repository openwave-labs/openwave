"""Exact and numerical rotation matrices D^3(n, theta) of section 1.2.

Exact route: n.J has eigenvalues -3..3 (it is conjugate to J_z), so
    D = exp(-i theta n.J) = sum_m e^{-i theta m} P_m,
    P_m = prod_{k != m} (n.J - k)/(m - k)     (Lagrange spectral projectors),
which is exact for algebraic n and exact cos/sin of theta.
Numerical route: scipy.linalg.expm on the same generator (used as a cross-check).
"""
import sympy as sp
import numpy as np
import scipy.linalg as sl
import core

Jz, Jp, Jm, Jx, Jy = core.Jmats()
JN = [np.array(sp.N(A, 20), dtype=complex) for A in (Jx, Jy, Jz)]


def nJ(n):
    return n[0] * Jx + n[1] * Jy + n[2] * Jz


def D3_exact(n, theta):
    A = nJ(n)
    c, s = sp.cos(theta), sp.sin(theta)
    D = sp.zeros(7)
    for m in range(-3, 4):
        Pm = sp.eye(7)
        for k in range(-3, 4):
            if k != m:
                Pm = Pm * (A - k * sp.eye(7)) / (m - k)
        D += sp.expand((c - sp.I * s) ** m) * Pm if m >= 0 else sp.expand((c + sp.I * s) ** (-m)) * Pm
    return D.applyfunc(core.exact)


def D3_num(n, theta):
    n = [float(x) for x in n]
    return sl.expm(-1j * float(theta) * (n[0] * JN[0] + n[1] * JN[1] + n[2] * JN[2]))


def R3(n, theta):
    """3x3 rotation (Rodrigues), numerical."""
    n = np.array([float(x) for x in n])
    n = n / np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    t = float(theta)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def Dz(theta):
    """exact z rotation: diagonal e^{-i m theta}."""
    return sp.diag(*[sp.expand_complex(sp.exp(-sp.I * m * theta)) for m in core.MS])


# R_x(pi): v_m -> -v_{-m}  (derived exactly by D3_exact, checked in item1.py)
