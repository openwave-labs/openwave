"""Numeric spin-j matrices built from an SU(2) matrix (own code).

Quaternion-to-matrix convention used throughout this room:
    w + x i + y j + z k  ->  w*I - i*(x*sx + y*sy + z*sz)
                          =  [[w - i z, -y - i x], [y - i x, w + i z]]
(i -> -i sx, j -> -i sy, k -> -i sz; a homomorphism, checked numerically).

V_j = Sym^{2j} C^2 with basis e_m = eps1^(j+m) eps2^(j-m) / sqrt((j+m)!(j-m)!),
U eps_b = sum_a U_ab eps_a, so D^j(U) e_m' = sum_m D_mm' e_m. Index order m = j, j-1, ..., -j.
"""
import numpy as np
from math import comb, factorial, sqrt

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def qmat(q):
    w, x, y, z = [complex(float(t)) for t in q]
    return w * np.eye(2) - 1j * (x * SX + y * SY + z * SZ)


def _pow_coeffs(a, b, p):
    """(a eps1 + b eps2)^p as array indexed by power of eps1."""
    c = np.zeros(p + 1, dtype=complex)
    for k in range(p + 1):
        c[k] = comb(p, k) * a ** k * b ** (p - k)
    return c


def D(twoj, U):
    n = twoj
    ms = [twoj - 2 * t for t in range(n + 1)]  # 2m values, descending
    out = np.zeros((n + 1, n + 1), dtype=complex)
    for col, tm in enumerate(ms):
        p = (n + tm) // 2
        q = n - p
        poly = np.convolve(_pow_coeffs(U[0, 0], U[1, 0], p), _pow_coeffs(U[0, 1], U[1, 1], q))
        for row, tm2 in enumerate(ms):
            p2 = (n + tm2) // 2
            q2 = n - p2
            out[row, col] = poly[p2] * sqrt(factorial(p2) * factorial(q2)) / sqrt(factorial(p) * factorial(q))
    return out


def Rz(a):
    return np.array([[np.exp(-0.5j * a), 0], [0, np.exp(0.5j * a)]])


def Ry(b):
    return np.array([[np.cos(b / 2), -np.sin(b / 2)], [np.sin(b / 2), np.cos(b / 2)]], dtype=complex)


def mvals(twoj):
    return np.array([(twoj - 2 * t) / 2 for t in range(twoj + 1)])
