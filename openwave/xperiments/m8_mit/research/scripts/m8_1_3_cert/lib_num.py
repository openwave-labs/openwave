"""High-precision numerics (mpmath): D^j on the orthonormal Condon-Shortley basis for a numeric
SU(2) matrix, Euler-angle parametrisation, an exact-degree cubature for averages over SU(2)
of polynomials in D^j entries, and Gauss-Legendre nodes."""
import mpmath as mpm
from math import factorial
import numpy as np


def Dp_num(a, b, c, d, twoj):
    """numeric analogue of lib_alg.Dp_exact (weighted basis)."""
    n2 = twoj

    def pmul(P, Q):
        R = [mpm.mpc(0)] * (len(P) + len(Q) - 1)
        for i, x in enumerate(P):
            for k, y in enumerate(Q):
                R[i + k] += x * y
        return R

    def ppow(P, n):
        R = [mpm.mpc(1)]
        for _ in range(n):
            R = pmul(R, P)
        return R

    M = [[mpm.mpc(0)] * (n2 + 1) for _ in range(n2 + 1)]
    for col in range(n2 + 1):
        ex, ey = n2 - col, col
        P = pmul(ppow([c, a], ex), ppow([d, b], ey))
        for k, coef in enumerate(P):
            M[n2 - k][col] = coef
    return M


def Dw_num(a, b, c, d, twoj):
    """D^j on the orthonormal CS basis: S M S^{-1}, S = diag sqrt((j+m)!(j-m)!)."""
    M = Dp_num(a, b, c, d, twoj)
    F = [mpm.mpf(factorial(twoj - i) * factorial(i)) for i in range(twoj + 1)]
    return [[M[r][s] * mpm.sqrt(F[r] / F[s]) for s in range(twoj + 1)] for r in range(twoj + 1)]


def su2_euler(al, be, ga):
    """U = Uz(al) Uy(be) Uz(ga), Uz(t) = diag(e^{-it/2}, e^{it/2}), Uy(t) = [[c,-s],[s,c]] (t/2)."""
    def mm(A, B):
        return [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
                [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]]
    Uz = lambda t: [[mpm.expj(-t / 2), 0], [0, mpm.expj(t / 2)]]
    cb, sb = mpm.cos(be / 2), mpm.sin(be / 2)
    Uy = [[cb, -sb], [sb, cb]]
    U = mm(mm(Uz(al), Uy), Uz(ga))
    return U[0][0], U[0][1], U[1][0], U[1][1]


def gauss_legendre(n):
    x0, _ = np.polynomial.legendre.leggauss(n)
    nodes, wts = [], []
    for x in x0:
        x = mpm.mpf(float(x))
        for _ in range(100):
            p = mpm.legendre(n, x)
            dp = n * (x * p - mpm.legendre(n - 1, x)) / (x * x - 1)
            dx = p / dp
            x -= dx
            if abs(dx) < mpm.mpf(10) ** (-mpm.mp.dps + 3):
                break
        dp = n * (x * mpm.legendre(n, x) - mpm.legendre(n - 1, x)) / (x * x - 1)
        nodes.append(x)
        wts.append(2 / ((1 - x * x) * dp * dp))
    return nodes, wts


def haar_average(func, twoj, nang=16, nbeta=16, double_cover=False):
    """Average over SU(2) (normalised Haar) of func(Dw) where func is a polynomial in the
    entries of Dw = D^j(g) and their conjugates of total degree <= 2*(nang-1) in angles.
    Euler angles: d(Haar) = sin(be) dal dbe dga / (8 pi^2) on SO(3) (integer j) or with
    ga in [0, 4pi) for the double cover.  Trapezoid in al, ga (exact for trigonometric
    polynomials of degree < nang) and Gauss-Legendre in cos(be)."""
    nodes, wts = gauss_legendre(nbeta)
    gamma_period = 4 * mpm.pi if double_cover else 2 * mpm.pi
    ng = 2 * nang if double_cover else nang
    tot = mpm.mpf(0)
    ms = [mpm.mpf(twoj) / 2 - i for i in range(twoj + 1)]
    n = twoj + 1
    for t, wt in zip(nodes, wts):
        be = mpm.acos(t)
        a, b, c, d = su2_euler(0, be, 0)
        dmat = Dw_num(a, b, c, d, twoj)
        for ia in range(nang):
            al = 2 * mpm.pi * ia / nang
            ea = [mpm.expj(-m * al) for m in ms]
            for ig in range(ng):
                ga = gamma_period * ig / ng
                eg = [mpm.expj(-m * ga) for m in ms]
                D = [[ea[r] * dmat[r][s] * eg[s] for s in range(n)] for r in range(n)]
                tot += wt * func(D)
    return tot / (2 * nang * ng)
