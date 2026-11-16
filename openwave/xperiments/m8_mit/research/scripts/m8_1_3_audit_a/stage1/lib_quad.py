"""Exact-degree quadrature for Haar averages over SU(2) of |phi_w|^2 and |phi_w|^4, in mpmath.
Hopf coordinates: U = [[z1, z2], [-conj z2, conj z1]], z1 = sqrt(t) e^{i a}, z2 = sqrt(1-t) e^{i b};
Haar measure = (dt/1) (da/2pi) (db/2pi), t uniform on [0, 1]. |phi|^4 is a polynomial of bidegree (4j, 4j)
in (z, conj z): a trapezoid rule with N > 8j nodes in each phase is exact, and the surviving terms are
polynomials of degree <= 4j in t, integrated exactly by Gauss-Legendre with ceil((4j+1)/2) nodes."""
import mpmath
from lib_common import ms, s2


def D_cs_num(U, j2):
    """numerical CS-basis D^j(U), (D f)(v) = f(U^T v), same convention as lib_common.Dw"""
    from math import comb
    ml = ms(j2); n = j2 + 1
    M = [[mpmath.mpc(0)] * n for _ in range(n)]
    sq = [mpmath.sqrt(s2(j2, m2)) for m2 in ml]
    for col, m2 in enumerate(ml):
        A = (j2 + m2) // 2; B = (j2 - m2) // 2
        p1 = [comb(A, k) * U[0][0] ** k * U[1][0] ** (A - k) for k in range(A + 1)]
        p2 = [comb(B, k) * U[0][1] ** k * U[1][1] ** (B - k) for k in range(B + 1)]
        for k1 in range(A + 1):
            for k2 in range(B + 1):
                row = ml.index(2 * (k1 + k2) - j2)
                M[row][col] += p1[k1] * p2[k2]
    for r in range(n):
        for c in range(n):
            M[r][c] = M[r][c] * sq[r] / sq[c]
    return M


def nodes(j2):
    N = 4 * j2 + 1  # > 8j
    ng = (2 * j2 + 2) // 2 + 1  # degree 4j = 2*j2 in t needs n >= (2*j2+1)/2
    xs, ws = gauss_legendre01(ng)
    out = []
    for t, wt in zip(xs, ws):
        r1 = mpmath.sqrt(t); r2 = mpmath.sqrt(1 - t)
        for ka in range(N):
            ea = mpmath.expjpi(mpmath.mpf(2 * ka) / N)
            for kb in range(N):
                eb = mpmath.expjpi(mpmath.mpf(2 * kb) / N)
                z1 = r1 * ea; z2 = r2 * eb
                U = [[z1, z2], [-mpmath.conj(z2), mpmath.conj(z1)]]
                out.append((U, wt / (N * N)))
    return out


def gauss_legendre01(n):
    # nodes/weights on [0,1] from Legendre polynomial roots (Newton on P_n), derived here
    xs = []; ws = []
    for i in range(1, n + 1):
        x = mpmath.cos(mpmath.pi * (i - mpmath.mpf(1) / 4) / (n + mpmath.mpf(1) / 2))
        for _ in range(100):
            p0, p1 = mpmath.mpf(1), x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < mpmath.mpf(10) ** (-mpmath.mp.dps - 5): break
        p0, p1 = mpmath.mpf(1), x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1)
        xs.append((1 + x) / 2); ws.append(1 / ((1 - x * x) * dp * dp))  # weight on [-1,1] is 2/((1-x^2)dp^2); /2 for [0,1]
    return xs, ws


def haar_avg_moments(j2, eta, wlist):
    """returns ([avg |phi_w|^2], [avg |phi_w|^4]) over Haar measure, phi_w(g)_a = sum w_m D_{mm'}(g) eta_{m'a}"""
    n = j2 + 1; d = len(eta[0])
    m2 = [mpmath.mpf(0)] * len(wlist); m4 = [mpmath.mpf(0)] * len(wlist)
    for U, wt in nodes(j2):
        D = D_cs_num(U, j2)
        Me = [[sum(D[r][k] * eta[k][a] for k in range(n)) for a in range(d)] for r in range(n)]
        for i, w in enumerate(wlist):
            phi = [sum(w[r] * Me[r][a] for r in range(n)) for a in range(d)]
            p2 = sum(abs(x) ** 2 for x in phi)
            m2[i] += wt * p2; m4[i] += wt * p2 * p2
    return m2, m4
