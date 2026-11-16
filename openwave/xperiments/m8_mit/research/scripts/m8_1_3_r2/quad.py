"""High-precision route: Haar averages over SU(2) by a product rule that is exact for the
polynomial integrands used here.

SU(2) element U = [[a, -conj(c)], [c, conj(a)]], a = sqrt(t) e^{i x1}, c = sqrt(1-t) e^{i x2};
normalized Haar measure = dt dx1 dx2 / (2 pi)^2, t uniform on [0, 1].
An integrand of bidegree (d, d) in (a, c; conj a, conj c) has phases |p - p'| <= d in each angle,
so an N-point trapezoid rule with N > d is exact there; what remains is a polynomial of degree d
in t, integrated exactly by Gauss-Legendre with ceil((d+1)/2) nodes.
|phi_w|^4 at level n has d = 4n: D(U) is a polynomial of degree n in (a, c, conj a, conj c)
jointly, so |phi|^2 has total degree 2n and |phi|^4 total degree 4n; after the angle rules the t-degree is <= 2n.

All index vectors here use k = j + m (k = 0 is m = -j).
"""
import mpmath as mpm
from lib import Mp


def gauss_legendre01(n):
    xs, ws = [], []
    # nodes of P_n on [-1, 1] via mpmath polyroots-free Newton iteration
    for i in range(1, n + 1):
        x = mpm.cos(mpm.pi * (i - mpm.mpf(1) / 4) / (n + mpm.mpf(1) / 2))
        for _ in range(100):
            p0, p1 = mpm.mpf(1), x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < mpm.mpf(10) ** (-mpm.mp.dps - 5):
                break
        p0, p1 = mpm.mpf(1), x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1)
        xs.append((1 + x) / 2); ws.append(1 / ((1 - x * x) * dp * dp))  # weights sum to 1 on [0,1]
    return xs, ws


def haar_D(n, n_ang, n_t):
    """List of (weight, D_w matrix) with D_w = N M N^-1 on Condon-Shortley components."""
    Nk = [mpm.sqrt(mpm.factorial(k) * mpm.factorial(n - k)) for k in range(n + 1)]
    ts, tw = gauss_legendre01(n_t)
    out = []
    zero, one = mpm.mpc(0), mpm.mpc(1)
    for t, wt in zip(ts, tw):
        ra, rc = mpm.sqrt(t), mpm.sqrt(1 - t)
        for i1 in range(n_ang):
            e1 = mpm.expjpi(mpm.mpf(2 * i1) / n_ang)
            for i2 in range(n_ang):
                e2 = mpm.expjpi(mpm.mpf(2 * i2) / n_ang)
                a = ra * e1; c = rc * e2
                U = (a, -mpm.conj(c), c, mpm.conj(a))
                M = Mp(n, U, zero, one)
                D = [[Nk[r] * M[r][s] / Nk[s] for s in range(n + 1)] for r in range(n + 1)]
                out.append((wt / (n_ang * n_ang), D))
    return out


def averages(nodes, eta, ws):
    """For each w (list over k), Haar averages of |phi_w|^2 and |phi_w|^4,
    phi_w(g)_a = sum_{k,k'} w_k D_{k k'}(g) eta_{k' a}."""
    n1 = len(eta); d = len(eta[0])
    res = []
    for w in ws:
        s2 = mpm.mpf(0); s4 = mpm.mpf(0)
        for wt, D in nodes:
            row = [mpm.fsum(w[k] * D[k][kp] for k in range(n1)) for kp in range(n1)]
            ph = [mpm.fsum(row[kp] * eta[kp][a] for kp in range(n1)) for a in range(d)]
            m2 = mpm.fsum(abs(x) ** 2 for x in ph)
            s2 += wt * m2; s4 += wt * m2 * m2
        res.append((s2, s4))
    return res
