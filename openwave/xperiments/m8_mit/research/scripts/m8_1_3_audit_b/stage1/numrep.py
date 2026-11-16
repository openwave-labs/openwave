"""Numerical (mpmath) versions of the representation matrices, for checks and for item 0(c), 0(e)."""
from math import comb
from mpmath import mp, matrix
from core import mlist, Fm, k_to_complex, D_mono, U_of


def D_ortho_num(a, b, j2):
    """D^j(U) in the orthonormal CS basis for U = [[a, -conj b],[b, conj a]] (a, b mpc)."""
    n = j2 + 1
    ms = mlist(j2)
    M = matrix(n, n)
    nb = -mp.conj(b); ca = mp.conj(a)
    for ip in range(n):
        alpha = j2 - ip; beta = ip
        for t1 in range(alpha + 1):
            c1 = comb(alpha, t1) * a ** t1 * b ** (alpha - t1)
            for t2 in range(beta + 1):
                c2 = comb(beta, t2) * nb ** t2 * ca ** (beta - t2)
                i = j2 - (t1 + t2)
                M[i, ip] += c1 * c2
    for i in range(n):
        for k in range(n):
            M[i, k] *= mp.sqrt(mp.mpf(Fm(j2, ms[i])) / Fm(j2, ms[k]))
    return M


def D_of_q(q, j2):
    a, b = U_of(q)
    ar, ai = k_to_complex(a, mp); br, bi = k_to_complex(b, mp)
    return D_ortho_num(mp.mpc(ar, ai), mp.mpc(br, bi), j2)


def P_ortho_num(Pmono, j2):
    n = j2 + 1
    ms = mlist(j2)
    M = matrix(n, n)
    for i in range(n):
        for k in range(n):
            re, im = k_to_complex(Pmono[i][k], mp)
            M[i, k] = mp.mpc(re, im) * mp.sqrt(mp.mpf(Fm(j2, ms[i])) / Fm(j2, ms[k]))
    return M


def W_theta(j2):
    """Theta u = W conj(u), W[mu, m] = (-1)^{j-mu} delta_{mu,-m}"""
    n = j2 + 1
    ms = mlist(j2)
    W = matrix(n, n)
    for a, mu in enumerate(ms):
        W[a, ms.index(-mu)] = (-1) ** int(ms[0] - mu)
    return W


def random_su2(rng):
    v = [mp.mpf(rng.gauss(0, 1)) for _ in range(4)]
    nv = mp.sqrt(sum(x * x for x in v))
    v = [x / nv for x in v]
    return mp.mpc(v[0], v[1]), mp.mpc(v[2], v[3])


def conjm(M):
    return M.apply(mp.conj)


def orthonormal_range(P, tol=None):
    """Gram-Schmidt on the columns of P (in order), keeping independent ones."""
    n = P.rows
    tol = tol or mp.mpf(10) ** (-mp.dps // 2)
    cols = []
    for k in range(n):
        v = P[:, k]
        for c in cols:
            v = v - c * (c.H * v)[0]
        nv = mp.sqrt(mp.re((v.H * v)[0]))
        if nv > tol:
            cols.append(v / nv)
    E = matrix(n, len(cols))
    for a, c in enumerate(cols):
        for i in range(n):
            E[i, a] = c[i]
    return E
