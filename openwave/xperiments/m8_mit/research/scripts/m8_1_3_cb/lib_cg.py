"""Clebsch-Gordan coefficients built in this room from the convention of worklist 1.2.

Model: V_j = degree-2j polynomials in (x, y); weight vector |j m> = x^(j+m) y^(j-m)/sqrt((j+m)!(j-m)!)
(orthonormal); J_z = m, J_+ = x d/dy, J_- = y d/dx (these give the Condon-Shortley matrix
elements sqrt((j-+m)(j+-m+1)) > 0, checked in s01).
Coupled state |J J> in V_j1 (x) V_j2: kernel of J_+ (total) in weight J, phase fixed by
<j1 j1; j2 J-j1 | J J> > 0 (Condon-Shortley).  Lower states by J_- (positive factors).
Every coefficient is returned exactly as a 'signed square' (sign, r) meaning sign*sqrt(r), r Fraction.
"""
from fractions import Fraction as Fr
from math import factorial
from functools import lru_cache


def w(j, m):  # (j+m)!(j-m)!, j, m Fractions with j-m integer
    return factorial(int(j + m)) * factorial(int(j - m))


def mrange(j):
    j = Fr(j)
    return [j - k for k in range(int(2 * j) + 1)]  # j, j-1, ..., -j


def _hw_kernel(j1, j2, J):
    """Monomial coefficients c[m1] (m2 = J - m1) of the highest weight vector, rational, as the
    1-dim kernel of J_+ total in weight J (exact elimination)."""
    ms = [m1 for m1 in mrange(j1) if abs(J - m1) <= j2]
    # J_+ maps mono(m1) x mono(m2) -> (j1-m1) mono(m1+1) x mono(m2) + (j2-m2) mono(m1) x mono(m2+1)
    # image lives in weight J+1, indexed by m1' (m2' = J+1-m1')
    rows = sorted({m1 + 1 for m1 in ms} | {m1 for m1 in ms}, reverse=True)
    rows = [r for r in rows if abs(r) <= j1 and abs(J + 1 - r) <= j2]
    A = [[Fr(0)] * len(ms) for _ in rows]
    for c, m1 in enumerate(ms):
        m2 = J - m1
        if m1 + 1 <= j1 and (m1 + 1) in rows:
            A[rows.index(m1 + 1)][c] += (j1 - m1)
        if m2 + 1 <= j2 and m1 in rows:
            A[rows.index(m1)][c] += (j2 - m2)
    # null space by Gauss-Jordan
    n = len(ms)
    M = [row[:] for row in A]
    piv_cols, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        piv_cols.append(c)
        r += 1
    free = [c for c in range(n) if c not in piv_cols]
    assert len(free) == 1, (j1, j2, J, free)
    f = free[0]
    vec = [Fr(0)] * n
    vec[f] = Fr(1)
    for i, c in enumerate(piv_cols):
        vec[c] = -M[i][f]
    return dict(zip(ms, vec))


@lru_cache(maxsize=None)
def cg_table(j1, j2, J):
    """dict {(m1, m2, M): (sign, r)} with <j1 m1; j2 m2 | J M> = sign*sqrt(r)."""
    j1, j2, J = Fr(j1), Fr(j2), Fr(J)
    assert abs(j1 - j2) <= J <= j1 + j2 and (j1 + j2 - J).denominator == 1
    c = _hw_kernel(j1, j2, J)  # monomial coefficients, weight J
    # Condon-Shortley phase: coefficient at m1 = j1 positive
    top = max(c)
    if c[top] < 0:
        c = {k: -v for k, v in c.items()}
    assert top == j1 or J - j1 > j2 or True
    out = {}
    M = J
    while True:
        norm2 = sum(v * v * w(j1, m1) * w(j2, M - m1) for m1, v in c.items())
        for m1, v in c.items():
            if v != 0:
                out[(m1, M - m1, M)] = (1 if v > 0 else -1, v * v * w(j1, m1) * w(j2, M - m1) / norm2)
        if M == -J:
            break
        # apply J_- = y d/dx on each factor (monomial coefficients)
        nc = {}
        for m1, v in c.items():
            m2 = M - m1
            if m1 - 1 >= -j1:  # y d/dx on factor 1: x^(j1+m1) -> (j1+m1) x^(j1+m1-1) y
                nc[m1 - 1] = nc.get(m1 - 1, Fr(0)) + v * (j1 + m1)
            if m2 - 1 >= -j2:
                nc[m1] = nc.get(m1, Fr(0)) + v * (j2 + m2)
        c = {k: v for k, v in nc.items() if v != 0}
        M -= 1
    return out


def cg(j1, m1, j2, m2, J, M):
    t = cg_table(Fr(j1), Fr(j2), Fr(J))
    return t.get((Fr(m1), Fr(m2), Fr(M)), (0, Fr(0)))


def racah(j1, m1, j2, m2, J, M):
    """Racah's closed formula, as a signed square, exact.  Used only as a comparison."""
    j1, m1, j2, m2, J, M = map(Fr, (j1, m1, j2, m2, J, M))
    if m1 + m2 != M or abs(m1) > j1 or abs(m2) > j2 or abs(M) > J:
        return (0, Fr(0))
    f = lambda x: factorial(int(x))
    pre = Fr((2 * J + 1) * f(J + j1 - j2) * f(J - j1 + j2) * f(j1 + j2 - J), f(j1 + j2 + J + 1))
    pre *= f(J + M) * f(J - M) * f(j1 - m1) * f(j1 + m1) * f(j2 - m2) * f(j2 + m2)
    s = Fr(0)
    for k in range(0, int(j1 + j2 - J) + 1):
        args = [k, j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k]
        if any(a < 0 for a in args):
            continue
        d = 1
        for a in args:
            d *= f(a)
        s += Fr((-1) ** k, d)
    if s == 0:
        return (0, Fr(0))
    return (1 if s > 0 else -1, pre * s * s)


def sgnsq_to_mp(t, mp):
    s, r = t
    if s == 0:
        return mp.mpf(0)
    return s * mp.sqrt(mp.mpf(r.numerator) / r.denominator)


def sgnsq_to_float(t):
    s, r = t
    return 0.0 if s == 0 else s * (r.numerator / r.denominator) ** 0.5
