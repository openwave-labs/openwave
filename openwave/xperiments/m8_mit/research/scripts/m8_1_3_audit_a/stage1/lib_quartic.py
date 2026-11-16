"""Exact channel coefficients and the quartic h in weighted coordinates.
q_J = tr((conj P (x) conj P) Pi_J) / ((2J+1) s_N^2), from Schur's lemma applied to the Haar average of
|phi_w|^4 = <u(x)u, (conj P (x) conj P) u(x)u>, u = conj(D(g)) w; every trace is taken in the weighted basis,
where P (exact isotypic projector) and Pi_J (Casimir polynomial) have entries in Q(sqrt5) and Q."""
from lib_rep import *


def channel_coeffs(slot, cache={}):
    if slot in cache: return cache[slot]
    name, j2, d = slot_info(slot)
    n = j2 + 1
    P = projector_w(name, j2)
    Pb = [[x.conj() for x in row] for row in P]
    proj = casimir_projectors_w(j2)
    sN = Fr(d, j2 + 1)
    q = {}
    for J2 in range(0, 2 * j2 + 1, 2):
        tr = ZERO
        for (r, c), v in proj[J2].items():
            a, b = divmod(r, n); cc, dd = divmod(c, n)
            # (Pb (x) Pb)[(cc,dd),(a,b)] * Pi[(a,b),(cc,dd)]
            tr = tr + Pb[cc][a] * Pb[dd][b] * v
        val = tr * Fr(1, J2 + 1) / (sN * sN)
        q[J2] = val
    cache[slot] = (q, proj)
    return q, proj


def sym_J2s(j2):
    """2J for the channels in Sym^2 V_j: J = 2j, 2j-2, ..."""
    return list(range(2 * j2, -1, -4))
