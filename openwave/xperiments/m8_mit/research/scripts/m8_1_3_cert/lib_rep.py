"""Representations of Gamma_0 on V_j: exact D^j (weighted basis) for all 120 elements,
exact isotypic projectors from the derived characters, and a numeric orthonormal intertwiner."""
from fractions import Fraction as Fr
from math import factorial
import mpmath as mpm
from lib_alg import (K, Q5, generators, closure, conjugacy_classes, element_order, Dp_exact,
                     matmulK, weights, qmul)
from s0a_group import derive_character_table

SLOTS = {"R2": (2, 7), "R4": (3, 6), "R5": (4, 6)}

_cache = {}


def group_data():
    if "g" in _cache:
        return _cache["g"]
    G = closure(list(generators()))
    classes = conjugacy_classes(G)
    classes.sort(key=lambda c: (element_order(c[0]), -float(c[0][0])))
    irr, sizes, reps = derive_character_table(G, classes)
    cls_of = {}
    for k, c in enumerate(classes):
        for g in c:
            cls_of[g] = k
    _cache["g"] = (G, classes, irr, sizes, reps, cls_of)
    return _cache["g"]


def irrep_char(name):
    """character (list over classes) of the slot `name`, identified by (dim, lowest level)."""
    from lib_alg import cheb_U, inner
    G, classes, irr, sizes, reps, cls_of = group_data()
    d, n = SLOTS[name]
    hits = []
    for chi in irr:
        if int(chi[0].a) != d:
            continue
        low = None
        for l in range(0, 40):
            ch = [cheb_U(l, r[0]) for r in reps]
            if inner(ch, chi, sizes, len(G)) != Q5(0):
                low = l
                break
        if low == n:
            hits.append(chi)
    assert len(hits) == 1
    return hits[0]


def all_D(twoj):
    key = ("D", twoj)
    if key not in _cache:
        G = group_data()[0]
        _cache[key] = [Dp_exact(g, twoj) for g in G]
    return _cache[key]


def projector_p(name):
    """P_p = (dim/|G|) sum_g conj(chi(g)) M(g), on weighted coordinates (exact, in K)."""
    G, classes, irr, sizes, reps, cls_of = group_data()
    d, n = SLOTS[name]
    chi = irrep_char(name)
    Ds = all_D(n)
    N = n + 1
    P = [[K(0)] * N for _ in range(N)]
    for g, M in zip(G, Ds):
        c = chi[cls_of[g]]  # real (checked in s0a: characters are real-valued)
        for r in range(N):
            for s in range(N):
                if not M[r][s].is_zero():
                    P[r][s] = P[r][s] + M[r][s] * c
    f = Fr(d, len(G))
    return [[P[r][s] * f for s in range(N)] for r in range(N)]


def to_w_num(Pp, twoj):
    F = weights(twoj)
    N = twoj + 1
    return [[Pp[r][s].mpc(mpm) * mpm.sqrt(mpm.mpf(F[r]) / F[s]) for s in range(N)] for r in range(N)]


def eta_num(name):
    """Orthonormal basis of the rho-isotypic subspace in the CS basis: Gram-Schmidt of the
    columns of P_w taken in the order m = j, j-1, ..., -j; returns N x dim list."""
    d, n = SLOTS[name]
    Pw = to_w_num(projector_p(name), n)
    N = n + 1
    cols = []
    for s in range(N):
        v = [Pw[r][s] for r in range(N)]
        for c in cols:
            ip = mpm.fsum(mpm.conj(c[r]) * v[r] for r in range(N))
            v = [v[r] - ip * c[r] for r in range(N)]
        nr = mpm.sqrt(mpm.fsum(abs(x) ** 2 for x in v))
        if nr > mpm.mpf(10) ** (-mpm.mp.dps // 2):
            cols.append([x / nr for x in v])
        if len(cols) == d:
            break
    assert len(cols) == d
    return [[cols[a][r] for a in range(d)] for r in range(N)]
