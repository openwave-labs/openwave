"""Group, exact character table, decompositions and the slot projectors, all derived from the generators."""
from fractions import Fraction as Fr
import functools
from core import *

SLOTS = {'R2': (2, 7), 'R4': (3, 6), 'R5': (4, 6)}   # name -> (dim rho, lowest level n)


@functools.lru_cache(None)
def group_data():
    G = generate_group()
    classes, idx = conj_classes(G)
    # order classes by (element order, real part) for a stable presentation
    classes.sort(key=lambda cl: (element_order(cl[0]), float(cl[0][0])))
    chars_num, cls_of, a = character_table(G, classes, idx)
    chars = []
    for chi in chars_num:
        chars.append([identify_zphi(complex(v)) for v in chi])
    # sort irreps by degree then by the value on the order-10 class with real part phi/2
    chars.sort(key=lambda c: (float(c[0]), [float(v) for v in c]))
    return G, classes, cls_of, chars


def class_of(q):
    G, classes, cls_of, chars = group_data()
    return cls_of[qkey(q)]


def inner_classfn(f1, f2):
    """(1/|G|) sum_g f1(g) conj f2(g) for real class functions given per class (Q5 values)."""
    G, classes, cls_of, chars = group_data()
    s = Q5(0)
    for r, cl in enumerate(classes):
        s = s + f1[r] * f2[r] * len(cl)
    return s / len(G)


def Vj_character(j2):
    G, classes, cls_of, chars = group_data()
    return [chebU(j2, cl[0][0]) for cl in classes]


def multiplicity(j2, irr):
    G, classes, cls_of, chars = group_data()
    m = inner_classfn(Vj_character(j2), chars[irr])
    assert m.b == 0 and m.a.denominator == 1, m
    return int(m.a)


def irrep_levels(irr, lmax=20):
    return [(l, multiplicity(l, irr)) for l in range(lmax + 1) if multiplicity(l, irr) > 0]


@functools.lru_cache(None)
def slot_irrep(name):
    """the unique irrep of the given dimension whose lowest level is the given n."""
    G, classes, cls_of, chars = group_data()
    d, n = SLOTS[name]
    hits = []
    for i, c in enumerate(chars):
        if c[0] == Q5(d):
            lv = irrep_levels(i, 30)
            if lv and lv[0][0] == n:
                hits.append(i)
    assert len(hits) == 1, (name, hits)
    return hits[0]


@functools.lru_cache(None)
def projector_mono(name):
    """P = (d/|G|) sum_h chi(h) D(h) in the monomial basis (K-valued, exact). chi is real for this group."""
    G, classes, cls_of, chars = group_data()
    d, n = SLOTS[name]
    irr = slot_irrep(name)
    chi = chars[irr]
    N = n + 1
    P = [[KZERO] * N for _ in range(N)]
    for g in G:
        c = chi[cls_of[qkey(g)]]
        if c.iszero():
            continue
        Dm = D_mono(g, n)
        ck = K(c)
        for i in range(N):
            for k in range(N):
                if not kiszero(Dm[i][k]):
                    P[i][k] = kadd(P[i][k], kmul(ck, Dm[i][k]))
    fac = Fr(d, len(G))
    return tuple(tuple(kscale(x, fac) for x in row) for row in P)


def s_N(name):
    d, n = SLOTS[name]
    return Fr(d, n + 1)
