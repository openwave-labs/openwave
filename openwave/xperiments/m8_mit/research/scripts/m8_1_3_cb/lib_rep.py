"""Spin representations in this room's convention, the group elements, projectors.

D^j(U)_{m m'} = <j m| U |j m'> with U = [[a, -conj b], [b, conj a]] acting on (x, y) by
x -> a x + b y, y -> -conj(b) x + conj(a) y, extended to degree-2j polynomials; basis
|j m> = x^(j+m) y^(j-m)/sqrt((j+m)!(j-m)!).  R_z(theta) = diag(e^{-i theta/2}, e^{i theta/2})
gives D = diag(e^{-i m theta}).  Index order everywhere: m = j, j-1, ..., -j.
"""
import json, os
from fractions import Fraction as Fr
from math import comb, factorial
from lib_exact import Q5, Q5i, PHI, qmul, qkey, quat_to_ab
from lib_cg import mrange, w as wfac

HERE = os.path.dirname(os.path.abspath(__file__))


def group_elements():
    half = Fr(1, 2)
    q1 = (Q5(half), Q5(half), Q5(half), Q5(half))
    q2 = (PHI * half, (PHI - 1) * half, Q5(half), Q5(0))
    one = (Q5(1), Q5(0), Q5(0), Q5(0))
    elems = {qkey(one): one}
    frontier = [one]
    while frontier:
        new = []
        for g in frontier:
            for s in (q1, q2):
                h = qmul(g, s)
                if qkey(h) not in elems:
                    elems[qkey(h)] = h
                    new.append(h)
        frontier = new
    return list(elems.values()), q1, q2


def slot_character(slot):
    """character of the slot as a function of the real part of the quaternion (classes are
    separated by real part: checked in s00).  Returns dict {(a, b) of real part: Q5}."""
    g = json.load(open(os.path.join(HERE, "out", "group.json")))
    idx = g["slot_index"][slot]
    irr = [d for d in g["irreducibles"] if d["index"] == idx][0]
    import sympy as sp
    out = {}
    for wstr, cstr in zip(g["class_real_parts"], irr["character_sympy"]):
        wv, cv = sp.sympify(wstr), sp.sympify(cstr)
        def toq5(e):
            e = sp.expand(e)
            b = e.coeff(sp.sqrt(5))
            a = sp.expand(e - b * sp.sqrt(5))
            return Q5(Fr(str(a)), Fr(str(b)))
        out[(toq5(wv).a, toq5(wv).b)] = toq5(cv)
    return out, irr["dim"]


def poly_D_entry_coeffs(j, m, mp_):
    """coefficients of x^(j+m) y^(j-m) in (a x + b y)^(j+mp) (-cb x + ca y)^(j-mp), as a list of
    (int coeff, (ea, eb, eca, ecb)) exponent tuples of a, b, conj a, conj b."""
    p, q = int(j + mp_), int(j - mp_)
    target = int(j + m)
    out = []
    for k in range(0, p + 1):  # from (a x + b y)^p choose x^k: a^k b^(p-k)
        l = target - k  # from (-cb x + ca y)^q choose x^l: (-cb)^l ca^(q-l)
        if l < 0 or l > q:
            continue
        c = comb(p, k) * comb(q, l) * (-1) ** l
        out.append((c, (k, p - k, q - l, l)))
    return out


def D_mono_exact(j, a, b):
    """matrix (list of lists) of coefficients in the monomial basis, entries Q5i, for exact a, b."""
    ca, cb = a.cc(), b.cc()
    ms = mrange(j)
    pw = {}

    def powr(x, n, key):
        k = (key, n)
        if k not in pw:
            r = Q5i(1)
            for _ in range(n):
                r = r * x
            pw[k] = r
        return pw[k]
    M = []
    for m in ms:
        row = []
        for mp_ in ms:
            s = Q5i(0)
            for c, (ea, eb, eca, ecb) in poly_D_entry_coeffs(j, m, mp_):
                s = s + powr(a, ea, 'a') * powr(b, eb, 'b') * powr(ca, eca, 'ca') * powr(cb, ecb, 'cb') * c
            row.append(s)
        M.append(row)
    return M


def D_mp(j, a, b, mp):
    """orthonormal Condon-Shortley D^j(U) at mp precision (a, b mp complex)."""
    ms = mrange(j)
    ca, cb = mp.conj(a), mp.conj(b)
    n = len(ms)
    D = mp.matrix(n, n)
    for r, m in enumerate(ms):
        for c, mp_ in enumerate(ms):
            s = mp.mpc(0)
            for co, (ea, eb, eca, ecb) in poly_D_entry_coeffs(j, m, mp_):
                s += co * a ** ea * b ** eb * ca ** eca * cb ** ecb
            D[r, c] = s * mp.sqrt(mp.mpf(wfac(j, m)) / wfac(j, mp_))
    return D


def theta_apply(j, u, mp):
    """(Theta u)_m = (-1)^(j-m) conj(u_{-m}), u an mp column (index order m = j..-j)."""
    ms = mrange(j)
    n = len(ms)
    out = mp.matrix(n, 1)
    for r, m in enumerate(ms):
        out[r] = (-1) ** int(j - m) * mp.conj(u[n - 1 - r])
    return out


def Jmats(j, mp):
    """J_z, J_+, J_-, J_x, J_y in the orthonormal basis (CS)."""
    ms = mrange(j)
    n = len(ms)
    Jz, Jp, Jm = mp.matrix(n, n), mp.matrix(n, n), mp.matrix(n, n)
    for r, m in enumerate(ms):
        Jz[r, r] = m
    for c, m in enumerate(ms):
        if m < j:  # J_+ |m> = sqrt((j-m)(j+m+1)) |m+1>, row index of m+1 is c-1
            Jp[c - 1, c] = mp.sqrt((j - m) * (j + m + 1))
        if m > -j:
            Jm[c + 1, c] = mp.sqrt((j + m) * (j - m + 1))
    Jx = (Jp + Jm) / 2
    Jy = (Jp - Jm) / (2 * mp.mpc(0, 1))
    return Jz, Jp, Jm, Jx, Jy
