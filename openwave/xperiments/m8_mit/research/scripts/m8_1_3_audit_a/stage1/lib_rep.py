"""Representation helpers: exact isotypic projectors in the weighted basis, Casimir projectors Pi_J,
numerical (mpmath) CS-basis objects. All derived here."""
from lib_common import *
import mpmath

_G = None


def group():
    global _G
    if _G is None:
        _G = build_group()
    return _G


def class_char(name):
    gj = load_json('group.json')
    ch = [K.real(Fr(a), Fr(b)) for a, b in gj['irr'][name]]
    res = [K.real(Fr(a), Fr(b)) for a, b in (c['re'] for c in gj['classes'])]
    return dict(zip(res, ch))  # classes of 2I are determined by Re(g) (checked in s01: 9 distinct values)


def slot_info(slot):
    gj = load_json('group.json')
    name = gj['slots'][slot]
    j2 = {'R2': 7, 'R4': 6, 'R5': 6}[slot]
    dim = int(Fr(gj['irr'][name][0][0]))
    return name, j2, dim


def projector_w(name, j2):
    """exact P_w = (d/|G|) sum_h conj(chi(h)) Dw(h), the rho-isotypic projector in weighted coordinates"""
    chi = class_char(name)
    G = group()
    n = j2 + 1
    d = list(chi.values())  # not used
    P = [[ZERO] * n for _ in range(n)]
    for h in G:
        c = chi[h[0]].conj()
        if c.iszero(): continue
        D = Dw(quat_to_U(h), j2)
        for a in range(n):
            for b in range(n):
                if not D[a][b].iszero():
                    P[a][b] = P[a][b] + c * D[a][b]
    dim = chi[ONE]
    f = dim * Fr(1, len(G))
    return [[f * x for x in row] for row in P]


def svec(j2):
    """s_m^2 list in descending m"""
    return [s2(j2, m2) for m2 in ms(j2)]


# ---- spin operators in the weighted basis (exact rationals) ----
def spin_ops_w(j2):
    ml = ms(j2); n = j2 + 1
    Jp = [[Fr(0)] * n for _ in range(n)]; Jm = [[Fr(0)] * n for _ in range(n)]; Jz = [[Fr(0)] * n for _ in range(n)]
    for c, m2 in enumerate(ml):
        A = (j2 + m2) // 2; B = (j2 - m2) // 2  # x^A y^B
        Jz[c][c] = Fr(m2, 2)
        if B > 0: Jp[c - 1][c] = Fr(B)      # x d/dy : -> B x^{A+1} y^{B-1}, index m+1 is c-1
        if A > 0: Jm[c + 1][c] = Fr(A)      # y d/dx
    return Jp, Jm, Jz


def casimir_projectors_w(j2):
    """Pi_J on V_j (x) V_j in the weighted product basis (index a*n+b), exact, by M-blocks.
    returns dict J2 -> dict {(row, col): Fraction}"""
    import sympy as sp
    n = j2 + 1; ml = ms(j2)
    Jp, Jm, Jz = spin_ops_w(j2)
    # total operators act on pairs; build per M block
    blocks = {}
    for a in range(n):
        for b in range(n):
            blocks.setdefault(ml[a] + ml[b], []).append((a, b))
    def tot(op, a, b):  # op (x) 1 + 1 (x) op applied to basis (a, b): dict {(a', b'): coeff}
        r = {}
        for a2 in range(n):
            if op[a2][a] != 0: r[(a2, b)] = r.get((a2, b), 0) + op[a2][a]
        for b2 in range(n):
            if op[b2][b] != 0: r[(a, b2)] = r.get((a, b2), 0) + op[b2][b]
        return r
    def apply(op, vec):
        out = {}
        for (a, b), c in vec.items():
            for k, v in tot(op, a, b).items():
                out[k] = out.get(k, 0) + c * v
        return out
    Js = list(range(0, 2 * j2 + 1, 2))  # 2J values 0..4j
    proj = {J2: {} for J2 in Js}
    for M2, basis in blocks.items():
        idx = {ab: i for i, ab in enumerate(basis)}
        nb = len(basis)
        C = sp.zeros(nb, nb)
        for i, ab in enumerate(basis):
            v = {ab: Fr(1)}
            # C = Jz^2 + (J+J- + J-J+)/2
            vz = apply(Jz, apply(Jz, v))
            v1 = apply(Jp, apply(Jm, v)); v2 = apply(Jm, apply(Jp, v))
            for k, c in vz.items(): C[idx[k], i] += sp.Rational(c)
            for k, c in v1.items(): C[idx[k], i] += sp.Rational(c) / 2
            for k, c in v2.items(): C[idx[k], i] += sp.Rational(c) / 2
        present = [J2 for J2 in Js if J2 >= abs(M2)]
        for J2 in present:
            lam = sp.Rational(J2 * (J2 + 2), 4)
            Pm = sp.eye(nb)
            for J2b in present:
                if J2b == J2: continue
                lb = sp.Rational(J2b * (J2b + 2), 4)
                Pm = Pm * (C - lb * sp.eye(nb)) / (lam - lb)
            for i in range(nb):
                for k in range(nb):
                    if Pm[i, k] != 0:
                        a, b = basis[i]; c, d = basis[k]
                        proj[J2][(a * n + b, c * n + d)] = Fr(int(sp.numer(Pm[i, k])), int(sp.denom(Pm[i, k])))
    return proj
