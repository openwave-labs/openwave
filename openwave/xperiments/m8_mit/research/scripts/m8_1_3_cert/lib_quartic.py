"""The reduced quartic h(w) = Q(w)||w||^4 of a slot at its lowest level, exactly.

  Q(w) ||w||^4 = avg_g |phi_w(g)|^4 / s_N^2,   phi_w(g) = w^T D(g) eta,
  avg_g |phi_w|^4 = (w (x) w)^T [ avg_g (D(x)D)(P(x)P)(D(x)D)^dagger ] (wbar (x) wbar)
                  = sum_J c_J ||[w (x) w]_J||^2,   c_J = tr(Pi_J (P (x) P)) / (2J+1)
(Schur: V_j (x) V_j is multiplicity-free).  Pi_J are built from the Casimir in the weighted
basis (rational), so everything is exact.  Then h is expanded as a real polynomial with
rational coefficients in the 14 chart coordinates x = (Re p_3..Re p_-3, Im p_3..Im p_-3)."""
from fractions import Fraction as Fr
import json, os
from lib_alg import K, Q5, weights
from lib_rep import projector_p, SLOTS

NV = 14


def casimir_projectors(twoj):
    j = Fr(twoj, 2)
    N = twoj + 1
    ms = [j - i for i in range(N)]
    # single-space generators on weighted coordinates (row = new index)
    Jz = {(i, i): ms[i] for i in range(N)}
    Jp = {}
    Jm = {}
    for i, m in enumerate(ms):
        if m < j:   # p_m contributes (j-m) p_m to coefficient m+1 (row i-1)
            Jp[(i - 1, i)] = j - m
        if m > -j:  # contributes (j+m) p_m to m-1 (row i+1)
            Jm[(i + 1, i)] = j + m
    NN = N * N

    def tens(A):
        T = {}
        for (r, s), v in A.items():
            for k in range(N):
                T[(r * N + k, s * N + k)] = T.get((r * N + k, s * N + k), 0) + v
                T[(k * N + r, k * N + s)] = T.get((k * N + r, k * N + s), 0) + v
        return T

    def smul(A, B):
        Bi = {}
        for (r, s), v in B.items():
            Bi.setdefault(r, []).append((s, v))
        C = {}
        for (r, k), v in A.items():
            for s, w in Bi.get(k, []):
                C[(r, s)] = C.get((r, s), 0) + v * w
        return {k: v for k, v in C.items() if v != 0}

    Tz, Tp, Tm = tens(Jz), tens(Jp), tens(Jm)
    C = smul(Tm, Tp)
    for k, v in smul(Tz, Tz).items():
        C[k] = C.get(k, 0) + v
    for k, v in Tz.items():
        C[k] = C.get(k, 0) + v
    Js = [Fr(J) for J in range(0, twoj + 1)]  # J ranges over 0..2j (integers) for j1 = j2 = j
    proj = {}
    for J in Js:
        P = {(i, i): Fr(1) for i in range(NN)}
        for Jq in Js:
            if Jq == J:
                continue
            Cs = dict(C)
            Cs2 = {}
            for k, v in Cs.items():
                Cs2[k] = v
            for i in range(NN):
                Cs2[(i, i)] = Cs2.get((i, i), 0) - Jq * (Jq + 1)
            den = J * (J + 1) - Jq * (Jq + 1)
            P = smul(P, {k: v / den for k, v in Cs2.items() if v != 0})
        proj[J] = P
    return proj, C


def channel_coefficients(name):
    d, n = SLOTS[name]
    N = n + 1
    P = projector_p(name)
    proj, C = casimir_projectors(n)
    cJ = {}
    for J, Pi in proj.items():
        # tr(Pi (P (x) P)) = sum Pi[(a,b),(c,d)] P[c][a] P[d][b]
        tot = K(0)
        for (r, s), v in Pi.items():
            a, b = divmod(r, N)
            c, dd = divmod(s, N)
            tot = tot + P[c][a] * P[dd][b] * v
        cJ[J] = tot * (Fr(1) / (2 * J + 1))
    return cJ, proj


def build(name, Jfilter=None):
    """returns dict: s_N, c_J (exact), q_J = c_J/s_N^2, and the real polynomial h as a
    dict {exponent tuple: Fraction}."""
    d, n = SLOTS[name]
    N = n + 1
    sN = Fr(d, N)
    cJ, proj = channel_coefficients(name)
    for J, v in cJ.items():
        assert v.im.is_zero() and v.re.b == 0, (J, v)
    c_rat = {J: v.re.a for J, v in cJ.items()}
    F = weights(n)
    # H[(a,b),(c,d)] = (1/sN^2) F_a F_b sum_J c_J Pi_J[(a,b),(c,d)]
    H = {}
    for J, Pi in proj.items():
        cj = c_rat[J]
        if cj == 0 or (Jfilter is not None and not Jfilter(J)):
            continue
        for (r, s), v in Pi.items():
            a, b = divmod(r, N)
            H[(r, s)] = H.get((r, s), 0) + cj * v * F[a] * F[b] / (sN * sN)
    H = {k: v for k, v in H.items() if v != 0}
    # p_a = X_a + i Y_a ; index of X_a = a, of Y_a = 7 + a  (a = 0..6 <-> m = 3..-3)
    def lin(a, conj=False):
        ex = [0] * NV
        ex[a] = 1
        ey = [0] * NV
        ey[N + a] = 1
        return {tuple(ex): (Fr(1), Fr(0)), tuple(ey): (Fr(0), Fr(-1) if conj else Fr(1))}

    def cmul(P, Q):
        R = {}
        for e1, (r1, i1) in P.items():
            for e2, (r2, i2) in Q.items():
                e = tuple(x + y for x, y in zip(e1, e2))
                r, i = R.get(e, (Fr(0), Fr(0)))
                R[e] = (r + r1 * r2 - i1 * i2, i + r1 * i2 + i1 * r2)
        return R

    pp = {}
    ppc = {}
    for a in range(N):
        for b in range(N):
            pp[(a, b)] = cmul(lin(a), lin(b))
            ppc[(a, b)] = cmul(lin(a, True), lin(b, True))
    poly = {}
    imag = {}
    for (r, s), v in H.items():
        a, b = divmod(r, N)
        c, dd = divmod(s, N)
        prod = cmul(pp[(a, b)], ppc[(c, dd)])
        for e, (re, im) in prod.items():
            poly[e] = poly.get(e, 0) + v * re
            imag[e] = imag.get(e, 0) + v * im
    poly = {e: c for e, c in poly.items() if c != 0}
    imag = {e: c for e, c in imag.items() if c != 0}
    return {"s_N": sN, "c_J": c_rat, "q_J": {J: c / (sN * sN) for J, c in c_rat.items()},
            "poly": poly, "imag_residue_terms": len(imag), "H": H}


def save(name, path):
    B = build(name)
    out = {"slot": name, "s_N": str(B["s_N"]),
           "c_J": {str(J): str(v) for J, v in B["c_J"].items()},
           "q_J": {str(J): str(v) for J, v in B["q_J"].items()},
           "imag_residue_terms": B["imag_residue_terms"],
           "poly": [[list(e), str(c)] for e, c in sorted(B["poly"].items())]}
    with open(path, "w") as f:
        json.dump(out, f)
    return B


def load(path):
    with open(path) as f:
        d = json.load(f)
    return {tuple(e): Fr(c) for e, c in d["poly"]}, d


# ---------------------------------------------------------------- polynomial utilities
def pdiff(poly, k):
    out = {}
    for e, c in poly.items():
        if e[k] == 0:
            continue
        e2 = list(e)
        e2[k] -= 1
        out[tuple(e2)] = out.get(tuple(e2), 0) + c * e[k]
    return {e: c for e, c in out.items() if c != 0}


class Compiled:
    """a polynomial with coefficients converted once by conv (e.g. exact interval enclosure
    of each rational coefficient); eval(x) works for floats, mpf or iv.mpf entries."""

    def __init__(self, poly, conv):
        self.terms = [(conv(c), [(i, k) for i, k in enumerate(e) if k]) for e, c in poly.items()]

    def eval(self, x, zero):
        pw = {}
        tot = zero
        for c, fac in self.terms:
            t = c
            for i, k in fac:
                key = (i, k)
                if key not in pw:
                    v = x[i]
                    for _ in range(k - 1):
                        v = v * x[i]
                    pw[key] = v
                t = t * pw[key]
            tot = tot + t
        return tot
