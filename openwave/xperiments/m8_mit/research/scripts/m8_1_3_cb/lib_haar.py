"""Route A (exact, from the definitions, no Clebsch-Gordan coefficients):
functions on SU(2) as polynomials in (a, b, conj a, conj b), U = [[a, -conj b], [b, conj a]];
Haar mean of a^p b^q conj(a)^r conj(b)^s = delta_pr delta_qs p! q!/(p+q+1)!.

Coefficients live in Z[sqrt5, i] as 4-tuples of ints (x0, x1, y0, y1) = (x0 + x1 sqrt5) + i (y0 + y1 sqrt5);
rational scale factors are carried separately as Fractions.

For a section psi_y = s_N^(-1/2) y^T D(g) eta with y given by weighted coordinates p (y_m = p_m sqrt(w_m)):
  y^T D_o P_o = A S^-1 with A = (p w)^T D_mono P_mono,   |psi|^2 = s^-1 A W^-1 A^dag,
  N(psi) eta^dag = s^-3/2 (A W^-1 A^dag) A S^-1, whose pointwise norm equals |N(psi)|.
Level-J norm of a scalar f: (2J+1) sum_{M M'} (w_M/w_M') |mean(f conj Dmono^J_{MM'})|^2.
"""
from fractions import Fraction as Fr
from math import factorial, lcm
from lib_rep import poly_D_entry_coeffs
from lib_cg import mrange, w as wfac

Z = (0, 0, 0, 0)


def zadd(u, v):
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2], u[3] + v[3])


def zmul(u, v):
    a0, a1, b0, b1 = u
    c0, c1, d0, d1 = v
    # (A + iB)(C + iD), A = a0 + a1 r, r^2 = 5
    AC0, AC1 = a0 * c0 + 5 * a1 * c1, a0 * c1 + a1 * c0
    BD0, BD1 = b0 * d0 + 5 * b1 * d1, b0 * d1 + b1 * d0
    AD0, AD1 = a0 * d0 + 5 * a1 * d1, a0 * d1 + a1 * d0
    BC0, BC1 = b0 * c0 + 5 * b1 * c1, b0 * c1 + b1 * c0
    return (AC0 - BD0, AC1 - BD1, AD0 + BC0, AD1 + BC1)


def zscale(u, k):
    return (u[0] * k, u[1] * k, u[2] * k, u[3] * k)


def zconj(u):
    return (u[0], u[1], -u[2], -u[3])


def padd(P, Q):
    R = dict(P)
    for k, v in Q.items():
        R[k] = zadd(R[k], v) if k in R else v
    return {k: v for k, v in R.items() if v != Z}


def pmul(P, Q):
    R = {}
    for k1, v1 in P.items():
        for k2, v2 in Q.items():
            k = (k1[0] + k2[0], k1[1] + k2[1], k1[2] + k2[2], k1[3] + k2[3])
            x = zmul(v1, v2)
            R[k] = zadd(R[k], x) if k in R else x
    return {k: v for k, v in R.items() if v != Z}


def pscale(P, c):
    return {k: zmul(v, c) for k, v in P.items()} if c != Z else {}


def pconj(P):  # complex conjugate of the function: a <-> conj a, b <-> conj b
    return {(k[2], k[3], k[0], k[1]): zconj(v) for k, v in P.items()}


def mean(P):
    """Haar mean as (Fraction x0, Fraction x1, Fraction y0, Fraction y1)."""
    acc = {}
    for (p, q, r, s), v in P.items():
        if p == r and q == s:
            K = p + q
            acc.setdefault(K, Z)
            acc[K] = zadd(acc[K], zscale(v, factorial(p) * factorial(q)))
    out = [Fr(0)] * 4
    for K, v in acc.items():
        for i in range(4):
            out[i] += Fr(v[i], factorial(K + 1))
    return tuple(out)


def Dmono_polys(j):
    """Dmono[m][m'] polynomial (dict) with integer coefficients."""
    ms = mrange(j)
    D = []
    for m in ms:
        row = []
        for mp_ in ms:
            P = {}
            for c, e in poly_D_entry_coeffs(j, m, mp_):
                P[e] = zadd(P.get(e, Z), (c, 0, 0, 0))
            row.append({k: v for k, v in P.items() if v != Z})
        D.append(row)
    return D


def frac_to_z(fr_tuple, den):
    """((a,b),(c,d)) Fractions -> ints scaled by den."""
    (a, b), (c, d) = fr_tuple
    out = []
    for x in (a, b, c, d):
        y = x * den
        assert y.denominator == 1
        out.append(int(y))
    return tuple(out)


class Section:
    """Exact data of psi_y for a slot.  p: list of Gaussian-integer pairs (re, im) per m = j..-j."""

    def __init__(self, j, Pm_frac, sN, p):
        self.j = j
        ms = mrange(j)
        self.ms = ms
        n = len(ms)
        dP = 1
        for row in Pm_frac:
            for (a, b), (c, d) in row:
                for x in (a, b, c, d):
                    dP = lcm(dP, x.denominator)
        self.dP = dP
        Pz = [[frac_to_z(t, dP) for t in row] for row in Pm_frac]
        Dm = Dmono_polys(j)
        wv = [wfac(j, m) for m in ms]
        # B_m' = sum_m p_m w_m Dmono[m][m']   (row (p w)^T Dmono)
        B = []
        for c in range(n):
            P = {}
            for r in range(n):
                coef = (p[r][0] * wv[r], 0, p[r][1] * wv[r], 0)
                if coef != Z:
                    P = padd(P, pscale(Dm[r][c], coef))
            B.append(P)
        # A = B Pmono  (scaled by dP)
        A = []
        for c in range(n):
            P = {}
            for k in range(n):
                if Pz[k][c] != Z:
                    P = padd(P, pscale(B[k], Pz[k][c]))
            A.append(P)
        self.A = A  # true A = A / dP
        Wl = 1
        for x in wv:
            Wl = lcm(Wl, x)
        self.Wl = Wl
        # Q = A W^-1 A^dag, scaled: true Q = Qz / (dP^2 Wl)
        Q = {}
        for c in range(n):
            Q = padd(Q, {k: zscale(v, Wl // wv[c]) for k, v in pmul(A[c], pconj(A[c])).items()})
        self.Q = Q
        self.Qscale = Fr(1, dP * dP * Wl)
        self.sN = sN
        self.wv = wv
        self.norm2 = sum(Fr(p[r][0] ** 2 + p[r][1] ** 2) * wv[r] for r in range(n))

    def h(self):
        """mean |psi|^4 (V = 1 measure), exact: s^-2 mean(Q^2)."""
        m = mean(pmul(self.Q, self.Q))
        sc = self.Qscale ** 2 / self.sN ** 2
        return tuple(x * sc for x in m)

    def mean_psi6(self):
        m = mean(pmul(pmul(self.Q, self.Q), self.Q))
        sc = self.Qscale ** 3 / self.sN ** 3
        return tuple(x * sc for x in m)

    def N_components(self):
        """G_m' = Q A_m' (scaled); true N eta^dag component m' = s^-3/2 G_m' /(dP^3 Wl) / sqrt(w_m')."""
        return [pmul(self.Q, Am) for Am in self.A]

    def level_norms(self, Js):
        """exact ||Pi_{2J} N(psi_y)||^2 for J in Js, as Q(sqrt5) pairs (Fraction, Fraction)."""
        Gs = self.N_components()
        scale = (self.Qscale / self.dP) ** 2 / self.sN ** 3  # (1/(dP^3 Wl))^2 s^-3
        idx = []
        for Gc in Gs:
            d = {}
            for (P, Q, R, S), v in Gc.items():
                d.setdefault((P - R, Q - S), []).append((P, Q, v))
            idx.append(d)
        out = {}
        for J in Js:
            DJ = Dmono_polys(J)
            msJ = mrange(J)
            wJ = [wfac(J, M) for M in msJ]
            tot = [Fr(0), Fr(0)]
            for c in range(len(Gs)):
                d = idx[c]
                for r1 in range(len(msJ)):
                    for r2 in range(len(msJ)):
                        cD = pconj(DJ[r1][r2])
                        acc = Z
                        K = None
                        for (p_, q_, r_, s_), u in cD.items():
                            for (P, Q, v) in d.get((r_ - p_, s_ - q_), ()):
                                K2 = P + p_ + Q + q_
                                assert K is None or K == K2
                                K = K2
                                acc = zadd(acc, zscale(zmul(v, u), factorial(P + p_) * factorial(Q + q_)))
                        if acc == Z:
                            continue
                        m = [Fr(x, factorial(K + 1)) for x in acc]
                        x0, x1, y0, y1 = m
                        re0 = x0 * x0 + 5 * x1 * x1 + y0 * y0 + 5 * y1 * y1
                        re1 = 2 * x0 * x1 + 2 * y0 * y1
                        f = Fr(wJ[r1], wJ[r2]) * Fr(1, self.wv[c]) * (2 * J + 1)
                        tot[0] += re0 * f
                        tot[1] += re1 * f
            out[J] = (tot[0] * scale, tot[1] * scale)
        return out
