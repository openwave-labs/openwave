"""Exact (rational) evaluation in weighted coordinates p of:
  f_J(w)      = |Pi_J (w(x)w)|^2/|w|^4 , Pi_J the Casimir projector (lib_rep)
  HS_K(w)     = |(w w^dag)_K|^2_HS/|w|^4 , projector onto spin K of End(V) under conjugation, built from the
                adjoint Casimir in the weighted basis
  rhat_6(w)   = |[w (x) Theta w]_6|^2/|w|^4 , via the Casimir projector on V_3 (x) V_3
All exact with Fractions when p has Gaussian-rational entries (complex numbers as pairs)."""
from lib_rep import *


class G:  # Gaussian rationals
    __slots__ = ('r', 'i')
    def __init__(self, r=0, i=0): self.r = Fr(r); self.i = Fr(i)
    def __add__(s, o): o = g(o); return G(s.r + o.r, s.i + o.i)
    __radd__ = __add__
    def __sub__(s, o): o = g(o); return G(s.r - o.r, s.i - o.i)
    def __mul__(s, o): o = g(o); return G(s.r * o.r - s.i * o.i, s.r * o.i + s.i * o.r)
    __rmul__ = __mul__
    def conj(s): return G(s.r, -s.i)
    def abs2(s): return s.r * s.r + s.i * s.i
    def __repr__(s): return f'({s.r}+{s.i}i)'


def g(x): return x if isinstance(x, G) else G(x)


def norm2_p(p, j2):
    return sum((x.abs2() * s for x, s in zip(p, svec(j2))), Fr(0))


def fJ_all(p, j2, proj):
    """dict 2J -> f_J (exact) ; proj from casimir_projectors_w(j2)"""
    n = j2 + 1; sv = svec(j2)
    pp = [p[a] * p[b] for a in range(n) for b in range(n)]
    G2 = [sv[a] * sv[b] for a in range(n) for b in range(n)]
    N4 = norm2_p(p, j2) ** 2
    out = {}
    for J2, P in proj.items():
        y = {}
        for (r, c), v in P.items():
            y[r] = y.get(r, G(0)) + pp[c] * v
        out[J2] = sum((G2[r] * yr.abs2() for r, yr in y.items()), Fr(0)) / N4
    return out


def ad_projectors_w(j2):
    """projector onto spin K of End(V_j) under X -> [J, X], in the weighted basis, per index pair (a,b) of X;
    returns dict K2 -> {((a,b),(c,d)): Fraction}"""
    import sympy as sp
    n = j2 + 1; ml = ms(j2)
    Jp, Jm, Jz = spin_ops_w(j2)
    def ad(op, X):  # X dict (a,b)->coeff ; [op, X]
        out = {}
        for (a, b), c in X.items():
            for a2 in range(n):
                if op[a2][a] != 0: out[(a2, b)] = out.get((a2, b), 0) + op[a2][a] * c
            for b2 in range(n):
                if op[b][b2] != 0: out[(a, b2)] = out.get((a, b2), 0) - c * op[b][b2]
        return out
    blocks = {}
    for a in range(n):
        for b in range(n):
            blocks.setdefault(ml[a] - ml[b], []).append((a, b))
    Ks = list(range(0, 2 * j2 + 1, 2))
    proj = {K2: {} for K2 in Ks}
    for M2, basis in blocks.items():
        idx = {ab: i for i, ab in enumerate(basis)}; nb = len(basis)
        C = sp.zeros(nb, nb)
        for i, ab in enumerate(basis):
            X = {ab: Fr(1)}
            for k, c in ad(Jz, ad(Jz, X)).items(): C[idx[k], i] += sp.Rational(c)
            for k, c in ad(Jp, ad(Jm, X)).items(): C[idx[k], i] += sp.Rational(c) / 2
            for k, c in ad(Jm, ad(Jp, X)).items(): C[idx[k], i] += sp.Rational(c) / 2
        present = [K2 for K2 in Ks if K2 >= abs(M2)]
        for K2 in present:
            lam = sp.Rational(K2 * (K2 + 2), 4)
            Pm = sp.eye(nb)
            for K2b in present:
                if K2b == K2: continue
                lb = sp.Rational(K2b * (K2b + 2), 4)
                Pm = Pm * (C - lb * sp.eye(nb)) / (lam - lb)
            for i in range(nb):
                for k in range(nb):
                    if Pm[i, k] != 0:
                        proj[K2][(basis[i], basis[k])] = Fr(int(sp.numer(Pm[i, k])), int(sp.denom(Pm[i, k])))
    return proj


def HS_K(p, j2, adproj, K2):
    """|(w w^dag)_K|^2_HS / |w|^4. CS operator X = w w^dag ; weighted-basis operator X_p = S^-1 X S = p p^dag S^2;
    HS product in weighted basis: <X, Y> = tr(X_p^dag S^2 Y_p S^-2)."""
    n = j2 + 1; sv = svec(j2)
    Xp = {(a, b): p[a] * p[b].conj() * sv[b] for a in range(n) for b in range(n)}
    Y = {}
    for ((a, b), (c, d)), v in adproj[K2].items():
        Y[(a, b)] = Y.get((a, b), G(0)) + Xp[(c, d)] * v
    # || Y ||^2_HS = sum_{a,b} |Y_ab|^2 sv[a]/sv[b]
    tot = sum((y.abs2() * Fr(sv[a], sv[b]) for (a, b), y in Y.items()), Fr(0))
    return tot / norm2_p(p, j2) ** 2


def theta_p(p, j2):
    """Theta in weighted coordinates: p'_m = (-1)^{j-m} conj(p_{-m})"""
    ml = ms(j2)
    return [p[ml.index(-m2)].conj() * (-1) ** ((j2 - m2) // 2) for m2 in ml]


def rhat_top(p, j2, proj):
    """|[w (x) Theta w]_{2j}|^2/|w|^4 via the Casimir projector of spin 2j"""
    n = j2 + 1; sv = svec(j2)
    t = theta_p(p, j2)
    v = [p[a] * t[b] for a in range(n) for b in range(n)]
    G2 = [sv[a] * sv[b] for a in range(n) for b in range(n)]
    y = {}
    for (r, c), x in proj[2 * j2].items():
        y[r] = y.get(r, G(0)) + v[c] * x
    return sum((G2[r] * yr.abs2() for r, yr in y.items()), Fr(0)) / norm2_p(p, j2) ** 2


def fischer_product_norm(p, q, j2):
    """Fischer norm^2 of the polynomial product (sum p_m x^{j+m}y^{j-m})(sum q_m ...), ||x^a y^b||^2 = a! b!"""
    n = j2 + 1; ml = ms(j2)
    prod = {}
    for a in range(n):
        for b in range(n):
            e = (j2 + ml[a]) // 2 + (j2 + ml[b]) // 2
            prod[e] = prod.get(e, G(0)) + p[a] * q[b]
    return sum((c.abs2() * fact(e) * fact(2 * j2 - e) for e, c in prod.items()), Fr(0))
