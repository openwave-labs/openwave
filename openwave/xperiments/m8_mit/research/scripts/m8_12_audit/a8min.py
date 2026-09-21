"""Audit item 8, minimum: N as a Hermitian form on degree-2 monomials; generalised eigenvalues vs |u|^4.
Coordinates b_m = c_m / sqrt(C(6,3+m)) (rational coefficients).  Exact."""
import itertools
import sympy as sp
from aud_core import MS, idx

Cb = {m: sp.binomial(6, 3 + m) for m in MS}
b = sp.symbols('b3 b2 b1 b0 bm1 bm2 bm3')
bb = sp.symbols('B3 B2 B1 B0 Bm1 Bm2 Bm3')   # stand-ins for conj(b)
def rho_b(Q):
    s = 0
    for m1 in MS:
        m2 = Q - m1
        if abs(m2) <= 3:
            s += Cb[m1] * Cb[m2] * (-1) ** (3 - m2) * b[idx(m1)] * bb[idx(-m2)]
    return sp.expand(s)
def conjpoly(p): return p.xreplace({**dict(zip(b, bb)), **dict(zip(bb, b))})
N = sp.expand(sum(sp.expand(rho_b(Q) * conjpoly(rho_b(Q))) / sp.binomial(12, 6 + Q) for Q in range(-6, 7)))
n2 = sum(Cb[m] * b[idx(m)] * bb[idx(m)] for m in MS)
# check against the c-route at a random exact point
from aud_core import N_exact
import random
random.seed(5)
cpt = [sp.Rational(random.randint(-9, 9), 7) + sp.I * sp.Rational(random.randint(-9, 9), 5) for _ in range(7)]
bpt = {**{b[idx(m)]: cpt[idx(m)] / sp.sqrt(Cb[m]) for m in MS}, **{bb[idx(m)]: sp.conjugate(cpt[idx(m)]) / sp.sqrt(Cb[m]) for m in MS}}
print('b-route N equals c-route N at a random exact point:', sp.expand(N.subs(bpt) - N_exact(cpt)) == 0)
monos = list(itertools.combinations_with_replacement(range(7), 2))
def mon(vs, al): return sp.Mul(*[vs[i] for i in al])
P = sp.Poly(N, *bb, *b)
Pn = sp.Poly(sp.expand(n2 ** 2), *bb, *b)
def coef(Pp, al, be):
    e = [0] * 14
    for i in al: e[i] += 1
    for i in be: e[7 + i] += 1
    return Pp.coeff_monomial(tuple(e))
K = sp.Matrix(len(monos), len(monos), lambda i, j: coef(P, monos[i], monos[j]))
G = sp.Matrix(len(monos), len(monos), lambda i, j: coef(Pn, monos[i], monos[j]))
print('K Hermitian (real symmetric):', K == K.T, '  Gamma diagonal positive:', G.is_diagonal() and all(G[i, i] > 0 for i in range(G.shape[0])))
# reconstruct N from K to confirm the representation is complete
rec = sp.expand(sum(K[i, j] * mon(bb, monos[i]) * mon(b, monos[j]) for i in range(len(monos)) for j in range(len(monos))))
print('N == sum K_ab conj(b^a) b^b :', sp.expand(rec - N) == 0)
Gi = G.inv()
ev = (Gi * K).eigenvals()
print('generalised eigenvalues of (K, Gamma) with multiplicities:', dict(ev))
print('min =', min(ev.keys()), '  => N >= |u|^4 * min  on all of V (since K - min*Gamma is PSD)')
lam = min(ev.keys())
S = K - lam * G
# PSD check exactly: all eigenvalues of Gamma^{-1/2} S Gamma^{-1/2} >= 0  <=> eigenvalues of Gi*S >= 0 (similar)
print('eigenvalues of Gamma^-1 (K - min Gamma):', sorted(set((Gi * S).eigenvals().keys())))
