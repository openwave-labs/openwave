"""Exact polynomial identity between the c-route N (aud_core, CG by lowering) and the b-route N used in
a8min.py / a8max.py, with c_m = sqrt(C(6,3+m)) b_m."""
import sympy as sp
from aud_core import N_exact, MS, idx

Cb = {m: sp.binomial(6, 3 + m) for m in MS}
xs = sp.symbols('x0:7', real=True); ys = sp.symbols('y0:7', real=True)
bv = [xs[k] + sp.I * ys[k] for k in range(7)]
c = [sp.sqrt(Cb[m]) * bv[idx(m)] for m in MS]
Nc = sp.expand(N_exact(c))
tot = 0
for Q in range(-6, 7):
    s = 0
    for m1 in MS:
        m2 = Q - m1
        if abs(m2) <= 3:
            s += Cb[m1] * Cb[m2] * (-1) ** (3 - m2) * bv[idx(m1)] * sp.conjugate(bv[idx(-m2)])
    s = sp.expand(s)
    tot += sp.expand(s * sp.conjugate(s)) / sp.binomial(12, 6 + Q)
print('exact polynomial identity N_c(sqrt(C) b) == N_b(b):', sp.expand(Nc - tot) == 0)
n2c = sp.expand(sum(sp.expand(z * sp.conjugate(z)) for z in c))
n2b = sp.expand(sum(Cb[m] * bv[idx(m)] * sp.conjugate(bv[idx(m)]) for m in MS))
print('exact identity |c|^2 == sum C_m |b_m|^2:', sp.expand(n2c - n2b) == 0)
