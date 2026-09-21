"""The locus span{v3, v-1}: exact restriction, interior critical points, linear coefficients at both ends."""
import sympy as sp
from aud_core import *

t = sp.Symbol('t', nonnegative=True)
x = sp.Symbol('x', real=True)
def v(m):
    c = [sp.Integer(0)] * 7; c[idx(m)] = sp.Integer(1); return c
for lab, a, b in (('chart v3 + z v-1 (omits v-1)', v(3), v(-1)), ('chart v-1 + w v3 (omits v3)', v(-1), v(3))):
    u = [p + x * q for p, q in zip(a, b)]   # radial restriction (R_z invariance): real z suffices
    N = sp.expand(sp.re(sp.expand(N_exact(u)))).subs(x, sp.sqrt(t))
    f = sp.expand(924 * N) / (924 * (1 + t) ** 2)
    print(lab)
    print('   924 N =', sp.expand(924 * N), ';  rhat6 = (%s)/(924 (1+t)^2)' % sp.expand(924 * N))
    ser = sp.series(f, t, 0, 3).removeO()
    print('   Taylor in t=|z|^2 :', sp.expand(ser), ';  linear coefficient =', sp.expand(ser).coeff(t, 1))
    fp = sp.factor(sp.diff(f, t))
    print("   d rhat6/dt =", fp, ';  zeros with t > 0:', [r for r in sp.solve(sp.numer(fp), t) if r.is_positive])
# is the restriction really radial?  check the full complex z dependence once
y = sp.Symbol('y', real=True)
u = [p + (x + I * y) * q for p, q in zip(v(3), v(-1))]
Nf = sp.expand(sp.re(sp.expand(N_exact(u))))
print('radial check: N(v3 + z v-1) - N at |z| on the real axis == 0:',
      sp.simplify(Nf - sp.expand(sp.re(sp.expand(N_exact([p + sp.sqrt(x**2 + y**2) * q for p, q in zip(v(3), v(-1))]))))) == 0)
