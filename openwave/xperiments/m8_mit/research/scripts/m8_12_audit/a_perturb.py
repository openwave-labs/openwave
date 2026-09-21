"""Exact tangential gradient at the four interior orbit representatives, and at slightly perturbed points."""
import sympy as sp
from aud_core import *

def vc(d):
    c = [sp.Integer(0)] * 7
    for m, z in d.items(): c[idx(m)] = sp.sympify(z)
    return c
r3, r5 = sp.sqrt(3), sp.sqrt(5)
reps = {
    'A*': vc({1: 2, -2: 1}),
    'D*': vc({2: sp.sqrt(13), -3: 2 * r3}),
    'F*': vc({0: sp.sqrt(6), 2: I * r5, -2: I * r5}),
    'G*': vc({0: sp.sqrt(23), 3: I * sp.sqrt(10), -3: -I * sp.sqrt(10)}),
}
eps = sp.Rational(1, 1000)
def tgrad2(c):
    n = sp.sqrt(nrm2(c)); u = [sp.radsimp(z / n) for z in c]
    g = grad_exact(u); Nv = sp.expand(N_exact(u)); x = to_real(u)
    tg = [sp.radsimp(sp.expand(a - 4 * Nv * b)) for a, b in zip(g, x)]
    return sp.radsimp(sp.expand(sum(t ** 2 for t in tg)))
for nm, c in reps.items():
    g0 = tgrad2(c)
    cp = list(c); k = next(i for i in range(7) if c[i] != 0); cp[k] = c[k] * (1 + eps)   # move off the critical point
    g1 = tgrad2(cp)
    print('%-3s |grad|^2 at representative = %s ;  after (1+1e-3) scaling of one coordinate: %s' % (nm, g0, sp.N(g1, 6)))
