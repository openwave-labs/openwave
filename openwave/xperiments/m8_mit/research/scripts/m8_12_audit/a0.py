"""Audit item 0: CG value, Theta^2, two exact rhat6 values, plus CG self-checks with a planted defect."""
import sympy as sp, mpmath as mp
from aud_core import *

tab = cg6_table()
print('<3 3; 3 -3 | 6 0> =', cg6(3, -3, 0), ' squared:', cg6(3, -3, 0) ** 2)

# check: each |6 Q> is an eigenvector of total J^2 with eigenvalue 42 (independent of how it was built)
Jx, Jy, Jz = Jmat()
E = sp.eye(7)
def kron(A, Bm): return sp.kronecker_product(A, Bm)
Tx, Ty, Tz = [kron(A, E) + kron(E, A) for A in (Jx, Jy, Jz)]
J2 = (Tx * Tx + Ty * Ty + Tz * Tz).applyfunc(sp.expand)
def vecQ(Q, table):
    v = sp.zeros(49, 1)
    for (a, b, q), c in table.items():
        if q == Q: v[idx(a) * 7 + idx(b)] = c
    return v
def J2_residual(table):
    worst = 0
    for Q in range(-6, 7):
        v = vecQ(Q, table)
        r = (J2 * v - 42 * v).applyfunc(sp.expand)
        worst = max(worst, max(abs(sp.N(x, 30)) for x in r))
    return worst
print('J^2 = 42 residual over all |6 Q> (exact expand, then |.|):', J2_residual(tab))
bad = dict(tab); bad[(1, -1, 0)] = -bad[(1, -1, 0)]
print('PLANTED sign flip of <3 1;3 -1|6 0>: J^2 residual =', J2_residual(bad), '(must be > 0)')
# normalisation
print('norms of |6 Q>:', set(sp.expand(sum(c ** 2 for (a, b, q), c in tab.items() if q == Q)) for Q in range(-6, 7)))

# Theta^2
cs = sp.symbols('c0:7')
u = list(cs)
tt = theta(theta(u))
print('Theta(Theta u) - u =', [sp.simplify(a - b) for a, b in zip(tt, u)])

def vec(d):
    c = [sp.Integer(0)] * 7
    for m, z in d.items(): c[idx(m)] = sp.sympify(z)
    return c
for name, d in [('2v3+v1-3v-2', {3: 2, 1: 1, -2: -3}), ('v3+i v0+2v-1', {3: 1, 0: I, -1: 2})]:
    c = vec(d)
    Nv = sp.expand(N_exact(c))
    r = sp.Rational(Nv) / nrm2(c) ** 2
    print(name, ' N =', Nv, ' |u|^2 =', nrm2(c), ' rhat6 =', r)
    for dps in (50, 80):
        mp.mp.dps = dps
        cc = [mp.mpc(complex(sp.N(z, dps))) if False else mp.mpc(sp.re(z), sp.im(z)) for z in c]
        val = mp_rhat(cc)
        print('   mp %d digits: |mp - exact| = %s' % (dps, mp.nstr(abs(val - mp.mpf(r.p) / r.q), 3)))
