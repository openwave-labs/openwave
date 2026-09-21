import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sympy as sp, common as C, hess
from orbits import ORBITS
lam = sp.Symbol('lam')
t0 = time.time()
hess.hessN_sym(); print('hess built', time.time() - t0)
for nm in sys.argv[1:]:
    c = list(ORBITS[nm]['u'])
    M, Nu = hess.M_u(c)
    B, dO = hess.N_basis(c)
    print(nm, 'dim O =', dO, 'dim N =', B.shape[1], time.time() - t0)
    p, K, G = hess.restricted_charpoly(M, B, lam)
    print(nm, sp.factor(p.as_expr()), time.time() - t0)
