import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sympy as sp, common as C, hess
from orbits import ORBITS
t0 = time.time()
hess.Nt_and_hess(); print('built', time.time() - t0, flush=True)
for nm in sys.argv[1:]:
    c = list(ORBITS[nm]['u'])
    a = hess.a_from_c(c)
    d = hess.exact_data(a)
    print(nm, d['field'], 'dimO', d['dimO'], 'rhat', d['rhat'], sp.factor(d['charpoly'].as_expr()), hess.signature(d['charpoly']), d['resid'], time.time() - t0, flush=True)
