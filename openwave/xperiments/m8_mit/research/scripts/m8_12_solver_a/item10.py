"""Item 10: Q_sigma = 1 + w6 * rhat6 with w6 = 28/39 and 21/52."""
import json, os
import sympy as sp
import common as C, hess
from orbits import ORBITS

ck = C.Checker('item10')
W = {'sector w6=28/39': sp.Rational(28, 39), 'sector w6=21/52': sp.Rational(21, 52)}
res = {}
item4 = json.load(open(os.path.join(C.OUT, 'item4.json')))
item5 = json.load(open(os.path.join(C.OUT, 'item5.json')))
item5b = json.load(open(os.path.join(C.OUT, 'item5b.json')))
lam, mu = sp.symbols('lam mu')
# structural check: Q = (|u|^4 + w N)/|u|^4 is degree-0 homogeneous; M^Q_u = w M_u + 8 u u^T, so on T_u it is w M_u
u = list(ORBITS['F*']['u'])
M, Nv, gN, xv = hess.M_c(u)
wq = sp.Rational(28, 39)
X = C.XS
n2 = sum(v ** 2 for v in X)
NQ = sp.expand(n2 ** 2 + wq * C.N_poly())
sub = dict(zip(X, list(xv)))
HQ = sp.Matrix(14, 14, lambda i, j: sp.radsimp(sp.expand(sp.diff(NQ, X[i], X[j]).subs(sub))))
MQ = HQ - 4 * sp.radsimp(NQ.subs(sub)) * sp.eye(14)
diff = (MQ - wq * M - 8 * xv * xv.T).applyfunc(sp.radsimp)
ck.check('M^Q_u = w M_u + 8 u u^T exactly at F* (so H^Q_u = w H_u on T_u)', diff == sp.zeros(14))
for sec, w in W.items():
    out = {'orbit_values_Q': {}, 'charpolys_Q': {}, 'signatures': {}, 'item5b_times_w': {}}
    for nm, d in item4['orbits'].items():
        out['orbit_values_Q'][nm] = str(1 + w * sp.Rational(d['rhat6']))
    for nm, d in item5.items():
        if nm.startswith('control'): continue
        p = sp.sympify(d['charpoly_monic_factored'].replace('lam', 'lam'))
        n = d['dim_N']
        pq = sp.factor(sp.expand(w ** n * p.subs(lam, mu / w)))   # monic char poly of w*H
        out['charpolys_Q'][nm] = str(pq).replace('mu', 'lam')
        out['signatures'][nm] = d['signature_(n-,n0,n+)']
    for nm, d in item5b.items():
        out['item5b_times_w'][nm] = {'H_u_matrix_times_w': [[str(w * sp.Rational(v)) for v in row] for row in d['H_u_matrix_on_unit_tangents']],
                                     'Gram_matrix_Re (unchanged)': d['Gram_matrix_Re'],
                                     'orbit_null_controls_times_w': {k: str(w * sp.Rational(v['H_u_on_unit_tangent_(orbit-null control)']))
                                                                     for k, v in d['directions'].items() if v['projection_onto_N_u_is_zero']}}
    print('\n==', sec)
    for nm, v in out['orbit_values_Q'].items(): print('  Q at %-4s = %s' % (nm, v))
    for nm, v in out['item5b_times_w'].items(): print('  5b x w  %-26s %s' % (nm, v['H_u_matrix_times_w']))
    res[sec] = out
C.save('item10', res)
print('item10 fails:', ck.fails)
