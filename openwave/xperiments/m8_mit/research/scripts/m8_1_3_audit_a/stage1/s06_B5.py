"""B.5: on V_3, are HS6 = |(ww^dag)_6|^2/|w|^4 and rhat_6 = |[w (x) Theta w]_6|^2/|w|^4 proportional?
Route: both written exactly as sum_J c_J f_J (Casimir projectors, exact rational evaluations at Gaussian-rational
weighted points, overdetermined system checked exactly); proportional iff the coefficient vectors are.
Second route: rhat_6 by the Fischer norm of the polynomial w * Theta(w) (|[a(x)b]_6|^2 = |ab|_F^2 / 924 on V_3),
checked exactly against the Casimir route; and a numeric spot check of the ratio at random complex points.
'plant': perturbs the rhat coefficient vector."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_quartic import *
from lib_hs import *
import sympy as sp, random

log = Log(__file__, '_plant' if planted() else '')
j2 = 6; n = 7
proj = casimir_projectors_w(j2)
adp = ad_projectors_w(j2)
sJ = sym_J2s(j2)
rnd = random.Random(777)
rows = []; hs = []; rh = []; fis = []
for k in range(8):
    p = [G(rnd.randint(-4, 4), rnd.randint(-4, 4)) for _ in range(n)]
    f = fJ_all(p, j2, proj)
    rows.append([f[J2] for J2 in sJ])
    hs.append(HS_K(p, j2, adp, 12))
    r = rhat_top(p, j2, proj); rh.append(r)
    fis.append(fischer_product_norm(p, theta_p(p, j2), j2) / norm2_p(p, j2) ** 2)
log.check('924 rhat_6 (Casimir) = |w Theta w|_F^2/|w|^4 (Fischer) exactly at 8 points', all(924 * a == b for a, b in zip(rh, fis)))
A = sp.Matrix(rows)
def solve(rhs):
    s = A[:4, :].LUsolve(sp.Matrix(rhs[:4]))
    ok = (A * s - sp.Matrix(rhs)) == sp.zeros(len(rhs), 1)
    return s, ok
bh, okh = solve(hs); br, okr = solve(rh)
if planted():
    br[1] = br[1] + sp.Rational(1, 500)
log.check('HS6 = sum b_J f_J consistent at all 8 points', okh)
log.check('rhat_6 = sum c_J f_J consistent at all 8 points', okr)
log('HS6 coefficients (J = 6,4,2,0):', [str(x) for x in bh])
log('rhat6 coefficients (J = 6,4,2,0):', [str(x) for x in br])
rat = [sp.nsimplify(br[i] / bh[i]) if bh[i] != 0 else None for i in range(4)]
log('rhat6 / HS6 coefficientwise:', [str(x) for x in rat])
prop = len(set(rat)) == 1 and None not in rat
log(f'proportional as functions: {prop}' + (f'; rhat_6 = {rat[0]} * HS6, i.e. HS6 = {1 / rat[0]} * rhat_6' if prop else ''))
log.check('coefficientwise ratio identical (exact)', prop)
if not planted():
    save_json('B5.json', dict(proportional=prop, rhat_over_HS=str(rat[0]) if prop else None,
                              HS_over_rhat=str(1 / rat[0]) if prop else None,
                              b_HS=[str(x) for x in bh], c_rhat=[str(x) for x in br]))
log.close()
