"""Audit of C.1: recompute the level norms at EACH solver's own y by my routes.
  cb's y has Gaussian-rational weighted coordinates -> my exact S^3 route (route E) and my CG route at 60 digits.
  ca's y has Gaussian-rational orthonormal components (irrational weighted coordinates) -> my CG route at 60 digits,
  compared with ca's exact surd expressions evaluated at 80 digits.  Both readings of ca's y (orthonormal or weighted
  components) are tried; only one can match.
Argument 'plant' perturbs one exact cb value by 1e-12 (relative) and the comparison must fail."""
import sys, os, json
sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
if len(sys.argv) > 1:
    os.environ['PLANT'] = sys.argv[1]
from fractions import Fraction as Fr
import sympy as sp
from mpmath import mp
from core import mlist, Fm
from reps import SLOTS, s_N
from s3poly import R0_levels, kfrac, wnorm2
from cgroute import CGRoute, CP
from chk import check, save_json, planted

mp.dps = 60
ca = json.load(open(os.path.join(here, '..', 'solver_ca', 'results.json')))['C.1']
cb = json.load(open(os.path.join(here, '..', 'solver_cb', 'results.json')))['C.1']
out = {}
cbp = {7: [(1, 2), (0, -1), (3, 1), (1, 0), (-2, 1), (1, 1), (0, 2), (1, -1)],
       6: [(1, 2), (0, -1), (3, 1), (1, 0), (-2, 1), (1, 1), (0, 2)]}
for nm in ['R2', 'R4', 'R5']:
    d, n = SLOTS[nm]; ms = mlist(n); N = n + 1
    cg = CGRoute(nm, mp)
    # ---------------- cb: weighted coordinates p (Gaussian integers)
    p = [kfrac(a, b) for a, b in cbp[n]]
    # consistency with cb's stated y (orthonormal components)
    ystated = [sp.sympify(cb[nm]['y'][str(m) if m.denominator == 1 else f'{m.numerator}/{m.denominator}'], locals={'I': sp.I}) for m in ms]
    mine_y = [sp.Rational(a) + sp.I * b for a, b in cbp[n]]
    nrm2 = wnorm2(nm, p)
    ok_y = all(sp.simplify(ystated[i] - mine_y[i] * sp.sqrt(Fm(n, ms[i])) / sp.sqrt(sp.Rational(nrm2.numerator, nrm2.denominator))) == 0 for i in range(N))
    check(f'C.1 {nm}: my reading of cb y (weighted p, Gaussian integers) reproduces cb stated unit y', ok_y)
    lv = R0_levels(nm, p)
    lvu = {L: v[0] / nrm2 ** 3 for L, v in lv.items()}
    check(f'C.1 {nm} (cb y): exact level norms real and rational', all(v[1:] == (0, 0, 0) for v in lv.values()))
    sq = mp.sqrt(mp.mpf(nrm2.numerator) / nrm2.denominator)
    yv = [CP(mp.mpf(a) * mp.sqrt(Fm(n, ms[i])) / sq, mp.mpf(b) * mp.sqrt(Fm(n, ms[i])) / sq) for i, (a, b) in enumerate(cbp[n])]
    lv2 = cg.levels(yv)
    worst = max(abs(lv2.get(L, 0) - mp.mpf(v.numerator) / v.denominator) for L, v in lvu.items())
    check(f'C.1 {nm} (cb y): my CG route agrees with my exact route at every level', worst < mp.mpf(10) ** -50, mp.nstr(worst, 3))
    rec = {'cb y': {}}
    for L in range(n + 1, 21):
        mine = lvu.get(L, Fr(0))
        theirs = Fr(cb[nm]['level norms'].get(str(L), '0'))
        if planted('plant') and nm == 'R4' and L == 14:
            theirs = theirs * (1 + Fr(1, 10 ** 12))
        check(f'C.1 {nm} (cb y) L={L}: cb exact value == my exact value', mine == theirs, f'{mine} vs {theirs}' if mine != theirs else str(mine))
        rec['cb y'][L] = str(mine)
    # ---------------- ca: y given as orthonormal components (reading A) or weighted coordinates (reading B)
    ys = [sp.sympify(s, locals={'I': sp.I}) for s in ca[nm]['y (unnormalized; y/||y|| used)']]
    for reading in ('A: orthonormal components', 'B: weighted coordinates'):
        if reading.startswith('A'):
            wv = [complex(v) for v in ys]
            wv = [(mp.mpf(int(sp.re(v))), mp.mpf(int(sp.im(v)))) for v in ys]
        else:
            wv = [(mp.mpf(int(sp.re(v))) * mp.sqrt(Fm(n, ms[i])), mp.mpf(int(sp.im(v))) * mp.sqrt(Fm(n, ms[i]))) for i, v in enumerate(ys)]
        nr = mp.sqrt(sum(a * a + b * b for a, b in wv))
        yv = [CP(a / nr, b / nr) for a, b in wv]
        lvA = cg.levels(yv)
        diffs = {}
        for L, s in ca[nm]['level norms'].items():
            val = mp.mpf(str(sp.N(sp.sympify(s), 80)))
            diffs[int(L)] = abs(val - lvA.get(int(L), 0))
        wd = max(diffs.values())
        print(f'{nm} ca y, reading {reading}: max |ca - mine| over levels = {mp.nstr(wd, 3)}')
        rec[f'ca y, reading {reading[0]}'] = {L: mp.nstr(lvA.get(L, 0), 30) for L in range(n + 1, 21)}
        if reading.startswith('A'):
            check(f'C.1 {nm} (ca y, orthonormal components): ca exact surd values agree with my CG route to 1e-50', wd < mp.mpf(10) ** -50, mp.nstr(wd, 3))
            nzA = [L for L in range(n + 1, 21) if lvA.get(L, 0) > mp.mpf(10) ** -40]
            check(f'C.1 {nm} (ca y): nonzero levels equal ca list', nzA == ca[nm]['levels where nonzero'], str(nzA))
    out[nm] = rec
save_json('a_C1.json', out)
