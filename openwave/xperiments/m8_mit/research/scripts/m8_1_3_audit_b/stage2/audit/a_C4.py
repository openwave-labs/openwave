"""Audit of C.4 (solver_ca): at every census point (R4, R5) and every R2 point:
  * exact criticality with sympy (my exact quartic h), plus a planted perturbation that must make it fail;
  * R0(u) by my CG route at 60 digits (mpmath), and by my exact S^3 route where the point has equal |p_m| on its support;
  * comparison with solver_ca's exact rational.
Then R0 over the Section-5 boxes O1, O2 (R4, R5) and O3 R5 by my CG route in mpmath.iv (90 digits), compared with ca's
enclosures (intersection) and sign.  Argument: PLANT tag (optional) -> 'perturb' makes one point non-critical."""
import sys, os, json, time
sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
if len(sys.argv) > 1:
    os.environ['PLANT'] = sys.argv[1]
from fractions import Fraction as Fr
from decimal import Decimal
import sympy as sp
from mpmath import mp, iv
from core import mlist, Fm, KZERO
from reps import SLOTS
from s3poly import R0_levels, R0_from_levels
from cgroute import CGRoute, CP
from hpoly import h_poly, PolyEval
from ivutil import ends, outward, width
from chk import check, save_json, planted

inp = json.load(open(os.path.join(here, '..', 'inputs.json')))
ca = json.load(open(os.path.join(here, '..', 'solver_ca', 'results.json')))
mp.dps = 60
out = {}


def comps_to_w(name, comps):
    d, n = SLOTS[name]; ms = mlist(n)
    w = [sp.Integer(0)] * (n + 1)
    for k, v in comps.items():
        w[ms.index(Fr(k))] = sp.sympify(v, locals={'I': sp.I, 'sqrt': sp.sqrt})
    return w


def exact_critical(name, w):
    d, n = SLOTS[name]; N = n + 1; ms = mlist(n)
    p = [w[i] / sp.sqrt(Fm(n, ms[i])) for i in range(N)]
    x = [sp.re(v) for v in p] + [sp.im(v) for v in p]
    he = PolyEval(h_poly(name), 2 * N)
    conv = lambda c: sp.Rational(c.numerator, c.denominator) if isinstance(c, Fr) else sp.Integer(c)
    hv = sp.nsimplify(sp.expand(he.value(x, conv)))
    g = [sp.expand(v) for v in he.grad(x, conv)]
    Gd = [Fm(n, m) for m in ms] * 2
    r = sp.expand(sum(Gd[i] * x[i] ** 2 for i in range(2 * N)))
    res = [sp.simplify(g[i] - 4 * hv / r * Gd[i] * x[i]) for i in range(2 * N)]
    return all(v == 0 for v in res), sp.simplify(hv / r ** 2), max(abs(sp.N(v, 30)) for v in res)


cg = {}
worst = {}
for name, pts in (('R4', inp['census']['points']), ('R5', inp['census']['points']), ('R2', inp['R2 points']['points'])):
    d, n = SLOTS[name]; N = n + 1; ms = mlist(n)
    if name not in cg:
        cg[name] = CGRoute(name, mp)
    for pname, comps in pts.items():
        w = comps_to_w(name, comps)
        if planted('perturb') and pname == 'hexagon':
            w[ms.index(Fr(1))] = sp.Rational(1, 10 ** 20)      # planted: a tiny off-orbit component
        crit, hu, res = exact_critical(name, w)
        check(f'C.4 {name} {pname}: critical (exact sympy)', crit, f'max residual {sp.N(res, 3)}')
        wn2 = sp.nsimplify(sum(sp.Abs(v) ** 2 for v in w))
        nrm = mp.sqrt(mp.mpf(sp.N(wn2, 80)))
        y = [CP(mp.mpf(sp.N(sp.re(v), 80)) / nrm, mp.mpf(sp.N(sp.im(v), 80)) / nrm) for v in w]
        R0c = cg[name].R0(y)
        rec = {'critical (exact)': bool(crit), 'h(u)': str(hu), 'R0 CG route (60 digits)': mp.nstr(R0c, 45)}
        # exact S^3 route where |p_m|^2 is constant on the support
        support = [i for i in range(N) if w[i] != 0]
        a2 = {i: sp.nsimplify(sp.Abs(w[i]) ** 2 / Fm(n, ms[i])) for i in support}
        R0ex = None
        if len(set(a2.values())) == 1:
            sc2 = list(a2.values())[0]
            p = [KZERO] * N
            for i in support:
                ph = sp.nsimplify(w[i] / sp.sqrt(Fm(n, ms[i])) / sp.sqrt(sc2))
                p[i] = (Fr(int(sp.fraction(sp.re(ph))[0]), int(sp.fraction(sp.re(ph))[1])), Fr(0),
                        Fr(int(sp.fraction(sp.im(ph))[0]), int(sp.fraction(sp.im(ph))[1])), Fr(0))
            v = R0_from_levels(name, R0_levels(name, p))
            assert v[1:] == (0, 0, 0)
            R0ex = sp.nsimplify(sp.Rational(v[0].numerator, v[0].denominator) * sc2 ** 3 / wn2 ** 3)
            rec['R0 exact (my S^3 route)'] = str(R0ex)
            e = abs(mp.mpf(sp.N(R0ex, 80)) - R0c)
            check(f'C.4 {name} {pname}: my exact S^3 value agrees with my CG route', e < mp.mpf(10) ** -50, mp.nstr(e, 3))
        cav = ca['C.4'][name][pname]
        cex = None
        for k, v in cav.items():
            if k.startswith('R0') and isinstance(v, str) and '/' in v:
                cex = sp.Rational(v)
        if cex is None:
            print('   ca entry:', cav)
        rec['ca exact'] = str(cex)
        if R0ex is not None:
            check(f'C.4 {name} {pname}: ca exact value == my exact value', cex == R0ex, f'{cex} vs {R0ex}')
        e2 = abs(mp.mpf(cex.p) / cex.q - R0c)
        worst[(name, pname)] = e2
        check(f'C.4 {name} {pname}: ca exact value agrees with my CG route to 1e-50', e2 < mp.mpf(10) ** -50, mp.nstr(e2, 3))
        rec['sign'] = '-' if R0c < 0 else '+'
        out.setdefault(name, {})[pname] = rec
        print(f'{name} {pname}: h = {hu}, R0 = {mp.nstr(R0c, 25)}', flush=True)

if planted('perturb'):
    sys.exit(0)

# ------------- boxes, interval CG route over the Section-5 box (x-part)
iv.dps = 90
rad = iv.mpf('1e-60')
for bname in ['O1 R4', 'O1 R5', 'O2 R4', 'O2 R5', 'O3 R4', 'O3 R5']:
    name = bname.split()[1]; d, n = SLOTS[name]; N = n + 1; ms = mlist(n)
    xc = inp['boxes']['boxes'][bname]['centre, x-part']
    X = [iv.mpf(s) + iv.mpf([-1, 1]) * rad for s in xc]
    t0 = time.time()
    cgi = CGRoute(name, iv)
    y = [CP(X[i] * iv.sqrt(Fm(n, ms[i])), X[N + i] * iv.sqrt(Fm(n, ms[i]))) for i in range(N)]
    nr2 = sum((Fm(n, ms[i]) * (X[i] ** 2 + X[N + i] ** 2) for i in range(N)), iv.mpf(0))
    R0box = cgi.R0(y) / nr2 ** 3          # R0 is homogeneous of degree 6: R0(x/|x|) = R0(x)/|x|^6
    lo, hi = ends(R0box)
    sign = '-' if hi < 0 else ('+' if lo > 0 else 'undetermined')
    check(f'C.4 {bname}: my interval enclosure of R0(x/|x|_G) over the box excludes 0', sign != 'undetermined')
    cae = ca['C.4'][name][bname]
    enc = None
    for k, v in cae.items():
        if isinstance(v, list) and len(v) == 2:
            enc = v
    clo, chi = Fr(Decimal(enc[0])), Fr(Decimal(enc[1]))
    from ivutil import mpf_to_fraction
    meet = max(clo, mpf_to_fraction(lo)) <= min(chi, mpf_to_fraction(hi))
    check(f'C.4 {bname}: ca enclosure intersects mine', meet, f'mine {outward(R0box, 50)}; ca {enc}')
    print(f'{bname}: R0 in {outward(R0box, 50)} width {mp.nstr(width(R0box), 3)} sign {sign} ({time.time() - t0:.0f}s)', flush=True)
    out.setdefault(name, {})[bname] = {'enclosure': outward(R0box, 50), 'width': mp.nstr(width(R0box), 3), 'sign': sign}
save_json('a_C4.json', out)
