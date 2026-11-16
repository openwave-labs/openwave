"""Item C.4: sign of R0(u) at the five cases (exact values) and at O3 in R4 over its Section-5 box (interval arithmetic).
R0(u) = -sum_{L > n} ||Pi_L N(psi_u)||^2 / (lambda_L - lambda_0), V = R = 1.
Exact route: S^3 harmonic decomposition at rational weighted coordinates (pentagonal pyramid: R0 there is a cubic form in
(|p_3|^2, |p_-2|^2) by z-rotation invariance; its 4 coefficients are fixed exactly from 4 rational points and the form is
verified exactly at a 5th).  Check route: CG coupling at 40 digits.  Interval route for O3: the CG route in mpmath.iv."""
import os, json, time
from fractions import Fraction as Fr
import sympy as sp
from mpmath import mp, iv
from core import mlist, Fm, KZERO
from reps import SLOTS
from s3poly import R0_levels, R0_from_levels, kfrac
from cgroute import CGRoute, CP
from hpoly import h_poly, PolyEval
from chk import check, save_json, planted, load_json

mp.dps = 40
here = os.path.dirname(os.path.abspath(__file__))
inp = json.load(open(os.path.join(here, 'inputs.json')))
out = {'C.4': {'R2': {}, 'R4': {}, 'R5': {}}}


def comps_to_w(name, comps):
    d, n = SLOTS[name]; ms = mlist(n)
    w = [sp.Integer(0)] * (n + 1)
    for k, v in comps.items():
        w[ms.index(Fr(k))] = sp.sympify(v.replace('I', 'I'), locals={'I': sp.I, 'sqrt': sp.sqrt})
    return w


def R0_exact_rational_p(name, p):
    lv = R0_levels(name, p)
    v = R0_from_levels(name, lv)
    assert v[1:] == (0, 0, 0)
    return v[0]


def exact_critical(name, w):
    """exact criticality check with sympy (entries may contain square roots)."""
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
    return all(v == 0 for v in res), sp.simplify(hv / r ** 2)


cases = inp['five cases']
cgr = {}
for case in cases:
    pname, slot = case.rsplit(' in ', 1)
    name = slot.replace('R', 'R')
    if name == 'R2':
        comps = inp['R2 points']['points'][pname]
    else:
        comps = inp['census']['points'][pname.replace('coherent v3', 'coherent v3')]
    d, n = SLOTS[name]; N = n + 1; ms = mlist(n)
    w = comps_to_w(name, comps)
    crit, hu = exact_critical(name, w)
    check(f'C.4 {case}: critical (exact)', crit)
    wn2 = sp.nsimplify(sum(sp.Abs(v) ** 2 for v in w))
    support = [i for i in range(N) if w[i] != 0]
    # weighted coordinates squared moduli
    a2 = {i: sp.nsimplify(sp.Abs(w[i]) ** 2 / Fm(n, ms[i])) for i in support}
    if pname == 'pentagonal pyramid':
        i3, i2 = support
        # R0 is sum_{a=0..3} c_a A^a B^(3-a), A = |p_3|^2, B = |p_-2|^2 ; fix c_a from 4 rational points, verify at a 5th
        pts = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 2)]
        vals = []
        for (s, t) in pts:
            p = [KZERO] * N; p[i3] = kfrac(s); p[i2] = kfrac(t)
            vals.append(R0_exact_rational_p(name, p))
        Mx = sp.Matrix([[sp.Integer(s) ** (2 * a) * sp.Integer(t) ** (2 * (3 - a)) for a in range(4)] for (s, t) in pts[:4]])
        cvec = Mx.solve(sp.Matrix([sp.Rational(v.numerator, v.denominator) for v in vals[:4]]))
        s, t = pts[4]
        pred = sum(cvec[a] * sp.Integer(s) ** (2 * a) * sp.Integer(t) ** (2 * (3 - a)) for a in range(4))
        ok5 = pred == sp.Rational(vals[4].numerator, vals[4].denominator)
        if planted('C4'):
            ok5 = pred == sp.Rational(vals[4].numerator, vals[4].denominator) * 2
        check(f'C.4 {case}: cubic form in (|p_3|^2, |p_-2|^2) verified exactly at a fifth point', ok5)
        A = a2[i3]; B = a2[i2]
        R0w = sp.nsimplify(sum(cvec[a] * A ** a * B ** (3 - a) for a in range(4)))
    else:
        # rational after a common scale: all |p_m|^2 equal on the support (checked)
        vals_ = set(a2.values())
        assert len(vals_) == 1, a2
        sc2 = vals_.pop()
        p = [KZERO] * N
        for i in support:
            ph = sp.nsimplify(w[i] / sp.sqrt(Fm(n, ms[i])) / sp.sqrt(sc2))
            p[i] = (Fr(int(sp.fraction(sp.re(ph))[0]), int(sp.fraction(sp.re(ph))[1])), Fr(0),
                    Fr(int(sp.fraction(sp.im(ph))[0]), int(sp.fraction(sp.im(ph))[1])), Fr(0))
        R0p = R0_exact_rational_p(name, p)
        R0w = sp.Rational(R0p.numerator, R0p.denominator) * sc2 ** 3
    R0u = sp.nsimplify(R0w / wn2 ** 3)
    # CG cross-check at 40 digits
    if name not in cgr:
        cgr[name] = CGRoute(name, mp)
    nrm = mp.sqrt(mp.mpf(sp.N(wn2, 60)))
    y = [CP(mp.mpf(sp.N(sp.re(w[i]), 60)) / nrm, mp.mpf(sp.N(sp.im(w[i]), 60)) / nrm) for i in range(N)]
    num = cgr[name].R0(y)
    err = abs(num - mp.mpf(sp.N(R0u, 60)))
    check(f'C.4 {case}: exact R0(u) agrees with the CG route', err < mp.mpf(10) ** -32, mp.nstr(err, 3))
    sign = '-' if R0u < 0 else ('+' if R0u > 0 else '0')
    print(f'C.4 {case}: h(u) = {hu}; R0(u) = {R0u} = {sp.N(R0u, 25)}; sign {sign}')
    out['C.4'][name][pname] = {'sign': sign, 'R0(u) (V=R=1)': str(R0u), 'decimal': str(sp.N(R0u, 25)), 'exact': True}

# ------------------------------------------------------------------ O3 in R4 over its box (interval arithmetic)
name = 'R4'; d, n = SLOTS[name]; N = n + 1; ms = mlist(n)
box = inp['boxes']['boxes']['O3 R4']
iv.dps = 90
rad = iv.mpf('1e-60')
xc = box['centre, x-part']
X = [iv.mpf(s) + iv.mpf([-1, 1]) * rad for s in xc]
check('C.4 O3 R4: the interval box encloses every centre coordinate', all(iv.mpf(s) in Xi for s, Xi in zip(xc, X)))
t0 = time.time()
cgi = CGRoute(name, iv)
y = [CP(X[i] * iv.sqrt(Fm(n, ms[i])), X[N + i] * iv.sqrt(Fm(n, ms[i]))) for i in range(N)]
R0box = cgi.R0(y)
print(f'   interval CG route built and evaluated in {time.time() - t0:.1f}s')
from ivutil import ends, outward, width
lo, hi = ends(R0box)
sign = '-' if hi < 0 else ('+' if lo > 0 else 'undetermined')
check('C.4 O3 R4: interval enclosure of R0 over the box excludes 0', sign != 'undetermined')
# consistency of the enclosure with a point evaluation at the centre (mp, 60 digits)
mp.dps = 60
cgm = CGRoute(name, mp)
yc = [CP(mp.mpf(xc[i]) * mp.sqrt(Fm(n, ms[i])), mp.mpf(xc[N + i]) * mp.sqrt(Fm(n, ms[i]))) for i in range(N)]
R0c = cgm.R0(yc)
check('C.4 O3 R4: point value at the centre lies inside the enclosure', lo <= R0c <= hi)
wid = hi - lo
print(f'C.4 O3 R4: R0 over the box in {outward(R0box, 70)}, width {mp.nstr(wid, 3)}; sign {sign}')
out['C.4']['R4']['O3 R4'] = {'sign': sign, 'enclosure': outward(R0box, 70), 'width': mp.nstr(wid, 5),
                          'route': 'interval CG coupling over the Section-5 box (radius 1e-60), mpmath.iv at 90 digits'}
save_json('out_C4.json', out)
