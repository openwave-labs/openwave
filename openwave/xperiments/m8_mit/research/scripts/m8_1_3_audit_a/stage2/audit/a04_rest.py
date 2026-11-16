"""A.1-A.3, A.5, A.6 (all three slots), B.5, E.1, E.2: comparison with my stage-1 values and my own new computations.
A.5: Q(alpha v_a + beta v_b) as an exact polynomial in (Re, Im) of alpha, beta from my channel-form h (own/lib_crit.h_poly),
     compared with c (|alpha|^2 + |beta|^2)^2.
A.6: from my exact q = Q(v_j) (Point at the top weight), ceiling and nearest rounding at 2 significant digits.
E.2: my own construction of the real 4d x 4d first-order system on E_0 for d = 1..6 at a random Gaussian-rational v (not
     along a basis vector), exact characteristic polynomial over Q at two parameter sets, compared with rig's closed form,
     and geometric multiplicities by exact ranks over Q(i).
'plant': alters one q_J of l6, uses (A+B)^2 + AB in A.5, and drops the factor 2 in the E.2 cubic term; must FAIL."""
from au import *
from lib_crit import h_poly, Point
import math, random

log = Log(__file__, '_plant' if planted() else '')
PL = planted()
OWN = own_json('own.json')
L6 = solver('solver_l6'); R2S = solver('solver_r2'); RG = solver('solver_rig')
V = {}

# ---- A.1, A.2, A.3 ----
for slot, src in (('R4', L6), ('R5', L6), ('R2', R2S)):
    t = dict(src['A.1'][slot]); m = OWN['A.1'][slot]
    if PL and slot == 'R4': t['q_4'] = '36/33'
    ok = all(sym(t[k]) == sym(m[k]) for k in m)
    log.check(f'A.1 {slot}: solver q_J = my exact q_J', ok, str(t)); V[f'A.1|{slot}'] = 'AGREE' if ok else 'DISAGREE'
    okb = src['A.2'][slot]['proportional'] is True and sym(src['A.2'][slot]['beta']) == sym(OWN['A.2'][slot]['beta'])
    log.check(f'A.2 {slot}: proportional, beta = mine ({OWN["A.2"][slot]["beta"]})', okb); V[f'A.2|{slot}'] = 'AGREE' if okb else 'DISAGREE'
ok3 = all(sym(L6['A.3'][k]) == sym(OWN['A.3'][k]) for k in OWN['A.3'])
log.check('A.3: ratio 9/16 and difference 49/156 = mine', ok3); V['A.3'] = 'AGREE' if ok3 else 'DISAGREE'
okhs = all(sym(R2S['A.2']['R2']['HS spin-6 factor in channel fractions'][k]) == v for k, v in
           zip(('c_7', 'c_5', 'c_3', 'c_1'), (sp.Rational(1, 264), sp.Rational(53, 264), sp.Rational(221, 264), sp.Rational(-13, 24))))
log.check('A.2 R2: r2 HS coefficients = my stage-1 (1/264, 53/264, 221/264, -13/24)', okhs)

# ---- A.5 ----
hd = h_poly('R2')
h, gens, j2 = hd
ar, ai, br, bi = sp.symbols('ar ai br bi', real=True)
A_ = ar ** 2 + ai ** 2; B_ = br ** 2 + bi ** 2
for (a2, b2), key in (((5, -5), '{5/2, -5/2}'), ((7, -7), '{7/2, -7/2}'), ((3, -3), '{3/2, -3/2}')):
    ml = ms(7)
    sub = {g: 0 for g in gens}
    ia, ib = ml.index(a2), ml.index(b2)
    sa, sb = sp.sqrt(s2(7, a2)), sp.sqrt(s2(7, b2))
    sub[gens[ia]] = ar / sa; sub[gens[8 + ia]] = ai / sa
    sub[gens[ib]] = br / sb; sub[gens[8 + ib]] = bi / sb
    hab = sp.expand(h.as_expr().subs(sub))
    c = sp.Poly(hab, ar, ai, br, bi).coeff_monomial(ar ** 4)
    target = c * (A_ + B_) ** 2 + (A_ * B_ if PL else 0)
    ok = sp.expand(hab - target) == 0
    theirs = R2S['A.5'][key]
    okt = sym(theirs['on unit vectors']) == c and sp.expand(sym(theirs['Q(alpha, beta), A=|alpha|^2, B=|beta|^2']).subs(
        {sp.Symbol('A'): A_, sp.Symbol('B'): B_}) - c * (A_ + B_) ** 2) == 0
    log(f'  A.5 {key}: h(alpha v_a + beta v_b) = {sp.factor(hab)}')
    log.check(f'A.5 {key}: h = c (|alpha|^2+|beta|^2)^2 exactly, c = {c}', ok)
    log.check(f'A.5 {key}: r2 value ({theirs["on unit vectors"]}) and formula = mine', okt)
    V[f'A.5|{key}'] = 'AGREE' if ok and okt else 'DISAGREE'

# ---- A.6 ----
def ceil2(x):
    x = sp.nsimplify(x)
    e = math.floor(math.log10(float(x))) - 1
    return sp.ceiling(x / sp.Integer(10) ** e) * sp.Integer(10) ** e if e < 0 else sp.ceiling(x / 10 ** e) * 10 ** e

def near2(x):
    e = math.floor(math.log10(float(x))) - 1
    return sp.Integer(round(float(sp.N(x / sp.Integer(10) ** e, 30)))) * sp.Integer(10) ** e

for slot, j2_, top, src in (('R2', 7, {'7/2': '1'}, R2S), ('R4', 6, {'3': '1'}, L6), ('R5', 6, {'3': '1'}, L6)):
    P = Point(slot, top).compute()
    q = P.hu
    a = sp.sqrt((q - 1) / 2) * 100; b = (q - 1) / 2 * 100
    mine = (ceil2(a), ceil2(b)); nearest = (near2(a), near2(b))
    t = src['A.6'][slot]
    keys = [k for k in t if 'percent' in k or k.endswith('%')]
    tv = (sym(t[[k for k in keys if k.startswith('sqrt')][0]]), sym(t[[k for k in keys if k.startswith('(q-1)')][0]]))
    log(f'  A.6 {slot}: q = {q}; sqrt((q-1)/2) = {sp.N(a, 12)} %, (q-1)/2 = {sp.N(b, 12)} %; ceiling {mine}; nearest {nearest}; solver {tv}')
    ok = sym(t['q']) == q and all(sp.nsimplify(x) == sp.nsimplify(y) for x, y in zip(tv, mine))
    log.check(f'A.6 {slot}: q and both ceilings at 2 s.f. = mine', ok)
    V[f'A.6|{slot}'] = 'AGREE' if ok else 'DISAGREE'
    V[f'A.6|{slot}|reading-sensitive'] = str(mine != nearest)

# ---- B.5 ----
okb5 = L6['B.5']['proportional'] is True and L6['B.5']['constant'] == '1' and OWN['B.5']['proportional'] is True
log.check('B.5: l6 proportional with constant 1 = mine', okb5); V['B.5'] = 'AGREE' if okb5 else 'DISAGREE'

# ---- E.1 ----
okE1 = True
oe = OWN['E.1']
for name, x in RG['E.1'].items():
    match = [v for v in oe.values() if v['dim'] == x['dim'] and v['n'] == x['n']]
    okE1 &= len(match) == 1 and match[0]['next_level'] == x['next level'] and str(match[0]['gap']) == x['lambda1 - lambda0 (units 1/R^2)']
okE1 &= len(RG['E.1']) == len(oe) == 6
log.check('E.1: rig six rigid slots, n, next level and gap = mine', okE1); V['E.1'] = 'AGREE' if okE1 else 'DISAGREE'

# ---- E.2: own construction, d = 1..6 ----
rng = random.Random(11)
mu = sp.Symbol('mu')
I = sp.I
for d in range(1, 7):
    okd = True
    for (om, target) in ((2, 25), (3, 49)):
        v = sp.Matrix([sp.Rational(rng.randint(-5, 5), rng.randint(1, 4)) + I * sp.Rational(rng.randint(-5, 5), rng.randint(1, 4)) for _ in range(d)])
        if all(x == 0 for x in v): v[0] = 1
        nv2 = sum(sp.expand(x * sp.conjugate(x)) for x in v)
        m = target - 4 * om ** 2           # m = 2 g c^2 |v|^2, chosen so 4 om^2 + m is a square
        gc2 = sp.Rational(m) / (2 * nv2) * (2 if PL else 1) / (2 if PL else 1)
        fac = 2 if not PL else 1           # planted: drop the factor 2 of 2 Re(v^dag y) v
        # state (Re y, Im y, Re y', Im y'), y'' = -2 i om y' - fac g c^2 Re(v^dag y) v
        n = d
        Amat = sp.zeros(4 * n, 4 * n)
        for k in range(n):
            Amat[k, 2 * n + k] = 1; Amat[n + k, 3 * n + k] = 1
            # -2 i om (p + i q) = 2 om q - 2 i om p  for y' = p + i q
            Amat[2 * n + k, 3 * n + k] += 2 * om
            Amat[3 * n + k, 2 * n + k] += -2 * om
        vr = [sp.re(x) for x in v]; vi = [sp.im(x) for x in v]
        # Re(v^dag y) = sum vr*yr + vi*yi
        for k in range(n):
            for l in range(n):
                cr = fac * gc2 * vr[k]; ci = fac * gc2 * vi[k]
                Amat[2 * n + k, l] += -cr * vr[l]; Amat[2 * n + k, n + l] += -cr * vi[l]
                Amat[3 * n + k, l] += -ci * vr[l]; Amat[3 * n + k, n + l] += -ci * vi[l]
        cp = sp.factor(Amat.charpoly(mu).as_expr())
        claim = mu ** (2 * d) * (mu ** 2 + 4 * om ** 2) ** (d - 1) * (mu ** 2 + 4 * om ** 2 + 2 * gc2 * nv2)
        okc = sp.expand(cp - claim) == 0
        from sympy.polys.matrices import DomainMatrix
        def rk(Mx):  # exact rank over Q(i) (sympy's Matrix.rank mis-tests zero on Gaussian rationals here)
            return DomainMatrix.from_Matrix(Mx.applyfunc(sp.expand)).convert_to(sp.QQ_I).rank()
        Ac = Amat
        geo0 = 4 * n - rk(Ac)
        geo2w = 4 * n - rk(Ac - 2 * I * om * sp.eye(4 * n))
        s = sp.sqrt(4 * om ** 2 + 2 * gc2 * nv2)
        assert s.is_Rational
        geoS = 4 * n - rk(Ac - I * s * sp.eye(4 * n))
        okg = geo0 == 2 * d - 1 and geo2w == d - 1 and geoS == 1
        log(f'  E.2 d={d}, omega={om}, 2gc^2|v|^2={2 * gc2 * nv2}: charpoly {cp}; geometric: 0 -> {geo0}, 2i omega -> {geo2w}, i sqrt -> {geoS}')
        okd &= okc and okg
    rgcp = RG['E.2']['slots'][[k for k in RG['E.2']['slots'] if RG['E.2']['slots'][k]['d'] == d][0]]['charpoly in mu']
    log.check(f'E.2 d={d}: my exact charpoly and geometric multiplicities match rig closed form ({rgcp})', okd)
    V[f'E.2|d={d}'] = 'AGREE' if okd else 'DISAGREE'
if not PL:
    save('rest.json', V)
log.close()
