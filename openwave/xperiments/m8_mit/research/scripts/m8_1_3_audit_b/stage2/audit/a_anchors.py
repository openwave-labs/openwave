"""Audit of the anchors 0(a)-0(e) in all three rooms against my own derived data (copied stage-1 library).
* 0(a): each solver's classes and characters are mapped to mine through the class's real part (which determines the
  class); the set of character rows must coincide exactly (sympy in Q(sqrt5)).
* 0(b): each solver's level table per irreducible (matched by its character) == my exact multiplicities, l <= 20.
* 0(c): cb and cert use my D^j convention (same quaternion map, same action): their exact eta must give exactly MY exact
  projector P = eta eta^T, and eta^T eta = I.  ca uses a different quaternion-to-matrix map and action: I rebuild ca's
  D^j myself from ca's stated convention and test ca's eta (D eta = eta rho, tr rho = chi, for all 120 elements) and
  ca's exact P (= its own projector) at 50 digits.
* 0(e): cb: (Theta eta) C = -eta exactly; ca: U conj(eta) C = eta in ca's convention (50 digits); C conj(C) = -1.
Argument 'plant': flips one sign in cert's eta; the exact P comparison must fail."""
import sys, os, json
sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
if len(sys.argv) > 1:
    os.environ['PLANT'] = sys.argv[1]
from fractions import Fraction as Fr
from math import comb, factorial
import sympy as sp
from mpmath import mp
from core import Q5, mlist, Fm, k_to_complex, qkey
from reps import group_data, slot_irrep, projector_mono, multiplicity, SLOTS
from chk import check, save_json, planted

mp.dps = 50
G, classes, cls_of, chars = group_data()
s5 = sp.sqrt(5)
tosym = lambda q: sp.Rational(q.a.numerator, q.a.denominator) + sp.Rational(q.b.numerator, q.b.denominator) * s5
my_re = [sp.nsimplify(tosym(cl[0][0])) for cl in classes]
my_rows = [tuple(sp.nsimplify(tosym(v)) for v in c) for c in chars]
R = {}
def parse(s):
    return sp.nsimplify(sp.sympify(str(s).replace('sqrt5', 'sqrt(5)'), locals={'sqrt': sp.sqrt}))

def cmp_table(tag, res, rows):
    perm = [my_re.index(r) for r in [sp.simplify(x) for x in res]]
    check(f'0(a) {tag}: class real parts are a permutation of mine', sorted(perm) == list(range(9)))
    mapped = set()
    for r in rows:
        v = [None] * 9
        for i, val in enumerate(r):
            v[perm[i]] = sp.simplify(val)
        mapped.add(tuple(v))
    check(f'0(a) {tag}: the 9 character rows equal mine exactly (as class functions)', mapped == set(my_rows))
    return perm

cad = json.load(open(os.path.join(here, '..', 'solver_ca', 'results.json')))
cbd = json.load(open(os.path.join(here, '..', 'solver_cb', 'results.json')))
ced = json.load(open(os.path.join(here, '..', 'solver_cert', 'results.json')))
# sizes / orders
mine_so = sorted((len(cl), ) for cl in classes)
pa = cmp_table('ca', [parse(c['real_part']) for c in cad['0(a)']['classes']], [[parse(v) for v in r] for r in cad['0(a)']['characters (columns = classes in the order above)'].values()])
pb = cmp_table('cb', [parse(c['real part']) for c in cbd['0(a)']['conjugacy classes']], [[parse(v) for v in r] for r in cbd['0(a)']['irreducible characters (label dim@lowest level; columns = classes above)'].values()])
pc = cmp_table('cert', [parse(c['real part w = cos(angle/2)']) for c in ced['0(a)']['conjugacy classes']], [[parse(v) for v in r['values']] for r in ced['0(a)']['irreducible characters (values in the class order above)']])
for tag, cls, key_s, key_o, perm in (('ca', cad['0(a)']['classes'], 'size', 'order', pa), ('cb', cbd['0(a)']['conjugacy classes'], 'size', 'element order', pb),
                                     ('cert', ced['0(a)']['conjugacy classes'], 'size', 'element order', pc)):
    ok = all(int(c[key_s]) == len(classes[perm[i]]) for i, c in enumerate(cls))
    from core import element_order
    ok2 = all(int(c[key_o]) == element_order(classes[perm[i]][0]) for i, c in enumerate(cls))
    check(f'0(a) {tag}: class sizes and element orders equal mine', ok and ok2)

# ---------------- 0(b)
def my_levels(row):
    irr = my_rows.index(row)
    return {l: multiplicity(l, irr) for l in range(21) if multiplicity(l, irr) > 0}

def cmp_levels(tag, perm, rows_levels):
    ok = True
    for row, lv in rows_levels:
        v = [None] * 9
        for i, val in enumerate(row):
            v[perm[i]] = sp.simplify(val)
        mine = my_levels(tuple(v))
        theirs = {int(k): int(x) for k, x in lv.items()}
        if mine != theirs:
            ok = False; print('   mismatch', tag, row, mine, theirs)
    check(f'0(b) {tag}: levels <= 20 with multiplicities equal mine for all 9 irreducibles', ok)

ca_rows = cad['0(a)']['characters (columns = classes in the order above)']
cmp_levels('ca', pa, [([parse(v) for v in ca_rows[k]], cad['0(b)']['levels l <= 20 with multiplicities, per irreducible'][k]) for k in ca_rows])
cb_rows = cbd['0(a)']['irreducible characters (label dim@lowest level; columns = classes above)']
cmp_levels('cb', pb, [([parse(v) for v in cb_rows[k]], cbd['0(b)'][k]['levels (l: multiplicity), l <= 20']) for k in cb_rows])
ce_rows = ced['0(a)']['irreducible characters (values in the class order above)']
cmp_levels('cert', pc, [([parse(v) for v in r['values']], lv['levels <= 20 with multiplicity']) for r, lv in zip(ce_rows, ced['0(b)']['irreducibles'])])
# slot identification
def slot_row(tag, perm, row):
    v = [None] * 9
    for i, val in enumerate(row):
        v[perm[i]] = sp.simplify(val)
    return my_rows.index(tuple(v))
for nm, k in cad['0(b)']['slot -> irreducible'].items():
    check(f'0(b) ca: {nm} -> {k} is my slot irreducible', slot_row('ca', pa, [parse(v) for v in ca_rows[k]]) == slot_irrep(nm))
for nm, k in cbd['0(b)']['slot names'].items():
    lab = k.split()[0]
    check(f'0(b) cb: {nm} -> {lab} is my slot irreducible', slot_row('cb', pb, [parse(v) for v in cb_rows[lab]]) == slot_irrep(nm))

# ---------------- 0(c): exact eta of cb and cert against my exact projector (same convention)
def my_P_ortho_exact(nm):
    d, n = SLOTS[nm]; ms = mlist(n); N = n + 1
    Pm = projector_mono(nm)
    out = sp.zeros(N, N)
    for a in range(N):
        for b in range(N):
            x = Pm[a][b]
            val = (sp.Rational(x[0].numerator, x[0].denominator) + sp.Rational(x[1].numerator, x[1].denominator) * s5
                   + sp.I * (sp.Rational(x[2].numerator, x[2].denominator) + sp.Rational(x[3].numerator, x[3].denominator) * s5))
            out[a, b] = sp.nsimplify(val * sp.sqrt(sp.Rational(Fm(n, ms[a]), Fm(n, ms[b]))))
    return out

for tag, dat, key in (('cb', cbd['0(c)'], 'eta (rows m = j..-j, columns a)'), ('cert', ced['0(c)'], 'eta (rows m = 3..-3, columns a), exact')):
    for nm in dat:
        eta = sp.Matrix([[parse(v) for v in row] for row in dat[nm][key]])
        if planted('plant') and tag == 'cert' and nm == 'R5':
            eta[2, 0] = -eta[2, 0]
        P = my_P_ortho_exact(nm)
        okI = sp.simplify(eta.H * eta - sp.eye(eta.shape[1])) == sp.zeros(eta.shape[1], eta.shape[1])
        okP = sp.simplify(eta * eta.H - P) == sp.zeros(P.shape[0], P.shape[0])
        check(f'0(c) {tag} {nm}: eta^dagger eta = I exactly', okI)
        check(f'0(c) {tag} {nm}: eta eta^dagger = MY exact isotypic projector (same D^j convention)', okP)
        check(f'0(c) {tag} {nm}: s_N = dim/(2j+1)', Fr(dat[nm]['s_N']) == Fr(SLOTS[nm][0], SLOTS[nm][1] + 1))
        check(f'0(c) {tag} {nm}: multiplicity 1', int(dat[nm][[k for k in dat[nm] if k.startswith('multiplicity')][0]]) == 1)

# ---------------- 0(c) ca, in ca's own convention (rebuilt here)
def D_ca(q, n):
    w, x, y, z = [mp.mpf(c.a.numerator) / c.a.denominator + mp.mpf(c.b.numerator) / c.b.denominator * mp.sqrt(5) for c in q]
    a = mp.mpc(w, x); b = mp.mpc(y, z); c = mp.mpc(-y, z); d = mp.mpc(w, -x)
    j2 = n; ms = mlist(n); N = n + 1
    M = mp.matrix(N, N)
    for col, mcol in enumerate(ms):
        p_ = int(Fr(j2, 2) + mcol); r_ = j2 - p_
        # (a x + c y)^p (b x + d y)^r
        for t1 in range(p_ + 1):
            c1 = comb(p_, t1) * a ** t1 * c ** (p_ - t1)
            for t2 in range(r_ + 1):
                c2 = comb(r_, t2) * b ** t2 * d ** (r_ - t2)
                px = t1 + t2
                row = j2 - px
                M[row, col] += c1 * c2
    for i in range(N):
        for k in range(N):
            M[i, k] *= mp.sqrt(mp.mpf(Fm(n, ms[i])) / Fm(n, ms[k]))
    return M

for nm in ['R2', 'R4', 'R5']:
    d, n = SLOTS[nm]; N = n + 1
    e = cad['0(c)'][nm]['eta (columns; rows m = j..-j; 40 digits, real and imaginary parts)']
    eta = mp.matrix(N, d)
    for i in range(N):
        for a in range(d):
            eta[i, a] = mp.mpc(e[i][a][0], e[i][a][1])
    chi = chars[slot_irrep(nm)]
    worst = 0; worst_tr = 0; worst_hom = 0
    for g in G:
        Dg = D_ca(g, n)
        rho = eta.H * Dg * eta
        worst = max(worst, mp.mnorm(Dg * eta - eta * rho, 1))
        worst_tr = max(worst_tr, abs(sum(rho[a, a] for a in range(d)) - mp.mpf(float(chi[cls_of[qkey(g)]]))))
    # unitarity / homomorphism of my rebuild of ca's D (so the test is meaningful)
    import random
    rng = random.Random(1)
    g1, g2 = G[rng.randrange(120)], G[rng.randrange(120)]
    from core import qmul
    worst_hom = mp.mnorm(D_ca(qmul(g1, g2), n) - D_ca(g1, n) * D_ca(g2, n), 1)
    check(f'0(c) ca {nm}: my rebuild of ca D^j is a homomorphism (sanity)', worst_hom < mp.mpf(10) ** -40, mp.nstr(worst_hom, 3))
    check(f'0(c) ca {nm}: D(h) eta = eta rho(h) for all 120 h, in ca convention (40-digit eta)', worst < mp.mpf(10) ** -35, mp.nstr(worst, 3))
    check(f'0(c) ca {nm}: tr rho(h) = chi(h) up to float precision of my chi (all h)', worst_tr < 1e-12, mp.nstr(worst_tr, 3))
    check(f'0(c) ca {nm}: eta^dagger eta = I', mp.mnorm(eta.H * eta - mp.eye(d), 1) < mp.mpf(10) ** -35)
    Pe = cad['0(c)'][nm]['P = eta eta^dagger, exact, orthonormal CS basis']
    Pm = mp.matrix([[mp.mpf(str(sp.N(parse(v), 60))) for v in row] for row in Pe])
    check(f'0(c) ca {nm}: ca exact P equals eta eta^dagger (40 digits)', mp.mnorm(Pm - eta * eta.H, 1) < mp.mpf(10) ** -35)
    # ca's P vs ca-convention isotypic projector
    Pc = mp.matrix(N, N)
    for g in G:
        Pc += mp.mpf(float(chi[cls_of[qkey(g)]])) * D_ca(g, n) if False else 0 * D_ca(g, n)
    R.setdefault('0(c) ca', {})[nm] = 'eta passes in ca convention'
    if nm == 'R2':
        C = cad['0(e)']['R2']['C (unitary, phase chosen so that the map is exactly Theta; 30 digits)']
        Cm = mp.matrix([[mp.mpc(C[i][k][0], C[i][k][1]) for k in range(2)] for i in range(2)])
        ms = mlist(n)
        W = mp.matrix(N, N)
        for a, mu in enumerate(ms):
            W[a, ms.index(-mu)] = (-1) ** int(Fr(n, 2) - mu)
        lhs = W * eta.apply(mp.conj) * Cm
        check('0(e) ca: U conj(eta) C = eta (map is exactly Theta) at 28 digits', mp.mnorm(lhs - eta, 1) < mp.mpf(10) ** -28, mp.nstr(mp.mnorm(lhs - eta, 1), 3))
        check('0(e) ca: C conj(C) = -1 at 28 digits', mp.mnorm(Cm * Cm.apply(mp.conj) + mp.eye(2), 1) < mp.mpf(10) ** -28)
        worstC = max(mp.mnorm((eta.H * D_ca(g, n) * eta).apply(mp.conj) * Cm - Cm * (eta.H * D_ca(g, n) * eta), 1) for g in G)
        check('0(e) ca: conj(rho(h)) C = C rho(h) for all h', worstC < mp.mpf(10) ** -28, mp.nstr(worstC, 3))

# ---------------- 0(e) cb, exact in my convention
eta = sp.Matrix([[parse(v) for v in row] for row in cbd['0(c)']['R2']['eta (rows m = j..-j, columns a)']])
C = sp.Matrix([[parse(v) for v in row] for row in cbd['0(e)']['R2']['C']])
ms = mlist(7)
W = sp.zeros(8, 8)
for a, mu in enumerate(ms):
    W[a, ms.index(-mu)] = (-1) ** int(Fr(7, 2) - mu)
check('0(e) cb: (Theta eta) C = -eta exactly (map w -> -Theta w)', sp.simplify(W * eta.conjugate() * C + eta) == sp.zeros(8, 2))
check('0(e) cb: C conj(C) = -1 exactly', C * C.conjugate() == -sp.eye(2))
# conj(rho)C = C rho for the generators, exact rho from my D (numerically at 50 digits)
from numrep import D_of_q
from core import Q1, Q2
etam = mp.matrix([[mp.mpf(str(sp.N(v, 60))) for v in row] for row in eta.tolist()])
Cm = mp.matrix([[mp.mpf(int(v)) for v in row] for row in C.tolist()])
wc = 0
for g in G:
    rho = etam.H * D_of_q(g, 7) * etam
    wc = max(wc, mp.mnorm(rho.apply(mp.conj) * Cm - Cm * rho, 1))
check('0(e) cb: conj(rho(h)) C = C rho(h) for all 120 h (my D, 50 digits)', wc < mp.mpf(10) ** -40, mp.nstr(wc, 3))
# 0(d)
from hpoly import cg_exact
sg, sq = cg_exact(3, 3, 3, -3, 6, 0)
check('0(d): my <3 3;3 -3|6 0> = +1/sqrt(924)', sg == 1 and sq == Fr(1, 924))
for tag, v in (('ca', cad['0(d)']['<3 3; 3 -3|6 0>']), ('cb', cbd['0(d)']['<3 3; 3 -3 | 6 0>']['value']), ('cert', ced['0(d)']['<3 3; 3 -3 | 6 0>'])):
    check(f'0(d) {tag}: value equals +1/sqrt(924) exactly', sp.simplify(parse(v) - 1 / sp.sqrt(924)) == 0, v)
save_json('a_anchors.json', R)
