"""Audit of anchors 0(a)-0(e) in the three rooms against my stage-1 values (own/group.json, own/anchors.json),
plus convention-faithful checks of the solvers' eta, rho and C.
Each solver's eta is tested in THAT solver's stated convention: quaternion -> U = [[w - iz, -y - ix], [y - ix, w + iz]]
(l6 and r2 both state this) and D acting by the substitution x -> U00 x + U10 y, y -> U01 x + U11 y (both state it;
it is also the substitution of my own Dw). The D matrices are built by MY code (own/lib_common.Dw) with their U.
'plant': corrupts one eta entry, one character value, C -> I, and the 0(e) sign; the matching checks must FAIL."""
from au import *

log = Log(__file__, '_plant' if planted() else '')
PL = planted()
G = own_json('group.json')

# ---------- my table, keyed by Re g (as sympy) ----------
def k_of(pair):
    return sp.Rational(pair[0]) + sp.Rational(pair[1]) * sp.sqrt(5)

my_re = [sp.nsimplify(k_of(c['re'])) for c in G['classes']]
my_rows = {name: {my_re[i]: sp.nsimplify(k_of(v)) for i, v in enumerate(vals)} for name, vals in G['irr'].items()}
my_size = {my_re[i]: c['size'] for i, c in enumerate(G['classes'])}
my_order = {my_re[i]: c['order'] for i, c in enumerate(G['classes'])}
slots_mine = G['slots']


def eqv(a, b):
    return sp.simplify(sp.nsimplify(a) - sp.nsimplify(b)) == 0


def match_row(row):
    for name, r in my_rows.items():
        if all(eqv(row[k], r[k]) for k in r):
            return name
    return None


def compare_table(label, classes, rows, rekey, names):
    re = [sp.nsimplify(sym(c[rekey])) for c in classes]
    log.check(f'{label} 0(a): 9 classes, Re g values = mine', len(re) == 9 and set(map(str, re)) == set(map(str, my_re)))
    log.check(f'{label} 0(a): class sizes and element orders = mine',
              all(c['size'] == my_size[r] and c['element order'] == my_order[r] for c, r in zip(classes, re)))
    found = {}
    for idx, row in enumerate(rows):
        vals = [sym(v) for v in row]
        if PL and label == 'l6' and idx == 4:
            vals[1] = vals[1] + 1  # planted: one character value altered
        m = match_row(dict(zip(re, vals)))
        found[idx] = m
    log.check(f'{label} 0(a): every character row equals one of my 9 rows, bijectively',
              None not in found.values() and len(set(found.values())) == 9, str(found))
    for slot, idx in names.items():
        log.check(f'{label} 0(b): name {slot} -> the same irreducible as mine ({slots_mine[slot]})', found.get(idx) == slots_mine[slot],
                  f'theirs -> {found.get(idx)}')
    return found


# l6
L6 = solver('solver_l6'); R2S = solver('solver_r2'); RG = solver('solver_rig')
f_l6 = compare_table('l6', L6['0(a)']['classes'], [r['values'] for r in L6['0(a)']['irreducible characters (rows; columns = classes in the order above)']],
                     'Re g', {'R2': 2, 'R4': 4, 'R5': 6})
f_r2 = compare_table('r2', R2S['0(a)']['classes'], [r['values'] for r in R2S['0(a)']['characters']], 'Re q', {'R2': 2, 'R4': 4, 'R5': 6})
rg_names = list(RG['0(a)']['characters'].keys())
f_rg = compare_table('rig', RG['0(a)']['classes'], [RG['0(a)']['characters'][n] for n in rg_names], 'Re g',
                     {s: rg_names.index(RG['0(b)']['names'][s]) for s in ('R2', 'R4', 'R5')})

# ---------- 0(b) multiplicities ----------
def mylevels(name):
    return {l: m for l, m in enumerate(G['mult'][name]) if m}

for label, found, lev in (
        ('l6', f_l6, [{int(k): v for k, v in r['levels <= 20 with multiplicity'].items()} for r in L6['0(b)']['irreducibles']]),
        ('r2', f_r2, [{a: b for a, b in r['levels <= 20 (level, multiplicity)']} for r in R2S['0(b)']['irreducibles']]),
        ('rig', f_rg, [{int(k): v for k, v in RG['0(b)'][n]['levels'].items()} for n in rg_names])):
    ok = all(lev[i] == mylevels(found[i]) for i in range(9) if found[i])
    log.check(f'{label} 0(b): levels <= 20 with multiplicities = mine for all 9 irreducibles', ok)

# ---------- 0(c) eta in the solvers' own convention ----------
def U_theirs(q):
    w, x, y, z = q
    I = K(0, 0, 1, 0)
    return [[w - I * z, -y - I * x], [y - I * x, w + I * z]]


def D_cs(Ufun, q, j2):
    Dm = Dw(Ufun(q), j2)
    s = [sp.sqrt(s2(j2, m2)) for m2 in ms(j2)]
    n = j2 + 1
    return sp.Matrix(n, n, lambda a, b: sp.radsimp(Dm[a][b].tosympy() * s[a] / s[b]))


from lib_common import GEN1, GEN2
gens = (GEN1, GEN2)


def chi(name, q):
    return my_rows[name][sp.nsimplify(q[0].tosympy())]


def check_eta(label, slot, eta, Ufun, conv, graded=True):
    j2 = 7 if slot == 'R2' else 6
    chk = log.check if graded else (lambda name, ok, detail='': log(f'  (informative, not a check) {name}: {ok}'))
    d = eta.cols
    ok_on = sp.simplify(eta.H * eta - sp.eye(d)) == sp.zeros(d, d)
    chk(f'{label} 0(c) {slot}: eta^H eta = I (exact)', ok_on)
    P = eta * eta.H
    inv = True; trc = True; rhos = []
    for q in gens:
        D = D_cs(Ufun, q, j2)
        inv &= sp.simplify((sp.eye(j2 + 1) - P) * D * eta) == sp.zeros(j2 + 1, d)
        rho = (eta.H * D * eta).applyfunc(sp.simplify)
        rhos.append(rho)
        trc &= eqv(rho.trace(), chi(slots_mine[slot], q))
    chk(f'{label} 0(c) {slot}: range(eta) invariant under D(q1), D(q2) in {conv} convention (exact)', inv)
    chk(f'{label} 0(c) {slot}: tr(eta^H D(q_i) eta) = chi_{slot}(q_i) (my character)', trc)
    return rhos, inv


def mat(rows):
    return sp.Matrix([[sym(x) for x in r] for r in rows])


e4 = mat(L6['0(c)']['R4']['eta (rows m=3..-3)']); e5 = mat(L6['0(c)']['R5']['eta (rows m=3..-3)'])
if PL:
    e4[2, 0] = -e4[2, 0]  # planted: one sign flipped
check_eta('l6', 'R4', e4, U_theirs, "l6's")
check_eta('l6', 'R5', e5, U_theirs, "l6's")
# the same eta in MY convention: informative only (eta is convention-dependent), reported not graded
_, inv_mine = check_eta('l6 eta read in MY convention', 'R4', e4, quat_to_U, 'my', graded=False)
log(f'  (informative) l6 eta R4 in my quaternion convention invariant: {inv_mine}')

# r2: rows listed m = -7/2 .. 7/2 (ascending): reverse to my descending order
er2 = mat(R2S['0(c)']['R2']['eta (rows m = -7/2..7/2)'])
er2 = sp.Matrix([er2.row(er2.rows - 1 - i) for i in range(er2.rows)])
rhos, _ = check_eta('r2', 'R2', er2, U_theirs, "r2's")
r1 = mat(R2S['0(c)']['R2']['rho(q1)']); r2m = mat(R2S['0(c)']['R2']['rho(q2)'])
log.check('r2 0(c): reported rho(q1), rho(q2) = eta^H D(q_i) eta (exact, my D in their convention)',
          sp.simplify(r1 - rhos[0]) == sp.zeros(2, 2) and sp.simplify(r2m - rhos[1]) == sp.zeros(2, 2))
# s_N
log.check('l6/r2 0(c): s_N = 3/7, 4/7, 1/4 equal mine', L6['0(c)']['R4']['s_N'] == '3/7' and L6['0(c)']['R5']['s_N'] == '4/7'
          and R2S['0(c)']['R2']['s_N'] == '1/4' and own_json('anchors.json')['R2']['s_N'] == '1/4')

# ---------- 0(d) ----------
A = own_json('anchors.json')
log.check('l6/r2 0(d): CG value sqrt(231)/462 and sign +; Theta^2 = +1 (V_3), -1 (V_7/2), all = mine',
          eqv(sym(L6['0(d)']['<3 3; 3 -3 | 6 0>']), sp.sqrt(231) / 462) and eqv(sym(R2S['0(d)']['<3 3; 3 -3|6 0>']), sp.sqrt(231) / 462)
          and L6['0(d)']['sign'] == '+' == R2S['0(d)']['sign <3 3; 3 -3|6 0>'] == A['CG sign']
          and L6['0(d)']['Theta^2 on V_3'] == '1' == R2S['0(d)']['Theta^2 on V_3'] and L6['0(d)']['Theta^2 on V_7/2'] == '-1' == R2S['0(d)']['Theta^2 on V_7/2'])
# my own CG by lowering (independent of both): |6 6> = |33>|33>; J- repeatedly; read <3 3; 3 -3|6 0>
def cg_6_0():
    j = sp.Integer(3)
    def jm(m):
        return sp.sqrt((j + m) * (j - m + 1))
    st = {(3, 3): sp.Integer(1)}
    for _ in range(6):
        new = {}
        for (a, b), c in st.items():
            if a > -3: new[(a - 1, b)] = new.get((a - 1, b), 0) + c * jm(a)
            if b > -3: new[(a, b - 1)] = new.get((a, b - 1), 0) + c * jm(b)
        nr = sp.sqrt(sum(v ** 2 for v in new.values()))
        st = {k: sp.simplify(v / nr) for k, v in new.items()}
    return st[(3, -3)]
cgv = cg_6_0()
log(f'  my CG <3 3; 3 -3|6 0> by lowering |6 6> = {cgv}')
log.check('0(d): my lowering route gives sqrt(231)/462', eqv(cgv, sp.sqrt(231) / 462))

# ---------- 0(e) ----------
C = mat(R2S['0(e)']['C'])
if PL:
    C = sp.eye(2)
ok = all(sp.simplify(r.conjugate() * C - C * r) == sp.zeros(2, 2) for r in rhos)
log.check('r2 0(e): conj(rho(h)) C = C rho(h) for both generators, with rho from their eta', ok)
log.check('r2 0(e): C unitary and C conj(C) = -I', sp.simplify(C * C.H - sp.eye(2)) == sp.zeros(2, 2) and
          sp.simplify(C * C.conjugate() + sp.eye(2)) == sp.zeros(2, 2))
# solution space of conj(rho) X = X rho : dimension
xs = sp.symbols('x0:8', real=True)
X = sp.Matrix(2, 2, lambda a, b: xs[2 * (2 * a + b)] + sp.I * xs[2 * (2 * a + b) + 1])
eqs = []
for r in rhos:
    E = (r.conjugate() * X - X * r).applyfunc(sp.expand)
    for e in E:
        eqs += [sp.re(e), sp.im(e)]
Mx = sp.Matrix([[sp.diff(e, v) for v in xs] for e in eqs]).applyfunc(sp.simplify)
log.check('r2 0(e): the real solution space of conj(rho) X = X rho is 2-dimensional (one complex line)', 8 - Mx.rank() == 2,
          f'nullity {8 - Mx.rank()}')

# the map psi -> conj(psi) C, read on w, at 50 digits on random group points (their convention, my code)
mpmath.mp.dps = 50
import random
rng = random.Random(7)
E_th = sp.zeros(8, 8); ml = ms(7)
for r_, m2 in enumerate(ml):
    E_th[r_, ml.index(-m2)] = (-1) ** ((7 - m2) // 2)
def Theta(wv):
    return E_th * wv.conjugate()
def Unum_theirs(Um):
    return Um
etaN = mpmath.matrix([[mpmath.mpc(complex(sp.N(er2[a, b], 60))) if False else mpmath.mpc(str(sp.N(sp.re(er2[a, b]), 60)), str(sp.N(sp.im(er2[a, b]), 60))) for b in range(2)] for a in range(8)])
CN = mpmath.matrix([[mpmath.mpc(str(sp.N(sp.re(C[a, b]), 60)), str(sp.N(sp.im(C[a, b]), 60))) for b in range(2)] for a in range(2)])
sign = 1 if PL else -1
maxres = 0
for trial in range(4):
    wv = sp.Matrix([sp.Rational(rng.randint(-9, 9), 7) + sp.I * sp.Rational(rng.randint(-9, 9), 5) for _ in range(8)])
    wp = sign * Theta(wv)
    wN = mpmath.matrix([[mpmath.mpc(str(sp.re(x)), str(sp.im(x))) for x in wv]])
    wpN = mpmath.matrix([[mpmath.mpc(str(sp.N(sp.re(x), 60)), str(sp.N(sp.im(x), 60))) for x in wp]])
    for g in range(3):
        U = rand_su2(rng)  # any SU(2) element; the identity must hold on all of SU(2)
        D = Dnum(U, 7)
        lhs = (wN * D * etaN).apply(mpmath.conj) * CN
        rhs = wpN * D * etaN
        maxres = max(maxres, mpmath.mnorm(lhs - rhs, 1))
log(f'  0(e): max |conj(psi_w) C - psi_(-Theta w)| over 12 (w, g) samples = {mpmath.nstr(maxres, 3)} (50 digits)')
log.check('r2 0(e): psi -> conj(psi) C equals w -> -Theta(w) on E_0 (to 1e-40 at 50 digits)', maxres < mpmath.mpf(10) ** -40)
log.close()
