"""E.1 from the derived multiplicity table (group.json); E.2 for two slots: trivial (n = 0, dim 1) and V_1/2 (n = 1, dim 2).
Route for E.2: psi = e^{i w T}(psi_v + zeta), zeta = psi_y in E_0 (psi_y(g) = y^T pi(g), pi = D^{n/2}(g) eta, eta unitary).
Because pi(g) is unitary, Re(psi_y . conj psi_v) = Re(v^dag y) is constant on SU(2) (checked numerically at random g),
so the linearized nonlinearity maps E_0 into E_0 as y -> 2 g4 Re(v^dag y) v, and -Delta - lambda_0 = 0 on E_0, with
w^2/c^2 = lambda_0 + g4 |v|^2. The first-order real system Z' = A Z, Z = (Y, Y'), Y = (Re y, Im y), is
  Y'' = -2 w i Y' - 2 g4 c^2 Re(v^dag y) v.
Eigenvalues: exact charpoly in sympy with symbols (w, g4, c, r = |v| > 0), v = (r, 0, ..) (SU(2) x phase moves any v there,
and the conjugation preserves the spectrum), and v generic numeric check; geometric multiplicities by exact rank.
'plant': drops the factor 2 in the nonlinear term (the numeric eigenvalue check must fire)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_common import *
from lib_quad import D_cs_num
import sympy as sp, mpmath, random

log = Log(__file__, '_plant' if planted() else '')
gj = load_json('group.json')
out = {'E.1': {}, 'E.2': {}}
log('E.1')
for name, row in gj['mult'].items():
    dim = int(Fr(gj['irr'][name][0][0]))
    n = min(l for l in range(21) if row[l])
    if row[n] == 1 and dim == n + 1:
        nxt = [l for l in range(n + 1, 21) if row[l]]
        l1 = nxt[0] if nxt else None
        lam = l1 * (l1 + 2) - n * (n + 2) if l1 is not None else None
        log(f'  {name:12s}: dim {dim}, lowest level n = {n}, V_n/2|Gamma0 = rho; next level {l1}; lambda1 - lambda0 = {lam} /R^2')
        out['E.1'][name] = dict(dim=dim, n=n, next_level=l1, gap=lam)
log.check('every listed slot has its next level within l <= 20', all(v['next_level'] is not None for v in out['E.1'].values()))

w, g4, c, r = sp.symbols('omega g_4 c r', positive=True)
lam = sp.Symbol('lambda')
for label, d in (('trivial (n=0)', 1), ('V_1/2 (n=1)', 2)):
    log(f'\nE.2 slot {label}, dim E_0 = {d} complex')
    # numerical structural check: |psi_v| constant, Re(psi_y conj psi_v) constant = Re(v^dag y)
    mpmath.mp.dps = 30
    rr = random.Random(5 + d)
    v = [mpmath.mpc(rr.uniform(-1, 1), rr.uniform(-1, 1)) for _ in range(d)]
    y = [mpmath.mpc(rr.uniform(-1, 1), rr.uniform(-1, 1)) for _ in range(d)]
    okc = True
    for _ in range(5):
        q = [mpmath.mpf(rr.gauss(0, 1)) for _ in range(4)]
        nq = mpmath.sqrt(sum(x * x for x in q)); q = [x / nq for x in q]
        U = [[mpmath.mpc(q[0], q[1]), mpmath.mpc(q[2], q[3])], [mpmath.mpc(-q[2], q[3]), mpmath.mpc(q[0], -q[1])]]
        D = D_cs_num(U, d - 1) if d > 1 else [[mpmath.mpc(1)]]
        pv = [sum(v[m] * D[m][a] for m in range(d)) for a in range(d)]
        py = [sum(y[m] * D[m][a] for m in range(d)) for a in range(d)]
        okc &= abs(sum(abs(x) ** 2 for x in pv) - sum(abs(x) ** 2 for x in v)) < 1e-25
        okc &= abs(mpmath.re(sum(a * mpmath.conj(b) for a, b in zip(py, pv))) - mpmath.re(sum(mpmath.conj(b) * a for a, b in zip(y, v)))) < 1e-25
    log.check(f'{label}: |psi_v| = |v| and Re(psi_y . conj psi_v) = Re(v^dag y) at random g', okc)
    # symbolic system
    n2 = 2 * d
    Jc = sp.Matrix(sp.BlockMatrix([[sp.zeros(d), -sp.eye(d)], [sp.eye(d), sp.zeros(d)]]))
    vre = [r] + [0] * (d - 1); vim = [0] * d
    vvec = sp.Matrix(vre + vim)
    L = vvec * vvec.T  # Re(v^dag y) v in real coordinates = (v . Y) v
    fac = 1 if planted() else 2
    A = sp.Matrix(sp.BlockMatrix([[sp.zeros(n2), sp.eye(n2)], [-fac * g4 * c ** 2 * L, -2 * w * Jc]]))
    cp = sp.factor((A - lam * sp.eye(2 * n2)).det())
    log(f'  characteristic polynomial det(A - lambda) = {cp}')
    roots = sp.roots(sp.Poly(sp.expand(cp), lam))
    res = []
    for rt, am in roots.items():
        gm = 2 * n2 - (A - rt * sp.eye(2 * n2)).applyfunc(sp.simplify).rank(simplify=True)
        res.append((sp.simplify(rt), am, gm))
        log(f'  eigenvalue {sp.simplify(rt)}: algebraic {am}, geometric {gm}')
    # numeric check with a generic complex v and random parameters, against the closed forms
    mpmath.mp.dps = 30
    wn, gn, cn = mpmath.mpf('1.3'), mpmath.mpf('0.7'), mpmath.mpf('1.9')
    vv = [mpmath.mpc(rr.uniform(-1, 1), rr.uniform(-1, 1)) for _ in range(d)]
    vr = [x.real for x in vv] + [x.imag for x in vv]
    rn = mpmath.sqrt(sum(x * x for x in vr))
    An = mpmath.matrix(2 * n2, 2 * n2)
    for i in range(n2): An[i, n2 + i] = 1
    for i in range(n2):
        for k in range(n2):
            An[n2 + i, k] = -2 * gn * cn ** 2 * vr[i] * vr[k]
    Jn = mpmath.matrix(n2, n2)
    for i in range(d): Jn[i, d + i] = -1; Jn[d + i, i] = 1
    for i in range(n2):
        for k in range(n2):
            An[n2 + i, n2 + k] = -2 * wn * Jn[i, k]
    ev = mpmath.eig(An)[0]
    pred = []
    for rt, am, gm in res:
        val = complex(sp.N(rt.subs({w: 1.3, g4: 0.7, c: 1.9, r: float(rn)}), 25))
        pred += [val] * am
    evs = sorted([complex(x) for x in ev], key=lambda z: (round(z.imag, 8), round(z.real, 8)))
    pred = sorted(pred, key=lambda z: (round(z.imag, 8), round(z.real, 8)))
    okn = len(evs) == len(pred) and all(abs(a - b) < 1e-6 for a, b in zip(evs, pred))
    log(f'  numeric (generic complex v, omega=1.3, g4=0.7, c=1.9): {[complex(round(z.real, 10), round(z.imag, 10)) for z in evs]}')
    log.check(f'{label}: numeric spectrum with generic v matches the closed forms', okn)
    out['E.2'][label] = [dict(eigenvalue=str(rt), algebraic=am, geometric=gm) for rt, am, gm in res]
    out['E.2'][label + ' charpoly'] = str(cp)
if not planted():
    save_json('E.json', out)
log.close()
