"""Item 0: <3 3; 3 -3 | 6 0>, Theta^2, and r6 at two points (exact + two independent 50-digit routes)."""
import sympy as sp, mpmath as mp
import common as C

ck = C.Checker('item0')
I = sp.I
res = {}

# --- CG: own lowering construction vs own Racah formula vs sympy library (lookup)
cg = C.CG()
bad = [k for k, v in cg.items() if sp.expand(C.cg_racah(3, k[0], 3, k[1], k[2], k[3]) ** 2 - v ** 2) != 0
       or sp.sign(C.cg_racah(3, k[0], 3, k[1], k[2], k[3])) != sp.sign(v)]
ck.check('lowering construction == Racah formula on all %d entries' % len(cg), not bad, str(bad[:3]))
from sympy.physics.quantum.cg import CG as LibCG
badlib = [k for k, v in cg.items() if sp.nsimplify(LibCG(3, k[0], 3, k[1], k[2], k[3]).doit() - v) != 0]
ck.check('lowering construction == sympy.physics.quantum.cg (library lookup)', not badlib, str(badlib[:3]))
# orthonormality of the full CG table (unitarity) - a genuine check of the construction
pairs = [(a, b) for a in C.MS for b in C.MS]
orth_bad = []
for (L1, L2) in [(6, 6), (6, 4), (5, 3), (2, 2), (0, 0)]:
    for M in range(-min(L1, L2), min(L1, L2) + 1):
        ip = sum(C.cgL(a, b, L1, M) * C.cgL(a, b, L2, M) for a, b in pairs if a + b == M)
        if sp.nsimplify(ip) != (1 if L1 == L2 else 0):
            orth_bad.append((L1, L2, M, ip))
ck.check('CG orthonormality on sampled (L1,L2,M)', not orth_bad, str(orth_bad[:3]))
# stretched formula used by the mp route
okst = all(sp.nsimplify(C.cg6(a, Q - a, Q) ** 2 - sp.binomial(6, 3 + a) * sp.binomial(6, 3 + Q - a) / sp.binomial(12, 6 + Q)) == 0
           and C.cg6(a, Q - a, Q) > 0 for Q in range(-6, 7) for a in C.MS if abs(Q - a) <= 3)
ck.check('L=6 coefficients equal sqrt(C(6,3+m1)C(6,3+m2)/C(12,6+Q)), all positive', okst)

v = cg[(3, -3, 6, 0)]
res['cg_33_3m3_60'] = str(v)
res['cg_33_3m3_60_squared'] = str(v ** 2)
ck.check('<3 3;3 -3|6 0>^2 == 1/924', v ** 2 == sp.Rational(1, 924), str(v))

# --- Theta^2
a = sp.symbols('a0:7'); b = sp.symbols('b0:7', real=True)
cs = [sp.Symbol('p%d' % k, real=True) + I * sp.Symbol('q%d' % k, real=True) for k in range(7)]
tt = C.theta(C.theta(cs))
ok = all(sp.expand(tt[k] - cs[k]) == 0 for k in range(7))
ck.check('Theta(Theta u) == u symbolically', ok)
res['Theta_squared'] = 'Theta(Theta u) = u   ((-1)^(3-m)(-1)^(3+m) = +1)'

# --- the two points
def vec(d):
    c = [sp.Integer(0)] * 7
    for m, z in d.items(): c[C.idx(m)] = sp.sympify(z)
    return c
pts = {'u1 = 2v3 + v1 - 3v-2': vec({3: 2, 1: 1, -2: -3}),
       'u2 = v3 + i v0 + 2v-1': vec({3: 1, 0: I, -1: 2})}
res['points'] = {}
for name, c in pts.items():
    ex = C.rhat_exact(c)
    N = C.N_exact(c); n2 = C.norm2(c)
    out = {'rhat6': str(ex), 'N': str(N), 'norm2': str(n2)}
    for dps in (50, 80):
        mp.mp.dps = dps
        cm = C.mp_vec(c)
        v1 = C.mp_rhat(cm)
        v2 = C.mp_rL_casimir(cm, 6)
        e = mp.mpf(ex.p) / ex.q
        out['dps%d' % dps] = {'cg_route': mp.nstr(v1, 30), 'casimir_route': mp.nstr(v2, 30),
                              'err_cg': mp.nstr(abs(v1 - e), 3), 'err_casimir': mp.nstr(abs(v2 - e), 3)}
        ck.check('%s dps=%d CG route agrees with exact' % (name, dps), abs(v1 - e) < mp.mpf(10) ** (-dps + 5), mp.nstr(abs(v1 - e), 3))
        ck.check('%s dps=%d Casimir route agrees with exact' % (name, dps), abs(v2 - e) < mp.mpf(10) ** (-dps + 8), mp.nstr(abs(v2 - e), 3))
    res['points'][name] = out
    print(name, 'rhat6 =', ex, '=', out['dps50']['cg_route'])

# --- total over L: sum_L r_L = 1 and r_0 = 1/7 (structural checks on the exact route)
for name, c in pts.items():
    tot = sum(C.rhat_exact(c, L) for L in range(7))
    ck.check('%s: sum_L r_L == 1' % name, sp.nsimplify(tot) == 1, str(tot))
    ck.check('%s: r_0 == 1/7' % name, C.rhat_exact(c, 0) == sp.Rational(1, 7))

C.save('item0', res)
print('item0 fails:', ck.fails)
