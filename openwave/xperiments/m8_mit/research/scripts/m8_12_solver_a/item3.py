"""Item 3: the six classes of complex dimension 1 - value of rhat6 and exact tangential gradient.
Tangential gradient of rhat6 at a unit u (real coordinates): grad N(u) - 4 N(u) u."""
import sympy as sp
import common as C, classes as K

ck = C.Checker('item3')
N = C.N_poly()
gradN = [sp.diff(N, v) for v in C.XS]

def tangential_gradient(c):
    xv = C.c_to_x(c)
    sub = dict(zip(C.XS, list(xv)))
    Nu = sp.nsimplify(sp.radsimp(N.subs(sub)))
    g = sp.Matrix([sp.radsimp(e.subs(sub)) for e in gradN])
    return (g - 4 * Nu * xv).applyfunc(sp.radsimp), Nu

res = {}
for nm, d in K.CLASSES1.items():
    c = list(d['u'])
    assert sp.simplify(C.norm2(c) - 1) == 0
    g, Nu = tangential_gradient(c)
    val = C.rhat_exact(c)
    ck.check('%s: rhat6 from polynomial N equals rhat6 from rho_6' % nm, sp.nsimplify(Nu - val) == 0)
    ck.check('%s: exact tangential gradient is zero' % nm, all(e == 0 for e in g), str([e for e in g if e != 0][:2]))
    res[nm] = {'rhat6': str(val), '924*rhat6': str(924 * val), 'tangential_gradient': 'exactly 0', 'stabilizer': d['H'], 'character': d['chi']}
    print('%-4s rhat6 = %-8s (= %s/924)  stabilizer %s' % (nm, val, 924 * val, d['H']))
# control: a point that is not the fixed vector of any positive-dimensional stabilizer character
ctrl = [sp.Integer(0)] * 7; ctrl[C.idx(3)] = 1; ctrl[C.idx(1)] = 1; ctrl[C.idx(-2)] = -1
ctrl = [z / sp.sqrt(3) for z in ctrl]
g, _ = tangential_gradient(ctrl)
nz = sp.sqrt(sp.nsimplify(sum(e ** 2 for e in g)))
ck.check('control u=(v3+v1-v-2)/sqrt3: tangential gradient is NOT zero (the criticality test can fail)', nz != 0, str(nz))
res['control_(v3+v1-v-2)/sqrt3_grad_norm'] = str(nz)
C.save('item3', res)
print('item3 fails:', ck.fails)
