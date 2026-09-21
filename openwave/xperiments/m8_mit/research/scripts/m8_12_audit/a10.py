"""Audit item 10 arithmetic: Q = 1 + w*rhat6 at each orbit, and item-5b entries times w (both solvers' bases).
The Hessian scaling H^Q = w H is checked exactly at one orbit (F*) by recomputing the form for N_Q = |u|^4 + w N."""
import sympy as sp
from fractions import Fraction as Fr
W = [Fr(28, 39), Fr(21, 52)]
vals = {'v3': Fr(1, 924), 'v2': Fr(3, 77), 'v1': Fr(75, 308), 'F*/Umin': Fr(75, 308), 'v0': Fr(100, 231),
        'xyz/oct': Fr(24, 77), 'cat/hex': Fr(463, 924), 'A*,D*': Fr(9, 35), 'G*/Wmin': Fr(200, 903)}
for k, v in vals.items(): print('Q at %-8s' % k, [str(1 + w * v) for w in W])
fivb = {'A* / L12circle': [Fr(-24, 55)], 'D* / L23circle': [Fr(-104, 55)],
        'F* (A basis)': [Fr(64, 33), Fr(10, 11)], 'Umin (B basis)': [Fr(8, 11), Fr(10, 11)],
        'G* (A basis)': [Fr(8, 11), Fr(920, 473)], 'Wmin (B basis)': [Fr(920, 473), Fr(184, 473)],
        'xyz in C / oct in L22': [Fr(-24, 11)], 'cat in E / hex in L33': [Fr(-4)],
        'xyz in G (A)': [Fr(40, 33), Fr(-8, 11)], 'oct in W (B)': [Fr(-40, 99), Fr(40, 33)],
        'cat in F (A)': [Fr(-10, 11), Fr(-64, 33)], 'hex in U (B)': [Fr(-10, 11), Fr(-40, 33)]}
for k, v in fivb.items(): print('5b x w %-22s' % k, [[str(w * x) for x in v] for w in W])

# exact check of H^Q = w H at F* using the a-coordinate machinery
from aud_core import idx, I
from aud_acoords import Na, HessNa, VARS, G0, realv, c_to_a
c = [0] * 7; c[idx(0)] = sp.sqrt(6); c[idx(2)] = I * sp.sqrt(5); c[idx(-2)] = I * sp.sqrt(5)
x = realv(c_to_a(c)); n2 = sp.expand((x.T * G0 * x)[0]); sub = dict(zip(VARS, list(x)))
w = sp.Rational(28, 39)
xv = sp.Matrix(VARS)
nq = sp.expand(((xv.T * G0 * xv)[0]) ** 2 + w * Na)
HQ = sp.hessian(nq, VARS).subs(sub).applyfunc(sp.expand)
NQ = sp.expand(nq.subs(sub))
MQ = (HQ / n2 - 4 * NQ / n2 ** 2 * G0).applyfunc(sp.expand)
M = (HessNa.subs(sub) / n2 - 4 * sp.expand(Na.subs(sub)) / n2 ** 2 * G0).applyfunc(sp.expand)
ux = x / sp.sqrt(n2)
D = (MQ - w * M).applyfunc(sp.expand)
# D should be 8 (G0 u)(G0 u)^T, which vanishes on T_u = {e : e^T G0 u = 0}
R = (D - 8 * (G0 * ux) * (G0 * ux).T).applyfunc(lambda t: sp.radsimp(sp.expand(t)))
print('M^Q - w M == 8 (G0 u)(G0 u)^T exactly at F* (so H^Q = w H on T_u):', all(t == 0 for t in R))
