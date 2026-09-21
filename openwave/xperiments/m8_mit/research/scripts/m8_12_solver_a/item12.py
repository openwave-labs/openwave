"""Item 12: H_u along a direction e in N_u computed two ways.

Orbit F* (u = unit vector at z = i sqrt(5/3) in class F).  Two directions in N_u:
 e1 = u_perp (lies in N_u exactly, item 5b), and
 e2 = a generic unit vector of N_u: the normalized projection onto N_u of the fixed rational vector
      w = v3 + 2 v2 - v1 + 3i v0 + v-2 - 2i v-3 (projection computed exactly).
Route 1: second derivative of s -> rhat6(u cos s + e sin s) at s = 0
   (1a) exactly, by symbolic differentiation of the exact rational function of s;
   (1b) numerically, mpmath.diff (high-order finite differences) at 50 and 80 digits.
Route 2: matrix formula e^T M_u e, M_u = Hess N(u) - 4 N(u) I, at 50 and 80 digits and exactly.
"""
import sympy as sp, mpmath as mp
import common as C, hess
from orbits import ORBITS

ck = C.Checker('item12')
s = sp.Symbol('s', real=True)
u = [sp.radsimp(z) for z in ORBITS['F*']['u']]
I = sp.I

# N_u basis and exact projection
O = [C.c_to_x(g).applyfunc(sp.radsimp) for g in C.orbit_gens(u)]
xu = C.c_to_x(u).applyfunc(sp.radsimp)
A = sp.Matrix.hstack(xu, *O)
def proj_N(vx):
    G = (A.T * A).applyfunc(sp.radsimp)
    coef = G.LUsolve((A.T * vx).applyfunc(sp.radsimp)).applyfunc(sp.radsimp)
    return (vx - A * coef).applyfunc(sp.radsimp)
nrm = sp.sqrt(1 + sp.Rational(5, 3))
a = [sp.Integer(0)] * 7; a[C.idx(0)] = 1
b = [sp.Integer(0)] * 7; b[C.idx(2)] = 1 / sp.sqrt(2); b[C.idx(-2)] = 1 / sp.sqrt(2)
z = I * sp.sqrt(sp.Rational(5, 3))
e1 = [sp.radsimp((-sp.conjugate(z) * a[k] + b[k]) / nrm) for k in range(7)]
w = [sp.Integer(0)] * 7
for m, cf in {3: 1, 2: 2, 1: -1, 0: 3 * I, -2: 1, -3: -2 * I}.items(): w[C.idx(m)] = cf
px = proj_N(C.c_to_x(w))
pn = sp.sqrt(sp.nsimplify(sp.radsimp((px.T * px)[0])))
e2x = (px / pn).applyfunc(sp.radsimp)
e2 = C.x_to_c(e2x)
res = {}
for name, e in [('e1 = u_perp', e1), ('e2 = generic unit vector of N_u', e2)]:
    ex_ = C.c_to_x(e)
    ck.check('%s lies in N_u (orthogonal to u and O_u, exactly)' % name, all(sp.simplify((ex_.T * v)[0]) == 0 for v in [xu] + O))
    # route 1a: exact
    curve = [sp.cos(s) * u[k] + sp.sin(s) * e[k] for k in range(7)]
    f = C.N_exact(curve) / C.norm2(curve) ** 2
    d2 = sp.nsimplify(sp.radsimp(sp.simplify(sp.diff(f, s, 2).subs(s, 0))))
    # route 2 exact: e^T M e
    M, Nv, gN, _ = hess.M_c(u)
    mat = sp.nsimplify(sp.radsimp(sp.expand((ex_.T * M * ex_)[0])))
    ck.check('%s: exact second derivative == exact matrix formula' % name, sp.simplify(d2 - mat) == 0, '%s vs %s' % (d2, mat))
    out = {'exact_second_derivative': str(d2), 'exact_matrix_formula': str(mat),
           'minimal_polynomial': str(sp.minimal_polynomial(d2, sp.Symbol('X')))}
    print(name, 'minimal polynomial of H_u(e,e):', out['minimal_polynomial'])
    for dps in (50, 80):
        mp.mp.dps = dps
        um = C.mp_vec(u); em = C.mp_vec(e)
        g = lambda t: C.mp_rhat([mp.cos(t) * um[k] + mp.sin(t) * em[k] for k in range(7)])
        d2n = mp.diff(g, 0, 2)
        Mm, xm, _ = hess.mp_M(u, dps)
        emv = mp.matrix([mp.re(q) for q in em] + [mp.im(q) for q in em])
        matn = (emv.T * Mm * emv)[0]
        exv = mp.mpf(sp.N(d2, dps + 10))
        out['dps%d' % dps] = {'finite_difference': mp.nstr(d2n, 30), 'matrix': mp.nstr(matn, 30),
                              '|fd - matrix|': mp.nstr(abs(d2n - matn), 3), '|fd - exact|': mp.nstr(abs(d2n - exv), 3),
                              '|matrix - exact|': mp.nstr(abs(matn - exv), 3)}
        ck.check('%s dps=%d: finite-difference and matrix routes agree' % (name, dps), abs(d2n - matn) < mp.mpf(10) ** (-dps + 10))
        print(name, dps, out['dps%d' % dps])
    print(name, 'exact H_u(e,e) =', d2)
    res[name] = out
C.save('item12', res)
print('item12 fails:', ck.fails)
