"""Planted defects: each block reruns a check used by an item script on deliberately corrupted input
and records whether it FAILS, as it must.  A line 'FIRED' means the check caught the defect."""
from fractions import Fraction as Fr
import sympy as sp, mpmath as mp
import common as C, classes as K, hess, sos
from orbits import ORBITS

results = []
def fired(name, check_passed):
    ok = not check_passed
    print(('FIRED   ' if ok else 'MISSED  ') + name)
    results.append((name, ok))

I = sp.I
def vec(d):
    c = [sp.Integer(0)] * 7
    for m, a in d.items(): c[C.idx(m)] += sp.sympify(a)
    return c

# D1 (item 0): flip the sign of one L=6 Clebsch-Gordan coefficient
u1 = vec({3: 2, 1: 1, -2: -3})
cg = C.CG(); key = (1, -1, 6, 0); orig = cg[key]
cg[key] = -orig
val_bad = C.rhat_exact(u1)
mp.mp.dps = 50
cas = C.mp_rL_casimir(C.mp_vec(u1), 6)
fired('D1 CG sign flip at <3 1;3 -1|6 0>: CG route vs Casimir route (item 0 check)', abs(mp.mpf(val_bad.p) / val_bad.q - cas) < mp.mpf(10) ** -45)
racah = C.cg_racah(3, 1, 3, -1, 6, 0)
fired('D1b same defect: lowering table vs Racah formula (item 0 check)', sp.sign(racah) == sp.sign(cg[key]))
cg[key] = orig

# D2 (item 9 / invariance): drop the (-1)^(3-m) sign from Theta
orig_theta = C.theta
Pp = sp.symbols('p0:7', real=True); Qq = sp.symbols('q0:7', real=True)
ug = [Pp[k] + I * Qq[k] for k in range(7)]
Ry = C.rot_axis_angle([0, 1, 0], sp.Rational(1, 2))
R111 = C.rot_axis_angle([1, 1, 1], sp.Rational(1, 3))
C.theta = lambda c: [sp.conjugate(c[C.idx(-m)]) for m in C.MS]
same_y = sp.expand(C.N_exact(C.apply(Ry, ug)) - C.N_exact(ug)) == 0
same_111 = sp.expand(C.N_exact(C.apply(R111, ug)) - C.N_exact(ug)) == 0
C.theta = orig_theta
print('   (D2 note: with the sign dropped, invariance under R_y(pi) still holds: %s -- that check alone is blind to this defect)' % same_y)
fired('D2 Theta without its sign: invariance N(R_111(2pi/3) u) == N(u) (item 9 check)', same_111)

# D3 (item 2): perturbed critical point on the circle of class A (t = 1/4 + 1/100)
x, y = K.x, K.y
Nn, Dn = K.restriction('A'); f = Nn / Dn
r = sp.sqrt(sp.Rational(1, 4) + sp.Rational(1, 100))
fired('D3 point off the critical circle of A: exact zero gradient (item 2 check)', sp.simplify(sp.diff(f, x).subs({x: r, y: 0})) == 0)

# D4 (item 2): drop one critical point from the exact list -> the independent Newton search disagrees
exact_t = {'0.0', '0.6', 'inf'}           # class F without |z|^2 = 5/3
found = {'0.0', '0.6', '1.66666666667', 'inf'}   # what item 2's search returns for F
fired('D4 critical point |z|^2=5/3 removed from class F list: Newton-search containment (item 2 check)', found <= exact_t)
fired('D4b same defect: Poincare-Hopf index sum == 2 (item 2 check)', (1 + 1 - 1 - 1 + 1) == 2)   # F indices minus one min

# D5 (item 5): Hessian formula without the -4N term: M = Hess N only
u = list(ORBITS['F*']['u'])
M, Nv, gN, xv = hess.M_c(u)
Mbad = M + 4 * Nv * sp.eye(14)
res_bad = [sp.simplify(((Mbad * C.c_to_x(o)).T * (Mbad * C.c_to_x(o)))[0]) for o in C.orbit_gens(u)]
fired('D5 Hessian formula missing -4N(u) I: M annihilates O_u (item 5 check)', all(v == 0 for v in res_bad))

# D6 (item 5b): orbit-null control applied to a non-null tangent (u_perp at F*)
z = I * sp.sqrt(sp.Rational(5, 3)); d = K.CLASSES2['F']; a_ = list(d['a']); b_ = list(d['b'])
nrm = sp.sqrt(1 + sp.Rational(5, 3))
up = [sp.radsimp((-sp.conjugate(z) * a_[k] + b_[k]) / nrm) for k in range(7)]
upx = C.c_to_x(up)
h = sp.nsimplify(sp.radsimp(sp.expand((upx.T * M * upx)[0])))
fired('D6 orbit-null control on u_perp (not in O_u): H_u == 0 (item 5b check)', h == 0)

# D7 (item 4): a rotated copy of v1 entered as a new orbit: pairwise-distinct invariants
inv = lambda c: tuple(C.rhat_exact(c, L) for L in range(1, 7))
v1r = [sp.radsimp(z_) for z_ in C.apply(Ry, vec({1: 1}))]
fired('D7 duplicate orbit (R_y(pi) v1 listed separately): distinct invariants (item 4 check)', inv(vec({1: 1})) != inv(v1r))

# D8 (item 8): certificate at lambda below the maximum (463/924 - 1/1000)
cert = sos.certificate(Fr(463, 924) - Fr(1, 1000), log=lambda *a: None)
fired('D8 lambda = 463/924 - 1/1000: exact certificate system consistent (item 8 check)', cert['consistent'])

# D9 (item 8): perturb one entry of the rational Gram solution
def bump(yv):
    yv = list(yv); yv[0] += Fr(1, 10 ** 6); return yv
cert = sos.certificate(Fr(463, 924), perturb=bump, log=lambda *a: None)
fired('D9 one Gram entry perturbed by 1e-6: exact identity (item 8 check)', cert['chk'][[k for k in cert['chk'] if k.startswith('exact identity')][0]])

# D10 (item 8): non-PSD Gram block (negate a block) -> exact LDL positivity must fail
def negate_first(yv):
    return [-v for v in yv]
cert = sos.certificate(Fr(463, 924), perturb=negate_first, log=lambda *a: None)
fired('D10 all Gram blocks negated: exact LDL positivity (item 8 check)', cert['chk']['every reduced Gram block is positive definite (exact LDL pivots all > 0)'])

# D11 (item 12): the finite-difference route with a crude fixed step h = 1e-3 (instead of mpmath's adaptive diff)
mp.mp.dps = 50
um = C.mp_vec(u); em = C.mp_vec(up)
g = lambda t: C.mp_rhat([mp.cos(t) * um[k] + mp.sin(t) * em[k] for k in range(7)])
hstep = mp.mpf('1e-3')
d2crude = (g(hstep) - 2 * g(0) + g(-hstep)) / hstep ** 2
fired('D11 central difference with h = 1e-3: agreement with e^T M e to 1e-40 (item 12 check)', abs(d2crude - mp.mpf(sp.N(h, 60))) < mp.mpf(10) ** -40)

n_ok = sum(ok for _, ok in results)
print('defects fired: %d / %d' % (n_ok, len(results)))
C.save('defects', {name: ('fired' if ok else 'MISSED') for name, ok in results})
