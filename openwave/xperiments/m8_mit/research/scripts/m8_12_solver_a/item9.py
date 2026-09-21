"""Item 9: exact reasons for the zeros met in items 2-7, with the symmetries shown to preserve rhat6."""
import sympy as sp
import common as C, classes as K
from orbits import ORBITS

ck = C.Checker('item9')
I = sp.I
res = {}
# generic symbolic vector
P = sp.symbols('p0:7', real=True); Qs = sp.symbols('q0:7', real=True)
u = [P[k] + I * Qs[k] for k in range(7)]
N_u = sp.expand(C.N_exact(u))

# (S1) Theta preserves rhat6: rho6(Theta u) = rho6(u) because <3 m1; 3 m2|6 Q> is symmetric in (m1, m2) and Theta^2 = 1
sym_cg = all(C.cg6(a, b, a + b) == C.cg6(b, a, a + b) for a in C.MS for b in C.MS if abs(a + b) <= 6)
ck.check('<3 m1; 3 m2|6 Q> = <3 m2; 3 m1|6 Q> for all m1, m2', sym_cg)
rt = C.rhoL(C.theta(u)); r0 = C.rhoL(u)
ck.check('rho6(Theta u) == rho6(u) as polynomials (so rhat6(Theta u) = rhat6(u), |Theta u| = |u|)',
         all(sp.expand(a - b) == 0 for a, b in zip(rt, r0)))
# (S2) rotations preserve rhat6: checked as exact polynomial identities for the rotations used as reasons
th = sp.Symbol('theta', real=True)
Rz = sp.diag(*[sp.exp(-I * m * th) for m in C.MS])
uz = C.apply(Rz, u)
ck.check('N(R_z(theta) u) == N(u) for symbolic theta', sp.simplify(sp.expand(C.N_exact(uz)) - N_u) == 0)
Ry = C.rot_axis_angle([0, 1, 0], sp.Rational(1, 2))
ck.check('N(R_y(pi) u) == N(u)', sp.expand(C.N_exact(C.apply(Ry, u)) - N_u) == 0)
R111 = C.rot_axis_angle([1, 1, 1], sp.Rational(1, 3))
ck.check('N(R_111(2pi/3) u) == N(u) (a rotation not about a coordinate axis)', sp.expand(C.N_exact(C.apply(R111, u)) - N_u) == 0)
# phase
ph = sp.Symbol('phi', real=True)
ck.check('N(e^{i phi} u) == N(u)', sp.simplify(sp.expand(C.N_exact([sp.exp(I * ph) * z for z in u])) - N_u) == 0)

# (Z1) item 2, class B, pole v1: Hessian of the restriction is exactly 0
x, y, t = K.x, K.y, K.t
Nn, _ = K.restriction('B')
g = sp.expand(924 * Nn).subs({x: sp.sqrt(t), y: 0})
ck.check('924 N(v1 + z v-3) = 225 (1+t)^2 - 224 t^2, t = |z|^2 (exact identity)', sp.expand(g - (225 * (1 + t) ** 2 - 224 * t ** 2)) == 0)
res['Z1'] = ('class B pole v1: rhat6 = 225/924 - (224/924) t^2/(1+t)^2; no linear term in t, hence zero chart Hessian. '
             'The cancellation is the identity 2<3 1;3 -1|6 0><3 -3;3 3|6 0> + <3 1;3 3|6 4>^2 + <3 -3;3 -1|6 -4>^2 = 2<3 1;3 -1|6 0>^2, i.e. 30 + 420 = 450 (in units 1/924).')
cgv = lambda a, b: C.cg6(a, b, a + b)
lhs = 2 * cgv(1, -1) * cgv(-3, 3) + cgv(1, 3) ** 2 + cgv(-3, -1) ** 2
ck.check('the CG identity behind Z1: 2<1,-1><-3,3> + <1,3>^2 + <-3,-1>^2 == 2<1,-1>^2', sp.nsimplify(lhs - 2 * cgv(1, -1) ** 2) == 0, str(lhs))

# (Z2) item 4: odd r_L vanish when Theta u is proportional to u (u (x) Theta u symmetric)
for nm in ('v0', 'xyz', 'cat'):
    c = list(ORBITS[nm]['u']); tc = C.theta(c)
    lam_ = next(tc[k] / c[k] for k in range(7) if c[k] != 0)
    ok = all(sp.simplify(tc[k] - lam_ * c[k]) == 0 for k in range(7))
    ck.check('%s: Theta u = (%s) u, so u (x) Theta u is symmetric and r_1 = r_3 = r_5 = 0' % (nm, lam_), ok and all(C.rhat_exact(c, L) == 0 for L in (1, 3, 5)))
ck.check('CG antisymmetry for odd L: <3 m1;3 m2|L Q> = -<3 m2;3 m1|L Q> for L = 1,3,5',
         all(C.cgL(a, b, L, a + b) == -C.cgL(b, a, L, a + b) for L in (1, 3, 5) for a in C.MS for b in C.MS if abs(a + b) <= L))
# r_1 = 0 at F*, G*: <J> is fixed by the stabilizer D2 / D3, and a vector fixed by D_n (n>=2) is 0; r_1 is proportional to |<J>|^2
for nm in ('F*', 'G*'):
    c = list(ORBITS[nm]['u'])
    Jexp = [sp.simplify(sum(sp.conjugate(c[a]) * (A * sp.Matrix(c))[a] for a in range(7))) for A in (C.JX, C.JY, C.JZ)]
    ck.check('%s: <J> = 0 exactly and r_1 = 0' % nm, all(v == 0 for v in Jexp) and C.rhat_exact(c, 1) == 0)
# r_1 is proportional to |<J>|^2 (checked on the six line classes and item-0 points)
ratios = set()
for nm in ('v3', 'v2', 'v1'):
    c = list(ORBITS[nm]['u'])
    J2 = sum(sp.Abs(sum(sp.conjugate(c[a]) * (A * sp.Matrix(c))[a] for a in range(7))) ** 2 for A in (C.JX, C.JY, C.JZ))
    ratios.add(sp.nsimplify(C.rhat_exact(c, 1) / J2))
ck.check('r_1 / |<J>|^2 is the same constant (%s) at v3, v2, v1' % ratios, len(ratios) == 1)
# r_2(v2) = 0: SO(2) leaves only the Q = 0 quadrupole, proportional to <3 Jz^2 - J^2> = 3*4 - 12 = 0
c2 = list(ORBITS['v2']['u'])
q = sp.simplify(sum(sp.conjugate(c2[a_]) * ((3 * C.JZ ** 2 - C.JX ** 2 - C.JY ** 2 - C.JZ ** 2) * sp.Matrix(c2))[a_] for a_ in range(7)))
ck.check('v2: <3 Jz^2 - J^2> = %s and r_2 = 0' % q, q == 0 and C.rhat_exact(c2, 2) == 0)
# r_2(xyz) = 0: the cubic group O has no nonzero invariant in the L=2 representation
c = list(ORBITS['xyz']['u'])
ck.check('xyz: r_1 = r_2 = r_3 = r_5 = 0', all(C.rhat_exact(c, L) == 0 for L in (1, 2, 3, 5)))

# (Z3) item 5b off-diagonal zeros: an antiunitary symmetry sigma fixing u, with sigma(u_perp) = -u_perp, sigma(i u_perp) = i u_perp
def check_sigma(label, cls, z, rot):
    d = K.CLASSES2[cls]; a = list(d['a']); b = list(d['b'])
    nrm = sp.sqrt(1 + sp.nsimplify(z * sp.conjugate(z)))
    uu = [sp.radsimp((a[k] + z * b[k]) / nrm) for k in range(7)]
    up = [sp.radsimp((-sp.conjugate(z) * a[k] + b[k]) / nrm) for k in range(7)]
    def sigma(v):
        w = C.theta(v)
        if rot is not None: w = C.apply(rot, w)
        return [sp.radsimp(z_) for z_ in w]
    su = sigma(uu)
    s0 = next(su[k] / uu[k] for k in range(7) if uu[k] != 0)
    ok_u = all(sp.simplify(su[k] - s0 * uu[k]) == 0 for k in range(7))
    # normalize: tau = s0^{-1/2}-free version: use sigma' = conj-phase adjusted; here s0 = +-1
    ok_s0 = s0 in (1, -1)
    sp_ = sigma(up); sip = sigma([I * v for v in up])
    # after dividing by s0: u_perp -> eps u_perp and i u_perp -> -eps i u_perp for one sign eps
    ok_perp = False; eps_found = None
    for eps in (1, -1):
        if all(sp.simplify(sp_[k] - s0 * eps * up[k]) == 0 for k in range(7)) and all(sp.simplify(sip[k] + s0 * eps * I * up[k]) == 0 for k in range(7)):
            ok_perp = True; eps_found = eps
    ck.check('%s: sigma = %s o Theta fixes u (factor %s); sigma/s0 sends u_perp -> %s u_perp and i u_perp -> %s i u_perp' % (
        label, 'R' if rot is not None else 'id', s0, eps_found, None if eps_found is None else -eps_found), ok_u and ok_s0 and ok_perp)
Rz4 = C.rot_axis_angle([0, 0, 1], sp.Rational(1, 4)); Rz6 = C.rot_axis_angle([0, 0, 1], sp.Rational(1, 6))
check_sigma('F*', 'F', I * sp.sqrt(sp.Rational(5, 3)), Rz4)
check_sigma('G*', 'G', I * sp.sqrt(sp.Rational(20, 23)), Rz6)
check_sigma('cat@F', 'F', sp.sqrt(sp.Rational(3, 5)), None)
check_sigma('xyz@G', 'G', 2 / sp.sqrt(5), None)
res['Z3'] = ('For F*, G*, cat@F, xyz@G the map sigma = (rotation) o Theta, times the sign s0, is a real-linear isometry that preserves rhat6 '
             '(S1, S2), fixes u, and acts as eps on u_perp and -eps on i u_perp (eps = +-1); hence H_u(u_perp, i u_perp) = -H_u(u_perp, i u_perp) = 0.')
C.save('item9', res)
print('item9 fails:', ck.fails)
