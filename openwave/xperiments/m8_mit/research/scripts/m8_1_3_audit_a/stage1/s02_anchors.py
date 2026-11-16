"""0(c), 0(d), 0(e).
0(c): multiplicity from s01; eta exact (Gram-Schmidt of the exact isotypic projector's columns); s_N exact by
      Schur orthogonality, s_N = dim rho/(2j+1); checked numerically by an exact-degree quadrature on SU(2)
      (Hopf coordinates: trapezoid in both phases, Gauss-Legendre in t = cos^2 eta) at 40 digits.
0(d): CG sign derived by lowering the highest weight vector |6 6> = |3 3>|3 3> with J_- (CS matrices);
      cross-checked with sympy's CG (a library implementation, not derived here). Theta^2 from the matrix E.
0(e): C solved from the linear equations conj(rho(h)) C = C rho(h) on both generators.
'plant': corrupts eta (swaps two rows) and Theta's sign rule, to show checks fire."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_sym import *
from lib_quad import haar_avg_moments
import mpmath

log = Log(__file__, '_plant' if planted() else '')
gj = load_json('group.json')
out = {}
log('0(c)')
for slot in ('R2', 'R4', 'R5'):
    name, j2, d = slot_info(slot)
    n = j2 + 1
    mlt = gj['mult'][name][j2]
    eta = eta_exact(slot)
    if planted():
        eta = eta.copy(); eta.row_swap(0, 1)
    log(f'{slot}: irreducible {name}, j = {j2}/2, multiplicity at lowest level = {mlt}')
    log('  eta (rows m = j..-j, columns a = 1..dim):')
    for r in range(n):
        log('   ', [str(x) for x in eta.row(r)])
    I = sp.simplify(eta.H * eta)
    log.check(f'{slot}: eta^dagger eta = I exactly', I == sp.eye(d))
    ok = True
    rhos = []
    for q in (GEN1, GEN2):
        D = D_cs_exact(q, j2)
        rho = (eta.H * D * eta).applyfunc(sp.radsimp)
        rhos.append(rho)
        ok &= sp.simplify(D * eta - eta * rho) == sp.zeros(n, d)
    log.check(f'{slot}: D(q) eta = eta rho(q) exactly for both generators', ok)
    chi = class_char(name)
    trok = all(sp.simplify(r.trace() - chi[q[0]].tosympy()) == 0 for r, q in zip(rhos, (GEN1, GEN2)))
    log.check(f'{slot}: tr rho(q_i) equals the derived character', trok)
    sN = sp.Rational(d, j2 + 1)
    log(f'  s_N = dim rho/(2j+1) = {sN}  (Schur orthogonality, exact)')
    # numerical check: average over SU(2) of |phi_w|^2 for a fixed unnormalised w
    mpmath.mp.dps = 40
    etan = [[mpmath.mpf(sp.N(eta[r, c], 60)) if eta[r, c].is_real else mpmath.mpc(sp.N(eta[r, c], 60)) for c in range(d)] for r in range(n)]
    w = [mpmath.mpc(k + 1, (-1) ** k * (k + 2) / 3) for k in range(n)]
    m2, m4 = haar_avg_moments(j2, etan, [w])
    nw = sum(abs(x) ** 2 for x in w)
    val = m2[0] / nw
    log(f'  quadrature: avg_SU(2)|phi_w|^2/|w|^2 = {mpmath.nstr(val, 35)}; difference from s_N = {mpmath.nstr(val - mpmath.mpf(d) / (j2 + 1), 5)}')
    log.check(f'{slot}: quadrature s_N agrees with Schur value to 1e-35', abs(val - mpmath.mpf(d) / (j2 + 1)) < mpmath.mpf(10) ** -35)
    out[slot] = dict(multiplicity=mlt, eta=[[str(x) for x in eta.row(r)] for r in range(n)], s_N=str(sN))

log('\n0(d)')
# lowering route
Jp, Jm, Jz = spin_ops_cs(6)
n = 7
Jm_tot = sp.kronecker_product(Jm, sp.eye(n)) + sp.kronecker_product(sp.eye(n), Jm)
v = sp.zeros(n * n, 1); v[0] = 1  # |3 3>|3 3>, index 0 = m=3
for _ in range(6):
    v = Jm_tot * v
v = v / sp.sqrt((v.T * v)[0])
v = v.applyfunc(sp.radsimp)
idx = 0 * n + 6  # m1 = 3 (row 0), m2 = -3 (row 6)
cg_low = v[idx]
from sympy.physics.quantum.cg import CG
cg_lib = CG(3, 3, 3, -3, 6, 0).doit()
log(f'<3 3; 3 -3 | 6 0> by lowering = {cg_low};  sympy CG = {cg_lib}')
log.check('lowering and sympy CG agree', sp.simplify(cg_low - cg_lib) == 0)
log(f'sign = {"+" if cg_low > 0 else "-"}')
out['CG sign'] = '+' if cg_low > 0 else '-'
for j2 in (6, 7):
    E = Theta_matrix(j2)
    if planted():
        E = E.applyfunc(abs)
    T2 = E * E.conjugate()  # Theta^2 u = E conj(E conj(u)) = E conj(E) u
    val = T2[0, 0]
    log(f'Theta^2 on V_{j2}/2 = {val} * I ; is scalar: {T2 == val * sp.eye(j2 + 1)}')
    log.check(f'Theta^2 on V_{j2}/2 equals (-1)^(2j)', T2 == (-1) ** j2 * sp.eye(j2 + 1))
    out[f'Theta^2 V_{j2}/2'] = str(val)
    # Theta commutes with D(q)
    okc = all(sp.simplify(D_cs_exact(q, j2) * E - E * D_cs_exact(q, j2).conjugate()) == sp.zeros(j2 + 1, j2 + 1) for q in (GEN1, GEN2))
    log.check(f'Theta commutes with D^j(q_i), V_{j2}/2 (D Theta = Theta D)', okc)

log('\n0(e)  R2')
eta = eta_exact('R2'); j2 = 7
if planted():
    eta = eta.copy(); eta.row_swap(0, 1)
rhos = [(eta.H * D_cs_exact(q, j2) * eta).applyfunc(sp.radsimp) for q in (GEN1, GEN2)]
c = sp.symbols('c0:4')
C = sp.Matrix(2, 2, c)
eqs = []
for r in rhos:
    eqs += list(r.conjugate() * C - C * r)
sol = sp.solve(eqs, c, dict=True)
log('general solution of conj(rho) C = C rho:', sol)
if not sol or all(v == 0 for v in sol[0].values()) and len(sol[0]) == 4:
    log.check('a nonzero C exists', False)
    log.close(); sys.exit(0)
Cg = C.subs(sol[0])
free = [x for x in c if Cg.has(x)]
Cs = Cg.subs({free[0]: 1})
nrm = sp.sqrt(sp.simplify((Cs.H * Cs)[0, 0]))
Cs = (Cs / nrm).applyfunc(sp.radsimp)
log('unitary C =', Cs.tolist())
log.check('C unitary', sp.simplify(Cs.H * Cs) == sp.eye(2))
log.check('conj(rho(q_i)) C = C rho(q_i)', all(sp.simplify(r.conjugate() * Cs - Cs * r) == sp.zeros(2, 2) for r in rhos))
CCb = sp.simplify(Cs * Cs.conjugate())
log('C conj(C) =', CCb.tolist())
out['R2 C'] = [[str(x) for x in row] for row in Cs.tolist()]
out['R2 C Cbar'] = str(CCb[0, 0]) + ' I'
# psi -> conj(psi) C on w: psi_w(g) = w^T D(g) eta ; conj(D) = E^{-1} D E (Theta commutes) ;
# conj(psi_w) C = conj(w)^T E^{-1} D E conj(eta) C ; E conj(eta) C = eta lam.
E = Theta_matrix(7)
Mx = (E * eta.conjugate() * Cs).applyfunc(sp.radsimp)
lam = (eta.H * Mx).applyfunc(sp.radsimp)
log('E conj(eta) C = eta * lam with lam =', lam.tolist())
log.check('E conj(eta) C lies in the span of eta (an intertwiner, multiplicity one)', sp.simplify(Mx - eta * lam) == sp.zeros(8, 2))
# w' with w'^T = conj(w)^T E^{-1} eta lam eta^dagger ... i.e. w' = lam_s * (E^{-1})^T conj(w), lam scalar
lam_s = lam[0, 0]
log.check('lam is a scalar multiple of I', sp.simplify(lam - lam_s * sp.eye(2)) == sp.zeros(2, 2))
A = (lam_s * (E.inv()).T).applyfunc(sp.simplify)
log('psi_w -> conj(psi_w) C equals psi_{w\'} with w\' = A conj(w), A =', lam_s, '* (E^{-1})^T ; compare Theta: w\' = E conj(w)')
ratio = sp.simplify(A * E.inv())
log('A E^{-1} =', ratio[0, 0], '* I ; scalar:', sp.simplify(ratio - ratio[0, 0] * sp.eye(8)) == sp.zeros(8, 8))
out['R2 map on w'] = f'w -> ({sp.simplify(ratio[0, 0])}) * Theta(w)'
if not planted():
    save_json('anchors.json', out)
log.close()
