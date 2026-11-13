import json, os, shutil, subprocess, sys, itertools
import sympy as sp, mpmath as mp

R = json.load(open('solver/results.json'))
qx, qy, qz, a = sp.symbols('q_x q_y q_z a')
loc = {'q_x': qx, 'q_y': qy, 'q_z': qz, 'a': a}
P = {d: sp.sympify(R['degree_%d' % d]['polynomial'], locals=loc) for d in (2, 4, 6)}
out = {}
mp.mp.dps = 70
ph = (1 + mp.sqrt(5)) / 2
raw = [(0, s1, s2*ph) for s1 in (1,-1) for s2 in (1,-1)] + [(s1, s2*ph, 0) for s1 in (1,-1) for s2 in (1,-1)] + [(s2*ph, 0, s1) for s1 in (1,-1) for s2 in (1,-1)]
U = []
for v in raw:
    nv = mp.sqrt(sum(c*c for c in v)); U.append([c/nv for c in v])
assert len(U) == 12

def unit(v):
    v = [mp.mpf(c) for c in v]; nv = mp.sqrt(sum(c*c for c in v)); return [c/nv for c in v]
# directions the solver did NOT use (solver used u0, z, (1,2,3), (3,-1,2), (1,1,-5))
DIRS = {'d1_(2,-3,7)': unit([2,-3,7]), 'd2_(5,1,1)': unit([5,1,1]), 'd3_(-4,6,3)': unit([-4,6,3]),
        'd4_(1,0,0)': unit([1,0,0]), 'd5_(1,1,1)': unit([1,1,1]), 'd6_(phi,1,0)': unit([ph,1,0]),
        'd7_(1,-1,0)': unit([1,-1,0]), 'd8_(7,2,-9)': unit([7,2,-9])}
fl = [sp.lambdify((qx, qy, qz, a), P[d], 'mpmath') for d in (2, 4, 6)]

def check(fns, scale=1, den_list=(4, 8, 16, 32)):
    rows = []
    for nm, n in DIRS.items():
        for den in den_list:
            a_ = mp.mpf(1)/den; k = mp.mpf(1)
            om = sum((1 - mp.cos(a_*k*sum(x*y for x, y in zip(u, n))))/(2*a_*a_) for u in U)
            ser = sum(f(k*n[0], k*n[1], k*n[2], a_*scale) for f in fns)
            b = 12*(a_*k)**8/(2*a_**2*mp.factorial(8))
            rows.append((nm, den, abs(om-ser), b, abs(om-ser) <= b))
    return rows

def summ(rows):
    return {'n': len(rows), 'all_inside': all(r[4] for r in rows), 'n_outside': sum(1 for r in rows if not r[4]),
            'max_diff_over_bound': mp.nstr(max(r[2]/r[3] for r in rows), 8),
            'min_diff_over_bound': mp.nstr(min(r[2]/r[3] for r in rows), 8),
            'outside': [[r[0], '1/%d' % r[1]] for r in rows if not r[4]][:6]}

# ---- C2
rows = check(fl)
c2 = summ(rows)
# halving ratio of differences (should approach 2^-6 since next term ~ a^6 k^8)
rat = {}
for nm in DIRS:
    d = {r[1]: r[2] for r in rows if r[0] == nm}
    rat[nm] = mp.nstr(d[8]/d[4], 6) + ',' + mp.nstr(d[16]/d[8], 6) + ',' + mp.nstr(d[32]/d[16], 6)
c2['diff_halving_ratios(1/64=0.015625)'] = rat
# solver's own rows
sr = R['numerical_check']
c2['solver_rows_all_inside'] = R['numerical_check_all_inside']
c2['solver_max_diff_over_bound'] = max(float(r['difference'])/float(r['bound']) for r in sr)
c2['bound_valid_derivation'] = ('1-cos t has 8th derivative -cos t; Lagrange remainder after t^6 term <= t^8/8!; '
                               '|u.n|<=1 and 12 terms give 12 (ak)^8/(2 a^2 8!)')
out['C2'] = c2
print('C2', 'dirs=%d samples=%d all_inside=%s max_ratio=%s' % (len(DIRS), len(rows), c2['all_inside'], c2['max_diff_over_bound']))

# ---- C3
c3 = {}; ok = True
for d, m in ((2, 1), (4, 2), (6, 3)):
    p = P[d]; ps = sp.expand(p.subs(a, a/2))
    f = sp.Rational(1, 2**(2*m-2))
    terms = sp.Poly(p, qx, qy, qz, a).terms()
    bad = 0
    for (i, j, k_, l), c in terms:
        if l != 2*m-2 or i+j+k_ != d: bad += 1
        c2_ = sp.Poly(ps, qx, qy, qz, a).coeff_monomial(qx**i*qy**j*qz**k_*a**l)
        if sp.simplify(c2_ - f*c) != 0: bad += 1
    exact = sp.expand(ps - f*p) == 0
    c3['degree_%d' % d] = {'terms': len(terms), 'bad_terms': bad, 'factor': str(f), 'exact': exact}
    ok = ok and exact and bad == 0
# invariant and I6 coefficients
c3['inv_coeff_ratio'] = [str(sp.simplify(sp.sympify(R['degree_%d' % d]['invariant_coefficient'], locals=loc).subs(a, a/2)/sp.sympify(R['degree_%d' % d]['invariant_coefficient'], locals=loc))) for d in (4, 6)]
k6 = sp.sympify(R['degree_6']['remainder_coefficient_in_I6'], locals=loc)
c3['I6_coeff_ratio'] = str(sp.simplify(k6.subs(a, a/2)/k6))
c3['ratio_unchanged'] = R['degree_6']['ratio']
c3['all_exact'] = ok
# wrong-factor mutation
wrong = sp.expand(P[6].subs(a, a/2) - sp.Rational(1, 8)*P[6]) == 0
c3['mutation_wrong_factor_1/8_for_deg6_exact'] = wrong
out['C3'] = c3
print('C3 exact=%s ; wrong factor 1/8 passes=%s' % (ok, wrong))

# ---- C4
def solver_test(vecs, w, N2):
    M = sp.zeros(3, 3)
    for wi, v in zip(w, vecs): M += wi*v*v.T/N2
    D = (M - sp.eye(3)*M.trace()/3).applyfunc(lambda e: sp.expand(sp.radsimp(sp.expand(e))))
    return D == sp.zeros(3, 3), D
s5 = sp.sqrt(5); phi = (1+s5)/2
rawS = []
for s1, s2 in itertools.product([1,-1],[1,-1]):
    rawS += [(0, s1, s2*phi), (s1, s2*phi, 0), (s2*phi, 0, s1)]
V = [sp.Matrix(v) for v in rawS]; N2 = sp.expand(2+phi)
c4 = {}
c4['original'] = solver_test(V, [1]*12, N2)[0]
c4['removed'] = solver_test(V[1:], [1]*11, N2)[0]
c4['doubled'] = solver_test(V, [2]+[1]*11, N2)[0]
# rotate one direction by exact rational rotation about z (cos 3/5, sin 4/5), norm preserved
Rz = sp.Matrix([[sp.Rational(3,5), -sp.Rational(4,5), 0], [sp.Rational(4,5), sp.Rational(3,5), 0], [0,0,1]])
Vr = [Rz*V[0]] + V[1:]
c4['own_rotate_one_direction'] = solver_test(Vr, [1]*12, N2)[0]
# replace one direction by a coordinate axis scaled to norm^2 = N2 (the test hard-codes N2)
Va = [sp.Matrix([0, 0, sp.sqrt(N2)])] + V[1:]
c4['own_replace_one_by_axis'] = solver_test(Va, [1]*12, N2)[0]
# rotate ALL by the same rotation: should still pass (sanity)
c4['sanity_rotate_all_passes'] = solver_test([Rz*v for v in V], [1]*12, N2)[0]
# can the test pass a non-invariant operator? 3 axes (cube) plus negatives: degree-2 isotropic, degree-4 not
cube = [sp.Matrix(v)*sp.sqrt(N2) for v in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]]
ok_cube, _ = solver_test(cube, [1]*6, N2)
w4 = sum((c[0]**4 + 0)/N2**2 for c in cube)  # degree-4 coefficient pieces
S4x = sp.expand(sum((v.dot(sp.Matrix([qx,qy,qz])))**4/N2**2 for v in cube))
inv4 = sp.expand(sp.Rational(6, 5)*(qx**2+qy**2+qz**2)**2)
c4['cube_axes_passes_solver_test'] = ok_cube
c4['cube_axes_degree4_is_invariant'] = sp.expand(S4x - inv4) == 0
out['C4'] = c4
print('C4', {k: v for k, v in c4.items()})

# ---- mutations
mut = {}
# one coefficient of degree-6 polynomial changed by 1e-3 relative (qz^6)
Pm = P[6]
cz = sp.Poly(Pm, qx, qy, qz, a).coeff_monomial(qz**6*a**4)
Pm2 = Pm + cz*sp.Rational(1, 1000)*qz**6*a**4
flm = [fl[0], fl[1], sp.lambdify((qx, qy, qz, a), Pm2, 'mpmath')]
rm = check(flm)
mut['coefficient_1e-3_change'] = summ(rm)
mut['coefficient_1e-3_change']['red_by_a*k'] = sorted(set('1/%d' % r[1] for r in rm if not r[4]))
rw = check(fl, scale=sp.Rational(1, 2)) if False else check(fl, scale=mp.mpf(1)/2)
mut['wrong_scaling_a_half_in_polys'] = summ(rw)
# wrong factor: degree-6 multiplied by 1/8 instead of 1/16 vs a/2 expectation, numeric
mut['c3_wrong_factor'] = {'passes': wrong}
out['mutations'] = mut
print('mut coeff 1e-3: all_inside=%s n_outside=%d red at %s' % (mut['coefficient_1e-3_change']['all_inside'], mut['coefficient_1e-3_change']['n_outside'], mut['coefficient_1e-3_change']['red_by_a*k']))
print('mut wrong scaling: all_inside=%s n_outside=%d' % (mut['wrong_scaling_a_half_in_polys']['all_inside'], mut['wrong_scaling_a_half_in_polys']['n_outside']))

# ---- script: rerun in scratch subfolder
sc = {}
os.makedirs('scratch_rerun', exist_ok=True)
shutil.copy('solver/solve.py', 'scratch_rerun/solve.py')
r = subprocess.run([sys.executable, 'solve.py'], cwd='scratch_rerun', capture_output=True, text=True)
sc['rerun_returncode'] = r.returncode
new = json.load(open('scratch_rerun/results.json'))
sc['rerun_identical_content'] = (new == R)
sc['keys_differing'] = [k for k in set(new) | set(R) if new.get(k) != R.get(k)]
src = open('solver/solve.py').read()
sc['duplicate_dps_line'] = src.count('mp.mp.dps = 80') == 2
sc['no_degree6_qx2qy2qz2_term'] = sp.Poly(P[6], qx, qy, qz, a).coeff_monomial(qx**2*qy**2*qz**2*a**4) == 0
sc['report_tightest_ratio_0.17_vs_mine_on_solver_rows'] = c2['solver_max_diff_over_bound']
out['script'] = sc
print('script rerun identical=%s rc=%d' % (sc['rerun_identical_content'], sc['rerun_returncode']))
json.dump(out, open('stage_two_audit.json', 'w'), indent=1, default=str)
