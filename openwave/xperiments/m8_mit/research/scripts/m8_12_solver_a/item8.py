"""Item 8: global minimum and maximum of rhat6 on the unit sphere.

Part A (minimum, exact identity):  924 N = |u|^4 + 1715 P0 + 714 P2 + 77 P4,
   P_L(u) = ||P_L (u (x) u)||^2 >= 0  (Sym^2 V3 = V6 + V4 + V2 + V0, sum_L P_L = |u|^4).
   Level-1 bound for the maximum: 11/21 |u|^4 - N = 29/18 R1 + 9/22 R5 + 7/18 P4, R_L = ||P_L(u (x) Theta u)||^2.
Part B (maximum, exact sextic certificate):  463/924 |u|^6 - N |u|^2 = sum of Hermitian squares,
   found numerically (barrier method), rounded to rationals, verified exactly (identity + LDL pivots).
Part C: multistart local search (evidence only).
"""
import time, itertools
from fractions import Fraction as Fr
import numpy as np, sympy as sp, mpmath as mp
import common as C, sos

ck = C.Checker('item8')
res = {}
t0 = time.time()
# ============================================================ Part A
X = C.XS
cvec = C.cvec_from_x()
def PL_poly(L):
    tot = 0
    for Q in range(-L, L + 1):
        s = sum(C.cgL(m1, Q - m1, L, Q) * cvec[C.idx(m1)] * cvec[C.idx(Q - m1)] for m1 in C.MS if abs(Q - m1) <= 3)
        re, im = sp.expand(s).as_real_imag()
        tot += sp.expand(re ** 2 + im ** 2)
    return sp.expand(tot)
P = {L: PL_poly(L) for L in range(7)}
n2 = sp.expand(sum(v ** 2 for v in X))
N = C.N_poly(6)
ck.check('P_1 = P_3 = P_5 = 0 (u (x) u is symmetric)', all(P[L] == 0 for L in (1, 3, 5)))
ck.check('P_0 + P_2 + P_4 + P_6 = |u|^4', sp.expand(P[0] + P[2] + P[4] + P[6] - n2 ** 2) == 0)
ident_min = sp.expand(924 * N - (n2 ** 2 + 1715 * P[0] + 714 * P[2] + 77 * P[4]))
ck.check('924 N = |u|^4 + 1715 P0 + 714 P2 + 77 P4 (exact polynomial identity)', ident_min == 0)
R = {L: C.N_poly(L) for L in (1, 5)}
ident_l1 = sp.expand(sp.Rational(11, 21) * n2 ** 2 - N - (sp.Rational(29, 18) * R[1] + sp.Rational(9, 22) * R[5] + sp.Rational(7, 18) * P[4]))
ck.check('11/21 |u|^4 - N = 29/18 R1 + 9/22 R5 + 7/18 P4 (exact polynomial identity)', ident_l1 == 0)
at_v3 = {x: (1 if x == X[C.idx(3)] else 0) for x in X}
ck.check('at v3: P0 = P2 = P4 = 0, so rhat6(v3) = 1/924 attains the lower bound',
         all(P[L].subs(at_v3) == 0 for L in (0, 2, 4)) and N.subs(at_v3) == sp.Rational(1, 924))
# every R_L is a fixed combination of P0, P2, P4, P6 (both families span the 4-dim space of
# invariant quartics); the matrix T is verified as exact polynomial identities
T = {0: ['1/7'] * 4, 1: ['-3/7', '-9/28', '-1/14', '9/28'], 2: ['5/7', '19/84', '-5/14', '25/84'],
     3: ['-1', '1/6', '1/6', '1/6'], 4: ['9/7', '-9/14', '97/154', '9/154'],
     5: ['-11/7', '55/84', '17/42', '1/84'], 6: ['13/7', '65/84', '13/154', '1/924']}
T = {L: [sp.Rational(v) for v in row] for L, row in T.items()}
okT = all(sp.expand(C.N_poly(L) - sum(t_ * P[K] for t_, K in zip(T[L], (0, 2, 4, 6)))) == 0 for L in range(7))
ck.check('R_L = sum_K T_LK P_K for L = 0..6 (exact polynomial identities)', okT)
# weak duality: the point p* = (P0,P2,P4,P6) = (1/7, 1/3, 0, 11/21) satisfies every level-1 constraint
pstar = [sp.Rational(1, 7), sp.Rational(1, 3), 0, sp.Rational(11, 21)]
rstar = [sum(t_ * p_ for t_, p_ in zip(T[L], pstar)) for L in range(7)]
ck.check('dual point p* = (1/7,1/3,0,11/21): p* >= 0, sum = 1, all R_L(p*) >= 0, R_6(p*) = 11/21, so no level-1 certificate beats 11/21',
         all(p_ >= 0 for p_ in pstar) and sum(pstar) == 1 and all(r_ >= 0 for r_ in rstar) and rstar[6] == sp.Rational(11, 21), str(rstar))
res['level1_optimality_dual_point'] = {'p0,p2,p4,p6': [str(v) for v in pstar], 'r0..r6': [str(v) for v in rstar]}
res['min'] = {'value': '1/924', 'identity': '924 N = |u|^4 + 1715 P0 + 714 P2 + 77 P4', 'attained_at': 'v3 (and exactly where P0=P2=P4=0)'}
res['level1_upper_bound'] = {'value': '11/21', 'identity': '11/21 |u|^4 - N = 29/18 R1 + 9/22 R5 + 7/18 P4'}
print('Part A done (%.0fs)' % (time.time() - t0))

# ============================================================ Part B
LAM = Fr(463, 924)
cert = sos.certificate(LAM)
for name, ok in cert['chk'].items(): ck.check(name, ok)
y = cert['y']; coefpolys = cert['coefpolys']; blocks = cert['blocks']; mats = cert['mats']
pivrep = cert['pivrep']; rank = cert['rank']; nun = cert['nun']
# numerical spot check of the identity in original c-coordinates at random points (independent of a-coords)
mp.mp.dps = 30
rng = np.random.default_rng(11)
worst = 0
for _ in range(5):
    cc = [mp.mpc(float(rng.normal()), float(rng.normal())) for _ in range(7)]
    n2v = sum(abs(z) ** 2 for z in cc)
    lhs = (mp.mpf(463) / 924 - C.mp_rhat(cc)) * n2v ** 3
    av = [mp.sqrt(sp.binomial(6, 3 + m)) * cc[C.idx(m)] for m in C.MS]
    rhs = sum(mp.mpf(v.numerator) / v.denominator * sos.evalc(cp, av) for v, cp in zip(y, coefpolys))
    worst = max(worst, abs(lhs - rhs))
ck.check('certificate identity holds numerically in the original coordinates (5 random points, 30 digits)', worst < mp.mpf(10) ** -20, mp.nstr(worst, 3))
res['max'] = {'value': '463/924', 'attained_at': 'cat = (v3+v-3)/sqrt2',
              'certificate': 'lambda |u|^6 - N|u|^2 = sum_b sum_M kappa_{L,M}^-1 c_bM^T Y_b conj(c_bM), Y_b = P_b Yt_b P_b^T, Yt_b rational PD',
              'reduced_blocks': [(b['kind'], b['L'], b['mult'], len(b['P'])) for b in blocks],
              'n_unknowns': nun, 'rank_of_identity_constraints': rank, 'ldl_pivots': pivrep,
              'Yt_blocks': {('kind=%d,L=%d' % (blocks[bi]['kind'], blocks[bi]['L'])): [[str(v) for v in row] for row in M] for bi, M in mats.items()},
              'numeric_margin_before_rounding': cert['margin']}
print('Part B done (%.0fs)' % (time.time() - t0))

# ============================================================ Part C (evidence only)
from scipy.optimize import minimize
MSl = C.MS
cg6f = {(m1, Q): float(C.cg6(m1, Q - m1, Q)) for Q in range(-6, 7) for m1 in MSl if abs(Q - m1) <= 3}
def r6(xv):
    c = xv[:7] + 1j * xv[7:]
    th = np.array([(-1) ** (3 - m) * np.conj(c[3 + m]) for m in MSl])
    tot = 0.0
    for Q in range(-6, 7):
        s = 0
        for m1 in MSl:
            if abs(Q - m1) <= 3: s += cg6f[(m1, Q)] * c[3 - m1] * th[3 - (Q - m1)]
        tot += abs(s) ** 2
    return tot / np.dot(xv, xv) ** 2
rng = np.random.default_rng(2026)
mins, maxs = [], []
for k in range(200):
    x0 = rng.normal(size=14)
    mins.append(minimize(r6, x0, method='BFGS', options={'gtol': 1e-10}).fun)
    maxs.append(-minimize(lambda z: -r6(z), x0, method='BFGS', options={'gtol': 1e-10}).fun)
res['search'] = {'starts': 200, 'best_min': min(mins), 'best_max': max(maxs),
                 'min_minus_1/924': min(mins) - 1 / 924, 'max_minus_463/924': max(maxs) - 463 / 924,
                 'distinct_local_max_values_found': sorted(set(round(v * 924, 4) for v in maxs)),
                 'distinct_local_min_values_found': sorted(set(round(v * 924, 4) for v in mins))}
print('search: best min*924 = %.10f, best max*924 = %.10f' % (min(mins) * 924, max(maxs) * 924))
print('  local max values (x924) reached:', res['search']['distinct_local_max_values_found'])
print('  local min values (x924) reached:', res['search']['distinct_local_min_values_found'])
C.save('item8', res)
print('item8 fails:', ck.fails, ' (%.0fs)' % (time.time() - t0))
