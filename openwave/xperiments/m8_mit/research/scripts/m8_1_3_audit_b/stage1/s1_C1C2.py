"""Items C.1 and C.2.  Level norms ||Pi_L N(psi_y)||^2 with V = R = 1 (Haar probability measure on X, psi_y = s_N^{-1/2} phi_y),
at a unit y with no symmetry.  Route 1 (exact): S^3 harmonic decomposition over Q(sqrt5, i).  Route 2 (numerical, 40 digits):
CG coupling of V_j^{(x)3}.  The two routes share only the exact projector P and s_N."""
from fractions import Fraction as Fr
from mpmath import mp
from core import *
from reps import *
from s3poly import R0_levels, kfrac, wnorm2, norm_inner_vec, Phi_polys, N_vec
from cgroute import CGRoute, CP
from chk import check, save_json, planted

mp.dps = 40
# the chosen y: weighted coordinates p_m (m = j, j-1, ...), Gaussian rationals, then normalised; no two |p_m| equal
YCHOICE = {
    7: [(1, 2), (-3, 1), (2, -1), (1, 3), (-2, -2), (3, 1), (-1, 2), (2, 5)],
    6: [(1, 2), (-3, 1), (2, -1), (1, 3), (-2, -2), (3, 1), (-1, 2)],
}
out = {'C.1': {}, 'C.2': {}}
for nm in ['R2', 'R4', 'R5']:
    d, n = SLOTS[nm]
    ms = mlist(n)
    p = [kfrac(Fr(a, 7), Fr(b, 5)) for (a, b) in YCHOICE[n]]
    nrm2 = wnorm2(nm, p)
    lv = R0_levels(nm, p)
    lv_unit = {L: v[0] / nrm2 ** 3 for L, v in lv.items()}
    check(f'C.1 {nm}: exact level norms have zero sqrt5 and imaginary parts', all(v[1:] == (0, 0, 0) for v in lv.values()))
    # total norm (all levels) vs sum of level norms
    Ph = Phi_polys(nm, p)
    Nv = N_vec(nm, Ph, Ph, Ph)
    tot = norm_inner_vec(nm, Nv, Nv, levels=False)[None][0] / s_N(nm) ** 3 / nrm2 ** 3
    check(f'C.1 {nm}: sum over levels equals the total norm of N (exact)', sum(lv_unit.values()) == tot, f'{float(tot)}')
    # route 2
    cg = CGRoute(nm, mp)
    sq = mp.sqrt(mp.mpf(nrm2.numerator) / nrm2.denominator)
    y = [CP(mp.mpf(p[i][0].numerator) / p[i][0].denominator * mp.sqrt(Fm(n, ms[i])) / sq,
            mp.mpf(p[i][2].numerator) / p[i][2].denominator * mp.sqrt(Fm(n, ms[i])) / sq) for i in range(n + 1)]
    lv2 = cg.levels(y)
    worst = max(abs(lv2.get(L, 0) - mp.mpf(v.numerator) / v.denominator) for L, v in lv_unit.items())
    if planted('C1'):
        worst = worst + abs(lv2[n]) * 1e-6
    check(f'C.1 {nm}: CG route agrees with exact route at every level (40 digits)', worst < mp.mpf(10) ** -32, mp.nstr(worst, 3))
    occ = [l for l, m in irrep_levels(slot_irrep(nm), 40)]
    above = [L for L in sorted(lv_unit) if n < L <= 20]
    nonzero = [L for L in above if lv_unit[L] != 0]
    check(f'C.1 {nm}: nonzero levels are exactly the occurrence levels of the slot (<= 3n)',
          nonzero == [L for L in occ if n < L <= min(20, 3 * n)], f'{nonzero}')
    print(f'{nm}: s_N = {s_N(nm)}; y (weighted p_m, m=j..-j, before normalisation, ||y||^2 = {nrm2}):',
          [f'{Fr(a, 7)}+{Fr(b, 5)}i' for a, b in YCHOICE[n]])
    print(f'   level norms (unit y, V=1):')
    for L in sorted(lv_unit):
        if L >= n:
            print(f'     L={L:2d}: {str(lv_unit[L]):>60s}  = {mp.nstr(mp.mpf(lv_unit[L].numerator) / lv_unit[L].denominator, 25)}')
    print(f'   nonzero above the lowest level, up to 20: {nonzero}')
    if n == 7:
        print(f'   (level 21 = 3n, beyond the range asked: {lv_unit.get(21)})')
    out['C.1'][nm] = {'s_N': str(s_N(nm)),
                      'y (weighted coordinates p_m, m = j..-j, unnormalised; unit y = y/||y||)': [f'{Fr(a, 7)}+{Fr(b, 5)}*I' for a, b in YCHOICE[n]],
                      '||y||^2': str(nrm2),
                      'level norms (unit y, V=1)': {str(L): str(lv_unit[L]) for L in sorted(lv_unit) if n < L <= 20},
                      'level norms decimal': {str(L): mp.nstr(mp.mpf(lv_unit[L].numerator) / lv_unit[L].denominator, 25) for L in sorted(lv_unit) if n < L <= 20},
                      'nonzero levels': nonzero,
                      'lowest-level norm (L=n), for reference': str(lv_unit[n])}
    nxt = [L for L in occ if L > n][0]
    out['C.2'][nm] = {'next level': nxt, 'lambda1 - lambda0': str(nxt * (nxt + 2) - n * (n + 2)) + ' / R^2'}
    print(f'C.2 {nm}: next level {nxt}, lambda1-lambda0 = {nxt * (nxt + 2) - n * (n + 2)}/R^2')
save_json('out_C1C2.json', out)
