"""A.1: channel coefficients q_J in R2, R4, R5.
Exact route: lib_quartic (Schur + Casimir projectors, exact in Q(sqrt5)).
Numerical route: Q(w) = avg|phi_w|^4 / (s_N^2 |w|^4) from the exact-degree SU(2) quadrature at 40 digits,
at 6 fixed generic points per slot; f_J(w) from Clebsch-Gordan coefficients of sympy (library, CS convention);
least-squares fit of q_J, residual and agreement with the exact values checked.
'plant': perturbs one exact q_J by 1/1000 before the comparison."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_quartic import *
from lib_quad import haar_avg_moments
from lib_sym import eta_exact
import sympy as sp, mpmath
from sympy.physics.quantum.cg import CG

log = Log(__file__, '_plant' if planted() else '')
mpmath.mp.dps = 40
res = {}
for slot in ('R2', 'R4', 'R5'):
    name, j2, d = slot_info(slot)
    n = j2 + 1; ml = ms(j2)
    q, proj = channel_coeffs(slot)
    log(f'{slot} (j = {j2}/2):')
    sJ = sym_J2s(j2)
    for J2 in range(0, 2 * j2 + 1, 2):
        tag = '' if J2 in sJ else '  (antisymmetric channel, not in Sym^2: does not enter Q)'
        log(f'   q_{Fr(J2, 2)} = {q[J2]}{tag}')
    allrat = all(q[J2].israt() for J2 in sJ)
    log.check(f'{slot}: every q_J (symmetric channels) is rational', allrat)
    qex = {J2: q[J2] for J2 in sJ}
    if planted():
        qex[sJ[0]] = qex[sJ[0]] + Fr(1, 1000)
    # numerical route
    eta = eta_exact(slot)
    etan = [[mpmath.mpf(str(sp.N(eta[r, c], 60))) for c in range(d)] for r in range(n)]
    pts = []
    for k in range(6):
        pts.append([mpmath.mpc(mpmath.mpf((7 * k + 3 * r + 1) % 11) / 5 - 1, mpmath.mpf((5 * k * r + 2 * r + k) % 13) / 7 - 0.9) for r in range(n)])
    m2, m4 = haar_avg_moments(j2, etan, pts)
    sN = mpmath.mpf(d) / (j2 + 1)
    cg = {}
    for J2 in sJ:
        for M2 in range(-J2, J2 + 1, 2):
            for a, ma in enumerate(ml):
                for b, mb in enumerate(ml):
                    if ma + mb == M2:
                        v = CG(sp.Rational(j2, 2), sp.Rational(ma, 2), sp.Rational(j2, 2), sp.Rational(mb, 2), sp.Rational(J2, 2), sp.Rational(M2, 2)).doit()
                        if v != 0: cg[(J2, M2, a, b)] = mpmath.mpf(str(sp.N(v, 60)))
    rows = []; rhs = []
    for w, a2, a4 in zip(pts, m2, m4):
        nw = sum(abs(x) ** 2 for x in w)
        Qn = a4 / (sN ** 2 * nw ** 2)
        fs = []
        for J2 in sJ:
            tot = 0
            for M2 in range(-J2, J2 + 1, 2):
                comp = sum(cg.get((J2, M2, a, b), 0) * w[a] * w[b] for a in range(n) for b in range(n))
                tot += abs(comp) ** 2
            fs.append(tot / nw ** 2)
        rows.append(fs); rhs.append(Qn)
        log(f'   quadrature Q at point {len(rows)}: {mpmath.nstr(Qn, 36)}; sum f_J - 1 = {mpmath.nstr(sum(fs) - 1, 3)}')
    A = mpmath.matrix(rows); bvec = mpmath.matrix(rhs)
    sol = mpmath.lu_solve(A.T * A, A.T * bvec)
    resid = max(abs(x) for x in (A * sol - bvec))
    log(f'   least-squares residual over 6 points: {mpmath.nstr(resid, 3)}')
    log.check(f'{slot}: Q from quadrature is exactly a combination of the f_J (residual < 1e-30)', resid < mpmath.mpf(10) ** -30)
    for J2, x in zip(sJ, sol):
        ex = qex[J2]
        exn = mpmath.mpf(ex.a.numerator) / ex.a.denominator + mpmath.sqrt(5) * mpmath.mpf(ex.b.numerator) / ex.b.denominator
        log(f'   fitted q_{Fr(J2, 2)} = {mpmath.nstr(x, 34)}   exact {ex}   diff {mpmath.nstr(x - exn, 3)}')
        log.check(f'{slot}: fitted q_{Fr(J2, 2)} agrees with exact to 1e-30', abs(x - exn) < mpmath.mpf(10) ** -30)
    res[slot] = {str(Fr(J2, 2)): str(q[J2].a) for J2 in sJ}
if not planted():
    save_json('A1.json', res)
log.close()
