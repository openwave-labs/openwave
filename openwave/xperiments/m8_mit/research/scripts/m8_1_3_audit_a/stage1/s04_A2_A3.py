"""A.2, A.3.
Exact route: HS6 written as sum_J b_J f_J, the b_J solved exactly from exact evaluations of HS6 and f_J at
Gaussian-rational weighted points (adjoint-Casimir projector on End(V_j), Casimir projectors on V_j(x)V_j);
consistency of the overdetermined system checked exactly. Then Q - 1 = sum (q_J - 1) f_J is a constant multiple
of HS6 iff (q_J - 1) = beta b_J for every symmetric J (the f_J are linearly independent functions, checked by
the rank of the evaluation matrix).
Numerical route: HS6 from its literal definition with T^K_M built from sympy's CG at 50 digits, and Q - 1 from
the channel form, at random points; ratio printed.
'plant': perturbs one b_J."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_quartic import *
from lib_hs import *
import sympy as sp, mpmath, random
from sympy.physics.quantum.cg import CG

log = Log(__file__, '_plant' if planted() else '')
out = {}
for slot in ('R2', 'R4', 'R5'):
    name, j2, d = slot_info(slot)
    n = j2 + 1
    q, proj = channel_coeffs(slot)
    sJ = sym_J2s(j2)
    adp = ad_projectors_w(j2)
    rnd = random.Random(12345 + j2)
    rows = []; rhs = []
    for k in range(len(sJ) + 3):
        p = [G(rnd.randint(-5, 5), rnd.randint(-5, 5)) for _ in range(n)]
        f = fJ_all(p, j2, proj)
        rows.append([f[J2] for J2 in sJ]); rhs.append(HS_K(p, j2, adp, 12))
    A = sp.Matrix(rows); bb = sp.Matrix(rhs)
    rk = A.rank()
    log.check(f'{slot}: evaluation matrix of the f_J has full column rank {len(sJ)} (f_J independent)', rk == len(sJ))
    sol = (A[:len(sJ), :]).LUsolve(bb[:len(sJ), :])
    if planted():
        sol[0] = sol[0] + sp.Rational(1, 997)
    log.check(f'{slot}: HS6 = sum b_J f_J exactly at all {A.rows} points', (A * sol - bb) == sp.zeros(A.rows, 1))
    b = {J2: sol[i] for i, J2 in enumerate(sJ)}
    log(f'{slot}: HS6 channel coefficients b_J:', {str(Fr(J2, 2)): str(b[J2]) for J2 in sJ})
    qm1 = {J2: sp.Rational(q[J2].a) - 1 for J2 in sJ}
    log(f'{slot}: q_J - 1:', {str(Fr(J2, 2)): str(qm1[J2]) for J2 in sJ})
    ratios = {J2: (qm1[J2] / b[J2] if b[J2] != 0 else (None if qm1[J2] != 0 else 'any')) for J2 in sJ}
    log(f'{slot}: (q_J - 1)/b_J:', {str(Fr(J2, 2)): str(v) for J2, v in ratios.items()})
    vals = set(v for v in ratios.values() if v != 'any')
    prop = len(vals) == 1 and None not in vals
    log(f'{slot}: Q - 1 proportional to HS6: {prop}')
    beta = vals.pop() if prop else None
    log(f'{slot}: beta = {beta}')
    out[slot] = dict(proportional=prop, beta=str(beta) if prop else None,
                     b_J={str(Fr(J2, 2)): str(b[J2]) for J2 in sJ})
    # numerical route at 50 digits from the literal T^K definition
    mpmath.mp.dps = 50
    ml = ms(j2); K = 6
    T = []
    for M2 in range(-2 * K, 2 * K + 1, 2):
        Tm = [[mpmath.mpf(0)] * n for _ in range(n)]
        for a, ma in enumerate(ml):
            for c, mc in enumerate(ml):
                if ma - mc == M2:
                    v = CG(sp.Rational(j2, 2), sp.Rational(ma, 2), sp.Rational(j2, 2), -sp.Rational(mc, 2), K, sp.Rational(M2, 2)).doit()
                    Tm[a][c] = (-1) ** ((j2 - mc) // 2) * mpmath.mpf(str(sp.N(v, 70)))
        T.append(Tm)
    hsnorm = max(abs(sum(T[x][a][c] * T[y][a][c] for a in range(n) for c in range(n)) - (1 if x == y else 0)) for x in range(len(T)) for y in range(len(T)))
    log.check(f'{slot}: T^6_M orthonormal in Hilbert-Schmidt (numerical, 1e-45)', hsnorm < mpmath.mpf(10) ** -45)
    sq = [mpmath.sqrt(s) for s in svec(j2)]
    rr = random.Random(99)
    for t in range(3):
        w = [mpmath.mpc(rr.uniform(-1, 1), rr.uniform(-1, 1)) for _ in range(n)]
        nw = sum(abs(x) ** 2 for x in w)
        hs = sum(abs(sum(T[x][a][c] * mpmath.conj(w[a]) * w[c] for a in range(n) for c in range(n))) ** 2 for x in range(len(T))) / nw ** 2
        # Q - 1 from exact channel form, f_J numerically through the weighted projectors
        p = [x / s for x, s in zip(w, sq)]
        sv = svec(j2)
        pp = [p[a] * p[c] for a in range(n) for c in range(n)]
        Qm1 = 0
        for J2 in sJ:
            y = {}
            for (r, c), v in proj[J2].items():
                y[r] = y.get(r, 0) + pp[c] * mpmath.mpf(v.numerator) / v.denominator
            fj = sum(sv[r // n] * sv[r % n] * abs(yr) ** 2 for r, yr in y.items()) / nw ** 2
            Qm1 += (mpmath.mpf(q[J2].a.numerator) / q[J2].a.denominator - 1) * fj
        log(f'   numeric point {t}: HS6 (literal T^K) = {mpmath.nstr(hs, 30)}, Q - 1 = {mpmath.nstr(Qm1, 30)}, ratio = {mpmath.nstr(Qm1 / hs, 30)}')
        if prop:
            bn = mpmath.mpf(sp.numer(beta)) / sp.denom(beta)
            log.check(f'{slot}: numeric (Q-1)/HS6 equals exact beta to 1e-40', abs(Qm1 / hs - bn) < mpmath.mpf(10) ** -40)
        # exact HS vs literal HS at this point: compare via b_J
        hs_ex = 0
        for J2 in sJ:
            y = {}
            for (r, c), v in proj[J2].items():
                y[r] = y.get(r, 0) + pp[c] * mpmath.mpf(v.numerator) / v.denominator
            fj = sum(sv[r // n] * sv[r % n] * abs(yr) ** 2 for r, yr in y.items()) / nw ** 2
            hs_ex += mpmath.mpf(sp.numer(b[J2])) / sp.denom(b[J2]) * fj
        log.check(f'{slot}: literal-definition HS6 equals sum b_J f_J to 1e-40', abs(hs_ex - hs) < mpmath.mpf(10) ** -40)

log('\nA.3')
if out['R4']['beta'] is None or out['R5']['beta'] is None:
    log.check('A.3 computable (both betas exist)', False); log.close(); sys.exit(0)
b4 = sp.Rational(out['R4']['beta']); b5 = sp.Rational(out['R5']['beta'])
log('beta_R5/beta_R4 =', b5 / b4, '; beta_R4 - beta_R5 =', b4 - b5)
out['A.3'] = dict(ratio=str(b5 / b4), difference=str(b4 - b5))
if not planted():
    save_json('A2.json', out)
log.close()
