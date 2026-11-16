"""A.4 (R2 at 'line {5/2, -3/2}', 'weight 7/2'), B.1/B.2 (R4, R5 at 'v1', 'pentagonal pyramid', 'D2 ray'),
B.3 (R2 at the same two points, with the four members).
Exact: h from the channel form (lib_crit), gradient/Hessian exact, characteristic polynomials exact (sympy),
signature by Descartes' rule on the exact N_u polynomial, k and r_u by exact Gram ranks, slow polynomial exact.
Numerical (60 digits): Krein signs (kernel of S^2 + t^2 by SVD, M_hat restricted), clause-3 ratios.
924 rhat_6 exact by the Fischer norm of the polynomial w * Theta(w); cross-checked numerically through the
Casimir spin-6 projector (lib_hs route).
'plant': uses 3 h(u) instead of 4 h(u) in M_hat (the K_u and Euler checks must fire)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_crit import *
import sympy as sp, mpmath

log = Log(__file__, '_plant' if planted() else '')
inp = load_inputs()
cens = inp['census']['points']; r2p = inp['R2 points']['points']
OUT = {}


def fischer_924rhat(P):
    """||w * Theta w||_F^2 / ||w||_F^4 ; w unit (CS norm = Fischer norm)."""
    j2 = P.j2; n = P.n; ml = ms(j2)
    p = P.p
    t = [sp.conjugate(p[ml.index(-m2)]) * (-1) ** ((j2 - m2) // 2) for m2 in ml]
    prod = {}
    for a in range(n):
        for b in range(n):
            e = (j2 + ml[a]) // 2 + (j2 + ml[b]) // 2
            prod[e] = prod.get(e, 0) + p[a] * t[b]
    return sp.simplify(sp.radsimp(sp.expand(sum(sp.Abs(sp.expand(c)) ** 2 * fact(e) * fact(2 * j2 - e) for e, c in prod.items()))))


def casimir_924rhat_num(P, proj):
    mpmath.mp.dps = 50
    j2 = P.j2; n = P.n; ml = ms(j2); sv = svec(j2)
    p = [mpmath.mpc(sp.N(sp.re(x), 60), sp.N(sp.im(x), 60)) for x in P.p]
    t = [mpmath.conj(p[ml.index(-m2)]) * (-1) ** ((j2 - m2) // 2) for m2 in ml]
    v = [p[a] * t[b] for a in range(n) for b in range(n)]
    y = {}
    for (r, c), x in proj[2 * j2].items():
        y[r] = y.get(r, 0) + v[c] * mpmath.mpf(x.numerator) / x.denominator
    return 924 * sum(sv[r // n] * sv[r % n] * abs(yr) ** 2 for r, yr in y.items())


def analyse(slot, pname, comps, hd, members=False):
    P = Point(slot, comps, hd).compute()
    if planted():
        P.Mp = P.Mp + P.hu * sp.eye(P.Mp.rows); P.Mw = P.Mw + P.hu * sp.eye(P.Mw.rows)
    N = 2 * P.n
    R = dict()
    log(f'\n[{slot}] {pname}: unit u (CS) = {P.w}')
    R['critical'] = bool(P.critical)
    R['Q'] = str(P.hu)
    log(f'  critical: {P.critical};  Q(u) = h(u) = {P.hu}')
    log.check(f'{slot} {pname}: grad h(u) = 4 h(u) u (Euler identity holds: u.grad = 4h)',
              sp.simplify(sum(g * x for g, x in zip(P.grad_w, P.u_real)) - 4 * P.hu) == 0)
    if not P.critical:
        return R, P
    u = sp.Matrix(P.u_real)
    log.check(f'{slot} {pname}: M_hat u = 8 h u', sp.simplify(P.Mw * u - 8 * P.hu * u) == sp.zeros(N, 1))
    vecs = P.tangent(with_sp1=(slot == 'R2'))
    k, Gm = gram_rank(vecs)
    kern_ok = all(sp.simplify(P.Mw * v) == sp.zeros(N, 1) for v in vecs)
    log.check(f'{slot} {pname}: M_hat vanishes on K_u (all {len(vecs)} generator vectors)', kern_ok)
    Om = P.Jc()
    A = sp.Matrix(len(vecs), len(vecs), lambda i2, k2: sp.simplify(sp.radsimp(sp.expand((vecs[i2].T * Om * vecs[k2])[0]))))
    # radical of sigma on K_u: restrict to a basis of K_u
    basis = []
    for v in vecs:
        if gram_rank(basis + [v])[0] > len(basis): basis.append(v)
    Ab = sp.Matrix(len(basis), len(basis), lambda i2, k2: sp.simplify(sp.radsimp(sp.expand((basis[i2].T * Om * basis[k2])[0]))))
    r_u = len(basis) - Ab.rank()
    R['k'] = k; R['r_u'] = r_u
    log(f'  k = dim K_u = {k};  r_u = {r_u}')
    cpM = charpoly(P.Mp)
    div = sp.Poly((X - 8 * P.hu) * X ** k, X)
    qN, rN = sp.div(cpM, div)
    log.check(f'{slot} {pname}: charpoly(M_hat) divisible by (x - 8h) x^k', rN.is_zero)
    sig = descartes(qN.as_expr())
    R['signature'] = list(sig)
    log(f'  signature on N_u (n-, n0, n+) = {sig}   [dim N_u = {N - 1 - k}]')
    log.check(f'{slot} {pname}: n- + n0 + n+ = dim N_u', sum(sig) == N - 1 - k)
    Mnum = tonum(P.Mw)

    def slow_block(tau, label):
        cp = charpoly(tau * P.Mp)
        z, Q = slow_split(cp.as_expr())
        fl = sp.factor_list(Q.as_expr())
        fac = sp.factor(Q.as_expr())
        rd = root_data(Q)
        wd = words(rd)
        tnum = tonum(tau)
        kr = krein(Mnum, tnum, rd)
        d = dict(zero_multiplicity=z, slow_polynomial=str(fac),
                 roots=[dict(nu=str(r) if r.is_Rational else mpmath.nstr(mpmath.mpmathify(sp.N(r, 40)), 30),
                             multiplicity=m, sign=kd) for r, m, kd in rd],
                 words=wd, krein=[dict(nu=str(x['nu']) if x['nu'].is_Rational else mpmath.nstr(mpmath.mpmathify(sp.N(x['nu'], 40)), 25),
                                       cluster_dim=x['dim'], sign=x['sign']) for x in kr])
        log(f'  [{label}] zero multiplicity = {z}; slow polynomial = {fac}')
        for r, m, kd in rd:
            log(f'      root nu = {r if r.is_Rational else sp.N(r, 30)}  mult {m}  {kd}')
        log(f'      words: {wd}')
        for x in kr:
            log(f'      Krein: nu = {x["nu"] if x["nu"].is_Rational else sp.N(x["nu"], 20)}: cluster dim {x["dim"]}, sign {x["sign"]}, '
                f'M_hat eigs there {x["eig"]}; kernel sv max {x["maxsmall"]}, gap {x["gap"]}')
            log.check(f'{slot} {pname} [{label}]: cluster dim = 2 x multiplicity', x['dim'] == 2 * x['mult'])
        log.check(f'{slot} {pname} [{label}]: zero multiplicity + 2 deg(slow) = {N}', z + 2 * Q.degree() == N)
        return d, rd
    d, rd = slow_block(P.Jc(), 'S = i M_hat')
    R.update(d)
    R['_rd'] = rd
    if members:
        Th = P.Th(); Kk = Th * P.Jc()
        R['members'] = {}
        for lab, tau in (('J', Th), ('k', Kk), ('(i+J)/sqrt2', (P.Jc() + Th) / sp.sqrt(2)), ('(i+k)/sqrt2', (P.Jc() + Kk) / sp.sqrt(2))):
            log.check(f'{slot} {pname}: tau = {lab} squares to -1', sp.simplify(tau * tau + sp.eye(N)) == sp.zeros(N, N))
            dm, _ = slow_block(tau, 'member ' + lab)
            R['members'][lab] = dm
    return R, P


hd = {s: h_poly(s) for s in ('R2', 'R4', 'R5')}
if planted():
    for slot, pname, comps in (('R2', 'line {5/2, -3/2}', r2p['line {5/2, -3/2}']), ('R4', 'v1', cens['v1'])):
        try:
            analyse(slot, pname, comps, hd[slot])
        except Exception as e:
            log(f'  (planted run stopped at {slot} {pname}: {type(e).__name__}: {e})')
    log.close(); sys.exit(0)
_, proj6 = channel_coeffs('R4')

log('==== A.4 and B.3 (R2) ====')
for pname in ('line {5/2, -3/2}', 'weight 7/2'):
    R, P = analyse('R2', pname, r2p[pname], hd['R2'], members=True)
    R.pop('_rd', None)
    OUT[('R2', pname)] = R

log('\n==== B.1 (R4) and B.2 (R5) ====')
store = {}
for pname in ('v1', 'pentagonal pyramid', 'D2 ray'):
    for slot in ('R4', 'R5'):
        R, P = analyse(slot, pname, cens[pname], hd[slot])
        r6 = fischer_924rhat(P)
        r6n = casimir_924rhat_num(P, proj6)
        log(f'  924 rhat_6 = {r6}  (Fischer route, exact);  Casimir route {mpmath.nstr(r6n, 40)}')
        log.check(f'{slot} {pname}: 924 rhat_6 Fischer = Casimir route to 1e-40', abs(mpmath.mpf(sp.N(r6, 60)) - r6n) < mpmath.mpf(10) ** -40)
        R['924 rhat_6'] = str(r6)
        store[(slot, pname)] = R
    # clause 3 ratios
    mpmath.mp.dps = 50
    def sorted_xi(R):
        lst = []
        for r, m, kd in R['_rd']:
            z = mpmath.mpc(sp.N(r, 60)) if not r.is_real else mpmath.mpc(mpmath.mpf(sp.N(r, 60)))
            for _ in range(m): lst.append(z)
        def keyf(z):
            re = z.real if abs(z.real) > mpmath.mpf(10) ** -40 else 0
            im = z.imag if abs(z.imag) > mpmath.mpf(10) ** -40 else 0
            return (re, im)
        lst.sort(key=keyf)
        return lst
    a4 = sorted_xi(store[('R4', pname)]); a5 = sorted_xi(store[('R5', pname)])
    ratios = []
    if len(a4) != len(a5):
        log(f'  B.2 {pname}: slot root counts differ ({len(a4)} vs {len(a5)}); no pairing')
    else:
        for z4, z5 in zip(a4, a5):
            x4 = plus_xi(z4); x5 = plus_xi(z5)
            rt = x5 / x4
            ratios.append(dict(nu_R4=mpmath.nstr(z4, 25), nu_R5=mpmath.nstr(z5, 25), xi_R4=mpmath.nstr(x4, 25), xi_R5=mpmath.nstr(x5, 25), ratio=mpmath.nstr(rt, 30)))
            log(f'  B.2 {pname}: +xi R5 / +xi R4 = {mpmath.nstr(rt, 30)}   (nu_R4 = {mpmath.nstr(z4, 20)}, nu_R5 = {mpmath.nstr(z5, 20)})')
        # exact: is P_R5(nu) proportional to P_R4(c nu) for c = (16/9)^2 ?  Then every root nu_R5 = (81/256) nu_R4,
        # the sort orders agree (scaling by a positive real), and +xi_R5 = (9/16) (+xi_R4) on every branch.
        P4 = sp.Poly(sp.sympify(store[('R4', pname)]['slow_polynomial']), NU)
        P5 = sp.Poly(sp.sympify(store[('R5', pname)]['slow_polynomial']), NU)
        c = sp.Rational(256, 81)
        P4s = sp.Poly(sp.expand(P4.as_expr().subs(NU, c * NU)), NU)
        prop_ok = sp.simplify(P4s.as_expr() * P5.LC() - P5.as_expr() * P4s.LC()) == 0
        log.check(f'B.2 {pname}: P_R5(nu) is proportional to P_R4((256/81) nu) exactly, so each ratio is exactly 9/16', prop_ok)
        allr = all(abs(plus_xi(z5) / plus_xi(z4) - mpmath.mpf(9) / 16) < mpmath.mpf(10) ** -40 for z4, z5 in zip(a4, a5))
        log.check(f'B.2 {pname}: numerical clause-3 ratios all equal 9/16 to 1e-40', allr)
        for x in ratios:
            x['exact'] = '9/16' if (prop_ok and allr) else None
    store[('R5', pname)]['ratio_R5_over_R4'] = ratios
for k_, R in store.items():
    R.pop('_rd', None); OUT[k_] = R
if not planted():
    save_json('B.json', {f'{a}|{b}': v for (a, b), v in OUT.items()})
log.close()
