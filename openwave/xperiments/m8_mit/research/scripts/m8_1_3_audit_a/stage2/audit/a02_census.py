"""B.1 (R4) and B.2 (R5) at all ten census points, by my own stage-1 machinery (own/lib_crit.py), with extra
attacks, then compared field by field with solver_l6/results.json.
Routes per point and slot:
  exact: Q, criticality, k, r_u, signature (Descartes on the exact N_u polynomial), zero multiplicity
         (characteristic polynomial) AND algebraic zero multiplicity from the exact rank sequence of S^p
         (independent of the characteristic polynomial), kernel dimension, slow polynomial, root signs by
         exact factorisation + real-root isolation, 924 rhat_6 (Fischer norm), B.2 proportionality.
  numeric: signature again from 60-digit eigenvalues of M_hat on an orthonormal basis of N_u, and
           Krein signs at 60 and at 100 digits (kernel of S^2 + tau^2 by SVD); a sign is accepted only if
           both precisions agree and the smallest |eigenvalue| on the cluster is > 1e-10.
  criticality attack: the gradient residual at u + 1e-30 * delta (delta a fixed generic vector) must be
           nonzero and O(1e-30), so 'critical' is an exact zero, not a tolerance.
'plant': the slow polynomial of R5 is compared with P_R4(nu * 256/81) after replacing 81 by 80, and l6's Q at one
         point is altered; the matching checks must FAIL. (Run on two points only.)"""
from au import *
from lib_crit import (h_poly, Point, gram_rank, charpoly, descartes, slow_split, root_data, words, krein, tonum,
                      X, NU, plus_xi)
from lib_sym import Theta_matrix
import time

log = Log(__file__, '_plant' if planted() else '')
PL = planted()
cens = inputs()['census']['points']
L6 = solver('solver_l6')
OUT = {}


def fischer_924rhat(P):
    j2 = P.j2; n = P.n; ml = ms(j2); p = P.p
    t = [sp.conjugate(p[ml.index(-m2)]) * (-1) ** ((j2 - m2) // 2) for m2 in ml]
    prod = {}
    for a in range(n):
        for b in range(n):
            e = (j2 + ml[a]) // 2 + (j2 + ml[b]) // 2
            prod[e] = prod.get(e, 0) + p[a] * t[b]
    return sp.nsimplify(sp.simplify(sp.radsimp(sp.expand(sum(sp.Abs(sp.expand(c)) ** 2 * fact(e) * fact(2 * j2 - e) for e, c in prod.items())))))


def num_sig_on_N(P, vecs, dps):
    """numeric signature of M_hat on N_u: orthonormal complement of u and K_u by QR, eigenvalues at dps."""
    mpmath.mp.dps = dps
    N = 2 * P.n
    M = tonum(P.Mw, dps)
    cols = [mpmath.matrix([mpmath.mpf(sp.N(x, dps + 20)) for x in P.u_real])] + \
           [mpmath.matrix([mpmath.mpf(sp.N(x, dps + 20)) for x in v]) for v in vecs]
    A = mpmath.matrix(N, len(cols))
    for c, v in enumerate(cols):
        for r in range(N): A[r, c] = v[r]
    # project out span(A) via SVD of A, then Gram-Schmidt the unit vectors against it
    U, S, V = mpmath.svd_r(A)
    span = [U[:, i] for i in range(len(cols)) if S[i] > mpmath.mpf(10) ** (-dps // 2)]
    basis = list(span)
    for e in range(N):
        v = mpmath.matrix(N, 1); v[e] = 1
        for b in basis:
            v = v - (b.T * v)[0] * b
        nv = mpmath.norm(v)
        if nv > mpmath.mpf('0.1'):
            basis.append(v / nv)
        if len(basis) == N: break
    comp = basis[len(span):]
    Bm = mpmath.matrix(N, len(comp))
    for c, v in enumerate(comp):
        for r in range(N): Bm[r, c] = v[r]
    R = Bm.T * M * Bm
    ev = mpmath.eigsy(R)[0]
    ev = [ev[i] for i in range(len(comp))]
    tol = mpmath.mpf(10) ** (-dps // 3)
    sig = (sum(1 for e in ev if e < -tol), sum(1 for e in ev if abs(e) <= tol), sum(1 for e in ev if e > tol))
    nz = [abs(e) for e in ev if abs(e) > tol]
    zs = [abs(e) for e in ev if abs(e) <= tol]
    return sig, (min(nz) if nz else None), (max(zs) if zs else None), len(span)


def alg_zero_mult(S):
    """algebraic multiplicity of 0 = N - rank(S^N) (S^p ranks stabilise); also kernel dim.
    Exact over Q when every entry of S is rational; otherwise numeric at 80 digits (SVD), with the gap reported."""
    N = S.rows
    ranks = []
    rational = all(x.is_Rational for x in S)
    if rational:
        Sp = sp.eye(N)
        for p in range(1, N + 1):
            Sp = Sp * S
            ranks.append(Sp.rank())
            if len(ranks) >= 2 and ranks[-1] == ranks[-2]:
                break
        return N - ranks[-1], N - ranks[0], ranks, 'exact'
    mpmath.mp.dps = 80
    Sn = tonum(S, 80)
    Sp = mpmath.eye(N); worst = []
    for p in range(1, N + 1):
        Sp = Sp * Sn
        sv = sorted(mpmath.svd_r(Sp, compute_uv=False))
        small = [x for x in sv if x < mpmath.mpf(10) ** -40]
        big = [x for x in sv if x >= mpmath.mpf(10) ** -40]
        worst.append((mpmath.nstr(max(small), 3) if small else '-', mpmath.nstr(min(big), 3) if big else '-'))
        ranks.append(len(big))
        if len(ranks) >= 2 and ranks[-1] == ranks[-2]:
            break
    return N - ranks[-1], N - ranks[0], ranks, f'numeric 80 dig, (max small sv, min large sv) per power {worst}'


def analyse(slot, pname, comps, hd):
    t0 = time.time()
    P = Point(slot, comps, hd).compute()
    N = 2 * P.n
    R = dict()
    R['critical'] = bool(P.critical); R['Q'] = str(P.hu)
    log(f'\n[{slot}] {pname}: u = {P.w}\n  critical: {P.critical};  Q = {P.hu}')
    # criticality attack: perturbed point
    mpmath.mp.dps = 60
    rng_delta = [sp.Rational((7 * i * i + 3) % 11 - 5, 7) for i in range(N)]
    eps = sp.Rational(1, 10 ** 30) if not PL else sp.Integer(0)  # planted: no perturbation -> residual 0 must FAIL
    sub2 = {}
    pr = [sp.re(x) for x in P.p] + [sp.im(x) for x in P.p]
    sv = [sp.sqrt(s2(P.j2, m2)) for m2 in ms(P.j2)] * 2
    for i, g in enumerate(P.gens):
        sub2[g] = pr[i] + eps * rng_delta[i] / sv[i]
    up = sp.Matrix([sp.N(pr[i] * sv[i], 80) + sp.N(eps * rng_delta[i], 80) for i in range(N)])
    hp = sp.N(P.h.as_expr().subs(sub2), 80)
    nrm2 = sum(x ** 2 for x in up)
    gp = [sp.N(P.h.diff(g).as_expr().subs(sub2), 80) / sv[i] for i, g in enumerate(P.gens)]
    lam = 4 * hp / nrm2
    res = max(abs(gp[i] - lam * up[i]) for i in range(N))
    R['perturbed residual'] = str(sp.N(res, 5))
    log(f'  criticality attack: residual of grad h - (4h/|u|^2) u at u + 1e-30 delta = {sp.N(res, 5)}')
    log.check(f'{slot} {pname}: perturbing by 1e-30 gives a nonzero residual of order 1e-30 (so the exact zero is not a tolerance)',
              sp.Float(1e-33) < res < sp.Float(1e-27))
    u = sp.Matrix(P.u_real)
    log.check(f'{slot} {pname}: M_hat u = 8 h u (exact)', sp.simplify(P.Mw * u - 8 * P.hu * u) == sp.zeros(N, 1))
    vecs = P.tangent(with_sp1=False)
    k, Gm = gram_rank(vecs)
    log.check(f'{slot} {pname}: M_hat vanishes on K_u (exact)', all(sp.simplify(P.Mw * v) == sp.zeros(N, 1) for v in vecs))
    Om = P.Jc()
    basis = []
    for v in vecs:
        if gram_rank(basis + [v])[0] > len(basis): basis.append(v)
    Ab = sp.Matrix(len(basis), len(basis), lambda a, b: sp.simplify(sp.radsimp(sp.expand((basis[a].T * Om * basis[b])[0]))))
    r_u = len(basis) - Ab.rank()
    R['k'] = k; R['r_u'] = r_u
    cpM = charpoly(P.Mp)
    qN, rN = sp.div(cpM, sp.Poly((X - 8 * P.hu) * X ** k, X))
    log.check(f'{slot} {pname}: charpoly(M_hat) divisible by (x - 8h) x^k', rN.is_zero)
    sig = descartes(qN.as_expr())
    R['signature'] = list(sig)
    if PL:  # planted: the numeric route sees M_hat + I/5, which must change the count at v2
        Psave = P.Mw; P.Mw = P.Mw + sp.Rational(1, 5) * sp.eye(N)
    nsig, mingap, maxz, nspan = num_sig_on_N(P, vecs, 60)
    if PL:
        P.Mw = Psave
    log(f'  k = {k}, r_u = {r_u}; signature exact (Descartes) {sig}; numeric 60-digit eigen-decomposition on N_u {nsig}, '
        f'min |nonzero eig| {mpmath.nstr(mingap, 5) if mingap is not None else None}, max |zero eig| {mpmath.nstr(maxz, 3) if maxz is not None else None}')
    log.check(f'{slot} {pname}: signature, exact Descartes = numeric eigen-decomposition on an explicit basis of N_u', tuple(sig) == tuple(nsig))
    # slow matrix
    S = P.Jc() * P.Mp
    cp = charpoly(S)
    z, Qs = slow_split(cp.as_expr())
    az, kd, ranks, how = alg_zero_mult(S)
    R['zero multiplicity'] = z; R['kernel dim of S'] = kd; R['alg zero mult (rank sequence)'] = az
    log(f'  zero multiplicity: charpoly {z}; rank sequence of S^p {ranks} ({how}) -> algebraic {az}, kernel dim {kd}')
    log.check(f'{slot} {pname}: algebraic zero multiplicity, charpoly = rank sequence', z == (kd if PL else az))  # planted: kernel dim in place of alg. mult.
    fac = sp.factor(Qs.as_expr())
    rd = root_data(Qs)
    R['slow polynomial'] = str(fac); R['words'] = words(rd)
    R['roots'] = [dict(nu=(str(r) if r.is_Rational else str(sp.N(r, 30))), mult=m, kind=kd_) for r, m, kd_ in rd]
    log(f'  slow polynomial {fac}\n  words: {R["words"]}')
    Mn60 = tonum(P.Mw, 60)
    kr60 = krein(Mn60, tonum(P.Jc(), 60), rd, dps=60)
    Mn100 = tonum(P.Mw, 100)
    kr100 = krein(Mn100, tonum(P.Jc(), 100), rd, dps=100)
    R['krein'] = []
    for a, b in zip(kr60, kr100):
        mineig = min(abs(mpmath.mpf(e)) for e in a['eig'])
        ok = a['sign'] == b['sign'] and a['dim'] == b['dim'] == 2 * a['mult'] and mineig > 1e-10
        R['krein'].append(dict(nu=str(a['nu']) if a['nu'].is_Rational else str(sp.N(a['nu'], 25)), mult=a['mult'], dim=a['dim'], sign=a['sign'],
                               min_abs_eig=mpmath.nstr(mineig, 5)))
        log(f'  Krein nu={R["krein"][-1]["nu"]}: {a["sign"]} (60 dig) / {b["sign"]} (100 dig); cluster dim {a["dim"]}; '
            f'min |M_hat eig| on cluster {mpmath.nstr(mineig, 5)}; kernel sv max {a["maxsmall"]} / {b["maxsmall"]}, gap {a["gap"]}')
        log.check(f'{slot} {pname}: Krein sign at nu={R["krein"][-1]["nu"]} agrees at 60/100 digits, semisimple cluster, definite margin', ok)
    r6 = fischer_924rhat(P)
    R['924 r6hat'] = str(r6)
    R['_rd'] = rd
    R['_Q'] = Qs
    log(f'  924 rhat_6 = {r6}   ({time.time() - t0:.0f}s)')
    return R


def same_poly(a, b):
    A = sp.Poly(sym(a), sp.Symbol('nu')); B = sp.Poly(sym(b), sp.Symbol('nu'))
    return sp.expand(A.as_expr() * B.LC() - B.as_expr() * A.LC()) == 0


def compare(slot, pname, R):
    blk = 'B.1' if slot == 'R4' else 'B.2'
    T = L6[blk][slot][pname]
    Qt = T['Q']
    if PL and pname == 'v2':
        Qt = '148/143'
    items = [('critical', T['critical'] == R['critical']), ('Q', sym(Qt) == sym(R['Q'])),
             ('924 rhat_6', sym(T['924 r6hat']) == sym(R['924 r6hat'])),
             ('signature', list(T['signature']) == R['signature']), ('k', T['k'] == R['k']), ('r_u', T['r_u'] == R['r_u']),
             ('zero multiplicity', T['zero multiplicity'] == R['zero multiplicity']),
             ('slow polynomial', same_poly(T['slow polynomial'], R['slow polynomial'])), ('words', T['words'] == R['words'])]
    # Krein: match by numeric nu
    mk = {round(float(sp.N(sym(x['nu']) if 'CRootOf' not in x['nu'] else sp.Float(x['nu_num']), 20)), 12): x['Krein'] for x in T['krein']}
    ok = len(mk) == len(R['krein']) and all(mk.get(round(float(sp.N(sym(x['nu']), 20)), 12)) == x['sign'] for x in R['krein'])
    items.append(('Krein signs', ok))
    res = {}
    for name, okk in items:
        log.check(f'l6 vs mine {slot} {pname}: {name}', okk)
        res[name] = 'AGREE' if okk else 'DISAGREE'
    return res


hd = {s: h_poly(s) for s in ('R4', 'R5')}
pts = list(cens.keys()) if not PL else ['v2', 'octahedron']
store = {}
for pname in pts:
    for slot in ('R4', 'R5'):
        R = analyse(slot, pname, cens[pname], hd[slot])
        R['verdicts'] = compare(slot, pname, R)
        store[(slot, pname)] = R
    # B.2 exact proportionality and clause-3 ratios
    P4 = store[('R4', pname)]['_Q']; P5 = store[('R5', pname)]['_Q']
    c = sp.Rational(256, 80 if PL else 81)
    P4s = sp.Poly(sp.expand(P4.as_expr().subs(NU, c * NU)), NU)
    prop = sp.expand(P4s.as_expr() * P5.LC() - P5.as_expr() * P4s.LC()) == 0
    log.check(f'B.2 {pname}: P_R5(nu) proportional to P_R4((256/81) nu) exactly -> every clause-3 ratio +xi_R5/+xi_R4 = 9/16', prop)
    mpmath.mp.dps = 50
    def sxi(rd):
        lst = []
        for r, m, kd_ in rd:
            z = mpmath.mpc(complex(sp.N(r, 60))) if not r.is_real else mpmath.mpc(mpmath.mpf(sp.N(r, 60)))
            if not r.is_real:
                z = mpmath.mpc(mpmath.mpf(sp.N(sp.re(r), 60)), mpmath.mpf(sp.N(sp.im(r), 60)))
            lst += [z] * m
        return sorted(lst, key=lambda z: (z.real if abs(z.real) > 1e-40 else 0, z.imag if abs(z.imag) > 1e-40 else 0))
    a4 = sxi(store[('R4', pname)]['_rd']); a5 = sxi(store[('R5', pname)]['_rd'])
    rat = [plus_xi(z5) / plus_xi(z4) for z4, z5 in zip(a4, a5)]
    numok = len(a4) == len(a5) and all(abs(r - mpmath.mpf(9) / 16) < 1e-40 for r in rat)
    log.check(f'B.2 {pname}: numeric clause-3 ratios (sort, +xi branch) all 9/16 to 1e-40 ({len(rat)} pairs)', numok)
    store[('R5', pname)]['ratio'] = '9/16' if prop and numok else 'not uniform'
    l6r = L6['B.2']['R5 vs R4 eigenvalue ratios (clause 3)'].get(pname, {}).get('common ratio +xi_R5/+xi_R4')
    log.check(f'l6 vs mine B.2 {pname}: ratio', l6r == store[('R5', pname)]['ratio'], f'l6 {l6r}')
    store[('R5', pname)]['verdicts']['ratio'] = 'AGREE' if l6r == store[('R5', pname)]['ratio'] else 'DISAGREE'

if not PL:
    save('census.json', {f'{a}|{b}': {k: v for k, v in R.items() if not k.startswith('_')} for (a, b), R in store.items()})
log.close()
