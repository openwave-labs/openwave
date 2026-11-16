"""A.4 and B.3 (R2) at all eight points, with the four members, by my own stage-1 machinery (own/lib_crit: h, the
exact gradient and Hessian, K_u) and an exact characteristic-polynomial route over Q or Q(sqrt51)
(sympy DomainMatrix over the algebraic field), then compared with solver_r2/results.json.
For the members (i+J)/sqrt2 and (i+k)/sqrt2 the exact route uses (i+tau')M (rational/Q(sqrt51) entries) and
nu = y^2/2. Roots are classified by exact factorisation over the field plus mpmath polyroots at 80 and 120 digits
(both must agree); every root list is also matched against the squares of 50-digit eigenvalues of tau*M_hat.
Krein signs: 60 and 100 digits, as in a02. Algebraic zero multiplicity: rank sequence of S^p.
Also: the alternative reading of the fibre group (no continuous Sp(1): K_u = phase + rotations only), and a
numeric test that h is invariant under the one-parameter group exp(t J) = cos t + sin t J.
'plant': runs two points; uses Theta without its signs for the Sp(1) invariance test, alters r2's Q at one point,
and shifts a Krein test by cluster union; the matching checks must FAIL."""
from au import *
from lib_crit import h_poly, Point, gram_rank, descartes, words, krein, tonum, X, NU
from sympy.polys.matrices import DomainMatrix
import time, random

log = Log(__file__, '_plant' if planted() else '')
PL = planted()
r2p = inputs()['R2 points']['points']
R2S = solver('solver_r2')
Y = sp.Symbol('y')


def field_of(M):
    rads = set()
    for x in M:
        for a in sp.preorder_traversal(x):
            if isinstance(a, sp.Pow) and a.exp == sp.Rational(1, 2):
                rads.add(a)
    if not rads:
        return sp.QQ, None
    ext = sp.sqrt(sp.prod([sp.Integer(r.base) for r in rads]))  # works here: a single square class (checked below)
    return sp.QQ.algebraic_field(*sorted(rads, key=str)), rads


def cp_exact(M):
    Kf, rads = field_of(M)
    dM = DomainMatrix.from_Matrix(M.applyfunc(sp.radsimp)).convert_to(Kf)
    cs = [sp.nsimplify(sp.radsimp(Kf.to_sympy(c))) if Kf != sp.QQ else sp.Rational(Kf.to_sympy(c)) for c in dM.charpoly()]
    return cs, Kf


def rank_exact_or_num(Mt):
    Kf, rads = field_of(Mt)
    if Kf == sp.QQ:
        return DomainMatrix.from_Matrix(Mt).convert_to(sp.QQ).rank(), 'exact'
    return DomainMatrix.from_Matrix(Mt.applyfunc(sp.radsimp)).convert_to(Kf).rank(), 'exact over ' + str(Kf)


def alg_zero(S):
    N = S.rows; ranks = []; Sp_ = sp.eye(N)
    for p in range(1, N + 1):
        Sp_ = (Sp_ * S).applyfunc(sp.radsimp)
        r, how = rank_exact_or_num(Sp_)
        ranks.append(r)
        if len(ranks) >= 2 and ranks[-1] == ranks[-2]:
            break
    return N - ranks[-1], N - ranks[0], ranks


def slow_from_cs(cs, half=False):
    """cs: charpoly coeffs (highest first) in the variable y. Return zero mult, monic slow poly in nu (sympy expr)."""
    n = len(cs) - 1
    z = 0
    while cs[n - z] == 0:
        z += 1
    P = sum(c * Y ** (n - i) for i, c in enumerate(cs))
    P = sp.expand(sp.cancel(P / Y ** z))
    Pp = sp.Poly(P, Y)
    odd_ok = all(sp.simplify(Pp.coeff_monomial(Y ** k)) == 0 for k in range(1, Pp.degree() + 1, 2))
    Qn = sp.expand(P.subs(Y, sp.sqrt(NU)))
    if half:
        Qn = sp.expand(P.subs(Y, sp.sqrt(2 * NU)))
    Qn = sp.Poly(Qn, NU)
    Qn = sp.Poly(sp.expand(Qn.as_expr() / Qn.LC()), NU)
    return z, Qn, odd_ok


def classify(Qn, Kf):
    """exact factorisation over Kf; numeric roots at 80 and 120 digits; returns root list and word."""
    ext = None if Kf == sp.QQ else [sp.sqrt(51)]
    if ext:
        fl = sp.factor_list(Qn.as_expr(), extension=ext)
    else:
        fl = sp.factor_list(Qn.as_expr())
    roots = []
    for fac, mult in fl[1]:
        F = sp.Poly(fac, NU)
        if F.degree() == 0: continue
        cs = [complex(0)] and F.all_coeffs()
        kinds = []
        for dps in (80, 120):
            mpmath.mp.dps = dps
            rr = mpmath.polyroots([mpmath.mpmathify(sp.N(c, dps + 20)) for c in cs], maxsteps=400, extraprec=2 * dps)
            kk = []
            for r in rr:
                r = mpmath.mpc(r)
                if abs(r.imag) < mpmath.mpf(10) ** (-dps // 2):
                    kk.append(('neg' if r.real < 0 else 'pos', mpmath.mpf(r.real)))
                else:
                    kk.append(('nonreal', r))
            kinds.append(kk)
        a = sorted(k for k, _ in kinds[0]); b = sorted(k for k, _ in kinds[1])
        assert a == b, 'root classification differs between 80 and 120 digits'
        mpmath.mp.dps = 50
        for k_, r in kinds[1]:
            roots.append(dict(kind=k_, mult=mult, val=r, fac=fac))
    return roots, fl


def wordof(roots):
    kinds = [r['kind'] for r in roots]
    if all(k == 'neg' for k in kinds):
        ms_ = [r['mult'] for r in roots]
        if all(m == 1 for m in ms_): return 'elliptic, simple'
        if sum(1 for m in ms_ if m == 2) == 1 and all(m in (1, 2) for m in ms_): return 'elliptic, one double frequency'
        return 'elliptic (not named)'
    nonreal = [r for r in roots if r['kind'] == 'nonreal']; pos = [r for r in roots if r['kind'] == 'pos']
    if not nonreal:
        tot = sum(r['mult'] for r in pos)
        if tot == 1: return 'hyperbolic, one real pair'
        if tot >= 2: return 'hyperbolic, real pairs'
    if not pos and len(nonreal) == 2 and all(r['mult'] == 1 for r in nonreal): return 'hyperbolic, a complex quartet'
    return 'hyperbolic (not named)'


def canon(wd):
    w = wd.lower()
    for key, c in (('elliptic, simple', 'ES'), ('elliptic, one double', 'E1D'), ('hyperbolic, one real pair', 'H1'),
                   ('hyperbolic, real pairs', 'HR'), ('hyperbolic, a complex quartet', 'HQ')):
        if w.startswith(key): return c
    return 'E?' if w.startswith('elliptic') else 'H?'


def krein_num(Mw, tau, roots, half=False):
    """Krein sign per negative root (tau*M_hat eigen pairs), at 60 and 100 digits."""
    out = []
    for r in roots:
        if r['kind'] != 'neg': continue
        rr = []
        for dps in (60, 100):
            mpmath.mp.dps = dps
            Mn = tonum(Mw, dps); Tn = tonum(tau, dps) * (1 / mpmath.sqrt(2) if half else 1)
            S = Tn * Mn
            N = S.rows
            t2 = -mpmath.mpf(r['val'])
            if isinstance(r['fac'], sp.Expr):
                # refine the root at this precision from its exact factor
                F = sp.Poly(r['fac'], NU)
                t2 = -mpmath.findroot(lambda x: sum(mpmath.mpmathify(sp.N(c, dps + 20)) * x ** (F.degree() - i) for i, c in enumerate(F.all_coeffs())), mpmath.mpf(r['val']))
            A = S * S + t2 * mpmath.eye(N)
            U, sv, V = mpmath.svd_r(A)
            svl = [sv[i] for i in range(N)]
            idx = [i for i in range(N) if svl[i] < mpmath.mpf(10) ** (-dps // 2)]
            gap = min([svl[i] for i in range(N) if i not in idx] or [mpmath.mpf(1)])
            B = mpmath.matrix(N, len(idx))
            for c, i in enumerate(idx):
                for k in range(N): B[k, c] = V[i, k]
            ev = mpmath.eigsy(B.T * Mn * B)[0]
            evl = [ev[i] for i in range(len(idx))]
            sgn = '+' if evl and all(e > 0 for e in evl) else ('-' if evl and all(e < 0 for e in evl) else 'indefinite')
            rr.append((sgn, len(idx), min(abs(e) for e in evl) if evl else 0, max([svl[i] for i in idx] or [0]), gap))
        ok = rr[0][0] == rr[1][0] and rr[0][1] == rr[1][1] == 2 * r['mult'] and rr[0][2] > 1e-10
        out.append(dict(nu=mpmath.nstr(r['val'], 20), mult=r['mult'], sign=rr[0][0], dim=rr[0][1], ok=ok,
                        min_abs_eig=mpmath.nstr(rr[0][2], 4), sv=mpmath.nstr(rr[1][3], 3), gap=mpmath.nstr(rr[0][4], 3)))
    return out


def eig_match(Mw, tau, roots, scale, z):
    """50-digit eigenvalues of tau*M_hat (times scale): zero count and squares vs roots (with multiplicity)."""
    mpmath.mp.dps = 50
    S = tonum(tau, 50) * tonum(Mw, 50) * scale
    ev = mpmath.eig(S, left=False, right=False)
    sq = sorted([e * e for e in ev if abs(e) > mpmath.mpf(10) ** -12], key=lambda c: (mpmath.re(c), mpmath.im(c)))
    nz = sum(1 for e in ev if abs(e) <= mpmath.mpf(10) ** -12)
    want = []
    for r in roots:
        want += [mpmath.mpc(r['val'])] * (2 * r['mult'])
    if len(want) != len(sq): return False, nz, None
    pool = list(sq); err = mpmath.mpf(0)
    for wv in want:  # greedy nearest matching (conjugate pairs would interleave under a plain sort)
        j = min(range(len(pool)), key=lambda i: abs(pool[i] - wv))
        err = max(err, abs(pool[j] - wv)); pool.pop(j)
    return err < mpmath.mpf(10) ** -8, nz, err


def structure(P, tau, label, half):
    Tm = tau if not half else tau
    S = (Tm * P.Mp).applyfunc(sp.radsimp)
    cs, Kf = cp_exact(S)
    z, Qn, odd_ok = slow_from_cs(cs, half=half)
    roots, fl = classify(Qn, Kf)
    wd = wordof(roots)
    az, kd, ranks = alg_zero(S)
    mpmath.mp.dps = 50
    scale = 1 / mpmath.sqrt(2) if half else 1
    kr = krein_num(P.Mw, Tm, roots, half=half)
    em, nz, err = eig_match(P.Mw, Tm, roots, scale, z)
    rational = all(sp.sympify(c).is_Rational for c in Qn.all_coeffs())
    log(f'  [{label}] zero mult {z} (alg via ranks {az}, kernel dim {kd}); slow poly rational: {rational}; '
        f'{"factored over Q: " + str(sp.factor(Qn.as_expr())) if rational else "monic over Q(sqrt51): " + str(Qn.as_expr())}')
    log(f'      roots: ' + '; '.join(f'{r["kind"]} x{r["mult"]} {mpmath.nstr(r["val"], 15)}' for r in roots) + f'  -> {wd}')
    for x in kr:
        log(f'      Krein nu={x["nu"]}: {x["sign"]} (dim {x["dim"]}, min|eig| {x["min_abs_eig"]}, ker sv {x["sv"]}, gap {x["gap"]})')
        log.check(f'R2 {P._name} [{label}]: Krein at nu={x["nu"]} stable at 60/100 digits, semisimple, definite margin', x['ok'])
    log.check(f'R2 {P._name} [{label}]: odd coefficients vanish (Hamiltonian charpoly)', odd_ok)
    log.check(f'R2 {P._name} [{label}]: algebraic zero multiplicity charpoly = rank sequence', z == az)
    log.check(f'R2 {P._name} [{label}]: 50-digit eigenvalues of tau*M_hat: squares match the slow roots, zero count = {z}', em and nz == z,
              f'max err {mpmath.nstr(err, 3) if err is not None else "count mismatch"}, numeric zeros {nz}')
    return dict(z=z, kernel=kd, Q=Qn, rational=rational, roots=roots, words=wd, krein=kr, Kf=str(Kf))


def same_monic(theirs, Qn, half=False):
    T = sp.Poly(sym(theirs), sp.Symbol('nu'))
    T = sp.Poly(sp.expand(T.as_expr() / T.LC()), NU)
    d = [sp.nsimplify(sp.radsimp(a - b)) for a, b in zip(T.all_coeffs(), Qn.all_coeffs())]
    return T.degree() == Qn.degree() and all(x == 0 for x in d)


def krein_match(theirs, mine):
    tk = {round(float(sp.N(sym(x['nu']) if 'nu' in x else 0)), 9): x['Krein sign'] for x in theirs}
    mk = {round(float(x['nu']), 9): x['sign'] for x in mine}
    return tk == mk, tk, mk


hd = h_poly('R2')
names = list(r2p.keys()) if not PL else ['weight 7/2', 'line {7/2, -5/2}']
OUT = {}
rng = random.Random(3)
for pname in names:
    t0 = time.time()
    P = Point('R2', r2p[pname], hd).compute()
    P._name = pname
    N = 16
    log(f'\n==== R2 {pname}: u = {P.w}\n  critical {P.critical}; Q = {P.hu}')
    R = dict(critical=bool(P.critical), Q=str(P.hu))
    # perturbation attack
    sv = [sp.sqrt(s2(7, m2)) for m2 in ms(7)] * 2
    pr = [sp.re(x) for x in P.p] + [sp.im(x) for x in P.p]
    delta = [sp.Rational((5 * i * i + 2) % 13 - 6, 9) for i in range(N)]
    eps = sp.Rational(1, 10 ** 30)
    sub2 = {g: pr[i] + eps * delta[i] / sv[i] for i, g in enumerate(P.gens)}
    up = [sp.N(pr[i] * sv[i] + eps * delta[i], 80) for i in range(N)]
    hp = sp.N(P.h.as_expr().subs(sub2), 80)
    lam = 4 * hp / sum(x ** 2 for x in up)
    res = max(abs(sp.N(P.h.diff(g).as_expr().subs(sub2), 80) / sv[i] - lam * up[i]) for i, g in enumerate(P.gens))
    log(f'  criticality attack: residual at u + 1e-30 delta = {sp.N(res, 4)}')
    log.check(f'R2 {pname}: perturbed residual nonzero and O(1e-30) (exact zero at u is not a tolerance)', sp.Float(1e-33) < res < sp.Float(1e-27))
    u = sp.Matrix(P.u_real)
    vecs = P.tangent(with_sp1=True)
    log.check(f'R2 {pname}: M_hat vanishes on all six generator vectors incl. Theta u, Theta(iu) (exact)',
              all(sp.simplify(P.Mw * v) == sp.zeros(N, 1) for v in vecs))
    # Sp(1) invariance of h, numeric: h(cos t u + sin t Theta u) = h(u) at random complex u
    mpmath.mp.dps = 50
    Th = P.Th()
    if PL:
        Th = sp.Matrix(sp.BlockMatrix([[sp.Matrix(8, 8, lambda a, b: 1 if a + b == 7 else 0), sp.zeros(8)], [sp.zeros(8), -sp.Matrix(8, 8, lambda a, b: 1 if a + b == 7 else 0)]]))
    hfun = sp.lambdify(P.gens, P.h.as_expr(), 'mpmath')
    worst = 0
    for _ in range(3):
        x = mpmath.matrix([mpmath.mpf(rng.gauss(0, 1)) for _ in range(N)])
        tN = tonum(Th, 50)
        t = mpmath.mpf(rng.random())
        y = mpmath.cos(t) * x + mpmath.sin(t) * (tN * x)
        def hval(v):
            return hfun(*[v[i] / mpmath.sqrt(s2(7, ms(7)[i % 8])) for i in range(N)])
        worst = max(worst, abs(hval(y) - hval(x)) / abs(hval(x)))
    log.check(f'R2 {pname}: h invariant under exp(tJ) = cos t + sin t Theta at random u (rel. err < 1e-40)', worst < mpmath.mpf(10) ** -40,
              f'rel err {mpmath.nstr(worst, 3)}')
    # K_u, r_u (Sp(1) reading)
    def kr_of(vs):
        k, _ = gram_rank(vs)
        basis = []
        for v in vs:
            if gram_rank(basis + [v])[0] > len(basis): basis.append(v)
        Om = P.Jc()
        Ab = sp.Matrix(len(basis), len(basis), lambda a, b: sp.simplify(sp.radsimp(sp.expand((basis[a].T * Om * basis[b])[0]))))
        return k, len(basis) - Ab.rank()
    k, r_u = kr_of(vecs)
    k_alt, r_alt = kr_of(vecs[:4])
    cpM, Kf = cp_exact(P.Mp)
    cpM_rat = all(sp.sympify(c).is_Rational for c in cpM)
    log.check(f'R2 {pname}: charpoly(M_hat) rational', cpM_rat)
    cpMp = sp.Poly(sum(c * X ** (16 - i) for i, c in enumerate(cpM)), X)
    def sig_for(kk):
        qN, rN = sp.div(cpMp, sp.Poly((X - 8 * P.hu) * X ** kk, X))
        return descartes(qN.as_expr()) if rN.is_zero else None
    sig = sig_for(k); sig_alt = sig_for(k_alt)
    # numeric signature on an explicit N_u basis
    mpmath.mp.dps = 60
    Mn = tonum(P.Mw, 60)
    cols = [mpmath.matrix([mpmath.mpf(sp.N(x, 80)) for x in P.u_real])] + [mpmath.matrix([mpmath.mpf(sp.N(x, 80)) for x in v]) for v in vecs]
    A = mpmath.matrix(N, len(cols))
    for c, v in enumerate(cols):
        for r in range(N): A[r, c] = v[r]
    Uu, Ss, Vv = mpmath.svd_r(A)
    basis = [Uu[:, i] for i in range(len(cols)) if Ss[i] > mpmath.mpf(10) ** -30]
    nspan = len(basis)
    for e in range(N):
        v = mpmath.matrix(N, 1); v[e] = 1
        for b in basis: v = v - (b.T * v)[0] * b
        if mpmath.norm(v) > 0.1: basis.append(v / mpmath.norm(v))
    comp = basis[nspan:]
    Bm = mpmath.matrix(N, len(comp))
    for c, v in enumerate(comp):
        for r in range(N): Bm[r, c] = v[r]
    ev = mpmath.eigsy(Bm.T * Mn * Bm)[0]
    ev = [ev[i] for i in range(len(comp))]
    tol = mpmath.mpf(10) ** -20
    nsig = (sum(1 for e in ev if e < -tol), sum(1 for e in ev if abs(e) <= tol), sum(1 for e in ev if e > tol))
    log(f'  k = {k}, r_u = {r_u}; signature {sig} (exact) / {nsig} (60-digit eigen-decomposition on N_u, '
        f'min |nonzero| {mpmath.nstr(min(abs(e) for e in ev if abs(e) > tol), 4)})')
    log(f'  alternative reading (no continuous Sp(1); K_u = phase + rotations): k = {k_alt}, r_u = {r_alt}, signature {sig_alt}')
    log.check(f'R2 {pname}: signature exact = numeric eigen-decomposition', tuple(sig) == tuple(nsig))
    R.update(k=k, r_u=r_u, signature=list(sig), alt=dict(k=k_alt, r_u=r_alt, signature=list(sig_alt) if sig_alt else None))
    # structures
    Jc = P.Jc(); Thm = P.Th(); Kk = Thm * Jc
    R['i'] = structure(P, Jc, 'i', False)
    R['J'] = structure(P, Thm, 'J', False)
    R['k'] = structure(P, Kk, 'k', False)
    R['(i+J)/sqrt2'] = structure(P, Jc + Thm, '(i+J)/sqrt2', True)
    R['(i+k)/sqrt2'] = structure(P, Jc + Kk, '(i+k)/sqrt2', True)
    # ---- compare with r2 ----
    T4 = R2S['A.4'][pname]; T3 = R2S['B.3'][pname]
    Qt = T4['Q'] if not (PL and pname == 'weight 7/2') else '145/143'
    v = {}
    def cmp(key, ok):
        log.check(f'r2 vs mine {pname}: {key}', ok); v[key] = 'AGREE' if ok else 'DISAGREE'
    cmp('A.4 critical', T4['critical'] == R['critical'] and T3['critical'] == R['critical'])
    cmp('A.4 Q', sym(Qt) == sym(R['Q']))
    cmp('A.4/B.3 signature', list(T4['signature']) == R['signature'] == list(T3['signature']))
    cmp('B.3 k', T3['k'] == k); cmp('B.3 r_u', T3['r_u'] == r_u)
    def cmp_struct(lab, T, M):
        cmp(f'{lab} zero multiplicity', T['zero multiplicity'] == M['z'])
        cmp(f'{lab} slow polynomial', same_monic(T['slow polynomial'], M['Q']))
        cmp(f'{lab} words', canon(T['slow-flow words']) == canon(M['words']))
        ok, tk, mk = krein_match(T['Krein signs'], M['krein'])
        cmp(f'{lab} Krein signs', ok)
        if not ok: log(f'      theirs {tk}\n      mine   {mk}')
    cmp_struct('i', T3, R['i'])
    for lab in ('J', 'k', '(i+J)/sqrt2', '(i+k)/sqrt2'):
        cmp_struct(lab, T3['members'][lab], R[lab])
    R['verdicts'] = v
    OUT[pname] = R
    log(f'  ({time.time() - t0:.0f}s)')

if PL:  # a Krein test on the union of a + and a - cluster must not come out definite
    P = Point('R2', r2p['weight 5/2'], hd).compute()
    mpmath.mp.dps = 60
    S = tonum(P.Jc(), 60) * tonum(P.Mw, 60); Mn = tonum(P.Mw, 60)
    vs = []
    for t2 in (sp.Rational(1600, 20449), sp.Rational(166464, 20449)):  # tau = 40/143 (-), 408/143 (+) per r2
        A = S * S + mpmath.mpf(sp.N(t2, 80)) * mpmath.eye(16)
        U, sv_, V = mpmath.svd_r(A)
        vs += [V[i, :].T for i in range(16) if sv_[i] < mpmath.mpf(10) ** -30]
    B = mpmath.matrix(16, len(vs))
    for c, x in enumerate(vs):
        for r in range(16): B[r, c] = x[r]
    ev = mpmath.eigsy(B.T * Mn * B)[0]
    log.check('planted: Krein definiteness on the union of a - and a + cluster', all(ev[i] > 0 for i in range(len(vs))) or all(ev[i] < 0 for i in range(len(vs))))

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items() if k not in ('fac',)}
    if isinstance(o, list): return [clean(x) for x in o]
    if isinstance(o, sp.Poly): return str(o.as_expr())
    if isinstance(o, (mpmath.mpf, mpmath.mpc)): return mpmath.nstr(o, 25)
    return o
if not PL:
    save('r2.json', clean(OUT))
log.close()
