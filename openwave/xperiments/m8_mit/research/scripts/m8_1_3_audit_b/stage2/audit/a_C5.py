"""Audit of C.5 (solver_cb) at all five cases, by my own route (stage-1 method generalised to any point):
  exact (sympy, entries may contain square roots): criticality, charpoly of Shat = J0 Mhat -> zero multiplicity and
  slow polynomial;  numerical (mpmath, 50 and 70 digits): R0 gradient/Hessian by my CG route, D^3 h from my exact
  quartic, zhat, D_u, T1 with every i = J0 (clause 1), biorthonormalised eigenvectors, c_g or eig(C_ab).
Also an independent semisimplicity test (rank of (S - s)^2 vs (S - s)) and, for each group, the invariance of the value
under a random change of basis inside the right eigenspace (normalisation independence).
Argument: optional case filter (substring)."""
import sys, os, json, time, random
sys.dont_write_bytecode = True
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from fractions import Fraction as Fr
import sympy as sp
from mpmath import mp, matrix
from core import mlist, Fm
from reps import SLOTS
from hpoly import h_poly, PolyEval
from exact_pt import group_generators
from cgroute import CGRoute, CP
from chk import check, save_json

inp = json.load(open(os.path.join(here, '..', 'inputs.json')))
FILTER = sys.argv[1] if len(sys.argv) > 1 else ''
out = {}


def comps(case):
    pname, slot = case.rsplit(' in ', 1)
    src = inp['R2 points']['points'] if slot == 'R2' else inp['census']['points']
    return pname, slot, src[pname]


def exact_part(name, cm):
    d, n = SLOTS[name]; N = n + 1; ms = mlist(n); D = 2 * N
    w = [sp.Integer(0)] * N
    for k, v in cm.items():
        w[ms.index(Fr(k))] = sp.sympify(v, locals={'I': sp.I, 'sqrt': sp.sqrt})
    wn = sp.sqrt(sp.nsimplify(sum(sp.Abs(v) ** 2 for v in w)))
    p = [sp.nsimplify(w[i] / sp.sqrt(Fm(n, ms[i])) / wn) for i in range(N)]
    x = [sp.re(v) for v in p] + [sp.im(v) for v in p]
    he = PolyEval(h_poly(name), D)
    conv = lambda c: sp.Rational(c.numerator, c.denominator) if isinstance(c, Fr) else sp.Integer(c)
    hv = sp.radsimp(sp.expand(he.value(x, conv)))
    Gd = [Fm(n, m) for m in ms] * 2
    g = [sp.expand(v) for v in he.grad(x, conv)]
    crit = all(sp.simplify(g[i] - 4 * hv * Gd[i] * x[i]) == 0 for i in range(D))
    H = he.hess(x, conv)
    M = sp.Matrix(D, D, lambda a, b: sp.expand(H[a][b] / Gd[a] - (4 * hv if a == b else 0)))
    J0 = sp.zeros(D, D)
    for i in range(N):
        J0[N + i, i] = 1; J0[i, N + i] = -1
    S = J0 * M
    xi = sp.Symbol('xi'); nu = sp.Symbol('nu')
    t0 = time.time()
    cp = S.charpoly(xi)
    cs = [sp.radsimp(sp.expand(c)) for c in cp.all_coeffs()]
    z = 0
    while cs and cs[-1] == 0:
        cs.pop(); z += 1
    rational = all(c.is_Rational for c in cs)
    odd_zero = all(c == 0 for i, c in enumerate(reversed(cs)) if i % 2 == 1)
    qn = sum(c * nu ** (i // 2) for i, c in enumerate(reversed(cs)) if i % 2 == 0)
    slow = sp.factor(sp.Poly(qn, nu).primitive()[1].as_expr()) if rational else qn
    return dict(crit=crit, h=hv, zero=z, slow=slow, rational=rational, odd_zero=odd_zero, w=w, x=x, secs=time.time() - t0)


def numeric_part(name, w_exact, dps):
    mp.dps = dps
    d, n = SLOTS[name]; N = n + 1; ms = mlist(n); D = 2 * N
    Fv = [Fm(n, m) for m in ms] * 2
    sF = [mp.sqrt(f) for f in Fv]
    wn = mp.sqrt(sum(abs(mp.mpc(sp.N(sp.re(v), dps + 20), sp.N(sp.im(v), dps + 20))) ** 2 for v in w_exact))
    wv = [mp.mpc(sp.N(sp.re(v), dps + 20), sp.N(sp.im(v), dps + 20)) / wn for v in w_exact]
    uxi = matrix([mp.re(c) for c in wv] + [mp.im(c) for c in wv])
    u_x = [uxi[i] / sF[i] for i in range(D)]
    he = PolyEval(h_poly(name), D)
    conv = lambda c: mp.mpf(c.numerator) / c.denominator if isinstance(c, Fr) else mp.mpf(c)
    hval = he.value(u_x, conv)
    gx = he.grad(u_x, conv)
    crit_res = max(abs(gx[i] / sF[i] - 4 * hval * uxi[i]) for i in range(D))
    Hx = he.hess(u_x, conv)
    H = matrix(D, D)
    for i in range(D):
        for k in range(D):
            H[i, k] = Hx[i][k] / (sF[i] * sF[k])
    M = H - 4 * hval * mp.eye(D)
    J0 = matrix(D, D)
    for i in range(N):
        J0[N + i, i] = 1; J0[i, N + i] = -1
    S = J0 * M
    gens = group_generators(name)
    Kb = []
    for g in gens:
        v = matrix(D, 1)
        for i in range(D):
            v[i] = sum(mp.mpf(sp.Rational(g[i, kk]).p) / sp.Rational(g[i, kk]).q * u_x[kk] for kk in range(D)) * sF[i]
        for c in Kb:
            v = v - c * (c.T * v)[0]
        nv = mp.norm(v)
        if nv > mp.mpf(10) ** (-dps // 2):
            Kb.append(v / nv)
    k = len(Kb)
    sig_K = matrix(k, k)
    for a in range(k):
        for b in range(k):
            sig_K[a, b] = (Kb[a].T * J0 * Kb[b])[0]
    sv = mp.svd_r(sig_K, compute_uv=False) if k else []
    rank_sig = sum(1 for s in sv if s > mp.mpf(10) ** (-dps // 2))
    r_u = k - rank_sig
    full = [uxi] + Kb
    for i in range(D):
        e = matrix(D, 1); e[i] = 1
        v = e
        for c in full:
            v = v - c * (c.T * v)[0]
        if mp.norm(v) > mp.mpf(10) ** -20:
            full.append(v / mp.norm(v))
    Nb = full[1 + k:]
    MN = matrix(len(Nb), len(Nb))
    for a in range(len(Nb)):
        for b in range(len(Nb)):
            MN[a, b] = (Nb[a].T * M * Nb[b])[0]
    evN = mp.eigsy(MN)[0]
    tol = mp.mpf(10) ** (-dps // 3)
    sig = [sum(1 for e in evN if e < -tol), sum(1 for e in evN if abs(e) <= tol), sum(1 for e in evN if e > tol)]
    gapN = min(abs(e) for e in evN)
    cols = [uxi] + Nb
    Bp = matrix(D, len(cols))
    for a, c in enumerate(cols):
        for i in range(D):
            Bp[i, a] = c[i]
    cg = CGRoute(name, mp)
    z0 = CP(mp.mpf(0), mp.mpf(0))
    uvec = [CP(mp.re(c), mp.im(c)) for c in wv]
    basis = []
    for i in range(D):
        v = [z0] * N
        v[i % N] = CP(mp.mpf(1), mp.mpf(0)) if i < N else CP(mp.mpf(0), mp.mpf(1))
        basis.append(v)
    g, HR = cg.R0_grad_hess(uvec, basis)
    gR = matrix(g); HRm = matrix(HR)
    gK = max([abs((c.T * gR)[0]) for c in Kb] + [mp.mpf(0)])
    zc = mp.lu_solve(Bp.T * M * Bp, Bp.T * gR)
    zhat = Bp * zc
    zres = mp.norm(M * zhat - gR)
    z_x = [zhat[i] / sF[i] for i in range(D)]
    T3x = he.third(u_x, conv, z_x)
    T3 = matrix(D, D)
    for i in range(D):
        for kk in range(D):
            T3[i, kk] = T3x[i][kk] / (sF[i] * sF[kk])
    Du = (HRm - T3) / hval ** 2
    om0 = mp.sqrt(n * (n + 2)); chi = 1 / (8 * om0 * hval)
    svS = mp.svd_r(S, compute_uv=False)
    kerdim = sum(1 for x in svS if x < mp.mpf(10) ** (-dps // 2))
    kergap = (max([x for x in svS if x < mp.mpf(10) ** (-dps // 2)] + [mp.mpf(0)]), min([x for x in svS if x >= mp.mpf(10) ** (-dps // 2)]))
    ev, Rv = mp.eig(S)
    evL, Lv = mp.eig(S.T)
    ztol = mp.mpf(10) ** (-dps // 5)
    nz = [i for i in range(D) if abs(ev[i]) > ztol]
    groups = []
    for i in nz:
        for gp in groups:
            if abs(ev[i] - ev[gp[0]]) < mp.mpf(10) ** (-dps // 2):
                gp.append(i); break
        else:
            groups.append([i])
    res = []
    rng = random.Random(7)
    for gp in groups:
        m = len(gp)
        s = sum(ev[i] for i in gp) / m
        Lidx = [i for i in range(D) if abs(evL[i] - s) < mp.mpf(10) ** (-dps // 2)]
        A1 = S - s * mp.eye(D)
        sv1 = mp.svd_c(A1, compute_uv=False); sv2 = mp.svd_c(A1 * A1, compute_uv=False)
        tl = mp.mpf(10) ** (-dps // 3)
        geo = sum(1 for x in sv1 if abs(x) < tl); alg2 = sum(1 for x in sv2 if abs(x) < tl)
        Rm = matrix(D, m); Lm = matrix(D, m)
        for a, i in enumerate(gp):
            for t in range(D):
                Rm[t, a] = Rv[t, i]
        for a, i in enumerate(Lidx[:m]):
            for t in range(D):
                Lm[t, a] = Lv[t, i]
        Lm = Lm * mp.inverse(Lm.T * Rm).T
        bi = mp.norm(Lm.T * Rm - mp.eye(m))
        mu0 = chi * s
        T1 = (1 / (4 * om0)) * (J0 * Du) - (mu0 / (2 * om0 ** 2)) * mp.eye(D) + (mu0 ** 2 / (2 * om0)) * J0
        C = Lm.T * T1 * Rm
        vals = sorted(mp.eig(C)[0] if m > 1 else [C[0, 0]], key=lambda z: (mp.re(z), mp.im(z)))
        # normalisation independence: random complex change of basis of the right eigenspace
        P = matrix(m, m)
        for a in range(m):
            for b in range(m):
                P[a, b] = mp.mpc(rng.uniform(-1, 1), rng.uniform(-1, 1)) + (3 if a == b else 0)
        R2 = Rm * P
        L2 = Lm * mp.inverse(Lm.T * R2).T
        C2 = L2.T * T1 * R2
        vals2 = sorted(mp.eig(C2)[0] if m > 1 else [C2[0, 0]], key=lambda z: (mp.re(z), mp.im(z)))
        dv = max(abs(a - b) for a, b in zip(vals, vals2))
        res.append(dict(s=s, m=m, nL=len(Lidx), geo=geo, alg2=alg2, vals=vals, bi=bi, basis_change=dv,
                        eigres=mp.norm(S * Rm - s * Rm)))
    res.sort(key=lambda t: (float(mp.im(t['s'])) if abs(mp.im(t['s'])) > 1e-30 else 0.0, float(mp.re(t['s'])) if abs(mp.re(t['s'])) > 1e-30 else 0.0))
    return dict(h=hval, crit=crit_res, k=k, r_u=r_u, sig=sig, gapN=gapN, gK=gK, zres=zres, zero=D - len(nz), groups=res, kerdim=kerdim, kergap=kergap)


for case in inp['five cases']:
    if FILTER and FILTER not in case:
        continue
    pname, name, cm = comps(case)
    t0 = time.time()
    E = exact_part(name, cm)
    print(f'\n===== {case}: exact part ({E["secs"]:.0f}s charpoly): critical {E["crit"]}, h = {E["h"]}, zero multiplicity {E["zero"]}, '
          f'charpoly rational {E["rational"]}, odd coefficients zero {E["odd_zero"]}', flush=True)
    print(f'   slow polynomial: {E["slow"]}', flush=True)
    check(f'C.5 {case}: critical (exact)', E['crit'])
    A = numeric_part(name, E['w'], 50)
    B = numeric_part(name, E['w'], 70)
    print(f'   numeric: k = {A["k"]}, r_u = {A["r_u"]}, signature {A["sig"]} (min |eig| on N_u {mp.nstr(A["gapN"], 5)}), '
          f'|grad R0 . K| = {mp.nstr(A["gK"], 3)}, zhat residual {mp.nstr(A["zres"], 3)}, numerical zero count {A["zero"]} / {B["zero"]}; dim ker Shat (geometric) = {A["kerdim"]} (sv gap {mp.nstr(A["kergap"][0], 3)} vs {mp.nstr(A["kergap"][1], 5)})')
    check(f'C.5 {case}: numerical zero count (50 and 70 digits) == exact zero multiplicity', A['zero'] == B['zero'] == E['zero'], f'{A["zero"]}, {B["zero"]}, {E["zero"]}')
    check(f'C.5 {case}: grad R0 orthogonal to K_u and zhat solves the system', A['gK'] < mp.mpf(10) ** -40 and A['zres'] < mp.mpf(10) ** -40)
    rec = {'h(u)': str(E['h']), 'zero multiplicity (exact)': E['zero'], 'slow polynomial': str(E['slow']),
           'k': A['k'], 'r_u': A['r_u'], 'signature': A['sig'], 'dim ker Shat (geometric multiplicity of 0)': A['kerdim'], 'groups': []}
    nu = sp.Symbol('nu')
    for ga, gb in zip(A['groups'], B['groups']):
        diff = max(abs(a - b) / abs(b) for a, b in zip(ga['vals'], gb['vals']))
        sd = abs(ga['s'] - gb['s'])
        ok_ss = ga['geo'] == ga['m'] == ga['alg2'] == ga['nL']
        check(f'C.5 {case}: s_g = {mp.nstr(gb["s"], 12)}: multiplicity {ga["m"]}, semisimple (geometric = algebraic-from-(S-s)^2 = left count)', ok_ss,
              f'geo {ga["geo"]}, ker (S-s)^2 {ga["alg2"]}, left {ga["nL"]}')
        check(f'C.5 {case}: s_g = {mp.nstr(gb["s"], 12)}: values agree 50 vs 70 digits to 1e-30 (relative), basis-change invariant',
              diff < mp.mpf(10) ** -30 and ga['basis_change'] < mp.mpf(10) ** -35, f'{mp.nstr(diff, 3)}, {mp.nstr(ga["basis_change"], 3)}')
        if E['rational'] and E['slow'] != 1:
            r2 = gb['s'] ** 2
            pv = sp.Poly(E['slow'], nu)
            val = sum(mp.mpf(int(c.p)) / int(c.q) * r2 ** k for k, c in enumerate(reversed(pv.all_coeffs())))
            check(f'C.5 {case}: s_g^2 is a root of the exact slow polynomial', abs(val) < mp.mpf(10) ** -40, mp.nstr(abs(val), 3))
        print(f'   s_g = {mp.nstr(gb["s"], 25)}  mult {ga["m"]}:  values ' + ', '.join(mp.nstr(v, 22) for v in gb['vals']))
        rec['groups'].append({'s_g': mp.nstr(gb['s'], 30), 'multiplicity': ga['m'], 'values': [mp.nstr(v, 25) for v in gb['vals']],
                              '50 vs 70 rel diff': mp.nstr(diff, 3)})
    out[case] = rec
save_json('a_C5' + ('_' + FILTER.replace(' ', '_') if FILTER else '') + '.json', out)
