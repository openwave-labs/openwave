"""Extra attacks.
(1) 924 rhat_6 at all ten census points by a second route, independent of the Fischer identity used in a02: the
    spin-6 component of w (x) Theta(w) through my exact Casimir projector Pi_6 on V_3 (x) V_3 (own/lib_rep), at 50 digits;
    compared with the exact Fischer value of a02 (and l6) to 1e-40.
(2) The reading 'Krein sign only for negative nu': on the hyperbolic clusters (positive nu, complex quartets) the real
    invariant subspace is ker(S^2 - nu) or ker((S^2 - nu)(S^2 - conj nu)); the form M_hat restricted there is computed at
    60 digits, to say what an alternative reading would report.
'plant': uses Pi_4 in place of Pi_6 in (1); the comparison must FAIL."""
from au import *
from lib_rep import casimir_projectors_w, svec
from lib_crit import h_poly, Point, tonum

log = Log(__file__, '_plant' if planted() else '')
PL = planted()
cens = inputs()['census']['points']
L6 = solver('solver_l6')
mine = load('census.json')
proj = casimir_projectors_w(6)
J2 = 8 if PL else 12
mpmath.mp.dps = 50
ml = ms(6); n = 7; sv = svec(6)
for pname, comps in cens.items():
    w = [sp.Integer(0)] * n
    for k, v in comps.items():
        w[ml.index(int(2 * sp.Rational(k)))] = sp.sympify(v, locals={'I': sp.I, 'sqrt': sp.sqrt})
    nrm2 = sum(sp.expand(x * sp.conjugate(x)) for x in w)
    p = [mpmath.mpc(sp.N(sp.re(x / sp.sqrt(nrm2)), 60), sp.N(sp.im(x / sp.sqrt(nrm2)), 60)) / mpmath.sqrt(s) for x, s in zip(w, sv)]
    t = [mpmath.conj(p[ml.index(-m2)]) * (-1) ** ((6 - m2) // 2) for m2 in ml]
    vv = [p[a] * t[b] for a in range(n) for b in range(n)]
    y = {}
    for (r, c), x in proj[J2].items():
        y[r] = y.get(r, 0) + vv[c] * mpmath.mpf(x.numerator) / x.denominator
    val = 924 * sum(sv[r // n] * sv[r % n] * abs(yr) ** 2 for r, yr in y.items())
    ex = sym(mine[f'R4|{pname}']['924 r6hat'])
    ok = abs(val - mpmath.mpf(sp.N(ex, 60))) < mpmath.mpf(10) ** -40 and sym(L6['B.1']['R4'][pname]['924 r6hat']) == ex
    log(f'  {pname}: 924 rhat_6 Casimir route {mpmath.nstr(val, 30)}; Fischer exact {ex}')
    log.check(f'924 rhat_6 {pname}: Casimir route = exact Fischer value (1e-40) = l6', ok)

if not PL:
    hd = {s: h_poly(s) for s in ('R4', 'R5')}
    for pname in ('D2 ray', 'octahedron', 'zonal v0', 'C3 ray', 'pentagonal pyramid'):
        for slot in ('R4',):
            P = Point(slot, cens[pname], hd[slot]).compute()
            mpmath.mp.dps = 60
            Mn = tonum(P.Mw, 60); S = tonum(P.Jc(), 60) * Mn
            N = S.rows
            ev = mpmath.eig(S, left=False, right=False)
            nus = []
            for e in ev:
                z = e * e
                if abs(e) > 1e-20 and (z.real > 1e-20 or abs(z.imag) > 1e-20):
                    if all(abs(z - q) > 1e-20 and abs(z - mpmath.conj(q)) > 1e-20 for q in nus):
                        nus.append(z)
            for z in nus:
                if abs(z.imag) < 1e-20:
                    A = S * S - z.real * mpmath.eye(N)
                    kind = f'positive nu = {mpmath.nstr(z.real, 12)}'
                else:
                    A = (S * S - z * mpmath.eye(N)) * (S * S - mpmath.conj(z) * mpmath.eye(N))
                    A = A.apply(mpmath.re)
                    kind = f'quartet nu = {mpmath.nstr(z, 12)}'
                U, s_, V = mpmath.svd_r(A)
                idx = [i for i in range(N) if s_[i] < mpmath.mpf(10) ** -25]
                B = mpmath.matrix(N, len(idx))
                for c, i in enumerate(idx):
                    for k in range(N): B[k, c] = V[i, k]
                e2 = mpmath.eigsy(B.T * Mn * B)[0]
                e2 = [e2[i] for i in range(len(idx))]
                npos = sum(1 for x in e2 if x > 1e-20); nneg = sum(1 for x in e2 if x < -1e-20)
                log(f'  {slot} {pname} {kind}: dim {len(idx)}, M_hat restricted: {npos} positive, {nneg} negative -> '
                    f'{"indefinite" if npos and nneg else "definite"}')
                log.check(f'{slot} {pname} {kind}: the hyperbolic cluster carries an indefinite form (alternative reading would report "indefinite")',
                          npos > 0 and nneg > 0)
log.close()
