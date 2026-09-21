"""Item 11: the non-critical point u = (v3 + v1 - v-2)/sqrt3."""
import sympy as sp, mpmath as mp
import common as C, hess

ck = C.Checker('item11')
u = [sp.Integer(0)] * 7; u[C.idx(3)] = 1; u[C.idx(1)] = 1; u[C.idx(-2)] = -1
u = [z / sp.sqrt(3) for z in u]
res = {}
val = C.rhat_exact(u)
res['924*rhat6'] = str(924 * val); res['rhat6'] = str(val)
M, Nv, gN, xv = hess.M_c(u)
ck.check('N(u) from the polynomial equals rhat6 from rho_6', sp.nsimplify(Nv - val) == 0)
g = (gN - 4 * Nv * xv).applyfunc(sp.radsimp)          # tangential gradient (see RETURN.md)
ck.check('gradient is tangent: Re<u, g> = 0', sp.simplify((xv.T * g)[0]) == 0)
gnorm = sp.sqrt(sp.nsimplify(sp.radsimp((g.T * g)[0])))
res['tangential_gradient_norm'] = str(gnorm)
res['tangential_gradient_norm_minpoly'] = str(sp.minimal_polynomial(gnorm, sp.Symbol('X')))
print('924*rhat6 =', 924 * val, '  rhat6 =', val)
print('|tangential gradient| =', gnorm, '=', sp.N(gnorm, 20))
labels = ['i u', '-i Jx u', '-i Jy u', '-i Jz u']
gens = C.orbit_gens(u)
gc = [g[k] + sp.I * g[k + 7] for k in range(7)]
res['||M_u d||'] = {}
for lb, d, X in zip(labels, gens, [None, C.JX, C.JY, C.JZ]):
    w = (M * C.c_to_x(d)).applyfunc(sp.radsimp)
    nrm = sp.sqrt(sp.nsimplify(sp.radsimp(sp.expand((w.T * w)[0]))))
    nrm = sp.radsimp(sp.sqrtdenest(nrm))
    # identity: M_u (X u) = X g for a skew generator X (derivative of invariance), so ||M_u X u|| = ||X g||
    Xg = [sp.I * z for z in gc] if X is None else [sp.expand(-sp.I * z) for z in C.apply(X, gc)]
    alt = sp.sqrt(sp.nsimplify(sp.radsimp(C.norm2(Xg))))
    ck.check('||M_u (%s)|| equals ||X g|| (generator applied to the gradient)' % lb, sp.simplify(nrm - alt) == 0)
    ck.check('||M_u (%s)|| is nonzero' % lb, nrm != 0)
    mp.mp.dps = 50
    Mm, xm, _ = hess.mp_M(u, 50)
    dm = mp.matrix([mp.mpf(sp.re(q).evalf(60)) for q in d] + [mp.mpf(sp.im(q).evalf(60)) for q in d])
    nm_ = mp.norm(Mm * dm)
    ck.check('||M_u (%s)|| 50-digit agrees' % lb, abs(nm_ - mp.mpf(sp.N(nrm, 60))) < mp.mpf(10) ** -45)
    mpoly = sp.minimal_polynomial(nrm, sp.Symbol('X'))
    print('   minimal polynomial:', mpoly)
    res['||M_u d||'][lb] = {'minimal_polynomial': str(mpoly), 'exact': str(nrm), 'numeric': mp.nstr(nm_, 25), 'squared': str(sp.nsimplify(sp.radsimp(sp.expand(nrm ** 2))))}
    print('||M_u (%s)|| = %s = %s   (squared: %s)' % (lb, nrm, mp.nstr(nm_, 20), sp.nsimplify(sp.radsimp(sp.expand(nrm ** 2)))))
C.save('item11', res)
print('item11 fails:', ck.fails)
