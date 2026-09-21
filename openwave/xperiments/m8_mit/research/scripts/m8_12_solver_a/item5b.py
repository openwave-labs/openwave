"""Item 5b: tangent directions of the plane class at interior orbit points.

u = (a + z b)/sqrt(1+|z|^2);  tangent space of W at u inside T_u = {w in W : Re<u,w> = 0}, real dim 3,
with the Re<,>-orthonormal basis  t1 = i u,  t2 = u_perp,  t3 = i u_perp,  u_perp = (-conj(z) a + b)/sqrt(1+|z|^2).
H_u(e, f) = Hess N(u)[e, f] - 4 N(u) Re<e, f>, with Hess N(u)[e, f] = d^2/ds dr N(u + s e + r f) at 0.
"""
import sympy as sp, mpmath as mp
import common as C, classes as K, hess
from orbits import INTERIOR, INTERIOR_EXTRA

ck = C.Checker('item5b')
s_, r_ = sp.symbols('s r', real=True)
I = sp.I

def Re_ip(e, f):
    return sp.radsimp(sp.expand(sp.re(sum(sp.conjugate(p) * q for p, q in zip(e, f)))))

def hessN(u, e, f):
    w = [u[k] + s_ * e[k] + r_ * f[k] for k in range(7)]
    Np = sp.expand(C.N_exact(w))
    return sp.radsimp(sp.Poly(Np, s_, r_).coeff_monomial(s_ * r_))

def H(u, e, f, Nu):
    return sp.nsimplify(sp.radsimp(hessN(u, e, f) - 4 * Nu * Re_ip(e, f)))

def in_O_span(u, t):
    """exact least squares of t on the O_u generators; returns (coefficients, residual vector)."""
    O = [C.c_to_x(g).applyfunc(sp.radsimp) for g in C.orbit_gens(u)]
    A = sp.Matrix.hstack(*O)
    tx = C.c_to_x(t).applyfunc(sp.radsimp)
    G = (A.T * A).applyfunc(sp.radsimp)
    # O generators may be dependent (not here: all interior points have finite stabilizer)
    coef = G.LUsolve((A.T * tx).applyfunc(sp.radsimp)).applyfunc(sp.radsimp)
    resid = (tx - A * coef).applyfunc(sp.radsimp)
    return coef, resid

res = {}
todo = dict(INTERIOR); todo.update(INTERIOR_EXTRA)
labels = {'A*': 'A*', 'D*': 'D*', 'F*': 'F*', 'G*': 'G*', 'xyz': 'xyz in C (supplementary)', 'cat': 'cat in E (supplementary)',
          'xyz@G': 'xyz in G (supplementary)', 'cat@F': 'cat in F (supplementary)'}
for nm, (cls, z) in todo.items():
    d = K.CLASSES2[cls]
    a = list(d['a']); b = list(d['b'])
    nrm = sp.sqrt(1 + sp.nsimplify(z * sp.conjugate(z)))
    u = [sp.radsimp((a[k] + z * b[k]) / nrm) for k in range(7)]
    up = [sp.radsimp((-sp.conjugate(z) * a[k] + b[k]) / nrm) for k in range(7)]
    tang = {'t1 = i u': [I * v for v in u], 't2 = u_perp': up, 't3 = i u_perp': [I * v for v in up]}
    Nu = C.N_exact(u)
    ck.check('%s: u is a unit vector' % nm, sp.simplify(C.norm2(u) - 1) == 0)
    names = list(tang)
    gram_all = sp.Matrix(3, 3, lambda i, j: Re_ip(tang[names[i]], tang[names[j]]))
    ck.check('%s: t1,t2,t3 orthonormal and orthogonal to u' % nm,
             gram_all == sp.eye(3) and all(Re_ip(u, tang[n]) == 0 for n in names))
    out = {'class': cls, 'chart_z': str(z), 'u': str(u), 'u_perp': str(up), 'n_directions': 3, 'directions': {}}
    rest = []
    labels_O = ['i u', '-i Jx u', '-i Jy u', '-i Jz u']
    for n in names:
        t = tang[n]
        coef, resid = in_O_span(u, t)
        zero = all(v == 0 for v in resid)
        entry = {'projection_onto_N_u_is_zero': zero}
        if zero:
            combo = ' + '.join('(%s)*[%s]' % (cf, lb) for cf, lb in zip(coef, labels_O) if cf != 0)
            h = H(u, t, t, Nu)
            entry.update({'equals_O_direction': combo, 'H_u_on_unit_tangent_(orbit-null control)': str(h)})
            ck.check('%s: orbit-null control H_u(%s) == 0' % (nm, n), h == 0, str(h))
        else:
            rest.append(n)
            pn = sp.sqrt(sp.nsimplify(sp.radsimp((resid.T * resid)[0])))
            entry['norm_of_projection_onto_N_u'] = str(pn)
        out['directions'][n] = entry
    Hm = sp.Matrix(len(rest), len(rest), lambda i, j: H(u, tang[rest[i]], tang[rest[j]], Nu))
    Gm = sp.Matrix(len(rest), len(rest), lambda i, j: Re_ip(tang[rest[i]], tang[rest[j]]))
    out['nonnull_directions'] = rest
    out['H_u_matrix_on_unit_tangents'] = [[str(v) for v in row] for row in Hm.tolist()]
    out['Gram_matrix_Re'] = [[str(v) for v in row] for row in Gm.tolist()]
    # high-precision route: e^T M f with M from the CG-based polynomial at 50 digits
    mp.mp.dps = 50
    M, xu, _ = hess.mp_M(u, 50)
    def mx(v): return mp.matrix([mp.mpf(sp.re(q).evalf(60)) for q in v] + [mp.mpf(sp.im(q).evalf(60)) for q in v])
    dev = 0
    for i, ni in enumerate(rest):
        for j, nj in enumerate(rest):
            num = (mx(tang[ni]).T * M * mx(tang[nj]))[0]
            dev = max(dev, abs(num - mp.mpf(sp.N(Hm[i, j], 60))))
    out['mp50_max_deviation'] = mp.nstr(dev, 3)
    ck.check('%s: 50-digit matrix route agrees with exact H entries (dev %s)' % (nm, mp.nstr(dev, 3)), dev < mp.mpf(10) ** -45)
    # consistency with item 2: in the chart z -> (a + z b)/sqrt(1+|z|^2) the Fubini-Study factor is
    # 1/(1+|z|^2), dz real -> u_perp, dz imaginary -> i u_perp (mod O_u, which H_u annihilates), so the
    # chart Hessian of item 2 at a critical point must equal H_u(t2,t3 block)/(1+|z|^2)^2.
    x, y = K.x, K.y
    Nn, Dn = K.restriction(cls)
    fch = Nn / Dn
    zx, zy = sp.re(z), sp.im(z)
    Hc = sp.Matrix([[sp.diff(fch, x, 2), sp.diff(fch, x, y)], [sp.diff(fch, x, y), sp.diff(fch, y, 2)]]).subs({x: zx, y: zy}).applyfunc(sp.simplify)
    full = sp.Matrix(2, 2, lambda i, j: H(u, tang[names[1 + i]], tang[names[1 + j]], Nu))
    tt = sp.nsimplify(z * sp.conjugate(z))
    ok = (Hc - full / (1 + tt) ** 2).applyfunc(sp.simplify) == sp.zeros(2)
    ck.check('%s: chart Hessian of item 2 == H_u on (u_perp, i u_perp) / (1+|z|^2)^2' % nm, ok, str(Hc.tolist()))
    out['chart_hessian_item2'] = str(Hc.tolist())
    print('\n%s (class %s, z = %s)' % (labels[nm], cls, z))
    for n in names: print('  ', n, out['directions'][n])
    print('   H_u on', rest, '=', Hm.tolist(), '  Gram =', Gm.tolist())
    res[labels[nm]] = out
C.save('item5b', res)
print('item5b fails:', ck.fails)
