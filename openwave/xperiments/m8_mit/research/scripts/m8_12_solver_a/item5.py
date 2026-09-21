"""Items 5, 6, 7: restricted Hessian at each orbit; Morse indices; kernel identification.

Exact route: a-coordinates (hess.py), DomainMatrix over Q or Q(sqrt d).
High-precision route: M_u = Hess N(u) - 4 N(u) I in the original c-coordinates from the
Clebsch-Gordan-built polynomial N, restricted to a numerically orthonormalized N_u, eigenvalues
by mpmath at 50 and 80 digits, compared with the roots of the exact characteristic polynomial.
"""
import sympy as sp, mpmath as mp
import common as C, hess
from orbits import ORBITS

ck = C.Checker('item5')
lam = sp.Symbol('lam')
res = {}

# ---- the a-coordinate polynomial is the same function as N (exact polynomial identity)
Nt, _, _ = hess.Nt_and_hess()
S = [sp.sqrt(hess.B6[m]) for m in C.MS]
sub = {hess.YS[k]: S[k] * C.XS[k] for k in range(7)}
sub.update({hess.YS[k + 7]: S[k] * C.XS[k + 7] for k in range(7)})
ident = sp.expand(Nt.subs(sub) - C.N_poly())
ck.check('Nt(S x) == N(x) as polynomials (a-coordinates are exact)', ident == 0)
ck.check('Nt has rational coefficients', all(cf.is_Rational for cf in sp.Poly(Nt, *hess.YS).coeffs()))

def mp_restricted_eigs(c, dps):
    M, xu, Nv = hess.mp_M(c, dps)
    cv = [xu[k] + 1j * xu[k + 7] for k in range(7)]
    # O_u generators numerically
    gens = []
    for G in [None, C.JX, C.JY, C.JZ]:
        if G is None: w = [1j * z for z in cv]
        else:
            Gm = [[C.to_mp(G[a, b]) for b in range(7)] for a in range(7)]
            w = [-1j * sum(Gm[a][b] * cv[b] for b in range(7)) for a in range(7)]
        gens.append([mp.re(z) for z in w] + [mp.im(z) for z in w])
    A = mp.matrix([list(xu)] + gens)            # 5 x 14
    U, Sv, V = mp.svd_r(A, full_matrices=True)
    rank = sum(1 for s in Sv if s > mp.mpf(10) ** (-dps // 2))
    Bn = mp.matrix(14, 14 - rank)
    for j in range(14 - rank):
        for i in range(14):
            Bn[i, j] = V[rank + j, i]
    K = Bn.T * M * Bn
    K = (K + K.T) / 2
    ev = mp.eigsy(K, eigvals_only=True)
    ev = sorted([ev[i] for i in range(len(ev))])
    # residual of M on O_u (numerical)
    res_o = max(mp.norm(M * mp.matrix(g)) for g in gens)
    return ev, rank - 1, res_o

for nm, d in ORBITS.items():
    c = list(d['u'])
    a = hess.a_from_c(c)
    ex = hess.exact_data(a, lam)
    p = ex['charpoly']
    sig = hess.signature(p)
    fac = sp.factor(p.as_expr())
    ck.check('%s: rhat6 from a-coordinates equals item-4 value' % nm, sp.nsimplify(ex['rhat'] - C.rhat_exact(c)) == 0)
    ck.check('%s: M_u annihilates all four generators of O_u exactly' % nm, all(r == 0 for r in ex['resid']), str(ex['resid']))
    ck.check('%s: dim N_u = 13 - dim O_u' % nm, ex['dimN'] == 13 - ex['dimO'])
    roots = []
    for fac_, mult_ in sp.factor_list(sp.Poly(p.as_expr(), lam))[1]:
        for r_ in sp.real_roots(sp.Poly(fac_, lam)):
            roots += [r_.evalf(100)] * mult_
    roots = sorted(roots)
    out = {'dim_O': ex['dimO'], 'dim_N': ex['dimN'], 'signature_(n-,n0,n+)': list(sig),
           'charpoly_monic_factored': str(fac), 'exact_field': ex['field'],
           'O_u_residuals_exact': [str(r) for r in ex['resid']]}
    for dps in (50, 80):
        ev, dimO_num, res_o = mp_restricted_eigs(c, dps)
        dev = max(abs(ev[k] - mp.mpf(str(sp.re(roots[k])))) for k in range(len(ev)))
        out['mp%d' % dps] = {'max_|eig - exact root|': mp.nstr(dev, 3), 'O_u_residual': mp.nstr(res_o, 3), 'dim_O': dimO_num}
        ck.check('%s: %d-digit eigenvalues match exact char-poly roots (dev %s)' % (nm, dps, mp.nstr(dev, 3)), dev < mp.mpf(10) ** (-dps + 15) and len(ev) == ex['dimN'])
    print('%-4s dimO=%d dimN=%d sig=%s  charpoly = %s' % (nm, ex['dimO'], ex['dimN'], sig, fac))
    print('      eigenvalues ~ %s' % [sp.N(sp.re(r), 8) for r in roots])
    # item 6
    out['item6'] = {'index_g>0': sig[0], 'index_g<0': sig[2], 'kernel': sig[1],
                    'local_min_of_g*rhat6_g>0': sig[0] == 0 and sig[1] == 0,
                    'local_min_of_g*rhat6_g<0': sig[2] == 0 and sig[1] == 0}
    # item 7: kernel identification
    if sig[1] > 0:
        from sympy.polys.matrices import DomainMatrix
        B = ex['B']; Qy = ex['Qy']
        Kf = (B.T * Qy * B).applyfunc(sp.radsimp)
        kern = Kf.nullspace()
        vecs = [(B * v).applyfunc(sp.radsimp) for v in kern]     # y-coordinates
        # back to c-coordinates: c_m = a_m / s_m
        cvecs = []
        for v in vecs:
            av = [v[k] + sp.I * v[k + 7] for k in range(7)]
            cvecs.append([sp.radsimp(av[k] / S[k]) for k in range(7)])
        out['item7_kernel_vectors_c_coords'] = [str(cv) for cv in cvecs]
        print('      kernel (c-coords):', cvecs)
        # structural test: kernel == tangent of class B at v1, i.e. span_R{v-3, i v-3}
        target = [sp.Matrix([1 if k == C.idx(-3) else 0 for k in range(7)] + [0] * 7),
                  sp.Matrix([0] * 7 + [1 if k == C.idx(-3) else 0 for k in range(7)])]
        kx = [C.c_to_x(cv) for cv in cvecs]
        ok = C.same_span(sp.Matrix.hstack(*kx), sp.Matrix.hstack(*target))
        ck.check('%s: kernel equals span_R{v-3, i v-3} = tangent of class B at v1' % nm, ok)
        out['item7_kernel_is_tangent_of_class_B'] = ok
    res[nm] = out

# ---- where the O_u check fails: a non-critical point (the item-11 point)
u11 = [sp.Integer(0)] * 7; u11[C.idx(3)] = 1; u11[C.idx(1)] = 1; u11[C.idx(-2)] = -1
u11 = [z / sp.sqrt(3) for z in u11]
Mc, Nv, gc, xv = hess.M_c(u11)
resid = [sp.sqrt(sp.nsimplify(sp.radsimp(sp.expand(((Mc * C.c_to_x(o)).T * (Mc * C.c_to_x(o)))[0])))) for o in C.orbit_gens(u11)]
ck.check('control (v3+v1-v-2)/sqrt3: M_u does NOT annihilate O_u (the check can fail)', any(r != 0 for r in resid), str(resid))
res['control_noncritical_O_residuals'] = [str(r) for r in resid]
C.save('item5', res)
print('item5 fails:', ck.fails)
