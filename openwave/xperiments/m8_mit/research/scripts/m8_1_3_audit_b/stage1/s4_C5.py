"""Item C.5 at 'coherent v3 in R4' and 'weight 7/2 in R2' (clauses 1 and 4).
Orthonormal real coordinates xi = (Re w_m, Im w_m) (metric = identity), c = 1, V = R = 1.
  Mhat = Hess h - 4h, Shat = J0 Mhat (J0 = multiplication by i),  zhat in R u + N_u with Mhat zhat = grad R0(u),
  D_u = kappa^-2 (Hess R0(u) - D^3 h(u)[zhat]),  omega0 = sqrt(n(n+2)),  chi = 1/(8 omega0 Q(u)),  mu0 = chi s_g,
  T1 = (1/(4 omega0)) J0 D_u - (1/(2 omega0^2)) mu0 + (mu0^2/(2 omega0)) J0   (clause 1: every i in T1 is J0),
  c_g = l^T T1 r (l^T r = 1), or the eigenvalues of C_ab = l_a^T T1 r_b for a semisimple s_g of multiplicity m.
h, its derivatives: exact polynomial; R0 gradient/Hessian: CG route at 50 digits, spot-checked against the exact S^3 route.
Zero multiplicity and the slow polynomial: exact (sympy charpoly)."""
from fractions import Fraction as Fr
import sympy as sp
from mpmath import mp, matrix
from core import mlist, Fm, KZERO
from reps import SLOTS
from hpoly import h_poly, PolyEval
from exact_pt import ExactPoint, group_generators
from s3poly import R0_hessian_entry, kfrac
from cgroute import CGRoute, CP
from chk import check, save_json, planted

mp.dps = 50
out = {'C.5': {'R2': {}, 'R4': {}, 'R5': {}}}


def fmp(x):
    if isinstance(x, Fr):
        return mp.mpf(x.numerator) / x.denominator
    x = sp.Rational(x)
    return mp.mpf(int(x.p)) / int(x.q)


def run(name, label, top_index):
    d, n = SLOTS[name]; N = n + 1; D = 2 * N; ms = mlist(n)
    Fv = [Fm(n, m) for m in ms] * 2
    sF = [mp.sqrt(f) for f in Fv]
    # exact data
    x0 = [0] * D; x0[top_index] = 1
    E = ExactPoint(name, x0)
    check(f'C.5 {label}: critical (exact)', E.critical)
    z, slow = E.slow()
    sig = E.signature()
    print(f'{label}: h(u) = {E.hu}; k = {E.k}; signature {sig}; zero multiplicity {z}; slow polynomial {slow}')
    # numeric objects in xi coordinates
    he = PolyEval(h_poly(name), D)
    u_x = [mp.mpf(0)] * D; u_x[top_index] = 1 / sF[top_index]          # unit vector, chart coordinates
    conv = fmp
    hval = he.value(u_x, conv)
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
    uxi = matrix([mp.mpf(0)] * D); uxi[top_index] = 1
    # K_u from the generators (transformed to xi coordinates), orthonormalised
    gens = group_generators(name)
    Kcols = []
    for g in gens:
        v = matrix(D, 1)
        for i in range(D):
            v[i] = sum(fmp(g[i, kk]) * u_x[kk] for kk in range(D)) * sF[i]
        Kcols.append(v)
    # orthonormal basis of K_u (Gram-Schmidt with rank detection)
    Kb = []
    for v in Kcols:
        for c in Kb:
            v = v - c * (c.T * v)[0]
        nv = mp.norm(v)
        if nv > mp.mpf(10) ** -30:
            Kb.append(v / nv)
    check(f'C.5 {label}: numerical k equals exact k', len(Kb) == E.k, f'{len(Kb)} vs {E.k}')
    # basis of K_u-perp = R u + N_u
    full = [uxi] + Kb
    for i in range(D):
        e = matrix(D, 1); e[i] = 1
        v = e
        for c in full:
            v = v - c * (c.T * v)[0]
        nv = mp.norm(v)
        if nv > mp.mpf(10) ** -20:
            full.append(v / nv)
    assert len(full) == D
    Bperp = matrix(D, D - len(Kb))
    cols = [full[0]] + full[1 + len(Kb):]
    for a, c in enumerate(cols):
        for i in range(D):
            Bperp[i, a] = c[i]
    # R0 gradient and Hessian (CG route), directions = xi unit vectors
    cg = CGRoute(name, mp)
    z0 = CP(mp.mpf(0), mp.mpf(0))
    uvec = [z0] * N; uvec[top_index % N] = CP(mp.mpf(1), mp.mpf(0))
    basis = []
    for i in range(D):
        v = [z0] * N
        v[i % N] = CP(mp.mpf(1), mp.mpf(0)) if i < N else CP(mp.mpf(0), mp.mpf(1))
        basis.append(v)
    g, HR = cg.R0_grad_hess(uvec, basis)
    gR = matrix(g); HRm = matrix(HR)
    # spot-check against the exact S^3 Hessian (chart directions e_i, e_k): Hess_xi = Hess_x / (sF_i sF_k)
    pu = [KZERO] * N; pu[top_index % N] = kfrac(1)
    worst = mp.mpf(0)
    rF = Fm(n, ms[top_index % N])
    for (i, k) in [(top_index, top_index), (1, 1), (N + 1, N + 1), (2, N + 2), (3, 3), (N - 1, N - 1), (1, N + 1)]:
        pi = [KZERO] * N; pk = [KZERO] * N
        pi[i % N] = kfrac(1) if i < N else kfrac(0, 1)
        pk[k % N] = kfrac(1) if k < N else kfrac(0, 1)
        hx, h5 = R0_hessian_entry(name, pu, pi, pk)
        # pu has |w|^2 = F_top; scale to unit u: Hess(u) = Hess(x0) / F_top^2
        val = fmp(hx) / rF ** 2 / (sF[i] * sF[k])
        worst = max(worst, abs(val - HRm[i, k]))
    if planted('C5'):
        worst += mp.mpf(10) ** -20
    check(f'C.5 {label}: CG-route Hessian of R0 agrees with exact S^3 entries', worst < mp.mpf(10) ** -35, mp.nstr(worst, 3))
    # grad R0 orthogonal to K_u
    gk = max(abs((c.T * gR)[0]) for c in Kb)
    check(f'C.5 {label}: grad R0(u) is orthogonal to K_u', gk < mp.mpf(10) ** -40, mp.nstr(gk, 3))
    # zhat
    Mr = Bperp.T * M * Bperp
    zc = mp.lu_solve(Mr, Bperp.T * gR)
    zhat = Bperp * zc
    resid = mp.norm(M * zhat - gR)
    check(f'C.5 {label}: Mhat zhat = grad R0 holds (nondegenerate on R u + N_u)', resid < mp.mpf(10) ** -40, mp.nstr(resid, 3))
    # D^3 h(u)[zhat] in xi coordinates
    z_x = [zhat[i] / sF[i] for i in range(D)]
    T3x = he.third(u_x, conv, z_x)
    T3 = matrix(D, D)
    for i in range(D):
        for k in range(D):
            T3[i, k] = T3x[i][k] / (sF[i] * sF[k])
    kappa = hval
    Du = (HRm - T3) / kappa ** 2
    om0 = mp.sqrt(n * (n + 2))
    Qu = hval
    chi = 1 / (8 * om0 * Qu)
    # eigen-decomposition of Shat
    ev, Rv = mp.eig(S)
    evL, Lv = mp.eig(S.T)
    tolz = mp.mpf(10) ** -10   # nonzero slow eigenvalues are O(0.1); a Jordan block at 0 splits only to ~1e-25 at 50 digits
    nz = [i for i in range(D) if abs(ev[i]) > tolz]
    zero_num = D - len(nz)
    check(f'C.5 {label}: numerical zero count equals exact zero multiplicity', zero_num == z, f'{zero_num} vs {z}')
    # group nonzero eigenvalues
    groups = []
    for i in nz:
        for gp in groups:
            if abs(ev[i] - ev[gp[0]]) < mp.mpf(10) ** -25:
                gp.append(i); break
        else:
            groups.append([i])
    # exact roots of the slow polynomial: s_g^2 must be roots
    nu = sp.Symbol('nu')
    roots = sp.Poly(slow, nu).all_roots() if slow != 1 else []
    res = []
    for gp in groups:
        s = ev[gp[0]]
        m = len(gp)
        # right eigenvectors: columns of Rv for gp; left: from S^T eigen with matching eigenvalue
        Lidx = [i for i in range(D) if abs(evL[i] - s) < mp.mpf(10) ** -25]
        check(f'C.5 {label}: s = {mp.nstr(s, 12)} has matching left/right eigenspaces of size {m}', len(Lidx) == m)
        Rm = matrix(D, m); Lm = matrix(D, m)
        for a, i in enumerate(gp):
            for t in range(D):
                Rm[t, a] = Rv[t, i]
        for a, i in enumerate(Lidx):
            for t in range(D):
                Lm[t, a] = Lv[t, i]
        # semisimplicity: rank of the right eigenvector block = m, and S Rm = s Rm
        semis = mp.norm(S * Rm - s * Rm) < mp.mpf(10) ** -35
        Cmat = Lm.T * Rm
        Lm = Lm * mp.inverse(Cmat).T      # now Lm^T Rm = I
        bi = mp.norm(Lm.T * Rm - mp.eye(m))
        check(f'C.5 {label}: s = {mp.nstr(s, 12)}: eigenpair residual small and l^T r = I after normalisation', semis and bi < mp.mpf(10) ** -35)
        mu0 = chi * s
        T1 = (1 / (4 * om0)) * (J0 * Du) - (mu0 / (2 * om0 ** 2)) * mp.eye(D) + (mu0 ** 2 / (2 * om0)) * J0
        C = Lm.T * T1 * Rm
        cev = mp.eig(C)[0] if m > 1 else [C[0, 0]]
        # exact s^2 identification
        s2 = s * s
        hit = [r for r in roots if abs(complex(sp.N(r, 30)) - complex(s2)) < 1e-20]
        check(f'C.5 {label}: s^2 = {mp.nstr(s2, 15)} is a root of the exact slow polynomial', len(hit) == 1)
        res.append((s, m, cev, hit[0] if hit else None))
    # sort by imaginary part of s
    res.sort(key=lambda t: (float(mp.im(t[0])), float(mp.re(t[0]))))
    od = {'h(u) = kappa = Q(u)': str(E.hu), 'k': E.k, 'signature': sig, 'zero multiplicity': z,
          'slow polynomial': str(slow), 'groups': []}
    for (s, m, cev, r2) in res:
        sexact = f'+sqrt({r2})' if True else ''
        print(f'   s_g = {mp.nstr(s, 25)}  (s_g^2 = {r2}), multiplicity {m}:')
        for c in cev:
            print(f'       value {mp.nstr(c, 20)}   [|Re| = {mp.nstr(abs(mp.re(c)), 3)}]')
        od['groups'].append({'s_g': mp.nstr(s, 30), 's_g^2 (exact root of the slow polynomial)': str(r2), 'multiplicity': m,
                             'values': [mp.nstr(c, 20) for c in cev], 'values Re': [mp.nstr(mp.re(c), 20) for c in cev], 'values Im': [mp.nstr(mp.im(c), 20) for c in cev]})
    print(f'   kappa = {mp.nstr(kappa, 20)}, omega0 = {mp.nstr(om0, 20)}, chi = {mp.nstr(chi, 20)}')
    od['kappa'] = mp.nstr(kappa, 30); od['omega0'] = mp.nstr(om0, 30); od['chi'] = mp.nstr(chi, 30)
    return od


ms6 = mlist(6); ms7 = mlist(7)
out['C.5']['R4']['coherent v3'] = run('R4', 'coherent v3 in R4', ms6.index(3))
out['C.5']['R2']['weight 7/2'] = run('R2', 'weight 7/2 in R2', ms7.index(Fr(7, 2)))
save_json('out_C5.json', out)
