"""Exact Hessian machinery (items 5, 5b, 7, 11, 12).

H_u(e, e) = e^T M_u e with M_u = Hess N(u) - 4 N(u) I  (u unit, e in T_u; proof in RETURN.md).

Exact route ("a-coordinates"): put a_m = s_m c_m with s_m = sqrt(C(6, 3+m)).  Then
  N(c) = Nt(a) = sum_Q |sum_m1 (-1)^(3-Q+m1) a_m1 conj(a_{m1-Q})|^2 / C(12, 6+Q)
has rational coefficients (identity verified exactly in item 5), the Hermitian metric becomes
<c, c'> = sum_m conj(a_m) a'_m / C(6,3+m), and J_+-, J_z have integer matrices.  For a vector u with
||u||^2 = n (not necessarily 1), the form H_{u/|u|} in the y = (Re a, Im a) coordinates is
  Qy = Hess Nt(y)/n - 4 Nt(y)/n^2 * Wm,   Wm = diag(1/C(6,3+m)) (twice),
and N_u is the Wm-orthogonal complement of {y, o_1..o_4} (o_k the four generators of O_u).
Everything then lives in Q or in Q(sqrt d) for a single d, where sympy's DomainMatrix is fast.
The high-precision route evaluates M_u = Hess N(u) - 4N(u) I directly in the original
c-coordinates with mpmath, from the unscaled polynomial N built from the Clebsch-Gordan table.
"""
import os, pickle
import sympy as sp, mpmath as mp
import common as C
from sympy.polys.matrices import DomainMatrix

B6 = {m: sp.binomial(6, 3 + m) for m in C.MS}
YS = sp.symbols('y0:14', real=True)

def Nt_poly():
    a = [YS[k] + sp.I * YS[k + 7] for k in range(7)]   # a_m at index idx(m)
    tot = 0
    for Q in range(-6, 7):
        s = 0
        for m1 in C.MS:
            k = m1 - Q
            if abs(k) <= 3:
                m2 = Q - m1
                s += (-1) ** (3 - m2) * a[C.idx(m1)] * sp.conjugate(a[C.idx(k)])
        re, im = sp.expand(s).as_real_imag()
        tot += sp.expand(re ** 2 + im ** 2) / sp.binomial(12, 6 + Q)
    return sp.expand(tot)

_cache = {}
def Nt_and_hess():
    if 'H' not in _cache:
        p = os.path.join(C.OUT, 'hessNt.pkl')
        if os.path.exists(p):
            with open(p, 'rb') as f: _cache.update(pickle.load(f))
        else:
            Nt = Nt_poly()
            H = [[sp.expand(sp.diff(Nt, u, v)) for v in YS] for u in YS]
            g = [sp.expand(sp.diff(Nt, u)) for u in YS]
            _cache.update(dict(Nt=Nt, H=H, g=g))
            with open(p, 'wb') as f: pickle.dump(dict(Nt=Nt, H=H, g=g), f)
    return _cache['Nt'], _cache['H'], _cache['g']

Wm = sp.diag(*([sp.Rational(1, B6[m]) for m in C.MS] * 2))

def a_from_c(c):
    return [sp.radsimp(sp.sqrt(B6[m]) * c[C.idx(m)]) for m in C.MS]

def y_from_a(a):
    return sp.Matrix([sp.re(z) for z in a] + [sp.im(z) for z in a]).applyfunc(sp.radsimp)

# J in a-coordinates: A J A^{-1}, A = diag(s_m); integer entries
def J_a():
    S = sp.diag(*[sp.sqrt(B6[m]) for m in C.MS]); Si = S.inv()
    return [(S * Jk * Si).applyfunc(sp.nsimplify) for Jk in (C.JX, C.JY, C.JZ)]
JXa, JYa, JZa = J_a()

def realify(A):
    return C.realify(A)

def orbit_y(a):
    """the four generators of O_u (i u, -i Jx u, -i Jy u, -i Jz u) in y-coordinates."""
    av = sp.Matrix(a)
    gens = [sp.I * av] + [(-sp.I) * (Jk * av) for Jk in (JXa, JYa, JZa)]
    return [y_from_a([sp.expand(g[k]) for k in range(7)]) for g in gens]

def field_for(exprs):
    """smallest algebraic field Q(sqrt d) containing the entries (at most one square root)."""
    rads = set()
    for e in exprs:
        for p in sp.preorder_traversal(sp.sympify(e)):
            if isinstance(p, sp.Pow) and p.exp == sp.Rational(1, 2):
                rads.add(p)
    if not rads: return sp.QQ, None
    assert len(rads) == 1, rads
    r = rads.pop()
    return sp.QQ.algebraic_field(r), r

def exact_data(a, lam=sp.Symbol('lam')):
    """a: exact a-coordinates of a (not necessarily unit) representative.
    Returns dict with the restricted form data, all exact."""
    Nt, H, g = Nt_and_hess()
    y = y_from_a(a)
    sub = dict(zip(YS, list(y)))
    n = sp.radsimp((y.T * Wm * y)[0])
    Nval = sp.radsimp(Nt.subs(sub))
    Hy = sp.Matrix(14, 14, lambda i, j: sp.radsimp(H[i][j].subs(sub)))
    gy = sp.Matrix([sp.radsimp(e.subs(sub)) for e in g])
    Qy = (Hy / n - 4 * Nval / n ** 2 * Wm).applyfunc(sp.radsimp)
    orb = orbit_y(a)
    cons = sp.Matrix.vstack(*[(Wm * v).T for v in [y] + orb])
    dimO = cons.rank(simplify=True) - 1
    ns = cons.nullspace(simplify=True)
    B = sp.Matrix.hstack(*ns).applyfunc(sp.radsimp)
    K, r = field_for(list(Qy) + list(B))
    dQ = DomainMatrix.from_Matrix(Qy).convert_to(K)
    dB = DomainMatrix.from_Matrix(B).convert_to(K)
    dW = DomainMatrix.from_Matrix(Wm).convert_to(K)
    Kf = dB.transpose() * dQ * dB
    G = dB.transpose() * dW * dB
    Mred = G.inv() * Kf
    cp = Mred.charpoly()
    cp = [K.to_sympy(cf) for cf in cp]
    poly = sp.Poly([sp.nsimplify(sp.radsimp(cf)) for cf in cp], lam)
    # annihilation of O_u: Qy o for each generator, measured in the dual metric (M_u o in c-coords)
    resid = []
    Si = Wm  # ||M_u d||^2 in c-coords equals (Qy o)^T Wm^{-1} (Qy o)
    for o in orb:
        w = (Qy * o).applyfunc(sp.radsimp)
        resid.append(sp.sqrt(sp.nsimplify(sp.radsimp((w.T * Wm.inv() * w)[0]))))
    return dict(n=n, N=Nval, rhat=sp.radsimp(Nval / n ** 2), Qy=Qy, B=B, dimO=dimO, dimN=B.shape[1],
                charpoly=poly, field=str(K), resid=resid, orb=orb, y=y, gy=gy)

def signature(poly):
    """exact (n-, n0, n+) of a real-rooted polynomial by exact real-root isolation per factor."""
    lam = poly.gen
    nneg = nzero = npos = 0
    p = sp.Poly(poly.as_expr(), lam)
    assert p.domain in (sp.QQ, sp.ZZ), 'char poly not rational: %s' % p.domain
    for fac, mult in sp.factor_list(p)[1]:
        f = sp.Poly(fac, lam)
        if f.degree() == 0: continue
        roots = sp.real_roots(f)
        assert len(roots) == f.degree(), 'non-real roots in %s' % fac
        for r in roots:
            if r == 0: nzero += mult
            elif r < 0: nneg += mult
            else: npos += mult
    return nneg, nzero, npos

_hc = {}
def M_c(c):
    """exact M_u = Hess N(u) - 4 N(u) I in the original c-coordinates, u unit (CG-built N)."""
    if 'H' not in _hc:
        N = C.N_poly()
        _hc['H'] = [[sp.expand(sp.diff(N, a, b)) for b in C.XS] for a in C.XS]
        _hc['g'] = [sp.expand(sp.diff(N, a)) for a in C.XS]
    xv = C.c_to_x(c).applyfunc(sp.radsimp)
    sub = dict(zip(C.XS, list(xv)))
    Nv = sp.radsimp(sp.expand(C.N_poly().subs(sub)))
    M = sp.Matrix(14, 14, lambda i, j: sp.radsimp(sp.expand(_hc['H'][i][j].subs(sub)))) - 4 * Nv * sp.eye(14)
    g = sp.Matrix([sp.radsimp(sp.expand(e.subs(sub))) for e in _hc['g']])
    return M, Nv, g, xv

# ------------------------------------------------------------------ high-precision route
_hn = {}
def mp_M(c, dps):
    """M_u = Hess N(u) - 4 N(u) I in the ORIGINAL c-coordinates (CG-based N), mpmath."""
    mp.mp.dps = dps
    if 'f' not in _hn:
        N = C.N_poly()
        H = [[sp.diff(N, a, b) for b in C.XS] for a in C.XS]
        _hn['f'] = sp.lambdify(C.XS, sp.Matrix(H), 'mpmath')
        _hn['N'] = sp.lambdify(C.XS, N, 'mpmath')
    x = [mp.mpf(sp.re(z).evalf(dps + 10)) for z in c] + [mp.mpf(sp.im(z).evalf(dps + 10)) for z in c]
    nrm = mp.sqrt(sum(v * v for v in x)); x = [v / nrm for v in x]
    H = mp.matrix(_hn['f'](*x)); Nv = _hn['N'](*x)
    return H - 4 * Nv * mp.eye(14), mp.matrix(x), Nv
