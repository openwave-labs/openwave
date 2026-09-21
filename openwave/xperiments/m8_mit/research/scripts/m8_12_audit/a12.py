"""Audit item 12: H_u along a direction of N_u, two ways, at my own choice (A*), and re-check of each
solver's own item-12 number with my code."""
import sympy as sp, mpmath as mp
from aud_core import *
from aud_acoords import form_at_c, realv, c_to_a

s = sp.Symbol('s', real=True)
def rip(p, q): return sp.expand(sum(sp.re(sp.expand(sp.conjugate(a) * b)) for a, b in zip(p, q)))

def NuProj(u, w):
    """exact Re<,>-orthogonal projection of w onto N_u = complement of span{u, O_u}."""
    span = [u] + O_gens(u)
    keep = []
    for k in range(len(span)):
        sub = keep + [k]
        Gm = sp.Matrix(len(sub), len(sub), lambda i, j: rip(span[sub[i]], span[sub[j]]))
        if sp.simplify(Gm.det()) != 0: keep = sub
    Gm = sp.Matrix(len(keep), len(keep), lambda i, j: rip(span[keep[i]], span[keep[j]]))
    rhs = sp.Matrix([rip(span[k], w) for k in keep])
    coef = (Gm.inv() * rhs).applyfunc(sp.radsimp)
    p = list(w)
    for cf, k in zip(coef, keep):
        p = [a - cf * b for a, b in zip(p, span[k])]
    return [sp.radsimp(sp.expand(t)) for t in p]

def route1_exact(u, e):
    """d^2/ds^2 rhat6(u cos s + e sin s) at 0, exact, from the Taylor data cos=1-s^2/2, sin=s."""
    g = [a * (1 - s ** 2 / 2) + b * s for a, b in zip(u, e)]
    Ns = sp.Poly(sp.expand(N_exact(g)), s)
    n2 = sp.Poly(sp.expand(nrm2(g)), s)
    N0, N1, N2 = [Ns.coeff_monomial(s ** k) for k in range(3)]
    q0, q1, q2 = [n2.coeff_monomial(s ** k) for k in range(3)]
    # f = N * q^-2 ; second derivative at 0 = 2 * [s^2 coefficient]
    i0 = 1 / q0 ** 2; i1 = -2 * q1 / q0 ** 3; i2 = (3 * q1 ** 2 - 2 * q0 * q2) / q0 ** 4
    return sp.radsimp(sp.expand(2 * (N0 * i2 + N1 * i1 + N2 * i0)))

def route2_exact(u, e):
    Mf, _, _, _ = form_at_c(u)
    ea = realv(c_to_a(e))
    return sp.radsimp(sp.expand((ea.T * Mf * ea)[0]))

def mp_routes(u, e, dps):
    mp.mp.dps = dps
    cg = mp_cg()
    un = [mp.mpc(sp.N(sp.re(z), dps + 20), sp.N(sp.im(z), dps + 20)) for z in u]
    en = [mp.mpc(sp.N(sp.re(z), dps + 20), sp.N(sp.im(z), dps + 20)) for z in e]
    F = lambda t: mp_rhat([a * mp.cos(t) + b * mp.sin(t) for a, b in zip(un, en)], cg)
    d1 = mp.diff(F, 0, 2)
    # matrix route numerically: e^T Hess N e - 4 N |e|^2 with Hess N[e,e] = d^2/dh^2 N(u + h e) (exact for a quartic
    # via its polynomial structure: evaluate N(u+he) at 5 points and take the h^2 coefficient)
    hs = [mp.mpf(k) for k in (-2, -1, 0, 1, 2)]
    vals = [mp_N([a + h * b for a, b in zip(un, en)], cg) for h in hs]
    # interpolate quartic exactly (Vandermonde) and read 2*coef(h^2)
    V = mp.matrix([[h ** k for k in range(5)] for h in hs])
    co = mp.lu_solve(V, mp.matrix(vals))
    Nu = mp_N(un, cg); e2 = sum(abs(z) ** 2 for z in en)
    d2 = 2 * co[2] - 4 * Nu * e2
    return d1, d2

def report(label, u, e, normalise=True):
    assert rip(u, u) == 1
    n2e = rip(e, e)
    h1 = route1_exact(u, e); h2 = route2_exact(u, e)
    diff = sp.radsimp(sp.expand(h1 - h2))
    val = sp.radsimp(h2 / n2e) if normalise else h2
    print(label)
    print('   |e|^2 =', n2e, ';  exact H(e/|e|) =', val, '≈', sp.N(val, 30))
    print('   route1 exact - route2 exact =', diff)
    try:
        print('   minimal polynomial of H(e/|e|):', sp.minimal_polynomial(val, sp.Symbol('X')))
    except Exception as ex:
        print('   minimal polynomial: not computed (%s)' % ex)
    for dps in (50, 80):
        d1, d2 = mp_routes(u, e, dps)
        ex = mp.mpf(sp.N(h2, dps + 20))
        print('   dps %d: |route1_num - exact| = %s  |route2_num - exact| = %s  |route1_num - route2_num| = %s' % (
            dps, mp.nstr(abs(d1 - ex), 3), mp.nstr(abs(d2 - ex), 3), mp.nstr(abs(d1 - d2), 3)))
    return val

def vc(d):
    c = [sp.Integer(0)] * 7
    for m, z in d.items(): c[idx(m)] = sp.sympify(z)
    return c
# ---- my own choice: A* and a generic real direction
uA = [z / sp.sqrt(5) for z in vc({1: 2, -2: 1})]
wv = [3, -1, 4, 1, -5, 9, 2, -6, 5, 3, -5, 8, 9, 7]
w = [sp.Integer(wv[k]) + I * wv[7 + k] for k in range(7)]
p = NuProj(uA, w)
print('check p in N_u:', rip(p, uA) == 0, all(sp.simplify(rip(p, g)) == 0 for g in O_gens(uA)))
report('AUDITOR: orbit A*, e = P_N w, w = %s' % wv, uA, p)
# planted defect: use a direction NOT in T_u (add u) -> the two routes must disagree
bad = [a + b for a, b in zip(p, uA)]
print('PLANTED (e not tangent): route1 - route2 =', sp.N(route1_exact(uA, bad) - route2_exact(uA, bad), 15))

# ---- solver A's item 12 e2: F*, w = v3 + 2v2 - v1 + 3i v0 + v-2 - 2i v-3
uF = [z / 4 for z in vc({0: sp.sqrt(6), 2: I * sp.sqrt(5), -2: I * sp.sqrt(5)})]
wA = vc({3: 1, 2: 2, 1: -1, 0: 3 * I, -2: 1, -3: -2 * I})
report("SOLVER A's e2 at F*", uF, NuProj(uF, wA))
# ---- solver B's item 12: Wmin, w = (1,...,14) in (Re, Im) coordinates
cW = vc({0: 1, 3: sp.sqrt(sp.Rational(10, 23)), -3: sp.sqrt(sp.Rational(10, 23))})
nW = sp.sqrt(rip(cW, cW)); uW = [sp.radsimp(z / nW) for z in cW]
wB = [sp.Integer(k + 1) + I * (k + 8) for k in range(7)]
report("SOLVER B's direction at Wmin", uW, NuProj(uW, wB))
