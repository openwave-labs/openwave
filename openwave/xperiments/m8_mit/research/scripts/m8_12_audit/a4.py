"""Audit items 1/3/4/7/9: exact rotations, critical lines, separations, kernel at v1, symmetries."""
import sympy as sp
from aud_core import *
from aud_acoords import form_at_c, realv, c_to_a, Na, VARS, A as Asym

Jx, Jy, Jz = Jmat()
def Dexact(n, cth, sth):
    """D(n,theta) = sum_m e^{-i m theta} P_m, exact; theta given by (cos, sin); n unit (exact)."""
    G = (n[0] * Jx + n[1] * Jy + n[2] * Jz).applyfunc(sp.radsimp)
    E = sp.eye(7)
    D = sp.zeros(7)
    eith = cth + I * sth
    for m in MS:
        Pm = E
        for k in MS:
            if k != m:
                Pm = (Pm * (G - k * E) / (m - k))
        Pm = Pm.applyfunc(lambda t: sp.radsimp(sp.expand(t)))
        D += sp.expand(eith ** (-m)) * Pm
    return D.applyfunc(lambda t: sp.radsimp(sp.expand(t)))
def app(D, c): return [sp.radsimp(sp.expand(t)) for t in list(D * sp.Matrix(c))]
def hip(p, q): return sp.expand(sum(sp.conjugate(a) * b for a, b in zip(p, q)))
def same_line(p, q):
    return sp.radsimp(sp.expand(hip(p, q) * sp.conjugate(hip(p, q)) - hip(p, p) * hip(q, q))) == 0
def vc(d):
    c = [sp.Integer(0)] * 7
    for m, z in d.items(): c[idx(m)] = sp.sympify(z)
    return c
r2, r3, r5 = sp.sqrt(2), sp.sqrt(3), sp.sqrt(5)
# sanity: D(z, 2pi) = 1 and D(x, pi) v_m = -v_{-m}
print('D(z,2pi) == 1:', Dexact((0, 0, 1), 1, 0) == sp.eye(7))
Dxpi = Dexact((1, 0, 0), -1, 0)
print('D(x,pi) v_m = -v_{-m}:', all(app(Dxpi, vc({m: 1})) == vc({-m: -1}) for m in MS))
n111 = (1 / r3, 1 / r3, 1 / r3)
D111 = Dexact(n111, sp.Rational(-1, 2), r3 / 2)
print('D(111, 2pi/3)^3 == 1:', (D111 ** 3).applyfunc(lambda t: sp.radsimp(sp.expand(t))) == sp.eye(7))
# item 1: the three D2 planes
F = [vc({0: 1}), vc({2: 1, -2: 1})]
def span_equal(B1, B2):
    M1 = sp.Matrix.hstack(*[sp.Matrix(b) for b in B1]); M2 = sp.Matrix.hstack(*[sp.Matrix(b) for b in B2])
    return sp.Matrix.hstack(M1, M2).rank(simplify=True) == 2
F1 = [app(D111, b) for b in F]; F2 = [app(D111, b) for b in F1]
other1 = [vc({3: 1, -3: -1}), vc({1: 1, -1: -1})]; other2 = [vc({3: 1, -3: 1}), vc({1: 1, -1: 1})]
print('R111 F  == span{v3-v-3, v1-v-1}?', span_equal(F1, other1), ' == span{v3+v-3, v1+v-1}?', span_equal(F1, other2))
print('R111^2 F == span{v3-v-3, v1-v-1}?', span_equal(F2, other1), ' == span{v3+v-3, v1+v-1}?', span_equal(F2, other2))
# item 4: A's exhibited rotations
cat = vc({3: 1, -3: 1}); xyz = vc({2: 1, -2: -1})
Dy90 = Dexact((0, 1, 0), 0, 1); Dz90 = Dexact((0, 0, 1), 0, 1)
Fpt = [a + sp.sqrt(sp.Rational(3, 5)) * b / r2 for a, b in zip(vc({0: 1}), vc({2: 1, -2: 1}))]
print('R_y(pi/2) R_z(pi/2) cat  ~ F point z=sqrt(3/5):', same_line(app(Dy90 * Dz90, cat), Fpt))
Dypi = Dexact((0, 1, 0), -1, 0)
print('control R_y(pi) cat ~ F point? (must be False):', same_line(app(Dypi, cat), Fpt))
Gpt = [a + (2 / r5) * b / r2 for a, b in zip(vc({0: 1}), vc({3: 1, -3: -1}))]
Dz45 = Dexact((0, 0, 1), 1 / r2, 1 / r2)
Dyb = Dexact((0, 1, 0), 1 / r3, sp.sqrt(sp.Rational(2, 3)))
Dz300 = Dexact((0, 0, 1), sp.Rational(1, 2), -r3 / 2)
print('R_z(5pi/3) R_y(arccos 1/sqrt3) R_z(pi/4) xyz ~ G point z=2/sqrt5:', same_line(app(Dz300 * Dyb * Dz45, xyz), Gpt))

# item 3: tangential gradient at six lines, exact
for nm, c in [('v3', vc({3: 1})), ('v2', vc({2: 1})), ('v1', vc({1: 1})), ('v0', vc({0: 1})),
              ('xyz', [t / r2 for t in xyz]), ('cat', [t / r2 for t in cat])]:
    g = grad_exact(c); Nv = sp.expand(N_exact(c)); xr = to_real(c)
    tg = [sp.radsimp(sp.expand(a - 4 * Nv * b)) for a, b in zip(g, xr)]
    print('item3 %-4s rhat6 = %-8s tangential gradient == 0: %s' % (nm, sp.radsimp(Nv), all(t == 0 for t in tg)))

# item 4 separations: <J> vector
def expJ(c):
    n2 = hip(c, c)
    return [sp.radsimp(sp.expand(hip(c, list(J * sp.Matrix(c))) / n2)) for J in (Jx, Jy, Jz)]
print('<J> at v1:', expJ(vc({1: 1})), '  at F*:', expJ(vc({0: sp.sqrt(6), 2: I * r5, -2: I * r5})))
print('<J> at A*:', expJ(vc({1: 2, -2: 1})), '  at D*:', expJ(vc({2: sp.sqrt(13), -3: 2 * r3})))

# item 7: kernel at v1 = span{v-3, i v-3}
Mf, _, _, _ = form_at_c(vc({1: 1}))
for nm, e in [('v-3', vc({-3: 1})), ('i v-3', vc({-3: I}))]:
    col = (Mf * realv(c_to_a(e))).applyfunc(sp.expand)
    print('item7: M_u applied to %-6s == 0 (full vector): %s' % (nm, all(t == 0 for t in col)))
e_bad = vc({-2: 1})
print('item7 control: M_u v-2 == 0 ?', all(t == 0 for t in (Mf * realv(c_to_a(e_bad))).applyfunc(sp.expand)))

# item 9: symmetries preserve N (exact polynomial identities in the a-coordinates)
cs = sp.symbols('w0:7')
conjA = [sp.conjugate(z) for z in Asym]
def Na_of(avec):
    sub = {}
    # express Na(avec) by re-evaluating the defining formula
    tot = 0
    for Q in range(-6, 7):
        s = 0
        for m1 in MS:
            m2 = Q - m1
            if abs(m2) <= 3:
                s += (-1) ** (3 - m2) * avec[idx(m1)] * sp.conjugate(avec[idx(-m2)])
        s = sp.expand(s)
        tot += sp.expand(s * sp.conjugate(s)) / sp.binomial(12, 6 + Q)
    return sp.expand(tot)
print('item9: N(conj u) == N(u):', sp.expand(Na_of(conjA) - Na) == 0)
thetaA = [(-1) ** (3 - m) * sp.conjugate(Asym[idx(-m)]) for m in MS]   # Theta commutes with the a-scaling (C(6,3+m)=C(6,3-m))
print('item9: N(Theta u) == N(u):', sp.expand(Na_of(thetaA) - Na) == 0)
# infinitesimal invariance: grad N . (X x) == 0 for X in {i, -iJx, -iJy, -iJz}  (c-coordinates, exact)
cvars = sp.symbols('x0:7', real=True) + sp.symbols('y0:7', real=True)
cvec = [cvars[k] + I * cvars[7 + k] for k in range(7)]
Nc = sp.expand(N_exact(cvec))
grad = [sp.diff(Nc, v) for v in cvars]
ok = True
for gen in O_gens(cvec):
    xr = [sp.re(sp.expand(t)) for t in gen] + [sp.im(sp.expand(t)) for t in gen]
    ok &= sp.expand(sum(a * b for a, b in zip(grad, xr))) == 0
print('item9: grad N . (X u) == 0 identically for the four generators (U(1)xSO(3) invariance, connected group):', ok)
# planted: a wrong Theta sign breaks rotation invariance
def N_badtheta(c):
    th = [(-1) ** (3 - m) * sp.conjugate(c[idx(-m)]) for m in MS]; th[idx(1)] = -th[idx(1)]
    tot = 0
    for Q in range(-6, 7):
        s = sum(cg6(m1, Q - m1, Q) * c[idx(m1)] * th[idx(Q - m1)] for m1 in MS if abs(Q - m1) <= 3)
        s = sp.expand(s); tot += sp.expand(s * sp.conjugate(s))
    return sp.expand(tot)
Nb = N_badtheta(cvec); gb = [sp.diff(Nb, v) for v in cvars]
xr = [sp.re(sp.expand(t)) for t in O_gens(cvec)[1]] + [sp.im(sp.expand(t)) for t in O_gens(cvec)[1]]
print('PLANTED wrong Theta: Jx-invariance identity still holds? (must be False):', sp.expand(sum(a * b for a, b in zip(gb, xr))) == 0)
