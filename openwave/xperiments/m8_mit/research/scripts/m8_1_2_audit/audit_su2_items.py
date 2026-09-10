"""SU(2)-only items 1-9, 15; constants c_K (G_K = c_K M_K) and N_J maps for items 14 and 21.

Exact sympy throughout (Condon-Shortley CG by the Racah formula, cross-checked against sympy's CG
in audit_item0.py). Vectors are lists indexed by m = -3..3 ascending (index = m + 3).
"""
import json
import pathlib
import itertools
from collections import Counter
import sympy as sp
import mpmath as mp
from audit_su2lib import cg, theta, couple, rho, M_map, basis, norm2, simp

HERE = pathlib.Path(__file__).parent
res = {}
I = sp.I
MS = list(range(-3, 4))


def ix(m):
    return m + 3


def vec(d):
    v = [sp.Integer(0)] * 7
    for m, c in d.items():
        v[ix(m)] = sp.nsimplify(c)
    return v


# ---------------- item 1 and item 2: weight multiplicities ----------------
def decomp_from_weights(wts):
    c = Counter(wts)
    top = max(c)
    out = {}
    for J in range(top, -1, -1):
        mlt = c[J] - c.get(J + 1, 0)
        if mlt:
            out[J] = mlt
    return out


sym3 = [sum(t) for t in itertools.combinations_with_replacement(MS, 3)]
d1 = decomp_from_weights(sym3)
assert sum((2 * J + 1) * m for J, m in d1.items()) == 84
res["item1_Sym3V3"] = {str(J): m for J, m in sorted(d1.items(), reverse=True)}
sym2 = [a + b for a, b in itertools.combinations_with_replacement(MS, 2)]
res["Sym2V3"] = {str(J): m for J, m in sorted(decomp_from_weights(sym2).items(), reverse=True)}
s2v = [a + b for a in sym2 for b in MS]  # conj V3 has the same weight multiset
d2 = decomp_from_weights(s2v)
res["item2_dim_Hom_J8"] = d2.get(8, 0)
res["item2_dim_Hom_J3"] = d2.get(3, 0)
res["V3tensor3_mult_V8"] = decomp_from_weights([a + b + c for a in MS for b in MS for c in MS]).get(8, 0)

# ---------------- symbolic state ----------------
u_s = list(sp.symbols("u_m3 u_m2 u_m1 u_0 u_1 u_2 u_3"))
a_s = list(sp.symbols("a0:7", real=True))
b_s = list(sp.symbols("b0:7", real=True))
ur = [a + I * b for a, b in zip(a_s, b_s)]
nrm2_r = sum(a * a + b * b for a, b in zip(a_s, b_s))

# ---------------- item 3 ----------------
M0 = M_map(ur, 0)
diff = [sp.expand(M0[k] + nrm2_r / sp.sqrt(7) * ur[k]) for k in range(7)]
assert all(sp.simplify(x) == 0 for x in diff)
rho0 = rho(ur, 0)
assert sp.simplify(rho0[0] + nrm2_r / sp.sqrt(7)) == 0
res["item3_M0_equals"] = "-(||u||^2/sqrt(7)) u"
res["item3_constant"] = str(-1 / sp.sqrt(7))
res["rho0_equals"] = "-||u||^2/sqrt(7)"

# ---------------- item 4 ----------------
lam = []
for m in MS:
    Mv = [simp(x) for x in M_map(basis(m), 6)]
    for k in range(7):
        if k != ix(m):
            assert Mv[k] == 0
    lam.append(Mv[ix(m)])
res["item4_M6_diag_m=-3..3"] = [str(x) for x in lam]
res["item4_M6_diag_times_sqrt7"] = [str(simp(x * sp.sqrt(7))) for x in lam]
res["item4_M6_diag_times_1001sqrt7"] = [str(simp(x * sp.sqrt(7) * 1001)) for x in lam]
# the linear operator A_6(v_m): diagonal?
A6diag = {}
for m in MS:
    r6 = rho(basis(m), 6)
    mat = []
    for k in MS:
        col = couple(r6, basis(k), 6, 3, 3)
        col = [simp(x) for x in col]
        for kk in range(7):
            if kk != ix(k):
                assert col[kk] == 0
        mat.append(col[ix(k)])
    A6diag[str(m)] = [str(x) for x in mat]
res["item4_A6(v_m)_is_diagonal_diag_entries_by_m_then_k"] = A6diag

# ---------------- item 5 ----------------
v3 = basis(3)
B2 = [simp(x) for x in couple(v3, v3, 3, 3, 2)]
r2 = [simp(x) for x in rho(v3, 2)]
res["item5_B2_v3_N=-2..2"] = [str(x) for x in B2]
res["item5_rho2_v3_N=-2..2"] = [str(x) for x in r2]
res["item5_Theta_v3"] = [str(x) for x in theta(v3)]
# holomorphic squares vanish at odd J (check on symbolic u)
for J in (1, 3, 5):
    assert all(sp.expand(x) == 0 for x in couple(u_s, u_s, 3, 3, J))
res["B_J_odd_vanish_checked"] = True

# ---------------- helpers: gradient / criticality ----------------
f6_r = None


def f_real(K):
    r = rho(ur, K)
    return sum(x * sp.conjugate(x) for x in r)


F6 = f_real(6)


def grad_complex(Fexpr, point):
    """g_m = dF/da_m + i dF/db_m at a numeric-exact point (list of 7 complex exact values)."""
    subs = {}
    for k in range(7):
        subs[a_s[k]] = sp.re(point[k])
        subs[b_s[k]] = sp.im(point[k])
    g = []
    for k in range(7):
        ga = sp.diff(Fexpr, a_s[k]).subs(subs)
        gb = sp.diff(Fexpr, b_s[k]).subs(subs)
        g.append(simp(ga + I * gb))
    return g


def tangential(g, u):
    nu = sum(x * sp.conjugate(x) for x in u)
    c = sum(sp.conjugate(u[k]) * g[k] for k in range(7)) / nu
    return [simp(g[k] - c * u[k]) for k in range(7)], simp(c)


def Gmap(u, K):
    """G_K(u)_k = (-1)^k sum_N CG(3,k+N;3,-k|K,N) u_{k+N} conj(rho_N(u)); <v,G_K> = <rho_K(u),[u (x) Theta v]_K>."""
    r = rho(u, K)
    out = []
    for k in MS:
        s = 0
        for iN, N in enumerate(range(-K, K + 1)):
            if abs(k + N) > 3:
                continue
            s += cg(3, k + N, 3, -k, K, N) * u[ix(k + N)] * sp.conjugate(r[iN])
        out.append((-1) ** abs(k) * s)
    return out


# ---------------- item 6 ----------------
states6 = {
    "v3": basis(3),
    "v0": basis(0),
    "(v2+v-2)/sqrt2": vec({2: 1 / sp.sqrt(2), -2: 1 / sp.sqrt(2)}),
    "(v3+v-3)/sqrt2": vec({3: 1 / sp.sqrt(2), -3: 1 / sp.sqrt(2)}),
}
item6 = {}
for name, u in states6.items():
    n6 = norm2(rho(u, 6))
    g = grad_complex(F6, u)
    tang, c = tangential(g, u)
    crit = all(x == 0 for x in tang)
    M6 = [simp(x) for x in M_map(u, 6)]
    t2, c2 = tangential(M6, u)
    item6[name] = {"924*||rho6||^2": str(simp(924 * n6)), "||rho6||^2": str(n6),
                   "grad_tangential_zero(critical)": crit, "grad_radial_coeff": str(c),
                   "M6(u)_parallel_u": all(x == 0 for x in t2), "M6_eigenvalue": str(c2)}
res["item6"] = item6

# relation gradient <-> G_6 <-> M_6 at a generic exact point
up = [sp.Integer(1) + I, sp.Integer(2), -I, sp.Rational(1, 2), sp.Integer(-1), 3 * I, sp.Integer(1) - 2 * I]
g = grad_complex(F6, up)
G6 = [simp(x) for x in Gmap(up, 6)]
M6p = [simp(x) for x in M_map(up, 6)]
ratios_gG = {simp(g[k] / G6[k]) for k in range(7) if G6[k] != 0}
res["grad_over_G6_ratio_set"] = [str(x) for x in ratios_gG]

# ---------------- c_K: G_K = c_K M_K ----------------
cK = {}
tests = [up, [sp.Integer(2) - I, sp.Integer(0), sp.Integer(1), I, sp.Integer(-3), sp.Integer(1) + I, sp.Rational(1, 3)]]
for K in range(7):
    rs = set()
    for u in tests:
        Gv = [simp(x) for x in Gmap(u, K)]
        Mv = [simp(x) for x in M_map(u, K)]
        for k in range(7):
            if Mv[k] == 0:
                assert Gv[k] == 0
            else:
                rs.add(simp(Gv[k] / Mv[k]))
    assert len(rs) == 1, (K, rs)
    cK[K] = rs.pop()
res["cK_G_K_over_M_K"] = {str(K): str(v) for K, v in cK.items()}

# ---------------- item 7 ----------------
u7 = states6["(v2+v-2)/sqrt2"]
row7 = [norm2(rho(u7, K)) for K in range(7)]
assert simp(sum(row7)) == 1
res["item7_row_K0..6"] = [str(x) for x in row7]

# ---------------- item 8 ----------------
s = sp.symbols("s", positive=True)
cT, sT = sp.symbols("cT sT", positive=True)  # cos t, sin t (t in (0, pi/2)); real so conj is trivial
u8c = vec({})
u8c[ix(2)] = cT
u8c[ix(-3)] = sT
f8cs = sp.expand(sp.nsimplify(sp.radsimp(sp.expand(sum(sp.expand(x * sp.conjugate(x)) for x in rho(u8c, 6))))))
Pcs = sp.Poly(f8cs, cT, sT)
assert all(e1 % 2 == 0 and e2 % 2 == 0 for (e1, e2) in Pcs.monoms()), "odd powers: not a function of s"
f8 = sp.expand(f8cs.subs({cT: sp.sqrt(1 - s), sT: sp.sqrt(s)}))
assert sp.Poly(f8, s).is_polynomial if hasattr(sp.Poly(f8, s), "is_polynomial") else True
u8 = vec({})
u8[ix(2)] = sp.sqrt(1 - s)
u8[ix(-3)] = sp.sqrt(s)
res["item8_norm_rho6_sq_in_cos_sin"] = str(f8cs)
P8 = sp.Poly(f8, s)
res["item8_norm_rho6_sq_poly_in_s"] = str(f8)
res["item8_924_times_poly"] = str(sp.expand(924 * f8))
df8 = sp.diff(f8, s)
crit8 = [r for r in sp.solve(df8, s) if r.is_real and 0 < r < 1]
res["item8_derivative"] = str(sp.factor(df8))
res["item8_interior_stationary"] = [{"s": str(r), "value": str(simp(f8.subs(s, r))), "924value": str(simp(924 * f8.subs(s, r))),
                                      "second_derivative": str(simp(sp.diff(f8, s, 2).subs(s, r)))} for r in crit8]
res["item8_endpoints"] = {"s=0": str(f8.subs(s, 0)), "s=1": str(f8.subs(s, 1)),
                          "924*s=0": str(924 * f8.subs(s, 0)), "924*s=1": str(924 * f8.subs(s, 1))}
# criticality of these rays on the full P(V3)
item8_full = []
for r in crit8:
    uu = [simp(x.subs(s, r)) for x in u8]
    M6 = [simp(x) for x in M_map(uu, 6)]
    t2, c2 = tangential(M6, uu)
    item8_full.append({"s": str(r), "M6_parallel_u(critical on P(V3))": all(x == 0 for x in t2)})
res["item8_full_space_criticality"] = item8_full

# ---------------- item 9 ----------------
x, y = sp.symbols("x y", real=True)
z = x + I * y
u9 = vec({})
u9[ix(3)] = sp.Integer(1)
u9[ix(0)] = z
u9[ix(-3)] = sp.Integer(1)
g9 = sp.expand(sp.nsimplify(sp.radsimp(sp.expand(sum(sp.expand(t * sp.conjugate(t)) for t in rho(u9, 6))))))
n9 = 2 + x ** 2 + y ** 2
r9 = g9 / n9 ** 2
res["item9_norm_rho6_sq"] = str(sp.factor(g9))
res["item9_rhat6"] = str(sp.factor(r9))
res["item9_924_rhat6"] = str(sp.factor(924 * r9))
ex = sp.factor(sp.numer(sp.together(sp.diff(r9, x))))
ey = sp.factor(sp.numer(sp.together(sp.diff(r9, y))))
res["item9_dr_dx_numerator"] = str(ex)
res["item9_dr_dy_numerator"] = str(ey)
GB = sp.groebner([sp.expand(ex), sp.expand(ey)], x, y, order="lex")
res["item9_groebner_lex_r"] = [str(sp.factor(p)) for p in GB.exprs]
sols = sp.solve([ex, ey], [x, y], dict=True)
crit9 = []
for so in sols:
    xv, yv = sp.nsimplify(so[x]), sp.nsimplify(so[y])
    if not (xv.is_real and yv.is_real):
        continue
    crit9.append((simp(xv), simp(yv)))
crit9 = sorted(set(crit9), key=lambda t: (float(t[0]), float(t[1])))
item9c = []
for xv, yv in crit9:
    val = simp(r9.subs({x: xv, y: yv}))
    uu = [simp(t.subs({x: xv, y: yv})) for t in u9]
    M6 = [simp(t) for t in M_map(uu, 6)]
    t2, _ = tangential(M6, uu)
    H = sp.hessian(r9, (x, y)).subs({x: xv, y: yv})
    ev = [simp(e) for e in H.eigenvals()]
    item9c.append({"x": str(xv), "y": str(yv), "rhat6": str(val), "924rhat6": str(simp(924 * val)),
                   "hessian_eigs": [str(e) for e in ev],
                   "critical_on_full_P(V3)": all(t == 0 for t in t2)})
res["item9_critical_rhat6"] = item9c
# unnormalised
gx = sp.factor(sp.diff(g9, x))
gy = sp.factor(sp.diff(g9, y))
solsU = sp.solve([gx, gy], [x, y], dict=True)
critU = sorted({(simp(so[x]), simp(so[y])) for so in solsU if sp.nsimplify(so[x]).is_real and sp.nsimplify(so[y]).is_real},
               key=lambda t: (float(t[0]), float(t[1])))
res["item9_unnormalised_grad"] = [str(gx), str(gy)]
res["item9_critical_unnormalised"] = [{"x": str(a), "y": str(b), "norm_rho6_sq": str(simp(g9.subs({x: a, y: b}))),
                                       "rhat6_there": str(simp(r9.subs({x: a, y: b})))} for a, b in critU]
# point at infinity of the chart: [v0]; chart w = p + i q: u = w (v3 + v-3) + v0
p_, q_ = sp.symbols("p q", real=True)
w = p_ + I * q_
uinf = vec({})
uinf[ix(3)] = w
uinf[ix(-3)] = w
uinf[ix(0)] = sp.Integer(1)
ginf = sp.expand(sp.nsimplify(sp.radsimp(sp.expand(sum(sp.expand(t * sp.conjugate(t)) for t in rho(uinf, 6))))))
rinf = ginf / (1 + 2 * (p_ ** 2 + q_ ** 2)) ** 2
res["item9_chart_at_infinity_rhat6"] = str(sp.factor(rinf))
res["item9_[v0]_rhat6"] = str(simp(rinf.subs({p_: 0, q_: 0})))
res["item9_[v0]_grad"] = [str(simp(sp.diff(rinf, p_).subs({p_: 0, q_: 0}))), str(simp(sp.diff(rinf, q_).subs({p_: 0, q_: 0})))]
res["item9_[v0]_hessian"] = str(sp.hessian(rinf, (p_, q_)).subs({p_: 0, q_: 0}))


# ---------------- item 15: constellations ----------------
zeta = sp.symbols("zeta")


def constellation(u):
    F = sum((-1) ** (3 - m) * sp.sqrt(sp.binomial(6, 3 + m)) * u[ix(m)] * zeta ** (3 - m) for m in MS)
    F = sp.expand(F)
    P = sp.Poly(F, zeta)
    deg = P.degree()
    # write F = zeta^a * H(zeta^g), solve H exactly, then take g-th roots
    exps = [e[0] for e in P.monoms()]
    a0 = min(exps)
    import math
    g = 0
    for e in exps:
        g = math.gcd(g, e - a0)
    g = max(g, 1)
    wv = sp.symbols("wv")
    H = sp.Poly(sum(c * wv ** ((e[0] - a0) // g) for e, c in zip(P.monoms(), P.coeffs())), wv)
    hr = sp.roots(H) if H.degree() > 0 else {}
    assert sum(hr.values()) == H.degree(), ("root finding incomplete", F)
    rts = Counter()
    if a0:
        rts[sp.Integer(0)] += a0
    for w0, mlt in hr.items():
        w0 = simp(w0)
        mod = simp(sp.Abs(w0) ** sp.Rational(1, g))
        ang = sp.arg(w0)
        for kk in range(g):
            r = simp(mod * sp.exp(I * (ang + 2 * sp.pi * kk) / g))
            rts[r] += mlt
    assert sum(rts.values()) == deg
    # verify every root numerically to 40 digits
    for r in rts:
        assert abs(sp.N(F.subs(zeta, r), 40)) < 1e-30, (F, r)
    pts = []
    for r, mult in rts.items():
        r = simp(r)
        rr = simp(sp.Abs(r) ** 2)
        vecp = (simp(2 * sp.re(r) / (1 + rr)), simp(2 * sp.im(r) / (1 + rr)), simp((1 - rr) / (1 + rr)))
        for _ in range(mult):
            pts.append({"root": str(r), "|r|": str(simp(sp.sqrt(rr))), "cos_theta": str(vecp[2]),
                        "theta_deg": float(sp.N(2 * sp.atan(sp.sqrt(rr)) * 180 / sp.pi, 15)),
                        "azimuth": str(simp(sp.arg(r))) if r != 0 else "undefined(pole)",
                        "azimuth_deg": (float(sp.N(sp.arg(r) * 180 / sp.pi, 15)) if r != 0 else None),
                        "xyz": [str(c) for c in vecp], "_v": vecp})
    for _ in range(6 - deg):
        pts.append({"root": "infinity", "cos_theta": "-1", "theta_deg": 180.0, "azimuth": "undefined(pole)",
                    "azimuth_deg": None, "xyz": ["0", "0", "-1"], "_v": (0, 0, -1)})
    dots = Counter()
    for i1 in range(6):
        for i2 in range(i1 + 1, 6):
            dv = simp(sum(a * b for a, b in zip(pts[i1]["_v"], pts[i2]["_v"])))
            dots[str(dv)] += 1
    for p in pts:
        del p["_v"]
    return {"F": str(F), "degree": deg, "points": pts, "pairwise_dot_multiset": dict(dots)}


const = {}
for name, u in states6.items():
    const[name] = constellation(u)
for r in crit8:
    uu = [simp(t.subs(s, r)) for t in u8]
    const[f"item8_s={r}"] = constellation(uu)
for xv, yv in crit9:
    uu = [simp(t.subs({x: xv, y: yv})) for t in u9]
    const[f"item9_x={xv}_y={yv}"] = constellation(uu)
res["item15_constellations"] = const

# ---------------- ranks of equivariant cubic maps (items 2, 14, 21) ----------------
def Nmap(u, J):
    """N_J(u)_n = sum_N CG(3,n;3,N-n|J,N) conj(u_{N-n}) B_J(u)_N,  <v,N_J(u)> = <[v (x) u]_J, B_J(u)>."""
    B = couple(u, u, 3, 3, J)
    out = []
    for n_ in MS:
        s_ = 0
        for iN, N in enumerate(range(-J, J + 1)):
            if abs(N - n_) > 3:
                continue
            s_ += cg(3, n_, 3, N - n_, J, N) * sp.conjugate(u[ix(N - n_)]) * B[iN]
        out.append(s_)
    return out


def fL(u, L):
    return couple(couple(u, u, 3, 3, L), theta(u), L, 3, 3)


us = u_s
ubar = [sp.conjugate(t) for t in us]
gens = us + ubar


def coeffvec(expr_list):
    """coefficients of the m=3 output component (determines an equivariant map)."""
    P = sp.Poly(sp.expand(expr_list[ix(3)]), *gens)
    return dict(zip(P.monoms(), P.coeffs()))


maps = {}
for K in range(7):
    maps[f"M{K}"] = coeffvec(M_map(us, K))
for L in (0, 2, 4, 6):
    maps[f"f{L}"] = coeffvec(fL(us, L))
for J in (0, 2, 4, 6):
    maps[f"N{J}"] = coeffvec(Nmap(us, J))
allmon = sorted(set().union(*[set(v) for v in maps.values()]))
mp.mp.dps = 60


def mrank(names):
    A = mp.matrix(len(names), len(allmon))
    for i, nm in enumerate(names):
        for k, mo in enumerate(allmon):
            A[i, k] = mp.mpc(sp.N(maps[nm].get(mo, 0), 70))
    sv = mp.svd_c(A, compute_uv=False)
    svl = sorted([float(abs(x)) for x in sv], reverse=True)
    r = sum(1 for x in svl if x > 1e-40)
    return r, svl


ranks = {}
for label, names in {"M0..M6": [f"M{K}" for K in range(7)], "f0,f2,f4,f6": ["f0", "f2", "f4", "f6"],
                     "all": list(maps), "M0,M6": ["M0", "M6"], "N0,N6": ["N0", "N6"],
                     "M0,M6,N0,N6": ["M0", "M6", "N0", "N6"], "M0,N0": ["M0", "N0"], "M6,N6": ["M6", "N6"],
                     "M1,M3,M5": ["M1", "M3", "M5"], "M0,M2,M4,M6": ["M0", "M2", "M4", "M6"]}.items():
    r, sv = mrank(names)
    ranks[label] = {"rank": r, "singular_values": sv}
res["equivariant_map_ranks"] = ranks

# exact linear relations: express each M_K in the f_L basis (exact solve on the coefficient vectors)
fb = ["f0", "f2", "f4", "f6"]
Fm = sp.Matrix([[maps[nm].get(mo, 0) for mo in allmon] for nm in fb]).T
expr_in_f = {}
for nm in [f"M{K}" for K in range(7)] + ["N0", "N6"]:
    bvec = sp.Matrix([maps[nm].get(mo, 0) for mo in allmon])
    # least squares normal equations exactly, then verify residual 0
    A = (Fm.T * Fm).applyfunc(simp)
    rhs = (Fm.T * bvec).applyfunc(simp)
    sol = A.LUsolve(rhs).applyfunc(simp)
    resid = (Fm * sol - bvec).applyfunc(simp)
    exact_zero = all(t == 0 for t in resid)
    # independent numeric solve: every coefficient evaluated individually to 100 digits
    mp.mp.dps = 100
    An = mp.matrix([[mp.mpc(sp.N(maps[f_].get(mo, 0), 110)) for f_ in fb] for mo in allmon])
    bn = mp.matrix([mp.mpc(sp.N(maps[nm].get(mo, 0), 110)) for mo in allmon])
    AhA = An.H * An
    xs = mp.lu_solve(AhA, An.H * bn)
    rn = An * xs - bn
    num_resid = max(abs(rn[i]) for i in range(rn.rows))
    mp.mp.dps = 60
    assert num_resid < mp.mpf(10) ** -80, (nm, num_resid)  # a genuine residual would be O(1)
    sympy_agrees = all(abs(xs[k] - mp.mpc(sp.N(sol[k], 110))) < mp.mpf(10) ** -60 for k in range(4))
    expr_in_f[nm] = {"sympy_LUsolve_coeffs(UNRELIABLE unless agrees)": [str(t) for t in sol],
                     "sympy_LUsolve_agrees_with_100digit_solve": sympy_agrees,
                     "coeffs_numeric": [mp.nstr(xs[k], 20) for k in range(4)],
                     "residual_exactly_simplified_to_zero(sympy)": exact_zero,
                     "residual_max_abs_100digit_solve": mp.nstr(num_resid, 5)}
res["maps_in_basis_f0_f2_f4_f6"] = expr_in_f

(HERE / "audit_res_su2.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
