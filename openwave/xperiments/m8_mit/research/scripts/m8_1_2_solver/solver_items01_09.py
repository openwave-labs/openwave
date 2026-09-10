"""Items 1-9: SU(2)-only computations at spin 3 (exact, sympy)."""
from collections import Counter
from itertools import combinations_with_replacement, product

import sympy as sp

from solver_lib import CG, mvals, save_results, s

J3 = 3
MS = list(range(-3, 4))
out = {}

# ------------------------------------------------------------------ item 1
cnt = Counter(sum(t) for t in combinations_with_replacement(MS, 3))
dec1 = {J: cnt.get(J, 0) - cnt.get(J + 1, 0) for J in range(0, 10)}
dec1 = {J: mu for J, mu in dec1.items() if mu}
print("item 1: Sym^3 V_3 =", dec1, " dim check", sum(mu * (2 * J + 1) for J, mu in dec1.items()))
out["item01_Sym3V3"] = {f"V_{J}": mu for J, mu in dec1.items()}

# ------------------------------------------------------------------ item 2
sym2 = Counter(m1 + m2 for m1, m2 in combinations_with_replacement(MS, 2))
w = Counter()
for (a, ca), m3 in product(sym2.items(), MS):   # conj(V_3) has the same weights as V_3
    w[a + m3] += ca
dec2 = {J: w.get(J, 0) - w.get(J + 1, 0) for J in range(0, 10)}
print("item 2: Sym^2 V_3 (x) conj V_3 =", {J: m for J, m in dec2.items() if m})
out["item02"] = {"dim_Hom_J8": dec2[8], "dim_Hom_J3": dec2[3],
                 "full_decomposition": {f"V_{J}": m for J, m in dec2.items() if m}}

# ------------------------------------------------------------------ symbolic machinery
A = sp.symbols("a_m3:4")   # placeholder, rebuilt below
a = list(sp.symbols("u0:7"))       # u_m at index m+3
b = list(sp.symbols("ub0:7"))      # conj(u_m) at index m+3


def rho(a, b, K):
    """rho_K(u)_N = sum CG(3n;3n'|KN) (-1)^{n'} u_n conj(u_{-n'})  (a = u, b = conj u)."""
    o = [sp.S(0)] * (2 * K + 1)
    for n in MS:
        for npr in MS:
            N = n + npr
            if abs(N) > K:
                continue
            c = CG(3, n, 3, npr, K, N)
            if c:
                o[N + K] += c * sp.Integer(-1) ** npr * a[n + 3] * b[-npr + 3]
    return o


def norm2_rho(a, b, K):
    r = rho(a, b, K)
    rc = rho(b, a, K)          # conj(rho_N): CG real
    return sp.expand(sum(x * y for x, y in zip(r, rc)))


def Mmap(a, b, K):
    """M_K(u)_M = [rho_K(u) (x) u]_3 = sum CG(K N; 3 m | 3 M) rho_N u_m."""
    r = rho(a, b, K)
    o = [sp.S(0)] * 7
    for N in range(-K, K + 1):
        for m in MS:
            M = N + m
            if abs(M) > 3:
                continue
            c = CG(K, N, 3, m, 3, M)
            if c:
                o[M + 3] += c * r[N + K] * a[m + 3]
    return [sp.expand(x) for x in o]


def vec(d):
    v = [sp.S(0)] * 7
    for m, c in d.items():
        v[m + 3] = sp.S(c)
    return v


def conjv(v):
    return [sp.conjugate(x) for x in v]


nrm2 = sum(a[i] * b[i] for i in range(7))

# ------------------------------------------------------------------ item 3
M0 = Mmap(a, b, 0)
ratio = [sp.simplify(M0[i] / (a[i] * nrm2)) for i in range(7)]
print("item 3: M_0(u)_m / (u_m ||u||^2) =", set(ratio))
out["item03"] = {"M0_equals_c_norm2_u": len(set(ratio)) == 1, "c": s(ratio[0])}

# ------------------------------------------------------------------ item 4
# diagonality: M_6 of a general u has M_6(u)_M containing only monomials of weight M
M6 = Mmap(a, b, 6)
diag = []
for m in MS:
    v = vec({m: 1})
    img = [x.subs(dict(zip(a, v))).subs(dict(zip(b, v))) for x in M6]
    off = [img[i] for i in range(7) if i != m + 3]
    assert all(o == 0 for o in off)
    diag.append(sp.radsimp(img[m + 3]))
print("item 4: M_6(v_m) = lambda_m v_m, lambda_m (m=-3..3) =", diag)
print("        times sqrt7:", [sp.radsimp(x * sp.sqrt(7)) for x in diag],
      " times 7*sqrt7*... ratio to lambda_3:", [sp.nsimplify(x / diag[-1]) for x in diag])
# also M_0(v_m) for comparison
diag0 = []
for m in MS:
    v = vec({m: 1})
    diag0.append(sp.radsimp(M0[m + 3].subs(dict(zip(a, v))).subs(dict(zip(b, v)))))
# is the cubic map M_6 'diagonal on the weight basis' also in the sense that the linear operator
# A_6(v_m) is diagonal? record A_6(v_m) diagonal entries
# (A_K(u) v = [rho_K(u) (x) v]_3; since rho_K(v_m) has weight 0 the operator is diagonal)
A6diag = {}
for m in MS:
    v = vec({m: 1})
    r = [x.subs(dict(zip(a, v))).subs(dict(zip(b, v))) for x in rho(a, b, 6)]
    row = []
    for mm in MS:
        row.append(sp.radsimp(sum(CG(6, N, 3, mm, 3, mm) * r[N + 6] for N in [0])))
    A6diag[m] = row
out["item04"] = {"M6_diagonal_on_weight_basis": True,
                 "lambda_m_for_m=-3..3": [s(x) for x in diag],
                 "lambda_m_times_sqrt7": [s(sp.radsimp(x * sp.sqrt(7))) for x in diag],
                 "M0_diag_for_comparison": [s(x) for x in diag0],
                 "A6(v_m)_diagonal_entries_on_v_mm_rows_m": {str(m): [s(x) for x in r]
                                                            for m, r in A6diag.items()}}

# ------------------------------------------------------------------ item 5
u = vec({3: 1})
ub = conjv(u)
B2 = [0] * 5
for n in MS:
    for npr in MS:
        N = n + npr
        if abs(N) <= 2:
            B2[N + 2] += CG(3, n, 3, npr, 2, N) * u[n + 3] * u[npr + 3]
rho2 = [sp.radsimp(x) for x in rho(u, ub, 2)]
print("item 5: B_2(v_3) =", B2, "  rho_2(v_3) =", rho2)
out["item05"] = {"B2_v3": [s(x) for x in B2], "rho2_v3_components_N=-2..2": [s(x) for x in rho2]}

# ------------------------------------------------------------------ items 6, 7 helpers
r2 = sp.sqrt(2)
states = {
    "v3": vec({3: 1}),
    "v0": vec({0: 1}),
    "(v2+v-2)/sqrt2": vec({2: 1 / r2, -2: 1 / r2}),
    "(v3+v-3)/sqrt2": vec({3: 1 / r2, -3: 1 / r2}),
}
f6 = norm2_rho(a, b, 6)
rhat6 = f6 / nrm2 ** 2
grad_b = [sp.diff(rhat6, b[i]) for i in range(7)]   # Wirtinger d/d conj(u_m)
item6 = {}
for name, v in states.items():
    sub = {**dict(zip(a, v)), **dict(zip(b, conjv(v)))}
    val = sp.radsimp(f6.subs(sub))
    g = [sp.simplify(x.subs(sub)) for x in grad_b]
    crit = all(x == 0 for x in g)
    print(f"item 6: {name}: 924*||rho_6||^2 = {sp.radsimp(924 * val)} ; rhat6 = {val}; "
          f"gradient zero: {crit}  grad = {g}")
    item6[name] = {"924_norm2_rho6": s(sp.radsimp(924 * val)), "norm2_rho6": s(val),
                   "Wirtinger_gradient_of_rhat6": [s(x) for x in g], "critical": crit}
out["item06"] = item6

# item 7
v = states["(v2+v-2)/sqrt2"]
sub = {**dict(zip(a, v)), **dict(zip(b, conjv(v)))}
row7 = [sp.radsimp(norm2_rho(a, b, K).subs(sub)) for K in range(7)]
print("item 7: ||rho_K||^2, K=0..6 at (v2+v-2)/sqrt2:", row7, " sum =", sp.nsimplify(sum(row7)))
out["item07"] = {"norm2_rhoK_K=0..6": [s(x) for x in row7], "sum": s(sp.nsimplify(sum(row7)))}

# also record the full row at all four states (used later)
rows = {}
for name, v in states.items():
    sub = {**dict(zip(a, v)), **dict(zip(b, conjv(v)))}
    rows[name] = [sp.radsimp(norm2_rho(a, b, K).subs(sub)) for K in range(7)]
    print("        row at", name, rows[name])
out["item07_extra_rows_all_four_states"] = {k: [s(x) for x in r] for k, r in rows.items()}

# ------------------------------------------------------------------ item 8
t, S = sp.symbols("t s", real=True)
c_, s_ = sp.symbols("c_ s_", real=True)
v = vec({2: c_, -3: s_})
sub = {**dict(zip(a, v)), **dict(zip(b, v))}
f8 = sp.expand(f6.subs(sub))
# rewrite in s = sin^2 t: f8 is a polynomial in c_^2, s_^2 (check) ; substitute c_^2 = 1 - s
P8 = sp.Poly(f8, c_, s_)
assert all(e1 % 2 == 0 and e2 % 2 == 0 for (e1, e2) in P8.monoms()), P8
f8s = sp.expand(sum(coef * (1 - S) ** (e1 // 2) * S ** (e2 // 2) for (e1, e2), coef in P8.terms()))
print("item 8: ||rho_6||^2(s) =", f8s, " = ", sp.factor(f8s))
d8 = sp.diff(f8s, S)
crit8 = [r for r in sp.solve(sp.Eq(d8, 0), S)]
crit8_in = [r for r in crit8 if r.is_real and 0 < r < 1]
print("        stationary s:", crit8, " interior:", crit8_in,
      " values:", [sp.nsimplify(sp.radsimp(f8s.subs(S, r))) for r in crit8_in])
ends = {"s=0": f8s.subs(S, 0), "s=1": f8s.subs(S, 1)}
print("        endpoints:", ends)
# second derivative (nature of stationary point)
out["item08"] = {"norm2_rho6_of_s": s(f8s), "factored": s(sp.factor(f8s)),
                 "stationary_points_all_roots_of_derivative": [s(r) for r in crit8],
                 "interior_stationary_points": [
                     {"s": s(r), "value": s(sp.radsimp(f8s.subs(S, r))),
                      "second_derivative": s(sp.radsimp(sp.diff(f8s, S, 2).subs(S, r))),
                      "float": float(f8s.subs(S, r))} for r in crit8_in],
                 "endpoint_values": {k: s(x) for k, x in ends.items()}}
# criticality of the item-8 stationary rays on the full P(V_3) (extra)
for r in crit8_in:
    cval, sval = sp.sqrt(1 - r), sp.sqrt(r)
    vv = vec({2: cval, -3: sval})
    sb = {**dict(zip(a, vv)), **dict(zip(b, vv))}
    g = [sp.simplify(x.subs(sb)) for x in grad_b]
    print("        item-8 stationary ray, full-P^6 Wirtinger gradient:", g)
    out["item08"].setdefault("full_P6_gradient_at_interior_stationary", []).append([s(x) for x in g])

# ------------------------------------------------------------------ item 9
x, y = sp.symbols("x y", real=True)
zz = sp.symbols("zz")
zb = sp.symbols("zb")
v = vec({3: 1, 0: 1, -3: 1})
va = list(v)
vb = list(v)
va[3] = zz
vb[3] = zb
f9c = sp.expand(f6.subs({**dict(zip(a, va)), **dict(zip(b, vb))}))
f9 = sp.expand(f9c.subs({zz: x + sp.I * y, zb: x - sp.I * y}))
n9 = 2 + x ** 2 + y ** 2
print("item 9: ||rho_6||^2 on the slice (in z, zbar):", sp.factor(f9c))
print("        in x,y:", f9, " ; factor:", sp.factor(f9))
R9 = f9 / n9 ** 2
# numerators of the gradient of rhat
gx = sp.factor(sp.numer(sp.together(sp.diff(R9, x))))
gy = sp.factor(sp.numer(sp.together(sp.diff(R9, y))))
print("        d/dx rhat numerator:", gx)
print("        d/dy rhat numerator:", gy)
G = sp.groebner([gx, gy], x, y, order="lex")
print("        Groebner (lex) of rhat gradient:", G.exprs)
sols = sp.solve(G.exprs, [x, y], dict=True)
real_sols = [d for d in sols if all(sp.im(sp.nsimplify(val)) == 0 for val in d.values())]
print("        all complex solutions:", sols)
crit9 = []
for d in sols:
    xv, yv = d[x], d[y]
    if sp.simplify(sp.im(xv)) != 0 or sp.simplify(sp.im(yv)) != 0:
        continue
    val = sp.radsimp(sp.simplify(R9.subs({x: xv, y: yv})))
    H = sp.hessian(R9, (x, y)).subs({x: xv, y: yv})
    H = sp.simplify(H)
    ev = [sp.nsimplify(sp.simplify(e)) for e in H.eigenvals()]
    crit9.append({"x": s(xv), "y": s(yv), "rhat6": s(val), "float": float(val),
                  "hessian_eigs": [s(e) for e in ev]})
print("        real critical points of rhat6 on chart:", crit9)
# unnormalised
Gu = sp.groebner([sp.diff(f9, x), sp.diff(f9, y)], x, y, order="lex")
solsu = sp.solve(Gu.exprs, [x, y], dict=True)
crit9u = []
for d in solsu:
    xv, yv = d[x], d[y]
    if sp.simplify(sp.im(xv)) != 0 or sp.simplify(sp.im(yv)) != 0:
        continue
    crit9u.append({"x": s(xv), "y": s(yv), "norm2_rho6": s(sp.simplify(f9.subs({x: xv, y: yv}))),
                   "rhat6_there": s(sp.simplify(R9.subs({x: xv, y: yv})))})
print("        Groebner unnormalised:", Gu.exprs)
print("        real critical points of unnormalised ||rho6||^2 on slice:", crit9u)
# point at infinity [v_0]: chart w = 1/z, u ~ w v3 + v0 + w v-3
wr, wi = sp.symbols("wr wi", real=True)
ww, wb = sp.symbols("ww wb")
va = [ww, 0, 0, 1, 0, 0, ww]
vb = [wb, 0, 0, 1, 0, 0, wb]
fw = sp.expand(f6.subs({**dict(zip(a, va)), **dict(zip(b, vb))}).subs({ww: wr + sp.I * wi,
                                                                     wb: wr - sp.I * wi}))
Rw = fw / (1 + 2 * (wr ** 2 + wi ** 2)) ** 2
gw = [sp.simplify(sp.diff(Rw, q).subs({wr: 0, wi: 0})) for q in (wr, wi)]
Hw = sp.simplify(sp.hessian(Rw, (wr, wi)).subs({wr: 0, wi: 0}))
print("        at [v_0] (w=0): rhat6 =", Rw.subs({wr: 0, wi: 0}), " gradient:", gw, " Hessian:", Hw)
# full-P^6 criticality of the chart critical points
full9 = []
for c in crit9 + [{"x": "oo", "y": "oo"}]:
    if c["x"] == "oo":
        vv = vec({0: 1})
    else:
        zv = sp.sympify(c["x"]) + sp.I * sp.sympify(c["y"])
        vv = vec({3: 1, 0: zv, -3: 1})
    sb = {**dict(zip(a, vv)), **dict(zip(b, conjv(vv)))}
    g = [sp.simplify(q.subs(sb)) for q in grad_b]
    full9.append({"point": (c["x"], c["y"]), "full_P6_gradient_zero": all(q == 0 for q in g)})
print("        full P^6 criticality:", full9)
out["item09"] = {"norm2_rho6_z_zbar": s(sp.factor(f9c)), "norm2_rho6_xy": s(f9),
                 "rhat6_xy": s(sp.factor(R9)),
                 "rhat_grad_numerators": [s(gx), s(gy)],
                 "rhat_groebner_lex": [s(e) for e in G.exprs],
                 "rhat_critical_points_real": crit9,
                 "unnormalised_groebner_lex": [s(e) for e in Gu.exprs],
                 "unnormalised_critical_points_real": crit9u,
                 "point_at_infinity_v0": {"rhat6": s(Rw.subs({wr: 0, wi: 0})),
                                          "gradient_in_w_chart": [s(q) for q in gw],
                                          "hessian_in_w_chart": s(Hw)},
                 "full_P6_criticality": full9}
save_results("items01_09", out)
