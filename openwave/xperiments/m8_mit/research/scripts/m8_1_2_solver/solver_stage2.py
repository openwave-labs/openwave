"""Stage 2: item 19 (continued) and item 20.

19c: L_8(u) = [rho_6(u) (x) u]_8 with CG <6 N; 3 m | 8 M>; report (L_8(v_3))_{M=3}.
20 : R_6(P) as a degree-12 Majorana form F(z) = sum_m (-1)^{6-m} sqrt(C(12,6+m)) R_m z^{6-m}
     (the handout's spin-3 formula with 3 -> 6); exact squarefreeness; root positions;
     injectivity of v -> J(F_R, F_v) (first transvectant), from multiplicities and by exact rank.
"""
from itertools import combinations

import numpy as np
import sympy as sp

from solver_lib import CG, rho_ab, transform, norm2, load_P, save_results, s

out = {}
# ------------------------------------------------------------------ 19 continued
v3 = [0] * 6 + [1]
r6 = rho_ab(v3, v3, 6)
L8 = [sp.S(0)] * 17
for N in range(-6, 7):
    if r6[N + 6] == 0:
        continue
    for m in range(-3, 4):
        M = N + m
        if abs(M) <= 8 and v3[m + 3] != 0:
            L8[M + 8] += CG(6, N, 3, m, 8, M) * r6[N + 6] * v3[m + 3]
L8 = [sp.radsimp(x) for x in L8]
print("rho_6(v_3) nonzero:", {N: sp.radsimp(r6[N + 6]) for N in range(-6, 7) if r6[N + 6] != 0})
print("CG(6 0; 3 3 | 8 3) =", CG(6, 0, 3, 3, 8, 3))
print("L_8(v_3) nonzero components:", {M: L8[M + 8] for M in range(-8, 9) if L8[M + 8] != 0})
l83 = L8[3 + 8]
print("(L_8(v_3))_3 =", l83, "~", float(l83))
out["item19c"] = {"rho6_v3_N0": s(sp.radsimp(r6[6])), "CG_60_33_83": s(CG(6, 0, 3, 3, 8, 3)),
                  "L8_v3_components_nonzero": {str(M): s(L8[M + 8]) for M in range(-8, 9) if L8[M + 8] != 0},
                  "L8_v3_N3": s(l83), "L8_v3_N3_float": float(l83)}

# ------------------------------------------------------------------ 20
z, X, Y = sp.symbols("z X Y")


def majorana(vec, j):
    return sp.expand(sum(sp.Integer(-1) ** (j - m) * sp.sqrt(sp.binomial(2 * j, j + m)) * vec[m + j] * z ** (j - m)
                         for m in range(-j, j + 1)))


def homog(f, deg):
    P = sp.Poly(f, z)
    return sp.expand(sum(c * X ** e[0] * Y ** (deg - e[0]) for e, c in P.terms()))


item20 = {}
forms = {}
for key in ("d=4", "d=3"):
    P = load_P(key)
    R6 = [sp.radsimp(sp.expand(x)) for x in transform(P, 6)]
    f = majorana(R6, 6)
    Pf = sp.Poly(f, z)
    deg = Pf.degree()
    lc = Pf.LC()
    fm = sp.Poly(sp.expand(f / lc), z)
    coeffs = [sp.nsimplify(sp.radsimp(c)) if False else sp.simplify(sp.radsimp(c)) for c in fm.all_coeffs()]
    fmon = sp.expand(sum(c * z ** (len(coeffs) - 1 - i) for i, c in enumerate(coeffs)))
    forms[key] = (R6, fmon, deg)
    print(f"\n{key}: R_6 = {R6}")
    print(f"   degree of F_R in z: {deg}; leading coefficient {sp.simplify(lc)}")
    print(f"   monic form: {fmon}")
    item20[key] = {"R6_components_N=-6..6": [s(x) for x in R6], "norm2_R6": s(sp.simplify(norm2(R6))),
                   "degree_in_z": deg, "roots_at_infinity_south_pole": 12 - deg,
                   "leading_coefficient": s(sp.simplify(lc)), "monic_form": s(fmon)}

# proportionality of the two sectors' R_6
R4, R3 = forms["d=4"][0], forms["d=3"][0]
i0 = next(i for i in range(13) if R4[i] != 0)
ratio = sp.simplify(R3[i0] / R4[i0])
prop = all(sp.simplify(R3[i] - ratio * R4[i]) == 0 for i in range(13))
print("\nR_6(d=3) = c R_6(d=4):", prop, " c =", ratio)
item20["sectors_R6_proportional"] = prop
item20["ratio_R6_d3_over_d4"] = s(ratio)

# exact squarefreeness of the monic form (use d=4; d=3 identical up to scale if proportional)
fmon, deg = forms["d=4"][1], forms["d=4"][2]
dom_ext = [sp.sqrt(5), sp.I]
try:
    Pm = sp.Poly(fmon, z, extension=dom_ext)
    dom = str(Pm.get_domain())
    g = sp.gcd(Pm, Pm.diff(z))
    disc = sp.discriminant(Pm)
    print("domain:", dom, " gcd(f, f') =", g.as_expr(), " discriminant =", sp.simplify(disc.as_expr() if hasattr(disc, 'as_expr') else disc))
    item20["exact_domain"] = dom
    item20["gcd_f_fprime"] = s(g.as_expr())
    item20["discriminant"] = s(sp.simplify(disc.as_expr() if hasattr(disc, 'as_expr') else disc))
    fac = sp.factor_list(fmon, extension=dom_ext)
    print("factor_list over Q(sqrt5, i):", fac)
    item20["factor_list_Q_sqrt5_i"] = s(fac)
except Exception as e:  # record and continue
    print("exact domain step failed:", repr(e))
    item20["exact_domain_error"] = repr(e)

# FLOAT roots
cf = [complex(sp.N(c, 50)) for c in sp.Poly(fmon, z).all_coeffs()]
rts = np.roots(cf)
pts = []
for r in rts:
    a2 = abs(r) ** 2
    pts.append(np.array([2 * r.real, 2 * r.imag, 1 - a2]) / (1 + a2))
for _ in range(12 - deg):
    pts.append(np.array([0.0, 0.0, -1.0]))
dots = sorted(round(float(p @ q), 10) for p, q in combinations(pts, 2))
from collections import Counter
dc = Counter(dots)
mind = min(np.linalg.norm(p - q) for p, q in combinations(pts, 2))
print("FLOAT roots:", np.round(rts, 10))
print("FLOAT points (x,y,z):", [tuple(np.round(p, 10)) for p in pts])
print("FLOAT pairwise dot multiset:", dict(dc), " 1/sqrt5 =", 1 / 5 ** 0.5, " min chord:", mind)
item20["FLOAT_roots"] = [[float(r.real), float(r.imag)] for r in rts]
item20["FLOAT_points_xyz"] = [[float(c) for c in p] for p in pts]
item20["FLOAT_pairwise_dots"] = {str(k): v for k, v in dc.items()}
item20["FLOAT_min_chord"] = float(mind)

# ------------------------------------------------------------------ transvectant map v -> J(F_R, F_v)
FR = homog(fmon, 12)
rows = []
for m in range(-3, 4):
    e = [0] * 7
    e[m + 3] = 1
    Fv = homog(majorana(e, 3), 6)
    Jv = sp.expand(sp.diff(FR, X) * sp.diff(Fv, Y) - sp.diff(FR, Y) * sp.diff(Fv, X))
    PJ = sp.Poly(Jv, X, Y)
    rows.append([PJ.coeff_monomial(X ** k * Y ** (16 - k)) for k in range(17)])
Mt = sp.Matrix(rows).T   # 17 x 7
rk = Mt.rank(simplify=True)
sv = np.linalg.svd(np.array([[complex(sp.N(e, 30)) for e in row] for row in Mt.tolist()]), compute_uv=False)
print("exact (symbolic) rank of v -> J(F_R, F_v):", rk, " FLOAT singular values:", sv)
# equivalence with the equivariant coupling [R_6 (x) v]_8: rank of that 17x7 matrix too
rowsC = []
for m in range(-3, 4):
    col = [sp.S(0)] * 17
    for N in range(-6, 7):
        M = N + m
        if abs(M) <= 8 and R4[N + 6] != 0:
            col[M + 8] += CG(6, N, 3, m, 8, M) * R4[N + 6]
    rowsC.append(col)
MC = sp.Matrix(rowsC).T
rkC = MC.rank(simplify=True)
print("exact rank of v -> [R_6 (x) v]_8:", rkC)
item20["transvectant_map_exact_rank"] = int(rk)
item20["transvectant_map_FLOAT_singular_values"] = [float(x) for x in sv]
item20["coupling_R6_v_to_8_exact_rank"] = int(rkC)
out["item20"] = item20
save_results("stage2", out)
