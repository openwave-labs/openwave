"""Stage 2 exact follow-up.
 (a) exact root multiset of the degree-12 Majorana form F_R of R_6(P):
     candidate points (from the FLOAT roots of solver_stage2.py) are the 12 points
     (+-a, 0, +-b), (0, +-b, +-a), (+-b, +-a, 0), a = sqrt((5-sqrt5)/10), b = sqrt((5+sqrt5)/10);
     each is mapped back by inverse stereographic projection r = (x + i y)/(1 + z) and F_R(r) = 0
     is verified EXACTLY (simplification to 0, confirmed by minimal polynomial = the variable).
     With deg F_R = 12 and gcd(F_R, F_R') = 1 (solver_stage2.py) this fixes the multiset exactly.
 (b) consistency of item 19c with stage 1: L_8(u) = [rho_6(u) (x) u]_8 = -(1/2) C_0(u) identically.
"""
from collections import Counter
from itertools import combinations, product

import sympy as sp

from solver_lib import CG, B_ab, theta_ab, rho_ab, save_results, s

z = sp.symbols("z")
fmon = (z**12 - 22*sp.sqrt(5)*z**10/5 - 33*z**8 + 44*sp.sqrt(5)*z**6/5 - 33*z**4
        - 22*sp.sqrt(5)*z**2/5 + 1)
a = sp.sqrt((5 - sp.sqrt(5)) / 10)
b = sp.sqrt((5 + sp.sqrt(5)) / 10)
cands = []
for s1, s2 in product((1, -1), repeat=2):
    cands += [(s1 * a, 0, s2 * b), (0, s1 * b, s2 * a), (s1 * b, s2 * a, 0)]
assert len(set(cands)) == 12
res = []
t = sp.Symbol("t")
for (x, y, zz) in cands:
    r = sp.radsimp((x + sp.I * y) / (1 + zz))
    val = sp.expand(fmon.subs(z, r))
    zero = sp.simplify(val) == 0
    mp = sp.minimal_polynomial(val, t) if not zero else t
    res.append({"xyz": (x, y, zz), "root": r, "F_at_root_is_zero": zero or mp == t,
                "root_float": complex(sp.N(r, 20))})
    print(f"point {(x, y, zz)}: r = {r}  ~ {complex(sp.N(r, 12))};  F(r) = 0 exactly: {zero or mp == t}")
allzero = all(q["F_at_root_is_zero"] for q in res)
dots = [sp.simplify(sp.radsimp(sum(p * q for p, q in zip(u["xyz"], v["xyz"])))) for u, v in combinations(res, 2)]
dc = Counter(dots)
norms = {sp.simplify(sum(c * c for c in u["xyz"])) for u in res}
print("all 12 candidates are exact roots:", allzero, "; unit norms:", norms)
print("pairwise dot products (exact multiset):", dict(dc))
# minimal chord / nearest neighbours: 5 neighbours per point at dot 1/sqrt5
nn = Counter(sum(1 for v in res if v is not u and
                 sp.simplify(sum(p * q for p, q in zip(u["xyz"], v["xyz"])) - 1 / sp.sqrt(5)) == 0) for u in res)
print("number of neighbours at dot 1/sqrt5 per point:", dict(nn))

# (b)
A = list(sp.symbols("u0:7"))
B = list(sp.symbols("ub0:7"))


def couple(X, jx, Y, jy, J):
    o = [sp.S(0)] * (2 * J + 1)
    for m1 in range(-jx, jx + 1):
        for m2 in range(-jy, jy + 1):
            M = m1 + m2
            if abs(M) <= J and X[m1 + jx] != 0 and Y[m2 + jy] != 0:
                o[M + J] += CG(jx, m1, jy, m2, J, M) * X[m1 + jx] * Y[m2 + jy]
    return [sp.expand(x) for x in o]


C0 = couple(B_ab(A, 6), 6, theta_ab(B), 3, 8)
L8 = couple(rho_ab(A, B, 6), 6, A, 3, 8)
ident = all(sp.expand(L8[i] + C0[i] / 2) == 0 for i in range(17))
print("L_8(u) == -(1/2) C_0(u) identically:", ident)
save_results("stage2_exact_roots", {
    "candidate_points_xyz": [[s(c) for c in q["xyz"]] for q in res],
    "roots_exact": [s(q["root"]) for q in res],
    "roots_float": [[q["root_float"].real, q["root_float"].imag] for q in res],
    "all_twelve_are_exact_roots": allzero,
    "multiplicities": "each 1 (12 distinct exact roots of a degree-12 form; also gcd(F,F')=1)",
    "unit_norms": [s(x) for x in norms],
    "pairwise_dots_exact": {s(k): v for k, v in dc.items()},
    "neighbours_at_dot_1_over_sqrt5": {str(k): v for k, v in nn.items()},
    "L8_equals_minus_half_C0_identity": ident})
