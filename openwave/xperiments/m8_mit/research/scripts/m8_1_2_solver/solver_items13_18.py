"""Items 13, 14, 17, 18: weights w_K, the projected cubic, Q_d as a function of rhat_6, beta.

Derivation (written out in solver_stage1_return.md) gives
   |psi(g)|^2 = sum_K rho_K(u)^T D^K(g) R_K,       R_K = M_K(P),
   B = d ||u||^2 / 7,   A = sum_K ||rho_K(u)||^2 ||R_K||^2 / (2K+1),
   w_K = 49 ||R_K||^2 / ((2K+1) d^2),
   N(u) = (1/d) sum_K c_K M_K(u),   c_K = Tr(P A(R_K)),  A(X) v = [X (x) v]_3.
This script evaluates them exactly and checks the identities exactly
(factorisation at a rational SU(2) element; (d/7)<N(u),u> = A(u) as a polynomial identity).
"""
import random
from fractions import Fraction as Fr

import sympy as sp

from solver_lib import (Quat, D_mono, mono_to_orthonormal, transform, norm2, A_op_matrix, rho_ab,
                        M_ab, load_P, save_results, s)

a = list(sp.symbols("u0:7"))
b = list(sp.symbols("ub0:7"))
nrm2 = sum(a[i] * b[i] for i in range(7))


def poly_zero(e):
    P = sp.Poly(sp.expand(e), *a, *b)
    return all(sp.simplify(sp.radsimp(c)) == 0 for c in P.coeffs())


rn2 = {K: sp.expand(sum(x * y for x, y in zip(rho_ab(a, b, K), rho_ab(b, a, K)))) for K in range(7)}
Ms = {K: M_ab(a, b, K) for K in range(7)}
pair = {K: sp.expand(sum(Ms[K][i] * b[i] for i in range(7))) for K in range(7)}   # <M_K(u), u>

out = {}
# structural facts independent of the sector
rhat0_ok = poly_zero(rn2[0] - nrm2 ** 2 / 7)
print("||rho_0(u)||^2 = ||u||^4/7 :", rhat0_ok)
# gradient of ||rho_K||^2 w.r.t. conj(u) versus M_K(u)
grad_rel = {}
for K in range(7):
    g = [sp.expand(sp.diff(rn2[K], b[m])) for m in range(7)]
    # find constant k with g = k * M_K
    i0 = 6
    P1 = sp.Poly(g[i0], *a, *b)
    P2 = sp.Poly(Ms[K][i0], *a, *b)
    mono = P2.monoms()[0]
    k = sp.simplify(P1.coeff_monomial(mono) / P2.coeff_monomial(mono)) if P2.coeff_monomial(mono) != 0 else None
    ok = k is not None and all(poly_zero(g[m] - k * Ms[K][m]) for m in range(7))
    pk = sp.simplify(sp.radsimp(sp.Poly(pair[K], *a, *b).coeff_monomial(sp.Poly(rn2[K], *a, *b).monoms()[0])
                                / sp.Poly(rn2[K], *a, *b).coeffs()[0])) if rn2[K] != 0 else None
    okp = pk is not None and poly_zero(pair[K] - pk * rn2[K])
    grad_rel[K] = {"d||rho_K||^2/d conj(u) = k M_K(u)": ok, "k": s(k),
                   "<M_K(u),u> = p ||rho_K||^2": okp, "p": s(pk)}
    print(f"K={K}: grad_ubar ||rho_K||^2 = {k} * M_K(u): {ok};  <M_K(u),u> = {pk} ||rho_K||^2: {okp}")
out["structure"] = {"norm2_rho0_is_norm4_over_7": rhat0_ok,
                    "gradient_and_pairing_relations": {str(K): v for K, v in grad_rel.items()}}

# rank of the span of M_0..M_6 as cubic maps (exact + float)
mon_keys = sorted({(m, mono) for K in range(7) for m in range(7)
                   for mono in sp.Poly(Ms[K][m], *a, *b).monoms()} if True else [])
def coeff_row(maps):
    row = []
    for (m, mono) in mon_keys:
        P = sp.Poly(maps[m], *a, *b)
        row.append(P.coeff_monomial(mono))
    return row
Mrows = sp.Matrix([coeff_row(Ms[K]) for K in range(7)])
rank_M = Mrows.rank(simplify=True)
import numpy as np
sv = np.linalg.svd(np.array(Mrows.evalf(30).tolist(), dtype=float), compute_uv=False)
print("rank span{M_0..M_6} (exact):", rank_M, " singular values (float):", sv)
M6_prop_M0 = Sp = sp.Matrix([coeff_row(Ms[0]), coeff_row(Ms[6])]).rank(simplify=True)
print("rank span{M_0, M_6}:", M6_prop_M0)
out["span_of_MK"] = {"rank_M0_to_M6": int(rank_M), "rank_M0_M6": int(M6_prop_M0),
                     "singular_values_float": [float(x) for x in sv]}

# rational SU(2) element for exact factorisation checks
gq = Quat(Fr(1, 5), Fr(2, 5), Fr(2, 5), Fr(4, 5))
assert gq.norm2() == 1
DK = {K: mono_to_orthonormal(D_mono(gq.su2(), 2 * K), 2 * K) for K in range(7)}
rnd = random.Random(3)
uval = [sp.Rational(rnd.randint(-9, 9), rnd.randint(1, 5)) + sp.I * sp.Rational(rnd.randint(-9, 9), rnd.randint(1, 5))
        for _ in range(7)]
uvec = sp.Matrix(uval)

rays = {
    "v3": {6: 1}, "v0": {3: 1}, "(v2+v-2)/sqrt2": {5: 1, 1: 1}, "(v3+v-3)/sqrt2": {6: 1, 0: 1},
    "item8 (sqrt13/5)v2+(2sqrt3/5)v-3": {5: sp.sqrt(13) / 5, 0: 2 * sp.sqrt(3) / 5},
    "item9 z=i sqrt10/2": {6: 1, 3: sp.I * sp.sqrt(10) / 2, 0: 1},
    "item9 z=sqrt230/10": {6: 1, 3: sp.sqrt(230) / 10, 0: 1},
    "item9 unnormalised-critical z=sqrt10/10 (not a rhat critical point)": {6: 1, 3: sp.sqrt(10) / 10, 0: 1},
}


def subs_for(d):
    v = [sp.S(d.get(i, 0)) for i in range(7)]
    return {**dict(zip(a, v)), **dict(zip(b, [sp.conjugate(x) for x in v]))}, v


sector_data = {}
for key in ("d=4", "d=3"):
    P = load_P(key)
    d = sp.simplify(P.trace())
    R = {K: [sp.radsimp(x) for x in transform(P, K)] for K in range(7)}
    RN = {K: sp.simplify(norm2(R[K])) for K in range(7)}
    wK = {K: sp.simplify(49 * RN[K] / ((2 * K + 1) * d ** 2)) for K in range(7)}
    cK = {K: sp.simplify(sp.radsimp((P * A_op_matrix(R[K], K)).trace())) for K in range(7)}
    # exact factorisation check at the rational element
    lhs = sp.expand((uvec.T * DK[3] * P * DK[3].H * uvec.conjugate())[0, 0])
    rhs = 0
    for K in range(7):
        rk = rho_ab(uval, [sp.conjugate(x) for x in uval], K)
        rhs += sum(rk[M] * DK[K][M, N] * R[K][N] for M in range(2 * K + 1) for N in range(2 * K + 1))
    fact_ok = sp.simplify(sp.radsimp(sp.expand(lhs - rhs))) == 0
    # (d/7) <N(u),u> = A(u)  with N = (1/d) sum c_K M_K   <=>  (1/7) sum c_K <M_K,u> = sum ||rho_K||^2 ||R_K||^2/(2K+1)
    ident_ok = poly_zero(sum(cK[K] * pair[K] for K in range(7)) / 7
                         - sum(rn2[K] * RN[K] / (2 * K + 1) for K in range(7)))
    # Q_d in terms of rhat_6
    Qslope = wK[6]
    Qconst = sp.simplify(wK[0] / 7)
    print(f"\n{key}: d = {d}; ||R_K||^2 = {[RN[K] for K in range(7)]}")
    print(f"   w_K = {[wK[K] for K in range(7)]};  w_0 = {wK[0]}, w_6/w_0 = {sp.simplify(wK[6] / wK[0])},"
          f" N = 924/w_6 = {sp.simplify(924 / wK[6])}")
    print(f"   c_K = Tr(P A(R_K)) = {[cK[K] for K in range(7)]};  coefficients c_K/d = "
          f"{[sp.simplify(cK[K] / d) for K in range(7)]}")
    print(f"   c_K / ||R_K||^2 (K=0,6): {sp.simplify(cK[0] / RN[0])}, {sp.simplify(cK[6] / RN[6])}")
    print(f"   exact factorisation |psi|^2 = sum_K rho_K^T D^K R_K at g=(1,2,2,4)/5: {fact_ok}")
    print(f"   exact identity (d/7)<N(u),u> = A(u): {ident_ok}")
    print(f"   Q_d = {Qconst} + {Qslope} * rhat_6")
    # eigen-rays and beta
    betas = {}
    for rname, dct in rays.items():
        sb, v = subs_for(dct)
        nv2 = sp.simplify(sum(x * sp.conjugate(x) for x in v))
        Nu = [sp.simplify(sum(cK[K] * Ms[K][m] for K in (0, 6)).subs(sb) / d) for m in range(7)]
        lam = sp.simplify(sum(Nu[m] * sp.conjugate(v[m]) for m in range(7)) / nv2)
        eig = all(sp.simplify(Nu[m] - lam * v[m]) == 0 for m in range(7))
        rh6 = sp.simplify(rn2[6].subs(sb) / nv2 ** 2)
        Qd = sp.simplify(Qconst + Qslope * rh6)
        beta_B1 = sp.simplify(lam * (7 / d) / nv2) if eig else None     # rescale u so that ||u||^2 = 7/d
        betas[rname] = {"eigenray_of_N": eig, "rhat6": s(rh6), "Q_d": s(Qd),
                        "beta_at_B=1": s(beta_B1), "beta_at_unit_fibre_norm": s(sp.simplify(lam / nv2)) if eig else None,
                        "beta_B1_equals_Q_d": (sp.simplify(beta_B1 - Qd) == 0) if eig else None}
        print(f"   ray {rname}: eigen {eig}; rhat6 = {rh6}; Q_d = {Qd}; beta(B=1) = {beta_B1}")
    sector_data[key] = {"d": s(d), "norm2_RK": [s(RN[K]) for K in range(7)], "w_K": [s(wK[K]) for K in range(7)],
                        "w_0": s(wK[0]), "w6_over_w0": s(sp.simplify(wK[6] / wK[0])),
                        "N": s(sp.simplify(924 / wK[6])), "c_K": [s(cK[K]) for K in range(7)],
                        "item14_coefficients_cK_over_d": [s(sp.simplify(cK[K] / d)) for K in range(7)],
                        "cK_over_norm2RK_K0_K6": [s(sp.simplify(cK[0] / RN[0])), s(sp.simplify(cK[6] / RN[6]))],
                        "factorisation_check_exact": fact_ok, "identity_dN_u_equals_A_exact": ident_ok,
                        "Q_d_constant": s(Qconst), "Q_d_slope_in_rhat6": s(Qslope), "beta": betas,
                        "_wK6": wK[6]}
d4, d3 = sector_data["d=4"], sector_data["d=3"]
diff_slope = sp.simplify(d3.pop("_wK6") - d4.pop("_wK6"))
print("\nbeta_{d=3} - beta_{d=4} at the same ray (B=1 in each sector) =", diff_slope, "* rhat_6([u])")
per_ray = {}
for rname in rays:
    b3, b4 = d3["beta"][rname]["beta_at_B=1"], d4["beta"][rname]["beta_at_B=1"]
    u3, u4 = d3["beta"][rname]["beta_at_unit_fibre_norm"], d4["beta"][rname]["beta_at_unit_fibre_norm"]
    if b3 != "None" and b4 != "None":
        db = sp.simplify(sp.sympify(b3) - sp.sympify(b4))
        du = sp.simplify(sp.sympify(u3) - sp.sympify(u4))
        rh = sp.sympify(d3["beta"][rname]["rhat6"])
        per_ray[rname] = {"beta3_minus_beta4_at_B=1": s(db), "check_equals_49/156*rhat6": sp.simplify(db - diff_slope * rh) == 0,
                          "beta3_minus_beta4_at_unit_fibre_norm": s(du),
                          "check_equals_-1/7+rhat6/13": sp.simplify(du - (-sp.Rational(1, 7) + rh / 13)) == 0}
        print(f"   {rname}: beta3-beta4 (B=1) = {db};  (unit fibre norm) = {du}")
out["sectors"] = sector_data
out["beta_difference_d3_minus_d4_coefficient_of_rhat6"] = s(diff_slope)
out["beta_differences_per_ray"] = per_ray
out["w6_over_w0_general_formula"] = "7 ||R_6||^2 / (13 d^2)  (= 12/(13 d^2) since ||R_6||^2 = 12/7 in both sectors)"
save_results("items13_18", out)
