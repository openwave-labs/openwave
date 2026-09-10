"""Refutation phase: item-by-item exact comparison of the solver's JSON with mine.

Reads solver_copy/solver_results.json (read-only) and audit_results.json. Every comparison is an
exact sympy equality (or a 1e-9 float multiset match for point sets) and is recorded, pass or fail.
"""
import json
import re
import pathlib
from collections import Counter
import sympy as sp

HERE = pathlib.Path(__file__).parent
S = json.loads((HERE / "solver_copy" / "solver_results.json").read_text())
A = json.loads((HERE / "audit_results.json").read_text())
out = {}


def r5(x):
    x = x.strip()
    m = re.fullmatch(r"\[\((-?[\d/]+)\+(-?[\d/]+)\*s5\) \+ i\((-?[\d/]+)\+(-?[\d/]+)\*s5\)\]", x)
    if m:
        a, b, c, d = [sp.Rational(t) for t in m.groups()]
        return a + b * sp.sqrt(5) + sp.I * (c + d * sp.sqrt(5))
    m = re.fullmatch(r"\((-?[\d/]+)\+(-?[\d/]+)\*s5\)", x)
    if m:
        return sp.Rational(m.group(1)) + sp.Rational(m.group(2)) * sp.sqrt(5)
    return sp.sympify(x)


def eq(a, b):
    a, b = (r5(a) if isinstance(a, str) else sp.S(a)), (r5(b) if isinstance(b, str) else sp.S(b))
    return sp.simplify(sp.radsimp(a - b)) == 0


def eqlist(la, lb):
    return len(la) == len(lb) and all(eq(x, y) for x, y in zip(la, lb))


def rec(item, what, ok, mine=None, theirs=None):
    out.setdefault(item, []).append({"what": what, "agree": bool(ok), "mine": mine, "solver": theirs})


su2, gam, der = A["su2"], A["gamma"], A["derived"]
# ---- item 0
i0 = S["item00"]
rec("0a", "Theta v2 = v_-2", A["item0"]["0a_Theta_v2_coeffs_m=-3..3"] == ["0", "1", "0", "0", "0", "0", "0"]
    and i0["0a_Theta_v2"] == {"coeff_of_v_-2": "1"}, "v_-2", i0["0a_Theta_v2"])
rec("0b", "CG", eq(A["item0"]["0b_CG_33_3m3_60"], i0["0b_CG_33_3m3_60"]), A["item0"]["0b_CG_33_3m3_60"], i0["0b_CG_33_3m3_60"])
rec("0c", "order/perfect/norms/histogram", A["item0"]["0c_order"] == i0["0c"]["order_Gamma"] == 120
    and A["item0"]["0c_perfect"] and i0["0c"]["Gamma_equals_derived_subgroup"]
    and {str(k): v for k, v in A["item0"]["0c_element_order_histogram"].items()} == i0["0c"]["element_order_histogram"],
    A["item0"]["0c_order"], i0["0c"]["order_Gamma"])
# ---- items 1-2
t1 = {k.replace("V_", ""): v for k, v in S["items01_09"]["item01_Sym3V3"].items()}
rec("1", "Sym^3 V3", t1 == su2["item1_Sym3V3"], su2["item1_Sym3V3"], t1)
rec("2", "J=8,J=3", (su2["item2_dim_Hom_J8"], su2["item2_dim_Hom_J3"]) == (S["items01_09"]["item02"]["dim_Hom_J8"], S["items01_09"]["item02"]["dim_Hom_J3"]),
    [su2["item2_dim_Hom_J8"], su2["item2_dim_Hom_J3"]], [S["items01_09"]["item02"]["dim_Hom_J8"], S["items01_09"]["item02"]["dim_Hom_J3"]])
# ---- 3-5
rec("3", "M0 constant", eq(su2["item3_constant"], S["items01_09"]["item03"]["c"]), su2["item3_constant"], S["items01_09"]["item03"]["c"])
rec("4", "lambda_m", eqlist(su2["item4_M6_diag_m=-3..3"], S["items01_09"]["item04"]["lambda_m_for_m=-3..3"]))
rec("5", "B2(v3)", eqlist(su2["item5_B2_v3_N=-2..2"], S["items01_09"]["item05"]["B2_v3"]))
rec("5", "rho2(v3)", eqlist(su2["item5_rho2_v3_N=-2..2"], S["items01_09"]["item05"]["rho2_v3_components_N=-2..2"]))
# ---- 6, 7
for nm in ("v3", "v0", "(v2+v-2)/sqrt2", "(v3+v-3)/sqrt2"):
    m, t = su2["item6"][nm], S["items01_09"]["item06"][nm]
    rec("6", f"{nm}: 924||rho6||^2 and critical", eq(m["924*||rho6||^2"], t["924_norm2_rho6"])
        and m["grad_tangential_zero(critical)"] == t["critical"] is True, m["924*||rho6||^2"], t["924_norm2_rho6"])
rec("7", "row", eqlist(su2["item7_row_K0..6"], S["items01_09"]["item07"]["norm2_rhoK_K=0..6"]))
# ---- 8
s = sp.Symbol("s")
f_m = sp.sympify(su2["item8_norm_rho6_sq_poly_in_s"], locals={"s": s})
f_t = sp.sympify(S["items01_09"]["item08"]["norm2_rho6_of_s"].replace("s", "s"), locals={"s": s})
rec("8", "polynomial in s", sp.expand(f_m - f_t) == 0, str(f_m), str(f_t))
ms_ = su2["item8_interior_stationary"]
ts_ = S["items01_09"]["item08"]["interior_stationary_points"]
rec("8", "interior stationary (s, value)", len(ms_) == len(ts_) == 1 and eq(ms_[0]["s"], ts_[0]["s"]) and eq(ms_[0]["value"], ts_[0]["value"]),
    [ms_[0]["s"], ms_[0]["value"]], [ts_[0]["s"], ts_[0]["value"]])
rec("8", "endpoints", eq(su2["item8_endpoints"]["s=0"], S["items01_09"]["item08"]["endpoint_values"]["s=0"])
    and eq(su2["item8_endpoints"]["s=1"], S["items01_09"]["item08"]["endpoint_values"]["s=1"]))
# ---- 9
def ptset(lst, kx="x", ky="y", kv="rhat6"):
    return sorted((float(sp.sympify(p[kx])), float(sp.sympify(p[ky])), sp.nsimplify(sp.sympify(p[kv]))) for p in lst)
pm = ptset(su2["item9_critical_rhat6"])
pt = ptset(S["items01_09"]["item09"]["rhat_critical_points_real"])
rec("9", "rhat6 critical set in chart", len(pm) == len(pt) and all(abs(a[0] - b[0]) < 1e-12 and abs(a[1] - b[1]) < 1e-12 and a[2] == b[2] for a, b in zip(pm, pt)),
    [str(p) for p in pm], [str(p) for p in pt])
um = ptset(su2["item9_critical_unnormalised"], kv="norm_rho6_sq")
ut = ptset(S["items01_09"]["item09"]["unnormalised_critical_points_real"], kv="norm2_rho6")
rec("9", "unnormalised critical set", len(um) == len(ut) and all(abs(a[0] - b[0]) < 1e-12 and a[2] == b[2] for a, b in zip(um, ut)),
    [str(p) for p in um], [str(p) for p in ut])
rec("9", "[v0] value", eq(su2["item9_[v0]_rhat6"], S["items01_09"]["item09"]["point_at_infinity_v0"]["rhat6"]))
# ---- 10, 11, 12, 16
g = S["group"]
rec("10", "dims K=0..6", [int(gam["item10_dim_invariants_K0to6"][str(K)]["char_average"]) for K in range(7)]
    == [g["item10"]["dim_VK_Gamma_K0_6"][str(K)] for K in range(7)])
rec("10", "extra K=7..30", {str(k): v for k, v in gam["item10_EXTRA_K7to30_char_average_only"].items()} == g["item10"]["EXTRA_beyond_K6_K7_30"])
lev6 = gam["item11_and_16_levels"]["level_6_j=3"]["summands"]
my_sum = {s_["dim"]: s_ for s_ in lev6}
th_sum = {int(c["rank"]): c for c in g["item11"]["components"]}
rec("11", "dims + multiplicities + FS", set(my_sum) == set(th_sum) == {3, 4}
    and all(th_sum[d]["multiplicity"] == "1" and th_sum[d]["Frobenius_Schur_indicator"] == "1" for d in (3, 4)))
# characters: match by the class real part w (solver records trace 2w)
my_w = [r5(x) for x in gam["class_real_parts"]]
th_w = [sp.sympify(c["trace_2w"]) / 2 for c in g["classes"]]
ok = True
for d in (3, 4):
    mych = [r5(x) for x in my_sum[d]["character_on_classes"]]
    thch = [sp.sympify(x) for x in th_sum[d]["character_on_classes"]]
    for wi, ci in zip(th_w, thch):
        j = [k for k, w in enumerate(my_w) if eq(w, wi)]
        ok = ok and len(j) == 1 and eq(mych[j[0]], ci)
rec("11", "characters class by class (matched by Re g)", ok)
for d in (3, 4):
    rec("12", f"d={d} norms K=0..6", eqlist([r5(x) for x in my_sum[d]["MK_norm2_K0..n"]], g["item12"][f"d={d}"]["norm2_MK_P_K0_6"]))
for n in range(1, 7):
    mylev = gam["item11_and_16_levels"][[k for k in gam["item11_and_16_levels"] if k.startswith(f"level_{n}_")][0]]
    thlev = g["item16"][f"level_{n}"]
    mys = sorted(([r5(x) for x in c["MK_norm2_K0..n"]] for c in mylev["summands"]), key=lambda v: float(v[0]))
    ths = sorted(([sp.sympify(x) for x in c["norm2_MjK_P_K0_2j"]] for c in thlev["components"]), key=lambda v: float(v[0]))
    rec("16", f"level {n} norms", len(mys) == len(ths) and all(eqlist(a, b) for a, b in zip(mys, ths)))
    rec("16", f"level {n} ranks carried", A["item16"][[k for k in A["item16"] if k.startswith(f"level_{n}_")][0]]["ranks_nonzero_generic"]
        == thlev["sample_state_ranks_nonzero"] == list(range(n + 1)))
# ---- 13, 14, 17, 18
sec = S["items13_18"]["sectors"]
for d in (3, 4):
    md = der["item13"][f"d={d}"]
    td = sec[f"d={d}"]
    rec("13", f"d={d} w_K", eqlist(md["w_K(K=0..6)"], td["w_K"]))
    rec("13", f"d={d} N", eq(md["N=924/w6"], td["N"]), md["N=924/w6"], td["N"])
    rec("14", f"d={d} coefficients", eqlist(der["item14"][f"d={d}"]["kappa_K(K=0..6)"], td["item14_coefficients_cK_over_d"]))
    rec("17", f"d={d} slope", eq(md["w6"], td["Q_d_slope_in_rhat6"]) and eq(td["Q_d_constant"], 1))
    for nm, v in der["item18_beta_at_item6_rays"].items():
        rec("18", f"d={d} beta at {nm}", eq(v[f"d={d}"], td["beta"][nm]["beta_at_B=1"]), v[f"d={d}"], td["beta"][nm]["beta_at_B=1"])
rec("18", "difference coefficient", eq("49/156", S["items13_18"]["beta_difference_d3_minus_d4_coefficient_of_rhat6"]))
# solver-only rays: recompute beta with MY slope and compare
w6 = {3: sp.Rational(28, 39), 4: sp.Rational(21, 52)}
for nm in ("item8 (sqrt13/5)v2+(2sqrt3/5)v-3", "item9 z=i sqrt10/2", "item9 z=sqrt230/10"):
    for d in (3, 4):
        t = sec[f"d={d}"]["beta"][nm]
        mine = 1 + w6[d] * sp.sympify(t["rhat6"])
        rec("18", f"d={d} beta at {nm} (my slope x its rhat6)", eq(mine, t["beta_at_B=1"]), str(mine), t["beta_at_B=1"])
# ---- 19, 19c
rec("19c", "(L8(v3))_3", eq(A["stage2_item19"]["L8_v3_component_N=3"], S["stage2"]["item19c"]["L8_v3_N3"]),
    A["stage2_item19"]["L8_v3_component_N=3"], S["stage2"]["item19c"]["L8_v3_N3"])
rec("19", "C(v3) norm: my ||C_iso(v3)||^2 = 1/1092 vs solver C_0(v3) = -sqrt273/546 (||C_0|| = ||Pi_8 x||)",
    eq(sp.Rational(1, 1092), sp.sympify(S["item19"]["C_v3_nonzero_components"][0]) ** 2))
# ---- 20
z = sp.Symbol("z")
mine_coeffs = [sp.sympify(c) for c in A["stage2_item20"]["G_d3"]["G_coeffs_z^0..z^12"]]
Gm = sum(c * z ** i for i, c in enumerate(mine_coeffs))
Gm_monic = sp.expand(Gm / mine_coeffs[12])
Gt = sp.sympify(S["stage2"]["item20"]["d=4"]["monic_form"], locals={"z": z})
rec("20", "forms identical as given", sp.expand(Gm_monic - Gt) == 0, str(Gm_monic), str(Gt))
rec("20", "my form with z -> i z equals solver's form", sp.expand(sp.expand(Gm_monic.subs(z, sp.I * z)) - Gt) == 0)
rec("20", "multiplicities all 1 (both)", A["stage2_item20"]["all_roots_simple"] and S["stage2"]["item20"]["gcd_f_fprime"] == "1")
rec("20", "injectivity rank 7 (both)", A["stage2_item20"]["transvectant_map_matrix_17x7_exact_rank_over_Q(sqrt5,i)"] == 7 == S["stage2"]["item20"]["transvectant_map_exact_rank"])
# hard-coded polynomial in solver_stage2_roots.py equals its computed monic form?
src = (HERE / "solver_copy" / "solver_stage2_roots.py").read_text()
mm = re.search(r"fmon = \((.*?)\)\n", src, re.S)
hard = sp.sympify(mm.group(1).replace("sp.sqrt", "sqrt").replace("\n", " "), locals={"z": z})
rec("20", "solver's hard-coded fmon == its computed monic form", sp.expand(hard - Gt) == 0)
# ---- 21
for d in (3, 4):
    mine = [sp.sympify(x) for x in der["item21"][f"d={d}"]["coef_N_J(J=0..6)"]]
    th = [sp.sympify(x) for x in S["item21"][f"d={d}"]["coefficients_cprime_over_d"]]
    # my N_0 = -(1/sqrt7) f_0, my N_6 = -(sqrt91/7) f_6 (100-digit solve); solver's N_J = f_J
    conv = {0: -1 / sp.sqrt(7), 6: -sp.sqrt(91) / 7}
    ok = all(eq(mine[J] * conv[J], th[J]) for J in (0, 6)) and all(eq(th[J], 0) for J in (1, 2, 3, 4, 5))
    rec("21", f"d={d} coefficients (after N_J basis conversion)", ok, [str(mine[J] * conv[J]) for J in (0, 6)], [str(th[J]) for J in (0, 6)])
sp_ = S["item21"]["spans"]
rec("21", "ranks N0N6=2, M0M6=2, all four=4, spans do not coincide", sp_["rank_N0_N6"] == 2 and sp_["rank_M0_M6"] == 2
    and sp_["rank_M0_M6_N0_N6"] == 4 and sp_["spans_coincide"] is False
    and su2["equivariant_map_ranks"]["M0,M6,N0,N6"]["rank"] == 4)
# ---- 15: point sets (independent of the quaternion map)
pairs = {"v3": "v3", "v0": "v0", "(v2+v-2)/sqrt2": "(v2+v-2)/sqrt2", "(v3+v-3)/sqrt2": "(v3+v-3)/sqrt2",
         "item8 s=12/25 (+): (sqrt13/5) v2 + (2sqrt3/5) v-3": "item8_s=12/25",
         "item9 z=+i sqrt10/2": "item9_x=0_y=sqrt(10)/2", "item9 z=-i sqrt10/2": "item9_x=0_y=-sqrt(10)/2",
         "item9 z=+sqrt230/10": "item9_x=sqrt(230)/10_y=0", "item9 z=-sqrt230/10": "item9_x=-sqrt(230)/10_y=0",
         "item9 z=0": "item9_x=0_y=0"}
for tn, mn in pairs.items():
    tp = sorted(tuple(round(c, 9) + 0.0 for c in p["xyz_float"]) for p in S["item15"][tn]["points"])
    mp_ = sorted(tuple(round(float(sp.N(sp.sympify(c), 30)), 9) + 0.0 for c in p["xyz"]) for p in su2["item15_constellations"][mn]["points"])
    rec("15", f"{tn} point multiset", tp == mp_)
n_dis = sum(1 for v in out.values() for r in v if not r["agree"])
out["_summary"] = {"comparisons": sum(len(v) for k, v in out.items() if not k.startswith("_")), "disagreements": n_dis}
(HERE / "audit_ref_compare.json").write_text(json.dumps(out, indent=1, default=str))
for k, v in out.items():
    if k.startswith("_"):
        continue
    for r in v:
        print(f"{k:4s} {'OK ' if r['agree'] else 'DIFF'} {r['what']}" + ("" if r["agree"] else f"   mine={r['mine']} solver={r['solver']}"))
print(out["_summary"])
