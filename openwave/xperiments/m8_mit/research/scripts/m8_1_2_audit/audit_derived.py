"""Items 13, 14, 17, 18, 21 from the exact outputs of audit_gamma.py and audit_su2_items.py,
asserted against the independent quadrature (audit_quadrature.py). Then merge everything into
audit_results.json.

Derived formulas (derivation written in audit_stage1_return.md):
  B = (d/7)|u|^2,  A = sum_K ||rho_K(u)||^2 ||R_K||^2 / (2K+1)
  w_K = 49 ||R_K||^2 / ((2K+1) d^2)
  N(u) = (7/d) sum_K ||R_K||^2/(2K+1) G_K(u),  G_K = c_K M_K  =>  kappa_K = (7/d) ||R_K||^2 c_K/(2K+1)
  psi psi^T channel: (7/d) sum_J ||R_J||^2/(2J+1) N_J(u)
"""
import json
import re
import pathlib
import sympy as sp

HERE = pathlib.Path(__file__).parent
gam = json.loads((HERE / "audit_res_gamma.json").read_text())
su2 = json.loads((HERE / "audit_res_su2.json").read_text())
quad = json.loads((HERE / "audit_res_quadrature.json").read_text())["quadrature"]
res = {}


def r5(sx):
    m = re.fullmatch(r"\((-?[\d/]+)\+(-?[\d/]+)\*s5\)", sx)
    return sp.Rational(m.group(1)) + sp.Rational(m.group(2)) * sp.sqrt(5)


R = {}
for summ in gam["item11_and_16_levels"]["level_6_j=3"]["summands"]:
    R[summ["dim"]] = [r5(t) for t in summ["MK_norm2_K0..n"]]
cK = {int(k): sp.sympify(v) for k, v in su2["cK_G_K_over_M_K"].items()}

item13, item14, item21, item17 = {}, {}, {}, {}
for d in (3, 4):
    wK = [sp.nsimplify(49 * R[d][K] / ((2 * K + 1) * d ** 2)) for K in range(7)]
    item13[f"d={d}"] = {"w_K(K=0..6)": [str(x) for x in wK], "w0": str(wK[0]), "w6": str(wK[6]),
                        "w6/w0": str(sp.nsimplify(wK[6] / wK[0])), "N=924/w6": str(sp.nsimplify(924 / wK[6]))}
    kap = [sp.nsimplify(sp.radsimp(sp.Rational(7, d) * R[d][K] * cK[K] / (2 * K + 1))) for K in range(7)]
    item14[f"d={d}"] = {"kappa_K(K=0..6)": [str(x) for x in kap]}
    lamJ = [sp.nsimplify(sp.Rational(7, d) * R[d][J] / (2 * J + 1)) for J in range(7)]
    item21[f"d={d}"] = {"coef_N_J(J=0..6)": [str(x) for x in lamJ]}
    item17[f"d={d}"] = f"Q_d = 1 + ({wK[6]}) * rhat_6"
    # ---- assertions against the quadrature ----
    q = quad[f"d{d}"]
    slope = q["Q_fit_intercept_slope"]
    assert abs(slope[0] - 1) < 1e-10 and abs(slope[1] - float(wK[6])) < 1e-10, (slope, wK[6])
    for fit in q["N_fits"]:
        assert abs(fit["coef_M0"][0] - float(kap[0])) < 1e-10 and abs(fit["coef_M0"][1]) < 1e-10
        assert abs(fit["coef_M6"][0] - float(kap[6])) < 1e-10 and abs(fit["coef_M6"][1]) < 1e-10
        assert fit["rel_resid"] < 1e-10
    for fit in q["item21"]:
        assert abs(fit["coef_N0"][0] - float(lamJ[0])) < 1e-10 and abs(fit["coef_N6"][0] - float(lamJ[6])) < 1e-10
        assert abs(fit["E_quad"] - fit["E_formula"]) < 1e-9 * fit["E_quad"]
        assert fit["invariance_psipsiT"]
    for K in range(7):
        assert abs(q["||R_K||^2_numeric"][str(K)] - float(R[d][K])) < 1e-12
res["item13"] = item13
res["item13_w6_over_w0_in_d"] = "12/(13 d^2)  (uses ||R_6||^2 = 12/7 in both sectors, ||R_0||^2 = d^2/7)"
res["item13_N_in_d"] = "143 d^2"
for d in (3, 4):
    assert sp.nsimplify(924 / (49 * R[d][6] / (13 * d ** 2))) == 143 * d ** 2
    assert sp.nsimplify(49 * R[d][6] / (13 * d ** 2) / 7) == sp.Rational(12, 13 * d ** 2)
res["item14"] = item14
res["item17"] = item17
res["item21"] = item21
res["cK"] = {str(k): str(v) for k, v in cK.items()}

# ---- item 18: beta = Q_d([u]) at B(u)=1; cross-check with kappa route at the item-6 critical rays ----
w6 = {d: sp.nsimplify(49 * R[d][6] / (13 * d ** 2)) for d in (3, 4)}
res["item18_beta"] = "beta = A(u) = Q_d([u]) = 1 + w6_d * rhat6([u])   (with B(u)=1, i.e. |u|^2 = 7/d)"
res["item18_beta_3_minus_beta_4"] = f"({sp.nsimplify(w6[3] - w6[4])}) * rhat6([u])"
chk = {}
for name, rec in su2["item6"].items():
    if not rec["M6(u)_parallel_u"]:
        continue
    mu = sp.sympify(rec["M6_eigenvalue"])            # M6(u) = mu u at |u| = 1
    r6 = sp.sympify(rec["||rho6||^2"])              # = rhat6 at |u| = 1
    per = {}
    for d in (3, 4):
        kap0 = sp.sympify(item14[f"d={d}"]["kappa_K(K=0..6)"][0])
        kap6 = sp.sympify(item14[f"d={d}"]["kappa_K(K=0..6)"][6])
        # at |u|^2 = 7/d: M0(u) = -(7/d)/sqrt7 u, M6(u) = (7/d) mu u
        beta_kappa = sp.nsimplify(sp.radsimp(kap0 * (-sp.Rational(7, d) / sp.sqrt(7)) + kap6 * sp.Rational(7, d) * mu))
        beta_Q = sp.nsimplify(1 + w6[d] * r6)
        assert sp.simplify(beta_kappa - beta_Q) == 0, (name, d, beta_kappa, beta_Q)
        per[f"d={d}"] = str(beta_Q)
    per["beta3-beta4"] = str(sp.nsimplify(sp.sympify(per["d=3"]) - sp.sympify(per["d=4"])))
    chk[name] = per
res["item18_beta_at_item6_rays"] = chk

(HERE / "audit_res_derived.json").write_text(json.dumps(res, indent=1))

# ---- merge ----
allres = {}
for f in ["audit_res_item0.json", "audit_res_gamma.json", "audit_res_su2.json", "audit_res_item19.json",
          "audit_res_quadrature.json", "audit_res_derived.json", "audit_res_item16.json"]:
    allres[f.replace("audit_res_", "").replace(".json", "")] = json.loads((HERE / f).read_text())
(HERE / "audit_results.json").write_text(json.dumps(allres, indent=1))
print(json.dumps(res, indent=1))
