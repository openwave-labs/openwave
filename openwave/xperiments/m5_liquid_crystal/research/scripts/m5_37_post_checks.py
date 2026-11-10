"""M5.37: arithmetic checks on the numbers printed in the author's post of 2026-11-10 on discussion #186.

Nothing here runs the author's card ("Card C"); every input is a number printed in the post or a PDG /
literature constant named below. The checks:

(1) The rigid radius ratio. With one inertia density rho(r) feeding both isovector Sachs form factors,
    G_E(q) ~ int rho(r) j0(qr) d^3r and G_M(q) ~ int rho(r) 3 j1(qr)/(qr) d^3r, the small-q expansions are
    1 - q^2 <r^2>/6 and 1 - q^2 <r^2>/10, so r_M^2 / r_E^2 = 3/5 for any rho. Checked symbolically (the
    series of j0 and 3 j1/x) and numerically on three profiles (Gaussian, exponential, hard shell).
(2) The rotor moment mu_V = M_p / Delta at Delta = 293 MeV, in nuclear magnetons, against mu_p - mu_n.
(3) The moment the rotor leaves open, in both conventions (mu_p - mu_n, and half of it), against the
    post's +0.66 mu_N.
(4) The two-component radius: a rigid piece (mu_rot at (3/5) r_E^2) plus a second piece (mu_x at r_x^2)
    must give the data ratio 0.887; mu_x solved for r_x^2 = 1.5 fm^2 at the PDG isovector charge radius.
(4b) Added on the audit: the data ratio from two input sets of the magnetic radii, and the ratio at which
    the solve of (4) returns the post's 0.66.
(5) The rho-pole radius 6 / m_rho^2, and the rms of 1.5 fm^2.
(6) Goldberger-Treiman at the post's g_A and g_piNN with physical M_N and f_pi: 1 - g_A M_N / (f_pi g).
(7) The Born scaling (9.55 / 13.1)^2.

Each check states the value that would make it fail (the post's printed number, with a tolerance).
Usage: python3 m5_37_post_checks.py [--write]   (writes ../data/m5_37_post_checks.json)
"""
import json, os, sys
import numpy as np
import sympy as sp

HBARC = 197.3269804          # MeV fm
MP, MN = 938.27208816, 939.56542052   # MeV
MU_P, MU_N = 2.79284734463, -1.91304273   # nuclear magnetons
M_RHO = 775.26               # MeV
RE2_P, RE2_N = 0.8409 ** 2, -0.1155   # fm^2 (PDG proton charge radius, neutron mean-square charge radius)

# numbers printed in the post
POST = dict(delta=293.0, mu_rot=3.20, mu_v=4.706, ratio_rigid=0.6, ratio_data=0.887, mu_missing=0.66,
            r2_missing=1.5, rms_missing=1.2, r2_rho=0.39, gA=0.904, gpiNN=9.55, gt_lo=0.01, gt_hi=0.025,
            g_phys=13.1, born=0.53)


def ratio_symbolic():
    x = sp.symbols("x", positive=True)
    j0 = sp.sin(x) / x
    j1 = sp.sin(x) / x ** 2 - sp.cos(x) / x
    e = sp.series(j0, x, 0, 6).removeO()
    m = sp.series(3 * j1 / x, x, 0, 6).removeO()
    ce = -sp.Poly(e, x).coeff_monomial(x ** 2)     # 1/6
    cm = -sp.Poly(m, x).coeff_monomial(x ** 2)     # 1/10
    return sp.Rational(cm, 1) / sp.Rational(ce, 1), sp.Poly(m, x).coeff_monomial(1)


def ratio_numeric(rho, rmax=40.0, n=400001):
    r = np.linspace(1e-9, rmax, n)
    w = rho(r) * r ** 2
    out = []
    for q in (1e-2, 2e-2):
        x = q * r
        ge = np.trapezoid(w * np.sin(x) / x, r) / np.trapezoid(w, r)
        gm = np.trapezoid(w * 3 * (np.sin(x) / x ** 2 - np.cos(x) / x) / x, r) / np.trapezoid(w, r)
        out.append((1 - gm) / (1 - ge))
    return out


def main():
    out, ok = {}, {}
    rat, m0 = ratio_symbolic()
    out["1_ratio_symbolic"] = str(rat)
    out["1_magnetic_normalization_at_q0"] = str(m0)
    prof = {"gaussian": lambda r: np.exp(-r ** 2), "exponential": lambda r: np.exp(-r),
            "shell": lambda r: np.exp(-((r - 3.0) / 0.05) ** 2)}
    out["1_ratio_numeric"] = {k: ratio_numeric(f) for k, f in prof.items()}
    ok["1_ratio_is_3_over_5"] = (rat == sp.Rational(3, 5)) and all(abs(v - 0.6) < 1e-3 for vs in out["1_ratio_numeric"].values() for v in vs)

    mu_rot = MP / POST["delta"]
    mu_v = MU_P - MU_N
    out["2_mu_rot"] = mu_rot
    out["2_mu_p_minus_mu_n"] = mu_v
    ok["2_mu_rot_is_3.20"] = abs(mu_rot - POST["mu_rot"]) < 0.005
    ok["2_mu_v_is_4.706"] = abs(mu_v - POST["mu_v"]) < 0.0005

    d_full = mu_v - mu_rot
    d_half = d_full / 2
    out["3_open_moment_full_convention"] = d_full
    out["3_open_moment_half_convention"] = d_half
    ok["3_open_moment_is_0.66"] = min(abs(d_full - POST["mu_missing"]), abs(d_half - POST["mu_missing"])) < 0.02

    re2 = RE2_P - RE2_N
    out["4_rE2_isovector_fm2"] = re2
    target = POST["ratio_data"] * re2
    rigid = 0.6 * re2
    mu_x = mu_rot * (target - rigid) / (POST["r2_missing"] - target)
    out["4_rM2_target_fm2"] = target
    out["4_mu_x_at_rx2_1.5_on_mu_rot_3.20"] = mu_x
    out["4_total_moment_then"] = mu_rot + mu_x
    # the same solve with the total pinned at 4.706 instead of the rotor at 3.20: mu_x = mu_v (target - rigid)/(rx2 - rigid)
    mu_x_tot = mu_v * (target - rigid) / (POST["r2_missing"] - rigid)
    out["4_mu_x_at_rx2_1.5_on_total_4.706"] = mu_x_tot
    # the radius the second piece needs if it carries exactly the open moment (full convention)
    out["4_rx2_if_mu_x_is_open_moment_fm2"] = (target * mu_v - rigid * mu_rot) / d_full
    # (4b) added on the audit: the data ratio itself. r_M,V^2 = (mu_p rMp^2 - mu_n rMn^2) / (mu_p - mu_n) for two
    # input sets of the magnetic radii (fm), and the ratio at which the two-component solve returns the post's 0.66
    sets = {"rMp_0.851_rMn_0.864": (0.851, 0.864), "rMp_0.777_rMn_0.862": (0.777, 0.862)}
    out["4b_data_ratio_by_input_set"] = {}
    out["4b_mu_x_on_rotor_by_input_set"] = {}
    for k, (a, b) in sets.items():
        rm2 = (MU_P * a ** 2 - MU_N * b ** 2) / mu_v
        out["4b_data_ratio_by_input_set"][k] = rm2 / re2
        out["4b_mu_x_on_rotor_by_input_set"][k] = mu_rot * (rm2 - rigid) / (POST["r2_missing"] - rm2)
    t66 = (POST["mu_missing"] * POST["r2_missing"] + mu_rot * rigid) / (mu_rot + POST["mu_missing"])
    out["4b_ratio_at_which_mu_x_is_0.66"] = t66 / re2
    ok["4b_printed_ratio_0.887_is_the_0.851_set"] = abs(out["4b_data_ratio_by_input_set"]["rMp_0.851_rMn_0.864"] - POST["ratio_data"]) < 0.01
    ok["4_mu_x_is_0.66"] = min(abs(mu_x - POST["mu_missing"]), abs(mu_x_tot - POST["mu_missing"])) < 0.02

    r2_rho = 6 * (HBARC / M_RHO) ** 2
    out["5_r2_rho_pole_fm2"] = r2_rho
    out["5_rms_of_1.5_fm2"] = float(np.sqrt(POST["r2_missing"]))
    ok["5_rho_pole_is_0.39"] = abs(r2_rho - POST["r2_rho"]) < 0.005
    ok["5_rms_is_1.2"] = abs(np.sqrt(POST["r2_missing"]) - POST["rms_missing"]) < 0.05

    mnuc = 0.5 * (MP + MN)
    gt = {}
    for fpi in (92.07, 92.28, 93.0):
        gt["f_pi_%.2f" % fpi] = 1 - POST["gA"] * mnuc / (fpi * POST["gpiNN"])
    out["6_GT_discrepancy_physical_inputs"] = gt
    out["6_f_pi_needed_for_1pct_and_2.5pct_MeV"] = [POST["gA"] * mnuc / (POST["gpiNN"] * (1 - d)) for d in (0.01, 0.025)]
    ok["6_GT_within_1_to_2.5_pct_at_physical_inputs"] = all(POST["gt_lo"] <= abs(v) <= POST["gt_hi"] for v in gt.values())

    born = (POST["gpiNN"] / POST["g_phys"]) ** 2
    out["7_born_scaling"] = born
    ok["7_born_is_0.53"] = abs(born - POST["born"]) < 0.005

    rec = {"task": "M5.37", "what": "arithmetic checks on the numbers printed in the post of 2026-11-10; no card run",
           "constants": dict(hbarc=HBARC, Mp=MP, Mn=MN, mu_p=MU_P, mu_n=MU_N, m_rho=M_RHO, rE2_p=RE2_P, rE2_n=RE2_N),
           "post": POST, "values": out, "checks": {k: bool(v) for k, v in ok.items()}}
    for k, v in out.items():
        print("%-48s %s" % (k, v))
    for k, v in ok.items():
        print("%-48s %s" % (k, "PASS" if v else "NOT REPRODUCED"))
    if "--write" in sys.argv:
        here = os.path.dirname(os.path.abspath(__file__))
        json.dump(rec, open(os.path.join(here, "..", "data", "m5_37_post_checks.json"), "w"), indent=1, default=float)
        print("written m5_37_post_checks.json")


if __name__ == "__main__":
    main()
