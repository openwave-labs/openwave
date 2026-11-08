"""M5.36 anchor close: W-coupling closed form vs exact SVD, dipole factor, and
diagnosis of a past unit mismatch. Written from the printed formulas only.

Block: B(r, M, b) = [[0, M r, 0], [0, M, b], [M r, 0, M]].
Light state = smallest singular value s (the lepton mass).
g = |a_L . nu|, nu = (1, -r)/sqrt(1+r^2) on the first two components.
Closed form: delta = (b/M)^2 r^2 / (1+r^2)^3, 1 - g = delta/2 + O((b/M)^4),
W_mu/W_e - 1 = -(delta_mu - delta_e)/2.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "m5_36_anchor_close.json")

M_E = 0.51099895069e-3  # GeV
M_MU = 105.6583755e-3   # GeV
ALPHA = 1.0 / 137.035999084


def block(r, M, b):
    return np.array([[0.0, M * r, 0.0], [0.0, M, b], [M * r, 0.0, M]])


def light_state(r, M, b):
    U, S, Vt = np.linalg.svd(block(r, M, b))
    k = int(np.argmin(S))
    return S[k], U[:, k]


def smin(r, M, b):
    return light_state(r, M, b)[0]


def one_minus_g(r, M, b):
    """1 - g computed from the small components (no cancellation).
    1 - g^2 = a2^2 + p^2, p = (r a0 + a1)/sqrt(1+r^2) (component along nu_perp)."""
    _, a = light_state(r, M, b)
    p = (r * a[0] + a[1]) / np.sqrt(1.0 + r * r)
    q = (a[0] - r * a[1]) / np.sqrt(1.0 + r * r)
    one_minus_g2 = a[2] ** 2 + p ** 2
    return one_minus_g2 / (1.0 + abs(q))


def solve_r(mass, M, b, lo=1e-8, hi=0.9):
    """Bisection in log r for smin(r) = mass (smin increasing in r here)."""
    flo, fhi = smin(lo, M, b) - mass, smin(hi, M, b) - mass
    assert flo < 0 < fhi, (flo, fhi)
    a, c = np.log(lo), np.log(hi)
    for _ in range(200):
        mid = 0.5 * (a + c)
        f = smin(np.exp(mid), M, b) - mass
        if f < 0:
            a = mid
        else:
            c = mid
        if c - a < 1e-16:
            break
    return float(np.exp(0.5 * (a + c)))


def delta(r, bM):
    return bM ** 2 * r ** 2 / (1.0 + r ** 2) ** 3


def point(M, b):
    re_ = solve_r(M_E, M, b)
    rm_ = solve_r(M_MU, M, b)
    bM = b / M
    omg_e, omg_m = one_minus_g(re_, M, b), one_minus_g(rm_, M, b)
    ge, gm = 1 - omg_e, 1 - omg_m
    exact = gm / ge - 1.0
    de, dm = delta(re_, bM), delta(rm_, bM)
    closed = -(dm - de) / 2.0
    return dict(M_GeV=M, b_GeV=b, b_over_M=bM, r_e=re_, r_mu=rm_,
                smin_e_check=smin(re_, M, b), smin_mu_check=smin(rm_, M, b),
                delta_e=de, delta_mu=dm,
                one_minus_g_e=omg_e, one_minus_g_mu=omg_m,
                exact_ratio_minus_1=exact, exact_ratio_minus_1_ppm=exact * 1e6,
                closed_form=closed, closed_form_ppm=closed * 1e6,
                exact_minus_closed=exact - closed,
                exact_minus_closed_ppm=(exact - closed) * 1e6,
                kappa_h_mu_minus_e_over_4=((1 - 2 * dm) - (1 - 2 * de)) / 4.0)


out = {}
out["baseline"] = point(1.0e4, 200.0)
out["LB1"] = point(3.0e4, 10.0)
out["printed_targets"] = dict(baseline_r_e=0.0015984, baseline_r_mu=0.0229907,
                              baseline_W_ppm=-0.1050, LB1_W=-5.72e-10)

# expansion check: 1 - g vs delta/2, residual should scale as (b/M)^4
grid = []
for r in (0.002, 0.02, 0.1, 0.5):
    for bM in (0.2, 0.1, 0.05, 0.02):
        M = 1.0
        b = bM
        ex = one_minus_g(r, M, b)
        cf = delta(r, bM) / 2.0
        grid.append(dict(r=r, b_over_M=bM, one_minus_g=ex, delta_over_2=cf,
                         rel_err=(ex - cf) / cf,
                         resid_over_bM4=(ex - cf) / bM ** 4))
out["expansion_grid"] = grid
out["expansion_max_rel_err_at_bM_0p02"] = max(
    abs(g["rel_err"]) for g in grid if g["b_over_M"] == 0.02)

# 2. closed form at the printed amplitudes
re_p, rm_p, bMp = 0.0015984, 0.0229907, 0.02
cf_p = -(delta(rm_p, bMp) - delta(re_p, bMp)) / 2.0
out["closed_form_at_printed_amplitudes"] = dict(
    r_e=re_p, r_mu=rm_p, b_over_M=bMp, W_ratio_minus_1=cf_p, ppm=cf_p * 1e6,
    expected_ppm=-0.10504)

# 3. dipole factor
e = np.sqrt(4 * np.pi * ALPHA)
fac = e * np.sqrt(M_MU / M_E)
earlier = 1.0 / 0.2294991888527181
out["dipole_factor"] = dict(e=e, e_sqrt_mmu_over_me=fac,
                            earlier_inverse_ratio=earlier,
                            relative_gap=(earlier - fac) / fac)

# 4. diagnosis of the earlier miss
# The earlier script used me=0.51099895, mm=105.6583755 (MeV numbers) in
# dl(m, b_, M_) while b_=200 and M_=1e4 are in GeV.
def dl_old(m, b_, M_):
    y = m / b_
    s = (m / M_) ** 2
    t2_ = y + s * (1 - y) ** 2 / (1 + s * (1 - y))
    r2 = t2_ / (1 - t2_)
    return (b_ / M_) ** 2 * r2 / (1 + r2) ** 3

old_e, old_mu = 0.51099895, 105.6583755
repro = dl_old(old_mu, 200, 1e4) - dl_old(old_e, 200, 1e4)
fixed = dl_old(M_MU, 200, 1e4) - dl_old(M_E, 200, 1e4)
out["r32_miss_diagnosis_numbers"] = dict(
    reproduced_dmu_minus_de_MeV_masses_with_GeV_b_M=repro,
    recorded_in_old_json=4.600026002691369e-05,
    same_formula_with_masses_in_GeV=fixed,
    exact_svd_delta_diff=out["baseline"]["delta_mu"] - out["baseline"]["delta_e"])
out["r32_miss_diagnosis"] = (
    "The earlier script passed the lepton masses in MeV (0.511, 105.66) into "
    "dl(m, b, M) while b=200 and M=1e4 are in GeV, so y=m/b was 1000 times too "
    "large and the amplitudes were r_mu ~ 1.0 instead of 0.023, giving "
    "delta_mu - delta_e = 4.6e-5 (reproduced here to the digit). With masses in "
    "GeV the same eq. (22)-style formula gives the printed amplitudes and the "
    "closed form -(delta_mu - delta_e)/2 = -0.105 ppm, so the printed W value "
    "stands and the factor ~200 was a unit slip, not a formula error.")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(out, fh, indent=1, default=float)

print(json.dumps(out, indent=1, default=float))
for k in ("baseline", "LB1"):
    d = out[k]
    print(f"[{k}] r_e={d['r_e']:.7g} r_mu={d['r_mu']:.7g} "
          f"exact={d['exact_ratio_minus_1']:.6e} closed={d['closed_form']:.6e} "
          f"diff={d['exact_minus_closed']:.2e}")
print(f"[printed amplitudes] closed form = {cf_p*1e6:.5f} ppm")
print(f"[dipole] e*sqrt(mmu/me)={fac:.5f} vs earlier {earlier:.5f}, "
      f"rel gap {(earlier-fac)/fac:.3e}")
print(f"[diagnosis] reproduced old miss: {repro:.6e}")
