"""M5.38 R3: the arithmetic of the moment bound and the Darwin statements of the author's post of 2026-11-10
(tasks/m5_38_task_details.md § 4). Printed numbers and symbolic identities only; nothing of the author's is run.

Sources of the printed numbers: part (3) of the author's `V84` (scorecard v1.7, rerun as shipped in M5.36):
mu_V and Delta0 on the two cards W and DN56, M_p = 938.2721 MeV; the post: mu_p - mu_n = 4.70589 mu_N,
Delta - N = 293 MeV, "the missing ~1.5 mu_N", <d^2> = 3 lambda_C^2 / 4, "the projection gives one third".

Run from this directory: python3 m5_38_post_checks.py [--write]   ->  ../data/m5_38_post_checks.json
"""
from __future__ import annotations

import json
import os
import sys

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

MP = 938.2721
CARDS = {"W": (286.9060, 3.270311), "DN56": (229.3050, 4.091808)}     # (Delta0 in MeV, mu_V in mu_N), V84 part (3)
MU_V_DATA, GAP = 4.70589, 293.0

out = {"versions": {"python": sys.version.split()[0], "sympy": sp.__version__}, "checks": {}}
C = out["checks"]

# 1. the identity mu_V Delta0 = M_p on the two cards
prods = {k: d * m for k, (d, m) in CARDS.items()}
C["1_identity_on_the_two_cards"] = {
    "products_MeV_muN": prods, "M_p": MP, "relative_deviation": {k: v / MP - 1 for k, v in prods.items()},
    "pass": all(abs(v / MP - 1) < 1e-5 for v in prods.values()),
    "fails_if": "either printed product differs from 938.2721 beyond the printed digits (1e-5)"}

# 2. the data against the bound (mu_p - mu_n)(Delta - N) <= M_N
prod = MU_V_DATA * GAP
C["2_data_against_the_bound"] = {
    "product_MeV_muN": prod, "over_M_p": prod / MP, "data_outside_the_bound": prod > MP,
    "gap_at_which_the_data_moment_saturates_MeV": MP / MU_V_DATA,
    "pass": prod > MP,
    "fails_if": "the product of the measured isovector moment and the 293 MeV gap is at or below M_p (then the data sit inside the bound and nothing is missing)"}

# 3. the missing moment
missing = MU_V_DATA - MP / GAP
C["3_missing_moment"] = {
    "rotor_moment_at_293": MP / GAP, "missing_muN": missing, "post_prints": "~1.5", "m5_37_row_3": 1.504,
    "pass": abs(missing - 1.504) < 5e-4 and abs(missing - 1.5) < 0.05,
    "fails_if": "4.70589 - M_p / 293 is not 1.504 (the M5.37 row) or does not round to the post's 1.5"}

# 4. the direction of the inequality: not derivable from the held material
C["4_inequality_direction"] = {
    "status": "NOT DERIVABLE from the held material: V84 part (3) states an identity on two cards (check 1); no script or text held "
              "on OpenWave derives '<=' for the class of locks the post names. Recorded as an ask.",
    "pass": None}

# 5. the Darwin statements
hbar, m, c, R, d2 = sp.symbols("hbar m c R delta2", positive=True)
lam = hbar / (m * c)
# (i) the Darwin coefficient equals isotropic smearing at <d^2> = (3/4) lambda_C^2
darwin = hbar ** 2 / (8 * m ** 2 * c ** 2)
i_ok = sp.simplify(darwin - (sp.Rational(3, 4) * lam ** 2) / 6) == 0
# (ii) isotropic smearing: the second-order term of <V(x + d)> is (<d^2> / 6) Laplacian V, on a general quadratic V
a = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"a{min(i, j)}{max(i, j)}"))
x = sp.Matrix(sp.symbols("x0:3"))
V = (x.T * a * x)[0]
lap = sum(sp.diff(V, xi, 2) for xi in x)
second = sp.Rational(1, 2) * sum((d2 / 3) * sp.diff(V, x[i], 2) for i in range(3))     # <d_i d_j> = (d2 / 3) delta_ij
ii_ok = sp.simplify(second - d2 / 6 * lap) == 0
# (iii) a planar orbit of radius R with normal n, averaged over orientations: <d_i d_j> = (R^2 / 3) delta_ij, so <d^2> = R^2
th, ph = sp.symbols("theta phi", real=True)
n = sp.Matrix([sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th)])
plane = (R ** 2 / 2) * (sp.eye(3) - n * n.T)
avg = plane.applyfunc(lambda e: sp.integrate(sp.integrate(e * sp.sin(th), (th, 0, sp.pi)), (ph, 0, 2 * sp.pi)) / (4 * sp.pi))
iii_ok = sp.simplify(avg - (R ** 2 / 3) * sp.eye(3)) == sp.zeros(3, 3)
# (iv) the Casimir radius at s = 1/2 and the projection: R^2 = s(s+1) lambda^2 = 3/4 lambda^2; (1/2)^2 / (3/4) = 1/3
s = sp.Rational(1, 2)
iv_ok = (s * (s + 1) == sp.Rational(3, 4)) and (sp.Rational(1, 4) / (s * (s + 1)) == sp.Rational(1, 3))
C["5_darwin_statements"] = {
    "i_darwin_coefficient_is_smearing_at_3_4_lambda2": bool(i_ok),
    "ii_isotropic_smearing_gives_d2_over_6_laplacian": bool(ii_ok),
    "iii_planar_orbit_averaged_over_orientations_gives_R2": bool(iii_ok),
    "iv_casimir_3_4_and_projection_one_third": bool(iv_ok),
    "pass": bool(i_ok and ii_ok and iii_ok and iv_ok),
    "fails_if": "any of: hbar^2/(8 m^2 c^2) != (3/4 lambda_C^2)/6; the smearing term != <d^2>/6 Laplacian; the orientation average != R^2/3 per axis; s(s+1) != 3/4 at s = 1/2"}

for k, v in C.items():
    print(k, "PASS" if v["pass"] else ("n/a" if v["pass"] is None else "FAIL"), json.dumps({a_: b_ for a_, b_ in v.items() if a_ not in ("fails_if", "pass")}))
out["all_pass"] = all(v["pass"] for v in C.values() if v["pass"] is not None)
if "--write" in sys.argv:
    with open(os.path.join(DATA, "m5_38_post_checks.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote ../data/m5_38_post_checks.json")
