"""Planted defects: each block reruns one of the room's checks on deliberately corrupted data and must print FAIL.
A planted check that prints PASS here is itself reported as a failure of this script ("DETECTOR BLIND")."""
import os, sys, json, pickle, subprocess, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import common
from common import *
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
blind = []
def expect_fail(label, cond, detail=""):
    """cond is the check as the production scripts evaluate it, applied to planted data."""
    fired = not cond
    print(f"[{'FAIL' if fired else 'PASS'}] planted: {label}" + (f"  ({detail})" if detail else ""),
          "-> detector fired" if fired else "-> DETECTOR BLIND")
    if not fired:
        blind.append(label)

# 1. generator norm, with q2 built from phi^{-1} replaced by phi
half = Fr(1, 2)
bad_q2 = Quat(PHI * half, PHI * half, half, 0)
expect_fail("0(a) |q2|^2 == 1 with q2 = (phi + phi i + j)/2", bad_q2.norm2() == Q5(1), str(bad_q2.norm2()))

# 2. character orthonormality with a corrupted character (one class value changed)
gd = json.load(open(os.path.join(OUT, "group_data.json"))); grp = json.load(open(os.path.join(OUT, "group.json")))
chars = {nm: [C5(Q5(Fr(a), Fr(b)), Q5(Fr(c), Fr(d))) for a, b, c, d in v] for nm, v in gd["chars"].items()}
sizes = [c["size"] for c in grp["classes"]]
def inner(c1, c2):
    s = C5(0)
    for k, nn in enumerate(sizes):
        s = s + nn * c1[k] * c2[k].conj()
    return C5(s.r / 120, s.i / 120)
bad = list(chars["3'"]); bad[4] = bad[4] + 1
expect_fail("0(a) <chi, chi> == 1 for 3' with one value shifted by 1", inner(bad, bad) == C5(1), str(inner(bad, bad)))

# 3. CG product rule with one CG coefficient sign-flipped
mp.mp.dps = 30
random.seed(1)
v = [mp.mpf(random.uniform(-1, 1)) for _ in range(4)]; nv = mp.sqrt(sum(x * x for x in v)); v = [x / nv for x in v]
g = [[mp.mpc(v[0], v[1]), mp.mpc(v[2], v[3])], [mp.mpc(-v[2], v[3]), mp.mpc(v[0], -v[1])]]
j1 = j2 = sp.Integer(3); J = sp.Integer(6)
W = mp.matrix([[mp.mpf(str(sp.N(x, 40))) for x in row] for row in W_matrix(J, j1, j2).tolist()])
W[6, 3 * 7 + 3] = -W[6, 3 * 7 + 3]              # <3 0; 3 0 | 6 0> sign flipped
D1 = D_mp(j1, g); DJ = D_mp(J, g)
K = mp.matrix(49, 49)
for a in range(7):
    for b in range(7):
        for c in range(7):
            for d in range(7):
                K[a * 7 + b, c * 7 + d] = D1[a, c] * D1[b, d]
err = max(abs(x) for x in (W * K - DJ * W))
expect_fail("CG: W_6 (D^3 (x) D^3) = D^6 W_6 with <3 0;3 0|6 0> sign-flipped", err < mp.mpf(10) ** -25, mp.nstr(err, 3))

# 4. Theta commuting with D, with the (-1)^(j-m) phase dropped from U
U = mp.matrix([[1 if c == 6 - r else 0 for c in range(7)] for r in range(7)])
Db = mp.matrix([[mp.conj(D1[a, b]) for b in range(7)] for a in range(7)])
err = max(abs(x) for x in (U * Db - D1 * U))
expect_fail("Theta commutes with D^3 when Theta drops the (-1)^(j-m) phase", err < mp.mpf(10) ** -25, mp.nstr(err, 3))

# 5. s_N: cubature value compared against a wrong closed form dim(rho)/(2j) instead of dim(rho)/(2j+1)
slots = pickle.load(open(os.path.join(OUT, "slots.pkl"), "rb"))
sN_num = sp.Rational(slots["R4"]["sN"])        # = cubature value to 1e-40 (s1_slots.log)
expect_fail("0(c) R4: s_N agrees with dim rho/(2j) (planted wrong formula)", abs(float(sN_num) - 3 / 6) < 1e-40)

# 6. invariance test of c_J: replace the isotypic projector P by a non-invariant projector (onto e_3, e_2, e_1)
consts = pickle.load(open(os.path.join(OUT, "consts.pkl"), "rb"))
n = 7; j = sp.Integer(3)
Pbad = sp.diag(1, 1, 1, 0, 0, 0, 0)
Z = Pbad * theta_U(j).T
z = sp.Matrix([Z[a, b] for a in range(n) for b in range(n)])
nz = [Jx for Jx in range(7) if (W_matrix(Jx, j, j) * z).norm() != 0]
expect_fail("c_J nonzero only for J in {0, 6}, with a non-invariant projector", set(nz) == {0, 6}, str(nz))

# 7. level norm comparison with one g_K perturbed by 1e-6 relative (R5, level 14)
log = open(os.path.join(HERE, "s2_constants.log")).read()
line = [l for l in log.splitlines() if l.strip().startswith("level 14") and "cubature" in l]
cub, ex = float(line[-1].split()[3]), float(line[-1].split()[6])
expect_fail("R5 level-14 norm: cubature == CG formula when g_7 is scaled by (1 + 1e-6)", abs(cub - ex * (1 + 1e-6)) < 1e-12,
            f"{cub} vs {ex*(1+1e-6)}")

# 8. exactness of criticality: a census point moved off its critical set must be reported non-critical
from crit import critical_exact   # the production test of s3_points.py
w = sp.Matrix([0, 0, 0, 0, 0, 0, 0]); w[1] = 1; w[0] = sp.Rational(1, 10)     # v2 + v3/10
nw = sp.sqrt(sum(abs(x) ** 2 for x in w))
crit, resid, Fv = critical_exact(sp.Integer(3), w / nw)
expect_fail("exact criticality test reports v2 + v3/10 critical", crit)

# 9/10. Krawczyk with a centre shifted by 1e-55 in one coordinate, and with c6 off by 1e-6 relative
for plant in ("shift-centre", "wrong-c6"):
    env = dict(os.environ); env["PLANT"] = plant
    r = subprocess.run([sys.executable, "-S", os.path.join(HERE, "s4_boxes.py")], capture_output=True, text=True, env=env)
    kl = [l for l in r.stdout.splitlines() if "Krawczyk" in l]
    print("   s4_boxes.py with PLANT=" + plant + ":", kl)
    expect_fail(f"Krawczyk on O1 R4 with PLANT={plant}", bool(kl) and all(l.startswith("[PASS]") for l in kl))

print("planted defects run:", "all detectors fired" if not blind else f"BLIND: {blind}")
