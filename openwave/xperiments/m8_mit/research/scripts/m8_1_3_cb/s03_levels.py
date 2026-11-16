"""Items C.1 and C.2.
C.1: s_N (from s02), then at a unit y with no symmetry the level norms ||Pi_L N(psi_y)||^2 (V = 1)
for every level n < L <= 20, exactly (route A: polynomial Haar moments from the definitions,
no CG) and at 50 digits (route B: CG engine); the two must agree.
C.2: next level at which the slot occurs (from the derived character table, s00) and
lambda_1 - lambda_0 in units of 1/R^2."""
import os, sys, json, pickle
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp
import numpy as np
from lib_check import check, summary, DEFECT
from lib_haar import Section
from lib_engine import Slot
from lib_cg import mrange, w as wfac
from lib_sym import stabilizer_search

mp.mp.dps = 50
grp = json.load(open(os.path.join(HERE, "out", "group.json")))
etaj = json.load(open(os.path.join(HERE, "out", "eta.json")))
res = {"C.1": {}, "C.2": {}}
# the chosen y, by weighted coordinates p_m (Gaussian integers), m = j..-j; y_m = p_m sqrt((j+m)!(j-m)!),
# then normalised.  Chosen with all components nonzero and no pattern.
PCHOICE = {
    "R2": [(1, 2), (0, -1), (3, 1), (1, 0), (-2, 1), (1, 1), (0, 2), (1, -1)],
    "R4": [(1, 2), (0, -1), (3, 1), (1, 0), (-2, 1), (1, 1), (0, 2)],
    "R5": [(1, 2), (0, -1), (3, 1), (1, 0), (-2, 1), (1, 1), (0, 2)],
}
for slot, low in (("R2", 7), ("R4", 6), ("R5", 6)):
    j = Fr(low, 2)
    idx = grp["slot_index"][slot]
    irr = [d for d in grp["irreducibles"] if d["index"] == idx][0]
    levels = {int(k): v for k, v in irr["levels"].items()}
    sN = Fr(etaj[slot]["s_N"])
    print(f"\n===== {slot}: lowest level n = {low}, s_N = {sN} =====")
    # ---- C.2
    above = sorted(l for l in levels if l > low)
    L1 = above[0]
    gap = L1 * (L1 + 2) - low * (low + 2)
    print(f"C.2 next level at which {slot} occurs: {L1} (multiplicity {levels[L1]}); lambda_1 - lambda_0 = {L1}*{L1+2} - {low}*{low+2} = {gap} / R^2")
    res["C.2"][slot] = {"next level": L1, "lambda_1 - lambda_0": str(gap)}
    # ---- C.1
    p = PCHOICE[slot]
    ms = mrange(j)
    norm2 = sum(Fr(a * a + b * b) * wfac(j, m) for (a, b), m in zip(p, ms))
    ystr = {str(m): f"({a} + {b}*I)*sqrt({wfac(j, m)})/sqrt({norm2})" for (a, b), m in zip(p, ms)}
    print(f"C.1 y (unit, CS components): y_m = (p_m) sqrt((j+m)!(j-m)!)/sqrt({norm2}), p = {p}")
    y = [mp.mpc(a, b) * mp.sqrt(wfac(j, m)) / mp.sqrt(norm2) for (a, b), m in zip(p, ms)]
    # no symmetry: maximal orbit dimension and no finite stabilizer
    yn = np.array([complex(c) for c in y])
    quat = slot == "R2"
    found = stabilizer_search(j, yn, quat, 0, seed=1)
    has_id = any(np.linalg.norm(R - np.eye(3)) < 1e-6 for R, q in found)
    nontriv = [R for R, q in found if np.linalg.norm(R - np.eye(3)) > 1e-6]
    found_t = [] if quat else stabilizer_search(j, yn, False, 1, seed=2)
    print(f"   symmetry search (2e5-point grid + least squares): stabilizing rotations found = {len(found)}"
          f" (identity among them: {has_id}); with charge conjugation: {len(found_t) if not quat else 'n/a (Sp(1) included above)'}")
    check(f"{slot}: symmetry search finds the identity (search is live)", has_id)
    check(f"{slot}: y has no nontrivial stabilizer in Gamma (search)", len(nontriv) == 0 and len(found_t) == 0)
    # route A exact
    P = pickle.load(open(os.path.join(HERE, "out", f"Pmono_{slot}.pkl"), "rb"))
    S = Section(j, P["Pm"], sN * (Fr(8, 7) if DEFECT == "wrong_sN_routeA" else 1), p)
    allJ = [Fr(L, 2) for L in range(low + 1, 22)]
    LA = S.level_norms(allJ)
    # route B 50 digits
    B = Slot(slot)
    LB = B.level_norms(y)
    out = {}
    nonzero = []
    agree = True
    for J in allJ:
        L = int(2 * J)
        a0, a1 = LA[J]
        exact = (a0 / norm2 ** 3, a1 / norm2 ** 3)
        vb = LB.get(J, mp.mpf(0))
        diff = abs(vb - (mp.mpf(exact[0].numerator) / exact[0].denominator + mp.mpf(exact[1].numerator) / exact[1].denominator * mp.sqrt(5)))
        agree &= diff < mp.mpf(10) ** -40
        check(f"{slot} L={L}: exact value has no sqrt5 part", exact[1] == 0)
        val = exact[0]
        if L <= 20:
            out[str(L)] = str(val)
        occ = levels.get(L, 0) if L <= 20 else None
        tag = "nonzero" if val != 0 else "zero"
        if val != 0:
            nonzero.append(L)
        print(f"   L={L:2d} ({'slot occurs x%d' % occ if occ else ('slot absent' if occ == 0 else 'L > 20, outside the asked range')}): "
              f"||Pi_L N||^2 = {val}  ~ {mp.nstr(vb, 30)}   [{tag}]")
        if L <= 20:
            check(f"{slot} L={L}: norm nonzero only where the slot occurs (0(b))", (val != 0) <= (levels.get(L, 0) > 0))
    check(f"{slot}: route A (exact) and route B (50 digits) agree on every level to 1e-40", agree)
    # totals: lowest level part and mean|psi|^6
    tot6 = S.mean_psi6()[0] / norm2 ** 3
    ln0 = LB[j]
    sumB = mp.fsum(LB.values())
    check(f"{slot}: sum of all level norms (route B) = mean|psi|^6 (route A, exact)", abs(sumB - mp.mpf(tot6.numerator) / tot6.denominator) < mp.mpf(10) ** -40)
    print(f"   lowest-level part ||Pi_n N||^2 = {mp.nstr(ln0, 30)};  mean|psi_y|^6 = {tot6} ~ {mp.nstr(mp.mpf(tot6.numerator)/tot6.denominator, 25)}")
    print(f"C.1 {slot}: levels n < L <= 20 with nonzero norm: {[L for L in nonzero if L <= 20]}")
    res["C.1"][slot] = {"s_N": str(sN), "y": ystr, "level norms": out, "nonzero levels": [L for L in nonzero if L <= 20]}
    if slot == "R2":
        res["C.1"][slot]["level 21 (outside the asked range)"] = str(LA[Fr(21, 2)][0] / norm2 ** 3)

if not DEFECT:
    json.dump(res, open(os.path.join(HERE, "out", "C12.json"), "w"), indent=1)
summary()
