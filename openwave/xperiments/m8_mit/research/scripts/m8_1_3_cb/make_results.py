"""Assemble results.json from the outputs of the scripts (out/*.json).  Prints what it writes."""
import os, sys, json
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp
from lib_check import check, summary

O = lambda f: json.load(open(os.path.join(HERE, "out", f)))
grp, eta, etax, c12, c3, c3x, c5 = (O("group.json"), O("eta.json"), O("eta_exact.json"), O("C12.json"),
                                    O("C3.json"), O("C3_exact.json"), O("C5.json"))
R = {}
lab = lambda d: f"{d['dim']}@{d['lowest_level']}"
R["0(a)"] = {
    "|q1|^2": "1", "|q2|^2": "1", "order": "120",
    "conjugacy classes": [{"size": s, "element order": o, "real part": w} for s, o, w in
                          zip(grp["class_sizes"], grp["class_orders"], grp["class_real_parts"])],
    "irreducible characters (label dim@lowest level; columns = classes above)": {lab(d): d["character_sympy"] for d in grp["irreducibles"]},
    "status": "DERIVED in this room from q1, q2 (closure, conjugation, characters of V_j restricted, Galois conjugation, orthogonality)",
}
R["0(b)"] = {lab(d): {"dim": d["dim"], "levels (l: multiplicity), l <= 20": {str(k): v for k, v in sorted(d["levels"].items(), key=lambda t: int(t[0]))}}
             for d in grp["irreducibles"]}
R["0(b)"]["slot names"] = {s: lab([d for d in grp["irreducibles"] if d["index"] == i][0]) + " (unique)" for s, i in grp["slot_index"].items()}
R["0(c)"] = {s: {"multiplicity at lowest level": eta[s]["multiplicity"], "eta (rows m = j..-j, columns a)": etax[s], "s_N": eta[s]["s_N"]}
             for s in ("R2", "R4", "R5")}
R["0(d)"] = {"<3 3; 3 -3 | 6 0>": {"sign": "+", "value": "sqrt(924)/924"}, "Theta^2 on V_3": "1", "Theta^2 on V_7/2": "-1"}
# 0(e): C of R2, exact after a check that the numeric C is the integer matrix
C = eta["R2"]["C"]
Cn = [[mp.mpc(mp.mpf(a), mp.mpf(b)) for a, b in row] for row in C]
Cint = [[int(mp.nint(mp.re(z))) for z in row] for row in Cn]
check("R2: C equals an integer matrix to 1e-35", max(abs(Cn[r][c] - Cint[r][c]) for r in range(2) for c in range(2)) < mp.mpf(10) ** -35)
cph = mp.mpc(mp.mpf(eta["R2"]["c_phase"][0]), mp.mpf(eta["R2"]["c_phase"][1]))
check("R2: (Theta eta) C = -eta", abs(cph + 1) < mp.mpf(10) ** -35)
R["0(e)"] = {"R2": {"C": [[str(x) for x in row] for row in Cint], "C conj(C)": str(eta["R2"]["CCbar"]),
                    "psi -> conj(psi) C on E_0, read on w": "w -> -Theta(w)  (conj(psi_w) C = psi_{-Theta w}; equals Theta up to the sign -1)"}}
R["C.1"] = {s: c12["C.1"][s] for s in ("R2", "R4", "R5")}
R["C.2"] = {s: c12["C.2"][s] for s in ("R2", "R4", "R5")}
r2 = dict(c3["R2 line {5/2, -3/2}"])
r2["gamma_u"] = c3x["R2 line {5/2, -3/2}"]["gamma_u"] + "  (times R^2)"
r2["R~0(u) (exact)"] = c3x["R2 line {5/2, -3/2}"]["R~0(u)"]
R["C.3"] = {"R2": {"line {5/2, -3/2}": r2}}
for s in ("R4", "R5"):
    v = dict(c3[f"v1 {s}"])
    v["gamma_u"] = c3x[f"v1 {s}"]["gamma_u"] + "  (times R^2)"
    v["R~0(v1) (exact)"] = c3x[f"v1 {s}"]["R~0(u)"]
    R["C.3"][s] = {"v1": v}
    check(f"C.3 {s}: exact and identified gamma agree", c3x[f"v1 {s}"]["gamma_u"] == c3[f"v1 {s}"]["gamma_u"])
check("C.3 R2: exact and identified gamma agree", c3x["R2 line {5/2, -3/2}"]["gamma_u"] == c3["R2 line {5/2, -3/2}"]["gamma_u"])
R["C.5"] = {}
for key, val in c5.items():
    pname, slot = key.rsplit(" in ", 1)
    R["C.5"].setdefault(slot, {})[pname] = val
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1)
print(json.dumps(R, indent=1)[:3000])
print("... results.json written")
summary()
