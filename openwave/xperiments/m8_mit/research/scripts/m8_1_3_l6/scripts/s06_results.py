"""Assemble results.json (room top level) from the outputs of s01-s05.  Prints nothing new except
a check that every value it writes came from out/*.json."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import cg, Checker
import sympy as sp

OUT = os.path.join(HERE, "out")
g = json.load(open(os.path.join(OUT, "group.json")))
it = json.load(open(os.path.join(OUT, "intertwiners.json")))
A = json.load(open(os.path.join(OUT, "blockA.json")))
B = json.load(open(os.path.join(OUT, "blockB.json")))

def q5(t):
    return str(sp.nsimplify(sp.Rational(t[0]) + sp.Rational(t[1]) * sp.sqrt(5)))

res = {}
res["0(a)"] = {
    "norms": {"||q1||": "1", "||q2||": "1"},
    "order": g["order"],
    "classes": [{"size": s, "element order": o, "Re g": q5(r)} for s, o, r in zip(g["class_sizes"], g["class_orders"], g["class_retr"])],
    "irreducible characters (rows; columns = classes in the order above)": [
        {"dim": d, "lowest level": fl, "values": row} for d, fl, row in zip(g["dims"], g["first_level"], g["table_str"])],
    "provenance": "DERIVED in this room (s01_group.py)",
}
res["0(b)"] = {"irreducibles": [{"dim": g["dims"][int(i)], "levels <= 20 with multiplicity": {l: k for l, k in m.items()}}
                                for i, m in g["mult"].items()],
               "slot names": {k: f"irreducible #{v} (dim {g['dims'][v]}, lowest level {g['first_level'][v]}), unique" for k, v in g["slots"].items()},
               "provenance": "DERIVED in this room (s01_group.py)"}
res["0(c)"] = {}
for s in ("R4", "R5"):
    res["0(c)"][s] = {"multiplicity at lowest level": it[s]["multiplicity"], "eta (rows m=3..-3)": it[s]["eta"],
                      "s_N": it[s]["s_N"], "P = eta eta^+ (rows/cols m=3..-3)": it[s]["P"]}
c = cg(3, 3, 3, -3, 6, 0)
res["0(d)"] = {"<3 3; 3 -3 | 6 0>": str(sp.radsimp(c)), "sign": "+", "Theta^2 on V_3": "1", "Theta^2 on V_7/2": "-1"}
res["A.1"] = {s: {k: v for k, v in A["A.1"][s].items() if k.startswith("q_")} for s in ("R4", "R5")}
res["A.2"] = A["A.2"]
res["A.3"] = A["A.3"]
res["A.6"] = A["A.6"]
def clean(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}
res["B.1"] = {"R4": {p: clean(v) for p, v in B["B.1"].items()}}
res["B.2"] = {"R5": {p: clean(v) for p, v in B["B.2"].items() if not p.startswith("_")},
              "R5 vs R4 eigenvalue ratios (clause 3)": {p: {"common ratio +xi_R5/+xi_R4": v.get("exact common ratio"),
                                                          "pairs": v.get("pairs")} for p, v in B["B.2"]["_comparison_with_R4"].items()}}
res["B.5"] = B["B.5"]
json.dump(res, open(os.path.join(HERE, "..", "results.json"), "w"), indent=1, default=str)
chk = Checker()
chk("results.json written and re-readable", json.load(open(os.path.join(HERE, "..", "results.json")))["0(a)"]["order"] == g["order"])
chk.summary()
