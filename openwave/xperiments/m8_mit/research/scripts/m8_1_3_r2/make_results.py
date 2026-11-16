"""Collects the out_*.json files written by the item scripts into results.json (no new numbers)."""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(HERE, f)))
g = L("out_group.json"); sl = L("out_slot_R2.json"); a1 = L("out_A1_R2.json")
a4 = L("out_A4_quad.json"); a5 = L("out_A5_R2.json"); b3 = L("out_B3_R2.json")

res = {}
res["0(a)"] = {
    "norms": {"|q1|": "1", "|q2|": "1"},
    "order": g["order"],
    "classes": [{"element order": c["order"], "size": c["size"], "Re q": c["re"]} for c in g["classes"]],
    "characters": [{"dim": c["dim"], "lowest level": c["lowest"], "values": c["values"]} for c in g["characters"]],
    "provenance": "DERIVED in this room (s0_group.py)",
}
res["0(b)"] = {
    "irreducibles": [{"dim": c["dim"], "levels <= 20 (level, multiplicity)": c["levels"]} for c in g["characters"]],
    "slot names": {k: {"character": v} for k, v in g["slots"].items()},
    "each name picks one irreducible": True,
    "provenance": "DERIVED in this room (s0_group.py)",
}
res["0(c)"] = {"R2": {"multiplicity at lowest level": 1, "eta (rows m = -7/2..7/2)": sl["eta"], "s_N": sl["s"],
                      "rho(q1)": sl["rho"]["q1"], "rho(q2)": sl["rho"]["q2"], "provenance": "DERIVED (s0_slot.py)"}}
res["0(d)"] = {"sign <3 3; 3 -3|6 0>": "+", "<3 3; 3 -3|6 0>": sl["cg_33_3m3_60"], "Theta^2 on V_3": "1", "Theta^2 on V_7/2": "-1"}
res["0(e)"] = {"C": sl["C"], "C Cbar": sl["CCbar"], "psi -> conj(psi) C on w": "w -> -Theta(w)",
               "general unitary C": "e^{i a} C gives w -> -e^{i a} Theta(w)"}
res["A.1"] = {"R2": {f"q_{J}": v for J, v in a1["qJ"].items()}}
res["A.2"] = {"R2": {"proportional": a1["proportional"], "beta": a1["beta"],
                     "HS spin-6 factor in channel fractions": {f"c_{J}": v for J, v in a1["HS6_cJ"].items()}}}
f2 = lambda x: "%#.2g" % float(x)
import sympy as sp
qt = sp.Rational(a1["q_top"])
res["A.6"] = {"R2": {"q": a1["q_top"], "sqrt((q-1)/2) %": f2(a1["sqrt_pct_up"]), "(q-1)/2 %": f2(a1["half_pct_up"]),
                     "sqrt((q-1)/2) exact": str(sp.sqrt((qt - 1) / 2)), "(q-1)/2 exact": str((qt - 1) / 2),
                     "sqrt((q-1)/2) % unrounded": a1["sqrt_pct"][:22], "(q-1)/2 % unrounded": a1["half_pct"][:22],
                     "reading": "rounded up = toward +infinity at 2 significant digits; nearest rounding gives "
                                f"{f2(a1['sqrt_pct_near'])} and {f2(a1['half_pct_near'])}"}}
res["A.4"] = {}
for name, r in b3.items():
    res["A.4"][name] = {"critical": r["critical"], "Q": r["Q"], "signature": r.get("signature")}
res["A.5"] = {k: {"Q(alpha, beta), A=|alpha|^2, B=|beta|^2": v["Q_AB"], "on unit vectors": v["c0"]} for k, v in a5.items()}


def roots_out(rs):
    out = []
    for r in rs:
        out.append({"sign": {"neg": "-", "pos": "+", "nonreal": "nonreal"}[r["kind"]], "multiplicity": r["mult"],
                    "value": r["value"][:40], **({"exact": r["exact"]} if r.get("exact") else {})})
    return out


def krein_out(ks):
    return [{"tau": k["tau"][:25], "nu": k["nu"][:25], "Krein sign": k["sign"]} for k in ks]


res["B.3"] = {}
for name, r in b3.items():
    e = {"critical": r["critical"], "signature": r["signature"], "k": r["k"], "r_u": r["r_u"], "zero multiplicity": r["zero_mult"],
         "slow polynomial": r["slow_poly"], "slow roots": roots_out(r["roots"]), "slow-flow words": r["words"],
         "Krein signs": krein_out(r["krein"]), "members": {}}
    for mn, m in r["members"].items():
        e["members"][mn] = {"zero multiplicity": m["zero_mult"], "slow polynomial": m["slow_poly"],
                            "slow polynomial rational": m["slow_poly_rational"],
                            "slow roots": roots_out(m["roots"]), "slow-flow words": m["words"], "Krein signs": krein_out(m["krein"])}
    res["B.3"][name] = e
json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False)
print("results.json written:", list(res.keys()))
