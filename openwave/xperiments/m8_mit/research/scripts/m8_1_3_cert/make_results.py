"""Assemble results.json from the outputs of the other scripts (no new computation except the
sympy simplification of the 0(d) coefficient to a radical string)."""
import json, os
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))


def rd(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def main():
    a, b, c, d, D, B4 = (rd(n) for n in ("out_0a.json", "out_0b.json", "out_0c.json", "out_0d.json",
                                         "out_D.json", "out_B4.json"))
    res = {}
    res["0(a)"] = {
        "provenance": "DERIVED in this room from the two generators (exact arithmetic in Q(sqrt5))",
        "norm q1": "1", "norm q2": "1",
        "order": str(a["order"]),
        "conjugacy classes": a["classes"],
        "irreducible characters (values in the class order above)": a["characters"],
    }
    res["0(b)"] = {
        "provenance": "DERIVED (character inner products with chi_{V_{l/2}} = U_l(w))",
        "irreducibles": [{"dim": t["dim"], "lowest level": t["lowest level"],
                          "levels <= 20 with multiplicity": {str(k): v for k, v in t["levels<=20 {level: multiplicity}"].items()}}
                         for t in b["table"]],
        "names": {nm: {"dim": v["dim"], "lowest level": v["lowest level"], "unique": True} for nm, v in b["names"].items()},
    }
    res["0(c)"] = {}
    for nm in ("R4", "R5"):
        res["0(c)"][nm] = {"multiplicity at lowest level": c[nm]["multiplicity"],
                           "eta (rows m = 3..-3, columns a), exact": c[nm]["eta exact (rows m = 3..-3, columns a)"],
                           "s_N": c[nm]["s_N"]}
    r = sp.Rational(d["cg_33_3m3_60"]["r"]) * sp.sqrt(sp.Rational(d["cg_33_3m3_60"]["R"]))
    res["0(d)"] = {"<3 3; 3 -3 | 6 0>": str(sp.radsimp(sp.nsimplify(r))),
                   "sign": "+" if d["cg_33_3m3_60"]["sign"] > 0 else "-",
                   "Theta^2 on V_3": d["Theta2_V3"], "Theta^2 on V_7/2": d["Theta2_V7/2"]}
    res["D"] = {"R4": {}, "R5": {}}
    for box, v in D.items():
        sl = box.split()[1]
        res["D"][sl][box] = {
            "Krawczyk test on the Section-5 box": "passed" if v["Krawczyk on Section-5 box"] else "failed",
            "Krawczyk data": v["Krawczyk info"],
            "certified box (refined) max width": v["certified box max width"],
            "lambda": v["lambda enclosure"],
            "924 r6": v["924 r6"],
            "signature": v["signature"],
            "Gram determinant": v["Gram det"],
            "k": v["k"],
            "r_u": v["r_u"],
            "zero multiplicity": v["zero multiplicity"],
            "slow roots (nu)": v["slow roots"],
            "slow root signs": ["-" if s < 0 else "+" for s in v["slow root signs"]],
            "slow roots real and simple": v["slow roots certified real simple"],
        }
    res["B.4"] = B4
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("results.json written with items:", list(res))


if __name__ == "__main__":
    main()
