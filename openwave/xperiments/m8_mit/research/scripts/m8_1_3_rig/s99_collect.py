"""Assemble results.json from out/s01..s04.json (items 0(a), 0(b), E.1, E.2 only)."""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda n: json.load(open(os.path.join(HERE, "out", n)))
s01, s02, s03, s04 = L("s01.json"), L("s02.json"), L("s03.json"), L("s04.json")
res = {"0(a)": s01["0(a)"], "0(b)": s01["0(b)"]}
res["E.1"] = {lab: {"dim": r["dim"], "n": r["n"], "next level": r["next level"],
                    "lambda1 - lambda0 (units 1/R^2)": r["lambda1 - lambda0"]} for lab, r in s02.items()}
res["E.2"] = {
    "reading": "eigenvalues of the real 4d x 4d generator A of d/dT (u, u_T), zeta = psi_u in E_0, rotating frame psi = e^{i omega T}(psi_v + zeta); r = |v|; valid for v != 0, omega > 0, g_4 > 0",
    "slots": {lab: {"d": r["d"], "n": r["n"], "charpoly in mu": r["charpoly"], "eigenvalues": r["eigenvalues"],
                    "E0 decoupled from next level (relative E0 component, quadrature)": s04[lab]["relative E0 component"]}
              for lab, r in s03.items()},
}
json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
print("wrote results.json with items", list(res))
