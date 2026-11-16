"""Collect the per-script outputs (out/*.json) into results.json."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
ld = lambda f: json.load(open(os.path.join(OUT, f)))
grp, cde, c12, pts, boxes = ld("group.json"), ld("anchors_cde.json"), ld("C12.json"), ld("C4_points.json"), ld("C4_boxes.json")

R = {}
R["0(a)"] = {"|q1|^2": "1", "|q2|^2": "1", "order": grp["order"],
             "classes": [{k: c[k] for k in ("size", "order", "real_part")} for c in grp["classes"]],
             "characters (columns = classes in the order above)": grp["characters"],
             "provenance": "DERIVED in this room (closure, explicit conjugation, characters of V_j, 2' = V_7/2 - 6, 4 = 2 x 2', 3' = V_3 - 4)"}
R["0(b)"] = {"levels l <= 20 with multiplicities, per irreducible": grp["levels"],
             "dimensions": {nm: int(v[0]) for nm, v in grp["characters"].items()},
             "slot -> irreducible": grp["slot_irreducible"], "provenance": "DERIVED (character inner products)"}
R["0(c)"] = {s: v for s, v in cde["0(c)"].items()}
R["0(d)"] = cde["0(d)"]
R["0(e)"] = {"R2": cde["0(e)"]}
R["C.1"] = c12["C.1"]
R["C.2"] = c12["C.2"]
C4 = {}
for s in ("R2", "R4", "R5"):
    C4[s] = {}
    for name, e in pts[s].items():
        C4[s][name] = {"critical": e["critical"]}
        if e["critical"]:
            C4[s][name].update({"sign": e["sign"], "R0~": e["R0~ exact (V = R = 1)"]})
    for name, e in boxes.get(s, {}).items():
        C4[s][f"{name} {s}"] = {"critical": e["critical (c = 0 at the certified zero)"], "sign": e["sign"],
                                "R0~ enclosure over the box": e["R0~ enclosure over the box"],
                                "width": e["R0~ enclosure width"]}
R["C.4"] = C4
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1)
print("results.json written with items:", list(R))
