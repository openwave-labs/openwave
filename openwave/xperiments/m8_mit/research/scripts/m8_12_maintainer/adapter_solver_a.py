"""Normalize solver_a's return. The orbit map is by (924*r6, stabilizer), both of
which the room computed itself, so it does not lean on the frozen spectra."""
import json, os, pathlib
R = json.loads(pathlib.Path(os.environ.get('ROOM_ROOT', '.') + '/solver_a/results.json').read_text())

MAP = {"v3": "coherent v3", "v2": "v2", "v1": "v1", "v0": "zonal v0",
       "xyz": "octahedron", "cat": "hexagon",
       "A*": "C3 ray",      # 1188/5, stabilizer C3
       "D*": "pyramid",     # 1188/5, stabilizer C5
       "F*": "D2 ray",      # 225,    stabilizer D2
       "G*": "prism"}       # 8800/43, stabilizer D3

def normalize():
    orbits = {}
    for k, v in R["headline"]["orbits"].items():
        orbits[MAP[k]] = {
            "r6_924": f"924*({v['rhat6']})",
            "dim_N": v["dim_N"],
            "sig": v["signature"],
            "spectrum": v["charpoly"].replace("lam", "l"),
        }
    i11 = R["item11"]
    keymap = {"i u": "i*u", "-i Jx u": "-i*Jx*u", "-i Jy u": "-i*Jy*u", "-i Jz u": "-i*Jz*u"}
    n1 = {"r6_924": i11["924*rhat6"],
          "grad": i11["tangential_gradient_norm"],
          "residuals": {keymap[k]: v["exact"] for k, v in i11["||M_u d||"].items()}}
    return {
        "orbits": orbits,
        "lines": {},                      # charts differ; verified against r6 in verify_lines.py
        "classification": {"n_points": len(R["item1"]["classes_dim1"]),
                           "n_lines": len(R["item1"]["classes_dim2"]),
                           "n_orbits": len(R["headline"]["orbits"]),
                           "proj_dim_ge_2": {"C1": [7], "C2": [4, 3], "C3": [3]}},
        "N1": n1,
        # line "B" is the {v1, v-3} locus: two critical loci, both endpoints, and the
        # room records the leading order as (-8/33)|z|^4, so the linear coefficient vanishes
        "N2": {"linear_coeff": "0",
               "interior_critical": any(c["where"] not in ("z=0", "omitted point [b]")
                                        for c in R["item2"]["B"]["critical_set"])},
        "H3": {"kernel_dim": R["headline"]["orbits"]["v1"]["signature"][1],
               "kernel_is_C4_line_direction": None},
        # DERIVED from the room's own evidence, never asserted: an end counts as
        # established by argument only if the room's proved bound EQUALS its claimed
        # value there. A search that lands on the right number is still a search.
        "G": {"min_orbit": "coherent v3", "max_orbit": "hexagon",
              "min_by": ("argument" if "identity" in R["item8"]["min"] else "search"),
              "max_by": ("argument" if ("certificate" in R["item8"]["max"]
                                        and R["item8"]["max"]["value"] == "463/924")
                         else "search")},
    }
