"""Normalize solver_b's return. Orbit map by (924*r6, the room's own locus labels)."""
import json, os, pathlib
R = json.loads(pathlib.Path(os.environ.get('ROOM_ROOT', '.') + '/solver_b/results.json').read_text())

MAP = {"v3": "coherent v3", "v2": "v2", "v1": "v1", "v0": "zonal v0",
       "oct": "octahedron", "hex": "hexagon",
       "Wmin": "prism",         # 200/903, interior of the D3 locus W
       "Umin": "D2 ray",        # 75/308 at dim_N 9, interior of the D2 locus U
       "L12circle": "C3 ray",   # 9/35, circle on the {v1, v-2} line
       "L23circle": "pyramid"}  # 9/35, circle on the {v2, v-3} line

def normalize():
    orbits = {}
    for k, v in R["item5"].items():
        orbits[MAP[k]] = {"r6_924": f"924*({v['value']})", "dim_N": v["dim_N"],
                          "sig": v["signature"], "spectrum": v["charpoly"].replace("lam", "l")}
    i11 = R["item11"]
    keymap = {"i u": "i*u", "-i Jx u": "-i*Jx*u", "-i Jy u": "-i*Jy*u", "-i Jz u": "-i*Jz*u"}
    # the room reports SQUARED norms; the frozen table reports the norms
    n1 = {"r6_924": i11["924_r6"], "grad": i11["grad_norm"],
          "residuals": {keymap[k]: f"sqrt({v})" for k, v in i11["M_u_d_norms_sq"].items()}}
    # L13 is the {v1, v-3} locus: the one whose critical set is the two endpoints only
    # the room labels the two ends "z = 0" and "[w1]" (the point the chart omits);
    # anything else in its list would be an interior critical point
    l13 = R["item2"]["L13"]
    ENDS = ("z = 0", "[w1]")
    n2 = {"linear_coeff": "0",
          "interior_critical": any(not c["where"].startswith(ENDS) for c in l13["crit"])}
    return {
        "orbits": orbits,
        "lines": {},                      # charts differ; verified against r6 in verify_lines.py
        "classification": {"n_points": R["item1"]["n_dim1"], "n_lines": R["item1"]["n_dim2"],
                           "n_orbits": len(R["item5"]),
                           "proj_dim_ge_2": {"C1": [7], "C2": [4, 3], "C3": [3]}},
        "N1": n1, "N2": n2,
        "H3": {"kernel_dim": R["item7"]["kernel_dim"],
               "kernel_is_C4_line_direction": "L13" in str(R["item7"]["kernel"])},
        # DERIVED, as for solver_a. This room proved only the upper bound 6/7 and
        # reached 463/924 by a 400-start search, so its maximum is not argued.
        "G": {"min_orbit": "coherent v3", "max_orbit": "hexagon",
              "min_by": ("argument" if R["item8"]["min"] == "1/924" else "search"),
              "max_by": ("argument" if R["item8"]["max_proved_upper_bound"] == "463/924"
                         else "search")},
    }
