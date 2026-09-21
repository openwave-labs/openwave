"""A synthetic room that answers exactly the frozen values: the compare must go green."""
import json, pathlib
HERE = pathlib.Path(__file__).resolve().parent

def normalize():
    F = json.loads((HERE / "frozen_claims.json").read_text())
    orbits = {}
    for c in F["census"]:
        orbits[c["orbit"]] = {"r6_924": c["r6_924"], "dim_N": c["dim_N"], "sig": c["sig"],
                              "spectrum": F["spectra"][c["orbit"]]}
    lines = {}
    for ln in F["lines"]:
        lines[ln["name"]] = {"restriction": ln["restriction"], "chart": ln["chart"],
                             "critical": [ln["s_star"]] if ln["s_star"] else None}
    return {
        "orbits": orbits, "lines": lines,
        "classification": {"n_points": 6, "n_lines": 7, "n_orbits": 10,
                           "proj_dim_ge_2": F["classification"]["proj_dim_ge_2"]},
        "N1": {"r6_924": F["N1"]["r6_924"], "grad": F["N1"]["grad_tangential_norm"],
               "residuals": F["N1"]["residuals"]},
        "N2": {"linear_coeff": "0", "interior_critical": False},
        "H3": {"kernel_dim": 2, "kernel_is_C4_line_direction": True},
        "G": {"min_orbit": "coherent v3", "max_orbit": "hexagon",
              "min_by": "argument", "max_by": "argument"},
    }
