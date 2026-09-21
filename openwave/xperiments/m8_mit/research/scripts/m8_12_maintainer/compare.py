"""Compare one room's return against the frozen M8.12 claims.

The room chooses its own labels, charts and output schema, so a thin per-room
ADAPTER (written after reading its return, and saved beside its snapshot as
`adapter_<room>.py`) normalizes the return into one dict:

    {
      "orbits": {
         "<frozen orbit name>": {
             "r6_924":  "<exact sympy source>",
             "dim_N":   <int>,
             "sig":     [n_minus, n_zero, n_plus],
             "spectrum": "<charpoly in l>"  or  "eigenvalues": ["<exact>", ...]
         }, ...
      },
      "lines":   {"<frozen line name>": {"restriction": "...", "chart": "s"|"z",
                                          "critical": ["<exact s or z>", ...]}},
      "classification": {"n_points": int, "n_lines": int, "n_orbits": int,
                          "proj_dim_ge_2": {"<group>": [dims]}},
      "N1": {"r6_924": "...", "grad": "...", "residuals": {...}},
      "N2": {"linear_coeff": "...", "interior_critical": bool},
      "H3": {"kernel_dim": int, "kernel_is_C4_line_direction": bool},
      "G":  {"min_orbit": "...", "max_orbit": "...", "min_by": "argument"|"search",
              "max_by": "argument"|"search"}
    }

Anything the room did not report is left absent, and is scored MISSING rather
than as a difference: a value a room did not reach is not a value that differs.

Usage:  python3 compare.py --adapter adapter_solver_a.py --label solver_a [--mutate N]
"""
import argparse
import importlib.util
import json
import pathlib
import sys

from sympy import (I, Poly, Rational, expand, factor, nsimplify, simplify,
                   sqrt, symbols, sympify)

HERE = pathlib.Path(__file__).resolve().parent
LAM = symbols("l", real=True)
ROWS = []


def ex(s):
    if isinstance(s, (int, float)):
        s = str(s)
    return sympify(s, locals={"Rational": Rational, "sqrt": sqrt, "I": I, "l": LAM,
                              "lam": LAM, "lambda_": LAM, "x": LAM})


def record(claim, item, verdict, detail=""):
    ROWS.append({"claim": claim, "item": item, "verdict": verdict, "detail": detail})
    mark = {"PASS": "PASS", "DIFFERS": "FAIL", "MISSING": "MISS"}[verdict]
    print(f"  {mark}  {claim} / {item}{('  ' + detail) if detail else ''}")


def cmp_exact(claim, item, got, want):
    if got is None:
        return record(claim, item, "MISSING")
    try:
        ok = simplify(ex(got) - ex(want)) == 0
    except Exception as e:                      # an unparseable return is a difference
        return record(claim, item, "DIFFERS", f"unparseable: {e}")
    record(claim, item, "PASS" if ok else "DIFFERS", "" if ok else f"got {got}, frozen {want}")


def monic(p):
    q = Poly(expand(ex(p)), LAM)
    return q.monic()


def cmp_poly(claim, item, got, want):
    """Equal up to a positive constant, so compare after monic normalization."""
    if got is None:
        return record(claim, item, "MISSING")
    try:
        ok = monic(got) == monic(want)
    except Exception as e:
        return record(claim, item, "DIFFERS", f"unparseable: {e}")
    record(claim, item, "PASS" if ok else "DIFFERS",
           "" if ok else f"got {monic(got).as_expr()}")


def cmp_roots(claim, item, got_roots, want_poly):
    """The room gave eigenvalues; compare the multiset against the frozen roots."""
    if got_roots is None:
        return record(claim, item, "MISSING")
    want = sorted(Poly(expand(ex(want_poly)), LAM).all_roots())
    got = sorted(ex(r) for r in got_roots)
    if len(got) != len(want):
        return record(claim, item, "DIFFERS", f"{len(got)} eigenvalues vs {len(want)} roots")
    bad = [(g, w) for g, w in zip(got, want) if simplify(g - w) != 0]
    record(claim, item, "PASS" if not bad else "DIFFERS",
           "" if not bad else f"{len(bad)} of {len(want)} differ, first {bad[0]}")


def cmp_int(claim, item, got, want):
    if got is None:
        return record(claim, item, "MISSING")
    record(claim, item, "PASS" if int(got) == int(want) else "DIFFERS",
           "" if int(got) == int(want) else f"got {got}, frozen {want}")


def cmp_list(claim, item, got, want):
    if got is None:
        return record(claim, item, "MISSING")
    record(claim, item, "PASS" if list(got) == list(want) else "DIFFERS",
           "" if list(got) == list(want) else f"got {got}, frozen {want}")


def cmp_valueset(claim, item, got, want):
    """Two sets of exact values, order-free."""
    if got is None:
        return record(claim, item, "MISSING")
    G = [ex(v) for v in got]
    W = [ex(v) for v in want]
    if len(G) != len(W):
        return record(claim, item, "DIFFERS", f"{len(G)} values vs {len(W)}")
    left = list(W)
    for g in G:
        hit = next((w for w in left if simplify(g - w) == 0), None)
        if hit is None:
            return record(claim, item, "DIFFERS", f"{g} is not in the frozen set")
        left.remove(hit)
    record(claim, item, "PASS")


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.normalize()


def apply_mutation(room, n):
    """Plant a defect in the ROOM's normalized return: the compare must catch it."""
    if n == 1:
        room["orbits"]["hexagon"]["sig"] = [8, 1, 0]
    elif n == 2:
        room["orbits"]["prism"]["r6_924"] = "Rational(8800,44)"
    elif n == 3:
        k = "spectrum" if "spectrum" in room["orbits"]["v1"] else "eigenvalues"
        if k == "spectrum":
            room["orbits"]["v1"][k] = "l**2*(3*l + 5)**2*(11*l - 3)**2*(363*l**2 - 374*l - 601)**2"
        else:
            room["orbits"]["v1"][k] = list(room["orbits"]["v1"][k])[:-1] + ["Rational(1,7)"]
    elif n == 4 and room.get("N1"):
        room["N1"]["grad"] = "2*sqrt(1003)/297"
    elif n == 5:
        room["classification"]["n_orbits"] = 11
    return room


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--mutate", type=int)
    a = ap.parse_args()

    F = json.loads((HERE / "frozen_claims.json").read_text())
    R = load_adapter(a.adapter)
    if a.mutate:
        print(f"### MUTATION {a.mutate} PLANTED in the room return: a red result is expected\n")
        R = apply_mutation(R, a.mutate)

    census = {c["orbit"]: c for c in F["census"]}

    print(f"=== P1 / H1 / H2: the census, orbit by orbit ({a.label}) ===")
    for name, c in census.items():
        got = R.get("orbits", {}).get(name)
        if got is None:
            record("H1", f"{name}: reported at all", "MISSING")
            continue
        cmp_exact("P1", f"{name}: 924*r6", got.get("r6_924"), c["r6_924"])
        cmp_int("H1", f"{name}: dim N_u", got.get("dim_N"), c["dim_N"])
        cmp_list("H1", f"{name}: signature", got.get("sig"), c["sig"])
        if got.get("spectrum") is not None:
            cmp_poly("H2", f"{name}: characteristic polynomial", got["spectrum"], F["spectra"][name])
        else:
            cmp_roots("H2", f"{name}: transverse eigenvalues", got.get("eigenvalues"), F["spectra"][name])

    print("\n=== L0 / L2: the classification counts ===")
    cl = R.get("classification", {})
    cmp_int("L0", "point classes", cl.get("n_points"), F["classification"]["n_points"])
    cmp_int("L0", "line classes", cl.get("n_lines"), F["classification"]["n_lines"])
    cmp_int("L2", "critical orbits", cl.get("n_orbits"), len(F["census"]))
    if cl.get("proj_dim_ge_2") is not None:
        got_dims = sorted(d for v in cl["proj_dim_ge_2"].values() for d in v)
        want_dims = sorted(d for v in F["classification"]["proj_dim_ge_2"].values() for d in v)
        cmp_list("L0", "dimensions of the spaces of projective dimension >= 2", got_dims, want_dims)
        cmp_int("L0", "number of groups carrying them", len(cl["proj_dim_ge_2"]),
                F["classification"]["n_such_groups"])
    else:
        record("L0", "the spaces of projective dimension >= 2", "MISSING")

    print("\n=== L1: the seven lines ===")
    for ln in F["lines"]:
        got = R.get("lines", {}).get(ln["name"])
        if got is None:
            record("L1", f"{ln['name']}: reported at all", "MISSING")
            continue
        if got.get("restriction") is not None and got.get("chart") == ln["chart"]:
            cmp_exact("L1", f"{ln['name']}: restriction", got["restriction"], ln["restriction"])
        elif got.get("restriction") is not None:
            record("L1", f"{ln['name']}: restriction", "MISSING",
                   f"room's chart is {got.get('chart')}, frozen chart is {ln['chart']}: compare by hand")
        if ln["s_star"] and got.get("critical") is not None:
            cmp_valueset("L1", f"{ln['name']}: interior critical value(s)", got["critical"], [ln["s_star"]])

    print("\n=== N1, N2: the negative controls ===")
    n1 = R.get("N1")
    if n1 is None:
        record("N1", "reported at all", "MISSING")
    else:
        cmp_exact("N1", "924*r6 at the control point", n1.get("r6_924"), F["N1"]["r6_924"])
        cmp_exact("N1", "tangential gradient norm", n1.get("grad"), F["N1"]["grad_tangential_norm"])
        for k, want in F["N1"]["residuals"].items():
            cmp_exact("N1", f"residual at {k}", (n1.get("residuals") or {}).get(k), want)
    n2 = R.get("N2")
    if n2 is None:
        record("N2", "reported at all", "MISSING")
    else:
        cmp_exact("N2", "linear coefficient", n2.get("linear_coeff"), F["N2"]["linear_coeff"])
        if n2.get("interior_critical") is not None:
            record("N2", "no interior critical point",
                   "PASS" if n2["interior_critical"] is False else "DIFFERS",
                   "" if n2["interior_critical"] is False else "room reports an interior point")

    print("\n=== H3: the degenerate orbit ===")
    h3 = R.get("H3")
    if h3 is None:
        record("H3", "reported at all", "MISSING")
    else:
        cmp_int("H3", "kernel dimension", h3.get("kernel_dim"), F["H3"]["kernel_dim"])
        if h3.get("kernel_is_C4_line_direction") is not None:
            record("H3", "kernel identified with the C4 line direction",
                   "PASS" if h3["kernel_is_C4_line_direction"] else "DIFFERS")

    print("\n=== G1, G2: the two ends, and how they were reached ===")
    g = R.get("G")
    if g is None:
        record("G1", "reported at all", "MISSING")
    else:
        record("G1", "the minimum is the coherent orbit",
               "PASS" if g.get("min_orbit") == "coherent v3" else "DIFFERS", str(g.get("min_orbit")))
        record("G2", "the maximum is the hexagon orbit",
               "PASS" if g.get("max_orbit") == "hexagon" else "DIFFERS", str(g.get("max_orbit")))
        for end, key in (("G1", "min_by"), ("G2", "max_by")):
            v = g.get(key)
            if v is None:
                record(end, f"{key}: argument or search", "MISSING")
            else:
                # a search establishes neither end; the pass condition is an argument
                record(end, f"{key}: argument rather than search",
                       "PASS" if v == "argument" else "DIFFERS", v)

    n_pass = sum(1 for r in ROWS if r["verdict"] == "PASS")
    n_diff = sum(1 for r in ROWS if r["verdict"] == "DIFFERS")
    n_miss = sum(1 for r in ROWS if r["verdict"] == "MISSING")
    print(f"\n{a.label}: {n_pass} pass, {n_diff} differ, {n_miss} missing, of {len(ROWS)}")
    out = HERE / f"compare_{a.label}{'_mut' + str(a.mutate) if a.mutate else ''}.json"
    out.write_text(json.dumps({"label": a.label, "rows": ROWS,
                               "pass": n_pass, "differs": n_diff, "missing": n_miss}, indent=1))
    if a.mutate:
        print("MUTATION RUN: at least one DIFFERS above is the expected outcome.")
        return 0 if n_diff else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
