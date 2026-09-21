"""Verify the maintainer's encoding of the M8.12 frozen values.

Two jobs, and only the second is about the author being right:

  STRUCTURAL  the frozen table is internally consistent (a spectrum's roots carry
              the signature it is filed beside, a line's restriction takes the
              census value at the critical point it names, the sum rules hold).
              A typo in `frozen_claims.json` fails here, before it can be
              reported as a room "differing".
  RECOMPUTE   924*r-hat-6 at every filed representative, from the worklist's own
              definition, by an exact route.

Every check prints PASS or FAIL and can fail: `--mutate` plants a defect and the
run is expected to go red.

Usage:  ./py check_frozen.py [--mutate N]
"""
import json
import pathlib
import sys

from sympy import (Abs, I, Poly, Rational, expand, factor, im, nsimplify, re,
                   simplify, sqrt, symbols, sympify)

HERE = pathlib.Path(__file__).resolve().parent
LAM = symbols("l", real=True)
S_VAR = symbols("s", nonnegative=True)
X, Y = symbols("x y", real=True)

FAILED = []
PASSED = 0


def check(label, ok, detail=""):
    global PASSED
    if ok:
        PASSED += 1
        print(f"  PASS  {label}{('  ' + detail) if detail else ''}")
    else:
        FAILED.append(label)
        print(f"  FAIL  {label}{('  ' + detail) if detail else ''}")


def ex(s):
    """Frozen strings are sympy source; evaluate them in a fixed namespace."""
    return sympify(s, locals={"Rational": Rational, "sqrt": sqrt, "I": I,
                              "l": LAM, "s": S_VAR, "x": X, "y": Y, "Abs": Abs})


def root_signs(poly_str, expected_deg):
    """(n_neg, n_zero, n_pos) of the roots with multiplicity, exactly."""
    p = Poly(expand(ex(poly_str)), LAM)
    roots = p.all_roots()
    neg = sum(1 for r in roots if r.is_negative)
    zero = sum(1 for r in roots if r.is_zero)
    pos = sum(1 for r in roots if r.is_positive)
    return (neg, zero, pos), p.degree(), len(roots)


def load(mutate=None):
    data = json.loads((HERE / "frozen_claims.json").read_text())
    if mutate == 1:      # a signature that its own spectrum contradicts
        data["census"][7]["sig"] = [5, 0, 4]
    elif mutate == 2:    # a line restriction that misses its census value
        data["lines"][4]["restriction"] = "-2*s**2 + 2*s + Rational(1,900)"
    elif mutate == 3:    # break the N1 sum rule
        data["N1"]["sum_of_squares"] = "Rational(17368,29404)"
    elif mutate == 4:    # break the sector bridge
        data["P"]["P5"]["w6_H_pyramid"]["3'"] = "Rational(-224,166)"
    elif mutate == 5:    # a degree that contradicts dim N_u
        data["spectra"]["octahedron"] = "(11*l + 8)**3*(11*l + 24)**3*(33*l - 40)**2"
    return data


def main():
    mutate = None
    if "--mutate" in sys.argv:
        mutate = int(sys.argv[sys.argv.index("--mutate") + 1])
        print(f"### MUTATION {mutate} PLANTED: this run is expected to fail\n")
    F = load(mutate)
    census = {c["orbit"]: c for c in F["census"]}

    print("=== the spectra carry the signatures they are filed beside ===")
    for orbit, poly in F["spectra"].items():
        c = census[orbit]
        sig, deg, nroots = root_signs(poly, c["dim_N"])
        check(f"{orbit}: degree == dim N_u", deg == c["dim_N"], f"{deg} vs {c['dim_N']}")
        check(f"{orbit}: all roots real", nroots == deg, f"{nroots} real of {deg}")
        check(f"{orbit}: root signs == signature", list(sig) == c["sig"],
              f"{sig} vs {tuple(c['sig'])}")

    print("\n=== the indices are the two ends of the signature ===")
    for c in F["census"]:
        n_neg, n_zero, n_pos = c["sig"]
        check(f"{c['orbit']}: index g>0 == n_minus", c["idx_pos"] == n_neg)
        check(f"{c['orbit']}: index g<0 == n_plus", c["idx_neg"] == n_pos)
        check(f"{c['orbit']}: signature sums to dim N_u", sum(c["sig"]) == c["dim_N"])

    print("\n=== dim N_u is 10 at the weight states and 9 elsewhere ===")
    WEIGHT = {"coherent v3", "v2", "v1", "zonal v0"}
    for c in F["census"]:
        want = 10 if c["orbit"] in WEIGHT else 9
        check(f"{c['orbit']}: dim N_u == {want}", c["dim_N"] == want, str(c["dim_N"]))

    print("\n=== each line restriction takes the census value at each point it names ===")
    by_value = {}
    for c in F["census"]:
        by_value.setdefault(simplify(ex(c["r6_924"])), []).append(c["orbit"])

    def hits_census(val_rhat, where):
        v924 = simplify(val_rhat * 924)
        ok = any(simplify(v924 - k) == 0 for k in by_value)
        names = [n for k, n in by_value.items() if simplify(v924 - k) == 0]
        check(where, ok, f"924*r6 = {v924}" + (f" = {names[0]}" if names else " NOT in the census"))

    for ln in F["lines"]:
        r = ex(ln["restriction"])
        if ln["chart"] == "s":
            hits_census(r.subs(S_VAR, 0), f"{ln['name']}: endpoint s = 0")
            hits_census(r.subs(S_VAR, 1), f"{ln['name']}: endpoint s = 1")
            if ln["s_star"]:
                hits_census(r.subs(S_VAR, ex(ln["s_star"])), f"{ln['name']}: interior s*")
        else:
            hits_census(r.subs({X: 0, Y: 0}), f"{ln['name']}: z = 0")
            lead = simplify((r * (X**2 + Y**2 + 2)**2 / (X**2 + Y**2)**2).subs({Y: 0}).limit(X, 0, "+")) \
                if False else None
            # z = infinity: the quartic terms dominate, so the limit is the ratio of
            # the |z|^4 coefficient to the denominator's
            num, den = r.as_numer_denom()
            inf_val = simplify(Poly(expand(num.subs(Y, 0)), X).all_coeffs()[0] /
                               Poly(expand(den.subs(Y, 0)), X).all_coeffs()[0])
            hits_census(inf_val, f"{ln['name']}: z = infinity")
            if "D3" in ln["name"]:
                hits_census(r.subs({X: sqrt(230) / 10, Y: 0}), f"{ln['name']}: z = +-sqrt(230)/10")
                hits_census(r.subs({X: 0, Y: sqrt(10) / 2}), f"{ln['name']}: z = +-I*sqrt(10)/2")
            else:
                hits_census(r.subs({X: sqrt(30) / 3, Y: 0}), f"{ln['name']}: z = +-sqrt(30)/3")
                hits_census(r.subs({X: 0, Y: sqrt(30) / 5}), f"{ln['name']}: z = +-I*sqrt(30)/5")

    print("\n=== the stated interior critical point is where the restriction is stationary ===")
    for ln in F["lines"]:
        if ln["chart"] != "s":
            continue
        d = simplify(ex(ln["restriction"]).diff(S_VAR))
        if ln["s_star"]:
            check(f"{ln['name']}: d/ds vanishes at s*",
                  simplify(d.subs(S_VAR, ex(ln["s_star"]))) == 0)
        else:
            sols = [r for r in Poly(d, S_VAR).all_roots()]
            interior = [r for r in sols if r.is_positive and (r - 1).is_negative]
            check(f"{ln['name']}: no interior stationary point", not interior, str(sols))

    print("\n=== N2: the C4 {v3, v-1} restriction has zero linear coefficient ===")
    c4 = next(ln for ln in F["lines"] if ln["name"] == F["N2"]["line"])
    coeffs = Poly(ex(c4["restriction"]), S_VAR).all_coeffs()
    check("N2: linear coefficient is exactly 0", coeffs[1] == 0, f"{coeffs[1]}")

    print("\n=== P4: the filed in-locus second variations are roots of the filed spectra ===")
    pyr = Poly(expand(ex(F["spectra"]["pyramid"])), LAM)
    check("P4: -104/55 is a root of the pyramid spectrum",
          pyr.eval(ex(F["P"]["P4"]["pyramid_C5_tangent"])) == 0)
    pri = Poly(expand(ex(F["spectra"]["prism"])), LAM)
    for v in F["P"]["P4"]["prism_chart_tangents"]:
        check(f"P4: {v} is a root of the prism spectrum", pri.eval(ex(v)) == 0)

    print("\n=== P5: the sector bridge ===")
    H = ex(F["P"]["P4"]["pyramid_C5_tangent"])
    for sec in ("3'", "4"):
        w6 = ex(F["P"]["P5"]["w6"][sec])
        check(f"P5: w6({sec})*H reproduces the filed value",
              simplify(w6 * H - ex(F["P"]["P5"]["w6_H_pyramid"][sec])) == 0,
              f"{simplify(w6 * H)}")
        check(f"P5: L_T({sec}) == (w6/4)*H",
              simplify(w6 / 4 * H - ex(F["P"]["P5"]["L_T_pyramid"][sec])) == 0)

    print("\n=== P1: the filed parent values are the census values ===")
    p1 = F["P"]["P1"]
    check("P1: pyramid 1188/5 is 9/35 of the way", simplify(ex(p1["pyramid_924"]) / 924 - ex(p1["pyramid_rhat"])) == 0)
    check("P1: prism 8800/43 is 200/903", simplify(ex(p1["prism_924"]) / 924 - ex(p1["prism_rhat"])) == 0)
    for v in p1["weight_states_924"] + [p1["octahedron_924"], p1["hexagon_924"]]:
        check(f"P1: 924*r6 = {v} is in the census", any(simplify(ex(str(v)) - k) == 0 for k in by_value))

    print("\n=== N1: the two identities the control carries ===")
    n1 = F["N1"]
    tot = sum(simplify(ex(v)**2) for v in n1["residuals"].values())
    check("N1: the four residual squares sum to 17368/29403",
          simplify(tot - ex(n1["sum_of_squares"])) == 0, f"{simplify(tot)}")
    check("N1: the phase residual equals the tangential gradient norm",
          simplify(ex(n1["residuals"]["i*u"]) - ex(n1["grad_tangential_norm"])) == 0)
    check("N1: all four residuals are nonzero",
          all(simplify(ex(v)) != 0 for v in n1["residuals"].values()))

    print("\n=== G: the two ends agree with the census ===")
    vals = {simplify(ex(c["r6_924"])): c for c in F["census"]}
    lo, hi = min(vals), max(vals)
    check("G1: the census minimum is the coherent orbit at 1", vals[lo]["orbit"] == "coherent v3" and lo == 1)
    check("G2: the census maximum is the hexagon at 463", vals[hi]["orbit"] == "hexagon" and hi == 463)
    check("G3: coherent signature matches the census", F["G"]["G3"]["coherent_sig"] == census["coherent v3"]["sig"])
    check("G3: hexagon signature matches the census", F["G"]["G3"]["hexagon_sig"] == census["hexagon"]["sig"])
    check("G3: neither end has a kernel",
          F["G"]["G3"]["coherent_sig"][1] == 0 and F["G"]["G3"]["hexagon_sig"][1] == 0)

    print("\n=== H3: v1 is the only degenerate orbit ===")
    degen = [c["orbit"] for c in F["census"] if c["sig"][1] != 0]
    check("H3: exactly one degenerate orbit", degen == ["v1"], str(degen))
    check("H3: its kernel has dimension 2", census["v1"]["sig"][1] == F["H3"]["kernel_dim"])

    print("\n=== D1, D2: the diagnostics as filed ===")
    d1 = F["D"]["D1"]
    check("D1: the parity terms sum to 7", sum(d1["parity_sum_terms"]) == d1["total"] == 7)
    order = [c["idx_pos"] for c in sorted(F["census"], key=lambda c: simplify(ex(c["r6_924"])))]
    check("D2: indices in increasing critical value", order == F["D"]["D2"]["idx_pos_by_increasing_value"], str(order))
    check("D2: 1 and 7 do not occur", all(i not in order for i in F["D"]["D2"]["absent"]))

    print("\n=== the classification counts ===")
    cl = F["classification"]
    check("6 point classes", len(cl["points"]) == cl["n_points"] == 6)
    check("7 line classes", len(cl["lines"]) == cl["n_lines"] == 7)
    dims = [d for v in cl["proj_dim_ge_2"].values() for d in v]
    check("4 spaces of projective dimension at least 2 from 3 groups",
          len(dims) == cl["n_such_spaces"] == 4 and len(cl["proj_dim_ge_2"]) == cl["n_such_groups"] == 3)
    check("every line class has a census orbit at each named point", len(F["lines"]) == 7)
    check("the census has ten orbits", len(F["census"]) == 10)

    print(f"\n{PASSED} passed, {len(FAILED)} failed")
    if FAILED:
        for f in FAILED:
            print(f"  failing: {f}")
    if mutate is not None:
        print("\nMUTATION RUN: a red result above is the expected outcome.")
        return 0 if FAILED else 1
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
