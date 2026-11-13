"""M10.1 check C1: the rooms' derivations, compared exactly.

Three result files came out of the clean rooms (research/m10_1/returns/):
  solver/results.json           a direct symbolic series expansion
  audit_a/audit_results.json    moments of the direction set, saved before solver/ was opened
  audit_b/audit_results.json    the same route, in a second room, saved before solver/ was opened

This script parses every expression with sympy and compares the three by exact
arithmetic: the degree-2, 4 and 6 polynomials, the invariant coefficients, the
remainders, the coefficient of I6 and the ratio, the first degree with a nonzero
remainder, and the scaling under a -> a/2. It then perturbs one coefficient of
one file and requires the comparison to go red (a check that cannot fail is not
a check).

Run from the repository root:
    python3 openwave/xperiments/m10_oph/research/scripts/m10_1_compare_rooms.py
Writes research/data/m10_1_compare_rooms.json and exits 1 on any disagreement.
"""

import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
RETURNS = HERE.parent / "m10_1" / "returns"
OUT = HERE.parent / "data" / "m10_1_compare_rooms.json"
SOURCES = {
    "solver": RETURNS / "solver" / "results.json",
    "audit_a": RETURNS / "audit_a" / "audit_results.json",
    "audit_b": RETURNS / "audit_b" / "audit_results.json",
}
x, y, z, a = sp.symbols("x y z a")
NAMES = {"x": x, "y": y, "z": z, "a": a, "q_x": x, "q_y": y, "q_z": z, "sqrt": sp.sqrt}
DEGREES = ("degree_2", "degree_4", "degree_6")
FIELDS = ("polynomial", "invariant_coefficient", "remainder", "remainder_coefficient_in_I6", "ratio")


def expr(text):
    """Parse one reported value. A trailing '* |q|^n' label is a unit, not part of the value."""
    text = re.sub(r"\s*\*?\s*\|q\|\s*(\^|\*\*)\s*\d+", "", str(text))
    return sp.simplify(sp.sympify(text, locals=NAMES))


def scaling(block):
    """The three factors a -> a/2 applies at degree 2, 4, 6, read from whatever shape the room used."""
    text = json.dumps(block)
    found = {}
    for deg in DEGREES:
        m = re.search(rf'"{deg}"\s*:\s*"[^"0-9]*(\d+(?:/\d+)?)', text)
        found[deg] = str(Fraction(m.group(1))) if m else None
    return found


def load(path):
    raw = json.loads(path.read_text())
    out = {deg: {f: expr(raw[deg][f]) for f in FIELDS} for deg in DEGREES}
    out["first_nonzero_remainder_degree"] = int(raw["first_nonzero_remainder_degree"])
    out["scaling_a_to_half"] = scaling(raw["scaling_a_to_half"])
    return out


def compare(data):
    """Every pairwise difference, as a list. Empty means the three agree exactly."""
    diffs = []
    names = list(data)
    for i, p in enumerate(names):
        for q in names[i + 1 :]:
            for deg in DEGREES:
                for f in FIELDS:
                    d = sp.simplify(sp.expand(data[p][deg][f] - data[q][deg][f]))
                    if d != 0:
                        diffs.append(f"{p} vs {q}: {deg}.{f} differs by {d}")
            for key in ("first_nonzero_remainder_degree", "scaling_a_to_half"):
                if data[p][key] != data[q][key]:
                    diffs.append(f"{p} vs {q}: {key}: {data[p][key]} against {data[q][key]}")
    return diffs


data = {name: load(path) for name, path in SOURCES.items()}
diffs = compare(data)

# Internal consistency of each file, by exact arithmetic: the polynomial is its
# invariant part plus its remainder, and the ratio is the quotient it claims to be.
internal = []
qq = x**2 + y**2 + z**2
for name, d in data.items():
    for n, deg in zip((1, 2, 3), DEGREES):
        if sp.expand(d[deg]["polynomial"] - d[deg]["invariant_coefficient"] * qq**n - d[deg]["remainder"]) != 0:
            internal.append(f"{name}: {deg} polynomial is not invariant part plus remainder")
        ratio = d[deg]["remainder_coefficient_in_I6"] / d[deg]["invariant_coefficient"]
        if sp.simplify(ratio - d[deg]["ratio"]) != 0:
            internal.append(f"{name}: {deg} ratio is not the quotient of its two coefficients")

# Mutations: each must turn the comparison red.
mutations = {}
bent = {k: {**v} for k, v in data.items()}
bent["audit_a"] = {**data["audit_a"], "degree_6": {**data["audit_a"]["degree_6"]}}
bent["audit_a"]["degree_6"]["polynomial"] = data["audit_a"]["degree_6"]["polynomial"] + a**4 * x**6 / 10**6
mutations["one degree-6 coefficient moved by 1e-6"] = "caught" if compare(bent) else "MISSED"
bent = {k: {**v} for k, v in data.items()}
bent["solver"] = {**data["solver"], "degree_6": {**data["solver"]["degree_6"]}}
bent["solver"]["degree_6"]["ratio"] = data["solver"]["degree_6"]["ratio"] + sp.Rational(1, 75)
mutations["the ratio moved by 1/75"] = "caught" if compare(bent) else "MISSED"
bent = {k: {**v} for k, v in data.items()}
bent["audit_b"] = {**data["audit_b"], "first_nonzero_remainder_degree": 4}
mutations["the first nonzero remainder moved to degree 4"] = "caught" if compare(bent) else "MISSED"

ok = not diffs and not internal and all(v == "caught" for v in mutations.values())
record = {
    "sources": {name: path.relative_to(HERE.parent).as_posix() for name, path in SOURCES.items()},
    "agreed_values": {
        deg: {f: str(data["solver"][deg][f]) for f in FIELDS if f not in ("polynomial", "remainder")}
        for deg in DEGREES
    },
    "first_nonzero_remainder_degree": data["solver"]["first_nonzero_remainder_degree"],
    "scaling_a_to_half": data["solver"]["scaling_a_to_half"],
    "differences": diffs,
    "internal_consistency": internal,
    "mutations": mutations,
    "verdict": "C1 holds: the three derivations agree exactly" if ok else "C1 fails",
}
OUT.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({k: record[k] for k in ("agreed_values", "first_nonzero_remainder_degree", "scaling_a_to_half")}, indent=1))
print("differences:", diffs or "none")
print("internal consistency:", internal or "holds in all three")
print("mutations:", mutations)
print(record["verdict"])
sys.exit(0 if ok else 1)
