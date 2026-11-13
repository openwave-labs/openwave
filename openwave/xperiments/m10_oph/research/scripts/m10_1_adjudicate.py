"""M10.1 check C5: the rooms' result against the author's frozen values.

The candidate is what the clean rooms returned and commit 2 fixed
(research/data/m10_1_compare_rooms.json, pushed before the author's values were
opened). The target is the maintainers' own transcription of row FZ-11 of the
author's frozen prediction register, read at the pinned commit after commit 2:

    omega^2 = k^2 + C4 k^4 + B0 k^6 + B6 k^6 I6(n) + O(k^8)

    C4 = -a^2/20        B0 = a^4/840        B6 = 2 a^4/7875
    B6/C4^2 = 32/315    B0/C4^2 = 10/21     B6/B0 = 16/75
    anisotropic angular ranks one to five vanish; rank six has the I6 shape

Every comparison is exact (sympy rationals). The comparison is then run with one
transcribed target value perturbed, and again with one candidate value perturbed;
both must go red.

Run from the repository root:
    python3 openwave/xperiments/m10_oph/research/scripts/m10_1_adjudicate.py
Writes research/data/m10_1_adjudication.json and exits 1 on any disagreement.
"""

import json
import sys
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent / "data" / "m10_1_compare_rooms.json"
OUT = HERE.parent / "data" / "m10_1_adjudication.json"
a = sp.Symbol("a")

# The target, transcribed by the maintainers from the register row (not copied
# from any earlier reconstruction). Source: the author's repository at commit
# 3c4e1e23fc32dd241e8c2bfba75de123fe5c6984, claims/frozen_prediction_register.json,
# row FZ-11, whose content hash is 8ac97d7c46199717ed031610efdda65c40f6a251e78715d6bc05888d598e66d8.
TARGET = {
    "C4": "-a**2/20",
    "B0": "a**4/840",
    "B6": "2*a**4/7875",
    "B6/C4^2": "32/315",
    "B0/C4^2": "10/21",
    "B6/B0": "16/75",
    "lowest anisotropic rank": "6",
}


def candidate_values(record):
    """The same seven quantities, built only from what commit 2 fixed."""
    v = record["agreed_values"]
    c4 = sp.sympify(v["degree_4"]["invariant_coefficient"], locals={"a": a})
    b0 = sp.sympify(v["degree_6"]["invariant_coefficient"], locals={"a": a})
    b6 = sp.sympify(v["degree_6"]["remainder_coefficient_in_I6"], locals={"a": a})
    lower_remainders_vanish = all(
        sp.sympify(v[d]["remainder_coefficient_in_I6"], locals={"a": a}) == 0 for d in ("degree_2", "degree_4")
    )
    rank = record["first_nonzero_remainder_degree"] if lower_remainders_vanish else None
    return {
        "C4": c4,
        "B0": b0,
        "B6": b6,
        "B6/C4^2": sp.simplify(b6 / c4**2),
        "B0/C4^2": sp.simplify(b0 / c4**2),
        "B6/B0": sp.simplify(b6 / b0),
        "lowest anisotropic rank": sp.Integer(rank) if rank is not None else sp.nan,
    }


def compare(target, cand):
    rows = []
    for key, text in target.items():
        t = sp.sympify(text, locals={"a": a})
        c = cand[key]
        agree = sp.simplify(t - c) == 0
        rows.append({"quantity": key, "author": str(t), "rooms": str(c), "verdict": "agree" if agree else "DISAGREE"})
    return rows


record = json.loads(CANDIDATE.read_text())
assert record["verdict"].startswith("C1 holds"), "the candidate is not a settled C1"
cand = candidate_values(record)
rows = compare(TARGET, cand)
agree = all(r["verdict"] == "agree" for r in rows)

# The quadratic term: the register normalizes it to k^2, and the rooms found the coefficient 1.
quadratic = sp.sympify(record["agreed_values"]["degree_2"]["invariant_coefficient"]) == 1

# Mutations, one on each side. Each must turn at least one row red.
bent_target = dict(TARGET, B6="2*a**4/7876")
bent_cand = dict(cand, C4=cand["C4"] * sp.Rational(21, 20))
mutations = {
    "target B6 transcribed as 2a^4/7876": "caught"
    if any(r["verdict"] != "agree" for r in compare(bent_target, cand))
    else "MISSED",
    "candidate C4 scaled by 21/20": "caught"
    if any(r["verdict"] != "agree" for r in compare(TARGET, bent_cand))
    else "MISSED",
}

ok = agree and quadratic and all(m == "caught" for m in mutations.values())
out = {
    "candidate": CANDIDATE.relative_to(HERE.parent).as_posix(),
    "target": "row FZ-11 of the author's frozen prediction register at the pinned commit, transcribed after commit 2",
    "rows": rows,
    "quadratic coefficient is 1": quadratic,
    "mutations": mutations,
    "verdict": "C5: every value agrees exactly" if ok else "C5: disagreement",
}
OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
for r in rows:
    print(f"{r['quantity']:26s} author {r['author']:14s} rooms {r['rooms']:14s} {r['verdict']}")
print("quadratic coefficient is 1:", quadratic)
print("mutations:", mutations)
print(out["verdict"])
sys.exit(0 if ok else 1)
