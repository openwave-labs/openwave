"""M10.1 packet check: nothing in the room's packet can carry a target.

Mechanical audit of the four files that enter the clean rooms
(research/m10_1/packet/), per dev_docs/CLEAN_ROOM_STANDARDS.md section 4:

  1. withheld vocabulary: no name of the framework, its author, this platform or
     the registered rows, and no word that states a sign or an outcome;
  2. numbers: every numeric token is one the definition itself needs, and no
     decimal literal appears;
  3. each check is mutation-tested: a planted violation must turn it red.

Run from the repository root:
    python3 openwave/xperiments/m10_oph/research/scripts/m10_1_packet_check.py
Writes research/data/m10_1_packet_check.json and exits 1 on any violation.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKET = HERE.parent / "m10_1" / "packet"
OUT = HERE.parent / "data" / "m10_1_packet_check.json"
FILES = ("TASK.md", "TASK_AUDIT.md", "OPENING_PROMPT.md", "OPENING_PROMPT_AUDIT.md")

# Names and outcome words the rooms must not be handed. Word-bounded, case-insensitive.
WITHHELD = [
    r"oph", r"observers?", r"patch(es)?", r"holograph\w*", r"mueller", r"pragma\w*",
    r"openwave", r"models\.md", r"m10(\.\d+)?", r"fz-?\d*", r"frozen", r"freeze",
    r"registers?", r"registered", r"predict\w*", r"authors?", r"icosa\w*", r"carriers?",
    r"ports?", r"lattice", r"dispersion", r"negative", r"sign of", r"anisotrop\w*",
    r"isotrop\w*", r"c4", r"c6", r"expected value", r"target values?",
]
# Every integer the definition, the requests and the file layout need. Nothing else may appear.
ALLOWED_NUMBERS = {"0", "1", "2", "3", "4", "5", "6", "8", "16", "32"}


def violations(text):
    found = []
    low = text.lower()
    for pat in WITHHELD:
        for m in re.finditer(rf"(?<![a-z0-9]){pat}(?![a-z0-9])", low):
            found.append(f"withheld word: {m.group(0)!r}")
    for m in re.finditer(r"\d+\.\d+", text):
        found.append(f"decimal literal: {m.group(0)}")
    for m in re.finditer(r"(?<![A-Za-z_\d.])\d+(?![\d.]*\d)", text):
        if m.group(0) not in ALLOWED_NUMBERS:
            found.append(f"number outside the declared set: {m.group(0)}")
    return found


def strip_names(text):
    """Identifiers that carry a digit by name, not a value (P6, I6, C1..C4, degree_2..)."""
    text = re.sub(r"\b(P6|I6|C[1-4])\b", "ID", text)
    return re.sub(r"\bdegree_[246]\b", "ID", text)


record = {"files": {}, "mutations": {}, "violations": {}}
ok = True
for name in FILES:
    raw = (PACKET / name).read_bytes()
    text = strip_names(raw.decode("utf-8"))
    v = violations(text)
    record["files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    record["violations"][name] = v
    ok &= not v

# Mutation tests: each planted line must be caught, on the real task file.
base = strip_names((PACKET / "TASK.md").read_text())
assert not violations(base), f"mutation baseline is not clean: {violations(base)}"
PLANTS = {
    "a framework name": "This is the OPH object.",
    "an outcome word": "The quartic coefficient is negative.",
    "a decimal value": "The ratio is close to 0.0125.",
    "an undeclared integer": "The coefficient is 7 over 480.",
    "a row id": "Compare with FZ-11 afterwards.",
}
for label, line in PLANTS.items():
    caught = bool(violations(base + "\n" + line + "\n"))
    record["mutations"][label] = "caught" if caught else "MISSED"
    ok &= caught

record["verdict"] = "clean" if ok else "violations"
OUT.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
for name, meta in record["files"].items():
    print(f"{meta['sha256']}  {meta['bytes']:5d}  packet/{name}  violations: {len(record['violations'][name])}")
for name, v in record["violations"].items():
    for item in v:
        print(f"  {name}: {item}")
print("mutations:", record["mutations"])
print(record["verdict"])
sys.exit(0 if ok else 1)
