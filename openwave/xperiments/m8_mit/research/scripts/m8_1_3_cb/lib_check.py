"""PASS/FAIL helper.  Every check is a boolean computed from data and prints FAIL
when false.  Planted defects (s99_planted.py) corrupt *data* through the
environment variable DEFECT, read by the scripts at the named places; the checks
themselves are never altered."""
import os

_counts = {"PASS": 0, "FAIL": 0}
DEFECT = os.environ.get("DEFECT", "")


def check(name, cond):
    cond = bool(cond)
    tag = "PASS" if cond else "FAIL"
    _counts[tag] += 1
    print(f"  {tag}: {name}")
    return cond


def summary():
    print(f"checks: {_counts['PASS']} PASS, {_counts['FAIL']} FAIL")
