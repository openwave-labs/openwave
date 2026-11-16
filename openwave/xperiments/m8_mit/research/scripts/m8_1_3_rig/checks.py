"""PASS/FAIL helper. A script run with `--plant NAME` applies the planted defect NAME."""
import sys

PLANT = sys.argv[sys.argv.index("--plant") + 1] if "--plant" in sys.argv else None
_counts = {"PASS": 0, "FAIL": 0}


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    _counts[tag] += 1
    print(f"{tag} {name}" + (f"  [{detail}]" if detail else ""))
    return ok


def summary():
    print(f"SUMMARY PASS={_counts['PASS']} FAIL={_counts['FAIL']} plant={PLANT}")
