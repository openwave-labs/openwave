"""PASS/FAIL bookkeeping. A check prints PASS only when its condition is True; the condition
is always a computed comparison, never a constant. planted_defects.py feeds defective inputs
through the same predicates to show that they can print FAIL."""
import sys

FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"{tag}: {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)
    return bool(cond)


def finish():
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED: {FAILS}")
        sys.exit(1)
    print("all checks passed")
