"""Run every script in order (run this file as `./py run_all.py`). Each script's output goes to <script>.log
beside it; each planted-defect run goes to <script>.plant_<name>.log. A plant run counts as 'fired' if it
prints at least one FAIL line; a clean run must print none. Finally s99_collect.py writes results.json."""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    ("s01_group.py", ["q2norm", "char", "mult", "trace", "names"]),
    ("s02_rigid_slots.py", ["restrict", "next"]),
    ("s03_rigid_lin.py", ["cubic", "frozen", "omega"]),
    ("s04_rigid_group.py", ["nonunitary", "inE0"]),
]


def run(script, plant=None):
    args = [sys.executable, "-S", os.path.join(HERE, script)] + (["--plant", plant] if plant else [])
    p = subprocess.run(args, cwd=HERE, capture_output=True, text=True)
    log = script[:-3] + (f".plant_{plant}.log" if plant else ".log")
    with open(os.path.join(HERE, log), "w") as f:
        f.write(p.stdout)
        if p.stderr:
            f.write("\n--- stderr ---\n" + p.stderr)
    nfail = sum(1 for l in p.stdout.splitlines() if l.startswith("FAIL"))
    npass = sum(1 for l in p.stdout.splitlines() if l.startswith("PASS"))
    return p.returncode, npass, nfail, log


ok = True
for script, plants in SCRIPTS:
    rc, npass, nfail, log = run(script)
    good = rc == 0 and nfail == 0
    ok &= good
    print(f"{'OK  ' if good else 'BAD '} clean  {script}: PASS={npass} FAIL={nfail} rc={rc} -> {log}")
for script, plants in SCRIPTS:
    for pl in plants:
        rc, npass, nfail, log = run(script, pl)
        fired = nfail > 0
        ok &= fired
        print(f"{'OK  ' if fired else 'BAD '} plant {pl:>10} {script}: FAIL lines={nfail} rc={rc} -> {log}")
rc, npass, nfail, log = run("s99_collect.py")
print(f"collect: rc={rc} -> {log}")
print("ALL OK" if ok and rc == 0 else "SOMETHING WRONG")
