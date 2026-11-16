"""Runs every script of this room in order through this interpreter (./py), keeping each script's
printed output in <script>.log beside it, then assembles results.json.  Usage: ./py run_all.py"""
import os, sys, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["s00_group.py", "s01_cg_theta.py", "s02_eta.py", "s02b_eta_exact.py", "s03_levels.py",
           "s04_c3.py", "s04b_c3_exact.py", "s05_c5.py", "s99_planted.py", "make_results.py"]
env = dict(os.environ)
env.pop("DEFECT", None)
total = {"PASS": 0, "FAIL": 0}
for s in SCRIPTS:
    t = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, s)], capture_output=True, text=True, cwd=HERE, env=env)
    log = p.stdout + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
    open(os.path.join(HERE, s.replace(".py", ".log")), "w").write(log)
    # a script's own checks print with exactly two leading spaces; s99 echoes the planted runs' FAIL
    # lines with six, and those are not counted
    np_ = sum(1 for l in log.splitlines() if l.startswith("  PASS:"))
    nf = sum(1 for l in log.splitlines() if l.startswith("  FAIL:"))
    total["PASS"] += np_
    total["FAIL"] += nf
    print(f"{s:22s} exit {p.returncode}  {np_:4d} PASS {nf:3d} FAIL  {time.time() - t:7.1f} s  -> {s.replace('.py', '.log')}")
print(f"all scripts: {total['PASS']} PASS, {total['FAIL']} FAIL "
      f"(the FAIL lines printed inside s99_planted.log come from the planted-defect subprocesses and are not counted here)")
