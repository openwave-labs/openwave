"""Run every script of this room in order; each script's printed output goes to <script>.log beside it.
Run as:  ./py run_all.py"""
import os, sys, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["s0_group.py", "s1_slots.py", "s2_constants.py", "s3_points.py", "s4_boxes.py", "planted.py", "assemble.py"]
env = dict(os.environ); env.pop("PLANT", None)
summary = []
for sc in SCRIPTS:
    t = time.time()
    r = subprocess.run([sys.executable, "-S", os.path.join(HERE, sc)], capture_output=True, text=True, env=env, cwd=HERE)
    log = os.path.join(HERE, sc[:-3] + ".log")
    open(log, "w").write(r.stdout + (("\n--- stderr ---\n" + r.stderr) if r.stderr else ""))
    npass = r.stdout.count("[PASS]"); nfail = r.stdout.count("[FAIL]")
    line = f"{sc:18s} exit {r.returncode}  PASS {npass:3d}  FAIL {nfail:3d}  {time.time()-t:7.1f}s  -> {os.path.relpath(log, HERE)}"
    print(line, flush=True); summary.append(line)
    if r.returncode != 0:
        print("stopping: script failed"); break
open(os.path.join(HERE, "run_all.log"), "w").write("\n".join(summary) + "\n")
