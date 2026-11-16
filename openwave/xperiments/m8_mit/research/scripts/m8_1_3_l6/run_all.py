"""Run every script in order, in-process (no subprocess), writing each script's printed output
to a .log file beside it.  Usage (from the room's top level):  ./py run_all.py"""
import os, sys, runpy, io, contextlib, time
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["s01_group.py", "s02_intertwiner.py", "s03_blockA.py", "s04_blockB.py", "s05_crosscheck.py", "s06_results.py"]
sys.path.insert(0, os.path.join(HERE, "scripts"))
summary = []
for s in SCRIPTS:
    path = os.path.join(HERE, "scripts", s)
    log = os.path.join(HERE, "scripts", s.replace(".py", ".log"))
    buf = io.StringIO(); t0 = time.time()
    with contextlib.redirect_stdout(buf):
        try:
            runpy.run_path(path, run_name="__main__"); status = "ok"
        except Exception as e:
            import traceback; traceback.print_exc(file=buf); status = "ERROR " + repr(e)
    out = buf.getvalue()
    open(log, "w").write(out)
    npass = out.count("\nPASS ") + out.startswith("PASS ")
    nfail = out.count("\nFAIL ") + out.startswith("FAIL ")
    nfired = out.count("PLANTED-DEFECT-FIRED"); nmissed = out.count("PLANTED-DEFECT-MISSED")
    line = f"{s}: {status}, {time.time() - t0:.0f}s, PASS {npass}, FAIL {nfail}, planted defects fired {nfired}, missed {nmissed}"
    print(line); summary.append(line)
open(os.path.join(HERE, "scripts", "run_all.log"), "w").write("\n".join(summary) + "\n")
