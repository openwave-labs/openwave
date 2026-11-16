"""Runs every script of this room in order, inside the ./py interpreter that runs this file
(./py run_all.py), keeping each script's printed output in <script>.log beside it.
The room's sandbox forbids a script from exec'ing ./py again, so each script is run in-process
with runpy (fresh module namespace, its own sys.argv).
Usage: ./py run_all.py [script ...]   (default: all, in order)"""
import contextlib, io, os, runpy, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ORDER = ["s0_group.py", "s0_slot.py", "sA1.py", "sA5.py", "sA4_quad.py", "sB3.py", "make_results.py", "plant_defects.py"]
todo = sys.argv[1:] or ORDER
for scr in todo:
    t = time.time()
    buf = io.StringIO()
    status = "ok"
    old_argv = sys.argv
    sys.argv = [os.path.join(HERE, scr)]
    with contextlib.redirect_stdout(buf):
        try:
            runpy.run_path(os.path.join(HERE, scr), run_name="__main__")
        except SystemExit as e:
            status = f"exit {e.code}"
        except Exception:
            status = "EXCEPTION"
            print(traceback.format_exc())
    sys.argv = old_argv
    with open(os.path.join(HERE, scr[:-3] + ".log"), "w") as f:
        f.write(buf.getvalue())
    summ = [l for l in buf.getvalue().splitlines() if l.startswith("checks:")]
    print(f"{scr}: {status}, {time.time() - t:.0f}s, {summ[-1] if summ else ''}", flush=True)
