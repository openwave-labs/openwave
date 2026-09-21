"""Run every script of this room in order, then run planted defects that must make checks fail.

Invoke as:  ./py run_all.py
Scripts are executed in this same interpreter process with runpy (the room's ./py wrapper cannot
be nested inside itself).  Output of each script is shown and PASS/FAIL lines are counted.
"""
import os, sys, io, runpy, importlib, contextlib, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
SCRIPTS = ["item0.py", "item1.py", "item2.py", "item3.py", "item4.py", "item5.py", "item5b.py",
           "item6_10.py", "item7.py", "item8.py", "item9.py", "item11.py", "item12.py", "collect.py"]
QUICK = "--quick" in sys.argv          # skip item12 when asked


class Tee(io.TextIOBase):
    def __init__(self, *s):
        self.s = s

    def write(self, t):
        for x in self.s:
            x.write(t)
        return len(t)

    def flush(self):
        for x in self.s:
            x.flush()


def run(script, echo=True):
    buf = io.StringIO()
    target = Tee(sys.__stdout__, buf) if echo else buf
    t0 = time.time()
    with contextlib.redirect_stdout(target):
        try:
            runpy.run_path(os.path.join(HERE, script), run_name="__main__")
            err = None
        except Exception as e:          # a crash counts as a failure
            err = repr(e)
    txt = buf.getvalue()
    return txt.count("\nPASS ") + txt.startswith("PASS "), txt.count("\nFAIL ") + txt.startswith("FAIL "), err, time.time() - t0


def reload_all():
    import core, hess, rot
    importlib.reload(core)
    importlib.reload(rot)
    importlib.reload(hess)


summary = []
for sc in SCRIPTS:
    if QUICK and sc == "item12.py":
        continue
    print("\n" + "=" * 20, sc, "=" * 20, flush=True)
    npass, nfail, err, dt = run(sc)
    summary.append((sc, npass, nfail, err, dt))

# planted defects: each must produce at least one FAIL (or crash) in the named script
PLANTS = [("cg_sign", "item0.py", "flip the sign of every CG with m1 < 0"),
          ("theta_sign", "item9.py", "flip the sign of the v1 component of Theta u"),
          ("hess_no4N", "item5.py", "drop the -4 N(u) I term of the Hessian formula")]
planted = []
for flag, sc, what in PLANTS:
    os.environ["PLANT"] = flag
    reload_all()
    npass, nfail, err, dt = run(sc, echo=False)
    fired = nfail > 0 or err is not None
    planted.append((flag, sc, what, npass, nfail, err, fired))
os.environ.pop("PLANT", None)
reload_all()

print("\n" + "=" * 60)
print("SUMMARY")
allok = True
for sc, npass, nfail, err, dt in summary:
    print("  %-12s PASS %3d  FAIL %3d  %s  (%.0f s)" % (sc, npass, nfail, "CRASH " + err if err else "", dt))
    allok &= nfail == 0 and err is None
print("PLANTED DEFECTS (each must fire):")
for flag, sc, what, npass, nfail, err, fired in planted:
    print("  PLANT=%-11s in %-9s (%s): PASS %d FAIL %d %s -> %s" % (flag, sc, what, npass, nfail,
                                                                   "crash" if err else "", "FIRED" if fired else "DID NOT FIRE"))
    allok &= fired
print("RUN_ALL", "OK" if allok else "PROBLEMS")
