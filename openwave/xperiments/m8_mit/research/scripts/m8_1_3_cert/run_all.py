"""Runs every script of this room in order through ./py, writing each script's printed output
to <script>.log beside it.  Paths are relative to this file's directory."""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "s0a_group.py",        # 0(a)
    "s0b_levels.py",       # 0(b)
    "s0c_intertwiner.py",  # 0(c) for R4, R5
    "s0d_cg_theta.py",     # 0(d)
    "s1_quartic.py",       # the reduced quartic h for R4, R5 (input to Block D)
    "sD_certify.py",       # Block D, six boxes
    "sB4.py",              # B.4
    "planted_defects.py",  # every planted defect must fire
    "make_results.py",     # results.json
]


def run_inprocess(s):
    """The room's sandbox does not allow ./py to spawn ./py, so each script is executed in this
    same ./py process as __main__ (runpy), with its printed output captured to its log."""
    import runpy, io, contextlib, traceback
    buf = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buf):
        try:
            runpy.run_path(os.path.join(HERE, s), run_name="__main__")
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        except Exception:
            traceback.print_exc(file=buf)
            code = 1
    # reset the shared PASS/FAIL ledger between scripts
    import lib_check
    lib_check.FAILS.clear()
    return code, buf.getvalue()


def main():
    sys.path.insert(0, HERE)
    status = []
    for s in SCRIPTS:
        t0 = time.time()
        code, text = run_inprocess(s)
        with open(os.path.join(HERE, s[:-3] + ".log"), "w") as f:
            f.write(text)
        dt = time.time() - t0
        status.append((s, code, dt))
        print(f"{s}: exit {code} ({dt:.1f} s) -> {s[:-3]}.log")
        if code != 0:
            print("stopping: a script failed")
            break
    sys.exit(0 if all(r == 0 for _, r, _ in status) and len(status) == len(SCRIPTS) else 1)


if __name__ == "__main__":
    main()
