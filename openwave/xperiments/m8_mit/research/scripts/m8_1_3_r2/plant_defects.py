"""Planted defects: re-run each script in-process with --plant=NAME and confirm that checks FAIL.
Planted runs never write the out_*.json files (each script guards its output with PLANT is None).
Usage: ./py plant_defects.py"""
import contextlib, io, os, runpy, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PLANTS = [
    ("s0_group.py", "q2norm", "q2 perturbed off the unit sphere"),
    ("s0_group.py", "extrachar", "a duplicated character in the derived table"),
    ("s0_slot.py", "eta", "second column of eta replaced by e_{-7/2}"),
    ("s0_slot.py", "C", "C replaced by the identity"),
    ("s0_slot.py", "s", "s set to 1/3 instead of dim/(2j+1)"),
    ("s0_slot.py", "cg", "one CG coefficient sign flipped"),
    ("sA1.py", "qJ", "q_3 shifted by 1e-6"),
    ("sA1.py", "hsval", "c_5 of the HS factor shifted by 1e-4"),
    ("sA1.py", "beta", "beta shifted by 1e-3"),
    ("sA5.py", "coef", "A.5 formula given an extra |alpha|^2|beta|^2/1000"),
    ("sA4_quad.py", "point", "quadrature point perturbed by 1e-3"),
    ("sB3.py", "thetasign", "Theta without its (-1)^(j-m) signs"),
    ("sB3.py", "kvec", "one tangent vector of K_u corrupted by 1e-6"),
    ("sB3.py", "nu", "cluster nu shifted by 1e-20 before the kernel"),
    ("sB3.py", "charpoly", "exact matrix perturbed by 1e-6 in one entry"),
    ("sB3.py", "numeric", "50-digit Mhat perturbed by 1e-10 (symmetric)"),
    ("sB3.py", "sturm", "positive-root count off by one"),
]
allfired = True
for scr, plant, what in PLANTS:
    buf = io.StringIO()
    old = sys.argv
    sys.argv = [os.path.join(HERE, scr), f"--plant={plant}"]
    with contextlib.redirect_stdout(buf):
        try:
            runpy.run_path(os.path.join(HERE, scr), run_name="__main__")
        except SystemExit:
            pass
        except Exception as e:
            print(f"EXCEPTION {type(e).__name__}: {e}")
    sys.argv = old
    lines = buf.getvalue().splitlines()
    fails = [l for l in lines if l.startswith("FAIL")]
    exc = [l for l in lines if l.startswith("EXCEPTION")]
    fired = bool(fails) or bool(exc)
    allfired &= fired
    print(f"{scr} --plant={plant} ({what}): {'FIRED' if fired else 'DID NOT FIRE'}; {len(fails)} FAIL line(s)", flush=True)
    for l in (fails[:4] + exc[:1]):
        print("    " + l[:150])
print("every planted defect fired:", allfired)
