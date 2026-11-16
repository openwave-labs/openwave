"""Planted defects: each re-runs one script with DEFECT set (the scripts corrupt *data* at the named
place and write no output files), and shows that checks fire (FAIL lines).  The corresponding
clean runs are the logs of run_all.py, with 0 FAIL."""
import os, sys, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib_check import check, summary

PLANTS = [
    ("s00_group.py", "wrong_generator", "q2 replaced by (1+i+j-k)/2: a different group"),
    ("s01_cg_theta.py", "cg_sign", "sign of <3 3; 3 -3|6 0> flipped in the built table"),
    ("s02_eta.py", "eta_perturb", "eta_{3,1} of R4 shifted by 1e-8"),
    ("s03_levels.py", "wrong_sN_routeA", "route A uses s_N * 8/7"),
    ("s04b_c3_exact.py", "wrong_lambda0", "exact route uses lambda_0 = n(n+1)"),
    ("s05_c5.py", "T1_mu0sq_sign", "sign of the mu0^2 term of T_1 flipped in the 70-digit run only"),
]
for script, defect, what in PLANTS:
    env = dict(os.environ, DEFECT=defect)
    p = subprocess.run([sys.executable, os.path.join(HERE, script)], env=env, capture_output=True, text=True, cwd=HERE)
    fails = [l.strip() for l in (p.stdout + p.stderr).splitlines() if "FAIL:" in l]
    print(f"[{script}] DEFECT={defect} ({what}): {len(fails)} FAIL line(s)")
    for l in fails[:6]:
        print("      " + l)
    check(f"planted defect '{defect}' in {script} makes at least one check fail", len(fails) > 0)
summary()
