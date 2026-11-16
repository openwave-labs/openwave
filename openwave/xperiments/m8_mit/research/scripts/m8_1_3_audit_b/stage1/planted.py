"""Planted-defect runs: each re-runs one script with a deliberate defect switched on (environment PLANT=<tag>) and reports
whether the corresponding check printed FAIL.  The planted runs' own output goes to <script>.PLANT-<tag>.log."""
import os, subprocess, sys
from chk import check

here = os.path.dirname(os.path.abspath(__file__))
plants = [
    ('s0_anchors.py', 'sN', '0(c) R2: s_N by Schur equals exact S^3 integral', 's_N altered by 1% (and w changed)'),
    ('s1_C1C2.py', 'C1', 'C.1 R2: CG route agrees with exact route', 'a 1e-6 relative error added to the CG level norm'),
    ('s2_C3.py', 'C3', "C.3 v1 R4: exact Hess R0 on K' agrees with the CG route", 'a 1e-10 error added to the comparison'),
    ('s3_C4.py', 'C4', 'C.4 pentagonal pyramid in R4: cubic form', 'the fifth-point value doubled'),
    ('s4_C5.py', 'C5', 'C.5 coherent v3 in R4: CG-route Hessian of R0 agrees', 'a 1e-20 error added to the Hessian comparison'),
    ('s5_D_B4.py', 'Dspan', 'D: the given generators span the same 4-dim space', 'one generator replaced by a G-skew map outside the span'),
    ('s5_D_B4.py', 'Dbox', 'D O3 R4: Krawczyk K(Z) lies in the interior of Z', 'the box shifted by 1e-40 (no longer contains the zero)'),
]
for script, tag, label, what in plants:
    env = dict(os.environ); env['PLANT'] = tag
    r = subprocess.run([sys.executable, '-S', os.path.join(here, script)], env=env, capture_output=True, text=True, cwd=here)
    txt = r.stdout
    fired = any(line.startswith('[FAIL]') and label in line for line in txt.splitlines())
    print(f'planted {tag} in {script} ({what}): the check "{label}..." printed FAIL: {fired}  (exit code {r.returncode})')
    check(f'planted defect {tag} fires', fired)
