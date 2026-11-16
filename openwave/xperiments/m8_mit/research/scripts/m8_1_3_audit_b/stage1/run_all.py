"""Run every script in order through this interpreter; each script keeps its own printed output in <script>.log beside it."""
import os, subprocess, sys, time
import chk  # noqa: tees this script's own output into run_all.log

here = os.path.dirname(os.path.abspath(__file__))
scripts = ['s0_anchors.py', 's1_C1C2.py', 's2_C3.py', 's3_C4.py', 's4_C5.py', 's5_D_B4.py', 'planted.py', 'assemble.py']
env = dict(os.environ); env.pop('PLANT', None)
tot_fail = 0
for s in scripts:
    t = time.time()
    r = subprocess.run([sys.executable, '-S', os.path.join(here, s)], env=env, capture_output=True, text=True, cwd=here)
    npass = r.stdout.count('[PASS]'); nfail = r.stdout.count('[FAIL]')
    tot_fail += nfail
    print(f'{s}: exit {r.returncode}, {npass} PASS, {nfail} FAIL, {time.time() - t:.0f}s  (output in {s[:-3]}.log)')
    if r.returncode != 0:
        print(r.stderr[-2000:])
print(f'total FAIL lines: {tot_fail}')
