"""Runs every script in order with this room's interpreter (run as: ./py run_all.py), then each script's
planted-defect mode ('plant', logs *_plant.log, which must show FAIL lines), then the collector."""
import os, sys, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ['s01_group.py', 's02_anchors.py', 's03_A1.py', 's04_A2_A3.py', 's05_A4_B.py', 's06_B5.py', 's07_E.py']
log = open(os.path.join(HERE, 'run_all.log'), 'w')


def run(args):
    t = time.time()
    p = subprocess.run([sys.executable, '-S'] + args, cwd=HERE, capture_output=True, text=True)
    out = p.stdout
    fails = sum(1 for l in out.splitlines() if l.startswith('FAIL'))
    passes = sum(1 for l in out.splitlines() if l.startswith('PASS'))
    msg = f'{" ".join(args):28s} rc={p.returncode} PASS={passes} FAIL={fails} ({time.time() - t:.0f}s)'
    if p.returncode != 0:
        msg += '\n' + p.stderr[-2000:]
    print(msg); log.write(msg + '\n'); log.flush()
    return p.returncode, fails


ok = True
for s in SCRIPTS:
    rc, f = run([s])
    ok &= (rc == 0 and f == 0)
log.write('--- planted defects (each must report FAIL >= 1) ---\n'); print('--- planted defects ---')
for s in SCRIPTS:
    rc, f = run([s, 'plant'])
    ok &= f >= 1
rc, f = run(['s08_collect.py'])
ok &= rc == 0
msg = 'ALL OK' if ok else 'SOMETHING WRONG'
print(msg); log.write(msg + '\n'); log.close()
