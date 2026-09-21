"""Run every script of this room in order, then collect results.json.
Usage (from the room):  ./py run_all.py
The scripts are executed in-process with runpy (a nested ./py call is refused by the room's sandbox:
'sandbox-exec: ... Operation not permitted', exit 65), so everything runs under the one ./py interpreter."""
import os, sys, io, time, runpy, contextlib, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE); sys.path.insert(0, HERE)
SCRIPTS = ['item0.py', 'item1.py', 'item2.py', 'item3.py', 'item4.py', 'item5.py', 'item5b.py',
           'item8.py', 'item9.py', 'item10.py', 'item11.py', 'item12.py', 'defects.py', 'collect.py']
# item 6 and item 7 numbers are produced inside item5.py (Morse indices, kernel identification)
os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
summary = []
for s in SCRIPTS:
    t = time.time()
    buf = io.StringIO(); err = ''
    try:
        with contextlib.redirect_stdout(buf):
            runpy.run_path(os.path.join(HERE, s), run_name='__main__')
        code = 0
    except SystemExit as e:
        code = e.code or 0
    except Exception:
        code = 1; err = traceback.format_exc()
    out = buf.getvalue()
    with open(os.path.join(HERE, 'out', s.replace('.py', '.log')), 'w') as f:
        f.write(out + err)
    npass = out.count('PASS ['); nfail = out.count('FAIL [')
    extra = ' fired=%d missed=%d' % (out.count('FIRED'), out.count('MISSED')) if s == 'defects.py' else ''
    line = '%-11s exit=%s  PASS=%d FAIL=%d%s  (%.0fs)' % (s, code, npass, nfail, extra, time.time() - t)
    print(line, flush=True)
    if err: print(err[-3000:])
    summary.append(line)
with open(os.path.join(HERE, 'out', 'run_all.log'), 'w') as f:
    f.write('\n'.join(summary) + '\n')
