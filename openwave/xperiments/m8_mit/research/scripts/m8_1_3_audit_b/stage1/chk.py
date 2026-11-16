"""Check reporting.  Every PASS line is a test that can fail: the condition is computed, not asserted.
Set the environment variable PLANT=<tag> to plant a defect in the code path carrying that tag (see planted.py)."""
import os, json, sys
PLANT = os.environ.get('PLANT', '')
RESULTS = []


class _Tee:
    def __init__(self, path):
        self.f = open(path, 'w'); self.o = sys.__stdout__

    def write(self, s):
        self.o.write(s); self.f.write(s); self.f.flush()

    def flush(self):
        self.o.flush(); self.f.flush()


# the printed output of each script is kept in <script>.log beside it (or <script>.PLANT-<tag>.log for planted runs)
_main = sys.modules.get('__main__')
if _main is not None and getattr(_main, '__file__', None):
    _p = os.path.abspath(_main.__file__)
    _log = _p[:-3] + (f'.PLANT-{PLANT}' if PLANT else '') + '.log'
    sys.stdout = _Tee(_log)


def check(label, cond, detail=''):
    st = 'PASS' if cond else 'FAIL'
    RESULTS.append((label, st))
    print(f'[{st}] {label}' + (f' :: {detail}' if detail else ''), flush=True)
    return cond


def planted(tag):
    return PLANT == tag


def save_json(path, obj):
    here = os.path.dirname(os.path.abspath(__file__))
    if PLANT:
        path = path[:-5] + f'.PLANT-{PLANT}.json'   # planted runs never overwrite the real results
    with open(os.path.join(here, path), 'w') as f:
        json.dump(obj, f, indent=1)


def load_json(path):
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, path)) as f:
        return json.load(f)
