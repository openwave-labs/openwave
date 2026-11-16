import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from pointdata import *
inp = json.load(open(os.path.join(HERE, "..", "inputs.json")))
gm = [sp.Matrix([[sp.Rational(e) for e in r] for r in M]) for M in inp['generators']['matrices']]
h4 = quartic({6: '1288/1287', 4: '35/33', 2: '14/9', 0: '7/3'})
for name, comp in inp['census']['points'].items():
    t = time.time()
    w = [sp.sympify(comp.get(str(m), '0')) for m in range(3, -4, -1)]
    d = PointData(h4, w, gm)
    print(name, d.critical, d.hu, d.k, d.r, d.MK_zero, d.Mu == sp.zeros(14, 1), field_of(list(d.Mop)), round(time.time() - t, 1), flush=True)
