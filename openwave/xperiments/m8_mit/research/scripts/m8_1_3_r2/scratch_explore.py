import json, time
from lib import *
from quartic import h_data, mat_i
from sympy.polys.matrices import DomainMatrix
n = 7
q = {7: sp.Rational(144, 143), 5: sp.Rational(196, 143), 3: sp.Rational(28, 11), 1: sp.Integer(0)}
pts = json.load(open("inputs.json"))["R2 points"]["points"]
Nk = [sp.sqrt(sp.factorial(k) * sp.factorial(n - k)) for k in range(8)] * 2
for name in ["line {5/2, -3/2}", "line {7/2, -5/2}"]:
    comp = pts[name]
    u = [sp.Integer(0)] * 8
    for m, v in comp.items():
        u[int(sp.Rational(m) + sp.Rational(7, 2))] = sp.sympify(v)
    h, g, H = h_data(n, q, u)
    Mh = H - 4 * h * sp.eye(16)
    Ms = sp.Matrix(16, 16, lambda a, b: sp.radsimp(Mh[a, b] * Nk[a] / Nk[b]))
    print(name, set(Ms))
    t = time.time()
    S = mat_i(n) * Ms
    dm = DomainMatrix.from_Matrix(S)
    print(dm.domain)
    cp = dm.charpoly()
    print(time.time() - t, cp)
