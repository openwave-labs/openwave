import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, sympy as sp
from scipy.linalg import expm
from scipy.optimize import minimize
import common as C, classes as K
Jx = np.array(C.JX.evalf(), dtype=complex); Jy = np.array(C.JY.evalf(), dtype=complex); Jz = np.array(C.JZ.evalf(), dtype=complex)
def D(a, b, g): return expm(-1j * a * Jz) @ expm(-1j * b * Jy) @ expm(-1j * g * Jz)
def nv(c): c = np.array([complex(z) for z in c]); return c / np.linalg.norm(c)
s2 = sp.sqrt(2)
Fx = K.vec({0: 1, 2: sp.sqrt(sp.Rational(3, 5)) / s2, -2: sp.sqrt(sp.Rational(3, 5)) / s2})
Gx = K.vec({0: 1, 3: sp.sqrt(sp.Rational(4, 5)) / s2, -3: -sp.sqrt(sp.Rational(4, 5)) / s2})
for name, src, tgt in [('cat->Fx', K.CLASSES1['cat']['u'], Fx), ('xyz->Gx', K.CLASSES1['xyz']['u'], Gx)]:
    s = nv(src); tg = nv(tgt)
    best = None
    rng = np.random.default_rng(0)
    for k in range(200):
        r = minimize(lambda p: 1 - abs(np.vdot(tg, D(*p) @ s)) ** 2, rng.uniform(0, 2 * np.pi, 3), method='BFGS')
        if best is None or r.fun < best.fun: best = r
        if r.fun < 1e-14:
            print(name, r.fun, np.round(np.array(r.x) / np.pi, 6), np.cos(r.x[1]))
