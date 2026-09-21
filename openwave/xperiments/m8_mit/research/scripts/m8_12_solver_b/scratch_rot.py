import numpy as np, scipy.optimize as so, math
import sympy as sp
import core, rot
def v(d):
    c = np.zeros(7, complex)
    for m, a in d.items(): c[core.IDX[m]] += a
    return c / np.linalg.norm(c)
hexa = v({3: 1, -3: 1}); octa = v({2: 1, -2: -1})
Umax = v({0: 1, 2: math.sqrt(0.3), -2: math.sqrt(0.3)})
Wsad = v({0: 1, 3: 1j * math.sqrt(0.4), -3: 1j * math.sqrt(0.4)})
def D(p):
    a, b, th = p
    n = (math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a))
    return rot.D3_num(n, th)
for src, tgt, nm in [(hexa, Umax, 'hex->Umax'), (octa, Wsad, 'oct->Wsad')]:
    best = None
    rng = np.random.default_rng(1)
    for _ in range(300):
        p0 = rng.uniform([0, 0, 0], [math.pi, 2 * math.pi, 2 * math.pi])
        r = so.minimize(lambda p: 1 - abs(np.vdot(tgt, D(p) @ src)) ** 2, p0, method='Nelder-Mead', options={'xatol': 1e-12, 'fatol': 1e-15, 'maxiter': 4000})
        if best is None or r.fun < best.fun: best = r
        if r.fun < 1e-12:
            a, b, th = r.x
            n = (math.sin(a) * math.cos(b), math.sin(a) * math.sin(b), math.cos(a))
            print(nm, r.fun, np.round(n, 6), th % (2 * math.pi) / math.pi)
