import sympy as sp, core
from core import IDX
c = [sp.Integer(0)] * 7
c[IDX[3]], c[IDX[1]], c[IDX[-2]] = 1, 1, -1
g, r, Nv = core.sphere_grad(c)
print("g =", list(g))
gc = sp.Matrix(core.from_real(list(g)))
Jz, Jp, Jm, Jx, Jy = core.Jmats()
for nm, X in (("i", sp.I * sp.eye(7)), ("Jx", -sp.I * Jx), ("Jy", -sp.I * Jy), ("Jz", -sp.I * Jz)):
    v = X * gc
    n2 = sp.radsimp(sp.expand(sum(sp.expand(q * sp.conjugate(q)) for q in v)))
    print(nm, n2, sp.N(n2, 20))
