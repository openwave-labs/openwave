"""Item 11: the non-critical point u = (v3 + v1 - v-2)/sqrt3.

Reports 924 r6, the tangential gradient g = grad N(u) - 4 N(u) u and |g|, and |M_u d| for the four
generators d of O_u exactly as section 1.5 writes them (not normalised).  Also checks the identity
M_u (X u) = X g for X in {i, -i J_x, -i J_y, -i J_z} (derived in RETURN.md, item 11).
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
import mpmath
import core, hess
from core import report, IDX
from hess import simp

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True
c = [sp.Integer(0)] * 7
c[IDX[3]], c[IDX[1]], c[IDX[-2]] = 1, 1, -1
cu, r = hess.unit_real(c)
val = core.rhat6(c)
print("924 r6 =", 924 * val)
g, r2, Nv = core.sphere_grad(c)
ok &= report("unit representative consistent", (sp.Matrix(r2) - r).applyfunc(simp) == sp.zeros(14, 1))
gn = sp.sqrt(simp(sum(t ** 2 for t in g)))
print("|tangential gradient| =", sp.radsimp(gn), "~", sp.N(gn, 20))
ok &= report("gradient is tangent (Re<u, g> = 0)", simp((r.T * g)[0, 0]) == 0)
Mu = hess.M_u(r, drop_4N=False)
og = hess.O_gens(cu)
Jz, Jp, Jm, Jx, Jy = core.Jmats()
Xs = {"i u": sp.I * sp.eye(7), "-i Jx u": -sp.I * Jx, "-i Jy u": -sp.I * Jy, "-i Jz u": -sp.I * Jz}
gc = sp.Matrix(core.from_real(list(g)))
res = {}
minpolys = {}
for k, d in og.items():
    v = (Mu * d).applyfunc(simp)
    nv2 = sp.radsimp(sp.expand(sum(t ** 2 for t in v)))
    nv = sp.sqrt(nv2)
    Xg = hess.cvec_to_real(list((Xs[k] * gc).applyfunc(simp)))
    ok &= report("identity M_u(%s) = X g holds exactly" % k, (v - Xg).applyfunc(simp) == sp.zeros(14, 1))
    res[k] = nv
    mp_ = sp.minimal_polynomial(nv, sp.Symbol("X"))
    print("  |M_u (%s)|^2 = %s ; |M_u (%s)| ~ %s ; minimal polynomial of the norm: %s" % (k, nv2, k, sp.N(nv, 20), mp_))
    minpolys[k] = str(mp_)
ok &= report("the O_u-annihilation check of item 5 fails here (some residual nonzero)", any(v != 0 for v in res.values()))
ok &= report("|M_u (i u)| equals |grad| exactly", core.is_zero_exact(res["i u"] ** 2 - gn ** 2))
# numerical route for |grad| : central difference of r6 along the tangent g/|g| (mpmath 50 digits)
mpmath.mp.dps = 50
un = [mpmath.mpc(sp.N(sp.re(q), 60), sp.N(sp.im(q), 60)) for q in cu]
gh = [mpmath.mpc(sp.N(sp.re(q) / gn, 60), sp.N(sp.im(q) / gn, 60)) for q in gc]
f = lambda s: core.rhat6_mp([a * mpmath.cos(s) + b * mpmath.sin(s) for a, b in zip(un, gh)], mpmath)
dnum = mpmath.diff(f, 0)
err = abs(dnum - mpmath.mpf(sp.N(gn, 60)))
ok &= report("numerical directional derivative along g/|g| equals |g| (mpmath 50 digits)", err < mpmath.mpf(10) ** -35,
             "diff %s" % mpmath.nstr(err, 3))
json.dump({"924_r6": str(924 * val), "grad_norm": str(sp.radsimp(gn)), "grad_norm_sq": str(simp(gn ** 2)),
           "M_u_d_norms_sq": {k: str(sp.radsimp(v ** 2)) for k, v in res.items()},
           "M_u_d_norm_minimal_polynomials": minpolys},
          open(os.path.join(HERE, "out", "item11.json"), "w"), indent=1)
print("ITEM11", "ALLPASS" if ok else "SOMEFAIL")
