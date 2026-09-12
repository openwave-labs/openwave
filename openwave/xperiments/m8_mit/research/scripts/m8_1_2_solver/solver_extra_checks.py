"""Extra exact checks used in the write-up of items 15 and 19.
 (a) item-9 rays z = +-i sqrt10/2: the two ring latitudes satisfy sin(lat) = -+1/sqrt3 exactly
     (octahedron along a 3-fold axis); item-8 ring: |r|^2 = (13/2)^(1/5); item-9 prism latitudes.
 (b) ||C(u)||^2, C(u) = [[u (x) u]_6 (x) Theta u]_8, at every item-15 ray (exact).
"""
import sympy as sp

from solver_lib import CG, B_ab, theta_ab, save_results, s

out = {}
# (a)
w1 = (5 * sp.sqrt(2) + 3 * sp.sqrt(6)) / 2          # |w| roots of w^2 - 5 sqrt2 i w + 1 (w = i*|w|, -i*|w'|)
w2 = (3 * sp.sqrt(6) - 5 * sp.sqrt(2)) / 2
chk = {}
for nm, ww in (("w1", w1), ("w2", w2)):
    r2 = ww ** sp.Rational(2, 3)
    sl = (1 - r2) / (1 + r2)
    mp = sp.minimal_polynomial(sl, sp.Symbol("t"))
    chk[nm] = {"|w|": s(ww), "sin_lat_minpoly": s(mp), "float": float(sl)}
    print("octahedron ring", nm, "sin(lat) minimal polynomial:", mp, " value", float(sl))
print("  w1*w2 =", sp.simplify(w1 * w2))
out["item9_octahedron_latitudes"] = chk
# prism
for sign in (+1, -1):
    ww = (sp.sqrt(46) + sign * sp.sqrt(42)) / 2
    r2 = ww ** sp.Rational(2, 3)
    sl = (1 - r2) / (1 + r2)
    print("prism ring |w| =", ww, " sin(lat) =", sp.simplify(sl), "~", float(sl), " minpoly:",
          sp.minimal_polynomial(sl, sp.Symbol("t")))
    out[f"item9_prism_ring_{'+' if sign > 0 else '-'}"] = {"|w|": s(ww), "sin_lat": s(sp.simplify(sl)),
                                                        "float": float(sl),
                                                        "minpoly": s(sp.minimal_polynomial(sl, sp.Symbol('t')))}
r2 = sp.Rational(13, 2) ** sp.Rational(1, 5)
sl8 = (1 - r2) / (1 + r2)
print("item8 ring sin(lat) =", sl8, "~", float(sl8), " latitude (deg) ~", float(sp.asin(sl8) * 180 / sp.pi))
out["item8_ring"] = {"abs_r_sq": s(r2), "sin_lat": s(sl8), "float": float(sl8),
                     "latitude_deg_float": float(sp.asin(sl8) * 180 / sp.pi)}


# (b)
def couple(A, ja, B, jb, J):
    o = [sp.S(0)] * (2 * J + 1)
    for m1 in range(-ja, ja + 1):
        for m2 in range(-jb, jb + 1):
            M = m1 + m2
            if abs(M) <= J and A[m1 + ja] != 0 and B[m2 + jb] != 0:
                o[M + J] += CG(ja, m1, jb, m2, J, M) * A[m1 + ja] * B[m2 + jb]
    return o


r2_, r3, r13 = sp.sqrt(2), sp.sqrt(3), sp.sqrt(13)
rays = {
    "v3": {3: 1}, "v0": {0: 1}, "(v2+v-2)/sqrt2": {2: 1 / r2_, -2: 1 / r2_},
    "(v3+v-3)/sqrt2": {3: 1 / r2_, -3: 1 / r2_},
    "item8 (+)": {2: r13 / 5, -3: 2 * r3 / 5}, "item8 (-)": {2: r13 / 5, -3: -2 * r3 / 5},
    "item9 z=+i sqrt10/2": {3: 1, 0: sp.I * sp.sqrt(10) / 2, -3: 1},
    "item9 z=+sqrt230/10": {3: 1, 0: sp.sqrt(230) / 10, -3: 1},
}
cn = {}
for nm, dct in rays.items():
    u = [sp.S(dct.get(m, 0)) for m in range(-3, 4)]
    ub = [sp.conjugate(x) for x in u]
    C = couple(B_ab(u, 6), 6, theta_ab(ub), 3, 8)
    n2 = sp.simplify(sum(x * sp.conjugate(x) for x in C))
    nu = sp.simplify(sum(x * sp.conjugate(x) for x in u))
    cn[nm] = {"norm2_C": s(n2), "norm2_C_over_norm_u^6": s(sp.simplify(n2 / nu ** 3))}
    print(f"||C(u)||^2 / ||u||^6 at {nm}: {sp.simplify(n2 / nu ** 3)}")
out["item19_C_at_item15_rays"] = cn
save_results("extra_checks", out)
