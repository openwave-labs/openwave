"""Stage 2, item 19 continued: L_8(u) = [rho_6(u) (x) u]_8 (CS phase), exact.

Also checks (not proofs): L_8 vanishes exactly on time-reversal-invariant rays and not elsewhere,
and ||L_8(u)||^2 / ||C_iso(u)||^2 is constant (C_iso = V_8-isotypic component of u (x) Theta u (x) u,
the stage-1 reading), i.e. the stage-1 characterisation is insensitive to this normalisation.
"""
import json
import pathlib
import itertools
from fractions import Fraction as Fr
from math import comb
import sympy as sp
from audit_su2lib import cg, couple, rho, theta, basis, simp, norm2

HERE = pathlib.Path(__file__).parent
res = {}

v3 = basis(3)
r6 = [simp(x) for x in rho(v3, 6)]
L8 = [simp(x) for x in couple(r6, v3, 6, 3, 8)]
comp3 = L8[3 + 8]
res["rho6_v3_N=-6..6"] = [str(x) for x in r6]
res["L8_v3_N=-8..8"] = [str(x) for x in L8]
res["L8_v3_component_N=3"] = str(comp3)
res["L8_v3_component_N=3_squared"] = str(simp(comp3 ** 2))
# independent closed-form assembly: rho6(v3) = -<33;3-3|60> e_0, then times <6 0;3 3|8 3>
alt = simp(-cg(3, 3, 3, -3, 6, 0) * cg(6, 0, 3, 3, 8, 3))
assert simp(alt - comp3) == 0
res["CG_60_33_83"] = str(cg(6, 0, 3, 3, 8, 3))
assert all(L8[i] == 0 for i in range(17) if i != 11)
res["L8_v3_norm2"] = str(norm2(L8))

# ---- checks vs the stage-1 isotypic C (exact monomial-basis Casimir route, re-implemented here) ----
from audit_field import K, R5, ZERO, ONE


def kq(re, im=0):
    return K(R5(Fr(re)), R5(Fr(im)))


def theta_mon(c):
    out = [ZERO] * 7
    for k in range(7):
        out[6 - k] = out[6 - k] + c[k].conj() * ((-1) ** abs(k - 3))
    return out


def cas(v):
    n = 6
    o = {}
    for key, val in v.items():
        mz = sum(key) - 9
        o[key] = o.get(key, ZERO) + val * (mz * mz)

    def sh(v, up):
        out = {}
        for key, val in v.items():
            for p in range(3):
                k = key[p]
                if up and k < n:
                    nk = list(key); nk[p] += 1
                    out[tuple(nk)] = out.get(tuple(nk), ZERO) + val * (n - k)
                if not up and k > 0:
                    nk = list(key); nk[p] -= 1
                    out[tuple(nk)] = out.get(tuple(nk), ZERO) + val * k
        return out
    half = kq(Fr(1, 2))
    for d in (sh(sh(v, False), True), sh(sh(v, True), False)):
        for key, val in d.items():
            o[key] = o.get(key, ZERO) + val * half
    return {k: x for k, x in o.items() if not x.iszero()}


def proj8(v):
    for Lp in range(10):
        if Lp == 8:
            continue
        cv = cas(v)
        den = Fr(1, 72 - Lp * (Lp + 1))
        v = {k: (cv.get(k, ZERO) - v.get(k, ZERO) * (Lp * (Lp + 1))) * kq(den) for k in set(cv) | set(v)}
        v = {k: x for k, x in v.items() if not x.iszero()}
    return v


def Cnorm2(c):
    th = theta_mon(c)
    w = {}
    for k1, k2, k3 in itertools.product(range(7), repeat=3):
        val = c[k1] * th[k2] * c[k3]
        if not val.iszero():
            w[(k1, k2, k3)] = val
    p = proj8(w)
    s = ZERO
    for key, val in p.items():
        s = s + val.conj() * val * kq(Fr(1, comb(6, key[0]) * comb(6, key[1]) * comb(6, key[2])))
    return s


def to_on(c):
    """monomial coords c_k (k = m+3) -> orthonormal u_m = c_k / sqrt(C(6,k))."""
    return [c[k].tosympy() / sp.sqrt(comb(6, k)) for k in range(7)]


states = {
    "generic A": [kq(1, 2), kq(-1), kq(0, 3), kq(2), kq(1, -1), kq(0), kq(3, 1)],
    "generic B": [kq(2, -1), kq(0), kq(1), kq(0, 1), kq(-3), kq(1, 1), kq(Fr(1, 3))],
    "v3 (e6)": [ZERO] * 6 + [ONE],
    "v2 (e5)": [ZERO] * 5 + [ONE, ZERO],
    "v3 + 2 v-3": [kq(2)] + [ZERO] * 5 + [ONE],
    "v0 (TR-invariant)": [ZERO] * 3 + [ONE] + [ZERO] * 3,
    "v3 + v-3 (TR-invariant)": [ONE] + [ZERO] * 5 + [ONE],
    "i*(Theta-fixed) (TR-invariant)": None,
}
# a Theta-fixed state times i
c = [kq(1, 2), kq(-2, 1), kq(3), kq(2), ZERO, ZERO, ZERO]
for k in range(3):
    c[6 - k] = c[k].conj() * ((-1) ** abs(k - 3))
states["i*(Theta-fixed) (TR-invariant)"] = [x * kq(0, 1) for x in c]

chk = {}
ratios = set()
import mpmath as mp
mp.mp.dps = 50
for name, cc in states.items():
    th = theta_mon(cc)
    lam = next(th[k] / cc[k] for k in range(7) if not cc[k].iszero())
    inv = all(th[k] == lam * cc[k] for k in range(7))
    u = to_on(cc)
    L = couple(rho(u, 6), u, 6, 3, 8)
    nL = sp.N(sum(sp.expand(x * sp.conjugate(x)) for x in L), 50)
    nL_exact_zero = all(simp(x) == 0 for x in L)
    nC = Cnorm2(cc)
    rec = {"TR_invariant": inv, "L8_exactly_zero": nL_exact_zero, "||L8||^2 (50 digits)": str(nL), "||C_iso||^2": repr(nC)}
    assert nL_exact_zero == inv == nC.iszero(), (name, rec)
    if not inv:
        r = mp.mpf(str(nL)) / mp.mpf(nC.p.tofloat()) if False else sp.N(nL / sp.Rational(nC.p.a.numerator, nC.p.a.denominator), 40)
        assert nC.p.b == 0 and nC.q.iszero()
        rec["ratio ||L8||^2/||C_iso||^2"] = str(r)
        ratios.add(str(sp.N(r, 30)))
    chk[name] = rec
res["checks"] = chk
res["ratio_set_30digits"] = sorted(ratios)
assert len(ratios) == 1
res["ratio_identified"] = str(sp.nsimplify(sp.Float(sorted(ratios)[0], 30), rational=True))
(HERE / "audit_res_stage2_item19.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
