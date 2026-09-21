"""Item 12: H_u along one N_u direction, two ways.

Orbit: Wmin, u = (v0 + sqrt(10/23)(v3 + v-3))/|.| (algebraic coordinates, finite stabiliser).
Direction: e = P_N w / |P_N w| with w = (1, 2, ..., 14) in the real coordinates (a generic
direction in N_u, exact).
Route A (definition): d^2/ds^2 r6(u cos s + e sin s) at s = 0,
   A1 exactly by sympy differentiation of the closed-form rational function of s,
   A2 numerically by mpmath.diff at 60 digits (independent of the matrix formula).
Route B (item-5 formula): e^T (Hess N(u) - 4 N(u) I) e, exact.
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
R = sp.Rational
c = [sp.Integer(0)] * 7
c[IDX[0]] = 1; c[IDX[3]] = sp.sqrt(R(10, 23)); c[IDX[-3]] = sp.sqrt(R(10, 23))
cu, r = hess.unit_real(c)
og = hess.O_gens(cu)
B = hess.N_basis(r, og)
Bm = sp.Matrix.hstack(*B)
PN = (Bm * (Bm.T * Bm).inv() * Bm.T).applyfunc(simp)
w = sp.Matrix(list(range(1, 15)))
p = (PN * w).applyfunc(simp)
e = (p / sp.sqrt(simp((p.T * p)[0, 0]))).applyfunc(simp)
ok &= report("e is a unit vector in N_u (orthogonal to u and to O_u)",
             simp((e.T * e)[0, 0]) == 1 and simp((r.T * e)[0, 0]) == 0 and all(simp((d.T * e)[0, 0]) == 0 for d in og.values()))
# route B
Mu = hess.M_u(r)
HB = simp((e.T * Mu * e)[0, 0])
print("route B (matrix formula, exact):", HB, "~", sp.N(HB, 40))
# route A1: exact second derivative of the definition.  r6(u cos s + e sin s) is a rational function
# of (cos s, sin s); its second derivative at 0 only needs the Taylor data cos s = 1 - s^2/2 + O(s^4),
# sin s = s + O(s^3), so f(s) = N(c(s)) / |c(s)|^4 is expanded exactly to order s^2.
s = sp.Symbol("s", real=True)
curve = [ri * (1 - s ** 2 / 2) + ei * s for ri, ei in zip(r, e)]
Ns = sp.Poly(sp.expand(core.N_poly().xreplace(dict(zip(core.RV, curve)))), s)
n2 = sp.Poly(sp.expand(sum(q ** 2 for q in curve)), s)
N0, N1, N2 = [simp(Ns.coeff_monomial(s ** k)) for k in range(3)]
q0, q1, q2 = [simp(n2.coeff_monomial(s ** k)) for k in range(3)]
# 1/|c|^4 = q^-2 with q = q0 + q1 s + q2 s^2 : series to s^2
inv0 = 1 / q0 ** 2
inv1 = -2 * q1 / q0 ** 3
inv2 = (3 * q1 ** 2 - 2 * q0 * q2) / q0 ** 4
HA1 = simp(2 * (N0 * inv2 + N1 * inv1 + N2 * inv0))
print("route A1 (exact d^2/ds^2 at 0):", HA1)
dA1 = simp(HA1 - HB)
ok &= report("route A1 (exact derivative) equals route B exactly", core.is_zero_exact(dA1), "difference = %s" % dA1)
# route A2: numerical derivative, mpmath at 60 digits
mpmath.mp.dps = 60
un = [mpmath.mpc(sp.N(sp.re(q), 80), sp.N(sp.im(q), 80)) for q in cu]
ec = core.from_real(list(e))
en = [mpmath.mpc(sp.N(sp.re(q), 80), sp.N(sp.im(q), 80)) for q in ec]
F = lambda t: core.rhat6_mp([a * mpmath.cos(t) + b * mpmath.sin(t) for a, b in zip(un, en)], mpmath)
HA2 = mpmath.diff(F, 0, 2)
dA2 = abs(HA2 - mpmath.mpf(sp.N(HB, 80)))
print("route A2 (mpmath.diff, 60 digits):", mpmath.nstr(HA2, 40), " |A2 - B| =", mpmath.nstr(dA2, 3))
ok &= report("route A2 agrees with route B to 1e-40", dA2 < mpmath.mpf(10) ** -40)
# planted control: the matrix formula without the -4N I term disagrees with the definition
HBbad = simp((e.T * hess.M_u(r, drop_4N=True) * e)[0, 0])
ok &= report("control: dropping -4N(u) I from the formula is detected by route A2",
             abs(mpmath.mpf(sp.N(HBbad, 60)) - HA2) > mpmath.mpf(10) ** -5,
             "wrong formula gives %s" % sp.N(HBbad, 15))
json.dump({"orbit": "Wmin", "direction": "P_N(1..14)/|.|", "H_matrix_formula": str(HB), "H_exact_derivative": str(HA1),
           "exact_difference": str(dA1), "H_mpmath_60digits": mpmath.nstr(HA2, 50), "numerical_difference": mpmath.nstr(dA2, 3)},
          open(os.path.join(HERE, "out", "item12.json"), "w"), indent=1)
print("ITEM12", "ALLPASS" if ok else "SOMEFAIL")
