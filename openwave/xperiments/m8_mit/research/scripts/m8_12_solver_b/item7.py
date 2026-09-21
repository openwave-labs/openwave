"""Item 7: the kernel of H_u on N_u at v1 (the only orbit with n0 > 0).

Checks exactly that ker(H_u|N_u) = span_R{v-3, i v-3} = the tangent space at v1 of the class
L13 = span{v1, v-3} (fixed space of C4 with a character), that r6 along L13 is
F(t) = 75/308 - (8/33) t^2 + O(t^3), t = |z|^2 (so the kernel is a genuine quartic degeneracy),
and the binomial identity behind F'(0) = 0.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
import core, hess
from core import report, IDX
from hess import simp

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True
c = [sp.Integer(0)] * 7; c[IDX[1]] = 1
cu, r = hess.unit_real(c)
Mu = hess.M_u(r)
og = hess.O_gens(cu)
B = hess.N_basis(r, og)
Bm = sp.Matrix.hstack(*B)
G = Bm.T * Bm
H = Bm.T * Mu * Bm
# kernel of the restricted form: vectors a with H a = 0 (G invertible) -> e = B a
ker = H.nullspace()
kv = [(Bm * a).applyfunc(simp) for a in ker]
print("dim ker(H_u | N_u) =", len(kv))
e1 = hess.cvec_to_real([0, 0, 0, 0, 0, 0, 1])            # v-3
e2 = hess.cvec_to_real([0, 0, 0, 0, 0, 0, sp.I])         # i v-3
Kspan = sp.Matrix.hstack(*kv)
ok &= report("kernel has dimension 2", len(kv) == 2)
ok &= report("kernel = span_R{v-3, i v-3} (rank test)",
             sp.Matrix.hstack(Kspan, e1, e2).rank() == 2 and sp.Matrix.hstack(e1, e2).rank() == 2)
ok &= report("M_u v-3 = 0 and M_u (i v-3) = 0 exactly (full 14-vector, not only on N_u)",
             (Mu * e1).applyfunc(simp) == sp.zeros(14, 1) and (Mu * e2).applyfunc(simp) == sp.zeros(14, 1))
ok &= report("v-3, i v-3 lie in N_u (orthogonal to u and O_u)",
             all(simp((v.T * w)[0, 0]) == 0 for v in (e1, e2) for w in [r] + list(og.values())))
t = sp.Symbol("t", nonnegative=True)
F = (t ** 2 + 450 * t + 225) / (924 * (t + 1) ** 2)
x = sp.Symbol("x", real=True)
cz = [0, 0, 1, 0, 0, 0, x]
ok &= report("r6(v1 + x v-3) = F(x^2) (item 2 restriction, recomputed)", sp.simplify(core.rhat6(cz) - F.subs(t, x ** 2)) == 0)
ser = sp.series(F, t, 0, 3).removeO()
print("F(t) =", sp.expand(ser), "+ O(t^3)")
ok &= report("F(t) = 75/308 - (8/33) t^2 + O(t^3): no t term, nonzero t^2 term",
             sp.expand(ser - (sp.Rational(75, 308) - sp.Rational(8, 33) * t ** 2)) == 0)
# binomial identity: t-coefficient of 924 N(v1 + z v-3) = 2 * 924 N(v1)
C = lambda a, b: sp.binomial(a, b)
cg1m1 = sp.sqrt(sp.Rational(C(6, 4) * C(6, 2), C(12, 6)))
cgm33 = sp.sqrt(sp.Rational(C(6, 0) * C(6, 6), C(12, 6)))
cg13 = sp.sqrt(sp.Rational(C(6, 4) * C(6, 6), C(12, 10)))
cgm3m1 = sp.sqrt(sp.Rational(C(6, 0) * C(6, 2), C(12, 2)))
ok &= report("these four CG equal <3 1;3 -1|6 0>, <3 -3;3 3|6 0>, <3 1;3 3|6 4>, <3 -3;3 -1|6 -4>",
             [cg1m1, cgm33, cg13, cgm3m1] == [core.CG6[(1, -1)], core.CG6[(-3, 3)], core.CG6[(1, 3)], core.CG6[(-3, -1)]])
lhs = 2 * cg1m1 * cgm33 + cg13 ** 2 + cgm3m1 ** 2
rhs = 2 * cg1m1 ** 2
print("B = 2<1,-1|0><-3,3|0> + <1,3|4>^2 + <-3,-1|-4>^2 =", simp(lhs), " ; 2C = 2<1,-1|0>^2 =", simp(rhs))
ok &= report("binomial identity 30/924 + 210/924 + 210/924 = 2 * 225/924 (so F'(0) = 0)", simp(lhs - rhs) == 0)
json.dump({"kernel_dim": len(kv), "kernel": "span_R{v-3, i v-3} = T_{v1} L13", "F_series": str(sp.expand(ser))},
          open(os.path.join(HERE, "out", "item7.json"), "w"), indent=1)
print("ITEM7", "ALLPASS" if ok else "SOMEFAIL")
