"""Item A.5 in R2: Q on the unit vectors alpha v_a + beta v_b, (a, b) = (5/2,-5/2), (7/2,-7/2), (3/2,-3/2).
Exact: [w x w]_J has components only at M = 2a, 0, -2a, so
  Q = sum_J q_J ( |alpha|^4 C(a,a)^2 + |alpha|^2 |beta|^2 (C(a,-a) + C(-a,a))^2 + |beta|^4 C(-a,-a)^2 ),
C(m1,m2) = <j m1; j m2 | J m1+m2>.  Checked against the generic f_J code (exact) and against the
50-digit quadrature of the defining integral at two points per span.
Usage: ./py sA5.py [--plant=coef]
"""
import sys, json
import mpmath as mpm
from lib import *
from quartic import fJ
import quad

PLANT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--plant=")), None)
chk = Checker()
n = 7; n1 = 8; j = sp.Rational(7, 2)
q = {int(J): sp.Rational(v) for J, v in json.load(open(rel("out_A1_R2.json")))["qJ"].items()}
d = json.load(open(rel("out_slot_R2.json")))
eta = sp.Matrix(n1, 2, lambda r, c: sp.sympify(d["eta"][r][c]))
s = sp.sympify(d["s"])
A, B = sp.symbols("A B", nonnegative=True)   # A = |alpha|^2, B = |beta|^2
P_ = sp.Symbol("p")                           # p = |alpha|^2 |beta|^2 on the unit circle A + B = 1
spans = [(sp.Rational(5, 2), -sp.Rational(5, 2)), (sp.Rational(7, 2), -sp.Rational(7, 2)), (sp.Rational(3, 2), -sp.Rational(3, 2))]
mpm.mp.dps = 50
nodes = quad.haar_D(n, 32, 16)
eta_mp = [[mpm.mpmathify(sp.N(eta[k, a], 70)) for a in range(2)] for k in range(n1)]
out = {}
for a, b in spans:
    expr = sp.Integer(0)
    for J, qJ in q.items():
        caa = cg(j, a, j, a, J, 2 * a)
        cbb = cg(j, b, j, b, J, 2 * b)
        cab = cg(j, a, j, b, J, 0) + cg(j, b, j, a, J, 0)
        expr += qJ * (A ** 2 * caa ** 2 + A * B * cab ** 2 + B ** 2 * cbb ** 2)
    expr = sp.expand(sp.radsimp(sp.expand(expr)))
    if PLANT == "coef":
        expr += A * B / 1000
    poly = sp.Poly(expr, A, B)
    chk(f"span ({a},{b}): coefficients rational", all(c.is_Rational for c in poly.coeffs()))
    on_circle = sp.expand(expr.subs(B, 1 - A))
    pform = sp.expand(sp.simplify(on_circle))
    # write as c0 + c1 p, p = A(1 - A)
    c0 = on_circle.subs(A, 0)
    c1 = sp.Poly(on_circle, A).coeff_monomial(A ** 2) * -1
    chk(f"span ({a},{b}): Q on unit vectors = c0 + c1 |alpha|^2|beta|^2 identically", sp.expand(c0 + c1 * A * (1 - A) - on_circle) == 0)
    label = f"{{{a}, {b}}}"
    print(f"A.5 span {label}:  Q = {sp.factor(expr)}   (A = |alpha|^2, B = |beta|^2, A + B = 1)")
    print(f"      = {c0} + ({c1}) |alpha|^2 |beta|^2 ;  range on the span: [{min(c0, c0 + c1 / 4)}, {max(c0, c0 + c1 / 4)}]")
    # checks: generic exact f_J code and quadrature at two points
    for (al, be) in [(sp.sqrt(sp.Rational(1, 3)), sp.sqrt(sp.Rational(2, 3)) * sp.exp(sp.I * sp.pi / 5)), (sp.Rational(3, 5), sp.Rational(4, 5) * sp.I)]:
        w = [sp.Integer(0)] * n1
        w[int(a + j)] = al; w[int(b + j)] = be
        f = fJ(n, w)
        Qg = sp.radsimp(sp.expand(sum(q[J] * f[J] for J in q)))
        Qf = expr.subs({A: sp.Abs(al) ** 2, B: sp.Abs(be) ** 2})
        dif = abs(sp.N(Qg - Qf, 50))
        chk(f"span {label}: formula = generic f_J evaluation at (|alpha|^2 = {sp.Abs(al)**2})", dif < 1e-45, str(sp.N(dif, 3)))
        wm = [mpm.mpmathify(sp.N(x, 70)) for x in w]
        (a2, a4), = quad.averages(nodes, eta_mp, [wm])
        Qn = a4 / ((mpm.mpf(s.p) / s.q) ** 2 * mpm.fsum(abs(x) ** 2 for x in wm) ** 2)
        dq = abs(Qn - mpm.mpmathify(sp.N(Qf, 60)))
        chk(f"span {label}: formula = 50-digit quadrature of the defining integral", dq < mpm.mpf(10) ** -40, mpm.nstr(dq, 3))
    out[label] = dict(Q_AB=str(expr), c0=str(c0), c1=str(c1), on_unit=f"{c0} + ({c1})*|alpha|^2*|beta|^2")
if PLANT is None:
    json.dump(out, open(rel("out_A5_R2.json"), "w"), indent=1)
chk.summary()
