"""Item 2: r-hat_6 restricted to each dimension-2 class; complete critical sets.

Chart for class span{w0, w1}: u = w0 + z w1, z = x + i y, omitting the point [w1];
the second chart u = w' w0 + w1 (w' = x' + i y') covers [w1] at w' = 0.
Completeness is by elimination (radial reduction to one variable for the five
classes with a circle symmetry; lex Groebner basis for U and W), not by a solver.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
import mpmath
import core
from core import report, IDX, MS

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True
x, y = sp.symbols("x y", real=True)
t = sp.symbols("t", nonnegative=True)
zc = x + sp.I * y


def vec(d):
    c = [sp.Integer(0)] * 7
    for m, val in d.items():
        c[IDX[m]] += sp.sympify(val)
    return c


def comb(w0, w1, a, b):
    return [sp.expand(a * p + b * q) for p, q in zip(w0, w1)]


CLASSES = {
    "L12": (vec({1: 1}), vec({-2: 1}), "C3", 3),
    "L13": (vec({1: 1}), vec({-3: 1}), "C4", 4),
    "L22": (vec({2: 1}), vec({-2: 1}), "C4", 4),
    "L23": (vec({2: 1}), vec({-3: 1}), "C5", 5),
    "L33": (vec({3: 1}), vec({-3: 1}), "C6", 6),
    "U":   (vec({0: 1}), vec({2: 1, -2: 1}), "D2", None),
    "W":   (vec({0: 1}), vec({3: 1, -3: 1}), "D3", None),
}
PLANT = core.PLANT


def restrict(w0, w1, a, b):
    return sp.factor(sp.cancel(core.rhat6(comb(w0, w1, a, b))))


out = {}
for name, (w0, w1, grp, _) in CLASSES.items():
    print("\n=== class", name, "(fixed space of", grp + ")")
    f = restrict(w0, w1, 1, zc)
    f2 = restrict(w0, w1, zc, 1)          # second chart: u = w' w0 + w1, (x, y) now mean w'
    print("chart u = w0 + z w1, omitted point [w1];  r6 =", f)
    rec = {"chart": "u = (%s) + z (%s)" % (w0, w1), "restriction": str(f), "second_chart": str(f2), "crit": []}
    crit = []    # (description, x, y, chart, value)
    if name in ("U", "W"):
        # ---- two-variable elimination
        gx = sp.numer(sp.together(sp.diff(f, x)))
        gy = sp.numer(sp.together(sp.diff(f, y)))
        G = sp.groebner([gx, gy], x, y, order="lex")
        print("lex Groebner basis:", [sp.factor(g) for g in G.exprs])
        ok &= report("%s: Groebner basis is zero-dimensional (finitely many complex solutions)" % name,
                     G.is_zero_dimensional)
        # last element is univariate in y; solve it exactly, back-substitute
        gl = [g for g in G.exprs if not g.has(x)]
        assert len(gl) == 1
        ysols = sp.roots(sp.Poly(gl[0], y))
        ok &= report("%s: univariate eliminant in y solved completely by radicals" % name,
                     sum(ysols.values()) == sp.Poly(gl[0], y).degree())
        sols = []
        for y0 in ysols:
            polys = [sp.Poly(g.subs(y, y0), x) for g in G.exprs if g.has(x)]
            polys = [p for p in polys if not p.is_zero]
            gg = polys[0]
            for p in polys[1:]:
                gg = sp.gcd(gg, p)
            for x0 in sp.roots(gg):
                if sp.im(x0) == 0 and sp.im(y0) == 0:
                    sols.append((core.exact(x0), core.exact(y0)))
        # verify each real solution exactly
        for (x0, y0) in sols:
            ok &= report("%s: gradient vanishes exactly at z = %s" % (name, x0 + sp.I * y0),
                         sp.simplify(gx.subs({x: x0, y: y0})) == 0 and sp.simplify(gy.subs({x: x0, y: y0})) == 0)
            crit.append(("point", x0, y0, 1))
        # Hessian type in the chart (2x2)
    else:
        # ---- circle symmetry: f depends on |z|^2 only
        F = sp.factor(f.subs(y, 0).subs(x, sp.sqrt(t)))
        ok &= report("%s: r6 restricted depends on |z|^2 only (f(x,y) - F(x^2+y^2) = 0 exactly)" % name,
                     sp.simplify(f - F.subs(t, x ** 2 + y ** 2)) == 0, "F(t) = %s" % F)
        dF = sp.factor(sp.diff(F, t))
        numer = sp.numer(sp.together(dF))
        print("F(t) =", F, "; F'(t) =", dF)
        troots = [r for r in sp.roots(sp.Poly(numer, t)) if r.is_real and r > 0] if sp.Poly(numer, t).degree() > 0 else []
        # grad f = 2 F'(|z|^2) (x, y): zero iff z = 0 or F'(|z|^2) = 0
        crit.append(("point", sp.Integer(0), sp.Integer(0), 1))
        for r in troots:
            crit.append(("circle |z|^2=%s" % r, sp.sqrt(r), sp.Integer(0), 1))
        rec["F"] = str(F); rec["dF_numerator"] = str(sp.factor(numer))
        rec["t_crit"] = [str(r) for r in troots]
    # ---- omitted point: second chart at w' = 0
    g2x = sp.diff(f2, x).subs({x: 0, y: 0}); g2y = sp.diff(f2, y).subs({x: 0, y: 0})
    ok &= report("%s: omitted point [w1] is critical (second-chart gradient at 0 is zero)" % name,
                 sp.simplify(g2x) == 0 and sp.simplify(g2y) == 0)
    crit.append(("omitted point [w1]", sp.Integer(0), sp.Integer(0), 2))
    # ---- values, local type on the line, and Euler characteristic check
    euler = 0
    for (kind, x0, y0, ch) in crit:
        ff = f if ch == 1 else f2
        val = core.exact(ff.subs({x: x0, y: y0}))
        if kind.startswith("circle"):
            # Morse-Bott: transverse second derivative along the radius
            Hrr = sp.simplify(sp.diff(ff.subs(y, 0), x, 2).subs(x, x0))
            typ = "max-circle" if Hrr < 0 else ("min-circle" if Hrr > 0 else "degenerate")
            contrib = 0      # chi(S^1) = 0
            hess = [Hrr]
        else:
            H = sp.hessian(ff, (x, y)).subs({x: x0, y: y0}).applyfunc(sp.simplify)
            evs = H.eigenvals()
            nneg = sum(m for e, m in evs.items() if e < 0)
            nzero = sum(m for e, m in evs.items() if e == 0)
            hess = [str(e) for e in evs]
            if nzero == 0:
                typ = {0: "min", 1: "saddle", 2: "max"}[nneg]
                contrib = (-1) ** nneg          # Poincare-Hopf index of a nondegenerate point
            elif name not in ("U", "W"):
                # degenerate centre of a circle-symmetric chart: grad f = 2 F'(|z|^2) (x, y) and F'
                # has one sign on (0, eps) (rational, nonzero there), so the index is +1
                Fl = sp.factor(ff.subs(y, 0).subs(x, sp.sqrt(t)))
                k = 1
                while sp.simplify(sp.diff(Fl, t, k).subs(t, 0)) == 0:
                    k += 1
                lead = sp.simplify(sp.diff(Fl, t, k).subs(t, 0))
                typ = "degenerate %s (F^(%d)(0) = %s, all lower t-derivatives zero)" % (
                    "max" if lead < 0 else "min", k, lead)
                contrib = 1
            else:
                typ = "degenerate"
                contrib = None
        if contrib is None:
            euler = None
        elif euler is not None:
            euler += contrib
        zval = x0 + sp.I * y0
        where = ("z = %s" % zval) if ch == 1 else "[w1] (w' = 0)"
        if kind.startswith("circle"):
            where = "|z| = %s (circle)" % core.exact(x0)
        print("  critical: %-28s r6 = %-10s ~ %.12f   type on line: %s" % (where, val, float(val), typ))
        rec["crit"].append({"where": where, "value": str(val), "type": typ, "hess_chart": [str(h) for h in hess]})
    ok &= report("%s: Poincare-Hopf index sum over the critical set equals chi(P^1) = 2" % name, euler == 2, "sum=%s" % euler)
    rec["euler_sum"] = euler
    out[name] = rec

json.dump(out, open(os.path.join(HERE, "out", "item2.json"), "w"), indent=1, default=str)

# ---- solver cross-check (not part of the completeness argument): Newton from a grid
mpmath.mp.dps = 30
extra = 0
for name in ("U", "W"):
    w0, w1, _, _ = CLASSES[name]
    f = sp.sympify(out[name]["restriction"], locals={"x": x, "y": y})
    gx = sp.lambdify((x, y), sp.diff(f, x), "mpmath"); gy = sp.lambdify((x, y), sp.diff(f, y), "mpmath")
    known = [complex(sp.N(sp.sympify(c["where"].split("=")[1]))) for c in out[name]["crit"] if c["where"].startswith("z =")]
    found = set()
    for i in range(-6, 7):
        for j in range(-6, 7):
            try:
                s = mpmath.findroot([lambda a, b: gx(a, b), lambda a, b: gy(a, b)], (0.37 * i + 0.01, 0.41 * j + 0.02))
            except Exception:
                continue
            zz = complex(float(s[0]), float(s[1]))
            if abs(zz) > 50:
                continue
            if min(abs(zz - k) for k in known) > 1e-8:
                extra += 1
                print("solver found an unlisted critical point", name, zz)
ok &= report("grid-Newton solver finds no critical point outside the eliminated list (U, W)", extra == 0)
print("ITEM2", "ALLPASS" if ok else "SOMEFAIL")
