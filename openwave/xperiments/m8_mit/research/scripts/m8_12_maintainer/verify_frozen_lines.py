"""The frozen line restrictions, checked against r-hat-6 itself in the frozen charts."""
import json, pathlib, sys
sys.path.insert(0, pathlib.Path(__file__).resolve().parent.as_posix())
from sympy import I, Rational, simplify, sqrt, symbols, sympify
from cg import MS, N_of, norm2

x, y, s = symbols('x y s', real=True)
NS = {'sqrt': sqrt, 'I': I, 'Rational': Rational, 'x': x, 'y': y, 's': s}
F = json.loads((pathlib.Path(__file__).resolve().parent/'frozen_claims.json').read_text())
E = lambda t: sympify(t, locals=NS)
SAMP = [Rational(1,3), Rational(2,5), Rational(3,7), Rational(5,6)]
ok = True

# the five s-charts: s = |c_first|^2 with the pair normalized
SPANS = {"C3 {v2, v-1}": (2, -1), "C4 {v3, v-1}": (3, -1), "C4 {v2, v-2}": (2, -2),
         "C5 {v3, v-2}": (3, -2), "C6 {v3, v-3}": (3, -3)}
for ln in F["lines"]:
    if ln["chart"] != "s":
        continue
    m1, m2 = SPANS[ln["name"]]
    f = E(ln["restriction"])
    bad = []
    for sv in SAMP:
        c = {m: Rational(0) for m in MS}
        c[m1] = sqrt(sv); c[m2] = sqrt(1-sv)
        want = simplify(N_of(c)/norm2(c)**2)
        got = simplify(f.subs(s, sv))
        if simplify(got-want) != 0:
            bad.append((sv, got, want))
    ok &= not bad
    print(f"  {'PASS' if not bad else 'FAIL'}  frozen {ln['name']}: restriction == r6 in s = |c_{m1}|^2"
          + ("" if not bad else f"  at s={bad[0][0]}: {bad[0][1]} vs {bad[0][2]}"))

# the two dihedral charts: u = b1 + z*b2, b1 and b2 unnormalized as the paper writes them
DIH = {"D3 {v3+v-3, v0}": ({3: 1, -3: 1}, {0: 1}), "D2 {v2+v-2, v0}": ({2: 1, -2: 1}, {0: 1})}
for ln in F["lines"]:
    if ln["chart"] == "s":
        continue
    b1, b2 = DIH[ln["name"]]
    f = E(ln["restriction"])
    bad = []
    for xv, yv in [(Rational(1,3), Rational(0)), (Rational(0), Rational(2,5)), (Rational(3,7), Rational(-2,3))]:
        z = xv + I*yv
        c = {m: Rational(b1.get(m,0)) + z*Rational(b2.get(m,0)) for m in MS}
        want = simplify(N_of(c)/norm2(c)**2)
        got = simplify(f.subs({x: xv, y: yv}))
        if simplify(got-want) != 0:
            bad.append((xv, yv, got, want))
    ok &= not bad
    print(f"  {'PASS' if not bad else 'FAIL'}  frozen {ln['name']}: restriction == r6 with b1, b2 unnormalized"
          + ("" if not bad else f"  at {bad[0][0]},{bad[0][1]}: {bad[0][2]} vs {bad[0][3]}"))

print("\nfrozen line restrictions:", "ALL ANCHORED" if ok else "A FAILURE ABOVE")
