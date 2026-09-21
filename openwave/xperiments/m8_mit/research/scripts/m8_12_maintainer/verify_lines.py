"""Check each room's line restriction against r-hat-6 ITSELF, not against the frozen string.

Each room states its own chart `u = a + z*b`. Rather than argue about conventions,
evaluate r-hat-6 at the vector that chart produces, by the maintainer's own exact
route, and compare to the room's own restriction formula at the same (x, y). A
formula that is right in the room's chart passes here whatever chart it chose, and a
formula that is wrong fails whatever chart it claims.
"""
import json, os, pathlib, sys, random
sys.path.insert(0, pathlib.Path(__file__).resolve().parent.as_posix())
from sympy import I, Rational, simplify, sqrt, symbols, sympify
from cg import MS, N_of, norm2

x, y = symbols('x y', real=True)
NS = {'sqrt': sqrt, 'I': I, 'Rational': Rational, 'x': x, 'y': y}
ROOT = pathlib.Path(os.environ.get('ROOM_ROOT', '.'))
FROZEN = json.loads((pathlib.Path(__file__).resolve().parent / 'frozen_claims.json').read_text())

def vec(lst):
    v = [sympify(str(t), locals=NS) for t in lst]
    return {m: v[3-m] for m in MS}

def r6_at(a, b, xv, yv):
    z = xv + I*yv
    c = {m: simplify(a[m] + z*b[m]) for m in MS}
    return simplify(N_of(c)/norm2(c)**2)

SAMPLES = [(Rational(1,3), Rational(0)), (Rational(0), Rational(2,5)),
           (Rational(3,7), Rational(-2,3)), (Rational(5,4), Rational(1,6))]

def check_room(label, entries):
    print(f"\n=== {label} ===")
    allok = True
    for name, a_list, b_list, restr in entries:
        a, b = vec(a_list), vec(b_list)
        f = sympify(restr, locals=NS)
        bad = []
        for xv, yv in SAMPLES:
            want = r6_at(a, b, xv, yv)
            got = simplify(f.subs({x: xv, y: yv}))
            if simplify(got - want) != 0:
                bad.append((xv, yv, got, want))
        ok = not bad
        allok &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: room restriction == r6 on its own chart"
              + ("" if ok else f"   first mismatch at {bad[0][0]},{bad[0][1]}: {bad[0][2]} vs {bad[0][3]}"))
    return allok

# --- solver A: basis vectors and rhat6_on_chart given per entry
A = json.loads((ROOT/'solver_a'/'results.json').read_text())
entA = []
for k, v in A['item2'].items():
    expr = v['rhat6_on_chart'].replace('^', '**')
    # "(NUM) / (1 + x^2 + y^2)^2"  ->  a sympy expression
    entA.append((k, json.loads(v['basis_a'].replace('sqrt(2)/2', '"sqrt(2)/2"')) if 'sqrt' in v['basis_a'] else json.loads(v['basis_a']),
                 json.loads(v['basis_b'].replace('sqrt(2)/2', '"sqrt(2)/2"').replace('-"sqrt(2)/2"', '"-sqrt(2)/2"')) if 'sqrt' in v['basis_b'] else json.loads(v['basis_b']),
                 expr))
okA = check_room('solver_a, its own chart', entA)

# --- solver B: chart string "u = ([..]) + z ([..])"
import re
B = json.loads((ROOT/'solver_b'/'results.json').read_text())
entB = []
for k, v in B['item2'].items():
    m = re.findall(r'\[([^\]]*)\]', v['chart'])
    a_list = [t.strip() for t in m[0].split(',')]
    b_list = [t.strip() for t in m[1].split(',')]
    entB.append((k, a_list, b_list, v['restriction']))
okB = check_room('solver_b, its own chart', entB)

print(f"\nsolver_a {'ALL PASS' if okA else 'HAS A FAILURE'};  solver_b {'ALL PASS' if okB else 'HAS A FAILURE'}")
