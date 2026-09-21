import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sympy as sp, common as C
I = sp.I; s2 = sp.sqrt(2)
def vec(d):
    c = [sp.Integer(0)] * 7
    for m, a in d.items(): c[C.idx(m)] += sp.sympify(a)
    return c
pts = {
 'v3': vec({3: 1}), 'v2': vec({2: 1}), 'v1': vec({1: 1}), 'v0': vec({0: 1}),
 'xyz': vec({2: 1, -2: -1}), 'cat': vec({3: 1, -3: 1}),
 'A t=1/4': vec({1: 1, -2: sp.Rational(1, 2)}),
 'D t=12/13': vec({2: 1, -3: sp.sqrt(sp.Rational(12, 13))}),
 'F y': vec({0: 1, 2: I * sp.sqrt(sp.Rational(5, 3)) / s2, -2: I * sp.sqrt(sp.Rational(5, 3)) / s2}),
 'F x': vec({0: 1, 2: sp.sqrt(sp.Rational(3, 5)) / s2, -2: sp.sqrt(sp.Rational(3, 5)) / s2}),
 'G y': vec({0: 1, 3: I * sp.sqrt(sp.Rational(20, 23)) / s2, -3: -I * sp.sqrt(sp.Rational(20, 23)) / s2}),
 'G x': vec({0: 1, 3: sp.sqrt(sp.Rational(4, 5)) / s2, -3: -sp.sqrt(sp.Rational(4, 5)) / s2}),
}
for k, c in pts.items():
    print(k, [str(C.rhat_exact(c, L)) for L in range(1, 7)])
