import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sympy as sp, common as C
x, y = sp.symbols('x y', real=True)
I = sp.I
def vec(d):
    c = [sp.Integer(0)] * 7
    for m, a in d.items(): c[C.idx(m)] += sp.sympify(a)
    return c
s2 = sp.sqrt(2)
classes = {
    'A C3 {v1,v-2}': ({1: 1}, {-2: 1}),
    'B C4 {v1,v-3}': ({1: 1}, {-3: 1}),
    'C C4 {v2,v-2}': ({2: 1}, {-2: 1}),
    'D C5 {v2,v-3}': ({2: 1}, {-3: 1}),
    'E C6 {v3,v-3}': ({3: 1}, {-3: 1}),
    'F D2 {v0,(v2+v-2)/rt2}': ({0: 1}, {2: 1 / s2, -2: 1 / s2}),
    'G D3 {v0,(v3-v-3)/rt2}': ({0: 1}, {3: 1 / s2, -3: -1 / s2}),
}
for name, (a, b) in classes.items():
    ca = vec(a); cb = vec(b)
    c = [ca[k] + (x + I * y) * cb[k] for k in range(7)]
    N = C.N_exact(c)
    n2 = sp.expand(C.norm2(c))
    f = sp.factor(sp.nsimplify(N))
    print(name, '\n  N =', f, '\n  |u|^2 =', sp.factor(n2))
