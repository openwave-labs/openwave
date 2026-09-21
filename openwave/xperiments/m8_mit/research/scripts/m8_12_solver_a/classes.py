"""The classes of item 1 and the exact analysis of rhat6 restricted to the planes (item 2)."""
import sympy as sp
import common as C

I = sp.I
s2 = sp.sqrt(2)

def vec(d):
    c = [sp.Integer(0)] * 7
    for m, a in d.items(): c[C.idx(m)] += sp.sympify(a)
    return sp.Matrix(c)

# planes: (a, b) orthonormal; chart u = a + z b, z = x + i y, omits the point [b]
CLASSES2 = {
    'A': dict(H='C3', chi='R_z(2pi/3) -> e^{-2pi i/3}', a=vec({1: 1}), b=vec({-2: 1})),
    'B': dict(H='C4', chi='R_z(pi/2) -> -i', a=vec({1: 1}), b=vec({-3: 1})),
    'C': dict(H='C4', chi='R_z(pi/2) -> -1', a=vec({2: 1}), b=vec({-2: 1})),
    'D': dict(H='C5', chi='R_z(2pi/5) -> e^{-4pi i/5}', a=vec({2: 1}), b=vec({-3: 1})),
    'E': dict(H='C6', chi='R_z(pi/3) -> -1', a=vec({3: 1}), b=vec({-3: 1})),
    'F': dict(H='D2', chi='R_z(pi) -> 1, R_y(pi) -> -1, R_x(pi) -> -1', a=vec({0: 1}), b=vec({2: 1 / s2, -2: 1 / s2})),
    'G': dict(H='D3', chi='R_z(2pi/3) -> 1, R_y(pi) -> -1', a=vec({0: 1}), b=vec({3: 1 / s2, -3: -1 / s2})),
}
CLASSES1 = {
    'v3': dict(H='SO(2)_z', chi='R_z(t) -> e^{-3it}', u=vec({3: 1})),
    'v2': dict(H='SO(2)_z', chi='R_z(t) -> e^{-2it}', u=vec({2: 1})),
    'v1': dict(H='SO(2)_z', chi='R_z(t) -> e^{-it}', u=vec({1: 1})),
    'v0': dict(H='O(2)_z', chi='R_z(t) -> 1, every flip -> -1', u=vec({0: 1})),
    'xyz': dict(H='O', chi='A2 (sign) character of O; T acts trivially', u=vec({2: 1 / s2, -2: -1 / s2})),
    'cat': dict(H='D6', chi='R_z(pi/3) -> -1, R_y(pi) -> +1', u=vec({3: 1 / s2, -3: 1 / s2})),
}

x, y, t = sp.symbols('x y t', real=True)

def chart_vector(nm, swap=False):
    d = CLASSES2[nm]
    a, b = (d['b'], d['a']) if swap else (d['a'], d['b'])
    return [a[k] + (x + I * y) * b[k] for k in range(7)]

def restriction(nm, swap=False):
    """exact rhat6 on the chart, as (numerator polynomial, denominator (1+x^2+y^2)^2)."""
    c = chart_vector(nm, swap)
    Nn = sp.expand(C.N_exact(c))
    n2 = sp.expand(C.norm2(c))
    assert sp.expand(n2 - (1 + x ** 2 + y ** 2)) == 0
    return Nn, n2 ** 2

def is_circle_symmetric(Nn):
    return sp.expand(x * sp.diff(Nn, y) - y * sp.diff(Nn, x)) == 0

def critical_set(nm):
    """Complete critical set of f = Nn/(1+x^2+y^2)^2 on P(W) = chart + the omitted point.
    Returns list of dicts; method is pure elimination (Groebner basis, lex order)."""
    Nn, Dn = restriction(nm)
    f = Nn / Dn
    fx = sp.factor(sp.numer(sp.together(sp.diff(f, x))))
    fy = sp.factor(sp.numer(sp.together(sp.diff(f, y))))
    out = []
    if is_circle_symmetric(Nn):
        g = sp.expand(Nn.subs({x: sp.sqrt(t), y: 0}))       # Nn = g(t), t = x^2+y^2
        ft = g / (1 + t) ** 2
        num = sp.factor(sp.numer(sp.together(sp.diff(ft, t))))
        # grad f = 2 f'(t) (x, y): critical iff (x,y)=0 or f'(t)=0 with t>0
        out.append(dict(where='z=0', t=sp.Integer(0), value=sp.nsimplify(ft.subs(t, 0)), kind='point'))
        for r in sp.Poly(num, t).all_roots() if sp.Poly(num, t).degree() > 0 else []:
            if r.is_real and r > 0:
                out.append(dict(where='|z|^2=%s' % r, t=r, value=sp.nsimplify(ft.subs(t, r)), kind='circle'))
        info = dict(symmetric=True, g_of_t=g, dfdt_numerator=num)
    else:
        G = sp.groebner([fx, fy], x, y, order='lex')
        sols = sp.solve(list(G), [x, y], dict=True)
        for s in sols:
            if all(sp.im(v) == 0 for v in s.values()):
                out.append(dict(where='z=%s' % sp.nsimplify(s[x] + I * s[y]), x=s[x], y=s[y],
                                value=sp.nsimplify(sp.radsimp(f.subs(s))), kind='point'))
        info = dict(symmetric=False, fx=fx, fy=fy, groebner=list(G), n_complex_solutions=len(sols))
    # the omitted point [b]: chart w with u = b + w a, check grad at w = 0
    Nw, Dw = restriction(nm, swap=True)
    fw = Nw / Dw
    gx = sp.diff(fw, x).subs({x: 0, y: 0}); gy = sp.diff(fw, y).subs({x: 0, y: 0})
    out.append(dict(where='omitted point [b]', value=sp.nsimplify(fw.subs({x: 0, y: 0})),
                    grad_at_omitted=(sp.simplify(gx), sp.simplify(gy)), kind='point'))
    info['f'] = f
    return out, info
