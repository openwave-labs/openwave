"""Audit item 2 and the span{v3, v-1} locus: exact restrictions and critical sets by elimination."""
import sympy as sp
from aud_core import *

x, y = sp.symbols('x y', real=True)
z = x + I * y
t = sp.Symbol('t', nonnegative=True)
def v(m):
    c = [sp.Integer(0)] * 7; c[idx(m)] = sp.Integer(1); return c
def add(p, q, s=1): return [a + s * b for a, b in zip(p, q)]
def scal(p, s): return [s * a for a in p]
r2 = sp.sqrt(2)
planes = {
    'A span{v1,v-2}': (v(1), v(-2)),
    'B span{v1,v-3}': (v(1), v(-3)),
    'Bswap span{v3,v-1}': (v(3), v(-1)),
    'C span{v2,v-2}': (v(2), v(-2)),
    'D span{v2,v-3}': (v(2), v(-3)),
    'E span{v3,v-3}': (v(3), v(-3)),
    'F span{v0,(v2+v-2)/r2}': (v(0), scal(add(v(2), v(-2)), 1 / r2)),
    'G span{v0,(v3-v-3)/r2}': (v(0), scal(add(v(3), v(-3), -1), 1 / r2)),
}
out = {}
for name, (a, b) in planes.items():
    u = add(a, scal(b, z))
    Nz = sp.expand(N_exact(u))
    Nz = sp.expand(sp.re(Nz))  # N is real; keep the real part exactly (imag part checked below)
    assert sp.expand(sp.im(sp.expand(N_exact(u)))) == 0
    den = (1 + x ** 2 + y ** 2) ** 2
    f = Nz / den
    # numerators of the gradient
    px = sp.factor(sp.numer(sp.together(sp.diff(f, x))))
    py = sp.factor(sp.numer(sp.together(sp.diff(f, y))))
    print('=' * 70); print(name)
    print('  924 N(a+zb) =', sp.factor(924 * Nz))
    radial = sp.expand(924 * Nz - (924 * Nz).subs({x: sp.sqrt(x ** 2 + y ** 2), y: 0})) == 0
    print('  radial (function of |z|^2 only):', radial)
    if radial:
        Nt = sp.expand((924 * Nz).subs({x: sp.sqrt(t), y: 0}))
        F = Nt / (924 * (1 + t) ** 2)
        Fp = sp.factor(sp.diff(F, t))
        print('  924 N as poly in t:', Nt, '   F\'(t) =', Fp)
        print('  Taylor of rhat6 at z=0 in t:', sp.series(F, t, 0, 3))
        # the other pole: w = 1/z  ->  N(b + w a) = |w|^4 N(a + b/w)... compute directly
        u2 = add(b, scal(a, z))
        N2 = sp.expand((924 * sp.re(sp.expand(N_exact(u2)))).subs({x: sp.sqrt(t), y: 0}))
        F2 = N2 / (924 * (1 + t) ** 2)
        print('  other chart 924 N(b + w a) =', N2, '  Taylor:', sp.series(F2, t, 0, 3))
        roots = [r for r in sp.solve(sp.numer(sp.together(Fp)), t) if r.is_positive]
        print('  interior critical |z|^2 (t>0 roots of F\' numerator):', roots, [F.subs(t, r) for r in roots])
    else:
        print('  d_x numerator:', px)
        print('  d_y numerator:', py)
        G = sp.groebner([sp.numer(sp.together(sp.diff(f, x))), sp.numer(sp.together(sp.diff(f, y)))], x, y, order='lex')
        print('  lex Groebner basis:', [sp.factor(g) for g in G.exprs])
        sols = sp.solve(G.exprs, [x, y], dict=True)
        real = [s for s in sols if all(val.is_real for val in s.values())]
        for s in real:
            print('   critical z =', sp.simplify(s[x] + I * s[y]), ' rhat6 =', sp.simplify(f.subs(s)))
        u2 = add(b, scal(a, z))
        N2 = sp.expand(sp.re(sp.expand(N_exact(u2))))
        f2 = N2 / den
        g2 = [sp.simplify(sp.diff(f2, s_).subs({x: 0, y: 0})) for s_ in (x, y)]
        print('  omitted point [b]: rhat6 =', sp.simplify(f2.subs({x: 0, y: 0})), ' gradient in swapped chart =', g2)
