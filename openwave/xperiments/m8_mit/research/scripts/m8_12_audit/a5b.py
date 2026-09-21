"""Audit item 5b: tangents of the plane at interior orbit points, in solver A's basis (iu, u_perp, i u_perp)
and in solver B's basis (iu, d_x uhat, d_y uhat).  Exact."""
import sympy as sp
from aud_core import *
from aud_acoords import form_at_c, realv, c_to_a

x, y = sp.symbols('x y', real=True)
def v(m):
    c = [sp.Integer(0)] * 7; c[idx(m)] = sp.Integer(1); return c
def lin(*pairs):
    out = [sp.Integer(0)] * 7
    for s, w in pairs:
        out = [o + s * ww for o, ww in zip(out, w)]
    return [sp.expand(o) for o in out]
def rip(p, q): return sp.radsimp(sp.expand(sum(sp.re(sp.expand(sp.conjugate(a) * b)) for a, b in zip(p, q))))
def hip(p, q): return sp.expand(sum(sp.conjugate(a) * b for a, b in zip(p, q)))
r2 = sp.sqrt(2)
CASES = [  # label, a, b (b possibly non-unit), z0, whose chart
    ('A* in A (A chart)', v(1), v(-2), sp.Rational(1, 2)),
    ('D* in D (A chart)', v(2), v(-3), 2 * sp.sqrt(39) / 13),
    ('F* in F (A chart)', v(0), lin((1 / r2, v(2)), (1 / r2, v(-2))), I * sp.sqrt(sp.Rational(5, 3))),
    ('G* in G (A chart)', v(0), lin((1 / r2, v(3)), (-1 / r2, v(-3))), 2 * I * sp.sqrt(115) / 23),
    ('xyz in C (A chart)', v(2), v(-2), sp.Integer(1)),
    ('cat in E (A chart)', v(3), v(-3), sp.Integer(1)),
    ('xyz in G (A chart)', v(0), lin((1 / r2, v(3)), (-1 / r2, v(-3))), 2 / sp.sqrt(5)),
    ('cat in F (A chart)', v(0), lin((1 / r2, v(2)), (1 / r2, v(-2))), sp.sqrt(sp.Rational(3, 5))),
    ('Wmin in W (B chart)', v(0), lin((1, v(3)), (1, v(-3))), sp.sqrt(230) / 23),
    ('Umin in U (B chart)', v(0), lin((1, v(2)), (1, v(-2))), I * sp.sqrt(30) / 6),
    ('L12circle (B chart)', v(1), v(-2), sp.Rational(1, 2)),
    ('L23circle (B chart)', v(2), v(-3), sp.sqrt(sp.Rational(12, 13))),
    ('hex in L33 (B chart)', v(3), v(-3), sp.Integer(1)),
    ('hex in U (B chart)', v(0), lin((1, v(2)), (1, v(-2))), sp.sqrt(30) / 10),
    ('oct in L22 (B chart)', v(2), v(-2), sp.Integer(-1)),
    ('oct in W (B chart)', v(0), lin((1, v(3)), (1, v(-3))), I * sp.sqrt(10) / 5),
]
def proj_O(u, t):
    """coefficients of the Re<,>-orthogonal projection of t onto O_u (w.r.t. the 4 generators, via a maximal
    independent subset) and the residual (projection onto N_u, since t is in T_u)."""
    gens = O_gens(u)
    Gm = sp.Matrix(4, 4, lambda i, j: rip(gens[i], gens[j]))
    # independent subset
    keep = []
    for k in range(4):
        sub = keep + [k]
        if Gm.extract(sub, sub).det() != 0: keep = sub
    Gs = Gm.extract(keep, keep)
    rhs = sp.Matrix([rip(gens[k], t) for k in keep])
    coef = (Gs.inv() * rhs).applyfunc(sp.radsimp)
    pr = lin(*[(coef[i], gens[k]) for i, k in enumerate(keep)])
    resid = [sp.expand(a - b) for a, b in zip(t, pr)]
    return dict(zip(keep, coef)), resid
names = ['iu', '-iJx u', '-iJy u', '-iJz u']
for label, a, b, z0 in CASES:
    u0 = lin((1, a), (z0, b))
    nn = sp.sqrt(rip(u0, u0))
    u = [sp.radsimp(t / nn) for t in u0]
    Mf, _, _, _ = form_at_c(u)
    def H(p, q): return sp.radsimp(sp.expand((realv(c_to_a(p)).T * Mf * realv(c_to_a(q)))[0]))
    if 'A chart' in label:
        tt = sp.Rational(1) + z0 * sp.conjugate(z0)
        up = [sp.radsimp(t / sp.sqrt(tt)) for t in lin((-sp.conjugate(z0), a), (1, b))]
        tang = [('iu', [I * t for t in u]), ('u_perp', up), ('i u_perp', [I * t for t in up])]
    else:
        zz = x + I * y
        U = lin((1, a), (zz, b))
        nU = sp.sqrt(sum(sp.expand(t * sp.conjugate(t)) for t in U))
        uh = [t / nU for t in U]
        sub = {x: sp.re(z0), y: sp.im(z0)}
        dx = [sp.radsimp(sp.simplify(sp.diff(t, x).subs(sub))) for t in uh]
        dy = [sp.radsimp(sp.simplify(sp.diff(t, y).subs(sub))) for t in uh]
        tang = [('iu', [I * t for t in u]), ('d_x uhat', dx), ('d_y uhat', dy)]
    print('=' * 80); print(label, '  rhat6 =', sp.simplify(sp.expand(N_exact(u))))
    nonnull = []
    for nm, t in tang:
        assert rip(u, t) == 0, (label, nm, 'not in T_u')
        coef, resid = proj_O(u, t)
        n2t = rip(t, t); n2r = rip(resid, resid)
        if n2r == 0:
            print('  %-9s projection onto N_u = 0 ; t = %s ; H(t/|t|) = %s' % (
                nm, ' + '.join('(%s)[%s]' % (c, names[k]) for k, c in coef.items()), sp.radsimp(H(t, t) / n2t)))
        else:
            ratio = sp.radsimp(n2r / n2t)
            print('  %-9s |P_N t|^2/|t|^2 = %s ; H(unit unprojected) = %s ; H(unit projection) = %s' % (
                nm, ratio, sp.radsimp(H(t, t) / n2t), sp.radsimp(H(resid, resid) / n2r)))
            nonnull.append((nm, t, resid))
    if len(nonnull) == 2:
        (n1, t1, r1), (n2, t2, r2_) = nonnull
        print('  off-diagonal H(unit t1, unit t2) =', sp.radsimp(H(t1, t2) / sp.sqrt(rip(t1, t1) * rip(t2, t2))),
              ' Gram off-diag Re<t1,t2>/(|t1||t2|) =', sp.radsimp(rip(t1, t2) / sp.sqrt(rip(t1, t1) * rip(t2, t2))))
