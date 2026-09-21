"""Item 2: rhat6 restricted to each dimension-2 class, and its complete critical set.

Chart: u = a + z b (a, b orthonormal basis of W from classes.py), z = x + i y; omits [b].
Elimination: (i) circle-symmetric classes (A-E): rhat6 = g(t)/(1+t)^2 with t = |z|^2, grad = 2 f'(t)(x, y);
(ii) F, G: the gradient numerators factor as x*P1, y*Q1, and P1 - Q1 is a nonzero multiple of (1+t),
so critical points lie on the axes; each axis gives one linear equation in t.  The same answer is
re-derived by a lex Groebner basis.  Independent checks: every critical point has exactly zero
gradient; Poincare-Hopf index sum; 60-digit Newton search from a grid of seeds.
"""
import itertools
import sympy as sp, mpmath as mp
import common as C, classes as K

ck = C.Checker('item2')
x, y, t = K.x, K.y, K.t
I = sp.I
res = {}
for nm, d in K.CLASSES2.items():
    Nn, Dn = K.restriction(nm)
    f = Nn / Dn
    crit, info = K.critical_set(nm)
    out = {'basis_a': str(list(d['a'])), 'basis_b': str(list(d['b'])),
           'chart': 'u = a + z b, z = x + i y; omitted point [b]',
           'rhat6_on_chart': '(%s) / (1 + x^2 + y^2)^2' % sp.factor(Nn)}
    print('\n=== class %s  (%s, %s)' % (nm, d['H'], d['chi']))
    print('  rhat6 = (%s)/(1+x^2+y^2)^2' % sp.factor(Nn))
    if info['symmetric']:
        print('  circle-symmetric: 924*N = %s (t = x^2+y^2);  numerator of df/dt: %s' % (sp.expand(924 * info['g_of_t']), info['dfdt_numerator']))
        out['924_N_of_t'] = str(sp.expand(924 * info['g_of_t']))
        out['dfdt_numerator'] = str(info['dfdt_numerator'])
    else:
        fx, fy = info['fx'], info['fy']
        print('  numer d/dx:', fx); print('  numer d/dy:', fy)
        # hand elimination: fx = x*P1*c, fy = y*Q1*c'
        P1 = sp.cancel(fx / x); Q1 = sp.cancel(fy / y)
        ck.check('%s: x divides numer(f_x) and y divides numer(f_y)' % nm,
                 sp.expand(fx.subs(x, 0)) == 0 and sp.expand(fy.subs(y, 0)) == 0)
        # normalize to the gradient of Nn/(1+t)^2 up to the same positive factor
        gx = sp.expand(sp.diff(Nn, x) * (1 + x ** 2 + y ** 2) - 4 * x * Nn)
        gy = sp.expand(sp.diff(Nn, y) * (1 + x ** 2 + y ** 2) - 4 * y * Nn)
        p1 = sp.expand(sp.cancel(gx / x)); q1 = sp.expand(sp.cancel(gy / y))
        diff = sp.factor(p1 - q1)
        print('  (1+t)^3 f_x = x*p1, (1+t)^3 f_y = y*q1,  p1 - q1 =', diff)
        ck.check('%s: p1 - q1 is a nonzero constant times (1 + x^2 + y^2)' % nm,
                 sp.simplify(diff / (1 + x ** 2 + y ** 2)).is_number and diff != 0)
        out['elimination'] = {'p1': str(p1), 'q1': str(q1), 'p1_minus_q1': str(diff),
                              'groebner_lex': [str(g) for g in info['groebner']],
                              'n_complex_solutions_of_groebner_system': info['n_complex_solutions']}
        # axes: x=0 -> q1(0,y)=0 ; y=0 -> p1(x,0)=0
        ax_y = sp.factor(q1.subs(x, 0)); ax_x = sp.factor(p1.subs(y, 0))
        print('  on x=0: q1 =', ax_y, ';  on y=0: p1 =', ax_x)
        out['axis_equations'] = {'x=0': str(ax_y), 'y=0': str(ax_x)}
        print('  Groebner basis (lex x>y):', info['groebner'])
    # list, verify exact zero gradient and classify
    rows = []
    fxx = sp.diff(f, x, 2); fyy = sp.diff(f, y, 2); fxy = sp.diff(f, x, y)
    fX = sp.diff(f, x); fY = sp.diff(f, y)
    idx_sum = 0; degenerate = False
    for e in crit:
        row = {'where': e['where'], 'value': str(e['value']), 'kind': e['kind']}
        if e['where'] == 'omitted point [b]':
            g = e['grad_at_omitted']
            ck.check('%s: gradient at the omitted point is exactly zero' % nm, g == (0, 0), str(g))
            Nw, Dw = K.restriction(nm, swap=True); fw = Nw / Dw
            H = sp.Matrix([[sp.diff(fw, x, 2), sp.diff(fw, x, y)], [sp.diff(fw, x, y), sp.diff(fw, y, 2)]]).subs({x: 0, y: 0})
            pts = [(0, 0)]
        elif e['kind'] == 'circle':
            r = sp.sqrt(e['t'])
            pts = [(r, 0), (0, r), (r / sp.sqrt(2), r / sp.sqrt(2))]
            H = None
        else:
            if 'x' in e: pts = [(e['x'], e['y'])]
            else: pts = [(0, 0)]
            H = sp.Matrix([[fxx, fxy], [fxy, fyy]]).subs({x: pts[0][0], y: pts[0][1]})
        if e['where'] != 'omitted point [b]':
            g = [sp.simplify(fX.subs({x: p[0], y: p[1]})) for p in pts] + [sp.simplify(fY.subs({x: p[0], y: p[1]})) for p in pts]
            ck.check('%s: exact zero gradient at %s' % (nm, e['where']), all(v == 0 for v in g))
            row['value_check'] = str(sp.nsimplify(sp.radsimp(f.subs({x: pts[0][0], y: pts[0][1]}))))
            ck.check('%s: value at %s re-evaluated' % (nm, e['where']), sp.nsimplify(sp.radsimp(f.subs({x: pts[0][0], y: pts[0][1]}))) == e['value'])
        if H is not None:
            H = H.applyfunc(sp.simplify)
            det = sp.simplify(H.det()); tr = sp.simplify(H.trace())
            if det > 0: kind = 'local max' if tr < 0 else 'local min'; ind = 1
            elif det < 0: kind = 'saddle'; ind = -1
            elif info['symmetric']:
                # pole of a circle-symmetric class: f = F(t), t = |w|^2 in the relevant chart;
                # if F(t) - F(0) = alpha t^k + ..., grad = 2 F'(t)(x, y) is radial, index +1
                Nc, Dc = K.restriction(nm, swap=(e['where'] == 'omitted point [b]'))
                Ft = sp.expand(Nc.subs({x: sp.sqrt(t), y: 0})) / (1 + t) ** 2
                ser = sp.series(Ft - Ft.subs(t, 0), t, 0, 4).removeO()
                kk = min(sp.Poly(ser, t).monoms())[0]; alpha = sp.Poly(ser, t).coeff_monomial(t ** kk)
                kind = 'degenerate (Hessian 0); F(t)-F(0) = (%s) t^%d, isolated local %s' % (alpha, kk, 'max' if alpha < 0 else 'min')
                ind = 1
                row['leading_order'] = '(%s) |z|^%d' % (alpha, 2 * kk)
            else: kind = 'degenerate'; ind = None; degenerate = True
            row.update({'hess_det': str(det), 'hess_trace': str(tr), 'type': kind, 'index': ind})
            idx_sum += ind if ind is not None else 0
        else:
            # circle: f(t) with f'(t*) = 0; second derivative in t decides max/min in the normal direction
            g_t = info['g_of_t']; ft = g_t / (1 + t) ** 2
            f2 = sp.simplify(sp.diff(ft, t, 2).subs(t, e['t']))
            row.update({'d2f_dt2': str(f2), 'type': 'circle of maxima (Morse-Bott)' if f2 < 0 else 'circle of minima (Morse-Bott)', 'index': 'circle, Euler characteristic 0'})
        rows.append(row)
        print('  crit:', row)
    # Poincare-Hopf on S^2: sum of indices of grad f over isolated points = 2, circles contribute chi(S^1)=0
    ck.check('%s: Poincare-Hopf index sum over isolated critical points == 2' % nm, (not degenerate) and idx_sum == 2, 'sum=%s' % idx_sum)
    out['critical_set'] = rows
    out['index_sum'] = idx_sum
    # independent numerical search: Newton on grad from a grid of seeds in both charts at 60 digits
    mp.mp.dps = 60
    found = set()
    for chart in (False, True):
        Nc, Dc = K.restriction(nm, swap=chart); fc = Nc / Dc
        gX = sp.lambdify((x, y), sp.diff(fc, x), 'mpmath'); gY = sp.lambdify((x, y), sp.diff(fc, y), 'mpmath')
        for sx, sy in itertools.product([-1.7, -0.9, -0.35, 0.05, 0.4, 1.1, 1.8], repeat=2):
            try:
                sol = mp.findroot([gX, gY], (mp.mpf(sx), mp.mpf(sy)), tol=mp.mpf(10) ** -50, maxsteps=200)
            except Exception:
                continue
            X, Y = sol[0], sol[1]
            if abs(X) > 3 or abs(Y) > 3: continue
            if max(abs(gX(X, Y)), abs(gY(X, Y))) > mp.mpf(10) ** -40: continue
            w2 = X ** 2 + Y ** 2
            if chart:   # u = b + w a  ~  a + (1/w) b, so |z|^2 = 1/|w|^2
                key = 'inf' if w2 < mp.mpf(10) ** -30 else mp.nstr(1 / w2, 12)
            else:
                key = '0.0' if w2 < mp.mpf(10) ** -30 else mp.nstr(w2, 12)
            found.add(key)
    exact_t = set()
    for e in crit:
        if e['where'] == 'omitted point [b]': exact_t.add('inf')
        elif e['kind'] == 'circle': exact_t.add(mp.nstr(mp.mpf(sp.Rational(e['t']).p) / sp.Rational(e['t']).q, 12))
        else:
            tt = sp.nsimplify(e['x'] ** 2 + e['y'] ** 2) if 'x' in e else sp.Integer(0)
            exact_t.add(mp.nstr(mp.mpf(tt.p) / tt.q, 12) if tt != 0 else '0.0')
    print('  numerical Newton search found |z|^2 in', sorted(found, key=str), ' exact:', sorted(exact_t, key=str))
    ck.check('%s: Newton search found no |z|^2 outside the exact critical set' % nm, found <= exact_t, str(found - exact_t))
    out['newton_search_abs_z2_found'] = sorted(found, key=str)
    res[nm] = out
C.save('item2', res)
print('item2 fails:', ck.fails)
