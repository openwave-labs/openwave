import json, itertools
import sympy as sp
import mpmath as mp

s5 = sp.sqrt(5)
phi = (1 + s5) / 2
qx, qy, qz, a, x = sp.symbols('q_x q_y q_z a x')
Q = sp.Matrix([qx, qy, qz])
q2 = qx**2 + qy**2 + qz**2

# Twelve vectors (0,+-1,+-phi), (+-1,+-phi,0), (+-phi,0,+-1); all have |v|^2 = 2+phi.
raw = []
for s1, s2 in itertools.product([1, -1], [1, -1]):
    raw += [(0, s1, s2 * phi), (s1, s2 * phi, 0), (s2 * phi, 0, s1)]
assert len(set(raw)) == 12
N2 = sp.expand(2 + phi)
for v in raw:
    assert sp.simplify(v[0]**2 + v[1]**2 + v[2]**2 - N2) == 0
V = [sp.Matrix(v) for v in raw]


def clean(e):
    return sp.expand(sp.radsimp(sp.expand(e)))


def poly_clean(e):
    P = sp.Poly(sp.expand(e), qx, qy, qz, a)
    return sum(clean(c) * qx**i * qy**j * qz**k * a**l for (i, j, k, l), c in P.terms())


def S(m, vecs=None, w=None):
    """sum_u w_u (u.q)^(2m), exact, using (u.q)^2 = (v.q)^2/|v|^2 (only even powers occur)."""
    vecs = V if vecs is None else vecs
    w = [1] * len(vecs) if w is None else w
    return sum(wi * sp.expand(v.dot(Q)**(2 * m)) / N2**m for wi, v in zip(w, vecs))


# omega^2 = sum_{m>=1} (-1)^(m+1) a^(2m-2) S(m) / (2 (2m)!)
def coef(m):
    return sp.Rational((-1)**(m + 1), 2 * sp.factorial(2 * m)) * a**(2 * m - 2)


P6 = sp.Poly(sp.legendre(6, x), x)
u0 = V[0]
# S6 = sum_u P6(u.u0) (P6 even, so only (u.u0)^2 needed)
S6 = clean(sum(sum(c * ((v.dot(u0))**2 / N2**2)**(i // 2) for (i,), c in P6.terms()) for v in V))
# I6 as a homogeneous degree-6 polynomial in q: |q|^6 * sum_u P6(u.q/|q|) / S6
I6q = sp.expand(sum(c * S(i // 2) * q2**(3 - i // 2) for (i,), c in P6.terms()) / S6)
I6q = poly_clean(I6q)

res = {'S6_sum_P6_at_vertex': str(S6), 'I6_polynomial_in_q (= |q|^6 I6(n))': str(I6q)}
rems = {}
for m, deg in [(1, 2), (2, 4), (3, 6)]:
    poly = poly_clean(coef(m) * S(m))
    # invariant part: c (q.q)^m with c fixed by the sphere average: avg (u.n)^(2m) = 1/(2m+1)
    inv_c = sp.Rational(12, 2 * m + 1) * coef(m)
    inv = poly_clean(inv_c * q2**m)
    rem = poly_clean(poly - inv)
    d = {'polynomial': str(poly), 'invariant_coefficient': str(inv_c),
         'invariant_part': str(inv), 'remainder': str(rem),
         'a_power': 2 * m - 2}
    if rem == 0:
        d['remainder_coefficient_in_I6'] = '0'
        d['ratio'] = '0'
    else:
        r0 = sp.expand(rem / a**(2 * m - 2))
        kap = clean(sp.Poly(r0, qx, qy, qz).coeff_monomial(qz**6) /
                    sp.Poly(I6q, qx, qy, qz).coeff_monomial(qz**6))
        d['remainder_exactly_kappa_times_I6poly'] = bool(poly_clean(r0 - kap * I6q) == 0)
        d['remainder_coefficient_in_I6'] = str(sp.simplify(kap * a**(2 * m - 2)))
        d['remainder_form'] = 'remainder = remainder_coefficient_in_I6 * |q|^6 * I6(n)'
        d['ratio'] = str(sp.simplify(kap * a**(2 * m - 2) / inv_c))
        # independent route: x^6 = 1/7 + (l=2,4 parts) + (16/231) P6(x); l=2,4 sums vanish
        route2 = sp.simplify(coef(m) * sp.Rational(16, 231) * S6)
        d['route2_coefficient'] = str(route2)
        d['routes_agree'] = bool(sp.simplify(route2 - kap * a**(2 * m - 2)) == 0)
    res['degree_%d' % deg] = d
    rems[deg] = rem

res['first_nonzero_remainder_degree'] = next(dg for dg in (2, 4, 6) if rems[dg] != 0)
scal = {}
for dg in (2, 4, 6):
    p = sp.sympify(res['degree_%d' % dg]['polynomial'])
    scal['degree_%d' % dg] = str(sp.simplify(p.subs(a, a / 2) / p))
res['scaling_a_to_half'] = {
    'factors_polynomial_ratio': scal,
    'note': 'a -> a/2 multiplies the degree-2m coefficient by 2^-(2m-2): 1, 1/4, 1/16. '
            'Invariant coefficient, remainder coefficient and the ratio-defining pieces scale alike; ratio unchanged.'}

# ---------------- numerical check ----------------
# Bound fixed before running: Taylor with Lagrange remainder for 1-cos t after t^6 term gives
# |1-cos t - (t^2/2 - t^4/24 + t^6/720)| <= t^8/8!.  With t = a k (u.n), |u.n|<=1:
# |omega^2 - series_deg<=6| <= 12 (a k)^8 / (2 a^2 8!).
mp.mp.dps = 80
mp.mp.dps = 80


def bound(a_, k_):
    return 12 * (a_ * k_)**8 / (2 * a_**2 * mp.factorial(8))


ph = (1 + mp.sqrt(5)) / 2
Um = []
for v in [(0, s1, s2 * ph) for s1 in (1, -1) for s2 in (1, -1)] + \
         [(s1, s2 * ph, 0) for s1 in (1, -1) for s2 in (1, -1)] + \
         [(s2 * ph, 0, s1) for s1 in (1, -1) for s2 in (1, -1)]:
    nv = mp.sqrt(sum(c * c for c in v))
    Um.append([c / nv for c in v])


def unit(v):
    v = [mp.mpf(c) for c in v]
    nv = mp.sqrt(sum(c * c for c in v))
    return [c / nv for c in v]


dirs = {'u_vertex': Um[0], 'axis_z': unit([0, 0, 1]), 'generic_1': unit([1, 2, 3]),
        'generic_2': unit([3, -1, 2]), 'generic_3': unit([1, 1, -5])}
polys = [sp.lambdify((qx, qy, qz, a), sp.sympify(res['degree_%d' % dg]['polynomial']), 'mpmath')
         for dg in (2, 4, 6)]
rows = []
for name, n in dirs.items():
    for den in (4, 8, 16, 32):
        k_ = mp.mpf(1)
        a_ = mp.mpf(1) / den
        om = sum((1 - mp.cos(a_ * k_ * sum(ui * ni for ui, ni in zip(u, n)))) / (2 * a_ * a_) for u in Um)
        ser = sum(p(k_ * n[0], k_ * n[1], k_ * n[2], a_) for p in polys)
        diff = abs(om - ser)
        b = bound(a_, k_)
        rows.append({'direction': name, 'a*k': '1/%d' % den, 'difference': mp.nstr(diff, 15),
                     'bound': mp.nstr(b, 15), 'inside': bool(diff <= b)})
res['numerical_check'] = rows
res['numerical_check_all_inside'] = all(r['inside'] for r in rows)

# ---------------- control ----------------
# Test (fixed before running): degree-2 polynomial is q^T M q / 4 with M = sum_u w_u u u^T.
# It is rotationally invariant iff D = M - (tr M / 3) I is exactly the zero matrix.


def test(vecs, w):
    M = sp.zeros(3, 3)
    for wi, v in zip(w, vecs):
        M += wi * v * v.T / N2
    D = (M - sp.eye(3) * M.trace() / 3).applyfunc(clean)
    return {'invariant': bool(D == sp.zeros(3, 3)), 'anisotropy_matrix': str(D.tolist())}


res['control'] = {
    'test': 'M = sum_u w_u u u^T; degree-2 polynomial q^T M q/4 is rotationally invariant iff M - (tr M/3) I == 0 exactly',
    'original': test(V, [1] * 12),
    'one_direction_removed': test(V[1:], [1] * 11),
    'one_weight_doubled': test(V, [2] + [1] * 11)}

json.dump(res, open('results.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
