# Step one: exact moment route.
# omega^2 = sum_m (-1)^(m+1) a^(2m-2) S_2m(q) / (2 (2m)!),  S_2m(q) = sum_u (u.q)^(2m)
import sympy as sp, json, itertools

x, y, z = sp.symbols('x y z')
a = sp.symbols('a', positive=True)
t = sp.symbols('t')
phi = (1 + sp.sqrt(5)) / 2
base = [(0, 1, phi), (1, phi, 0), (phi, 0, 1)]
vs = set()
for v in base:
    for s in itertools.product([1, -1], repeat=3):
        vs.add(tuple(sp.simplify(si * c) for si, c in zip(s, v)))
U = sorted(vs, key=str)
assert len(U) == 12
nrm2 = sp.expand(1 + phi ** 2)  # |v|^2 of every unnormalised vertex
q = (x, y, z)
r2 = x ** 2 + y ** 2 + z ** 2


def dot(v, w):
    return sum(c * d for c, d in zip(v, w))


def S(p):  # sum over normalised u of (u.q)^p, exact in Q(sqrt5)
    e = sum(sp.expand(dot(v, q) ** p) for v in U) / nrm2 ** (p // 2)
    return sp.expand(sp.radsimp(sp.expand(e)))


def clean(e):
    return sp.expand(sp.radsimp(sp.simplify(e)))


S_ = {p: clean(S(p)) for p in (2, 4, 6)}
P6 = sp.legendre(6, t)


def homog_P6(d):  # |q|^6 P6(d/|q|) with d = u.q (u normalised)
    res = 0
    for (j,), c in sp.Poly(P6, t).terms():
        res += c * d ** j * r2 ** ((6 - j) // 2)
    return res


raw = sum(homog_P6(dot(v, q) / sp.sqrt(nrm2)) for v in U)
raw = clean(raw)
# value of the same sum at a vertex n = U[0]/|U[0]|
n0 = [c / sp.sqrt(nrm2) for c in U[0]]
sumvert = sp.simplify(sum(P6.subs(t, dot(v, n0) / sp.sqrt(nrm2)) for v in U))
sumvert = sp.nsimplify(sp.radsimp(sumvert))
I6poly = clean(raw / sumvert)   # |q|^6 * I6(q/|q|)

out = {}
mons = None
for m in (1, 2, 3):
    pol = clean((-1) ** (m + 1) * a ** (2 * m - 2) * S_[2 * m] / (2 * sp.factorial(2 * m)))
    inv = sp.Rational(12, 2 * m + 1) * (-1) ** (m + 1) * a ** (2 * m - 2) / (2 * sp.factorial(2 * m))
    rem = clean(pol - inv * r2 ** m)
    if rem == 0:
        rc = sp.Integer(0)
    else:
        P = sp.Poly(rem, x, y, z)
        mon = P.monoms()[0]
        rc = sp.simplify(sp.radsimp(P.coeffs()[0] / sp.Poly(I6poly, x, y, z).coeff_monomial(mon)))
        assert clean(rem - rc * I6poly) == 0
    out['degree_%d' % (2 * m)] = dict(
        polynomial=str(pol), invariant_coefficient=str(sp.simplify(inv)),
        remainder=str(rem), remainder_coefficient_in_I6=str(rc),
        ratio=str(sp.simplify(rc / inv)) if rem != 0 else '0')
out['first_nonzero_remainder_degree'] = 6
out['scaling_a_to_half'] = {'degree_2': '1 (unchanged)', 'degree_4': '1/4', 'degree_6': '1/16',
                            'rule': 'coefficient of degree 2m scales by 2^-(2m-2)'}
out['notes'] = ('ratio = remainder_coefficient_in_I6 / invariant_coefficient; invariant part = sphere '
                'average, coef*|q|^(2m); I6 polynomial = |q|^6 I6(q/|q|); sum of P6 at a vertex = ' + str(sumvert))
if __name__ == '__main__':
    json.dump(out, open('audit_results.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))
