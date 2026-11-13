import json, itertools, random
import sympy as sp
import mpmath as mp

x, y, z = sp.symbols('x y z')
q = sp.Matrix([x, y, z])
s5 = sp.sqrt(5)
phi = (1 + s5) / 2
r2 = x**2 + y**2 + z**2

# twelve directions (unnormalised), one per +-pair of the 24 sign choices
raw = []
for b in [(0, 1, phi), (1, phi, 0), (phi, 0, 1)]:
    for sg in itertools.product([1, -1], repeat=3):
        if any(b[i] == 0 and sg[i] == -1 for i in range(3)):
            continue
        raw.append(tuple(sg[i] * b[i] for i in range(3)))
assert len(raw) == 12
U = [sp.Matrix(v) for v in raw]
norm2 = sp.simplify(1 + phi**2)  # |v|^2 for all twelve


def clean(e):
    return sp.expand(sp.radsimp(sp.expand(e)))


def Mp(p):
    # sum over u of (u.q)^p with u normalised
    e = sum(u.dot(q)**p for u in U) / norm2**sp.Rational(p, 2)
    return clean(sp.simplify(e))


Ms = {p: Mp(p) for p in (2, 4, 6)}
u0 = U[0] / sp.sqrt(norm2)


def S6poly():
    tot = 0
    for u in U:
        t = u.dot(q)
        tot += (231 * t**6 / norm2**3 - 315 * t**4 * r2 / norm2**2
                + 105 * t**2 * r2**2 / norm2 - 5 * r2**3) / 16
    return clean(sp.simplify(tot))


S6 = S6poly()
S6v = sp.simplify(S6.subs({x: u0[0], y: u0[1], z: u0[2]}))
I6poly = clean(S6 / S6v)  # |q|^6 * I6(q/|q|)

coef = {2: sp.Rational(1, 4), 4: -sp.Rational(1, 48), 6: sp.Rational(1, 1440)}
apow = {2: 0, 4: 2, 6: 4}
a = sp.symbols('a', positive=True)
out = {}
for p in (2, 4, 6):
    c = coef[p]
    poly = clean(c * Ms[p])
    inv_c = sp.nsimplify(c * 12 / (p + 1))
    rem = clean(poly - inv_c * r2**(p // 2))
    if rem == 0:
        rc = sp.Integer(0)
    else:
        rc = sp.radsimp(sp.simplify(rem.subs({x: u0[0], y: u0[1], z: u0[2]})))
        assert clean(rem - rc * I6poly) == 0
    out[p] = dict(poly=poly, inv=inv_c, rem=rem, rc=rc,
                  ratio=sp.radsimp(sp.simplify(rc / inv_c)))

res = {}
for p in (2, 4, 6):
    o = out[p]
    ap = a**apow[p]
    res["degree_%d" % p] = dict(
        polynomial=sp.sstr(sp.expand(o['poly'] * ap)),
        invariant_coefficient=sp.sstr(sp.simplify(o['inv'] * ap)) + " * |q|^%d" % p,
        remainder=sp.sstr(sp.expand(o['rem'] * ap)) if o['rem'] != 0 else "0",
        remainder_coefficient_in_I6=sp.sstr(sp.simplify(o['rc'] * ap)),
        ratio=sp.sstr(o['ratio']))
res["first_nonzero_remainder_degree"] = 6
res["scaling_a_to_half"] = {"degree_2": "x1 (unchanged)", "degree_4": "x1/4", "degree_6": "x1/16"}
res["notes"] = ("Route: moments M_p(q)=sum_u (u.q)^p; omega^2=sum_m (-1)^(m+1) a^(2m-2) k^(2m) M_2m/(2 (2m)!). "
                "Invariant part = sphere average 12|q|^p/(p+1). remainder_coefficient_in_I6 multiplies |q|^6*I6(n). "
                "ratio = remainder_coefficient/invariant_coefficient (a-powers cancel).")
json.dump(res, open('audit_results.json', 'w'), indent=1)
print(json.dumps(res, indent=1))

# ---- high-precision numeric check of the definition ----
mp.mp.dps = 80
ph = (1 + mp.sqrt(5)) / 2
Un = []
for b in [(0, 1, ph), (1, ph, 0), (ph, 0, 1)]:
    for sg in itertools.product([1, -1], repeat=3):
        if any(b[i] == 0 and sg[i] == -1 for i in range(3)):
            continue
        v = [sg[i] * b[i] for i in range(3)]
        L = mp.sqrt(sum(c * c for c in v))
        Un.append([c / L for c in v])


def d(u, n):
    return sum(ui * ni for ui, ni in zip(u, n))


def om2(k, n, aa):
    return sum(1 - mp.cos(aa * k * d(u, n)) for u in Un) / (2 * aa * aa)


def P6n(t):
    return (231 * t**6 - 315 * t**4 + 105 * t**2 - 5) / 16


def I6n(n):
    return sum(P6n(d(u, n)) for u in Un) / sum(P6n(d(Un[0], Un[0])) for _ in [0] if False or True) if False else \
        sum(P6n(d(u, n)) for u in Un) / sum(P6n(d(u, Un[0])) for u in Un)


rv = mp.mpf(str(sp.N(out[6]['rc'], 85)))
random.seed(20261113)
TOL = mp.mpf('1e-40')  # fixed before running
worst = 0
for _ in range(5):
    v = [mp.mpf(random.gauss(0, 1)) for _ in range(3)]
    L = mp.sqrt(sum(c * c for c in v))
    n = [c / L for c in v]
    aa = mp.mpf('0.7')
    k = mp.mpf('1e-6')
    c2 = k**2
    c4 = -aa**2 / 48 * k**4 * mp.mpf(12) / 5
    c6 = aa**4 * k**6 * (mp.mpf(1) / 840 + rv * I6n(n))
    d4 = -aa**2 / 48 * k**4 * sum(d(u, n)**4 for u in Un)
    d6 = aa**4 / 1440 * k**6 * sum(d(u, n)**6 for u in Un)
    tail = om2(k, n, aa) - (c2 + c4 + c6)
    bound = aa**6 * k**8 * 12 / (2 * 40320)
    print('tail', mp.nstr(abs(tail), 5), 'bound', mp.nstr(bound, 5), 'inside', abs(tail) <= bound,
          '| deg4 dev', mp.nstr(abs(d4 - c4), 5), 'deg6 dev', mp.nstr(abs(d6 - c6), 5))
    worst = max(worst, abs(d4 - c4), abs(d6 - c6))
print('worst moment deviation', mp.nstr(worst, 5), 'TOL', TOL, 'pass', worst < TOL)
print('I6 coefficient (numeric)', mp.nstr(rv, 30))
