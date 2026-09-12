"""Item 15: Majorana constellations in the handout convention
F_u(z) = sum_m (-1)^{3-m} sqrt(C(6,3+m)) u_m z^{3-m}; root r -> theta = 2 arctan|r|, phi = arg r;
each missing degree -> a point at theta = pi. Cartesian point of r: (2Re r, 2Im r, 1-|r|^2)/(1+|r|^2).

Roots are built exactly in polar form: F = z^k Q(z^n) with Q of degree <= 2 for every state here;
each root w of Q is written |w| e^{i alpha} (alpha exact), giving z = |w|^{1/n} e^{i(alpha+2 pi t)/n}.
"""
from collections import Counter
from functools import reduce
from itertools import combinations
from math import gcd

import sympy as sp

from solver_lib import save_results, s

z, w = sp.symbols("z w")


def F(u):
    return sp.expand(sum(sp.Integer(-1) ** (3 - m) * sp.sqrt(sp.binomial(6, 3 + m)) * u[m] * z ** (3 - m)
                         for m in range(-3, 4) if u.get(m, 0) != 0))


def wrap(phi):
    phi = sp.simplify(phi)
    while phi > sp.pi:
        phi = phi - 2 * sp.pi
    while phi <= -sp.pi:
        phi = phi + 2 * sp.pi
    return sp.simplify(phi)


def constellation(u):
    f = F(u)
    P = sp.Poly(f, z)
    deg = P.degree()
    terms = P.terms()
    exps = [e[0] for e, _ in terms]
    k = min(exps)
    diffs = [e - k for e in exps if e != k]
    pts = []
    for _ in range(k):   # z = 0 roots -> north pole
        pts.append({"rho2": sp.S(0), "phi": sp.nan})
    if diffs:
        n = reduce(gcd, diffs)
        Q = sp.Poly(sum(c * w ** ((e[0] - k) // n) for e, c in terms), w)
        wr = sp.roots(Q)
        assert sum(wr.values()) == Q.degree(), "Q roots incomplete"
        for wroot, mult in wr.items():
            wroot = sp.radsimp(wroot)
            mod = sp.simplify(sp.Abs(wroot))
            alpha = sp.simplify(sp.arg(wroot))
            assert alpha.is_number and (alpha / sp.pi).is_rational, alpha
            for _ in range(mult):
                for t in range(n):
                    pts.append({"rho2": sp.simplify(mod ** sp.Rational(2, n)),
                                "phi": wrap((alpha + 2 * sp.pi * t) / n)})
    for _ in range(6 - deg):   # missing degree -> south pole
        pts.append({"rho2": sp.oo, "phi": sp.nan})
    out = []
    for p in pts:
        r2 = p["rho2"]
        if r2 == sp.oo:
            xyz = (sp.S(0), sp.S(0), sp.S(-1))
            theta = sp.pi
        else:
            rr = sp.sqrt(r2)
            ph = p["phi"] if p["phi"] is not sp.nan else 0
            xyz = (sp.simplify(2 * rr * sp.cos(ph) / (1 + r2)), sp.simplify(2 * rr * sp.sin(ph) / (1 + r2)),
                   sp.simplify((1 - r2) / (1 + r2)))
            theta = sp.simplify(2 * sp.atan(rr))
        out.append({"abs_root_sq": r2, "xyz": xyz, "theta": theta, "sin_latitude": xyz[2],
                    "latitude": sp.simplify(sp.pi / 2 - theta),
                    "azimuth": p["phi"] if r2 not in (0, sp.oo) else sp.nan})
    return f, out


def theta_prop(u):
    """Exact test: Theta u = lambda u ?  (Theta v_m = (-1)^m v_{-m}, antilinear)."""
    vec = [sp.S(u.get(m, 0)) for m in range(-3, 4)]
    tv = [sp.Integer(-1) ** abs(m) * sp.conjugate(vec[-m + 3]) for m in range(-3, 4)]
    i0 = next(i for i in range(7) if vec[i] != 0)
    lam = sp.simplify(tv[i0] / vec[i0])
    ok = all(sp.simplify(tv[i] - lam * vec[i]) == 0 for i in range(7))
    return ok, (lam if ok else None)


r2_, r3, r13 = sp.sqrt(2), sp.sqrt(3), sp.sqrt(13)
cases = {
    "v3": {3: 1},
    "v0": {0: 1},
    "(v2+v-2)/sqrt2": {2: 1 / r2_, -2: 1 / r2_},
    "(v3+v-3)/sqrt2": {3: 1 / r2_, -3: 1 / r2_},
    "item8 s=12/25 (+): (sqrt13/5) v2 + (2sqrt3/5) v-3": {2: r13 / 5, -3: 2 * r3 / 5},
    "item8 s=12/25 (-): (sqrt13/5) v2 - (2sqrt3/5) v-3": {2: r13 / 5, -3: -2 * r3 / 5},
    "item9 z=0": {3: 1, -3: 1},
    "item9 z=+i sqrt10/2": {3: 1, 0: sp.I * sp.sqrt(10) / 2, -3: 1},
    "item9 z=-i sqrt10/2": {3: 1, 0: -sp.I * sp.sqrt(10) / 2, -3: 1},
    "item9 z=+sqrt230/10": {3: 1, 0: sp.sqrt(230) / 10, -3: 1},
    "item9 z=-sqrt230/10": {3: 1, 0: -sp.sqrt(230) / 10, -3: 1},
    "item9 point at infinity [v0]": {0: 1},
}
out = {}
for name, u in cases.items():
    f, pts = constellation(u)
    ds = [sp.simplify(sum(x * y for x, y in zip(p["xyz"], q["xyz"]))) for p, q in combinations(pts, 2)]
    dcount = Counter(round(float(d), 12) for d in ds)
    lat = Counter()
    for p in pts:
        lat[p["sin_latitude"]] += 1
    tr, lam = theta_prop(u)
    print(f"\n== {name}:  F = {f};  Theta u = lambda u: {tr} (lambda = {lam})")
    for p in pts:
        print(f"   xyz = {tuple(p['xyz'])}; theta = {p['theta']}; sin(latitude) = {p['sin_latitude']};"
              f" azimuth = {p['azimuth']}   [theta ~ {float(p['theta']):.9f}]")
    print("   latitude groups (sin lat: count):", {str(k): v for k, v in lat.items()})
    print("   pairwise dot products (float, rounded 1e-12; multiset):", dict(sorted(dcount.items())))
    out[name] = {"F": s(f), "Theta_u_proportional_to_u": tr, "lambda": s(lam),
                 "points": [{"xyz": [s(c) for c in p["xyz"]], "abs_root_sq": s(p["abs_root_sq"]),
                             "theta": s(p["theta"]), "sin_latitude": s(p["sin_latitude"]),
                             "latitude": s(p["latitude"]), "azimuth": s(p["azimuth"]),
                             "theta_float": float(p["theta"]),
                             "xyz_float": [float(c) for c in p["xyz"]]} for p in pts],
                 "pairwise_dot_products_float_multiset": {str(k): v for k, v in sorted(dcount.items())}}
save_results("item15", out)
