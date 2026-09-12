"""Item 19 checks (exact, over Q(i) inside K = Q(sqrt5, i)), monomial basis e_k = x^k y^(6-k).

C(u) := V_8-isotypic component of u (x) Theta u (x) u in V3^{(x)3}, computed with the Casimir
projector (exact). Compared with the Jacobian J(p_u^2, p_{Theta u}). These are CHECKS; the
proof is written in the return file.
"""
import json
import pathlib
import itertools
import random
from fractions import Fraction as Fr
from math import comb
from audit_field import K, R5, ZERO, ONE

HERE = pathlib.Path(__file__).parent
res = {}
n = 6


def kq(re, im=0):
    return K(R5(Fr(re)), R5(Fr(im)))


def theta_mon(c):
    # Theta e_k = (-1)^(k-3) e_{6-k}, antilinear
    out = [ZERO] * 7
    for k in range(7):
        out[6 - k] = out[6 - k] + c[k].conj() * ((-1) ** abs(k - 3))
    return out


def triple(a, b, c):
    w = {}
    for k1, k2, k3 in itertools.product(range(7), repeat=3):
        v = a[k1] * b[k2] * c[k3]
        if not v.iszero():
            w[(k1, k2, k3)] = v
    return w


def jp(v):
    o = {}
    for key, val in v.items():
        for p in range(3):
            k = key[p]
            if k < n:
                nk = list(key); nk[p] += 1; nk = tuple(nk)
                o[nk] = o.get(nk, ZERO) + val * (n - k)
    return o


def jm(v):
    o = {}
    for key, val in v.items():
        for p in range(3):
            k = key[p]
            if k > 0:
                nk = list(key); nk[p] -= 1; nk = tuple(nk)
                o[nk] = o.get(nk, ZERO) + val * k
    return o


def cas(v):
    o = {}
    for key, val in v.items():
        mz = sum(key) - 9
        o[key] = o.get(key, ZERO) + val * (mz * mz)
    half = kq(Fr(1, 2))
    for d in (jp(jm(v)), jm(jp(v))):
        for key, val in d.items():
            o[key] = o.get(key, ZERO) + val * half
    return {k: x for k, x in o.items() if not x.iszero()}


def proj(v, L, spins=range(0, 10)):
    for Lp in spins:
        if Lp == L:
            continue
        cv = cas(v)
        den = Fr(1, L * (L + 1) - Lp * (Lp + 1))
        v = {k: (cv.get(k, ZERO) - v.get(k, ZERO) * (Lp * (Lp + 1))) * kq(den) for k in set(cv) | set(v)}
        v = {k: x for k, x in v.items() if not x.iszero()}
    return v


def ipG(a, b):
    s = ZERO
    for key, val in b.items():
        if key in a:
            s = s + a[key].conj() * val * kq(Fr(1, comb(6, key[0]) * comb(6, key[1]) * comb(6, key[2])))
    return s


def pmul(p, q):
    o = [ZERO] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            o[i + j] = o[i + j] + a * b
    return o


def jac(f, g):
    """J(f,g) = f_x g_y - f_y g_x for homogeneous f (deg a), g (deg b); lists indexed by power of x."""
    a, b = len(f) - 1, len(g) - 1

    def dx(p):
        return [p[k] * k for k in range(1, len(p))]  # x^k y^(d-k) -> k x^(k-1) y^(d-k)

    def dy(p):
        d = len(p) - 1
        return [p[k] * (d - k) for k in range(0, len(p) - 1)]
    t1 = pmul(dx(f), dy(g))
    t2 = pmul(dy(f), dx(g))
    return [x - y for x, y in zip(t1, t2)]


def jnorm2(p):
    d = len(p) - 1
    return sum((x * x.conj() * kq(Fr(1, comb(d, k))) for k, x in enumerate(p)), ZERO)


random.seed(19)


def rnd():
    return kq(random.randint(-3, 3), random.randint(-3, 3))


cases = []
# generic states
for t in range(4):
    cases.append(("generic", [rnd() for _ in range(7)]))
# Theta-fixed states times phases (time-reversal invariant rays)
phases = [ONE, kq(0, 1), kq(Fr(3, 5), Fr(4, 5)), kq(Fr(5, 13), Fr(-12, 13))]
for t in range(4):
    c = [ZERO] * 7
    for k in range(3):
        c[k] = rnd()
        c[6 - k] = c[k].conj() * ((-1) ** abs(k - 3))
    c[3] = kq(random.randint(-3, 3))  # c3 = (-1)^0 conj(c3): real
    ph = phases[t]
    cc = [x * ph for x in c]
    # verify Theta cc = conj(ph)^2 ... i.e. proportional
    th = theta_mon(cc)
    lam = None
    for k in range(7):
        if not cc[k].iszero():
            lam = th[k] / cc[k]
            break
    assert all(th[k] == lam * cc[k] for k in range(7))
    cases.append(("time-reversal invariant ray", cc))
# basis-vector cases
for k in range(7):
    c = [ZERO] * 7; c[k] = ONE
    cases.append((f"monomial e_{k} (v_m, m={k-3})", c))
# a Theta-eigen state built from v3 + v-3 with a relative phase that breaks invariance
c = [ZERO] * 7; c[0] = ONE; c[6] = kq(0, 1)
cases.append(("v3 + i v-3 (Theta u = -i u? check)", c))
c = [ZERO] * 7; c[0] = ONE; c[6] = kq(2)
cases.append(("v3 + 2 v-3 (not invariant)", c))

out = []
ratios = set()
for label, c in cases:
    th = theta_mon(c)
    # is the ray time-reversal invariant?
    lam = None
    inv = True
    for k in range(7):
        if not c[k].iszero():
            lam = th[k] / c[k]
            break
    inv = all(th[k] == lam * c[k] for k in range(7))
    w = triple(c, th, c)
    p8 = proj(w, 8)
    nC = ipG(p8, p8)
    F = c[:]  # p_u coefficients by power of x
    Jv = jac(pmul(F, F), th)
    nJ = jnorm2(Jv)
    JF = jac(F, th)
    rec = {"case": label, "TR_invariant": inv, "||C||^2": repr(nC), "C_zero": nC.iszero(),
           "J(F^2,G)_zero": all(x.iszero() for x in Jv), "J(F,G)_zero": all(x.iszero() for x in JF)}
    if not nC.iszero():
        ratio = nC / nJ
        rec["ratio_||C||^2/||J||^2"] = repr(ratio)
        ratios.add(ratio)
    assert rec["C_zero"] == inv == rec["J(F^2,G)_zero"], rec
    out.append(rec)
res["item19_cases"] = out
res["item19_ratio_set(constant => C = const * Jacobian map)"] = [repr(r) for r in ratios]
assert len(ratios) == 1
(HERE / "audit_res_item19.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
