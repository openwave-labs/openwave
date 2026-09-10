"""Stage 2, item 20: R_6(P) as a degree-12 binary form (Majorana correspondence extended to spin 6),
root multiset with exact multiplicities, and injectivity of v -> (G, F_v)_1.

Route A (exact, Q(sqrt5, i)): a_N := sqrt(C(12,6+N)) R_N = sum_{k1+k2=N+6} w_mon[k1,k2], using the
stretched CS coupling <3n;3n'|6N> = sqrt(C(6,3+n)C(6,3+n')/C(12,6+N)) and the monomial-basis w of
stage 1. Route B (sympy CG, orthonormal basis) cross-checks every component.
Majorana (spin 6): G(z) = sum_N (-1)^(6-N) sqrt(C(12,6+N)) R_N z^(6-N); homogeneous G(X,Y) with z = X/Y.
"""
import json
import pathlib
from fractions import Fraction as Fr
from math import comb
import sympy as sp
import mpmath as mp
from audit_field import K, R5, ZERO, ONE, rank
from audit_su2lib import MK_matrix

HERE = pathlib.Path(__file__).parent
Pj = json.loads((HERE / "audit_projectors_level6.json").read_text())
res = {}


def kfrom(t):
    a, b, c, d = [Fr(x) for x in t]
    return K(R5(a, b), R5(c, d))


def kq(re, im=0):
    return K(R5(Fr(re)), R5(Fr(im)))


def kstr(x):
    return str(sp.nsimplify(x.tosympy()))


# ---------- polynomial helpers over K (coefficient lists, index = power) ----------
def trim(p):
    p = list(p)
    while len(p) > 1 and p[-1].iszero():
        p.pop()
    return p


def pdivmod(a, b):
    a, b = trim(a), trim(b)
    q = [ZERO] * max(1, len(a) - len(b) + 1)
    r = a[:]
    inv = b[-1].inv()
    while len(r) >= len(b) and not (len(r) == 1 and r[0].iszero()):
        if len(r) < len(b):
            break
        c = r[-1] * inv
        sh = len(r) - len(b)
        q[sh] = c
        for i, bb in enumerate(b):
            r[sh + i] = r[sh + i] - c * bb
        r = trim(r)
        if len(r) < len(b) or (len(r) == 1 and r[0].iszero()):
            break
    return trim(q), r


def pgcd(a, b):
    a, b = trim(a), trim(b)
    while not (len(b) == 1 and b[0].iszero()):
        _, r = pdivmod(a, b)
        a, b = b, r
    inv = a[-1].inv()
    return [x * inv for x in a]


def pderiv(p):
    return trim([p[i] * i for i in range(1, len(p))] or [ZERO])


out = {}
forms = {}
for d in (3, 4):
    Pm = [[kfrom(t) for t in row] for row in Pj[f"d{d}"]]
    n = 6
    eps = -1
    w = [[Pm[k1][n - k2] * (eps * (-1) ** (n - k2) * comb(n, k2)) for k2 in range(7)] for k1 in range(7)]
    a = {}
    for N in range(-6, 7):
        s = ZERO
        for k1 in range(7):
            k2 = N + 6 - k1
            if 0 <= k2 <= 6:
                s = s + w[k1][k2]
        a[N] = s
    # norm check: ||R_6||^2 = sum |a_N|^2 / C(12,6+N) must be 12/7
    nrm = sum((a[N] * a[N].conj() * kq(Fr(1, comb(12, 6 + N))) for N in range(-6, 7)), ZERO)
    assert nrm == kq(Fr(12, 7)), nrm
    # route B: sympy CG on the orthonormal-basis projector
    Pon = sp.Matrix(7, 7, lambda k, l: Pm[k][l].tosympy() * sp.sqrt(sp.Rational(comb(6, l), comb(6, k))))
    RB = MK_matrix(Pon, 6)
    for i, N in enumerate(range(-6, 7)):
        diff = sp.N(sp.sqrt(comb(12, 6 + N)) * RB[i] - a[N].tosympy(), 40)
        assert abs(diff) < 1e-30, (d, N, diff)
    # Majorana coefficients: G(z) = sum_N g_{6-N} z^{6-N}, g_{6-N} = (-1)^{6-N} a_N
    g = [ZERO] * 13
    for N in range(-6, 7):
        g[6 - N] = a[N] * ((-1) ** abs(6 - N))
    forms[d] = g
    out[f"d={d}"] = {"a_N=sqrt(C(12,6+N)) R_N, N=-6..6": [kstr(a[N]) for N in range(-6, 7)],
                     "G_coeffs_z^0..z^12": [kstr(x) for x in g]}
assert all(forms[3][i] == -forms[4][i] for i in range(13))
res["R6(P3) = -R6(P4) exactly"] = True

g = forms[3]
gt = trim(g)
deg = len(gt) - 1
low = next(i for i in range(13) if not g[i].iszero())
res["G_d3"] = out["d=3"]
res["G_d4"] = out["d=4"]
res["degree_in_z"] = deg
res["multiplicity_at_infinity(12-deg)"] = 12 - deg
res["multiplicity_at_z=0(lowest power)"] = low
gg = pgcd(gt, pderiv(gt))
res["gcd(G,G')_degree"] = len(gg) - 1
# squarefree => all finite roots simple; together with infinity multiplicity
res["all_roots_simple"] = (len(gg) - 1 == 0) and (12 - deg) <= 1
res["number_of_distinct_roots_on_sphere"] = deg + (1 if deg < 12 else 0) if len(gg) == 1 else None

# exact factorisation attempt
zs = sp.symbols("z")
Gsym = sp.expand(sum(sp.nsimplify(g[i].tosympy()) * zs ** i for i in range(13)))
res["G_sympy"] = str(Gsym)
try:
    fac = sp.factor_list(Gsym, extension=[sp.sqrt(5), sp.I])
    res["factor_list_over_Q(sqrt5,i)"] = [[str(f), m] for f, m in fac[1]]
    res["factor_const"] = str(fac[0])
except Exception as e:  # record, do not hide
    res["factor_list_error"] = repr(e)

# numeric roots (50 digits) and geometry
mp.mp.dps = 50
coeffs = [mp.mpc(*[mp.mpf(str(sp.N(sp.re(x.tosympy()), 60))), mp.mpf(str(sp.N(sp.im(x.tosympy()), 60)))]) for x in gt]
rts = mp.polyroots(coeffs[::-1], maxsteps=500, extraprec=400)
pts = []
for r in rts:
    rr = abs(r) ** 2
    pts.append((2 * r.real / (1 + rr), 2 * r.imag / (1 + rr), (1 - rr) / (1 + rr)))
for _ in range(12 - deg):
    pts.append((mp.mpf(0), mp.mpf(0), mp.mpf(-1)))
rootinfo = []
for r in rts:
    rootinfo.append({"root": mp.nstr(r, 25), "theta_deg": mp.nstr(2 * mp.atan(abs(r)) * 180 / mp.pi, 20),
                     "azimuth_deg": mp.nstr(mp.arg(r) * 180 / mp.pi, 20) if abs(r) > 1e-40 else "pole"})
res["roots_numeric"] = rootinfo
dots = []
for i in range(len(pts)):
    for j in range(i + 1, len(pts)):
        dots.append(sum(pts[i][k] * pts[j][k] for k in range(3)))
cls = {}
for dv in dots:
    key = mp.nstr(dv, 25)
    cls[key] = cls.get(key, 0) + 1
res["pairwise_dot_classes"] = cls
res["min_pairwise_chordal_distance"] = mp.nstr(min(mp.sqrt(2 - 2 * dv) for dv in dots), 25)
s5 = 1 / mp.sqrt(5)
res["dots_all_in_{-1,+-1/sqrt5}"] = all(min(abs(dv - t) for t in (-1, s5, -s5)) < mp.mpf(10) ** -35 for dv in dots)

# ---------- transvectant map v -> J(G, F_v), homogeneous, exact rank ----------
# G(X,Y) = sum_i g_i X^i Y^(12-i) (z = X/Y); F_{v_m} = (-1)^(3-m) sqrt(C(6,3+m)) X^(3-m) Y^(3+m).
# Column rescaling by the nonzero constants (-1)^(3-m) sqrt(C) does not change the rank, so use X^a Y^(6-a).


def jac_h(f, gpoly):
    """J = f_X g_Y - f_Y g_X, f, g homogeneous; lists index = power of X."""
    df, dg = len(f) - 1, len(gpoly) - 1

    def dX(p):
        return [p[k] * k for k in range(1, len(p))]

    def dY(p):
        dd = len(p) - 1
        return [p[k] * (dd - k) for k in range(len(p) - 1)]

    def mul(p, q):
        o = [ZERO] * (len(p) + len(q) - 1)
        for i, x in enumerate(p):
            for j, y in enumerate(q):
                o[i + j] = o[i + j] + x * y
        return o
    t1, t2 = mul(dX(f), dY(gpoly)), mul(dY(f), dX(gpoly))
    return [x - y for x, y in zip(t1, t2)]


cols = []
for a_ in range(7):
    F = [ZERO] * 7
    F[a_] = ONE
    cols.append(jac_h(g, F))
M = [[cols[c][r] for c in range(7)] for r in range(17)]
rk = rank(M)
res["transvectant_map_matrix_17x7_exact_rank_over_Q(sqrt5,i)"] = rk
res["transvectant_map_injective_by_exact_rank"] = (rk == 7)
# numeric SVD as a third, weaker piece of evidence
import numpy as np
Mn = np.array([[M[r][c].tocomplex() for c in range(7)] for r in range(17)])
res["transvectant_map_singular_values_double"] = [float(x) for x in np.linalg.svd(Mn, compute_uv=False)]
# control: a perfect square G0 = F_w^2 must give a nontrivial kernel (w itself)
w0 = [kq(1), kq(0), kq(2), kq(0, 1), kq(0), kq(-1), kq(1)]
G0 = [ZERO] * 13
for i in range(7):
    for j in range(7):
        G0[i + j] = G0[i + j] + w0[i] * w0[j]
cols0 = []
for a_ in range(7):
    F = [ZERO] * 7
    F[a_] = ONE
    cols0.append(jac_h(G0, F))
M0 = [[cols0[c][r] for c in range(7)] for r in range(17)]
res["control_square_form_rank"] = rank(M0)
assert rank(M0) == 6

(HERE / "audit_res_stage2_item20.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
