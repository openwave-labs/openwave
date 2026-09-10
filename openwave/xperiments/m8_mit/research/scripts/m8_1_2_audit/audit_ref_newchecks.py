"""Refutation phase: new checks where the two workers were silent or used the same route.

(1) items 1, 2 by Weyl integration of exact characters (a route neither worker used).
(2) item 11 by Burnside: dim span{D^3(g) : g in Gamma} = sum of dim^2 of distinct constituents.
(3) item 13: rank of span{||rho_K(u)||^2, K = 0..6} as quartic polynomials (the solver's
    non-uniqueness remark: is the one relation sum_K rhat_K = 1 = 7 rhat_0 the only one?).
(4) item 8: the second ray with the same s (the solver's (-) ray), exactly.
(5) item 9: numeric multi-start search for ALL critical points of rhat6 on the projective line.
(6) item 19: exact bridge between my normalisation (1/1296) and the solver's (504 sqrt286).
(7) item 20: the quaternion->SU(2) maps of the two workers differ by an SU(2) conjugation V; check
    that the solver's projectors are D(V) P_mine D(V)^dag and its form is my form rotated.
(8) item 4: lambda_m = p_6 ||rho_6(v_m)||^2 with p_6 = -sqrt91/13 (solver's pairing constant).
(9) item 21: left factor B_J(u) = M_J(u (Theta u)^dag) (solver's claim), with my code.
"""
import json
import pathlib
import itertools
from fractions import Fraction as Fr
from math import comb, factorial
import numpy as np
import sympy as sp
import mpmath as mp
from audit_su2lib import rho, couple, theta, MK_matrix, M_map, basis, simp, norm2
from audit_field import Q1, Q2, closure, K as KF, R5, ZERO, rank

HERE = pathlib.Path(__file__).parent
res = {}

# ---------------- (1) Weyl integration ----------------
t = sp.Symbol("t", real=True)


def chi(j2, ang):  # character of spin j2/2 at rotation angle ang (element diag(e^{i ang/2}, ...)): sin((j2+1)ang/2)/sin(ang/2)
    return sum(sp.exp(sp.I * (j2 - 2 * k) * ang / 2) for k in range(j2 + 1))


c3 = lambda ang: chi(6, ang)
chiSym2 = lambda ang: (c3(ang) ** 2 + c3(2 * ang)) / 2
chiSym3 = lambda ang: (c3(ang) ** 3 + 3 * c3(2 * ang) * c3(ang) + 2 * c3(3 * ang)) / 6


def mult(chW, J):
    # Weyl: m_J = (1/pi) int_0^{2pi} chW(t) chi_J(t) sin^2(t/2) dt   (class function in the angle t)
    f = sp.expand(chW(t) * chi(2 * J, t) * (2 - sp.exp(sp.I * t) - sp.exp(-sp.I * t)) / 4)
    # constant term of the Laurent polynomial in e^{i t/2} gives the average over t in [0, 4pi]
    # integrate numerically-exactly: average over a fine uniform grid is exact for Laurent polys of bounded degree
    N = 200
    val = sum(complex(sp.N(f.subs(t, 4 * sp.pi * k / N), 30)) for k in range(N)) / N * 2
    return round(val.real), val


res["item1_Weyl"] = {J: mult(chiSym3, J)[0] for J in range(0, 10)}
sym2V3 = lambda ang: chiSym2(ang) * c3(ang)
res["item2_Weyl_J8"], res["item2_Weyl_J3"] = mult(sym2V3, 8)[0], mult(sym2V3, 3)[0]

# ---------------- (2) Burnside for item 11 ----------------
G = closure([Q1, Q2])


def Dmon(q, n):
    (a, b), (c, d) = q.su2()
    X, Y = [c, a], [d, b]

    def pm(p, q_):
        o = [ZERO] * (len(p) + len(q_) - 1)
        for i, x in enumerate(p):
            for k, y in enumerate(q_):
                o[i + k] = o[i + k] + x * y
        return o

    def pw(p, e):
        o = [KF(1)]
        for _ in range(e):
            o = pm(o, p)
        return o
    M = [[ZERO] * (n + 1) for _ in range(n + 1)]
    for k in range(n + 1):
        col = pm(pw(X, k), pw(Y, n - k))
        for kp in range(n + 1):
            M[kp][k] = col[kp]
    return M


rows = [[x for r in Dmon(g, 6) for x in r] for g in G]
res["item11_Burnside_dim_span_D3(Gamma)"] = rank(rows)   # expect 3^2 + 4^2 = 25 if two inequivalent irreducibles

# ---------------- (3) span of ||rho_K||^2 ----------------
us = list(sp.symbols("u0:7"))
ub = [sp.conjugate(x) for x in us]
polys = []
for Kk in range(7):
    r = rho(us, Kk)
    polys.append(sp.expand(sum(x * sp.conjugate(x) for x in r)))
gens = us + ub
mons = sorted(set().union(*[set(sp.Poly(p, *gens).monoms()) for p in polys]))
mp.mp.dps = 60
Mx = mp.matrix(7, len(mons))
for i, p in enumerate(polys):
    Pp = sp.Poly(p, *gens)
    dct = dict(zip(Pp.monoms(), Pp.coeffs()))
    for k, mo in enumerate(mons):
        Mx[i, k] = mp.mpf(str(sp.N(dct.get(mo, 0), 70)))
sv = sorted([float(x) for x in mp.svd_r(Mx, compute_uv=False)], reverse=True)
nrm4 = sp.expand(sum(a * b for a, b in zip(us, ub)) ** 2)
res["item13_span_rhoK_norms_singular_values"] = sv
res["item13_span_rhoK_norms_rank"] = sum(1 for x in sv if x > 1e-40)
res["item13_relation_sum_K_equals_norm4"] = sp.expand(sum(polys) - nrm4) == 0
res["item13_relation_7rho0_equals_norm4"] = sp.expand(7 * polys[0] - nrm4) == 0

# ---------------- (4) item 8 second ray ----------------
uminus = [0] * 7
uminus[2 + 3] = sp.sqrt(13) / 5
uminus[-3 + 3] = -2 * sp.sqrt(3) / 5
uminus = [sp.S(x) for x in uminus]
M6m = [simp(x) for x in M_map(uminus, 6)]
lam = simp(sum(sp.conjugate(uminus[k]) * M6m[k] for k in range(7)))
res["item8_minus_ray_rhat6"] = str(norm2(rho(uminus, 6)))
res["item8_minus_ray_critical(M6 || u)"] = all(simp(M6m[k] - lam * uminus[k]) == 0 for k in range(7))
zeta = sp.Symbol("zeta")
Fm = sp.expand(sum((-1) ** (3 - m) * sp.sqrt(sp.binomial(6, 3 + m)) * uminus[m + 3] * zeta ** (3 - m) for m in range(-3, 4)))
res["item8_minus_ray_F"] = str(Fm)
fifth = sp.solve(sp.Eq(sp.Poly(Fm, zeta).coeff_monomial(zeta ** 6) * zeta ** 5 + sp.Poly(Fm, zeta).coeff_monomial(zeta), 0), zeta)
res["item8_minus_ray_ring_azimuths_deg"] = sorted(round(float(sp.N(sp.arg(r) * 180 / sp.pi, 20)), 9) for r in sp.Poly(Fm / zeta, zeta).nroots(n=30))

# ---------------- (5) item 9 numeric critical-point census ----------------
from audit_su2lib import cg as cgx
CGt = {}


def cgf(*k):
    if k not in CGt:
        CGt[k] = float(cgx(*k))
    return CGt[k]


def rhat6_num(u):
    th_ = np.array([(-1) ** abs(m) * np.conj(u[-m + 3]) for m in range(-3, 4)])
    tot = 0.0
    for N in range(-6, 7):
        s_ = 0j
        for n_ in range(-3, 4):
            npr = N - n_
            if abs(npr) <= 3:
                s_ += cgf(3, n_, 3, npr, 6, N) * u[n_ + 3] * th_[npr + 3]
        tot += abs(s_) ** 2
    return tot / np.vdot(u, u).real ** 2


def chartA(x, y):  # u = v3 + z v0 + v-3  (misses only [v0])
    u = np.zeros(7, complex)
    u[6] = u[0] = 1.0
    u[3] = x + 1j * y
    return u, (np.sqrt(2), x + 1j * y)


def chartB(x, y):  # u = w (v3 + v-3) + v0  (misses only [v3 + v-3])
    u = np.zeros(7, complex)
    u[6] = u[0] = x + 1j * y
    u[3] = 1.0
    return u, (np.sqrt(2) * (x + 1j * y), 1.0)


def bloch(ab):
    a_, b_ = ab
    nrm = abs(a_) ** 2 + abs(b_) ** 2
    n = np.array([2 * (np.conj(a_) * b_).real, 2 * (np.conj(a_) * b_).imag, abs(a_) ** 2 - abs(b_) ** 2]) / nrm
    return tuple(round(float(c), 6) + 0.0 for c in n)


rng = np.random.default_rng(9)
found = []
for chart in (chartA, chartB):
    f = lambda p: rhat6_num(chart(p[0], p[1])[0])

    def gr(p, h=1e-6):
        return np.array([(f(p + np.array([h, 0])) - f(p - np.array([h, 0]))) / (2 * h),
                         (f(p + np.array([0, h])) - f(p - np.array([0, h]))) / (2 * h)])
    for trial in range(250):
        p = rng.uniform(-2.5, 2.5, size=2)
        for it in range(80):
            g_ = gr(p)
            H = np.column_stack([(gr(p + np.array([1e-4, 0])) - gr(p - np.array([1e-4, 0]))) / 2e-4,
                                 (gr(p + np.array([0, 1e-4])) - gr(p - np.array([0, 1e-4]))) / 2e-4])
            try:
                step = np.linalg.solve(H, g_)
            except np.linalg.LinAlgError:
                break
            if np.linalg.norm(step) > 1.0:
                step = step / np.linalg.norm(step)
            p = p - step
            if np.linalg.norm(step) < 1e-12 or np.linalg.norm(p) > 1e3:
                break
        if np.linalg.norm(p) < 1e3 and np.linalg.norm(gr(p)) < 1e-7:
            found.append((bloch(chart(p[0], p[1])[1]), round(f(p), 9)))
from collections import Counter
cen = Counter(found)
res["item9_numeric_census_bloch_point_and_rhat6"] = {str(k): c for k, c in sorted(cen.items())}
res["item9_numeric_distinct_critical_rays"] = len({k[0] for k in cen})

# ---------------- (6) item 19 normalisation bridge ----------------
bridge = sp.Rational(720 ** 3 * 504 ** 2 * 286, factorial(16))
res["item19_bridge_720^3*504^2*286/16!"] = str(bridge)
res["item19_bridge_equals_1296"] = bridge == 1296

# ---------------- (7) item 20: conjugation between the two quaternion maps ----------------
def U_mine(w, x, y, z_):
    return np.array([[w - 1j * z_, -y - 1j * x], [y - 1j * x, w + 1j * z_]])


def U_sol(w, x, y, z_):
    return np.array([[w + 1j * x, y + 1j * z_], [-y + 1j * z_, w - 1j * x]])


rows_ = []
for e in [(0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]:
    Am, As = U_mine(*e), U_sol(*e)
    # V Am - As V = 0, unknown V (2x2) flattened
    for i in range(2):
        for k in range(2):
            row = np.zeros(4, complex)
            for l in range(2):
                row[i * 2 + l] += Am[l, k]
                row[l * 2 + k] -= As[i, l]
            rows_.append(row)
_, svv, Vh = np.linalg.svd(np.array(rows_))
V = Vh[-1].conj().reshape(2, 2)
V = V / np.sqrt(np.linalg.det(V))
res["item20_V_nullity_singular_values"] = [float(x) for x in svv]
sqc = np.array([comb(6, k) ** 0.5 for k in range(7)])


def D3num(U):
    a, b, c, d = U[0, 0], U[0, 1], U[1, 0], U[1, 1]
    Dm = np.zeros((7, 7), complex)
    for k in range(7):
        for i in range(k + 1):
            for l in range(7 - k):
                Dm[i + l, k] += comb(k, i) * a ** i * c ** (k - i) * comb(6 - k, l) * b ** l * d ** (6 - k - l)
    return Dm * sqc[None, :] / sqc[:, None]


# solver's D uses v_m = x^{j+m} y^{j-m}/sqrt((j+m)!(j-m)!) -> same orthonormal basis up to the constant sqrt(6!)
Pmine = {}
Pj = json.loads((HERE / "audit_projectors_level6.json").read_text())
for d in (3, 4):
    Pm = np.array([[complex(float(Fr(t[0])) + float(Fr(t[1])) * 5 ** 0.5, float(Fr(t[2])) + float(Fr(t[3])) * 5 ** 0.5) for t in row] for row in Pj[f"d{d}"]])
    Pmine[d] = Pm * sqc[None, :] / sqc[:, None]
S = json.loads((HERE / "solver_copy" / "solver_results.json").read_text())
ok = {}
DV = D3num(V)
for d in (3, 4):
    Psol = np.array([[complex(sp.N(sp.sympify(e), 20)) for e in r] for r in S["group"]["item12"][f"d={d}"]["P_orth_rows"]])
    ok[d] = float(np.max(np.abs(DV @ Pmine[d] @ DV.conj().T - Psol)))
res["item20_max|D(V) P_mine D(V)^dag - P_solver|"] = ok
# the rotation of V
sig = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
Rv = np.array([[0.5 * np.trace(sig[i] @ V @ sig[k] @ V.conj().T).real for k in range(3)] for i in range(3)])
res["item20_rotation_of_V (sigma-frame)"] = np.round(Rv, 12).tolist()

# ---------------- (8) item 4 via pairing constant ----------------
p6 = -sp.sqrt(91) / 13
res["item4_lambda_equals_p6_norm_rho6"] = all(simp(M_map(basis(m), 6)[m + 3] - p6 * norm2(rho(basis(m), 6))) == 0 for m in range(-3, 4))

# ---------------- (9) item 21 left factor ----------------
ut = [sp.Integer(1) + sp.I, sp.Integer(2), -sp.I, sp.Rational(1, 2), sp.Integer(-1), 3 * sp.I, sp.Integer(1) - 2 * sp.I]
th_u = theta(ut)
X = sp.Matrix(7, 1, ut) * sp.Matrix(7, 1, th_u).H
res["item21_B_J_equals_M_J(u Theta u^dag)"] = all(
    all(simp(a_ - b_) == 0 for a_, b_ in zip(couple(ut, ut, 3, 3, J), MK_matrix(X, J))) for J in range(7))

(HERE / "audit_ref_newchecks.json").write_text(json.dumps(res, indent=1, default=str))
print(json.dumps(res, indent=1, default=str))
