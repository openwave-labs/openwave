"""Item 8: global minimum and maximum of r6 on the unit sphere.

(a) Minimum, argument 1: N(u) = <u(x)u, K u(x)u> with K Hermitian on Sym^2(C^7); K is computed
    exactly from N (polarisation), its exact eigenvalues give N(u) >= lambda_min |u|^4.
(b) Minimum, argument 2 (independent): rho_6(u) = (1/sqrt 924) p_u p_{Theta u} in the Fock-normalised
    basis of binary forms (identity checked exactly here); then the product inequality
    ||pq||_F >= ||p||_F ||q||_F (proved in RETURN.md) gives r6 >= 1/924.
(c) Maximum: exact upper bounds that are proved (6/7 from |rho_0|^2 = 1/7) and a multi-start
    numerical search.  The search is evidence, not a proof.
"""
import os, sys, json, math, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.optimize as so
import sympy as sp
import core
from core import report, IDX, MS

HERE = os.path.dirname(os.path.abspath(__file__))
ok = True
out = {}
# ------------------------------------------------------------------ (a) Sym^2 form
c = sp.symbols("c0:7")
d = sp.symbols("d0:7")            # d_k stands for conj(c_k)
rh = core.rho6(list(c), conj=lambda zz: d[c.index(zz)])
swap = {**{c[k]: d[k] for k in range(7)}, **{d[k]: c[k] for k in range(7)}}
Nc = sp.expand(sum(v * v.xreplace(swap) for v in rh.values()))   # CG real, so conj(rho) = swap(rho)
P = sp.Poly(Nc, *c, *d)
pairs = [(a, b) for a in range(7) for b in range(a, 7)]
K = sp.zeros(28)
for i, (a, b) in enumerate(pairs):
    for j, (e, f) in enumerate(pairs):
        coeff = P.coeff_monomial(d[a] * d[b] * c[e] * c[f])
        K[i, j] = coeff / ((1 if a == b else sp.sqrt(2)) * (1 if e == f else sp.sqrt(2)))
K = K.applyfunc(core.exact)
# check the representation: N = sum conj(s_i) K_ij s_j with s = orthonormal coordinates of u (x) u in Sym^2
s = [c[a] * c[b] * (1 if a == b else sp.sqrt(2)) for (a, b) in pairs]
sb = [x_.xreplace(swap) for x_ in s]
rep = sp.expand(sum(sb[i] * K[i, j] * s[j] for i in range(28) for j in range(28) if K[i, j] != 0))
ok &= report("N(u) = <u(x)u, K u(x)u> exactly, K Hermitian on Sym^2", sp.expand(rep - Nc) == 0 and K.is_hermitian)
ok &= report("|u(x)u|^2 = |u|^4 in these coordinates",
             sp.expand(sum(sb[i] * s[i] for i in range(28)) - sum(c[k] * d[k] for k in range(7)) ** 2) == 0)
ev = K.eigenvals()
print("exact eigenvalues of K (value: multiplicity):", ev)
lmin = min(ev)
ok &= report("lambda_min(K) = 1/924 with multiplicity 13 (the spin-6 part of Sym^2)", lmin == sp.Rational(1, 924) and ev[lmin] == 13)
ok &= report("r6(v3) = 1/924 attains the bound", core.rhat6([1, 0, 0, 0, 0, 0, 0]) == sp.Rational(1, 924))
out["K_eigenvalues"] = {str(k): v for k, v in ev.items()}
out["min"] = "1/924"

# ------------------------------------------------------------------ (b) binary-form identity
X, Y = sp.symbols("X Y")
def form(coeffs, j):
    return sum(coeffs[IDX[m]] * X ** (j + m) * Y ** (j - m) / sp.sqrt(sp.factorial(j + m) * sp.factorial(j - m)) for m in MS)
cr = sp.symbols("p0:7", real=True); ci = sp.symbols("q0:7", real=True)
u = [cr[k] + sp.I * ci[k] for k in range(7)]
tu = core.theta(u)
prod = sp.expand(form(u, 3) * form(tu, 3))
rho = core.rho6(u)
bad = 0
for Q in range(-6, 7):
    coeffQ = sp.Poly(prod, X, Y).coeff_monomial(X ** (6 + Q) * Y ** (6 - Q)) * sp.sqrt(sp.factorial(6 + Q) * sp.factorial(6 - Q))
    if sp.expand(coeffQ / sp.sqrt(924) - rho[Q]) != 0:
        bad += 1
ok &= report("rho_6(u)_Q = (1/sqrt 924) x [Fock coordinate Q of p_u p_{Theta u}] for symbolic u, all Q", bad == 0,
             "mismatches %d" % bad)

# ------------------------------------------------------------------ (c) maximum
rho0 = sum(core.cg(3, m, 3, -m, 0, 0) * u[IDX[m]] * tu[IDX[-m]] for m in MS)
ok &= report("|rho_0(u)|^2 = |u|^4 / 7 exactly (so r6 <= 1 - 1/7 = 6/7)",
             sp.expand(sp.expand(rho0 * sp.conjugate(rho0)) - sp.expand(sum(x_ * sp.conjugate(x_) for x_ in u)) ** 2 / 7) == 0)
out["max_proved_upper_bound"] = "6/7"

# numerical multi-start search on R^14 (float64, BFGS with analytic gradient)
Nf = sp.lambdify([core.RV], core.N_poly(), "numpy")
gN = sp.lambdify([core.RV], core.grad_N_sym(), "numpy")
def f(xv, sgn):
    n2 = xv @ xv
    return sgn * Nf(xv) / n2 ** 2
def g(xv, sgn):
    n2 = xv @ xv
    return sgn * (np.array(gN(xv), dtype=float) / n2 ** 2 - 4 * Nf(xv) * xv / n2 ** 3)
rng = np.random.default_rng(20260921)
res = {"max": [], "min": []}
NSTART = 400
for sgn, key in ((-1, "max"), (1, "min")):
    for _ in range(NSTART):
        x0 = rng.normal(size=14)
        r_ = so.minimize(f, x0, args=(sgn,), jac=g, method="BFGS", options={"gtol": 1e-12, "maxiter": 5000})
        xv = r_.x / np.linalg.norm(r_.x)
        gt = np.linalg.norm(g(xv, 1))
        res[key].append((sgn * r_.fun, gt))
known = {"1/924": 1 / 924, "3/77": 3 / 77, "200/903": 200 / 903, "75/308": 75 / 308, "9/35": 9 / 35,
         "24/77": 24 / 77, "100/231": 100 / 231, "463/924": 463 / 924}
for key in ("max", "min"):
    vals = np.array([v for v, _ in res[key]])
    best = vals.max() if key == "max" else vals.min()
    clusters = {}
    for v, gt in res[key]:
        lab = min(known, key=lambda k: abs(known[k] - v))
        lab = lab if abs(known[lab] - v) < 1e-9 else "other %.12f" % v
        clusters[lab] = clusters.get(lab, 0) + 1
    print("search for local %s (%d starts): best %.16f ; end values: %s" % (key, NSTART, best, clusters))
    out["search_" + key] = {"best": repr(best), "clusters": clusters, "max_final_grad": repr(max(gt for _, gt in res[key]))}
ok &= report("search: best local max equals 463/924 to 1e-12", abs(max(v for v, _ in res["max"]) - 463 / 924) < 1e-12)
ok &= report("search: no start exceeded 463/924 + 1e-12", all(v < 463 / 924 + 1e-12 for v, _ in res["max"]))
ok &= report("search: best local min equals 1/924 to 1e-12 (consistent with the proof)",
             abs(min(v for v, _ in res["min"]) - 1 / 924) < 1e-12)

# second level of the symmetric-extension bound (numerical, float64; NOT a certificate)
Kn = np.zeros((49, 49), dtype=complex)
for a, b, e_, f_ in itertools.product(range(7), repeat=4):
    aa, bb = sorted((a, b)); ee, ff = sorted((e_, f_))
    Kn[a * 7 + b, e_ * 7 + f_] = float(P.coeff_monomial(d[aa] * d[bb] * c[ee] * c[ff])) / ((1 if aa == bb else 2) * (1 if ee == ff else 2))
def symproj(k):
    n = 7 ** k
    Pm = np.zeros((n, n))
    idx = np.arange(n).reshape([7] * k)
    perms = list(itertools.permutations(range(k)))
    for p in perms:
        Pm[np.arange(n), np.transpose(idx, p).reshape(-1)] += 1
    return Pm / len(perms)
P3 = symproj(3)
A3 = P3 @ np.kron(Kn, np.eye(7)) @ P3
b3 = np.linalg.eigvalsh((A3 + A3.conj().T) / 2).max()
print("numerical level-3 symmetric-extension upper bound on max r6: %.12f (not certified)" % b3)
out["level3_bound_numerical"] = repr(b3)
json.dump(out, open(os.path.join(HERE, "out", "item8.json"), "w"), indent=1)
print("ITEM8", "ALLPASS" if ok else "SOMEFAIL")
