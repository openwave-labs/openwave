"""Item E.2: the linearization at a member of a rigid family, restricted to E_0.

Setting (worklist Section 3, Block E). E_0 = { psi_u(g) = u^T pi(g) : u in C^d }, pi = D^{n/2} unitary, d = n+1.
Pointwise |psi_z(g)|^2 = |z|^2, hence N(psi_z) = psi_{|z|^2 z}: the cubic term maps E_0 into E_0 with
coefficient map N_c(z) = |z|^2 z on C^d (checked numerically with explicit D^j in s04).
With psi = e^{i w T}(psi_v + zeta), zeta = psi_u(T) in E_0, -Lap = lambda0 on E_0:
    c^-2 (u'' + 2 i w u' - w^2 (v+u)) + lambda0 (v+u) + g N_c(v+u) = 0,
and the member condition w^2/c^2 = lambda0 + g|v|^2 is used to eliminate lambda0.
First-order real system in (Re u, Im u, Re u', Im u'), real dimension 4d; eigenvalues are those of its
real 4d x 4d generator A, d/dT (u,u') = A (u,u').
Two independent routes:
  (S) sympy: exact Jacobian at v = r e_1 (allowed by U(d) invariance of the equation), char poly, factors,
      algebraic and geometric multiplicities;
  (N) numpy: A by central finite differences of the full nonlinear right-hand side at a random complex v
      (not along e_1), eigenvalues and ranks compared with the closed form.
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import sympy as sp
from checks import check, summary, PLANT

rig = json.load(open(os.path.join(HERE, "out", "s02.json")))
w, g, c, r = sp.symbols("omega g_4 c r", positive=True)  # r = |v|
mu = sp.symbols("mu")
K = 2 * g * c ** 2 * r ** 2
Omega = sp.sqrt(4 * w ** 2 + K)


def expected(d):
    """closed-form hypothesis: {eigenvalue: (alg, geom)}"""
    e = {sp.Integer(0): (2 * d, 2 * d - 1), sp.I * Omega: (1, 1), -sp.I * Omega: (1, 1)}
    if d > 1:
        e[2 * sp.I * w] = (d - 1, d - 1)
        e[-2 * sp.I * w] = (d - 1, d - 1)
    return e


def Nc_sym(zr, zi):
    n2 = sum(a * a + b * b for a, b in zip(zr, zi))
    if PLANT == "frozen":
        n2 = r ** 2  # planted: |z|^2 frozen at |v|^2 -> member still exact, drops the 2 Re<u,v> v term
    fac = 2 if PLANT == "cubic" else 1  # planted defect: wrong nonlinearity
    return [fac * n2 * a for a in zr], [fac * n2 * b for b in zi]


def A_sym(d):
    lam0 = w ** 2 / c ** 2 - g * r ** 2
    x = sp.symbols(f"x0:{d}", real=True)
    y = sp.symbols(f"y0:{d}", real=True)
    X = sp.symbols(f"X0:{d}", real=True)
    Y = sp.symbols(f"Y0:{d}", real=True)
    vr = [r] + [0] * (d - 1)
    vi = [0] * d
    zr = [vr[a] + x[a] for a in range(d)]
    zi = [vi[a] + y[a] for a in range(d)]
    Nr, Ni = Nc_sym(zr, zi)
    # u'' = -2 i w u' + w^2 z - c^2 lam0 z - c^2 g N(z);  -2 i w (X + iY) = 2 w Y - 2 i w X
    accr = [2 * w * Y[a] + w ** 2 * zr[a] - c ** 2 * lam0 * zr[a] - c ** 2 * g * Nr[a] for a in range(d)]
    acci = [-2 * w * X[a] + w ** 2 * zi[a] - c ** 2 * lam0 * zi[a] - c ** 2 * g * Ni[a] for a in range(d)]
    state = list(x) + list(y) + list(X) + list(Y)
    rhs = list(X) + list(Y) + accr + acci
    at0 = {s: 0 for s in state}
    resid = [sp.simplify(e.subs(at0)) for e in rhs]
    J = sp.Matrix(rhs).jacobian(state).subs(at0)
    return sp.simplify(J), resid


results = {}
all_sym_ok = True
for lab, info in rig.items():
    d = info["dim"]
    A, resid = A_sym(d)
    check(f"E.2 {lab}: the member solves the reduced equation (residual 0)", all(e == 0 for e in resid))
    cp = sp.factor(A.charpoly(mu).as_expr())
    exp = expected(d)
    target = sp.Integer(1)
    for ev, (alg, _) in exp.items():
        target *= (mu - ev) ** alg
    cp_ok = sp.simplify(sp.expand(A.charpoly(mu).as_expr() - target)) == 0
    check(f"E.2 {lab}: char poly equals prod (mu - ev)^alg of the closed form", cp_ok, str(cp))
    geo = {}
    for ev, (alg, gm) in exp.items():
        M = (A - ev * sp.eye(4 * d)).applyfunc(sp.simplify)
        geo[ev] = 4 * d - M.rank(simplify=True)
    geo_ok = all(geo[ev] == exp[ev][1] for ev in exp)
    check(f"E.2 {lab}: geometric multiplicities (symbolic rank)", geo_ok, str({str(k): v for k, v in geo.items()}))
    # second exact evaluation of ranks at rational parameter values where Omega is rational
    vals = {w: sp.Integer(1), g: sp.Integer(1), c: sp.Integer(1), r: sp.sqrt(6)}  # Omega = sqrt(4+12) = 4
    An = A.subs(vals)
    geo2 = {ev.subs(vals): 4 * d - (An - ev.subs(vals) * sp.eye(4 * d)).rank() for ev in exp}
    check(f"E.2 {lab}: geometric multiplicities at (w,g,c,|v|)=(1,1,1,sqrt6), exact ranks", all(geo2[ev.subs(vals)] == exp[ev][1] for ev in exp), str(geo2))
    results[lab] = {"d": d, "n": info["n"], "charpoly": str(cp),
                    "eigenvalues": {str(ev): {"algebraic": exp[ev][0], "geometric": geo[ev]} for ev in exp}}
    print(f"{lab}: A ({4*d}x{4*d}) char poly = {cp}")
    for ev in exp:
        print(f"    eigenvalue {ev}: alg {exp[ev][0]}, geom {geo[ev]}")


# ---------- (N) numeric, random complex v, finite differences of the full nonlinear RHS ----------
rng = np.random.default_rng(7)


def rhs_num(state, v, om, gg, cc, lam0):
    d = len(v)
    u = state[:d] + 1j * state[d:2 * d]
    up = state[2 * d:3 * d] + 1j * state[3 * d:]
    z = v + u
    N = np.vdot(z, z).real * z
    acc = -2j * om * up + om ** 2 * z - cc ** 2 * lam0 * z - cc ** 2 * gg * N
    return np.concatenate([up.real, up.imag, acc.real, acc.imag])


for lab, info in rig.items():
    d, n = info["dim"], info["n"]
    for trial in range(2):
        v = rng.normal(size=d) + 1j * rng.normal(size=d)
        gg, cc = 1.0, rng.uniform(0.5, 2.0)
        lam0 = n * (n + 2)  # R = 1
        om = cc * np.sqrt(lam0 + gg * np.vdot(v, v).real)
        if PLANT == "omega":
            om *= 1.01  # planted: not a member
        h = 1e-6
        f0 = rhs_num(np.zeros(4 * d), v, om, gg, cc, lam0)
        A = np.zeros((4 * d, 4 * d))
        for k in range(4 * d):
            e = np.zeros(4 * d); e[k] = h
            A[:, k] = (rhs_num(e, v, om, gg, cc, lam0) - rhs_num(-e, v, om, gg, cc, lam0)) / (2 * h)
        rv = np.linalg.norm(v)
        Om = np.sqrt(4 * om ** 2 + 2 * gg * cc ** 2 * rv ** 2)
        exp_list = [0.0] * (2 * d) + [1j * Om, -1j * Om] + [2j * om] * (d - 1) + [-2j * om] * (d - 1)
        ev = np.linalg.eigvals(A)
        # match as multisets
        rem = list(ev)
        err = 0.0
        for t in exp_list:
            k = int(np.argmin([abs(t - s) for s in rem]))
            err = max(err, abs(t - rem.pop(k)))
        # geometric multiplicity: nullity of A - t I by SVD
        def nullity(t):
            s = np.linalg.svd(A - t * np.eye(4 * d), compute_uv=False)
            return int(np.sum(s < 1e-6 * s[0]))
        geo = {"0": nullity(0), "iOm": nullity(1j * Om)}
        if d > 1:
            geo["2iw"] = nullity(2j * om)
        geo_exp = {"0": 2 * d - 1, "iOm": 1}
        if d > 1:
            geo_exp["2iw"] = d - 1
        # Jordan check at 0: nullity of A^2 equals algebraic multiplicity 2d
        s2 = np.linalg.svd(A @ A, compute_uv=False)
        null2 = int(np.sum(s2 < 1e-6 * s2[0]))
        ok = np.abs(f0).max() < 1e-9 and err < 1e-4 and geo == geo_exp and null2 == 2 * d
        check(f"E.2 {lab}: numeric FD at random v (trial {trial}): member residual, eigenvalues, nullities", ok,
              f"resid {np.abs(f0).max():.1e}, ev err {err:.1e}, nullities {geo}, null(A^2) {null2}")

if PLANT is None:
    json.dump(results, open(os.path.join(HERE, "out", "s03.json"), "w"), indent=1)
summary()
