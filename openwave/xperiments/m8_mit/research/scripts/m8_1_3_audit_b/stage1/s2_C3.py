"""Item C.3 (with clause 5).  Exact: h, Mhat, K', signatures, slow polynomials (sympy rationals in the weighted chart);
gamma_u from the exact S^3 Hessian of R0 on K' (cross-checked by the CG route at 40 digits).
Gamma_u at R2's line point: numerical search over SU(2) for the stabiliser of the quaternionic line H u, then the
Sp(1) factor solved for each element; the group is checked for closure; its action on K' and the commutant computed."""
from fractions import Fraction as Fr
import random
import numpy as np
import sympy as sp
from scipy.optimize import minimize
from mpmath import mp
from core import mlist, Fm, KZERO
from reps import SLOTS
from exact_pt import ExactPoint, R
from s3poly import R0_levels, R0_from_levels, R0_hessian_entry, kfrac
from cgroute import CGRoute, CP
from chk import check, save_json, planted

mp.dps = 40
out = {'C.3': {'R2': {}, 'R4': {}, 'R5': {}}}


def p_of_x(x, n):
    N = n + 1
    return [(Fr(int(sp.fraction(x[i])[0]), int(sp.fraction(x[i])[1])), Fr(0),
             Fr(int(sp.fraction(x[N + i])[0]), int(sp.fraction(x[N + i])[1])), Fr(0)) for i in range(N)]


def gamma_on_K(E, Kvecs, label):
    """exact gamma_u = kappa^-2 (Hess R0(u) - 6 R0(u)) on K', relative to the inner product."""
    name, n = E.name, E.n
    pu = p_of_x(list(E.x0), n)
    lv = R0_levels(name, pu)
    R0x = R0_from_levels(name, lv)
    assert R0x[1:] == (0, 0, 0)
    R0u = R(R0x[0]) / E.r ** 3
    m = len(Kvecs)
    A = sp.zeros(m, m)
    for a in range(m):
        for b in range(a, m):
            v, v5 = R0_hessian_entry(name, pu, p_of_x(list(Kvecs[a]), n), p_of_x(list(Kvecs[b]), n))
            assert v5 == 0
            A[a, b] = A[b, a] = R(v) / E.r ** 2
    Gram = sp.Matrix(m, m, lambda a, b: (Kvecs[a].T * E.G * Kvecs[b])[0])
    op = Gram.inv() * (A - 6 * R0u * Gram) / E.hu ** 2
    # numerical cross-check of the Hessian entries with the CG route
    cg = CGRoute(name, mp)
    ms = mlist(n); N = n + 1
    sq = mp.sqrt(mp.mpf(int(sp.fraction(E.r)[0])) / int(sp.fraction(E.r)[1]))

    def tow(x, scale):
        return [CP(mp.mpf(int(sp.fraction(x[i])[0])) / int(sp.fraction(x[i])[1]) * mp.sqrt(Fm(n, ms[i])) / scale,
                   mp.mpf(int(sp.fraction(x[N + i])[0])) / int(sp.fraction(x[N + i])[1]) * mp.sqrt(Fm(n, ms[i])) / scale) for i in range(N)]
    u = tow(list(E.x0), sq)
    g, H = cg.R0_hessian_entry_free = cg.R0_grad_hess(u, [tow(list(v), 1) for v in Kvecs])
    err = max(abs(H[a][b] - mp.mpf(int(sp.fraction(A[a, b])[0])) / int(sp.fraction(A[a, b])[1])) for a in range(m) for b in range(m))
    if planted('C3'):
        err += 1e-10
    check(f'C.3 {label}: exact Hess R0 on K\' agrees with the CG route', err < mp.mpf(10) ** -30, mp.nstr(err, 3))
    errR = abs(cg.R0(u) - mp.mpf(int(sp.fraction(R0u)[0])) / int(sp.fraction(R0u)[1]))
    check(f'C.3 {label}: exact R0(u) agrees with the CG route', errR < mp.mpf(10) ** -30, mp.nstr(errR, 3))
    grad_err = max(abs(x) for x in g)
    check(f'C.3 {label}: grad R0(u) vanishes along K\' (CG route)', grad_err < mp.mpf(10) ** -30, mp.nstr(grad_err, 3))
    return op, R0u, A, Gram


# ===================================================================== v1 in R4 and R5
for name in ['R4', 'R5']:
    d, n = SLOTS[name]; N = n + 1
    ms = mlist(n)
    x0 = [0] * (2 * N); x0[ms.index(1)] = 1
    E = ExactPoint(name, x0)
    check(f'C.3 v1 {name}: v1 is critical (exact)', E.critical)
    sig = E.signature()
    Kp = E.kernel_in_N()
    # identify K' as weight spaces
    supp = sorted(set(ms[i % N] for v in Kp for i in range(2 * N) if v[i] != 0))
    # circle generator: d/dtheta of R_z(theta) e^{i theta}: w_m -> i(1-m) w_m  (on p as well)
    circ = sp.zeros(2 * N, 2 * N)
    for i, m in enumerate(ms):
        circ[N + i, i] = R(1 - m); circ[i, N + i] = -R(1 - m)
    Kb = sp.Matrix.hstack(*Kp)
    # restriction of the circle generator to K'
    sol = (Kb.T * Kb).inv() * Kb.T * circ * Kb
    check(f'C.3 v1 {name}: K\' is invariant under the circle', circ * Kb == Kb * sol)
    weights = [1 - m for m in supp]
    print(f'v1 {name}: critical {E.critical}; h(u) = {E.hu}; k = {E.k}; r_u = {E.r_u}; signature {sig}')
    print(f'   K\' = span of the m = {supp} components (real dim {len(Kp)}); circle generator on K\' = {sol.tolist()} -> weight {weights}')
    op, R0u, A, Gram = gamma_on_K(E, Kp, f'v1 {name}')
    is_scalar = op == op[0, 0] * sp.eye(op.shape[0])
    check(f'C.3 v1 {name}: gamma_u is a multiple of the identity on K\'', is_scalar, str(op.tolist()))
    print(f'   R0(u) = {R0u};  gamma_u on K\' = {op.tolist()}  (V = R = 1; i.e. gamma_u = {op[0, 0]} R^2)')
    # Shat off K': the complement spanned by the other chart coordinates
    off = [i for i in range(2 * N) if ms[i % N] not in supp]
    Sfull = E.S
    block_ok = all(Sfull[i, k] == 0 for i in range(2 * N) for k in off if i not in off)
    check(f'C.3 v1 {name}: Shat preserves the components off K\'', block_ok)
    Soff = Sfull.extract(off, off)
    zoff, qoff = E.slow(Soff)
    zfull, qfull = E.slow()
    print(f'   off K\': zero multiplicity {zoff}, slow polynomial {qoff};  (whole E0: zero multiplicity {zfull}, slow {qfull})')
    out['C.3'][name]['v1'] = {
        'critical': True, 'h(u)': str(E.hu), 'k': E.k, 'r_u': E.r_u, 'signature': sig,
        "K'": f'the m = {supp[0]} weight space (Re p_{supp[0]}, Im p_{supp[0]}), real dimension {len(Kp)}',
        "dim K'": len(Kp), 'weight under the circle': str(weights[0]),
        'R0(u) (V=R=1)': str(R0u),
        'gamma_u': f'{op[0, 0]} R^2' if is_scalar else str(op.tolist()),
        'gamma_u (number, V=R=1)': str(op[0, 0]) if is_scalar else None,
        "zero multiplicity off K'": zoff, "slow polynomial off K'": str(qoff),
        'zero multiplicity on all of E0 (for reference)': zfull}

# ===================================================================== R2's point of the line {5/2, -3/2}
name = 'R2'; d, n = SLOTS[name]; N = n + 1
ms = mlist(n)
# w = sqrt(3/4) v_{5/2} + sqrt(1/4) v_{-3/2}  ->  p_{5/2} = p_{-3/2} = 1/sqrt(960): chart x0 = e_{5/2} + e_{-3/2}, r = 960
x0 = [0] * (2 * N); x0[ms.index(Fr(5, 2))] = 1; x0[ms.index(Fr(-3, 2))] = 1
E = ExactPoint(name, x0)
check('C.3 R2 line: the given point equals x0/sqrt(r) (p_{5/2} = p_{-3/2} = 960^{-1/2})',
      E.r == 960 and sp.Rational(3, 4) / 720 == sp.Rational(1, 960) and sp.Rational(1, 4) / 240 == sp.Rational(1, 960))
check('C.3 R2 line: critical (exact)', E.critical)
sig = E.signature()
Kp = E.kernel_in_N()
print(f'R2 line {{5/2,-3/2}}: critical {E.critical}; h(u) = {E.hu}; k = {E.k}; r_u = {E.r_u}; signature {sig}; dim K\' = {len(Kp)}')
for v in Kp:
    print('   K\' basis vector (chart x):', list(v.T))
op, R0u, A, Gram = gamma_on_K(E, Kp, 'R2 line')
is_scalar = op == op[0, 0] * sp.eye(op.shape[0])
check('C.3 R2 line: gamma_u is a multiple of the identity on K\'', is_scalar, str(op.tolist()))
print(f'   R0(u) = {R0u};  gamma_u on K\' = {op.tolist()}')

# ---- Gamma_u, numerically (double precision search, then verification)
Fv = np.array([float(Fm(n, m)) for m in ms])
u = np.zeros(N, complex); u[ms.index(Fr(5, 2))] = np.sqrt(3 / 4); u[ms.index(Fr(-3, 2))] = np.sqrt(1 / 4)
sgn = np.array([(-1) ** int(Fr(7, 2) - m) for m in ms])
rev = [ms.index(-m) for m in ms]


def theta(w):
    return sgn * np.conj(w[rev])


def Dnum(a, b):
    from math import comb
    M = np.zeros((N, N), complex)
    for ip in range(N):
        al = n - ip; be = ip
        for t1 in range(al + 1):
            c1 = comb(al, t1) * a ** t1 * b ** (al - t1)
            for t2 in range(be + 1):
                c2 = comb(be, t2) * (-np.conj(b)) ** t2 * np.conj(a) ** (be - t2)
                M[n - (t1 + t2), ip] += c1 * c2
    S = np.sqrt(Fv)
    return M * S[:, None] / S[None, :]


Bq = np.stack([u, theta(u)], axis=1)
Qb, _ = np.linalg.qr(Bq)
Proj = Qb @ Qb.conj().T


def resvec(v):
    v = v / np.linalg.norm(v)
    a = v[0] + 1j * v[1]; b = v[2] + 1j * v[3]
    Du = Dnum(a, b) @ u
    r = Du - Proj @ Du
    return np.concatenate([r.real, r.imag])


from scipy.optimize import least_squares
rng = np.random.default_rng(2024)
sols = []
for trial in range(600):
    v0 = rng.normal(size=4)
    r = least_squares(resvec, v0, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    if np.linalg.norm(r.fun) < 1e-11:
        v = r.x / np.linalg.norm(r.x)
        if v[0] < 0 or (abs(v[0]) < 1e-9 and v[np.argmax(np.abs(v) > 1e-9)] < 0):
            v = -v
        if not any(min(np.linalg.norm(v - s), np.linalg.norm(v + s)) < 1e-7 for s in sols):
            sols.append(v)
print(f'   stabiliser search: {len(sols)} elements of SU(2)/(+-1) preserve H u')


def quat_map_matrix(a, b):
    """the real-linear map q o D(g) fixing u, as a real 2N x 2N matrix on (Re w, Im w), with q in Sp(1) solved."""
    Dg = Dnum(a, b)
    Du = Dg @ u
    # q = c0 + c1 i + c2 Theta + c3 Theta i ; solve q(Du) = u
    cols = [Du, 1j * Du, theta(Du), theta(1j * Du)]
    Am = np.stack([np.concatenate([c.real, c.imag]) for c in cols], axis=1)
    rhs = np.concatenate([u.real, u.imag])
    c, *_ = np.linalg.lstsq(Am, rhs, rcond=None)
    res = np.linalg.norm(Am @ c - rhs)
    T = np.zeros((2 * N, 2 * N))
    for kk in range(2 * N):
        e = np.zeros(2 * N); e[kk] = 1
        w = e[:N] + 1j * e[N:]
        y = Dg @ w
        z = c[0] * y + c[1] * 1j * y + c[2] * theta(y) + c[3] * theta(1j * y)
        T[:, kk] = np.concatenate([z.real, z.imag])
    return T, res, c


maps = []
for v in sols:
    T, res, c = quat_map_matrix(v[0] + 1j * v[1], v[2] + 1j * v[3])
    maps.append(T)
    assert res < 1e-9 and abs(np.linalg.norm(c) - 1) < 1e-9
# distinct as maps
uniq = []
for T in maps:
    if not any(np.linalg.norm(T - S) < 1e-7 for S in uniq):
        uniq.append(T)
closed = len(uniq) > 0 and all(any(np.linalg.norm(A1 @ A2 - S) < 1e-7 for S in uniq) for A1 in uniq for A2 in uniq)
check('C.3 R2 line: the stabiliser found is non-empty and closed under composition', closed)
check('C.3 R2 line: the identity is among the elements found', any(np.linalg.norm(T - np.eye(2 * N)) < 1e-7 for T in uniq))
uvec = np.concatenate([u.real, u.imag])
check('C.3 R2 line: every element fixes u and is orthogonal', all(np.linalg.norm(T @ uvec - uvec) < 1e-9 and np.linalg.norm(T.T @ T - np.eye(2 * N)) < 1e-9 for T in uniq))
orders = sorted(int(min(k for k in range(1, 50) if np.linalg.norm(np.linalg.matrix_power(T, k) - np.eye(2 * N)) < 1e-7)) for T in uniq)
print(f'   |Gamma_u| (as maps of E0) = {len(uniq)}; element orders {orders}')
# action on K' (convert chart vectors to orthonormal w-coordinates)
Sq = np.sqrt(np.concatenate([Fv, Fv]))
Kw = np.stack([np.array([float(x) for x in v]) * Sq for v in Kp], axis=1)
Qk, _ = np.linalg.qr(Kw)
inv_ok = all(np.linalg.norm(T @ Qk - Qk @ (Qk.T @ T @ Qk)) < 1e-8 for T in uniq)
check("C.3 R2 line: K' is invariant under Gamma_u", inv_ok)
reps_K = [Qk.T @ T @ Qk for T in uniq]
# commutant dimension: X R = R X for all R
m = Qk.shape[1]
rows = []
for Rm in reps_K:
    for a in range(m):
        for b in range(m):
            row = np.zeros(m * m)
            for c_ in range(m):
                row[a * m + c_] += Rm[c_, b]
                row[c_ * m + b] -= Rm[a, c_]
            rows.append(row)
sv = np.linalg.svd(np.array(rows), compute_uv=False)
comm_dim = int(np.sum(sv < 1e-8))
chars_K = sorted(round(float(np.trace(Rm)), 6) for Rm in reps_K)
print(f'   commutant of Gamma_u on K\' has dimension {comm_dim}; traces on K\': {chars_K}')
# also: is the action faithful / nontrivial
check("C.3 R2 line: commutant dimension computed with a clear singular-value gap", sv[-comm_dim - 1] > 1e-3 if comm_dim < len(sv) else True,
      f'smallest sv {sv[-3:]}')
out['C.3']['R2']['line {5/2, -3/2}'] = {
    'critical': True, 'h(u)': str(E.hu), 'k': E.k, 'r_u': E.r_u, 'signature': sig, 'n_-': sig[0],
    '|Gamma_u|': len(uniq), 'element orders': orders,
    "K' basis (chart x = (Re p_7/2..Re p_-7/2, Im p_7/2..Im p_-7/2))": [[str(c) for c in v] for v in Kp],
    "dim K'": len(Kp),
    "Gamma_u acts on K' absolutely irreducibly": comm_dim == 1, "commutant dimension": comm_dim,
    'R0(u) (V=R=1)': str(R0u),
    'gamma_u': f'{op[0, 0]} R^2' if is_scalar else str(op.tolist()),
    'gamma_u (number, V=R=1)': str(op[0, 0]) if is_scalar else None}
save_json('out_C3.json', out)
