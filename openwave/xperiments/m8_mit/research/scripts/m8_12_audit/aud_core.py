"""Auditor's own core: CG coefficients, Theta, rho6, N, gradient, Hessian (exact + mpmath).

Written from worklist.md definitions only.  Basis order v3, v2, v1, v0, v-1, v-2, v-3.
Real coordinates: e-directions  v_m  and  i*v_m ;  Re<u,w> is the Euclidean product.
"""
import sympy as sp
import mpmath as mp
import functools

MS = [3, 2, 1, 0, -1, -2, -3]
def idx(m): return 3 - m
I = sp.I

def jminus_coef(m):   # J- v_m = sqrt(12 - m(m-1)) v_{m-1}
    return sp.sqrt(12 - m * (m - 1))
def jplus_coef(m):    # J+ v_m = sqrt(12 - m(m+1)) v_{m+1}
    return sp.sqrt(12 - m * (m + 1))

@functools.lru_cache(None)
def cg6_table():
    """<3 m1; 3 m2 | 6 Q> by lowering |6 6> = v3 (x) v3 with total J- (convention <33;33|66> = +1)."""
    state = {(3, 3): sp.Integer(1)}
    tab = {}
    Q = 6
    while True:
        for k, v in state.items():
            tab[(k[0], k[1], Q)] = v
        if Q == -6:
            break
        new = {}
        for (a, b), v in state.items():
            if a - 1 >= -3:
                new[(a - 1, b)] = new.get((a - 1, b), 0) + v * jminus_coef(a)
            if b - 1 >= -3:
                new[(a, b - 1)] = new.get((a, b - 1), 0) + v * jminus_coef(b)
        norm = sp.sqrt(6 * 7 - Q * (Q - 1))
        # exact: sympy keeps q*sqrt(squarefree) canonical, so after expand each coefficient
        # must be a single such term (asserted, no numerics involved)
        state = {k: sp.expand(v / norm) for k, v in new.items()}
        for k, v in state.items():
            assert (v ** 2).is_Rational, (k, v)
        Q -= 1
    return tab

def cg6(m1, m2, Q):
    if m1 + m2 != Q or abs(m1) > 3 or abs(m2) > 3:
        return sp.Integer(0)
    return cg6_table().get((m1, m2, Q), sp.Integer(0))

def theta(c):
    """(Theta u)_m = (-1)^(3-m) conj(c_{-m});  c indexed by idx."""
    return [(-1) ** (3 - m) * sp.conjugate(c[idx(-m)]) for m in MS]

def B(x, y, Q, conj=sp.conjugate):
    """B_Q(x,y) = sum_m1 CG(m1,Q-m1) x_m1 (Theta y)_{Q-m1};  linear in x, antilinear in y."""
    tot = 0
    for m1 in MS:
        m2 = Q - m1
        if abs(m2) <= 3:
            tot += cg6(m1, m2, Q) * x[idx(m1)] * (-1) ** (3 - m2) * conj(y[idx(-m2)])
    return tot

def rho(c):
    return [B(c, c, Q) for Q in range(-6, 7)]

def N_exact(c):
    return sum(sp.expand(r * sp.conjugate(r)) for r in rho(c))

def nrm2(c):
    return sum(sp.expand(z * sp.conjugate(z)) for z in c)

def rhat(c):
    return sp.radsimp(sp.simplify(N_exact(c) / nrm2(c) ** 2))

# real directions
def basis14():
    out = []
    for k in range(7):
        e = [0] * 7; e[k] = sp.Integer(1); out.append(e)
    for k in range(7):
        e = [0] * 7; e[k] = I; out.append(e)
    return out

def to_real(c):
    c = [sp.expand(z) for z in c]
    return [sp.re(z) for z in c] + [sp.im(z) for z in c]

def from_real(x):
    return [x[k] + I * x[7 + k] for k in range(7)]

def reip(u, w):
    return sp.expand(sum(sp.re(sp.conjugate(a) * b) for a, b in zip(u, w)))

def L_Q(u, e, Q):
    return B(u, e, Q) + B(e, u, Q)

def grad_exact(u):
    """gradient of N at u, as 14 real components (Euclidean)."""
    rh = rho(u)
    g = []
    for e in basis14():
        s = 0
        for qi, Q in enumerate(range(-6, 7)):
            s += 2 * sp.re(sp.expand(sp.conjugate(rh[qi]) * L_Q(u, e, Q)))
        g.append(sp.radsimp(sp.expand(s)))
    return g

def hessN_exact(u):
    """14x14 Hessian of N at u (Euclidean real coordinates)."""
    rh = rho(u)
    E = basis14()
    Ls = [[sp.expand(L_Q(u, e, Q)) for Q in range(-6, 7)] for e in E]
    H = sp.zeros(14, 14)
    for a in range(14):
        for b in range(a, 14):
            s = 0
            for qi, Q in enumerate(range(-6, 7)):
                s += 2 * sp.re(sp.expand(sp.conjugate(Ls[a][qi]) * Ls[b][qi]))
                s += 2 * sp.re(sp.expand(sp.conjugate(rh[qi]) * (B(E[a], E[b], Q) + B(E[b], E[a], Q))))
            v = sp.radsimp(sp.expand(s))
            H[a, b] = v; H[b, a] = v
    return H

# ---------------------------------------------------------------- mpmath route
def mp_cg():
    return {k: mp.mpf(sp.N(v, mp.mp.dps + 20)) for k, v in cg6_table().items()}

def mp_N(c, cg=None):
    cg = cg or mp_cg()
    th = [(-1) ** (3 - m) * mp.conj(c[idx(-m)]) for m in MS]
    tot = mp.mpf(0)
    for Q in range(-6, 7):
        s = mp.mpc(0)
        for m1 in MS:
            m2 = Q - m1
            if abs(m2) <= 3:
                s += cg[(m1, m2, Q)] * c[idx(m1)] * th[idx(m2)]
        tot += abs(s) ** 2
    return tot

def mp_rhat(c, cg=None):
    n2 = sum(abs(z) ** 2 for z in c)
    return mp_N(c, cg) / n2 ** 2

# ---------------------------------------------------------------- rotation generators
def Jz_vec(c):  return [m * c[idx(m)] for m in MS]
def Jp_vec(c):
    out = [0] * 7
    for m in MS:
        if m + 1 <= 3: out[idx(m + 1)] += jplus_coef(m) * c[idx(m)]
    return out
def Jm_vec(c):
    out = [0] * 7
    for m in MS:
        if m - 1 >= -3: out[idx(m - 1)] += jminus_coef(m) * c[idx(m)]
    return out
def Jx_vec(c): return [sp.expand((a + b) / 2) for a, b in zip(Jp_vec(c), Jm_vec(c))]
def Jy_vec(c): return [sp.expand((a - b) / (2 * I)) for a, b in zip(Jp_vec(c), Jm_vec(c))]

def O_gens(u):
    """the four generators of O_u exactly as worklist 1.5 writes them."""
    return [[I * z for z in u],
            [-I * z for z in Jx_vec(u)],
            [-I * z for z in Jy_vec(u)],
            [-I * z for z in Jz_vec(u)]]

def Jmat():
    """7x7 sympy matrices Jx, Jy, Jz."""
    Jz = sp.diag(*[m for m in MS])
    Jp = sp.zeros(7); Jm = sp.zeros(7)
    for m in MS:
        if m + 1 <= 3: Jp[idx(m + 1), idx(m)] = jplus_coef(m)
        if m - 1 >= -3: Jm[idx(m - 1), idx(m)] = jminus_coef(m)
    return (Jp + Jm) / 2, (Jp - Jm) / (2 * I), Jz
