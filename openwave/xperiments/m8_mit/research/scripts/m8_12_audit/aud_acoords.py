"""Auditor's rational-coordinate machinery (a_m = sqrt(C(6,3+m)) c_m); identity with aud_core.N checked in a5.py."""
import sympy as sp
from aud_core import MS, idx, I

P = sp.symbols('p3 p2 p1 p0 pm1 pm2 pm3', real=True)
Qs = sp.symbols('q3 q2 q1 q0 qm1 qm2 qm3', real=True)
A = [P[k] + I * Qs[k] for k in range(7)]
VARS = list(P) + list(Qs)
Cb = {m: sp.binomial(6, 3 + m) for m in MS}
S = {m: sp.sqrt(Cb[m]) for m in MS}

def _Na():
    tot = 0
    for Q in range(-6, 7):
        s = 0
        for m1 in MS:
            m2 = Q - m1
            if abs(m2) <= 3:
                s += (-1) ** (3 - m2) * A[idx(m1)] * sp.conjugate(A[idx(-m2)])
        s = sp.expand(s)
        tot += sp.expand(s * sp.conjugate(s)) / sp.binomial(12, 6 + Q)
    return sp.expand(tot)
Na = _Na()
HessNa = sp.hessian(Na, VARS)
G0 = sp.diag(*([1 / Cb[m] for m in MS] * 2))

def realv(a): return sp.Matrix([sp.re(sp.expand(z)) for z in a] + [sp.im(sp.expand(z)) for z in a])
def c_to_a(c): return [S[m] * c[idx(m)] for m in MS]

def form_at_c(c):
    """For a c-vector c (any norm): returns (Mform_a, x_a, n2, N) with Mform_a the matrix of H_u in a-coords
    at u = c/|c|, so that H_u(e,f) = e_a^T Mform_a f_a for e,f in T_u given in c-coords (e_a = realv(c_to_a(e)))."""
    a = c_to_a(c)
    x = realv(a)
    n2 = sp.expand((x.T * G0 * x)[0])
    sub = dict(zip(VARS, list(x)))
    Nval = sp.expand(Na.subs(sub))
    Hs = HessNa.subs(sub).applyfunc(sp.expand)
    return (Hs / n2 - 4 * Nval / n2 ** 2 * G0).applyfunc(sp.expand), x, n2, Nval

def Ogens_c(u):
    from aud_core import O_gens
    return O_gens(u)
