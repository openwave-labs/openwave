"""Exact criticality test for h = c0 ||w||^4 + c6 ||[w (x) Theta w]_6||^2 at a unit u (any c6 != 0):
grad h = 4 h u  <=>  dF/dconj(w) = 2 F(u) u, F = ||[w (x) Theta w]_6||^2  (the ||w||^4 part satisfies it identically)."""
from common import *

def simp(x):
    return sp.radsimp(sp.expand(x))

_gradcache = {}
def gradF(j):
    if j in _gradcache:
        return _gradcache[j]
    n = int(2 * j + 1)
    w = sp.symbols(f"w0:{n}"); wb = sp.symbols(f"wb0:{n}")
    U = theta_U(j)
    Tw = U * sp.Matrix(wb)        # Theta w  (U real)
    Twb = U * sp.Matrix(w)        # conj(Theta w)
    W6 = W_matrix(6, j, j)
    B = W6 * sp.Matrix([w[a] * Tw[c] for a in range(n) for c in range(n)])
    Bb = W6 * sp.Matrix([wb[a] * Twb[c] for a in range(n) for c in range(n)])
    F = sp.expand(sum((B[k] * Bb[k] for k in range(B.rows)), sp.Integer(0)))
    g = [sp.diff(F, wb[k]) for k in range(n)]
    _gradcache[j] = (w, wb, F, g)
    return _gradcache[j]

def critical_exact(j, u):
    """(is_critical, residual vector, F(u)) for a unit u, exact."""
    w, wb, F, g = gradF(j)
    sub = {w[k]: u[k] for k in range(len(u))}
    sub.update({wb[k]: sp.conjugate(u[k]) for k in range(len(u))})
    Fv = simp(F.subs(sub))
    res = [simp(g[k].subs(sub) - 2 * Fv * u[k]) for k in range(len(u))]
    return all(r == 0 for r in res), res, Fv
