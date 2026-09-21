"""Hessian machinery for item 5 (formula proved in RETURN.md, item 5):

    for unit u and e in T_u:  H_u(e, e) = e^T (Hess N(u) - 4 N(u) I) e,   M_u := Hess N(u) - 4 N(u) I,

with N(u) = sum_Q |rho_6(u)_Q|^2 the quartic numerator of r-hat_6, in the real
coordinates (x_3..x_-3, y_3..y_-3) of core.RV.
"""
import sympy as sp
import core

LAM = sp.Symbol("lam")
Jz, Jp, Jm, Jx, Jy = core.Jmats()


def simp(t):
    return core.exact(t)


def unit_real(c):
    nrm2 = sum(sp.expand(z * sp.conjugate(z)) for z in c)
    cu = [simp(z / sp.sqrt(nrm2)) for z in c]
    return cu, sp.Matrix([simp(v) for v in core.to_real(cu)])


def N_at(r):
    return simp(core.N_poly().xreplace(dict(zip(core.RV, list(r)))))


def M_u(r, drop_4N=False):
    H = sp.Matrix(14, 14, lambda i, j: 0)
    h = core.hess_N_sym()
    sub = dict(zip(core.RV, list(r)))
    for i in range(14):
        for j in range(i, 14):
            H[i, j] = H[j, i] = simp(h[i][j].xreplace(sub))
    if core.PLANT == "hess_no4N" or drop_4N:
        return H
    return H - 4 * N_at(r) * sp.eye(14)


def cvec_to_real(c):
    return sp.Matrix([simp(v) for v in core.to_real(list(c))])


def O_gens(cu):
    """the four generators of O_u exactly as section 1.5 writes them (real 14-vectors)."""
    u = sp.Matrix(cu)
    gens = {"i u": sp.I * u, "-i Jx u": -sp.I * Jx * u, "-i Jy u": -sp.I * Jy * u, "-i Jz u": -sp.I * Jz * u}
    return {k: cvec_to_real([simp(t) for t in v]) for k, v in gens.items()}


def N_basis(r, og):
    A = sp.Matrix.hstack(r, *og.values()).T
    return A.nullspace(simplify=True)


def restricted(Mu, B):
    """Gram G = B^T B and compressed form H = B^T M B, and the characteristic polynomial of the
    restriction of H_u to span(B) in an orthonormal basis: det(lam G - H) / det G."""
    Bm = sp.Matrix.hstack(*B)
    G = (Bm.T * Bm).applyfunc(simp)
    H = (Bm.T * Mu * Bm).applyfunc(simp)
    cp = sp.factor(sp.expand(simp((LAM * G - H).det(method="berkowitz")) / simp(G.det(method="berkowitz"))))
    return G, H, cp


def signature_from_charpoly(cp, deg):
    """(n-, n0, n+) with multiplicity: square-free factorisation, then Sturm counting on each factor
    (all roots are real because the restricted form is symmetric)."""
    P = sp.Poly(sp.expand(cp), LAM)
    nneg = n0 = npos = 0
    for f, mult in sp.sqf_list(P)[1]:
        f = sp.Poly(f, LAM)
        z = 0
        while f.eval(0) == 0:
            f = sp.Poly(sp.quo(f.as_expr(), LAM), LAM)
            z += 1
        n0 += z * mult
        if f.degree() > 0:
            p_ = f.count_roots(0, None)
            npos += p_ * mult
            nneg += (f.degree() - p_) * mult
    assert n0 + npos + nneg == deg, (n0, npos, nneg, deg)
    return nneg, n0, npos
