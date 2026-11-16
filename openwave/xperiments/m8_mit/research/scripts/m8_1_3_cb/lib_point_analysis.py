"""Shared analysis of a point u in a slot: criticality, K_u, k, r_u, N_u, signature of M^ on N_u,
K' = ker M^ cap N_u, slow matrix S^ = i M^.  Real coordinates x = (Re w, Im w) on orthonormal CS
components; Re<.,.> is the Euclidean product; the complex structure i is lib_engine.Jcx."""
import mpmath as mp
from lib_engine import Slot, grad, hess, Jcx, to_real, orbit_tangents, orth_basis


def complement(vecs, dim, tol):
    A = mp.matrix(dim, len(vecs))
    for c, v in enumerate(vecs):
        for r in range(dim):
            A[r, c] = v[r]
    U, S, V = mp.svd_r(A, full_matrices=True)
    rank = sum(1 for k in range(min(dim, len(vecs))) if S[k] > tol)
    return [U[:, k] for k in range(rank, dim)]


def mat_from_cols(cols, dim):
    B = mp.matrix(dim, len(cols))
    for c, v in enumerate(cols):
        for r in range(dim):
            B[r, c] = v[r]
    return B


def analyze(slot_obj, u, tol_exp=None):
    S = slot_obj
    n = S.n
    dim = 2 * n
    dps = mp.mp.dps
    tol = mp.mpf(10) ** (-(dps // 2)) if tol_exp is None else mp.mpf(10) ** (-tol_exp)
    x = to_real(u)
    xl = [x[k] for k in range(dim)]
    h0 = S.hx(xl)
    g = grad(S.hx, xl, 4)
    crit = mp.norm(g - 4 * h0 * x)
    H = hess(S.hx, xl, 4)
    Mh = H - 4 * h0 * mp.eye(dim)
    tang = orbit_tangents(S, u)
    Kb, svals = orth_basis(tang, tol)
    k = len(Kb)
    Jm = Jcx(n)
    KB = mat_from_cols(Kb, dim) if k else None
    if k:
        Sig = KB.T * Jm * KB
        sv = mp.svd_r(Sig, compute_uv=False)
        rank_sig = sum(1 for s in sv if s > tol)
    else:
        rank_sig = 0
    r_u = k - rank_sig
    Mh_on_K = max([mp.norm(Mh * v) for v in Kb] + [mp.mpf(0)])
    Nb = complement([x] + Kb, dim, tol)
    NB = mat_from_cols(Nb, dim)
    MN = NB.T * Mh * NB
    MN = (MN + MN.T) / 2
    ev, EV = mp.eigsy(MN)
    neg = sum(1 for e in ev if e < -tol)
    zer = sum(1 for e in ev if abs(e) <= tol)
    pos = sum(1 for e in ev if e > tol)
    Kp = [NB * EV[:, i] for i in range(len(ev)) if abs(ev[i]) <= tol]
    return dict(x=x, h=h0, crit_residual=crit, H=H, Mh=Mh, Kb=Kb, k=k, r_u=r_u, Mh_on_K=Mh_on_K,
                Nb=Nb, NB=NB, signature=(neg, zer, pos), evN=ev, Kp=Kp, tol=tol, J=Jm, tang_svals=svals)


def second_derivs_along(f, x, basis, deg):
    """matrix of D^2 f(x)[b_a, b_b] for a list of directions."""
    from lib_engine import _coeffs_along
    m = len(basis)
    xl = [x[k] for k in range(len(x))]
    c2 = [_coeffs_along(f, xl, [b[r] for r in range(len(x))], deg)[2] for b in basis]
    Hm = mp.matrix(m, m)
    for a in range(m):
        Hm[a, a] = 2 * c2[a]
        for b in range(a + 1, m):
            v = [basis[a][r] + basis[b][r] for r in range(len(x))]
            Hm[a, b] = Hm[b, a] = _coeffs_along(f, xl, v, deg)[2] - c2[a] - c2[b]
    return Hm
