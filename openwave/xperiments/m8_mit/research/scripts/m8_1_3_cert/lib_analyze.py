"""Certified quantities at a certified box (Block D / B.4), by interval arithmetic.

Structural facts used (each proved exactly elsewhere: s1_quartic.py checks h is invariant and
the generators are G-skew with X_1^2 = -1 commuting with the rotations):
 * at the true critical point u (|u|_G = 1, lam = 4h(u)):  A = Hess h(u) - 4h(u) G satisfies
   A k_i = 0 (k_i = X_i u) and A u = 8 h(u) G u (Euler: Hess h.u = 3 grad h);
 * S = X_1 G^{-1} A (the slow matrix i*M_hat in the chart) maps W = span(u, k_1..k_4) into
   span(k_1) subset W, nilpotently, so char_S(xi) = xi^5 * char(S on E0/W);
 * char_S is even in xi (S is J'A' with J' antisymmetric, A' symmetric after the G^{1/2}
   change of basis), so char(S on E0/W) is odd and vanishes at 0: zero multiplicity >= 6,
   with equality iff the coefficient a_6 of xi^6 in char_S is nonzero;
 * sigma(k_1, k_j) = x^T G X_j x = 0 for all j, so the 4x4 matrix of sigma on K_u has a zero
   first row and column; its rank is 0 or 2, and is 2 iff some rotation entry is nonzero.
"""
from math import comb
import numpy as np
import mpmath as mpm
from mpmath import iv
from lib_cert import (lo, hi, midf, width, excludes_zero, sign_of, matvec, matmul, transpose,
                      gauss_jordan_inverse, GDIAG, ivq)
from lib_quartic import NV


def r6_924(x):
    """924 * ||[w (x) Theta w]_6||^2 / ||w||^4 on the chart x (exact formula, see RETURN.md):
    [w(x)Theta w]_{6,M} = 720/sqrt(C(12,6+M)) sum_{m1+m2=M} (-1)^{3-m2} p_{m1} conj(p_{-m2})."""
    def P(m):
        a = 3 - m
        return x[a], x[7 + a]
    tot = iv.mpf(0)
    for M in range(-6, 7):
        re, im = iv.mpf(0), iv.mpf(0)
        for m1 in range(-3, 4):
            m2 = M - m1
            if abs(m2) > 3:
                continue
            sg = (-1) ** (3 - m2)
            a, b = P(m1)
            c, d = P(-m2)
            # p_{m1} conj(p_{-m2}) = (a + ib)(c - id)
            re += sg * (a * c + b * d)
            im += sg * (b * c - a * d)
        tot += (re * re + im * im) * ivq(__import__("fractions").Fraction(720 * 720, comb(12, 6 + M)))
    nrm = sum((GDIAG[a] * x[a] * x[a] for a in range(NV)), iv.mpf(0))
    return 924 * tot / (nrm * nrm)


def det_interval(A):
    n = len(A)
    M = [list(r) for r in A]
    det = iv.mpf(1)
    for c in range(n):
        if not excludes_zero(M[c][c]):
            raise ArithmeticError("pivot contains zero")
        det *= M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return det


def charpoly_leverrier(S):
    """coefficients a_0..a_n of det(xi I - S), interval Faddeev-LeVerrier."""
    n = len(S)
    a = [None] * (n + 1)
    a[n] = iv.mpf(1)
    Mk = [[iv.mpf(0)] * n for _ in range(n)]
    for k in range(1, n + 1):
        # M_k = S M_{k-1} + a_{n-k+1} I
        SM = matmul(S, Mk) if k > 1 else [[iv.mpf(0)] * n for _ in range(n)]
        Mk = [[SM[i][j] + (a[n - k + 1] if i == j else 0) for j in range(n)] for i in range(n)]
        SMk = matmul(S, Mk)
        a[n - k] = -sum((SMk[i][i] for i in range(n)), iv.mpf(0)) / k
    return a


def poly_eval_iv(coeffs, t):
    """coeffs a_0..a_d (intervals), t interval: Horner."""
    acc = iv.mpf(0)
    for c in reversed(coeffs):
        acc = acc * t + c
    return acc


def certify_real_roots(coeffs, delta_exp=-80):
    """Certify all roots of the degree-d polynomial with interval coefficients as real and simple
    by d disjoint sign changes.  Returns (ok, list of root intervals, approx roots)."""
    d = len(coeffs) - 1
    mids = [midf(c) for c in coeffs]
    try:
        approx = mpm.polyroots(list(reversed(mids)), maxsteps=400, extraprec=400)
    except mpm.libmp.libhyper.NoConvergence:
        # no approximate roots (e.g. a multiple root stalls the iteration): nothing certified
        return False, None, []
    approx = sorted(approx, key=lambda z: (mpm.re(z), mpm.im(z)))
    tol = mpm.mpf(10) ** (-mpm.mp.dps // 2)
    if any(abs(mpm.im(z)) > tol for z in approx):
        return False, None, approx
    delta = mpm.mpf(10) ** delta_exp
    encl = []
    for z in approx:
        r = mpm.re(z)
        a, b = r - delta, r + delta
        fa = poly_eval_iv(coeffs, iv.mpf(a))
        fb = poly_eval_iv(coeffs, iv.mpf(b))
        if sign_of(fa) == 0 or sign_of(fb) == 0 or sign_of(fa) == sign_of(fb):
            return False, None, approx
        encl.append(iv.mpf([a, b]))
    for i in range(len(encl) - 1):
        if not hi(encl[i]) < lo(encl[i + 1]):
            return False, None, approx
    return len(encl) == d, encl, approx


def analyze(slot, gens_iv, Zx):
    """Zx: 14 intervals (x-part of a certified box)."""
    x = Zx
    out = {}
    out["924 r6"] = r6_924(x)
    h = slot.h(x)
    out["h"] = h
    nrm2 = sum((GDIAG[a] * x[a] * x[a] for a in range(NV)), iv.mpf(0))
    out["|x|_G^2"] = nrm2
    k = [matvec(X, x) for X in gens_iv]
    Gram = [[sum((GDIAG[a] * k[i][a] * k[j][a] for a in range(NV)), iv.mpf(0)) for j in range(4)] for i in range(4)]
    gdet = det_interval(Gram)
    out["Gram det"] = gdet
    out["k"] = 4 if lo(gdet) > 0 else None
    # sigma on K_u: Omega_ij = k_i^T G X_1 k_j
    X1k = [matvec(gens_iv[0], kj) for kj in k]
    Om = [[sum((k[i][a] * GDIAG[a] * X1k[j][a] for a in range(NV)), iv.mpf(0)) for j in range(4)] for i in range(4)]
    out["Omega"] = Om
    rot_nonzero = [(i, j) for i in range(1, 4) for j in range(i + 1, 4) if excludes_zero(Om[i][j])]
    out["r_u"] = 2 if (out["k"] == 4 and rot_nonzero) else None
    out["Omega rotation entries certified nonzero"] = rot_nonzero
    # A = Hess h - 4 h G (as a quadratic form in the chart)
    H = slot.hess(x)
    A = [[H[a][b] - (4 * h * GDIAG[a] if a == b else 0) for b in range(NV)] for a in range(NV)]
    # N_u basis: B = P_perp B2, P_perp = I - W (W^T G W)^{-1} W^T G
    W = [[x[a]] + [k[i][a] for i in range(4)] for a in range(NV)]  # 14 x 5
    GW = [[GDIAG[a] * W[a][c] for c in range(5)] for a in range(NV)]
    WtGW = matmul(transpose(W), GW)
    WtGWinv = gauss_jordan_inverse(WtGW)
    Pperp = matmul(W, matmul(WtGWinv, transpose(GW)))
    Pperp = [[(1 if i == j else 0) - Pperp[i][j] for j in range(NV)] for i in range(NV)]
    Pm = mpm.matrix([[midf(v) for v in row] for row in Pperp])
    Am = mpm.matrix([[midf(v) for v in row] for row in A])
    U, s, Vt = np.linalg.svd(np.array([[float(Pm[i, j]) for j in range(NV)] for i in range(NV)]))
    B0 = mpm.matrix([[mpm.mpf(float(U[i, c])) for c in range(9)] for i in range(NV)])
    B1 = Pm * B0
    AN = B1.T * Am * B1
    AN = (AN + AN.T) / 2
    ev, V = mpm.eigsy(AN)
    B2 = B1 * V
    B2iv = [[iv.mpf(B2[i, c]) for c in range(9)] for i in range(NV)]
    B = matmul(Pperp, B2iv)
    ANiv = matmul(transpose(B), matmul(A, B))
    discs = []
    for i in range(9):
        rad = sum((max(abs(lo(ANiv[i][j])), abs(hi(ANiv[i][j]))) for j in range(9) if j != i), mpm.mpf(0))
        discs.append((lo(ANiv[i][i]) - rad, hi(ANiv[i][i]) + rad))
    neg = sum(1 for a, b in discs if b < 0)
    pos = sum(1 for a, b in discs if a > 0)
    out["signature"] = [neg, 0, pos] if neg + pos == 9 else None
    out["Gershgorin discs (N_u, congruent form)"] = discs
    # also the eigenvalue u-direction: A u = 8 h G u -> u^T A u = 8h |u|^2 (sanity)
    # slow matrix S = X_1 G^{-1} A
    GinvA = [[A[a][b] / GDIAG[a] for b in range(NV)] for a in range(NV)]
    S = matmul(gens_iv[0], GinvA)
    cp = charpoly_leverrier(S)
    out["charpoly"] = cp
    zero_consistent = all(not excludes_zero(cp[i]) for i in range(6)) and all(
        not excludes_zero(cp[i]) for i in range(7, 15, 2))
    out["charpoly: a_0..a_5 and odd a_k enclose 0 (consistency)"] = zero_consistent
    out["zero multiplicity"] = 6 if excludes_zero(cp[6]) else None
    slow = [cp[6 + 2 * i] for i in range(5)]
    out["slow coefficients (nu^0..nu^4)"] = slow
    ok, encl, approx = certify_real_roots(slow)
    out["slow roots certified real simple"] = ok
    out["slow roots"] = encl
    out["slow roots approx"] = approx
    return out
