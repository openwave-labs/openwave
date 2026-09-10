"""Item 21: the psi psi^T construction.

Checks per sector (exact):
  * Theta commutes with P; a Theta-fixed orthonormal basis eta of the sector exists; with it
    sigma(h) = eta^dag D(h) eta is real orthogonal (sigma sigma^T = I) at both generators, and
    psi psi^T is right-Gamma-invariant; with a non-real basis (eta U, U = diag(1, i, 1, ..)) it is not.
  * psi psi^T = sum_J B_J(u)^T D^J(g) R'_J with R'_J = sum CG(3n;3n'|JN)(eta eta^T)_{nn'}; compare R'_J
    with R_J = M_J(P); left factor B_J(u) = M_J(u (Theta u)^dag).
  * surviving channels, and whether span{N_0, N_6} = span{M_0, M_6}.
"""
import random
from fractions import Fraction as Fr

import sympy as sp

from solver_lib import (Q1, Q2, Quat, D_mono, mono_to_orthonormal, transform, norm2, A_op_matrix, CG,
                        B_ab, M_ab, N_ab, rho_ab, load_P, load_results, save_results, s)

a = list(sp.symbols("u0:7"))
b = list(sp.symbols("ub0:7"))
T = sp.zeros(7, 7)
for m in range(-3, 4):
    T[-m + 3, m + 3] = sp.Integer(-1) ** abs(m)


def zero(M):
    return all(sp.simplify(sp.radsimp(sp.expand(e))) == 0 for e in M)


def Dorth(q, n):
    return mono_to_orthonormal(D_mono(q.su2(), n), n)


D1, D2 = Dorth(Q1, 6), Dorth(Q2, 6)
gq = Quat(Fr(1, 5), Fr(2, 5), Fr(2, 5), Fr(4, 5))
DK = {J: Dorth(gq, 2 * J) for J in range(7)}
rnd = random.Random(11)
uval = [sp.Rational(rnd.randint(-9, 9), rnd.randint(1, 5)) + sp.I * sp.Rational(rnd.randint(-9, 9), rnd.randint(1, 5))
        for _ in range(7)]
uvec = sp.Matrix(uval)

# left factor identity B_J(u) = M_J(u (Theta u)^dag), symbolic check with u, conj u independent:
# (u (Theta u)^dag)_{n,k} = u_n * conj((Theta u)_k) = u_n (-1)^k u_{-k}
X = sp.Matrix(7, 7, lambda i, k: a[i] * sp.Integer(-1) ** abs(k - 3) * a[-(k - 3) + 3])
left_ok = all(all(sp.expand(x - y) == 0 for x, y in zip(transform(X, J), B_ab(a, J))) for J in range(7))
print("B_J(u) == M_J(u (Theta u)^dag) for J=0..6:", left_ok)
oddB = all(all(sp.expand(x) == 0 for x in B_ab(a, J)) for J in (1, 3, 5))
print("B_J == 0 identically for odd J:", oddB)

FS = {c["rank"]: c["Frobenius_Schur_indicator"] for c in load_results()["group"]["item11"]["components"]}
out = {"left_factor_is_B_J_equals_M_J_of_u_Thetau_dag": left_ok, "B_odd_J_vanish": oddB,
       "Frobenius_Schur_from_group_script": FS}

Ms = {K: M_ab(a, b, K) for K in range(7)}
Ns = {J: N_ab(a, b, J) for J in range(7)}

mon_keys = sorted({(m, mono) for mp in list(Ms.values()) + list(Ns.values()) for m in range(7)
                   for mono in (sp.Poly(mp[m], *a, *b).monoms() if mp[m] != 0 else [])})


def coeff_row(mp):
    return [sp.Poly(mp[m], *a, *b).coeff_monomial(mono) if mp[m] != 0 else 0 for (m, mono) in mon_keys]


def rank_of(maps):
    return sp.Matrix([coeff_row(mp) for mp in maps]).rank(simplify=True)


for key in ("d=4", "d=3"):
    P = load_P(key)
    d = int(sp.simplify(P.trace()))
    theta_comm = zero(T * P.conjugate() - P * T)
    # Theta-fixed spanning vectors of the sector and exact Gram-Schmidt (inner products real)
    cands = []
    for k in range(7):
        y = P[:, k]
        for ph in (1, sp.I):
            x = ph * y + T * (ph * y).conjugate()
            cands.append(x.applyfunc(lambda e: sp.radsimp(sp.expand(e))))
    basis = []
    for x in cands:
        v = x
        for e in basis:
            v = v - (e.H * v)[0, 0] * e
        v = v.applyfunc(lambda e: sp.radsimp(sp.expand(e)))
        n2 = sp.simplify((v.H * v)[0, 0])
        if n2 != 0:
            basis.append((v / sp.sqrt(n2)).applyfunc(lambda e: sp.radsimp(sp.expand(e))))
        if len(basis) == d:
            break
    eta = sp.Matrix.hstack(*basis)
    checks = {
        "Theta_commutes_with_P": theta_comm,
        "eta_dag_eta_is_I": zero(eta.H * eta - sp.eye(d)),
        "eta_eta_dag_is_P": zero(eta * eta.H - P),
        "Theta_eta_equals_eta": zero(T * eta.conjugate() - eta),
    }
    sig = {}
    for nm, Dh in (("q1", D1), ("q2", D2)):
        S = (eta.H * Dh * eta).applyfunc(lambda e: sp.simplify(sp.radsimp(e)))
        sig[nm] = {"intertwines": zero(Dh * eta - eta * S), "real": zero(S - S.conjugate()),
                   "sigma_sigmaT_is_I": zero(S * S.T - sp.eye(d))}
    # non-real basis for contrast
    U = sp.diag(*([1, sp.I] + [1] * (d - 2)))
    eta2 = eta * U
    S2 = (eta2.H * D1 * eta2).applyfunc(lambda e: sp.simplify(sp.radsimp(e)))
    contrast = zero(S2 * S2.T - sp.eye(d))
    # right-invariance of psi psi^T at a rational g, h = q1, q2
    def psipsiT(e, D):
        row = uvec.T * D * e
        return sp.expand((row * row.T)[0, 0])
    inv_real = all(sp.simplify(sp.radsimp(psipsiT(eta, DK[3] * Dh) - psipsiT(eta, DK[3]))) == 0 for Dh in (D1, D2))
    inv_nonreal = all(sp.simplify(sp.radsimp(psipsiT(eta2, DK[3] * Dh) - psipsiT(eta2, DK[3]))) == 0 for Dh in (D1, D2))
    # right factor R'_J
    E = (eta * eta.T).applyfunc(lambda e: sp.radsimp(sp.expand(e)))
    Rp = {}
    for J in range(7):
        o = [sp.S(0)] * (2 * J + 1)
        for n in range(-3, 4):
            for n2 in range(-3, 4):
                N = n + n2
                if abs(N) <= J:
                    o[N + J] += CG(3, n, 3, n2, J, N) * E[n + 3, n2 + 3]
        Rp[J] = [sp.radsimp(x) for x in o]
    R = {J: [sp.radsimp(x) for x in transform(P, J)] for J in range(7)}
    Rp_eq_R = {J: all(sp.simplify(x - y) == 0 for x, y in zip(Rp[J], R[J])) for J in range(7)}
    Rp_norm = {J: sp.simplify(norm2(Rp[J])) for J in range(7)}
    # exact expansion psi psi^T = sum_J B_J(u)^T D^J R'_J at g
    lhs = psipsiT(eta, DK[3])
    rhs = 0
    for J in range(7):
        BJ = B_ab(uval, J)
        rhs += sum(BJ[M] * DK[J][M, N] * Rp[J][N] for M in range(2 * J + 1) for N in range(2 * J + 1))
    exp_ok = sp.simplify(sp.radsimp(sp.expand(lhs - rhs))) == 0
    cp = {J: sp.simplify(sp.radsimp((P * A_op_matrix(Rp[J], J)).trace())) for J in range(7)}
    surviving = [J for J in range(7) if Rp_norm[J] != 0]
    print(f"\n{key}: checks {checks}")
    print(f"   sigma at generators: {sig};  non-real basis gives sigma sigma^T = I: {contrast}")
    print(f"   psi psi^T right-invariant: Theta-fixed basis {inv_real}, non-real basis {inv_nonreal}")
    print(f"   R'_J == R_J = M_J(P): {Rp_eq_R};  ||R'_J||^2 = {[Rp_norm[J] for J in range(7)]}")
    print(f"   exact expansion psi psi^T = sum_J B_J^T D^J R'_J: {exp_ok}")
    print(f"   c'_J = Tr(P A(R'_J)) = {[cp[J] for J in range(7)]}; coefficients c'_J/d = {[sp.simplify(cp[J] / d) for J in range(7)]}")
    print(f"   surviving channels J: {surviving}")
    out[key] = {"checks": checks, "sigma_at_generators": sig, "nonreal_basis_sigma_sigmaT_is_I": contrast,
                "psipsiT_invariant_Theta_fixed_basis": inv_real, "psipsiT_invariant_nonreal_basis": inv_nonreal,
                "Rprime_equals_R": {str(J): v for J, v in Rp_eq_R.items()},
                "norm2_Rprime": [s(Rp_norm[J]) for J in range(7)], "expansion_exact": exp_ok,
                "cprime_J": [s(cp[J]) for J in range(7)],
                "coefficients_cprime_over_d": [s(sp.simplify(cp[J] / d)) for J in range(7)],
                "surviving_J": surviving,
                "eta_columns": [[s(e) for e in eta[:, k]] for k in range(d)]}

rN06 = rank_of([Ns[0], Ns[6]])
rM06 = rank_of([Ms[0], Ms[6]])
rall = rank_of([Ms[0], Ms[6], Ns[0], Ns[6]])
rNall = rank_of([Ns[J] for J in (0, 2, 4, 6)])
rMall = rank_of([Ms[K] for K in range(7)])
rboth = rank_of([Ms[K] for K in range(7)] + [Ns[J] for J in (0, 2, 4, 6)])
# explicit witness: N_0(v_3)
v3 = {**{a[i]: (1 if i == 6 else 0) for i in range(7)}, **{b[i]: (1 if i == 6 else 0) for i in range(7)}}
wit = {"N0(v3)": [s(sp.radsimp(x.subs(v3))) for x in Ns[0]], "M0(v3)": [s(sp.radsimp(x.subs(v3))) for x in Ms[0]],
       "M6(v3)": [s(sp.radsimp(x.subs(v3))) for x in Ms[6]], "N6(v3)": [s(sp.radsimp(x.subs(v3))) for x in Ns[6]]}
print(f"\nrank span{{N_0,N_6}} = {rN06}; rank span{{M_0,M_6}} = {rM06}; rank span{{M_0,M_6,N_0,N_6}} = {rall}")
print(f"rank span{{N_0,N_2,N_4,N_6}} = {rNall}; rank span{{M_0..M_6}} = {rMall}; rank of all together = {rboth}")
print("witness values at v_3:", wit)
out["spans"] = {"rank_N0_N6": int(rN06), "rank_M0_M6": int(rM06), "rank_M0_M6_N0_N6": int(rall),
                "spans_coincide": int(rall) == int(rM06) == int(rN06),
                "rank_N0_N2_N4_N6": int(rNall), "rank_M0_to_M6": int(rMall), "rank_all": int(rboth),
                "witness_at_v3": wit}
save_results("item21", out)
