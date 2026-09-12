"""FLOATING-POINT CROSS-CHECK (labelled): Haar quadrature on SU(2) of the functionals
B = int |psi|^2, A = int |psi|^4, the projection N(u) of |psi|^2 psi onto level 6, and the psi psi^T
analogue N'(u) (item 21), compared with the closed forms obtained exactly elsewhere.

Quadrature: Euler angles U = Rz(alpha) Ry(beta) Rz(gamma); integrands are functions on SO(3)
(integer spin), alpha, gamma uniform grids (exact for trigonometric polynomials of degree < n),
Gauss-Legendre in cos(beta).
"""
import numpy as np
import sympy as sp

from solver_lib import (transform, norm2, A_op_matrix, rho_ab, M_ab, N_ab, load_P, load_results,
                        save_results)

rng = np.random.default_rng(2026)
NA, NB, NG = 28, 28, 28


def D_num(U, n):
    """Spin n/2 matrix, orthonormal basis, same convention as solver_lib.D_mono (vectorised over U)."""
    from math import factorial, sqrt
    U11, U12, U21, U22 = U[..., 0, 0], U[..., 0, 1], U[..., 1, 0], U[..., 1, 1]
    out = np.zeros(U.shape[:-2] + (n + 1, n + 1), dtype=complex)
    for aa in range(n + 1):
        # coefficients of x^k in (U21 + U11 x)^aa (U22 + U12 x)^(n-aa)
        poly = np.ones(U.shape[:-2] + (1,), dtype=complex)
        for _ in range(aa):
            new = np.zeros(poly.shape[:-1] + (poly.shape[-1] + 1,), dtype=complex)
            new[..., :-1] += poly * U21[..., None]
            new[..., 1:] += poly * U11[..., None]
            poly = new
        for _ in range(n - aa):
            new = np.zeros(poly.shape[:-1] + (poly.shape[-1] + 1,), dtype=complex)
            new[..., :-1] += poly * U22[..., None]
            new[..., 1:] += poly * U12[..., None]
            poly = new
        for ap in range(n + 1):
            out[..., ap, aa] = poly[..., ap] * sqrt(factorial(ap) * factorial(n - ap) / (factorial(aa) * factorial(n - aa)))
    return out


al = 2 * np.pi * np.arange(NA) / NA
ga = 2 * np.pi * np.arange(NG) / NG
xg, wg = np.polynomial.legendre.leggauss(NB)
be = np.arccos(xg)
AL, BE, GA = np.meshgrid(al, be, ga, indexing="ij")
W = np.broadcast_to((wg / 2)[None, :, None], AL.shape) / (NA * NG)


def Rz(t):
    z = np.zeros(t.shape + (2, 2), dtype=complex)
    z[..., 0, 0] = np.exp(-0.5j * t)
    z[..., 1, 1] = np.exp(0.5j * t)
    return z


def Ry(t):
    z = np.zeros(t.shape + (2, 2), dtype=complex)
    c, s_ = np.cos(t / 2), np.sin(t / 2)
    z[..., 0, 0] = c
    z[..., 0, 1] = -s_
    z[..., 1, 0] = s_
    z[..., 1, 1] = c
    return z


Ug = Rz(AL) @ Ry(BE) @ Rz(GA)
D3 = D_num(Ug, 6)
print("quadrature normalisation check, int 1 =", W.sum())
# Schur orthogonality sanity: int |D_{00}|^2 = 1/7
print("int |D^3_{00}|^2 * 7 =", (W * np.abs(D3[..., 3, 3]) ** 2).sum() * 7)

a = list(sp.symbols("u0:7"))
b = list(sp.symbols("ub0:7"))
u = rng.normal(size=7) + 1j * rng.normal(size=7)
sub = {**{a[i]: complex(u[i]) for i in range(7)}, **{b[i]: complex(np.conj(u[i])) for i in range(7)}}
T = np.zeros((7, 7))
for m in range(-3, 4):
    T[-m + 3, m + 3] = (-1) ** abs(m)

out = {}
for key in ("d=4", "d=3"):
    P = load_P(key)
    d = int(sp.simplify(P.trace()))
    Pn = np.array(P.evalf(30).tolist(), dtype=complex)
    # Theta-fixed orthonormal basis (numerical) of the sector
    cands = []
    for k in range(7):
        y = Pn[:, k]
        for ph in (1, 1j):
            cands.append(ph * y + T @ np.conj(ph * y))
    basis = []
    for x in cands:
        v = x.copy()
        for e in basis:
            v = v - (np.conj(e) @ v) * e
        if np.linalg.norm(v) > 1e-8:
            basis.append(v / np.linalg.norm(v))
        if len(basis) == d:
            break
    eta = np.array(basis).T
    assert np.allclose(eta @ eta.conj().T, Pn)
    psi = np.einsum("m,ijkmn,na->ijka", u, D3, eta)
    dens = np.sum(np.abs(psi) ** 2, axis=-1)
    Bq = (W * dens).sum()
    Aq = (W * dens ** 2).sum()
    R = {K: transform(P, K) for K in range(7)}
    RN = {K: sp.simplify(norm2(R[K])) for K in range(7)}
    nu2 = float(np.sum(np.abs(u) ** 2))
    Bf = d * nu2 / 7
    Af = sum(complex(sp.expand(sum(x * y for x, y in zip(rho_ab(a, b, K), rho_ab(b, a, K)))).subs(sub)).real
             * float(RN[K]) / (2 * K + 1) for K in range(7))
    # N(u) by quadrature: N_m = (7/d) int sum_a |psi|^2 psi_a conj(psi_{v_m, a})
    psim = np.einsum("ijkmn,na->ijkma", D3, eta)          # psi_{v_m}(g)_a
    Nq = (7 / d) * np.einsum("ijk,ijk,ijka,ijkma->m", W, dens, psi, np.conj(psim))
    cK = {K: complex(sp.N((P * A_op_matrix(R[K], K)).trace(), 30)) for K in range(7)}
    Nf = np.array([sum(cK[K] * complex(M_ab(a, b, K)[m].subs(sub)) for K in range(7)) / d for m in range(7)])
    # psi psi^T analogue: N'_m = (7/d) int (psi psi^T) sum_b conj(psi_b) conj(psi_{v_m,b})
    ppT = np.sum(psi * psi, axis=-1)
    Npq = (7 / d) * np.einsum("ijk,ijk,ijka,ijkma->m", W, ppT, np.conj(psi), np.conj(psim))
    Npf = np.array([sum(cK[J] * complex(N_ab(a, b, J)[m].subs(sub)) for J in (0, 2, 4, 6)) / d for m in range(7)])
    res = {"B_quad": Bq, "B_formula": Bf, "A_quad": Aq, "A_formula": Af,
           "N_max_abs_diff": float(np.max(np.abs(Nq - Nf))), "N_max_abs": float(np.max(np.abs(Nf))),
           "Nprime_max_abs_diff": float(np.max(np.abs(Npq - Npf))), "Nprime_max_abs": float(np.max(np.abs(Npf)))}
    print(key, res)
    out[key] = res
save_results("crosscheck_quadrature_FLOAT", out)
