import numpy as np, itertools, math
import sympy as sp
exec(open('scratch_K.py').read().split("print(M.is_hermitian)")[0])
# K on V⊗V (49x49) symmetric representative
K = np.zeros((49, 49), dtype=complex)
for a in range(7):
    for b in range(7):
        for e in range(7):
            for f in range(7):
                aa, bb = sorted((a, b)); ee, ff = sorted((e, f))
                coeff = float(P.coeff_monomial(d[aa] * d[bb] * c[ee] * c[ff]))
                # coefficient of conj(c_a c_b) c_e c_f spread over orderings
                na = 1 if aa == bb else 2
                ne = 1 if ee == ff else 2
                K[a * 7 + b, e * 7 + f] = coeff / (na * ne)
print("herm", np.abs(K - K.conj().T).max())
rng = np.random.default_rng(0)
for _ in range(3):
    u = rng.normal(size=7) + 1j * rng.normal(size=7); u /= np.linalg.norm(u)
    uu = np.kron(u, u)
    print(np.vdot(uu, K @ uu).real, float(sp.N(core.rhat6([sp.Float(x.real) + sp.I * sp.Float(x.imag) for x in u]))))
def symproj(k):
    n = 7 ** k
    Pm = np.zeros((n, n))
    perms = list(itertools.permutations(range(k)))
    idx = np.arange(n).reshape([7] * k)
    for p in perms:
        perm_idx = np.transpose(idx, p).reshape(-1)
        Pm[np.arange(n), perm_idx] += 1
    return Pm / len(perms)
for k in [2, 3, 4]:
    Pk = symproj(k)
    Kk = np.kron(K, np.eye(7 ** (k - 2)))
    A = Pk @ Kk @ Pk
    w = np.linalg.eigvalsh((A + A.conj().T) / 2)
    print(k, w[-3:], 463 / 924)
