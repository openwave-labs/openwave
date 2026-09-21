import core, sympy as sp
c = sp.symbols('c0:7')
d = sp.symbols('d0:7')   # d_k stands for conj(c_k)
def cj(z):
    k = c.index(z)
    return d[k]
rh = core.rho6(list(c), conj=cj)
swap = {**{c[k]: d[k] for k in range(7)}, **{d[k]: c[k] for k in range(7)}}
Nc = sp.expand(sum(v * v.xreplace(swap) for v in rh.values()))
P = sp.Poly(Nc, *c, *d)
pairs = [(a, b) for a in range(7) for b in range(a, 7)]
M = sp.zeros(28)
for i, (a, b) in enumerate(pairs):
    for j, (e, f) in enumerate(pairs):
        coeff = P.coeff_monomial(d[a] * d[b] * c[e] * c[f])
        sa = 1 if a == b else sp.sqrt(2)
        se = 1 if e == f else sp.sqrt(2)
        M[i, j] = coeff / (sa * se)
print(M.is_hermitian)
print(M.eigenvals())
