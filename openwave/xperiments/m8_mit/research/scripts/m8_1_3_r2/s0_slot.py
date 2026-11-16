"""Items 0(c) for R2, 0(d), 0(e).
Usage: ./py s0_slot.py [--plant=NAME]  planted defects: eta (break the intertwiner), C (wrong C), s (wrong s)
"""
import sys, json
import mpmath as mpm
from lib import *
from quartic import mat_theta, cg_table
import quad

PLANT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--plant=")), None)
chk = Checker()
G, chi = load_group_data()
n = 7; n1 = 8; j = sp.Rational(7, 2)
Nk = [sp.sqrt(sp.factorial(k) * sp.factorial(n - k)) for k in range(n1)]

def Dw_exact(g):
    M = Mp(n, U_of_q(g), K(0), K(1))
    return sp.Matrix(n1, n1, lambda r, c: sp.radsimp(Nk[r] * M[r][c].sym() / Nk[c]))

# isotypic projector P = (dim/|G|) sum chi(h) D(h), on CS components
Pp = [[K(0)] * n1 for _ in range(n1)]
for g in G:
    M = Mp(n, U_of_q(g), K(0), K(1)); c = K(chi("R2", g))
    for a in range(n1):
        for b in range(n1):
            Pp[a][b] = Pp[a][b] + M[a][b] * c
P = sp.Matrix(n1, n1, lambda r, c: sp.radsimp((Nk[r] * Pp[r][c].sym() * sp.Rational(2, 120) / Nk[c])))
chk("0(c) P is an orthogonal projector of rank 2 (P^2 = P = P^H)",
    sp.simplify(P * P - P) == sp.zeros(n1) and sp.simplify(P - P.H) == sp.zeros(n1) and P.rank() == 2)
print("multiplicity of R2 in V_7/2 = rank P / dim R2 =", P.rank() // 2)

# eta = [e1, Theta e1], e1 = normalized P column (index k = 0, i.e. m = -7/2)
col = P[:, 0]
e1 = sp.simplify(col / sp.sqrt(sp.simplify((col.H * col)[0])))
Th = mat_theta(n)
def theta_vec(v):
    return sp.Matrix([(-1) ** (n - k) * sp.conjugate(v[n - k]) for k in range(n1)])
e2 = sp.simplify(theta_vec(e1))
eta = sp.Matrix.hstack(e1, e2)
if PLANT == "eta":
    eta = sp.Matrix.hstack(e1, sp.Matrix([1, 0, 0, 0, 0, 0, 0, 0]))
etaHeta = sp.simplify(eta.H * eta)
chk("0(c) eta^H eta = I_2", etaHeta == sp.eye(2), str(etaHeta))
print("eta (rows m = -7/2 .. 7/2):")
for k in range(n1):
    print(f"   m = {k - j}: {[sp.radsimp((eta[k, a])) for a in range(2)]}")
gens = {"q1": GEN_Q1, "q2": GEN_Q2}
rho = {}
for nm, g in gens.items():
    D = Dw_exact(g)
    chk(f"D(gen {nm}) unitary", sp.simplify(D * D.H) == sp.eye(n1))
    r = sp.simplify(eta.H * D * eta)
    rho[nm] = r
    chk(f"0(c) intertwiner: D({nm}) eta = eta rho({nm})", sp.simplify(D * eta - eta * r) == sp.zeros(n1, 2))
    print(f"   rho({nm}) =", [[sp.radsimp((r[a, b])) for b in range(2)] for a in range(2)])
# trace of rho on generators equals the R2 character
for nm, g in gens.items():
    chk(f"tr rho({nm}) = chi_R2({nm})", sp.simplify(rho[nm].trace() - chi("R2", g).sym()) == 0)

# s: exact (Schur orthogonality) and high-precision quadrature
s_exact = sp.Rational(2, n1)
if PLANT == "s":
    s_exact = sp.Rational(1, 3)
mpm.mp.dps = 50
nodes = quad.haar_D(n, 32, 16)  # exact for |phi|^4 too (phases <= 28, t-degree <= 14)
eta_mp = [[mpm.mpc(complex(0)) + mpm.mpmathify(sp.N(eta[k, a], 70)) for a in range(2)] for k in range(n1)]
tests = [[1, 0, 0, 0, 0, 0, 0, 0], [sp.Rational(1, 3), 2, -1, sp.I, 0, 5, sp.Rational(-2, 7), 1 + sp.I]]
ws = [[mpm.mpmathify(sp.N(x, 70)) for x in t] for t in tests]
avs = quad.averages(nodes, eta_mp, ws)
for t, w, (a2, a4) in zip(tests, ws, avs):
    nw2 = mpm.fsum(abs(x) ** 2 for x in w)
    s_num = a2 / nw2
    err = abs(s_num - mpm.mpf(s_exact.p) / s_exact.q)
    print(f"   quadrature s at w = {t}: {mpm.nstr(s_num, 45)}  |diff| = {mpm.nstr(err, 3)}")
    chk("0(c) s exact (dim rho/(2j+1)) agrees with 50-digit quadrature", err < mpm.mpf(10) ** -45)
print("0(c) s_N(R2) =", s_exact)

# ------------------------------------------------------------ 0(d)
c3360 = cg(3, 3, 3, -3, 6, 0)
print("0(d) <3 3; 3 -3 | 6 0> =", c3360, "=", sp.N(c3360, 20))
chk("0(d) <3 3; 3 -3|6 0> > 0", c3360 > 0)
c2 = cg(sp.Rational(7, 2), sp.Rational(7, 2), sp.Rational(7, 2), -sp.Rational(7, 2), 7, 0)
chk("<7/2 7/2; 7/2 -7/2|7 0> > 0 (convention fixed by the formula)", c2 > 0, str(c2))
T3 = mat_theta(6); T7 = mat_theta(7)
print("0(d) Theta^2 on V_3 = +1:", T3 * T3 == sp.eye(14), "; Theta^2 on V_7/2 = -1:", T7 * T7 == -sp.eye(16))
chk("0(d) Theta^2 = +1 on V_3", T3 * T3 == sp.eye(14))
chk("0(d) Theta^2 = -1 on V_7/2", T7 * T7 == -sp.eye(16))
# Theta commutes with the group action (D on generators), as a real-linear map
from quartic import complex_to_real_matrix
for nm, g in gens.items():
    Dr = complex_to_real_matrix(Dw_exact(g))
    chk(f"Theta commutes with D({nm}) on V_7/2", sp.simplify(T7 * Dr - Dr * T7) == sp.zeros(16))

# CG table of V_7/2 x V_7/2 (Racah formula): the 64 x 64 matrix is orthogonal (50 digits)
from quartic import cg_table
tabc = cg_table(n)
mpm.mp.dps = 50
U64 = mpm.matrix(64, 64)
row = 0
for J in range(0, n + 1):
    for M in range(-J, J + 1):
        for k1, k2, c in tabc[J][M]:
            val = mpm.mpmathify(sp.N(c, 70))
            if PLANT == "cg" and (J, M, k1) == (3, 0, 2):
                val = -val
            U64[row, k1 * n1 + k2] = val
        row += 1
devo = mpm.mnorm(U64 * U64.T - mpm.eye(64), 1)
chk("CG coefficients <7/2 m1; 7/2 m2 | J M> form an orthogonal 64 x 64 matrix (50 digits)", row == 64 and devo < mpm.mpf(10) ** -45, mpm.nstr(devo, 3))
# Condon-Shortley ladder: J+ |J M> computed from CG equals sqrt((J-M)(J+M+1)) |J M+1>, for J = 6 on V_3 x V_3
from quartic import J_ops
tab6 = cg_table(6)
Jz3, Jp3, Jm3 = J_ops(6)
okl = True
for J in (6, 4, 2):
    for M in range(-J, J):
        v = sp.zeros(49, 1); vp = sp.zeros(49, 1)
        for k1, k2, c in tab6[J][M]:
            v[k1 * 7 + k2] = c
        for k1, k2, c in tab6[J][M + 1]:
            vp[k1 * 7 + k2] = c
        Jtot = sp.kronecker_product(Jp3, sp.eye(7)) + sp.kronecker_product(sp.eye(7), Jp3)
        okl &= all(sp.radsimp(x) == 0 for x in (Jtot * v - sp.sqrt((J - M) * (J + M + 1)) * vp))
chk("CG vectors |J M> on V_3 x V_3 obey the Condon-Shortley ladder relation (exact, J = 6, 4, 2)", okl)

# ------------------------------------------------------------ 0(e)
# C with conj(rho(h)) C = C rho(h) for both generators: solve the linear system
cs = sp.symbols("c0:4")
Cs = sp.Matrix(2, 2, cs)
eqs = []
for nm in gens:
    eqs += list(sp.simplify(rho[nm].conjugate() * Cs - Cs * rho[nm]))
sol = sp.solve(eqs, cs, dict=True)
print("solutions of conj(rho)C = C rho:", sol)
C = sp.simplify(Cs.subs(sol[0]))
free = [x for x in cs if C.has(x)]
C = C.subs({free[0]: 1}) if free else C
C = sp.simplify(C / sp.sqrt(sp.simplify((C.H * C)[0, 0])))
if PLANT == "C":
    C = sp.eye(2)
print("0(e) C =", C.tolist())
chk("0(e) C unitary", sp.simplify(C.H * C) == sp.eye(2))
for nm in gens:
    chk(f"0(e) conj(rho({nm})) C = C rho({nm})", sp.simplify(rho[nm].conjugate() * C - C * rho[nm]) == sp.zeros(2))
CCb = sp.simplify(C * C.conjugate())
print("0(e) C conj(C) =", CCb.tolist())
# map psi_w -> conj(psi_w) C on w: find w' with w'^T D(g) eta = conj(w)^T conj(D(g)) conj(eta) C for all g.
# Y with conj(D(g)) = Y D(g) Y^-1: check on generators, Y_{k,k'} = (-1)^(n-k) delta_{k', n-k}
Y = sp.zeros(n1)
for k in range(n1):
    Y[k, n - k] = (-1) ** (n - k)
for nm, g in gens.items():
    D = Dw_exact(g)
    chk(f"conj(D({nm})) = Y D({nm}) Y^-1", sp.simplify(D.conjugate() * Y - Y * D) == sp.zeros(n1))
# conj(w)^T Y D Y^-1 conj(eta) C ; Y^-1 conj(eta) C = lam * eta
Z = sp.simplify(Y.inv() * eta.conjugate() * C)
lam = sp.simplify((eta.H * Z))
print("eta^H Y^-1 conj(eta) C =", lam.tolist())
chk("Y^-1 conj(eta) C = lam eta (Hom is one-dimensional)", sp.simplify(Z - eta * lam) == sp.zeros(n1, 2) and lam[0, 1] == 0 and lam[1, 0] == 0 and sp.simplify(lam[0, 0] - lam[1, 1]) == 0)
lam0 = lam[0, 0]
# w' = lam Y^T conj(w); compare with Theta: (Theta w)_k = (-1)^(n-k) conj(w_{n-k})
# (Y^T conj w)_k = sum_k' Y_{k',k} conj(w_k') = (-1)^k conj(w_{n-k}) ; ratio to Theta: (-1)^k / (-1)^(n-k) = (-1)^(n) = -1
ratio = sp.simplify(lam0 * (-1) ** n)
print(f"0(e) on w: psi_w -> conj(psi_w) C reads w -> ({ratio}) * Theta(w)")
# direct numeric verification at random g: phi_{w'} = conj(phi_w) C
import random
random.seed(1)
wt = sp.Matrix([sp.Rational(random.randint(-5, 5), 3) + sp.I * random.randint(-4, 4) for _ in range(n1)])
wp = ratio * theta_vec(wt)
mpm.mp.dps = 40
okn = True
for wt_, D in nodes[::97]:
    Dm = sp.Matrix(n1, n1, lambda r, c: sp.Float(str(D[r][c].real), 40) + sp.I * sp.Float(str(D[r][c].imag), 40))
    lhs = (wp.T * Dm * eta).evalf(40)
    rhs = ((wt.T * Dm * eta).conjugate() * C).evalf(40)
    okn &= max(abs(complex(x)) for x in (lhs - rhs)) < 1e-30
chk("0(e) conj(psi_w) C = psi_{w'} at sampled group points (numeric, 40 digits)", okn)

out = dict(eta=[[str(sp.radsimp((eta[k, a]))) for a in range(2)] for k in range(n1)],
           P=[[str(P[r, c]) for c in range(n1)] for r in range(n1)],
           s=str(s_exact), C=[[str(x) for x in row] for row in C.tolist()], CCbar=[[str(x) for x in row] for row in CCb.tolist()],
           fibre_map_on_w=f"w -> ({ratio}) * Theta(w)", rho={k: [[str(x) for x in row] for row in v.tolist()] for k, v in rho.items()},
           cg_33_3m3_60=str(c3360))
if PLANT is None:
    json.dump(out, open(rel("out_slot_R2.json"), "w"), indent=1)
chk.summary()
