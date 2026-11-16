"""Items 0(c), 0(d), 0(e): isotypic projectors (exact), intertwiners eta, the constant s_N, CG sign, Theta^2,
the quaternionic structure in R2.  Also verifies the D^j / CG / Theta conventions every later script relies on."""
import os, sys, json, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
from common import *
from sympy.physics.quantum.cg import CG as LibCG

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
mp.mp.dps = 50
gd = json.load(open(os.path.join(OUT, "group_data.json")))
grp = json.load(open(os.path.join(OUT, "group.json")))
order = grp["order"]
chars = {nm: [C5(Q5(Fr(a), Fr(b)), Q5(Fr(c), Fr(d))) for a, b, c, d in v] for nm, v in gd["chars"].items()}
G = closure(list(generators()))
re_to_class = {}
for k, cl in enumerate(gd["classes"]):
    re_to_class[sp.nsimplify(cl[0][0])] = k
def cls_of(q):
    return re_to_class[sp.nsimplify(q.c[0].sym())]

SLOTS = {"R2": (sp.Rational(7, 2), "2'"), "R4": (sp.Integer(3), "3'"), "R5": (sp.Integer(3), "4")}
q1, q2 = generators()

def mat_c5_mul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    return [[sum((A[i][t] * B[t][jj] for t in range(k)), C5(0)) for jj in range(m)] for i in range(n)]

# ------------------------------------------------ convention checks (numerical, 50 digits)
random.seed(1)
def rand_su2():
    v = [mp.mpf(random.uniform(-1, 1)) for _ in range(4)]
    nv = mp.sqrt(sum(x * x for x in v)); v = [x / nv for x in v]
    w, x, y, z = v
    return [[mp.mpc(w, x), mp.mpc(y, z)], [mp.mpc(-y, z), mp.mpc(w, -x)]]
def mm2(g, h):
    return [[g[r][0] * h[0][c] + g[r][1] * h[1][c] for c in range(2)] for r in range(2)]
def maxabs(M):
    return max(abs(M[i, j]) for i in range(M.rows) for j in range(M.cols))
def to_mp(Ms):
    return mp.matrix([[complex(Ms[i, j]) if False else mp.mpf(str(sp.N(Ms[i, j], 70))) for j in range(Ms.cols)] for i in range(Ms.rows)])

g, h = rand_su2(), rand_su2()
for j in (sp.Rational(1, 2), sp.Integer(3), sp.Rational(7, 2)):
    Dg, Dh, Dgh = D_mp(j, g), D_mp(j, h), D_mp(j, mm2(g, h))
    check(f"D^{j}: homomorphism D(gh) = D(g)D(h)", maxabs(Dgh - Dg * Dh) < mp.mpf(10) ** -45)
    check(f"D^{j}: unitary", maxabs(Dg.H * Dg - mp.eye(Dg.rows)) < mp.mpf(10) ** -45)
    U = to_mp(theta_U(j))
    Dbar = mp.matrix([[mp.conj(Dg[a, b]) for b in range(Dg.cols)] for a in range(Dg.rows)])
    check(f"D^{j}: Theta commutes with D (conj(D) = U^-1 D U)", maxabs(U * Dbar - Dg * U) < mp.mpf(10) ** -45)
# CG product rule D^{j1} (x) D^{j2} = W^T (+) D^J W, for (j1,j2) = (6,7/2), (3,3), (7/2,7/2)
for j1, j2 in ((sp.Integer(3), sp.Integer(3)), (sp.Rational(7, 2), sp.Rational(7, 2)), (sp.Integer(6), sp.Rational(7, 2))):
    D1, D2 = D_mp(j1, g), D_mp(j2, g)
    n1, n2 = D1.rows, D2.rows
    K = mp.matrix(n1 * n2, n1 * n2)
    for a in range(n1):
        for b in range(n2):
            for c in range(n1):
                for d in range(n2):
                    K[a * n2 + b, c * n2 + d] = D1[a, c] * D2[b, d]
    err = mp.mpf(0)
    Jmin, Jmax = abs(j1 - j2), j1 + j2
    J = Jmin; Wall = []
    while J <= Jmax:
        W = to_mp(W_matrix(J, j1, j2))
        DJ = D_mp(J, g)
        # W K = DJ W  (V_J sits in V_j1 (x) V_j2 via the CG coefficients)
        err = max(err, maxabs(W * K - DJ * W))
        Wall.append(W)
        J += 1
    check(f"CG: W_J (D^{j1} (x) D^{j2}) = D^J W_J for all J (my Racah CG and my D^j agree)", err < mp.mpf(10) ** -45, mp.nstr(err, 3))
# CG library comparison (lookup, reported in the manifest)
bad = 0; tot = 0
for (j1, j2) in ((3, 3), (sp.Rational(7, 2), sp.Rational(7, 2)), (6, 3), (6, sp.Rational(7, 2))):
    for J in range(int(abs(sp.Rational(j1) - j2) * 2), int((sp.Rational(j1) + j2) * 2) + 1, 2):
        J = sp.Rational(J, 2)
        for m1 in mlist(j1):
            for m2 in mlist(j2):
                M = m1 + m2
                if abs(M) > J:
                    continue
                tot += 1
                if sp.simplify(cg(j1, m1, j2, m2, J, M) - LibCG(j1, m1, j2, m2, J, M).doit()) != 0:
                    bad += 1
check("CG: my Racah CG == sympy.physics.quantum.cg.CG on all coefficients used", bad == 0, f"{tot} compared, {bad} differ")

# ------------------------------------------------ 0(d)
c33 = cg(3, 3, 3, -3, 6, 0)
print("0(d) <3 3; 3 -3 | 6 0> =", c33, "=", sp.N(c33, 20))
check("0(d) <3 3; 3 -3 | 6 0> > 0", c33 > 0)
theta2 = {}
for j in (sp.Integer(3), sp.Rational(7, 2)):
    U = theta_U(j)
    T2 = U * U   # Theta^2 u = U conj(U conj u) = U U u, U real
    n = U.rows
    val = T2[0, 0]
    check(f"0(d) Theta^2 on V_{j} is a scalar", T2 == val * sp.eye(n))
    theta2[str(j)] = int(val)
    print(f"0(d) Theta^2 on V_{j} = {val}")

# ------------------------------------------------ 0(c): exact isotypic projectors
res = {"0(c)": {}, "0(d)": {"<3 3; 3 -3|6 0>": str(c33), "sign": "+", "Theta^2 on V_3": str(theta2["3"]),
                                 "Theta^2 on V_7/2": str(theta2["7/2"])}}
slotdata = {}
Dgens_w = {}
for s, (j, irr) in SLOTS.items():
    n = int(2 * j + 1)
    dimr = int(chars[irr][0].r.a)
    Pw = [[C5(0)] * n for _ in range(n)]
    for q in G:
        chi = chars[irr][cls_of(q)].conj()
        Dw = D_weighted_exact(j, q)
        for a in range(n):
            for b in range(n):
                if not Dw[a][b].is0():
                    Pw[a][b] = Pw[a][b] + chi * Dw[a][b]
    Pw = [[C5(x.r * Fr(dimr, order), x.i * Fr(dimr, order)) for x in row] for row in Pw]
    P2 = mat_c5_mul(Pw, Pw)
    check(f"0(c) {s}: P^2 = P exactly (weighted coordinates)", all(P2[a][b] == Pw[a][b] for a in range(n) for b in range(n)))
    tr = sum((Pw[a][a] for a in range(n)), C5(0))
    check(f"0(c) {s}: trace P = dim rho = {dimr} (multiplicity one)", tr == C5(dimr), str(tr))
    for nm, qq in (("q1", q1), ("q2", q2)):
        Dq = D_weighted_exact(j, qq)
        A, B = mat_c5_mul(Dq, Pw), mat_c5_mul(Pw, Dq)
        check(f"0(c) {s}: P commutes with D({nm}) exactly", all(A[a][b] == B[a][b] for a in range(n) for b in range(n)))
    # orthonormal basis: w_m = sqrt(N_m) p_m, so P_on = S Pw S^-1
    ms = mlist(j)
    Nm = [sp.factorial(j + m) * sp.factorial(j - m) for m in ms]
    Pon = sp.Matrix(n, n, lambda a, b: sp.sqrt(Nm[a]) * Pw[a][b].sym() / sp.sqrt(Nm[b]))
    Pon = Pon.applyfunc(sp.radsimp)
    herm = (Pon - Pon.H).applyfunc(lambda x: sp.simplify(sp.expand(x)))
    check(f"0(c) {s}: P is Hermitian in the orthonormal CS basis (exact)", herm == sp.zeros(n, n))
    # eta: orthonormal basis of range(P), numerically at 50 digits (Gram-Schmidt on columns of P)
    Pm = mp.matrix([[mp.mpc(str(sp.N(sp.re(Pon[a, b]), 70)), str(sp.N(sp.im(Pon[a, b]), 70))) for b in range(n)] for a in range(n)])
    cols = []
    for b in range(n):
        v = Pm[:, b]
        for u in cols:
            v = v - u * (u.H * v)[0]
        nv = mp.sqrt(mp.re((v.H * v)[0]))
        if nv > mp.mpf(10) ** -20:
            cols.append(v / nv)
        if len(cols) == dimr:
            break
    eta = mp.matrix(n, dimr)
    for a in range(dimr):
        for r in range(n):
            eta[r, a] = cols[a][r]
    check(f"0(c) {s}: eta^dagger eta = I", maxabs(eta.H * eta - mp.eye(dimr)) < mp.mpf(10) ** -45)
    check(f"0(c) {s}: eta eta^dagger = P", maxabs(eta * eta.H - Pm) < mp.mpf(10) ** -45)
    # rho(h) = eta^dagger D(h) eta and D(h) eta = eta rho(h), character check
    errs = mp.mpf(0); cherr = mp.mpf(0)
    rho = {}
    for qq in G:
        Dq = D_mp(j, quat_mp(qq))
        r_ = eta.H * Dq * eta
        errs = max(errs, maxabs(Dq * eta - eta * r_))
        tr_ = sum(r_[a, a] for a in range(dimr))
        cherr = max(cherr, abs(tr_ - chars[irr][cls_of(qq)].mpc()))
        rho[qq] = r_
    check(f"0(c) {s}: D(h) eta = eta rho(h) for all 120 h", errs < mp.mpf(10) ** -40, mp.nstr(errs, 3))
    check(f"0(c) {s}: trace rho(h) = chi_{irr}(h) for all 120 h", cherr < mp.mpf(10) ** -40, mp.nstr(cherr, 3))
    # s_N: exact route (Schur orthogonality) dim rho/(2j+1); numerical route: cubature over SU(2)
    sN_exact = sp.Rational(dimr, n)
    random.seed(7)
    wv = mp.matrix([mp.mpc(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(n)])
    nw2 = mp.re((wv.H * wv)[0])
    deg = 2 * (n - 1)               # |phi_w|^2 has degree 2*2j in the entries
    Nxi = deg + 2; Nt = (n - 1) + 2
    import numpy as np
    xs = [mp.findroot(lambda x: mp.legendre(Nt, x), mp.mpf(float(x0))) for x0 in np.polynomial.legendre.leggauss(Nt)[0]]
    ws = [2 * (1 - x * x) / (Nt * mp.legendre(Nt - 1, x)) ** 2 for x in xs]
    check(f"0(c) {s}: {Nt} distinct Gauss-Legendre nodes, weights sum to 2", len({mp.nstr(x, 30) for x in xs}) == Nt and abs(sum(ws) - 2) < mp.mpf(10) ** -45)
    tn = [(x + 1) / 2 for x in xs]; tw = [wt / 2 for wt in ws]
    total = mp.mpf(0)
    wT = wv.T
    for t, wt in zip(tn, tw):
        ce, se = mp.sqrt(1 - t), mp.sqrt(t)
        for k1 in range(Nxi):
            e1 = mp.expjpi(mp.mpf(2 * k1) / Nxi)
            for k2 in range(Nxi):
                e2 = mp.expjpi(mp.mpf(2 * k2) / Nxi)
                a_, b_ = ce * e1, se * e2         # w + x i = a_, y + z i = b_
                gg = [[a_, b_], [-mp.conj(b_), mp.conj(a_)]]
                phi = wT * D_mp(j, gg) * eta
                total += wt * sum(abs(phi[0, a]) ** 2 for a in range(dimr))
    integral = total / (Nxi * Nxi)          # normalized Haar measure: dt dxi1 dxi2 / (4 pi^2), t in [0,1]
    sN_num = integral / nw2
    print(f"0(c) {s}: s_N exact (Schur) = {sN_exact};  cubature = {mp.nstr(sN_num, 40)}")
    check(f"0(c) {s}: cubature s_N agrees with dim rho/(2j+1) to 1e-40", abs(sN_num - mp.mpf(sN_exact.p) / sN_exact.q) < mp.mpf(10) ** -40)
    slotdata[s] = dict(j=j, n=n, irr=irr, Pon=Pon, eta=eta, rho=rho, sN=sN_exact)
    res["0(c)"][s] = {
        "irreducible (table name)": irr, "multiplicity at lowest level": 1,
        "s_N": str(sN_exact),
        "eta (columns; rows m = j..-j; 40 digits, real and imaginary parts)":
            [[[mp.nstr(mp.re(eta[r, a]), 40), mp.nstr(mp.im(eta[r, a]), 40)] for a in range(dimr)] for r in range(n)],
        "P = eta eta^dagger, exact, orthonormal CS basis": [[str(Pon[a, b]) for b in range(n)] for a in range(n)],
    }
    print(f"0(c) {s}: P exact (orthonormal basis) row 0:", [str(Pon[0, b]) for b in range(n)])

# ------------------------------------------------ 0(e) R2 quaternionic structure
d = slotdata["R2"]; j = d["j"]; eta = d["eta"]; rho = d["rho"]
# solve conj(rho(h)) C = C rho(h) for h = q1, q2: linear system for vec(C)
rows = []
for qq in (q1, q2):
    r_ = rho[qq]
    rb = mp.matrix([[mp.conj(r_[a, b]) for b in range(2)] for a in range(2)])
    for a in range(2):
        for b in range(2):
            row = [mp.mpc(0)] * 4
            # (rb C - C r)_{ab} = sum_c rb[a,c] C[c,b] - C[a,c] r[c,b]
            for c in range(2):
                row[c * 2 + b] += rb[a, c]
                row[a * 2 + c] -= r_[c, b]
            rows.append(row)
A = mp.matrix(rows)
U_, S_, V_ = mp.svd_c(A)
print("0(e) singular values of the intertwining system:", [mp.nstr(S_[k], 5) for k in range(len(S_))])
check("0(e) the space of C with conj(rho)C = C rho is one-dimensional", S_[3] < mp.mpf(10) ** -40 and S_[2] > mp.mpf(10) ** -5)
cv = V_.H[:, 3]
C = mp.matrix([[cv[0], cv[1]], [cv[2], cv[3]]])
CC = C.H * C
C = C / mp.sqrt(mp.re(CC[0, 0]))
check("0(e) C is unitary after scaling", maxabs(C.H * C - mp.eye(2)) < mp.mpf(10) ** -40)
check("0(e) conj(rho(h)) C = C rho(h) for all 120 h",
      max(maxabs(mp.matrix([[mp.conj(rho[qq][a, b]) for b in range(2)] for a in range(2)]) * C - C * rho[qq]) for qq in G) < mp.mpf(10) ** -40)
Cbar = mp.matrix([[mp.conj(C[a, b]) for b in range(2)] for a in range(2)])
CCb = C * Cbar
print("0(e) C conj(C) =", CCb)
check("0(e) C conj(C) = -1", maxabs(CCb + mp.eye(2)) < mp.mpf(10) ** -40)
U = to_mp(theta_U(j))
etab = mp.matrix([[mp.conj(eta[a, b]) for b in range(2)] for a in range(8)])
Mx = eta.H * U * etab * C
print("0(e) eta^dagger U conj(eta) C =", Mx)
cph = Mx[0, 0]
check("0(e) U conj(eta) C = c eta with c a scalar of modulus 1",
      maxabs(U * etab * C - eta * cph) < mp.mpf(10) ** -40 and abs(abs(cph) - 1) < mp.mpf(10) ** -40)
Uinv_T = (theta_U(j).inv()).T
check("0(e) (U^-1)^T = U on V_7/2 (exact)", Uinv_T == theta_U(j))
# rephase C so that c = 1: C0 = C / c
C0 = C / cph
# direct test of psi_w -> conj(psi_w) C0 against psi_{Theta w} at random g
random.seed(3)
wv = mp.matrix([mp.mpc(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(8)])
Tw = U * mp.matrix([mp.conj(x) for x in wv])
err = mp.mpf(0)
for _ in range(5):
    gg = rand_su2(); Dg = D_mp(j, gg)
    lhs = mp.matrix([[mp.conj(x) for x in (wv.T * Dg * eta)]]) * C0
    rhs = Tw.T * Dg * eta
    err = max(err, maxabs(lhs - rhs))
check("0(e) conj(psi_w) C0 = psi_{Theta w} at random points (map = Theta when c = 1)", err < mp.mpf(10) ** -40, mp.nstr(err, 3))
res["0(e)"] = {
    "C (unitary, phase chosen so that the map is exactly Theta; 30 digits)":
        [[[mp.nstr(mp.re(C0[a, b]), 30), mp.nstr(mp.im(C0[a, b]), 30)] for b in range(2)] for a in range(2)],
    "C conj(C)": "-1",
    "map psi -> conj(psi) C on w": "w -> c Theta w, c = the unimodular scalar with U conj(eta) C = c eta; c = 1 for the C given (C -> e^{ia} C multiplies c by e^{ia})",
    "Theta^2 on V_7/2": "-1",
    "Sp(1) action": "w -> a w + b Theta w with |a|^2+|b|^2 = 1 (unit quaternions a + b j, i and j = Theta acting on E_0, Theta i = -i Theta); E_0 = V_7/2 becomes a left H-module of quaternionic dimension 4",
}
# save numerical eta and the exact projectors for later scripts
import pickle
pickle.dump({s: dict(j=str(d["j"]), n=d["n"], irr=d["irr"], sN=str(d["sN"]),
                     Pon=[[str(d["Pon"][a, b]) for b in range(d["n"])] for a in range(d["n"])],
                     eta=[[(str(mp.re(d["eta"][r, a])), str(mp.im(d["eta"][r, a]))) for a in range(d["eta"].cols)] for r in range(d["n"])])
             for s, d in slotdata.items()}, open(os.path.join(OUT, "slots.pkl"), "wb"))
json.dump(res, open(os.path.join(OUT, "anchors_cde.json"), "w"), indent=1)
print("failures:", n_fails())
