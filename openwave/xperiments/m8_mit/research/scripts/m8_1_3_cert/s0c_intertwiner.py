"""Item 0(c) for R4 and R5: multiplicity at the lowest level, an intertwiner eta with
eta^dagger eta = I, and the constant s_N with int_X |phi_w|^2 dV = s_N V ||w||^2.
Exact route for s_N: Schur orthogonality, avg_g D_mm'(g) conj(D_nn'(g)) = delta delta/(2j+1),
so avg |phi_w|^2 = ||w||^2 tr(eta^dagger eta)/(2j+1) = ||w||^2 dim(rho)/(2j+1).
High-precision route: exact-degree cubature over SU(2) at 40 digits for random w."""
import json, os, random
from fractions import Fraction as Fr
import mpmath as mpm
from lib_alg import K, Q5, Dp_exact, matmulK, su2_of_quat, generators, qmul, weights
from lib_rep import group_data, projector_p, eta_num, all_D, irrep_char, SLOTS, to_w_num
from lib_num import haar_average, Dw_num, su2_euler
from lib_check import check, finish

HERE = os.path.dirname(os.path.abspath(__file__))
mpm.mp.dps = 60


def kstr(z):
    return f"({z.re.sympy_str()}) + I*({z.im.sympy_str()})"


def main():
    G, classes, irr, sizes, reps, cls_of = group_data()
    q1, q2 = generators()
    # --- the 2x2 map is a homomorphism (exact), and D^j is a homomorphism (exact)
    def m2(q):
        a, b, c, d = su2_of_quat(q)
        return [[a, b], [c, d]]
    ok2 = all(m2(qmul(g, h)) == matmulK(m2(g), m2(h)) for g in G[:30] for h in G[:30])
    check("quaternion -> SU(2) map is multiplicative on 30x30 group elements (exact)", ok2)
    for twoj in (1, 6):
        okD = all(Dp_exact(qmul(g, h), twoj) == matmulK(Dp_exact(g, twoj), Dp_exact(h, twoj))
                  for g in G[:8] for h in G[:8])
        check(f"D^(j) (2j={twoj}) multiplicative on 8x8 elements (exact)", okD)
    # D^j on the CS basis is unitary: exact criterion on the weighted matrix M:
    # (S M S^-1) unitary  <=>  M^dagger F M = F  with F = diag((j+m)!(j-m)!)
    F6 = weights(6)
    def unitary_ok(M, F):
        N = len(M)
        for r in range(N):
            for s in range(N):
                acc = K(0)
                for k in range(N):
                    acc = acc + M[k][r].conj() * M[k][s] * F[k]
                if not (acc == K(F[r] if r == s else 0)):
                    return False
        return True
    check("D^3(q1), D^3(q2) unitary in the CS basis (exact)",
          unitary_ok(Dp_exact(q1, 6), F6) and unitary_ok(Dp_exact(q2, 6), F6))
    # Euler factorisation used by the cubature agrees with the direct numeric D
    a, b, c, d = su2_euler(mpm.mpf("0.3"), mpm.mpf("1.1"), mpm.mpf("2.5"))
    Ddir = Dw_num(a, b, c, d, 6)
    a0, b0, c0, d0 = su2_euler(0, mpm.mpf("1.1"), 0)
    dm = Dw_num(a0, b0, c0, d0, 6)
    ms = [3 - i for i in range(7)]
    Dfac = [[mpm.expj(-ms[r] * mpm.mpf("0.3")) * dm[r][s] * mpm.expj(-ms[s] * mpm.mpf("2.5"))
             for s in range(7)] for r in range(7)]
    err = max(abs(Ddir[r][s] - Dfac[r][s]) for r in range(7) for s in range(7))
    check("Euler factorisation D(al,be,ga) = Dz(al) d(be) Dz(ga)", err < mpm.mpf(10) ** -50, mpm.nstr(err, 3))

    out = {}
    for name in ("R4", "R5"):
        dim, n = SLOTS[name]
        N = n + 1
        P = projector_p(name)
        P2 = matmulK(P, P)
        check(f"{name}: P^2 == P (exact)", P2 == P)
        tr = K(0)
        for r in range(N):
            tr = tr + P[r][r]
        mult = Fr(tr.re.a, dim)
        check(f"{name}: tr P is a rational multiple of dim", tr.im.is_zero() and tr.re.b == 0, kstr(tr))
        print(f"{name}: tr P = {kstr(tr)}  -> multiplicity at level {n} = tr P / dim = {mult}")
        F = weights(n)
        herm = all(P[r][s] * F[r] == P[s][r].conj() * F[s] for r in range(N) for s in range(N))
        check(f"{name}: P is Hermitian in the CS basis (exact: P_p[r][s] F_r == conj(P_p[s][r]) F_s)", herm)
        comm = all(matmulK(Dp_exact(q, n), P) == matmulK(P, Dp_exact(q, n)) for q in (q1, q2))
        check(f"{name}: P commutes with D^3(q1), D^3(q2) (exact)", comm)
        # P in the CS basis: P_w[r][s] = P_p[r][s] * sqrt(F_r/F_s)
        Pw_exact = [[f"({kstr(P[r][s])})*sqrt({Fr(F[r], F[s])})" for s in range(N)] for r in range(N)]
        eta = eta_num(name)
        # checks on eta
        etaH_eta = [[mpm.fsum(mpm.conj(eta[r][a]) * eta[r][b] for r in range(N)) for b in range(dim)]
                    for a in range(dim)]
        e1 = max(abs(etaH_eta[a][b] - (1 if a == b else 0)) for a in range(dim) for b in range(dim))
        check(f"{name}: eta^dagger eta == I (60 digits)", e1 < mpm.mpf(10) ** -55, mpm.nstr(e1, 3))
        Pw = to_w_num(P, n)
        e2 = max(abs(mpm.fsum(eta[r][a] * mpm.conj(eta[s][a]) for a in range(dim)) - Pw[r][s])
                 for r in range(N) for s in range(N))
        check(f"{name}: eta eta^dagger == P_w", e2 < mpm.mpf(10) ** -55, mpm.nstr(e2, 3))
        # intertwining and the character of rho(h) = eta^dagger D(h) eta, on all 120 elements
        Ds = all_D(n)
        worst_int, worst_chi = mpm.mpf(0), mpm.mpf(0)
        chi = irrep_char(name)
        for g, M in zip(G, Ds):
            Dw = [[M[r][s].mpc(mpm) * mpm.sqrt(mpm.mpf(F[r]) / F[s]) for s in range(N)] for r in range(N)]
            De = [[mpm.fsum(Dw[r][k] * eta[k][a] for k in range(N)) for a in range(dim)] for r in range(N)]
            rho = [[mpm.fsum(mpm.conj(eta[r][a]) * De[r][b] for r in range(N)) for b in range(dim)]
                   for a in range(dim)]
            eR = [[mpm.fsum(eta[r][k] * rho[k][b] for k in range(dim)) for b in range(dim)] for r in range(N)]
            worst_int = max(worst_int, max(abs(De[r][b] - eR[r][b]) for r in range(N) for b in range(dim)))
            trr = mpm.fsum(rho[a][a] for a in range(dim))
            worst_chi = max(worst_chi, abs(trr - chi[cls_of[g]].mp(mpm)))
        check(f"{name}: D(h) eta == eta rho(h) for all 120 h", worst_int < mpm.mpf(10) ** -50, mpm.nstr(worst_int, 3))
        check(f"{name}: tr rho(h) == chi_{name}(h) for all 120 h", worst_chi < mpm.mpf(10) ** -50, mpm.nstr(worst_chi, 3))
        # s_N: exact by Schur, numeric by cubature
        s_exact = Fr(dim, N)
        mpm.mp.dps = 40
        rnd = random.Random(7 + dim)
        worst = mpm.mpf(0)
        for trial in range(2):
            w = [mpm.mpc(rnd.uniform(-1, 1), rnd.uniform(-1, 1)) for _ in range(N)]
            nw2 = mpm.fsum(abs(x) ** 2 for x in w)

            def f2(D):
                tot = mpm.mpf(0)
                wD = [mpm.fsum(w[r] * D[r][s] for r in range(N)) for s in range(N)]
                for a_ in range(dim):
                    ph = mpm.fsum(wD[s] * eta[s][a_] for s in range(N))
                    tot += abs(ph) ** 2
                return tot
            avg = haar_average(f2, n, nang=8, nbeta=8)
            ratio = avg / nw2
            worst = max(worst, abs(ratio - mpm.mpf(s_exact.numerator) / s_exact.denominator))
            print(f"{name}: cubature avg|phi_w|^2/||w||^2 = {mpm.nstr(ratio, 35)}  (trial {trial})")
        mpm.mp.dps = 60
        check(f"{name}: s_N exact {s_exact} agrees with 40-digit cubature", worst < mpm.mpf(10) ** -35,
              mpm.nstr(worst, 3))
        print(f"{name}: s_N = {s_exact}")
        # exact eta: the same Gram-Schmidt done in sympy on the exact P_w
        import sympy as sp
        Pw_s = sp.Matrix(N, N, lambda r, s: sp.radsimp(sp.nsimplify(sp.sympify(Pw_exact[r][s]))))
        cols = []
        for s in range(N):
            v = Pw_s[:, s]
            for cc in cols:
                v = v - (cc.T * v)[0] * cc
            v = v.applyfunc(sp.radsimp)
            n2 = sp.radsimp(sp.simplify((v.T * v)[0]))
            if n2 != 0:
                cols.append((v / sp.sqrt(n2)).applyfunc(lambda x: sp.radsimp(sp.simplify(x))))
            if len(cols) == dim:
                break
        E = sp.Matrix.hstack(*cols)
        check(f"{name}: exact eta^T eta == I (sympy, exact)", sp.simplify(E.T * E - sp.eye(dim)) == sp.zeros(dim, dim))
        check(f"{name}: exact eta eta^T == P_w (sympy, exact)", sp.simplify(E * E.T - Pw_s) == sp.zeros(N, N))
        e3 = max(abs(mpm.mpf(sp.N(E[r, a], 70)) - eta[r][a]) for r in range(N) for a in range(dim))
        check(f"{name}: exact eta agrees with the 60-digit numeric eta", e3 < mpm.mpf(10) ** -55, mpm.nstr(e3, 3))
        print(f"{name}: exact eta (rows m = 3..-3): {E.tolist()}")
        out[name] = {"multiplicity": str(mult), "s_N": str(s_exact),
                     "eta exact (rows m = 3..-3, columns a)": [[str(E[r, a]) for a in range(dim)] for r in range(N)],
                     "P_w exact (entries, sympy strings), rows/cols m = 3..-3": Pw_exact,
                     "eta (rows m = 3..-3, columns a), 40 digits": [[mpm.nstr(x, 40) for x in row] for row in eta]}
        print(f"{name}: eta (rows m=3..-3), 20 digits:")
        for r in range(N):
            print("   ", [mpm.nstr(eta[r][a], 20) for a in range(dim)])
        print(f"{name}: P_p (weighted-basis projector, exact):")
        for r in range(N):
            print("   ", [kstr(P[r][s]) for s in range(N)])
    with open(os.path.join(HERE, "out_0c.json"), "w") as f:
        json.dump(out, f, indent=1)
    finish()


if __name__ == "__main__":
    main()
