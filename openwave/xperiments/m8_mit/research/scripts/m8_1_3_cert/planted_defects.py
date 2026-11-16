"""Planted defects: the same predicates the scripts print as PASS, fed defective inputs, must
print FAIL.  Each block names the check family it exercises.  This script succeeds (exit 0)
only if every planted defect fires."""
import json, os, random, sys
from fractions import Fraction as Fr
import mpmath as mpm
from mpmath import iv
import lib_check
from lib_check import check

HERE = os.path.dirname(os.path.abspath(__file__))
fired = []


def expect_fail(name, cond, detail=""):
    print(f"--- planted: {name}")
    r = check(name, cond, detail)
    fired.append((name, not r))
    lib_check.FAILS.clear()


def main():
    # 1. exact norm check (s0a): q2 with phi replaced by 8/5
    from lib_alg import Q5, qnorm2, HALF, PHI_INV
    q2bad = (Q5(Fr(8, 5)) * HALF, PHI_INV * HALF, Q5(HALF), Q5(0))
    expect_fail("|q2|^2 == 1 exactly  (defect: phi -> 8/5)", qnorm2(q2bad) == Q5(1), repr(qnorm2(q2bad)))

    # 2. character-table orthonormality (s0a): perturb one value of one derived character
    from s0b_levels import setup
    from lib_alg import inner
    G, classes, irr, sizes, reps = setup()
    bad = [list(c) for c in irr]
    bad[3][4] = bad[3][4] + Q5(Fr(1, 10))
    ok = all(inner(bad[a], bad[b], sizes, len(G)) == Q5(1 if a == b else 0)
             for a in range(len(bad)) for b in range(len(bad)))
    expect_fail("row orthonormality of the character table (defect: one value + 1/10)", ok)

    # 3. name picks exactly one irrep (s0b): a name (dim 4, lowest level 7) that matches none
    from lib_alg import cheb_U
    def lowest(chi):
        for l in range(41):
            if inner([cheb_U(l, r[0]) for r in reps], chi, sizes, len(G)) != Q5(0):
                return l
    hits = [c for c in irr if int(c[0].a) == 4 and lowest(c) == 7]
    expect_fail("name picks exactly one irreducible (defect: name (4, 7))", len(hits) == 1, f"{len(hits)} hits")

    # 4. projector idempotence (s0c): 2P
    from lib_rep import projector_p
    from lib_alg import matmulK
    P = projector_p("R4")
    P2 = [[v * 2 for v in row] for row in P]
    expect_fail("P^2 == P (defect: P -> 2P)", matmulK(P2, P2) == P2)

    # 5. CG vs library (s0d): flip the sign of one coefficient
    from lib_alg import cg_exact
    from sympy.physics.quantum.cg import CG
    from sympy import Rational, sqrt, simplify
    a, A = cg_exact(3, 1, 3, -1, 2, 0)
    mine = -Rational(a.numerator, a.denominator) * sqrt(Rational(A.numerator, A.denominator))
    lib = CG(3, 1, 3, -1, 2, 0).doit()
    expect_fail("own CG == sympy CG (defect: sign flipped)", simplify(mine - lib) == 0)

    # 6. Theta^2 a single scalar (s0d): a defective Theta with no sign on m >= 0
    from s0d_cg_theta import half_range
    j = Fr(3)
    def theta_bad(u):
        out = {}
        for m in half_range(j):
            re, im = u[-m]
            sgn = (-1) ** int(j - m) if m < 0 else 1
            out[m] = (sgn * re, -sgn * im)
        return out
    facs = set()
    for m0 in half_range(j):
        u = {m: (Fr(0), Fr(0)) for m in half_range(j)}
        u[m0] = (Fr(1), Fr(0))
        t2 = theta_bad(theta_bad(u))
        facs.add(t2[m0])
    expect_fail("Theta^2 on V_3 is a single scalar (defect: sign dropped for m >= 0)", len(facs) == 1, str(facs))

    # 7. tr rho(h) == chi(h) (s0c): R4's intertwiner against R5's character
    from lib_rep import eta_num, group_data, irrep_char, all_D
    from lib_alg import weights
    mpm.mp.dps = 30
    G_, classes_, irr_, sizes_, reps_, cls_of = group_data()
    eta = eta_num("R4")
    chi5 = irrep_char("R5")
    F = weights(6)
    worst = mpm.mpf(0)
    for g, M in list(zip(G_, all_D(6)))[:20]:
        Dw = [[M[r][s].mpc(mpm) * mpm.sqrt(mpm.mpf(F[r]) / F[s]) for s in range(7)] for r in range(7)]
        tr = mpm.fsum(mpm.conj(eta[r][a]) * Dw[r][k] * eta[k][a] for a in range(3) for r in range(7) for k in range(7))
        worst = max(worst, abs(tr - chi5[cls_of[g]].mp(mpm)))
    expect_fail("tr rho(h) == chi(h) (defect: R4's eta against R5's character)", worst < mpm.mpf(10) ** -25, mpm.nstr(worst, 3))

    # 8. invariance of h (s1): add x_0^4 to h
    from lib_quartic import load, pdiff, NV
    from lib_cert import generators_exact
    poly, meta = load(os.path.join(HERE, "quartic_R4.json"))
    bad = dict(poly)
    e = tuple([4] + [0] * 13)
    bad[e] = bad.get(e, 0) + Fr(1, 1000)
    gens = generators_exact()
    grads = [pdiff(bad, k) for k in range(NV)]
    inv_ok = True
    for X in gens:
        acc = {}
        for a in range(NV):
            for b in range(NV):
                if X[a][b] == 0:
                    continue
                for ee, c in grads[a].items():
                    e2 = list(ee); e2[b] += 1
                    acc[tuple(e2)] = acc.get(tuple(e2), 0) + c * X[a][b]
        inv_ok &= all(v == 0 for v in acc.values())
    expect_fail("grad h . X_i x == 0 (defect: h + x_0^4/1000)", inv_ok)

    # 9. polynomial h == definition by cubature (s1): wrong s_N (1/2 instead of 3/7)
    from lib_quartic import Compiled
    from lib_num import haar_average
    from s1_quartic import x_to_w
    mpm.mp.dps = 30
    hc = Compiled(poly, lambda c: mpm.mpf(c.numerator) / c.denominator)
    rnd = random.Random(5)
    x = [mpm.mpf(rnd.uniform(-0.1, 0.1)) for _ in range(NV)]
    w = x_to_w(x)
    eta = eta_num("R4")
    def f4(D):
        wD = [mpm.fsum(w[r] * D[r][s] for r in range(7)) for s in range(7)]
        return mpm.fsum(abs(mpm.fsum(wD[s] * eta[s][a] for s in range(7))) ** 2 for a in range(3)) ** 2
    avg = haar_average(f4, 6, nang=16, nbeta=16)
    hdef = avg / mpm.mpf(0.5) ** 2
    hp = hc.eval(x, mpm.mpf(0))
    expect_fail("h == definition (defect: s_N = 1/2)", abs(hdef - hp) / abs(hp) < mpm.mpf(10) ** -25,
                mpm.nstr(abs(hdef - hp) / abs(hp), 3))

    # 10-12. Krawczyk (sD): (a) centre shifted by 3e-60 in one coordinate, (b) a perturbed h,
    #        (c) box radius 1e-45 -> 924 r6 width check
    from lib_cert import set_prec, load_inputs, Slot, SliceSystem, make_box, width
    from lib_analyze import analyze, r6_924, certify_real_roots
    set_prec()
    inp = load_inputs()["boxes"]["boxes"]["O3 R4"]
    slot = Slot("R4")
    S = SliceSystem(slot, inp["x0"])
    cx = list(inp["centre, x-part"])
    cx[0] = str(mpm.mpf(cx[0]) + mpm.mpf("3e-60"))
    Z, c, lam = make_box(S, cx)
    ok, K, info = S.krawczyk(Z)
    expect_fail("Krawczyk K(Z) in int(Z) (defect: centre x_0 shifted by 3e-60)", ok)
    badpoly = dict(poly)
    # a monomial in the real parts only (the O3 point is real, so it moves the critical point)
    some = sorted(e for e in badpoly if sum(e[7:]) == 0 and e[0] > 0 and e[2] > 0)[0]
    badpoly[some] = badpoly[some] * Fr(1001, 1000)
    slot_bad = Slot("R4", poly=badpoly)
    Sb = SliceSystem(slot_bad, inp["x0"])
    Zb, cb, lamb = make_box(Sb, inp["centre, x-part"])
    okb, Kb, infob = Sb.krawczyk(Zb)
    expect_fail("Krawczyk K(Z) in int(Z) (defect: one coefficient of h times 1.001)", okb)
    Zw, cw, lamw = make_box(S, inp["centre, x-part"], radius="1e-45")
    r6 = r6_924(Zw[:14])
    expect_fail("924 r6 enclosure width <= 1e-50 (defect: box radius 1e-45)", width(r6) <= mpm.mpf("1e-50"),
                mpm.nstr(width(r6), 3))

    # 13. k, r_u, signature, zero multiplicity at a point with a stabiliser (zonal v0, R4,
    #     a thin box): the Gram determinant must not certify k = 4
    x_v0 = [iv.mpf(0)] * 14
    x_v0[3] = 1 / iv.sqrt(iv.mpf(36))
    A = analyze(slot, S.Xiv, x_v0) if False else None
    from lib_analyze import det_interval
    from lib_cert import matvec, GDIAG, lo
    kk = [matvec(X, x_v0) for X in S.Xiv]
    Gram = [[sum((GDIAG[a] * kk[i][a] * kk[j][a] for a in range(14)), iv.mpf(0)) for j in range(4)] for i in range(4)]
    try:
        gd = det_interval(Gram)
        kval = 4 if lo(gd) > 0 else None
    except ArithmeticError:
        kval = None
    expect_fail("k = 4 from Gram determinant (defect: zonal v0, fixed by rotations about z)", kval == 4)

    # 14. slow roots certified real and simple: polynomials with a complex pair / a double root
    ivc = lambda v: iv.mpf(v)
    cpx = [ivc(2), ivc(3), ivc(3), ivc(3), ivc(1)]          # (nu^2+1)(nu+1)(nu+2)
    okc, _, _ = certify_real_roots(cpx)
    expect_fail("slow roots certified real and simple (defect: (nu^2+1)(nu+1)(nu+2))", okc)
    dbl = [ivc(6), ivc(17), ivc(17), ivc(7), ivc(1)]       # (nu+1)^2 (nu+2)(nu+3)
    okd, _, _ = certify_real_roots(dbl)
    expect_fail("slow roots certified real and simple (defect: (nu+1)^2(nu+2)(nu+3))", okd)

    # 15. Gershgorin signature certification on a matrix with a zero eigenvalue
    M = [[iv.mpf(-1), iv.mpf("1e-30"), iv.mpf(0)], [iv.mpf("1e-30"), iv.mpf(0), iv.mpf(0)], [iv.mpf(0), iv.mpf(0), iv.mpf(2)]]
    from lib_cert import hi
    discs = []
    for i in range(3):
        rad = sum((max(abs(lo(M[i][j])), abs(hi(M[i][j]))) for j in range(3) if j != i), mpm.mpf(0))
        discs.append((lo(M[i][i]) - rad, hi(M[i][i]) + rad))
    neg = sum(1 for a, b in discs if b < 0)
    pos = sum(1 for a, b in discs if a > 0)
    expect_fail("signature certified: all Gershgorin discs exclude 0 (defect: a zero eigenvalue)", neg + pos == 3)

    n_fired = sum(1 for _, f in fired if f)
    print(f"planted defects fired: {n_fired}/{len(fired)}")
    for nm, f in fired:
        if not f:
            print("  DID NOT FIRE:", nm)
    sys.exit(0 if n_fired == len(fired) else 1)


if __name__ == "__main__":
    main()
