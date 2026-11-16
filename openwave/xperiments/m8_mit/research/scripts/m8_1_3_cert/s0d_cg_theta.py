"""Item 0(d): sign of <3 3; 3 -3 | 6 0> from the Racah formula built in lib_alg (Condon-Shortley),
and Theta^2 on V_3 and V_{7/2} from the definition (Theta u)_m = (-1)^{j-m} conj(u_{-m}).
Also: self-tests of the Clebsch-Gordan builder (orthogonality, the CS sign condition), and a
comparison with sympy's library CG (a lookup, reported separately)."""
import json, os
from fractions import Fraction as Fr
from lib_alg import cg_exact
from lib_check import check, finish

HERE = os.path.dirname(os.path.abspath(__file__))


def half_range(j):
    return [j - k for k in range(int(2 * j) + 1)]


def theta(u, j):
    """u: dict m -> complex (python complex with exact small ints is enough here; we use Fractions pairs)."""
    out = {}
    for m in half_range(j):
        re, im = u[-m]
        sgn = (-1) ** int(j - m)
        out[m] = (sgn * re, -sgn * im)
    return out


def main():
    r, R = cg_exact(3, 3, 3, -3, 6, 0)
    val_sq = r * r * R
    sign = 1 if r > 0 else -1
    print(f"<3 3; 3 -3 | 6 0> = {r} * sqrt({R});  square = {val_sq};  sign = {sign:+d}")
    check("<3 3;3 -3|6 0>^2 == 1/924 (= C(6,6)C(6,0)/C(12,6), stretched-coupling formula)",
          val_sq == Fr(1, 924))
    # CG builder self-tests: orthogonality in (J,M) for j1 = j2 = 3 and 7/2, and CS sign rule
    for j in (Fr(3), Fr(7, 2), Fr(1), Fr(1, 2)):
        ms = half_range(j)
        Js = [Fr(J) for J in range(0, int(2 * j) + 1)]
        ok = True
        for J in Js:
            for Jp in Js:
                for M in [J - k for k in range(int(2 * J) + 1)]:
                    if abs(M) > Jp:
                        continue
                    s = Fr(0)
                    # sum over m1 of <..|J M><..|J' M>, both real r*sqrt(R): products are rational
                    # only up to sqrt(R R'); we test the exact squared sum when J == J' and use
                    # floating test for J != J' below.
                    if J == Jp:
                        for m1 in ms:
                            m2 = M - m1
                            if abs(m2) > j:
                                continue
                            a, A = cg_exact(j, m1, j, m2, J, M)
                            s += a * a * A
                        ok &= (s == 1)
        check(f"sum_m1 <j m1; j M-m1|J M>^2 == 1 for all J, M (j = {j})", ok)
        okx = True
        import mpmath as mpm
        mpm.mp.dps = 40
        for J in Js:
            for Jp in Js:
                if J >= Jp:
                    continue
                for M in [Jp - k for k in range(int(2 * Jp) + 1)]:
                    if abs(M) > J:
                        continue
                    s = mpm.mpf(0)
                    for m1 in ms:
                        m2 = M - m1
                        if abs(m2) > j:
                            continue
                        a, A = cg_exact(j, m1, j, m2, J, M)
                        b, B = cg_exact(j, m1, j, m2, Jp, M)
                        s += mpm.mpf(a.numerator) / a.denominator * mpm.sqrt(mpm.mpf(A.numerator) / A.denominator) * \
                            mpm.mpf(b.numerator) / b.denominator * mpm.sqrt(mpm.mpf(B.numerator) / B.denominator)
                    okx &= abs(s) < mpm.mpf(10) ** -35
        check(f"orthogonality between J != J' (40 digits, j = {j})", okx)
        rr, RR = cg_exact(j, j, j, -j, 2 * j, 0)
        cs_ok = rr > 0 and RR > 0
        check(f"CS sign rule <j j; j -j|2j 0> > 0 (j = {j})", cs_ok)
    # library comparison (lookup): sympy.physics.quantum.cg
    from sympy.physics.quantum.cg import CG
    from sympy import Rational, sqrt, nsimplify, simplify
    agree = True
    nchecked = 0
    for j in (Fr(3), Fr(7, 2)):
        for J in range(0, int(2 * j) + 1):
            for m1 in half_range(j):
                for m2 in half_range(j):
                    M = m1 + m2
                    if abs(M) > J:
                        continue
                    a, A = cg_exact(j, m1, j, m2, J, M)
                    mine = Rational(a.numerator, a.denominator) * sqrt(Rational(A.numerator, A.denominator))
                    lib = CG(Rational(j.numerator, j.denominator), Rational(m1.numerator, m1.denominator),
                             Rational(j.numerator, j.denominator), Rational(m2.numerator, m2.denominator),
                             J, Rational(M.numerator, M.denominator)).doit()
                    if simplify(mine - lib) != 0:
                        agree = False
                    nchecked += 1
    print(f"sympy library CG compared on {nchecked} coefficients (j = 3 and 7/2): agree = {agree}")
    check("own Racah CG == sympy library CG on all j=3, 7/2 couplings", agree)

    # Theta^2 on V_3 and V_{7/2}, applied to every basis vector and to i*basis vector
    res = {}
    for j in (Fr(3), Fr(7, 2)):
        ms = half_range(j)
        factors = set()
        for m0 in ms:
            for ph in ((Fr(1), Fr(0)), (Fr(0), Fr(1))):
                u = {m: (Fr(0), Fr(0)) for m in ms}
                u[m0] = ph
                t2 = theta(theta(u, j), j)
                # find c with t2 = c u
                c = (t2[m0][0] * ph[0] + t2[m0][1] * ph[1], t2[m0][1] * ph[0] - t2[m0][0] * ph[1])
                for m in ms:
                    if m != m0:
                        assert t2[m] == (0, 0)
                factors.add(c)
        res[str(j)] = factors
        print(f"Theta^2 on V_{j}: multiplies every basis vector (and i times it) by {factors}")
    check("Theta^2 on V_3 is a single scalar", len(res["3"]) == 1)
    check("Theta^2 on V_7/2 is a single scalar", len(res["7/2"]) == 1)
    t3 = list(res["3"])[0][0]
    t72 = list(res["7/2"])[0][0]
    with open(os.path.join(HERE, "out_0d.json"), "w") as f:
        json.dump({"cg_33_3m3_60": {"r": str(r), "R": str(R), "square": str(val_sq), "sign": sign},
                   "Theta2_V3": str(t3), "Theta2_V7/2": str(t72),
                   "sympy_library_agrees": agree}, f, indent=1)
    finish()


if __name__ == "__main__":
    main()
