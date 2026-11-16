"""The reduced quartic h for R4 and R5, exact, saved as quartic_R4.json / quartic_R5.json for
Block D.  Checks: (i) imaginary part cancels; (ii) exact invariance under the four bench
generators of inputs.json (grad h . X_i x == 0 as polynomials); (iii) h agrees with the
definition Q||w||^4 = V int_X |psi_w|^4 / ... computed by exact-degree cubature over SU(2) at
40 digits (independent route); (iv) the Casimir projectors agree with the Racah CG norms
||[w (x) w]_J||^2; (v) generators are G-skew and X_1^2 = -1 (structure used in Block D).
The channel coefficients printed here are ingredients of h, not answers to item A.1."""
import json, os, random
from fractions import Fraction as Fr
import mpmath as mpm
from lib_quartic import save, pdiff, Compiled, NV
from lib_alg import cg_exact, weights
from lib_rep import eta_num, SLOTS
from lib_num import haar_average
from lib_check import check, finish

HERE = os.path.dirname(os.path.abspath(__file__))
mpm.mp.dps = 50


def load_generators():
    with open(os.path.join(HERE, "inputs.json")) as f:
        inp = json.load(f)
    return [[[Fr(v) for v in row] for row in M] for M in inp["generators"]["matrices"]]


GDIAG = [720, 120, 48, 36, 48, 120, 720] * 2


def x_to_w(x):
    F = weights(6)
    return [mpm.mpc(x[a], x[7 + a]) * mpm.sqrt(F[a]) for a in range(7)]


def main():
    gens = load_generators()
    # (v) structure of the generators: G X skew-symmetric, X_1^2 = -1, [X_1, X_i] = 0
    def mat(M, N):
        return [[sum(M[i][k] * N[k][j] for k in range(NV)) for j in range(NV)] for i in range(NV)]
    skew = all(GDIAG[i] * X[i][j] + GDIAG[j] * X[j][i] == 0 for X in gens for i in range(NV) for j in range(NV))
    check("G X_i is antisymmetric for the four bench generators (exact)", skew)
    X1sq = mat(gens[0], gens[0])
    check("X_1^2 == -I (phase generator is the complex structure i)",
          all(X1sq[i][j] == (-1 if i == j else 0) for i in range(NV) for j in range(NV)))
    comm = all(mat(gens[0], X) == mat(X, gens[0]) for X in gens[1:])
    check("X_1 commutes with the three rotation generators (exact)", comm)

    for name in ("R4", "R5"):
        B = save(name, os.path.join(HERE, f"quartic_{name}.json"))
        poly = B["poly"]
        print(f"{name}: s_N = {B['s_N']}; c_J = {{{', '.join(f'{J}: {v}' for J, v in B['c_J'].items())}}}")
        print(f"{name}: q_J = c_J/s_N^2 = {{{', '.join(f'{J}: {v}' for J, v in B['q_J'].items())}}}; "
              f"{len(poly)} monomials in h")
        check(f"{name}: imaginary part of h cancels exactly", B["imag_residue_terms"] == 0)
        from lib_quartic import build
        Bodd = build(name, Jfilter=lambda J: J % 2 == 1)
        check(f"{name}: odd-J channels contribute the zero polynomial (w (x) w is symmetric)",
              len(Bodd["poly"]) == 0 and any(B["c_J"][J] != 0 for J in B["c_J"] if J % 2 == 1),
              f"{len(Bodd['poly'])} odd-J monomials")
        # (ii) invariance
        grads = [pdiff(poly, k) for k in range(NV)]
        inv_ok = True
        for X in gens:
            acc = {}
            for a in range(NV):
                for b in range(NV):
                    if X[a][b] == 0:
                        continue
                    for e, c in grads[a].items():
                        e2 = list(e)
                        e2[b] += 1
                        acc[tuple(e2)] = acc.get(tuple(e2), 0) + c * X[a][b]
            inv_ok &= all(v == 0 for v in acc.values())
        check(f"{name}: grad h . X_i x == 0 identically for i = 1..4 (exact)", inv_ok)
        # (iii) definition by cubature, (iv) CG route
        hc = Compiled(poly, lambda c: mpm.mpf(c.numerator) / c.denominator)
        eta = eta_num(name)
        d, n = SLOTS[name]
        rnd = random.Random(11 + d)
        worst_q, worst_cg = mpm.mpf(0), mpm.mpf(0)
        mpm.mp.dps = 40
        for trial in range(2):
            x = [mpm.mpf(rnd.uniform(-0.1, 0.1)) for _ in range(NV)]
            w = x_to_w(x)
            hpoly = hc.eval(x, mpm.mpf(0))

            def f4(D):
                wD = [mpm.fsum(w[r] * D[r][s] for r in range(7)) for s in range(7)]
                tot = mpm.mpf(0)
                for a_ in range(d):
                    ph = mpm.fsum(wD[s] * eta[s][a_] for s in range(7))
                    tot += abs(ph) ** 2
                return tot ** 2
            avg = haar_average(f4, n, nang=16, nbeta=16)
            sN = mpm.mpf(B["s_N"].numerator) / B["s_N"].denominator
            hdef = avg / sN ** 2
            worst_q = max(worst_q, abs(hdef - hpoly) / abs(hpoly))
            # CG route
            tot = mpm.mpf(0)
            for J, q in B["q_J"].items():
                if q == 0:
                    continue
                for M in range(-int(J), int(J) + 1):
                    comp = mpm.mpc(0)
                    for m1 in range(-3, 4):
                        m2 = M - m1
                        if abs(m2) > 3:
                            continue
                        r, R = cg_exact(3, m1, 3, m2, J, M)
                        if R == 0:
                            continue
                        cgv = mpm.mpf(r.numerator) / r.denominator * mpm.sqrt(mpm.mpf(R.numerator) / R.denominator)
                        comp += cgv * w[3 - m1] * w[3 - m2]
                    tot += mpm.mpf(q.numerator) / q.denominator * abs(comp) ** 2
            worst_cg = max(worst_cg, abs(tot - hpoly) / abs(hpoly))
            nw = mpm.fsum(abs(z) ** 2 for z in w)
            print(f"{name}: trial {trial}: Q(w) from cubature = {mpm.nstr(hdef / nw ** 2, 35)}, "
                  f"from polynomial = {mpm.nstr(hpoly / nw ** 2, 35)}")
        mpm.mp.dps = 50
        check(f"{name}: polynomial h == definition (40-digit SU(2) cubature)", worst_q < mpm.mpf(10) ** -35,
              mpm.nstr(worst_q, 3))
        check(f"{name}: polynomial h == sum_J q_J ||[w(x)w]_J||^2 with Racah CG", worst_cg < mpm.mpf(10) ** -35,
              mpm.nstr(worst_cg, 3))
    finish()


if __name__ == "__main__":
    main()
