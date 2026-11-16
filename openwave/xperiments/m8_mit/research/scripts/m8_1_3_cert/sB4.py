"""B.4 at O1, O2, O3 in R4 and R5, at the points certified by sD_certify.py (each box of
Section 5 passed our own Krawczyk test; enclosures are over our refined certified boxes,
which contain the certified point -- clause 2).  Signature, k, r_u, zero multiplicity, slow
roots (enclosed, sign, simple), slow-flow words, the R5/R4 ratio of each nonzero eigenvalue
of S_hat paired by clause 3, and 924 r6."""
import json, os
import mpmath as mpm
from mpmath import iv
from lib_cert import set_prec, lo, hi, midf, width, sign_of
from sD_certify import out_round
from lib_check import check, finish

HERE = os.path.dirname(os.path.abspath(__file__))


def words(signs, simple):
    """Section 3 words from the certified real roots: signs list (+1/-1), all simple (bool)."""
    if all(s < 0 for s in signs):
        return "elliptic, simple" if simple else "elliptic (multiplicities: see roots)"
    npos = sum(1 for s in signs if s > 0)
    if npos == 1 and simple:
        return "hyperbolic, one real pair"
    if npos >= 2:
        return "hyperbolic, real pairs"
    return "unclassified"


def plus_xi(nu):
    """+xi = the square root of nu with positive real part, or (real part 0) positive imaginary
    part (clause 3).  Returns (re, im) intervals for a real certified root interval nu."""
    if sign_of(nu) > 0:
        return iv.sqrt(nu), iv.mpf(0)
    if sign_of(nu) < 0:
        return iv.mpf(0), iv.sqrt(-nu)
    raise ArithmeticError("sign of nu not certified")


def main():
    set_prec()
    with open(os.path.join(HERE, "out_D.json")) as f:
        D = json.load(f)
    res = {"R4": {}, "R5": {}}
    for o in ("O3", "O1", "O2"):
        roots = {}
        for sl in ("R4", "R5"):
            box = f"{o} {sl}"
            d = D[box]
            check(f"{box}: point certified by our Krawczyk test", d["Krawczyk on Section-5 box"])
            rts = [iv.mpf(r) for r in d["slow roots"]]
            # clause 3 sort: real part, then imaginary part (all certified real here)
            order = sorted(range(len(rts)), key=lambda i: midf(rts[i]))
            for a, b in zip(order, order[1:]):
                assert hi(rts[a]) < lo(rts[b])
            rts = [rts[i] for i in order]
            roots[sl] = rts
            signs = [sign_of(r) for r in rts]
            w = words(signs, d["slow roots certified real simple"])
            res[sl][box] = {
                "signature": d["signature"], "k": d["k"], "r_u": d["r_u"],
                "zero multiplicity": d["zero multiplicity"],
                "slow roots (nu), sorted": [out_round(r, 70) for r in rts],
                "slow root widths": [mpm.nstr(width(r), 3) for r in rts],
                "slow root signs": ["-" if s < 0 else "+" for s in signs],
                "slow roots simple": d["slow roots certified real simple"],
                "slow-flow words": w,
                "924 r6": d["924 r6"],
                "924 r6 width": mpm.nstr(width(iv.mpf(d["924 r6"])), 3),
            }
            print(f"{box}: signature {d['signature']}, k {d['k']}, r_u {d['r_u']}, zero mult {d['zero multiplicity']}, "
                  f"words: {w}; signs {signs}")
        ratios = []
        for i, (n4, n5) in enumerate(zip(roots["R4"], roots["R5"])):
            check(f"{o}: root {i} has the same certified sign in R4 and R5", sign_of(n4) == sign_of(n5))
            r4, i4 = plus_xi(n4)
            r5, i5 = plus_xi(n5)
            if sign_of(n4) > 0:
                q = r5 / r4
            else:
                q = i5 / i4
            ratios.append({"pair": i, "+xi(R4)": out_round(r4 if sign_of(n4) > 0 else i4, 60),
                           "+xi(R4) is": "real" if sign_of(n4) > 0 else "i times the given positive number",
                           "+xi(R5)": out_round(r5 if sign_of(n5) > 0 else i5, 60),
                           "ratio +xi(R5)/+xi(R4) = -xi(R5)/-xi(R4)": out_round(q, 60),
                           "ratio width": mpm.nstr(width(q), 3)})
            print(f"  {o} pair {i}: ratio xi(R5)/xi(R4) in [{mpm.nstr(lo(q), 50)}, {mpm.nstr(hi(q), 50)}]  width {mpm.nstr(width(q), 3)}")
            check(f"{o}: ratio enclosure width <= 1e-50 (pair {i})", width(q) <= mpm.mpf("1e-50"))
        res["R5"][f"{o} R5"]["eigenvalue ratios R5/R4 (clause 3 pairing)"] = ratios
        res["R4"][f"{o} R4"]["eigenvalue ratios R5/R4 (clause 3 pairing)"] = ratios
    with open(os.path.join(HERE, "out_B4.json"), "w") as f:
        json.dump(res, f, indent=1)
    finish()


if __name__ == "__main__":
    main()
