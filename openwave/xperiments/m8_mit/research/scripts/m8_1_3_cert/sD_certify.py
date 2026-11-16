"""Block D: the six boxes O1..O3 in R4 and R5 (order: O3 R4, O3 R5 first, as the brief says).
Krawczyk's test for the slice system on each box of Section 5 (radius 1e-60, centre's lambda
from our own h, c = 0); then refinement to a smaller certified box, and at the certified point:
924 r6_hat, signature, k (Gram determinant), r_u, zero multiplicity of S_hat, slow roots."""
import json, os
from fractions import Fraction as Fr
import mpmath as mpm
from mpmath import iv
from lib_cert import (set_prec, load_inputs, Slot, SliceSystem, make_box, refine, lo, hi, midf,
                      width, excludes_zero, sign_of, NV, GDIAG)
from lib_analyze import analyze
from lib_check import check, finish

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["O3 R4", "O3 R5", "O1 R4", "O1 R5", "O2 R4", "O2 R5"]


def out_round(I, digits=70):
    """decimal enclosure [floor, ceil] of interval I at `digits` significant digits (outward)."""
    a, b = lo(I), hi(I)
    m = max(abs(a), abs(b))
    if m == 0:
        return ["0", "0"]
    e = int(mpm.floor(mpm.log10(m))) - digits + 1
    sc = mpm.mpf(10) ** (-e)
    la = mpm.floor(a * sc)
    ub = mpm.ceil(b * sc)

    def fmt(n):
        n = int(n)
        s = str(abs(n))
        sign = "-" if n < 0 else ""
        if e >= 0:
            return sign + s + "0" * e
        k = -e
        s = s.rjust(k + 1, "0")
        return sign + s[:-k] + "." + s[-k:]
    return [fmt(la), fmt(ub)]


def main():
    set_prec()
    inp = load_inputs()["boxes"]["boxes"]
    slots = {nm: Slot(nm) for nm in ("R4", "R5")}
    results = {}
    for box in ORDER:
        oname, sname = box.split()
        slot = slots[sname]
        S = SliceSystem(slot, inp[box]["x0"])
        Z, centre, lam_c = make_box(S, inp[box]["centre, x-part"])
        ok, K, info = S.krawczyk(Z)
        check(f"{box}: Krawczyk K(Z) in int(Z) on the Section-5 box (outward-rounded)", ok,
              ", ".join(f"{k} = {mpm.nstr(v, 4)}" for k, v in info.items()))
        # the zero lies in K(Z); check K(Z) lies inside the exact decimal box (margin 1e-150)
        tx = [mpm.mpf(s) for s in inp[box]["centre, x-part"]] + [lam_c] + [mpm.mpf(0)] * 4
        r = mpm.mpf("1e-60")
        margin = mpm.mpf("1e-150")
        inside_exact = all(lo(K[i]) > tx[i] - r + margin and hi(K[i]) < tx[i] + r - margin for i in range(19))
        check(f"{box}: K(Z) lies inside the exact box of Section 5", inside_exact)
        c_has0 = all(not excludes_zero(K[NV + 1 + i]) for i in range(4))
        check(f"{box}: the c-part of K(Z) contains 0 (consistency: c = 0 at the zero)", c_has0)
        Zc, hist, okr = refine(S, Z)
        ok_steps = [h[0] for h in hist]
        check(f"{box}: refined box is certified by its own Krawczyk test", okr and any(ok_steps[1:]),
              f"steps {ok_steps}, widths {[mpm.nstr(h[1]['max width K'], 3) for h in hist]}")
        wc = max(width(v) for v in Zc)
        A = analyze(slot, S.Xiv, Zc[:NV])
        lam = Zc[NV]
        r6 = A["924 r6"]
        check(f"{box}: 924 r6 enclosure width <= 1e-50", width(r6) <= mpm.mpf("1e-50"), mpm.nstr(width(r6), 3))
        check(f"{box}: k = 4 (Gram determinant > 0)", A["k"] == 4, mpm.nstr(midf(A["Gram det"]), 20))
        check(f"{box}: r_u certified (rank of sigma|K_u = 2)", A["r_u"] == 2, str(A["Omega rotation entries certified nonzero"]))
        check(f"{box}: signature certified (all 9 Gershgorin discs exclude 0)", A["signature"] is not None,
              str(A["signature"]))
        check(f"{box}: zero multiplicity certified (a_6 excludes 0)", A["zero multiplicity"] == 6,
              mpm.nstr(midf(A["charpoly"][6]), 10))
        check(f"{box}: structural zeros of char_S enclose 0 (consistency)",
              A["charpoly: a_0..a_5 and odd a_k enclose 0 (consistency)"])
        check(f"{box}: slow roots certified real and simple (4 disjoint sign changes)",
              A["slow roots certified real simple"])
        roots = A["slow roots"] or []
        rw = max((width(v) for v in roots), default=None)
        if roots:
            check(f"{box}: slow-root enclosures width <= 1e-50", rw <= mpm.mpf("1e-50"), mpm.nstr(rw, 3))
        res = {
            "Krawczyk on Section-5 box": ok,
            "Krawczyk info": {k: mpm.nstr(v, 6) for k, v in info.items()},
            "zero inside exact Section-5 box": inside_exact,
            "certified box max width": mpm.nstr(wc, 6),
            "lambda enclosure": out_round(lam),
            "h(u) = lambda/4 enclosure": out_round(lam / 4),
            "924 r6": out_round(r6),
            "Gram det": out_round(A["Gram det"], 40),
            "k": A["k"],
            "r_u": A["r_u"],
            "sigma|K_u nonzero rotation entries (indices of generators 1..4 from 0)": A["Omega rotation entries certified nonzero"],
            "signature": A["signature"],
            "Gershgorin discs": [[mpm.nstr(a, 8), mpm.nstr(b, 8)] for a, b in A["Gershgorin discs (N_u, congruent form)"]],
            "zero multiplicity": A["zero multiplicity"],
            "a_6 (coefficient of xi^6 in char S)": out_round(A["charpoly"][6], 40),
            "slow polynomial coefficients nu^0..nu^4": [out_round(c, 60) for c in A["slow coefficients (nu^0..nu^4)"]],
            "slow roots certified real simple": A["slow roots certified real simple"],
            "slow roots": [out_round(v, 70) for v in roots],
            "slow root signs": [sign_of(v) for v in roots],
            "slow roots (approx, 30 digits)": [mpm.nstr(mpm.re(z), 30) for z in A["slow roots approx"]],
        }
        results[box] = res
        print(f"{box}: lambda = {mpm.nstr(midf(lam), 30)}, 924 r6 = {mpm.nstr(midf(r6), 40)}")
        print(f"   k = {A['k']}, r_u = {A['r_u']}, signature = {A['signature']}, zero mult = {A['zero multiplicity']}")
        print(f"   slow roots: {res['slow roots (approx, 30 digits)']}, signs {res['slow root signs']}")
        print(f"   certified box width {mpm.nstr(wc, 4)}; Krawczyk steps {[(h[0], mpm.nstr(h[1]['max width K'], 3)) for h in hist]}")
    with open(os.path.join(HERE, "out_D.json"), "w") as f:
        json.dump(results, f, indent=1)
    finish()


if __name__ == "__main__":
    main()
