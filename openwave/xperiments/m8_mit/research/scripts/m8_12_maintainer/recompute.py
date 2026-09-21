"""The maintainer's independent route to the M8.12 frozen values.

Recomputes, from the worklist's own definitions and nothing else:

  * 924*r-hat-6 at each of the ten filed representatives and at the N1 control
    point, by an exact sympy route through the Condon-Shortley coefficients;
  * the transverse operator M_u = Hess N(u) - 4N(u)*I restricted to N_u, at 50
    digits, its dimension, its signature, and its eigenvalues;
  * the four N1 orbit residuals.

Each is then set against `frozen_claims.json`. This is the route the room results
are also compared against, so it runs before any room opens.

Usage:  python3 recompute.py [--only <orbit>]
"""
import json
import pathlib
import sys
import time

import mpmath as mp
from sympy import I, Poly, Rational, expand, simplify, sqrt, symbols, sympify

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, HERE.as_posix())

from cg import MS, N_of, norm2  # noqa: E402
from hess import analyse, c_to_v14, cplx, hessian, orbit_dirs, v14_to_c, Nv  # noqa: E402

mp.mp.dps = 50
LAM = symbols("l", real=True)
FAILED = []
PASSED = 0
OUT = {}


def check(label, ok, detail=""):
    global PASSED
    if ok:
        PASSED += 1
        print(f"  PASS  {label}{('  ' + detail) if detail else ''}", flush=True)
    else:
        FAILED.append(label)
        print(f"  FAIL  {label}{('  ' + detail) if detail else ''}", flush=True)


def ex(s):
    return sympify(s, locals={"Rational": Rational, "sqrt": sqrt, "I": I, "l": LAM})


def rep_to_dict(rep):
    return {m: ex(rep[i]) for i, m in enumerate(MS)}


def main():
    F = json.loads((HERE / "frozen_claims.json").read_text())
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]

    print("=== exact 924*r-hat-6 at each filed representative ===", flush=True)
    for c in F["census"]:
        if only and c["orbit"] != only:
            continue
        t0 = time.time()
        cd = rep_to_dict(c["rep"])
        val = simplify(924 * N_of(cd) / norm2(cd) ** 2)
        want = ex(c["r6_924"])
        check(f"{c['orbit']}: 924*r6 == {want}", simplify(val - want) == 0,
              f"got {val}  [{time.time() - t0:.1f}s]")
        OUT.setdefault(c["orbit"], {})["r6_924_exact"] = str(val)

    print("\n=== exact 924*r-hat-6 at the N1 control point ===", flush=True)
    cd = rep_to_dict(F["N1"]["point"])
    val = simplify(924 * N_of(cd) / norm2(cd) ** 2)
    check(f"N1: 924*r6 == {F['N1']['r6_924']}", simplify(val - ex(F["N1"]["r6_924"])) == 0, f"got {val}")

    print("\n=== the transverse operator at 50 digits ===", flush=True)
    for c in F["census"]:
        if only and c["orbit"] != only:
            continue
        t0 = time.time()
        a = analyse(rep_to_dict(c["rep"]), c["orbit"])
        dt = time.time() - t0
        # ex(...) may be a rational such as 8800/43, so evaluate it at working
        # precision; a float64 cast here would make an exact value read as off by 1e-17
        want924 = mp.mpf(str(ex(c["r6_924"]).evalf(40)))
        check(f"{c['orbit']}: 924*N(u) numeric == filed", mp.fabs(a["val924"] - want924) < mp.mpf("1e-25"),
              f"{mp.nstr(a['val924'], 20)}  [{dt:.0f}s]")
        check(f"{c['orbit']}: dim N_u == {c['dim_N']}", a["dimN"] == c["dim_N"], f"got {a['dimN']} (dim O_u {a['dimO']})")
        check(f"{c['orbit']}: signature == {tuple(c['sig'])}", list(a["sig"]) == c["sig"], f"got {a['sig']}")
        check(f"{c['orbit']}: M_u annihilates O_u", a["resid"] < mp.mpf("1e-22"),
              f"residual {mp.nstr(a['resid'], 6)}")

        # the eigenvalues must be the roots of the filed characteristic polynomial
        p = Poly(expand(ex(F["spectra"][c["orbit"]])), LAM)
        roots = sorted(mp.mpf(str(r.evalf(40))) for r in p.all_roots())
        got = sorted(a["ev"])
        worst = max(mp.fabs(x - y) for x, y in zip(got, roots)) if len(got) == len(roots) else None
        check(f"{c['orbit']}: eigenvalues are the roots of the filed spectrum",
              worst is not None and worst < mp.mpf("1e-22"),
              f"largest disagreement {mp.nstr(worst, 6) if worst is not None else 'DEGREE MISMATCH'}")
        OUT.setdefault(c["orbit"], {}).update(
            dimN=a["dimN"], sig=list(a["sig"]), ev=[mp.nstr(x, 30) for x in got],
            orbit_residual=mp.nstr(a["resid"], 6))

    if not only:
        print("\n=== N1: the four orbit residuals, un-normalized ===", flush=True)
        cd = rep_to_dict(F["N1"]["point"])
        cnum = {m: cplx(cd[m]) for m in MS}
        nrm = mp.sqrt(mp.fsum([mp.re(cnum[m] * mp.conj(cnum[m])) for m in MS]))
        cnum = {m: cnum[m] / nrm for m in MS}
        u = c_to_v14(cnum)
        Nu = Nv(u)
        M = hessian(u) - 4 * Nu * mp.eye(14)
        names = ["i*u", "-i*Jx*u", "-i*Jy*u", "-i*Jz*u"]
        got = [mp.norm(M * d) for d in orbit_dirs(u)]
        for nm, g in zip(names, got):
            want = mp.mpf(str(ex(F["N1"]["residuals"][nm]).evalf(40)))
            check(f"N1 residual at {nm}", mp.fabs(g - want) < mp.mpf("1e-25"),
                  f"{mp.nstr(g, 25)} vs {mp.nstr(want, 25)}")
        tot = mp.fsum([g ** 2 for g in got])
        want = mp.mpf(str(ex(F["N1"]["sum_of_squares"]).evalf(40)))
        check("N1: the four squares sum to 17368/29403", mp.fabs(tot - want) < mp.mpf("1e-25"),
              f"{mp.nstr(tot, 25)}")
        OUT["N1"] = {"residuals": [mp.nstr(g, 30) for g in got], "sum": mp.nstr(tot, 30)}

    (HERE / "recompute_out.json").write_text(json.dumps(OUT, indent=1))
    print(f"\n{PASSED} passed, {len(FAILED)} failed", flush=True)
    for f in FAILED:
        print(f"  failing: {f}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
