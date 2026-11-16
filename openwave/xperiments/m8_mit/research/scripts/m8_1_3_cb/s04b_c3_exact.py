"""Item C.3: exact route for gamma_u (route A of lib_haar: polynomial Haar moments, no CG).

gamma_u = kappa^-2 (D^2 R~0(u)[e, e] - 6 R~0(u)) for a unit e in K' (gamma is a multiple of the
identity on K', s04).  With u = u'/sqrt(Nu), e = e'/sqrt(Ne), u', e' of integer weighted
coordinates, R~0 of bidegree (3,3):
  R~0(u + t e) = Nu^-3 R~0(u' + tau e'),  tau = t sqrt(Nu/Ne),
so with R~0(u' + tau e') = sum_k c_k tau^k (exact, from 7 exact evaluations, tau = -3..3):
  R~0(u) = c_0/Nu^3,  D^2 R~0(u)[e, e] = 2 c_2 /(Nu^2 Ne),  kappa = h(u) = h(u')/Nu^2.
Directions e (from s04, numerically): R2 line point: e' = 13 mono_{1/2} + mono_{-7/2} (the vector of
K' in the (-1)-eigenspace of the 4-fold element of Gamma_u; checked here to lie in K' at 50 digits);
v1 in R4, R5: e' = mono_{-3} (K' = C v_{-3})."""
import os, sys, json, pickle
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp
from lib_check import check, summary, DEFECT
from lib_haar import Section
from lib_cg import mrange, w as wfac
from lib_engine import Slot, to_real
from lib_point_analysis import analyze, mat_from_cols
from lib_points import point

c3 = json.load(open(os.path.join(HERE, "out", "C3.json")))
etaj = json.load(open(os.path.join(HERE, "out", "eta.json")))
OUT = {}


def exact_Rt0(slot, j, Pm, sN, p):
    S = Section(j, Pm, sN, p)
    low = int(2 * j)
    Js = [Fr(L, 2) for L in range(low + 1, 3 * low + 1)]
    ln = S.level_norms(Js)
    tot = Fr(0)
    for J, (a, b) in ln.items():
        assert b == 0
        L = int(2 * J)
        lam0 = low * (low + 1) if DEFECT == "wrong_lambda0" else low * (low + 2)
        tot -= a / (L * (L + 2) - lam0)
    return tot, S


cases = [
    ("R2", "line {5/2, -3/2}", {Fr(5, 2): 1, Fr(-3, 2): 1}, {Fr(1, 2): 13, Fr(-7, 2): 1}, "R2 line {5/2, -3/2}"),
    ("R4", "v1", {Fr(1): 1}, {Fr(-3): 1}, "v1 R4"),
    ("R5", "v1", {Fr(1): 1}, {Fr(-3): 1}, "v1 R5"),
]
for slot, pname, up, ep, key in cases:
    j = Fr(7, 2) if slot == "R2" else Fr(3)
    ms = mrange(j)
    Pm = pickle.load(open(os.path.join(HERE, "out", f"Pmono_{slot}.pkl"), "rb"))["Pm"]
    sN = Fr(etaj[slot]["s_N"])
    Nu = sum(Fr(c * c) * wfac(j, m) for m, c in up.items())
    Ne = sum(Fr(c * c) * wfac(j, m) for m, c in ep.items())
    # the direction e lies in K' (numerical check, 50 digits)
    mp.mp.dps = 50
    Sl = Slot(slot)
    A = analyze(Sl, point(pname, j))
    KB = mat_from_cols(A["Kp"], 2 * Sl.n)
    ev = [mp.mpf(0)] * (2 * Sl.n)
    for m, c in ep.items():
        ev[ms.index(m)] = c * mp.sqrt(wfac(j, m)) / mp.sqrt(Ne)
    evm = mp.matrix(ev)
    resid = mp.norm(evm - KB * (KB.T * evm))
    check(f"{key}: e lies in K' (|e - P_K' e| < 1e-40)", resid < mp.mpf(10) ** -40)
    # the point u matches inputs.json
    uvec = to_real(point(pname, j))
    uu = [mp.mpf(0)] * (2 * Sl.n)
    for m, c in up.items():
        uu[ms.index(m)] = c * mp.sqrt(wfac(j, m)) / mp.sqrt(Nu)
    check(f"{key}: u'/sqrt(Nu) is the input point", mp.norm(mp.matrix(uu) - uvec) < mp.mpf(10) ** -40)
    vals = {}
    for tau in range(-3, 4):
        p = [(up.get(m, 0) + tau * ep.get(m, 0), 0) for m in ms]
        vals[tau], S = exact_Rt0(slot, j, Pm, sN, p)
        if tau == 0:
            hu = S.h()
            assert hu[1] == hu[2] == hu[3] == 0
            kappa = hu[0] / Nu ** 2
    # exact interpolation: solve Vandermonde over Q
    taus = list(range(-3, 4))
    M = [[Fr(t) ** k for k in range(7)] + [vals[t]] for t in taus]
    for c in range(7):
        piv = next(r for r in range(c, 7) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        M[c] = [x / M[c][c] for x in M[c]]
        for r in range(7):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    coef = [M[k][7] for k in range(7)]
    R0 = coef[0] / Nu ** 3
    D2 = 2 * coef[2] / (Nu ** 2 * Ne)
    gamma = (D2 - 6 * R0) / kappa ** 2
    print(f"{key}: kappa_u = {kappa};  R~0(u) = {R0};  D^2 R~0(u)[e,e] = {D2};  gamma_u = {gamma} R^2  ~ {float(gamma):.15g}")
    ident = c3[key]["gamma_u"]
    check(f"{key}: exact gamma_u equals the value identified from 50/80-digit numerics ({ident})", str(gamma) == ident)
    check(f"{key}: exact kappa_u equals the identified one", str(kappa) == str(c3[key].get("kappa_u", c3[key].get("kappa_u = h(v1)"))))
    rnum = mp.mpf(c3[key]["R~0(u)" if "R~0(u)" in c3[key] else "R~0(v1)"])
    check(f"{key}: exact R~0(u) agrees with route B to 1e-25", abs(rnum - mp.mpf(R0.numerator) / R0.denominator) < mp.mpf(10) ** -25)
    OUT[key] = {"gamma_u": str(gamma), "kappa_u": str(kappa), "R~0(u)": str(R0), "D2R~0(u)[e,e]": str(D2)}
if not DEFECT:
    json.dump(OUT, open(os.path.join(HERE, "out", "C3_exact.json"), "w"), indent=1)
summary()
