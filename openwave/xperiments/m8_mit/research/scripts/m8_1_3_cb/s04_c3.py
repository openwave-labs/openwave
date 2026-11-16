"""Item C.3 (with clause 5).
(1) R2, the point of the line {5/2, -3/2}: |Gamma_u|, K' and its dimension, whether Gamma_u acts on
    K' absolutely irreducibly, n_-, and gamma_u.
(2) v1 in R4 and R5: K', its weight under the circle R_z(theta) e^{i theta}, gamma_u, and on the
    components of E_0 off K' the zero multiplicity of S^ and its slow polynomial; also k and r_u.
Everything at 50 and at 80 digits; exact values identified at both (rational, q <= 1e15)."""
import os, sys, json
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp
import numpy as np
import sympy as sp
from lib_check import check, summary, DEFECT
from lib_engine import Slot, theta, Jcx, to_real
from lib_points import point, ident_rational, ident_q5
from lib_point_analysis import analyze, second_derivs_along, mat_from_cols
from lib_rep import D_mp
from lib_cg import mrange
from lib_sym import stabilizer_search, is_group

RES = {}


def gamma_on_Kp(S, A, Kp):
    """gamma_u = kappa^-2 (Hess R~0(u) - 6 R~0(u)) restricted to K' (orthonormal basis Kp)."""
    x = A["x"]
    R0 = S.Rx([x[k] for k in range(len(x))])
    HR = second_derivs_along(S.Rx, x, Kp, 6)
    m = len(Kp)
    Gm = (HR - 6 * R0 * mp.eye(m)) / A["h"] ** 2
    return Gm, R0


def snap(v):
    """snap a float to 0, +-1/2, +-1/sqrt2, +-1 (quaternion components of the octahedral group)."""
    for c in (0.0, 0.5, -0.5, 2 ** -0.5, -(2 ** -0.5), 1.0, -1.0):
        if abs(v - c) < 1e-7:
            return {0.0: mp.mpf(0), 0.5: mp.mpf(1) / 2, -0.5: -mp.mpf(1) / 2, 1.0: mp.mpf(1), -1.0: mp.mpf(-1)}.get(c, mp.sqrt(2) / 2 * (1 if c > 0 else -1))
    return None


def real_map(fun, dim):
    """real matrix of a real-linear map on R^dim given as a function on mp column vectors."""
    M = mp.matrix(dim, dim)
    for c in range(dim):
        e = mp.matrix(dim, 1)
        e[c] = 1
        col = fun(e)
        for r in range(dim):
            M[r, c] = col[r]
    return M


def run_R2(dps, quats=None):
    mp.mp.dps = dps
    S = Slot("R2")
    j, n, dim = S.j, S.n, 2 * S.n
    u = point("line {5/2, -3/2}", j)
    A = analyze(S, u)
    out = dict(h=A["h"], crit=A["crit_residual"], k=A["k"], r_u=A["r_u"], sig=A["signature"])
    Kp = A["Kp"]
    out["dimKp"] = len(Kp)
    # Gamma_u: rotations g with D(g) u in H u; for each, the unique q = alpha + beta Theta with D(g) q u = u
    maps = []
    if quats is not None:
        uc = mp.matrix(u)
        tu = mp.matrix(theta(j, u))
        for qf in quats:
            q = [snap(c) for c in qf]
            assert all(c is not None for c in q), qf
            a = mp.mpc(q[0], -q[3]); b = mp.mpc(q[2], -q[1])
            Dg = D_mp(j, a, b, mp)
            xv = Dg.H * uc  # = D(g)^-1 u = alpha u + beta Theta u
            al = (uc.H * xv)[0]; be = (tu.H * xv)[0]
            resid = mp.norm(xv - al * uc - be * tu)
            def fmap(e, Dg=Dg, al=al, be=be):
                wv = [mp.mpc(e[k], e[n + k]) for k in range(n)]
                tw = theta(j, wv)
                z = Dg * mp.matrix([al * wv[k] + be * tw[k] for k in range(n)])
                return to_real([z[k] for k in range(n)])
            Mq = real_map(fmap, dim)
            maps.append((Mq, resid, abs(al) ** 2 + abs(be) ** 2))
        out["max_stab_residual"] = max(m[1] for m in maps)
        out["max_fix_u"] = max(mp.norm(m[0] * A["x"] - A["x"]) for m in maps)
        out["max_orth"] = max(mp.mnorm(m[0].T * m[0] - mp.eye(dim)) for m in maps)
        out["max_commute_Mh"] = max(mp.mnorm(m[0] * A["Mh"] - A["Mh"] * m[0]) for m in maps)
        # distinct maps
        distinct = []
        for Mq, _, _ in maps:
            if not any(mp.mnorm(Mq - M2) < mp.mpf(10) ** -20 for M2 in distinct):
                distinct.append(Mq)
        out["n_distinct_maps"] = len(distinct)
        if Kp:
            KB = mat_from_cols(Kp, dim)
            reps = [KB.T * M * KB for M in distinct]
            out["Kp_invariant"] = max(mp.mnorm(M * KB - KB * (KB.T * M * KB)) for M in distinct)
            # commutant dimension of the action on K'
            m = len(Kp)
            rows = []
            for Rm in reps:
                for r in range(m):
                    for c in range(m):
                        row = [mp.mpf(0)] * (m * m)
                        for t in range(m):
                            row[r * m + t] += Rm[t, c]   # (X R)_{rc} = sum_t X_{rt} R_{tc}
                            row[t * m + c] -= Rm[r, t]   # (R X)_{rc} = sum_t R_{rt} X_{tc}
                        rows.append(row)
            sv = mp.svd_r(mp.matrix(rows), compute_uv=False)
            out["commutant_dim"] = sum(1 for s in sv if s < mp.mpf(10) ** -(dps // 2))
            out["char_on_Kp"] = sorted(mp.nstr(sum(Rm[t, t] for t in range(m)), 8) for Rm in reps)
    if Kp:
        Gm, R0 = gamma_on_Kp(S, A, Kp)
        out["gamma_matrix"] = Gm
        out["R0"] = R0
    return out, A


def run_v1(slot, dps):
    mp.mp.dps = dps
    S = Slot(slot)
    j, n, dim = S.j, S.n, 2 * S.n
    ms = mrange(j)
    u = point("v1", j)
    A = analyze(S, u)
    out = dict(h=A["h"], crit=A["crit_residual"], k=A["k"], r_u=A["r_u"], sig=A["signature"], dimKp=len(A["Kp"]))
    # circle generator: w_m -> i (1 - m) w_m
    Xc = mp.matrix(dim, dim)
    for r, m in enumerate(ms):
        Xc[r, n + r] = -(1 - m)
        Xc[n + r, r] = (1 - m)
    out["circle_fixes_u"] = mp.norm(Xc * A["x"])
    out["circle_commutes_Mh"] = mp.mnorm(Xc * A["Mh"] - A["Mh"] * Xc)
    Kp = A["Kp"]
    if Kp:
        KB = mat_from_cols(Kp, dim)
        # support of K' on components m, and the circle weights there
        supp = sorted({str(ms[r]) for r in range(n) for c in range(len(Kp)) if abs(KB[r, c]) > 1e-20 or abs(KB[n + r, c]) > 1e-20})
        out["Kp_support_m"] = supp
        out["Kp_weights"] = sorted({int(1 - Fr(m)) for m in supp})
        out["Kp_complex"] = mp.mnorm(A["J"] * KB - KB * (KB.T * A["J"] * KB))  # K' invariant under i?
        Gm, R0 = gamma_on_Kp(S, A, Kp)
        out["gamma_matrix"] = Gm
        out["R0"] = R0
        # off K': the real circle-blocks |1-m| not meeting K'
        kw = {abs(int(1 - Fr(m))) for m in supp}
    else:
        kw = set()
    off = [r for r, m in enumerate(ms) if abs(int(1 - m)) not in kw]
    idx = off + [n + r for r in off]
    out["off_m"] = [str(ms[r]) for r in off]
    # block structure check: M^ does not couple off-K' blocks to the rest
    rest = [r for r in range(dim) if r not in idx]
    out["block_coupling"] = max([abs(A["Mh"][a, b]) for a in idx for b in rest] + [mp.mpf(0)])
    Ssub = mp.matrix(len(idx), len(idx))
    Sfull = A["J"] * A["Mh"]
    for a, ia in enumerate(idx):
        for b, ib in enumerate(idx):
            Ssub[a, b] = Sfull[ia, ib]
    ev = mp.eig(Ssub, left=False, right=False)
    # zero eigenvalues of the non-normal S^ sit in Jordan blocks and are perturbed to ~eps^(1/b):
    # threshold 10^(-dps/5); the gap to the nonzero ones is recorded
    tol = mp.mpf(10) ** (-(mp.mp.dps // 5))
    zeros = sum(1 for e in ev if abs(e) < tol)
    nz = [e for e in ev if abs(e) >= tol]
    out["zero_gap"] = (max([abs(e) for e in ev if abs(e) < tol] + [mp.mpf(0)]), min([abs(e) for e in nz] + [mp.inf]))
    # characteristic polynomial of the nonzero part in xi, then in nu = xi^2
    poly = [mp.mpc(1)]
    for e in nz:
        poly = [a - e * b for a, b in zip(poly + [0], [0] + poly)]
    out["zero_mult_off"] = zeros
    out["slow_coeffs_xi"] = poly  # highest degree first
    out["S_eigs_off"] = nz
    out["full_S_zero_mult"] = sum(1 for e in mp.eig(Sfull, left=False, right=False) if abs(e) < tol)
    return out, A


def rat_or_q5(x50, x80, d1=50, d2=80):
    f1, f2 = ident_rational(x50, d1), ident_rational(x80, d2)
    if f1 is not None and f1 == f2:
        return str(f1), "rational, q <= 1e15, same at 50 and 80 digits"
    a1, a2 = ident_q5(x50, 50), ident_q5(x80, 80)
    if a1 is not None and a1 == a2:
        return f"{a1[0]} + ({a1[1]})*sqrt(5)", "Q(sqrt5) by PSLQ, coefficients <= 1e12, same at 50 and 80 digits"
    return None, "not identified"


# ======================= (1) R2 point of the line {5/2, -3/2} =======================
print("===== C.3 (1): R2, point of the line {5/2, -3/2} =====")
mp.mp.dps = 30
u0 = point("line {5/2, -3/2}", Fr(7, 2))
found = stabilizer_search(Fr(7, 2), np.array([complex(c) for c in u0]), True, 0, seed=5)
print(f"stabilizer search: {len(found)} rotations; closed under products: {is_group(found)}")
check("R2 line point: rotation set found is a group", is_group(found))
quats = [q for R, q in found]
r50, A50 = run_R2(50, quats)
r80, A80 = run_R2(80, quats)
for tag, r in (("50", r50), ("80", r80)):
    print(f"[dps {tag}] h(u) = {mp.nstr(r['h'], 30)}; |grad h - 4 h u| = {mp.nstr(r['crit'], 5)}; k = {r['k']}; r_u = {r['r_u']}; "
          f"signature on N_u = {r['sig']}; dim K' = {r['dimKp']}")
check("R2 line point is critical (|grad h - 4h u| < 1e-40 at 50 digits)", r50["crit"] < mp.mpf(10) ** -40)
check("R2 line point: k = 6 (finite stabilizer)", r50["k"] == 6)
check("R2 line point: signature and dim K' agree at 50 and 80 digits", r50["sig"] == r80["sig"] and r50["dimKp"] == r80["dimKp"])
print(f"Gamma_u: {r50['n_distinct_maps']} distinct maps of E_0; max residual D(g)u in Hu: {mp.nstr(r50['max_stab_residual'], 3)}; "
      f"max |A u - u| = {mp.nstr(r50['max_fix_u'], 3)}; max |A^T A - 1| = {mp.nstr(r50['max_orth'], 3)}; max |[A, M^]| = {mp.nstr(r50['max_commute_Mh'], 3)}")
check("Gamma_u elements verified at 50 digits (fix u, orthogonal, commute with M^)",
      r50["max_fix_u"] < mp.mpf(10) ** -40 and r50["max_orth"] < mp.mpf(10) ** -40 and r50["max_commute_Mh"] < mp.mpf(10) ** -35)
check("|Gamma_u| = number of rotations found (2-to-1 count of (g, q) pairs)", r50["n_distinct_maps"] == len(found))
res1 = {"|Gamma_u|": r50["n_distinct_maps"], "dim K'": r50["dimKp"], "n_-": r50["sig"][0], "k": r50["k"], "signature": list(r50["sig"])}
if r50["dimKp"]:
    print(f"K' invariant under Gamma_u: max residual {mp.nstr(r50['Kp_invariant'], 3)}; commutant dimension of Gamma_u on K' = {r50['commutant_dim']}")
    print(f"traces of Gamma_u on K': {r50['char_on_Kp']}")
    check("K' invariant under Gamma_u", r50["Kp_invariant"] < mp.mpf(10) ** -35)
    check("commutant dimension the same at 50 and 80 digits", r50["commutant_dim"] == r80["commutant_dim"])
    absirr = r50["commutant_dim"] == 1
    G50, G80 = r50["gamma_matrix"], r80["gamma_matrix"]
    m = G50.rows
    lam50 = G50[0, 0]
    off50 = mp.mnorm(G50 - lam50 * mp.eye(m))
    print(f"gamma_u matrix on K' (50 digits): diag {[mp.nstr(G50[i, i], 25) for i in range(m)]}; |gamma - gamma_11 I| = {mp.nstr(off50, 3)}")
    print(f"R~0(u) = {mp.nstr(r50['R0'], 30)}")
    val, how = rat_or_q5(lam50, r80["gamma_matrix"][0, 0])
    print(f"gamma_u = {val} R^2   [{how}]   ~ {mp.nstr(lam50, 30)}")
    check("gamma_u is a multiple of the identity on K' (1e-35)", off50 < mp.mpf(10) ** -35)
    check("gamma_u identified exactly at two precisions", val is not None)
    res1.update({"K'": f"dimension {m}", "absolutely irreducible": absirr, "commutant dimension": r50["commutant_dim"],
                 "gamma_u": val, "gamma_u (decimal, 30 digits)": mp.nstr(lam50, 30), "R~0(u)": mp.nstr(r50["R0"], 30), "kappa_u": mp.nstr(r50["h"], 30)})
    # describe K' in components
    KB = mat_from_cols(A50["Kp"], 16)
    ms = mrange(Fr(7, 2))
    supp = sorted({str(ms[r]) for r in range(8) for c in range(m) if abs(KB[r, c]) > 1e-20 or abs(KB[8 + r, c]) > 1e-20})
    print(f"K' is supported on the components m in {supp}")
    res1["K' support (m)"] = supp
    # exact form of K' (identified from the numerical kernel, then checked): weighted coordinates p_m
    #   e1: p_{1/2} = 13, p_{-7/2} = 1;   e2(z): p_{7/2} = 3z, p_{-1/2} = -z, p_{3/2} = 15 conj z, p_{-5/2} = -5 conj z
    from lib_cg import w as wfac
    def wvec(d):
        y = [mp.mpc(0)] * 8
        for mm, c in d.items():
            y[ms.index(Fr(mm))] = c * mp.sqrt(wfac(Fr(7, 2), Fr(mm)))
        xx = to_real(y)
        return xx / mp.norm(xx)
    I_ = mp.mpc(0, 1)
    cand = [wvec({"1/2": 13, "-7/2": 1}), wvec({"7/2": 3, "-1/2": -1, "3/2": 15, "-5/2": -5}),
            wvec({"7/2": 3 * I_, "-1/2": -I_, "3/2": -15 * I_, "-5/2": 5 * I_})]
    res_in = max(mp.norm(v - KB * (KB.T * v)) for v in cand)
    gram = mat_from_cols(cand, 16)
    indep = min(mp.svd_r(gram, compute_uv=False))
    print(f"exact form of K' checked: max |v - P_K' v| = {mp.nstr(res_in, 3)}, smallest singular value of the 3 vectors {mp.nstr(indep, 6)}")
    check("R2 line point: the exact 3 vectors lie in K' and span it", res_in < mp.mpf(10) ** -40 and indep > mp.mpf(10) ** -3 and m == 3)
    res1["K' (exact)"] = ("real span of: e1 with weighted coordinates p_{1/2} = 13, p_{-7/2} = 1; and e2(z), z in C, with "
                          "p_{7/2} = 3z, p_{-1/2} = -z, p_{3/2} = 15 conj(z), p_{-5/2} = -5 conj(z)  (p_m = w_m/sqrt((j+m)!(j-m)!)); "
                          "not a complex subspace")
    # kappa
    kap, how = rat_or_q5(r50["h"], r80["h"])
    print(f"kappa_u = h(u) = {kap}  [{how}]")
    res1["kappa_u"] = kap
RES["R2 line {5/2, -3/2}"] = res1

# ======================= (2) v1 in R4 and R5 =======================
for slot in ("R4", "R5"):
    print(f"\n===== C.3 (2): v1 in {slot} =====")
    # v1 runs at 80 and 130 digits (the slow-polynomial coefficients have denominators ~1e25)
    o50, A50 = run_v1(slot, 80)
    o80, A80 = run_v1(slot, 130)
    print(f"h(v1) = {mp.nstr(o50['h'], 30)}; |grad h - 4 h u| = {mp.nstr(o50['crit'], 5)}")
    check(f"v1 critical in {slot}", o50["crit"] < mp.mpf(10) ** -40)
    print(f"clause 5: k = {o50['k']}, r_u = {o50['r_u']}  (130 digits: k = {o80['k']}, r_u = {o80['r_u']}); singular values of the 4 tangent vectors: {[mp.nstr(s, 6) for s in A50['tang_svals']]}")
    check(f"{slot} v1: k and r_u agree at 80 and 130 digits", (o50["k"], o50["r_u"]) == (o80["k"], o80["r_u"]))
    print(f"signature on N_u = {o50['sig']}; dim K' = {o50['dimKp']}; circle fixes u: {mp.nstr(o50['circle_fixes_u'], 3)}; [circle, M^] = {mp.nstr(o50['circle_commutes_Mh'], 3)}")
    check(f"{slot} v1: the circle fixes u and commutes with M^", o50["circle_fixes_u"] < mp.mpf(10) ** -40 and o50["circle_commutes_Mh"] < mp.mpf(10) ** -35)
    r = {"k": o50["k"], "r_u": o50["r_u"], "signature": list(o50["sig"]), "dim K'": o50["dimKp"]}
    if o50["dimKp"]:
        G50 = o50["gamma_matrix"]
        m = G50.rows
        lam50 = G50[0, 0]
        off50 = mp.mnorm(G50 - lam50 * mp.eye(m))
        val, how = rat_or_q5(lam50, o80["gamma_matrix"][0, 0], 80, 130)
        how = how.replace("50 and 80", "80 and 130")
        print(f"K' supported on m in {o50['Kp_support_m']}; circle weights there (w_m -> e^(i(1-m)theta) w_m): {o50['Kp_weights']}; "
              f"K' complex (i-invariant) residual {mp.nstr(o50['Kp_complex'], 3)}")
        print(f"gamma_u on K': diag {[mp.nstr(G50[i, i], 25) for i in range(m)]}; |gamma - gamma_11 I| = {mp.nstr(off50, 3)}")
        print(f"gamma_u = {val} R^2  [{how}]  ~ {mp.nstr(lam50, 30)};  R~0(v1) = {mp.nstr(o50['R0'], 30)}")
        check(f"{slot} v1: gamma_u a multiple of the identity on K'", off50 < mp.mpf(10) ** -35)
        check(f"{slot} v1: gamma_u identified exactly at two precisions", val is not None)
        r.update({"K'": f"real dimension {m}, components m in {o50['Kp_support_m']}", "K' weight under the circle": o50["Kp_weights"],
                  "K' is a complex subspace": bool(o50["Kp_complex"] < mp.mpf(10) ** -30),
                  "gamma_u": val, "gamma_u (decimal, 30 digits)": mp.nstr(lam50, 30), "R~0(v1)": mp.nstr(o50["R0"], 30)})
        kap, _ = rat_or_q5(o50["h"], o80["h"], 80, 130)
        r["kappa_u"] = kap
    else:
        r["K'"] = "0"
    print(f"off K': components m in {o50['off_m']}; coupling of M^ to the rest: {mp.nstr(o50['block_coupling'], 3)}")
    check(f"{slot} v1: off-K' components form an M^-invariant block", o50["block_coupling"] < mp.mpf(10) ** -35)
    print(f"   zero multiplicity of S^ off K' = {o50['zero_mult_off']} (130 digits: {o80['zero_mult_off']}); whole E_0: {o50['full_S_zero_mult']}; "
          f"largest |zero eigenvalue| {mp.nstr(o50['zero_gap'][0], 3)} vs smallest |nonzero| {mp.nstr(o50['zero_gap'][1], 6)} (80 digits); "
          f"at 130 digits {mp.nstr(o80['zero_gap'][0], 3)} vs {mp.nstr(o80['zero_gap'][1], 6)}")
    check(f"{slot} v1: zero multiplicity off K' agrees at 80 and 130 digits", o50["zero_mult_off"] == o80["zero_mult_off"])
    # slow polynomial in nu: coefficients of xi^(2k) (odd ones must vanish), identify
    c50, c80 = o50["slow_coeffs_xi"], o80["slow_coeffs_xi"]
    deg = len(c50) - 1
    odd = max([abs(c50[i]) for i in range(1, deg + 1, 2)] + [mp.mpf(0)])
    imag = max(abs(mp.im(c)) for c in c50)
    check(f"{slot} v1: characteristic polynomial off K' is even in xi with real coefficients", odd < mp.mpf(10) ** -35 and imag < mp.mpf(10) ** -35)
    coeffs = []
    ok = True
    for i in range(0, deg + 1, 2):
        f1 = ident_rational(mp.re(c50[i]), 80, qmax=10 ** 30)
        f2 = ident_rational(mp.re(c80[i]), 130, qmax=10 ** 30)
        ok &= f1 is not None and f1 == f2
        coeffs.append(f1)
    check(f"{slot} v1: slow polynomial coefficients identified as rationals at two precisions", ok)
    nu = sp.Symbol("nu")
    if ok:
        poly = sum(sp.Rational(c.numerator, c.denominator) * nu ** (deg // 2 - k) for k, c in enumerate(coeffs))
        prim = sp.Poly(poly, nu).primitive()[1]
        lc = prim.LC()
        prim = prim * (1 if lc > 0 else -1)
        fac = sp.factor(prim.as_expr())
        roots = sorted(sp.Poly(prim, nu).nroots(n=30), key=lambda z: (sp.re(z), sp.im(z)))
        print(f"   slow polynomial (off K', nu = xi^2): {fac}")
        print(f"   roots in nu: {[sp.N(z, 20) for z in roots]}")
        r["off K': zero multiplicity of S^"] = o50["zero_mult_off"]
        r["off K': slow polynomial"] = str(fac)
        r["off K': components m"] = o50["off_m"]
        r["off K': slow roots (nu)"] = [str(sp.N(z, 20)) for z in roots]
    r["whole E_0: zero multiplicity of S^ (for reference)"] = o50["full_S_zero_mult"]
    r["kappa_u = h(v1)"] = rat_or_q5(o50["h"], o80["h"], 80, 130)[0]
    RES[f"v1 {slot}"] = r

if not DEFECT:
    json.dump(RES, open(os.path.join(HERE, "out", "C3.json"), "w"), indent=1, default=str)
summary()
