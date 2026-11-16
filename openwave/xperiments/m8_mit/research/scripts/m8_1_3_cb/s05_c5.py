"""Item C.5 (with clauses 1 and 4).  At the five cases: the zero multiplicity of S^ = i M^, and for each
slow group (nonzero eigenvalue s_g of S^, multiplicity m) the value c_g = l^T T_1 r (m = 1) or the
eigenvalues of C_ab = l_a^T T_1 r_b (semisimple m > 1), with
  T_1 = c^2/(4 w0) i D_u - c^2/(2 w0^2) mu0 + mu0^2/(2 w0) i,   mu0 = chi s_g,  chi = c^2/(8 w0 Q(u)),
  D_u = kappa^-2 (Hess R~0(u) - D^3 h(u)[z]),  M^ z = grad R~0(u) on R u + N_u,
clause 1: each i in T_1 is the complex structure of E_0 (the real matrix J), extended complex-linearly;
mu0, s_g scalars.  c = R = V = 1, g4 = 1.  Computed at 50 and at 70 digits; reported digits are those
on which the two agree (at least 15 required)."""
import os, sys, json
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mpmath as mp
from lib_check import check, summary, DEFECT
from lib_engine import Slot, grad, hess, dir_deriv, to_real
from lib_points import point, INP
from lib_point_analysis import analyze, mat_from_cols

CASES = [("pentagonal pyramid", "R4"), ("coherent v3", "R4"), ("coherent v3", "R5"), ("weight 7/2", "R2"), ("octahedron", "R4")]
check("the five cases are those of inputs.json", [f"{p} in {s}" for p, s in CASES] == INP["five cases"])


def nullspace_c(A, m, tol):
    U, S, V = mp.svd_c(A)
    n = A.cols
    small = [k for k in range(n) if S[k] < tol]
    return [V.H[:, k] for k in small], [S[k] for k in range(n)]


def run(pname, slot, dps):
    mp.mp.dps = dps
    S = Slot(slot)
    n, dim = S.n, 2 * S.n
    u = point(pname, S.j)
    A = analyze(S, u)
    x = A["x"]
    xl = [x[k] for k in range(dim)]
    out = dict(h=A["h"], crit=A["crit_residual"], k=A["k"], r_u=A["r_u"], sig=A["signature"], dimKp=len(A["Kp"]))
    kappa = A["h"]
    Q = A["h"]  # unit u, V = 1
    Jm = A["J"]
    Mh = A["Mh"]
    Sh = Jm * Mh
    # zero multiplicity (Jordan-tolerant threshold) and nonzero eigenvalues
    ev = mp.eig(Sh, left=False, right=False)
    ztol = mp.mpf(10) ** (-(dps // 5))
    zeros = [e for e in ev if abs(e) < ztol]
    nz = [e for e in ev if abs(e) >= ztol]
    out["zero_mult"] = len(zeros)
    out["zero_gap"] = (max([abs(e) for e in zeros] + [mp.mpf(0)]), min([abs(e) for e in nz] + [mp.inf]))
    if out["dimKp"]:
        out["degenerate"] = True
        return out
    # R~0 data
    gR = grad(S.Rx, xl, 6)
    HR = hess(S.Rx, xl, 6)
    out["R0"] = S.Rx(xl)
    # grad R~0 orthogonal to K_u (invariance)
    out["gradR_on_K"] = max([abs((v.T * gR)[0]) for v in A["Kb"]] + [mp.mpf(0)])
    B = mat_from_cols([x] + A["Nb"], dim)
    MB = B.T * Mh * B
    cz = mp.lu_solve(MB, B.T * gR)
    z = B * cz
    out["z_residual"] = mp.norm(B.T * (Mh * z - gR))
    # D^3 h(u)[z] = Hess of x -> Dh(x)[z]
    zl = [z[k] for k in range(dim)]
    gfun = dir_deriv(S.hx, zl, 4)
    D3 = hess(gfun, xl, 3)
    Du = (HR - D3) / kappa ** 2
    Du = (Du + Du.T) / 2
    w0 = mp.sqrt(S.lam0)
    chi = 1 / (8 * w0 * Q)
    # cluster nonzero eigenvalues
    groups = []
    for e in nz:
        for g in groups:
            if abs(g[0] - e) < mp.mpf(10) ** (-(dps // 4)):
                g[1].append(e)
                break
        else:
            groups.append([e, [e]])
    res = []
    I = mp.eye(dim)
    for g0, members in groups:
        mult = len(members)
        s = mp.fsum(members) / mult
        stol = mp.mpf(10) ** (-(dps // 3))
        Rv, sv = nullspace_c(Sh - s * I, mult, stol)
        Lv, _ = nullspace_c(Sh.T - s * I, mult, stol)
        semisimple = len(Rv) == mult and len(Lv) == mult
        if not semisimple:
            res.append(dict(s=s, mult=mult, semisimple=False, geo=len(Rv)))
            continue
        Rm = mat_from_cols(Rv, dim)
        Lm = mat_from_cols(Lv, dim)
        X = Lm.T * Rm
        Lm = Lm * mp.inverse(X).T
        mu0 = chi * s
        T1 = (1 / (4 * w0)) * (Jm * Du) - (mu0 / (2 * w0 ** 2)) * I + (mu0 ** 2 / (2 * w0)) * Jm
        if DEFECT == "T1_mu0sq_sign" and dps == 70:  # planted: sign of the mu0^2 term flipped in one run
            T1 = T1 - (mu0 ** 2 / w0) * Jm
        C = Lm.T * T1 * Rm
        vals = [C[0, 0]] if mult == 1 else list(mp.eig(C, left=False, right=False))
        bi = mp.mnorm(Lm.T * Rm - mp.eye(mult))
        res.append(dict(s=s, mult=mult, semisimple=True, vals=vals, biorth=bi,
                        eig_resid=mp.mnorm(Sh * Rm - s * Rm), mu0=mu0))
    out["groups"] = res
    out["chi"] = chi
    out["w0"] = w0
    out["Du_sym"] = mp.mnorm(Du - Du.T)
    return out


def fmtc(z, d):
    """sympy-parsable string 'a + b*I'; a part below 1e-40 in size is numerical noise (both runs agree it
    vanishes to that level) and is written 0."""
    re_, im_ = mp.re(z), mp.im(z)
    a = "0" if abs(re_) < mp.mpf(10) ** -40 else mp.nstr(re_, d)
    b = "0" if abs(im_) < mp.mpf(10) ** -40 else mp.nstr(im_, d)
    return f"{a} + ({b})*I"


def sortkey(g):
    """order slow groups by (Im s_g, Re s_g), parts below 1e-30 in size counted as 0 (numerical noise)."""
    chop = lambda v: 0.0 if abs(v) < mp.mpf(10) ** -30 else float(v)
    return (chop(mp.im(g["s"])), chop(mp.re(g["s"])))


RES = {}
for pname, slot in (CASES[1:2] if DEFECT else CASES):
    key = f"{pname} in {slot}"
    print(f"\n===== C.5: {key} =====")
    a = run(pname, slot, 50)
    b = run(pname, slot, 70)
    print(f"h(u) = Q(u) = kappa_u = {mp.nstr(a['h'], 25)};  |grad h - 4hu| = {mp.nstr(a['crit'], 3)};  k = {a['k']}, r_u = {a['r_u']}, "
          f"signature {a['sig']}, dim K' = {a['dimKp']}")
    check(f"{key}: critical", a["crit"] < mp.mpf(10) ** -40)
    check(f"{key}: zero multiplicity the same at 50 and 70 digits", a["zero_mult"] == b["zero_mult"])
    print(f"zero multiplicity of S^ = {a['zero_mult']}  (largest |zero eig| {mp.nstr(a['zero_gap'][0], 3)}, smallest |nonzero| {mp.nstr(a['zero_gap'][1], 6)})")
    r = {"zero multiplicity": a["zero_mult"], "kappa_u = Q(u)": mp.nstr(a["h"], 30), "k": a["k"], "r_u": a["r_u"], "signature": list(a["sig"])}
    if a.get("degenerate"):
        print("DEGENERATE point (K' != 0): the formula of Section 3 for nondegenerate points does not apply; not computed")
        r["slow groups"] = "not computed: point is degenerate (K' != 0)"
        RES[key] = r
        continue
    check(f"{key}: nondegenerate (K' = 0)", a["dimKp"] == 0)
    print(f"R~0(u) = {mp.nstr(a['R0'], 25)};  |<grad R~0, K_u>| = {mp.nstr(a['gradR_on_K'], 3)};  |M^ z - grad R~0| on Ru+N_u = {mp.nstr(a['z_residual'], 3)}")
    check(f"{key}: grad R~0(u) orthogonal to K_u", a["gradR_on_K"] < mp.mpf(10) ** -35)
    ga = sorted(a["groups"], key=sortkey)
    gb = sorted(b["groups"], key=sortkey)
    check(f"{key}: same slow-group structure at 50 and 70 digits", [(g["mult"], g["semisimple"]) for g in ga] == [(g["mult"], g["semisimple"]) for g in gb])
    rows = []
    for g, h_ in zip(ga, gb):
        if not g["semisimple"]:
            print(f"  s_g = {mp.nstr(g['s'], 20)}  multiplicity {g['mult']}, NOT semisimple (geometric {g['geo']}): formula not applicable")
            rows.append({"s_g": mp.nstr(g["s"], 20), "multiplicity": g["mult"], "c_g": "not semisimple; not computed"})
            continue
        va = sorted(g["vals"], key=lambda z: (mp.re(z), mp.im(z)))
        vb = sorted(h_["vals"], key=lambda z: (mp.re(z), mp.im(z)))
        diff = max(abs(p - q) / max(abs(q), mp.mpf(10) ** -30) for p, q in zip(va, vb))
        sdiff = abs(g["s"] - h_["s"])
        digits = int(-mp.log10(max(diff, mp.mpf(10) ** -45)))
        check(f"{key}: s_g = {mp.nstr(g['s'], 8)}: c_g agrees at 50/70 digits to >= 15 significant digits (rel. diff {mp.nstr(diff, 3)})", diff < mp.mpf(10) ** -15)
        print(f"  s_g = {mp.nstr(h_['s'], 25)}   multiplicity {g['mult']}   mu0 = {mp.nstr(h_['mu0'], 20)}")
        for v in vb:
            print(f"      value: {mp.nstr(v, 25)}")
        print(f"      (biorthogonality residual {mp.nstr(g['biorth'], 3)}, eigen residual {mp.nstr(g['eig_resid'], 3)}, 50-vs-70 rel. diff {mp.nstr(diff, 3)}, s_g diff {mp.nstr(sdiff, 3)})")
        rows.append({"s_g": fmtc(h_["s"], 25), "multiplicity": g["mult"],
                     "c_g" if g["mult"] == 1 else "eigenvalues of C_ab": [fmtc(v, 20) for v in vb],
                     "agreement 50 vs 70 digits (relative)": mp.nstr(diff, 3)})
    r["slow groups"] = rows
    r["R~0(u)"] = mp.nstr(b["R0"], 25)
    RES[key] = r
if not DEFECT:
    json.dump(RES, open(os.path.join(HERE, "out", "C5.json"), "w"), indent=1, default=str)
summary()
