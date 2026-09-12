"""M5.32 R15-H adversarial audit: the tilt channel on a rotating split background.

Independent sympy implementation built from the audit brief's DEFINITIONS only
(no producer script was read). Every density is expanded to second order in
the tilt by differentiation in a scaling parameter e (theta -> e*th,
theta_t -> e*tht, theta_z -> e*thz), which avoids sympy `series` on the
projector trig products.

Run:
    /opt/anaconda3/envs/master312/bin/python3 m5_32_r15_h_audit.py
Writes ../data/m5_32_r15_h_audit.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import sympy as sp

t0 = time.time()

# ---------------------------------------------------------------- symbols
g, delta, s, omega, w, cP = sp.symbols("g delta s omega w c_P", real=True)
tau = sp.symbols("tau", real=True)  # tau = omega t (rotation phase)
th, tht, thz = sp.symbols("theta theta_t theta_z", real=True)
e = sp.symbols("e", real=True)  # tilt scaling parameter

ETA = sp.diag(-1, 1, 1, 1)
ID4 = sp.eye(4)


def rot(p: int, q: int, x):
    """Rotation by x in the (p, q) plane: cos on (p,p),(q,q); -sin on (p,q); +sin on (q,p)."""
    R = sp.eye(4)
    R[p, p] = sp.cos(x)
    R[q, q] = sp.cos(x)
    R[p, q] = -sp.sin(x)
    R[q, p] = sp.sin(x)
    return R


def D_split():
    return sp.diag(g, 1, delta + s, delta - s)


def h_metric(M):
    """h = eta + 2 (eta u)(eta u)^T with u the timelike unit eigenvector of N = M eta."""
    N = M * ETA
    # For spatial-rotation backgrounds u = e0 exactly; verify rather than assume.
    u = sp.Matrix([1, 0, 0, 0])
    assert sp.simplify(N * u - (N * u)[0] * u) == sp.zeros(4, 1), "u = e0 is not an eigenvector"
    assert (u.T * ETA * u)[0] == -1
    eu = ETA * u
    return ETA + 2 * eu * eu.T


def bracket_eta(X, Y):
    return X * ETA * Y - Y * ETA * X


def inner_G(F, G):
    return (G * F * G * F.T).trace()


def densities(M, A0, Az, Q, sel=None):
    """All densities as functions of M, A_0, A_z and the eigenframe Q.

    Returns dict name -> expression (not yet simplified).
    """
    out = {}
    F = bracket_eta(A0, Az)
    h = h_metric(M)
    assert sp.simplify(h - ID4) == sp.zeros(4, 4), "h != identity on this background"
    if sel is None or "I1" in sel:
        out["-4I1"] = 4 * inner_G(F, ETA)
    if sel is None or "I1h" in sel:
        out["-4I1_h"] = 4 * inner_G(F, h)
    if sel is None or "KP" in sel:
        P23 = Q * sp.diag(0, 0, 1, 1) * Q.T
        Om0 = P23 * A0 * ETA * P23
        Omz = P23 * Az * ETA * P23
        out["K_P^23"] = (cP / 2) * ((Om0.T * ETA * Om0 * ETA).trace() - (Omz.T * ETA * Omz * ETA).trace())
    if sel is None or "reg" in sel:
        for name, G in (("reg_eta", ETA), ("reg_h", h)):
            out[name] = w * ((A0 * G * A0 * G).trace() - (Az * G * Az * G).trace())
    return out


def taylor_in_e(expr, order=2):
    """Coefficients c0, c1, c2 of expr = c0 + c1 e + c2 e^2 + O(e^3)."""
    coeffs = []
    cur = expr
    for k in range(order + 1):
        val = cur.subs(e, 0)
        val = sp.simplify(sp.trigsimp(sp.expand(val)))
        val = sp.simplify(sp.fu(sp.expand(val)))
        coeffs.append(val / sp.factorial(k))
        cur = sp.diff(cur, e)
    return coeffs


def poly_coeffs(L2):
    """Split a quadratic form in (th, tht, thz) into named coefficients."""
    P = sp.Poly(sp.expand(L2), th, tht, thz)
    d = {}
    for mon, c in P.terms():
        d[mon] = sp.factor(sp.simplify(c))
    names = {
        (0, 2, 0): "alpha (theta_t^2)",
        (0, 0, 2): "gamma (theta_z^2)",
        (2, 0, 0): "eps (theta^2)",
        (0, 1, 1): "theta_t theta_z",
        (1, 1, 0): "theta theta_t",
        (1, 0, 1): "theta theta_z",
    }
    return {names.get(k, str(k)): v for k, v in d.items()}


def build_background(tilt_plane=(1, 2), tilt_inside=True, rot_plane=(2, 3)):
    """M(tau, theta), A_0 = omega d_tau M + theta_t d_theta M, A_z = theta_z d_theta M.

    All tilt variables are scaled by e for the Taylor expansion.
    """
    thf = sp.symbols("vartheta", real=True)  # the tilt angle before scaling
    Rrot = rot(*rot_plane, tau)
    Rtilt = rot(*tilt_plane, thf)
    if tilt_inside:
        Q = Rrot * Rtilt
    else:
        Q = Rtilt * Rrot
    D = D_split()
    M = Q * D * Q.T
    M_tau = sp.diff(M, tau).subs(thf, e * th)
    M_th = sp.diff(M, thf).subs(thf, e * th)  # d/d(theta), then theta = e*th
    M = M.subs(thf, e * th)
    Q = Q.subs(thf, e * th)
    A0 = omega * M_tau + e * tht * M_th
    Az = e * thz * M_th
    return M, A0, Az, Q, M_tau, M_th


results: dict = {"claims": {}, "mutations": {}, "not_claimed_but_found": []}
SYM: dict = {}


def S(x):
    return str(sp.factor(sp.simplify(x)))


def expand_all(M, A0, Az, Q, label):
    dens = densities(M, A0, Az, Q)
    rec = {}
    for name, expr in dens.items():
        c0, c1, c2 = taylor_in_e(expr)
        L2sym = poly_coeffs(c2) if c2 != 0 else {}
        rec[name] = {
            "L0": S(c0),
            "L1": S(c1),
            "L2": {k: str(v) for k, v in L2sym.items()},
            "tau_dependent": bool(any(x.has(tau) for x in (c0, c1, c2))),
        }
        SYM[(label, name)] = {"L0": c0, "L1": c1, "L2": L2sym}
    print(f"\n=== {label} ===")
    for name, r in rec.items():
        print(f"  {name}: L0={r['L0']}  L1={r['L1']}  tau_dep={r['tau_dependent']}")
        for k, v in r["L2"].items():
            print(f"      {k}: {v}")
    return rec, dens


# ---------------------------------------------------------------- MAIN: (1,2) tilt inside rotating (2,3) frame
M, A0, Az, Q, M_tau, M_th = build_background()
main, dens_main = expand_all(M, A0, Az, Q, "MAIN: tilt (1,2) inside rotating (2,3) frame")
results["main"] = main

dsm1 = delta + s - 1

# H1: alpha
MAIN = "MAIN: tilt (1,2) inside rotating (2,3) frame"
def L2of(name, key, label=MAIN):
    return SYM[(label, name)]["L2"].get(key, sp.Integer(0))
def L0of(name, label=MAIN):
    return SYM[(label, name)]["L0"]
h1 = {name: str(L2of(name, "alpha (theta_t^2)")) for name in main}
h1_ok = all(sp.simplify(L2of(n, "alpha (theta_t^2)")) == 0 for n in ("-4I1", "-4I1_h", "K_P^23")) and all(
    sp.simplify(L2of(n, "alpha (theta_t^2)") - 2 * w * dsm1**2) == 0 for n in ("reg_eta", "reg_h")
)
results["claims"]["H1"] = {"alpha_by_term": h1, "verdict": "CONFIRMED" if h1_ok else "REFUTED"}

# H2: -4I1 and -4I1_h second-order content, and |C|^2
Ct = bracket_eta(M_tau, M_th).subs(e, 0)
C2 = sp.simplify(sp.trigsimp(inner_G(Ct, ETA)))
C2_h = sp.simplify(sp.trigsimp(inner_G(Ct, ID4)))
h2_target = {"gamma (theta_z^2)": 32 * omega**2 * s**2 * dsm1**2}
h2_ok = True
for name in ("-4I1", "-4I1_h"):
    L2 = SYM[(MAIN, name)]["L2"]
    for k, v in L2.items():
        if sp.simplify(v - h2_target.get(k, 0)) != 0:
            h2_ok = False
    if "gamma (theta_z^2)" not in L2:
        h2_ok = False
C2_ok = sp.simplify(C2 - 8 * s**2 * dsm1**2) == 0
# the author's <F,F> = 4 k^2 omega^2 s^2 (...)^2: plane-wave theta_z^2 -> k^2 |theta|^2, so
# <F,F> = omega^2 theta_z^2 |C|^2 = 8 omega^2 s^2 (...)^2 k^2 |theta|^2, a factor 2 above the author's 4.
results["claims"]["H2"] = {
    "C2_eta": S(C2),
    "C2_h": S(C2_h),
    "C2_matches_8s2": bool(C2_ok),
    "L2_matches": bool(h2_ok),
    "FF_per_k2_theta2": S(omega**2 * C2),
    "verdict": "CONFIRMED" if (h2_ok and C2_ok) else "REFUTED",
}

# H3: kappa_2, regulator L2, gamma_total, hyperbolicity
Y = M_th.subs(e, 0)
k2_eta = sp.simplify(sp.trigsimp((Y * ETA * Y * ETA).trace()))
k2_h = sp.simplify(sp.trigsimp((Y * Y).trace()))
reg_target = {
    "alpha (theta_t^2)": 2 * w * dsm1**2,
    "gamma (theta_z^2)": -2 * w * dsm1**2,
    "eps (theta^2)": -2 * omega**2 * w * (3 * s + 1 - delta) * dsm1,
}
h3_ok = sp.simplify(k2_eta - 2 * dsm1**2) == 0 and sp.simplify(k2_h - 2 * dsm1**2) == 0
for name in ("reg_eta", "reg_h"):
    L2 = SYM[(MAIN, name)]["L2"]
    for k in set(L2) | set(reg_target):
        if sp.simplify(L2.get(k, 0) - reg_target.get(k, 0)) != 0:
            h3_ok = False
    if sp.simplify(L0of(name) - 8 * omega**2 * s**2 * w) != 0:
        h3_ok = False
# total gamma with all three physical terms + regulator (eta)
gamma_tot = sum(L2of(n, "gamma (theta_z^2)") for n in ("-4I1", "K_P^23", "reg_eta"))
alpha_tot = sum(L2of(n, "alpha (theta_t^2)") for n in ("-4I1", "K_P^23", "reg_eta"))
eps_tot = sum(L2of(n, "eps (theta^2)") for n in ("-4I1", "K_P^23", "reg_eta"))
gamma_ok = sp.simplify(gamma_tot - 2 * (16 * omega**2 * s**2 - w) * dsm1**2) == 0
Omega2 = sp.simplify(-(gamma_tot * sp.Symbol("k") ** 2 + eps_tot) / alpha_tot)
results["claims"]["H3"] = {
    "kappa2_eta": S(k2_eta),
    "kappa2_h": S(k2_h),
    "regulator_L2_matches": bool(h3_ok),
    "alpha_total(I1+KP+reg_eta)": S(alpha_tot),
    "gamma_total(I1+KP+reg_eta)": S(gamma_tot),
    "eps_total(I1+KP+reg_eta)": S(eps_tot),
    "gamma_total_matches": bool(gamma_ok),
    "Omega2_dispersion": str(Omega2),
    "hyperbolic_iff": "alpha>0 and gamma<0: w>0 (with delta+s!=1) and w > 16 omega^2 s^2",
    "verdict": "CONFIRMED" if (h3_ok and gamma_ok) else "REFUTED",
}

# H4: K_P^23 background + L2; static twist sheet
kp_L0_ok = sp.simplify(L0of("K_P^23") - 4 * cP * omega**2 * s**2) == 0
kp_L2 = SYM[(MAIN, "K_P^23")]["L2"]
kp_L2_ok = set(kp_L2) == {"eps (theta^2)"} and sp.simplify(kp_L2["eps (theta^2)"] + 4 * cP * omega**2 * s**2) == 0
# static twist sheet: omega = 0, psi(z) finite amplitude, no time dependence
psi = sp.Function("psi")
zz = sp.symbols("z", real=True)
Qs = rot(1, 2, psi(zz))
Ms = Qs * D_split() * Qs.T
A0s = sp.zeros(4, 4)
Azs = sp.diff(Ms, zz)
ds = densities(Ms, A0s, Azs, Qs)
static = {k: S(v) for k, v in ds.items()}
# also the static (2,3) twist sheet and the static (1,3) sheet (not claimed)
Qs13 = rot(1, 3, psi(zz))
Ms13 = Qs13 * D_split() * Qs13.T
ds13 = densities(Ms13, A0s, sp.diff(Ms13, zz), Qs13)
static13 = {k: S(v) for k, v in ds13.items()}
Qs23 = rot(2, 3, psi(zz))
Ms23 = Qs23 * D_split() * Qs23.T
ds23 = densities(Ms23, A0s, sp.diff(Ms23, zz), Qs23)
static23 = {k: S(v) for k, v in ds23.items()}
h4_ok = kp_L0_ok and kp_L2_ok and sp.simplify(ds["K_P^23"]) == 0 and sp.simplify(ds["-4I1"]) == 0
results["claims"]["H4"] = {
    "KP_L0": main["K_P^23"]["L0"],
    "KP_L2": main["K_P^23"]["L2"],
    "static_12_sheet": static,
    "static_13_sheet": static13,
    "static_23_sheet": static23,
    "verdict": "CONFIRMED" if h4_ok else "REFUTED",
}

# H5: tau independence + L1 = 0
h5_ok = all((not main[n]["tau_dependent"]) and sp.simplify(SYM[(MAIN, n)]["L1"]) == 0 for n in main)
results["claims"]["H5"] = {
    "tau_dependent": {n: main[n]["tau_dependent"] for n in main},
    "L1": {n: main[n]["L1"] for n in main},
    "verdict": "CONFIRMED" if h5_ok else "REFUTED",
}

# ---------------------------------------------------------------- MUTATIONS
# (a) tilt in the (1,3) plane inside the rotating frame
M13, A013, Az13, Q13, _, _ = build_background(tilt_plane=(1, 3))
mut13, _ = expand_all(M13, A013, Az13, Q13, "MUT a: tilt (1,3) inside rotating (2,3) frame")
results["mutations"]["a_tilt_13_inside"] = mut13

# (b) tilt (1,2) OUTSIDE the rotating frame
Mo, A0o, Azo, Qo, _, _ = build_background(tilt_inside=False)
muto, _ = expand_all(Mo, A0o, Azo, Qo, "MUT b: tilt (1,2) OUTSIDE rotating (2,3) frame")
results["mutations"]["b_tilt_12_outside"] = muto

# (c) cross-term check: sign-flip theta_z -> -theta_z on the MAIN L2 must be invariant
cross = {n: {k: v for k, v in main[n]["L2"].items() if "theta_t theta_z" in k or "theta theta_" in k} for n in main}
results["mutations"]["c_cross_terms_main"] = cross

# (d) s = 0 degenerate background
main_s0 = {
    n: {
        "L0": S(L0of(n).subs(s, 0)),
        "L2": {k: S(v.subs(s, 0)) for k, v in SYM[(MAIN, n)]["L2"].items()},
    }
    for n in main
}
results["mutations"]["d_s_equals_0"] = main_s0

# (e) not claimed: the (2,3) tilt inside the frame, i.e. a tilt in the ROTATION plane (should decouple)
M23, A023, Az23, Q23, _, _ = build_background(tilt_plane=(2, 3))
mut23, _ = expand_all(M23, A023, Az23, Q23, "MUT e: tilt (2,3) inside rotating (2,3) frame")
results["mutations"]["e_tilt_23_inside"] = mut23

# (f) not claimed: delta + s = 1 (tilt between two EQUAL eigenvalues 1 and delta+s): tilt is pure gauge
main_deg = {
    n: {k: S(v.subs(delta, 1 - s)) for k, v in SYM[(MAIN, n)]["L2"].items()} for n in main
}
# (g) mutation b, evaluated at two phases to show the tau dependence explicitly
MUTB = "MUT b: tilt (1,2) OUTSIDE rotating (2,3) frame"
results["mutations"]["b_tilt_12_outside_at_phases"] = {
    n: {
        k: {"tau=0": S(v.subs(tau, 0)), "tau=pi/2": S(v.subs(tau, sp.pi / 2)), "tau=pi/4": S(v.subs(tau, sp.pi / 4))}
        for k, v in SYM[(MUTB, n)]["L2"].items()
    }
    for n in main
}
# (h) I1 total energy sign check: -4 I1 gives gamma > 0 (anti-hyperbolic on its own); the regulator alone gamma<0
results["not_claimed_but_found"] = [
    "the (1,3) tilt is the mirror channel: every coefficient maps under s -> -s, i.e. (delta+s-1) -> (delta-s-1); "
    "the K_P^23 and I1 background values are even in s so they are unchanged",
    "a tilt IN the rotation plane (2,3) inside the frame has a nonzero L1 = 8 c_P omega s^2 theta_t (+16 omega s^2 w theta_t "
    "for the regulator): it is the rotation phase itself; it is a total time derivative and drops from the action, "
    "and its L2 is a pure regulator/K_P wave term with alpha = -gamma = 4 c_P s^2 (K_P) and 8 s^2 w (regulator); I1 gives nothing",
    "at delta + s = 1 the (1,2) tilt rotates two EQUAL eigenvalues (1 and delta+s): every quadratic coefficient vanishes "
    "except the K_P^23 mass term -4 c_P omega^2 s^2 theta^2, so the K_P^23 eps is NOT proportional to kappa_2 and does not "
    "vanish on the gauge direction; it comes from the projector P23 moving with theta while the spectrum is degenerate",
    "the tilt OUTSIDE the rotating frame is tau-dependent for I1 and the regulator (kappa_2 oscillates between "
    "2(delta+s-1)^2 at tau=0 and 2(delta-s-1)^2 at tau=pi/2), while K_P^23 gives NO quadratic term at all "
    "(the projector then rotates rigidly with the tilt and Om_z vanishes); the producer's tau-independence rests on the "
    "tilt being inside the frame, as the brief states",
    "the static (2,3) twist sheet (twist in the ROTATION plane, omega = 0) gives -4 I1 = 0 as well, but the static K_P^23 "
    "density there is NOT zero: -4 c_P s^2 psi'(z)^2 for any amplitude (the twist stays inside the P23 eigenplane, so "
    "Om_z survives); the static K_P^23 vanishes only on the (1,2) and (1,3) sheets, where A_z is off-diagonal between the "
    "eigenplanes and P23 A_z eta P23 = 0; the static regulator is -2 w (delta+s-1)^2 psi'^2 on the (1,2) sheet",
    "at s = 0 the I1, I1_h and K_P^23 quadratic forms vanish as expected, but the regulator's theta^2 coefficient does NOT: "
    "eps(reg) = +2 w omega^2 (delta-1)^2 with alpha = 2 w (delta-1)^2, so the single-tilt dispersion at s = 0 is "
    "Omega^2 = k^2 - omega^2 in the rotating frame; a static D = diag(g,1,delta,delta) has lab-frame massless tilts "
    "Omega_lab^2 = k^2 and the rotating frame should give (k -/+ omega)^2, so the single-theta ansatz drops the Coriolis "
    "coupling to the (1,3) partner tilt (the cross term theta_12_t theta_13 of the two-tilt problem); this does not touch "
    "the principal symbol (hyperbolicity) but the k = 0 gap Omega^2(0) = -eps_tot/alpha_tot should not be read as physical",
]
results["Omega2_at_k0_total"] = str(sp.factor(sp.simplify(-eps_tot / alpha_tot)))
results["Omega2_at_k0_total_s0"] = str(sp.factor(sp.simplify((-eps_tot / alpha_tot).subs(s, 0))))
assert "nan" not in json.dumps(results, default=str), "nan leaked into results"
results["mutations"]["f_delta_plus_s_equals_1"] = main_deg

results["runtime_s"] = round(time.time() - t0, 1)
out = Path(__file__).resolve().parent.parent / "data" / "m5_32_r15_h_audit.json"
out.write_text(json.dumps(results, indent=2, default=str))
print("\nverdicts:", {k: v["verdict"] for k, v in results["claims"].items()})
print("wrote", out, "runtime", results["runtime_s"], "s")
