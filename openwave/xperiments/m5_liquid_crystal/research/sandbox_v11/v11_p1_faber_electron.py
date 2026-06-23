"""M5.11 P1 - reproduce FABER's electron (the credibility gate).

Faber & Golubich 2026 (arXiv:2604.12021), the SU(2) topological-fermion soliton:

    Q(x) = cos a(x) - i sigma.n(x) sin a(x),     |n| = 1            (Eq. 1)
    L    = -(alpha_f hbar c /4pi) ( 1/4 R.R + Lambda ),            (Eq. 2)
    Lambda = q0^6 / r0^4,   q0 = cos a,                            (Eq. 4)
    R_munu = Gamma_mu x Gamma_nu,                                  (Eq. 5)
    Gamma_mu = (d_mu a) n + sin a cos a d_mu n + sin^2 a (n x d_mu n)   (Eq. 6)

Single electron = hedgehog n = r_hat, a(r) = arctan(r/r0)          (Eq. 7),
rest energy E0 = (alpha_f hbar c / r0)(pi/4)                       (Eq. 8),
calibrated by r0 = 2.21320516 fm -> E0 = m_e c^2 = 0.510999 MeV    (Eq. 10).

RADIAL REDUCTION (derived here, verified analytically -> pi/4):
  with G_ij = Gamma_i . Gamma_j = a'^2 r_i r_j + (sin^2 a / r^2)(d_ij - r_i r_j),
  curvature density u_curv = 1/4[(Tr G)^2 - Tr(G^2)] = a'^2 sin^2 a / r^2 + sin^4 a /(2 r^4),
  so (with u = r/r0):
    E = (alpha_f hbar c / r0) * I[a],
    I[a] = integral_0^inf [ (da/du)^2 sin^2 a + sin^4 a /(2 u^2) + u^2 cos^6 a ] du,
  and I[arctan u] = 2*(pi/16) + (1/2)*(pi/4) = pi/4   (exact).

This is NON-circular: we MINIMIZE I[a] from a GENERIC seed (not arctan) with the
P0-style relaxer and show it (a) descends monotonically to a stationary profile,
(b) that profile = arctan(u), (c) I_min = pi/4, (d) the energy scale alpha_f hbar c
= 1.43996 MeV.fm gives E0 = 511 keV at r0 = 2.2132 fm. 511 keV at that r0 is Faber's
CALIBRATION; the physics content is the pi/4 integral + the relaxed profile.

Modes:  analytic | gradcheck | minimize | all   (all -> plot + checkpoint JSON)
The non-trivial Coulomb/fine-structure reproduction (alpha_sol ~ 1.4387 MeV.fm,
alpha^-1 ~ 137) is the TWO-soliton dipole -> P1b (v11_p1b_dipole.py).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "_checkpoints")
os.makedirs(CKPT, exist_ok=True)

# ---- physical constants (CODATA 2018, as Faber cites) ----------------------
HBAR_C = 197.3269804          # MeV.fm
ALPHA_F = 1.0 / 137.035999084
ALPHA_F_HBAR_C = ALPHA_F * HBAR_C     # = 1.43996 MeV.fm  (= e0^2/4pi eps0)
M_E_C2 = 0.51099895000        # MeV
R0_FABER = 2.21320516         # fm  (Faber Eq. 10)
PI_4 = np.pi / 4.0


# ---- radial grid: u = sinh(s), clusters near 0, reaches u ~ 200 ------------
def make_grid(s_max=6.0, n=1600):
    s = np.linspace(0.0, s_max, n)
    ds = s[1] - s[0]
    u = np.sinh(s)
    J = np.cosh(s)            # du/ds
    return s, ds, u, J


def du_deriv(alpha, ds, J):
    """da/du via central difference in s divided by J = du/ds. Endpoints one-sided."""
    p = np.zeros_like(alpha)
    p[1:-1] = (alpha[2:] - alpha[:-2]) / (2 * ds) / J[1:-1]
    p[0] = (alpha[1] - alpha[0]) / ds / J[0]
    p[-1] = (alpha[-1] - alpha[-2]) / ds / J[-1]
    return p


def I_functional(alpha, ds, u, J):
    """dimensionless I[a] = integral [ (da/du)^2 sin^2 a + sin^4 a/(2u^2) + u^2 cos^6 a ] du,
    trapezoid in s with weight J ds (du = J ds). u^-2 term is regular (a~u near 0)."""
    p = du_deriv(alpha, ds, J)
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    u2 = u * u
    u2_safe = np.where(u2 < 1e-12, 1e-12, u2)
    dens = p * p * sa * sa + (sa ** 4) / (2.0 * u2_safe) + u2 * ca ** 6
    dens[0] = 0.0  # r=0: all terms -> 0 (a ~ u)
    w = J * ds
    bulk = float(np.sum(dens * w))
    # analytic exterior (Coulomb) tail beyond u_max, like Faber's H_out (Eq. 24):
    # the relaxed/arctan profile asymptotes integrand ~ 1/(2u^2) + 2/u^4, so
    # int_{u_max}^inf = 1/(2 u_max) + 2/(3 u_max^3)  (exact to O(u^-5)).
    u_max = u[-1]
    tail = 1.0 / (2.0 * u_max) + 2.0 / (3.0 * u_max ** 3)
    return bulk + tail


def I_grad(alpha, ds, u, J):
    """analytic dI/d alpha_j (interior). Derivative-coupling + local terms (see header)."""
    p = du_deriv(alpha, ds, J)
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    u2 = u * u
    u2_safe = np.where(u2 < 1e-12, 1e-12, u2)
    w = J * ds
    g = np.zeros_like(alpha)
    # local terms at j
    local = (p * p * np.sin(2 * alpha)
             + 2.0 * sa ** 3 * ca / u2_safe
             - 6.0 * u2 * ca ** 5 * sa)
    g += w * local
    # derivative coupling: from p_{j+1} and p_{j-1} (central stencil)
    coup = np.zeros_like(alpha)
    coup[1:-1] = p[:-2] * sa[:-2] ** 2 - p[2:] * sa[2:] ** 2
    g += coup
    g[0] = 0.0
    g[-1] = 0.0
    return g


def analytic_profile(u):
    return np.arctan(u)


# ---- 1D FIRE relaxer (boundary pinned) -------------------------------------
def fire_1d(alpha0, energy_fn, grad_fn, max_iter=60000, tol=1e-9,
            dt0=0.01, dt_max=0.05, log_every=2000):
    a = alpha0.copy()
    v = np.zeros_like(a)
    dt = dt0
    alpha_mix = 0.1
    n_pos = 0
    hist = {"I": [], "gnorm": [], "iter": []}
    for it in range(max_iter):
        g = grad_fn(a)
        F = -g
        v = v + dt * F
        P = float(np.sum(F * v))
        fn = np.sqrt(np.sum(F * F))
        vn = np.sqrt(np.sum(v * v))
        if fn > 0:
            v = (1 - alpha_mix) * v + alpha_mix * (vn / (fn + 1e-30)) * F
        if P > 0:
            n_pos += 1
            if n_pos > 5:
                dt = min(dt * 1.1, dt_max)
                alpha_mix *= 0.99
        else:
            n_pos = 0
            dt *= 0.5
            alpha_mix = 0.1
            v[:] = 0.0
        a = a + dt * v
        a[0] = alpha0[0]
        a[-1] = alpha0[-1]
        gn = float(np.sqrt(np.sum(g[1:-1] ** 2)))
        if it % log_every == 0:
            hist["I"].append(energy_fn(a)); hist["gnorm"].append(gn); hist["iter"].append(it)
        if gn < tol:
            hist["I"].append(energy_fn(a)); hist["gnorm"].append(gn); hist["iter"].append(it)
            break
    return a, hist


def gate_analytic():
    s, ds, u, J = make_grid()
    a = analytic_profile(u)
    I = I_functional(a, ds, u, J)
    return {"gate": "analytic", "I_analytic_profile": I, "pi_4": PI_4,
            "rel_err": abs(I - PI_4) / PI_4, "ok": bool(abs(I - PI_4) / PI_4 < 5e-4)}


def gate_gradcheck(seed=0):
    s, ds, u, J = make_grid(s_max=5.0, n=300)
    rng = np.random.default_rng(seed)
    a = analytic_profile(u) + 0.05 * rng.standard_normal(u.size)
    a[0] = 0.0; a[-1] = analytic_profile(u)[-1]
    g_an = I_grad(a, ds, u, J)
    eps = 1e-6
    errs = []
    for j in [20, 60, 100, 150, 200, 250]:
        ap = a.copy(); am = a.copy()
        ap[j] += eps; am[j] -= eps
        num = (I_functional(ap, ds, u, J) - I_functional(am, ds, u, J)) / (2 * eps)
        errs.append(abs(num - g_an[j]) / (abs(num) + abs(g_an[j]) + 1e-12))
    me = float(np.max(errs))
    return {"gate": "gradcheck", "max_rel_err": me, "ok": bool(me < 1e-5)}


def gate_minimize():
    s, ds, u, J = make_grid()
    a_an = analytic_profile(u)
    # GENERIC seed: monotone 0 -> pi/2 but NOT arctan (a stretched tanh)
    a0 = (np.pi / 2.0) * np.tanh(0.45 * u)
    a0[0] = 0.0
    a0[-1] = a_an[-1]   # pin the far boundary to the known asymptote
    I0 = I_functional(a0, ds, u, J)

    def E(a):
        return I_functional(a, ds, u, J)

    def G(a):
        return I_grad(a, ds, u, J)

    a_f, hist = fire_1d(a0, E, G)
    I_fire = E(a_f)
    # L-BFGS cross-check
    from scipy.optimize import minimize
    free = np.arange(1, u.size - 1)
    base = a0.copy()

    def fun(x):
        aa = base.copy(); aa[free] = x; return E(aa)

    def jac(x):
        aa = base.copy(); aa[free] = x; return G(aa)[free]

    res = minimize(fun, a0[free].copy(), jac=jac, method="L-BFGS-B",
                   options={"maxiter": 5000, "gtol": 1e-10, "ftol": 1e-16})
    a_lb = base.copy(); a_lb[free] = res.x
    I_lbfgs = E(a_lb)

    prof_err = float(np.max(np.abs(a_f - a_an)))
    monotone = all(hist["I"][i + 1] <= hist["I"][i] + 1e-9 for i in range(len(hist["I"]) - 1))
    # physical units
    I_min = min(I_fire, I_lbfgs)
    E0_at_faber = ALPHA_F_HBAR_C / R0_FABER * I_min      # MeV, should be ~0.511
    r0_for_511 = ALPHA_F_HBAR_C * I_min / M_E_C2          # fm, should be ~2.2132
    ok = (abs(I_fire - PI_4) / PI_4 < 5e-4
          and abs(I_lbfgs - PI_4) / PI_4 < 5e-4
          and prof_err < 2e-2 and monotone)
    return {"gate": "minimize", "ok": bool(ok),
            "I_seed": I0, "I_fire": I_fire, "I_lbfgs": I_lbfgs, "pi_4": PI_4,
            "rel_err_fire": abs(I_fire - PI_4) / PI_4,
            "rel_err_lbfgs": abs(I_lbfgs - PI_4) / PI_4,
            "profile_max_err_vs_arctan": prof_err, "monotone": bool(monotone),
            "E0_at_r0_faber_MeV": E0_at_faber, "m_e_c2_MeV": M_E_C2,
            "E0_rel_err": abs(E0_at_faber - M_E_C2) / M_E_C2,
            "r0_for_511keV_fm": r0_for_511, "r0_faber_fm": R0_FABER,
            "alpha_f_hbar_c_MeVfm": ALPHA_F_HBAR_C,
            "_profiles": {"u": u.tolist(), "a_min": a_f.tolist(),
                          "a_analytic": a_an.tolist(), "a_seed": a0.tolist()}}


def make_plot(minres):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        return f"(plot skipped: {e})"
    pr = minres["_profiles"]
    u = np.array(pr["u"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(u, pr["a_seed"], "k:", lw=1, label="generic seed")
    ax[0].plot(u, pr["a_min"], "b-", lw=2, label="relaxed (FIRE)")
    ax[0].plot(u, pr["a_analytic"], "r--", lw=1.2, label="arctan(u) [Faber Eq.7]")
    ax[0].set_xlim(0, 12); ax[0].set_xlabel("u = r / r0"); ax[0].set_ylabel("alpha(r)")
    ax[0].set_title("Faber electron profile: seed -> relaxed = arctan"); ax[0].legend(loc="lower right")
    # energy density of the relaxed profile
    s, ds, ug, J = make_grid()
    a = np.array(pr["a_min"])
    p = du_deriv(a, ds, J); sa = np.sin(a); ca = np.cos(a)
    u2 = np.where(ug ** 2 < 1e-12, 1e-12, ug ** 2)
    dens = (p * p * sa * sa + sa ** 4 / (2 * u2) + ug ** 2 * ca ** 6) * J
    ax[1].plot(ug, dens, "b-", lw=2)
    ax[1].set_xlim(0, 12); ax[1].set_xlabel("u = r / r0")
    ax[1].set_ylabel("dI/ds  (energy per shell)")
    ax[1].set_title(f"I_min = {minres['I_fire']:.5f}  vs  pi/4 = {PI_4:.5f}")
    fig.suptitle(f"M5.11 P1 - Faber electron reproduced: E0 = {minres['E0_at_r0_faber_MeV']*1000:.1f} keV "
                 f"at r0 = {R0_FABER:.4f} fm", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(HERE, "p1_faber_electron.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if mode == "analytic":
        res = [gate_analytic()]
    elif mode == "gradcheck":
        res = [gate_gradcheck()]
    elif mode == "minimize":
        res = [gate_minimize()]
    elif mode == "all":
        res = [gate_analytic(), gate_gradcheck(), gate_minimize()]
    else:
        print("modes: analytic | gradcheck | minimize | all"); return
    plot_path = None
    for r in res:
        if r["gate"] == "minimize":
            plot_path = make_plot(r)
            r.pop("_profiles", None)
    all_ok = all(r["ok"] for r in res)
    for r in res:
        flag = "PASS" if r["ok"] else "FAIL"
        print(f"[{flag}] {r['gate']:10s} " +
              " ".join(f"{k}={v}" for k, v in r.items()
                       if k not in ("gate", "ok", "_profiles")))
    print(f"\nP1 {'ALL PASS' if all_ok else 'FAILURES PRESENT'}  ({round(time.time()-t0,1)}s)")
    if plot_path:
        print(f"plot -> {plot_path}")
    if mode == "all":
        out = {"phase": "P1a", "all_ok": bool(all_ok),
               "wall_s": round(time.time() - t0, 1), "gates": res}
        path = os.path.join(CKPT, "p1_faber_electron.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"checkpoint -> {path}")


if __name__ == "__main__":
    main()
