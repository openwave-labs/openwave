"""M5.11 P1b' - the quantitative two-soliton running alpha(d) (Faber Fig. 3), via Taichi AD.

Builds a DIFFERENTIABLE Taichi version of the axisymmetric Faber SU(2) energy validated
(in numpy) in v11_p1b_dipole.py, then:
  - mode `single` : GATE - reproduce the single soliton energy on the differentiable kernel
                    (must match 4*pi*I_radial(<R_cut), i.e. the numpy axisym result).
  - mode `pair`   : two solitons (centers pinned at +-d/2), AD-FIRE relax, interaction energy.
  - mode `sweep`  : E_int(d) over a range of d, fit  E_int = +- alpha_sol * hbar c / d,
                    -> alpha_sol(d) (the running). Reproduces the SHAPE of Faber Fig. 2/3.

Field: the SU(2) singlet axisym ansatz Q=(q0,q_rho,q_z) on the (rho,z) half-plane, |Q|=1.
Constraint handled by projected-gradient FIRE (renormalize + tangent-project each step).
Energy uses the hand-derived + validated Gamma_x/Gamma_y(winding)/Gamma_z (see v11_p1b_dipole).

Units: r0 = 1 (lengths in r0). E = (alpha_f hbar c / r0) * I_dimless ; the keV factor below.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=True)

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "_checkpoints")
os.makedirs(CKPT, exist_ok=True)
PI = np.pi

ALPHA_F_HBAR_C = 1.4399645474109544   # MeV.fm
M_E_C2 = 0.51099895                    # MeV
KEV_PER_Idim = ALPHA_F_HBAR_C / 2.21320516 * 1000.0   # E_keV = I_dim * this (I=pi/4 -> 511)

# grid (rho x z), set big enough; pair mode extends z by d
MODE = sys.argv[1] if len(sys.argv) > 1 else "single"
H = 0.18
NR = 60
NZ = 220 if MODE in ("pair", "sweep") else 100

Q = ti.Vector.field(3, ti.f64, shape=(NR, NZ), needs_grad=True)
loss = ti.field(ti.f64, shape=(), needs_grad=True)
par = ti.field(ti.f64, shape=4)        # [h, R_cut, zc(unused here), 0]


@ti.kernel
def compute_energy():
    for i, j in ti.ndrange((1, NR - 1), (1, NZ - 1)):
        h = par[0]
        rcut = par[1]
        rho = (i + 0.5) * h
        z = (j - (NZ - 1) / 2.0) * h
        r = ti.sqrt(rho * rho + z * z)
        inb = 1.0
        if r > rcut:
            inb = 0.0
        inv2h = 1.0 / (2.0 * h)
        dr = (Q[i + 1, j] - Q[i - 1, j]) * inv2h     # d/drho (vector)
        dz = (Q[i, j + 1] - Q[i, j - 1]) * inv2h     # d/dz   (vector)
        q0 = Q[i, j][0]; qr = Q[i, j][1]; qz = Q[i, j][2]
        q0r = dr[0]; qrr = dr[1]; qzr = dr[2]
        q0z = dz[0]; qrz = dz[1]; qzz = dz[2]
        Gx = ti.Vector([q0 * qrr - q0r * qr, qz * qrr - qr * qzr, q0 * qzr - q0r * qz])
        Gy = (qr / rho) * ti.Vector([-qz, q0, qr])
        Gz = ti.Vector([q0 * qrz - q0z * qr, qz * qrz - qr * qzz, q0 * qzz - q0z * qz])
        Gxx = Gx.dot(Gx); Gyy = Gy.dot(Gy); Gzz = Gz.dot(Gz)
        Gxy = Gx.dot(Gy); Gxz = Gx.dot(Gz); Gyz = Gy.dot(Gz)
        trG = Gxx + Gyy + Gzz
        trG2 = Gxx * Gxx + Gyy * Gyy + Gzz * Gzz + 2.0 * (Gxy * Gxy + Gxz * Gxz + Gyz * Gyz)
        u_curv = 0.25 * (trG * trG - trG2)
        Lam = q0 ** 6
        loss[None] += 2.0 * PI * (u_curv + Lam) * rho * h * h * inb


def energy_and_grad(Qnp, h, rcut):
    Q.from_numpy(Qnp)
    par.from_numpy(np.array([h, rcut, 0.0, 0.0]))
    loss[None] = 0.0
    with ti.ad.Tape(loss=loss):
        compute_energy()
    return float(loss[None]), Q.grad.to_numpy()


def energy_only(Qnp, h, rcut):
    Q.from_numpy(Qnp)
    par.from_numpy(np.array([h, rcut, 0.0, 0.0]))
    loss[None] = 0.0
    compute_energy()
    return float(loss[None])


def radial_cumulative_I(U_cut, n=400000):
    u = np.linspace(1e-6, U_cut, n)
    a = np.arctan(u); ap = 1.0 / (1.0 + u * u)
    sa, ca = np.sin(a), np.cos(a)
    dens = ap * ap * sa * sa + sa ** 4 / (2 * u * u) + u * u * ca ** 6
    return float(np.sum(0.5 * (dens[1:] + dens[:-1]) * np.diff(u)))


def grid_coords():
    i = np.arange(NR); j = np.arange(NZ)
    RHO = (i[:, None] + 0.5) * H * np.ones((1, NZ))
    Z = (j[None, :] - (NZ - 1) / 2.0) * H * np.ones((NR, 1))
    return RHO, Z


def hedgehog(RHO, Z, zc, charge):
    """single (anti)soliton centered at (0, zc); charge=+1 soliton, -1 antisoliton."""
    rr = np.sqrt(RHO ** 2 + (Z - zc) ** 2)
    rs = np.where(rr < 1e-12, 1e-12, rr)
    al = np.arctan(rs)
    sa = np.sin(al)
    q0 = np.cos(al)
    qr = charge * RHO / rs * sa
    qz = charge * (Z - zc) / rs * sa
    return np.stack([q0, qr, qz], axis=-1)


def gate_single():
    RHO, Z = grid_coords()
    Qnp = hedgehog(RHO, Z, 0.0, +1)
    R_cut = 5.0
    target = 4 * PI * radial_cumulative_I(R_cut)
    J = energy_only(Qnp, H, R_cut)
    rel = (J - target) / target
    # AD self-consistency: gradient vs central finite-difference at a few points
    E0, g = energy_and_grad(Qnp, H, 1e9)
    eps = 1e-6
    errs = []
    for (i, j) in [(8, 50), (15, 42), (20, 60), (5, 55)]:
        for c in range(3):
            Qp = Qnp.copy(); Qm = Qnp.copy()
            Qp[i, j, c] += eps; Qm[i, j, c] -= eps
            num = (energy_only(Qp, H, 1e9) - energy_only(Qm, H, 1e9)) / (2 * eps)
            errs.append(abs(num - g[i, j, c]) / (abs(num) + abs(g[i, j, c]) + 1e-9))
    gmax = float(np.max(errs))
    ok = abs(rel) < 0.05 and gmax < 1e-5
    print(f"[single] J(<{R_cut})={J:.4f} vs 4pi*I={target:.4f} ({100*rel:+.2f}%)  "
          f"AD gradcheck max_rel={gmax:.1e}  E0(keV~{E0*KEV_PER_Idim:.0f})")
    print(f"single-soliton differentiable kernel: [{'PASS' if ok else 'FAIL'}]")
    out = {"phase": "P1b'-single", "ok": bool(ok), "J": J, "target": target,
           "rel_err": rel, "ad_gradcheck_max_rel": gmax}
    with open(os.path.join(CKPT, "p1b_running_single.json"), "w") as f:
        json.dump(out, f, indent=2)
    return ok


def normalize(Qnp):
    n = np.linalg.norm(Qnp, axis=-1, keepdims=True)
    return Qnp / np.where(n < 1e-12, 1e-12, n)


def seed_pair(RHO, Z, d):
    """soliton (+1) at +d/2, antisoliton (-1) at -d/2 (singlet); SMOOTH tanh blend
    across z=0 (a hard half-and-half seed leaves a huge junction artifact that relaxes
    to scattered metastable states)."""
    sol = hedgehog(RHO, Z, +d / 2.0, +1)
    anti = hedgehog(RHO, Z, -d / 2.0, -1)
    w = 0.5 * (1.0 + np.tanh(Z / 1.5))          # 0 (bottom) -> 1 (top) over ~1.5 r0
    Qs = w[..., None] * sol + (1.0 - w[..., None]) * anti
    return normalize(Qs)


def pin_mask_pair(RHO, Z, d, rcore=1.2):
    """free = movable DOF; pin the outer boundary + a small disk around each core."""
    free = np.ones((NR, NZ), bool)
    free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
    for zc in (d / 2.0, -d / 2.0):
        free[np.sqrt(RHO ** 2 + (Z - zc) ** 2) < rcore] = False
    return free


def relax_pair(Q_seed, free, n_iter=4000, rcut=1e9):
    Qn = normalize(Q_seed.copy())
    free3 = free[..., None]
    v = np.zeros_like(Qn)
    dt, dt_max, alpha, n_pos = 0.008, 0.06, 0.1, 0
    gn = 0.0
    for it in range(n_iter):
        E, g = energy_and_grad(Qn, H, rcut)
        gt = g - np.sum(g * Qn, axis=-1, keepdims=True) * Qn   # tangent-project
        gt = np.where(free3, gt, 0.0)
        gn = float(np.sqrt(np.sum(gt * gt)))
        F = -gt
        v = v + dt * F
        P = float(np.sum(F * v)); fn = np.sqrt(np.sum(F * F)); vn = np.sqrt(np.sum(v * v))
        if fn > 0:
            v = (1 - alpha) * v + alpha * (vn / (fn + 1e-30)) * F
        if P > 0:
            n_pos += 1
            if n_pos > 5:
                dt = min(dt * 1.1, dt_max); alpha *= 0.99
        else:
            n_pos = 0; dt *= 0.5; alpha = 0.1; v[:] = 0.0
        Qn = normalize(Qn + dt * v)
        Qn = np.where(free3, Qn, Q_seed)
        if gn < 1e-6:
            break
    return Qn, energy_only(Qn, H, rcut), gn


def run_sweep():
    RHO, Z = grid_coords()
    ds = np.array([6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.])
    E2 = []
    gns = []
    for d in ds:
        Qs = seed_pair(RHO, Z, d)
        free = pin_mask_pair(RHO, Z, d)
        _, E, gn = relax_pair(Qs, free)
        E2.append(E); gns.append(gn)
        print(f"  d={d:5.1f} r0   J2={E:.5f}   |g|={gn:.2e}")
    E2 = np.array(E2)
    # fit J2(d) = A + B/d ; kappa = -B/(4pi) = charge-product magnitude (e^2);
    # opposite unit charges -> J_int = -4pi/d -> B=-4pi -> kappa=+1 -> 1/alpha_sol=137
    Amat = np.vstack([np.ones_like(ds), 1.0 / ds]).T
    (A, B), *_ = np.linalg.lstsq(Amat, E2, rcond=None)
    kappa = -B / (4 * PI)
    alpha_sol_hbarc = abs(kappa) * ALPHA_F_HBAR_C
    inv_alpha_sol = 137.035999 / abs(kappa) if kappa != 0 else 0.0
    # running: local kappa(d) from a 2-pt slope of J2 vs 1/d on adjacent points
    run = []
    for k in range(len(ds) - 1):
        bloc = (E2[k + 1] - E2[k]) / (1.0 / ds[k + 1] - 1.0 / ds[k])
        kloc = -bloc / (4 * PI)
        dmid = 2.0 / (1.0 / ds[k] + 1.0 / ds[k + 1])
        run.append({"d": float(dmid), "kappa": float(kloc),
                    "inv_alpha_sol": (137.035999 / abs(kloc) if kloc else 0.0)})
    out = {"phase": "P1b'-running", "ds": ds.tolist(), "J2": E2.tolist(),
           "gnorm": gns, "fit_A": A, "fit_B": B, "kappa_global": kappa,
           "alpha_sol_hbarc_MeVfm": alpha_sol_hbarc, "inv_alpha_sol_global": inv_alpha_sol,
           "coulomb_hbarc": ALPHA_F_HBAR_C, "faber_alpha_sol_hbarc": "1.4387(8)",
           "faber_inv_alpha_sol": "137.1(1)", "attractive": bool(B < 0), "running": run,
           "caveats": ("2nd-order axisym, finite box, no H_out exterior, pinned cores; "
                       "shape + magnitude of the running, not Faber's 137.1(1) precision")}
    with open(os.path.join(CKPT, "p1b_running.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nfit J2 = {A:.4f} + ({B:.4f})/d   ->  kappa=|q1q2|={abs(kappa):.4f} e^2 "
          f"({'attractive' if B < 0 else 'repulsive'})")
    print(f"alpha_sol*hbarc = {alpha_sol_hbarc:.4f} MeV.fm (Coulomb {ALPHA_F_HBAR_C:.4f}; "
          f"Faber 1.4387(8))")
    print(f"1/alpha_sol (global) = {inv_alpha_sol:.2f}  (CODATA 137.036; Faber 137.1(1))")
    print("running 1/alpha_sol(d):  " +
          "  ".join(f"d={r['d']:.1f}:{r['inv_alpha_sol']:.1f}" for r in run))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(1.0 / ds, E2, "bo", label="J2(d)")
        ax[0].plot(1.0 / ds, A + B / ds, "r-", label=f"fit A+B/d, B={B:.2f}")
        ax[0].set_xlabel("1/d  (1/r0)"); ax[0].set_ylabel("J2 (dimensionless)")
        ax[0].set_title("two-soliton energy vs 1/d (Coulomb)"); ax[0].legend()
        dd = [r["d"] for r in run]; ia = [r["inv_alpha_sol"] for r in run]
        ax[1].plot(dd, ia, "bo-", label="1/alpha_sol(d)")
        ax[1].axhline(137.036, color="r", ls="--", label="137.036")
        ax[1].set_xlabel("d (r0)"); ax[1].set_ylabel("1/alpha_sol")
        ax[1].set_title("running coupling (cf. Faber Fig. 2/3)"); ax[1].legend()
        fig.suptitle("M5.11 P1b' - two-soliton running alpha(d)", fontweight="bold")
        fig.tight_layout()
        p = os.path.join(HERE, "p1b_running.png"); fig.savefig(p, dpi=110); plt.close(fig)
        print(f"plot -> {p}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    print(f"checkpoint -> {os.path.join(CKPT, 'p1b_running.json')}")


def main():
    t0 = time.time()
    if MODE == "single":
        gate_single()
    elif MODE == "pair":
        RHO, Z = grid_coords()
        d = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
        Qs = seed_pair(RHO, Z, d)
        free = pin_mask_pair(RHO, Z, d)
        E_seed = energy_only(Qs, H, 1e9)
        Qr, E, gn = relax_pair(Qs, free)
        print(f"[pair d={d}] J2 seed={E_seed:.4f} -> relaxed={E:.4f}  |g|={gn:.2e}")
    elif MODE == "sweep":
        run_sweep()
    print(f"({round(time.time()-t0,1)}s)")


if __name__ == "__main__":
    main()
