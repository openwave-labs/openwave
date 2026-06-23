"""M5.11 P2 - the regularized vortex LOOP: does it stabilize or collapse?

The neutrino object Duda said was missing. Build a closed +1/2 disclination RING in
the M5 tensor field (the engine has only point/line defects), regularize the core with
the LdG potential V_M, and relax under the FULL functional with the validated P0
minimizer. THE question (Derrick): does the loop radius stabilize at finite R (the
4th-order curvature term providing outward pressure) or collapse to R->0 (line tension
wins)?

Seeder: loop core = circle radius R0 in the z=0 plane. Meridional angle around the core
chi = atan2(z, s-R0), s = sqrt(x^2+y^2). +1/2 disclination director (apolar, n ~ -n):
    n_hat = cos(chi/2) s_hat + sin(chi/2) z_hat
melted at the core (rho_m -> 0) toward isotropic. M_spatial = uniaxial(n_hat) melted;
M[0,0] = g (time axis, frozen). Energy/gradient/minimizer reused from v11_p0_minimizer.

Diagnostic: track the loop radius R(iter) = argmin_s Tr(M^2) in the z=0 mid-plane, the
energy split (curvature vs potential), and |grad|. Report HONESTLY:
  - R stabilizes at finite value, |grad|->0  -> Derrick evaded, stable regularized loop
  - R -> 0 (collapse)                         -> plain loop unstable; FORK: needs the
                                                 chiral/knot structure (flag, do not fake)

Run:  python v11_p2_vortex_loop.py [N] [R0] [iters]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v11_p0_minimizer as p0

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "_checkpoints")
os.makedirs(CKPT, exist_ok=True)

G_TIME = 8.0
DELTA = 0.3
# LdG amplitude well V = a*Tr2 + c*Tr2^2 (b=0): min at Tr2 = -a/(2c) = uniaxial 1+2*delta^2
VAC_TR2 = 1.0 + 2.0 * DELTA ** 2          # ~1.18
LDG_C = 0.5
LDG_A = -2.0 * LDG_C * VAC_TR2            # so dV/dTr2 = 0 at Tr2 = VAC_TR2
LDG_B = 0.0
C2 = 1.0


def seed_vortex_loop_M(n, dx, R0, r_core):
    """+1/2 disclination ring in the tensor field, index-0, core melted to isotropic."""
    xs = (np.arange(n) - (n - 1) / 2.0) * dx
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    s = np.sqrt(X ** 2 + Y ** 2)
    s_safe = np.where(s < 1e-9, 1e-9, s)
    shat = np.stack([X / s_safe, Y / s_safe, np.zeros_like(X)], axis=-1)
    zhat = np.zeros(X.shape + (3,)); zhat[..., 2] = 1.0
    u = s - R0                       # meridional radial offset from the core circle
    w = Z
    chi = np.arctan2(w, u)
    rho_m = np.sqrt(u ** 2 + w ** 2)            # distance from the loop core
    nhat = np.cos(chi / 2.0)[..., None] * shat + np.sin(chi / 2.0)[..., None] * zhat
    nn = np.linalg.norm(nhat, axis=-1, keepdims=True)
    nhat = nhat / np.where(nn < 1e-9, 1e-9, nn)
    # uniaxial M_sp = delta*I + (1-delta)*nn ; melt the anisotropy at the core
    melt = rho_m / np.sqrt(rho_m ** 2 + r_core ** 2)           # ->0 at core, ->1 far
    nn_out = nhat[..., :, None] * nhat[..., None, :]
    iso = np.eye(3) / 3.0 * (1.0 + 2.0 * DELTA)
    Msp = iso + melt[..., None, None] * ((DELTA * np.eye(3) + (1 - DELTA) * nn_out) - iso)
    M = np.zeros((n, n, n, 4, 4))
    M[..., 1:4, 1:4] = Msp
    M[..., 0, 0] = G_TIME
    return M, (X, Y, Z, s)


def loop_radius(M, n, dx):
    """R = argmin_s Tr(M_sp^2) in the z=0 mid-plane (the melted core ring)."""
    mid = n // 2
    Msp = M[:, :, mid, 1:4, 1:4]
    tr2 = np.trace(Msp @ Msp, axis1=-2, axis2=-1)
    xs = (np.arange(n) - (n - 1) / 2.0) * dx
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    s = np.sqrt(X ** 2 + Y ** 2)
    # azimuthally average tr2 into s-bins, find the minimum-amplitude radius
    sbin = np.round(s / dx).astype(int)
    rmax = int(0.7 * (n / 2.0))          # exclude the boundary region (Dirichlet pin)
    prof = np.array([tr2[sbin == k].mean() if np.any(sbin == k) else np.nan
                     for k in range(rmax + 1)])
    valid = ~np.isnan(prof)
    ks = np.arange(rmax + 1)[valid]
    pv = prof[valid]
    # the core is the interior minimum of Tr2 (most melted) at s>0
    inner = ks > 1
    if not np.any(inner):
        return 0.0, prof
    kmin = ks[inner][np.argmin(pv[inner])]
    return float(kmin * dx), prof


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    R0 = float(sys.argv[2]) if len(sys.argv) > 2 else 9.0   # in voxels
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
    dx = 1.0
    r_core = 2.0
    t0 = time.time()

    M0, _ = seed_vortex_loop_M(n, dx, R0 * dx, r_core)

    def E(MM):
        return p0.total_energy(MM, LDG_A, LDG_B, LDG_C, dx, 3, C2)

    def G(MM):
        return p0.energy_gradient(MM, LDG_A, LDG_B, LDG_C, dx, 3, C2)

    def Ecurv(MM):
        return float(np.sum(p0.curvature_energy_density(MM, dx, 3, C2)) * dx ** 3)

    R_init, prof0 = loop_radius(M0, n, dx)
    E_init = E(M0)
    interior = p0._interior_mask((n, n, n), 3)
    g0 = float(np.sqrt(np.sum(np.where(interior[..., None, None], G(M0), 0.0) ** 2)))
    print(f"seed: N={n} R0={R0} vox  R_measured={R_init:.2f}  E={E_init:.2f}  "
          f"Ecurv={Ecurv(M0):.2f}  |g|={g0:.3f}")

    # relax in chunks, tracking the loop radius
    track = {"iter": [], "R": [], "E": [], "Ecurv": [], "gnorm": []}
    M = M0
    chunk = 250
    done = 0
    while done < iters:
        M, hist = p0.fire_relax(M, E, G, 3, max_iter=chunk, tol=1e-7,
                                dt0=0.02, dt_max=0.2, log_every=chunk)
        done += chunk
        R, _ = loop_radius(M, n, dx)
        gn = hist["gnorm"][-1]
        track["iter"].append(done); track["R"].append(R); track["E"].append(E(M))
        track["Ecurv"].append(Ecurv(M)); track["gnorm"].append(gn)
        print(f"  iter {done:5d}  R={R:.2f}  E={E(M):.2f}  Ecurv={Ecurv(M):.2f}  |g|={gn:.4f}")
        if R < 1.5 * dx:   # collapsed to the axis
            print("  -> loop radius collapsed below 1.5 vox")
            break

    R_final = track["R"][-1]
    R_series = track["R"]
    Ecurv_init = Ecurv(M0)
    Ecurv_final = track["Ecurv"][-1]
    curv_retention = Ecurv_final / max(Ecurv_init, 1e-9)
    # classify HONESTLY:
    #   collapsed  -> R ran to the axis
    #   dissolved  -> the disclination curvature combed out (texture gone) or R ran to the
    #                 boundary cap: NOT a stable localized loop
    #   stable     -> the loop RETAINS its curvature energy AND R is finite + steady + interior
    boundary_cap = 0.7 * (n / 2.0) * dx
    collapsed = R_final < max(2.0 * dx, 0.25 * R0 * dx)
    dissolved = (curv_retention < 0.3) or (R_final > 0.95 * boundary_cap)
    tail = R_series[-4:] if len(R_series) >= 4 else R_series
    steady = (max(tail) - min(tail) < 1.0 * dx)
    stable = (not collapsed) and (not dissolved) and steady and (curv_retention > 0.5)
    verdict = ("stable_finite_loop" if stable else
               "collapsed" if collapsed else
               "dissolved_curvature_combed_out" if dissolved else
               "drifting_inconclusive")
    out = {"phase": "P2-vortex-loop", "verdict": verdict, "N": n, "R0_seed_vox": R0,
           "R_init": R_init, "R_final": R_final, "iters_run": done,
           "E_init": E_init, "E_final": track["E"][-1],
           "Ecurv_init": Ecurv_init, "Ecurv_final": Ecurv_final,
           "curv_retention": curv_retention, "gnorm_final": track["gnorm"][-1],
           "ldg": {"a": LDG_A, "b": LDG_B, "c": LDG_C, "vac_Tr2": VAC_TR2},
           "track": track,
           "note": ("plain +1/2 disclination ring, uniaxial seed, amplitude-only V_M; "
                    "FORK if collapsed -> the chiral/knot structure (Duda) is the next "
                    "choice, flagged not assumed")}
    with open(os.path.join(CKPT, "p2_vortex_loop.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nP2 verdict: {verdict.upper()}  (R {R_init:.2f} -> {R_final:.2f} vox, "
          f"{done} iters, {round(time.time()-t0,1)}s)")
    print(f"checkpoint -> {os.path.join(CKPT, 'p2_vortex_loop.json')}")


if __name__ == "__main__":
    main()
