"""M5.11 P2 (AD) - does a smooth knot (Hopfion) stabilize under the M5 functional?

Uses the Taichi-AD gradient (validated == P0 to 1e-13 in v11_ad_energy.py) to drive a
FIRE relaxer on the full 4x4 tensor field. Two purposes:
  1. validate the AD-driven minimizer end-to-end (monotone energy descent, |grad|->0);
  2. the physics test: the M5 functional is U = c2*4*sum||[dM,dM]||^2 + V_M, i.e.
     4th-order curvature (scales lambda^-1 under x->lambda x) + a potential. It has NO
     2nd-order Frank term. So Derrick gives E(lambda) = lambda^-1 E_curv + lambda^3 E_pot.
     A SINGULAR defect (electron hedgehog) forces a core melt -> E_pot>0 -> finite size
     (Faber's electron, P1a). A SMOOTH Hopfion keeps Tr(M^2) at vacuum everywhere ->
     E_pot~0 -> only lambda^-1 survives -> it should EXPAND. Confirming this sharpens the
     conclusion: the stable neutrino needs a KNOTTED SINGULAR disclination (melt for the
     lambda^3 + knotting for protection), i.e. Duda's chiral nematic vortex knot.

Hopfion director (standard Hopf map, charge 1): inverse-stereographic R^3->S^3 (scale R0)
then S^3->S^2:  n -> (0,0,-1) at infinity, linked preimages near the origin.

Run:  python v11_p2_hopfion.py [N] [R0] [iters]
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

N = int(sys.argv[1]) if len(sys.argv) > 1 else 36
R0 = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 800

G_TIME = 8.0
DELTA = 0.3
VAC_TR2 = 1.0 + 2.0 * DELTA ** 2
LDG_C = 0.5
LDG_A = -2.0 * LDG_C * VAC_TR2
LDG_B = 0.0
C2 = 1.0
DX = 1.0

M = ti.Matrix.field(4, 4, ti.f64, shape=(N, N, N), needs_grad=True)
loss = ti.field(ti.f64, shape=(), needs_grad=True)
par = ti.field(ti.f64, shape=5)


@ti.kernel
def compute_energy():
    # validated == P0 (v11_ad_energy.py): 3 curvature loops (per axis-pair) + V_M loop
    for i, j, k in ti.ndrange((1, N - 1), (1, N - 1), (0, N)):
        c2 = par[3]; dx = par[4]
        inv2dx = 1.0 / (2.0 * dx); dV = dx * dx * dx
        Mx = (M[i + 1, j, k] - M[i - 1, j, k]) * inv2dx
        My = (M[i, j + 1, k] - M[i, j - 1, k]) * inv2dx
        cxy = Mx @ My - My @ Mx
        loss[None] += c2 * 4.0 * cxy.norm_sqr() * dV
    for i, j, k in ti.ndrange((1, N - 1), (0, N), (1, N - 1)):
        c2 = par[3]; dx = par[4]
        inv2dx = 1.0 / (2.0 * dx); dV = dx * dx * dx
        Mx = (M[i + 1, j, k] - M[i - 1, j, k]) * inv2dx
        Mz = (M[i, j, k + 1] - M[i, j, k - 1]) * inv2dx
        cxz = Mx @ Mz - Mz @ Mx
        loss[None] += c2 * 4.0 * cxz.norm_sqr() * dV
    for i, j, k in ti.ndrange((0, N), (1, N - 1), (1, N - 1)):
        c2 = par[3]; dx = par[4]
        inv2dx = 1.0 / (2.0 * dx); dV = dx * dx * dx
        My = (M[i, j + 1, k] - M[i, j - 1, k]) * inv2dx
        Mz = (M[i, j, k + 1] - M[i, j, k - 1]) * inv2dx
        cyz = My @ Mz - Mz @ My
        loss[None] += c2 * 4.0 * cyz.norm_sqr() * dV
    for i, j, k in M:
        a = par[0]; b = par[1]; c = par[2]; dx = par[4]
        dV = dx * dx * dx
        m = M[i, j, k]
        tr2 = 0.0; tr3 = 0.0
        for p in ti.static(range(1, 4)):
            for q in ti.static(range(1, 4)):
                tr2 += m[p, q] * m[q, p]
        for p in ti.static(range(1, 4)):
            for q in ti.static(range(1, 4)):
                for r in ti.static(range(1, 4)):
                    tr3 += m[p, q] * m[q, r] * m[r, p]
        loss[None] += (a * tr2 - b * tr3 + c * tr2 * tr2) * dV


def energy_and_grad(M_np):
    M.from_numpy(M_np)
    par.from_numpy(np.array([LDG_A, LDG_B, LDG_C, C2, DX]))
    loss[None] = 0.0
    with ti.ad.Tape(loss=loss):
        compute_energy()
    return float(loss[None]), M.grad.to_numpy()


def curvature_energy_only(M_np):
    M.from_numpy(M_np)
    par.from_numpy(np.array([0.0, 0.0, 0.0, C2, DX]))   # V off -> curvature only
    loss[None] = 0.0
    compute_energy()
    return float(loss[None])


def seed_hopfion(N, dx, R0):
    xs = (np.arange(N) - (N - 1) / 2.0) * dx
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    r2 = X ** 2 + Y ** 2 + Z ** 2
    d = r2 + R0 ** 2
    X1 = 2 * R0 * X / d; X2 = 2 * R0 * Y / d; X3 = 2 * R0 * Z / d
    X4 = (r2 - R0 ** 2) / d
    n1 = 2 * (X1 * X3 + X2 * X4)
    n2 = 2 * (X2 * X3 - X1 * X4)
    n3 = X1 ** 2 + X2 ** 2 - X3 ** 2 - X4 ** 2
    nhat = np.stack([n1, n2, n3], axis=-1)
    nhat = nhat / np.linalg.norm(nhat, axis=-1, keepdims=True)
    nn = nhat[..., :, None] * nhat[..., None, :]
    Msp = DELTA * np.eye(3) + (1 - DELTA) * nn          # uniaxial, smooth (Tr2 = vacuum)
    Mout = np.zeros((N, N, N, 4, 4))
    Mout[..., 1:4, 1:4] = Msp
    Mout[..., 0, 0] = G_TIME
    return Mout, (X, Y, Z, n3)


def core_radius(M_np, n3_field):
    """radius of the Hopfion core ring (n3 max = preimage of +z) in the z=0 plane."""
    mid = N // 2
    n3 = n3_field[:, :, mid]
    xs = (np.arange(N) - (N - 1) / 2.0)
    Xg, Yg = np.meshgrid(xs, xs, indexing="ij")
    s = np.sqrt(Xg ** 2 + Yg ** 2)
    # the core ring sits where n3 is largest off-axis
    mask = s > 1.5
    if not np.any(mask):
        return 0.0
    idx = np.argmax(n3[mask])
    return float(s[mask][idx])


def amp_dev(M_np):
    Msp = M_np[..., 1:4, 1:4]
    tr2 = np.trace(Msp @ Msp, axis1=-2, axis2=-1)
    return float(np.max(np.abs(tr2 - VAC_TR2)))


def interior_mask():
    m = np.ones((N, N, N), dtype=bool)
    for ax in range(3):
        idx = [slice(None)] * 3; idx[ax] = 0; m[tuple(idx)] = False
        idx[ax] = -1; m[tuple(idx)] = False
    return m


def main():
    t0 = time.time()
    M0, (X, Y, Z, n3_0) = seed_hopfion(N, DX, R0)
    free = interior_mask()[..., None, None]

    E0, g0 = energy_and_grad(M0)
    Ec0 = curvature_energy_only(M0)
    R_init = core_radius(M0, n3_0)
    print(f"seed: N={N} R0={R0}  core_R={R_init:.2f}  E={E0:.1f}  Ecurv={Ec0:.3f}  "
          f"|g|={np.linalg.norm(np.where(free, g0, 0.0)):.2f}")

    # FIRE on the AD gradient (boundary pinned)
    M_np = M0.copy()
    v = np.zeros_like(M_np)
    dt, dt_max, alpha, n_pos = 0.02, 0.2, 0.1, 0
    track = {"iter": [], "Ecurv": [], "E": [], "coreR": [], "ampdev": [], "gnorm": []}
    n3_field = n3_0  # n3 recomputed from director would need eigsolve; track seed ring proxy
    for it in range(ITERS):
        E, g = energy_and_grad(M_np)
        g = np.where(free, g, 0.0)
        F = -g
        v = v + dt * F
        P = float(np.sum(F * v))
        fn = np.sqrt(np.sum(F * F)); vn = np.sqrt(np.sum(v * v))
        if fn > 0:
            v = (1 - alpha) * v + alpha * (vn / (fn + 1e-30)) * F
        if P > 0:
            n_pos += 1
            if n_pos > 5:
                dt = min(dt * 1.1, dt_max); alpha *= 0.99
        else:
            n_pos = 0; dt *= 0.5; alpha = 0.1; v[:] = 0.0
        M_np = M_np + dt * v
        M_np = np.where(free, M_np, M0)
        if it % 50 == 0 or it == ITERS - 1:
            Ec = curvature_energy_only(M_np)
            gn = float(np.sqrt(np.sum(g * g)))
            track["iter"].append(it); track["Ecurv"].append(Ec); track["E"].append(E)
            track["coreR"].append(core_radius(M_np, n3_field)); track["ampdev"].append(amp_dev(M_np))
            track["gnorm"].append(gn)
            print(f"  it {it:4d}  E={E:.1f}  Ecurv={Ec:.3f}  ampdev={amp_dev(M_np):.3f}  |g|={gn:.4f}")

    Ec_final = track["Ecurv"][-1]
    retention = Ec_final / max(Ec0, 1e-9)
    monotone = all(track["E"][i + 1] <= track["E"][i] + 1e-6 for i in range(len(track["E"]) - 1))
    # Hopfion expanding: curvature DROPS as the texture spreads (toward uniform) under lambda^-1
    verdict = ("expanded_curvature_dropped" if retention < 0.5 else
               "held_finite_size" if retention > 0.8 else "partial")
    out = {"phase": "P2-hopfion-AD", "verdict": verdict, "N": N, "R0": R0, "iters": ITERS,
           "Ecurv_init": Ec0, "Ecurv_final": Ec_final, "curv_retention": retention,
           "E_init": E0, "E_final": track["E"][-1], "ad_minimizer_monotone": monotone,
           "ampdev_final": track["ampdev"][-1], "gnorm_final": track["gnorm"][-1],
           "track": track,
           "note": ("smooth Hopfion, uniaxial (Tr2=vacuum everywhere -> E_pot~0). Under the "
                    "4th-order-only functional (no Frank term) Derrick predicts expansion; "
                    "stable neutrino needs a knotted SINGULAR disclination (melt for lambda^3 "
                    "+ knotting for protection) = Duda's chiral nematic vortex knot.")}
    with open(os.path.join(CKPT, "p2_hopfion.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nP2-Hopfion verdict: {verdict.upper()}  (Ecurv {Ec0:.2f}->{Ec_final:.2f}, "
          f"retention {retention:.2%}, AD-FIRE monotone={monotone}, {round(time.time()-t0,1)}s)")
    print(f"checkpoint -> {os.path.join(CKPT, 'p2_hopfion.json')}")


if __name__ == "__main__":
    main()
