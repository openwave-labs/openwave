"""M5.11 - Taichi-AD energy + gradient for the M5 tensor functional (the gradient engine).

Taichi (the engine's backbone) HAS reverse-mode autodiff (ti.ad.Tape / needs_grad).
This ports the production functional into a differentiable Taichi kernel so the gradient
delta E / delta M comes from autodiff, not a hand-derived adjoint. Mirrors P0 exactly:

    U(M) = c2 * 4 * sum_{mu<nu} || [d_mu M, d_nu M] ||_F^2  +  V_M
    V_M  = a Tr(Msp^2) - b Tr(Msp^3) + c (Tr Msp^2)^2     (spatial 3x3 block)

GATE: the AD gradient must match (a) P0's numpy analytic gradient (itself FD-validated)
at interior points, and (b) the energy VALUE must match P0's total_energy. If both hold,
the AD path is trustworthy for the loop relaxer + the running-alpha minimizer.

Run:  python v11_ad_energy.py [N]    (default N=12)  -> gradcheck vs P0
"""
from __future__ import annotations

import os
import sys

import numpy as np
import taichi as ti

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v11_p0_minimizer as p0

ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32, random_seed=0,
        print_ir=False, offline_cache=True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12

M = ti.Matrix.field(4, 4, ti.f64, shape=(N, N, N), needs_grad=True)
loss = ti.field(ti.f64, shape=(), needs_grad=True)
par = ti.field(ti.f64, shape=5)        # [a, b, c, c2, dx] (runtime constants, no grad)


@ti.kernel
def compute_energy():
    # AD rule: top level = for-loops ONLY (no mixed top-level statements); read par inside
    # curvature: one loop PER axis-pair, each over the voxels where BOTH its gradients are
    # valid (interior in those 2 axes, any index in the third) -> matches P0's coverage
    # exactly (P0 includes a pair's curvature on the faces normal to the third axis).
    for i, j, k in ti.ndrange((1, N - 1), (1, N - 1), (0, N)):       # (x,y) pair
        c2 = par[3]; dx = par[4]
        inv2dx = 1.0 / (2.0 * dx); dV = dx * dx * dx
        Mx = (M[i + 1, j, k] - M[i - 1, j, k]) * inv2dx
        My = (M[i, j + 1, k] - M[i, j - 1, k]) * inv2dx
        cxy = Mx @ My - My @ Mx
        loss[None] += c2 * 4.0 * cxy.norm_sqr() * dV
    for i, j, k in ti.ndrange((1, N - 1), (0, N), (1, N - 1)):       # (x,z) pair
        c2 = par[3]; dx = par[4]
        inv2dx = 1.0 / (2.0 * dx); dV = dx * dx * dx
        Mx = (M[i + 1, j, k] - M[i - 1, j, k]) * inv2dx
        Mz = (M[i, j, k + 1] - M[i, j, k - 1]) * inv2dx
        cxz = Mx @ Mz - Mz @ Mx
        loss[None] += c2 * 4.0 * cxz.norm_sqr() * dV
    for i, j, k in ti.ndrange((0, N), (1, N - 1), (1, N - 1)):       # (y,z) pair
        c2 = par[3]; dx = par[4]
        inv2dx = 1.0 / (2.0 * dx); dV = dx * dx * dx
        My = (M[i, j + 1, k] - M[i, j - 1, k]) * inv2dx
        Mz = (M[i, j, k + 1] - M[i, j, k - 1]) * inv2dx
        cyz = My @ Mz - Mz @ My
        loss[None] += c2 * 4.0 * cyz.norm_sqr() * dV
    # V_M over all voxels (spatial 3x3 block only; M symmetric)
    for i, j, k in M:
        a = par[0]
        b = par[1]
        c = par[2]
        dx = par[4]
        dV = dx * dx * dx
        m = M[i, j, k]
        tr2 = 0.0
        tr3 = 0.0
        for p in ti.static(range(1, 4)):
            for q in ti.static(range(1, 4)):
                tr2 += m[p, q] * m[q, p]
        for p in ti.static(range(1, 4)):
            for q in ti.static(range(1, 4)):
                for r in ti.static(range(1, 4)):
                    tr3 += m[p, q] * m[q, r] * m[r, p]
        loss[None] += (a * tr2 - b * tr3 + c * tr2 * tr2) * dV


def energy_and_grad(M_np, a, b, c, c2, dx):
    """returns (energy_value, dE/dM) via Taichi autodiff."""
    M.from_numpy(M_np)
    par.from_numpy(np.array([a, b, c, c2, dx], dtype=np.float64))
    loss[None] = 0.0
    with ti.ad.Tape(loss=loss):
        compute_energy()
    return float(loss[None]), M.grad.to_numpy()


def main():
    rng = np.random.default_rng(0)
    dx = 0.7
    a, b, c, c2 = 0.5, 0.3, 0.2, 1.3
    Mnp = np.zeros((N, N, N, 4, 4))
    R = rng.standard_normal((N, N, N, 3, 3)) * 0.4
    Mnp[..., 1:4, 1:4] = 0.5 * (R + np.swapaxes(R, -1, -2)) + np.eye(3) * 0.6
    Mnp[..., 0, 0] = 8.0

    E_ad, g_ad = energy_and_grad(Mnp, a, b, c, c2, dx)
    E_p0 = p0.total_energy(Mnp, a, b, c, dx, 3, c2)
    g_p0 = p0.energy_gradient(Mnp, a, b, c, dx, 3, c2)

    # energy value match
    e_rel = abs(E_ad - E_p0) / (abs(E_p0) + 1e-12)
    # gradient match at DEEP interior points (>=3 from boundary, where stencils agree)
    lo, hi = 3, N - 4
    sl = (slice(lo, hi + 1),) * 3
    # symmetric DOF: compare the symmetric-summed gradient (P0 returns per-entry; the
    # AD gradient treats M[p,q],M[q,p] as independent inputs, so symmetrize both)
    def sym(g):
        return 0.5 * (g + np.swapaxes(g, -1, -2))
    ga = sym(g_ad)[sl]
    gp = sym(g_p0)[sl]
    denom = np.abs(ga) + np.abs(gp) + 1e-9
    rel = np.abs(ga - gp) / denom
    # focus on the meaningful (large) gradient components
    big = denom > 1e-3
    max_rel = float(np.max(rel[big])) if np.any(big) else float(np.max(rel))

    print(f"N={N}")
    print(f"energy: AD={E_ad:.8f}  P0={E_p0:.8f}  rel_err={e_rel:.2e}")
    print(f"gradient (deep interior, sym): max_rel_err={max_rel:.2e}  "
          f"(over {int(big.sum())} components > 1e-3)")
    ok = e_rel < 1e-10 and max_rel < 1e-6
    print(f"\nTaichi-AD == P0 functional: [{'PASS' if ok else 'FAIL'}]")
    return ok


if __name__ == "__main__":
    main()
