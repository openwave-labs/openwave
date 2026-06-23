"""M5.11 P2 - the heliknoton: does the CHIRAL term + a helical background stabilize a knot?

Run 2 showed a smooth Hopfion EXPANDS under the bare M5 functional (uniform far-field, no
chiral term): U = c2*4*sum||[dM,dM]||^2 + V_M has uniform n as its minimum, so any smooth
knot combs out (Derrick: only lambda^-1 survives at vacuum amplitude). The Smalyukh/Tai 2020
thesis (../../theory/liquid_crystal_defects/) names the missing ingredient: the CHIRAL term, not a
Skyrme term. This script adds exactly the two missing pieces and runs the clean A/B test.

The missing term (LdG Q-tensor chiral Lifshitz invariant, thesis Eq. 6):

    F_chiral = integral  2 q0 L  eps_ikl  Q_ij  d_k Q_lj   d^3r ,     q0 = 2 pi / p

Q = Msp - (1/3 Tr Msp) I is the traceless LdG tensor, but the trace-shift drops out of the
eps-contraction exactly (eps_ikl symmetric-Q antisymmetric-index => 0), so the term is
computed directly on the spatial block Msp. It is LINEAR in one derivative (a DM-like term):
it makes the medium prefer a helical twist of pitch p and frustrates the uniform state.

Why this can stabilize a finite knot where run 2 could not. Derrick under x->lambda x:
  curvature (4th order) ~ lambda^-1  (>0)
  chiral (1 derivative)  ~ lambda^2 * (sign)   (NEGATIVE when the twist matches q0)
  V_M (potential)        ~ lambda^3  (~0 at vacuum amplitude, uniaxial)
Chiral alone does NOT beat Derrick-Hobart (E = A/lambda - C lambda^2 -> wants lambda->inf).
The thesis (p.9) is explicit: "in addition to chiral interactions, confinement and surface
interactions help overcome the constraints of the Derrick-Hobart theorem." So the FINITE BOX
with the boundary pinned to the helix IS the confinement (faithful to Smalyukh's finite-cell
heliknotons). The test: in that finite chiral cell, does a seeded Hopf knot survive as a
localized structure ABOVE the bare-helix background, or does it comb back to the bare helix?

Clean A/B (the scientific content): chiral strength Lc swept. Lc=0 = the run-2 control
(knot combs to the bare helix); Lc>0 should retain localized excess curvature if the chiral
term + confinement stabilize the heliknoton.

Modes:
  chiral_check : FD gradcheck of the chiral term vs Taichi AD + helix-favorability gate
  helix        : relax the bare helix in the box (confirm it is a near-stationary background)
  sweep        : seed heliknoton, AD-FIRE relax, sweep Lc, report excess-curvature retention
  all (default): run all three in order

Run:  python v11_p2_heliknoton.py [mode] [N] [pitch] [iters]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=True, random_seed=0)

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "_checkpoints")
os.makedirs(CKPT, exist_ok=True)

MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 32
PITCH = float(sys.argv[3]) if len(sys.argv) > 3 else 24.0
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 1200

# LdG / amplitude (same as the run-2 Hopfion: uniaxial, vacuum amplitude => V_M ~ 0,
# so the ONLY changes vs run 2 are the helical background + the chiral term)
G_TIME = 8.0
DELTA = 0.3
VAC_TR2 = 1.0 + 2.0 * DELTA ** 2
LDG_C = 0.5
LDG_A = -2.0 * LDG_C * VAC_TR2
LDG_B = 0.0
C2 = 1.0
DX = 1.0
Q0 = 2.0 * np.pi / PITCH
R0 = PITCH / 4.0          # Hopfion size ~ quarter pitch (knot core ~ half pitch across)

# the 6 nonzero Levi-Civita permutations (i,k,l,sign) on spatial indices 1,2,3
PERMS = ((1, 2, 3, 1.0), (2, 3, 1, 1.0), (3, 1, 2, 1.0),
         (1, 3, 2, -1.0), (3, 2, 1, -1.0), (2, 1, 3, -1.0))

M = ti.Matrix.field(4, 4, ti.f64, shape=(N, N, N), needs_grad=True)
loss = ti.field(ti.f64, shape=(), needs_grad=True)
par = ti.field(ti.f64, shape=8)        # [a, b, c, c2, dx, q0, Lchiral, K_frank]


@ti.kernel
def compute_energy():
    # --- curvature: 3 loops (per axis-pair), validated == P0 in v11_ad_energy.py ---
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
    # --- V_M over all voxels (spatial 3x3 block; M symmetric) ---
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
    # --- Frank elastic K|grad Msp|^2 + chiral Lifshitz 2 q0 L eps_ikl Msp_ij d_k Msp_lj ---
    # these two are the cholesteric free energy (L/2)|grad Q - q0(...)|^2 (one constant);
    # the chiral cross-term needs its Frank partner or it is unbounded below (helix runaway).
    for i, j, k in ti.ndrange((1, N - 1), (1, N - 1), (1, N - 1)):
        q0 = par[5]; Lc = par[6]; Kf = par[7]; dx = par[4]
        inv2dx = 1.0 / (2.0 * dx); dV = dx * dx * dx
        gx = (M[i + 1, j, k] - M[i - 1, j, k]) * inv2dx
        gy = (M[i, j + 1, k] - M[i, j - 1, k]) * inv2dx
        gz = (M[i, j, k + 1] - M[i, j, k - 1]) * inv2dx
        loss[None] += Kf * (gx.norm_sqr() + gy.norm_sqr() + gz.norm_sqr()) * dV
        m = M[i, j, k]
        chi = 0.0
        for pi, pk, pl, s in ti.static(PERMS):
            gk = gx if pk == 1 else (gy if pk == 2 else gz)   # d along spatial axis pk
            for jj in ti.static(range(1, 4)):
                chi += s * m[pi, jj] * gk[pl, jj]
        loss[None] += 2.0 * q0 * Lc * chi * dV


def set_par(a, b, c, c2, dx, q0, Lc, Kf):
    par.from_numpy(np.array([a, b, c, c2, dx, q0, Lc, Kf], dtype=np.float64))


def energy_and_grad(M_np, a, b, c, c2, dx, q0, Lc, Kf):
    M.from_numpy(M_np)
    set_par(a, b, c, c2, dx, q0, Lc, Kf)
    loss[None] = 0.0
    with ti.ad.Tape(loss=loss):
        compute_energy()
    return float(loss[None]), M.grad.to_numpy()


def energy_only(M_np, a, b, c, c2, dx, q0, Lc, Kf):
    M.from_numpy(M_np)
    set_par(a, b, c, c2, dx, q0, Lc, Kf)
    loss[None] = 0.0
    compute_energy()
    return float(loss[None])


def curvature_only(M_np):
    return energy_only(M_np, 0.0, 0.0, 0.0, C2, DX, 0.0, 0.0, 0.0)


# ---------- numpy reference for the chiral term (independent of Taichi, for the gradcheck) ----
def chiral_energy_np(M_np, q0, Lc, dx):
    Msp = M_np[..., 1:4, 1:4]               # (N,N,N,3,3)
    inv2dx = 1.0 / (2.0 * dx)
    gx = np.zeros_like(Msp); gy = np.zeros_like(Msp); gz = np.zeros_like(Msp)
    gx[1:-1] = (Msp[2:] - Msp[:-2]) * inv2dx
    gy[:, 1:-1] = (Msp[:, 2:] - Msp[:, :-2]) * inv2dx
    gz[:, :, 1:-1] = (Msp[:, :, 2:] - Msp[:, :, :-2]) * inv2dx
    grads = {1: gx, 2: gy, 3: gz}
    dens = np.zeros(Msp.shape[:3])
    for (pi, pk, pl, s) in PERMS:
        g = grads[pk]
        dens += s * np.einsum('...j,...j->...', Msp[..., pi - 1, :], g[..., pl - 1, :])
    interior = np.zeros_like(dens, dtype=bool)
    interior[1:-1, 1:-1, 1:-1] = True
    return float(np.sum(2.0 * q0 * Lc * dens[interior]) * dx ** 3)


def frank_energy_np(M_np, K, dx):
    Msp = M_np[..., 1:4, 1:4]
    inv2dx = 1.0 / (2.0 * dx)
    gx = (Msp[2:] - Msp[:-2]) * inv2dx
    gy = (Msp[:, 2:] - Msp[:, :-2]) * inv2dx
    gz = (Msp[:, :, 2:] - Msp[:, :, :-2]) * inv2dx
    gxi = gx[:, 1:-1, 1:-1]; gyi = gy[1:-1, :, 1:-1]; gzi = gz[1:-1, 1:-1, :]
    dens = (np.sum(gxi ** 2, axis=(-2, -1)) + np.sum(gyi ** 2, axis=(-2, -1))
            + np.sum(gzi ** 2, axis=(-2, -1)))
    return float(K * np.sum(dens) * dx ** 3)


# ---------- seeds ----------
def helix_director(X, Y, Z, q0, hand=1.0):
    """bare cholesteric helix, axis z, director rotating in the xy-plane.
    hand=+1/-1 selects the handedness (sign of the twist) to match the chiral term."""
    c = np.cos(q0 * Z); s = np.sin(hand * q0 * Z)
    nh = np.stack([c, s, 0.0 * Z], axis=-1)
    return nh / np.linalg.norm(nh, axis=-1, keepdims=True)


def hopf_director(X, Y, Z, R0):
    """standard Hopf map R^3->S^2, charge 1, -> (0,0,-1) at infinity."""
    r2 = X ** 2 + Y ** 2 + Z ** 2
    d = r2 + R0 ** 2
    X1 = 2 * R0 * X / d; X2 = 2 * R0 * Y / d; X3 = 2 * R0 * Z / d
    X4 = (r2 - R0 ** 2) / d
    n1 = 2 * (X1 * X3 + X2 * X4)
    n2 = 2 * (X2 * X3 - X1 * X4)
    n3 = X1 ** 2 + X2 ** 2 - X3 ** 2 - X4 ** 2
    nh = np.stack([n1, n2, n3], axis=-1)
    return nh / np.linalg.norm(nh, axis=-1, keepdims=True)


def singular_hopfion_tensor(X, Y, Z, R0, r_c, melt_floor):
    """a Hopf-charge-1 director (linked preimages = the topological PROTECTION) with a SINGULAR
    melt along its core ring (preimage of +z at rho=R0, z=0) = the lambda^3 SIZE-FIX. Combines
    the two ingredients each prior negative lacked: run-2's smooth Hopfion had the linking but no
    melt (expanded); run-4's melted loop had the melt but no linking (dissolved).
    melt_floor: amplitude at the ring core (1.0 = smooth control = run-2; 0.0 = full melt).
    Derrick: E(lambda) = A/lambda (Hopf curvature) + B*lambda^3 (melt) => a finite-size minimum."""
    nH = hopf_director(X, Y, Z, R0)
    rho = np.sqrt(X ** 2 + Y ** 2)
    d_ring = np.sqrt((rho - R0) ** 2 + Z ** 2)              # distance to the Hopf core ring
    amp = melt_floor + (1.0 - melt_floor) * np.tanh(d_ring / r_c)   # ->melt_floor at the ring
    nn = nH[..., :, None] * nH[..., None, :]
    Msp = amp[..., None, None] * (DELTA * np.eye(3) + (1 - DELTA) * nn)
    Mout = np.zeros(X.shape + (4, 4))
    Mout[..., 1:4, 1:4] = Msp
    Mout[..., 0, 0] = G_TIME
    return Mout, nH


def heliknoton_director(X, Y, Z, R0, q0):
    """Hopf knot embedded in the helix: rotate the Hopf director so its (0,0,-1) far-field
    maps onto the local helix director. Ry(pi/2): (x,y,z)->(z,y,-x); then Rz(q0 z).
    At infinity nH->(0,0,-1) -> (-cos q0 z, -sin q0 z, 0) = the helix line (n == -n)."""
    nH = hopf_director(X, Y, Z, R0)
    a1 = nH[..., 2]; a2 = nH[..., 1]; a3 = -nH[..., 0]      # Ry(pi/2)
    c = np.cos(q0 * Z); s = np.sin(q0 * Z)                  # Rz(q0 z)
    b1 = c * a1 - s * a2
    b2 = s * a1 + c * a2
    b3 = a3
    nh = np.stack([b1, b2, b3], axis=-1)
    return nh / np.linalg.norm(nh, axis=-1, keepdims=True)


def disclination_loop_tensor(X, Y, Z, R, r_c):
    """singular +1/2 disclination LOOP (ring radius R in the z=0 plane), meridional winding
    psi/2, with a SINGULAR melted core: Msp = f(d)*(delta I + (1-delta) nn), f=tanh(d/r_c),
    f(0)=0 so the core melts (Msp->0). The melt costs V_M>0 at the core = the lambda^3
    stabilizer, exactly as for the Faber electron (a topologically-forced melt). Returns (M, n).
    psi = atan2(z, rho-R) is the angle around the ring core in the meridional (rho,z) plane;
    the director winds by pi over one loop around the core (a +1/2 disclination)."""
    rho = np.sqrt(X ** 2 + Y ** 2)
    rho_safe = np.where(rho < 1e-9, 1e-9, rho)
    rhat = np.stack([X / rho_safe, Y / rho_safe, np.zeros_like(X)], axis=-1)
    zhat = np.zeros_like(rhat); zhat[..., 2] = 1.0
    psi = np.arctan2(Z, rho - R)
    n = np.cos(psi / 2)[..., None] * rhat + np.sin(psi / 2)[..., None] * zhat
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    d = np.sqrt((rho - R) ** 2 + Z ** 2)            # distance to the ring core
    f = np.tanh(d / r_c)                             # melt profile (0 at core -> 1 far)
    nn = n[..., :, None] * n[..., None, :]
    Msp = f[..., None, None] * (DELTA * np.eye(3) + (1 - DELTA) * nn)
    Mout = np.zeros(X.shape + (4, 4))
    Mout[..., 1:4, 1:4] = Msp
    Mout[..., 0, 0] = G_TIME
    return Mout, n


def melt_diag(M_np):
    """diagnose the disclination core via the melt: voxels with Tr(Msp^2) well below vacuum.
    Returns (melt_volume, melt_ring_R, min_tr2). A persistent ring at finite R = loop held;
    melt_volume->0 = healed/dissolved; melt_ring_R->0 = collapsed to a point."""
    Msp = M_np[..., 1:4, 1:4]
    tr2 = np.trace(Msp @ Msp, axis1=-2, axis2=-1)
    melted = tr2 < (VAC_TR2 - 0.3)
    vol = int(np.sum(melted))
    if vol == 0:
        return vol, 0.0, float(np.min(tr2))
    X, Y, Z = grid()
    rho = np.sqrt(X ** 2 + Y ** 2)
    depth = np.clip(VAC_TR2 - tr2, 0.0, None) * melted
    ring_R = float(np.sum(rho * depth) / (np.sum(depth) + 1e-12))
    return vol, ring_R, float(np.min(tr2))


def tensor_from_director(nhat):
    nn = nhat[..., :, None] * nhat[..., None, :]
    Msp = DELTA * np.eye(3) + (1 - DELTA) * nn             # uniaxial, vacuum amplitude
    Mout = np.zeros(nhat.shape[:3] + (4, 4))
    Mout[..., 1:4, 1:4] = Msp
    Mout[..., 0, 0] = G_TIME
    return Mout


def grid():
    xs = (np.arange(N) - (N - 1) / 2.0) * DX
    return np.meshgrid(xs, xs, xs, indexing="ij")


def interior_mask():
    m = np.ones((N, N, N), dtype=bool)
    for ax in range(3):
        idx = [slice(None)] * 3; idx[ax] = 0; m[tuple(idx)] = False
        idx[ax] = -1; m[tuple(idx)] = False
    return m


def amp_dev(M_np):
    Msp = M_np[..., 1:4, 1:4]
    tr2 = np.trace(Msp @ Msp, axis1=-2, axis2=-1)
    return float(np.max(np.abs(tr2 - VAC_TR2)))


def excess_curv_density(M_np, helix_M):
    """per-voxel curvature density of M_np minus that of the bare helix (the heliknoton
    structure ABOVE the background). Uses the validated curvature kernel on a difference of
    densities computed by a quick numpy 4th-order-block proxy is avoided; instead we use the
    global curvature_only and a localized field-difference norm as the localization proxy."""
    dM = M_np[..., 1:4, 1:4] - helix_M[..., 1:4, 1:4]
    return np.sqrt(np.sum(dM ** 2, axis=(-2, -1)))         # |Msp - Msp_helix| per voxel


# ---------- FIRE relax on the AD gradient (boundary pinned) ----------
def fire_relax(M0, Lc, Kf, iters, label=""):
    free = interior_mask()[..., None, None]
    M_np = M0.copy()
    v = np.zeros_like(M_np)
    dt, dt_max, alpha, n_pos = 0.02, 0.2, 0.1, 0
    track = {"iter": [], "E": [], "Ecurv": [], "gnorm": [], "ampdev": []}
    E_last = None
    for it in range(iters):
        E, g = energy_and_grad(M_np, LDG_A, LDG_B, LDG_C, C2, DX, Q0, Lc, Kf)
        # the physical LdG tensor is symmetric; the chiral gradient is not, so project the
        # spatial 3x3 block back onto the symmetric subspace (keeps M a valid Q-tensor)
        g[..., 1:4, 1:4] = 0.5 * (g[..., 1:4, 1:4] + np.swapaxes(g[..., 1:4, 1:4], -1, -2))
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
        if it % 100 == 0 or it == iters - 1:
            gn = float(np.sqrt(np.sum(g * g)))
            track["iter"].append(it); track["E"].append(E)
            track["Ecurv"].append(curvature_only(M_np)); track["gnorm"].append(gn)
            track["ampdev"].append(amp_dev(M_np))
            E_last = E
            print(f"  {label} it {it:4d}  E={E:.2f}  Ecurv={track['Ecurv'][-1]:.3f}  "
                  f"|g|={gn:.4f}  ampdev={track['ampdev'][-1]:.3f}")
    return M_np, track


# ---------- modes ----------
def run_chiral_check():
    print("=" * 70)
    print("MODE chiral_check : FD gradcheck of the chiral term (the real compute_energy loop)")
    rng = np.random.default_rng(0)
    q0t, Lct, Kt = 0.2618, 1.7, 0.85
    Mnp = np.zeros((N, N, N, 4, 4))
    R = rng.standard_normal((N, N, N, 3, 3)) * 0.5
    Mnp[..., 1:4, 1:4] = 0.5 * (R + np.swapaxes(R, -1, -2)) + np.eye(3) * 0.6
    Mnp[..., 0, 0] = G_TIME
    eps = 1e-6
    c0 = N // 2
    pts = [(c0, c0, c0, 1, 1), (c0 - 2, c0 + 1, c0, 1, 2), (c0 + 2, c0 - 2, c0 - 1, 2, 3),
           (c0, c0, c0 + 2, 3, 3), (c0 - 1, c0, c0 + 2, 2, 2)]

    def gradcheck(eref, ad_grad):
        mx = 0.0
        for (i, j, k, p, q) in pts:
            Mp = Mnp.copy(); Mm = Mnp.copy()
            Mp[i, j, k, p, q] += eps; Mm[i, j, k, p, q] -= eps
            if p != q:
                Mp[i, j, k, q, p] += eps; Mm[i, j, k, q, p] -= eps
            fd = (eref(Mp) - eref(Mm)) / (2 * eps)
            ad = ad_grad[i, j, k, p, q] + (ad_grad[i, j, k, q, p] if p != q else 0.0)
            mx = max(mx, abs(fd - ad) / (abs(fd) + abs(ad) + 1e-9))
        return mx

    # chiral term only (Kf=0)
    E_ad, g_ad = energy_and_grad(Mnp, 0.0, 0.0, 0.0, 0.0, DX, q0t, Lct, 0.0)
    e_rel_c = abs(E_ad - chiral_energy_np(Mnp, q0t, Lct, DX)) / (abs(chiral_energy_np(Mnp, q0t, Lct, DX)) + 1e-12)
    fd_c = gradcheck(lambda Mx: chiral_energy_np(Mx, q0t, Lct, DX), g_ad)
    # Frank term only (Lc=0)
    E_adf, g_adf = energy_and_grad(Mnp, 0.0, 0.0, 0.0, 0.0, DX, q0t, 0.0, Kt)
    e_rel_f = abs(E_adf - frank_energy_np(Mnp, Kt, DX)) / (abs(frank_energy_np(Mnp, Kt, DX)) + 1e-12)
    fd_f = gradcheck(lambda Mx: frank_energy_np(Mx, Kt, DX), g_adf)

    # helix favorability: E_chiral(helix) should be NEGATIVE (favored), E_chiral(uniform)=0
    X, Y, Z = grid()
    E_helix = chiral_energy_np(tensor_from_director(helix_director(X, Y, Z, q0t)), q0t, Lct, DX)
    nh_unif = np.zeros((N, N, N, 3)); nh_unif[..., 0] = 1.0
    E_unif = chiral_energy_np(tensor_from_director(nh_unif), q0t, Lct, DX)

    # Frank FD tol is looser: |grad M|^2 has larger 2nd-derivatives, so central-difference FD
    # at eps=1e-6 carries ~1e-5 roundoff (the term itself matches numpy to 5e-16, AD is exact).
    ok = (e_rel_c < 1e-10 and fd_c < 1e-5 and e_rel_f < 1e-10 and fd_f < 5e-5
          and E_helix < 0 < E_unif + 1e-12)
    print(f"  chiral: energy AD vs np rel={e_rel_c:.2e}  FD gradcheck max_rel={fd_c:.2e}")
    print(f"  frank:  energy AD vs np rel={e_rel_f:.2e}  FD gradcheck max_rel={fd_f:.2e}")
    print(f"  helix favorability: E_chiral(helix)={E_helix:.2f} (want <0)  "
          f"E_chiral(uniform)={E_unif:.4f} (==0)")
    print(f"  chiral + Frank terms VALIDATED: [{'PASS' if ok else 'FAIL'}]")
    return {"e_rel_chiral": e_rel_c, "fd_chiral": fd_c, "e_rel_frank": e_rel_f, "fd_frank": fd_f,
            "E_helix": E_helix, "E_unif": E_unif, "pass": bool(ok)}


def run_helix():
    print("=" * 70)
    print(f"MODE helix : is the pitch-{PITCH} helix the ground state with Frank+chiral? (N={N})")
    L = 1.7; Kf = L / 2.0                  # Frank partner locked to the chiral constant
    X, Y, Z = grid()
    nh = helix_director(X, Y, Z, Q0)
    Mh = tensor_from_director(nh)
    free = interior_mask()[..., None, None]
    # no-Frank control (Kf=0): the chiral term alone is unbounded -> the helix is NOT stationary
    _, g0 = energy_and_grad(Mh, LDG_A, LDG_B, LDG_C, C2, DX, Q0, L, 0.0)
    g0n = float(np.sqrt(np.sum(np.where(free, g0, 0.0) ** 2)))
    # with the Frank partner (Kf=L/2): the helix should be a near-stationary ground state
    E, g = energy_and_grad(Mh, LDG_A, LDG_B, LDG_C, C2, DX, Q0, L, Kf)
    gnorm = float(np.sqrt(np.sum(np.where(free, g, 0.0) ** 2)))
    gper = gnorm / max(np.sqrt(free.sum()), 1)
    print(f"  |g|_int : no-Frank(Kf=0)={g0n:.2f}   with-Frank(Kf=L/2)={gnorm:.2f} "
          f"(|g|/dof={gper:.4f})  ampdev={amp_dev(Mh):.2e}")
    Mh_relaxed, track = fire_relax(Mh, L, Kf, 400, label="helix")
    dn = np.max(np.abs(Mh_relaxed[..., 1:4, 1:4] - Mh[..., 1:4, 1:4]))
    runaway = (not np.isfinite(track["E"][-1])) or abs(track["E"][-1]) > 10 * abs(track["E"][0])
    print(f"  after 400 FIRE steps: E {track['E'][0]:.1f}->{track['E'][-1]:.1f}  "
          f"max|dMsp|={dn:.3f}  runaway={runaway}  (stable helix => good background)")
    return {"E": E, "gnorm_noFrank": g0n, "gnorm_Frank": gnorm, "gper": gper,
            "relax_dMsp_max": float(dn), "E0": track["E"][0], "Ef": track["E"][-1],
            "runaway": bool(runaway)}


def run_sweep():
    print("=" * 70)
    print(f"MODE sweep : seed heliknoton, AD-FIRE relax, sweep Lc (N={N}, pitch={PITCH}, R0={R0})")
    X, Y, Z = grid()
    nh_helix = helix_director(X, Y, Z, Q0)
    Mhelix = tensor_from_director(nh_helix)          # the bare-helix background (baseline)
    nh_seed = heliknoton_director(X, Y, Z, R0, Q0)
    M_seed = tensor_from_director(nh_seed)

    # localization: excess |Msp - Msp_helix|, the heliknoton structure above the background
    exc0 = excess_curv_density(M_seed, Mhelix)
    exc0_tot = float(np.sum(exc0))
    Ecurv_helix = curvature_only(Mhelix)
    Ecurv_seed = curvature_only(M_seed)
    print(f"  baseline: Ecurv(bare helix)={Ecurv_helix:.3f}  Ecurv(seed)={Ecurv_seed:.3f}  "
          f"seed excess|Msp| total={exc0_tot:.2f}")

    # L is the single chiral constant; the Frank partner is locked K = L/2.
    # L=0 = control (no chiral, no Frank = pure M5 4th-order + V_M) -> knot should comb out.
    L_list = [0.0, 1.7, 5.0, 12.0]
    results = []
    for L in L_list:
        Kf = L / 2.0
        print(f"\n -- L={L} (K_frank={Kf}) --")
        Mf, track = fire_relax(M_seed, L, Kf, ITERS, label=f"L{L}")
        exc = excess_curv_density(Mf, Mhelix)
        exc_tot = float(np.sum(exc))
        retention = exc_tot / max(exc0_tot, 1e-9)
        # localization of the surviving excess: peak/mean (high => a localized core remains)
        nz = exc[interior_mask()]
        loc = float(np.max(nz) / (np.mean(nz) + 1e-12))
        Ecurv_f = curvature_only(Mf)
        excess_curv = Ecurv_f - Ecurv_helix
        gf = track["gnorm"][-1]
        finite = bool(np.isfinite(track["E"][-1]) and np.isfinite(gf))
        verdict = ("HOLDS" if finite and retention > 0.35 and loc > 4.0 else
                   "DISSOLVES" if finite and retention < 0.18 else
                   "RUNAWAY" if not finite else "PARTIAL")
        results.append({"L": L, "K_frank": Kf, "retention": retention, "localization": loc,
                        "Ecurv_final": Ecurv_f, "excess_curv_over_helix": excess_curv,
                        "gnorm_final": gf, "finite": finite, "verdict": verdict})
        print(f"  L={L}: excess|Msp| retention={retention:.2%}  localization(peak/mean)={loc:.1f}  "
              f"excessEcurv={excess_curv:.3f}  |g|={gf:.4f}  => {verdict}")

    # the A/B story: does a finite-size localized knot survive ONLY with chiral+Frank on?
    r0 = results[0]["retention"]
    holds = [r for r in results if r["verdict"] == "HOLDS"]
    rmax = max(r["retention"] for r in results)
    chiral_helps = (any(r["L"] > 0 and r["verdict"] == "HOLDS" for r in results)
                    and results[0]["verdict"] != "HOLDS")
    out = {"phase": "P2-heliknoton", "N": N, "pitch": PITCH, "q0": Q0, "R0": R0,
           "iters": ITERS, "Ecurv_helix": Ecurv_helix, "seed_excess_total": exc0_tot,
           "L_sweep": results, "chiral_frank_stabilizes": bool(chiral_helps),
           "retention_L0": r0, "retention_max": rmax, "n_holds": len(holds),
           "note": ("A/B: L=0 = control (pure M5 4th-order + V_M, no chiral/Frank) -> the seeded "
                    "knot combs back to the pinned-helix boundary. L>0 turns on the cholesteric "
                    "free energy (Frank K=L/2 + chiral 2 q0 L eps Q dQ), ground state = the "
                    "pitch-2pi/q0 helix; the box boundary pinned to that helix = the finite-cell "
                    "confinement (thesis p.9) needed to beat Derrick-Hobart. retention = surviving "
                    "excess |Msp - Msp_helix| over the seed; localization = peak/mean of that "
                    "excess (a localized core => a real heliknoton).")}
    with open(os.path.join(CKPT, "p2_heliknoton.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  chiral+Frank stabilizes a localized knot (control L=0 does not): {chiral_helps} "
          f"(L=0 -> {r0:.2%}, best -> {rmax:.2%}, n_holds={len(holds)})")
    print(f"  checkpoint -> {os.path.join(CKPT, 'p2_heliknoton.json')}")
    return out


def calibrate_q0(Lc, Kf, hand=1.0, n=21):
    """find the chiral-term q0 that makes the pitch-PITCH helix STATIONARY (|g|_seed min).
    The Q-tensor amplitude rescales the chiral pitch-preference vs Frank, so the stationary
    q0 differs from the geometric 2pi/PITCH; this measures it. Returns (q0_star, g_min)."""
    X, Y, Z = grid()
    Mh = tensor_from_director(helix_director(X, Y, Z, Q0, hand))
    free = interior_mask()[..., None, None]
    q0s = np.linspace(0.3 * Q0, 2.2 * Q0, n)
    gs = []
    for q0c in q0s:
        _, g = energy_and_grad(Mh, LDG_A, LDG_B, LDG_C, C2, DX, q0c, Lc, Kf)
        gs.append(float(np.sqrt(np.sum(np.where(free, g, 0.0) ** 2))))
    gs = np.array(gs)
    i = int(np.argmin(gs))
    return float(q0s[i]), float(gs[i]), (q0s, gs)


def run_calib():
    """1) find the chiral q0 that zeroes the force on the pitch-PITCH helix (per amplitude);
    2) relax that helix to confirm it is a stable simple-helix background."""
    print("=" * 70)
    print(f"MODE calib : chiral-q0 that makes the pitch-{PITCH} helix stationary (N={N})")
    X, Y, Z = grid()
    results = []
    for hand in (1.0, -1.0):
        for Lc in (0.5, 1.0):
            Kf = Lc / 2.0
            q0_star, g_min, (q0s, gs) = calibrate_q0(Lc, Kf, hand)
            # relax the helix using the calibrated chiral q0
            Mh = tensor_from_director(helix_director(X, Y, Z, Q0, hand))
            free = interior_mask()[..., None, None]
            g_geo = float(np.sqrt(np.sum(np.where(free,
                       energy_and_grad(Mh, LDG_A, LDG_B, LDG_C, C2, DX, Q0, Lc, Kf)[1], 0.0) ** 2)))
            # temporary relax at q0_star (fire_relax uses module Q0, so relax inline here)
            M_np = Mh.copy(); v = np.zeros_like(M_np)
            dt, dt_max, alpha, n_pos = 0.02, 0.2, 0.1, 0
            Efin = None
            for it in range(500):
                E, g = energy_and_grad(M_np, LDG_A, LDG_B, LDG_C, C2, DX, q0_star, Lc, Kf)
                g[..., 1:4, 1:4] = 0.5 * (g[..., 1:4, 1:4] + np.swapaxes(g[..., 1:4, 1:4], -1, -2))
                g = np.where(free, g, 0.0); Fc = -g
                v = v + dt * Fc
                P = float(np.sum(Fc * v)); fn = np.sqrt(np.sum(Fc * Fc)); vn = np.sqrt(np.sum(v * v))
                if fn > 0:
                    v = (1 - alpha) * v + alpha * (vn / (fn + 1e-30)) * Fc
                if P > 0:
                    n_pos += 1
                    if n_pos > 5:
                        dt = min(dt * 1.1, dt_max); alpha *= 0.99
                else:
                    n_pos = 0; dt *= 0.5; alpha = 0.1; v[:] = 0.0
                M_np = M_np + dt * v; M_np = np.where(free, M_np, Mh)
                Efin = E
            dn = float(np.max(np.abs(M_np[..., 1:4, 1:4] - Mh[..., 1:4, 1:4])))
            ec = curvature_only(M_np)
            gfin = float(np.sqrt(np.sum(np.where(free,
                     energy_and_grad(M_np, LDG_A, LDG_B, LDG_C, C2, DX, q0_star, Lc, Kf)[1], 0.0) ** 2)))
            stable = ec < 5.0 and dn < 0.3 and np.isfinite(Efin)
            results.append({"hand": hand, "Lc": Lc, "Kf": Kf, "q0_geo": Q0, "q0_star": q0_star,
                            "g_seed_geo": g_geo, "g_seed_star": g_min, "Ecurv_relaxed": ec,
                            "max_dMsp": dn, "gnorm_final": gfin, "stable_simple_helix": bool(stable)})
            print(f"  hand={hand:+.0f} Lc={Lc}: q0_geo={Q0:.4f} q0*={q0_star:.4f}  "
                  f"|g|_seed {g_geo:.1f}->{g_min:.2f}  | relax: Ecurv={ec:.2f} max|dMsp|={dn:.3f} "
                  f"|g|={gfin:.2f}  stable={stable}")
    any_stable = any(r["stable_simple_helix"] for r in results)
    with open(os.path.join(CKPT, "p2_helix_calib.json"), "w") as f:
        json.dump({"phase": "P2-helix-calib", "N": N, "pitch": PITCH, "q0_geo": Q0,
                   "results": results, "any_stable_simple_helix": any_stable}, f, indent=2)
    print(f"  any stable simple helix found: {any_stable}")
    print(f"  checkpoint -> {os.path.join(CKPT, 'p2_helix_calib.json')}")
    return {"results": results, "any_stable": any_stable}


def relax_track_melt(M0, Lc, Kf, iters, label=""):
    """AD-FIRE relax (boundary pinned, symmetric-projected gradient) tracking the melt core."""
    free = interior_mask()[..., None, None]
    M_np = M0.copy(); v = np.zeros_like(M_np)
    dt, dt_max, alpha, n_pos = 0.02, 0.2, 0.1, 0
    track = {"iter": [], "E": [], "Ecurv": [], "melt_vol": [], "ring_R": [], "min_tr2": [], "gnorm": []}
    for it in range(iters):
        E, g = energy_and_grad(M_np, LDG_A, LDG_B, LDG_C, C2, DX, Q0, Lc, Kf)
        g[..., 1:4, 1:4] = 0.5 * (g[..., 1:4, 1:4] + np.swapaxes(g[..., 1:4, 1:4], -1, -2))
        g = np.where(free, g, 0.0); Fc = -g
        v = v + dt * Fc
        P = float(np.sum(Fc * v)); fn = np.sqrt(np.sum(Fc * Fc)); vn = np.sqrt(np.sum(v * v))
        if fn > 0:
            v = (1 - alpha) * v + alpha * (vn / (fn + 1e-30)) * Fc
        if P > 0:
            n_pos += 1
            if n_pos > 5:
                dt = min(dt * 1.1, dt_max); alpha *= 0.99
        else:
            n_pos = 0; dt *= 0.5; alpha = 0.1; v[:] = 0.0
        M_np = M_np + dt * v; M_np = np.where(free, M_np, M0)
        if it % 100 == 0 or it == iters - 1:
            vol, ring_R, mintr2 = melt_diag(M_np)
            ec = curvature_only(M_np)
            gn = float(np.sqrt(np.sum(g * g)))
            track["iter"].append(it); track["E"].append(E); track["Ecurv"].append(ec)
            track["melt_vol"].append(vol); track["ring_R"].append(ring_R)
            track["min_tr2"].append(mintr2); track["gnorm"].append(gn)
            print(f"  {label} it {it:4d}  E={E:.1f}  Ecurv={ec:.2f}  melt_vol={vol}  "
                  f"ring_R={ring_R:.2f}  min_tr2={mintr2:.3f}  |g|={gn:.4f}")
    return M_np, track


def director_ring_R(M_np):
    """radius of the Hopf core ring (director ~ +z) in the z~0 slab, from the leading
    eigenvector of Msp; works for smooth or melted fields. 0 if no off-axis core."""
    mid = N // 2
    Msp = M_np[:, :, mid - 1:mid + 2, 1:4, 1:4]
    Msym = 0.5 * (Msp + np.swapaxes(Msp, -1, -2))
    w, vecs = np.linalg.eigh(Msym)                 # ascending eigenvalues
    nz = np.abs(vecs[..., 2, -1])                  # |n_z| of the leading eigenvector
    X, Y, Z = grid()
    rho = np.sqrt(X[:, :, mid - 1:mid + 2] ** 2 + Y[:, :, mid - 1:mid + 2] ** 2)
    mask = rho > 1.5
    if not np.any(mask):
        return 0.0
    nz_m = np.where(mask, nz, -1.0)
    flat = int(np.argmax(nz_m))
    return float(rho.ravel()[flat])


def run_disc():
    print("=" * 70)
    R_seed = N * DX / 4.0          # ring radius ~ quarter box (well inside, room to grow/shrink)
    r_c = 2.5                      # melted-core radius
    print(f"MODE disc : singular +1/2 disclination loop (R={R_seed}, r_c={r_c}); "
          f"L=0 control vs chiral on (N={N})")
    X, Y, Z = grid()
    M0, _ = disclination_loop_tensor(X, Y, Z, R_seed, r_c)
    v0, rR0, mt0 = melt_diag(M0)
    print(f"  seed: melt_vol={v0}  ring_R={rR0:.2f}  min_tr2={mt0:.3f}  "
          f"Ecurv={curvature_only(M0):.2f}")
    results = []
    for L in (0.0, 1.7, 5.0):
        Kf = L / 2.0
        print(f"\n -- L={L} (K_frank={Kf}) --")
        Mf, track = relax_track_melt(M0, L, Kf, ITERS, label=f"L{L}")
        vol_f, rR_f, mt_f = track["melt_vol"][-1], track["ring_R"][-1], track["min_tr2"][-1]
        vol_keep = vol_f / max(v0, 1)
        # held = a melt ring persists at finite R (not healed, not collapsed to a point)
        held = vol_keep > 0.25 and rR_f > 0.4 * R_seed and np.isfinite(track["E"][-1])
        verdict = ("HELD" if held else
                   "DISSOLVED" if vol_keep < 0.15 else
                   "COLLAPSED" if rR_f < 0.3 * R_seed and vol_keep > 0.15 else "PARTIAL")
        results.append({"L": L, "K_frank": Kf, "melt_vol_seed": v0, "melt_vol_final": vol_f,
                        "melt_vol_retained": vol_keep, "ring_R_seed": R_seed, "ring_R_final": rR_f,
                        "min_tr2_final": mt_f, "gnorm_final": track["gnorm"][-1], "verdict": verdict})
        print(f"  L={L}: melt_vol {v0}->{vol_f} ({vol_keep:.0%})  ring_R {R_seed:.1f}->{rR_f:.2f}  "
              f"min_tr2={mt_f:.3f}  => {verdict}")
    chiral_helps = (any(r["L"] > 0 and r["verdict"] == "HELD" for r in results)
                    and results[0]["verdict"] != "HELD")
    out = {"phase": "P2-disclination-loop", "N": N, "R_seed": R_seed, "r_c": r_c, "iters": ITERS,
           "results": results, "chiral_helps": bool(chiral_helps),
           "note": ("singular +1/2 disclination LOOP with a melted core (Msp->0 at the ring), the "
                    "lambda^3 melt = the electron's stabilizer. L=0 = pure M5 control (run-1 plain "
                    "ring dissolved); L>0 adds the validated chiral+Frank terms. held = a melt ring "
                    "persists at finite R; dissolved = melt heals to vacuum; collapsed = R->0.")}
    with open(os.path.join(CKPT, "p2_disclination_loop.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  chiral protects the loop (control L=0 does not hold): {chiral_helps}")
    print(f"  checkpoint -> {os.path.join(CKPT, 'p2_disclination_loop.json')}")
    return out


def run_shopf():
    print("=" * 70)
    R0 = N * DX / 5.0          # Hopf core-ring radius (~6.4 for N=32; room to grow/shrink)
    r_c = 2.0                  # melt-ring thickness
    print(f"MODE shopf : singular (melted-core) Hopfion (R0={R0:.1f}, r_c={r_c}); Hopf charge "
          f"(protection) + melt (lambda^3 size). N={N}")
    X, Y, Z = grid()
    # A/B over melt depth: 1.0 = smooth control (run-2, should expand), 0.0 = full melt.
    # plus one chiral run on the full melt to see if it helps or breaks the linking.
    configs = [(1.0, 0.0), (0.5, 0.0), (0.0, 0.0), (0.0, 1.7)]   # (melt_floor, L)
    results = []
    for melt_floor, L in configs:
        Kf = L / 2.0
        M0, _ = singular_hopfion_tensor(X, Y, Z, R0, r_c, melt_floor)
        Ec0 = curvature_only(M0)
        v0, _, mt0 = melt_diag(M0)
        dR0 = director_ring_R(M0)
        print(f"\n -- melt_floor={melt_floor} L={L} -- seed: Ecurv={Ec0:.2f} melt_vol={v0} "
              f"min_tr2={mt0:.3f} dirRing={dR0:.2f}")
        Mf, track = relax_track_melt(M0, L, Kf, ITERS, label=f"mf{melt_floor}L{L}")
        Ecf = track["Ecurv"][-1]
        curv_keep = Ecf / max(Ec0, 1e-9)
        dRf = director_ring_R(Mf)
        vol_f, rR_f, mt_f = track["melt_vol"][-1], track["ring_R"][-1], track["min_tr2"][-1]
        finite = bool(np.isfinite(track["E"][-1]) and np.isfinite(Ecf))
        # a localized hold RETAINS the seed curvature at a finite ring radius. Guard against the
        # two false positives: curvature combed to ~0 (expanded) and curvature BLOWN UP (>2.5x =
        # the chiral blue-phase global texture, delocalized, NOT a localized knot).
        blue_phase = finite and L > 0 and curv_keep > 2.5
        held = finite and not blue_phase and 0.3 < curv_keep < 2.5 and 0.35 * R0 < dRf < 1.8 * R0
        verdict = ("BLUE_PHASE" if blue_phase else
                   "HELD" if held else
                   "EXPANDED" if finite and (curv_keep < 0.2 or dRf > 1.8 * R0) else
                   "COLLAPSED" if finite and dRf < 0.35 * R0 else "PARTIAL")
        results.append({"melt_floor": melt_floor, "L": L, "Ecurv_seed": Ec0, "Ecurv_final": Ecf,
                        "curv_retention": curv_keep, "dirRing_seed": dR0, "dirRing_final": dRf,
                        "melt_vol_final": vol_f, "min_tr2_final": mt_f,
                        "gnorm_final": track["gnorm"][-1], "verdict": verdict})
        print(f"  mf={melt_floor} L={L}: Ecurv {Ec0:.1f}->{Ecf:.2f} ({curv_keep:.0%})  "
              f"dirRing {dR0:.2f}->{dRf:.2f}  melt_vol={vol_f}  min_tr2={mt_f:.3f}  => {verdict}")
    smooth = next(r for r in results if r["melt_floor"] == 1.0 and r["L"] == 0.0)
    fullmelt = next(r for r in results if r["melt_floor"] == 0.0 and r["L"] == 0.0)
    melt_stabilizes = fullmelt["verdict"] == "HELD" and smooth["verdict"] != "HELD"
    out = {"phase": "P2-singular-hopfion", "N": N, "R0": R0, "r_c": r_c, "iters": ITERS,
           "results": results, "melt_stabilizes_hopfion": bool(melt_stabilizes),
           "note": ("singular Hopfion = Hopf charge 1 (linked preimages = protection) + a melted "
                    "core ring (lambda^3 size-fix). melt_floor=1 = smooth control (run-2, expands); "
                    "melt_floor=0 = full melt. Derrick: E=A/lambda + B lambda^3 -> finite size IF "
                    "the melt is present. held = Hopf curvature retained at a finite director-ring "
                    "radius; expanded = curvature combs out / ring -> box; collapsed = ring -> 0.")}
    with open(os.path.join(CKPT, "p2_singular_hopfion.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  the melt stabilizes the Hopfion (smooth expands, melted holds): {melt_stabilizes}")
    print(f"  checkpoint -> {os.path.join(CKPT, 'p2_singular_hopfion.json')}")
    return out


def main():
    t0 = time.time()
    summary = {}
    if MODE in ("chiral_check", "all"):
        summary["chiral_check"] = run_chiral_check()
    if MODE in ("calib", "all"):
        summary["calib"] = run_calib()
    if MODE in ("helix", "all"):
        summary["helix"] = run_helix()
    if MODE in ("sweep", "all"):
        summary["sweep"] = run_sweep()
    if MODE in ("disc", "all"):
        summary["disc"] = run_disc()
    if MODE in ("shopf", "all"):
        summary["shopf"] = run_shopf()
    print("=" * 70)
    print(f"done ({round(time.time() - t0, 1)}s)")
    return summary


if __name__ == "__main__":
    main()
