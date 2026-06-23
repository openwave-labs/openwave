"""M5.11 P0 - energy minimizer + V_M/LdG validation (index-0, standalone f64).

Builds the piece the production engine lacks: a true static energy MINIMIZER
(FIRE + L-BFGS to |dE/dM| -> 0). The leapfrog `evolve_M` only does dynamics; the
relax route also sidesteps the V-on leapfrog instability (11a risk table).

Mirrors the production functional exactly (engine2_pde.py:280, :291, :327):

    U(M) = c2 * 4 * sum_{mu<nu} || [d_mu M, d_nu M] ||_F^2   +   V_M(M)
    V_M  = a*Tr(Msp^2) - b*Tr(Msp^3) + c*(Tr Msp^2)^2     (spatial 3x3 block only)
    G_a  = 8 * sum_nu [[M_a, M_nu], M_nu]   (= dU_curv/dM_a, the curvature flux)
    dU/dM = dV_M - c2 * div(G)                            (static EL gradient)

Index-0 convention: M is 4x4, time/g axis = index 0, spatial block = [1:4,1:4];
V acts on the spatial block only (the engine freezes the boost-decoupled g axis).

P0 validation gates (run: `python v11_p0_minimizer.py <mode>`):
  gradcheck : analytic dU/dM matches central finite-difference   (rel err < 1e-6)
  vm_ldg    : this V_M reproduces the engine2_pde.V_M formula     (bit-level + ref)
  phi4      : the minimizer descends monotonically to the         (E -> 2*sqrt2/3)
              analytic phi^4-kink energy (independent textbook problem)
  hedgehog  : relax a seeded 3x3 hedgehog under V_M -> |grad|->0   (smoke test)
  all       : run every gate, write a checkpoint JSON

Checkpoints -> _checkpoints/.  No file > 1 MB is written.
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

# index-0 vacuum spectrum D = diag(g, 1, delta, 0); g decoupled from V (spatial block)
G_TIME = 8.0          # placeholder boost scale for the smoke tests (real run: graded 1e10)
DELTA = 0.3           # placeholder; Duda's physical delta ~ 1e-10 enters at P1/P5


# ----------------------------------------------------------------------------
# core algebra (per-voxel 4x4, vectorized over a grid of arbitrary leading shape)
# ----------------------------------------------------------------------------
def _spatial(M):
    """spatial 3x3 block [1:4,1:4] of the 4x4 field (index-0)."""
    return M[..., 1:4, 1:4]


def V_M(M, a, b, c):
    """Landau-de Gennes potential density, per voxel. Mirrors engine2_pde.V_M:291
    exactly: a*Tr(M^2) - b*Tr(M^3) + c*(Tr M^2)^2 on the spatial 3x3 block."""
    msp = _spatial(M)
    m2 = msp @ msp
    tr2 = np.trace(m2, axis1=-2, axis2=-1)
    tr3 = np.trace(m2 @ msp, axis1=-2, axis2=-1)
    return a * tr2 - b * tr3 + c * tr2 * tr2


def dV_M(M, a, b, c):
    """dV/dM, embedded in 4x4 (time row/col = 0). Mirrors engine2_pde.dV_M:311:
    2a*M - 3b*M^2 + 4c*Tr(M^2)*M on the spatial block."""
    msp = _spatial(M)
    m2 = msp @ msp
    tr2 = np.trace(m2, axis1=-2, axis2=-1)[..., None, None]
    dsp = 2.0 * a * msp - 3.0 * b * m2 + 4.0 * c * tr2 * msp
    out = np.zeros_like(M)
    out[..., 1:4, 1:4] = dsp
    return out


def _commutator(A, B):
    return A @ B - B @ A


def _central_diff(M, axis, dx):
    """central difference d_axis M over a spatial grid axis (0=x,1=y,2=z of the grid).
    Dirichlet: one-sided would bias the boundary, so callers mask the boundary."""
    f = np.zeros_like(M)
    sl_p = [slice(None)] * M.ndim
    sl_m = [slice(None)] * M.ndim
    sl_o = [slice(None)] * M.ndim
    sl_p[axis] = slice(2, None)
    sl_m[axis] = slice(0, -2)
    sl_o[axis] = slice(1, -1)
    f[tuple(sl_o)] = (M[tuple(sl_p)] - M[tuple(sl_m)]) / (2.0 * dx)
    return f


def grad_axes(M, dx, ndim_space):
    """list of d_mu M for the spatial grid axes mu = 0..ndim_space-1."""
    return [_central_diff(M, ax, dx) for ax in range(ndim_space)]


def curvature_energy_density(M, dx, ndim_space, c2=1.0):
    """U_curv density = c2 * 4 * sum_{mu<nu} ||[d_mu M, d_nu M]||_F^2 (Frobenius)."""
    Ms = grad_axes(M, dx, ndim_space)
    acc = np.zeros(M.shape[:-2])
    for mu in range(ndim_space):
        for nu in range(mu + 1, ndim_space):
            comm = _commutator(Ms[mu], Ms[nu])
            acc = acc + np.sum(comm * comm, axis=(-2, -1))
    return c2 * 4.0 * acc


def curvature_flux(M, dx, ndim_space):
    """G_a = 8 * sum_{nu != a} [[M_a, M_nu], M_nu], one per spatial axis a.
    Mirrors engine2_pde.compute_curvature_flux:327."""
    Ms = grad_axes(M, dx, ndim_space)
    G = []
    for a in range(ndim_space):
        acc = np.zeros_like(M)
        for nu in range(ndim_space):
            if nu == a:
                continue
            acc = acc + _commutator(_commutator(Ms[a], Ms[nu]), Ms[nu])
        G.append(8.0 * acc)
    return G


def energy_density(M, a, b, c, dx, ndim_space, c2=1.0):
    return curvature_energy_density(M, dx, ndim_space, c2) + V_M(M, a, b, c)


def total_energy(M, a, b, c, dx, ndim_space, c2=1.0):
    """U = integral of the density (sum * dV). Boundary voxels carry no valid
    central-diff curvature; they are pinned (Dirichlet) so excluding them from the
    integral is consistent with pinning them in the gradient."""
    dens = energy_density(M, a, b, c, dx, ndim_space, c2)
    dV = dx ** ndim_space
    return float(np.sum(dens) * dV)


def energy_gradient(M, a, b, c, dx, ndim_space, c2=1.0):
    """dU/dM = dV_M - c2 * div(G), times dV (so it is the gradient of total_energy).
    div(G) = sum_a d_a G_a (central diff)."""
    G = curvature_flux(M, dx, ndim_space)
    divG = np.zeros_like(M)
    for a_ in range(ndim_space):
        divG = divG + _central_diff(G[a_], a_, dx)
    dU = dV_M(M, a, b, c) - c2 * divG
    return dU * (dx ** ndim_space)


# ----------------------------------------------------------------------------
# minimizers
# ----------------------------------------------------------------------------
def _interior_mask(shape_space, ndim_space):
    """True on interior voxels (boundary pinned, Dirichlet)."""
    m = np.ones(shape_space, dtype=bool)
    for ax in range(ndim_space):
        idx = [slice(None)] * ndim_space
        idx[ax] = 0
        m[tuple(idx)] = False
        idx[ax] = -1
        m[tuple(idx)] = False
    return m


def fire_relax(M0, energy_fn, grad_fn, n_space, max_iter=4000, tol=1e-7,
               dt0=0.05, dt_max=0.5, f_inc=1.1, f_dec=0.5, alpha0=0.1,
               f_alpha=0.99, n_min=5, log_every=200, pin=None):
    """FIRE (Bitzek 2006) static relaxer. Returns (M, history). `pin` masks DOF
    held fixed (boundary + any seeded constraints). Velocity-Verlet with the FIRE
    velocity mixing; energy must descend monotonically once dt stabilizes."""
    M = M0.copy()
    shape_space = M.shape[:-2]
    interior = _interior_mask(shape_space, n_space)
    free = interior if pin is None else (interior & ~pin)
    free4 = free[..., None, None]

    v = np.zeros_like(M)
    dt = dt0
    alpha = alpha0
    n_pos = 0
    hist = {"E": [], "gnorm": [], "iter": []}
    g = grad_fn(M)
    g = np.where(free4, g, 0.0)
    for it in range(max_iter):
        F = -g
        v = v + dt * F
        P = float(np.sum(F * v))
        fnorm = np.sqrt(np.sum(F * F))
        vnorm = np.sqrt(np.sum(v * v))
        if fnorm > 0:
            v = (1.0 - alpha) * v + alpha * (vnorm / (fnorm + 1e-30)) * F
        if P > 0:
            n_pos += 1
            if n_pos > n_min:
                dt = min(dt * f_inc, dt_max)
                alpha = alpha * f_alpha
        else:
            n_pos = 0
            dt = dt * f_dec
            alpha = alpha0
            v = np.zeros_like(v)
        M = M + dt * v
        M = np.where(free4, M, M0)  # re-pin fixed DOF
        g = grad_fn(M)
        g = np.where(free4, g, 0.0)
        gn = float(np.sqrt(np.sum(g * g)))
        if it % log_every == 0 or it == max_iter - 1:
            E = energy_fn(M)
            hist["E"].append(E)
            hist["gnorm"].append(gn)
            hist["iter"].append(it)
        if gn < tol:
            E = energy_fn(M)
            hist["E"].append(E)
            hist["gnorm"].append(gn)
            hist["iter"].append(it)
            break
    return M, hist


def lbfgs_relax(M0, energy_fn, grad_fn, n_space, pin=None, maxiter=2000, gtol=1e-8):
    """scipy L-BFGS-B over the free DOF. Independent cross-check of FIRE."""
    from scipy.optimize import minimize
    shape = M0.shape
    shape_space = shape[:-2]
    interior = _interior_mask(shape_space, n_space)
    free = interior if pin is None else (interior & ~pin)
    free4 = free[..., None, None]
    base = M0.copy()

    def unpack(x):
        M = base.copy()
        M[free4] = x
        return M

    def fun(x):
        M = unpack(x)
        return energy_fn(M)

    def jac(x):
        M = unpack(x)
        g = grad_fn(M)
        return g[free4]

    x0 = base[free4].copy()
    res = minimize(fun, x0, jac=jac, method="L-BFGS-B",
                   options={"maxiter": maxiter, "gtol": gtol, "ftol": 1e-15})
    return unpack(res.x), {"E": [float(res.fun)], "gnorm": [float(np.linalg.norm(res.jac))],
                           "nit": int(res.nit), "success": bool(res.success), "msg": str(res.message)}


# ----------------------------------------------------------------------------
# validation gates
# ----------------------------------------------------------------------------
def gate_gradcheck(seed=0):
    """analytic dU/dM vs central finite-difference on a small random 3D field."""
    rng = np.random.default_rng(seed)
    n = 6
    dx = 0.7
    a, b, c, c2 = 0.5, 0.3, 0.2, 1.3
    M = np.zeros((n, n, n, 4, 4))
    # random symmetric spatial block + g on the time axis
    R = rng.standard_normal((n, n, n, 3, 3)) * 0.4
    M[..., 1:4, 1:4] = 0.5 * (R + np.swapaxes(R, -1, -2)) + np.eye(3) * 0.6
    M[..., 0, 0] = G_TIME

    def E(MM):
        return total_energy(MM, a, b, c, dx, 3, c2)

    g_an = energy_gradient(M, a, b, c, dx, 3, c2)
    # finite-diff a handful of interior, symmetric (i,j)>=(1,1) components
    interior = _interior_mask((n, n, n), 3)
    eps = 1e-6
    pts = [(2, 2, 2), (3, 2, 4), (2, 4, 3), (4, 4, 4), (3, 3, 2)]
    comps = [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]
    errs = []
    for (i, j, k) in pts:
        if not interior[i, j, k]:
            continue
        for (p, q) in comps:
            Mp = M.copy(); Mm = M.copy()
            # perturb symmetric pair together (DOF is the symmetric entry)
            Mp[i, j, k, p, q] += eps; Mp[i, j, k, q, p] += eps if p != q else 0.0
            Mm[i, j, k, p, q] -= eps; Mm[i, j, k, q, p] -= eps if p != q else 0.0
            num = (E(Mp) - E(Mm)) / (2 * eps)
            ana = g_an[i, j, k, p, q] + (g_an[i, j, k, q, p] if p != q else 0.0)
            errs.append(abs(num - ana) / (abs(num) + abs(ana) + 1e-12))
    max_rel = float(np.max(errs))
    ok = max_rel < 1e-6
    return {"gate": "gradcheck", "ok": bool(ok), "max_rel_err": max_rel, "n_checks": len(errs)}


def gate_vm_ldg(seed=1):
    """this V_M reproduces the engine formula: a*Tr2 - b*Tr3 + c*Tr2^2 on the
    spatial block, verified vs an independent einsum implementation + a hand ref."""
    rng = np.random.default_rng(seed)
    a, b, c = 0.5, 0.3, 0.2
    M = np.zeros((50, 4, 4))
    R = rng.standard_normal((50, 3, 3))
    M[..., 1:4, 1:4] = 0.5 * (R + np.swapaxes(R, -1, -2))
    M[..., 0, 0] = G_TIME  # must be ignored by V_M
    mine = V_M(M, a, b, c)
    # independent: explicit traces via einsum on the spatial block
    msp = M[..., 1:4, 1:4]
    tr2 = np.einsum("...ij,...ji->...", msp, msp)
    tr3 = np.einsum("...ij,...jk,...ki->...", msp, msp, msp)
    indep = a * tr2 - b * tr3 + c * tr2 ** 2
    match = float(np.max(np.abs(mine - indep)))
    # g-decoupling: doubling g must not change V
    M2 = M.copy(); M2[..., 0, 0] = 999.0
    g_decoupled = float(np.max(np.abs(V_M(M2, a, b, c) - mine)))
    # hand reference: M = diag(_, 2,1,0) -> Tr2=5, Tr3=9, V=.5*5-.3*9+.2*25=2.3-2.7+5=4.6
    Mh = np.zeros((1, 4, 4)); Mh[0] = np.diag([G_TIME, 2.0, 1.0, 0.0])
    ref = float(V_M(Mh, a, b, c)[0]); ref_exp = 0.5 * 5 - 0.3 * 9 + 0.2 * 25
    ok = match < 1e-12 and g_decoupled < 1e-12 and abs(ref - ref_exp) < 1e-12
    return {"gate": "vm_ldg", "ok": bool(ok), "indep_match": match,
            "g_decoupled": g_decoupled, "hand_ref": ref, "hand_ref_expected": ref_exp}


def gate_phi4():
    """independent textbook check of the minimizer: the phi^4 kink.
    E[phi] = sum [ 0.5 phi'^2 + 0.25 (phi^2-1)^2 ] dx ; analytic E_kink = 2*sqrt2/3,
    profile phi=tanh(x/sqrt2). Minimize from a crude ramp, BC phi(+-L)=+-1."""
    L = 12.0
    N = 401
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]

    def E(phi):
        d = np.gradient(phi, dx)
        return float(np.sum(0.5 * d * d + 0.25 * (phi * phi - 1.0) ** 2) * dx)

    def grad(phi):
        # dE/dphi = -phi'' + (phi^2-1)phi  (with the dx integration weight)
        lap = np.zeros_like(phi)
        lap[1:-1] = (phi[2:] - 2 * phi[1:-1] + phi[:-2]) / dx ** 2
        g = (-lap + (phi * phi - 1.0) * phi) * dx
        g[0] = 0.0; g[-1] = 0.0
        return g

    phi0 = np.clip(x / 3.0, -1, 1)   # crude ramp, far from tanh
    phi0[0] = -1.0; phi0[-1] = 1.0
    # FIRE on the 1D field (reuse the generic driver via a degenerate 4x4 embed
    # would be overkill; run a direct FIRE here on the scalar line)
    phi = phi0.copy(); v = np.zeros_like(phi)
    dt = 0.02; dt_max = 0.1; alpha = 0.1; n_pos = 0
    Es = [E(phi)]; gns = []
    for it in range(20000):
        g = grad(phi); F = -g
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
        phi = phi + dt * v
        phi[0] = -1.0; phi[-1] = 1.0
        gn = float(np.sqrt(np.sum(g * g)))
        if it % 200 == 0:
            Es.append(E(phi)); gns.append(gn)
        if gn < 1e-9:
            Es.append(E(phi)); gns.append(gn); break
    E_final = E(phi)
    E_analytic = 2.0 * np.sqrt(2.0) / 3.0
    # monotone descent (allow tiny FIRE overshoot wiggle < 1e-6)
    monotone = all(Es[i + 1] <= Es[i] + 1e-6 for i in range(len(Es) - 1))
    # profile match to tanh(x/sqrt2)
    prof_err = float(np.max(np.abs(phi - np.tanh(x / np.sqrt(2.0)))))
    ok = monotone and abs(E_final - E_analytic) < 5e-3 and prof_err < 2e-2
    return {"gate": "phi4", "ok": bool(ok), "E_final": E_final, "E_analytic": E_analytic,
            "rel_err": abs(E_final - E_analytic) / E_analytic, "profile_max_err": prof_err,
            "monotone": bool(monotone), "n_energy_samples": len(Es)}


def gate_hedgehog():
    """smoke test: relax a seeded uniaxial hedgehog spatial block under V_M (b=0
    amplitude well) on a small grid; gradient norm must fall by >> 1 decade and
    energy must descend. Not a physics gate (that is P1), just exercises the 3D
    minimizer end to end."""
    n = 12
    dx = 0.5
    a, b, c, c2 = -0.5, 0.0, 0.5, 1.0  # double-well amplitude potential (min at Tr2>0)
    xs = (np.arange(n) - (n - 1) / 2) * dx
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2) + 1e-9
    nhat = np.stack([X / r, Y / r, Z / r], axis=-1)  # radial director
    amp = np.tanh(r / (2 * dx))  # melt the core (n->0 at r=0)
    M = np.zeros((n, n, n, 4, 4))
    nn = nhat[..., :, None] * nhat[..., None, :]  # n outer n
    M[..., 1:4, 1:4] = amp[..., None, None] * nn
    M[..., 0, 0] = G_TIME

    def E(MM):
        return total_energy(MM, a, b, c, dx, 3, c2)

    def g(MM):
        return energy_gradient(MM, a, b, c, dx, 3, c2)

    g0 = float(np.sqrt(np.sum(np.where(_interior_mask((n, n, n), 3)[..., None, None], g(M), 0.0) ** 2)))
    E0 = E(M)
    Mf, hist = fire_relax(M, E, g, 3, max_iter=3000, tol=1e-7, log_every=500)
    gf = hist["gnorm"][-1]
    Ef = hist["E"][-1]
    ok = (gf < g0 / 10.0) and (Ef <= E0 + 1e-9)
    return {"gate": "hedgehog", "ok": bool(ok), "g0": g0, "gf": gf, "E0": E0, "Ef": Ef,
            "descent_decades": float(np.log10(g0 / (gf + 1e-30)))}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    gates = {"gradcheck": gate_gradcheck, "vm_ldg": gate_vm_ldg,
             "phi4": gate_phi4, "hedgehog": gate_hedgehog}
    if mode in gates:
        res = [gates[mode]()]
    elif mode == "all":
        res = [gates[k]() for k in ["gradcheck", "vm_ldg", "phi4", "hedgehog"]]
    else:
        print("modes: gradcheck | vm_ldg | phi4 | hedgehog | all")
        return
    all_ok = all(r["ok"] for r in res)
    out = {"phase": "P0", "all_ok": bool(all_ok), "wall_s": round(time.time() - t0, 1),
           "gates": res}
    for r in res:
        flag = "PASS" if r["ok"] else "FAIL"
        print(f"[{flag}] {r['gate']:10s} " +
              " ".join(f"{k}={v}" for k, v in r.items() if k not in ("gate", "ok")))
    print(f"\nP0 {'ALL PASS' if all_ok else 'FAILURES PRESENT'}  ({out['wall_s']}s)")
    if mode == "all":
        path = os.path.join(CKPT, "p0_minimizer.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"checkpoint -> {path}")


if __name__ == "__main__":
    main()
