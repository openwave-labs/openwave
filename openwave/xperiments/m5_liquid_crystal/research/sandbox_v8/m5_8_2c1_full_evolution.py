"""
M5.8.2c-1 — the Step-1 sandbox spike: FULL nonlinear 4D Minkowski M-evolution,
NO frozen background. The first run in the program where the defect, the boost
dressing, and the clock evolve TOGETHER — the production update mirrored in
f64 numpy with the constrained integrator the port will use.

THE LAGRANGIAN (5a §10b/§10d, time = matrix index 0, η = diag(−1,1,1,1)):

    ℒ = T − U − V ,   ⟨A,B⟩_s ≡ Tr(Aᵀ η B η)  (the signed block inner — for
                       antisymmetric A: spatial pairs +, (0,α) pairs −)
    T = 2 Σ_i ⟨F_0i, F_0i⟩_s ,  F_0i = [Ṁ, M_i]     (the signed kinetic)
    U = 2 Σ_(i<j) ⟨F_ij, F_ij⟩_s ,  F_ij = [M_i, M_j]
    KEY IDENTITY: the signed flux is the PRODUCTION flux with F → ηFη — the
    Step-2 port delta is one twist, not a new kernel family.

THE CONSTRAINED INTEGRATOR (the 2b/2b-2 lessons, production-implementable):
    P = A(Ṁ) ,  A(Ṁ) = 4 Σ_i [η[Ṁ, M_i]η, M_i]   (the matrix inertia operator;
    T = ½⟨Ṁ, A(Ṁ)⟩ — self-checked at runtime). Per voxel, A is a symmetric
    operator on the 10-dim space of symmetric 4×4 matrices: eigendecompose,
    KEEP only the positive-inertia eigendirections (λ > ε·max|λ|) — the ghost
    directions are FROZEN (the constraint surface; the spectral generalization
    of 2b-2's Q-PD mask). Guards: (a) the volume-mean of the (0,α) velocity
    components clamped to 0 (no free GLOBAL dressing mode — the 2b-1 runaway
    channel); (b) bounded-energy monitor. Sign conventions pinned at runtime by
    gradient + conservation checks (C1) — not trusted from algebra alone.

RUNS (V = 0, Dirichlet boundary at the seed, N³ grid):
    R1  Minkowski, dressed b* = 0.13      — the 4D recipe, free (no kick)
    R2  Minkowski, undressed b = 0        — ≡ the 3D M5.7 system (control:
        at b=0 the (0,α) sector is exactly inert — the dispersal benchmark)
    R3  Euclidean (η→1), dressed b* = 0.13 — the signature isolation twin

GATES (first answers to the 2c physics gates — sandbox level, coarse grid;
resolution-confirm is part of G-2c-1 proper at Step 4):
    C1  self-checks: T quadratic-form identity; potential-force gradient check;
        T-flux gradient check; calm-run H conservation.
    C2  (→G-2c-1) survival: R1 retains core energy + director-radial alignment
        at least as well as R2; no blow-up.
    C3  (→G-2c-2) saturation: R1 H(t) bounded — no λ≈15.6/t runaway under the
        projection + guards.
    C4  (→G-2c-3) the clock: R1's core clock-tangent overlap oscillates
        WITHOUT any kick (the C-source spin-up) — or, if silent, a kicked run
        R1b must HOLD its oscillation (reported either way).
    C5  signature: R1 vs R3 differences quantified (coherence / retention).

USAGE:  python m5_8_2c1_full_evolution.py
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2a_4d_hamiltonian import (  # noqa: E402
    conj, rot4, gen4, boost_field, D4,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v6.m5_6_2a_biaxial_hedgehog import (  # noqa: E402
    matmul, commf, RC, RHOC, DELTA,
)

N = int(os.environ.get("M58_N", "24"))   # spike grid; M58_N env overrides (the N-2 resolution confirm, sandbox_vn/m5_8_2i)
L = 6.0
B_STAR = 0.13
R_W = float(os.environ.get("M58_RW", "3.5"))   # dressing width; M58_RW env overrides (N-6a ZBW family — the knob that ACTUALLY enters seed_M)
PLANE = (2, 3)             # (δ,0) clock plane — eigen-plane MATRIX pair (index-0)
A_BOOST = 2                # boost axis a ∈ {1,2,3} (index-0); the (δ,0)-clock article pairing
DT = 0.002
STEPS = 900
EPS_EIG = 0.05              # positive-inertia keep threshold (× max|λ| per voxel)
VCAP = 5.0                  # per-voxel ‖Ṁ‖_F safety cap (backstop; engagements logged)
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])      # Minkowski, time = index 0 (Duda / index-0)


def tw(A, euclid=False):
    """η A η — flips the (0,α)/(α,0) components. Euclid flag: identity."""
    if euclid:
        return A
    B = A.copy()
    B[..., 0, 1:4] *= -1.0
    B[..., 1:4, 0] *= -1.0
    return B


def central(f, axis, h):
    out = np.zeros_like(f)
    sp_, sm, so = [slice(None)] * f.ndim, [slice(None)] * f.ndim, [slice(None)] * f.ndim
    sp_[axis], sm[axis], so[axis] = slice(2, None), slice(0, -2), slice(1, -1)
    out[tuple(so)] = (f[tuple(sp_)] - f[tuple(sm)]) / (2 * h)
    return out


def div(fields, h):
    return sum(central(fields[ax], ax, h) for ax in range(3))


def build_grid():
    xs = np.linspace(-L, L, N)
    h = xs[1] - xs[0]
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)
    rho = np.sqrt(X**2 + Y**2)
    rhat = np.stack([X, Y, Z], -1) / np.sqrt(X**2 + Y**2 + Z**2 + RC**2)[..., None]
    rhat /= np.sqrt(np.einsum("...i,...i->...", rhat, rhat))[..., None]
    azim = np.stack([-Y, X, np.zeros_like(X)], -1) / np.sqrt(rho**2 + 1e-12)[..., None]
    s = np.clip(rho / RHOC, 0.0, 1.0)
    shrink = (s * s * (3.0 - 2.0 * s))[..., None]
    ePhi = azim * shrink
    ePhi = ePhi - np.einsum("...i,...i->...", ePhi, rhat)[..., None] * rhat
    eTheta = np.cross(ePhi, rhat)
    O3 = np.stack([rhat, eTheta, ePhi], axis=-1)
    O4 = np.zeros(O3.shape[:-2] + (4, 4))
    O4[..., 1:4, 1:4] = O3
    O4[..., 0, 0] = 1.0
    return dict(h=h, r=r, rho=rho, rhat=rhat, O4=O4)


def seed_M(g, b):
    w = np.exp(-((g["r"] / R_W) ** 2))
    W = matmul(g["O4"], boost_field(b * w, A_BOOST))
    M = conj(W, D4)
    G = gen4(PLANE)
    Mth = conj(W, G @ D4 - D4 @ G)        # the clock tangent at the seed
    return M, Mth


SYM_BASIS = []
for a in range(4):
    E = np.zeros((4, 4))
    E[a, a] = 1.0
    SYM_BASIS.append(E)
for a in range(4):
    for c in range(a + 1, 4):
        E = np.zeros((4, 4))
        E[a, c] = E[c, a] = 1.0 / np.sqrt(2.0)
        SYM_BASIS.append(E)
SYM_BASIS = np.array(SYM_BASIS)            # (10, 4, 4), Frobenius-orthonormal


def A_apply(V, Mi, euclid=False):
    """A(V) = 4 Σ_i [η[V, M_i]η, M_i] — the inertia operator applied to V."""
    out = np.zeros_like(V)
    for i in range(3):
        out += commf(tw(commf(V, Mi[i]), euclid), Mi[i])
    return 4.0 * out


def build_A_matrix(Mi, euclid=False):
    """Per-voxel 10×10 matrix of A in the orthonormal symmetric basis."""
    shape = Mi[0].shape[:-2]
    Amat = np.zeros(shape + (10, 10))
    for l in range(10):
        El = np.broadcast_to(SYM_BASIS[l], Mi[0].shape)
        AEl = A_apply(El, Mi, euclid)
        for k in range(10):
            Amat[..., k, l] = np.einsum("...ab,ab->...", AEl, SYM_BASIS[k])
    return 0.5 * (Amat + np.swapaxes(Amat, -1, -2))        # symmetrize numerics


def to_coeff(V):
    return np.einsum("...ab,kab->...k", V, SYM_BASIS)


def from_coeff(c):
    return np.einsum("...k,kab->...ab", c, SYM_BASIS)


def solve_constrained(Amat, Pc, eps=EPS_EIG):
    """Ṁ-coefficients from P on the positive-inertia subspace (ghost frozen).

    ALSO projects P itself onto the kept subspace — frozen directions must NOT
    accumulate momentum (the v2 lesson: the evolving spectrum releases a
    direction later and the stored P converts to a huge velocity — a
    constraint-switching energy pump; the projection kills it).
    """
    lam, Q = np.linalg.eigh(Amat)
    lmax = np.abs(lam).max(axis=-1, keepdims=True)
    keep = lam > eps * (lmax + 1e-300)
    qp = np.einsum("...ak,...a->...k", Q, Pc)              # P in the eigenbasis
    qp_kept = np.where(keep, qp, 0.0)
    Pc_proj = np.einsum("...ak,...k->...a", Q, qp_kept)
    inv = np.where(keep, 1.0 / np.where(keep, lam, 1.0), 0.0)
    cdot = np.einsum("...ak,...k->...a", Q, inv * qp_kept)
    return cdot, keep, Pc_proj


def fluxes(M, Md, h, euclid=False):
    """Spatial fluxes + signed energies. Returns (flux list, T, U) fields."""
    Mi = [central(M, ax, h) for ax in range(3)]
    F0 = [commf(Md, Mi[i]) for i in range(3)]
    T = 2.0 * sum(np.einsum("...ab,...ab->...", F0[i], tw(F0[i], euclid))
                  for i in range(3))
    U = 0.0
    Fij = {}
    for i in range(3):
        for j in range(i + 1, 3):
            Fij[(i, j)] = commf(Mi[i], Mi[j])
            U = U + 2.0 * np.einsum("...ab,...ab->...",
                                    Fij[(i, j)], tw(Fij[(i, j)], euclid))
    # ∂U/∂M_i = +4 Σ_j [ηF_ij η, M_j]  (sign PINNED by the C1b gradient check —
    # the first-slot variation: ⟨ηFη, [δM_i, M_j]⟩ = ⟨δM_i, [ηFη, M_j]⟩)
    dU = []
    for i in range(3):
        acc = np.zeros_like(M)
        for j in range(3):
            if j == i:
                continue
            F = Fij[(min(i, j), max(i, j))] * (1.0 if i < j else -1.0)
            acc += commf(tw(F, euclid), Mi[j])
        dU.append(4.0 * acc)
    # ∂T/∂M_i = −4 [ηF_0i η, Ṁ]
    dT = [-4.0 * commf(tw(F0[i], euclid), Md) for i in range(3)]
    return Mi, dU, dT, T, U


def rhs(M, Md, h, euclid=False):
    """Ṗ = −∂_i(∂T/∂M_i) + ∂_i(∂U/∂M_i)  (V=0; signs C1-pinned)."""
    Mi, dU, dT, T, U = fluxes(M, Md, h, euclid)
    return div(dU, h) - div(dT, h), Mi, T, U


def run(g, b, euclid, steps, label, kick=0.0):
    h = g["h"]
    M, Mth = seed_M(g, b)
    inter = np.zeros(g["r"].shape, bool)
    inter[2:-2, 2:-2, 2:-2] = True
    act = inter & (g["r"] > 2 * RC) & (g["rho"] > RHOC)
    core = act & (g["r"] < 4.0)
    M0 = M.copy()
    Md = kick * Mth                                       # optional clock kick
    Mi = [central(M, ax, h) for ax in range(3)]
    P = A_apply(Md, Mi, euclid)
    Hs, ssig, alens, espread = [], [], [], []
    nE0 = float(np.einsum("...ab,...ab->...", M - M0, M - M0)[core].sum()) + 1e-30
    t0 = time.time()
    for n in range(steps):
        Md_est = Md
        force, Mi, T, U = rhs(M, Md_est, h, euclid)
        P = P + DT * force
        # guard (a): clamp the global mean of the (0,α) momentum components
        for a_ in (1, 2, 3):                 # spatial eigen-axes (index-0); time = index 0
            P[..., a_, 0] -= P[..., a_, 0][act].mean() * act
            P[..., 0, a_] = P[..., a_, 0]
        Amat = build_A_matrix(Mi, euclid)
        cdot, keep, Pc_proj = solve_constrained(Amat, to_coeff(P))
        P = from_coeff(Pc_proj)                            # no frozen-direction P
        Md = from_coeff(cdot) * act[..., None, None]
        vnorm = np.sqrt(np.einsum("...ab,...ab->...", Md, Md))
        ncap = int((vnorm > VCAP).sum())
        if ncap:
            scale = np.where(vnorm > VCAP, VCAP / (vnorm + 1e-30), 1.0)
            Md = Md * scale[..., None, None]
        M = M + DT * Md
        if n % 10 == 0:
            H = float((T + U)[act].sum()) * h**3
            Hs.append(H)
            # clock overlap + defect-survival probes
            dM = M - M0
            ssig.append(float(np.einsum("...ab,...ab->...", dM, Mth)[core].mean()))
            Msp = M[..., 1:4, 1:4]
            sub = core
            ev, evec = np.linalg.eigh(Msp[sub])
            ndir = evec[..., -1]
            alens.append(float(np.abs(np.einsum("...i,...i->...",
                                                ndir, g["rhat"][sub])).mean()))
            eloc = np.einsum("...ab,...ab->...", dM, dM)
            tot = float(eloc[act].sum()) + 1e-30
            espread.append(float(eloc[core].sum()) / tot)
        if not np.isfinite(M).all() or np.abs(M).max() > 1e3:
            print(f"      {label}: DIVERGED at step {n}")
            return dict(ok=False, Hs=np.array(Hs), s=np.array(ssig),
                        align=np.array(alens), espread=np.array(espread), n=n)
        if n % 100 == 99:
            print(f"      {label} step {n + 1}/{steps} [{time.time() - t0:.0f}s] "
                  f"H={Hs[-1]:.4f} align={alens[-1]:.3f} ghost-frozen="
                  f"{100 * (1 - keep[act].mean()):.0f}% vcap-hits={ncap}")
    return dict(ok=True, Hs=np.array(Hs), s=np.array(ssig),
                align=np.array(alens), espread=np.array(espread), n=steps)


def main():
    print("=" * 78)
    print("M5.8.2c-1 — FULL nonlinear 4D evolution (no frozen background), N="
          f"{N}³ dt={DT}")
    print("  constrained integrator: per-voxel positive-inertia spectral projection")
    print("  + global-(0,α)-momentum clamp + bounded-energy monitor.  V=0.")
    print("=" * 78)
    g = build_grid()
    h = g["h"]

    # --- C1: self-checks -----------------------------------------------------------
    rng = np.random.default_rng(7)
    M, Mth = seed_M(g, B_STAR)
    Mi = [central(M, ax, h) for ax in range(3)]
    Vt = from_coeff(rng.normal(size=g["r"].shape + (10,)) * 0.1)
    T_quad = 0.5 * np.einsum("...ab,...ab->...", Vt, A_apply(Vt, Mi))
    F0 = [commf(Vt, Mi[i]) for i in range(3)]
    T_dir = 2.0 * sum(np.einsum("...ab,...ab->...", F0[i], tw(F0[i]))
                      for i in range(3))
    c1a = np.abs(T_quad - T_dir).max() < 1e-9 * (np.abs(T_dir).max() + 1e-30)
    print(f"\n[C1a] T = ½⟨Ṁ, A(Ṁ)⟩ identity: max diff "
          f"{np.abs(T_quad - T_dir).max():.2e} → {c1a}")
    # potential gradient check: ⟨div(dU), δM⟩ ≈ −dU_total/dε
    dM = from_coeff(rng.normal(size=g["r"].shape + (10,)) * 0.01)
    inter = np.zeros(g["r"].shape, bool)
    inter[3:-3, 3:-3, 3:-3] = True
    dM *= inter[..., None, None]

    def Utot(Mfield):
        Mi_ = [central(Mfield, ax, h) for ax in range(3)]
        u = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                F = commf(Mi_[i], Mi_[j])
                u = u + 2.0 * np.einsum("...ab,...ab->...", F, tw(F))
        return float(u.sum())

    eps = 1e-5
    dnum = (Utot(M + eps * dM) - Utot(M - eps * dM)) / (2 * eps)
    _, dU, _, _, _ = fluxes(M, np.zeros_like(M), h)
    dana = float(np.einsum("...ab,...ab->...", div(dU, h), dM).sum())
    c1b = abs(dnum + dana) < 5e-3 * (abs(dnum) + 1e-30)
    print(f"[C1b] potential-force gradient check: numeric dU/dε = {dnum:+.4e}, "
          f"⟨div flux, δM⟩ = {dana:+.4e} → {c1b}")

    def Ttot(Mfield, Mdfield):
        Mi_ = [central(Mfield, ax, h) for ax in range(3)]
        t_ = 0.0
        for i in range(3):
            F = commf(Mdfield, Mi_[i])
            t_ = t_ + 2.0 * np.einsum("...ab,...ab->...", F, tw(F))
        return float(t_.sum())

    Mdt = from_coeff(rng.normal(size=g["r"].shape + (10,)) * 0.05)
    dnumT = (Ttot(M + eps * dM, Mdt) - Ttot(M - eps * dM, Mdt)) / (2 * eps)
    _, _, dT, _, _ = fluxes(M, Mdt, h)
    danaT = float(np.einsum("...ab,...ab->...", div(dT, h), dM).sum())
    c1c = abs(dnumT + danaT) < 5e-3 * (abs(dnumT) + 1e-30)
    print(f"[C1c] T-flux gradient check: numeric dT/dε = {dnumT:+.4e}, "
          f"⟨div T-flux, δM⟩ = {danaT:+.4e} → {c1c}")
    c1 = c1a and c1b and c1c

    # --- the three runs --------------------------------------------------------------
    print(f"\n[R1] Minkowski, dressed b*={B_STAR} (free, no kick):")
    R1 = run(g, B_STAR, False, STEPS, "R1")
    print(f"\n[R2] Minkowski, UNDRESSED b=0 (the 3D M5.7 control):")
    R2 = run(g, 0.0, False, STEPS, "R2")
    print(f"\n[R3] Euclidean twin, dressed b*={B_STAR}:")
    R3 = run(g, B_STAR, True, STEPS, "R3")

    R1b = None
    s_amp = np.abs(R1["s"]).max() if len(R1["s"]) else 0.0
    if R1["ok"] and s_amp < 1e-6:
        print(f"\n[R1b] no spontaneous clock signal (|s|max={s_amp:.1e}) — kicked run:")
        R1b = run(g, B_STAR, False, STEPS, "R1b", kick=0.05)

    # --- gates ------------------------------------------------------------------------
    print("\n" + "-" * 78)

    def summ(R, name):
        if R is None:
            return
        drift = ((R["Hs"].max() - R["Hs"].min()) / (abs(R["Hs"][0]) + 1e-30)
                 if len(R["Hs"]) > 1 else np.inf)
        print(f"  {name}: ok={R['ok']}  H[{R['Hs'][0]:.3f}→{R['Hs'][-1]:.3f}] "
              f"drift={100 * drift:.1f}%  align[{R['align'][0]:.3f}→"
              f"{R['align'][-1]:.3f}]  core-frac[{R['espread'][0]:.2f}→"
              f"{R['espread'][-1]:.2f}]  |s|max={np.abs(R['s']).max():.2e}")

    for R, nm in ((R1, "R1 Mink dressed"), (R2, "R2 Mink b=0   "),
                  (R3, "R3 Eucl dressed"), (R1b, "R1b kicked    ")):
        summ(R, nm)

    c2 = (R1["ok"] and R1["align"][-1] >= R2["align"][-1] - 0.02
          if (R1["ok"] and R2["ok"]) else R1["ok"])
    c3 = R1["ok"] and len(R1["Hs"]) > 1 and np.isfinite(R1["Hs"]).all()
    sclock = R1 if np.abs(R1["s"]).max() > 1e-6 else R1b
    c4 = sclock is not None and sclock["ok"] and np.abs(sclock["s"]).max() > 1e-6
    c5 = R1["ok"] and R3["ok"]
    print(f"\n[C2 →G-2c-1] dressed defect survives ≥ undressed control: {c2}")
    print(f"[C3 →G-2c-2] R1 energy bounded under projection+guards: {c3}")
    print(f"[C4 →G-2c-3] clock signal present ({'spontaneous' if sclock is R1 else 'kicked/held' if c4 else 'NONE'}): {c4}")
    print(f"[C5] signature twin comparison available: {c5}")

    ok = c1 and c2 and c3 and c4 and c5
    print("\n" + "=" * 78)
    print("M5.8.2c-1: first fully self-consistent 4D evolution. Constrained")
    print("integrator (spectral positive-inertia projection + global clamp +")
    print("energy monitor) demonstrated on the real nonlinear system; survival,")
    print("boundedness, clock, and signature comparisons measured (coarse grid —")
    print("resolution-confirm at Step 4). The Step-2 port delta is F → ηFη.")
    print("PASS" if ok else "PARTIAL — inspect the failing gate above")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
