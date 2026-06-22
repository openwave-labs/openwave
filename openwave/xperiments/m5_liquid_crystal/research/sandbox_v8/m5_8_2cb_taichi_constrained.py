"""
M5.8.2c OPTION B — Step-1 spike: the CONSTRAINED spectral-projection integrator
(2c-1) implemented in TAICHI and validated against the f64 numpy reference.

THE JOB (5a §10e / roadmap M5.8.2c Step-2-FOLLOW-UP): the Minkowski-signed flux
is only stable under the 2c-1 constrained kernel — per voxel, the inertia
operator A(Ṁ) = 4 Σ_i [η[Ṁ,M_i]η, M_i] is eigendecomposed in the 10-dim
orthonormal symmetric-matrix basis, only positive-inertia directions evolve
(λ > ε·max|λ|), and the momentum P is PROJECTED onto the kept subspace every
step (frozen directions must not accumulate momentum — the constraint-switching
pump). This spike ports that kernel to Taichi: ONE fused kernel per step builds
the 10×10 A, runs a cyclic JACOBI eigensolve on LOCAL matrices (own
implementation — ti.sym_eig is broken on Metal/f32 for non-degenerate spectra),
projects, and writes (P, Ṁ). The validation ladder separates implementation
bugs from precision loss from backend bugs:

    [J1]  per-run Jacobi self-check: ‖QΛQᵀ − A₀‖/‖A₀‖ + ‖QᵀQ − I‖
    [B1]  ti-cpu-f64 vs numpy-f64   — implementation fidelity (should be exact
          up to eigensolver tolerance + atomic summation order)
    [B2]  ti-cpu-f32 vs ti-cpu-f64  — the f32-precision question (THE risk)
    [B2m] ti-metal-f32 vs ti-cpu-f32 — the Metal-backend trap class
    [B3]  ti-metal-f32 vs ref finals — physics reproduction on the production
          target (bounded 900 steps, H plateau, align/core-frac bands)
    [B4]  perf at 64³ Metal f32      — measure the ~5–10× step-cost claim
          (report-only; per-kernel phase timing)

Step-0 SHARP comparisons (pre-trajectory, no chaos amplification): the built
A₀ matrix field, the sorted spectrum λ₀, the keep mask, the first force, the
first solve output Ṁ₀, and M after one step. Trajectory gates are then
checked over an early horizon (divergence growth is physical, not a bug) plus
gate-level finals (H drift, alignment, core energy fraction).

MODES (subprocess-isolated — exactly one ti.init per process, the segfault
lesson; `all` orchestrates):

    python m5_8_2cb_taichi_constrained.py all
    python m5_8_2cb_taichi_constrained.py ref
    python m5_8_2cb_taichi_constrained.py ti cpu f64
    python m5_8_2cb_taichi_constrained.py ti cpu f32
    python m5_8_2cb_taichi_constrained.py ti metal f32
    python m5_8_2cb_taichi_constrained.py compare
    python m5_8_2cb_taichi_constrained.py perf

    CB_STEPS=30 python ... all      # quick smoke (env override of 900 steps)

The run configuration is the 2c-1 R1: Minkowski, dressed b* = 0.13, free (no
kick), V = 0, N=24³, dt = 0.002 — constants imported from the reference module
(single source of truth).
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ti-free import chain (verified: no `import taichi` anywhere below)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2c1_full_evolution import (  # noqa: E402
    N, L, DT, STEPS, EPS_EIG, VCAP, B_STAR, R_W, PLANE, A_BOOST, SYM_BASIS,
    seed_M, build_grid, fluxes, rhs, build_A_matrix, solve_constrained,
    A_apply, to_coeff, from_coeff, central,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v6.m5_6_2a_biaxial_hedgehog import (  # noqa: E402
    matmul, RC, RHOC,
)
from openwave.xperiments.m5_liquid_crystal.research.sandbox_v8.m5_8_2a_4d_hamiltonian import (  # noqa: E402
    conj, gen4, boost_field, D4,
)

RUN_STEPS = int(os.environ.get("CB_STEPS", STEPS))
# CB_DT overrides the step size (e.g. CB_DT=0.001 for the dt-halving
# onset-scaling discriminator — the fixed-τ runaway proof, 2026-06-05)
DT = float(os.environ.get("CB_DT", DT))
CHECK_NS = tuple(n_ for n_ in (0, 10, 50, 100, 200, 400, 890) if n_ < RUN_STEPS)
PERF_N = 64
PERF_STEPS = 20
MAX_SWEEPS = 20


def npz_path(tag):
    sfx = "" if N == 24 else f"_N{N}"   # off-default grids get their own caches (N-2)
    if RC != 0.8:
        sfx += f"_RC{RC:g}"             # off-default core radii too (historical N-6a misfire)
    if R_W != 3.5:
        sfx += f"_RW{R_W:g}"            # the REAL seed-structure knob (N-6a redesign)
    return HERE / f"_m5_8_2cb_{tag}{sfx}.npz"


def make_masks(g):
    """The exact 2c-1 run() masks."""
    inter = np.zeros(g["r"].shape, bool)
    inter[2:-2, 2:-2, 2:-2] = True
    act = inter & (g["r"] > 2 * RC) & (g["rho"] > RHOC)
    core = act & (g["r"] < 4.0)
    return act, core


def probe_state(Mpost, M0, Mth, act, core, rhat):
    """The exact 2c-1 survival/clock/alignment probes on the POST-update M.
    (H is computed separately per mode from the PRE-update (M, Ṁ_prev) pair —
    the reference takes T,U from the rhs() call of the same iteration.)"""
    dM = Mpost - M0
    s = float(np.einsum("...ab,...ab->...", dM, Mth)[core].mean())
    _, evec = np.linalg.eigh(Mpost[..., 1:4, 1:4][core])
    ndir = evec[..., -1]
    align = float(np.abs(np.einsum("...i,...i->...", ndir, rhat[core])).mean())
    eloc = np.einsum("...ab,...ab->...", dM, dM)
    espread = float(eloc[core].sum()) / (float(eloc[act].sum()) + 1e-30)
    return s, align, espread


# ════════════════════════════════════════════════════════════════════════════
# ref — the f64 numpy reference (the 2c-1 R1 loop, literally, + captures)
# ════════════════════════════════════════════════════════════════════════════


def run_ref():
    print("=" * 78)
    print(f"[ref] numpy f64 — the 2c-1 R1 loop, N={N}³ dt={DT} steps={RUN_STEPS}")
    print("=" * 78)
    g = build_grid()
    h = g["h"]
    M, Mth = seed_M(g, B_STAR)
    act, core = make_masks(g)
    M0 = M.copy()
    Md = np.zeros_like(M)
    Mi = [central(M, ax, h) for ax in range(3)]
    P = A_apply(Md, Mi)
    Hs, ssig, alens, espread = [], [], [], []
    checks, sharps = {}, {}
    t0 = time.time()
    for n in range(RUN_STEPS):
        force, Mi, T, U = rhs(M, Md, h)
        if n == 0:
            sharps["force0"] = force.copy()
        P = P + DT * force
        for a_ in (1, 2, 3):                 # spatial eigen-axes (index-0); time = matrix axis 0
            P[..., a_, 0] -= P[..., a_, 0][act].mean() * act
            P[..., 0, a_] = P[..., a_, 0]
        Amat = build_A_matrix(Mi)
        cdot, keep, Pc_proj = solve_constrained(Amat, to_coeff(P))
        P = from_coeff(Pc_proj)
        Md = from_coeff(cdot) * act[..., None, None]
        vnorm = np.sqrt(np.einsum("...ab,...ab->...", Md, Md))
        if (vnorm > VCAP).any():
            Md = Md * np.where(vnorm > VCAP, VCAP / (vnorm + 1e-30), 1.0)[..., None, None]
        if n == 0:
            lam0, _ = np.linalg.eigh(Amat)
            sharps["A0"] = Amat
            sharps["lam0"] = np.sort(lam0, axis=-1)
            sharps["keep0"] = keep.astype(np.uint8)
            sharps["Md0"] = Md.copy()
        M = M + DT * Md
        if n % 10 == 0:
            # H from the rhs-returned T,U (the pre-update M with the previous
            # Md — exact 2c-1 semantics); the rest from the post-update M
            Hs.append(float((T + U)[act].sum()) * h**3)
            s, al, es = probe_state(M, M0, Mth, act, core, g["rhat"])
            ssig.append(s)
            alens.append(al)
            espread.append(es)
        if n in CHECK_NS:
            checks[f"M_{n}"] = M.copy()
        if not np.isfinite(M).all() or np.abs(M).max() > 1e3:
            print(f"   DIVERGED at step {n}")
            return 1
        if n % 100 == 99:
            print(f"   step {n + 1}/{RUN_STEPS} [{time.time() - t0:.0f}s] "
                  f"H={Hs[-1]:.4f} align={alens[-1]:.3f}")
    wall = time.time() - t0
    np.savez(npz_path("ref"),
             M_seed=M0, Mth=Mth, act=act, core=core, rhat=g["rhat"], h=h,
             Hs=np.array(Hs), s=np.array(ssig), align=np.array(alens),
             espread=np.array(espread), check_ns=np.array(CHECK_NS),
             ok=True, wall=wall, ms_step=1e3 * wall / RUN_STEPS,
             **sharps, **checks)
    print(f"   done [{wall:.0f}s, {1e3 * wall / RUN_STEPS:.0f} ms/step] → "
          f"{npz_path('ref').name}")
    return 0


# ════════════════════════════════════════════════════════════════════════════
# the Taichi simulation factory (shared by validation + perf)
# ════════════════════════════════════════════════════════════════════════════


def build_sim(ti, fp, n, with_diag):
    """Builds fields + kernels for grid n³ at precision fp. Returns a dict."""
    TINY = 1e-300 if fp == ti.f64 else 1e-30
    JTOL2 = (1e-13 if fp == ti.f64 else 2e-7) ** 2

    Mf = ti.Matrix.field(4, 4, dtype=fp, shape=(n, n, n))
    Mdf = ti.Matrix.field(4, 4, dtype=fp, shape=(n, n, n))
    Pf = ti.Matrix.field(4, 4, dtype=fp, shape=(n, n, n))
    Gf = ti.Matrix.field(4, 4, dtype=fp, shape=(3, n, n, n))
    actf = ti.field(ti.i32, shape=(n, n, n))
    Bas = ti.Matrix.field(4, 4, dtype=fp, shape=10)
    red = ti.field(fp, shape=3)
    cnt = ti.field(ti.i32, shape=2)          # [vcap hits, frozen directions]
    # diag fields hoisted to plain locals — dict subscripts inside a @ti.kernel
    # body are a compile risk; locals resolve cleanly at kernel-compile time
    A0f = Q0f = lam0f = keep0f = None
    if with_diag:
        A0f = ti.field(fp, shape=(n, n, n, 10, 10))
        Q0f = ti.field(fp, shape=(n, n, n, 10, 10))
        lam0f = ti.field(fp, shape=(n, n, n, 10))
        keep0f = ti.field(ti.i32, shape=(n, n, n, 10))

    @ti.func
    def twm(A):
        """η A η — flips the (α,0)/(0,α) components (Minkowski only; index-0, time = matrix axis 0, spatial-eigen axes {1,2,3})."""
        B = A
        for a_ in ti.static((1, 2, 3)):
            B[a_, 0] = -B[a_, 0]
            B[0, a_] = -B[0, a_]
        return B

    @ti.func
    def comm(A, B):
        return A @ B - B @ A

    @ti.func
    def gx(i, j, k, inv2h: fp):
        out = ti.Matrix.zero(fp, 4, 4)
        if 0 < i < n - 1:
            out = (Mf[i + 1, j, k] - Mf[i - 1, j, k]) * inv2h
        return out

    @ti.func
    def gy(i, j, k, inv2h: fp):
        out = ti.Matrix.zero(fp, 4, 4)
        if 0 < j < n - 1:
            out = (Mf[i, j + 1, k] - Mf[i, j - 1, k]) * inv2h
        return out

    @ti.func
    def gz(i, j, k, inv2h: fp):
        out = ti.Matrix.zero(fp, 4, 4)
        if 0 < k < n - 1:
            out = (Mf[i, j, k + 1] - Mf[i, j, k - 1]) * inv2h
        return out

    @ti.kernel
    def k_flux(inv2h: fp):
        """G_i = ∂U/∂M_i − ∂T/∂M_i flux fields (the 2c-1 fluxes(), C1-pinned
        signs): dU_i = 4 Σ_{j≠i} [η F̃_ij η, M_j], dT_i = −4 [η F_0i η, Ṁ]."""
        for i, j, k in Mf:
            Mx = gx(i, j, k, inv2h)
            My = gy(i, j, k, inv2h)
            Mz = gz(i, j, k, inv2h)
            Md = Mdf[i, j, k]
            Fxy = comm(Mx, My)
            Fxz = comm(Mx, Mz)
            Fyz = comm(My, Mz)
            Gf[0, i, j, k] = 4.0 * (comm(twm(Fxy), My) + comm(twm(Fxz), Mz)
                                    + comm(twm(comm(Md, Mx)), Md))
            Gf[1, i, j, k] = 4.0 * (comm(twm(-Fxy), Mx) + comm(twm(Fyz), Mz)
                                    + comm(twm(comm(Md, My)), Md))
            Gf[2, i, j, k] = 4.0 * (comm(twm(-Fxz), Mx) + comm(twm(-Fyz), My)
                                    + comm(twm(comm(Md, Mz)), Md))

    @ti.kernel
    def k_force(inv2h: fp, dt: fp):
        """P += dt · div(G) (central, border-zero — the 2c-1 div())."""
        for i, j, k in Pf:
            f = ti.Matrix.zero(fp, 4, 4)
            if 0 < i < n - 1:
                f += (Gf[0, i + 1, j, k] - Gf[0, i - 1, j, k]) * inv2h
            if 0 < j < n - 1:
                f += (Gf[1, i, j + 1, k] - Gf[1, i, j - 1, k]) * inv2h
            if 0 < k < n - 1:
                f += (Gf[2, i, j, k + 1] - Gf[2, i, j, k - 1]) * inv2h
            Pf[i, j, k] += dt * f

    @ti.kernel
    def k_clamp_sum():
        for i, j, k in Pf:
            if actf[i, j, k] == 1:
                p = Pf[i, j, k]
                red[0] += p[1, 0]
                red[1] += p[2, 0]
                red[2] += p[3, 0]

    @ti.kernel
    def k_clamp_sum_planes():
        """Production-style 3-mid-plane sampling of the (α,0) momentum mean
        (index-0: time = matrix axis 0, spatial-eigen axes {1,2,3}) —
        the full-grid atomic reduction stalls on Metal (the known gotcha;
        production's sample_v03_drift uses exactly this pattern)."""
        for j, k in ti.ndrange(n, n):
            if actf[n // 2, j, k] == 1:
                p = Pf[n // 2, j, k]
                red[0] += p[1, 0]
                red[1] += p[2, 0]
                red[2] += p[3, 0]
        for i, k in ti.ndrange(n, n):
            if actf[i, n // 2, k] == 1:
                p = Pf[i, n // 2, k]
                red[0] += p[1, 0]
                red[1] += p[2, 0]
                red[2] += p[3, 0]
        for i, j in ti.ndrange(n, n):
            if actf[i, j, n // 2] == 1:
                p = Pf[i, j, n // 2]
                red[0] += p[1, 0]
                red[1] += p[2, 0]
                red[2] += p[3, 0]

    @ti.kernel
    def k_clamp_apply(m0: fp, m1: fp, m2: fp):
        """Guard (a): subtract the act-mean of the (α,0) momentum (act only;
        index-0: time = matrix axis 0, spatial-eigen axes {1,2,3}),
        then restore P symmetry everywhere — the exact 2c-1 order."""
        for i, j, k in Pf:
            if actf[i, j, k] == 1:
                Pf[i, j, k][1, 0] -= m0
                Pf[i, j, k][2, 0] -= m1
                Pf[i, j, k][3, 0] -= m2
            Pf[i, j, k][0, 1] = Pf[i, j, k][1, 0]
            Pf[i, j, k][0, 2] = Pf[i, j, k][2, 0]
            Pf[i, j, k][0, 3] = Pf[i, j, k][3, 0]

    @ti.kernel
    def k_solve(inv2h: fp, diag: ti.i32):
        """THE constrained solve, fused per voxel: build the 10×10 A in the
        sym basis → cyclic Jacobi on LOCAL matrices → positive-inertia keep →
        project P AND solve Ṁ = A⁺P on the kept subspace → act mask + v-cap."""
        for i, j, k in Mf:
            Mx = gx(i, j, k, inv2h)
            My = gy(i, j, k, inv2h)
            Mz = gz(i, j, k, inv2h)
            # --- build A[kk,li] = ⟨A_op(E_li), E_kk⟩, then symmetrize --------
            Aloc = ti.Matrix.zero(fp, 10, 10)
            for li in range(10):
                El = Bas[li]
                AE = 4.0 * (comm(twm(comm(El, Mx)), Mx)
                            + comm(twm(comm(El, My)), My)
                            + comm(twm(comm(El, Mz)), Mz))
                for kk in range(10):
                    Aloc[kk, li] = (AE * Bas[kk]).sum()
            for p in range(10):
                for q in range(p + 1, 10):
                    v = 0.5 * (Aloc[p, q] + Aloc[q, p])
                    Aloc[p, q] = v
                    Aloc[q, p] = v
            if ti.static(with_diag):
                if diag == 1:
                    for p in range(10):
                        for q in range(10):
                            A0f[i, j, k, p, q] = Aloc[p, q]
            # --- cyclic Jacobi: A ← JᵀAJ, Q ← QJ ------------------------------
            Qloc = ti.Matrix.identity(fp, 10)
            normf2 = (Aloc * Aloc).sum()
            sweep = 0
            off2 = normf2
            while sweep < MAX_SWEEPS and off2 > JTOL2 * normf2:
                for p in range(9):
                    for q in range(p + 1, 10):
                        apq = Aloc[p, q]
                        if ti.abs(apq) > 0:
                            tau = (Aloc[q, q] - Aloc[p, p]) / (2.0 * apq)
                            t = 1.0 / (ti.abs(tau) + ti.sqrt(1.0 + tau * tau))
                            if tau < 0:
                                t = -t
                            c = 1.0 / ti.sqrt(1.0 + t * t)
                            s = t * c
                            for r in range(10):       # A ← A·J (columns p,q)
                                arp = Aloc[r, p]
                                arq = Aloc[r, q]
                                Aloc[r, p] = c * arp - s * arq
                                Aloc[r, q] = s * arp + c * arq
                            for r in range(10):       # A ← Jᵀ·A (rows p,q)
                                apr = Aloc[p, r]
                                aqr = Aloc[q, r]
                                Aloc[p, r] = c * apr - s * aqr
                                Aloc[q, r] = s * apr + c * aqr
                            for r in range(10):       # Q ← Q·J
                                qrp = Qloc[r, p]
                                qrq = Qloc[r, q]
                                Qloc[r, p] = c * qrp - s * qrq
                                Qloc[r, q] = s * qrp + c * qrq
                off2 = 0.0
                for p in range(9):
                    for q in range(p + 1, 10):
                        off2 += 2.0 * Aloc[p, q] * Aloc[p, q]
                sweep += 1
            # --- keep mask + projection + A⁺ solve ----------------------------
            lmax = ti.cast(0.0, fp)
            for a_ in range(10):
                lmax = ti.max(lmax, ti.abs(Aloc[a_, a_]))
            Pm = Pf[i, j, k]
            Pc = ti.Vector.zero(fp, 10)
            for a_ in range(10):
                Pc[a_] = (Pm * Bas[a_]).sum()
            Pproj = ti.Vector.zero(fp, 10)
            cdot = ti.Vector.zero(fp, 10)
            nfroz = 0
            for kk in range(10):
                lamk = Aloc[kk, kk]
                qp = ti.cast(0.0, fp)
                for a_ in range(10):
                    qp += Qloc[a_, kk] * Pc[a_]
                if lamk > EPS_EIG * (lmax + TINY):
                    for a_ in range(10):
                        Pproj[a_] += Qloc[a_, kk] * qp
                        cdot[a_] += Qloc[a_, kk] * qp / lamk
                else:
                    nfroz += 1
            cnt[1] += nfroz
            if ti.static(with_diag):
                if diag == 1:
                    for kk in range(10):
                        lam0f[i, j, k, kk] = Aloc[kk, kk]
                        keep0f[i, j, k, kk] = (
                            1 if Aloc[kk, kk] > EPS_EIG * (lmax + TINY) else 0)
                        for a_ in range(10):
                            Q0f[i, j, k, a_, kk] = Qloc[a_, kk]
            Pnew = ti.Matrix.zero(fp, 4, 4)
            Mdnew = ti.Matrix.zero(fp, 4, 4)
            for a_ in range(10):
                Pnew += Pproj[a_] * Bas[a_]
                Mdnew += cdot[a_] * Bas[a_]
            if actf[i, j, k] == 0:
                Mdnew = ti.Matrix.zero(fp, 4, 4)
            vn = ti.sqrt((Mdnew * Mdnew).sum())
            if vn > VCAP:
                Mdnew = Mdnew * (VCAP / (vn + 1e-30))
                cnt[0] += 1
            Pf[i, j, k] = Pnew
            Mdf[i, j, k] = Mdnew

    @ti.kernel
    def k_update(dt: fp):
        for i, j, k in Mf:
            Mf[i, j, k] += dt * Mdf[i, j, k]

    return dict(Mf=Mf, Mdf=Mdf, Pf=Pf, actf=actf, Bas=Bas, red=red, cnt=cnt,
                k_flux=k_flux, k_force=k_force, k_clamp_sum=k_clamp_sum,
                k_clamp_sum_planes=k_clamp_sum_planes,
                k_clamp_apply=k_clamp_apply, k_solve=k_solve, k_update=k_update,
                A0=A0f, Q0=Q0f, lam0=lam0f, keep0=keep0f)


def step(sim, inv2h, n_act, diag=0):
    """One full constrained step — the exact 2c-1 loop order."""
    sim["k_flux"](inv2h)
    sim["k_force"](inv2h, DT)
    sim["red"].fill(0.0)
    sim["k_clamp_sum"]()
    r = sim["red"].to_numpy()
    sim["k_clamp_apply"](float(r[0]) / n_act, float(r[1]) / n_act,
                         float(r[2]) / n_act)
    sim["cnt"].fill(0)
    sim["k_solve"](inv2h, diag)
    sim["k_update"](DT)


# ════════════════════════════════════════════════════════════════════════════
# ti — the validated Taichi run (loads the ref seed; saves the same captures)
# ════════════════════════════════════════════════════════════════════════════


def run_ti(arch_name, fp_name):
    import taichi as ti
    arch = dict(cpu=ti.cpu, metal=ti.metal)[arch_name]
    fp = dict(f32=ti.f32, f64=ti.f64)[fp_name]
    ti.init(arch=arch, default_fp=fp)
    tag = f"ti_{arch_name}_{fp_name}"
    ref_file = npz_path("ref")
    if not ref_file.exists():
        print(f"[{tag}] ERROR: run `ref` first ({ref_file.name} missing)")
        return 1
    d = np.load(ref_file)
    M0, Mth, act, core = d["M_seed"], d["Mth"], d["act"], d["core"]
    rhat, h = d["rhat"], float(d["h"])
    n_act = int(act.sum())
    print("=" * 78)
    print(f"[{tag}] N={N}³ dt={DT} steps={RUN_STEPS} (seed: ref npz)")
    print("=" * 78)
    sim = build_sim(ti, fp, N, with_diag=True)
    sim["Mf"].from_numpy(M0.astype(np.float64 if fp == ti.f64 else np.float32))
    sim["Mdf"].fill(0.0)
    sim["Pf"].fill(0.0)
    sim["actf"].from_numpy(act.astype(np.int32))
    sim["Bas"].from_numpy(SYM_BASIS.astype(
        np.float64 if fp == ti.f64 else np.float32))
    inv2h = 1.0 / (2.0 * h)
    Hs, ssig, alens, espread = [], [], [], []
    checks, sharps = {}, {}
    ok = True
    t0 = time.time()
    for n in range(RUN_STEPS):
        if n % 10 == 0:
            Mnp = sim["Mf"].to_numpy().astype(np.float64)
            Mdnp = sim["Mdf"].to_numpy().astype(np.float64)
        if n == 0:
            # sharp: the first force (P was 0 ⇒ force = P/dt after k_force)
            sim["k_flux"](inv2h)
            sim["k_force"](inv2h, DT)
            sharps["force0"] = sim["Pf"].to_numpy().astype(np.float64) / DT
            sim["red"].fill(0.0)
            sim["k_clamp_sum"]()
            r = sim["red"].to_numpy()
            sim["k_clamp_apply"](float(r[0]) / n_act, float(r[1]) / n_act,
                                 float(r[2]) / n_act)
            sim["cnt"].fill(0)
            sim["k_solve"](inv2h, 1)
            sim["k_update"](DT)
            A0 = sim["A0"].to_numpy().astype(np.float64)
            Q0 = sim["Q0"].to_numpy().astype(np.float64)
            lam_d = sim["lam0"].to_numpy().astype(np.float64)
            # [J1] Jacobi self-check: reconstruction + orthonormality
            rec = np.einsum("...ak,...k,...bk->...ab", Q0, lam_d, Q0)
            j1r = float(np.linalg.norm(rec - A0) / (np.linalg.norm(A0) + 1e-30))
            ortho = np.einsum("...ak,...al->...kl", Q0, Q0)
            j1o = float(np.abs(ortho - np.eye(10)).max())
            j1_tol = 1e-10 if fp == ti.f64 else 5e-4
            j1 = j1r < j1_tol and j1o < (1e-10 if fp == ti.f64 else 1e-4)
            print(f"   [J1] Jacobi self-check: ‖QΛQᵀ−A‖/‖A‖={j1r:.2e} "
                  f"‖QᵀQ−I‖∞={j1o:.2e} → {'PASS' if j1 else 'FAIL'}")
            ok = ok and j1
            sharps["A0"] = A0
            # Jacobi's diagonal is UNSORTED — reorder λ AND keep into ascending-λ
            # order so they compare 1:1 against numpy eigh's convention
            order = np.argsort(lam_d, axis=-1)
            sharps["lam0"] = np.take_along_axis(lam_d, order, axis=-1)
            sharps["keep0"] = np.take_along_axis(
                sim["keep0"].to_numpy().astype(np.uint8), order, axis=-1)
            sharps["Md0"] = sim["Mdf"].to_numpy().astype(np.float64)
            sharps["j1_rec"] = j1r
            sharps["j1_ortho"] = j1o
        else:
            step(sim, inv2h, n_act)
        if n % 10 == 0:
            Mpost = sim["Mf"].to_numpy().astype(np.float64)
            # H from the PRE-step (M, Ṁ_prev) pair — the rhs() T,U semantics
            _, _, _, T, U = fluxes(Mnp, Mdnp, h)
            Hs.append(float((T + U)[act].sum()) * h**3)
            s, al, es = probe_state(Mpost, M0, Mth, act, core, rhat)
            ssig.append(s)
            alens.append(al)
            espread.append(es)
            if not np.isfinite(Mpost).all() or np.abs(Mpost).max() > 1e3:
                print(f"   DIVERGED at step {n}")
                ok = False
                break
        if n in CHECK_NS:
            checks[f"M_{n}"] = sim["Mf"].to_numpy().astype(np.float64)
        if n % 100 == 99:
            c = sim["cnt"].to_numpy()
            print(f"   step {n + 1}/{RUN_STEPS} [{time.time() - t0:.0f}s] "
                  f"H={Hs[-1]:.4f} align={alens[-1]:.3f} "
                  f"frozen={100 * c[1] / (N**3 * 10):.0f}% vcap-hits={c[0]}")
    wall = time.time() - t0
    np.savez(npz_path(tag),
             Hs=np.array(Hs), s=np.array(ssig), align=np.array(alens),
             espread=np.array(espread), check_ns=np.array(CHECK_NS),
             ok=ok, wall=wall, ms_step=1e3 * wall / max(1, RUN_STEPS),
             **sharps, **checks)
    print(f"   done ok={ok} [{wall:.0f}s, {1e3 * wall / RUN_STEPS:.0f} ms/step]"
          f" → {npz_path(tag).name}")
    return 0 if ok else 1


# ════════════════════════════════════════════════════════════════════════════
# compare — the B-gates
# ════════════════════════════════════════════════════════════════════════════


def relF(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    return float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-300))


def sharp_block(a, b, name, tA, tlam, tkeep, tMd, tM1):
    """Step-0 sharp comparisons between npz dicts a (test) and b (base)."""
    rA = relF(a["A0"], b["A0"])
    dl = np.abs(a["lam0"] - b["lam0"])
    lmax = np.abs(b["lam0"]).max(axis=-1, keepdims=True) + 1e-30
    rl = float((dl / lmax).max())
    rk = float((a["keep0"] != b["keep0"]).mean())
    rMd = relF(a["Md0"], b["Md0"])
    rM1 = relF(a["M_0"], b["M_0"])
    rF = relF(a["force0"], b["force0"])
    gates = [rA < tA, rl < tlam, rk < tkeep, rMd < tMd, rM1 < tM1, rF < tA]
    print(f"   [{name}] sharps: A0={rA:.2e}(<{tA:.0e}) λ0={rl:.2e}(<{tlam:.0e})"
          f" keep0-mism={100 * rk:.3f}%(<{100 * tkeep:.1f}%) "
          f"force0={rF:.2e}(<{tA:.0e})")
    print(f"   [{name}]         Md0={rMd:.2e}(<{tMd:.0e}) "
          f"M@1step={rM1:.2e}(<{tM1:.0e}) → "
          f"{'PASS' if all(gates) else 'FAIL'}")
    return all(gates)


def traj_block(a, b, name, tH_early, t_align, t_espread):
    """Trajectory: early-horizon H agreement + gate-level finals."""
    m = min(len(a["Hs"]), len(b["Hs"]))
    ne = min(10, m)                                  # first 100 steps
    rH = float(np.abs(a["Hs"][:ne] - b["Hs"][:ne]).max()
               / (np.abs(b["Hs"][:ne]).max() + 1e-30))
    rH_all = float(np.abs(a["Hs"][:m] - b["Hs"][:m]).max()
                   / (np.abs(b["Hs"][:m]).max() + 1e-30))
    da = abs(float(a["align"][-1]) - float(b["align"][-1]))
    de = abs(float(a["espread"][-1]) - float(b["espread"][-1]))
    cks = [f"M_{k}" for k in a["check_ns"] if f"M_{k}" in b]
    cstr = " ".join(f"@{k.split('_')[1]}:{relF(a[k], b[k]):.1e}" for k in cks)
    gates = [bool(a["ok"]), rH < tH_early, da < t_align, de < t_espread]
    print(f"   [{name}] traj: ok={bool(a['ok'])} H-early={rH:.2e}(<{tH_early:.0e})"
          f" H-full={rH_all:.2e} Δalign={da:.4f}(<{t_align}) "
          f"Δcore-frac={de:.3f}(<{t_espread})")
    print(f"   [{name}]       field rel-diff {cstr}")
    return all(gates)


def run_compare():
    print("=" * 78)
    print("[compare] Option B gates — implementation / precision / backend / physics")
    print("=" * 78)
    runs = {}
    for tag in ("ref", "ti_cpu_f64", "ti_cpu_f32", "ti_metal_f32"):
        f = npz_path(tag)
        if not f.exists():
            print(f"   MISSING {f.name} — run that mode first")
            return 1
        runs[tag] = dict(np.load(f))
    ref, t64, t32, tm32 = (runs["ref"], runs["ti_cpu_f64"],
                           runs["ti_cpu_f32"], runs["ti_metal_f32"])

    print("\n[B1] ti-cpu-f64 vs numpy-f64 — implementation fidelity:")
    b1 = sharp_block(t64, ref, "B1", 1e-11, 1e-8, 1e-3, 1e-7, 1e-9)
    b1 = traj_block(t64, ref, "B1", 1e-6, 0.005, 0.01) and b1

    print("\n[B2] ti-cpu-f32 vs ti-cpu-f64 — the f32-precision question:")
    b2 = sharp_block(t32, t64, "B2", 5e-5, 5e-4, 0.02, 5e-3, 5e-5)
    b2 = traj_block(t32, t64, "B2", 1e-3, 0.02, 0.05) and b2

    print("\n[B2m] ti-metal-f32 vs ti-cpu-f32 — Metal backend:")
    b2m = sharp_block(tm32, t32, "B2m", 1e-5, 1e-4, 0.01, 1e-2, 1e-4)
    b2m = traj_block(tm32, t32, "B2m", 1e-3, 0.02, 0.05) and b2m

    print("\n[B3] ti-metal-f32 vs ref finals — physics on the production target:")
    drift_r = float((ref["Hs"].max() - ref["Hs"].min())
                    / (abs(ref["Hs"][0]) + 1e-30))
    drift_m = float((tm32["Hs"].max() - tm32["Hs"].min())
                    / (abs(tm32["Hs"][0]) + 1e-30))
    da = abs(float(tm32["align"][-1]) - float(ref["align"][-1]))
    de = abs(float(tm32["espread"][-1]) - float(ref["espread"][-1]))
    b3 = (bool(tm32["ok"]) and abs(drift_m - drift_r) < 0.05
          and da < 0.02 and de < 0.05)
    print(f"   [B3] ok={bool(tm32['ok'])} H-drift ref={100 * drift_r:.1f}% "
          f"metal={100 * drift_m:.1f}% Δalign={da:.4f} Δcore-frac={de:.3f} → "
          f"{'PASS' if b3 else 'FAIL'}")
    print(f"   finals: align ref={float(ref['align'][-1]):.3f} "
          f"metal={float(tm32['align'][-1]):.3f} | core-frac "
          f"ref={float(ref['espread'][-1]):.3f} "
          f"metal={float(tm32['espread'][-1]):.3f} | "
          f"|s|max ref={np.abs(ref['s']).max():.2e} "
          f"metal={np.abs(tm32['s']).max():.2e}")
    print(f"   ms/step (24³): numpy={float(ref['ms_step']):.0f} "
          f"cpu-f64={float(t64['ms_step']):.0f} "
          f"cpu-f32={float(t32['ms_step']):.0f} "
          f"metal-f32={float(tm32['ms_step']):.0f}")

    ok = b1 and b2 and b2m and b3
    print("\n" + "=" * 78)
    print(f"OPTION B SPIKE: {'PASS' if ok else 'FAIL — inspect the gate above'} "
          f"(B1={b1} B2={b2} B2m={b2m} B3={b3})")
    print("=" * 78)
    return 0 if ok else 1


# ════════════════════════════════════════════════════════════════════════════
# perf — [B4] 64³ Metal f32 per-kernel step cost (report-only)
# ════════════════════════════════════════════════════════════════════════════


def build_grid_n(n, box):
    """The 2c-1 build_grid, parametric in n (for the 64³ perf seed)."""
    xs = np.linspace(-box, box, n)
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
    O4[..., 1:4, 1:4] = O3   # index-0: spatial block in axes {1,2,3}, time = axis 0
    O4[..., 0, 0] = 1.0
    return dict(h=h, r=r, rho=rho, rhat=rhat, O4=O4)


def run_perf():
    import taichi as ti
    ti.init(arch=ti.metal, default_fp=ti.f32)
    n = PERF_N
    print("=" * 78)
    print(f"[B4 perf] {n}³ Metal f32 — per-kernel constrained-step cost")
    print("=" * 78)
    g = build_grid_n(n, L)
    h = g["h"]
    w = np.exp(-((g["r"] / R_W) ** 2))
    W = matmul(g["O4"], boost_field(B_STAR * w, A_BOOST))
    M0 = conj(W, D4)
    inter = np.zeros(g["r"].shape, bool)
    inter[2:-2, 2:-2, 2:-2] = True
    act = inter & (g["r"] > 2 * RC) & (g["rho"] > RHOC)
    n_act = int(act.sum())
    n_act_pl = int(act[n // 2].sum() + act[:, n // 2].sum() + act[:, :, n // 2].sum())
    sim = build_sim(ti, ti.f32, n, with_diag=False)
    sim["Mf"].from_numpy(M0.astype(np.float32))
    sim["Mdf"].fill(0.0)
    sim["Pf"].fill(0.0)
    sim["actf"].from_numpy(act.astype(np.int32))
    sim["Bas"].from_numpy(SYM_BASIS.astype(np.float32))
    inv2h = 1.0 / (2.0 * h)
    for _ in range(3):                                  # warmup + compile
        step(sim, inv2h, n_act)
    ti.sync()
    ph = dict(flux=0.0, force=0.0, clamp=0.0, solve=0.0, update=0.0)
    for _ in range(PERF_STEPS):
        t = time.time()
        sim["k_flux"](inv2h)
        ti.sync()
        ph["flux"] += time.time() - t
        t = time.time()
        sim["k_force"](inv2h, DT)
        ti.sync()
        ph["force"] += time.time() - t
        t = time.time()
        sim["red"].fill(0.0)
        sim["k_clamp_sum_planes"]()
        r = sim["red"].to_numpy()
        sim["k_clamp_apply"](float(r[0]) / n_act_pl, float(r[1]) / n_act_pl,
                             float(r[2]) / n_act_pl)
        ti.sync()
        ph["clamp"] += time.time() - t
        t = time.time()
        sim["cnt"].fill(0)
        sim["k_solve"](inv2h, 0)
        ti.sync()
        ph["solve"] += time.time() - t
        t = time.time()
        sim["k_update"](DT)
        ti.sync()
        ph["update"] += time.time() - t
    # the documented full-grid atomic stall, measured separately (NOT in the
    # production path — the timed loop above uses the 3-plane pattern)
    t = time.time()
    for _ in range(3):
        sim["red"].fill(0.0)
        sim["k_clamp_sum"]()
        ti.sync()
    t_atomic = (time.time() - t) / 3 * 1e3
    Mn = sim["Mf"].to_numpy()
    sane = bool(np.isfinite(Mn).all() and np.abs(Mn).max() < 1e3)
    tot = sum(ph.values()) / PERF_STEPS * 1e3
    rest = (sum(ph.values()) - ph["solve"]) / PERF_STEPS * 1e3
    print(f"\n   per-step phase cost ({PERF_STEPS} timed steps, ms):")
    for k_, v in ph.items():
        note = " (3-plane sampling, the production pattern)" if k_ == "clamp" else ""
        print(f"      {k_:7s} {1e3 * v / PERF_STEPS:9.1f}{note}")
    print(f"      total   {tot:9.1f}  →  {1e3 / tot:.1f} steps/s")
    print(f"   solve / rest-of-step ratio: {ph['solve'] / PERF_STEPS * 1e3:.1f} / "
          f"{rest:.1f} = {ph['solve'] * 1e3 / PERF_STEPS / (rest + 1e-30):.1f}×")
    print(f"   full-grid atomic clamp alternative: {t_atomic:.0f} ms — the known "
          f"Metal stall; NOT for production")
    print(f"   {PERF_STEPS}-step field sanity: {'finite/bounded' if sane else 'NOT SANE'}")
    return 0 if sane else 1


# ════════════════════════════════════════════════════════════════════════════


def run_all():
    script = str(Path(__file__).resolve())
    stages = [["ref"], ["ti", "cpu", "f64"], ["ti", "cpu", "f32"],
              ["ti", "metal", "f32"], ["compare"], ["perf"]]
    rc_compare = 1
    for st in stages:
        print(f"\n>>> stage: {' '.join(st)}")
        rc = subprocess.run([sys.executable, "-u", script] + st).returncode
        if st == ["compare"]:
            rc_compare = rc
        elif rc != 0 and st != ["perf"]:
            print(f"!!! stage {' '.join(st)} failed (rc={rc}) — aborting chain")
            return rc
    return rc_compare


if __name__ == "__main__":
    a = sys.argv[1:]
    mode = a[0] if a else "all"
    if mode == "ref":
        raise SystemExit(run_ref())
    elif mode == "ti":
        raise SystemExit(run_ti(a[1], a[2]))
    elif mode == "compare":
        raise SystemExit(run_compare())
    elif mode == "perf":
        raise SystemExit(run_perf())
    elif mode == "all":
        raise SystemExit(run_all())
    else:
        print(__doc__)
        raise SystemExit(2)
