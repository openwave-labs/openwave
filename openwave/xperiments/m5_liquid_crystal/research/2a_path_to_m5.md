# PATH TO M5 — LIQUID-CRYSTAL MODEL

Implementation plan for **M5 / LIQUID-CRYSTAL MODEL** (directory `openwave/xperiments/m5_liquid_crystal/`), the production field engine that graduates sandbox-validated Lagrangian / topological physics onto the GPU-accelerated OpenWave platform. This document references concrete code in the current engines and defines what M5 inherits, replaces, and adds.

**Naming**: **M5 / LIQUID-CRYSTAL MODEL**. History — originally "Lagrangian-Wave Model", renamed 2026-04-19 to "Lagrangian-Field Model", renamed again 2026-05-15 to "Liquid-Crystal Model". The current name identifies the **framework** the method implements (the Landau-de Gennes liquid-crystal-based particle framework of Duda's `arxiv:2108.07896` v7) rather than the mathematical formalism. This matters because M6 (Ouroboros, Werbos's chaoiton framework) is also a Lagrangian field theory — calling M5 "Lagrangian Field" would conflict. Methods are named after the candidate framework they implement (M3 = Wolff-LaFreniere, M5 = Liquid-Crystal, M6 = Ouroboros), not after the math they share. The Python module stays as `lagrangian_engine.py` because the module name describes *what the code does* (integrates a Lagrangian-derived PDE `∂²_tψ = c²∇²ψ − ∂V/∂ψ` on the matrix field `M = ODO^T`), which is still accurate. See [3b § What wave equation does M5 solve?](_overview.md#what-wave-equation-does-m5-solve-is-force-still-e) for the engine architecture.

**Spec inputs**:

- [1a_lagrangian_framework.md](1a_lagrangian_framework.md) — Lagrangian framework evaluation, 8 experiments, Duda/Close context
- [1c_lagrangian_experiments.md](1c_lagrangian_experiments.md) — sandbox numerical results (experiment-by-experiment)
- [0_WAVE_EQUATION.md](../../m3_wolff_lafreniere/research/0_WAVE_EQUATION.md) — the M2/M3 vs Lagrangian comparison and why the equation is the *consequence*, not the goal

**Production code references**:

- `openwave/xperiments/m2_free_wave/wave_engine.py` — existing PDE-stepping engine (free-wave, scalar)
- `openwave/xperiments/m4_ewt/wave_engine.py` — existing analytical-superposition engine (vector, WPSW)

---

## WHY M5 IS A NEW ENGINE, NOT AN M4 UPGRADE

M4's architecture is **analytical superposition**:

- Each voxel evaluates the closed-form Weighted Partial Standing Wave directly (`m4_ewt/wave_engine.py:197–211`)
- Total field = linear sum over active WCs (`wave_engine.py:209`, `+= oscillator * direction`)
- Each WC is a query parameter (position + phase offset) that every voxel reads each frame
- Energy is postulated: `E = ρ·V·(f·A)²` from EMA on RMS amplitude (`wave_engine.py:287–293`)

This pattern has a **fundamental incompatibility** with the Lagrangian framework:

> Linear superposition `ψ = Σ ψ_i` only works when the wave equation is linear. Any nonlinear potential V(ψ) (Smolinski `−k·ψ³`, Sine-Gordon `sin(φ)`, Landau-de Gennes, Close's vector equation) destroys superposition. The field must be evolved *as a whole*, not composed from per-WC contributions.

Further, topology requires the field to carry **winding information**. In M4, the field at each voxel is reconstructed fresh every frame from WC positions — the field has no memory of its own topological state. A topological defect (hedgehog, kink, knot) is a property of the *field configuration*, which M4 doesn't persist.

M5 therefore replaces the core physics kernel while keeping the production infrastructure (Taichi, grid, visualization, force motion, GUI).

---

## WHAT M5 INHERITS FROM M2

M2 (`m2_free_wave/wave_engine.py`) already solves the **free-wave equation** via leapfrog time-stepping on a 3D grid. It is, in essence, a linear-Lagrangian PDE engine — the simplest case of what M5 becomes. The following patterns port directly to M5:

| M2 element | Location | Reuse in M5 |
| --- | --- | --- |
| 6-point discrete Laplacian | `compute_laplacianL`, `wave_engine.py:527–562` | Same stencil, computes `∇²ψ` per component (x, y, z in vector case) |
| Triple-buffer pattern (`psi_prev`, `psi`, `psi_new`) | Field allocation in `WaveField` + update at `wave_engine.py:649–660` + swap at line 743 | Same buffering needed for leapfrog; extended to vector component by component |
| Leapfrog update formula `ψ_new = 2ψ − ψ_prev + (c·dt)²·∇²ψ` | `wave_engine.py:649–653` | Replace RHS with `(c·dt)²·∇²ψ − dt²·dV/dψ` (add nonlinear potential term) |
| Dirichlet BC via skipped boundary voxels | `wave_engine.py:641` (`ndrange((1, nx-1), …)`) | Same approach for clamping ψ=0 at edges, or absorbing layer (commented block at lines 662–699 is a partial PML start) |
| Zero-crossing frequency detection with EMA | `wave_engine.py:724–741` | Transfers directly for per-voxel frequency tracking |
| EMA on ψ² for RMS amplitude | `wave_engine.py:711–722` | Same tracker — one per vector component or per scalar field |
| `charge_*` initial-condition kernels (Gaussian, 1/r falloff, radial sinusoidal, 1λ bubble) | `wave_engine.py:30–257` | Pattern is directly applicable, but **renamed in M5 to `seed_*`** (see note below) — kernels like `seed_hedgehog`, `seed_sine_gordon_kink` configure the initial topological state instead of a spherical energy pulse |
| CFL-aware timestep via `c_amrs * dt_rs` | `wave_engine.py:606–607, 652` | Same dt budgeting |

**Key conceptual point**: M2 is to a linear Lagrangian what M5 is to a full Lagrangian. The only physics difference is the absence of the `V(ψ) ≠ 0` term. M5 is "M2 with teeth" — nonlinear potential, vector field, topological ICs.

---

## WHAT M5 INHERITS FROM M4

M4's vector-field infrastructure and production polish transfer wholesale:

| M4 element | Location | Reuse in M5 |
| --- | --- | --- |
| Vector-valued displacement (3 components per voxel) | `displacement_am[i,j,k]` as `ti.Vector([0,0,0])`, `wave_engine.py:152` | Same — but M5 evolves each component via independent PDE stepping (no longer constrained to `* direction` radial projection at line 211) |
| Flux-mesh visualization (XY/XZ/YZ planes, colormaps) | `update_flux_mesh_values`, `wave_engine.py:544–703` | Unchanged — visualizes any scalar tracker regardless of source |
| Granule position rendering from displacement | `sample_position_to_render`, `wave_engine.py:379–403` | Unchanged |
| 3-plane sampling for global averages | `sample_avg_trackers`, `wave_engine.py:410–536` | Unchanged — computes RMS / freq / energy averages |
| WC position + phase offset + active flag tracking | `WaveCenter` data class (referenced at `wave_engine.py:52–58`) | **Role flips**: WC is no longer a query parameter but a *tracked measurement* of where a topological defect sits in the field |
| Force = −∇E via neighbor sampling | `select_voxels`, `wave_engine.py:31–101` (and downstream in `force_motion.py`) | Keep the gradient computation; the *source* of E changes (now Hamiltonian density, not postulated `ρV(fA)²`) |
| GUI, xperiments scaffold, scale_factor, constants | throughout | Unchanged |

**What M4 gives up**: the analytical WPSW kernel and the selective-voxel shortcut (`select_voxels`, `propagate_wave_neighbors`). The PDE is spatially coupled (neighbors via Laplacian), so voxel skipping is no longer valid — M5 runs full grid, same as M2 does today.

---

## WHAT M5 ADDS (NEW PHYSICS)

Everything below is net-new infrastructure, gated on sandbox experiment results.

### 1. Nonlinear potential term in the EoM

The M5 leapfrog step becomes:

```text
ψ_new = 2·ψ − ψ_prev + dt²·[c²·∇²ψ − ∂V/∂ψ]
```

The specific `V(ψ)` is selected from sandbox results:

| Candidate | Potential term | Status |
| --- | --- | --- |
| Sine-Gordon family | `V(φ) = (m²c⁴/ℏ²)(1 − cos φ)` → `∂V/∂φ = (m²c⁴/ℏ²) sin(φ)` | ✅ validated (Exp 1) — useful for 1D analogs, not 3D hedgehogs |
| Smolinski Ψ³ | `V(ψ) = (k/4)·ψ⁴` → `∂V/∂ψ = k·ψ³` | ❌ falsified for K-selectivity (Exp 8); usable only as in-configuration stabilizer |
| Landau-de Gennes | `V(M) = a·Tr(M²) − b·Tr(M³) + c·(Tr M²)²` | ⚠️ mechanism validated (Exp 6); specific parameters deferred to M5.6 |
| Close's elastic solid | Eq. 19 `∂²Q = −c²·∇×∇×Q` (linear limit) + **Eq. 23** (particle equation, preserves `∇·s = 0`, Close's recommendation 2026-04-18) — optional Eq. 21 nonlinear terms `−u·∇s + w×s` for comparison | ✅ Eq. 19 validated (Exp 7 v2); Eq. 23 selected per Close's explicit guidance — **M5's base wave dynamics layer** |
| Klein-Gordon mass term | `V(ψ) = ½m²·ψ²` → `∂V/∂ψ = m²·ψ` | ✅ validated (Exp 4) — **selected as M5's perturbation-mass mechanism** |
| Mixed / composite | linear combination (topology seeding + Close dynamics + KG mass + optional Skyrme) | **adopted** — full recipe in [3a § Winning Approach](1c_lagrangian_experiments.md#winning-approach-for-m5) |

### 2. True independent vector components (no radial constraint)

M4 collapses its vector to radial (`oscillator * direction`, line 211). M5 evolves each of `(ψ_x, ψ_y, ψ_z)` independently via its own PDE step. No radial projection; topology emerges from the full vector field.

**Storage**: either three coupled scalar fields, or one `ti.Vector.field(3, dtype=ti.f32, shape=(nx,ny,nz))` with a triple buffer (`psi_prev`, `psi`, `psi_new`).

### 3. Director field extraction (for topology)

Optional dedicated field `n(i,j,k) = unit vector` extracted either:

- From the major axis of the local displacement ellipse (the 6-phasor data M4 already implicitly carries), or
- From a separate director field evolved in tandem (LdG style)

This is the field whose winding integer = topological charge.

### 4. Topological initial-condition kernels

New `seed_*` kernels in the M2 style (renamed from M2's `charge_*` to better match IC formalism — these configure the field's topological state, not inject energy). Every scenario is a composition: **first `seed_vacuum` to fill the grid with the Lagrangian's ground state, then `seed_<defect>` to overwrite a localized region with the topological feature**:

- `seed_vacuum(n0)` → fill entire grid with ground state (e.g. `n(x) = ẑ` for a uniaxial vacuum, or `ψ = 0` for a scalar field). **Always the first step** — topology requires a reference state to wind around
- `seed_hedgehog(center, sign)` → `n(x) = ±(x − center)/|x − center|` near center (Experiment 2 ported to production)
- `seed_sine_gordon_kink(position, velocity)` → 1D / axisymmetric kink seeding (Experiment 1 ported to production)
- `seed_biaxial_hedgehog(center, axis)` → three-mass-scale lepton-family seeding (Experiment 6)
- `seed_spherical_harmonic(l, m)` → Close's spherical-harmonic seed for Experiment 7

See the **[Architectural Decision: Background Vacuum + Defects](#architectural-decision-background-vacuum--defects-m2-philosophy)** section below for why the vacuum step is mandatory and how it differs from M2's oscillating base wave.

**Naming note**: M2's kernels are called `charge_*` (e.g. `charge_gaussian`, `charge_falloff`) because they inject an energy pulse into the field — "charging" it up. M5's kernels don't inject energy; they configure the initial *topological configuration* (which way the director points at each voxel, which kink profile fills the domain). The right verb is **seed** (same as random-number generators, initial-condition generators in PDE literature). Keeping `charge_*` would be misleading — "seeding a hedgehog" is accurate; "charging a hedgehog" is not.

### 5. Hamiltonian energy density

Replace `E = ρV(fA)²` with the Hamiltonian density derived from the Lagrangian:

```text
H(i,j,k) = ½|ψ_dot|²  +  ½c²|∇ψ|²  +  V(ψ)
            kinetic     gradient     potential
```

where `ψ_dot = (ψ − ψ_prev)/dt`. This energy is conserved by Noether's theorem (barring numerical drift), and its gradient gives force exactly the same way M4 does today (`F = −∇E`). The `rho_qgam * dx³ * (f·A)²` postulate becomes an *approximation* that holds in the linear limit.

### 6. Winding-number tracker

New diagnostic: compute topological charge `Q = (1/4π) ∮_S (∂_u n × ∂_v n) · n du dv` on spheres around each tracked WC location, as per Experiment 3. Returns integer values; becomes a new tracker alongside `amp_local_emarms_am`, `freq_local_cross_rHz`, `energy_local_aJ`.

---

## ARCHITECTURAL DECISION: BACKGROUND VACUUM + DEFECTS (M2-PHILOSOPHY)

A question that has to be settled before any kernel is written: **what is the state of M5's grid at t=0 when nothing is happening?**

OpenWave's existing methods answer this differently:

- **M2** — the field is a self-stabilized standing wave throughout the grid. "Emptiness" is still a wave field in equilibrium. WCs are disturbances *in* this field
- **M3** — the field is literally zero everywhere except where WCs emit. "Emptiness" is nothing; each WC generates its own wave into the void
- **M4** — same as M3 (analytical, zero background), just with a vector rather than a scalar output

**M5 must commit to M2's philosophy** — the field exists as a **background vacuum state** that is the ground state of the Lagrangian's potential V(ψ). A topological defect (hedgehog, kink) is a *deformation of this vacuum*, and the winding number is measured *relative to the vacuum orientation*. This is a hard mathematical requirement, not a stylistic preference:

> **A winding number only makes sense against a reference state.** If there is no vacuum field to wind around, there is no topological charge to measure. M3's "WCs emit into void" approach is structurally incompatible with topology.

### Static Vacuum vs. Oscillating Base Wave

Duda's vacuum is **not** the same as M2's base wave. The distinction:

| Aspect | M2 base wave (current) | M5 vacuum (planned) |
| --- | --- | --- |
| Field state when "empty" | `ψ_base = A₀·cos(kx)·cos(ωt)` — oscillating | `n(x) = ẑ` — static, aligned everywhere |
| What causes oscillation | Assumed intrinsic (all matter in universe, EWT) | Emerges from defect-vacuum interaction (time crystal, phi-4 mechanism) |
| Time dependence | Built in (ω·t) | None — oscillations arise from the dynamics, not the ground state |
| Mathematical role | Initial condition + driving term | Minimum of V(ψ) — the potential well |

So M5's vacuum is **more fundamental** than M2's base wave: it explains *why* fields oscillate (defects oscillate because they have mass and interact with V(ψ)) rather than assuming the universe comes pre-oscillating. See [1a_lagrangian_framework.md § Impact on Base Wave Architecture](1a_lagrangian_framework.md#impact-on-base-wave-architecture--m2-vs-m3-philosophy) for the full comparison and how this resolves Duda's time-crystal insight.

### Concrete Consequence for M5 `seed_*` Kernels

Every M5 initial-condition kernel follows a two-step pattern:

```text
Step 1: seed_vacuum(n0)            # fill the entire grid with the ground state
                                     (e.g. n(x) = ẑ everywhere, or ψ = 0)

Step 2: seed_<defect>(center, ...)  # overwrite a localized region with the defect
                                     (hedgehog, kink, spherical harmonic, etc.)
```

This mirrors how the sandbox experiments seed their fields (a hedgehog is "uniform director + radial override near the center"). The vacuum step is separate and shared — it's what Duda's framework requires to make topology measurable.

### Why This Does NOT Invalidate M3's Results

M3's lock-in, annihilation, and K=10 stability are real near-field wave physics — they happen regardless of whether the far field is modeled as vacuum or void. What M3 cannot do, because of its "no background" philosophy, is measure topological charge or produce far-field Coulomb without sinc barriers. M5 keeps M3's near-field physics as validated phenomena and adds the background-vacuum architecture where it matters (topology, charge quantization, far-field interactions). Cf. [1a_lagrangian_framework.md § Practical Implication](1a_lagrangian_framework.md).

---

## CONCRETE CODE MAPPING

Translating M4's `compute_voxel_wave` → M5's `step_voxel_pde`:

| M4 current | M5 equivalent | Delta |
| --- | --- | --- |
| Loop over WCs, evaluate closed-form per-WC (`wave_engine.py:161–211`) | No WC loop — voxel reads only its own `ψ_prev`, `ψ`, and 6 neighbors for Laplacian | Replace analytical eval with leapfrog step |
| `oscillator * direction` summation | `∇²ψ` via 6-point stencil (port from M2 `compute_laplacianL`, `wave_engine.py:549–560`) | Nonlocal via neighbors |
| N/A (linear wave) | `+ V'(ψ)` nonlinear term | New |
| `displacement_am[i,j,k] = A·oscillator·direction` (analytical result) | `psi_new[i,j,k] = 2·psi − psi_prev + dt²·(c²·∇²ψ − V'(ψ))` | PDE step replaces direct evaluation |
| EMA on RMS amplitude (`wave_engine.py:244–255`) | **Same** — keep as tracker, but not as energy source | Unchanged |
| `E = ρV(fA)²` postulate (`wave_engine.py:287–293`) | `H = ½·ψ̇² + ½c²·(∇ψ)² + V(ψ)` Hamiltonian density | New computation, same scalar output |

Buffer layout change:

```text
M4 (current):
    displacement_am[nx,ny,nz]     ← single Vector(3) buffer, overwritten each frame

M5 (target):
    psi_prev[nx,ny,nz]            ← Vector(3) at t−dt
    psi[nx,ny,nz]                 ← Vector(3) at t (current)
    psi_new[nx,ny,nz]             ← Vector(3) at t+dt (computed)
    n_director[nx,ny,nz]          ← Vector(3) optional (for topology)
    # Swap at end of each step, same pattern as M2 wave_engine.py:743
```

---

## SANDBOX VERDICTS (all 8 experiments complete)

M5 sandbox is complete (2026-04-16 / 2026-04-17). The architectural decisions for M5 are now resolved:

| Decision | Resolution | Source |
| --- | --- | --- |
| Does nonlinearity alone give K-selectivity? | ❌ **No** — Smolinski Ψ³ produces breathing oscillation only, no K-dependence | Exp 8 |
| Does topology alone give clean 1/r Coulomb? | ✅ **Yes** — R² = 0.993 post-relax, no sinc | Exp 2 |
| Does topology give integer charge quantization under perturbation? | ✅ **Yes** — Q = ±1 surface-independent, robust to 50% noise | Exp 3 |
| Does the biaxial-hedgehog mass hierarchy match lepton ratios? | ⚠️ Mechanism validated (E ∝ K linear); specific ratios achievable by choice of K, not derived. Full Q-tensor derivation = Exp 6.1 (deferred) | Exp 6 |
| Does Close's nonlinear vector equation produce localized solitons? | ⚠️ No static solitons from harmonic seeds — but Close's framework doesn't actually require them. Close's Eq. 19 IS a valid transverse vector wave equation for M5's base dynamics | Exp 7 v2 |
| Can our Combined W-L be derived from a Lagrangian, or must we replace it? | ⚠️ The M4 code's sum form IS a free-wave solution. **The documented product form `2A·sin(kr/2)·cos(kr/2−(ωt+φ))/r` is NOT** — it has a quadrature residual. M5 keeps the sum form, discards the product form | Exp 5 |
| Does the PDE evolution produce Klein-Gordon dispersion? | ✅ **Yes** — ω² = c²k² + m² validated to R² = 0.999982 across 9 modes | Exp 4 |
| Is leapfrog + nonlinear potential stable with Lorentz-correct kinematics? | ✅ **Yes** — kink v = 0.4997c (input 0.9c), width L/γ matched, energy drift 1.5e-6 | Exp 1 |

**Headline finding**: topology (Exps 1, 2, 3) is load-bearing; pure nonlinearity (Exps 7, 8) is insufficient on its own; Klein-Gordon wave dynamics (Exp 4) are the correct perturbative layer. Full context in [1c_lagrangian_experiments.md § OVERALL CONCLUSIONS](1c_lagrangian_experiments.md#overall-conclusions).

---

## IMPLEMENTATION PHASES (post-sandbox, ready to start)

The winning recipe is now known: **topology + Klein-Gordon dynamics + Close's vector wave equation + M3 near-field physics + Skyrme stabilizer + (eventually) LdG biaxial potential**. Phases below are scoped accordingly.

---

## RESOLUTION & PERFORMANCE PLAN

A separate, mandatory engineering plan that determines which phases run on the local M4 Max and which need rented GPU time. M2 / M3 hit a hard ceiling at ~350M voxels because they had to resolve the EWT base wave at `λ₀ = 2.85 × 10⁻¹⁷ m`. M5 changes this fundamentally.

### Why M5's resolution budget is much friendlier than M2/M3's

| Constraint | M2 / M3 | M5 |
| --- | --- | --- |
| Vacuum is | An oscillating base wave at `f₀ = 1.05 × 10²⁵ Hz` | A **static** ground state of `V(ψ)` (no oscillation in vacuum) |
| Grid must resolve | `λ₀ = 2.85 × 10⁻¹⁷ m` (the EWT base wave) | `λ_C = ℏ/(mc)` of the **defect** (its Compton wavelength) |
| Min `dx` (Nyquist ~10–12× / wavelength) | `~3 × 10⁻¹⁸ m` | `~3 × 10⁻¹⁴ m` for an electron — **10⁴× larger** |
| Min `dt` (CFL `dt ≤ dx / (c·√3)`) | `~6 × 10⁻²⁷ s` | `~6 × 10⁻²³ s` for an electron — **10⁴× larger** |
| Reachable particle on M4 Max | Below electron (couldn't reach electron radius) | Electron well in scope; muon tight; tau requires AMR |

In one sentence: **M5 doesn't simulate the EWT vacuum's oscillations — there are none; the vacuum is a static ground state. Only the defect's intrinsic length scale needs to be resolved.**

### Per-particle defect scales (the "what does M5 actually need to resolve" table)

| Particle | Mass | Compton λ_C = ℏ/(mc) | Zitterbewegung ω = 2mc²/ℏ | T_Z = 2π/ω |
| --- | --- | --- | --- | --- |
| **Electron** | 0.511 MeV | **3.86 × 10⁻¹³ m** (~386 fm) | 1.55 × 10²¹ rad/s | 4.05 × 10⁻²¹ s |
| **Muon** | 105.7 MeV | 1.87 × 10⁻¹⁵ m | 3.21 × 10²³ rad/s | 1.96 × 10⁻²³ s |
| **Tau** | 1777 MeV | 1.11 × 10⁻¹⁶ m | 5.39 × 10²⁴ rad/s | 1.17 × 10⁻²⁴ s |
| **Up quark** (constituent ~2.2 MeV) | 2.2 MeV | 8.97 × 10⁻¹⁴ m | 6.7 × 10²¹ rad/s | 9.4 × 10⁻²² s |
| **Proton** (composite) | 938 MeV | 2.10 × 10⁻¹⁶ m | 2.85 × 10²⁴ rad/s | 2.20 × 10⁻²⁴ s |
| **Neutrino** (~0.1 eV) | 0.1 eV | ~2 × 10⁻⁶ m | ~3 × 10¹⁴ rad/s | ~2 × 10⁻¹⁴ s |

### Per-phase grid budgets (target on M4 Max 48 GB / 18.4 TFLOPS)

| Phase | Particle | dx (≈λ_C/12) | Domain (≈8·λ_C) | N (per side) | Voxels | Memory* | Steps for 10 T_Z | Wall-clock estimate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M5.0 / M5.1 / M5.2 (KG dispersion, Coulomb) | n/a (geometric tests) | natural-units `1` | `64` | 256 | 17M | ~1 GB | n/a (relaxation, not Zitterbewegung) | minutes |
| **M5.7 + M5.8** (resonance + electron Zitterbewegung) | electron | 3.2e-14 m | 3.1e-12 m | 256–384 | 17M–57M | 1–4 GB | ~500k steps | **~12–24h** (uniform grid, no AMR) |
| M5.9 (Cornell, light quarks) | u/d quark | ~7.5e-15 m (≈λ_C(u)/12) | ~7e-13 m | 96–128 | 1M–2M | <1 GB | ~50k–500k | hours |
| M5.9 (muon, tau biaxial) | muon, tau | ~1.5e-16 m / ~9e-18 m | tight | uniform grid impractical | — | — | — | **needs AMR** |
| M5.8 (Zitterbewegung table for all particles) | all | per-particle `dx` | per-particle | varies | varies | varies | 100×T_Z | **needs AMR + cloud burst** |

*Memory assumes Vector(3) for Q + 3 buffers (prev/curr/new) + winding/director field + Hamiltonian tracker, ~64 bytes/voxel.

### Performance optimizations to bake into M5.0 from Day 1

> **Profile before optimizing.** Tier 2 + Tier 3 optimizations below are options, not commitments. Apply them only after a profiling pass identifies the actual bottleneck. The default sequence is:
>
> 1. Land **Tier 1** architectural decisions before any kernel is written (these are hard to retrofit).
> 2. After the M5.0 scaffold + the gating physics-invariant test pass, run a **profiling pass**: instrument the leapfrog kernel, swap_buffers, Hamiltonian / tracker computations; measure per-frame wall time at production-relevant grid sizes (256³ baseline, then electron-scale ~384³); identify which operations dominate.
> 3. Apply **Tier 2** optimizations selectively in the order that addresses the measured bottleneck — not the order they appear in this list. (E.g., if the Laplacian dominates, BlockLocal tiling first; if buffer swap dominates, replace `copy_from` with a single Taichi kernel; if trackers dominate, skip-N-frames first.)
> 4. **Tier 3** is for production-scale runs (M5.5+); deferred until profiling shows the simpler tiers are insufficient.
>
> Premature optimization without profiling guidance is the historical failure mode of physics simulators — every optimization complicates the code; only those that target measured bottlenecks justify the complexity.

#### Tier 1 — architectural decisions (fix before any kernel is written)

1. **Native `ti.Vector.field(3, ...)`** with single triple buffer (`psi_prev`, `psi`, `psi_new`) — halves memory traffic vs three independent scalar fields and lets Taichi vectorize 3-component operations
2. **Unit system** — two complementary scalings to keep numerics in float32's sweet spot:
    - **Inherit OpenWave's existing scaled SI** from `openwave/common/constants.py` (lines 10–29): spatial in **attometers** (`1e-18 m`, suffix `_am`) and temporal in **rontoseconds** (`1e-27 s`, suffixes `_rs`, `_amrs`, `_rHz`). Already proven to keep 6–7 significant digits at f32 across M2/M3/M4 — base wavelength `~28.5 am`, wave speed `~0.3 am/rs`, etc. M5 must **continue using these units** for all stored fields and all I/O — don't fork a parallel unit system, don't reinvent precision-protection
    - **Add a separate, internal natural-unit kernel scaling** (`c = 1`, `λ_C(defect-of-interest) = 1`, `ℏ = 1`) for *intra-kernel* arithmetic only. The defect's Compton wavelength is the Lagrangian's natural reference length (electron `λ_C ≈ 386,000 am` is no longer near-1.0 in attometers — leaving f32 sweet spot if used directly in CFL / dispersion / amplitude-sweep math). Conversion happens at kernel entry/exit; no field is ever stored in natural units
    - **Net effect**: data layer stays in proven `_am` / `_rs` units (compatible with all OpenWave tooling, diagnostics, visualization); kernel arithmetic runs in natural units (CFL trivially `dt ≤ dx/√3`, amplitude sweeps `A ∈ {0.25, 0.5, 1, 2}` are O(1), Klein-Gordon mass term is O(1)). Both scalings are needed — neither alone covers M5's precision range across the lepton mass hierarchy `0 < δ ≪ 1 ≪ g`
3. **AMR-ready storage layer** — even if M5.0 ships with a uniform grid, design the field-storage abstraction so a future swap to octree-based AMR (M5.6 / M5.8) doesn't require a rewrite. Don't bake in fixed `(nx, ny, nz)` everywhere
4. **Domain size by defect physics**, not by EWT vacuum — far-field Coulomb cleanly captured at ~8·λ_C, so the box stays small

#### Tier 2 — kernel-level performance hacks

5. **Tile-and-reuse for the 6-point stencil** via Taichi `BlockLocal` / shared-memory loads. The Laplacian is bandwidth-bound, not flop-bound; tiling cuts global-memory traffic ~6×. Free 3–5× speedup
6. **Symplectic / Verlet integrator** instead of vanilla leapfrog. Same cost per step, but ~2–4× larger usable `dt` while preserving energy conservation. **Critical for M5.8** (Zitterbewegung needs energy-clean integration over hundreds of periods)
7. **Operator splitting**: `c²∇²ψ` (cheap, vectorized) and `−∂V/∂ψ` (potentially expensive nonlinear) in separate sub-steps. Each operator individually has better stability → larger overall `dt`
8. **Float32 by default; float64 only for diagnostic snapshots**. 2× speedup + 2× memory headroom; the leapfrog is well-conditioned at f32 in our scale
9. **Skip diagnostic trackers (Hamiltonian + winding) on most frames** — compute every Nth frame (N=10–100). They're observables, not load-bearing on the dynamics
10. **Rotating-pointer `swap_buffers`** — the current `medium.WaveField.swap_buffers()` uses two `field.copy_from()` calls (~5 GB/step memory traffic at 100M voxels, ~25% of per-step total). Replace with three field handles cycled by reference (zero copies). Implementation note: Taichi field handles are immutable references, so the rotation must be Python-side bookkeeping plus a kernel that always writes to a logical "next" buffer chosen via integer modulo. Already flagged in `medium.py:swap_buffers` docstring
11. **Merge `update_trackers` into `evolve_psi`** — currently a SECOND full-grid pass over ψ (~5 GB/step, another ~25% of per-step total). M2's analytical kernel had no neighbor dependency so trackers piggybacked on the propagation loop trivially; M5's leapfrog can do the same — write `psi_new` then immediately use it (with `psi_prev` already in registers) for tracker EMA / zero-crossing in the same iteration. Bonus: gives true central-difference `ψ̇` (more accurate than the forward diff currently used in `update_trackers`). Splitting was deliberate in M5.0d for clarity + AMR-readiness; merge is the right move once profiling confirms the bandwidth win matters
12. **Dirty-tile mask** (the M5 reincarnation of M4's "selective voxel" optimization). Voxels far from any defect are at exactly the vacuum value `ψ_vacuum`; the leapfrog update there is a no-op. Use a chunk-level mask (e.g. 16³ blocks) — skip blocks whose `‖ψ - ψ_vacuum‖` is below epsilon. Realistic 2–10× speedup for sparse single-defect scenarios. Note: unlike M4, this is an *optimization*, not a different physics — a "skipped" block must produce identical output to a computed one (modulo floating-point epsilon)

#### Tier 3 — long-run techniques (deferred to M5.5+ as needed)

13. **Adaptive Mesh Refinement (AMR)** — fine grid near the defect core, coarse grid in the far field. **The single biggest win for M5.6 (muon/tau) and M5.8 (full Zitterbewegung table)**. Cell size scales with local `λ_C`. Realistic 10²–10³× voxel reduction vs uniform fine-grid for a localized defect
14. **Comoving frame for moving defects** — track the defect's center, evolve in its rest frame. Eliminates needing the grid to span the defect's full traveled distance
15. **Multi-scale time stepping** — subcycle small-`dx` (defect core) at fine `dt`, subcycle coarse far-field at large `dt`. Pairs naturally with AMR
16. **Spectral / pseudo-spectral solver for the linear part** (Klein-Gordon) — exact dispersion at any `dx`, FFT-based, ~10× larger usable `dx`. Worth evaluating as an alternative validator for M5.2
17. **Absorbing boundaries / PML** (M2 has a partial start at `wave_engine.py:662–699`). Lets us shrink the domain without reflections — typically ~2× linear / ~8× volume reduction

### Compute strategy — when to leave the M4 Max

The M4 Max is excellent for M5.0–M5.4 (uniform grid, electron-scale runs). The 12–24 hour electron headline run is borderline (long jobs at high GPU utilization risk thermal events; ties up the dev machine for a day). **A laptop is not the right tool for multi-day production runs.**

Per the existing analysis in [`dev_docs/distributed_computing/m4_max_vs_aws.md`](../dev_docs/distributed_computing/m4_max_vs_aws.md) and [`system_upgrade_options.md`](../dev_docs/distributed_computing/system_upgrade_options.md), key constraints:

- **Taichi is single-GPU bound**. Multi-GPU AWS instances (`p3.8xlarge`, `p4d.24xlarge`) only use 1 of N GPUs without an MPI / domain-decomposition rewrite (~2–3 months engineering)
- **Single-GPU AWS instances are barely faster than M4 Max** — V100 is ~1.16× M4 Max FP32 at $3.06/hr; A10G is ~2.3× at $1.01/hr. None are step-change improvements
- **The right cloud play is parameter sweeps**, not single-run speedup. Run 100 amplitude/seed/coupling configurations in parallel on 100 cheap spot instances, finish in the wall-clock time of one
- **The hardware step-change targets are M5 Ultra (2026, ~50–60 TFLOPS, 512 GB) or AMD MI300X (~163 TFLOPS, 192 GB, $15k+)** — both single-GPU, no code rewrite

#### Decision tree per M5 phase

| Phase | Workload | Recommended platform | Rationale |
| --- | --- | --- | --- |
| M5.0 / M5.1 / M5.2 / M5.3 | Small-grid validation, dispersion fits, Coulomb sweep | **M4 Max local** | Minutes-to-hours; tight dev loop is the limiting factor, not compute |
| M5.4 (electron) — single run | 17M–57M voxels, ~500k steps, ~12–24h | **M4 Max overnight, with caveats** | Run when machine isn't needed; monitor for thermal throttling. Pre-validate on small grid first to catch bugs before long run |
| M5.4 — parameter sweep (Y_1^0 amplitude, axis tuning, lifetime characterization) | 4–20 variations × ~6h each | **AWS spot fleet (`g5.xlarge` × N)** | Embarrassingly parallel; total wall-clock = single-run time. Spot pricing (~$0.30/hr discounted) makes a 20-config sweep ~$36 |
| M5.5 (Skyrme) | Conditional; same scale as M5.4 | **Same as M5.4** | — |
| M5.6 (muon, tau, biaxial LdG, Faber regularization) | AMR essential; without it `dx ~ 10⁻¹⁸ m`, voxels astronomical | **AWS A100 (`p4d.24xlarge` 1 GPU)** OR **wait for M5 Ultra (2026)** | Higher-mass particles need AMR + more memory. AMR development on M4 Max, production runs on cloud-or-future-Ultra |
| M5.7 (Cornell potential) | 1D-line topology, lighter than M5.4 | **M4 Max local** | u/d quark scale (~few MeV) is friendlier than muon |
| M5.8 (Zitterbewegung table for 7 particles) | AMR essential; long runs (≥10 T_Z each) for FFT freq extraction | **Cloud or M5 Ultra** | Each particle is a separate run; parallelize across cloud spot instances |

#### Cloud burst recipe (for when local isn't enough)

When a phase requires cloud compute, the playbook is:

1. **Develop and validate on M4 Max** at small scale (256³ or smaller). Confirm correctness end-to-end
2. **Containerize** the run (Taichi + CUDA backend on a Docker image; switch from `ti.metal` to `ti.cuda` is a single-line change)
3. **Use AWS spot instances** for the heavy run. `g5.xlarge` (1× A10G, 24 GB, 31 TFLOPS) at spot pricing ~$0.30/hr is the price/performance sweet spot for OpenWave's single-GPU workloads
4. **Persist results to S3**, pull diagnostics back to M4 Max for analysis. Don't try to render or interactively debug remotely
5. **For parameter sweeps**: AWS Batch or Step Functions launches N spot instances simultaneously, each running one config. Total wall-clock = single-run time

**Estimated cloud spend for the full M5 roadmap**: at spot pricing, the entire M5.4 + M5.7 + M5.8 production run sequence (including parameter sweeps and the 7-particle Zitterbewegung table) is **a few hundred to ~$2,000** total. Not a budget killer, but worth tracking as a separate cost line.

#### Hardware upgrade decision points

| Trigger | Action |
| --- | --- |
| M5.4 production runs become routine (multiple per week) | Evaluate Mac Studio M3 Ultra 80-core / 256–512 GB or wait for M5 Ultra |
| M5.6 / M5.8 require routine production runs and AMR is implemented | Same — single-GPU step change is the right play, not multi-GPU |
| Project moves to fluent multi-particle ensemble simulation (M6.x, M7.x) | Re-evaluate the multi-GPU rewrite; the 2–3 month MPI investment becomes economically viable when single-GPU runs become a regular bottleneck |
| Need >48 GB memory specifically (not compute) | M4 Max 128 GB upgrade (~$4,000) is the cheapest unlock to ~1.5B voxels |

> **Bottom line**: M5.0–M5.4 are M4-Max-feasible with the Tier 1+2 optimizations in place. M5.6 and M5.8 are the phases where AMR is mandatory and cloud burst (or an M5 Ultra) becomes the right tool. Plan for that pivot now so the AMR-ready storage layer is in M5.0's scaffold rather than a retrofit later.

---

> **Per-phase task lists migrated to [`0c_roadmap.md`](0c_roadmap.md) § DETAILED**. This document keeps design rationale, M2/M4 inheritance, code mapping, resolution & performance plan, layered validation, external-comms milestones, and group feedback.

### What M5 does NOT implement

From the sandbox findings, these are ruled out or de-prioritized:

- **The documented Combined W-L product form `2A·sin(kr/2)·cos(kr/2−(ωt+φ))/r`** — Exp 5 showed this isn't a free-wave solution. M5 uses the M4 *sum form* `A·[sin(kr+ωt+φ) + sin(kr−ωt−φ)]/(kr)` if analytical standing waves are needed
- **Smolinski's Ψ³ as primary K-selectivity mechanism** — Exp 8 falsified. Ψ³ may still be a stabilizer *inside* a topologically protected configuration, but M5 does not rely on it for geometric selectivity
- **Seeded-soliton emergence from spherical harmonics as a physics mechanism** — Exp 7 v2 confirmed Close's framework doesn't predict this. Particles in M5 = topological defects (from seed), not emergent solitons from wave seeds

---

## LAYERED VALIDATION ROADMAP — VACUUM TO ATOMS

**OpenWave's distinguishing value**: an **integrated simulator that demonstrates each layer of physics built from previously validated primitives** in a single platform — vacuum → leptons → mesons → nucleons → nuclei → atoms. Each layer depends on the layer below being already validated; cross-layer dependency checks catch inconsistencies between scales that single-layer simulators (QCD-only, atomic-only, liquid-crystal-only) cannot. This integrated cross-scale validation is what makes OpenWave unique.

The full physics hierarchy maps cleanly onto M5 phases:

| Layer | Physics primitive | M5 phase that validates it | Status |
| --- | --- | --- | --- |
| **0 — Vacuum** | LdG ground state, static field configuration | M5.0 (scaffold) + M5.1 (`seed_vacuum`) | ✅ |
| **0' — Static topology / Coulomb** | Hedgehog pair → 1/d Coulomb from Frank elastic | M5.1 (Coulomb + visual confirmation) | ✅ |
| **1a — Matrix-substrate metastable defect** | First long-lived hedgehog on `M = ODO^T` field | M5.4 + M5.5 + M5.6 + M5.7 (resonance hunt) | 🚧 |
| **1b — Intrinsic oscillation (Zitterbewegung)** | de Broglie clock at `ω = 2mc²/ℏ` from 4D Lorentz-signature negative-energy terms | M5.8 (GROUP HEADLINE) | 🚧 |
| **1c — Lepton family hierarchy** | Three biaxial axes `(δ, 1, g)` → e/μ/τ from same Lagrangian | M5.9 | 🚧 |
| **1d — Closed vortex loop** | Neutrino topology variant (SO(3)~SU(2)) | M5.9 (alternative seed) | 🚧 |
| **1e — Two-defect annihilation** | Dynamic Coulomb 1/d² + e⁺/e⁻ annihilation | M5.7+ pair test | 🚧 |
| **2 — Open vortex string** | Quark-antiquark (meson), Cornell potential `V(r) = −α/r + σ·r` with σ ≈ 1 GeV/fm | M5.9 (Cornell) | 🚧 |
| **3 — Baryons (3 string endpoints)** | Proton (uud), neutron (udd) — color-neutral 3-quark composites | 9d (DEFERRED) | not planned |
| **4 — Multi-nucleon nucleus** | Nucleus binding via residual strong force | 9d (DEFERRED) | not planned |
| **5 — Atom** | Z electrons in standing-wave orbital shells around nucleus | 9d (DEFERRED — also needs cross-mass-class machinery) | not planned |
| **6 — Molecules / bulk matter** | Multi-atom bonding | Long-term | not planned |

### Layer-by-layer dependency chain

Each higher layer requires the layer below to be working correctly:

- **Layer 1** (lepton) requires Layer 0 (vacuum that supports topological defects) — checked by M5.0/M5.1 invariant tests (✅)
- **Layer 2** (meson / vortex string) requires Layer 1 (defect machinery + force diagnostics) — checked by M5.9 Cornell-potential validation passing
- **Layer 3** (nucleon) requires Layer 2 (validated string-tension physics + 3-endpoint Y-configuration handling) — gated on M5.9 success
- **Layer 4** (nucleus) requires Layer 3 (working nucleon + residual-force model)
- **Layer 5** (atom) requires Layer 4 (working nucleus) + Layer 1 (working electron) + the standing-wave near-field physics (already validated in M3, retained in M5)

**The cross-layer integration is the unique value**: a successful Layer 5 (atom simulation) provides simultaneous validation that the lepton physics, string-string-tension physics, residual-force binding, and orbital-force standing-wave interference all work *together* — something a single-scale simulator cannot test by construction.

### Why the hierarchy matters for the project's positioning

This integrated layered scope is what differentiates OpenWave from comparable simulators:

| Simulator class | What it covers | What it misses |
| --- | --- | --- |
| **QCD lattice** (mainstream particle physics) | Quark/gluon dynamics, confinement, hadron masses | Leptons, atoms, vacuum-as-LdG-medium, classical interpretability |
| **Liquid-crystal solvers** (LdG mathematics) | Topological defects, hedgehogs, disclinations | Particle-physics calibration, atom-scale dynamics, lepton families |
| **QED / atomic simulators** | Atoms, molecules, EM | Quark structure, lepton ⊃ defect identity, vacuum mechanics |
| **OpenWave (M5 → M6)** | All of the above, **integrated, classical-field-theoretic** | Calibration to high-precision data still in progress (validation against experimental ratios) |

The integration is the value. Reading the layered table top-to-bottom is also a reading-order suggestion for new contributors: validated primitives at the top, frontier work at the bottom.

For the conceptual companion to this roadmap, see [_overview.md § Where do quarks, protons, nuclei, and atoms fit?](_overview.md#where-do-quarks-protons-nuclei-and-atoms-fit) — same hierarchy, framed as a Q&A explanation rather than an implementation checklist.

### Long-term research directions — phase, Berry, and entanglement experiments

A separate research direction prompted by the 2026-04 Models of Particles thread on the Orion–Akkermans paper *"Topological sum rule for geometric phases of quantum gates"* (arxiv:2603.29795). The paper's headline corollary is that nontrivial Hamiltonian topology (`ν_H ≠ 0`) is a **necessary condition for quantum entanglement** — and M5's defect framework satisfies it by construction (every defect has nonzero winding). This opens up a class of experiments that test whether M5's twist degree of freedom (see [1b_topological_defect.md § The twist degree of freedom](1b_topological_defect.md#the-twist-degree-of-freedom--quantum-phase-as-a-derived-field-state)) reproduces the geometric-phase / entanglement structure of standard QM.

These are exploratory targets, not committed milestones. They naturally slot in after M5.7 (resonance hunt) provides the validated metastable-defect substrate and M5.9 (Cornell quark strings) provides validated string-defect dynamics. Prioritized below composites (9d).

| Phase candidate | Headline test | Layered on |
| --- | --- | --- |
| **Single-defect Berry phase** | Drive a single defect adiabatically along a closed loop in parameter space (e.g., rotate the LdG axis assignment). Measure the accumulated twist phase around the loop. Should equal the geometric solid angle × topological factor — the Berry phase formula falls out of the field's own dynamics, not as a postulate | M5.4 (single biaxial hedgehog stable) + M5.6 (axis-rotation machinery) |
| **Hydrodynamic Aharonov-Bohm analog** | Seed a topological vortex string (M5.7 setup); send a plane wave past it; measure the phase shift on the far side as a function of impact parameter. Should reproduce the Berry-1980 hydrodynamic AB result and the recent 2026 reproduction (phys.org/news/2026-04-simulation-famous-quantum-effect-reveals.html) | M5.7 (vortex string seeding) |
| **Two-defect concurrence analog** | Seed a positronium-like e⁺/e⁻ pair (M5.4 setup). Vary the initial conditions over a complete basis of "computational states." Measure the redistribution of geometric phases across initial states. Compare to the Orion sum rule prediction `(1/2π) Σ γ_n = 2m·ν_H` and the Wootters-concurrence-vs-phase pattern from the paper | M5.4 (e⁺/e⁻ pair test) |
| **Topological sum rule test** | Implement the same "gate" (e.g. defect-pair SWAP-analog) via two M5 evolution paths from different topological classes. Measure the per-state Berry phases; verify the sum is the same (`2m·ν_H` invariant) but the redistribution differs. Direct analog of the SWAP₁ vs SWAP₂ comparison in the paper's Fig. 3 | M5.4 + post-M5 (gate-analog protocols defined) |
| **CPT symmetry validation** | M5's Lagrangian is Hermitian and Lorentz-invariant by construction → must satisfy CPT (theorem). Verify numerically that pair creation and annihilation operate as CPT mirrors, that same-winding pairs are CPT-forbidden from annihilating (already a planned M5.4 test, here recognized as a CPT consequence), and that absorption / stimulated-emission analogs are CPT-paired (per Duda's 2026-04 deck on negative radiation pressure). Failure mode = a hidden T-symmetry-breaking term in the implementation | M5.4 (pair dynamics) |

#### Why this matters for OpenWave

These experiments do three things at once:

- **Validate that M5's twist DoF correctly carries quantum-phase physics.** If the Berry phase falls out of the M5 dynamics with the standard geometric-phase formula, that's strong evidence the framework is structurally complete: charge (topology), mass (amplitude), AND quantum phase (twist) all derived from one Lagrangian
- **Engage productively with the topology-and-entanglement literature.** The Orion paper sits in mainstream condensed-matter physics; M5 reproducing its predictions from a different starting point (classical-field-theoretic, topological defects) is the kind of cross-validation that makes the framework credible to physicists outside the wave-mechanics community
- **Connect to hydrodynamic quantum analogs.** Berry's hydrodynamic Aharonov-Bohm (1980, recreated 2026) and the broader walking-droplet HQA literature have struggled with multi-particle entanglement as the missing piece. M5's defect framework — where multi-defect entanglement-like correlations are automatic from shared topology — may be the structural bridge. If validated, this connects HQA, EWT, LdG-topology, and standard QM into one coherent picture

#### Open questions

- Does the twist field in M5 produce **exactly 2π-quantized** Berry phase, or only approximately? (Quantization should be geometric, but needs numerical confirmation)
- Does the two-defect Wootters concurrence analog quantitatively match the paper's predictions for entangling Hamiltonians, or only the qualitative pattern?
- How does the result depend on the choice of LdG potential, regularization (Faber), and Skyrme coefficient? Are there parameter regimes where the topology-entanglement link breaks?
- Do the standing-wave + topology + twist primitives also reproduce the **double-slit** and **Casimir** patterns Marc Fleury cites — i.e., is the framework consistent with the broader EM-fluid / standing-wave-electron literature beyond just the Orion paper?
- For Bell-inequality experimental analogs: does the twist-field path-dependence reproduce the experimental phase-compensation patterns, supporting Marc's claim that "Bell completely disappears with phase modulation"?

These are deferred research questions — concrete numerical work waits until M5.4 + M5.7 establish the defect substrate. Documented here so the framework is in scope; specific experiments will be planned when the prerequisites land.

### Beyond matter — forces, EM waves, heat

The matter-particle hierarchy above (Layers 0–6) is OpenWave's *foundational layer*. On top of validated matter primitives, the simulator's full scope covers four output domains:

| Output domain | What M5+ must compute | Where it sits in the roadmap |
| --- | --- | --- |
| **MATTER** (foundation) | Particle emergence (leptons → quarks → nucleons → atoms) via topological defects + wave dynamics | Layers 0–5+ above |
| **FORCES** | Electric (topology), strong (string tension + standing waves), magnetic (transverse waves from spin), gravitational (density deficit / 4D boost-axis topology) — all derived from a single Lagrangian | Validated cumulatively across M5.4 (electric), M5.7 (strong), Phase 4 (magnetic), Phase 5 (gravity) |
| **ELECTROMAGNETIC WAVES** | Photon emission/absorption, EM dispersion, polarization, antenna-medium coupling | Phase 4 (alongside magnetic-force validation) |
| **HEAT** | Thermal energy at the wave / spin-coherence level (not bulk kinetic temperature); thermal-EM coupling; Wien's law / blackbody emergence | Phase 6+ (long-term, after the matter layers are stable) |

**Why these four are co-equal scientific goals**:

- The matter hierarchy alone is incomplete physics. You can simulate an electron beautifully and still not understand how it interacts with surrounding fields, photons, or thermal environments
- Force / EM / heat are how matter *interacts* with the rest of the world — these are the observable, measurable outputs against which a wave-mechanics simulator is validated
- Each output domain is its own validation target — not a derived consequence of M5 matter physics, but an *independent* phenomenon that requires its own emergent-physics chain. **Validating that EM waves emerge correctly (Phase 4) is a separate undertaking from validating particle masses (M5.8); validating gravity (Phase 5) is separate from both. Each downstream application that uses any of these as a tool — for resonance probing, for force coupling, for thermal modulation — depends on the corresponding output-domain phase landing first.** This is why the roadmap explicitly names them as parallel goals rather than collapsing them into "stuff M5 will eventually compute"
- A simulator that does only matter is a particle-physics tool. A simulator that does matter + forces + EM + heat from one Lagrangian is a **complete classical-field-theoretic description** — that's the integrated value of OpenWave

### How matter layers feed forces / EM / heat outputs

Each output domain depends on validated matter layers being in place. The dependency chain:

| Output | Depends on matter Layer(s) | M5 phase chain |
| --- | --- | --- |
| Electric force (Coulomb 1/d²) | Layer 1 (single defects) | M5.1 + M5.4 |
| Strong force / confinement | Layer 1 + Layer 2 (vortex strings) | M5.1 + M5.7 |
| Magnetic force | Layer 1 + spin dynamics | Phase 4 (post-M5) |
| Gravitational force | Layer 1 + density-deficit / 4D extension | Phase 5+ (post-M5) |
| Photon / EM wave dynamics | Layer 1 + tilt-axis curl (Maxwell from director field) | Phase 4 (alongside magnetic) |
| Thermal energy / heat | Layer 5 (atoms) + spin-coherence dynamics | Phase 6+ (post-M5, post-atoms) |

Heat is *deepest* in the dependency chain because it requires simulated atoms to model thermal phenomena meaningfully (temperature is a statistical property of many-particle systems).

### Phase 4 — explicit goals (refined 2026-05)

Phase 4 (EM / magnetic emergence) is not yet detail-scoped at the per-experiment level the way M5.0–M5.8 are; it sits as a "post-M5" phase. The goals below are the concrete physics targets Phase 4 must meet — refined 2026-05. They are physics-only.

| Phase 4 goal | What it validates | Why it matters |
| --- | --- | --- |
| **L+T decomposition of defect-emitted wave as separable observables** | The defect's outgoing wave carries longitudinal (electric / scalar) AND transverse (magnetic) components, measurable independently and controllable independently | Already implicit in [1b_topological_defect.md § Outgoing-wave L+T decomposition](1b_topological_defect.md). Phase 4 elevates from "stated" to "numerically verified". Required for any method that targets L vs T independently |
| **Polarization-selective response in dielectric / ferromagnetic / metasurface analogs** | A simulated material with anisotropic structure exhibits selective transmission / reflection of L vs T components of the defect-emitted wave | Validates that the L+T decomposition is physically actionable — the components can be filtered, rotated, mode-converted by structures analogous to optical polarizers. Without this validation, "manipulate L and T separately" is a hopeful claim |
| **Frequency-downshift inertial-response test** | Apply a heterodyne / low-pass / mixing operation on the high-ω T-component of a single defect's outgoing wave; measure whether the downshifted-effective-frequency variable mag field exerts measurable force on a test charged particle (electron analog) | Tests the underlying physics. The "averaged-out at high ω" intuition is correct; whether a downshift operation defeats that averaging is the falsifiable open question. Pass = downshift principle works; fail = the averaging is not recoverable, magnetism stays inertially invisible at all frequencies |
| **Maxwell from director-field curl** (already implied) | Standard Maxwell equations emerge from the LdG director field's curl + boost-axis dynamics in the appropriate continuum limit | Existence proof that the M5 framework reproduces conventional EM at the macroscopic limit; required for credibility with the broader physics community |

This document is OpenWave's M5 roadmap and stops at the physics.

---

## STATUS

Current M5 phase status (✅ / 🔶 / [ ]) lives in [`0c_roadmap.md`](0c_roadmap.md) § SUMMARY. See [`0c_roadmap.md`](0c_roadmap.md) § DETAILED for per-phase task lists. Architecture analysis (this document) covers everything *around* the linear roadmap: design rationale, inheritance, performance, validation, comms.

---

## EXTERNAL-COMMS MILESTONES — TRIGGER GATES FOR MODELS-OF-PARTICLES UPDATES

Strategic decision 2026-05-11: M5.1 Coulomb result (R²=0.978 + visual EM-field-line geometry, documented in [`3a_coulomb_visual_geometry.md`](3a_coulomb_visual_geometry.md)) is correct but **not yet ready for an external update to the Models-of-Particles group**. Reasons for holding:

| Concern | Why it's not email-ready |
| --- | --- |
| Static energy only (not dynamic) | Duda will ask "can you SEE attraction?" — answer should be yes |
| R² = 0.978 (Dirichlet BC) vs Exp 2's 0.993 (periodic BC) | First impression should not look like a regression on a worked example |
| Topology dissipates under heavy relax | Admits the result is fragile — undermines the demo |

Per `feedback_external_comms.md` memory: scientific-venue communications are Rodrigo's voice only, no AI text generation. Below are the **content-trigger milestones** that would lift the email-readiness threshold. Send when any of these lands, in increasing impact order:

| Trigger | What's added since M5.1 | Email framing (in Rodrigo's voice when composed) |
| --- | --- | --- |
| **M5.2 V(ψ) lands + defect survives EVOLVE PSI** | Stable particle in simplest dynamic sense; topology no longer dissolves | "Topological hedgehog is now a long-lived resonance under V(ψ)" |
| **M5.2 + dynamic attraction visible** | Pair moves toward each other under wave dynamics (defect motion is observable, not just static energy bookkeeping) | "Coulomb attraction observed dynamically" |
| **M5.4 headline test passes** | Stable single defect + dynamic pair Coulomb + possibly annihilation | "M5.4 complete — single hedgehog stable, pair reproduces dynamic 1/d²" |
| **M5.5 Skyrme protection** | Topology robust under any relaxation / perturbation; the "topology dissipation" caveat goes away | "Topology fully protected (Skyrme stabilizer)" |

Recommended first send: **M5.2 + defect survives EVOLVE PSI** (the lowest bar that addresses all three current concerns at once). M5.4 would be even stronger but is months out; communicate before then. When the email goes out, it should reference M5.1 Coulomb as the FIRST DATA POINT in a sequence — the M5.1 work isn't wasted, it becomes more compelling when paired with M5.2+ dynamics.

**Pre-send checklist** (per `feedback_external_comms.md`):

| Item | Status when ready |
| --- | --- |
| Composed in Rodrigo's voice (no AI prose draft) | Required |
| Visual screenshots / plots as the lead | Available — `images/coulomb_visual/` + future M5.2 dynamic captures |
| Honest caveats (what's still TBD) | Required — Duda values transparency |
| Reference Jarek's April 2026 framing | The April thread set the "recreate EM via topological charges" milestone; closing the loop |
| Single screen-height max | Mailing-list etiquette |
| Reply-in-thread to the original 2026-04 discussion | Continuity |

---

## GROUP FEEDBACK (2026-04-17/18) — REFINEMENTS TO M5 PLAN

Replies from Jarek Duda, Jeff Yee, and Robert Close to the Apr 17 sandbox-summary email produced four targeted refinements to the M5 plan. Full email thread in [1a_lagrangian_framework.md § EMAIL THREAD](1a_lagrangian_framework.md#email-thread).

### Refinement 1 — Use Close's Eq. 23 as the particle equation (Robert Close)

Exp 7 implemented Close's Eq. 21 ("Equation of Everything"). Close's recommendation for the **particle equation** is actually **Eq. 23** — it preserves `∇·s = 0` (zero divergence of spin density). M5.2 adopts Eq. 23 as the particle equation, with Eq. 19 as the V=0 linear limit, and enforces the divergence constraint per time step.

### Refinement 2 — Particles are resonances, not stable solitons (Robert Close)

Close: *"Unless you have a good way to model an infinite system, I doubt that you will find completely stable non-radiating solutions."* M5's success criterion is reframed: **long-lived resonance with measurable lifetime**, not perfect stability. M5.2 includes an amplitude-sweep protocol (`Y_1^0` seed at `A ∈ {0.25, 0.5, 1.0, 2.0}·λ`) to hunt for these resonances.

### Refinement 3 — Lepton mass hierarchy from `0 < δ << 1 << g` axis lengths (Jarek Duda)

The ~3477:1 biaxial ratio the sandbox required is **physically motivated**, not ad-hoc. Three naturally-separated physical scales map to the three axis eigenvalues: `δ ~ ℏ` (QM / twist), `1` (unity / tilt / matter), `g` (gravity / boost). M5.6 parameterizes the Q-tensor this way from the start. Jarek also flagged that **LdG regularization is the hardest numerical step** — a known challenge that M5.6 budgets accordingly.

### Refinement 4 — New validation phases M5.7 and M5.8 (Jarek Duda)

- **M5.7 — Cornell potential**: after M5.4 succeeds, extend from point hedgehogs to a **topological vortex string** with fractional-charge end points. Measure `V(r) = −α/r + σ·r` and confirm the `~1 GeV/fm` linear confinement coefficient. This is the QCD confinement analog in the Lagrangian-topological framework
- **M5.8 — De Broglie clock / Zitterbewegung**: measure the intrinsic oscillation frequency of a single LdGS defect (electron: non-dual; neutrino: dual) and validate `ω = 2·m·c²/ℏ`. The 1+1D phi-4 toy (arxiv:2501.04036) showed the *mechanism*; M5.8 validates it in full LdGS for real particles

### Refinement 5 — M3 near-field carries three force regimes (Jeff Yee)

Jeff confirmed the M3 + topology coexistence decision and added a crucial scope expansion: M3's standing-wave lock-in is the mechanism for **three** force regimes, not just one — (a) intra-particle binding of K=1 WCs into standalone particles, (b) intra-nucleus strong force between K=10 particles, and (c) **orbital force** (electron-nucleus binding in atoms). This strengthens the rationale for keeping M3 physics intact in M5 — it's load-bearing well beyond the K=10 electron problem.

> **Note on vocabulary (2026-04-19)**: Jeff's regime enumeration uses EWT's K-count vocabulary (K=1 = neutrino, K=10 = electron) as was standard before the Duda framework was adopted. In the M5 paradigm, the mapping shifts: (a) single-particle stability is *not* a wave lock-in problem in M5 (the particle is a single defect, held together by topology, not by K=1-to-K=10 wave interference); (b) intra-nucleus strong force is between *quarks* (single defects of a different type) bound into nucleons, not between electrons; (c) orbital force unchanged — electrons around nuclei. Jeff's underlying point stands: M3's near-field physics is load-bearing for composite-particle and atomic-scale dynamics in M5.

### Summary impact table

| Area | Change | Source |
| --- | --- | --- |
| M5.2 Wave dynamics | Close's **Eq. 23** (not Eq. 21) as particle equation; enforce `∇·s = 0`; amplitude-sweep resonance hunt | Robert |
| M5.4 Headline test | Success criterion = long-lived resonance (lifetime ratio), not perfect stability | Robert |
| M5.6 Biaxial LdG | Axis hierarchy `0 < δ << 1 << g` (physics-motivated); LdG regularization budgeted as hardest step | Jarek |
| **New M5.7** | Cornell potential via topological vortex string; `V(r) = −α/r + σ·r` with `σ ≈ 1 GeV/fm` | Jarek |
| **New M5.8** | De Broglie clock / Zitterbewegung test in full LdGS for electron + neutrino | Jarek |
| M3 retention rationale | M3 near-field covers *three* force regimes: intra-particle, strong, and orbital | Jeff |
| M5.6 `(δ, g)` treatment | Calibrated numerically (no ab-initio form exists); iterate against observed lepton-mass ratios | Jarek (2026-04-19) |
| M5.6 regularization baseline | Port Manfried Faber's scheme (arxiv:2604.12021, Universe 11/2025/113) — slightly different potential but produces running-coupling effect; adapt rather than reinvent | Jarek (2026-04-19) |
