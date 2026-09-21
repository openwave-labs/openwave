# M4 Pipeline Engine — Implementation Plan

> **Status:** DRAFT (working document)
> **Scope:** The pipeline_engine as the realisation substrate for Enhanced EWT.
> **Audience:** Contributors implementing, testing, or extending M4.
> **Related:** `M4_engine_upgrade.md`, `M4_k_selectivity_Formalization.md`,
> `__M4_model_briefing.md`, manuscript v5.0.x (Zenodo), Yee's EWT corpus.

---

## 0. Executive Summary

The pipeline_engine is a **composition root** for M4 simulations: a list of
stateless processors executed against a shared `Context`, with explicit
`requires`/`provides` contracts, swappable sinks, and a typed `FeatureBag`
for runtime state. It exists because the previous monolithic `wave_engine.py`
made it impossible to express the actual physics of Enhanced EWT — every
mechanism was a hard-coded branch, every variant a new file, every comparison
a code duplication.

This document captures:

1. **Why** the previous implementation only *simulated* EWT and did not
   *realise* it (Section 1).
2. **What** the core conceptual commitments are that the new engine must
   honour (Sections 2–5).
3. **How** to build the engine (Block 1) and the physics (Block 2), with
   concrete, checkable work items (Sections 6–7).
4. **What** from the old implementation is obsolete and should not be ported
   (Section 8).
5. **Which** questions remain genuinely open and require author input or
   further research (Section 9).

Nothing in this document is a claim about nature. It is a **plan for a tool**,
written so that the tool can express the hypotheses we intend to test.

---

## 1. Why the previous implementation faked EWT

This section is diagnostic, not accusatory. The old `wave_engine.py` was an
honest attempt to translate EWT into code. It failed for structural reasons,
not for lack of effort.

### 1.1. Wave centres as pinned sources, not reflectors

Yee's EWT defines a wave centre as a **point at which incoming waves are
reflected to become outgoing waves**. The soliton is the standing-wave
interference of `Ψ_in` and `Ψ_out`, bounded by the particle radius.

The old implementation defined wave centres as **hard-pinned regions**:

```text
ψ = A · sin(ω t + offset) · r̂    inside a ball of radius R around each WC
```

This is a driven antenna, not a reflector. The field does not participate in
the WC's existence; the WC simply overwrites voxels. There is no reflection,
no `Ψ_in`/`Ψ_out` distinction, no conservation argument.

### 1.2. One field, not two

Yee's EWT distinguishes **longitudinal** (mass, charge) from **transverse**
(spin, magnetism), coupled through the fine-structure constant at the WC.
The old implementation had a single vector field `ψ` with no mode split.

Consequence: the fine-structure constant had to be **imposed** through
`cos(offset)` and the mass of the electron through an analytic formula.
Neither emerged from dynamics.

### 1.3. Wavelength structure ignored

Yee's standing-wave geometry prescribes *decreasing* wavelengths from the core:

```text
r_wavelength(n) = 2Kλ − 2nλ
r_x = (K + 2·Σ_{n=1..x}(K−n)) · λ
```

with a maximum radius `r_particle = K²λ`. The old implementation used a
single `base_wavelength` throughout. The electron's characteristic ring
structure — visible in the Lund stroboscope image — was absent.

### 1.4. Density profile fixed, not self-consistent

Enhanced EWT's gravitational sector is built on the **push-out** mechanism:
the soliton's energy density `ρ_E` displaces Elastic Medium Constituents
(EMC), creating a local deficit `ρ(r) < N_ν,stat`. The deficit is what the
outside vacuum presses against; the pressure gradient is what we call gravity.

The old implementation had `ρ(r)` as a **fixed profile relative to the domain
centre**. When a WC drifted, the well did not follow. There was no
self-consistency between `|Ψ|²` and `ρ(r)`.

### 1.5. Nonlinearity as an external potential, not a consequence of `c(ρ)`

M4.9 established the microscopic relation:

```text
v_phys(η) = a(η) · sqrt(k(η)/m₀)  ∝  η^{+1/2}
```

where `η = ρ/ρ₀`. The local wave speed depends on the local EMC density.
Because the soliton itself depletes EMC, `c²(r) = c₀²·ρ(r)/ρ₀`, and the
wave equation becomes *automatically nonlinear*:

```text
∂²Ψ/∂t² = ∇·(c²(ρ) ∇Ψ) + c₀² (β_ρ/ρ₀) |∇Ψ|² Ψ − 2 κ β_ρ² Ψ³
```

where the second term is the Euler-Lagrange correction that keeps the
gradient energy `½c²|∇Ψ|²` conserved when `c²` varies with the field. The last two terms are the Euler-Lagrange corrections. 
See Section 5.2.

The old implementation instead injected a Klein–Gordon-style potential
`V(ψ) = (c₁/4)u² − (c₂/6)u³` with `u = |ψ|²`. Mathematically this gives an
NLS soliton. Physically it is a prosthesis: the coupling constants `c₁`, `c₂`
were fitted, not derived.

### 1.6. Gravity as `−∇E`, not as `−∇ρ`

The old `compute_force_vector` used `F = −∇E_local`, where `E_local` was a
heuristic `ρ·V·(f·A)²` with `f` hard-coded to the base frequency. This
measured the *internal* energy gradient, not the *external* EMC density
gradient. The push-out picture requires `F_pressure = −α_p·∇ρ(r)`, where
`ρ(r)` is the EMC packing density, not the soliton's energy density.

### 1.7. Summary table

| EWT primitive | Old implementation | Why it is a fake |
|---|---|---|
| WC as reflector | WC as hard pin | No reflection, no in/out |
| `Ψ_in` + `Ψ_out` | Single `ψ` | No distinction, no closure |
| Decreasing `λ(n)` | One `base_wavelength` | Ring structure absent |
| `Ψ_long` + `Ψ_trans` | One vector field | Spin not emergent |
| `ρ(r)` self-consistent | Fixed profile | No push-out feedback |
| `c²(ρ)` nonlinearity | External `V(ψ)` | Coupling fitted |
| `F = −∇ρ` gravity | `F = −∇E` | Measures wrong gradient |
| Boundary as EMC wall | Dirichlet `ψ = 0` | No profile, no sign change |
| BCC lattice | Simple cubic stencil | Medium ≠ medium of theory |

---

## 2. Conceptual commitments

These are the principles the new engine and physics must respect. They are
*not* claims about nature; they are constraints on the tool.

### 2.1. Conservation is structural, not imposed

Energy conservation in EWT is a consequence of the **unitarity of reflection
at the WC** and the **Hermiticity of the wave equation**, not an external
constraint. If the WC reflection satisfies

```text
|Ψ_out|² + |Ψ_spin|² = |Ψ_in|²
```

and the wave equation derives from a real Hamiltonian density, then energy is
conserved by construction. Any deviation is a bug, not a feature.

The engine must therefore expose enough structure to *measure* the energy
budget and *verify* that `dE/dt + flux ≈ 0`.

### 2.2. The soliton is an open system in steady state

The medium carries an always-on base wave (Yee: "waves flow through all of
matter"). This wave supplies `Ψ_in` to the WC. The WC reflects part of it as
`Ψ_out` and converts part of it into transverse spin. In steady state:

```text
flux_in = flux_out + flux_spin
```

The soliton is a **dissipative structure** in the Prigogine sense — a local
concentration of energy maintained by continuous exchange with the medium,
not a closed system isolated from it.

This is a deliberate commitment to **variant B** (coupled dynamics), not
variant A (static background). It is harder to implement, but it is what the
manuscript describes.

### 2.3. The EMC density field is dynamic and self-consistent

`ρ(r)` is not an input. It is a solution of

```text
∂ρ/∂t = D ∇²ρ − γ_ρ (ρ − ρ₀) − β_ρ |Ψ|²
```

in the simplest model, or a richer evolution with inertia. In steady state,
`ρ(r)` and `|Ψ(r)|²` are mutually consistent. The EMC Wall — the local peak
`ρ > ρ₀` that reverses the sign of the nonlinearity — is a **consequence**
of the dynamics, not a parameter.

### 2.4. The tail is analytic; the soliton neighbourhood is simulated

The far-field deficit

```text
ρ(r) → N_stat · (1 − r_s/r)   for r ≫ r_core
```

is taken as analytic input (granted in M4.4; derivation in the manuscript).
The pipeline does not re-derive it and does not simulate it. The simulated
domain covers the soliton and its immediate neighbourhood; the tail beyond
is analytic.

This separation is deliberate: the soliton neighbourhood is where the
wave-centre structure lives and where K-selectivity must emerge; the tail
is a solved analytic problem that does not need a discretised field.

### 2.5. Units are pluggable

No processor contains a dimensional constant. Every constant comes from a
single `UnitSystem` feature. Three implementations are anticipated:

- `NaturalUnitSystem` — `λ = 1`, `c = 1`. For research and algorithm testing.
- `OpenWaveUnitSystem` — `am`, `rs`, as in the legacy xparameters.
- `SIUnitSystem` — for output conversion and cross-checking against CODATA.

The same pipeline runs under any of them. Results are comparable after
conversion.

### 2.6. Topology and geometry are hypotheses, not assumptions

The 1-3-6 arrangement and the `n·λ` spacing of wave centres are **candidate**
configurations, not given truths. The engine must allow systematic variation
of:

- Topology (1-3-6, golden-angle, BCC, line, random).
- Spacing (exact `n·λ`, exact `(n+½)λ`, perturbed, swept).
- Coupling (none, push-out only, push-out with feedback).

Only then can we say which of these factors is necessary, sufficient, or
irrelevant for stability.

### 2.7. What α is, and what it is not

The manuscript defines `α` as a purely geometric ratio:

```text
α = x² / S = 1 / (4π³ + π² + π)
```

where `S` is the emission surface (sphere plus cone) and `x` is a bookkeeping
amplitude that cancels. With the BCC correction:

```text
α = 1 / (A_π − ε_M) ≈ 7.29733855 × 10⁻³
```

This is the value the engine loads from `GeometricConstants`. It is a
**parameter**, not a derived quantity. The engine uses it as the reflection
coefficient at a wave centre; it does not derive it from the field.

A dynamic counterpart — whether some field ratio at a WC coincides with this
value — is a **consistency observation**, not a second derivation. The
engine measures several candidate ratios (Section 7, item 2.2, variant B2b)
so that the coincidence, if any, can be recorded. A mismatch is not a failure
of the plan; it means the geometric ratio has no direct dynamic manifestation
in this engine.

---

## 3. The unit system

### 3.1. Contract

A `UnitSystem` is a feature providing at least:

```text
c            : wave speed in these units
lambda       : fundamental wavelength (λ_ν)
dx           : grid step in these units
dt           : time step in these units
rho_0        : statutory background density (N_ν,stat)
A_pi         : 4π³ + π² + π
eps_M        : 1 / (N_geom · π³)
N_geom       : 8π⁴ · (1 − ζ)
gamma        : 1 / eps_M
X_eff        : geometric dilution factor
N_nu_eff     : effective volume deficit
A_base       : base wave amplitude
r_domain     : simulated domain radius
r_core       : soliton extent (K²λ), theoretical scale
```

Plus conversion methods:

```text
to_physical_length(x)   -> metres
to_physical_time(t)     -> seconds
to_physical_energy(E)   -> joules
to_physical_density(r)  -> 1/m³
```

### 3.2. Implementations

**NaturalUnitSystem** (default for research):

```text
c      = 1
lambda = 1
dx     = 1 / K_grid     (e.g. 0.05 for 20 voxels/λ)
dt     = CFL_SAFETY · dx / (c · √3),   CFL_SAFETY ≈ 0.9
```

The 3D leapfrog bound is `dx / (c · √3)`. `CFL_SAFETY = 0.9` of that bound
gives `dt ≈ 0.52 · dx / c`. A 1D run may use `dt = CFL_SAFETY · dx / c` as
a validation figure; the default is 3D and takes the `√3` factor.

**OpenWaveUnitSystem**:

```text
c      = 0.3 am/rs
lambda = EWAVE_LENGTH / ATTOMETER   (≈ 28.5 am)
dx     = lambda / 12
dt     = CFL_SAFETY · dx / (c · √3)
```

**SIUnitSystem**:

```text
c      = 299792458 m/s
lambda = 2.8540965e-17 m
dx     = r_e / 200
dt     = CFL_SAFETY · dx / (c · √3)
```

Used primarily for output conversion.

### 3.3. Consequences for processors

A processor must never write `c = 1` or `gamma = 2.4e4` or `lambda = 28.5`.
It must always write:

```python
units = ctx.data.require(UnitSystem)
c = units.c
gamma = units.gamma
```

This rule is checkable by inspection: grep for numeric literals in
`physics/*.py`. If a processor contains a dimensional constant, it is a bug.

---

## 4. Simulation domain and boundary

The pipeline simulates the soliton and its immediate neighbourhood. The
far-field tail of the EMC deficit extends far beyond any tractable domain
and is treated analytically (Section 2.4); it is not simulated.

Two length scales matter, and they are not the same:

- **The soliton extent**, `r_core = K²λ`. For K = 10 this is 100 λ_ν, for
  K = 12 it is 144 λ_ν. It is a theoretical scale of the standing-wave
  region. The EMC tail continues beyond it, and the tail is analytic.

- **The wave-centre neighbourhood.** The region the K-sweep actually
  measures. Its size is set by the largest distance from the
  configuration centre to a wave centre, plus a buffer of a few λ. From
  the geometry tests, the measured configuration radii at K = 10 are
  2.00 λ_ν (tetrahedron_10_locked) and 0.36 λ_ν (golden angle); the
  `line` negative control at K = 12 has radius 5.50 λ_ν. A domain of
  `r_domain ~ 10 λ_ν` covers every topology in the sweep with a wide
  margin.

The simulation runs on the wave-centre neighbourhood. The soliton extent
and the tail are not resolved; they are analytic inputs.

At `dx = 0.05` (20 voxels per λ_ν) and `r_domain = 10 λ_ν`, the box is
400 voxels in diameter, about 6.4 × 10⁷ voxels in 3D. Tractable. A domain
that instead resolved the soliton extent `r_core = K²λ` would be
1.9 × 10¹¹ voxels at K = 12 and is not.

The boundary at `r = r_domain` is one of:

- **Absorbing** — waves leave and do not return. Default for isolated
  soliton tests.
- **Periodic** — the domain wraps. No leakage, at the cost of wrap-around
  artefacts when the soliton's tail meets itself.
- **Reflecting** — waves bounce. Use only when the test specifically
  requires reflections.

The choice is per experiment, not global. There is no attempt to model the
tail at the boundary; the tail is analytic elsewhere.

**Validation with K = 1.** The neutrino has a single wave centre and no
extended soliton structure. It fits inside the domain trivially and serves
as the validation case for the local dynamics before the K > 1 runs begin.

---

## 5. Vacuum layer

### 5.1. What the base wave is

The base wave is the always-on ground-state oscillation of the medium. In
natural units:

```text
Ψ_base(r, t) = A_0 · cos(k·r − ω·t)
```

with `k = 2π/λ = 2π` and `ω = 2π·c/λ = 2π`.

### 5.2. The conservation check

The energy budget excludes `E_base`. The check is:

```text
E_soliton     = ∫ (½ |∂Ψ/∂t|² + ½ c²(ρ) |∇Ψ|²) dV
E_deformation = ∫ ½ κ (ρ − ρ₀)² dV
E_total       = E_soliton + E_deformation
check:          dE_total/dt + flux_through_boundary = 0
```

The kinetic term is required: without it, a standing wave's gradient-only
integral oscillates at `2ω` and the conservation test fails by construction.
The deformation term uses `κ`, the stiffness supplied by the unit system.

The wave equation is implemented in the Euler-Lagrange form derived from
the Lagrangian

```text
L = ½ |∂Ψ/∂t|² − ½ c²(|Ψ|²) |∇Ψ|² − V(Ψ)
```

`V` is the sum of the potentials the selected variants contribute, by the
same rule the budget follows. Under B4a the slaved density
`ρ = ρ₀ − β_ρ|Ψ|²` makes the deformation energy a potential in the field,
`V = ½ κ (ρ − ρ₀)² = ½ κ β_ρ² Ψ⁴` in the real one-component case, which
gives

```text
∂²Ψ/∂t² = ∇·(c²(ρ) ∇Ψ) + c₀² (β_ρ/ρ₀) |∇Ψ|² Ψ − 2 κ β_ρ² Ψ³
```
where the second term is the variation of `c²` with respect to `|Ψ|²`
(one-component case, `c² = c₀²(1 − β_ρ|Ψ|²/ρ₀)`) and the third is
`− dV/dΨ`. This is the same equation as the one in Section 1.5, and the
Hamiltonian of this `L` is `E_total` term for term, which is what makes
the budget a check rather than a restatement of the stepper. The plain
divergence form
`∇·(c²∇Ψ)` alone does not conserve the gradient energy `½c²|∇Ψ|²` when `c²`
depends on the field: the exchange term `½ ∫ ∂(c²)/∂t · |∇Ψ|²` appears on
the right-hand side of the identity and does not vanish when `c²` moves.
The Euler-Lagrange form above cancels it exactly, so the budget holds as
written.

The alternative was to drop `E_deformation` under B4a entirely, treating
the slaved density as bookkeeping rather than a second energy. This plan
takes R2: the density has energy in every variant, and the equation pays
for it, by the `V` term above.

For the EMC density dynamics variants (item 2.4):

- **B4a (instantaneous):** the density is a function of `|Ψ|²` at each
  step, and the equation carries the extra term `− 2 κ β_ρ² Ψ³` from the
  deformation potential. `E_total` is conserved to numerical tolerance.
- **B4b (relaxation):** the `D∇²ρ` and `−γ_ρ(ρ − ρ₀)` terms dissipate.
  The budget carries a ledger entry `P_deform` for the rate at which the
  deformation energy is lost, and the check becomes
  `dE_total/dt + flux + P_deform = 0`. `P_deform > 0` means the
  deformation is dissipating; `P_deform < 0` means it is being driven.
- **B4c (inertial):** the density carries its own kinetic energy. The
  deformation energy becomes
  `E_deformation = ∫ (½ |∂ρ/∂t|²/c_ρ² + ½ κ |∇ρ|² + ½ κ (ρ − ρ₀)²) dV`,
  and the budget includes it. B4c is deferred until the extra term is
  added to the tracker. ⚠️ The three terms above do not share a single
  `[κ]`: the gradient and spring terms differ by a factor of length
  squared and `κ` multiplies both, so it cancels between them and no
  choice of `[κ]` makes all three energy densities. Pinning the density
  wave speed at `√κ · c_ρ`, or redefining `c_ρ`, reaches only the kinetic
  term. The formula needs a length scale, or a second stiffness, before
  the variant is implemented from it.

**Rule.** The budget enumerates every term the selected variants put in
the equation. A variant that adds a potential to the Lagrangian adds its
matching term to the stepper and its energy to `E_total`. The B6a
recommended start, for instance, adds `F = γ_nl (1 − ρ/ρ₀) |Ψ|² Ψ`, which
under B4a is `γ_nl (β_ρ/ρ₀) Ψ⁵` and carries its own potential
`− γ_nl (β_ρ/ρ₀) Ψ⁶ / 6`; without that term the drift does not converge.
⚠️ That potential is unbounded below, so B6a has no ground state and a
large-amplitude run can collapse rather than converge. The term is kept
on the empirical ground above; the property is recorded here so it is a
known limit of the variant rather than a surprise in a sweep.
The budget is not a fixed formula, it is the sum of the active terms.


### 5.3. Coupling to the soliton

The soliton interacts with the base wave only **at the WC**, through
reflection. There is no volume coupling. This preserves the locality of the
interaction and avoids the "pumping" problem: a volume-coupled base wave
would continuously inject energy everywhere, and the soliton would either
grow without bound or require an explicit damping term.

### 5.4. The vacuum layer is swappable

Energy should not disappear. But the simulation domain is finite, waves
reflect off its boundaries, and there is no obvious way to keep a
non-equilibrium steady state running forever without either injecting energy
or bleeding it. The problem is real and not resolved here.

The plan's response is to make the vacuum layer **swappable**. The interface
is fixed; the implementations are variants:

```text
VacuumProvider:
    seed(ctx)         # called once, at t = 0
    step(ctx)         # called every step; may inject or absorb
    energy_budget()   # returns what the provider contributed or removed
```

Candidate implementations, in order of increasing physical fidelity:

- **V1 — static vacuum.** `Ψ_base` is set once and does not evolve. The
  soliton evolves on top of a fixed background. No energy exchange, no
  reflection problem from the vacuum side, but the background does not
  respond to the soliton.
- **V2 — passive vacuum.** `Ψ_base` evolves with the wave equation but is
  never re-driven. Energy leaks through the boundary and is lost. Simple,
  but not conservative.
- **V3 — periodic box.** The domain wraps. No leakage. The vacuum is
  conserved by construction, at the cost of wrap-around artefacts when the
  soliton's tail meets itself.
- **V4 — absorbing boundary with re-injection.** Waves leaving the domain
  are absorbed; an equal flux is injected at the boundary to keep the total
  energy constant. Conservative in the budget sense, but the injection is
  artificial and can seed artifacts.
- **V5 — self-consistent vacuum.** `Ψ_base` and the soliton are coupled;
  the base wave responds to the soliton's presence and the total energy is
  conserved by a Hamiltonian structure. The most physical, but also the
  hardest to implement and the most likely to be numerically stiff.

The plan starts with **V1** for testing the engine, moves to **V3** for
K-selectivity (where the wrap-around artefacts may be tolerable if the
domain is large enough), and leaves **V4** and **V5** as research targets.
Which one is physically correct is author-gated (Section 9, Q2).

---

## 6. Block 1 — Engine implementation

Infrastructure only. No specific physics. Each item is a work unit.

### 1.0 — `UnitSystem` feature

- [ ] Define the `UnitSystem` protocol (fields + conversion methods).
- [ ] Implement `NaturalUnitSystem`, `OpenWaveUnitSystem`, `SIUnitSystem`.
- [ ] Add `UnitSystem` to `Pipeline.external_provides`.
- [ ] Wire `Runner.run(..., initial_features=[units])`.
- [ ] Add a linter rule: no dimensional literal in `physics/*.py`.
- [ ] Document in `README.md`.

### 1.1 — Multi-field FeatureBag

- [ ] Define `PsiBaseField`, `PsiLongField`, `PsiTransField`,
      `EMCDensityField`, `EMCFluxField` (optional) as separate features.
- [ ] Each field is a `ti.Vector.field(3, ti.f32, shape=grid)` with a
      triple-buffer variant for time integration.
- [ ] Ensure `FeatureBag` handles these as distinct types without aliasing.
- [ ] Add a test: allocate all five, write distinct values, read back.

### 1.2 — Trackers per voxel

- [ ] Define `TrackerFields` feature.
- [ ] Fields: `energy_long_local`, `energy_trans_local`, `amp_local`,
      `freq_local`, `rho_local`.
- [ ] Implement `TrackersUpdate` processor in `Stage.MEASURE`.
- [ ] Implement 3-plane sampling (as in the legacy `sample_avg_trackers`)
      to compute global averages without full reductions.
- [ ] Add a test: uniform field → uniform tracker values.

### 1.3 — Multi-field evolution

- [ ] Generalise `LaplacianProcessor` to accept a field name in `__init__`.
- [ ] Generalise `LeapfrogProcessor` similarly.
- [ ] Define a coupling contract: `Ψ_long → Ψ_trans` conversion at WCs,
      with a tunable coefficient.
- [ ] Add a test: two coupled fields, no coupling coefficient → independent
      evolution; with coefficient → energy transfers.

### 1.4 — Source terms (additive, not overwrite)

- [ ] Define the contract: a source term **adds** to `psi_new`, never
      overwrites `psi_am`.
- [ ] Implement `SeedBaseWave` — seeds `Ψ_base` once.
- [ ] Implement `SourceTermInterface` — base class for additive sources.
- [ ] Add a test: two sources, superposition holds.

### 1.5 — Reflector interface

- [ ] Extend `WCState` with `reflect_coeff_long`, `reflect_coeff_trans`,
      `phase_shift`.
- [ ] These are *attributes*, not yet *behaviours*. The values are set by
      the experiment, not computed.
- [ ] Document the contract clearly: a reflector is a WC that satisfies
      unitarity on `Ψ_in`, `Ψ_out`, `Ψ_spin`.
- [ ] Add a test: reflection of a plane wave from a single reflector
      preserves energy.

### 1.6 — WC motion (drift)

- [ ] Extend `WCState` with `velocity`, `force`.
- [ ] Implement `WCMotionProcessor` in `Stage.POST_UPDATE`.
- [ ] Implement `WCDriftRule` contract: a callable that returns a force
      vector given local field values.
- [ ] Provide a default rule (`F = −∇ρ`) and a no-op (`F = 0`).
- [ ] Add a test: no-op drift → positions unchanged; default drift on a
      static `ρ` → WCs move to minimum.

### 1.7 — Boundary condition

- [ ] Define `BoundaryCondition` feature: `kind` (`absorbing` | `periodic`
      | `reflecting`), `r_domain`.
- [ ] Implement `BoundaryProcessor` in `Stage.POST_UPDATE`.
- [ ] Absorbing: waves leave without reflection. Periodic: wrap. Reflecting:
      mirror.
- [ ] Add a test: wave hitting absorbing boundary leaves the domain without
      reflection, energy accounted for in `flux_boundary`.

### 1.8 — Energy budget tracker

- [ ] Define `EnergyBudget` feature.
- [ ] Fields: `E_kin`, `E_grad`, `E_deform`, `flux_boundary`, `dE_dt`.
- [ ] `E_soliton = E_kin + E_grad` with `E_kin = ∫ ½ |∂Ψ/∂t|² dV` and
      `E_grad = ∫ ½ c²(ρ) |∇Ψ|² dV`; `E_total = E_soliton + E_deform`.
- [ ] Implement `EnergyBudgetUpdate` in `Stage.MEASURE`.
- [ ] Add a test: 1D harmonic oscillator → `dE/dt ≈ 0` to machine precision.

### 1.9 — Stability metrics

- [ ] Define `StabilityMetrics` feature.
- [ ] Fields: `sol_lifetime`, `localization`, `sphericity`, `freq_drift`,
      `wc_drift`.
- [ ] Implement `StabilityMetricsUpdate` in `Stage.MEASURE`.
- [ ] Add a test: static Gaussian → `localization` stays constant;
      spreading Gaussian → `localization` decreases.

### 1.10 — Experiment runner

- [ ] Implement `ExperimentRunner`:
      takes a list of configuration dicts, runs each, records summary.
- [ ] Output: CSV with one row per run, columns = config + final metrics.
- [ ] Support deterministic seeds per run.
- [ ] Add a test: 3-run sweep produces 3-row CSV.

### 1.11 — Geometric constants provider

- [ ] Define `GeometricConstants` feature.
- [ ] Implement `ComputeGeometry` lifecycle processor that imports from
      `m4_7_ewt_emergence_engine.py` and populates the feature.
- [ ] Wire as `external_provides` for pipelines that need it.
- [ ] Add a test: computed `A_pi`, `eps_M`, `N_geom` match the engine.

### 1.12 — Units and conversions

This is folded into 1.0. Listed separately only for traceability.

### 1.13 — Checkpoint / restart

- [ ] Define `CheckpointProcessor` (lifecycle + periodic).
- [ ] Serialise `FeatureBag` to disk (Taichi fields → numpy → npz).
- [ ] Deserialise on startup.
- [ ] Add a test: run 100 steps, checkpoint, run 100 more;
      run 200 from scratch; compare.

### 1.14 — Live monitor

- [ ] Adapt `live_monitor_viewer.py` for pipeline_engine.
- [ ] Panels: energy (long/trans/emc), boundary flux, WC drift, freq.
- [ ] Reads `live.json` written by `LiveJsonSink`.
- [ ] Add a test: launch monitor, run 100 steps, monitor updates.

### 1.15 — Research logging schema

- [ ] Define the output layout:
      `run_meta.json`, `timeseries.parquet` (or CSV), `summary.json`,
      `events.json`.
- [ ] `run_meta.json`: config, seed, code hash, unit system.
- [ ] `timeseries`: full time series per metric, with configurable cadence.
- [ ] `summary`: final metrics for sweep aggregation.
- [ ] `events`: annihilation, boundary hits, instability detected.
- [ ] Add a test: schema validates against a JSON schema.

### 1.16 — Simulation domain configuration

This is the decision documented in Section 4. Work items:

- [ ] Add `r_domain` and `r_core` to `UnitSystem`.
- [ ] Implement the boundary condition processor (1.7).
- [ ] Document the choice: `r_domain ~ 10 λ_ν`.
- [ ] Add a test: for K = 1, the whole soliton fits inside `r_domain`.

### 1.17 — Vacuum layer provider

This is the decision documented in Section 5. Work items:

- [ ] Define `VacuumProvider` interface: `seed`, `step`, `energy_budget`.
- [ ] Implement V1 (static), V3 (periodic), and V4 (absorbing with
      re-injection). V2 and V5 are deferred.
- [ ] Wire the provider as an external feature.
- [ ] Implement the conservation check excluding `E_base`, including the
      kinetic term.
- [ ] Add a test: base wave alone → `dE_soliton/dt = 0` trivially.

### 1.18 — Diagnostic hooks

- [ ] Define a stop-condition contract: `StopCondition` callable.
- [ ] Implement `DiagnosticProcessor` in `Stage.MEASURE`.
- [ ] Built-in conditions: `dE/dt > threshold`, `localization < threshold`,
      `sphericity < threshold`.
- [ ] Add a test: run with a forced violation → simulation stops early.

### 1.19 — Deterministic seeds

- [ ] Add `seed` to `RunContext` (already present).
- [ ] All random initialisation reads from `ctx.run.seed`.
- [ ] Add a test: two runs with same seed within a backend → bit-identical
      output; two runs across backends → equal to a stated tolerance at the
      parsed-value level.

### 1.20 — Parameter sweep DSL

- [ ] Define YAML/JSON schema for sweeps:
      `topologies`, `spacings`, `couplings`, `K`, `seeds`.
- [ ] Implement `SweepRunner` that consumes the schema and drives
      `ExperimentRunner`.
- [ ] Add a test: 2×2 sweep produces 4 runs.

### 1.21 — Artifact versioning

- [ ] Hash the configuration + code state.
- [ ] Store results in `output_dir / <hash> /`.
- [ ] Provide `list_runs()` and `load_run(hash)` utilities.
- [ ] Add a test: same config → same hash; different config → different.

---

## 7. Block 2 — Physics implementation

Each item is a *variant*. The engine (Block 1) is variant-agnostic; the
physics layer supplies the specific mechanisms.

### 2.0 — Soliton assembly

Before any specific mechanism, define how the pieces compose:

```text
Ψ_total = Ψ_base + Ψ_soliton
ρ(r)    = ρ₀ − β_ρ |Ψ_soliton|²        (initial guess)
c²(r)   = c₀² · ρ(r) / ρ₀
```

The soliton exists as a *fixed point* of the coupled system:
`Ψ_soliton` and `ρ` are mutually consistent. This composition is the
foundation; all variants below are refinements.

- [ ] Implement `SolitonAssembly` as a documented contract.
- [ ] Add a test: static `Ψ_soliton`, no coupling → no soliton, only
      dispersion.

### 2.1 — Vacuum implementation

- [ ] **V1**: static vacuum.
- [ ] **V2**: passive vacuum (leaky).
- [ ] **V3**: periodic box.
- [ ] **V4**: absorbing boundary with re-injection.
- [ ] **V5**: self-consistent vacuum (deferred).
- [ ] Recommended start: **V1** for engine tests, **V3** for K-selectivity.

### 2.2 — WC as reflector

The reflection coefficient `α` is **loaded from `GeometricConstants`**, not
derived. Its value is `α = 1/(A_π − ε_M) ≈ 7.29733855 × 10⁻³`, fixed by the
geometric derivation in the manuscript.

- [ ] **B2a**: perfect reflection, no spin conversion
      (`reflect_coeff_trans = 0`).
- [ ] **B2b**: reflection with conversion, `|Ψ_spin| = √α · |Ψ_in|`
      (coefficient multiplies amplitude). Unitarity:
      `|Ψ_out|² + |Ψ_spin|² = |Ψ_in|²`.
- [ ] **B2c**: geometry-dependent reflection (local `α`, if variants warrant).
- [ ] Recommended start: **B2b**.

**Optional consistency observation (not a derivation).** The engine may
record three candidate ratios near a WC:

```text
r1 = |Ψ_spin|² / |Ψ_in|²   (Yee's spin energy fraction)
r2 = |Ψ_out|  / |Ψ_in|     (amplitude reflection ratio)
r3 = |Ψ_out|² / |Ψ_in|²    (energy reflection ratio)
```

Each is compared with the geometric `α`. A match is evidence that the
corresponding field quantity is what the geometric ratio describes, but
only on a variant that does not load `α` at the wave centre (B2c, if
implemented). On B2b the conversion is set to the loaded `α`, so r1
equals the loaded value by construction and r2, r3 follow from unitarity.
A mismatch for all three means the geometric `α` has no direct dynamic
counterpart in this engine; that is a valid observation, not a failure of
the plan.

### 2.3 — Longitudinal ↔ transverse coupling

- [ ] **B3a**: conversion at WCs only.
- [ ] **B3b**: volume conversion proportional to `|Ψ_long|²`.
- [ ] **B3c**: with relaxation (`Ψ_trans → Ψ_long` possible).
- [ ] Recommended start: **B3a** — local, clean, matches Yee.

### 2.4 — EMC density dynamics

- [ ] **B4a**: instantaneous (`ρ = ρ₀ − β_ρ|Ψ|²`).
- [ ] **B4b**: relaxation dynamics
      (`∂ρ/∂t = D∇²ρ − γ_ρ(ρ−ρ₀) − β_ρ|Ψ|²`).
- [ ] **B4c**: inertial dynamics (full wave equation for `ρ`).
- [ ] Recommended start: **B4a** for tests; promote to **B4b** for
      self-consistent solitons.

### 2.5 — Wave speed modulation

- [ ] **B5a**: `c²(ρ) = c₀² · ρ/ρ₀`.
- [ ] **B5b**: power law `c² = c₀² · (ρ/ρ₀)^n`.
- [ ] **B5c**: anisotropic `c(ρ, ∇ρ)`.
- [ ] Recommended start: **B5a** — matches M4.9.

### 2.6 — Density-modulated nonlinearity

- [ ] **B6a**: `F = γ_nl · (1 − ρ/ρ₀) · |Ψ|² · Ψ`.
- [ ] **B6b**: explicit profile `mod(r)` instead of local `ρ`.
- [ ] **B6c**: nonlinearity in `c²(ρ)` instead of in `F`.
- [ ] Recommended start: **B6a** — matches manuscript Variant B.

### 2.7 — WC motion rule

- [ ] **B7a**: `F = −∇ρ` (EMC density gradient).
- [ ] **B7b**: `F = −∇|Ψ|²` (energy gradient).
- [ ] **B7c**: `F = −∇(ρ + |Ψ|²)`.
- [ ] **B7d**: `F = 0` (control).
- [ ] Recommended start: **B7a** — matches push-out.

### 2.8 — WC topology

- [ ] **B8a**: `tetrahedron_10_locked` (r1 = 1λ, r2 = 2λ).
- [ ] **B8b**: `tetrahedron_10_unlocked` (legacy r1, r2).
- [ ] **B8c**: `golden_angle`.
- [ ] **B8d**: `bcc_lattice`.
- [ ] **B8e**: `line` (negative control).
- [ ] **B8f**: `random` (negative control).

### 2.9 — WC spacing

- [ ] **B9a**: sweep spacing at fixed topology.
- [ ] **B9b**: `n·λ` vs `(n+½)·λ`.
- [ ] **B9c**: perturbation ±10%, ±20%.

### 2.10 — K-selectivity

The sweep runs on both a conservative and a dissipative vacuum, and
measures two observables.

**Structural (V3, conservative).** For each K, run from three perturbed
initial conditions at matched initial energy. Measure:

- localization: does the configuration stay bounded, or spread?
- sphericity: does it stay compact?
- WC return-to-initial: after the perturbation, do the wave centres
  return to their starting configuration?
- configuration fidelity: does the final state resemble the initial
  topology, or has it drifted?

K = 10 is structurally selected if it is the only K whose configuration
survives all three perturbations.

**Energetic (V2 or V4, dissipative).** For each K, run from three
perturbed initial conditions and let the dynamics settle. Measure:

- final energy: is E(K = 10) below E(K = 9) and E(K = 11)?
- convergence: do the three seeds land in the same final state?

K = 10 is energetically selected if it has the lowest final energy and
its seeds converge.

- [ ] B10a: sweep K = 2..12 at fixed topology, spacing, coupling, on V3.
- [ ] B10b: same sweep on V2 or V4.
- [ ] B10c: K × topology sweep.
- [ ] B10d: K × spacing sweep.
- [ ] B10e: structural comparison (V3) — three initial conditions at
      matched energy.
- [ ] B10f: energetic comparison (V2/V4) — three initial conditions,
      final energies and convergence.

### 2.11 — Energy conservation verification

- [ ] **B11a**: measure `dE_total/dt` for isolated soliton, using the
      definition from Section 5.2 (`E_kin` without `/c²`, `E_grad` with
      `c²(ρ)`) and including the deformation term of the active variant.
- [ ] **B11b**: measure boundary flux.
- [ ] **B11c**: compare stable vs unstable K.

---

## 8. What is dropped from the old implementation

The following should **not** be ported. Each is listed with the reason.

| Old element | Reason for dropping |
|---|---|
| `interact_wc_dirichlet` | Hard pin, superseded by reflector (1.5, 2.2) |
| `interact_wc_neumann` | Hard pin variant, superseded |
| `interact_wc_soft` | Additive pin, at most one source variant, not the mechanism |
| `V_MODE = 1` (pure cubic) | Prosthesis, superseded by 2.6 |
| `V_MODE = 2` (quintic) | Prosthesis, superseded by 2.6 |
| `V_MODE = 3` (double-well) | Not relevant to EWT |
| `V_MODE = 4,5,6,7,9,10` | Simplified profiles, superseded by 2.4 |
| `energy_local_aJ` with hardcoded `base_frequency` | Superseded by `EnergyBudget` |
| `compute_force_vector` (`F = −∇E`) | Superseded by `F = −∇ρ` (2.7) |
| `DirichletBoundaryProcessor` (`ψ = 0`) | Superseded by `BoundaryCondition` (1.7) |
| Simple cubic Laplacian | To be replaced by BCC stencil if needed |
| `seed_wave` modes 0, 1 | Kept as utilities, not central |
| `detect_annihilation` | Deferred until reflectors work |
| `select_voxels` | Already removed |

**Kept from the old implementation** (as utilities, not as core):

- Idea of flux mesh (rendering, Block 3).
- Idea of granule motion (rendering, Block 3).
- `constants.EWAVE_*` (absorbed into `OpenWaveUnitSystem`).
- `m4_7_ewt_emergence_engine.py` formulas (absorbed into `GeometricConstants`).

---

## 9. Open questions and working assumptions

Each entry is either an open question (author-gated, do not resolve by
inference) or a working assumption (a draft answer that can be revised as
the work progresses). Assumptions are marked as such.

### Q2. Which vacuum implementation is physically correct?

Open question. The plan offers five (`V1`–`V5`, Section 5.4) and starts
with `V1` for engine tests and `V3` for K-selectivity. The choice is a
compromise: physical fidelity against numerical tractability. Author-gated.

### Q4. What is the correct unit system for research?

Natural units are recommended for tractability. The manuscript's
predictions are in SI. The choice affects how `γ` enters the dynamics. Not
blocking; the `UnitSystem` abstraction (item 1.0) makes the choice
revisitable.

### Q5. What is the correct definition of "stability"?

Several candidates: lifetime, localization, sphericity, frequency stability.
All are measured (item 1.9). The primary definition is author-gated.

### Q6. Should the WC motion be continuous or discrete?

Open question. Yee's picture suggests continuous drift toward amplitude
minima; "lock-in" language suggests discrete jumps. Both are testable
(item 2.7). Author-gated.

### Q7 (working assumption). K = 10: topological or energetic?

The two readings are distinguished by what the K-sweep can measure, and
that in turn depends on the vacuum implementation (Section 5.4).

- **On a conservative box (V3):** final energy equals initial energy by
  construction. Comparing final energies across K compares the seeds, not
  the physics, and "different seeds give different final states" is
  expected on any conservative dynamics — not a falsifier. The observable
  that does discriminate on V3 is **structural**: does the configuration
  stay localized, keep its shape, and avoid collapse under perturbation?
  If only K = 10 survives at fixed initial energy, the selection is
  structural.

- **On a dissipative box (V2, V4):** the dynamics can relax, so final
  energies and attractors are meaningful. Different seeds converging to
  the same low-energy state is evidence for energetic selection.

**Working assumption:** run the sweep on both, and let the two
observables stand as separate tests.

- Primary: structural comparison on V3 (localization, sphericity,
  WC return-to-initial after perturbation, configuration fidelity).
- Secondary: energy comparison on V2 or V4 (final energies across K,
  convergence to a shared attractor across seeds).

If the structural test shows K = 10 is uniquely stable, the selection is
topological or geometric regardless of the energy. If the energy test
shows K = 10 is the unique minimum on a dissipative box, the selection is
energetic. If neither holds, neither reading is supported by the engine
as written.

This is a working assumption, not a settled answer. Update as the sweep
runs.

### Q8 (working assumption). Does spin stabilise the soliton?

Draft: start with longitudinal-only dynamics. If K-selectivity emerges
without spin (item 2.10), spin is not necessary for the selection mechanism.
Add spin later (items 2.2b, 2.3) to see if it changes the picture.

The rationale: the K-selectivity question is separable from the spin
question. If spin turns out to be necessary, the test in 2.10 will show
different results with and without the transverse coupling. Until then,
starting without spin keeps the first round of tests simpler.

This is a working assumption, not a settled answer. Update if the K-sweep
or the stability metrics show that spin is load-bearing.

---

## 10. Recommended execution order

**Phase A — Engine foundation (Block 1.0–1.3)**

1.0 (UnitSystem) → 1.1 (Multi-field) → 1.2 (Trackers) → 1.3 (Multi-field
evolution)

**Phase B — Physics interfaces (Block 1.4–1.7)**

1.4 (Source terms) → 1.5 (Reflector interface) → 1.6 (WC motion) → 1.7
(Boundary condition)

**Phase C — Measurement (Block 1.8–1.9)**

1.8 (Energy budget) → 1.9 (Stability metrics)

**Phase D — Research infrastructure (Block 1.10–1.21)**

1.10 (Experiment runner) → 1.11 (Geometry provider) → 1.13 (Checkpoint) →
1.14 (Live monitor) → 1.15 (Logging schema) → 1.16 (Domain config) → 1.17
(Vacuum layer) → 1.18 (Diagnostics) → 1.19 (Seeds) → 1.20 (Sweep DSL) →
1.21 (Artifacts)

**Phase E — Physics variants (Block 2)**

In the order 2.0 → 2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6 → 2.7 → 2.8 → 2.9 →
2.10 → 2.11.

**Phase F — Rendering (Block 3, out of scope here)**

Port the flux mesh, granule motion, and interactive controls from the old
launcher. Only after the physics is validated in headless mode.

---

## 11. Glossary

- **EMC** — Elastic Medium Constituent. The spherical unit of the vacuum lattice.
- **BCC** — Body-Centred Cubic. The lattice geometry of the EMC arrangement.
- **WC** — Wave Centre. A point that reflects incoming waves into outgoing waves.
- **`K`** — Number of wave centres in a soliton. `K = 1` neutrino, `K = 10` electron.
- **`K²λ`** — Maximum standing-wave radius of a soliton with `K` centres.
- **`λ_ν`** — Neutrino wavelength. The fundamental length scale in natural units.
- **`ρ_E`** — Energy density (high inside a soliton).
- **`ρ`** — EMC packing density (low inside a soliton).
- **`N_ν,stat`** — Statutory background EMC density (undisturbed vacuum).
- **`N_ν,eff`** — Effective EMC density inside the soliton.
- **`X_eff`** — Geometric dilution factor. `X_eff = A_π · 3 · K_WC · √2 / C_unif`,
  with `C_unif = 1/K_WC + 1 + α/(π L_p)`. Converts `N_ν,stat` into `N_ν,eff`.
- **Push-out** — The mechanism by which `ρ_E` displaces EMC, creating `ρ < N_stat`.
- **Tail** — The far-field deficit `ρ(r) → N_stat (1 − r_s/r)`. Analytic;
  not simulated (Section 2.4).
- **`α`** — Fine-structure constant. Geometric value: `α = 1/(A_π − ε_M) ≈ 7.29733855e-03`.
  Loaded from `GeometricConstants`, used as the reflection coefficient at a WC.
  The dynamic interpretation (whether a field ratio coincides with this value)
  is a consistency observation, not a derivation.
- **`ε_M`** — Magnetic deficit. `1/(N_geom π³) ≈ 1/(8π⁷(1−ζ))`.
- **`A_π`** — Geometric core of the soliton. `4π³ + π² + π`.
- **`N_geom`** — Effective BCC stiffness. `8π⁴(1−ζ)`.
- **`γ`** — Nonlinear coupling. `1/ε_M`.
- **`β_ρ`** — Rate coefficient in the EMC density evolution (Section 2.3,
  item 2.4).
- **`γ_nl`** — Coupling coefficient in the density-modulated nonlinearity
  (item 2.6, variant B6a). Under B4a it contributes `γ_nl (β_ρ/ρ₀) Ψ⁵` to
  the equation and `− γ_nl (β_ρ/ρ₀) Ψ⁶ / 6` to the deformation potential.
- **`κ`** — Stiffness of the EMC density deformation. Enters `E_deformation`
  as `½ κ (ρ − ρ₀)²`.
- **`c_ρ`** — Characteristic wave speed of the density field, used by B4c.
- **NESS** — Non-Equilibrium Steady State. The soliton's dynamical regime.
- **Reflector** — A WC that satisfies `|Ψ_out|² + |Ψ_spin|² = |Ψ_in|²`.
- **Feature** — A typed object stored in `FeatureBag`, keyed by its class.
- **Processor** — A stateless pipeline stage that reads and writes features.
- **VacuumProvider** — The swappable vacuum layer (Section 5.4).

---

## 12. How to use this document

This document is a **working plan**, not a specification. It records:

- **Why** the tool exists (Sections 1–2).
- **What** the tool must express (Sections 3–5).
- **How** to build it (Sections 6–7).
- **What** to skip (Section 8).
- **What** remains unresolved (Section 9).

Update it as decisions are made. Each work item in Blocks 1 and 2 should be
promoted to a `tasks/m4_<n>_task_details.md` when it is picked up, with
pre-registered pass/fail criteria. The roadmap row in `m4_roadmap.md`
references that task document.

When a work item is complete, mark the checkbox and add a one-line note in
Section 15 (Changelog).

---

## 13. Document scope and precedence

**What this document is** — a design rationale and a work plan for the
pipeline_engine. It explains *why* the tool exists, *what* it must express,
and *how* the work is organised.

**What this document is not** — a roadmap, a task document, or a findings
note. It does not replace `m4_roadmap.md`, `tasks/m4_<n>_task_details.md`,
or `findings/`. Those documents carry the criteria, the numbers, and the
verdicts.

The correct flow for a work item is:

```text
this document  →  m4_roadmap.md row  →  tasks/m4_<n>_task_details.md
                (design intent)        (preview)               (the record)
                                       ↓
                              scripts/m4_<n>_*.py
                              data/m4_<n>_*.csv
                              plots/m4_<n>_*.png
                              findings/m4_<n>_*.md
```

When this document and a task document disagree, the task document wins. When
this document and the manuscript disagree, the manuscript wins. When the
manuscript and the model author disagree, the author wins.

---

## 14. TaskID mapping

The following TaskIDs are proposed for the roadmap. They are assigned in
creation order and are never reused. The list is a proposal until the pull
request that adds the rows; the IDs are allocated there and re-checked
against the live roadmap for collisions.

**Block 1 — Engine**

| Proposed ID | Item | Depends on |
|---|---|---|
| M4.20 | UnitSystem feature | — |
| M4.21 | Multi-field FeatureBag | M4.20 |
| M4.22 | Trackers per voxel | M4.21 |
| M4.23 | Multi-field evolution | M4.21 |
| M4.24 | Source terms (additive) | M4.23 |
| M4.25 | Reflector interface | M4.23 |
| M4.26 | WC motion (drift) | M4.25 |
| M4.27 | Boundary condition | M4.20 |
| M4.28 | Energy budget tracker | M4.22, M4.23 |
| M4.29 | Stability metrics | M4.22 |
| M4.30 | Experiment runner | M4.20 |
| M4.31 | Geometric constants provider | M4.20 |
| M4.32 | Checkpoint / restart | M4.21 |
| M4.33 | Live monitor | M4.22 |
| M4.34 | Research logging schema | M4.30 |
| M4.35 | Simulation domain configuration | M4.20 |
| M4.36 | Vacuum layer provider | M4.20 |
| M4.37 | Diagnostic hooks | M4.29 |
| M4.38 | Deterministic seeds | M4.30 |
| M4.39 | Parameter sweep DSL | M4.30 |
| M4.40 | Artifact versioning | M4.30 |

**Block 2 — Physics**

| Proposed ID | Item | Depends on |
|---|---|---|
| M4.41 | Soliton assembly contract | M4.23, M4.24 |
| M4.42 | Vacuum implementation variants | M4.36 |
| M4.43 | WC reflector variants | M4.25, M4.41 |
| M4.44 | Longitudinal↔transverse coupling | M4.25 |
| M4.45 | EMC density dynamics | M4.21, M4.27 |
| M4.46 | Wave speed modulation | M4.45 |
| M4.47 | Density-modulated nonlinearity | M4.45, M4.46 |
| M4.48 | WC motion rule variants | M4.26, M4.45 |
| M4.49 | WC topology variants | M4.23 |
| M4.50 | WC spacing variants | M4.49 |
| M4.51 | K-selectivity sweep | M4.47–M4.50 |
| M4.52 | Energy conservation verification | M4.28, M4.51 |

IDs `M4.1`–`M4.13` are used or reserved by the existing roadmap. IDs
`M4.14`–`M4.19` are currently unassigned. The proposed assignment continues
the sequence without collision.

---

## 15. Changelog

| Date | Change | Author |
|---|---|---|
| 2026-09-17 | Initial draft. | Lukasz Smolinski |
| 2026-09-18 | Renamed to `M4_PIPELINE_PLAN.md`. B1: CFL bound with `√3`. B2: item 2.13 removed; `α` treated as loaded geometric parameter; glossary and Section 2.7 updated. B3: kinetic term added to energy budget. Vacuum layer made swappable (V1–V5). Tail treated as analytic; wall peak not simulated. Section 4 shortened. Q1, Q3 removed. Q2 reformulated as vacuum-choice question. Q7, Q8 converted to working assumptions with drafts. Renamed `β` to `β_nl` / `β_ρ`. Added `X_eff`, `N_nu_eff`, `VacuumProvider` to glossary. | Lukasz Smolinski |
| 2026-09-20 | Section 4 rewritten: soliton neighbourhood simulated (`r_domain ~ 10 λ_ν`), soliton extent `K²λ` and the tail treated as analytic input. Section 2.4 header and body aligned. Section 5.2: equation stated in divergence form, dissipation ledger added for B4b. Item 2.2: coefficient multiplies amplitude; consistency observation conditional on not loading `α`. Item 2.10 rewritten: structural (V3) and energetic (V2/V4) tests. Q7 rewritten as two-observable test. Section 1.7 table row for `α` removed. Section 3.1: `r_core` labelled theoretical scale. | Lukasz Smolinski |
| 2026-09-20 | Round three. Section 1.5 and Section 5.2: equation in Euler-Lagrange form with the exchange term `c₀²(β_ρ/ρ₀)\|∇Ψ\|²Ψ`; the plain divergence form does not conserve the gradient energy when `c²` depends on the field. B4c: `E_deformation` gains the density kinetic term `½\|∂ρ/∂t\|²/c_ρ²`, deferred until added. Section 4: `r_domain` sized by half the largest wave-centre pair separation plus a buffer, with measured numbers. Section 2.7: reference to "item 2.2, variant B2b". Section 14 preamble: IDs allocated by the author at row creation. | Lukasz Smolinski |
| 2026-09-21 | Round four. R2 applied: `E_total` carries the deformation energy in every variant; the equation gains `− 2 κ β_ρ² Ψ³` under B4a. Section 5.2 states the budget as the sum of the active variant's terms (B6a rule). Section 1.5 equation aligned. Section 4: `r_domain` sized by the configuration radius about its centre, with the measured values. Glossary: `β_nl` removed; `γ_nl`, `κ`, `c_ρ` added. Item 1.8 and item 2.11 B11a aligned with the new definition. | Lukasz Smolinski |

---

*End of document.*
