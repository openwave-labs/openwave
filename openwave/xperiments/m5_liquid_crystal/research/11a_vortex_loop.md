# M5.11 , the regularized, stable, stationary topological vortex loop (the real simulation)

> **Purpose.** Build the simulation Dr. Jarek Duda expects: particles as REGULARIZED topological defects of the
> Landau-de Gennes tensor field, relaxed to STATIONARY energy-minimizing configurations, with MASS read off as
> the regularized field energy. First reproduce FABER's electron (the regularized hedgehog → 511 keV) to prove
> the machinery, then the new piece , the neutrino as a regularized chiral vortex LOOP/KNOT, mass = loop-length
> density , then re-derive the mixing on the real relaxed loops. This grounds the symmetry/overlap result of
> [`10e_findings_N4c.md`](10e_findings_N4c.md) in real dynamics and directly answers Duda's 2026-06-22 critique.
> Findings as we execute: [`11b_findings.md`](11b_findings.md). Code + checkpoints: `sandbox_v11/`.
> Convention: index-0 (`D = diag(g, 1, δ, 0)`, `η = diag(-1,1,1,1)`).

## 0. The challenge (Duda, 2026-06-22, verbatim)

> "I am trying to read these Python files, but they look very far from simulations I was expecting. E.g. for
> neutrinos require topological vortex loop configurations, with main problem being regularization requiring
> potential ... the code is much too simple ... while these results look great, personally I am very far from
> trusting them."

He is right. [`sandbox_v10`](sandbox_v10/) computes mixing from SEEDED loop ansatze via symmetry + overlap; it
never regularizes, relaxes, or stabilizes a loop. M5.11 does the real thing , and it is grounded in Duda's OWN
theory (his slides [`../theory/liquid_crystal_particles.pdf`](../theory/liquid_crystal_particles.pdf), the
[`4a`](4a_convo_2026.05.12.md)/[`4c`](4c_convo_2026.06.08.md)/[`4d`](4d_convo_2026.06.11.md) exchanges, his
[`9a`](9a_lepton_mass_planning.md) review). **Goal: a simulation a working physicist trusts. No cut corners.**

## 1. Duda's theory (what we are simulating, in his own terms)

Particles are non-perturbative field configurations of a real symmetric tensor field `M(x) = O D O^T`,
`D = diag(g, 1, δ, 0)` (the preferred biaxial "shape": `g ≫ 1 ≫ δ > 0`; index-0, time/g axis = 0). The director
tilts → EM, the second-axis twists → quantum phase (Klein-Gordon), the time-axis boosts → GEM gravity. The
Lagrangian (Duda's slides, his Mathematica on p.19 of [`liquid_crystal_particles.pdf`](../theory/liquid_crystal_particles.pdf)):

```text
ℒ = − F_μναβ F^μναβ − V_Higgs(M) ,        F_μναβ = [∂_μ M, ∂_ν M]_αβ   (matrix commutator of gradients)

  Duda's exact Hamiltonian (his Mathematica, regularized 2-charge calc):
    H = Σ_{i<j} ‖[∂_i M, ∂_j M]‖²_F   +   V_Higgs(M)

  V_Higgs = the Landau-de Gennes form  (A/2)Tr(M²) − (B/3)Tr(M³) + (C/4)(Tr M²)²
            OR the eigenvalue-pinning  Σ_i (λ_i − Λ_i)²  ,  Λ = (g, 1, δ, 0)   (Duda: "tough choice!")
```

This is **Faber's quantized-EM model** (`ℒ = −(αℏc/16π) R_μν R^μν`, `R_μν = Γ_μ × Γ_ν`,
`Γ_i = (∂_i n) × n`) extended from the uniaxial director `n` to the biaxial tensor `M` and to 4D
(teleparallelism). It is M5's production functional already (`engine2_pde.V_M` + the signed commutator
curvature). The scales are Duda's: `g ~ 10^{10}`, `δ ~ 10^{-10}`, `g·δ ≈ 1` (NOT `δ = 0.3`, which he flagged);
the N1 graded-precision method carries the `10^{20}` range.

### Regularization (Duda's "main problem")

The Higgs potential lets the order parameter MELT at the defect core (for a hedgehog `n(0) = 0`, via
`V = (‖n‖² − 1)²`; for the tensor, the eigenvalues deform from `(g,1,δ,0)`), trading divergent gradient energy
for finite bulk energy and setting the core size. Faber's specific core potential is `Λ = q0⁶/r0⁴` (`q0` = the
SU(2) scalar part `cos α`); the soliton rest energy is `E0 = (α_f ℏc/r0)(π/4)`, which set to `m_e c² = 0.511 MeV`
fixes the core radius `r0 = 2.2132 fm` (the classical electron radius × π/4). The recent high-precision lattice
paper **Faber & Golubich 2026** ([`../theory/faber_universe_2025.pdf`](../theory/faber_universe_2025.pdf),
arXiv:2604.12021) extracts the two-soliton dipole energy `E(d) = 2 m_e c² − α_sol ℏc/d` with
`α_sol ℏc = 1.4387(8) MeV·fm` (vs the Coulomb `1.43996`), the inverse fine-structure constant
`α_sol⁻¹ = 137.1(1)`, and the QED running of `α(d)`, via conjugate-gradient energy minimization on a cylindrical
lattice. **Reproducing those numbers is the canonical validation** , it answers "too simple" directly, and it is
Faber's own work (Duda's collaborator). Note: Faber's solitons are NOT Skyrmions (the Skyrme model is
short-range; these reproduce long-range Coulomb via Gauss's law) , the 4th-order curvature is Skyrme-LIKE but
the physics is QED.

### The objects (Duda's defect classification, his slide 10: `r = s + d + 1`)

| Particle | Defect | Stabilized by | Mass |
| --- | --- | --- | --- |
| electron | regularized **hedgehog** (`s=2`, point) | Higgs-regularized Skyrme curvature (Faber) | `∫ ½\|E\|² = 511 keV` |
| neutrino | regularized **vortex LOOP / chiral knot** (`s=1`, line/loop) | same functional + the chiral nematic vortex-knot structure | **mass/length density × loop length** (Duda 2026-06-21) |
| nuclei | **linked vortex knots** (Borromean / Efimov halo) | linking number | sum of loop energies |

The neutrino is specifically the chiral nematic vortex knot family (Duda's Abrikosov-vortex / QCD-flux-tube /
Smalyukh "fusion and fission of particle-like chiral nematic vortex knots" slides). Mass scales ~linearly with
loop length (a density), which is why flavour oscillation conserves energy , the loop changes length.

## 2. Derrick's theorem , RESOLVED by the functional itself (the key physics)

Earlier we feared a static loop must collapse (Derrick). That is WRONG for this functional. Under `x → λx`:
the plain-gradient energy scales as `λ`, the **commutator-curvature `‖[∂M,∂M]‖²` (FOUR derivatives) scales as
`λ^{-1}`**, the potential as `λ³`. So `E(λ) = λ E_grad + λ^{-1} E_curv + λ³ E_pot` has an interior minimum:
the 4th-order curvature term (the Skyrme term, which IS Duda's `F²` curvature) provides the outward pressure
that balances collapse. **Static, finite-size, stable regularized solitons EXIST** , Faber's electron is the
proof of existence. So the target is to FIND the stable regularized loop by energy minimization, not merely to
show collapse.

| Layer | Role |
| --- | --- |
| Skyrme-like curvature `‖[∂M,∂M]‖²` + Higgs `V` | the STATIC stabilizer (Derrick-evading) + core regularization. Gives the regularized hedgehog/loop and its rest mass. |
| the de Broglie CLOCK (Zitterbewegung) | an ADDITIONAL dynamical energy-LOWERING (Duda: "oscillation reduces energy; the minima are the preferred frequencies"). Our M5.8 result (the `(0,α)` negative-energy boost fuel). It does NOT prevent collapse; it lowers the mass and gives the resting oscillation = neutrino flavour oscillation (loop-length change). |

The earlier `dE/dL = +6.74 > 0` ([`n2_closed_loop.py`](sandbox_v10/n2_closed_loop.py)) was a SEEDED, V-off,
unrelaxed loop , exactly the regime where Derrick bites. With the full Skyrme + Higgs functional and energy
minimization, a stationary loop should exist. M5.11 demonstrates it.

## 3. The phased plan (each phase gated against a KNOWN result; reproduce Faber first)

| Phase | What | Validation gate |
| --- | --- | --- |
| **P0 , infrastructure** | turn the Higgs/LdG potential ON in the index-0 engine (`V_M`, the hardest numerical step); the right scales (`g~1e10, δ~1e-10`) via the N1 graded method; add a true energy MINIMIZER (FIRE / L-BFGS to `\|δE/δM\| → 0`). | minimizer descends monotonically to a stationary point; `V_M` reproduces the analytic LdG free energy. |
| **P1 , reproduce FABER's electron** | the regularized uniaxial hedgehog under Higgs `V = (‖n‖²−1)²`: melt the core (`n(0)=0`), minimize, integrate the field energy. | **`∫ ½\|E\|² = 511 keV` at the Faber core radius; running coupling `α(d)` curve; Coulomb `E(d) ≈ const + C/d`.** Match [`../theory/faber_universe_2025.pdf`](../theory/faber_universe_2025.pdf). THE trust-rebuilder. |
| **P2 , the vortex LOOP (heliknoton)** | add the **chiral term `q0`** to the functional + a **helical far-field background**, seed an elementary heliknoton (Hopf `Q=1`), relax with the Taichi-AD minimizer. (Run-1 plain disclination ring dissolved; run-2 smooth Hopfion expanded , both because the functional lacked the chiral term; the Smalyukh heliknoton recipe supplies it.) | stable localized knot at size `~p`; Hopf index `Q=1` (linked preimages); `⟨δF/δn⟩→0`. Full build spec + caveats: [`11b_findings.md`](11b_findings.md) § "P2 fork RESOLVED". |
| **P3 , stability + the clock** | prove stable (Hessian `≥ 0` / real-time Minkowski evolution over many periods); add the M5.8 clock dressing (the `(0,α)` boost fuel) → the oscillating loop lowers energy. | no collapse mode; the loop persists; the clock lowers the energy (the de Broglie oscillation), `ω` measured. |
| **P4 , mass from the loop** | mass = regularized loop field energy; the mass/length density; vary loop size/knot → the 3 neutrino masses. | the spectrum + `Δm²` hierarchy (the N4c spectrum was ~6× too compressed); the 6.2 pm loop scale; honest pass/fail vs data. |
| **P5 , the parameter search (Duda's assignment)** | find the Higgs coefficients `(A,B,C)` / `Λ` and `g, δ` that give physical lepton + neutrino masses , the work Duda explicitly handed us ("you should start here, finding these parameters/details for agreements"). | the parameters that reproduce the masses + Faber's electron simultaneously; reported as the deliverable. |
| **P6 , mixing on real loops** | recompute the PMNS overlap on the RELAXED stable loops at 3 SO(3) orientations (not seeds). | the [`10e`](10e_findings_N4c.md) mixing re-derived from real solutions; whether the substrate now does work. |

P0-P1 are tractable and decisive (reproducing Faber is the credibility gate). P2-P3 are the research core
(stable vortex loop + clock). P4-P5 are the mass + parameter search Duda assigned. P6 closes back to mixing.

## 4. Infrastructure: reuse vs build

**Reuse (the index-0 engine already implements Duda's functional):** `engine2_pde.V_M` (the LdG `a Tr M² − b Tr
M³ + c (Tr M²)²` + `dV_M`), the signed commutator curvature `compute_curvature_flux(_4d)`, `signed_dot4`
(`η = diag(-1,1,1,1)`), the seeders (`seed_hedgehog_M`, `seed_biaxial_hedgehog_M`, `seed_dressed_hedgehog_M`
with the clock), the 4D Minkowski + constrained integrators, `compute_winding_number`, `compute_energyH_density_M`,
the N1 graded-precision method. The M5.8 clock machinery (`sandbox_v8`/`vn`, now index-0) is P3's clock layer.

**Build (the gaps the engine-inventory flagged):**

| Add | For |
| --- | --- |
| `seed_vortex_loop_M(center, R, axis, twist/chirality, ...)` , a closed-disclination LOOP | P2 (engine has only point/line defects) |
| a true energy MINIMIZER (`relax_to_equilibrium`: FIRE/L-BFGS, `‖∇E‖ < tol`, line search) | P0-P2 (engine has dissipative relax + leapfrog, no stationary-state solver) |
| the eigenvalue-pinning Higgs variant `Σ(λ_i − Λ_i)²` (alongside the a/b/c LdG) | P0/P5 (Duda's alternative potential) |
| Hessian / stability tool (2nd variation, mode spectrum) | P3 |
| Faber validation harness (`∫½\|E\|²`, running coupling) | P1 |
| loop diagnostics (radius, length, linking/Hopf charge) | P2-P4 |

## 5. Do we ask Duda first? , NO, he already assigned the parameter search

Figure it out ourselves. His theory (slides + exchanges) answers every STRUCTURAL question: the functional, the
Higgs regularization, the scales, the defect classification, mass-as-field-energy/loop-length. The one genuinely
open piece , the exact potential coefficients , **Duda explicitly handed to us**:

> "finding the exact ones turned out surprisingly difficult, even worse for details of potential , you should
> start here, finding these parameters/details for agreements." (Duda, 2026-06-21)
>
> "There are required serious simulations ... we need the details: parameters, potential, minimizing energy
> configurations." (Duda, 2026-06-20)

So we do not ask before starting; we DELIVER what he asked , parameters, potential, energy-minimized
configurations, starting from Faber's electron. The reserved question (only if P2-P3 genuinely stalls) is sharp
and shows competence: whether the neutrino loop is a plain disclination loop or a specific chiral knot
(Hopf/linking number), since his slides show both. Backed by a working Faber-electron reproduction, that question
rebuilds trust; asking up front would not.

## 6. Honest risks / unknowns

| Risk | Status |
| --- | --- |
| The parameter search is genuinely hard (Duda: "surprisingly difficult") | accepted; it IS the assigned work; report what we find, including partial. |
| Turning `V_M` ON without destabilizing the leapfrog (the engine's hardest step) | known M5 roadblock; the minimizer route (relax, not evolve) sidesteps the leapfrog instability for the static phases. |
| A stable vortex LOOP (vs the hedgehog) may need the chiral/knot structure | **RESOLVED (direction):** the Smalyukh **heliknoton** recipe ([`../theory/liquid_crystal_defects/`](../theory/liquid_crystal_defects/) 2020 Tai thesis) , a smooth Hopf knot is stabilized by the **chiral term `q0` + a helical background**, NOT a Skyrme term (our run-1 ring dissolved + run-2 Hopfion expanded, both for lack of the chiral term). Build spec'd in `11b`. Caveat: biaxial-material singular knots stay hard (thesis p.132) → smooth heliknoton first. |
| The masses (P4) may stay in tension with `Δm²` | the N4c spectrum was ~6× too compressed; reported honestly. |
| Cost: P2-P3 energy minimization + evolution are multi-hour GPU research runs | accepted; full machine, run for the time it needs. |

## 7. Deliverables

| Artifact | Where |
| --- | --- |
| This plan | `11a_vortex_loop.md` |
| Findings (filled per phase, mirrors `10e`) | [`11b_findings.md`](11b_findings.md) |
| Code (one script per phase) + checkpoints | `sandbox_v11/` (`v11_p0_minimizer`, `v11_p1_faber_electron`, `v11_p2_loop`, `v11_p3_stability_clock`, `v11_p4_mass`, `v11_p5_params`, `v11_p6_mixing`) |
| Figures | the regularized core, the Faber 511 keV field-energy curve, the stationary loop, the stability evolution, the mass family, the re-derived mixing |

## 8. Theory sources reviewed (2026-06-22)

[`../theory/liquid_crystal_particles.pdf`](../theory/liquid_crystal_particles.pdf) (READ IN FULL , the functional,
Faber regularization, the vortex-knot neutrino, the clock) · [`../theory/faber_universe_2025.pdf`](../theory/faber_universe_2025.pdf)
(READ IN FULL , Faber & Golubich 2026 / arXiv:2604.12021, the exact P1 target: `r0=2.2132 fm`,
`α_sol ℏc=1.4387 MeV·fm`, `α⁻¹=137.1`, CG minimizer) · [`../theory/time_crystal_toy_model.pdf`](../theory/time_crystal_toy_model.pdf)
(READ IN FULL , Duda arXiv:2501.04036, the P3 clock: `−αR²` curvature + `+βR⁴` saturation → finite `ω=√(70α/(96β−35α²))`,
the time crystal; the model "combines Skyrme + Landau-de Gennes") · [`../theory/liquid_crystal_model.pdf`](../theory/liquid_crystal_model.pdf) + the Wolfram time-crystal writeup (the broader-model + online-discussion versions of the above; covered by proxy) ·
the [`4a`](4a_convo_2026.05.12.md)/[`4c`](4c_convo_2026.06.08.md)/[`4d`](4d_convo_2026.06.11.md)
Duda exchanges + [`9a`](9a_lepton_mass_planning.md) review · the M5 framework
([`1a_lagrangian_framework.md`](1a_lagrangian_framework.md), [`5a_lagrangian_evolution.md`](5a_lagrangian_evolution.md),
[`99_summary_report.md`](99_summary_report.md)) · the index-0 engine (`engine2_pde.py` `V_M` + curvature).

## Cross-refs

[`10e_findings_N4c.md`](10e_findings_N4c.md) (the symmetry result this makes dynamical) ·
[`9a_lepton_mass_planning.md`](9a_lepton_mass_planning.md) (the lepton-mass program, #200) ·
[`sandbox_v8`](sandbox_v8/)/[`sandbox_vn`](sandbox_vn/) (the M5.8 clock machinery) ·
[#199](https://github.com/openwave-labs/openwave/issues/199) · [#236](https://github.com/openwave-labs/openwave/issues/236) (HELD).
