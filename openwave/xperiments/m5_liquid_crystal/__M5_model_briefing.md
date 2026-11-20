# M5 Liquid Crystal (LC): Model Briefing

> **What M5 brings.** Particles as topological defects of a Landau-de Gennes matrix
> field (extended to a 4×4 tensor, with a Skyrme kinetic term): the electron is a biaxial hedgehog whose de Broglie clock is the
> energy-minimizing state, charge is the defect's topological winding, and the four
> forces plus gravity emerge from the geometry of the same field. It is OpenWave's
> most-validated model (live counts on the [`MODELS.md`](../../../MODELS.md) score-board).

![M5_hero](research/images/clock.gif)

## Identity

| Field | Value |
| --- | --- |
| Model ID | M5 |
| Name | Liquid Crystal (LC) |
| Author | Jarek Duda (Jagiellonian University, Kraków) |
| Author contact | GitHub [@JarekDuda](https://github.com/JarekDuda), for author-gated questions (definitions, intent, what the model does and does not claim); routing convention in [`dev_docs/CROSS_MODEL_TESTING.md`](../../../dev_docs/CROSS_MODEL_TESTING.md) § 6 |
| Key inputs | Manfried Faber (TU Wien) EM and core regularization; liquid-crystal discussions with [Samo Kralj](https://scholar.google.com/citations?user=Bojz5CkAAAAJ&hl=en) |
| Lineage | Landau-de Gennes nematics + Skyrme term + Einstein teleparallel gravity |
| Primary sources | Duda arXiv:2501.04036 (time crystal), arXiv:2108.07896 (superfluid / KG-around-hedgehog); Faber arXiv:2201.13262, Faber & Golubich [Phys. Rev. D 114, 014510](https://doi.org/10.1103/8zn4-rwth) (2026; preprint arXiv:2604.12021); liquid-crystal particle-analog experiments ([Wikipedia: Liquid crystal particle analogs](https://en.wikipedia.org/wiki/Draft:Liquid_crystal_particle_analogs)) |
| In-repo | `medium.py` + `engine1-4` + `_launcher.py` (production); `research/sandbox_v1..v11` (headless); [`__M5_course.md`](__M5_course.md) (intuition) |

## Model Profile (what it brings, short form)

| Attribute | M5 |
| --- | --- |
| Substrate | `M`, a 4×4 real symmetric tensor field (`M = O·D·Oᵀ`, `O ∈ SO(1,3)`); the eigenspectrum `D` is fixed only in vacuum (the potential minimum). ⚠️ `D` is the **η-spectrum** (the generalized spectrum of `η·M`, `η = diag(−1, 1, 1, 1)`), not the literal diagonal of `M`: carry the sign per the Vacuum row |
| Vacuum | the preferred η-spectrum `D = diag(g, 1, δ, 0)` (the author's `Λ` notation, see [`research/tasks/m5_27_convo.md`](research/tasks/m5_27_convo.md)), the minimum of the potential, SO(1,3) vacuum dynamics unifies EM (tilts) + QM (twists) + GEM (boosts). ⚠️ **The tensor itself is `M_vac = diag(−g, 1, δ, 0)`, time-time NEGATIVE**; `diag(+g, 1, δ, 0)` is NOT a vacuum (`V ≈ 1.05e6` at g = 8, machine-exact), so any background built from this row must carry that sign: [`research/m5_theory_canonical.md`](research/m5_theory_canonical.md) (covariant-vacuum row), measured in [`research/findings/m5_18_verification_note.md`](research/findings/m5_18_verification_note.md) § 5 |
| Particle | defect of `M`: a point-like topological charge (elementary electric charge), a topological vortex line (quark string) |
| Charge | integer winding of the director (Gauss-Bonnet), `\|Q\| = ±1`, additive |
| Derrick escape | time-periodic resonance (no static soliton; the stable object is 4D) |
| de Broglie clock | bounded, self-starting, frequency-rigid time crystal = the energy-minimizing state |
| EM | Maxwell from tilts of the 1-axis (two independent routes, at c) |
| Quantum | Klein-Gordon from biaxial twist, mass geometric (no added mass term) |
| Gravity | curved space (perpendicular to the local time axis) in flat spacetime; boost of the time axis → gravitoelectromagnetism (GEM), no dynamical metric required |
| Free parameters | δ (QM/EM contribution), g (EM/GEM contribution), 1-2 potential coefficients; calibration knobs: Faber r₀ (mass), Coulomb units |
| Lab anchor | hopfions/skyrmions in real media (Liu et al. 2026, Nature Physics); pilot-wave droplets (Bush-Couder) |
| Formal artifacts | every MODELS.md cell backed by a runnable script + note; negatives preserved (M5.2, M5.7) |
| Next falsifier | g-factor ≈ 2 (order confirmed), the absolute-ω calibration chain, neutrino δ_CP = 180° (JUNO/DUNE) |

## Decision-Relevant Attributes

Model-level attributes to weigh the column: parameter economy, the formal artifacts that
back the claims, and what would falsify the model next. (Held here for now; a consolidated
cross-model version may return to `MODELS.md` later.)

| Attribute | M5 |
| --- | --- |
| Free parameters | δ (QM/EM contribution), g (EM/GEM contribution), plus 1-2 potential coefficients; the boost dressing b enters the clock sector. Calibration handles: Faber r₀ (mass), Coulomb units<br>[`research/4c_convo_2026.06.08.md`](research/tasks/m5_4c_convo_2026.06.08.md) |
| Formal artifacts | Every claim backed by a runnable open script + research note; documented negatives (M5.2, the M5.7 nulls) preserved as results<br>[`research/archive/m5_summary_report.md`](research/archive/m5_summary_report.md) § Reproduction |
| Falsifiable near-term tests | Unit-free g-factor ≈ 2 from the fixed-clock electron (**RE-BASED ❌ at verified-L, M5.21.5 2026-07-21**: the first-principles bridge is now derived (q = e exact, no free factor) and g does NOT close (spans 8.5e-4-1.45); the earlier #219 "[1.97, 2.22] brackets 2.0023" is retro-flagged as a box-truncated observable through the underived K = 4/α; the test STANDS as the model's sharpest open falsifier, now riding the long-evolution texture read of the production fixed-J port); the absolute-ω calibration chain (Coulomb units + LdG-to-rest-energy). **Calibration Phase A+B (2026-06-17, #208):** the scale-free unit map has two free dials (`c` emergent, `ℏ`=action unit); **α = \|b\|/(ℏc) ≈ 1/178** from the Coulomb coupling (right order vs 137.036, pure-topology un-tuned, c-cone factor pending); the clock runs **28.2× below** the electron ZBW under the energy anchor (structural, ω rigid vs energy). Faber's `E₀=α·ℏc·π/(4r₀)` contains α, so the clock is the lone independent absolute-frequency test. **Lever budget (#217, 2026-06-17):** V-on is **null** on frequency (V is rotation-invariant; the clock is a pure rotation, so `∂V/∂clock=0` exactly), the **faithful F-commutator kinetic** is the one real lever but bounded **×3.04**, leaving **~9-20×** = the missing length anchor. **Length anchor (#218, 2026-06-17):** anchoring on the Faber `r₀` (length) gives ω **~36× HIGH** (mirror of the energy route's 28× low); the two anchors **BRACKET the ZBW** and their **geometric mean reproduces it to ~13%** (c=2). The model's particle obeys `E·r₀ = α·(π/4)·ℏc` **exactly** (the classical-radius relation, `r₀ = α·(π/4)·ƛ_C` ~174× below the Compton wavelength). **So the gap is a calibration SPLIT, not an irreducible deficit**: the ZBW scale is recoverable by anchoring energy + length **jointly** (the `E·r₀=const` line), not the energy postulate alone; the geometric-mean law awaits a mass-family test (**#220**). **Neutrino sector (#199 ✅, 2026-06-18):** the SO(3) route's **δ_CP = 180°** is a sharp near-term falsifier: JUNO/DUNE settling δ_CP far from 180° kills the pure-SO(3) neutrino structure (currently CONSISTENT, NuFIT 6.0 best fit ~177°)<br>[`research/m5_roadmap.md`](research/m5_roadmap.md) § ELECTRON-ID, [`research/findings/m5_8_2x_findings.md`](research/findings/m5_8_2x_findings.md), [`research/findings/m5_8_2y_findings.md`](research/findings/m5_8_2y_findings.md), [`research/findings/m5_8_2z_findings.md`](research/findings/m5_8_2z_findings.md) |

## Field Configuration of Particles

Standing demand of any particle model: *state the field configuration of each
particle, and say whether it uses topological vortices.* M5's answers:

| Particle | Field configuration in M5 | Topological vortex? |
| --- | --- | --- |
| Electron | biaxial hedgehog defect of `M`, winding `Q = -1`, with internal de Broglie clock | ✅ disclination hedgehog |
| Positron | the `Q → -Q` reflected defect | ✅ |
| Neutrinos | light, charge-0 loop of quark string; flavour = SO(3) spatial rotation | ✅ loop |
| Quarks | fractional-charge segment of a 1D quark string (fraction-of-π rotation) | ✅ string |
| Baryons / mesons | knots of quark strings (baryon = simplest knot; pion = twist / reconnection) | ✅ knot |
| Photon / EM wave | a twist-like wave, like the wake behind a marine propeller (at c) | ❌ (radiation, not a defect) |

The de Broglie clock is *derived*, not assumed: the Zitterbewegung frequency is the one
that minimizes the field energy (M5.8), exactly the "preferred time derivative". Neutrino mixing (PMNS, δ_CP) falls out of the SO(3) flavour rotation, a
testable parameter reachable in the model that mainstream physics reaches only through
large neutrino experiments.

## Implementation Status

The most-validated column in the coverage matrix; the live per-criterion counts are on
the [`MODELS.md`](../../../MODELS.md) score-board, and phase progress lives in
[`research/m5_roadmap.md`](research/m5_roadmap.md).

| Sector | Status |
| --- | --- |
| Electron (4 observables) | ⚠️ charge (the seeded integer holds through evolution; pulled apart and released with free faces the unit relaxes back whole, [M5.46](research/tasks/m5_46_task_details.md); a split with the separation held is not excluded); ⚠️ mass (one anchor; the clock quantity that would test it does not come out); 🚧 spin-½ statistics (the apolar 720° is an identity of the declared form); all three re-read 2026-11-18 ([T11](../../../dev_docs/tasks/t11_task_details.md)); μ ❌ RE-BASED + J ⚠️ (see Falsifiable tests row) |
| Coulomb 1/r | ⚠️ re-based 2026-11-18: on the verified-L stack an antipair attracts and annihilates, the 1/d law is not confirmed at reachable boxes, and the charge e is an anchor (lineage: R² = 0.978 on the one-constant director) |
| Maxwell EM | 🚧 re-based 2026-11-18: the homogeneous equations hold as identities of the declared frame (hydrodynamic dictionary, Faber curvature); no transverse wave is computed |
| Klein-Gordon (QM) | ⚠️ read in a time evolution on 2026-11-19 ([M5.47](research/tasks/m5_47_task_details.md), the time part supplied by the port): in the vacuum six branches follow ω² = k² + m² within 0.16 %, as the declared action implies; on the relaxed unit, in boxes up to 10, no single law holds (above a local cutoff the mid branch has c² near 0.87 and a mass from 34.5 to 8.9, against 60.09). Geometric mass from the biaxial twist stays a derivation (Fig 9) |
| de Broglie clock | ⚠️ re-based 2026-11-18. Retired-stack result: a bounded self-starting time crystal (M5.8-era quartic engine); ZBW scale recovered to ~13% (geo-mean of energy + length anchors). Era qualifier (2026-07-20): on the current certified verified-L stack the free clock does NOT self-start (free minimization gives ω\* = 0 or −∞, author-conceded); the clock is CONSTRAINT-CARRIED (fixed-J, ω\* = J/2kin, M5.21.9) with its thermodynamics measured exact. Second era qualifier (2026-07-24, M5.27): a background-oscillation drive of the spectrum does NOT supply the missing clock power (structural null: the drive cannot torque the clock frame on block-diagonal states); the coupling channel that could (the mixed (0,i) block) is unstable under the current L. Third era qualifier (2026-08-11, M5.21.15): the free-clock negative is now a THEOREM (the energy is exactly affine in ω² per configuration, so free minimization can never select nonzero ω, with or without the boost dressing), and the constraint-carried clock SURVIVES the guarded (1/g) dressing: interior minimum at nonzero ω at positive energy on the narrow guard, the energy sign pending the regularization potential |
| Neutrino mixing | ⚠️ PMNS from SO(3): tri-bimaximal + δ_CP ≈ 180° parameter-free; θ₁₃ ≈ 8.5° = the SO(3)-breaking |
| Lepton mass spectrum | ⚠️ mass law `E ∝ Λ³` fixed; the 1 : 5.9 : 15.1 hierarchy origin open |
| Gravity | ⚠️ GEM ∝ (b·g)² measured (boost-tilt route); no dynamical metric required (curved space ⊥ the local time axis, in flat spacetime) |
| Strong / weak | ⚠️ short-range running-coupling onset ✅; Cornell confinement + SU(2) chiral open |
| Quarks · baryons · mesons · nuclei | ⚠️ the toy census MEASURED the first baryon analogs (M5.22, 2026-07-30, audited 9/9): a protected proton-analog (\|Q\| = 1, vortex column + equatorial ring; identity author-confirmed) and a protected neutral state (column + two rings) whose rings carry EXACTLY opposite unit topological charges, a bound ring-antiring pair surviving every kick-apart probe (M5.22.1, 2026-08-01: its neutron-vs-dineutron identity is the author-gated Q40); mass ordering neutral > proton, δ-robust, though the ratios do not converge by δ-steps alone (the Q33 bridge holds the quantitative claims). A deuteron candidate exists at n = 32 (electric +1, two rings, 28% below its constituent sum; electric-quadrupole sign NEGATIVE vs the physical positive; unsettled at n = 48). The in-baryon quark-shift cost measured 50-57× below the inter-hedgehog string tension (Q38 first data point). Nuclei remain the most practical direction |
| Orbital quantization | 🚧 a different level: the coupled pilot wave set up by the electron clock |
| Production launcher engines (interactive) | ✅ (2026-07-19, M5.24) the launcher runs the VERIFIED-L canonical stack as its `canonical` integrator: η-bracket curvature + the universal spectral potential on the symmetrized stencil, covariant vacuum, the certified full-3D τ-step, a FIRE relaxer (frozen-time-row recipe, topology-holding) and a boundary sponge; gated 14/14 against the audited M5.21.3 reference, so the interactive demos show the same physics the research stack measures (including its instabilities). The earlier era paths (the Eq.18 leapfrog, the M5.8.2c 4D experiments) remain as labeled modes. **The fixed-J ZBW clock is LIVE (2026-07-24, M5.23.1)**: the launcher's RELAX → SET-J → EVOLVE flow carries the M5.21.9 constraint-carried isorotation state as simulated dynamics only (gated 12/12 against the research reference, Legendre closure reproduced; the carried charge ⟨Ṁ, a0(M)⟩ is the hold observable, and the on-screen axis rate is ω\*/‖a0_raw‖ by the flow-normalization convention). **The render package landed (2026-07-24, M5.23.2, gated 13/13)**: converged research endpoints load from disk as LIVE launcher states (npz loader with the measured covariant-convention pin; the certified electron, and the μ/τ census states, are now on-screen objects), the energy-density ISOSURFACE renders from the live Hamiltonian density (taichi marching tetrahedra, watertight on the sphere anchor; the electron's surface is the rod tube / equatorial ring, open at the box because the rod runs through: that is the physics), and the J/μ twist demo shows the disclination rods twisting under the carried clock (outer ring rows visibly lag: the clock envelope scales the local rate) |
| Verified-L minimization chain (2026-07-16) | ⚠️ the author's minimization-first electron chain ran end-to-end: the author's structure claims land asymptotically (axis split ∝ δ^1.03, core pinned by g = 32) but toy-parameter statics does not converge, the endpoint is a saddle against time-mixing, and rigid-rotation minimization cannot supply J; the same-day spec-review retrospective (audited) bracketed the mechanism: soft potential = the paper-anticipated amplitude escape, hard spectrum-pin = Derrick expansion; the stable window is a Faber-type virial balance, and the potential FORM (the paper's own open choice, incl. its det-constraint hedge) is the live question to the author. **The 3D-first run (M5.21.2, 2026-07-17, audited 8/8) sharpened it**: topological protection is real but box-fed; the three rotation seeds give three distinct levels with the electron lowest (the qualitative 3-lepton signature, cross-instrument robust); the charged disclination ring (third defect type) ties the point hedgehog within instrument limits; and NO stencil-consistent minimizer exists at toy parameters (each discretization hides curvature in its blind directions), making the potential-form question one of WELL-POSEDNESS. **M5.21.2b (same night, audited 8/8) resolved it constructively**: the stencil-symmetrized functional gives genuine consistent stationary minima; the author's Eq-12 eigenvalue-penalty potential is the term set whose bare minimum is also virial-balanced; the three rotation minima persist (A < C < B, B/C certified stationary); the point hedgehog and the charged ring merge into ONE object whose core is two braided +½ vortex lines at a dynamically selected transverse scale growing as δ falls (the Q30/Q31 answers). **The 4D lift (M5.21.3, 2026-07-18, audited 6/6): the static electron is a genuine saddle toward the time sector, yet free 4×4 minimization lands no stable dynamical state (rotations cost energy, the boost slope never lands); the surviving route to spin is the fixed-J isorotation state with the measured clock inertia.** **The decay + verification rounds (M5.21.6 + M5.21.8, 2026-07-18/19, both audited): in the honest free arena the μ-candidate DECAYS to the electron by exactly the author's rotation mechanism (rotation-dominant, no melt; energy ledger closed to 3 decimals with 23% radiated; one equatorial ring released vs the conjectured two; the τ-candidate drains at this box size), and the author's analytical benchmark was verified with an independent pipeline: the gravitational-mass law m* ≈ 1/g is essentially exact (0.009%) and lattice-real (the dressed family tracks it at 0.83× across g = 8-64), while NO finite nonzero clock ω exists in the author's own formulas or on the lattice (ω* = 0 or −∞, incl. at physical (1e10, 1e-10)): the fixed-J constraint-carried state is the two-stack consensus route to spin, with the author's Larmor protocol as the staged acceptance test.** **The fixed-J round (M5.21.9, 2026-07-20, audited + round-2 addendum): the constraint-carried electron EXISTS and HOLDS (three J rungs, rel_move ≤ 1.3e-4, a t = 80 dynamics-grade window) with exact clock thermodynamics, dE/dJ = ω\* closing at ~1%: the literature-standard isorotating-soliton construction (Coleman → Radu-Volkov). The Larmor LINEAR read proved unmeasurable in the first instrument configuration (the preparation-texture systematic dominates, measured decisively at ± field pairs; the adiabatic ramp-on redesign rides the production port), while a clean QUADRATIC field-induced clock-drift slowdown emerged as the first resolved field coupling. The author's negative-δ suggestion verified as bounded-but-still-static (both runaway channels flip bounded at δ < 0; free descent still collapses through the boost channel at both signs, profile-independent).** **The decay-grade box extension (M5.21.10 + the Q35 read M5.21.12, 2026-07-20, audited): at n = 64 the electron HOLDS in free dynamics (4.4% absorbed over t = 150) while the τ-candidate's f48 "drain" is SUPERSEDED: it disintegrates through the rotation channel (20°, half its budget radiated); the μ-decay ejects a real symmetric direction-separated pair whose loop identity the tracer then RESOLVED (M5.23.2, 2026-07-24: two mirror on-axis INTERIOR OPEN filaments that evaporate, never closing and never reaching the box: the two-closed-loops picture unsupported at 64³ grade, sub-voxel closure not excludable); the point-vs-ring question closed as a measured near-degeneracy; and the fixed-J construction got its literature spine: a first-order ghost sector cured by the energy-Casimir method (the standard isorotation practice), with the radiation window ω\* < μ imported as the sharpest standing falsifier.** **The μ + g closure attempt (M5.21.5, 2026-07-21, audited): the FIRST-PRINCIPLES bridge was derived from the Coulomb anchor (g = μE/2πS, the anchored hedgehog carries exactly charge e, the length unit cancels, audit-exact) and the μ channel exists with a radially-converged envelope read, but g does NOT close: the moment is a parity-cancellation residue tracking preparation basin and transverse texture across 4 orders (box ladder non-monotone), g spans 8.5e-4 to 1.45, and the canonical-era g = 1.97 bracket is retro-flagged as a box-truncated read through the underived K = 4/α; the honest 4-observable electron stands at mass, charge, J validated, μ measured-but-fragile, g open, with the long-evolution texture read (the production fixed-J port) as the named next rung.** **The two-particle program (M5.21.4, 2026-07-21, audited two rounds, the series' physics close): the full 3D dynamical antipair capture-to-annihilation film DELIVERED at verified-L rigor with both ledgers (charge sigmoid ±1 → ≤ 0.005, energy physical to 0.1%): the mechanism is CONDUIT ANNIHILATION (charges cancel in place through the connecting topological line, ballistic core-walk refuted); there is no static separated-antipair regime (descent IS attraction), the seed-level binding tracks the derived zero-free-parameter Coulomb form as an outer-window trend, and the same-charge sector produced the structural surprise: separated like charges are STRING-CONFINED (E_int linear in d, two boxes) and merge into a central charge-2 ring compound, the model's first measured multi-charge bound state and the direct on-ramp to the nuclei-as-vortex-knots program.** **The background-scalar pilot (M5.27, 2026-07-24, gates 8/8 + independent audit 5/5): the third fork branch closed with a STRUCTURAL null: a prescribed background scalar driving the spectral targets owns the eigenvalues exactly (time-time excursion = drive amplitude, slope 1.000) yet cannot torque the eigenframe carrying the clock (the drive force commutes with M on the block-diagonal sector, which is also dynamically invariant), so no Arnold tongue exists (NULL 40/40, 7 false positives refuted); the mixed (0,i) block is thereby identified as the load-bearing time-sector coupling channel, and it is unstable under the current L: the time sector is a Lagrangian-level problem, not a coupling-design one.** |

Deep dives: [`research/m5_theory_canonical.md`](research/m5_theory_canonical.md) (**the
canonical working-recipe registry**: verified equations, locked parameters, recipes of
record, anti-recipes; consolidated from the method notes, maintained at every task
review), [`research/archive/m5_summary_report.md`](research/archive/m5_summary_report.md) (results of
record), [`research/m5_roadmap.md`](research/m5_roadmap.md) (full program),
[`research/m5_question_tracker.md`](research/m5_question_tracker.md) (open questions).

## Roadmap

| Next | What lands |
| --- | --- |
| Lepton hierarchy origin (#200) | why the eigenvalue ratios give 1 : 5.9 : 15.1 (the mass law `E ∝ Λ³` is already fixed) |
| Absolute-ω calibration (#220) | close the ZBW scale from a mass-family test + the Coulomb-unit map |
| Neutrino field follow-up (#200) | the charged-lepton second-rotation matrix behind θ₁₃ |
| Quarks / hadrons / nuclei (15a) | Cornell confinement via 1D vortex string; baryons / mesons as knots; nuclei (the most practical direction) |
| Weak force | a clean SU(2) chiral reconnection mechanism; the static-kick beta-decay probe measured NEGATIVE at the 3×3 truncation (the neutral states are deep minima), and the constant-ω long-axis-twist arm of the dynamical sector also measured negative on all four baryon-sector states (M5.22.4: no minimum at ω > 0, the electron decoupling reproduces exactly); the mechanism now rides free full-4×4 dynamics and/or physical parameters |
| Orbital quantization | pilot-wave standing-wave resonance around the electron clock |

## Help Wanted

M5 is an open column in an open arena. Anyone can extend it, and a documented *negative*
(a runnable script showing "this doesn't work, here's why") counts as much as a positive.

| You can contribute | How |
| --- | --- |
| A new validation of an M5 cell | runnable script + short note, pass/fail against the shared criteria |
| A harder falsifier | pick an open ⚠️ / 🚧 cell (hierarchy origin, weak force, quarks) |
| A rival field configuration | propose an alternative ansatz for a particle, test it head-to-head |
| A whole new model | a new `openwave/xperiments/<model>/` column, same shared criteria |

Flow: open a discussion → fork → branch → PR with a DCO sign-off
(`git commit -s`), under Apache 2.0. Light review checks only reproducibility + honest
documentation, not orthodoxy. Start here: [`../../../MODELS.md`](../../../MODELS.md)
§ Contributing, [`../../../ONBOARDING_MODELS.md`](../../../ONBOARDING_MODELS.md),
[`../../../CONTRIBUTING.md`](../../../CONTRIBUTING.md). Model discussion runs in the
Models-of-Particles group.

## Rich Context for Deep Reader

This is a top level documentation and orientation content. For additional context on this model, a detailed read in the /theory and /research folders is recommended, as well as the production files in this model root folder, that may contain the canonical full PDE implementation of the theory.
