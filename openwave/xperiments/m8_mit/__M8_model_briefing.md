# M8 MIT (Mode Identity Theory): Model Briefing

> **What M8 brings.** Particles as samples of a single standing wave on a fixed
> topology: the three-sphere, its binary icosahedral quotient S³/2I, and a Möbius
> edge. It is a top-down structural model, not an emergent-dynamics one. Where the
> other columns start from a Lagrangian and evolve fields until particles appear, MIT
> starts from the topology and reads the spectrum off it: the couplings, the fermion
> mass ratios, and Λ come from representation theory on S³/2I (McKay distance,
> Reidemeister torsion, the 120-cell). It is strong on the origin of the numbers and
> does not yet carry field dynamics of its own; supplying them IS the M8 program
> ([`research/m8_background.md`](research/m8_background.md)).
>
> **Status: research mode first.** This column was scaffolded by the
> maintainers on 2026-07-21 from the author's onboarding proposal
> ([discussion #312](https://github.com/openwave-labs/openwave/discussions/312));
> the content below is adapted from the briefing the author submitted there. The
> author owns the science; the platform supplies the arena, the standards, and the
> cross-model pointers ([`research/m8_platform_pointers.md`](research/m8_platform_pointers.md)).
> The first sector is now VALIDATED in-platform: M8.1 (2026-07-21) verified the
> twisted-Möbius first-eigenvalue theorem at 10-digit precision, blind and audited.
> The field-dynamics half has its contract: the pre-registration the Lagrangian
> survey is graded against was locked before any numerics existed to tune it toward
> (M8.2, the author's first pull request). Task-by-task state is in the roadmap, not
> restated here. MIT's other results remain analytic or externally
> computed, and three of them are pre-registered documented negatives the author
> reports as results. Work runs headless (scripts + research notes) first; the 3D
> rendering port is a later stage, gated on field dynamics validating in-platform
> ([`research/m8_roadmap.md`](research/m8_roadmap.md) M8.7).

![Möbius manifold topological universe blueprint, by Blake Shatto (asset from the author's mode-identity-theory repo, MIT license)](research/images/blueprint.png)

## Identity

| Field | Value |
| --- | --- |
| Model ID | M8 |
| Name | MIT (Mode Identity Theory) |
| Author | Blake Shatto (independent researcher, sole author) |
| Author contact | GitHub [@dmobius3](https://github.com/dmobius3), for author-gated questions (definitions, intent, what the model does and does not claim); routing convention in [`dev_docs/CROSS_MODEL_TESTING.md`](../../../dev_docs/CROSS_MODEL_TESTING.md) § 6 |
| Lineage | Spectral geometry on S³/2I + Möbius boundary conditions + representation theory (McKay correspondence, Kostant partition, Reidemeister torsion); Einstein's field equations kept unchanged |
| Key inputs | Standalone math papers: the twisted-Möbius first-positive eigenvalue, the coexact spectral gap from McKay distance, the E8-filling Galois pair, and channel selection for a cubic self-interaction on S³/2I |
| Primary sources | Author repo: [github.com/dmobius3/mode-identity-theory](https://github.com/dmobius3/mode-identity-theory); framework deposit [10.5281/zenodo.18064856](https://doi.org/10.5281/zenodo.18064856); full registry in [`theory/_CITATIONS.md`](theory/_CITATIONS.md) (10 Zenodo DOIs machine-verified 2026-07-21) |
| Author-side artifacts | `calculator.html` (recomputes couplings, the 24-entry mass spectrum, and cosmology from the postulate); `mass-null-test.py` (pre-registered torsion null test, frozen tag); `claim-ledger.md` (the framework's own freedom audit: calibration web, cycles, overclaim checks) |
| Onboarding record | [Discussion #312](https://github.com/openwave-labs/openwave/discussions/312) (2026-07-21); maintainer evaluation against [`ONBOARDING_MODELS.md`](../../../ONBOARDING_MODELS.md) STEP 1 (self-evaluation, parameter-count test, red flags) passed on artifact verification |

## Model Profile (what it brings, short form)

| Attribute | M8 |
| --- | --- |
| Substrate | not a dynamical medium: a standing wave `Ψ = cos(t/2)` sampled on a fixed static geometry (S³, the quotient S³/2I, the Möbius edge `S¹ = ∂(Möbius)`). Topology is the input |
| Vacuum / dynamics | none native. No field Lagrangian and no equation of motion; Einstein's field equations are kept unchanged as the geometry's dynamics. This is the defining gap versus the emergent-field columns, and the M8 program's target |
| Particle | a sampled standing-wave mode: an irreducible representation of 2I at a McKay-lattice position, carrying a Reidemeister-torsion / flat-connection (vacuum) assignment. Not a soliton and not an evolved defect |
| Charge | from the 2I stabilizer structure: the Z₃ face stabilizers give QCD color (singlet / triplet per irrep). A group-theoretic assignment, not an integrated winding |
| Derrick escape | not applicable in current MIT: no soliton, so the collapse question does not arise. Stability is asserted spectrally (the first positive level is stable across the cone's self-adjoint extensions; matter modes return under the Möbius double cover), not derived from dynamics |
| Clock | the Waltz clock `dt/dτ = S^(-1/2)`, `S = sin(t/2)`, mapping phase to time. Assumed, not derived: the exponent −1/2 is empirically forced (integer alternatives excluded at Δχ² > 60) but not yet derived from the embedding. The opposite of M5, where the clock is the measured energy-minimizing state |
| EM | α read from the first Fibonacci well (13/60) at grid depth. Calibration-webbed: α is the calibration input that fixes the curvature radius R, so the 0.5% match is a consistency check, not a free prediction (the author's own Cycle 2) |
| Quantum | not applicable: no field quantization. MIT is spectral geometry on a fixed manifold, not a QFT |
| Gravity | Einstein's equations unchanged. The Möbius-surface first-positive eigenvalue `2/R²` is the spectral seed (✅ M8.1); under the stated Gauss lift it becomes `6/R²`, and with the de Sitter `½` imported from GR the reference value is `ΛR² = 3`. Two gaps remain: the value of R is the open R-problem (two routes disagree by ~4×, author-flagged), and the physical identification of `Λ` is gated on an undetermined stress tensor. Pre-registered falsifier: `Λ_obs R_ind² = 3`, where >5σ falsifies that identification and not the seed |
| Free parameters | not zero. One measured calibration anchor per sector: H₀ (edge), Λ or α (surface), m_e (mass-sector normalization). Plus named selection choices (the first-positive eigenvalue branch, the anti-periodic boundary condition, the well set {13, 21, 34, 55}, the torsion-to-slot map), several of which the author's own audits show are chosen or null, not forced |
| Lab anchor | m_e (mass benchmark); measured Λ, H₀, α; PDG fermion masses as comparison targets |
| Formal artifacts | the calculator reassembles every quoted number from the postulate; `mass-null-test.py` is a pre-registered, frozen-tag null test; the claim ledger runs the parameter-count and red-flag self-checks. Documented negatives preserved |
| Next falsifier | Euclid DR1 (full release, mid 2027): a₀(z) ∝ H(z), Λ epoch-independence, and the sign-fixed negative (1+z)¹ term in H²(z). Nearer-term structural: ν₂ neutrino mass at 8.6 meV (JUNO / DUNE) |

## Decision-Relevant Attributes

MIT's parameter economy is audited in the author's own `claim-ledger.md`, which runs the
[`ONBOARDING_MODELS.md`](../../../ONBOARDING_MODELS.md) parameter-count test on the framework before
any maintainer does. The honest summary, kept at the ledger's own weight:

| Attribute | M8 |
| --- | --- |
| Free parameters | one calibration anchor per sector (H₀, Λ-or-α, m_e), plus the four named selection choices above. The ledger's shared-freedom audit finds several are not forced: the well set {13, 21, 34, 55} ranks 30,420 / 249,900 (12.2%) under its own eight-functional variational test, never extremal (selection null); the torsion-to-slot map is a pre-registered null (`mass-null-v1.1`, corrected table, p_A = 0.690; supersedes `mass-null-v1.0`'s p = 0.174 on the pre-correction table); α is input-and-output (Cycle 2, consistency check); and the two independent routes to R disagree by ~4× (Cycle 7, the framework's most significant internal tension) |
| Honest residuals | present and listed, not smoothed: down quark 3.2×, top quark compatible within ×3 (0.93, symmetrized 1.07), tau the weakest adjudicated within-×3 entry (~2.74×); the m_e-to-Λ calibration loop closes only to ~11%; charm is unplaced, and 8 of 24 mass slots are unassigned |
| Formal artifacts | every claim recomputes from its own definition (McKay distances, Reidemeister torsions, the C_geom weights) via the calculator; documented negatives kept as results |
| Falsifiable near-term tests | Euclid DR1 (mid 2027) is the live gate, with pre-registered thresholds on a five-row contender card; ν₂ = 8.6 meV is the sharp particle-sector falsifier |

## Field Configuration of Particles

Standing demand of any particle model: *state the field configuration of each particle,
and say whether it uses topological vortices.* MIT's honest answer is that it does
**not** supply field configurations of that kind: its particles are
representation-theory slots, not field defects. This is exactly the half of the program
MIT lacks and the M8 program supplies
(the gap map: [`research/m8_background.md`](research/m8_background.md)).

| Particle | Configuration in MIT | Topological vortex? |
| --- | --- | --- |
| Electron / leptons | an irrep of 2I at a fixed McKay distance, trivial-vacuum flat connection, with a Kostant geometric weight `C_geom` | ❌ a spectral / representation-theory slot, not a field defect |
| Three generations | the same irrep read in the three flat connections (trivial, standard, Galois) of S³/2I | ❌ three vacua, a structural label |
| Quarks / color | color charge from the Z₃ face stabilizer of 2I (triplet vs singlet) | ❌ a group-theoretic assignment |
| Photon / gluon | massless at the edge-only (S¹) level of the layer split | ❌ (radiation, and no field model of it here) |

The clock is **assumed** (empirically pinned), not derived, the opposite of M5 where the
de Broglie clock is the energy-minimizing state.

## Implementation Status

The in-platform verifications are the analytic results marked ✅ below; native field
dynamics is the open problem (last row), and the live icon tally is in
[`MODELS.md`](../../../MODELS.md). The table records honest external status and marks the
in-platform work planned. The three ❌ rows are
pre-registered negatives the author already owns, offered in the spirit that a
documented negative is a result.

| Sector | Status |
| --- | --- |
| Λ = first-positive eigenvalue 2/R² (twisted Möbius Laplacian) | ✅ VERIFIED in-platform (M8.1, 2026-07-21): blind two-agent eigensolve per [`ONBOARDING_MODELS.md`](../../../ONBOARDING_MODELS.md) STEP 0 agent roles, adversarially audited (6/6 fidelity checks); 2/R², the α₀(α₀+1)/R² wide branch, the 2R/e stability threshold and the −4e^(−2γ)/δ₀² defect state all confirmed at 10-digit precision by agents that never saw the claimed values. The Λ = 3/R² inference (Gauss-Codazzi step + the R-problem) remains open ([`research/findings/m8_1_method_note.md`](research/findings/m8_1_method_note.md)) |
| Fermion mass spectrum (24 entries) | 🚧 planned in-platform / analytic-only today. Reproducible as a script from recomputed constants (M8.3), the same category the platform scores EWT's masses under ("from analytic equations, not in-sim dynamics"). Residuals listed above; evidence graded at the ledger's own weight (the torsion null caps the ×3 hit-rate claim) |
| Yang-Mills mass gap 4/R² on S³/2I | ✅ the ANALYTIC statement is VERIFIED in-platform (M8.1.1, 2026-07-28): blind, adversarially audited, the adjoint coexact gap is 4/R² for every irreducible flat SU(2) connection across the ADE family with the single 36/R² exception on the Galois connection, uniqueness held over 41 connections. ⚠️ Confinement itself (a linear inter-charge potential) is untested and the MODELS.md row stays 🚧 ([`research/findings/m8_1_1_method_note.md`](research/findings/m8_1_1_method_note.md)) |
| Charge / color quantization | 🚧 planned in-platform. Structural in MIT (2I stabilizers); not run in the engine |
| Torsion mass-scorecard (the within-3× hit rate) | ❌ documented negative (author-side): the pre-registered null test on the corrected table (`mass-null-v1.1`, frozen tag, one run) finds random torsion reassignment reproduces or beats the observed coverage 69.0% of the time (p_A = 0.690), superseding `mass-null-v1.0`'s p = 0.174 on the pre-correction table. The compatible-coverage count is typical under random reassignment and carries no evidential weight for the torsion map; the structural outputs are the evidence |
| a₀(z) coherence-scale trigger (SPARC) | ❌ documented negative (author-side): pre-registered pipeline ([10.5281/zenodo.20271702](https://doi.org/10.5281/zenodo.20271702)), run once on 123 galaxies. Transition radius tracks L_f at slope 0.23 (registered [0.7, 1.3]); the coherence-scale mechanism is falsified. The lattice arithmetic is untouched |
| H₀ bimodality (discrete-vs-continuous fork) | ❌ documented negative (author-side): dip test fails to reject unimodality (p = 0.217); H₀ data sorts by calibration class but does not quantize |
| Native field dynamics | 🚧 planned / absent, the defining open problem and the M8 program's core: MIT has no Lagrangian, so masses are assigned by structure rather than emerging from evolution. The platform supplies the Lagrangian-family candidates and the simulation engineering ([`research/m8_platform_pointers.md`](research/m8_platform_pointers.md)) |

## Roadmap

Full program with gates and ownership: [`research/m8_roadmap.md`](research/m8_roadmap.md). Short form:

| Task | What lands |
| --- | --- |
| M8.1 | ✅ DONE (2026-07-21): the certification gate PASSED: independent blind eigensolve + adversarial audit confirmed 2/R², the wide branch, the 2R/e threshold and the defect-state asymptotic at 10-digit precision ([`research/findings/m8_1_method_note.md`](research/findings/m8_1_method_note.md)) |
| M8.2 | ✅ DONE (2026-07-27): the field-dynamics pre-registration LOCKED, a modular contract (immutable core + per-family modules + signed execution appendices carrying the numerics) with targets, success ladder, four-axis outcome language and the no-search rule frozen before any run ([`research/findings/m8_2_preregistration.md`](research/findings/m8_2_preregistration.md)) |
| M8.3 | ✅ COMPLETE (author-contributed, [#362](https://github.com/openwave-labs/openwave/pull/362)): mass-formula reproducer, every constant recomputed from its definition, assembly scripted, 23 gates with a coverage-guarded mutation registry. The reproduction FOUND A DEFECT in the published page (a dropped scalar-zeta term for half-integer bundles); corrected upstream, the null re-run as v1.1 at `p_A = 0.690`, and no MODELS.md icon moved ([`research/findings/m8_3_method_note.md`](research/findings/m8_3_method_note.md)) |
| M8.1.1 | ✅ BOTH PAPERS VERIFY (2026-07-28): 18 pre-registered claims confirmed blind, two adversarial audits refuting nothing. The adjoint coexact gap is `4/R²` across the ADE family with exactly one exception, `36/R²`, uniqueness held over 41 connections. ⚠️ The affine conversion identity is algebraically forced, a platform-side overclaim corrected at review ([`research/findings/m8_1_1_method_note.md`](research/findings/m8_1_1_method_note.md)) |
| M8.1.2 | ✅ ALL 21 FROZEN CLAIMS REPRODUCE (2026-09-10): the fourth bedrock paper, channel selection on S³/2I, verified blind from a two-stage packet with the group given only by its generators; the adversarial audit refuted no value. The spin-8 zero set was proved by both agents with the normalizing hint withheld. ⚠️ The expansion weights are fixed only under the Peter-Weyl reading ([`research/findings/m8_1_2_method_note.md`](research/findings/m8_1_2_method_note.md)) |
| M8.4 | First result landed ([kinematic close](research/findings/m8_4_kinematic_close.md), 2026-08-18): native fields on S³/2I carry no nontrivial McKay slot at any level, closing OQ1's native branch by theorem; the slot survey retargets onto the M8-owned twisted object `M4_int` (three frozen connections, `σ_0` null control). **CLOSED UNRESOLVED 2026-08-26**: the preregistration was FILED 2026-08-23 naming `M4L_Erho`, nine flat bundles `E_ρ` (eight target-bearing plus the `E_R0` control), its § 2 giving `M4_int`'s eight-slot comparison a structural N/A; P1A the pre-target qualification phase ran and closed, the nonlinear pilot is BLOCKED with the one substrate built failing and the alternate never built, so no target configuration ran and no sector was spent ([closeout](research/findings/m8_4_closeout.md)) |
| M8.5 | Quotient-manifold simulation engineering. Grid built and adjudicated. The spectral half ran M8.5-C (attempt A1 terminated without adjudication on a frozen-arm defect, #501) and then M8.5-C2, ADJUDICATED `M8.5-C2-FAILED` 2026-08-31: instrument-attributed STOP-QUAL, no gate measured red and the attribution is to the commissioned unit's conduct (#506, protocol addendum 1). The spectral route and the M8.4 reopening path are closed; nothing in M8.5 is pending |
| M8.6 | ❌ CLOSED WITHOUT RUNNING (2026-08-07, row retired 2026-08-08): McKay-distance rule vs M5's measured lepton hierarchy. The named target proved circular (a readiness audit found it mass-derived, 2026-07-29), and the amended condition's last admissible route, M5.21.11 route (b), failed terminally, so the comparison is permanently inadmissible as instrumented ([`research/findings/m8_6_readiness_note.md`](research/findings/m8_6_readiness_note.md)) |
| M8.7 | LATER, gated on validated field dynamics: the 3D rendering port (M5-style launcher + shared GGUI stack). The route through M8.5-C is closed by the C2 verdict. The gate's condition is unchanged and would run through a future programme under platform conditions, which nothing currently authorises or solicits (#512); no route to it exists today |
| M8.8 | ✅ REPRODUCED (2026-08-22) under the protocol's `convention difference` category, closed out with the author's provenance layer ([#459](https://github.com/openwave-labs/openwave/pull/459)): a context-isolated fresh implementer rebuilt the half-integer torsions from the based chain complex, and the adjudication found 8/8 rows and 4/4 identities exact in `Q(φ)` under the global inverse. Attempt 1, a `structural failure` on a one-string interface mismatch, stays on record ([`research/findings/m8_8_adjudication_record.md`](research/findings/m8_8_adjudication_record.md)) |
| M8.9 | ✅ ADJUDICATED `S1b-SPECTRAL` (2026-08-26): a trivial fibre at high harmonic level already gives a non-real compressed action, so nontrivial fibre transport is NOT NECESSARY to produce the discrete-spectrum signature. Base RBF-FD discretization or scalar quotient reduction implicated, NOT separated; S1 closed on an instrument defect, S2 not triggered, and the task closed at that claim ceiling ([`research/findings/m8_9_s1b_closeout.md`](research/findings/m8_9_s1b_closeout.md)) |

## Help Wanted

M8 is an open column in an open arena. These asks replace the original #312 list. The
independent recompute of `2/R²` is delivered and retired (M8.1, blind and audited), and the
old catch-all "a field dynamics" is now split into questions specific enough to fail.

| Ask | The test, stated so it can return NO |
| --- | --- |
| **Does any candidate field theory here derive MIT's source fingerprint?** | Every other column in this arena evolves a Lagrangian; MIT has none, and that is its central gap. What is new is that the target is pinned rather than gestured at. Supply an ACTION with its metric coupling, and therefore a defined `T_μν`: an equation of motion alone does not fix a stress tensor, and an EOM plus a separately chosen `T_μν` tests the choice, not the theory. Restrict the theory to the static `S³` or the `S³/2I` quotient, whichever your construction needs; that is a choice of arena, not the fork. Separately, declare which metric your `T_μν` sources: physical-static or effective-metric, branches (a) and (b) of § III. Branch (c) there, shifted coefficient, is an outcome about the coefficient rather than a metric to source, so it is not a third placement to declare here. That sourcing choice is open, and it is not ours to assume on your behalf. **For the effective-metric placement with `a_eff = a₀S` held**, the target is pinned: ONE tied object, `Ψ²/S³`, whose `S⁻³` and `S⁻¹` sectors carry coefficient ratio exactly `−1` rather than an independently normalized `w = −2/3` component; PLUS a separate genuine `w = −1` constant; with NO `S⁻²` sector on flat effective slices. Fingerprint, notation and conditionality: [stress-tensor-bridge.md](https://github.com/dmobius3/mode-identity-theory/blob/main/files/framework/files/working/files/stress-tensor-bridge.md) § VI, "First results: the pinned target". A NO on a proposed theory is a result and will be recorded as one: it closes that route rather than being tuned into a YES, and it is not a proof that no such theory exists |
| **Can `R` be fixed independently of `Λ`, the CMB, and de Sitter?** | `Λ = 3/R²` is a forward prediction only if `R` arrives from elsewhere; reading `R` off `Λ` makes the whole chain a tautology. Two author-side routes are live: measured `α` → `Ω_Λ` → `R` at about 23%, and the fermion mass spectrum at order of magnitude. Wanted, either way round: a third route that touches none of the three excluded inputs, or a hostile audit that kills one of the two live routes by exhibiting a hidden dependence on them. [r-problem.md](https://github.com/dmobius3/mode-identity-theory/blob/main/files/framework/files/working/files/r-problem.md) |
| **An adversarial parameter count** | A hostile § 4 pass on the mass and coupling sectors. The author runs a standing claim ledger and expects the freedom to be non-trivial; an independent counter is welcome to disagree with it |

Flow: open a discussion → fork → branch → PR with a DCO sign-off
(`git commit -s`), under Apache 2.0. Light review checks only reproducibility + honest
documentation, not orthodoxy. Start here: [`../../../MODELS.md`](../../../MODELS.md)
§ Contributing, [`../../../ONBOARDING_MODELS.md`](../../../ONBOARDING_MODELS.md),
[`../../../CONTRIBUTING.md`](../../../CONTRIBUTING.md).

## Rich Context for Deep Reader

This is top-level orientation content. **AI agents working for the author start at
[`research/m8_agent_orientation.md`](research/m8_agent_orientation.md)** (the
one-prompt bootstrap: reading list, task template, workflow suggestion, completion
protocol). For additional context: the spec of record in
[`research/m8_theory_canonical.md`](research/m8_theory_canonical.md) (canonical when
docs disagree), the gap map and program rationale in
[`research/m8_background.md`](research/m8_background.md), the cross-model pointer map
in [`research/m8_platform_pointers.md`](research/m8_platform_pointers.md) (written to
be consumed by the author's AI agents), the citations registry in
[`theory/_CITATIONS.md`](theory/_CITATIONS.md), and the author's repo (calculator,
claim ledger, null tests) at
[github.com/dmobius3/mode-identity-theory](https://github.com/dmobius3/mode-identity-theory).
