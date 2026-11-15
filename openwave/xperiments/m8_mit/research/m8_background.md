# M8 background: what MIT has, what it lacks, and why the combination is the program

> **Purpose.** This is the gap map behind the M8 column: an honest statement of what
> Mode Identity Theory brings, what it is missing, and why a field-dynamics program on
> its arena is worth running on this platform. It was written by the maintainers at
> scaffold time (2026-07-21) from the author's onboarding submission
> ([discussion #312](https://github.com/openwave-labs/openwave/discussions/312)), the
> author's repo artifacts (claim ledger, null tests, calculator), and the platform's
> own onboarding evaluation. The author owns the science; corrections via PR or
> discussion are welcome.

## 1. What MIT is

MIT is a **top-down structural model**: a fixed arena (the three-sphere S³, its quotients
S³/{±I} and S³/2I, and a Möbius edge), a single background standing wave
`Ψ = cos(t/2)` sampled on it, and a phase-to-time map (the Waltz clock
`dt/dτ_H = S^(-1/2)`). Particles are representation-theory slots: an irrep of 2I at a
McKay-lattice position, read in one of the three flat connections (read as the three
generations), with color from the Z₃ face stabilizers. The couplings and the fermion
mass ratios are read off this structure (McKay distance, Reidemeister torsion, the
120-cell), and the conditional relation `Λ_ref = 3/R²` is seeded by the carrier's first
positive level `2/R²` and lifted by the Gauss relation and the de Sitter `½` imported
from GR. None of these is evolved from a dynamics.

The existing OpenWave columns run the same program from the opposite end: they start
from a nonlinear field Lagrangian and evolve it until particles emerge, and their
hardest open item is exactly where MIT is strongest, the **origin of the numbers**
(M5's lepton row names its open piece as "the eigenvalue-hierarchy origin,
1 : 5.9 : 15.1"). MIT is strong where the emergent models stay honestly open, and
absent where they are strong.

## 2. The gap map (what the M8 program must supply)

| Gap | The honest statement | What supplies it |
| --- | --- | --- |
| No field equation | MIT has no Lagrangian and no equation of motion; Einstein's equations are kept unchanged as the geometry's dynamics. Masses are assigned by structure, not evolved. The author's frozen slot action (2026-10-04) is an effective model for an M8 run, not MIT's physical matter action, so it leaves this gap standing | M8.4: the Lagrangian-family survey, drawing candidates from M4/M5/M7 ([`m8_platform_pointers.md`](m8_platform_pointers.md) § 2), CLOSED UNRESOLVED 2026-08-26 before any candidate ran; the gap stands; the target-free chassis program's first protocol (M8.5-C) terminated without adjudication on a frozen-arm defect, and its derived successor was adjudicated `M8.5-C2-FAILED` 2026-08-31, instrument-attributed ([#506](https://github.com/openwave-labs/openwave/issues/506)); no route is authorized today, and the gap stays open ([#512](https://github.com/openwave-labs/openwave/discussions/512)) |
| No defect concept | MIT's topology is the ARENA (the manifold itself), not topology IN a field (M5's sense: defects, windings, vortex strings). Its particles are explicitly not vortices | M8.4 was to put a dynamical field ON S³/2I and ask whether its defect or standing-wave sectors realize the McKay slots; it CLOSED UNRESOLVED 2026-08-26 with no target configuration executed. On the standing-wave side, [M8.11](tasks/m8_11_task_details.md) established, as an audited argument, that its pinned stationary cubic equation has local branch germs at six of the level-6 critical rays in each of sectors `3′` and `4`, for sufficiently small amplitude (2026-09-18). Whether any defect or standing-wave sector realizes the McKay slots is untested |
| Stability not derived | MIT's spectral statement, the first positive level `2/R²` in the narrow conic regime, under the Friedrichs extension it adopts at the cone point (and under bridging with `δ₀ > 2R/e`), concerns a level, not the stability of a matter state. MIT's physical theory has no soliton and so no Derrick analysis; under the author's frozen slot action, a Pohozaev argument shows localized standing waves vanish in flat three-space, so no lump much smaller than R is expected, and states at the curvature scale are not excluded | M8.4 was to run a Derrick analysis on a compact arena (see § 4 below); it CLOSED UNRESOLVED 2026-08-26 without a candidate-family Derrick analysis. The branch germs of [M8.11](tasks/m8_11_task_details.md) carry no stability claim. Under the frozen slot action, the author's Slot Ground States (2026-10-05) derives an orbitally stable set of fixed-charge ground states in every slot, and The Soft-Slot Branches (2026-10-06) the small-amplitude linear spectrum of the soft slots' branches, unstable where a slow eigenvalue leaves the imaginary axis, and at a nondegenerate minimum spectrally stable and orbitally stable relative to its stationary family; M8.15 to M8.15.2 recorded the linear type of five of those branches at finite amplitude, and their persistence, at the row 6 ceiling: three are Elliptic wherever their type is decided, and two, the prism in `R4` and the hexagon in `R5`, turn Hyperbolic. On the harmonic Galerkin instrument M8.17 and M8.18 then located the prism's change of type in `R4` and that of the third outside orbit (T4) in `R5`; M8.19's persistence test on the coherent branch in `R5` Persists, the Elliptic stretches of `v₂` (T1) and T4 end untested, and the author's census closed on 2026-11-10 with four rows instrument-limited ([roadmap](m8_roadmap.md#done)). These are global standing waves, not particle states, so the stability of a matter state remains unaddressed |
| Clock not derived from the postulate | The Waltz exponent −1/2 enters through the GR correspondence (standard matter dilution and the Friedmann relation, imported from GR) and is empirically unique in its family (Δχ² > 60 vs integer alternatives), but is not derived at the postulate layer. The opposite of M5, where the clock is the measured energy-minimizing state | The M8 dynamics question: does the background mode + field coupling SELECT the clock |
| Radiation unmodeled | photon/gluon are "massless at the edge-only level", a layer label with no wave mechanics behind it | downstream of a validated field dynamics; not an early target |

## 3. Evidence weight (the author's own ledger, adopted as the column's grading)

The author's `claim-ledger.md` runs the platform's
[`ONBOARDING_MODELS.md`](../../../../ONBOARDING_MODELS.md) parameter-count test on the
framework itself, and the M8 column adopts its verdicts as the grading baseline:

| Layer | Weight | Why |
| --- | --- | --- |
| Structural results | the durable core | the count of 3 flat connections of S³/2I (read as the 3 generations, an identification the ledger cordons off as interpretation), the coexact mass gap 4/R², color from Z₃ stabilizers, chirality: retrodictions with zero adjustable freedom (the ledger's "bones") |
| The 2/R² eigenvalue | decidable, verified | the M8.1 certification gate PASSED (2026-07-21): blind two-agent eigensolve, adversarially audited |
| The 24-slot mass table | LOW evidential weight | the author's own pre-registered null on the corrected table (`mass-null-v1.1`, p_A = 0.690, superseding `mass-null-v1.0`'s p = 0.174 on the pre-correction table): the compatible-coverage count is typical under random torsion reassignment and carries no evidential weight for the torsion map; the well set {13, 21, 34, 55} is never extremal under its own variational test; 19 of 24 entries carry no adjudicated SM assignment, the up and charm quarks unassigned and the bottom compatible only out of its sector; down 3.2× residual, top compatible within ×3 (0.93, symmetrized 1.07) |
| The calibration web | consistency, not prediction | α is input-and-output (Cycle 2); m_e ↔ Λ closes to ~11%; the two routes to R disagree by 3.2× on the canonical pair and 1.7× on the best assigned pair (Cycle 7, the framework's flagged master-scale tension) |
| Forward bets | the falsifiable layer | Euclid DR1 pre-registered thresholds, four rows at the full release (mid 2027) and Row IV on the first qualifying measurement, from DR1-Foundation data or later; the JWST measurements examined lie inside Row IV's registered ceiling, the side on which the row fails, and none is its score |

The practical rule for M8 tasks: **target the structural ladder (slot structure, gap
ratios, generation count), not the numeric mass table.** Slot placement alone is
kinematic: any Laplace-type kinetic term on the flat bundles puts each slot at its McKay
level, which is calibration, never evidence ([M8.4 kinematic close](findings/m8_4_kinematic_close.md)). A
common finite-amplitude branch or stability structure across the eight slots under one
action, without per-slot tuning, would be a result; MIT files no relation among slot
energies for it to meet (the author's [slot relation](https://github.com/dmobius3/mode-identity-theory/blob/main/files/framework/files/working/files/slot-relation.md#viii-report),
reported OPEN 2026-10-04). Chasing the 24 numbers would inherit freedom the author's own
audit already flagged.

## 4. Why the combination is technically attractive

| Feature | Why it matters |
| --- | --- |
| Compact arena vs Derrick | On a compact manifold the Derrick scaling argument loses its free dilation (the radius R sets a scale), so the collapse pressure that constrains flat-space solitons is structurally weakened, though only at the curvature scale: under the author's frozen slot action, localized standing waves vanish in flat three-space (Pohozaev), so no lump much smaller than R is expected. And the platform's measured lesson (M6: oscillation is a genuine third Derrick escape; M5: stability requires time-periodicity) rides naturally on a background clock: the two mechanisms stack |
| The central −I and the Möbius sign | 2I ⊂ SU(2) carries the spinor sign: its central `−I` acts as −1 on half-integer-spin representations, the 720-degree return behind the spin-statistics row that most columns leave 🚧. On MIT's projective carrier one loop carries that sign and the Möbius orientation sign together, which makes the spin sign geometric on the carrier; through the adopted transverse sampler, the integer-spin samples are the anti-periodic ones. That derives neither spin-statistics nor anticommutation; a field dynamics on the arena would still have to supply them |
| Spectral basis = target ladder | Simulating on S³/2I naturally expands fields in 2I-symmetric harmonics, the same representation theory the McKay ladder lives in. The simulation basis and the slot structure speak one language, so slot identity is manifest. Placement on the slots is then kinematic, calibration and never evidence; the well-posed, pre-registrable question is dynamical, whether one action carries a common finite-amplitude branch or stability structure across the eight slots |
| Clean division of labor | MIT supplies the arena and the slot structure (top-down), and files no relation among slot energies (the author's [slot relation](https://github.com/dmobius3/mode-identity-theory/blob/main/files/framework/files/working/files/slot-relation.md#viii-report), reported OPEN 2026-10-04); the platform supplies Lagrangian candidates, simulation engineering, and grading standards (bottom-up). Either outcome is a result: a common branch or stability structure across the eight slots under one action is a dynamical result with no claim on a relation; its absence is a clean documented negative, which this author demonstrably accepts |
| Cross-model payoff, closed on the M5 side | M8.6 (the McKay rule vs M5's lepton hierarchy) was the ONE candidate mechanism on the table for M5's open hierarchy origin. GATED 2026-07-29 (`1 : 5.9 : 15.1` is `Λ := m^(1/3)` from the masses, not measured M5 output) and CLOSED 2026-08-07: the M5.21.11 bridge's last admissible route (the pre-registered extrapolation) failed terminally, so no admissible M5-side target exists as instrumented ([finding 9](tasks/m8_6_task_details.md), [readiness note](findings/m8_6_readiness_note.md)) |

## 5. Onboarding evaluation of record (2026-07-21)

| Check | Result |
| --- | --- |
| Provenance | author account + repo real and active; all artifacts named in the submission exist (`calculator.html`, `claim-ledger.md`, `mass-null-test.py` + frozen inputs/results, per-paper test scripts); all 10 Zenodo DOIs machine-verified resolving (SSRN IDs author-attested, see [`../theory/_CITATIONS.md`](../theory/_CITATIONS.md)) |
| [`ONBOARDING_MODELS.md`](../../../../ONBOARDING_MODELS.md) STEP 1 fit | partial criteria coverage (particle rows yes, dynamics rows structurally absent); genuine forward predictions exist; reproducible (calculator + scripts + public data); strongly falsifiable (dated pre-registrations with thresholds, 3 documented negatives on record) |
| § 4 parameter count | run by the author in advance; freedom found non-trivial and DISCLOSED (the nulls and cycles above), which is the honest shape the test exists to find |
| § 5.1 red flags | one real hit: unrefereed sourcing (all venues preprint-tier), recorded in the citations registry; the other flags largely pass BECAUSE residuals and negatives are preserved rather than smoothed |
| Rigor culture | pre-registration with frozen tags, one-shot runs, negatives reported unprompted: compatible with [`AI_HYGIENE.md`](../../../../AI_HYGIENE.md) and the platform's method-note standard from day one |

Scaffold decision: admit as a research-mode column at 21 🚧, with the certification gate
(M8.1) maintainer-run before any deeper commitment, and the author driving the column
from there ([`m8_roadmap.md`](m8_roadmap.md)).
