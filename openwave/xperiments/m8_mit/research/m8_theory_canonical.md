# M8 / MIT, THEORY CANONICAL (specs of record)

> **Purpose.** The single place that pins the MIT spec the M8 column is graded
> against: the arena, the background mode, the operator layer, the particle map, the
> mass formula, and the known tensions, each version-pinned to a published record
> ([`../theory/_CITATIONS.md`](../theory/_CITATIONS.md)). When other docs disagree,
> this file is canonical; when THIS file changes, the change is dated and the record
> id updated. The author owns the science; edits to the spec itself come from the
> author (PR or discussion), edits to verification status come from platform runs.
>
> **Scaffold-stage caveat (2026-07-21).** Everything below is TRANSCRIBED from the
> author's records and onboarding submission
> ([discussion #312](https://github.com/openwave-labs/openwave/discussions/312)),
> not yet verified in-platform. Verification status is tracked per item; the first
> flip is the M8.1 certification gate ([`m8_roadmap.md`](m8_roadmap.md)).

## 1. The arena (topology is the input)

| Object | Definition | Record |
| --- | --- | --- |
| S³ | the three-sphere of curvature radius R (the master length scale; see the R-problem, § 6) | framework deposit [10.5281/zenodo.18064856](https://doi.org/10.5281/zenodo.18064856) |
| S³/2I | the quotient by the binary icosahedral group 2I (\|2I\| = 120; the Poincaré homology sphere) | framework deposit; coexact-gap paper |
| The Möbius edge | a conic Möbius band whose boundary circle `S¹ = ∂(Möbius)` carries the edge layer; the cone point admits a family of self-adjoint extensions | first-eigenvalue paper (SSRN 6968741, author registry) |
| Boundary condition | anti-periodic (twisted): matter modes return only under the double cover. The structural home of 720-degree return | first-eigenvalue paper |

## 2. The background mode + the clock

| Object | Definition | Status |
| --- | --- | --- |
| The standing wave | `Ψ = cos(t/2)` sampled on the fixed geometry; not a dynamical field, the sampling structure particles are read from | assumed (the postulate) |
| The Waltz clock | `dt/dτ = S^(−1/2)` with `S = sin(t/2)`, mapping phase to time | assumed; the exponent −1/2 is empirically pinned (integer alternatives excluded at Δχ² > 60) but NOT derived from the embedding (OQ3) |
| Dynamics | none native: no Lagrangian, no equation of motion; Einstein's field equations kept unchanged as the geometry's dynamics | the defining gap; supplying it is the M8 program ([`m8_background.md`](m8_background.md)) |

## 3. The operator layer (the decidable mathematics)

| Result | Statement | Record | Platform status |
| --- | --- | --- | --- |
| First positive eigenvalue | the twisted Laplacian on the conic Möbius band has first positive eigenvalue `λ₁ = 2/R²`, stable across the cone's self-adjoint extensions | SSRN 6968741 (author registry); operator statement: [first-eigenvalue.md](https://github.com/dmobius3/mode-identity-theory/blob/main/files/framework/files/bedrock/files/first-eigenvalue.md) (author repo, shared on #312, 2026-07-21) | ✅ VERIFIED IN-PLATFORM (M8.1, 2026-07-21): blind two-agent verification, adversarially audited; λ₁⁺ = 2/R² (narrow), α₀(α₀+1)/R² (wide), the 2R/e stability threshold and the −4e^(−2γ)/δ₀² defect state confirmed at 10-digit precision ([`findings/m8_1_method_note.md`](findings/m8_1_method_note.md)) |
| Λ from the edge mode | `Λ = 3/R²`, the eigenvalue `2/R²` carried by the Gauss-Codazzi factor 3/2 (3 = Ricci trace derived, 1/2 imported from GR) | Λ ground-mode paper [10.5281/zenodo.18307314](https://doi.org/10.5281/zenodo.18307314) | 🚧 the 2/R² input is now verified (M8.1 ✅); the Gauss-Codazzi 3/2 step and the R-problem (§ 6) remain untested |
| Coexact spectral gap | Yang-Mills-sector mass gap `4/R²` on S³/2I from McKay distance for flat bundles | SSRN 6968698 (author registry); YM paper [10.5281/zenodo.18463584](https://doi.org/10.5281/zenodo.18463584) | ✅ VERIFIED IN-PLATFORM (M8.1.1, 2026-07-28): blind, adversarially audited. First occurrence equals McKay distance for all 344 irreducibles tested; the adjoint gap is `4/R²` for every irreducible flat SU(2) connection across the ADE family with exactly ONE exception, the Galois connection on S³/2I at `36/R²`, uniqueness holding over 41 connections ([`findings/m8_1_1_method_note.md`](findings/m8_1_1_method_note.md)) |
| Galois pair | the affine rho-index conversion and the E8-filling Galois pair on the Poincaré homology sphere | SSRN 7129118 (author registry) | ✅ ARITHMETIC VERIFIED IN-PLATFORM (M8.1.1, 2026-07-28): blind, adversarially audited, the author's own test script quarantined and unused. Rho and character sums, the golden-class support, the charges, the vector `H` and its augmentation `−72`, and the E8 mod-2 package all reproduce exactly. ⚠️ The conversion identity itself is algebraically FORCED (§ 5.1 of the note), and the paper's § 7 topology proofs are outside a numerical check ([`findings/m8_1_1_method_note.md`](findings/m8_1_1_method_note.md)) |
| Channel selection | binary-icosahedral symmetry confines a cubic self-interaction of spin 3 to `span{M_0, M_6}` with `M_0` radial, so one projective ray survives; separately, the spin-8 channel vanishes exactly on the time-reversal-invariant rays | Zenodo [10.5281/zenodo.22681502](https://doi.org/10.5281/zenodo.22681502) (version DOI) | ✅ VERIFIED IN-PLATFORM (M8.1.2, 2026-09-10): blind, two-stage packet with the group given by generators, adversarially audited; all 21 frozen claims reproduce and no value was refuted. ⚠️ The expansion weights are fixed only under the Peter-Weyl reading ([`findings/m8_1_2_method_note.md`](findings/m8_1_2_method_note.md)) |

## 4. The particle map (representation theory, not field configurations)

| Object | Definition | Status |
| --- | --- | --- |
| Particle | an irreducible representation of 2I at a McKay-lattice position (a mode slot, NOT a soliton or defect) | structural |
| Generations | the same irrep read in the three flat connections of S³/2I (trivial, standard, Galois): exactly three, zero freedom | structural, the ledger's strongest class |
| Color | the Z₃ face stabilizers of 2I: triplet vs singlet per irrep | structural |
| Torsion-to-slot map | Reidemeister-torsion assignment of irreps to mass slots | ❌ author's pre-registered null on the corrected table (`mass-null-v1.1`: p_A = 0.690, superseding `mass-null-v1.0`'s p = 0.174 on the pre-correction table); the compatible-coverage count is typical under random torsion reassignment and carries no evidential weight for the torsion map; slots are graded structurally only |
| Unassigned | charm unplaced; 8 of 24 slots empty; rank-16 / dead-zone states named but uncomputed | open, listed honestly |

## 5. The mass formula (analytic sector)

```text
m = μ_Λ · C_geom · (√Ω)^(dist/30) · T²

μ_Λ  = ρ_Λ^(1/4) ≈ 2.25 meV     (the mass-sector anchor, from measured Λ: an INPUT)
C_geom = Kostant geometric weight of the irrep
Ω     = (R/ℓ_P)² ≈ 10¹²²          (the calibration-web hub)
dist  = McKay distance of the slot
T     = Reidemeister torsion of the assignment (evidence-null, § 4)
```

Calibration anchors, one per sector: H₀ (edge), Λ or α (surface), m_e (mass
normalization). Record: mass-spectrum paper
[10.5281/zenodo.18603975](https://doi.org/10.5281/zenodo.18603975). Platform status:
🚧 analytic only: the M8.3 reproducer (✅ 2026-07-28) recomputes every constant from its
definition; graded at the ledger weight regardless of hit rate (§ 6).

## 6. Known tensions + evidence grading (adopted from the author's claim ledger)

| Item | The honest statement |
| --- | --- |
| Cycle 2 (α) | α is input AND output: it calibrates R, then the scaling law "predicts" α. The 0.5% match is a consistency check; the non-circular content is Λ at ~24% downstream |
| Cycle 7 (the R-problem) | the two independent routes to R disagree ~4× (coupling route ≈ 5.3 Gpc, mass route ≈ 20 Gpc; ~14× in Λ). Every R-dependent number inherits this until one route is shown correct (OQ2) |
| m_e ↔ Λ | the closure loop holds only to ~11% |
| Residuals | down quark 3.2×, top compatible within ×3 (0.93, symmetrized 1.07), tau the weakest adjudicated within-×3 entry (~2.74×); not smoothed |
| Documented negatives | torsion null (`mass-null-v1.1`, p_A = 0.690 on the corrected table; supersedes `mass-null-v1.0`'s p = 0.174 on the pre-correction table), SPARC a₀(z) coherence-scale mechanism (falsified, pre-registered pipeline [10.5281/zenodo.20271702](https://doi.org/10.5281/zenodo.20271702)), H₀ bimodality (unimodal, p = 0.217) |
| Grading rule | platform tasks target the STRUCTURAL ladder (slot structure, gap ratios, generation count), never the 24-entry numeric table ([`m8_background.md § 3`](m8_background.md)) |

## 7. Consumption rules (standing)

| Rule | Meaning |
| --- | --- |
| Version-pin | every result cites the record id it was computed against; the author's repo is context, the deposits are the citable layer |
| Pre-register | gates, conventions, and success criteria BEFORE numerics; forks reported with all numbers, never tuned toward published values |
| No calibrated conventions | derive and pre-register, or record as a fit with its search space |
| Audit | substantive claims get an independent adversarial pass before they are trusted ([`AI_HYGIENE.md`](../../../../AI_HYGIENE.md) § 1) |
| Author-gated | intent, provenance, and definition questions go to the author; the platform does not resolve them by inference |

## OPEN QUESTIONS

| ID | Question | Route | Status |
| --- | --- | --- | --- |
| OQ1 | Does ANY reasonable nonlinear field equation on S³/2I have defect or standing-wave solutions whose energies realize the McKay slot structure? | M8.4, CLOSED UNRESOLVED 2026-08-26 | 🚧 open, and NEVER TESTED on the twisted branch: M8.4 closed with no target configuration executed and no nontrivial sector spent ([closeout](findings/m8_4_closeout.md)), and the spectral chassis that was to reopen it adjudicated `M8.5-C2-FAILED` 2026-08-31 ([#506](https://github.com/openwave-labs/openwave/issues/506)), so OQ1's dynamical branch has no authorized route and stays open ([#512](https://github.com/openwave-labs/openwave/discussions/512)). The NATIVE branch closed by theorem 2026-08-18 ([kinematic close](findings/m8_4_kinematic_close.md)): single-valued quotient fields carry no nontrivial slot at any level, so the live branch runs on M8-owned twisted objects, `M4_int` first |
| OQ2 | The R-problem: which of the two routes (coupling ≈ 5.3 Gpc vs mass ≈ 20 Gpc) is the correct determination of the master scale, and what breaks in the other? | author-side; M8.1 and M8.3 have landed (✅), and no platform task has bounded it yet | 🚧 open |
| OQ3 | Can the Waltz exponent −1/2 be DERIVED from the embedding (or from a field dynamics) rather than empirically pinned? | M8.4 lineage, CLOSED UNRESOLVED 2026-08-26; the M8.5-C route closed 2026-08-31 ([#506](https://github.com/openwave-labs/openwave/issues/506)), and no field-dynamics route is authorized today ([#512](https://github.com/openwave-labs/openwave/discussions/512)) | 🚧 open |
| OQ4 | Does the McKay-distance rule map onto M5's lepton hierarchy? `1 : 5.9 : 15.1` is NOT the target (it is `Λ := m^(1/3)` from the masses); the admissible one is the measured A < C < B census, once physically parameterized | M8.6, CLOSED on the M5 side 2026-08-07: the bridge's last admissible route failed terminally ([finding 9](tasks/m8_6_task_details.md)) | ❌ closed as instrumented |
| OQ5 | Which topological-defect sectors exist at all on S³/2I (homotopy of the target space restricted to the quotient), and does the anti-periodic double-cover condition create sectors flat space lacks? | M8.4 prerequisite analysis | 🚧 open |
| OQ6 | Does the double-cover return supply a genuine spin-½ mechanism once a dynamical field lives on the arena (the 720-degree row most columns leave 🚧)? | downstream of OQ1 | 🚧 open |
