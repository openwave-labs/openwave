# M6 Ouroboros: Model Briefing

> **What M6 brings.** A classical field theory in which every particle is a localized,
> time-periodic soliton (a "chaoiton"): in the two-vector formulation the integer
> Chern-Simons / Hopf linking number *is* electric charge, and the theory claims the
> electron, the lepton spectrum, and a sub-MeV dark-matter candidate from one Lagrangian.
> **REFRESHED 2026-07-20**: the former "permanent hold" is lifted; the model column now
> tracks the author's CURRENT theory (Ouroboros+Eli+Fable v4, published 2026-07-20), and a
> new validation program ([`research/m6_roadmap.md`](research/m6_roadmap.md), M6.1+) will
> re-earn or re-grade every cell at M5-series rigor.

![M6_hero](research/images/ouroboros_icon.jpg)

## Identity

| Field | Value |
| --- | --- |
| Model ID | M6 |
| Name | Ouroboros (Chaoiton framework) |
| Author | Paul J. Werbos (NSF program director, retired; QAGI LLC) |
| Author contact | GitHub [@pwerbos](https://github.com/pwerbos), for author-gated questions (definitions, intent, what the model does and does not claim); routing convention in [`dev_docs/CROSS_MODEL_TESTING.md`](../../../dev_docs/CROSS_MODEL_TESTING.md) § 6 |
| Collaboration | AI co-authorship disclosed on the papers (DeepSeek; Claude Sonnet 4.6 + Claude Code; "Nuclear Gemini"); OpenWave contributed the neutral-chaoiton BVP profile + scaling symmetry, and the Test 1 corrections acknowledged in the July 8 record |
| Lineage | Schwinger 1969 dyons + Maxwell extension (toroidal-poloidal mutual confinement); shares the Schwinger ancestor with M5 |
| Primary sources | The full latest-version Zenodo corpus, 29 records: manifest [`theory/_CITATIONS.md`](theory/_CITATIONS.md) (local corpus, gitignored; every record openly downloadable at its DOI). Current spec: Zenodo [21447590](https://zenodo.org/records/21447590) (v4) |
| In-repo | [`research/m6_theory_canonical.md`](research/m6_theory_canonical.md) (specs of record + provenance ledger), [`research/m6_roadmap.md`](research/m6_roadmap.md), [`research/m6_particle_hunt.md`](research/m6_particle_hunt.md); archive-era validated record in [`research/archive/`](research/archive/) (sandbox v1-v11) |

## Model Profile (what it brings, short form)

| Attribute | M6 |
| --- | --- |
| Substrate (two-vector era, all validated results) | two coupled Lorenz-constrained vector fields `A_μ, J_μ` on Minkowski; `ℒ_JA = −F·F − G·G + J·A − g(J·J)²` (v11 normalization; notation drifts across records, see the canonical § 2) |
| Substrate (current spec, v4, paper-level) | ONE scalar chaoiton field φ coupled to nucleon density: `L = ½∂φ∂φ − ½m_φ²φ² − (λ/4!)φ⁴ − gφρ_n(x) + λC[φ]`, m_φ ≈ 0.460 MeV; A-linear nuclear coupling derived from a surface-saturation constraint; the two-theories relation is open (canonical OQ1) |
| Particle | chaoiton = localized time-periodic soliton; charge = mutual Chern-Simons linking Q_CS (= Hopf invariant) in the two-vector era |
| Derrick escape | time-periodicity / oscillation (the "third escape"): genuine and in-platform |
| EM | `A_μ` IS the EM 4-potential; Maxwell exact in the J→0 linear limit (two-vector era; v4's scalar has no EM sector) |
| Electron benchmark | **CLOSED (M6.2 decision gate, 2026-07-20, branch (b))**: the pre-registered no-search re-derivation found the published H/Q = 1.689 to be a code artifact (H not the spec's energy, Q no Noether charge, ODE = the z20274505 mass-term variant) evaluated on a provably non-localized state at one integration window; full record [`research/findings/m6_2_method_note.md`](research/findings/m6_2_method_note.md), ledger canonical § 4 |
| DM candidate | neutral chaoiton m_χ = 0.460 MeV + mediator m_J = 0.6184 MeV parameter-free (in-platform, archive era); v4 promotes 0.460 MeV to the fundamental field mass |
| Free parameters | two-vector era: 3 claimed (g, λ, ω), with ω scan-fitted per lepton and no discreteness mechanism (the standing open question); v4: m_φ, λ, g plus the unspecified constraint C[φ] |
| Formal artifacts | Lean 4: theorem STATEMENTS formalized; the existence proof is `sorry`-discharged and the formalized ODE differs from the integrated one (canonical § 4); the June GF-stability script is the corpus's most reproducible artifact |
| Solve method | 1D radial ODE / BVP (scipy) + statistics notebooks; no production port (the full-3D continuation is M7's program) |
| Next falsifier | A-linear vs A² nuclear discrimination (v4's own six-domain program, M6.5); the solar-wind sidereal protocol P1-P3 ([`research/ai_analysis/2026-07-11_1630_dm_solar_wind_review.md`](research/ai_analysis/2026-07-11_1630_dm_solar_wind_review.md)); sub-MeV direct detection |

## Decision-Relevant Attributes

| Attribute | M6 |
| --- | --- |
| Parameter economy | "3 parameters" is the era claim; the refresh's corpus read documented scan-fitted ω per lepton, a recalibrated g (1.0625 → 1.0000 with the harmonics silently moving), and fit-grade conventions behind the benchmark, and M6.2 then closed the question: the benchmark number is not derivable from the published Lagrangian under any parameter count ([`research/m6_theory_canonical.md § 4`](research/m6_theory_canonical.md)) |
| Formal artifacts | Lean 4 statements + the attached `chaoiton_gf_verification.py` scan (319/360 radial-GF-stable); our independent numerical reproduction of the canonical neutral profile ([`research/archive/0d_canonical.md`](research/archive/0d_canonical.md)) |
| Self-correction record | The record contains genuine public corrections: the July 5 retraction notice (LLM-generated placeholder numbers), the July 8 Test 1 walk-back, the in-era artifact retractions (23-solution and 1,208-solution claims). The refresh treats this as the working norm: claims are graded by what survives independent rerun |
| Falsifiable near-term tests | v4's A-linear vs A² six-domain suite; DAMA per-bin amplitude table; solar-wind modulation (sidereal split); sub-MeV searches; the A-scaling sensor discriminator (R = 1.125 nuclear vs 1.108 EM) from the author's detection-disclosure record |

## Field Configuration of Particles

Standing demand of any particle model: *state the field configuration of each particle,
and say whether it uses topological vortices.* M6's answers, with the v4 caveat that the
current spec's particle sector is silent (details: [`research/m6_particle_hunt.md`](research/m6_particle_hunt.md)):

| Particle | Field configuration in M6 (two-vector era) | Topological vortex? |
| --- | --- | --- |
| Electron | charged chaoiton, `Q_CS = +1`, ω = 1 | ✅ knot (linking) |
| Positron | `Q_CS = −1` (opposite linking sign) | ✅ (statement only) |
| Muon / tau | higher-ω states (values version-dependent: 11.0/40.7 → 12.82/50.0 → "bifurcation islands"); no ω-discreteness mechanism | ✅ claimed |
| Pion+ (candidate) | chaoiton at ω = 15.0 (3.25%) | ✅ claimed |
| Neutral chaoiton (DM) | `Q_CS = 0`, l = 1 p-wave → m_χ = 0.460 MeV, m_J = 0.6184 MeV, C = 770 MeV·fm | ⚠️ neutral static profile by a boundary-value solve (in-platform; re-based 2026-11-18) |
| Proton | three-chaoiton bound state (Schwinger H-particle lineage) | 🚧 narrative-level |

Honest caveats preserved from the archive era and confirmed by the refresh read: the clock
is *built into* the `e^{iωt}` ansatz (assumed, not derived as energy minimizer); no
discrete-spectrum mechanism selects the lepton ω values; charge quantization is
Lean-stated plus author-claimed, not re-derived in-platform, and the underlying proof is
for static configurations while the solutions are time-periodic.

## Implementation Status

The 2026-07-20 refresh day settled which cells the column has earned; the live tally is
the [`MODELS.md`](../../../MODELS.md) score-board. The three honest negatives (electron mass,
lepton spectrum, pion) were earned by the M6.2 decision gate's pre-registered no-search
re-derivation, exactly the "cells move in both directions" contract; the three archive-era ✅
(neutral-sector stability, DM candidate, Maxwell-by-construction) were re-read under the
written bar on 2026-11-18 ([T11](../../../dev_docs/tasks/t11_task_details.md)) and read 🚧, ⚠️ and 🚧. Nothing from the July-era theory (+Eli, v4) is validated
in-platform.

| Sector | Status |
| --- | --- |
| Particle stability | 🚧 re-based 2026-11-18 (neutral sector): a nodeless neutral radial profile by BVP (K₁ tail), never evolved in time, on a row that asks for a run in time; the CHARGED sector is closed: no localized state at the calibration point (M6.2) nor anywhere in the localization window (M6.4), and the published census evidence is dead (62/319/62 irreproducible or imposed-by-construction; radial-only gap closed within the reduction) |
| Dark-matter candidate | ⚠️ re-based 2026-11-18: parameter-free m_J = 0.6184 MeV; m_χ = 0.460 MeV is set by the chosen λ; stability untested; the phenomenology chain above it (suppression factors, modulation inference) is unverified → M6.6 |
| Maxwell EM | 🚧 re-based 2026-11-18: `A_μ` is the 4-potential by construction (two-vector spec; absent from v4), a declared property; no wave is computed |
| Electron rest mass | ❌ M6.2 (2026-07-20): the benchmark H/Q is a code artifact on a non-localized state; honest negative of record ([`research/findings/m6_2_method_note.md`](research/findings/m6_2_method_note.md)) |
| Charge quantization | ⚠️ Lean-stated + claimed; static-proof vs time-periodic gap; never computed in-platform; |
| de Broglie clock | ⚠️ built into the ansatz, not emergent; the Q in L = ωQ is a coded convention (no internal U(1) exists, M6.2) |
| Lepton spectrum | ❌ the ω ladder rides the invalidated H/Q machinery and the non-localized charged-state family (M6.2); the discreteness question is CLOSED negative (M6.4): no localized state exists for a mechanism to select, every ladder ω sits above every window, and the system the ladder was computed in cannot select by construction |
| Magnetic moment + spin | the published g-factor match is the identity L = ωQ with ω chosen (v11's own footnote concedes this); honest status: not independently evidenced |
| Nuclear sector (v4's stake) | 🚧 the six-domain A-linear program is public-notebook grade, unaudited; the Sawada anomaly is described three inconsistent ways across the record → M6.5 |
| Quarks · baryons · weak · gravity · KG · orbitals | 🚧 not addressed (gravity explicitly out of scope "prior to gravity"; a Carmeli 5D embedding is asserted without equations) |

Deep dives: [`research/m6_theory_canonical.md`](research/m6_theory_canonical.md) (specs of
record, version lineage, THE PROVENANCE LEDGER § 4, consumption rules),
[`research/m6_roadmap.md`](research/m6_roadmap.md) (the M6.1+ program),
[`research/m6_particle_hunt.md`](research/m6_particle_hunt.md) (identification scorecards),
[`research/archive/0d_canonical.md`](research/archive/0d_canonical.md) (archive-era numerical
record).

## Roadmap (refresh era, re-scoped 2026-07-20)

The pre-analysis decision (recorded in [`research/m6_roadmap.md`](research/m6_roadmap.md)):
the program CLOSES THE LEDGER ON THE FROZEN two-vector v11 spec rather than tracking the
July-era moving target (three theory versions in 18 days; the drift evidence is the
canonical § 4 ledger). [M6.2](research/m6_roadmap.md), the decision gate, ran the same
day and landed **branch (b)**: the electron sector closed honestly.

| Next | What lands |
| --- | --- |
| ✅ [M6.1](research/tasks/m6_1_task_details.md) spec certification gate (DONE 2026-07-20) | the v11 spec certified: [convention sheet](research/m6_1_v11_convention_sheet.md) with the FIXED-vs-GAP pre-registration checklist; v4 characterized script-backed (EL unclosable, boundedness verified, A-linearity generic); four print-level v11 defects ledgered (canonical § 4); adversarial audit 9/10 CONFIRMED ([method note](research/findings/m6_1_method_note.md)) |
| ✅ [M6.2](research/tasks/m6_2_task_details.md) THE DECISION GATE (DONE 2026-07-20, branch (b)) | pre-registered no-search H/Q re-derivation, audited 8/8: the derived pairing gives 0.1429 vs the claimed 1.689; the state is provably non-localized and the number window-defined; electron sector closed as three honest negatives ([method note](research/findings/m6_2_method_note.md)) |
| [M6.3](research/m6_roadmap.md) charge quantization in-platform | gating condition (M6.2 branch (a)) unmet; |
| ✅ [M6.4](research/tasks/m6_4_task_details.md) stability census + ω-selection (2026-07-21) | all three published counts fail reproduction (May 0/1280; June 318/360 only under imposed-tail backgrounds, window-defined; July = a citation of the dead May count); the record's numerics do not implement its own Lean theorem; OQ3 closed negative: no localized charged-sector state at any ω. Adversarial audit 9/10 CONFIRMED + 1 PARTIAL. The decidable core is exhausted; M6 returns to hold (DM sector + M7 lineage = the durable legacy) |
| [M6.5](research/m6_roadmap.md) / [M6.6](research/m6_roadmap.md) | the July-era nuclear + DM fitting programs, until the author freezes a spec (reopening condition in the roadmap); M6.6's solar-wind arm is spec-independent and can be pulled forward alone |
| OPTIONAL: [M6.7](research/m6_roadmap.md) the two-theories fork | only if continued engagement with the author is wanted |

The full-3D route remains **M7 HydroBoros**, which carries the two-vector Lagrangian,
`m_J`, `g`, and the benchmark onto a 3D lattice
([`../m7_hydroboros/__M7_model_briefing.md`](../m7_hydroboros/__M7_model_briefing.md)).

## Help Wanted

M6 is an open column mid-refresh; the highest-leverage contributions are exactly the
roadmap's re-derivations. A documented *negative* counts as much as a positive.

| You can contribute | How |
| --- | --- |
| Refute the M6.2 close | the decision gate's derivation chain, numbers, and audit are fully published ([`research/findings/m6_2_method_note.md`](research/findings/m6_2_method_note.md)): find a convention under which the published H/Q IS derivable from the published Lagrangian, or a bound charged state the far-field analysis missed |
| Refute the M6.4 close | the census rerun, the ODE comparison, and the existence scans are fully published with the audit ([`research/findings/m6_4_method_note.md`](research/findings/m6_4_method_note.md)): reproduce any published stability count under a faithful reading of its record, find a localized charged-sector state in the frozen system's window, or exhibit a discreteness mechanism the term set supports |
| A nuclear-domain audit | rerun any of v4's six A-linear domains from the named datasets with stated metrics (M6.5) |
| The solar-wind protocol | execute P1-P3 from [`research/ai_analysis/2026-07-11_1630_dm_solar_wind_review.md`](research/ai_analysis/2026-07-11_1630_dm_solar_wind_review.md) on public OMNI/L1 data (M6.6b) |
| A full-3D validation | M7's program; see its roadmap |

Flow: open a discussion → fork → branch → PR with a DCO sign-off
(`git commit -s`), under Apache 2.0. See [`../../../MODELS.md`](../../../MODELS.md)
§ Contributing, [`../../../ONBOARDING_MODELS.md`](../../../ONBOARDING_MODELS.md),
[`../../../CONTRIBUTING.md`](../../../CONTRIBUTING.md). Model discussion runs in the
Models-of-Particles group.

## Rich Context for Deep Reader

This is top-level orientation content. For the full picture read
[`research/m6_theory_canonical.md`](research/m6_theory_canonical.md) first (its § 4
provenance ledger and § 6 consumption rules govern how every M6 number may be used), then
the roadmap and hunt pages, then the archive-era record in
[`research/archive/`](research/archive/). The author's paper corpus itself is local-only
([`theory/_CITATIONS.md`](theory/_CITATIONS.md)) with every record openly available at its
Zenodo DOI.
