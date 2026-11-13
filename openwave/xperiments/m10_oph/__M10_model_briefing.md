# M10 OPH (Observer Patch Holography): Model Briefing

> **What M10 brings.** Observers first, fields second. An observer is a feedback
> loop that reads the ports of its patch, keeps a finite record, and acts on that
> record. The observers sit on a holographic screen of twelve-port icosahedral
> carriers and run one local repair and memory law, written as update rules with no
> action behind it. Fields, particles and spacetime are the effective descriptions
> the observers arrive at when they compare records. Its substrate is a discrete
> carrier with update rules, not a field, and it was admitted with every criterion
> not yet tested in-platform (discussion #712, 2026-11-13). Model ID M10 was assigned at admission.
>
> **Hero image: placeholder.** Every model briefing opens with one. The author
> chooses the image and the link it opens, and adds it here by pull request.

## Identity

| Field | Value |
| --- | --- |
| Model ID | M10 |
| Name | OPH (Observer Patch Holography) |
| Author | Bernhard Mueller, Pragma Research Inc., corresponding author. Not the sole author: the flagship paper lists fourteen authors (the author's statement in the application) |
| Author contact | GitHub [@muellerberndt](https://github.com/muellerberndt), for author-gated questions (definitions, intent, which construction is the specification of record); routing convention in [`dev_docs/CROSS_MODEL_TESTING.md`](../../../dev_docs/CROSS_MODEL_TESTING.md) § 6 |
| Disclosure | A co-author of the flagship is the applicant of K7 ([discussion #525](https://github.com/openwave-labs/openwave/discussions/525)), and the author's repository vendors K7 comparison files. The two are separate frameworks, and K7's standing is unchanged by this column (the author's disclosure, and the admission reply) |
| Lineage | Wheeler's law without law; the relational and Darwinist readings of quantum theory (Rovelli, Zurek); distributed agreement (Lamport, Dijkstra's self-stabilization); Jacobson and Van Raamsdonk for gravity from entanglement thermodynamics; Doplicher and Roberts for gauge groups from superselection; Bisognano and Wichmann for modular clocks; Wilson's lattice gauge theory; 't Hooft's cellular automaton interpretation as the closest relative among discrete deterministic substrates |
| Primary sources | Author repository [FloatingPragma/observer-patch-holography](https://github.com/FloatingPragma/observer-patch-holography), pinned at `3c4e1e2` for this scaffold; registry in [`theory/_CITATIONS.md`](theory/_CITATIONS.md). For every number, the specification of record is the executable model at the pinned commit ([`research/m10_theory_canonical.md`](research/m10_theory_canonical.md) § 1) |
| In-repo | Headless only. No launcher. The first task is the certification gate [`research/tasks/m10_1_task_details.md`](research/tasks/m10_1_task_details.md) |
| Application | [discussion #712](https://github.com/openwave-labs/openwave/discussions/712); admission reply [discussioncomment-19595538](https://github.com/openwave-labs/openwave/discussions/712#discussioncomment-19595538) |

## Model Profile (what it brings, short form)

| Attribute | M10 |
| --- | --- |
| Substrate | A holographic screen of twelve-port icosahedral carriers. Each carrier holds integer loads on its ports, neighbouring carriers share seam links, and a six-class flux register records the protected classes. Port symmetry: the icosahedral group A5, with its binary cover for spinors. Not a field and not a dynamical medium. The design is declared, with no derivation claimed |
| Vacuum / dynamics | Tier 2: a local repair and memory law written as update rules, run by the native observers. No Lagrangian, no action, no field equation. Conservation is demonstrated per source, not asserted |
| Particle | A protected flux defect: a twist in the carrier network that no local repair removes, with a centre class in Z6. Several species share a class, so the class does not identify a species |
| Charge | The declared generator Q = T3 + Y on a declared fifteen-state module per generation. Quantization is assigned, not derived |
| Derrick escape | Not applicable as posed: no soliton of a field. A self-bound particle is not claimed (the dressed band needs a declared trap) |
| Clock | Event order generated from read-from edges is the framework's reading of time. A self-starting de Broglie clock is open by the author's own theorem |
| EM | A declared Maxwell cochain complex on the finite sources. Not derived from the carrier |
| Quantum | Finite quantum Hamiltonians on declared lattices; projection events with the measurement identities on finite event algebras |
| Gravity | A declared energy-lapse coupling on a finite source; the conditional Einstein limit is paper-tier. The one tested unit attachment fails the strength of G and the speed of light |
| Free parameters | Zero free parameters is not claimed. The ledger of inputs, calibration targets and predictions is in [`research/m10_theory_canonical.md`](research/m10_theory_canonical.md) § 6 |
| Lab anchor | The electron mass (the energy anchor); the fine-structure constant (fixes the pixel coordinate P) |
| Formal artifacts | A Lean library, an executable model with per-package proof files and receipts, a frozen prediction register with custody files, a portable packet of five headless computations (all in the author's repository, none run in-platform) |
| Next falsifier | The author's frozen scalar tilt against the first Simons Observatory primary-CMB release. No frozen row is target-blind, distinct from the null and free of exposed choices at once (the author's own reading) |

## Decision-Relevant Attributes

| Attribute | M10 |
| --- | --- |
| The defining open problem | By the author's own account, each criteria row runs on its own declared finite source and none runs on the twelve-port screen. One process carrying the rows is the column's open join ([`research/m10_background.md`](research/m10_background.md) § 2) |
| How a cell moves | A declared or installed source is inherited and leaves its row not yet tested. A row moves when its observable is computed on the carrier's own process in-platform ([`research/m10_theory_canonical.md`](research/m10_theory_canonical.md) § 8) |
| Predictions | None passes the prediction test of [`ONBOARDING_MODELS.md`](../../../ONBOARDING_MODELS.md) § 1.1, on the author's own audit. The lepton ratios, the hadron spectrum and the tau width ratio are comparisons with known data |
| Residuals on record | The author lists them unprompted: the muon ratio 442 standard uncertainties from measurement, the held-out hadrons and light nuclei failing, the gravity unit attachment off by 1.18 × 10⁴⁰, the weak mixing angle of the declared coupling chain at 0.23074 against 0.23129(4) |

## Field Configuration of Particles

Standing demand: state the field configuration of each particle, and whether it
uses topological vortices. The author's answer has two layers, and they are not
yet joined.

| Particle | Configuration in OPH | Topological vortex? |
| --- | --- | --- |
| Native picture, any species | A protected flux defect of the carrier network, labelled by a centre class in Z6 | The intended reading is a topological defect of overlap transport. No localized defect is established for any species (the author's hypothesis page) |
| Electron and charged leptons, in the row constructions | A state of a declared finite source carrying the fifteen-state module; the three families from a register on one carrier face | No |
| Quarks | Phase lifts of a closed loop of (SU(3)×U(1))/Z3; no localized texture | No |
| Baryons, mesons, light nuclei | States of a calibrated hyperfine action on two-cell blocks and boxes | No |
| Photon | A transverse mode of the declared Maxwell cochain complex | No |

## Implementation Status

Registered column. Every `MODELS.md` cell is at 🚧, and the per-row summaries are
in [`MODELS.md` § OPH (M10)](../../../MODELS.md#oph-m10). Nothing below is an
in-platform result.

| Sector | Status |
| --- | --- |
| Native carrier and repair law | Author-side: confluent repair with one endpoint, event order from provenance, a rank-three slowest band of the repair mean on the twelve-port carrier, exact fluctuation identities. Statements about declared finite objects, machine-checked in the author's Lean library by the author's account; not rebuilt in-platform |
| All criteria rows | 🚧 The author's constructions exist for every row, each on its own declared finite source. Inherited until computed on the carrier's own process |
| Lepton register | 🚧 Koide's 2/3 and the phase 2/9 from a register on one carrier face. The headline numbers were recomputed at admission from the stated formulas; the result is a fit by the author's own parameter count ([`research/m10_background.md`](research/m10_background.md) § 4) |
| Hadron source | 🚧 Calibrated to seven measured inputs; the held-out comparisons fail and are on record |
| Certification gate | 🚧 M10.1, planned: a blind recompute of the twelve-direction dispersion the author offered |

## Roadmap

| Task | What lands |
| --- | --- |
| M10.1 | Certification gate: a blind, maintainer-run recompute of the twelve-direction scalar dispersion, from the author's definition-only statement. Certifies arithmetic on a declared object and moves no cell |
| Later tasks | Created and ordered by the author on [`research/m10_roadmap.md`](research/m10_roadmap.md) |
| Rendering | Not before a cell exists. Headless first |

## Help Wanted

| Contribution | What it would settle |
| --- | --- |
| A row derived on the carrier | The column's open join: one criteria row computed on the common process, with no field, action, radial direction, clock or coupling supplied for that row |
| Independent recompute of the hadron fit | The author's own load-bearing offer: the seven-input hyperfine action with its held-out failures (neutral pion, helion, alpha), recomputed by a second implementation |
| A falsification attempt on the frozen scalar tilt | Whether the author's frozen value survives the first Simons Observatory primary-CMB release, under the kill rule the author froze |
| The author's premise list | The premises and the standing of each headline result, in the author's own words on the application thread; the canonical spec takes its premise list from it |

Flow: [`ONBOARDING_MODELS.md`](../../../ONBOARDING_MODELS.md) STEP 3, then
[`CONTRIBUTING.md`](../../../CONTRIBUTING.md): fork, branch, pull request, DCO
sign-off (`git commit -s`).

## Rich Context for Deep Reader

Spec of record: [`research/m10_theory_canonical.md`](research/m10_theory_canonical.md).
Gap map and onboarding evaluation: [`research/m10_background.md`](research/m10_background.md).
Program: [`research/m10_roadmap.md`](research/m10_roadmap.md).
Agent front door: [`research/m10_agent_orientation.md`](research/m10_agent_orientation.md).
Cross-model reading map: [`research/m10_platform_pointers.md`](research/m10_platform_pointers.md).
Citations: [`theory/_CITATIONS.md`](theory/_CITATIONS.md).
Author-gated questions stay with the author.
