# M10 OPH (Observer Patch Holography): Model Briefing

The core intuition behind OPH is that reality is the self-answering question:
“What would a universe that creates itself consistently be like?” The key word
is *consistently*: mathematical and logical consistency, free of contradiction.
OPH aims to show that the universe it constructs on this basis is precisely
the one we inhabit.

## From measurements to spacetime

Start with two observers who meet, separate and meet again. In Observer Patch Holography each is a finite patch on a spherical screen of twelve-port icosahedral cells. At each meeting they can read the same event on their overlap. Between meetings each keeps its own records, compares readings with neighbours and repairs disagreements locally. A reading that survives every allowed comparison is a fact the observers can share.

Each observer reads time from within its own history. The quantities accessible to its patch form its readout algebra. Its state determines a modular flow on the support of that algebra, and a local clock calibrated against this flow measures its elapsed time. At reunion the observers can compare their hands. The simulator’s schedule supplies no universal present. [Observer clocks, p. 6, Definition 1.3 and Remark 1.4](https://github.com/FloatingPragma/observer-patch-holography/releases/download/r2072/observers_are_all_you_need.pdf#page=6)

Why call this *holography*? The records live on a two-dimensional screen. Under OPH’s reconstruction premises, a spacetime with three spatial dimensions and one time dimension can account for the measurements observers make. The simulation shows the two views side by side. On the left are the cells and their clock hands. On the right is the view one observer builds as signals reach it. [Watch the OPH simulation](https://floatingpragma.ai/oph/simulation/) · [Spacetime reconstruction, §1, p. 2, paragraph “What a test of the model is”](https://github.com/FloatingPragma/observer-patch-holography/releases/download/r2072/executable_observer_construction.pdf#page=2)

The executable uses local repair and memory rules on this discrete carrier. OpenWave admitted OPH as M10 on 13 November 2026. All criteria entered untested. The M10.3 review awards partial credit for particle stability on Interval Echo. [Admission discussion](https://github.com/openwave-labs/openwave/discussions/712) · [M10.3 decision](research/findings/m10_3_maintainer_review.md)

[![OPH conceptual schematic: overlapping observer patches on a spherical carrier network, with one patch mapped to an observer's complete history in emergent spacetime](https://raw.githubusercontent.com/FloatingPragma/observer-patch-holography/bb1de05927d8c6bb1285182cc91e97410092ec8a/assets/oph-schematic.png)](https://github.com/FloatingPragma/observer-patch-holography/blob/3c4e1e23fc32dd241e8c2bfba75de123fe5c6984/flagship/from_observer_consensus_to_standard_physics.pdf)

*Conceptual schematic selected by Bernhard Mueller, generated with AI assistance.
The image illustrates OPH's proposed patch/history interpretation; the depicted
human history and curved spacetime are the physical target. Finite static
experiments do not establish that reconstruction. Click for the synthesis paper.
[Image source](https://github.com/FloatingPragma/observer-patch-holography/blob/bb1de05927d8c6bb1285182cc91e97410092ec8a/assets/oph-schematic.png),
[CC BY-NC-SA 4.0](https://github.com/FloatingPragma/observer-patch-holography/blob/bb1de05927d8c6bb1285182cc91e97410092ec8a/LICENSE).*

## Identity

| Field | Value |
| --- | --- |
| Model ID | M10 |
| Name | OPH (Observer Patch Holography) |
| Author | Bernhard Mueller, Pragma Research Inc., corresponding author. Not the sole author: the flagship paper lists fourteen authors (the author's statement in the application) |
| Author contact | GitHub [@muellerberndt](https://github.com/muellerberndt), for author-gated questions (definitions, intent, which construction is the specification of record); routing convention in [`dev_docs/CROSS_MODEL_TESTING.md`](../../../dev_docs/CROSS_MODEL_TESTING.md) § 6 |
| Disclosure | A co-author of the flagship is the applicant of K7 ([discussion #525](https://github.com/openwave-labs/openwave/discussions/525)), and the author's repository vendors K7 comparison files. The two are separate frameworks, and K7's standing is unchanged by this column (the author's disclosure, and the admission reply) |
| Lineage | Wheeler's law without law; the relational and Darwinist readings of quantum theory (Rovelli, Zurek); distributed agreement (Lamport, Dijkstra's self-stabilization); Jacobson and Van Raamsdonk for gravity from entanglement thermodynamics; Doplicher and Roberts for gauge groups from superselection; Bisognano and Wichmann for modular clocks; Wilson's lattice gauge theory; 't Hooft's cellular automaton interpretation as the closest relative among discrete deterministic substrates |
| Primary sources | The author’s [OPH repository](https://github.com/FloatingPragma/observer-patch-holography) contains the executable model. The [canonical specification](research/m10_theory_canonical.md#1-specification-of-record-and-the-pin) distinguishes the admission snapshot, the earlier M10.2 pin and the adopted M10.3 pin. The [bibliography](theory/_CITATIONS.md) lists the admission sources. |
| In-repo | Headless only. No launcher. The certification gate's record: [`research/tasks/m10_1_task_details.md`](research/tasks/m10_1_task_details.md) |
| Application | [discussion #712](https://github.com/openwave-labs/openwave/discussions/712); admission reply [discussioncomment-19595538](https://github.com/openwave-labs/openwave/discussions/712#discussioncomment-19595538) |

## Active candidate submission

**Interval Echo is the pinned candidate for M10** (task M10.3), with particle stability in free evolution (row 4) at ⚠️ partial. A shared seam balances its endpoint loads and records their previous signed difference. At exposure 37 the computed probability that both loads occupy a seam in the launch cell is **0.980** at the supplied coupling $\kappa = 8$ and **0.203** with the contrast interaction switched off. Of 4,096 seeded draws from each computed outcome distribution, 4,028 and 824 respectively returned a launch-cell seam. The pair’s charge, spin and species are unassigned. The pin was adopted and the row decided on 2026-11-21. [Maintainer decision](research/findings/m10_3_maintainer_review.md) · [Pinned candidate](research/m10_theory_canonical.md#12-interval-echo-candidate-for-m103) · [Task and replay](research/tasks/m10_3_task_details.md) · [Method, control and simulated counts](research/findings/m10_3_method_note.md) · [Row 4 animation with a cell-0 release](https://floatingpragma.ai/oph/simulation/openwave/row-4/)

**M10.2 is the earlier signed-source candidate, kept as ancestry.** Its graph and energies reproduced. Integer loads, the 511 keV calibration and the half-turn symmetry limited what those results established about charge, rest energy and spin. [M10.2 task](research/tasks/m10_2_task_details.md) · [Maintainer decision](research/findings/m10_2_maintainer_review.md)

## Model Profile (what it brings, short form)

The table records the model at admission. For the scored candidate's carrier, seam law, preparation and observer, see the [Interval Echo specification](research/m10_theory_canonical.md#12-interval-echo-candidate-for-m103).

| Attribute | Admission construction |
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
| The defining open problem | Row 4 has an in-platform computation on the pinned carrier. Extending it to other criteria requires computing their observables on this candidate. The separate sources at admission and the M10.2 static assays retain their own scope. [M10.3 review](research/findings/m10_3_maintainer_review.md) |
| How a cell moves | A declared or installed source is inherited and leaves its row not yet tested. A row moves when its observable is computed on the carrier's own process in-platform ([`research/m10_theory_canonical.md`](research/m10_theory_canonical.md) § 8) |
| Predictions | None passes the prediction test of [`ONBOARDING_MODELS.md`](../../../ONBOARDING_MODELS.md) § 1.1, on the author's own audit. The lepton ratios, the hadron spectrum and the tau width ratio are comparisons with known data |
| Residuals on record | The author lists them unprompted: the muon ratio 442 standard uncertainties from measurement, the held-out hadrons and light nuclei failing, the gravity unit attachment off by 1.18 × 10⁴⁰, the weak mixing angle of the declared coupling chain at 0.23074 against 0.23129(4) |

## Field Configuration of Particles

Interval Echo tests a prepared pair of unit loads with unassigned species. The other entries describe the admission construction and its separate sources.

| Particle | Configuration in OPH | Topological vortex? |
| --- | --- | --- |
| Interval Echo, prepared pair | Two units start on one seam. The terminal observer reads one of 30 seams in the launch cell or quiet. | The assay does not measure a vortex. [M10.3 review](research/findings/m10_3_maintainer_review.md) |
| Admission picture, any species | A protected flux defect of the carrier network, labelled by a centre class in Z6 | The intended reading is a topological defect of overlap transport. No localized defect is established for any species (the author's hypothesis page) |
| Electron and charged leptons, in the row constructions | A state of a declared finite source carrying the fifteen-state module; the three families from a register on one carrier face | No |
| Quarks | Phase lifts of a closed loop of (SU(3)×U(1))/Z3; no localized texture | No |
| Baryons, mesons, light nuclei | States of a calibrated hyperfine action on two-cell blocks and boxes | No |
| Photon | A transverse mode of the declared Maxwell cochain complex | No |

## Implementation Status

The official M10 matrix marks particle stability ⚠️ (partial) and every other criterion 🚧 (not yet tested). The table below records earlier author constructions and the M10.1 certification. [Official M10 matrix](../../../MODELS.md#oph-m10)

| Sector | Status |
| --- | --- |
| Native carrier and repair law | Author-side: confluent repair with one endpoint, event order from provenance, a rank-three slowest band of the repair mean on the twelve-port carrier, exact fluctuation identities. Statements about declared finite objects, machine-checked in the author's Lean library by the author's account; not rebuilt in-platform |
| All criteria rows except row 4 | 🚧 The author's constructions exist for every row, each on its own declared finite source. Inherited until computed on the carrier's own process |
| Lepton register | 🚧 Koide's 2/3 and the phase 2/9 from a register on one carrier face. The headline numbers were recomputed at admission from the stated formulas; the result is a fit by the author's own parameter count ([`research/m10_background.md`](research/m10_background.md) § 4) |
| Hadron source | 🚧 Calibrated to seven measured inputs; the held-out comparisons fail and are on record |
| Certification gate | ✅ M10.1 closed 2026-11-13: the author's frozen coefficients of the twelve-direction dispersion are reproduced exactly in clean rooms. Arithmetic on a declared object, so no cell moves. Note: [`research/findings/m10_1_method_note.md`](research/findings/m10_1_method_note.md) |

## Roadmap

| Task | What lands |
| --- | --- |
| M10.1 | Certification gate, closed 2026-11-13: a maintainer-run recompute of the twelve-direction scalar dispersion in clean rooms. The author's frozen coefficients agree exactly. It certified arithmetic on a declared object and moved no cell |
| [M10.2](research/findings/m10_2_maintainer_review.md) | The static assays on the signed-source candidate reproduced. Charge, rest energy and spin each depended on a declared input and kept their admission marks. |
| [M10.3](research/findings/m10_3_maintainer_review.md) | OpenWave adopted Interval Echo. The free pair on its two-load generator earned partial credit for row 4 at the supplied coupling. |
| Later tasks | Created and ordered by the author on [`research/m10_roadmap.md`](research/m10_roadmap.md) |
| Rendering | The platform replay is headless. The author provides a separate [cell-0 animation](https://floatingpragma.ai/oph/simulation/openwave/row-4/). |

## Help Wanted

| Contribution | What it would settle |
| --- | --- |
| A row derived on the carrier | The column's open join: one criteria row computed on the common process, with no field, action, radial direction, clock or coupling supplied for that row |
| Independent recompute of the hadron fit | The author's own load-bearing offer: the seven-input hyperfine action with its held-out failures (neutral pion, helion, alpha), recomputed by a second implementation |
| A falsification attempt on the frozen scalar tilt | Whether the author's frozen value survives the first Simons Observatory primary-CMB release, under the kill rule the author froze |

The author's [premise/status comment](https://github.com/openwave-labs/openwave/discussions/712#discussioncomment-19624288) supplies the
promised premise list and headline-claim standing; its condensed register is in
[canonical § 11.1](research/m10_theory_canonical.md#111-author-premise-and-claim-register). The
[94-entry theorem and observable inventory](research/m10_theorem_observable_inventory.md)
gives an intuition, source evidence and boundary for each selected result,
including the Einstein tower, de Sitter capacity-transfer analysis, Koide and
published native field progress. Its public RER snapshot is separate from the
source scored by a platform task.

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
