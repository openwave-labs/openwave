# M6 / OUROBOROS, ROADMAP (the refresh era)

> **Refresh baseline (2026-07-20).** The M6 program restarts against the author's current corpus. The archive era (sandbox v1-v11, the "permanent hold" verdict) validated the two-vector May 2026 Lagrangian and is closed, preserved read-only in [`archive/`](archive/) (its roadmap: [`archive/0b_M6_roadmap.md`](archive/0b_M6_roadmap.md)). The refresh corpus is the full latest-version Zenodo record set ([`../theory/_CITATIONS.md`](../theory/_CITATIONS.md), 29 records, harvested 2026-07-20); the specs of record and the provenance ledger live in [`m6_theory_canonical.md`](m6_theory_canonical.md) (FIRST READ).
>
> **THE PRE-ANALYSIS DECISION (2026-07-20, user-approved).** The full-corpus read split the theory in two, and the scope follows the split: the frozen two-vector v11 spec (Zenodo 20357670) is a real, well-posed model, while the July trajectory (three versions in 18 days, the particle sector silently abandoned) is not a stable validation target. **Scope = close the ledger on the frozen v11 spec, not track the moving theory.**
>
> **What the program is for.** Grade the M6 column of [`MODELS.md`](../../../../MODELS.md) (live counts on its score-board) and the particle-hunt scorecards ([`m6_particle_hunt.md`](m6_particle_hunt.md)) at M5-series rigor: pre-registered gates and conventions BEFORE each run, adversarial audit of every substantive claim, method notes per [`dev_docs/METHOD_NOTE.md`](../../../../dev_docs/METHOD_NOTE.md), honest negatives as results. Cells may move in BOTH directions; a downgrade is a deliverable.
>
> **Standing rules (canonical § 6):** version-pin every result to a Zenodo record id; no calibrated conventions (derive and pre-register, or record as a fit with its search space); tautology-screen every suite; papers are the citable layer; outbound to the author is user-gated. Task docs at `tasks/m6_<id>_task_details.md`, artifacts `m6_<id>_` prefixed, newly done rows APPEND at the END of DONE.

---

## IN PROGRESS

| TaskID | Title | Description | Gated By |
| --- | --- | --- | --- |

## BACKLOG (the active, decidable core: frozen-spec work)

| TaskID | Title | Description | Gated By |
| --- | --- | --- | --- |
| (none) | | the decidable core is exhausted; | |

> **Reopening condition:** the author freezes a spec (one record declared canonical, equations complete, stable across ≥ 1 month) OR a task below is explicitly pulled forward by user decision. Rationale: three theory versions in 18 days mean any verdict risks obsolescence on arrival (canonical § 4).

| TaskID | Title | Description | Reopen note |
| --- | --- | --- | --- |
| 🚧 M6.3 | Charge quantization in-platform: Q_CS on a converged chaoiton | The MODELS.md charge cell is ⚠️ (Lean artifacts statement-level, existence `sorry`-discharged, proof static-only). Compute the mutual Chern-Simons linking number numerically on a converged profile: integer? deformation-invariant? | 2026-07-20: the M6.2 branch-(a) gating condition was NOT met; reopens only under the condition above or an explicit user pull |
| 🚧 M6.5 | The nuclear-sector suite, rebuilt honestly | v4's six-domain A-linear case (DAMA, thermal neutron, A-scaling, NIF residuals, Sawada, global constraints) with named datasets, stated metrics, pre-registered pass/fail, train/test hygiene, tautology screen; includes checking the g_J = 0.0054 extraction against the Sawada source data and reconciling the three Sawada redefinitions + the DAMA reversal | entirely July-era, spec-dependent |
| 🚧 M6.6 | The DM sector under the current spec | (a) Re-derive the neutral-sector chain under v4 (m_φ = 0.460 MeV as fundamental): what survives of the parameter-free m_J, β(r), and the dipole-suppression chain (the \|F_dipole\|⁴ independence is asserted, not derived). (b) Execute P1-P3 of our adversarial solar-wind review ([`ai_analysis/2026-07-11_1630_dm_solar_wind_review.md`](ai_analysis/2026-07-11_1630_dm_solar_wind_review.md)): solar-vs-sidereal splitting, raw L1 rerun, synthetic controls | NOTE: arm (b) is a DATA claim on public data, spec-independent, and can be pulled forward alone on user call |
| 🚧 M6.7 | The two-theories fork: v4 vs the two-vector spec | Characterize what is lost if v4 supersedes the two-vector theory vs what it must reduce to as an effective description; produces the decision-grade question set for the author (canonical OQ1/OQ2/OQ5) | OPTIONAL: run only if continued engagement with the author is wanted; otherwise the fork answers itself by parking |

## STATUS AT A GLANCE (2026-07-21)

| Question | Answer |
| --- | --- |
| Where is M6? | The refresh arc (2026-07-20/21) ran to completion: refreshed from "permanent hold" → re-scoped (frozen-v11 ledger) → M6.1 ✅ (the certified convention sheet) → **M6.2 ✅ THE DECISION GATE, branch (b)** (electron sector closed, three MODELS.md cells → ❌) → **M6.4 ✅** (the census evidence dead, OQ3 closed negative). The decidable core is EXHAUSTED: M6 is back on HOLD with the DM sector + the M7 lineage as its durable legacy, exactly the branch-(b) contract; only the July-era programs remain, behind the reopening condition |
| Why the re-scope? | The corpus read found the July trajectory (+Eli → v4 → v5) carrying a strong drift signature: version churn faster than any validation cycle, the particle sector silently dropped in v4, fitted conventions, cannot-fail tests (canonical § 4). The frozen two-vector spec is the real, decidable object |
| What decided the program? | M6.2 (2026-07-20): the pre-registered no-search H/Q derivation gave 0.1429 vs the claimed 1.689 (91.5%), the audit confirmed 8/8 and proved the calibration state non-localized (the number is window-defined), and AF3 showed the benchmark implements the z20274505 mass-term spec rather than v11. M6.4 (2026-07-21) then closed the remaining live questions: no published stability count reproduces, the census record's numerics do not implement its own Lean theorem, and no localized charged-sector state exists at any ω for a discreteness mechanism to select (audit 9/10 + 1 partial) |
| Relation to M7 | M7 HydroBoros carries the two-vector Lagrangian in full 3D and owns that program; M6 owns the frozen-spec ledger + the provenance-hardened grading of the published record |
| Rigor bar | M5-series: pre-registered gates, adversarial audits, method notes, honest negatives as results; canonical § 6 consumption rules standing |

## DONE (refresh era)

> Newly done tasks APPEND at the end. The archive-era validated record (sandbox v1-v11, gates G1+, the DM-paper supplement) stays in [`archive/`](archive/), read-only; do not conflate the two eras.

| TaskID | Title | Description | Completed |
| --- | --- | --- | --- |
| ✅ (setup) | The 2026-07-20 refresh setup | Corpus harvested (29 latest-version Zenodo records + local context corpus, manifest [`../theory/_CITATIONS.md`](../theory/_CITATIONS.md)); full corpus read digested; [`m6_theory_canonical.md`](m6_theory_canonical.md) + [`m6_particle_hunt.md`](m6_particle_hunt.md) + [`../__M6_model_briefing.md`](../__M6_model_briefing.md) refresh + this roadmap written | 2026-07-20 |
| ✅ (pre-analysis) | The scope decision: frozen-spec ledger, not the moving target | Corpus-wide drift assessment presented and user-approved: active core = M6.1/M6.2 (decision gate)/M6.3/M6.4 against the frozen v11 spec; M6.5/M6.6 with the reopening condition; M6.7 optional. Evidence base: canonical § 4 ledger + the local provenance notes | 2026-07-20 |
| ✅ [M6.1](tasks/m6_1_task_details.md) | The spec certification gate | v11 certified: [`m6_1_v11_convention_sheet.md`](m6_1_v11_convention_sheet.md) (FIXED-vs-GAP pre-registration checklist; ℒ_ref reconstruction; targets pinned) + v4 characterized script-backed (EL unclosable, boundedness verified, A-linearity generic). Four print-level v11 defects added to canonical § 4; OQ6 resolved. Adversarial audit 9/10 CONFIRMED + 1 PARTIAL, auditor findings AF1/AF2 recorded. [Method note](findings/m6_1_method_note.md) | 2026-07-20 |
| ✅ [M6.2](tasks/m6_2_task_details.md) | THE DECISION GATE: H/Q under pre-registered conventions | **Branch (b)**, the electron sector closed. Pre-registered ([`m6_2_preregistration.md`](m6_2_preregistration.md)) before numerics; the derived no-search pairing misses the v11 point by 91.5%, and the audit proved the calibration state is NOT localized (no decaying channel at any amplitude). Audit 8/8 CONFIRMED. MODELS.md 3 cells → ❌, column re-sorted. [Method note](findings/m6_2_method_note.md) | 2026-07-20 |
| ✅ [M6.4](tasks/m6_4_task_details.md) | The stability census rerun + the ω-selection hunt | All three published stability counts fail reproduction, the record's numerics do not implement its own Lean theorem. **OQ3 CLOSED negative in its strongest form**: no localized charged-sector state at ANY in-window ω, every ladder ω above every window, the electron's ω = 1 exactly on the boundary. Audit 9/10 CONFIRMED + 1 PARTIAL, 0 refuted. [Method note](findings/m6_4_method_note.md) | 2026-07-21 |
