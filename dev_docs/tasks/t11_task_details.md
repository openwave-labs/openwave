# T11: One written bar: every validated cell re-read

> Platform task **T11**. Status: 🔶 **RUN 2026-11-18, the pull request waits for the author of the M5 column** (ruled 2026-11-15, the day of the trigger; staged and run 2026-11-18) · Roadmap: [`../platform_roadmap.md`](../platform_roadmap.md) · Deliverable lands in [`MODELS.md`](../../MODELS.md) (the bar, the `static` sentence, the restated row tests, the re-read cells, the score-board) and in this record.

## PLANNING

**Origin.** The review of [#723](https://github.com/openwave-labs/openwave/pull/723) declined three M10 rows. In each the native process ran and the numbers were right, and the property the row names came from what the candidate declares: an integer state, a single anchor, the symmetry of the cell. The review said in the thread that an M5 cell on the same row records a statement of the same kind, and that the same question goes to that cell. The user's ruling: the bar is one bar for every column, the platform's longest-standing column included, and it belongs in `MODELS.md` itself, where a contributor reads it before filing, not in a review, where it is read after.

**Why now, and what it is.** The cells that carry a validated icon were set over several months, on the review practice of their day. That practice has since gained the independent recomputation, the adversarial audit by a second model, the mutation test of every printed pass line, and the declared-source rule. T11 reads every validated cell under the bar, in every column; any cell that moves does so on its own evidence. It is not a judgment on any model: a cell that moves keeps its result in its text, and says what would restore the icon.

**Scope.**

| Item | Content |
| --- | --- |
| The bar | One short paragraph in `MODELS.md` beside "Each icon is earned": what a validated, a partial and a not-yet-tested cell each require, in terms of what the model's own process returns against what the model declares before any run. Its text is fixed in § 1 below, before any cell is read |
| The `static` sentence | `MODELS.md` defines a static row as "earnable from structure or spectrum alone". Restated so that the regime says what kind of run a row needs, not what counts as earned (§ 1) |
| The row tests | Where a row's simplest test can be passed by a declared property, the sentence is restated; which rows, if any, is found in the re-read. Two candidates are written in § 1 so that they are on record before the cells are read |
| The re-read | Every cell that carries a validated icon, in every column: the evidence it cites today, what kind of evidence that is, the best current evidence, the icon under the bar, the cell text, and what would restore the icon. On the day of staging that is twelve cells in two columns |
| The score-board | Recounted from the rows, with the linter |

**Method, per cell.** One table row each, with five entries:

| Entry | Content |
| --- | --- |
| Cited evidence | The scripts and records the cell links today, each opened and read |
| Kind | An outcome of the model's process; an identity of the declared form; an anchor with nothing left to compare; or a result on a retired engine or stack |
| Best current evidence | What the column's own record holds on the current stack, with paths |
| Icon and text | The icon under the bar, and the cell text inside the 65-word budget, state of the art only |
| The way back | The concrete run or evidence that would restore the validated icon, specific enough to write a task from |

The table for a column is drafted by whoever knows the column's record best and audited adversarially by a different model, which re-opens every cited script and tries to refute each reading in both directions: a cell marked down that should stand, and a cell left standing that should not.

**The authors.** A reading of a past result is author-gated ([`PR_REVIEW_STANDARDS.md` § 8.1](../PR_REVIEW_STANDARDS.md#81-when-the-contributor-is-not-an-author-of-the-column)); the icon is the platform's, on the evidence bar of [Gate E](../PR_REVIEW_STANDARDS.md#7-gate-e-modelsmd-cell-changes). No cell moves on `main` before its column's author has seen the row as it would read. The work is committed to the working branch first; the pull request that lands it tags, on its first line, the author of each column whose rows that author filed or reviewed, and states, in order: what triggered the re-read, why the cell reads differently under the bar, and what evidence would move it back. An author's reply that supplies evidence the re-read missed is exactly what the tag asks for. A column whose rows the maintainers ran themselves, before the present review practice, has the maintainer as the person to inform, and needs no tag.

**Definition of done.** The bar, the `static` sentence and any restated row test are in `MODELS.md`; every validated cell has its row in the table of the findings and its adversarial audit; each cell that moves carries the way back in its text; the score-board, the criteria counts stated across the repository and both linters are clean; this record holds the tables and the ruling on the row tests; the task review is presented in the terminal. The pull request and its merge are a separate step, after the authors.

**Gating.** The user's go. Docs-only: no engine run, no new result. A cell whose way back needs a run gets a roadmap row on its own column, filed by that column's author or with the author's agreement.

**Model and effort.** The session's model for the tables; one audit agent on a different model, named at the spawn, in both directions.

**Research body.** This document (`## FINDINGS` after the run); the edits land in [`MODELS.md`](../../MODELS.md) and in the files that restate its counts; no script and no data file.

**Consumers.**

| Sub-result | Consumer | The check that consumer needs |
| --- | --- | --- |
| The bar text | every contributor who files a row; review Gate E | it agrees with the rule already published in the M10 section of `MODELS.md` and quotes no cell as its example before that cell is read |
| The per-cell tables | the column authors; the pull request that lands the rows | every cited script opened, not read from its header; the audit in both directions |
| The cell texts | a reader of the matrix | inside the word budget, state of the art only, each moved cell naming what would move it back |
| The score-board | the column order | recounted by the linter from the rows |

### 1. The texts fixed before any cell is read

These are written here first, so that the cells are read against a written text and the text is not tuned to an outcome.

**The bar** (to stand beside "Each icon is earned"):

> A cell reports what the model's own process returns. A property fixed by what the model declares before any run (an assigned integer, a calibrated value, a symmetry of the declared field or cell) is recorded in the cell's text and earns no icon: a declared or installed source is inherited, and the row moves when its observable is computed on the model's own process in-platform. **Validated**: the property comes out of that process and meets the row's test. **Partial**: it comes out with something still supplied, or with part of the row open. **Not yet tested**: nothing runnable returns it yet, whatever the model declares.

**The `static` sentence:** `static` = the row can be earned without time evolution, from a state the model's own process selects and the structure of that state.

**Two candidate restatements of a row's simplest test,** adopted only if the re-read finds that the present sentence can be passed by a declared property:

| Row | The sentence today | The candidate |
| --- | --- | --- |
| Electron rest energy (mass) | localized state anchored to 511 keV | a localized state the dynamics selects, set to 511 keV by one anchor; validated when a second quantity fixed by that anchor agrees with measurement |
| Spin-½ statistics (720° double cover) | the field returns under a π rotation while its frame needs 2π | a measured sign of −1 under a 2π rotation or under the exchange of two cores; a half turn that the dynamics selects against a polar control earns partial |

The first test sentence is the platform's; the second was proposed for its map by the author of the column that first carried the row ([T1](t1_task_details.md)), and the candidate follows that author's wording.

### 2. Blindspot pass, the audit and the shape of the work

| Risk | Guard |
| --- | --- |
| A partial re-read is worse than none: moving the cells the trigger named while others of the same kind stand would repeat the inconsistency at a larger size | the scope is every validated cell, in every column |
| The bar tuned to an outcome | § 1 is written before the cells are read, and the cells are read against it |
| A re-read by the session that knows a column best is also the session most used to its claims | the audit is by a different model and runs in both directions |
| Strictness becoming a second standard for older cells | a cell that passes keeps its icon without a caveat it did not need before |
| A cell read from a script's header | every cited file is opened; the table's "Kind" entry names what the script computes, with the line |
| The column order | it sorts on validated plus partial, so a move from validated to partial does not reorder the matrix; a move to not yet tested can, and is flagged before it lands |
| A statement about a model's physics | none is made by this task; it reads what each cell's own cited evidence shows |
| Counts restated elsewhere | the linter's count check runs over every live document that states a column tally |

The audit: one agent on a different model receives the bar of § 1 and the drafted tables, opens every cited file itself, and returns a verdict per cell in both directions, with the line it rests on.

The shape of the work: desk work, no computation. The cells are read in sequence by column; the audit runs once on the full tables.

## FINDINGS (2026-11-18)

> Re-read of 2026-11-18. Twelve cells carried a validated icon on that day: nine in the M5 column and three in the M6 column. Every file a cell cited was opened and read in full; the reading of each file is in the tables below, and the cells as they now stand are in [`MODELS.md`](../../MODELS.md). No script was run and no result is new.

**TL;DR.** (1) The bar, the restated `static` sentence and two restated row tests are in `MODELS.md`. (2) M5: two cells keep the validated icon on evidence from the verified-L stack (particle stability, antimatter), four read as partial (charge quantization, electron rest energy, de Broglie clock, Coulomb) and three as not yet tested (spin-½ statistics, EM waves, Klein-Gordon). (3) M6: one cell reads as partial (dark matter candidate) and two as not yet tested (particle stability, EM waves). (4) The score-board: M5 goes from 9 validated and 11 partial to 2 and 15, M6 from 3 and 2 to 0 and 3; the column order does not change. (5) No script of either column was found wrong: in each cell that moved, the cited evidence is an identity of a declared form, an anchor, or a result on a retired stack, and the cell now says so and names what would move it back.

### 3. The M5 column

| Cell | Cited evidence, and its kind | Best current evidence | Icon under the bar | The way back |
| --- | --- | --- | --- | --- |
| Charge quantization | [`m5_1_winding.py`](../../openwave/xperiments/m5_liquid_crystal/research/scripts/m5_1_winding.py): the winding number read on a seeded hedgehog of the retired vector engine, no relaxation; the integer is the sign passed to the seed. An evaluation on a seeded form | The seeded integer is an outcome in the sense that it holds: q = 1.000 at every radius through one free axisymmetric evolution without a clock, with the outer cells pinned (the canonical stack that preceded verified-L), which the record itself calls trivial for a topological number, and where the axes are near-degenerate the fixed-radius read is ambiguous ([`m5_21_films.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_21_films.md), the gate of [`m5_21_c_clockrun.py`](../../openwave/xperiments/m5_liquid_crystal/research/scripts/m5_21_c_clockrun.py)); on the verified-L stack the seeded states read integer to 2e-4 and an antipair's ±1 goes to 0 under damped dynamics ([`m5_21_4_note.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_21_4_note.md)); net degree 1 kept through annealing, carried by two +½ lines ([`m5_21_2b_note.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_21_2b_note.md)) | ⚠️: the integer is assigned in the seed and the far winding is held by pinned cells; what the dynamics returns is that it holds and that an antipair goes to zero. The test's clause on a split is open: the relaxed unit is two +½ lines | a free-boundary run in which the unit stays whole when its two +½ lines are pulled apart, the charge followed by an instrument that tracks the branch |
| Electron rest energy | [`m5_6_3b_faber_on_M.py`](../../openwave/xperiments/m5_liquid_crystal/research/scripts/m5_6_3b_faber_on_M.py): energies of an analytic profile on rescaled grids, so E·r₀ is constant by construction. [`m5_16_task_details.md`](../../openwave/xperiments/m5_liquid_crystal/research/tasks/m5_16_task_details.md): a relaxation in the spherically constrained class (the unconstrained one leaves the hedgehog) in which c₂ and the 511 keV scale are the two inputs and the half-energy radius, 2.93 fm, is the output; the 2.2132 fm of the old cell text is the parameter of the reference soliton, not of this anchor. An anchor, and one output that lies far above the bounds on electron structure | The second quantity fixed by the anchor is the clock, ħΩ = m_e c²; at fixed J it does not come out, the ratio 2JΩ/E being set by the box ([M5.42](../../openwave/xperiments/m5_liquid_crystal/research/tasks/m5_42_task_details.md)) | ⚠️: a state comes out of a relaxation once the anchor is set; the test's second quantity does not | a charge whose size a run fixes, the same on two boxes and under the bound; then ħΩ/E = 1 on that state, the clock's motion being a symmetry of it |
| de Broglie clock | [`m5_9_lepton_mass_clock_findings.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_9_lepton_mass_clock_findings.md): the principle rests on an analytic toy with an assumed ω ∝ 1/r₀, and its own text says the dynamical readout is blocked. [`m5_27_note.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_27_note.md): time evolution on the certified action, a null for entrainment, with a free release that loses 97 percent of J. [`m5_21_14_note.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_21_14_note.md), [`m5_34_kit_checks.py`](../../openwave/xperiments/m5_liquid_crystal/research/scripts/m5_34_kit_checks.py): symbolic and arithmetic. "Self-starting" and "energy-minimizing" are results of the retired stack | On the verified-L stack free descent has no stationary frequency ([`m5_21_3_note.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_21_3_note.md), [`m5_21_15_note.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_21_15_note.md)); at fixed J the clocked state exists with dE/dJ = ω* at the 1 percent level ([`m5_21_9_note.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_21_9_note.md)); with the rigid twist the ratio 2JΩ/E is set by the box ([M5.42](../../openwave/xperiments/m5_liquid_crystal/research/tasks/m5_42_task_details.md), [M5.44](../../openwave/xperiments/m5_liquid_crystal/research/tasks/m5_44_task_details.md)) | ⚠️: the clock exists as a state with J supplied; the frequency at mc²/ħ and the self-start do not come out | the same as the rest-energy row |
| Particle stability | [`m5_8_2g_spontaneity.py`](../../openwave/xperiments/m5_liquid_crystal/research/scripts/m5_8_2g_spontaneity.py): a time evolution on the retired quartic stack whose outcome is growth from a state at rest. [`m5_21_10_note.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_21_10_note.md): free dynamics on the verified-L stack, an outcome | The control electron of M5.21.10, started from a mid-descent snapshot, holds over t = 150 at N = 64 (4.4 percent of its starting energy absorbed, a decline of 5.6 percent against the starting budget), one window at one box, audited; its half-energy radius follows the box ([`m5_21_5_note.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_21_5_note.md)) | ✅, with the evidence base restated to the verified-L stack and the non-compact core named | not needed; a second box and a longer window would say whether the hold has a size of its own |
| Spin-½ statistics | [`m5_8_2s_spin_half_apolar.py`](../../openwave/xperiments/m5_liquid_crystal/research/scripts/m5_8_2s_spin_half_apolar.py): M evaluated at φ, φ + π and φ + π/2 on an analytic seed; M is quadratic in its frame, so the pass line is true by construction. An identity of the declared form | From runs: the relaxed unit carries two +½ lines ([`m5_21_2b_note.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_21_2b_note.md)). No sign under a 2π rotation or an exchange is measured | 🚧: the property is declared; the test, as restated, asks for a measured sign | half lines selected by the dynamics against a polar control, for partial; a measured −1, for validated |
| Antimatter + annihilation | [`m5_21_4_note.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_21_4_note.md), [`m5_21_4_a_pair.py`](../../openwave/xperiments/m5_liquid_crystal/research/scripts/m5_21_4_a_pair.py): descent and a capture run on the verified-L stack, with interior damping and a sponge, audited twice. An outcome | the same | ✅; the cell now names the damping under which the pair annihilates and that the outgoing radiation is not identified | not needed |
| Electric force (Coulomb) | [`m5_21_4_note.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_21_4_note.md): an attractive interaction at every separation at seed level, the pair closing and annihilating under descent and in the capture run, the 1/d form not confirmed at reachable separations, like charges forming a string at ansatz grade. [`m5_21_5_note.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_21_5_note.md): the charge e is "exact by construction" through the anchor. [`m5_1_coulomb.py`](../../openwave/xperiments/m5_liquid_crystal/research/scripts/m5_1_coulomb.py), [`m5_4_coulomb_matrix.py`](../../openwave/xperiments/m5_liquid_crystal/research/scripts/m5_4_coulomb_matrix.py): a director relaxed on the one-constant Frank energy with pinned core signs, not the verified-L stack | the two-body outcome of M5.21.4 | ⚠️: an attraction comes out; the far field's strength is an anchor and the 1/d law is open | the 1/r⁴ tail and the 1/d law on relaxed states, extrapolated in the box at fixed spacing, with α not fitted |
| EM waves (Maxwell) | [`m5_6_4a_hydro_em.py`](../../openwave/xperiments/m5_liquid_crystal/research/scripts/m5_6_4a_hydro_em.py): vector identities on a generic periodic field with no model field in it. [`m5_6_4b_faber_curvature_em.py`](../../openwave/xperiments/m5_liquid_crystal/research/scripts/m5_6_4b_faber_curvature_em.py): the curvature of an analytic hedgehog. Identities; the cell's sentence on tilt modes at c had no script behind it | [`m5_8_c_isotropy_gate.py`](../../openwave/xperiments/m5_liquid_crystal/research/scripts/m5_8_c_isotropy_gate.py): the uniform vacuum has no linear propagation (a cubic force), and the one time evolution there is a linear twist equation on a frozen analytic background. The author's results on the founding quartic are in the thread record ([`m5_32_convo.md`](../../openwave/xperiments/m5_liquid_crystal/research/tasks/m5_32_convo.md), 2026-11-16) | 🚧: no transverse wave is computed on the model's field | a transverse wave at c computed on the model's own action |
| Klein-Gordon | [`m5_6_1_kg_operator_check.py`](../../openwave/xperiments/m5_liquid_crystal/research/scripts/m5_6_1_kg_operator_check.py): a symbolic identity on the analytic connection, which itself records the bare phase as massless. [`m5_6_1b_twist_evolution.py`](../../openwave/xperiments/m5_liquid_crystal/research/scripts/m5_6_1b_twist_evolution.py): a linear scalar evolved on a prescribed background with an imposed regularization. A derivation | One spectral read on a relaxed field, at toy parameters and on a sliding background ([`m5_21_1_method_note.md`](../../openwave/xperiments/m5_liquid_crystal/research/findings/m5_21_1_method_note.md)): a gapped branch with a roton-like minimum, which its own verdict calls not Klein-Gordon-like | 🚧: the row is `dynamic`, which a spectrum alone cannot earn; the one read on a relaxed field is not Klein-Gordon-like, and no time evolution of the model's field returns the relation | the dispersion of small perturbations about a converged relaxed state, read in a time evolution, giving ω² = k² + m²; the rate Ω/γ of a moving clocked core earns no more than partial |

### 4. The M6 column

The three cells date from the column's first months and were set by the maintainers on the archive-era record.

| Cell | Cited evidence, and its kind | Best current evidence | Icon under the bar | The way back |
| --- | --- | --- | --- | --- |
| Particle stability | [`sandbox_v11/`](../../openwave/xperiments/m6_ouroboros/research/archive/sandbox_v11/): a boundary-value solve of a static radial equation for the neutral profile (nodeless; the K₁ tail is set by the outer boundary condition). The neutral equation has no frequency in it, so the "escape via oscillation" of the old text is not a property of the solved state. No file evolves a field in time; the two method notes concern the charged sector | the same; the charged-sector negatives of [`m6_2_method_note.md`](../../openwave/xperiments/m6_ouroboros/research/findings/m6_2_method_note.md) and [`m6_4_method_note.md`](../../openwave/xperiments/m6_ouroboros/research/findings/m6_4_method_note.md) stand | 🚧: the row is `dynamic`; a localized profile comes out of a static solve, and nothing evolves it in time | free evolution of the neutral profile under the certified v11 spec |
| Dark matter candidate | [`sandbox_v11/`](../../openwave/xperiments/m6_ouroboros/research/archive/sandbox_v11/): the same solve. The mediator mass follows from m_e through the scaling symmetry and a conversion recipe; the 0.460 MeV is the target of a root find on λ. The supplement recomputes from the same profile, reproduces one integral of the draft paper and differs on the others. An outcome with one value supplied and stability untested | the same; the column's roadmap lists the re-derivation as [M6.6](../../openwave/xperiments/m6_ouroboros/research/m6_roadmap.md) | ⚠️ | a stability run, and the mass chain re-derived |
| EM waves (Maxwell) | [`0d_canonical.md`](../../openwave/xperiments/m6_ouroboros/research/archive/0d_canonical.md): "structural fact", an assertion about the A field as J → 0. The linear spectrum is not computed, and the second audit's hand check doubts the assertion for the coupled pair. No script computes a dispersion, a speed or a polarization. Declared | none | 🚧 | a propagating transverse mode at c computed under the certified v11 spec |

### 5. The texts, the tests and the score-board

| Item | Outcome |
| --- | --- |
| The bar | in `MODELS.md` as fixed in § 1, with two additions made after the cells were read: a sentence that dates the re-read and links this record, and a parenthesis in the partial sentence ("a value the model is calibrated to earns nothing by itself; a state the process returns once that value is set can"), added because the first audit found the fixed text ambiguous between "earns no icon" and "something still supplied" |
| The `static` sentence | restated as in § 1 |
| The rest-energy test | restated as in § 1: the re-read found that the earlier sentence is passed by the anchor alone |
| The spin-½ test | restated as in § 1: the re-read found that the earlier sentence is passed by an identity of the declared form |
| Statements elsewhere | brought in step with the cells: three sentences of `MODELS.md` outside the twelve cells (the M5 Lorentz cell's clause on tilt modes at c and an emergent Klein-Gordon dispersion, which had no measurement behind it; the opening of the M5 orbital cell, which took the clock as established; and the paragraph "Reading the table", which tied stability to time-periodicity in every framework), the two model briefings, the two criteria pages ([`m5_particle_hunt.md`](../../openwave/xperiments/m5_liquid_crystal/research/m5_particle_hunt.md), [`m6_particle_hunt.md`](../../openwave/xperiments/m6_ouroboros/research/m6_particle_hunt.md)) and one line of the M5 question tracker |
| The score-board | recounted by the linter: M5 2 validated, 15 partial, 2 negative, 12 not yet tested; M6 0, 3, 3, 25. Validated plus partial: M5 20 → 17, M6 5 → 3. The column order is unchanged |

### 6. The audit

The audit of record was made by an agent on a model other than the one that drafted the tables. It opened every cited file itself, gave equal weight to a mark set too high and a mark set too low, applied the bar as written, and counted a cell's mark as supported where two adjacent marks were both defensible.

| Cell | The drafted mark | The audit's mark | Outcome |
| --- | --- | --- | --- |
| M5 charge quantization | ✅ | ⚠️ | adopted: the integer is seeded and the far winding is held by pinned cells; the test's clause on a split is open |
| M5 Klein-Gordon | ⚠️ | 🚧 | adopted: a `dynamic` row; the one output of a process is a spectral read that is not Klein-Gordon-like |
| M6 particle stability | ⚠️ | 🚧 | adopted: a `dynamic` row; a boundary-value solve with no evolution and no stability analysis |
| The other nine | as in §§ 3 and 4 | the same | supported; for seven of them the audit also calls an adjacent mark defensible (M5 particle stability and antimatter toward partial, spin-½ toward partial, Coulomb and M6 dark matter toward not yet tested, rest energy and the clock toward a negative) |

The marks in `MODELS.md` are the audit's on all twelve cells. The audit found no cell marked too low. It confirmed the score-board against the rows, and found both restated tests needed: each earlier sentence could be passed by an anchor or by an identity of the declared form.

**An earlier pass.** A first audit pass ran on a brief that named three cells to press on, the three M5 cells then drafted as validated. It returned lower marks on eight cells; the maintainer set it aside as led by its brief and had the audit repeated on the neutral brief above. Its checks of the cell texts were kept: of the nineteen statements corrected in the drafted texts, ten came from it and nine from the audit of record, none of them changing a mark. Among the corrections: the stack named for the charge evolution, the anchor of the rest-energy cell (the 2.2132 fm of the old text is the reference soliton's parameter), the energy decline and the starting snapshot of the stability window, the damping under which the antipair annihilates, the seed-level grade of the two-body attraction, the forms on which the Maxwell identities were checked, and the assertion behind the M6 EM cell.

**Left open by the audit.** The clock cell's way back names a precondition and not the frequency itself; the validated route of the spin cell turns on an assignment that is itself a choice; the mass recipe of the M6 dark-matter cell uses a ratio the M6.2 review questioned; the scattering bound named in the rest-energy cell is not sourced in the cited files.

**The clock cell's way back, reworded 2026-11-19.** The first point above was closed with the column's author before the pull request: the cell's way back named a precondition (a charge of finite size, then a clock whose motion is a symmetry of that state, the same as the rest-energy row in the table above) and now names the frequency itself, worded from the author's suggestion: ħΩ/E = 1 on a converged state at the physical J, the inertia grid-converged and not carried by a few cells, unchanged between two spacings and two boxes. The icon did not change.

**The marks of the M5 column and its author.** Two of the adopted marks, charge quantization and Klein-Gordon, are lower than the marks the column's author had seen before the re-read. The rows are on the working branch and do not land on `main` before the author has seen them and had the chance to supply evidence, or to name a run, that would read them differently.

**Scope**: a re-read of what existing cells' cited evidence shows, under a bar written down first; it licenses icon and text changes in `MODELS.md` and does not license any new result, any change to the criteria set, or any statement about a model's physics.

## TASK REVIEW (2026-11-18)

**Task Duration:** 00:31 (from 2026-11-18 13:10 EST to 2026-11-18 13:41 EST)
**Usage Cap Triggered:** NO

**Results.**

- ✅ The bar, the restated `static` sentence and two restated row tests are in `MODELS.md` (§ 1, § 5).
- ✅ Twelve validated cells re-read, every cited file opened (§§ 3 and 4). M5: particle stability and antimatter stay validated; charge quantization, electron rest energy, the de Broglie clock and Coulomb read as partial; spin-½ statistics, EM waves and Klein-Gordon as not yet tested. M6: the dark matter candidate reads as partial; particle stability and EM waves as not yet tested.
- ✅ The score-board: M5 2 validated and 15 partial (17, was 20); M6 0 and 3 (3, was 5). The column order is unchanged.
- ✅ The audit of record supports nine of the twelve drafted marks and lowered three, all adopted (§ 6). Nineteen statements of the drafted cell texts were corrected.
- ✅ No script of either column was found wrong: each cell that moved cited an identity of a declared form, an anchor, or a result on a retired stack.

**Issues / blockers.** The first audit brief named cells to press on and its marks were set aside (unvalidated instrument: a brief that carries a suspicion returns it). A parenthesis was added to the bar's partial sentence after the cells were read, at the audit's finding that the fixed text was ambiguous (§ 5). Two marks of the M5 column are lower than the ones its author had seen.

**Deviations from plan.** Two audit passes ran, not one. The cited files were read by two agents on a smaller model and the tables were drafted from their file-by-file reports. The way back of each moved cell sits in the cell's pointer tail. Five other documents were brought in step with the cells (§ 5).

**Action needed.** The rows go to the author of the M5 column before any pull request; the pull request, with the author tagged, follows the author's answer. Each way back that needs a run gets a roadmap row on its own column, filed with the author.

**Scope.** A re-read of what twelve cells' cited evidence shows, under a bar written first. It licenses the icon and text changes in `MODELS.md` and does not license any new result, any change to the criteria set, or any statement about either model's physics.

**Findings.** Under one written bar M5 keeps two validated cells and M6 none. Ten cells moved because their evidence was a declared property, an anchor, a retired-stack result, or a static result on a row that asks for a run in time; each now says so and names what would move it back.

**Research docs created / updated.** [`t11_task_details.md`](t11_task_details.md) (this document) · [`MODELS.md`](../../MODELS.md) · [`platform_roadmap.md`](../platform_roadmap.md) · [`__M5_model_briefing.md`](../../openwave/xperiments/m5_liquid_crystal/__M5_model_briefing.md), [`m5_particle_hunt.md`](../../openwave/xperiments/m5_liquid_crystal/research/m5_particle_hunt.md), [`m5_question_tracker.md`](../../openwave/xperiments/m5_liquid_crystal/research/m5_question_tracker.md) · [`__M6_model_briefing.md`](../../openwave/xperiments/m6_ouroboros/__M6_model_briefing.md), [`m6_particle_hunt.md`](../../openwave/xperiments/m6_ouroboros/research/m6_particle_hunt.md).

**Landing (2026-11-19).** The author of the M5 column read the rows and answered on 2026-11-19; the work landed on `main` the same day by pull request [#741](https://github.com/openwave-labs/openwave/pull/741), with the author tagged on its first line, and the roadmap row moved to Done. The runs that would move the charge-quantization and Klein-Gordon cells were filed as rows of the [M5 roadmap](../../openwave/xperiments/m5_liquid_crystal/research/m5_roadmap.md) (M5.46, M5.47).
