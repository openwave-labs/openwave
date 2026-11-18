# T11: One written bar: every validated cell re-read

> Platform task **T11**. Status: 🚧 **STAGED 2026-11-18** (ruled 2026-11-15, the day of the trigger) · Roadmap: [`../platform_roadmap.md`](../platform_roadmap.md) · Deliverable lands in [`MODELS.md`](../../MODELS.md) (the bar, the `static` sentence, the restated row tests, the re-read cells, the score-board) and in this record.

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

**The authors.** A reading of a past result is author-gated ([`PR_REVIEW_STANDARDS.md` § 8.1](../PR_REVIEW_STANDARDS.md#81-when-the-contributor-is-not-an-author-of-the-column)); the icon is the platform's, on the evidence bar of [Gate E](../PR_REVIEW_STANDARDS.md#7-gate-e-modelsmd-cell-changes). No cell moves on `main` before its column's author has seen the row as it would read. The work is committed to the working branch first; the pull request that lands it tags each affected author on its first line and states, in order: what triggered the re-read, why the cell reads differently under the bar, and what evidence would move it back. An author's reply that supplies evidence the re-read missed is exactly what the tag asks for.

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

## FINDINGS

Empty until the go.

**Scope**: a re-read of what existing cells' cited evidence shows, under a bar written down first; it licenses icon and text changes in `MODELS.md` and does not license any new result, any change to the criteria set, or any statement about a model's physics.
