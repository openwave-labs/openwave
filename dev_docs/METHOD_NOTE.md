# METHOD NOTE: the audit page standard

> **Standing rule (2026-07-03, mandatory; renamed from "method report" 2026-07-04).** Every report, summary doc, or email addressed to a model's theory owner (M5: Dr. Jarek Duda; M7: Marc Fleury; any external physicist or advisor) carries a method note built to this standard. The bar: the reader must be able to **audit the result by reading**, without trusting us and without reverse-engineering Python. A result whose method cannot be found is not a result to that audience.
>
> Why a NOTE, not a report: the method note is a **component, not the report itself**. Sometimes it ships attached to a results report alongside findings; sometimes it stands alone as the internal audit page (the maintainer re-verifying what an instrument computes, months later, is the same reader as the theory owner). Either way it is the page where the equations and their code locations live.

## The motivating incident (2026-07-03, M5.16)

The M5.16 calibration report led with results and buried the physics: the energy functional and potential lived inside a ~1000-line script's docstring, and the report's inspection list pointed at a driver file containing no physics. The theory owner read the code and replied, verbatim: *"this median energy radius requires choice of potential (to be found) - I don't see anywhere there, and this minimized energy is integral of Hamiltonian - I also don't see ... so honestly still I have no idea what does it calculate."* None of the five questions attached to the report were answered. The failure was **legibility of the verification surface, not correctness**: his same reply restated a prescription that largely matched what the task had implemented. Full record: [`m5_4h_convo_2026.07.03.md`](../openwave/xperiments/m5_liquid_crystal/research/tasks/m5_4h_convo_2026.07.03.md).

The lesson generalizes: physicists verify by reading the Lagrangian / Hamiltonian / potential in mathematical notation and checking that the code implements it term by term. A rendered simulation, a results table, or a well-written prose summary is not a verification surface for this audience.

## The required shape (in this order)

| # | Element | Requirement |
| --- | --- | --- |
| 1 | **Equations first** | The report OPENS with the physics in math notation: the Lagrangian / Hamiltonian (or free-energy) density, the potential with its explicit form and coefficients, the ansatz / boundary conditions, the discretization, the minimization or evolution scheme, and the DEFINITION of every reported observable. All before any result. Written as typeset math, in the format of [§ Equation format](#equation-format). |
| 2 | **Equation-to-code map** | A table mapping every term of the functional (and every observable) to `function name` + `script.py:line` + a clickable GitHub **permalink** (commit-pinned `https://github.com/.../blob/<commit>/...#L<n>-L<m>`, so line anchors never drift). Test: any term findable in under a minute by a human clicking links. |
| 3 | **Small auditable physics module** | The energy functional / Hamiltonian lives in a single-purpose module (target ~200 lines, not a monolith), whose docstring carries the same equations as the report. Drivers, sweeps, and plotting import it; they never re-implement physics. |
| 4 | **Results after methods** | Each result states its pre-registered gate and convergence evidence next to the number. |
| 5 | **Minimal inspection set** | At most ~4 artifacts, ordered physics-first: the functional module FIRST, the driver LAST. Never send a reader into a driver to find physics. |
| 6 | **Scope honesty** | State explicitly what was NOT computed (observables, regimes, anchors), in the owner's own vocabulary for the model. |
| 7 | **The scope line** (T9, 2026-10-27) | One line, `**Scope**:`, in every results post, task review and method note, stating what the result licenses and what it does not: the object it was computed on, the box and spacing or the sample, and the claims it does not support. Shape and two filled examples in [§ The scope line](#the-scope-line). Checked at PR review ([`PR_REVIEW_STANDARDS.md` C9](PR_REVIEW_STANDARDS.md#5-gate-c-claim-to-artifact)). |

## Pre-send checklist

| Check | Pass condition |
| --- | --- |
| Can the owner find `V(...)` and the Hamiltonian density from the report in one click? | yes, via the equation-to-code map |
| Does the report open with equations, not results? | yes |
| Are the equations typeset math that renders on GitHub, with none left inside a code block? | yes ([§ Equation format](#equation-format)) |
| Is every code link clickable (absolute https) with a file:line anchor? | yes. `blob/main` links are the DEFAULT for task-scoped files (frozen after task close, anchors stable; The maintainer 2026-07-05); commit-pin ONLY when linking into live/evolving files whose lines will drift |
| Is the functional in a small module whose docstring matches the report's equations? | yes |
| Is each headline number next to its gate + convergence evidence? | yes |
| Is the not-computed list explicit? | yes |
| Is the scope line present, one sentence, naming the exclusions and not repeating the title? | yes (T9) |
| For a reproduction or verification run: is its receipt written beside the data? | yes (T10, [§ The validation receipt](#the-validation-receipt)) |
| REQUIRED (upgraded from RECOMMENDED 2026-07-05): has an independent second agent ADVERSARIALLY audited the note (own script, different method/seed, tried to REFUTE each claim, re-walked the equation-to-code map)? | yes, before sending; adopted from the model owner's own advice ("careful small steps, maybe multiple agents verifying each other", Duda 2026-07-03). Upgraded after the audit's first real catch (M5.18: a witness overclaim refuted + a missed vacuum-branch structure found, `m5_18_verification_note.md § 10`). Record the audit outcome IN the note (the § 10 pattern): the audit record is part of the rigor surface |

## Scope

Applies to every model in `openwave/xperiments/` when reporting outward to its theory owner or any external physicist. It does not constrain internal working notes; it constrains anything that crosses the repo boundary toward a scientist.

## Equation format

Adopted 2026-11-18. Every equation in a method note, a task document, a pull request, an issue or a discussion post is written as **LaTeX math in Markdown**, which GitHub typesets in the browser. The reference is the M10.2 method note, the first note in the repository written this way: [`m10_2_method_note.md`](../openwave/xperiments/m10_oph/research/findings/m10_2_method_note.md). An equation typed as plain text inside a code block is the form this replaces: a reader checks a typeset formula against a paper at a glance, and a row of Unicode symbols in a fixed-width box only with effort.

| Form | How it is written | Use it for |
| --- | --- | --- |
| Display | a line holding only `$$`, the LaTeX on the lines after it, a line holding only `$$`, with a blank line before and after the block | every equation a reader will check: the functional, the potential, an observable's definition, a stencil |
| Inline | `$ ... $` inside a sentence | a symbol or a short expression in running text, such as $E_J = U + J^2/2I$ |
| Fenced | a code fence whose language is `math` | a display equation that Markdown would otherwise damage (see the last rows of the next table) |

A worked example, the static energy of the M5 functional. The source:

```markdown
$$
U = h^3 \sum_n \Big[ L_1 \sum_i \lVert \partial_i M \rVert^2
  + \kappa \sum_{i \lt j} \lVert [\partial_i M, \partial_j M] \rVert^2
  + v \sum_k \big( \operatorname{tr} M^k - c_k \big)^2 \Big],
\qquad c_k = 1 + \delta^k
$$
```

and what GitHub shows for it:

$$
U = h^3 \sum_n \Big[ L_1 \sum_i \lVert \partial_i M \rVert^2
  + \kappa \sum_{i \lt j} \lVert [\partial_i M, \partial_j M] \rVert^2
  + v \sum_k \big( \operatorname{tr} M^k - c_k \big)^2 \Big],
\qquad c_k = 1 + \delta^k
$$

What it needs, and where it renders:

| Question | Answer |
| --- | --- |
| Is this LaTeX? | Yes: LaTeX math syntax, typeset by MathJax, which GitHub runs in the reader's browser |
| Does the writer or the reader install anything? | No. Nothing is installed in the repository's environment or on the reader's computer; a browser on github.com is enough |
| Where does it render? | In every Markdown surface of GitHub: files in a repository, pull requests, issues, discussions and their comments. Editors with a Markdown preview that supports math show it too |
| Where does it not render? | In a terminal, in a plain-text editor, and in the body of an e-mail: there the LaTeX source is shown as typed, which stays readable. Inside a code block of any language other than `math` it is never typeset |
| And in an e-mail to a theory owner? | Mail clients typeset no LaTeX. The equations go in the mail as plain Unicode text, one per line, and the mail links the note on GitHub where they are typeset |
| And in a docstring? | A docstring keeps its equations as plain text, readable in an editor. The note carries the same equations typeset (requirement 3 of the table above compares the two) |

Rules that keep an equation from breaking:

| Rule | Detail |
| --- | --- |
| No math in a code block | A block of language `text`, `python` or none shows the source, not the formula. Pseudocode and stencils made of words stay in code blocks; equations leave them |
| Norms and absolute values without the bar character | `\lVert x \rVert` and `\lvert x \rvert`, never the keyboard bar: inside a table cell the bar ends the cell, and at the start of a line it reads as a table row |
| Comparisons | `\lt` and `\gt`, or the signs with a space on each side. A `<` followed at once by a letter can be read as the start of an HTML tag |
| One equation per display block | Several statements on one line are separated by `\qquad`, as in the reference note. Alignment environments are used only inside a `math` fence |
| No backslash before punctuation | Markdown reads a backslash before a punctuation mark as an escape and removes it, inside `$ ... $` and inside `$$` blocks alike: `\{0,1\}` reaches the typesetter as `{0,1}` and the braces vanish, and the thin space `\,` becomes a comma. Write `\lbrace` and `\rbrace`, `\operatorname{tr}` or `\quad` in place of a `\,`, `\vert` for a bar; a backslash before a letter (`\sum`, `\qquad`) is untouched |
| When Markdown still damages the source | An underscore or an asterisk pair read as emphasis, or a `\\` line break that loses a backslash: move the equation into a `math` fence, where the source is taken as typed |
| Text inside math | `\mathrm{...}` or `\operatorname{...}` for a name with more than one letter (`\operatorname{tr}`, `\mathrm{invalid}`), so it is not set as a product of variables |
| Look at it rendered | Before a note is sent, its equations are read on the rendered page, since a formula that fails to parse is shown as raw source with no error |

## The scope line

Adopted 2026-10-27 ([T9](tasks/t9_task_details.md)) from the claim registry of the repository that hosts the M5 discussion, where every accepted claim carries a `does_not_license` field fixed before the run. The M5.32 rungs had been stating the same thing in prose ("what the poll does not do", "not in R27"); the line makes it a fixed slot a reader can find. One sentence, at most two; it names the exclusions, never repeats the title, and it is not a status label (the ✅ / 🔶 / ⚠️ labels stay as they are).

| Where | The line |
| --- | --- |
| A reproduction verdict (M5.35) | **Scope**: a rerun of the author's scripts as shipped on pinned versions; it licenses "the package reproduces on this stack" and the path-dependence finding on the ℓ = 0 refit, and does not license any statement about the record field, any cell icon, or the physical correctness of the author's stacks. |
| A lattice gap (the R29 plan's reads) | **Scope**: the lowest kinetic-positive ω² of the projected Hessian about one relaxed hedgehog at the toy point, extrapolated in 1/L over four rungs at one spacing with one mesh check; it licenses one of the four pre-registered words and does not license a continuum value, a physical-unit gap, or a statement about the author's vacuum. |

## The validation receipt

Adopted 2026-10-27 ([T10](tasks/t10_task_details.md)) from the per-run `validation-receipt.yaml` of the same repository, reduced to what a reproduction needs. Every reproduction or verification run writes one small tracked JSON beside its record at FINISH, `research/data/<id>_receipt.json`, with [`dev_docs/utils/write_receipt.py`](utils/write_receipt.py) from the run's logs and comparator output. Two receipts diff without reading two task docs; a receipt is a record of what ran, never evidence of correctness (the verdict words stay in the task record with their gates).

| Field | Content |
| --- | --- |
| `task`, `written` | the task id; the UTC stamp of the receipt |
| `base` | this repo's commit the run started from |
| `package` | the commit, DOI or URL of what was reproduced, as pinned in the record |
| `machine` | the class only (`workstation`, `laptop`, `container`); never a hostname, a path or a core count |
| `versions` | the interpreter and the pinned libraries, read from the running interpreter |
| `jobs` | per job: the command, the exit code, the wall seconds, the log file name |
| `comparator` | per regenerated file: leaves compared, max relative difference, the leaf |
| `verdict` | the pre-registered words of the task record, verbatim |
