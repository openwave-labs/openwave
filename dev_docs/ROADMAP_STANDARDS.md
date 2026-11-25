# ROADMAP STANDARDS

> How every roadmap in this repository is structured and how much it is allowed to say. Binds the [platform roadmap](platform_roadmap.md) and every model roadmap (`openwave/xperiments/<model>/research/m<n>_roadmap.md`). Enforced by [`utils/check_roadmaps.py`](utils/check_roadmaps.py), which a maintainer runs before merging any roadmap edit ([`PR_REVIEW_STANDARDS.md`](PR_REVIEW_STANDARDS.md)). There is no CI: the checker runs when someone runs it.

## 1. What a roadmap is, and what it is not

| A roadmap IS | A roadmap is NOT |
| --- | --- |
| The index of work: what is running, what is queued, what closed | The record of results |
| One row per task, each row a **preview** that gets a reader to the right document | A findings note, a task review, or a lab log |
| The place to see run order and gates at a glance | The place to read what was measured |

Every row links its task document, and **that document is the record**. If a fact matters and does not fit the row, it belongs in the task document, not in a longer row. A roadmap row that has to be read slowly has already failed.

## 2. Required structure

Sections in this order. Bracketed ones are optional; nothing else is a top-level section.

| # | Section | Contents |
| --- | --- | --- |
| 1 | `# <NAME> ROADMAP` + intro blockquote | one blockquote: what this roadmap covers, links to its own sections |
| 2 | `## IN PROGRESS` | the single running task; **header row only when nothing is running**, never a `(none)` placeholder and never status prose parked in one (that belongs in the backlog row, the status snapshot, or the change-log) |
| 3 | `## BACKLOG` | queued tasks, **row order = run sequence** |
| 4 | [`## STATUS AT A GLANCE`] | dated snapshot, question-and-answer table |
| 5 | [`## CONVENTIONS`] | ID scheme, local reading rules, anything a newcomer needs before the tables make sense |
| 6 | `## DONE` | closed tasks, appended in order of completion |
| 7 | [`## CHANGE-LOG`] | dated prose entries: reorganizations, decisions, routing |
| 8 | [`## ARCHIVE` / `## LEGACY DONE` / `## PRE-REGISTERED ...`] | frozen content, see § 7 |

**Table shape.** Live tables (IN PROGRESS, BACKLOG, phase tables) carry `| TaskID | Title | Description | Gated By |`; the fourth column may be renamed to the local gate vocabulary (`Gate`, `Validation gate`, `Owner` may be inserted before it). DONE tables carry `| TaskID | Title | Description | Completed |`. The column named `Description` is the one the budget below applies to, and it must be named exactly that so the checker can find it.

**Status icon.** Every task row opens its `TaskID` cell with one status icon, a space, then the ID: `🚧 [M5.47](tasks/m5_47_task_details.md)`. The icon says where the row stands, and it is said once, there:

| Icon | Status | Where the row sits |
| --- | --- | --- |
| 🔶 | in progress, worked on today | `IN PROGRESS` |
| 🔶🔶 | in progress over several days, waiting for a later day (nights of data, a person home) | `IN PROGRESS` |
| 🚧 | backlog | `BACKLOG`, `LATER`, and any phase table that follows them |
| ✅ | done | `DONE`, and any phase table that follows it |

The icon follows the section, so moving a row between tables changes its icon in the same edit, and a parked or deferred task that sits in `BACKLOG` carries 🚧. The icon is not part of the ID: links, file names, commit messages and prose keep the bare ID. The `Description` never opens with one of these three icons. A closed row whose outcome is negative or carries a caveat may still open its `Description` with ❌ or ⚠️: that mark is the verdict, the icon beside the ID is the status. A `(none)` placeholder row carries no icon.

**Blockquote placement.** One blockquote directly under the heading it explains, before the table. Never between the header row and the data rows, never after a table.

**Escaped pipes.** A literal `|` inside a cell is written `\|`. An unescaped one silently shifts every later column.

## 3. Word budgets

The hard rule, and the reason this document exists.

| Element | Cap | Grounding |
| --- | --- | --- |
| **`Description` cell** | **65 words** | half the length of the row that triggered the rule; M7's closed rows already run at a 21-word median, so the budget is proven livable rather than aspirational |
| Row `Title` cell | 15 words | the 90th percentile across all roadmaps is 13 |
| Every other cell in the row (`Gated By`, `Gate`, `Owner`, `Completed`) | 35 words | these are pointers, not records; the median is 1 word. Without this cap the description budget is avoidable by moving the text one column right, which is what one roadmap had already done |
| Section blockquote | 50 words | the median is 33 |
| Intro blockquote (under the H1) | 80 words | it carries navigation, so it gets more room than a section note |
| Change-log entry (one paragraph) | 200 words | the narrative home, and still not a findings note |

**What the blockquote budgets do and do not cover.** They apply to a **section note**: the `>` block directly under a heading, above the table it introduces. They do not apply to body prose (a reading guide, a standing rule, a source-document table), which is reference material rather than a preview and carries no budget. So a standing rule belongs in body prose or in `CONVENTIONS`, never in a section note, and moving one there is a fix rather than an evasion.

**How words are counted** (identical in the checker): markdown link *labels* count, link targets do not; `<br>` counts as a space; backticks, asterisks, underscores and hashes are stripped; everything else splits on whitespace. So `[M5.22](tasks/m5_22_task_details.md)` is one word, and formatting is never a way to buy room.

## 4. What belongs in a Description cell

| Include | Leave to the task document |
| --- | --- |
| The state in a word (`STAGED`, `Closed`) and, for a closed row, the close date; the status icon sits beside the ID (§ 2) | run logs, durations, cap events |
| One sentence on what the task does or found | the method, the ladder of sub-results, the deferred items |
| The single decisive number or verdict, if there is one | every other number |
| What it unblocks or is blocked by, when not already in `Gated By` | the blindspots table, the audit record, the cross-links |

Anti-patterns, all of them observed in this repository before the rule existed: a row that reproduces its own title; a row that enumerates every sub-result of a closed run; a row that narrates how the task came to exist (that is a change-log entry); a row that carries measured values a reader cannot check without opening the task document anyway.

## 5. DONE rows

Appended at the **end** of the table, so it reads in order of completion. A closed row states the verdict, not the evidence for it: `Closed <date>. <one-sentence verdict, with the decisive number if there is one>. Full record: [note] + [task doc].`, in a row whose `TaskID` cell opens with ✅ (§ 2). The temptation to keep the full close-out summary in the row is what produced the 288-word rows this standard replaces; the summary already exists, in the task document's `TASK REVIEW` section.

## 6. Task IDs

| Scheme | Used by |
| --- | --- |
| `M<model>.<n>[.<m>]` | model roadmaps, e.g. `M5.21.11` |
| `T<n>` | the platform roadmap |

IDs are assigned in creation order and **never reused**, including after a renumber or a migration between roadmaps. A migrated task carries a `(was <old ID>)` note in its row and a change-log entry; records written before a renumber keep the old ID and are not rewritten.

**Whoever creates the row allocates its ID**, in the same pull request. On a model roadmap that is the column's author or co-author, who defines the task and therefore numbers it; there is no request step and no maintainer assignment, because an ID nobody can pick without asking turns every new task into a round trip. The reviewer verifies instead of assigning: the ID is the next free number in creation order, it is not a retired or reused one, and the scheme matches the table above. A collision with a row that landed first is renumbered by whoever merges second, as a [maintainer edit](PR_REVIEW_STANDARDS.md#10-maintainer-edits). The merge is the approval.

**Proposing is not allocating.** A design document may list the IDs it expects its future tasks to take. They stay proposals until the pull request that creates the rows, and the collision check runs then, because other rows can land in between.

## 7. Frozen sections

Three section kinds are **exempt from every budget above**, are not rewritten to comply, and are skipped by the checker. Editing any of them to meet a current standard would destroy the only thing it is for.

| Section | Why it is frozen |
| --- | --- |
| `ARCHIVE` | preserves how the work read at the time |
| `LEGACY DONE` | the same, for a superseded record format |
| `PRE-REGISTERED ...` | a protocol, outcome matrix or verdict definition **filed on a date, before the run it governs**. Its evidential value is that it was written down in advance and has not been edited since. Amendments are appended and dated, never applied in place, and a word budget cannot be allowed to force a silent rewrite |

A `PRE-REGISTERED` section must carry its filing date in the heading, so a reader can see what was fixed when.

## 8. Enforcement

```bash
python3 dev_docs/utils/check_roadmaps.py          # all roadmaps
python3 dev_docs/utils/check_roadmaps.py <path>   # one file
```

Exit 0 clean, 1 with a line-numbered list of violations. Run it after any roadmap edit and before merging one. It checks the six budgets, the required `Description` column, each row's cell count against its header (the unescaped-pipe bug), and the status icon of every task row against the section it sits in (§ 2). It does not check whether a row is *accurate*; that is what review is for.
