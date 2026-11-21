# M10 / OPH (OBSERVER PATCH HOLOGRAPHY), ROADMAP

> Column roadmap. Index of work only; results live in the task documents. Live:
> [IN PROGRESS](#in-progress). Queued: [BACKLOG](#backlog). Closed: [DONE](#done).
> Reading rules: [§ CONVENTIONS](#conventions). The author creates and orders the
> tasks here.

## IN PROGRESS

| TaskID | Title | Description | Gated By |
| --- | --- | --- | --- |

## BACKLOG

> Every row here is the author's to create, in the pull request that brings the
> task.

| TaskID | Title | Description | Gated By |
| --- | --- | --- | --- |

## STATUS AT A GLANCE (2026-11-21)

| Question | Answer |
| --- | --- |
| Where is M10? | Registered column, admitted 2026-11-13 ([discussion #712](https://github.com/openwave-labs/openwave/discussions/712)). Scored on the Interval Echo pin since 2026-11-21 ([M10.3](tasks/m10_3_task_details.md)); particle stability partial, every other `MODELS.md` cell not yet tested |
| What kind of column? | Tier 2: a local repair and memory law on a screen of twelve-port carriers, written as update rules with no action |
| What decides the next cell? | In the words of [`m10_theory_canonical.md`](m10_theory_canonical.md) § 8: "Its observable computed on the carrier's own process in-platform. With something still supplied, that is a case for ⚠️ with the caveat in the cell. With the source derived from the carrier, it meets the bar for ✅ set in the #482 answer." Row 4 is the first, at partial, its coupling supplied ([review](findings/m10_3_maintainer_review.md) § 5) |
| Who runs what? | M10.1, the certification gate, was the one maintainer-run task and is closed. The author runs and brings every task from here, on the author's own compute. Maintainers review, at maintainer pace |

## CONVENTIONS

> Standing reading rules. Dated events live in [§ CHANGE-LOG](#change-log).

**Spec of record.** [`m10_theory_canonical.md`](m10_theory_canonical.md) wins when
documents disagree. The column scores one pinned candidate at a time.

**Task IDs.** `M10.<n>`, allocated by whoever creates the row, in the same pull
request ([`ROADMAP_STANDARDS.md`](../../../../dev_docs/ROADMAP_STANDARDS.md) § 6).
Every script, data file and finding carries the id of a registered row.

**Mode of work.** Headless first. No launcher and no GUI until a cell exists.

**Standing rules.** Terms before runs: a result that will claim a cell files its
gates and inputs first. The run record stays in the author's repository at a
pinned commit; the pull request brings the method note
([`METHOD_NOTE.md`](../../../../dev_docs/METHOD_NOTE.md)) and a small script that
regenerates the headline number. Adversarial audit on substantive claims
([`AI_HYGIENE.md`](../../../../AI_HYGIENE.md) § 1). One pull request per concern.

## DONE

| TaskID | Title | Description | Completed |
| --- | --- | --- | --- |
| ✅ [M10.1](tasks/m10_1_task_details.md) | Certification gate: twelve-direction dispersion | Closed 2026-11-13. The author's frozen expansion coefficients are reproduced exactly in clean rooms, seven of seven values, the comparison made after the rooms' result was pushed. Arithmetic on a declared object; no cell moves. Full record: [note](findings/m10_1_method_note.md) + [task](tasks/m10_1_task_details.md) | 2026-11-13 |
| ✅ [M10.2](tasks/m10_2_task_details.md) | Signed-source static tests | Closed 2026-11-15. The author's static results on the signed-source candidate reproduce by an independent in-platform recomputation: the graph, the compact energies, the moments and the stopping times. The charge, rest-energy and spin rows stay not yet tested, since each property comes from what the candidate declares; the record says what would move each. Full record: [review](findings/m10_2_maintainer_review.md) + [task](tasks/m10_2_task_details.md) | 2026-11-15 |
| ✅ [M10.3](tasks/m10_3_task_details.md) | Free particle persistence | Closed 2026-11-21. The author's free-pair numbers on Interval Echo reproduce to nine digits by an independent in-platform recomputation. The pin was adopted and the particle-stability row moved to partial: the pair's persistence comes out of the generator the candidate's spec states, with the coupling a supplied choice. Full record: [maintainer review](findings/m10_3_maintainer_review.md) + [task](tasks/m10_3_task_details.md) | 2026-11-21 |

## CHANGE-LOG

2026-11-13. Column admitted and scaffolded from the application
([discussion #712](https://github.com/openwave-labs/openwave/discussions/712)).
Model ID M10. The admission reply answers the application's two questions:
update rules with no action are tier 2 and admissible, and a row on a declared
source is inherited until its observable is computed on the carrier's own
process. M10.1 registered as the certification gate, from the author's own offer.
The briefing's hero image is a placeholder for the author to fill.

2026-11-13. M10.1 started on the maintainers' go: the packet for the two clean
rooms committed before either opens.

2026-11-13. M10.1 closed. Three clean rooms, two routes, one exact result; the
author's seven frozen values for the twelve-direction dispersion agree with it
exactly. Two audit rooms stopped early for a cause in the packet's layout, logged
with the fix. The backlog is empty: the next row is the author's to create.

2026-11-15. M10.2 closed. The author brought a new pinned candidate, a signed unit
pair on a federation of twenty cells, and asked for a decision on three static rows.
Every number the maintainers recomputed agrees with the author's (the graph, the
compact energies, the moments, the stopping times), by a script that reads one data
file of the author's and runs none of the author's code. The package's own counts
were not recomputed. The three rows stay not yet tested:
the integer loads, the single anchor and the cell's symmetry are declared, and the
run returns what they fix. The review names what would move each row. The backlog
is empty: the next row is the author's to create.

2026-11-21. M10.3 closed, with pull request
[#745](https://github.com/openwave-labs/openwave/pull/745) merged. Interval Echo at
`4fe10f9a` is adopted as the column's pin, and the signed-source candidate of M10.2
is kept as ancestry. Row 4, particle stability, moved to partial: the pair's
persistence comes out of the generator the candidate's spec states, and the
coupling is a supplied choice that answers to this row. What would move it is
recorded in the cell. The maintainers' recompute, a script that reads one data file
of the author's and runs none of the author's code, reproduces the author's numbers
to nine digits and is the check of record. The backlog is empty: the next row is
the author's to create.
