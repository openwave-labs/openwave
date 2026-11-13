# M10 / OPH (OBSERVER PATCH HOLOGRAPHY), ROADMAP

> Column roadmap. Index of work only; results live in the task documents. Live:
> [IN PROGRESS](#in-progress). Queued: [BACKLOG](#backlog). Closed: [DONE](#done).
> Reading rules: [§ CONVENTIONS](#conventions). The author creates and orders the
> tasks here.

## IN PROGRESS

| TaskID | Title | Description | Gated By |
| --- | --- | --- | --- |
| [M10.1](tasks/m10_1_task_details.md) | Certification gate: twelve-direction dispersion | 🔶 Running since 2026-11-13. Blind recompute of the scalar dispersion on the twelve icosahedral directions, in two clean rooms, with the comparison done last. Certifies arithmetic on a declared object and moves no cell. | the rooms' returns, then the adjudication |

## BACKLOG

> Every row here is the author's to create, in the pull request that brings the
> task.

| TaskID | Title | Description | Gated By |
| --- | --- | --- | --- |

## STATUS AT A GLANCE (2026-11-13)

| Question | Answer |
| --- | --- |
| Where is M10? | Registered column, admitted 2026-11-13 ([discussion #712](https://github.com/openwave-labs/openwave/discussions/712)). Every `MODELS.md` cell not yet tested |
| What kind of column? | Tier 2: a local repair and memory law on a screen of twelve-port carriers, written as update rules with no action |
| What decides the first cell? | One criteria row computed on the carrier's own process in-platform. Rows on declared sources are inherited ([`m10_theory_canonical.md`](m10_theory_canonical.md) § 8) |
| Who runs what? | The author runs and brings every task after M10.1, on the author's own compute. Maintainers review, at maintainer pace |

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
