# CROSS-MODEL TESTING: borrowing one model's field family into another model's framework

> **Standing rule (2026-07-24).** When a column tests a field family drawn from another
> column, the family's structure is taken as its author declared it, and nothing more.
> Two rules follow: a **uniform default** (§ 1, no silent internal structure) and an
> **attribution clause** (§ 3, an extended family is a different object). Origin: the
> M8.2 pre-registration surfaced the question at platform level
> ([discussion #312](https://github.com/openwave-labs/openwave/discussions/312)).

## Why this doc exists

The platform's columns are independent frameworks that happen to share a table. As soon
as one column's program tests candidates drawn from the others (the M8 field-dynamics
survey is the first case), a structural question appears that neither column's own docs
answer: **does the borrowed field carry the structure the testing framework needs?**

The concrete instance: the M8 arena carries three flat SU(2) connections on S³/2I (the
trivial one and a Galois pair). To be twisted by a flat connection at all, a field must
carry an internal representation the connection can act on. The M8.2 audit reported that
the candidate families do not obviously carry one:

| Family | What the pinned code establishes | Internal adjoint index |
| --- | --- | --- |
| M4 | a 3-component field seeded as a radial spatial displacement, evolved by a componentwise vector Laplacian | not established either way |
| M5 | a rank-two Lorentz-covariant tensor (a frame law) | no separate internal index |
| M7 | spatial one-forms with divergence-based charge | none |

Assuming an internal index to make the program run would have been an un-pre-registered
structural assumption, and it would have been invisible in the result. Hence the rules
below.

## 1. The uniform default

**A borrowed family is native and untwisted unless its author explicitly declares an
internal representation, or a soldering prescription is supplied and pre-registered.**

This is not a new principle. It is three existing ones applied to a new situation:

| Existing rule | Application here |
| --- | --- |
| No calibrated conventions: derive and pre-register, or record it as a fit with its search space | An assumed internal index is an unregistered structural assumption |
| Author-gated unknowns close only by an author answer ([`../AI_HYGIENE.md`](../AI_HYGIENE.md)) | "Is this field an internal triplet or a geometric displacement?" is a definition question, and definitions belong to whoever wrote the model |
| Honest negatives are results | "Not applicable" is a recorded status, not a failure |

The default is **uniform across families**. Per-family overrides are recorded as they
arrive; the default carries the burden of proof, not the exception.

## 2. What counts as a declaration

| Accepted | Not accepted |
| --- | --- |
| The family author states the field's representation in writing (discussion, issue, PR, or the model's own canonical doc) | Inference from component count |
| A soldering prescription, named and pre-registered with its search space (§ 3) | Inference from a variable's name, or from what would make the test work |
| The model's canonical spec already fixes the representation unambiguously | An AI agent's reading of the code, unconfirmed by the author |

Component count is the trap worth naming: a 3-component field may be an internal triplet
or three tangent-space components, and under a twist those are different physics.

## 3. The soldering clause

**A soldered or extended family is a different object. Any result obtained through the
soldering grades the extension, not the original family.**

If a family becomes testable in the borrowing framework only through an added structure
(an adjoint extension, a soldering of geometric indices to an internal bundle), then the
result is a result about "family + prescription P". It is named that way, recorded that
way, and scored that way. Otherwise the coverage matrix quietly starts crediting or
debiting a model for structure its author never wrote.

This is not a formality. Soldering is often genuinely available: on S³ the manifold is
parallelizable, so a geometric field can be identified with a bundle an internal group
acts on. But the identification requires a **choice of framing**, the choice is content,
and different choices are different physics. So a soldering prescription is
pre-registered like anything else: named, with its search space stated, and reported
under its own name.

| Consequence | Statement |
| --- | --- |
| Coverage matrix | A cell is never filled from a soldered run without the extension named in the cell |
| Falsification | A negative result under prescription P falsifies P applied to the family, not the family |
| Ownership | The prescription belongs to whoever wrote it, i.e. usually the borrowing column, not the family author |

## 4. "Not applicable" is a neutral status

Where no declaration and no prescription exist, the observable is recorded **not
applicable** for that family. On this platform that carries the same weight as a 🚧 cell:
it says the test does not apply, not that the family failed it. It is never counted as a
negative against the family in any summary count.

## 5. Author silence is a valid terminal state

If the family author does not answer, the default of § 1 stands, the row records "not
applicable", and the testing program proceeds and locks its pre-registration on schedule.

**No program blocks on an inbox.** Authors work at their own pace, several are academics
with conference seasons, and some columns are parked by their own roadmaps. A silence is
not a refusal and is not read as one.

## 6. Routing author-gated questions

The question travels **from the asking author to the family author directly**, not
through a maintainer relay.

| Reason | Statement |
| --- | --- |
| Relay drift | Every paraphrase hop is lossy while raising confidence. The platform's countermeasure is already explicit: "route the human-only questions back to the human, smaller" ([`../AI_HYGIENE.md`](../AI_HYGIENE.md), relay-drift detail) |
| Citable vs hearsay | An answer in the author's own words in a public thread is evidence the asking program can cite in its pre-registration. A relayed answer is not |
| It is the platform working | Model authors meeting each other is what the coverage matrix exists to produce |

**Mechanism:** open a **Q&A discussion**, one per family, mention the author, keep it
ownership-scoped, and state the pre-registered consequence so that a one-line answer
settles a registered branch.

Use the Q&A category rather than an issue. The question is not a defect in the family,
an open issue against another model's column reads as an outstanding problem in that
column, and Q&A's accepted-answer marking produces exactly the canonical citable answer
the pre-registration needs. If an answer then generates real platform work (implementing
a prescription, adding an extension), that work opens as an issue at that point.

Author contact handles are listed in each model briefing's Identity table, and so is
the `Co-author` row where a column carries one. A question inside a co-author's named
scope goes to the co-author; anything else goes to the author. Where that scope is the
whole column, the question goes to both handles. Where a column carries
two authors instead, the `Author contact` row names the scope of each.

## See also

| Doc | Why |
| --- | --- |
| [`../AI_HYGIENE.md`](../AI_HYGIENE.md) | author-gated unknowns, relay drift, the verification contract |
| [`METHOD_NOTE.md`](METHOD_NOTE.md) | the audit-page standard any cross-model result reports under |
| [`../MODELS.md`](../MODELS.md) | the coverage matrix these rules protect |
| [`../ONBOARDING_MODELS.md`](../ONBOARDING_MODELS.md) | the model-author entry path |
