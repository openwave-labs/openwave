# M10 platform pointers: where to read, for the author and the author's AI agents

> **Purpose.** A curated map of the OpenWave repository written to be CONSUMED BY AI
> AGENTS working for the M10 author (and by the author directly). Each entry gives
> a linked short name and what it teaches for the M10 program (deriving criteria
> rows on the carrier's own process: [`m10_background.md`](m10_background.md)).
> Every link was verified at scaffold time (2026-11-13).
>
> **If you are an AI agent:** load § 1 before generating any claim about this
> repository, then read only the sections the current task needs. Model output is a
> draft, never a result, until backed by a runnable script (the repo-wide contract).

## 1. Read first (the platform contract)

| Doc | Why |
| --- | --- |
| [`AI_HYGIENE.md`](../../../../AI_HYGIENE.md) | the working contract for AI-assisted research here: script-backed claims only, adversarial audit of substantive derivations, author-gated questions stay with the author |
| [`ONBOARDING_MODELS.md`](../../../../ONBOARDING_MODELS.md) | the onboarding standard M10 was admitted under: the four tiers (M10 is tier 2, so conservation is demonstrated, not asserted), bring your own compute, and § 3.6 headless-first |
| [`MODELS.md`](../../../../MODELS.md) | the shared coverage matrix and its status semantics. Every M10 cell opened not yet tested; a cell flips only via a runnable script plus a research note, decided at review |
| [`dev_docs/METHOD_NOTE.md`](../../../../dev_docs/METHOD_NOTE.md) | the reporting standard for any substantive result: equations first, equation-to-code map, adversarial audit recorded |
| [`dev_docs/PR_REVIEW_STANDARDS.md`](../../../../dev_docs/PR_REVIEW_STANDARDS.md) | what a review here verifies. § 7 is the evidence bar for a `MODELS.md` cell; Gate A covers large files and task ids; code is read for intent before anything is executed |
| [`dev_docs/ROADMAP_STANDARDS.md`](../../../../dev_docs/ROADMAP_STANDARDS.md) | the shape and word budgets of [`m10_roadmap.md`](m10_roadmap.md): a row is a preview, the task document is the record, and whoever creates a row allocates its id |
| [`REPRODUCE.md`](../../../../REPRODUCE.md) | the task-id glue convention: one id prefixes the task doc, its scripts, its data and its findings |
| [`CONTRIBUTING.md`](../../../../CONTRIBUTING.md) | setup, fork → branch → pull request flow, DCO sign-off, and how review works (one document, one merge) |

M10's own spec of record is [`m10_theory_canonical.md`](m10_theory_canonical.md);
the program is [`m10_roadmap.md`](m10_roadmap.md).

## 2. The scoring precedents M10's cells are read against

The admission reply answered how a row on a declared source is scored by pointing
at cells that already exist. Read them before proposing a cell move.

| Pointer | What it teaches |
| --- | --- |
| [`MODELS.md` § NSM (M9)](../../../../MODELS.md#nsm-m9) | installed physics is inherited: M9's Standard Model rows sit at not yet tested, with the cell saying the matter is installed and not emerged |
| [`MODELS.md` § MIT (M8)](../../../../MODELS.md#mit-m8) | importing field equations is not importing a solution: M8's gravity rows stay not yet tested while Einstein's equations are imported unchanged |
| [`MODELS.md` § EWT (M4)](../../../../MODELS.md#ewt-m4), the local metric row | partial with the caveat in the cell: profile and amplitude in-platform, the encodings taken from the manuscript. The route to validated is to derive the encodings in-platform ([issue #482](https://github.com/openwave-labs/openwave/issues/482)) |
| [`MODELS.md` § Liquid Crystal (M5)](../../../../MODELS.md#liquid-crystal-m5), the lepton row | the other column with Koide's relation in it: there it is imposed on a circulant fixed by the electron and muon, with the tau held out. M10's register offers a reason for the balance; comparing the two is a natural cross-model question |

## 3. The certification-column precedents (M8 and M9)

| Pointer | What it teaches |
| --- | --- |
| [`m8_1_task_details.md`](../../m8_mit/research/tasks/m8_1_task_details.md) + [`m8_1_method_note.md`](../../m8_mit/research/findings/m8_1_method_note.md) | the certification-gate template: pre-registered pass and fail criteria before numerics, a blind solver, an independent audit, the comparison done last. [M10.1](tasks/m10_1_task_details.md) follows this shape |
| [`m9_1_task_details.md`](../../m9_emergent_gravity/research/tasks/m9_1_task_details.md) | a certification that shipped a documented miss against the author's own paper as a scored result |
| [`m8_roadmap.md`](../../m8_mit/research/m8_roadmap.md) | a mature author-driven roadmap: terms filed before runs, the run record in the author's repository at pinned commits, one pull request per task |
| [`m8_theory_canonical.md`](../../m8_mit/research/m8_theory_canonical.md) | the other column built on icosahedral structure (the binary icosahedral quotient of the three-sphere). M10's carrier has the icosahedral group as its port symmetry; where the two meet is theirs to work out |
| [`dev_docs/CLEAN_ROOM_STANDARDS.md`](../../../../dev_docs/CLEAN_ROOM_STANDARDS.md) | how a blind run is built and audited here: what enters the packet, what the room cannot see, commit before unseal |
| [`dev_docs/CROSS_MODEL_TESTING.md`](../../../../dev_docs/CROSS_MODEL_TESTING.md) | the rules if M10 work ever borrows another column's structure, and the routing of author-gated questions |

## 4. What M10 consumes from shared code (deliberately little)

| Item | Status |
| --- | --- |
| `openwave/common/` engines and the GGUI stack | not consumed. M10's instruments are exact algebra and finite Hamiltonians; headless-first resolves to headless-only until a cell exists ([`m10_roadmap.md`](m10_roadmap.md) conventions) |
| `numpy`, `scipy`, `sympy`, Taichi | already platform dependencies (`pyproject.toml`); research scripts stay dependency-light and self-contained, one script one claim |
| Shared docs layer | fully consumed: the § 1 contract, the roadmap checker (`dev_docs/utils/check_roadmaps.py`), the `MODELS.md` linter (`dev_docs/utils/check_models_md.py`) and the review standards are the interfaces this column plugs into |

## 5. House standards for M10 tasks (so contributions merge smoothly)

| Standard | Rule |
| --- | --- |
| Naming | scripts, data and findings under `research/` with `m10_<id>_` prefixes, where `<id>` is a registered row on [`m10_roadmap.md`](m10_roadmap.md) |
| Task ownership | the author creates and orders roadmap tasks; registration is a roadmap row plus a `tasks/m10_<id>_task_details.md`, in the same pull request as the artifacts |
| PR shape | one concern, one pull request, from the author's fork, draft until quiescent; review and merging follow maintainer schedule |
| What a PR brings | the method note and a small script that regenerates the headline number. The run record stays in the author's repository at a pinned commit, and nothing is imported in bulk |
| Compute | the author's own. Maintainers read code for intent before executing any of it, and run contributor code only when a review calls for it |
| Status | `MODELS.md` cells are decided by the maintainers at review. A pull request states its result and may propose a cell; it does not edit another column's cells |
| Terms before runs | gates, inputs and what fails them, written in the task document before the run; every number reported, none tuned toward a target |
| Audit | substantive claims get an independent adversarial pass (own script, own method) before they are trusted; record the audit in the deliverable |
| Sources | papers stay in the author's repository; [`../theory/_CITATIONS.md`](../theory/_CITATIONS.md) is the tracked registry; never fabricate an identifier |
| Style | markdown tables for structured content; status icons only ✅ ⚠️ ❌ 🔶 🚧; escape literal pipes as `\|` inside table cells; relative links; English throughout; no machine-local paths |
