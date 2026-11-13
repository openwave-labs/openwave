# M10 Agent Orientation: the agent front door

> **If you are an AI agent and were told "read the m10_agent_orientation.md": this
> page is your bootstrap.** Load every document in § 1 into context, in order. Then
> follow the completion protocol in § 4: confirm you are oriented, summarize what
> you read, and declare yourself ready. From that point the author can ask you
> anything about the M10 column, plan next moves with you, or start work with a
> simple **"go task m10.2"** (or any roadmap task).
>
> Humans are welcome to read this page too; the model's human front door is
> [`__M10_model_briefing.md`](../__M10_model_briefing.md).

## 1. The orientation reading list (load ALL, in this order)

| # | Doc | What it gives you |
| --- | --- | --- |
| 1 | [`__M10_model_briefing.md`](../__M10_model_briefing.md) | the column at a glance: identity, model profile, status, help wanted |
| 2 | [`research/m10_theory_canonical.md`](m10_theory_canonical.md) | THE SPEC OF RECORD (canonical when docs disagree): the pin, the arena, the dynamics, the author's five obligations, the row map, and § 8, the rules this column is scored by |
| 3 | [`research/m10_background.md`](m10_background.md) | the gap map (what OPH has, what it lacks), the evidence-weight grading, the onboarding evaluation of record, and the admission terms |
| 4 | [`research/m10_roadmap.md`](m10_roadmap.md) | the program: tasks, gates, current status. The author creates and orders these tasks |
| 5 | [`research/m10_platform_pointers.md`](m10_platform_pointers.md) | the cross-model reading map and the house standards; ALSO load its § 1 platform-contract docs, [`AI_HYGIENE.md`](../../../../AI_HYGIENE.md) above all |
| 6 | [`research/tasks/m10_1_task_details.md`](tasks/m10_1_task_details.md) | the certification gate's plan: pre-registered checks, a blind protocol, the comparison done last. ⚠️ If you work for the author you already know the frozen values it withholds; never write them into this repository before that task has run |
| 7 | [`theory/_CITATIONS.md`](../theory/_CITATIONS.md) | the source registry: which record backs which claim; papers live in the author's repository, never here; the never-fabricate identifier policy |

## 2. How tasks run here

| Phase | What to do |
| --- | --- |
| PLAN | a roadmap row and a task document first: scope, definition of done, and **the gates, inputs and what fails them, written BEFORE any numerics**. Say which pieces of the construction are supplied (a field, an action, a radial direction, a clock, a coupling) and which run on the carrier's own process |
| EXECUTE | on the author's compute. Scripts, data and findings under `research/` with the task's `m10_<id>_` prefix; every claim backed by a runnable script; the heavy run record stays in the author's repository at a pinned commit |
| FINISH | a method note per [`dev_docs/METHOD_NOTE.md`](../../../../dev_docs/METHOD_NOTE.md): equations first, equation-to-code map, the audit record. A negative is a result and is reported the same way |
| REVIEW | one pull request, from the author's fork, DCO signed. The maintainers review it under [`dev_docs/PR_REVIEW_STANDARDS.md`](../../../../dev_docs/PR_REVIEW_STANDARDS.md) and decide any cell |

## 3. The working rules this column runs under

These bind the author's agents exactly as they bind the maintainers'. They are the
admission terms of [discussion #712](https://github.com/openwave-labs/openwave/discussions/712)
plus the platform contract, in one place.

| Rule | What it means at the keyboard |
| --- | --- |
| GitHub thread FIRST, at every session start | before any other work, read the open pull request's conversation and review state (`gh pr view <N> --repo openwave-labs/openwave --comments`). Maintainer decisions land there; work done blind to that thread can void itself |
| One concern, one pull request | each roadmap task lands as its own pull request; nothing else rides along, and nothing is imported in bulk |
| Draft until quiescent | while commits are still landing, the pull request is a draft; review runs on a settled head |
| Task ids resolve | a new task gets its roadmap row and `tasks/m10_<id>_task_details.md` BEFORE its scripts take the id |
| One candidate at a time | the column scores the executable model at a pinned commit. A changed law, memory, state or decoder is a new candidate: say so, move the pin by pull request, and keep the ancestry visible |
| Declared is inherited | a row on a declared or installed source stays not yet tested. Do not propose a cell move for it. Propose one when the observable is computed on the carrier's own process, and name what is still supplied |
| Calibration is labelled | a value used to tune any choice is a calibration target, and agreement with it is not a prediction. Held-out failures are reported, not retired into inputs silently |
| Maintainer pacing | review and merging follow maintainer schedule and resource availability; the queue absorbs throughput as per-task pull requests |
| Author gates | questions of intent, provenance or definition are the author's alone; an agent never resolves them by inference |
| Adversarial audit | before any substantive claim is trusted or shared: an independent second pass tries to REFUTE it with its own script and its own method; the audit is recorded in the deliverable |
| Public wording | the platform's claims are the cells in `MODELS.md` and the scripts behind them. Statuses on the author's own site are the author's estimates and are labelled as such |

And the one non-negotiable, from [`AI_HYGIENE.md`](../../../../AI_HYGIENE.md):
**model output is a draft or hypothesis, never a result, until verified by
something that is not a language model**: a runnable script, a hand-checked
derivation, a measurement, or the author's own authority. Agents show the script
and the number, never a verdict alone.

## 4. Completion protocol (what to print after reading)

Once §§ 1-3 are consumed, print, in this order:

| # | Print |
| --- | --- |
| 1 | A confirmation that you are ORIENTED on the M10 column and the platform contract |
| 2 | A one-line summary of EACH document you read (so the author can verify nothing was skipped) |
| 3 | The column's current status from the roadmap: what is done, what is startable, what is gated |
| 4 | A readiness statement: you can now (a) answer the author's questions about the M10 column and its OpenWave context, (b) help plan next moves, and (c) execute a roadmap task on command: **"go task m10.2"** means open that task's document under [`research/tasks/`](tasks/), confirm its pre-registered gates with the author, and run it per §§ 2-3 |
