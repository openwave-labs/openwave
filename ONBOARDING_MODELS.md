# Onboarding a Model to OpenWave

## What OpenWave is (for a first-time reader)

OpenWave is an open-source platform for **testing candidate field-theoretic models of matter against empirical observation, uniformly**. It is not one theory with one substrate. It is a **model database**: each candidate framework becomes a **column** scored against a shared set of criteria (particles, forces, waves, quantum emergence) in [`MODELS.md`](MODELS.md), and every cell in that matrix is earned by a **runnable script plus an honest research note**, or it stays marked "not yet tested".

The columns are deliberately heterogeneous. M5 is a 4x4 real symmetric tensor field. M6 is two coupled Lorenz-constrained vector fields. M8 is spectral geometry on a fixed quotient manifold, admitted with no dynamical field of its own. Each model gets its own directory, its own solver, and its own roadmap. **A new framework does not have to look like any of the existing ones, and it does not have to arrive finished.**

| The platform supplies | The model author supplies |
| --- | --- |
| The shared criteria and the honest scoring legend | The model: its equations, its claims, its numbers |
| The repository, the scaffold, and the review | The runs, the scripts, and the data behind every claim |
| Cross-model reading maps and prior art | The compute and the AI tokens to produce them |
| Maintainer help pointing the author's AI agent at the right places | The answers when someone challenges a claim on the column |

### Bring your own compute (BYOC)

That table is the platform's resource contract, and it is worth stating plainly because it decides what an author can plan around. There are four roles here, and each one brings something different:

| Role | What the role brings |
| --- | --- |
| **Platform maintainers** | The shared criteria and the honest status legend, the scaffold, the review, the best practices, the cross-model prior art, the onboarding of new models, and pull-request review |
| **Model authors** | The runs behind their own column: their own AI tokens and their own hardware, plus the answers when a claim on that column is challenged |
| **Model co-authors** | The same, over the scope named in that column's briefing ([§ Model co-authors](#model-co-authors)) |
| **Anyone else, contributing by pull request** | The same, for whatever they take on: a validation, an independent recompute, a falsification attempt |
| **Resource contributors** | Neither theory nor code, but what the runs cost: sponsorship, GPU or cluster hours, AI token grants, earmarked to named work and credited against it ([README § Resource Contributors](README.md#resource-contributors)) |

**What this means in practice.** Early on, the maintainer side funded most of the runs, because a comparison table with nothing in it convinces nobody and the database had to become non-zero first. That phase is over. Maintainer resources are now prioritized for running the platform: onboarding new models, reviewing pull requests, keeping the standards and the cross-model work in order, and honoring commitments already made. When resources are free, a maintainer can still help an author get scripts running, but that is help, not a guarantee, and it cannot be planned around. Bringing your own compute is what gives an author consistency.

**Why the resources are not pooled.** Sending tokens or compute to a maintainer to spend on someone else's behalf adds administration and puts a maintainer between a contributor and their own runs. The direct path is the useful one: spend the budget on your own runs, then open a pull request with the script and the note behind the result ([`CONTRIBUTING.md`](CONTRIBUTING.md) has the flow). Larger runs here are not blocked on permission, they are blocked on compute, so a contributor who brings tokens or machine time is bringing exactly the scarce resource.

**The one exception: resource contributors.** Someone who wants to fund the science without running it is not writing scripts, so "spend it on your own runs" does not apply to them. That contribution is still not pooled into a general fund: it is earmarked to a named run, that run publishes its script and result like any other cell, and the contributor is credited against it. Lanes, current cost centers, and how credit works: [README § Resource Contributors](README.md#resource-contributors).

**What the platform offers instead** is not compute, tokens, or admin work: a plain field for comparison, the same criteria applied to every column, the evidence shown for every claim, and openness to anyone who wants to put a model on the table or test one that is already there. Authors own their claims, and everyone owns the runs behind what they submit. The author-side detail is in [§ What the model author owns](#what-the-model-author-owns).

**The bar: reproducibility, not orthodoxy.** Unconventional frameworks are explicitly in scope. A documented negative (a runnable script showing "this does not work, and here is why") is as valuable as a positive. What is *not* in scope is an unfalsifiable claim, or a numerical agreement that cannot be independently reproduced from stated inputs.

This guide is for two readers:

| Reader | Use it to |
| --- | --- |
| **A model author** | Self-screen whether the framework fits, apply, and get a column scaffolded with rigor. |
| **An OpenWave maintainer** | Run a consistent evaluation and scaffold when a new model is proposed, so every column is admitted on the same terms. |

It complements and does not replace: [`MODELS.md`](MODELS.md) (the comparison table and the shared criteria), [`CONTRIBUTING.md`](CONTRIBUTING.md) (the canonical setup, pull-request, and DCO reference), [`TUTORIAL.md`](TUTORIAL.md) (zero to first contribution, for contributors generally), and [`REPRODUCE.md`](REPRODUCE.md) (the clean-clone path from any published claim back to the command that regenerates it, which is the reproducibility bar a model's results are held to).

## Contents

| Step | What happens |
| --- | --- |
| [**STEP 0:<br>drive it with an AI agent**](#step-0-drive-it-with-an-ai-agent) | How the work runs here, and what the model author owns |
| [**STEP 1:<br>self-evaluation**](#step-1-self-evaluation) | Does the model fit? Five criteria, the parameter-count test, the red flags, the hostile read |
| [**STEP 2:<br>application**](#step-2-application) | One discussion in the New Model category, with the fields a maintainer needs |
| [**STEP 3:<br>scaffolding and first PR**](#step-3-scaffolding-and-first-pr) | What a maintainer creates, what the author fills in, how the first pull request lands, and the headless-first mode of work |
| [Reference: status legend](#reference-status-legend) | The four icons a matrix cell can carry |
| [See also](#see-also) | The rest of the doc set |
| [DEEP READER ORIENTATION](#deep-reader-orientation) | For AI agents and deep readers landing on this page |

---

## STEP 0: drive it with an AI agent

### The single biggest recommendation

**Use an AI coding agent from day one.** Point it at this file and let it do the work alongside the author. Independent reproduction, parameter counting, and adversarial review are exactly what agents are good at, and the volume of work involved in earning a column honestly is more than most authors want to do by hand.

The fastest possible start, for an author with an agent that has repository access:

```text
Read ONBOARDING_MODELS.md in the OpenWave repo (github.com/openwave-labs/openwave)
and walk me through STEP 1 on my model. My papers are: <links>.
Show me every script and number, never a verdict alone.
```

**One firm rule when an agent does the science: the agent must show its work, the script and the numbers, never a verdict alone.** Language models will happily assert an agreement that does not exist. Every claim an agent makes must be backed by a runnable artifact the author can re-run.

Before doing any AI-assisted work here, read the repository-wide contract: [`AI_HYGIENE.md`](AI_HYGIENE.md). It is the single source for what AI is for in this project, what it must never be trusted with alone, and the verification habits that keep the record trustworthy. This section is the model-onboarding application of that page.

### What the model author owns

Being a column in OpenWave is a commitment, not a listing. Five things belong to the model author:

| Responsibility | What it means concretely |
| --- | --- |
| **Supply the compute** | The model author builds and runs the validation scripts on their own AI tokens and hardware. The platform supplies the standards, the scaffold, the review, and the prior art. It does not supply the token budget (the full resource contract, and why it is not pooled: [§ Bring your own compute](#bring-your-own-compute-byoc)) |
| **Own the model** | The column carries the author's name. When someone challenges a claim in a discussion or an issue, the author answers. A maintainer will not defend an author's physics for them, and should not |
| **Engage the other authors** | Cross-model questions, comparisons, and critiques are part of the deal. Other authors will read the column and ask hard questions; the author does the same for theirs |
| **Accept the collaborator invitation** | Once an application is accepted, a maintainer sends the author a GitHub invitation as an **external collaborator** on the repository. Accepting it is what makes someone a model author here rather than a cited reference |
| **Have a GitHub account** | Prerequisite for that invitation. An author without one creates it at [github.com/signup](https://github.com/signup) and posts the handle in the application discussion. It is the model briefing's **Author contact** field anyway, the route for author-gated questions |

**Author-gated questions.** Some questions only the author can answer: what a term in the Lagrangian is intended to mean, which version of a paper is the specification of record, whether a step is forced by the structure or chosen. The repository routes those to the author rather than guessing, per [`dev_docs/CROSS_MODEL_TESTING.md`](dev_docs/CROSS_MODEL_TESTING.md) section 6. Expect them, and answer them in writing where the answer becomes part of the record.

### Model co-authors

A column can carry more than one name. When a contributor's work becomes a standing part of a column rather than a one-off contribution, the column's author can promote that contributor to **co-author** over a named scope. Three steps, and no other machinery:

| Step | Who |
| --- | --- |
| Approve the promotion, in a public thread | the column's author |
| Accept it, in the same thread | the contributor |
| Turn the briefing's `Extension` row into a `Co-author` row, carrying the handle and the scope | a maintainer |

A co-author carries the author's duties over that scope and no further: owns the claims inside it, answers the challenges to it, and receives its author-gated questions. Everything outside the scope stays the author's. The role takes effect on the briefing edit, and it is removed the way it was added, by the author saying so in a public thread.

**What this changes for review.** An author-gated finding on a pull request opened by an author or co-author of that column goes to that person in the thread, and the review writes no model-author note at all ([`dev_docs/PR_REVIEW_STANDARDS.md`](dev_docs/PR_REVIEW_STANDARDS.md#81-when-the-contributor-is-not-an-author-of-the-column) section 8.1). A column can equally carry two authors from the start, as [M7](openwave/xperiments/m7_hydroboros/__M7_model_briefing.md) does, with a scope named against each handle; the routing is the same.

### Agent roles worth running (as separate, non-colluding passes)

| Agent role | Prompt it to | Guardrail |
| --- | --- | --- |
| Reproducer | Implement the stated formula from the stated constants and report the output number, with the script. | It must print the script and the value; the author re-runs it. |
| Independent recomputer | Recompute each quoted mathematical constant (eigenvalue, torsion, zeta value, volume) *from its own definition*, not from the paper's printed value. | Forbid it from reading the paper's number for that constant first. |
| Red-team / adversary | Argue, as hard as it can, that the model is over-fit. Find the hidden free parameters and the chosen-not-forced steps. | Reward finding problems, not confirming the result. |
| Citation checker | Verify each load-bearing citation says what the model claims it says. | Require quotes and locations, not paraphrase. |
| Parameter counter | Execute [the parameter-count test](#the-parameter-count-test) explicitly: list `N_obs`, enumerate every `N_free` choice, and compare. | Make it justify each item it counts or refuses to count. |

A practical pattern: run the reproducer and the independent recomputer first (do the numbers even hold?), then the red-team and the parameter counter (are they predictions or fits?), then synthesize. Disagreement between the reproducer and the recomputer is itself a finding. Treat a unanimous "all confirmed" from a single agent with suspicion, which is what the separate adversarial pass is for.

---

## STEP 1: self-evaluation

Work through this before applying. If the model author can answer all of it concretely, the model fits and the next step is [the application](#step-2-application). If not, the gaps are exactly what to work on first, and naming them honestly in the application is better than papering over them.

### 1.1 The one question that matters: prediction or post-fit?

Every other criterion is downstream of this one. For each number the model reproduces (a mass, a coupling, a mixing angle, a cross-section), ask:

| | Prediction | Post-fit |
| --- | --- | --- |
| Definition | The number is fixed by the model's structure and could have come out wrong. | The number was used, directly or indirectly, to set the model's structure. |
| Test | Was the value of the target known to the author *before* the formula that yields it was fixed? | Would changing the target force a change to some "choice" inside the model? |
| Status in OpenWave | Counts as a validation. | Counts as a calibration, and must be labelled as such. |

A model with many post-fit numbers is not disqualified, it is just scored honestly: calibration is not prediction. The danger is calling a post-fit a prediction, which [the parameter-count test](#the-parameter-count-test) is designed to catch.

### 1.2 The honest ledger: inputs vs calibration targets vs predictions

The model author produces this table for the model. Maintainers will ask for it, and it is a required field in the application.

| Category | What goes here | Example |
| --- | --- | --- |
| **Inputs** | Quantities the author assumes or fixes by hand (axioms, normalizations, one unit scale). | a single energy scale for unit conversion; integer topological classes |
| **Calibration targets** | Experimental numbers used to tune any choice. | a coupling that was fit, then reused |
| **Predictions** | Numbers fixed by structure alone, compared to data *after* the formula was closed. | a mass ratio that falls out with no further freedom |

"Zero free parameters" is a strong claim. It means: **no dimensionless quantity is adjusted after the inputs are stated.** If a sector-specific choice (which operator, which eigenvalue branch, which sign, which normalization) is selected to match data, that is a free parameter even if it is described in geometric or topological language. Count it.

### 1.3 Reproducibility: every claim is backed by a runnable script

OpenWave is a numerical platform. A claim that cannot be reproduced in code is not yet a column entry.

| Requirement | Means |
| --- | --- |
| Stated constants are recomputable | Any mathematical constant the model quotes (an eigenvalue, a torsion, a zeta value, a volume) can be recomputed from its own definition by someone who has not read the derivation. |
| The assembly is a script | Plugging the constants into the formula and getting the observable is a script another person can run and get the same number. |
| No hidden datasets | All compared values come from public sources (e.g. PDG, public cosmological catalogs). No private fit. |
| Standard tooling | Python / NumPy / SciPy, or a widely available CAS. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the environment. |

### 1.4 Falsifiers: what would kill the model

State, up front, the observations that would refute the framework, with the current experimental bound and the refutation threshold. A model with no falsifier is not yet science the platform can score.

| Signature | The model's prediction (with order of magnitude) | Current bound | Refutation criterion |
| --- | --- | --- | --- |
| (example row) | a deviation of order X in observable Y | current measurement / null | value outside [a, b] kills mechanism Z |

Sharp, near-term falsifiers are the most valuable. "Falsifiable in principle, someday" is weak. "This specific experiment, at this sensitivity, already constrains it" is strong.

### 1.5 Answer the structural criticisms in advance (the FAQ)

Strong submissions include a short FAQ that anticipates the obvious objections and answers each with a pointer to where in the work it is addressed. At minimum, answer:

| Objection a reviewer will raise | What the answer must show |
| --- | --- |
| "With enough machinery you can fit any numbers." | Why the agreements are predictions, not fits (section 1.1, and the count below). |
| "Is one input really one input?" | That the other 'derived' quantities are genuinely derived, not quietly re-introduced. |
| "This theorem is really a definition / an identification." | A clean separation of axioms, theorems (logical deductions), and physical identifications. |
| "Isn't the argument circular?" | The logical chain, with each step independent of the conclusion it supports. |
| "This reproduces a known numerical coincidence." | What is new in the derivation beyond the coincidence, and that the coincidence is a check, not the basis. |

### The parameter-count test

This is the single most useful test for any model that claims to derive observables, and the one a maintainer runs before admitting a column. It separates a genuine prediction from a laundered fit. The model author runs it first.

| Step | Action |
| --- | --- |
| 1. Count the outputs | `N_obs` = number of independent observables the model reproduces (e.g. three lepton masses). |
| 2. Count the genuine inputs | `N_free` = number of structural choices that could have been otherwise: which operator, which eigenvalue branch, which knot / topology / representation, which normalization, which sign, each calibration scale. Count a choice as free if the model's own framework admits an inequivalent alternative. |
| 3. Compare | If `N_free` >= `N_obs`, "zero free parameters" is illusory: the freedom is hidden inside the 'choices'. The agreement is then a fit, however it is described. |
| 4. Recompute independently | Recompute every quoted constant from its own definition, not from the number printed in the paper. Confirm the assembled formula actually yields the observable. |
| 5. Forced vs chosen | For each step the model calls "forced", find whether the structure permits an inequivalent alternative. If it does, it was chosen. Genuine forcing means: fixed by something independent of the target being reproduced. |

If a model passes step 3 (few genuine inputs), survives step 4 (constants recompute independently), and step 5 shows the key choices are forced by structure independent of the data, that is a real result and a strong ✅. If it fails any step, document precisely where the freedom hides: that is a valuable ⚠️ or ❌, and exactly the platform's product.

### The red-flag checklist

These are the patterns that most often distinguish an over-claimed framework from a sound one. The model author runs this before a maintainer or a hostile reader does.

| Red flag | Why it matters | Self-check |
| --- | --- | --- |
| Everything matches to ~0 sigma with "zero free parameters" | A framework that reproduces *every* observable to the central value, with no residuals anywhere, is far more likely over-fit than uniquely correct. Real first-principles theories leave residuals. | Are there *any* honest residuals or tensions? If not, suspect hidden tuning. |
| Knobs dressed as structure | Sector-specific choices (a framing number, a knot assignment, a normalization) each given a separate post-hoc justification that happens to land on the answer. | Could a reader have derived each choice *before* seeing the target? List them and check. |
| Reproducing a known coincidence | Recovering a historically famous numerical coincidence (and presenting it as derivation) without adding new, independent content. | What does the derivation add beyond the coincidence? Is the coincidence a check or the load-bearing step? |
| Rhetorical certainty | Near-unity Bayesian posteriors, claims of being the unique possible theory, appeals to authority or destiny. Reviewers read these as overreach, not evidence. | Remove the rhetoric. Does the case still stand on the numbers alone? |
| Unrefereed / fringe sourcing | Leaning on preprint-commons or non-refereed sources as support. | Are the load-bearing citations to refereed, checkable results? |
| Theorems that are identifications | Physical assignments written in theorem-like language, blurring "proved" with "interpreted". | Label every statement: axiom, theorem (logical deduction), definition, or physical identification. |
| No falsifier | A model that cannot be wrong cannot be scored. | Section 1.4. If it cannot be filled in, the model is not yet testable here. |

### The hostile cold-reader pass

Before applying, put the work through a reader who *wants it to be wrong* and has not been softened by the author's narrative. If no such person is available, simulate one with the red-team agent role from [STEP 0](#agent-roles-worth-running-as-separate-non-colluding-passes). The protocol:

| Step | What the hostile reader does |
| --- | --- |
| Strip the prose | Discard the motivation and keep only four lists: what is input, what is derived, what is predicted, what would falsify it. |
| Re-derive one headline number cold | Pick the most impressive result and recompute it from scratch, using only the stated constants and definitions, never the paper's intermediate values. |
| Attack the forced steps | For every "uniquely forced" claim, search for an inequivalent alternative the framework also permits. One counterexample downgrades "forced" to "chosen". |
| Count the bits | Apply [the parameter-count test](#the-parameter-count-test). Does the structural freedom exceed the information content of the data reproduced? |
| Check the citations | Verify each cited theorem is used as stated and actually supports the step it is attached to. |
| Demand a residual | If nothing disagrees with experiment anywhere, treat that as evidence of over-fit until shown otherwise. |

A model that survives a genuine hostile pass is ready. One that has only been read by sympathetic readers is not.

### Fit scorecard

| Criterion | Pass condition |
| --- | --- |
| Maps onto shared criteria | The model addresses at least some rows in [`MODELS.md`](MODELS.md) (particles, forces, waves, quantum emergence). |
| Predictions exist | At least one genuine prediction (section 1.1), not only calibrations. |
| Reproducible | Stated constants and the assembly run in code (section 1.3). |
| Falsifiable | At least one concrete falsifier with a current bound (section 1.4). |
| Survives the FAQ | The structural objections (section 1.5) have answers. |

Partial coverage is normal and welcome. Most cells in a new column begin as 🚧 "not yet tested in-platform" and deepen over time.

**A model with no Lagrangian is still admissible.** [M8](openwave/xperiments/m8_mit/__M8_model_briefing.md) was admitted with no field Lagrangian and no equation of motion, scored honestly, with the missing dynamics named as its defining open problem and a field-dynamics collaboration set up around it. That collaboration was not left as a promise: the contract the Lagrangian survey is graded against was pre-registered and locked before any numerics existed to tune it toward ([`m8_2_preregistration.md`](openwave/xperiments/m8_mit/research/findings/m8_2_preregistration.md)), contributed by the author's own first pull request. The author supplies the arena and the target structure; the platform supplies the candidate Lagrangians, the simulation engineering, and the grading. Where that program stands on any given day is in its roadmap ([`m8_roadmap.md`](openwave/xperiments/m8_mit/research/m8_roadmap.md)), not here. A framework that is not formulated as an action principle should say so plainly in the application rather than working around it. What the platform cannot substitute for is a **closed set of equations**: something that can be integrated, solved, or eigen-decomposed to produce a number.

**The four tiers.** "Has a Lagrangian" is not the bar, and never was. Frameworks arrive at one of four tiers, and the first three are all admissible:

| Tier | What the model supplies | Admissible |
| --- | --- | --- |
| 1 | An action principle, varied to Euler-Lagrange field equations | Yes, with the guarantees below |
| 2 | Evolution equations written directly, with no action behind them | Yes, without those guarantees |
| 3 | A spectral or eigenvalue problem, no time evolution at all | Yes, for static observables only |
| 4 | No closed system: nothing to integrate, solve, or eigen-decompose | No. This is the only disqualifying case |

Tier 2 is ordinary physics, not a compromise. Maxwell's equations were written down and solved for decades before anyone formulated the electromagnetic action, and Navier-Stokes has no standard Lagrangian while being among the most heavily simulated systems in existence. An author whose framework is tier 2 should say so plainly and will not be marked down for it.

**What tier 2 gives up, and must therefore measure.** Deriving equations from an action buys four things for free. A tier-2 column has to establish each one by hand instead:

| Guarantee lost | What it costs the column |
| --- | --- |
| Noether's theorem | Energy, momentum and charge conservation stop being automatic. They become quantities to verify numerically, and drift is a result to report rather than a bug to hide |
| Canonical stress-energy tensor | `T_μν` no longer falls out of varying against the metric, so the gravity rows need their source term supplied another way |
| Self-consistency by construction | A hand-assembled system can be quietly overdetermined or mutually inconsistent. Worth an explicit check before any physics claim rests on it |
| Symplectic structure | Structure-preserving integrators are unavailable, so long-run energy drift has to be measured and bounded rather than assumed away |

None of that blocks admission. It changes what the column's method notes have to show: a tier-1 model may assert conservation, a tier-2 model has to demonstrate it.

**Why tier 3 is bounded.** A spectral model produces real, checkable numbers, but its observables are the static catalogue: what states exist, what they weigh, what quantum numbers they carry. Anything about persistence, propagation or exchange (stability under perturbation, angular momentum, magnetic moment, annihilation, radiation, a clock rate) needs time evolution, which is why those rows sit at 🚧 in a tier-3 column and why the [`MODELS.md`](MODELS.md) coverage matrix carries a `regime` column marking which rows demand it. A tier-3 model is admitted on the understanding that the gap is named, not worked around. Naming it is also what opens the program to close it: a stated gap becomes a task with a pre-registration and a place on the roadmap, which is how a column moves up a tier.

---

## STEP 2: application

**There is no form.** Open one discussion in the **[New Model](https://github.com/openwave-labs/openwave/discussions/categories/new-model)** category. A maintainer picks it up from there.

The application body is the raw material for the model briefing (STEP 3), so the fields below are the briefing's own fields. Fill in what is available; write "none native" or "not yet" where that is the honest answer. An agent can draft this from the papers in one pass, and the author corrects it.

### Before posting (author checklist)

| Check | Why |
| --- | --- |
| A GitHub account exists and the handle is in the post | It is how the collaborator invitation reaches the author, and it is the briefing's Author contact field |
| [STEP 1](#step-1-self-evaluation) has been run on the author's own work | The application asks for its outputs directly |
| The bedrock papers are reachable by link | DOI, arXiv, Zenodo, or a public repository. A file on request is not enough for a public record |
| Which paper backs which claim can be named | "It is all in the book" is not a citation a reviewer can check |

### What to put in the application

| Field | What to write |
| --- | --- |
| **Model name** | The full name plus the short name for the column (e.g. "Mode Identity Theory (MIT)") |
| **Author** | Name, affiliation or "independent researcher", and whether the author is the sole author |
| **Author contact** | GitHub handle (required, for author-gated questions), ORCID if there is one |
| **Lineage** | The tradition and prior work the framework builds on, named |
| **Bedrock papers** | Links (DOI / arXiv / Zenodo / repository) **and which specific paper backs which claim**. Name the one that is the specification of record if there are several versions |
| **Substrate** | What the field is, concretely: how many components, what geometry, what symmetry. "None, this is not a dynamical medium" is a valid answer |
| **Dynamics** | The Lagrangian, action, or equations of motion. If there is none native, say so and say what plays that role instead |
| **Particle** | What a particle *is* in the framework |
| **Charge** | Where charge comes from, and whether quantization is derived or assigned |
| **Free parameters ledger** | The three-column table from [section 1.2](#12-the-honest-ledger-inputs-vs-calibration-targets-vs-predictions): inputs, calibration targets, predictions |
| **Honest residuals** | The places the model disagrees with experiment, listed rather than smoothed. Their presence is a credibility signal, not a weakness |
| **Falsifiers** | The table from [section 1.4](#14-falsifiers-what-would-kill-the-model), with thresholds, and dates where the test is a specific upcoming measurement |
| **Formal artifacts** | Any code, calculator, notebook, proof-assistant formalization, or dataset that already exists, with links |
| **Which MODELS.md rows the model addresses** | Point at the actual criteria rows in [`MODELS.md`](MODELS.md). Partial coverage is expected |
| **Help wanted** | What the author wants from the platform and from other authors: an independent recompute, a Lagrangian candidate, a simulation route, a falsification attempt |

The M8 application, [discussion #312](https://github.com/openwave-labs/openwave/discussions/312), is the worked example: the author posted a complete model briefing as the discussion body, including the negatives, and the column was scaffolded from it directly.

### What happens next

| Who | Does |
| --- | --- |
| Maintainer | Reads the application, runs [the parameter-count test](#the-parameter-count-test) as a first pass, and asks clarifying questions in the thread |
| Maintainer | Accepts (or explains what is missing), then scaffolds the column and **sends the author the external-collaborator invitation** |
| Model author | Accepts the invitation, then works through [STEP 3](#step-3-scaffolding-and-first-pr) |

A decision is about testability, not about whether anyone agrees with the physics.

---

## STEP 3: scaffolding and first PR

### 3.1 What the scaffold contains

A maintainer creates the directory and the starting documents; the model author fills them in and deepens them. The file set below is what the M8 onboarding actually needed, which is more than a briefing alone:

| File | What it is | Who writes it first |
| --- | --- | --- |
| Model Briefing | The human front door: identity, profile, per-particle field configs, status, roadmap, help wanted (section 3.3)<br>`openwave/xperiments/<your-model>/__<MID>_model_briefing.md` | Maintainer drafts from the application, the author corrects |
| Canonical | **The specification of record**, canonical when documents disagree: the arena, the equations, the particle map, known tensions, consumption rules, open questions<br>`research/<mid>_theory_canonical.md` | Maintainer drafts, the author owns it |
| Background | The gap map: what the framework has, what it lacks, why the program exists, and the onboarding evaluation of record<br>`research/<mid>_background.md` | Maintainer |
| Roadmap | The program: tasks, gates, per-task ownership (author-driven or maintainer-driven), current status<br>`research/<mid>_roadmap.md` | Maintainer drafts, the author reorders |
| Cross-Model | A cross-model reading map written **for the author's AI agents**: where in the other columns to find Lagrangian families, defect taxonomies, clock and stability lessons, engine routes, with the platform-contract docs first<br>`research/<mid>_platform_pointers.md` | Maintainer |
| Agent Orientation | **The agent front door**: the ordered load list, then a completion protocol so an agent can declare itself oriented and start work from "go task `<mid>`.2"<br>`research/<mid>_agent_orientation.md` | Maintainer |
| Task Planning | One planning document per task: scope, definition of done, pre-registered gates<br>`research/tasks/<mid>_<n>_task_details.md` | Maintainer for task 1, then the author |
| Citations | The source registry (section 3.4). The papers themselves stay local and gitignored<br>`theory/_CITATIONS.md` | Maintainer drafts from the application |
| Additional Folders | Empty directories, ready for the first run<br>`research/{scripts,data,plots,findings,images,checkpoints}/` | Maintainer |

**Lesson from the M8 onboarding, applied here:** the agent-orientation document was added three days *after* that scaffold, once it became clear the author's agent needed a bootstrap. It is now scaffolded on day one. If an agent cannot orient itself from one file, the scaffold is incomplete, and the author should say so.

### 3.2 The maintainer sequence

For maintainers, and so authors know what to expect. This is the M8 order, which worked:

| # | Step |
| --- | --- |
| 1 | Scaffold the column and the file set above from the application discussion |
| 2 | **Send the author the GitHub external-collaborator invitation** (requires the handle from the application; if the author has no account, ask for one to be created and posted in the thread) |
| 3 | Pin the first task's specification from the author's own words in the application thread, so the run tests what the author actually claims |
| 4 | Promote task 1 to active and link its planning document |
| 5 | Run task 1 as a **blind certification gate**: an independent recompute of one bedrock claim, gates pre-registered before the run, by an agent that has not seen the claimed values, adversarially audited afterwards |
| 6 | Record the result honestly, negatives included, then add the agent-orientation document so the author can take over |
| 7 | Add the column to [`MODELS.md`](MODELS.md) with its cells at their earned status, most of them 🚧 |
| 8 | Register the author wherever the roster is kept: a row in [`README.md` § Theoretical Contributors](README.md#theoretical-contributors), and the advisors and project-architecture tables in [`CLAUDE.md`](CLAUDE.md) |

**Certification first** is the pattern: the first task on a new column is not a new result, it is an independent check that one existing headline number holds up. It costs little, it establishes the working relationship, and it is the strongest possible start for a column's credibility.

### 3.3 The model briefing

Every model directory carries a one-page briefing, `__<MID>_model_briefing.md` (e.g. `__M5_model_briefing.md`), that summarizes what the model brings. It is the front door a reader, a maintainer, or a mailing-list reader hits first, and drafting it honestly is itself a **self-test**: it forces the same statements STEP 1 asks for, in a form anyone can scan in a minute.

Use [`m5_liquid_crystal/__M5_model_briefing.md`](openwave/xperiments/m5_liquid_crystal/__M5_model_briefing.md) as the worked template. Its six sections and the self-test each one enforces:

| Section | What it states | Self-test it enforces |
| --- | --- | --- |
| Identity | ID, name, author, author contact, lineage, primary sources, in-repo files | provenance is explicit and citable |
| Model Profile | short-form rows: substrate, particle, charge, Derrick escape, clock, EM, gravity, free parameters, lab anchor, next falsifier | the honest ledger (1.2) plus a falsifier (1.4) at a glance |
| Field Configuration of Particles | per-particle field config and "topological vortex?"; whether the clock is derived or assumed | each particle can actually be specified, not just named |
| Implementation Status | per-sector ✅ / ⚠️ / ❌ / 🚧 against the shared criteria, with deep-dive links | honest status, negatives included |
| Roadmap | what lands next | the open work is named, not hidden |
| Help Wanted | how others can contribute a validation, a falsifier, or a rival configuration | the column stays open |

A new column's briefing may be mostly 🚧 at first, and that is expected. The point is that it exists, is honest, and reads as a scannable one-pager. It does not replace the deeper documents (roadmap, question tracker, canonical); it is the index to them.

### 3.4 The theory corpus and `_CITATIONS.md`

Every model directory has a `theory/` folder holding the foundational papers and author documents the framework builds on. Two rules govern it: a copyright and git rule for the files, and a format rule for the record.

**Copyright and git (non-negotiable).** OpenWave is a public repository. Third-party papers are copyrighted and MUST NOT be committed. Keep the PDFs and author documents **local-only and gitignored**; only the author's own `.md` notes and code are tracked. The repository's `.gitignore` already ignores the common document types under any `theory/` folder:

```gitignore
openwave/xperiments/*/theory/**/*.pdf     # also .docx .doc .epub .ppt .pptx .djvu .tex .bib
```

The one tracked artifact in `theory/` is `_CITATIONS.md`. It is the durable record that a source existed and where it came from, even though the file itself is never committed. Anyone who needs a paper obtains it from its original venue (DOI / arXiv) or from the author.

**`_CITATIONS.md` format.** Name it with a leading underscore so it sorts to the top of the folder. It is dual-purpose: a readable bibliography *and* a file-inventory manifest. Copy [`m7_hydroboros/theory/_CITATIONS.md`](openwave/xperiments/m7_hydroboros/theory/_CITATIONS.md) as the worked template. Structure:

| Part | What it holds |
| --- | --- |
| Provenance blockquote | the "NOT in git, obtain from the original venue" note (the copyright reminder) |
| Total line | document count plus an as-of date |
| `## Bibliography` | the readable citation list, ordered by **year ascending** (undated entries last); columns `Author(s) \| Year \| Title \| Venue / ID` |
| `## Local corpus` | the gitignored-file inventory; columns `Author (Year) \| Path \| Size`, size formatted `X.X MB` / `XXX KB` |

**Rules that keep it honest and usable:**

| Rule | Why |
| --- | --- |
| Never fabricate an identifier | Verify DOIs and arXiv IDs; unresolved becomes `n/a`, an author-shared draft becomes `author copy`, a guessed venue gets a trailing `(?)`. A wrong DOI is worse than an honest `n/a`. |
| Keep the manifest matching disk | Add, remove, or rename a source file and `_CITATIONS.md` is updated in the same commit. Every `Path` in the Local corpus must resolve to a real file. |
| ASCII, clean filenames | Prefer `YEAR - Author - Title.pdf`. No em or en dashes and no unicode ligatures in a source filename or in this manifest's cells: they break byte-exact paths, and a pasted `Path` value stops resolving. The scope is filenames and manifest cells. Prose elsewhere in the repo is not covered by this rule. |
| One work, one Bibliography entry | If a work has more than one local file (e.g. a published PDF plus its LaTeX source), it is a single citation but multiple Local-corpus rows. |
| Cite as Author (Year) | Other documents reference sources this way, so no separate citation-key column is needed. |

### 3.5 Setup and the first pull request

The canonical, always-current setup is [`CONTRIBUTING.md`](CONTRIBUTING.md). The short path:

```bash
# 1. Fork the repo on GitHub (all work happens on the fork; there is no direct push).

# 2. Clone the fork
git clone https://github.com/YOUR-USERNAME/openwave.git
cd openwave

# 3. Environment (conda recommended)
conda create -n openwave python=3.12
conda activate openwave
pip install -e .            # installs dependencies from pyproject.toml

# 4. One-time: enable the auto DCO sign-off hook
git config core.hooksPath .githooks

# 5. Branch for the model
git checkout -b add-<your-model>

# 6. Commit with a DCO sign-off (required), then push and open a PR
git commit -s -m "Add <your-model> column and scaffold"
git push origin add-<your-model>
```

The `-s` flag adds the `Signed-off-by:` line that certifies the contributor has the right to contribute the work under [Apache 2.0](LICENSE). The hook in step 4 adds it automatically if it is forgotten.

**A good first PR** adds the column plus a model directory with one or two cells actually backed (a runnable script plus a note), and the rest marked 🚧 honestly. Finite-difference or first-pass now, fuller validation later, is fine and expected.

A maintainer reviews with a **light PR review** focused on two things only: (1) a runnable script that reproduces the claim, and (2) a research note documenting pass or fail honestly. It is not ideological gatekeeping.

Review runs on **one document, one merge**: comments are settled in the thread before merging, and a maintainer applies small fixes to the branch at merge rather than sending them back as another cycle, so a contribution does not land across a chain of follow-up pull requests. What that means case by case, including for a frozen document: [`CONTRIBUTING.md § How review works`](CONTRIBUTING.md#how-review-works).

### 3.6 Mode of work: headless first, rendering last

Every column runs the same order, and it is worth stating because the appealing thing to do is the opposite one.

OpenWave has two solvers over the same field equations. The **headless** solver (also called sandbox or research: plain scripts under a model's `research/scripts/`, no GUI) is where a question gets answered. The **rendered** solver (also called production, launcher, or GGUI: a model's `_launcher.py` and its `xparameters/` configs) is where an answered question gets shown. [`TUTORIAL.md § 3`](TUTORIAL.md#3-the-two-solvers-headless-sandbox-vs-rendered) lays out what each is good for.

| | Headless solver | Rendered solver |
| --- | --- | --- |
| Answers | "which configuration is selected, and by how much" | "what does this mechanism look like, and how does it respond when perturbed" |
| Runs as | `python script.py`, unattended, loopable over a parameter grid | `openwave -x`, one interactive window at a time |
| Produces | numbers, JSON/CSV, PNGs, a written note anyone can regenerate | an on-screen render |

**Do the research headless, port to the rendered engine afterwards.** Two reasons, and the first is the practical one:

| Why | Detail |
| --- | --- |
| A rendered study does not leave a result behind | A parameter sweep driven through the launcher is one window per configuration, watched by a person. Its conclusion lives with whoever watched it. The same sweep as a script plus a loop runs unattended and ends in a table anyone can regenerate, which is the only form that can move a cell in [`MODELS.md`](MODELS.md) |
| Rendering an unvalidated dynamics showcases nothing | Until there is a statement about what the simulation does, a viewer has no way to tell a result from an artifact, and neither does the person who built it |

Where each column writes it down: M5 holds "rendering gates nothing" as a standing convention and reached its `_launcher.py` through headless rounds first; [M7](openwave/xperiments/m7_hydroboros/research/m7_roadmap.md) states "headless first, rendering graduates once the electron is canonical, no GUI work before the physics is canonical"; [M8](openwave/xperiments/m8_mit/research/m8_roadmap.md) makes its rendering port a task gated on validated dynamics, and the port instructions themselves open with "Do not start here".

When a result does graduate, the launcher runs **the same kernels the headless work validated**. An interactive demo shows the physics of record, including its instabilities, rather than a prettier separate thing.

This applies to contributions to existing columns exactly as it does to a new one. [`m4_ewt/research/README.md`](openwave/xperiments/m4_ewt/research/README.md) is a compact worked statement of the whole convention if a single page is more useful than this section.

---

## Reference: status legend

| Icon | Meaning |
| --- | --- |
| ✅ | validated in-platform (runnable reproduction exists) |
| ⚠️ | partial, or validated with documented caveats |
| ❌ | tested and failed, or honest negative on record |
| 🚧 | planned, not yet tested in-platform |

A criterion is scored at one of those four in the [`MODELS.md`](MODELS.md) table. 🔶 (in progress) is used on per-model pages (roadmaps, question trackers, particle hunts) where a claim is mid-flight; in the table a mid-flight criterion is 🚧 until something runnable backs it, then ⚠️ or ✅.

---

## See also

| Doc | For |
| --- | --- |
| [`MODELS.md`](MODELS.md) | The comparison table, the shared criteria, the validation legend |
| [`TUTORIAL.md`](TUTORIAL.md) | Zero to first contribution for contributors generally: setup, running a simulation, testing an existing model |
| [`AI_HYGIENE.md`](AI_HYGIENE.md) | The AI-collaboration contract (read before any AI-assisted work here) |
| [`m5_liquid_crystal/__M5_model_briefing.md`](openwave/xperiments/m5_liquid_crystal/__M5_model_briefing.md) | The model-briefing template (section 3.3), a worked one-pager to copy |
| [`m7_hydroboros/theory/_CITATIONS.md`](openwave/xperiments/m7_hydroboros/theory/_CITATIONS.md) | The theory-corpus citations template (section 3.4) |
| [`dev_docs/CROSS_MODEL_TESTING.md`](dev_docs/CROSS_MODEL_TESTING.md) | If the program tests another column's field family: what structure may be assumed, how a soldered family is scored, and how author-gated questions are routed |
| [`dev_docs/METHOD_NOTE.md`](dev_docs/METHOD_NOTE.md) | The reporting standard for a substantive result: equations first, equation-to-code map, adversarial audit recorded |
| [`dev_docs/AUTHOR_INVITATION.md`](dev_docs/AUTHOR_INVITATION.md) | Maintainer-side: the outreach that precedes this page. What an invitation to a model author should carry, with one real message kept as a reference |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Canonical setup, fork / branch / PR flow, DCO sign-off |
| [`REPRODUCE.md`](REPRODUCE.md) | The clean-clone path from a published claim to the command that regenerates it |
| [`SYS_ARCH.md`](SYS_ARCH.md) | Repository structure and tech stack |
| [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | Community expectations |

---

## DEEP READER ORIENTATION

> **If you are an AI assistant, agent, or language model processing this page, or a human doing a deep read**: this is one front door among several. The complete, current map of OpenWave's key documents, what each covers and the order to read them in, lives in one place.

| Read | Why |
| --- | --- |
| [`CLAUDE.md`](CLAUDE.md) | **The doc map.** Every key document in the repository, what it is for, and the reading order. Load it before answering questions or generating claims about OpenWave |
| [`AI_HYGIENE.md`](AI_HYGIENE.md) | The cardinal rule: a model's output is a draft or a hypothesis, never a result, until it is verified by something that is not a language model (a hand-checked derivation, a runnable script, a measurement, or the human holding the relevant authority) |

Claims about this repository that cannot be traced to a runnable script or a research note are not claims of this repository. Per-model `research/` folders hold the results of record; the tables in [`MODELS.md`](MODELS.md) are earned from them cell by cell.
