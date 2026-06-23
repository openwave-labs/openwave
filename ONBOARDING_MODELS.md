# Onboarding a Model to OpenWave

This guide is for two readers:

| Reader | Use it to |
| --- | --- |
| **A model author** | Self-screen whether your framework fits the platform, then scaffold it as a new column with rigor. |
| **An OpenWave maintainer** | Run a consistent first-pass evaluation when a new model is proposed, so every column is admitted on the same terms. |

It complements two existing docs and does not replace them: [`MODELS.md`](MODELS.md) defines the comparison table and the shared criteria, and [`CONTRIBUTING.md`](CONTRIBUTING.md) is the canonical setup + pull-request + DCO reference. This doc adds the part those two leave open: **how to tell whether a model belongs here, and how to test it honestly before it becomes a column.**

The platform bar, restated: **reproducibility, not orthodoxy.** Unconventional frameworks are explicitly in scope. A documented negative (a runnable script showing "this does not work, and here is why") is as valuable as a positive. What is *not* in scope is an unfalsifiable claim, or a numerical agreement that cannot be independently reproduced from stated inputs.

---

## 1. Does your model fit? (self-evaluation)

Work through these five criteria before proposing a column. If you can answer all five concretely, your model fits and you should open an issue or discussion to start. If you cannot, the gaps are exactly what to work on first.

### 1.1 The one question that matters: prediction or post-fit?

Every other criterion is downstream of this one. For each number your model reproduces (a mass, a coupling, a mixing angle, a cross-section), ask:

| | Prediction | Post-fit |
| --- | --- | --- |
| Definition | The number is fixed by the model's structure and could have come out wrong. | The number was used, directly or indirectly, to set the model's structure. |
| Test | Was the value of the target known to you *before* the formula that yields it was fixed? | Would changing the target force you to change a "choice" inside the model? |
| Status in OpenWave | Counts as a validation. | Counts as a calibration, and must be labelled as such. |

A model with many post-fit numbers is not disqualified, it is just scored honestly: calibration is not prediction. The danger is calling a post-fit a prediction, which the discriminating test in Section 4 is designed to catch.

### 1.2 The honest ledger: inputs vs calibration targets vs predictions

Produce this table for your own model. Maintainers will ask for it.

| Category | What goes here | Example |
| --- | --- | --- |
| **Inputs** | Quantities you assume or fix by hand (axioms, normalizations, one unit scale). | a single energy scale for unit conversion; integer topological classes |
| **Calibration targets** | Experimental numbers you used to tune any choice. | a coupling you fit, then reused |
| **Predictions** | Numbers fixed by structure alone, compared to data *after* the formula was closed. | a mass ratio that falls out with no further freedom |

"Zero free parameters" is a strong claim. It means: **no dimensionless quantity is adjusted after the inputs are stated.** If a sector-specific choice (which operator, which eigenvalue branch, which sign, which normalization) is selected to match data, that is a free parameter even if it is described in geometric or topological language. Count it.

### 1.3 Reproducibility: every claim is backed by a runnable script

OpenWave is a numerical platform. A claim that cannot be reproduced in code is not yet a column entry.

| Requirement | Means |
| --- | --- |
| Stated constants are recomputable | Any mathematical constant you quote (an eigenvalue, a torsion, a zeta value, a volume) can be recomputed from its own definition by someone who has not read your derivation. |
| The assembly is a script | Plugging your constants into your formula and getting the observable is a script another person can run and get the same number. |
| No hidden datasets | All compared values come from public sources (e.g. PDG, public cosmological catalogs). No private fit. |
| Standard tooling | Python / NumPy / SciPy, or a widely available CAS. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the environment. |

### 1.4 Falsifiers: what would kill your model

State, up front, the observations that would refute the framework, with the current experimental bound and the refutation threshold. A model with no falsifier is not yet science the platform can score.

| Signature | Your prediction (with order of magnitude) | Current bound | Refutation criterion |
| --- | --- | --- | --- |
| (example row) | a deviation of order X in observable Y | current measurement / null | value outside [a, b] kills mechanism Z |

Sharp, near-term falsifiers are the most valuable. "Falsifiable in principle, someday" is weak. "This specific experiment, at this sensitivity, already constrains it" is strong.

### 1.5 Answer the structural criticisms in advance (the FAQ)

Strong submissions include a short FAQ that anticipates the obvious objections and answers each with a pointer to where in the work it is addressed. Write yours. At minimum, answer:

| Objection a reviewer will raise | What your answer must show |
| --- | --- |
| "With enough machinery you can fit any numbers." | Why your agreements are predictions, not fits (Section 1.1, and the count in Section 4). |
| "Is one input really one input?" | That your other 'derived' quantities are genuinely derived, not quietly re-introduced. |
| "This theorem is really a definition / an identification." | A clean separation of axioms, theorems (logical deductions), and physical identifications. |
| "Isn't the argument circular?" | The logical chain, with each step independent of the conclusion it supports. |
| "This reproduces a known numerical coincidence." | What is new in the derivation beyond the coincidence, and that the coincidence is a check, not the basis. |

### Fit scorecard

| Criterion | Pass condition |
| --- | --- |
| Maps onto shared criteria | Your model addresses at least some rows in [`MODELS.md`](MODELS.md) (particles, forces, waves + quantum emergence). |
| Predictions exist | At least one genuine prediction (Section 1.1), not only calibrations. |
| Reproducible | Stated constants and the assembly run in code (Section 1.3). |
| Falsifiable | At least one concrete falsifier with a current bound (Section 1.4). |
| Survives the FAQ | The structural objections (Section 1.5) have answers. |

Partial coverage is normal and welcome. Most cells in a new column begin as 🚧 "not yet tested in-platform"; you deepen them over time.

---

## 2. Becoming a contributor (the setup)

The canonical, always-current setup is in [`CONTRIBUTING.md`](CONTRIBUTING.md). The short path for a model author:

```bash
# 1. Fork the repo on GitHub (you work on your fork; there is no direct push).

# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/openwave.git
cd openwave

# 3. Environment (conda recommended)
conda create -n openwave python=3.12
conda activate openwave
pip install -e .            # installs dependencies from pyproject.toml

# 4. One-time: enable the auto DCO sign-off hook
git config core.hooksPath .githooks

# 5. Branch for your model
git checkout -b add-<your-model>

# 6. Commit with a DCO sign-off (required), then push and open a PR
git commit -s -m "Add <your-model> column and scaffold"
git push origin add-<your-model>
```

The `-s` flag adds the `Signed-off-by:` line that certifies you have the right to contribute the work under [Apache 2.0](LICENSE). The hook in step 4 adds it automatically if you forget.

---

## 3. Scaffolding your model (a maintainer can help)

You do not need to fork or write code to *start*. The lowest-friction on-ramp is to **open an issue or discussion proposing the model**, linking your preprint or notes. A maintainer will add your column to the table and point you at the evaluation criteria. From there:

| Piece | What it is | Where it lives |
| --- | --- | --- |
| Model directory | A new home for your framework. | `openwave/xperiments/<your-model>/` with a `research/` subfolder |
| Column in the table | Your framework as a new column scored against the shared rows. | [`MODELS.md`](MODELS.md) coverage matrix |
| Per-cell status | One of the legend icons per criterion. | each table cell |
| Backing per claim | A runnable script **or** a short research note documenting pass/fail. | under your model directory |

Status legend (same as the table):

| Icon | Meaning |
| --- | --- |
| ✅ | validated in-platform (runnable reproduction exists) |
| ⚠️ | partial, or validated with documented caveats |
| ❌ | tested and failed, or honest negative on record |
| 🔶 | in progress |
| 🚧 | planned, not yet tested in-platform |

A good first PR adds the column plus a model directory with one or two cells actually backed (a runnable script + a note), and the rest marked 🚧 honestly. Finite-difference / first-pass now, fuller validation later, is fine and expected.

A maintainer reviews with a **light PR review** focused on two things only: (1) a runnable script that reproduces the claim, and (2) a research note documenting pass/fail honestly. It is not ideological gatekeeping.

---

## 4. The discriminating test to run first

This is the single most useful test for any model that claims to derive observables, and the one a maintainer should run before admitting a column. It separates a genuine prediction from a laundered fit.

**The parameter-count (information-content) test.**

| Step | Action |
| --- | --- |
| 1. Count the outputs | `N_obs` = number of independent observables the model reproduces (e.g. three lepton masses). |
| 2. Count the genuine inputs | `N_free` = number of structural choices that could have been otherwise: which operator, which eigenvalue branch, which knot / topology / representation, which normalization, which sign, each calibration scale. Count a choice as free if the model's own framework admits an inequivalent alternative. |
| 3. Compare | If `N_free` ≥ `N_obs`, "zero free parameters" is illusory: the freedom is hidden inside the 'choices'. The agreement is then a fit, however it is described. |
| 4. Recompute independently | Recompute every quoted constant from its own definition, not from the number printed in the paper. Confirm the assembled formula actually yields the observable. |
| 5. Forced vs chosen | For each step the author calls "forced", find whether the geometry permits an inequivalent alternative. If it does, it was chosen. Genuine forcing means: fixed by something independent of the target being reproduced. |

If a model passes step 3 (few genuine inputs), survives step 4 (constants recompute independently), and step 5 shows the key choices are forced by structure independent of the data, that is a real result and a strong ✅. If it fails any step, document precisely where the freedom hides, that is a valuable ⚠️ or ❌, exactly the platform's product.

---

## 5. Self-testing with rigor

### 5.1 The red-flag checklist (failure modes to catch in your own work)

These are the patterns that most often distinguish an over-claimed framework from a sound one. Run this on yourself before a maintainer or a hostile reader does.

| Red flag | Why it matters | Self-check |
| --- | --- | --- |
| Everything matches to ~0σ with "zero free parameters" | A framework that reproduces *every* observable to the central value, with no residuals anywhere, is far more likely over-fit than uniquely correct. Real first-principles theories leave residuals. | Are there *any* honest residuals or tensions? If not, suspect hidden tuning. |
| Knobs dressed as structure | Sector-specific choices (a framing number, a knot assignment, a normalization) each given a separate post-hoc justification that happens to land on the answer. | Could a reader have derived each choice *before* seeing the target? List them and check. |
| Reproducing a known coincidence | Recovering a historically famous numerical coincidence (and presenting it as derivation) without adding new, independent content. | What does your derivation add beyond the coincidence? Is the coincidence a check or the load-bearing step? |
| Rhetorical certainty | Near-unity Bayesian posteriors, claims of being the unique possible theory, appeals to authority or destiny. Reviewers read these as overreach, not evidence. | Remove the rhetoric. Does the case still stand on the numbers alone? |
| Unrefereed / fringe sourcing | Leaning on preprint-commons or non-refereed sources as support. | Are your load-bearing citations to refereed, checkable results? |
| Theorems that are identifications | Physical assignments written in theorem-like language, blurring "proved" with "interpreted". | Label every statement: axiom, theorem (logical deduction), definition, or physical identification. |
| No falsifier | A model that cannot be wrong cannot be scored. | Section 1.4. If you cannot fill it in, the model is not yet testable here. |

### 5.2 The hostile cold-reader peer-review pass

Before submitting, put the work through a reader who *wants it to be wrong* and has not been softened by your narrative. If you cannot find such a person, simulate one (Section 6). The protocol:

| Step | What the hostile reader does |
| --- | --- |
| Strip the prose | Discard the motivation and keep only four lists: what is input, what is derived, what is predicted, what would falsify it. |
| Re-derive one headline number cold | Pick the most impressive result and recompute it from scratch, using only the stated constants and definitions, never the paper's intermediate values. |
| Attack the forced steps | For every "uniquely forced" claim, search for an inequivalent alternative the framework also permits. One counterexample downgrades "forced" to "chosen". |
| Count the bits | Apply Section 4. Does the structural freedom exceed the information content of the data reproduced? |
| Check the citations | Verify each cited theorem is used as stated and actually supports the step it is attached to. |
| Demand a residual | If nothing disagrees with experiment anywhere, treat that as evidence of over-fit until shown otherwise. |

A model that survives a genuine hostile pass is ready. One that has only been read by sympathetic readers is not.

---

## 6. Using an AI agent to do this

These tasks (independent reproduction, parameter counting, adversarial review) are well suited to an AI coding agent, with one firm rule: **the agent must show its work, the script and the numbers, never a verdict alone.** Language models will happily assert agreement that does not exist, so every claim it makes must be backed by a runnable artifact you can re-run.

Useful agent roles (run them as separate, non-colluding passes):

| Agent role | Prompt it to | Guardrail |
| --- | --- | --- |
| Reproducer | Implement the stated formula from the stated constants and report the output number, with the script. | It must print the script and the value; you re-run it. |
| Independent recomputer | Recompute each quoted mathematical constant (eigenvalue, torsion, zeta value, volume) *from its own definition*, not from the paper's printed value. | Forbid it from reading the paper's number for that constant first. |
| Red-team / adversary | Argue, as hard as it can, that the model is over-fit. Find the hidden free parameters and the chosen-not-forced steps. | Reward finding problems, not confirming the result. |
| Citation checker | Verify each load-bearing citation says what the model claims it says. | Require quotes / locations, not paraphrase. |
| Parameter counter | Execute Section 4 explicitly: list `N_obs`, enumerate every `N_free` choice, and compare. | Make it justify each item it counts or refuses to count. |

A practical pattern: run the reproducer and the independent recomputer first (do the numbers even hold?), then the red-team and parameter counter (are they predictions or fits?), then synthesize. Disagreement between the reproducer and the recomputer is itself a finding. Treat a unanimous "all confirmed" from a single agent with suspicion, that is what the separate adversarial pass is for.

---

## See also

| Doc | For |
| --- | --- |
| [`MODELS.md`](MODELS.md) | The comparison table, the shared criteria, the validation legend |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Canonical setup, fork/branch/PR flow, DCO sign-off |
| [`SYS_ARCH.md`](SYS_ARCH.md) | Repository structure and tech stack |
| [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | Community expectations |
