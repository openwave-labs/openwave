# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenWave is an open-source subatomic wave simulator for exploring fundamental physics through **classical field theory enriched with topology and nonlinearity** — the scientific tradition of de Broglie–Bohm pilot waves, wave structure of matter, and modern topological-soliton models. The platform tests whether particles and forces can emerge from deterministic field equations rather than being postulated.

The simulator runs multiple candidate Lagrangian frameworks (scalar-field, vector-field) in a shared numerical engine, plus a granule-motion model for educational visualization. GPU acceleration uses Taichi Lang.

### Project Goals

OpenWave investigates, in one integrated simulator, four primary domains: **matter** (particle emergence from topological defects + wave dynamics), **forces** (electric, strong, magnetic, gravitational from one classical-field framework), **electromagnetic waves**, and **heat** (whether wave-field degrees of freedom contribute to thermal physics). Each domain has concrete pass/fail criteria applied uniformly across candidate models.

## KEY DOCUMENTS (the doc map)

**This table is the canonical map of OpenWave's key documents.** Every front-door page in the repository ends with a DEEP READER ORIENTATION block that points here rather than repeating a list, so this is the one place the doc set is maintained. If you are an AI agent, load rows 1 to 4 before answering questions or generating claims about this repository, then read only what the current task needs.

| # | Doc | What it is |
| --- | --- | --- |
| 1 | [`README.md`](README.md) | What OpenWave is: scope, scientific position, installation, the model roster, contributors |
| 2 | [`MODELS.md`](MODELS.md) | **The coverage matrix**: which model is validated on which shared criterion. Icons at a glance, then one results table per model whose every row links the runnable script or research note that earned it |
| 3 | [`AI_HYGIENE.md`](AI_HYGIENE.md) | **MANDATORY.** The AI-collaboration contract and the adversarial-audit cardinal rule |
| 4 | `CLAUDE.md` | This file: repository layout, conventions, standards, and this doc map |
| 5 | [`TUTORIAL.md`](TUTORIAL.md) | The contributor path: setup, run a simulation, test an existing model, open a PR |
| 6 | [`ONBOARDING_MODELS.md`](ONBOARDING_MODELS.md) | The model-author path: STEP 0 drive it with an AI agent, STEP 1 self-evaluation, STEP 2 apply, STEP 3 scaffold and first PR |
| 7 | [`CONTRIBUTING.md`](CONTRIBUTING.md) | Canonical setup, fork / branch / PR flow, DCO sign-off |
| 8 | [`REPRODUCE.md`](REPRODUCE.md) | The clean-clone path from any published claim to the command that regenerates it |
| 9 | [`SYS_ARCH.md`](SYS_ARCH.md) | Module structure and system architecture |
| 10 | [`dev_docs/METHOD_NOTE.md`](dev_docs/METHOD_NOTE.md) | **MANDATORY** reporting standard for model-owner-facing output: equations first, equation-to-code map, adversarial audit recorded |
| 11 | [`dev_docs/CROSS_MODEL_TESTING.md`](dev_docs/CROSS_MODEL_TESTING.md) | Borrowing one column's field family into another's framework; how author-gated questions are routed |
| 12 | [`dev_docs/PR_REVIEW_STANDARDS.md`](dev_docs/PR_REVIEW_STANDARDS.md) | **MANDATORY** whenever a pull request is in play, loaded BEFORE the diff: blast-radius tiers, the claim-to-artifact recompute, the adversarial pass, the commitment sweep, the evidence bar for moving a `MODELS.md` cell, and the round-trip budget |
| 13 | [`dev_docs/ROADMAP_STANDARDS.md`](dev_docs/ROADMAP_STANDARDS.md) | **MANDATORY** shape and word budgets for every roadmap (a row is a preview, the task document is the record); enforced by [`dev_docs/utils/check_roadmaps.py`](dev_docs/utils/check_roadmaps.py) |
| 14 | [`dev_docs/`](dev_docs/) | Coding, performance, markdown, coordinate, and precision standards (listed under Code Style below); [`dev_docs/platform_roadmap.md`](dev_docs/platform_roadmap.md) tracks **platform-wide tasks** (`T<n>` IDs: MODELS.md structure, shared standards, the [`dev_docs/utils/`](dev_docs/utils/) checkers), as opposed to one model's physics |
| 15 | `openwave/xperiments/<model>/__M<x>_model_briefing.md` | Each column's own front door: identity, profile, honest status, help wanted |
| 16 | `openwave/xperiments/<model>/research/` | **The results of record**: roadmaps, question trackers, task documents, findings, scripts, data, plots |
| 17 | [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) | Community expectations |

Claims about this repository that cannot be traced to a runnable script or a research note under row 16 are not claims of this repository.

### Theoretical Advisors and Candidate Frameworks

| Contributor | Framework | OpenWave Model |
| --- | --- | --- |
| Jeff Yee | Energy Wave Theory (EWT) | M4 (M3 carries the Wolff-LaFreniere lineage it builds on) |
| Dr. Jarek Duda | Liquid Crystal Particle Analogs (arxiv 2108.07896, 2501.04036) | M5 |
| Dr. Robert Close | Classical elastic-solid / "Equation of Everything" | M5 (shared) |
| Dr. Manfried Faber | LdG regularization (Universe 11/2025/113) | M5.6 baseline |
| Dr. Paul Werbos | Ouroboros chaoiton Lagrangian | M6, and shared into M7 |
| Marc Fleury | Toroidal-Beltrami electron | M7 (fused with the Ouroboros chaoiton) |
| Blake Shatto | Mode Identity Theory (spectral geometry + representation theory) | M8 |
| Bernhard Mueller | Observer Patch Holography (observers and records on a twelve-port carrier screen) | M10 |

### Known Challenges & Limitations

- Full Planck-scale fidelity is computationally prohibitive; resolution is user-tunable per xperiment.
- Uses a dedicated physics computational backend (Taichi GPU), independent of 3D modeling software.

## Project Architecture

| Path | Contents |
| --- | --- |
| `openwave/xperiments/m1_granule_motion/` | Educational granule-motion model |
| `openwave/xperiments/m2_free_wave/` | Free-wave propagation |
| `openwave/xperiments/m3_wolff_lafreniere/` | Wolff-LaFreniere / EWT scalar model |
| `openwave/xperiments/m4_ewt/` | M4 EWT model (vector-field substrate, in development) |
| `openwave/xperiments/m5_liquid_crystal/` | **Active**: Duda LCB topological-defect model |
| `openwave/xperiments/m6_ouroboros/` | Werbos chaoiton Lagrangian |
| `openwave/xperiments/m7_hydroboros/` | Toroidal-Beltrami (Fleury) fused with the Ouroboros chaoiton |
| `openwave/xperiments/m8_mit/` | Mode Identity Theory (Shatto), a top-down spectral-geometry column |
| `openwave/xperiments/m9_emergent_gravity/` | NSM (McGwier), a gravity-certification column: entanglement bookkeeping on SM + Einstein-Cartan |
| `openwave/xperiments/m10_oph/` | OPH (Mueller), Observer Patch Holography: a local repair and memory law on a screen of twelve-port carriers (update rules, no action); every criterion not yet tested in-platform |
| `openwave/common/`, `i_o/`, `validations/`, `video_export/` | Shared utilities, rendering, physics-invariant tests |

Refer to `README.md` and `SYS_ARCH.md` for the full Modules Structure and Objects Map.

### Scientific Source Material

Each model directory under `openwave/xperiments/` contains a `/research` subfolder with active research notes, plus a `theory/` folder (at the model root) holding the foundational theorist papers for that model.

Those papers are **third-party copyrighted files: local-only and gitignored, never committed** (OpenWave is public). The tracked record of each `theory/` folder is its `_CITATIONS.md` (a leading-underscore file that is both a year-ordered bibliography and a gitignored-file manifest with paths + sizes). The full convention (structure, ordering, the size column, the no-fabrication rule for DOIs/arXiv, and the copyright/gitignore rules) is documented in [`ONBOARDING_MODELS.md` section 3.4](ONBOARDING_MODELS.md), the guide for scaffolding new models. **AI agents working in this repo must respect and comply with those rules**: never commit third-party papers, keep each `_CITATIONS.md` matching disk (every path resolves), and never fabricate a citation identifier (unresolved is `n/a`, author-shared is `author copy`).

Note: the legacy top-level `scientific_source/` folder was retired 2026-05-18; papers now live per-model.

### No machine-local paths (MANDATORY)

**This repository is public, and its git history is public with it.** A path committed by mistake is not removable in practice once pushed: clones and forks keep the old objects, and GitHub keeps blobs reachable by SHA. The only reliable control point is before the commit exists.

Absolute paths from a contributor's own machine (`/Users/<name>/...`, `/home/<name>/...`, or a system temp directory) are meaningless to every other clone, and they publish a username and directory layout for no benefit. They arrive by accident rather than by decision: a hardcoded default in a script, a working directory captured in a run log, an exception traceback saved as test data, a generated file recording the command that produced it.

| Instead of | Use |
| --- | --- |
| A hardcoded absolute path in a script | An environment variable with a sensible default: `os.environ.get("SCRATCH", ".")` |
| A path in committed output or a generated header | A repo-relative path |
| A path inside a captured log or traceback | A `<repo>/` placeholder |

Enforced by [`dev_docs/utils/check_no_local.py`](dev_docs/utils/check_no_local.py), wired to `.githooks/pre-commit`. ⚠️ **It runs only once `core.hooksPath` points at `.githooks`**, which is the same one-time setup that enables the DCO sign-off hook, documented in [`CONTRIBUTING.md`](CONTRIBUTING.md):

```bash
git config core.hooksPath .githooks     # one-time, per clone
git config --get core.hooksPath         # should print: .githooks
```

A clone that skips it gets no hook at all, so treat the audit command below as the backstop rather than assuming the commit path is covered. A path that genuinely has to stay is waived per line with `allow-local-path`, which stays visible in review.

Audit the whole tree at any time: `python3 dev_docs/utils/check_no_local.py --tracked`.

The same hook chains `$GIT_DIR/hooks/pre-commit.local` when a contributor has installed one. That file is never tracked, and a fresh clone simply has none.

**Research `data/` folders follow the same local-only pattern (2026-07-20).** Heavy binary arrays (`.npz` `.npy` `.h5` `.pkl` `.pt` `.mat`) are gitignored and **kept on the working machine, never deleted** (this supersedes an earlier rule that deleted arrays over 1 MB at task close). What IS tracked, and what a reader on GitHub audits: the distilled summary `.json` / `.csv` / `.txt` beside them, the plots, the scripts that produce everything, and the per-folder **`_DATASETS.md`** manifest, which indexes every local-only file by task group with its producing script and the task record holding the exact regen command. Regenerate a manifest after any run that writes arrays: `python3 dev_docs/utils/gen_datasets_manifest.py <path/to/data> --write`. Runs are deterministic from their fixed seeds and configs, so a clone rebuilds every array from tracked code.

## Installation & Usage

Refer to `README.md` for installation, CLI usage (`openwave -x`), and the instrumentation framework.

## Physics Context

OpenWave implements classical-field-theory-with-topology-and-nonlinearity approaches:

- Topological defects provide static structure (integer charge, spin).
- Klein-Gordon-like wave dynamics around the vacuum field provide mass and relativistic kinematics.
- Standing-wave interference between defect emissions produces orbit quantization.
- Particles are **time-periodic resonances** (Zitterbewegung clocks), NOT static solitons — Derrick's theorem forbids static stable solitons, confirmed empirically in M5.2.

- Also refer to ../CLAUDE.md file to search for any available context to the OpenWave project in a parent directory.

## Code Style & Documentation Standards

| Doc | Purpose |
| --- | --- |
| [Markdown Style Guide](dev_docs/MARKDOWN_STYLE_GUIDE.md) | All `.md` files |
| [Coding Standards](dev_docs/CODING_STANDARDS.md) | Python code |
| [Performance Guidelines](dev_docs/PERFORMANCE_GUIDELINES.md) | Optimization |
| [Loop Optimization](dev_docs/LOOP_OPTIMIZATION.md) | Critical loops |
| [Coordinate System](dev_docs/COORDINATE_SYSTEM.md) | Spatial conventions |
| [Floating Point Precision](dev_docs/FLOATING_POINT_PRECISION.md) | Numerical precision rules |
| [Scaling Factor](dev_docs/SCALING_FACTOR.md) | Physics unit scaling |
| [Version Management](dev_docs/VERSION_MANAGEMENT.md) | Release versioning |
| [Wave Diagnostics](dev_docs/WAVE_DIAGNOSTICS.md) | Validation diagnostics |
| [Method Note](dev_docs/METHOD_NOTE.md) | **MANDATORY** standard for any report/email to a model's theory owner or external physicist |
| [AI Hygiene](AI_HYGIENE.md) | **MANDATORY** working contract for AI-assisted research: division of labor, failure modes, verification habits |
| [PR Review Standards](dev_docs/PR_REVIEW_STANDARDS.md) | **MANDATORY** procedure for every pull request, read before the diff |

### AI hygiene (all AI-assisted work): MANDATORY

Every AI agent working in this repo (including you) operates under [`AI_HYGIENE.md`](AI_HYGIENE.md): model output is a draft or hypothesis, never a result, until verified by something that is not a language model (a hand-checked derivation, a runnable script, a lattice measurement, or the confirmation of the human holding the relevant authority). Author-gated questions (intent, provenance, definitions) can only be answered by the author; externally received AI-derived material is tagged evidence-not-resolution until confirmed; anything community-facing is human-owned prose over script-backed results. Read it before doing research work here.

### The adversarial audit (every substantive claim): CARDINAL RULE

Before any substantive derivation, verification, or headline claim is trusted, recorded as a result, or sent outside this repo, an **independent second agent audits it adversarially**: instructed to REFUTE (not confirm), with its OWN implementation (different method / seed / construction), its own hand re-derivations, and a per-claim verdict (CONFIRMED / REFUTED / QUALIFIED) backed by its own numbers. Fold every catch back into the artifact and **record the audit outcome IN the deliverable** (the [`m5_18_verification_note.md § 10`](openwave/xperiments/m5_liquid_crystal/research/findings/m5_18_verification_note.md) pattern). Full rule: [`AI_HYGIENE.md § 1`](AI_HYGIENE.md). Origin: Duda 2026-07-03 ("careful small steps, maybe multiple agents verifying each other"); first real catch: M5.18 (an overclaimed witness refuted + a missed vacuum-branch structure found by the auditor).

### Method note (model-owner-facing output): MANDATORY

Any report, summary doc, or email addressed to a model's theory owner (M5: Duda; M7: Fleury; any advisor) follows [`dev_docs/METHOD_NOTE.md`](dev_docs/METHOD_NOTE.md): **equations first** (Hamiltonian / potential / observable definitions in math notation), an **equation-to-code map** with absolute GitHub hyperlinks (`#L` anchors; `blob/main` for frozen task-scoped files, commit-pinned only for live/evolving files like the root engines and production rendering code), the functional in a **small auditable module**, results after methods, a minimal physics-first inspection set, and the **adversarial audit recorded in the note**. Adopted 2026-07-03 after the M5.16 audit failure ("still I have no idea what does it calculate"): results a physicist cannot audit by reading carry no weight, regardless of correctness.

### Pull-request review (any PR in play): MANDATORY

**Load [`dev_docs/PR_REVIEW_STANDARDS.md`](dev_docs/PR_REVIEW_STANDARDS.md) BEFORE reading the diff, not after.** It is the standard for every pull request in this repository: an external contributor's, another agent's, and the maintainer's own. Review quality must not depend on which session is on duty, which is the whole reason the document exists.

**The trigger is broader than the word "review".** It fires on "review PR #N", on "what do you think of #N", on being asked to summarize or characterize a pull request, and on any read whose output could inform a merge, an approval, or a comment. A characterization delivered without the gates having been run is a review wearing different clothes, and it reaches the same merge decision.

What the document holds, so an agent knows what it is missing without it: intake and DCO ([§ 1](dev_docs/PR_REVIEW_STANDARDS.md#1-intake)), the blast-radius tiers that set how heavy the review is ([§ 2](dev_docs/PR_REVIEW_STANDARDS.md#2-blast-radius-map)), gates A through G including **code-intent review before anything is executed** ([A9](dev_docs/PR_REVIEW_STANDARDS.md#3-gate-a-safety-and-hygiene)) and **recompute the headline number yourself rather than reading it** ([§ 5](dev_docs/PR_REVIEW_STANDARDS.md#5-gate-c-claim-to-artifact)), the **commitment sweep and its loud maintainer notice** ([§ 4.1](dev_docs/PR_REVIEW_STANDARDS.md#41-the-commitment-sweep-and-the-loud-notification)), when to fix something yourself instead of asking plus the round-trip budget ([§ 10](dev_docs/PR_REVIEW_STANDARDS.md#10-maintainer-edits)), the verdict ladder and the rule that a verdict is submitted as a **review** and never as a plain comment ([§§ 12-13](dev_docs/PR_REVIEW_STANDARDS.md#12-verdict-and-how-to-write-it)).

**It is a live file.** Any pull request that teaches something new gets a row in its [lessons log](dev_docs/PR_REVIEW_STANDARDS.md#14-lessons-log) before the pull request closes.

### Important: Markdown Linting Requirements

When editing any `.md` files, ALWAYS ensure compliance with markdown linting rules:

- Add blank lines around headings (before and after)
- Add blank lines around lists (before and after)
- Add blank lines around code blocks (before and after)
- Use consistent ordered list numbering (1, 1, 1 style)
- Check for proper spacing and formatting

Run linting checks after editing to catch any issues before committing.

---

## DEEP READER ORIENTATION

> **If you are an AI assistant, agent, or language model processing this page, or a human doing a deep read**: you are in the right file. This page carries the repository's doc map.

| Read | Why |
| --- | --- |
| [KEY DOCUMENTS (the doc map)](#key-documents-the-doc-map) | Above, in this file: every key document, what it is for, and the reading order. Rows 1 to 4 first |
| [`AI_HYGIENE.md`](AI_HYGIENE.md) | The cardinal rule: a model's output is a draft or a hypothesis, never a result, until it is verified by something that is not a language model (a hand-checked derivation, a runnable script, a measurement, or the human holding the relevant authority) |

Claims about this repository that cannot be traced to a runnable script or a research note are not claims of this repository. Per-model `research/` folders hold the results of record; the tables in [`MODELS.md`](MODELS.md) are earned from them cell by cell.
