# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenWave is an open-source subatomic wave simulator for exploring fundamental physics through **classical field theory enriched with topology and nonlinearity** — the scientific tradition of de Broglie–Bohm pilot waves, wave structure of matter, and modern topological-soliton models. The platform tests whether particles and forces can emerge from deterministic field equations rather than being postulated.

The simulator runs multiple candidate Lagrangian frameworks (scalar-field, vector-field) in a shared numerical engine, plus a granule-motion model for educational visualization. GPU acceleration uses Taichi Lang.

### Project Goals

OpenWave investigates, in one integrated simulator, four primary domains: **matter** (particle emergence from topological defects + wave dynamics), **forces** (electric, strong, magnetic, gravitational from one classical-field framework), **electromagnetic waves**, and **heat** (whether wave-field degrees of freedom contribute to thermal physics). Each domain has concrete pass/fail criteria applied uniformly across candidate models.

### What is OpenWave?

| Reference | Purpose |
| --- | --- |
| `README.md` | Full description, scope, scientific position, installation |
| `WELCOME.md` | Quick intro |
| `SYS_ARCH.md` | Module structure and system architecture |

### Theoretical Advisors and Candidate Frameworks

| Contributor | Framework | OpenWave Model |
| --- | --- | --- |
| Jeff Yee | Energy Wave Theory (EWT) | M3 |
| Dr. Jarek Duda | Liquid Crystal Particle Analogs (arxiv 2108.07896, 2501.04036) | M5 |
| Dr. Robert Close | Classical elastic-solid / "Equation of Everything" | M5 (shared) |
| Manfried Faber | LdG regularization (Universe 11/2025/113) | M5.6 baseline |
| Dr. Paul Werbos | Ouroboros chaoiton Lagrangian | M6 |

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
| `openwave/xperiments/m5_liquid_crystal/` | **Active** — Duda LCB topological-defect model |
| `openwave/xperiments/m6_ouroboros/` | Werbos chaoiton Lagrangian |
| `openwave/common/`, `i_o/`, `validations/`, `video_export/` | Shared utilities, rendering, physics-invariant tests |

Refer to `README.md` and `SYS_ARCH.md` for the full Modules Structure and Objects Map.

### Scientific Source Material

Each model directory under `openwave/xperiments/` contains a `/research` subfolder with active research notes, plus a `theory/` folder (at the model root) holding the foundational theorist papers for that model (e.g. `m5_liquid_crystal/theory/liquid_crystal_model.pdf`, `Equation-of-Everything.pdf`, `faber_universe_2025.pdf`).

Note: the legacy top-level `scientific_source/` folder was retired 2026-05-18 — papers now live per-model.

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
| [Markdown Style Guide](/dev_docs/MARKDOWN_STYLE_GUIDE.md) | All `.md` files |
| [Coding Standards](/dev_docs/CODING_STANDARDS.md) | Python code |
| [Performance Guidelines](/dev_docs/PERFORMANCE_GUIDELINES.md) | Optimization |
| [Loop Optimization](/dev_docs/LOOP_OPTIMIZATION.md) | Critical loops |
| [Coordinate System](/dev_docs/COORDINATE_SYSTEM.md) | Spatial conventions |
| [Floating Point Precision](/dev_docs/FLOATING_POINT_PRECISION.md) | Numerical precision rules |
| [Scaling Factor](/dev_docs/SCALING_FACTOR.md) | Physics unit scaling |
| [Version Management](/dev_docs/VERSION_MANAGEMENT.md) | Release versioning |
| [Wave Diagnostics](/dev_docs/WAVE_DIAGNOSTICS.md) | Validation diagnostics |

### Important: Markdown Linting Requirements

When editing any `.md` files, ALWAYS ensure compliance with markdown linting rules:

- Add blank lines around headings (before and after)
- Add blank lines around lists (before and after)
- Add blank lines around code blocks (before and after)
- Use consistent ordered list numbering (1, 1, 1 style)
- Check for proper spacing and formatting

Run linting checks after editing to catch any issues before committing.
