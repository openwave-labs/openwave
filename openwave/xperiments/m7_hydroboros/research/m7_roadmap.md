# M7 / HYDROBOROS, ROADMAP

> Local task roadmap for the **M7 (HydroBoros)** model, replicating the M5 local-roadmap structure (2026-07-01). M7 has **no GitHub issues**; tasks are tracked here. **taskID = M7.N**. Full planning + findings live in each [`tasks/m7_N_*.md`](tasks/) detail doc; scripts / data / plots live in [`scripts/`](scripts/) · [`data/`](data/) · [`plots/`](plots/), all named `m7_N_*.{py,npz,png}`. Ordering is the build sequence (M7.0 → M7.23, Phases 1 → 5; task IDs are stable, table order = build order); the **top of Backlog is the next task**.
>
> The program runs in five **phases** (renumbered 2026-07-04, extended after the Phase-1-review call). **Phase 1** builds the electron and drafts the coverage column (M7.0-M7.7 ✅), then extends through the author-collaboration track (M7.8 → M7.9 → M7.10, with M7.11-M7.14 reserved for the pure-Maxwell/Beltrami orbit work). **Phase 2** expands across the forces (M7.15-M7.20); **Phase 3** publishes the column (M7.21 ✅); **Phase 4** the composites (M7.22); **Phase 5** production rendering (M7.23). Each M7.N is gated against a KNOWN result. Migrated from [`m7_background.md`](m7_background.md) 2026-07-01.

## IN PROGRESS

| TaskID | Task title | Description | Validation gate |
| --- | --- | --- | --- |

## BACKLOG

> **M7 (2026-07-07, post-M7.10)** until the author returns from the conference and reviews the [walkthrough § 7](tasks/m7_phase1_walkthrough.md) package; an adversarial pass may re-scope the band below. Sequencing rationale lives in the task docs themselves ([M7.11 § 0](tasks/m7_11_variable_lambda.md) carries the full post-M7.10 analysis).

| TaskID | Task title | Description | Validation gate |
| --- | --- | --- | --- |
| 🚧 [M7.11](tasks/m7_11_variable_lambda.md) | The variable-λ vacuum-cure check (Q14) | The two computable cure candidates from the M7.10 constraint (the bilinear carries BOTH the tachyon and the confinement): (a) linearize the vacuum WITH the scalar/Gauss sector; (b) the author's variable-λ mass profile as a fixed background, bound interior + massless healthy exterior. Plus the first ISOLATED-orbit hunt, non-degenerate once the flow is inhomogeneous | exterior vacuum noise does not grow while a localized minimizer survives ≥ 10 T, or the failure documented; genuine `find_cycle` period recovery. Unblocks M7.18 |

## PHASE 1 EXTENSION, reserved band (M7.11-M7.14): the pure-Maxwell / Beltrami orbit track

**M7.10 is DELIVERED** (2026-07-07: the no-Lagrangian test, row in the author-collaboration DONE table below) and **M7.11 is now defined** (the variable-λ vacuum-cure check, § BACKLOG above, top = next); M7.12-M7.14 stay reserved for the Fleury-track work as it firms up (the author's re-entry is post-August): plain **linear Maxwell evolution** (the `m_J = 0, f = 0` sector of the existing engine, bootstrapped by M7.10), Beltrami / variable-λ ansatz as seeds, a defined notion of a stable orbit in the field flow (the ChaosBook methodology from M7.9), and the static→dynamic charge transition the author plans to lead. Two-track principle (2026-07-06): the Maxwell track runs alongside the harmonic/functional track that produced Phase 1, on the same lattice; measurements adjudicate. Honest boundary, stated up front: pure free-space linear Maxwell cannot hold a localized finite-energy state indefinitely (Nadirashvili; our Q11 dispersion work; M7.10 turns this into a measurement), so this track searches and verifies with the variable-λ profile as the localizer, and what sustains that profile (back-reaction) remains the open physics.

## PHASE 2, forces and the remaining particle sectors (M7.15-M7.20)

| TaskID | Task title | Description | Validation gate |
| --- | --- | --- | --- |
| 🚧 [M7.15](tasks/m7_15_magnetic.md) | magnetic force | the per-defect magnetic structure carried by the electron's clock (Coulomb already landed in M7.4/M7.6) | magnetic force from the clock's `Γ₀` (pure twist is EM-silent; the M5 mechanism) |
| 🚧 [M7.16](tasks/m7_16_gravity.md) | gravity | the time-axis boost of the field (the M5 4×4 route) | a GEM coupling that vanishes at zero boost; honest pass / fail (Ouroboros stops before gravity, so this is genuinely hard for M7) |
| 🚧 M7.17 | nuclear forces | strong = the 4th-order short-range roll-off + linking tension; weak = a topology-reconnection (defect-class transition) | running-coupling onset at the core; a reconnection channel; partial, mirroring M5 |
| 🚧 M7.18 | antimatter + annihilation | **pre-step = the Q14 vacuum-cure check (self-determine, 2026-07-05)**: linearize the vacuum WITH the scalar/Gauss sector (does it lift `det M(0) = −1`?) + test the variable-λ mass profile; then seed a soliton + anti-soliton (`Q → −Q`); evolve | Q14 answered or cured in-model (real-time runs blocked until then); charge ledger `±1 → 0`; rest energy released as outgoing waves; pair → vacuum |
| 🚧 [M7.19](tasks/m7_19_lepton_family.md) | the lepton + neutrino family | vary knot size / linking: charged = self-linked torus, neutrino = the lighter loop; carries the **Q11 localized-branch hunt** with the 2026-07-05 scan design: the decaying-channel condition `ω²(ω² + λ) < m_c⁴` is `ω < ω*` at canonical `(λ, m_c)` (localization and existence mutually exclusive there), so the scan moves **`(λ, m_c)`**, not `(ω, g)` ([`m7_13_q11_window.py`](scripts/m7_13_q11_window.py) pre-work) | the lepton mass family (μ, τ); light neutral neutrino loops; flavour-rotation mixing; a localized charged branch found or excluded over the `(λ, m_c)` plane |
| 🚧 [M7.20](tasks/m7_20_dark_matter.md) | dark matter | the **neutral** knot (helicity-only, zero net `∇·F`), inheriting M6's neutral chaoiton | a stable neutral soliton; sub-MeV mass à la M6's `m_χ = 0.460 MeV` |

## PHASE 4, the composites still untested in M5, grouped (M7.22)

Per the M5 prescription, the cells that remain 🚧 [not yet tested] in M5 itself are grouped into one later task, after the column is published. They depend on the electron + force primitives of Phases 1-2 already being in place.

| TaskID | Task title | Description | Validation gate |
| --- | --- | --- | --- |
| 🚧 M7.22 | composites + atomic structure | quark = fractional-charge string segment; baryon / meson = linked / twisted knots; atom = pilot-wave orbital quantization | Quarks · Baryons (p, n) · Mesons (π, K) · Orbital quantization |

## PHASE 5, production rendering upgrade (post-canonical)

After the electron is canonical (the M7.7 milestone), graduate the winning recipe from the research scripts to the production engine, exactly the path M5 followed.

| TaskID | Task title | Description | Validation gate |
| --- | --- | --- | --- |
| 🚧 M7.23 | graduate to the production engine + rendering | fold the canonical recipe (M7.7) into `medium.py` + `engine1_seeds` / `engine2_pde` / `engine3_observables` / `engine4_render` + `_launcher.py`; the Phase 2 + 4 observables feed the renderer as they land | HydroBoros runs via `openwave -x` with toroidal field-line / vorticity rendering; first-try reproducible |

```text
research/scripts/ data/ plots/   headless Taichi scripts + data + PNG diagnostics   (tasks M7.1-M7.22)
research/tasks/                  per-task detail docs  m7_N_*.md
        │  winning recipe →
medium.py                  the (A,J) / RS substrate definition
engine1_seeds.py           Bateman/Hopf + toroidal-Beltrami seeders
engine2_pde.py             the nonlinear PDE evolution + minimizer
engine3_observables.py     charge (∇·F), helicity, energy, spin, μ
engine4_render.py          toroidal field-line / vorticity rendering
_launcher.py               registers HydroBoros for `openwave -x`
```

Reference layout: [`../../m5_liquid_crystal/`](../../m5_liquid_crystal/) (`medium.py`, `engine1_seeds.py` … `_launcher.py`). Headless first (matplotlib PNG diagnostics in `research/plots/`); rendering graduates once the electron is canonical (the M7.7 milestone), and the Phase 2 + 4 coverage tasks feed their observables into the renderer as they land, identical to how M5 reached `_launcher.py`. No GUI / viz work before the physics is canonical.

## DONE

## PHASE 1, the electron and the M7 column (M7.0-M7.7) ✅ DONE 2026-07-04

M7.1-M7.3 are the decisive credibility gates (reproduce **both** parents from the same lattice code). M7.4 is the research core (the thing neither parent did). **M7.7 is the milestone: the column drafted; publication = M7.21 (Phase 3, ✅ 2026-07-18, now live in [`MODELS.md`](../../../../MODELS.md)).** Note: **Coulomb rides with the electron**, not as a later forces task: once the divergence charge `∇·F` exists (M7.4), its `1/r` field is immediate (Gauss's law), and the two-body `E(d) ~ 1/d` is confirmed at M7.6, exactly as M5 got Coulomb in its first `m5_1` milestone.

**Method frame (2026-07-02 plan refactor, [`m7_background.md § 5`](m7_background.md)).** Both parents are time-periodic, so the whole Phase 1 runs in the **time-harmonic (fixed-ω) frame**: the minimizer works on the period-averaged functional `E_ω` at fixed ω, and the de Broglie clock is in the soliton from M7.1 (M7.5 validates it in real time). Stabilization = **helicity anti-collapse + Ouroboros confinement anti-expansion** (the drafted Faddeev-Niemi term is inert on Beltrami fields and demoted to an optional Q2 experiment). **Sequencing note:** M7.2 is pure quadrature (no minimizer), so it can run in parallel with the M7.1 minimizer build and land the first trust-builder early; M7.3's verbatim-ODE pre-gate must pass before any 3D relaxation run is trusted.

**Prescription (Jarek Duda, 2026-06-29 models-of-particles thread; Ouroboros [#247](https://github.com/openwave-labs/openwave/issues/247)).** Standing demand of any particle-model framework: *specify the field configuration of each particle* (electron, neutrino, photon, mesons, baryons), and *does it use topological vortices?* HydroBoros answers head-on, the electron's field configuration **is** the self-linked toroidal Beltrami vortex ([`m7_background.md § 0`](m7_background.md)), a topological vortex. The author's two concrete tests map onto the plan exactly:

| Test | What it prescribes | M7 task |
| --- | --- | --- |
| **Coulomb** | assume **two** charge field-configurations, read the `1/r` interaction energy `E(d)` | M7.4 (single-charge `1/r`) + M7.6 (two-charge `E(d)~1/d`), the M5 `m5_1` way |
| **the clock** | the de Broglie frequency is the one that **minimizes the energy** | M7.5 (clock + stability), = the M5.8 energy-minimizing-clock mechanism |

The author's deeper point, that pinning the field configurations is the **precondition to passing OpenWave's tests** (not assuming them = "just hallucinations"), is precisely M7's premise: M7 specifies the config (the toroidal Beltrami seed) **and** earns it as the energy-minimizer of the functional ([`m7_background.md § 5`](m7_background.md)), then reads observables off the relaxed field. That sidesteps the Werbos-vs-Duda "assume vs predict" tension in the thread (M7 does both: a seeded config, relaxed to the true minimum).

| TaskID | Task title | Description | Validation gate |
| --- | --- | --- | --- |
| ✅ [M7.0](tasks/m7_0_bootstrap.md) | bootstrap | Collect the theory-source corpus (Fleury torus, Ouroboros/M6, Beltrami/Trkalian, Marc's evolving `electron_beltrami/` library, 66 docs as of 2026-07-05) and stand up the planning scaffolding (background, question-tracker, roadmap, the `research/` folder structure). | ✅ corpus consolidated (64 docs) in [`../theory/`](../theory/) + [manifest](../theory/_CITATIONS.md); plan + roadmap + tracker in place (2026-07-02) |
| ✅ [M7.1](tasks/m7_1_infra.md) | infra | A-primary doublet on a 3D lattice in the time-harmonic (fixed-ω) frame; helicity observable + fixed-helicity relaxation; Taichi-AD energy gradient; FIRE minimizer; six seeders. **Woltjer-Taylor**: random seed → constant-λ eigenfield, `λ → 2π/L` at `5.5e-6`, `E = λH`; AD vs complex-step `2.3e-15`; the M6 ledger `H/Q = 1.68897` reproduced by the seeder pipeline | ✅ ALL GATES PASS (2026-07-02, [findings](tasks/m7_1_infra.md)) |
| ✅ [M7.2](tasks/m7_2_fleury_torus.md) | reproduce Fleury on the lattice | Lattice quadrature of the FLDB toroidal ansatz under the pinned conventions contract; grid-convergence study; Bessel-envelope stretch. `Q/μ/L/U` to closed forms at `1.4e-4`, order ~2.5; printed solution reconstructed digit-for-digit; **Q10: corrected-convention `U = 0.958 m_ec²`** (was 0.795); the stretch exposed the mask's hidden surface charge (~18× bulk) | ✅ ALL GATES PASS (2026-07-02, [findings](tasks/m7_2_fleury_torus.md)) |
| ✅ [M7.3](tasks/m7_3_ouroboros_3d.md) | reproduce M6's electron in full 3D | Pre-gate: the 3D harmonic functional restricted to M6's ansatz reproduces the M6 ODE verbatim; then the M6 profile embedded as a 3D seed. Windowed `H/Q = 1.68889` vs ledger `1.68897` (dev **4.7e-5**); three discoveries: the charged ledger is WINDOWED (Q11), `0d_canonical § 2.2` is not an EL reduction (Q12), the M6 electron is a **3D saddle with focusing collapse** (Q13) | ✅ ALL GATES (2026-07-03, [findings](tasks/m7_3_ouroboros_3d.md)) |
| ✅ [M7.4](tasks/m7_4_charged_soliton.md) | the charged soliton + its Coulomb field (NEW physics) | Relax the full functional from multiple seeds at fixed ω + helicity; charge = the measured `∇·F`, never an imposed ansatz. **First stable finite-size 3D soliton**: a Taylor-dressed spheromak family, `E = 0.802\|H_A\|`, `H_cross = −0.390 H_A` universal across 3 seeds; helicity load-bearing (zero-helicity seeds evaporate); ⚠️ net-Gauss charge + Coulomb deferred (scalar-sector instability) | ✅ PRIMARY GATE PASS (2026-07-03, [findings](tasks/m7_4_charged_soliton.md)) |
| ✅ [M7.5](tasks/m7_5_clock_stability.md) | validate the clock (real-time) + stability | Minkowski leapfrog evolution of the relaxed harmonic states. Translation exact (`⟨E_real⟩ = E_ω` to 1.85e-14) and the reduction vindicated, BUT the vacuum is **unconditionally tachyonic** at long wavelength (`det M(0) = −1`, rate 0.785 vs 0.786 analytic, **no `β*` threshold: Q14 opened**); solitons exist only above **`ω* = 0.786`**, so the clock IS the stabilizer | ✅/❌ SPLIT, honest (2026-07-03, [findings](tasks/m7_5_clock_stability.md)) |
| ✅ [M7.6](tasks/m7_6_observables.md) | electron observables | Mass, spin, μ, KG and two-charge Coulomb `E(d)` on the M7.5 design inputs. **The rotating electron candidate** (blend_m1, E = 6.3246) is a clean **`j_z = 1` per-quantum wave (0.6%)**; **Coulomb landed** (Gauss 99.1%, two-charge `1/d` reference-matched with a measured 1.17 dressing); KG exact (`m_eff² = φ`); 🔶 μ charge-unit-blocked (scalar sector) | ✅ 6 of 7 gates (2026-07-03, [findings](tasks/m7_6_observables.md)) |
| ✅ [M7.7](tasks/m7_7_consolidation.md) | consolidate the M7 column (MILESTONE) | Canonical spec + small auditable physics module (METHOD_NOTE standard), the column, the units-contract decision table, both comms packages. [`m7_theory_canonical.md`](m7_theory_canonical.md) + `m7_functional.py` + `m7_7_canonical.py` passed **all gates first try at N = 64 AND N = 48, engine-vs-reference 1.4e-14**; the column drafted honestly (0✅/8⚠️/13🚧), published 2026-07-18<!-- count:historical --> | ✅ MILESTONE (2026-07-04, [findings](tasks/m7_7_consolidation.md)) |

## PHASE 1 EXTENSION, the author-collaboration track (M7.8 + M7.9 + M7.10 ✅: the track is COMPLETE, the walkthrough § 7 report ready for the author)

| TaskID | Task title | Description | Validation gate |
| --- | --- | --- | --- |
| ✅ [M7.8](tasks/m7_8_helicity_pair.md) | helicity-pair 3:1 test + the Phase 1 walkthrough | The amplitude-ratio ladder on the repaired CK/LG helicity pair at fixed `(Q_can, H_A)`, plus the [walkthrough](tasks/m7_phase1_walkthrough.md) and canonical-spec refresh #1. **The closure pair is NOT stationary at any seeded asymmetry**: the minus mode is expelled at all 5 rungs (`U₊/U₋` 104 → 1077 vs the predicted 3), so the **pair-asymmetry spin reads one quantum, not ℏ/2**. Audit CONFIRMED | ✅ DELIVERED (2026-07-07, [findings](tasks/m7_8_helicity_pair.md)) |
| ✅ [M7.9](tasks/m7_9_chaosbook.md) | ChaosBook study + canonical-exercise benchmark | The self-test from the Phase-1-review call plus the orbit-hunting toolkit for the Maxwell track, every value transcribed and cited per the [hygiene ledger](tasks/m7_9_chaosbook_index.md). Pre-book analytic gates 4/4, **5/5 benchmarks green**, headlined by the **full exercise 16.9 Rössler cycle table, 14/14 periodic orbits to length 7** (worst dev 2.0e-6). Audit CONFIRMED by an independent route. Toolkit: [`m7_9_orbits.py`](scripts/m7_9_orbits.py) | ✅ DELIVERED (2026-07-07, [findings](tasks/m7_9_chaosbook.md)) |
| ✅ [M7.10](tasks/m7_10_pure_maxwell.md) | Pure-Maxwell sector: the no-Lagrangian test | Coupling terms switched off, leaving plain Maxwell evolution; four experiments (discrete ABC eigenmode + Trkalian cavity, free-space evaporation, electron destruction, the coupling ladder). All four green: Theorem 2 at the integrator floor (clock to 5.8e-14), orbits non-isolated BY PHYSICS, and **Q14 sealed**: rates on `0.786√ε` to 0.1-0.3% on all 7 rungs, so the tachyon and the confinement have ONE source, the bilinear | ✅ DELIVERED (2026-07-07, [findings](tasks/m7_10_pure_maxwell.md)); audit CONFIRMED |

## PHASE 3, the MODELS.md column (the coverage scoreboard, M7.21) ✅ DONE 2026-07-18

M7.21 published the drafted 21-cell HydroBoros column into repo-root [`MODELS.md`](../../../../MODELS.md), reordered per the ✅+⚠️ rule (ties break to more ✅, then fewer ❌): the model columns are now **M5 · M6 · M7 · M4** (M7 ties M4 on 8 and wins on 0 ❌ vs 3, so it lands 3rd). Icons entered unchanged (0✅ / 8⚠️ / 13🚧)<!-- count:historical -->; every filled cell is script-backed with a repo-root-relative link; the composites cells stay honestly 🚧 until Phase 4. The GitHub issue + PR governance flow stays the maintainer's to run. The M7.7-era `preview_models.md` staging draft was removed at publication to avoid duplication.

| TaskID | Task title | Description | Validation gate |
| --- | --- | --- | --- |
| ✅ [M7.21](tasks/m7_21_models_column.md) | complete + govern the MODELS.md column | published the 21-cell HydroBoros column into repo-root [`MODELS.md`](../../../../MODELS.md), reordered M5·M6·M7·M4 per the ✅+⚠️ rule; This published the results-so-far honestly (0✅/8⚠️/13🚧)<!-- count:historical --> | ✅ DONE (2026-07-18, [record](tasks/m7_21_models_column.md)): all 21 cells carry an honest, script-backed icon; the M7 column is live in MODELS.md as the 3rd column; doc-checker clean, no cross-repo content |

The M5 prescription that governed every cell (the column is now published):

| Rule | What it means for M7 |
| --- | --- |
| Every cell is **script-backed** | each filled cell links to a runnable `scripts/m7_N_*.py` script (or a research note), reproducible by anyone |
| **Honest status icons** | ✅ validated in-platform · ⚠️ partial / caveats · ❌ tested + failed · 🔶 in progress · 🚧 planned |
| **Negatives are results** | a divergence-ful field that refuses to hold a knot (Q5) lands as a documented ❌, not a silence |
| **The column is earned, cell by cell** | M7 cells start 🚧, go 🔶 during a task, settle to ✅ / ⚠️ / ❌ when the run lands |
| **New-model governance** | M7 is a new column: open a discussion first so a maintainer adds it, then a script-backed PR + DCO + light review, per [`MODELS.md`](../../../../MODELS.md) § Contributing, [`ONBOARDING_MODELS.md`](../../../../ONBOARDING_MODELS.md), [`CONTRIBUTING.md`](../../../../CONTRIBUTING.md) |

Each task filled specific cells, the running scoreboard of the program:

| Task | MODELS.md cells targeted | Backing script |
| --- | --- | --- |
| **M7.1** | (infrastructure, no cell) | `scripts/m7_1_*.py` |
| **M7.2** | Charge quantization · Electron rest energy · Magnetic moment μ + spin J · EM waves (Maxwell) · de Broglie clock | `scripts/m7_2_fleury_torus.py` |
| **M7.3** | Electron rest energy (`H/Q = 1.6969` in full 3D) · Particle stability | `scripts/m7_3_ouroboros_3d.py` |
| **M7.4** | Particle stability (Derrick escape) · Charge quantization (helicity / linking + divergence) · Electric force (Coulomb 1/r, single-charge field) | `scripts/m7_4_linked_vortex.py` |
| **M7.5** | de Broglie clock (Zitterbewegung) · Particle stability | `scripts/m7_5_clock_stability.py` |
| **M7.6** | Magnetic moment μ + spin J · Spin-½ statistics · Quantum wave equation (Klein-Gordon) · Electric force (Coulomb, two-charge `E(d)~1/d`) | `scripts/m7_6_observables.py` |
| **M7.7 (milestone)** | consolidate the column + the M7 deep-dive ([`m7_theory_canonical.md`](m7_theory_canonical.md), the future "Per-model results of record" row) | `scripts/m7_7_canonical.py` + `scripts/m7_functional.py` |
| **M7.8** | Spin-½ statistics · Magnetic moment μ + spin J (the `U₊/U₋` reading) | `scripts/m7_8_helicity_pair.py` |
| **M7.9** | (validation infrastructure, no cell: the ChaosBook benchmark suite) | `scripts/m7_9_*.py` (orbits module + gates/benchmark/audit drivers) |
| **M7.15** | Magnetic moment μ, the intrinsic-magnetostatics completion (clock-carried per-defect B + Larmor response) | `scripts/m7_15_magnetic_force.py` |
| **M7.16** | Gravity | `scripts/m7_16_gravity.py` |
| **M7.17** | Strong force / confinement · Weak force | `scripts/m7_17_nuclear_forces.py` |
| **M7.18** | Antimatter + annihilation | `scripts/m7_18_annihilation.py` |
| **M7.19** | Neutrinos · Lepton mass spectrum (μ, τ) | `scripts/m7_19_lepton_neutrino.py` |
| **M7.20** | Dark matter candidate | `scripts/m7_20_dark_matter.py` |
| **M7.22** | Quarks · Baryons (p, n) · Mesons (π, K) · Orbital quantization | `scripts/m7_22_composites.py` |

The phase plan maps to the [`MODELS.md`](../../../../MODELS.md) criteria (live counts on its score-board; the shared set has grown since this plan was drafted): Phase 1 (M7.0-M7.7) earns the electron cells, **including Coulomb** (tied to the electron's charge, M5-style), and drafts the column at the M7.7 milestone (published at M7.21); Phase 2 (M7.15-M7.20) fills the remaining forces (magnetism, gravity, nuclear) + annihilation / neutrinos / dark matter, opening with the M7.8 helicity-pair check of the Fleury closure notes; Phase 3 (M7.21) publishes the column; Phase 4 (M7.22) fills the cells still 🚧 in M5 (quarks, baryons, mesons, orbital quantization). Each task upgrades its cells from 🚧 to a verified icon, as M5's column grew.

---

Artifacts follow the `research/` structure: scripts in [`scripts/`](scripts/), data in [`data/`](data/), plots in [`plots/`](plots/), per-task detail docs in [`tasks/`](tasks/), all named `m7_N_*.{py,npz,png,md}` (e.g. `scripts/m7_1_harmonic_lattice.py`, `scripts/m7_2_fleury_torus.py`, … `scripts/m7_22_composites.py`). Cross-refs: [`m7_background.md`](m7_background.md) (full background: § 5 dynamics; the MODELS.md cell map + production path are **Phases 3 + 5** above), the open questions + risks in [`m7_question_tracker.md`](m7_question_tracker.md), the Phase 1 report [`tasks/m7_phase1_report.md`](tasks/m7_phase1_report.md), the published column at repo-root [`MODELS.md`](../../../../MODELS.md).
