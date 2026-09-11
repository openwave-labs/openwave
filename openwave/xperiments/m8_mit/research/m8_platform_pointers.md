# M8 platform pointers: where to read, for the author and the author's AI agents

> **Purpose.** This is a curated map of the OpenWave repository written to be
> CONSUMED BY AI AGENTS working for the M8 author (and by the author directly). Each
> entry gives a linked short name (hover / follow the link for the real path) and what
> it teaches for the M8 program (the field-dynamics half MIT lacks:
> [`m8_background.md`](m8_background.md)). Every link was verified at scaffold time
> (2026-07-21).
>
> **If you are an AI agent:** load § 1 before generating any claim about this repo,
> then read only the sections the current task needs. Model output is a draft, never
> a result, until backed by a runnable script (the repo-wide contract).

## 1. Read first (the platform contract)

| Doc | Why |
| --- | --- |
| [`AI_HYGIENE.md`](../../../../AI_HYGIENE.md) | the working contract for AI-assisted research here: script-backed claims only, adversarial audit of substantive derivations, author-gated questions stay with the author |
| [`ONBOARDING_MODELS.md`](../../../../ONBOARDING_MODELS.md) | the onboarding standard M8 was admitted under: STEP 1 self-evaluation (parameter-count test, red flags), STEP 0 how to drive AI agents through reproduction |
| [`MODELS.md`](../../../../MODELS.md) | the shared 31-criteria coverage matrix M8 is scored against; the M8 column's cells flip only via runnable script + honest research note |
| [`dev_docs/METHOD_NOTE.md`](../../../../dev_docs/METHOD_NOTE.md) | the reporting standard for any substantive result: equations first, equation-to-code map, embedded figures, adversarial audit recorded |
| [`CONTRIBUTING.md`](../../../../CONTRIBUTING.md) + [`SYS_ARCH.md`](../../../../SYS_ARCH.md) | setup (conda, `pip install -e .`), fork → branch → PR with DCO sign-off, repo structure, tech stack (Python ≥ 3.12, Taichi GPU, NumPy/SciPy) |

M8's own spec of record lives in [`m8_theory_canonical.md`](m8_theory_canonical.md);
the program in [`m8_roadmap.md`](m8_roadmap.md).

## 2. Lagrangian families on the platform (candidates for S³/2I, task M8.4)

Four working Lagrangian families exist in this repo, each with code AND honest failure
records. Read the failures as carefully as the successes: they are the cheapest
guidance on what a compact-arena port must avoid.

| Family | Code | Documentation | What it teaches for M8 |
| --- | --- | --- | --- |
| M5 Landau-de Gennes matrix field (3×3 → 4×4 tensor, Skyrme kinetic term, Frank elastic energy) | [`engine2_pde.py`](../../m5_liquid_crystal/engine2_pde.py) (potential `V_M`, evolution kernels), [`engine3_observables.py`](../../m5_liquid_crystal/engine3_observables.py) (energy densities incl. Frank `H_F = (K/2)\|∇n̂\|²`), [`medium.py`](../../m5_liquid_crystal/medium.py) (the action of record, in the docstring) | [M5.1a Lagrangian framework](../../m5_liquid_crystal/research/tasks/m5_1a_lagrangian_framework.md) (the 8-experiment survey + the winning recipe), [M5.5a Lagrangian evolution](../../m5_liquid_crystal/research/tasks/m5_5a_lagrangian_evolution.md) | the platform's most-validated family; its defect spectrum (hedgehogs, vortex lines) is the natural "topological defect on a manifold" candidate; its potential-form well-posedness lessons (stencil-consistent minimizers) transfer directly to any curved-space discretization |
| M4 nonlinear vector wave (EWT) | [`wave_engine.py`](../../m4_ewt/wave_engine.py) (`∂²ψ/∂t² = c²∇²ψ − dV(ψ)` with swappable potentials), [`medium.py`](../../m4_ewt/medium.py), [`force_motion.py`](../../m4_ewt/force_motion.py) | [M4 briefing](../../m4_ewt/__M4_model_briefing.md); validation record in [M3/M4 status](../../m3_wolff_lafreniere/research/0_STATUS.md) | the standing-wave-first family, closest in spirit to MIT's background mode; its honest negatives (charge sign imposed, lock-in fragile under perturbation) mark exactly the traps a structure-first model must not repeat |
| M7 two-vector (Maxwell A + massive J, quartic self-coupling) | [`m7_functional.py`](../../m7_hydroboros/research/scripts/m7_functional.py) (the Lagrangian + period-averaged energy functional, small and auditable), [`m7_7_canonical.py`](../../m7_hydroboros/research/scripts/m7_7_canonical.py) (the canonical single-script electron) | [M7 theory canonical](../../m7_hydroboros/research/m7_theory_canonical.md) | the cleanest "fixed-ω harmonic frame" implementation (MIT's clock is also fixed-phase); its measured lesson: the truncated vacuum carries a tachyonic band, so CHECK THE VACUUM SPECTRUM FIRST on any new arena before hunting solitons |
| M6 charged-sector ODE lineage | [M6 archive](../../m6_ouroboros/research/archive/) (the record) | [M6 theory canonical](../../m6_ouroboros/research/m6_theory_canonical.md) (the provenance ledger) | a cautionary record: sign conventions, window-defined observables, and non-localized states can silently manufacture spectra; the M6.2/M6.4 method notes show the audit discipline that catches this |

## 3. Topological defect options (what "particle as defect" can mean on the arena)

| Pointer | Content |
| --- | --- |
| [M5.1b topological defect](../../m5_liquid_crystal/research/tasks/m5_1b_topological_defect.md) | the primary taxonomy: hedgehog / kink / vortex / knot, winding number → charge quantization, and the oscillation mechanism |
| [M5 background](../../m5_liquid_crystal/research/m5_background.md) | the narrative map: point defects as leptons, vortex strings as quarks, knots as baryons |
| [`compute_winding_number`](../../m5_liquid_crystal/engine3_observables.py) + [`m5_1_winding.py`](../../m5_liquid_crystal/research/scripts/m5_1_winding.py) | topological charge as an integral, the thing MIT's group-theoretic charge assignment is NOT yet; porting this to S³/2I is how the two notions get compared |
| [M5 seed catalog](../../m5_liquid_crystal/xparameters/) | the concrete defect seeds (`_topo_*.py`: canonical 4D, uniaxial, biaxial, charged ring, dressed): worked examples of "how to seed a defect configuration" that an S³/2I port will need equivalents of |
| The M8-specific question | which defect sectors exist at all on S³/2I (π₂ and π₃ of the target space RESTRICTED to the quotient's symmetry), and whether the anti-periodic (double-cover) boundary condition creates sectors flat space does not have. Genuinely new ground: no existing column has run a compact-quotient arena (OQ5 in [`m8_theory_canonical.md`](m8_theory_canonical.md)) |

## 4. Clock + stability insights (the platform's measured lessons for a background-clock model)

| Pointer | Lesson |
| --- | --- |
| [M5.1b topological defect](../../m5_liquid_crystal/research/tasks/m5_1b_topological_defect.md) + [M5.9 findings](../../m5_liquid_crystal/research/findings/m5_9_lepton_mass_clock_findings.md) | the derived de Broglie clock: `ω ∝ m` scaling and the calibration chain. MIT's Waltz clock is ASSUMED; this record shows what "derived" looks like and what it costs |
| [M6 theory canonical](../../m6_ouroboros/research/m6_theory_canonical.md) + [M6 briefing](../../m6_ouroboros/__M6_model_briefing.md) (Derrick rows) | oscillation as the genuine third Derrick escape: the platform-validated stability route a background clock naturally provides |
| [`m7_5_clock_stability.py`](../../m7_hydroboros/research/scripts/m7_5_clock_stability.py) | the clock as stabilizer, measured: solitons exist only above the vacuum's tachyonic band (ω* = 0.786); the direct precedent for "the background mode gates existence" |
| Compact-arena Derrick note | on S³/2I the Derrick dilation argument is modified (no global scaling freedom; R is a scale). Do the scaling analysis EXPLICITLY for each candidate family before numerics; do not import flat-space Derrick conclusions |

## 5. The lepton-hierarchy target (task M8.6, CLOSED WITHOUT RUNNING 2026-08-07)

| Pointer | Content |
| --- | --- |
| [M8.6 readiness note](findings/m8_6_readiness_note.md) | **READ FIRST.** The provenance audit that gated this task, and the seven conditions (§ 8) that reopen it. Written because the row below named a target that turned out to be circular |
| ⚠️ [M5.9 findings](../../m5_liquid_crystal/research/findings/m5_9_lepton_mass_clock_findings.md) | **NOT an admissible target.** The eigenvalue hierarchy `1 : 5.9 : 15.1` is `Λ := m^(1/3)` from the charged-lepton masses, so comparing it against those masses compares a number to its own definition: the source calls it "near-tautological" and the values "Yukawa-like input". The mass law `E ∝ Λ³` is a real M5 result; the hierarchy VALUES are not |
| [M5 particle hunt](../../m5_liquid_crystal/research/m5_particle_hunt.md) | the MEASURED three-minima census (A < C < B, C/A ≈ 4.2, B/A ≈ 16.0): the only non-circular M5 target, and not usable yet, since it is consistency-converged rather than value-converged and carries no physical-parameter bridge |
| [M5.21.11](../../m5_liquid_crystal/research/tasks/m5_21_11_task_details.md) | the bridge M8.6 waited on: it ran 2026-08-07 under the frozen route-(b) framework ([`m5_21_11_framework.md`](../../m5_liquid_crystal/research/findings/m5_21_11_framework.md)) and FAILED TERMINALLY on the pre-committed F3 + F4 criteria, closing the gate permanently on the M5 side ([finding 9](tasks/m8_6_task_details.md)): no physically parameterized census exists or is promised |
| Discipline | the pre-registration discipline (mapping frozen before any number) remains the standard for any future revival, but no revival is admissible from the existing M5 instrument generation (route (b) was the last admissible route) |

## 6. Quotient-manifold simulation engineering (task M8.5)

No existing column runs a curved compact arena, so this is new infrastructure. Two
routes, both viable; prototype before committing.

**Status 2026-08-31.** Route (a) was built and adjudicated (`M85B-ADJ-07`) and its dynamics
substrate did NOT qualify for the M8.4 observables; see
[`findings/m8_4_closeout.md`](findings/m8_4_closeout.md). Route (b) was never built as a
simulation backend, only as character-averaging certification machinery, so its dynamics evidence
was never gathered. It is now M8.5-C, target-free, with its qualification protocol FILED 2026-08-28 and frozen at merge ([protocol](findings/m8_5c_protocol.md)); attempt A1 terminated without adjudication 2026-08-29 on a frozen-arm defect ([addendum 1](findings/m8_5c_protocol.md), [record](m8_5c/a1/)); the derived successor [M8.5-C2](findings/m8_5c2_protocol.md) is FILED 2026-08-29 and frozen at merge. Its attempt C2-A1 was commissioned, terminated 2026-08-30 on unit misconduct that destroyed the output ledger's integrity, filed byte-identical ([record](m8_5c2/a1/), [#508](https://github.com/openwave-labs/openwave/pull/508)), and the ruling ([#506](https://github.com/openwave-labs/openwave/issues/506)) adjudicated `M8.5-C2-FAILED`, instrument-attributed STOP-QUAL ([addendum 1](findings/m8_5c2_protocol.md)). The spectral route is closed, and the M8.4 reopening path closed with the verdict.

| Route | How | Trade-offs |
| --- | --- | --- |
| (a) 2I-equivariant grid | simulate on an S³ grid (embedding or intrinsic coordinates) and impose the 120-element 2I identification as an equivariance constraint or ghost-cell map | reuses the platform's finite-difference habits (see M5's stencil lessons in § 2); the identification map is fiddly; watch the cone/edge structure where MIT's Möbius boundary lives |
| (b) Spectral method in 2I-symmetric harmonics | expand fields in S³ harmonics restricted to 2I-invariant (or covariant) subspaces; evolve mode coefficients | elegant: the basis IS the McKay representation theory, so slot structure is manifest; nonlinear terms need convolution handling; the anti-periodic sector enters as the double-cover representations |
| Shared infrastructure | [`../../../common/constants.py`](../../../common/constants.py) (units), [`../../../common/equations.py`](../../../common/equations.py) (core equations), per-model `_launcher.py` pattern for later | production kernels are Taichi-first on this platform (per-frame paths in `ti.kernel`); NumPy/SciPy is fine for research scripts, and M8.1's eigensolve is plain SciPy |

## 7. The 3D rendering port (LATER: gated on validated field dynamics, task M8.7)

Do not start here. Rendering comes after an M8 dynamics validates in-platform
(the [`m8_roadmap.md`](m8_roadmap.md) gate), a condition the M8.4 closure did not change.
No such dynamics exists yet: M8.4 closed unresolved on 2026-08-26 with no target
configuration executed, and the route through M8.5-C closed with the C2 verdict of
2026-08-31 ([#506](https://github.com/openwave-labs/openwave/issues/506)). Any route now
runs through a future program under platform conditions, which nothing currently
authorizes ([#512](https://github.com/openwave-labs/openwave/discussions/512)). If the
gate opens, this is the port path an AI agent should follow:

| Step | Pointer |
| --- | --- |
| 1. Study the M5 architecture split | [`../../m5_liquid_crystal/_launcher.py`](../../m5_liquid_crystal/_launcher.py): the pattern is medium → seeds (`engine1`) → PDE (`engine2`) → observables (`engine3`) → render (`engine4`), wired to the shared GGUI stack |
| 2. The shared rendering stack | [`../../../i_o/render.py`](../../../i_o/render.py) (Taichi GGUI window / camera / scene every model reuses), [`../../../i_o/flux_mesh.py`](../../../i_o/flux_mesh.py) (3-plane cross-section detectors), [`../../../i_o/video.py`](../../../i_o/video.py) (export) |
| 3. The M5 worked example of render design | [`../../m5_liquid_crystal/engine4_render.py`](../../m5_liquid_crystal/engine4_render.py) (ellipsoid shells, defect-line rods, director glyphs) + the design doc [`m5_4b_rendering_features.md`](../../m5_liquid_crystal/research/tasks/m5_4b_rendering_features.md) |
| 4. The M8-specific problem | S³/2I is not a box: choose a chart (e.g. render a fundamental domain, or a stereographic projection of S³ with the 120-fold identification indicated); nothing on the platform solves this yet, so it is a design task, not a copy |
| 5. Standard | the launcher must run the SAME kernels the research stack validates (M5's rule: interactive demos show the physics of record, including its instabilities) |

## 8. House standards for M8 tasks (so contributions merge smoothly)

| Standard | Rule |
| --- | --- |
| Naming | scripts / data / plots under `research/` with `m8_<id>_` prefixes; per-task notes `tasks/m8_<id>_task_details.md`; substantive results get `findings/m8_<id>_method_note.md` (the pre-created folders: `scripts/`, `plots/`, `tasks/`, `data/`, `findings/`) |
| Status honesty | MODELS.md cells flip only with a runnable script + a research note documenting pass/fail; negatives are results and are wired into the column the same day |
| Pre-registration | gates, conventions, and success criteria are written down BEFORE numerics, in the task note; forks are reported with all numbers, never tuned toward published values |
| Audit | substantive derivations and claims get an independent adversarial pass (own script, own method) before they are trusted; record the audit in the deliverable |
| Style | markdown tables for structured content; status icons only ✅ ⚠️ ❌ 🔶 🚧; escape literal pipes as `\|` inside table cells; relative links |
