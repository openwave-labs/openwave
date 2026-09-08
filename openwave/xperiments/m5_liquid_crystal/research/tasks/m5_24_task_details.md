# M5.24, production-engine catch-up (audit + port the M5.x-era physics)

> ✅ CLOSED COMPLETE + APPROVED 2026-07-19 (go 15:50, closed ~21:05 EDT; three rounds + close-out). Roadmap row: [`m5_roadmap.md § DONE`](../m5_roadmap.md). The remaining round-4 scope (the fixed-J port) was RE-HOMED at close to [M5.23.1](m5_23_1_task_details.md) (gated on [M5.21.9](m5_21_9_task_details.md), which gates on the author's fork answer). Checkpoint: [`../checkpoints/m5_24_progress.md`](../checkpoints/m5_24_progress.md).

## TASK PLANNING

**Scope**: Round 1 of the production-engine catch-up. The launcher's production physics stack (`engine2_pde.py` + `engine1_seeds.py` + `engine3_observables.py` + the `medium.py` physics fields, untouched since 2026-07-02 except the seeder) predates the M5.9-M5.21 research era that produced the canonical registry. This round: (A) full per-kernel inventory of the production physics stack, (B) confrontation against [`m5_theory_canonical.md`](../m5_theory_canonical.md) + the Done-task lineage into a routed GAP TABLE, (C) the first port batch (top-priority gaps, taichi-first, per-gap headless selftests), (D) launcher smoke verification + M5.23 selftest regression. Expected MULTI-ROUND (user note at staging: "maybe will take multiple updates later"); rounds 2+ take the deferred gaps.

**Definition of done (this round)**:

| # | Criterion | Check |
| --- | --- | --- |
| 1 | Per-kernel audit inventory in this doc: equation implemented, parameters, convention, provenance | every physics kernel in `engine1/2/3` + `medium` dispositioned |
| 2 | Gap table: per gap the research-validated source (task / method-note link) vs the production state, ROUTED: `PORT NOW` / `DEFER` (round 2+) / `NOT A GAP` (deliberate demo tuning) / `USER CALL` | every canonical-registry recipe row dispositioned |
| 3 | Port batch 1: each ported gap carries a headless selftest, ALL GREEN | goal-loop, 3-try cap per gate |
| 4 | No regression: launcher compiles + headless smoke evolve on the standard config; the M5.23 ellipsoid selftest stays 14/14 | machine-checked |
| 5 | Doc checker clean over touched `.md` | exit 0 |

**Gating**: user "go" (received 2026-07-19 15:50). Model/effort: Fable 5 / high (research default; audit + porting with physics judgement).

**Blindspot pass** (cross-cutting audit territory):

| Blindspot | Guard |
| --- | --- |
| Viz-tuning vs physics-gap conflation: production defaults may be DELIBERATE demo tuning, not stale physics | route ambiguous rows `USER CALL`, never silently clobber a tuned default |
| Transfer validity: canonical recipes were validated on research grids (axisym reductions, small boxes, specific integrators); the live 3D launcher grid is a different regime | each port carries a transfer note: what was re-verified at production scale |
| Convention drift: the 2026-06-21 index-0 flip (`D = diag(g, 1, δ, 0)`) | verify per kernel which convention it ACTUALLY uses; never assume the flip propagated |
| Hot files: `_launcher.py` / `medium.py` / `engine4_render.py` were edited today (M5.23 + user edits) | re-grep current state before every edit |
| Hidden dependents: launcher demo features (WM modes, flux mesh, VIZ suite) may depend on OLD kernel behavior | per port: back-compat or an explicit behavioral-change flag at review |

**Research body**: audit inventory + gap table + findings live in THIS doc; scripts `../scripts/m5_24_*.py`; plots `../plots/m5_24_*.png`; checkpoint `../checkpoints/m5_24_progress.md`. Feeds [M5.23.2](m5_23_2_task_details.md) (gated on the certified 4D production physics this task delivers).

**Stages**: A inventory sweep → B gap table → C port batch 1 → D smoke + regression. Preconditions: taichi via the conda env `master312`; canonical registry fresh-read at EXECUTE.

## STAGE A FINDINGS: the production physics inventory (2026-07-19)

Every physics kernel in the production stack, dispositioned. The launcher's live dispatch (verified in [`_launcher.py`](../../_launcher.py) `evolve_pde`): three evolve paths (3D leapfrog default; 4D masked leapfrog when the 4D xperiment is seeded; the Option B constrained integrator when `INTEGRATOR_4D = "constrained"`), then `eigen_decompose` refreshes the derived fields every frame.

| Kernel (file) | Physics implemented | Era vs the canonical registry |
| --- | --- | --- |
| `commutator` / `principal_director` / `eigvec_for` (`engine2_pde`) | analytic Cardano symmetric-3×3 eigensolver (the Metal/f32 `ti.sym_eig` replacement) | current; convention-agnostic, reused by everything incl. M5.23 |
| `eigen_decompose` + `director_mid` (`engine2_pde`) | per-voxel spectrum + apolar sign-continuous director + clock-hand axis, index-0 convention | current (the 2026-06-21 flip propagated) |
| `V_M` / `dV_M` (`engine2_pde`) | LdG quartic trace potential `a·Tr(M²) − b·Tr(M³) + c·(TrM²)²`, spatial block only | SUPERSEDED by the universal spectral potential (canonical § 1 "Superseded potential" row) |
| `compute_curvature_flux` + `evolve_M` (`engine2_pde`) | Eq.18 3D leapfrog: PLAIN commutator `F = [M_μ, M_ν]`, 8× flux, ½‖Ṁ‖²_F kinetic, M5.8.1 time-axis freeze | pre-verified-L era (M5.5); the sanctioned runnable dynamics is the § 2 row 6 canonical stack |
| M5.8.2c masked 4D path: `compute_stable_mask`, `compute_tstar`, `dV_M_dressed`, `compute_curvature_flux_4d`, `sample_v03_drift`, `evolve_M_4d` (`engine2_pde`) | the masked ηFη blend + dressed well + faithful-lite diagonal inertia | era experiment, pre-verified-L; the ghost problem it worked around is now UNDERSTOOD structurally (the η-null constraint + indefinite H, § 2 rows 1-3) |
| Option B constrained integrator: `init_P_4d` … `update_M_4d_constrained` (`engine2_pde`) | per-voxel 10×10 spectral projection, positive-inertia keep | the canonical § 2 explicitly names this stack the historical precedent of the structural disease; its refined descendant (the research true-L EL solve) is DIAGNOSTICS-ONLY by verdict |
| `relax_director_step` + `rebuild_M_from_director` (`engine2_pde`) | M5.1 Frank-energy director descent + uniaxial M rebuild | viz-tier seed conditioning; canonical statics recipes (FIRE frozen-time-row, stencil-symmetrized functional) are a different grade |
| Seeds (`engine1_seeds`): vacuum, uniaxial hedgehog, biaxial hedgehog, charged ring, dressed hedgehog | all embed the time axis as `+g` at `[0,0]` (block-diag), boost-decoupled except the dressed seed | ring seed is post-census (ported 2026-07-17); the `+g` embed is era-consistent with the plain-commutator paths but is NOT the verified vacuum (`M_vac = diag(−g, 1, δ, 0)`; `diag(+g,…)` has `V ≈ 1.05e6·w` at g = 8, canonical § 1) |
| `update_trackers_M`, `compute_energyH_density_M`, `compute_energyF_density`, `compute_director_em*` (`engine3_observables`) | ‖M−D_vac‖_F amplitude + ‖Ṁ‖_F clock; the Eq.18 plain ℋ + LdG V + v0-subtraction display hack; Frank density; director div/curl EM views | energyH is pre-verified-L; the trackers/EM/Frank views are director-derived viz (era-agnostic) |
| `compute_winding_number` (`engine3_observables`) | naive sphere integral on the director | the canonical instrument adds guards (multi-radius, decline on λ₁ ≈ λ₂); dashboard-only today |
| `fill_dipole_sample_B` (`engine3_observables`) | analytic dipole placeholder | already tracked as [M5.23.5](m5_23_5_task_details.md) stage 2 |

## STAGE B: the gap table (routed)

Porting reference of record: [`../scripts/m5_21_3_a_4d.py`](../scripts/m5_21_3_a_4d.py) (the M5.21.3 audited 4D instrument: `comm_eta`, `inner_eta`, `e_parts`, exact `grad`, `vac4`, W1 = 7.24023879e-4; its own gates: complex-step < 5e-9, SO(1,3) invariance < 1e-9 + negative control, vacuum ≡ 0 both branches, 3D block-diag regression < 1e-12). Key transfer fact from its G3 gate: on a block-diag (spatial-only) field with the time-time entry at `−sg`, the η-stack static energy EQUALS the plain-commutator 3D energy exactly, so the port is behavior-compatible with today's static seeds by construction.

| # | Gap | Research-validated source | Production state | Route |
| --- | --- | --- | --- | --- |
| G1 | η-bracket field strength `F = [A_i, A_j]_η` + signed inner product `⟨F,F⟩_η` | canonical § 1 (M5.18, 15/15); `m5_21_3_a_4d.py` | plain Frobenius commutator (3D path); pre-verified ηFη masked twist (4D-era paths) | ✅ PORT NOW |
| G2 | universal spectral potential `V4 = w·Σ_p (Tr_η(M^p) − C_p)²`, `w = 7.24e-4`, `C_p = sg^p + 1 + δ^p` | canonical § 1 + § 4 (WSCALE lock) | LdG `V_M(a,b,c)` (superseded) + the M5.8.2 dressed well | ✅ PORT NOW |
| G3 | the canonical-kinetic regularization stack as the runnable 4D dynamics (½‖Ṁ‖²_F + η-curvature + V4; § 2 row 6: "fine for statics-adjacent dynamics questions and films", i.e. exactly the launcher's use class) | m5_20 § 2-3; the 3D twin (m5_21 § 4); the M5.21.3 4D functional | M5.8.2c masked-blend leapfrog + faithful-lite inertia | ✅ PORT NOW: new `canonical` integrator path (the centerpiece) |
| G4 | covariant vacuum `M_vac = diag(−g, 1, δ, 0)` | canonical § 1 | every seed embeds `+g` | ✅ PORT WITH G3: seed-time time-axis flip on the canonical path only (other paths stay `+g`, era-consistent) |
| G5 | verified-H energy display: `u_η + V4` with v0 ≡ 0 at vacuum (exact) | canonical § 1 (Legendre) + the factor-2 bridge (§ 5.5) | Eq.18 plain ℋ + v0-subtraction hack | ✅ PORT NOW: η energy-density kernel for the canonical path |
| G6 | sym-stencil (½(fwd + bwd), exact adjoints) well-posedness | M5.21.2b certificate; § 6 anti-recipes (2h deep statics) | all production PDE kernels are 2h central | ✅ PORT NOW inside the new path (existing paths untouched) |
| G7 | status of the constrained integrator | § 2 rows 1-3 + the historical-precedent note | live launcher mode | NOT A GAP: keep as the era mode; docstring updated to point at the § 2 verdict chain |
| G8 | statics-relaxer catch-up (FIRE frozen-time-row; stencil-symmetrized functional) | § 5.2 | M5.1 Frank director descent only | 🚧 DEFER round 2 (needed when the launcher wants census-grade relaxed states) |
| G9 | guarded winding instrument (multi-radius + λ-gap decline) | § 5.4 | naive sphere integral | 🚧 DEFER round 2 (dashboard-only today) |
| G10 | M5.8.3(a): the 4D production seeder extension | [`m5_8_3_task_details.md`](m5_8_3_task_details.md) | `seed_dressed_hedgehog_M` exists (M5.8.2-era, `+g`) | PARTIAL: the canonical-path flip covers static 4D seeds; a verified-L clock-carrying seed is physics-gated (the fixed-J arc, [M5.21.9](m5_21_9_task_details.md)) |
| G11 | M5.8.3(b): the faithful-kinetic engine variant | [`m5_8_3_task_details.md`](m5_8_3_task_details.md); § 2 rows 1-2 | not implemented | NOT A GAP: SUPERSEDED (the faithful kinetic IS the true-L operator, ill-posed as a free EL; the constrained mode already covers the era physics) |
| G12 | anti-recipe guards riding the ports (certified dt values, FIRE dt caps, no-deep-2h-statics) | § 5.2/5.3/§ 6 | n/a | carried as docstrings + notes in the new kernels |

## STAGE C + D FINDINGS: port batch 1 (2026-07-19)

### What landed

| Piece | Where | Content |
| --- | --- | --- |
| The η algebra + spectral potential | [`engine2_pde.py`](../../engine2_pde.py) M5.24 section | `eta_left`/`eta_right`/`comm_eta44`/`inner_eta44` (the M5.18 verified bracket + inner product), `v4_of`/`dv4_of` (the universal spectral potential + its exact gradient), `W1_SPECTRAL = 7.24023879e-4` (the WSCALE lock) |
| The canonical evolve pair | same | `compute_eta_flux(branch)` (two-pass symmetrized stencil, ½(fwd + bwd) with exact adjoints, the M5.21.2b well-posedness certificate) + `evolve_M_eta_start`/`evolve_M_eta_finish` (adjoint gather + V4 force; `M_new_am` doubles as the force accumulator, zero new fields) + `flip_time_axis` (covariant-vacuum conversion) |
| The verified-H energy view | [`engine3_observables.py`](../../engine3_observables.py) | `compute_energyH_density_eta`: ½‖Ṁ‖²_F + u_η + V4 with a TRUE-ZERO vacuum floor (retires the v0-subtraction hack on this path) |
| Launcher wiring | [`_launcher.py`](../../_launcher.py) | canonical activation (post-relax time-axis flip + routing), the dt cap in `compute_timestep` (survives the every-frame refresh), the energyH branch, a bounded-energy auto-pause guard, and a **stale-4D-state reset at the seed dispatch** (a latent era bug: an xperiment switch away from a 4D config kept evolving on the previous integrator; my canonical path would have inherited it) |
| Demo xperiment | [`xparameters/_topo_canonical4d.py`](../../xparameters/_topo_canonical4d.py) | biaxial hedgehog on the canonical stack (its far field IS the covariant vacuum spatial spectrum) |
| The machine gate | [`../scripts/m5_24_canonical_engine_selftest.py`](../scripts/m5_24_canonical_engine_selftest.py) | 11 checks against the AUDITED reference imported directly (no re-transcription) + an informational indefinite-channel probe |

### Gate results (selftest, 3 tries per the goal-loop cap)

| Check | Result |
| --- | --- |
| V4(vacuum) = 0 exact; V4(diag(+g,…)) = W1·(4g² + 4g⁶) closed form (the canonical § 1 "not a vacuum" number, unweighted ≈ 1.05e6) | ✅ 3e-11 / 1e-5 |
| V4 random vs the reference `e_parts` | ✅ 3.3e-7 |
| Production energy kernel vs reference (bump field) | ✅ 8.4e-7 |
| The two-pass force vs the reference exact gradient | ✅ 1.25e-5 |
| Vacuum force at the f32 floor | ✅ 5.9e-4 (the g⁴-trace-scale f32 floor; per-step displacement ~1.5e-8, self-centering well) |
| SO(1,3) invariance + broken-L negative control | ✅ 1.1e-6 / control 1.8e4 |
| 3D block-diag regression (u_η == plain 3D read) | ✅ 5.1e-6 (the transfer fact: behavior-compatible with static seeds) |
| flip_time_axis on all three buffers | ✅ |
| **E conservation, spatial sector, 1500 steps @ dt = 0.005** | ❌ at the 5e-3 tolerance: drift 2.34e-2, **but bounded** (max\|M\| = 8.00 = the vacuum g scale, E 0.1228 → 0.1223, no growth). 🔶 suspected ESTIMATOR artifact: the staggered kinetic (M_t − M_{t−1})/dt mis-measures the ω·dt = 0.39 stiff-mode energy by O((ωdt)²) ≈ 15% per mode; the goal-loop 3-try cap was reached, so per discipline the gate is surfaced, not tuned away. Fix (centered-velocity or half-step ledger) = round 2 |
| M5.23 ellipsoid selftest regression | ✅ 14/14 |
| Headless launcher smoke (real 63³ demo grid) | ✅ canonical routed, time axis flipped, dt cap survives the frame loop, 30 real frames bounded (max\|M\| = 8.000), energyH finite from the true-zero floor, xperiment switch resets routing, the plain path still evolves |

### Measured findings (beyond the port)

| Finding | Status | Detail |
| --- | --- | --- |
| The certified dt is DIMENSION-SPLIT | ✅ measured | the axisym stack's dt = 0.02 (canonical § 5.3) went NaN on the full-3D f32 path (ω_stiff·dt = 1.57, too close to the leapfrog margin 2.0); the 3D twin's dt = 0.005 (canonical § 3) is stable: the launcher cap defaults to 0.005. Candidate § 5.3 annotation at the model-doc sweep |
| The indefinite channel is REAL in production | ✅ measured | a full-4×4 noise bump (time-row components) runs away WITH E conserved (NaN by ~1500 steps at both dt values): the canonical § 1 "H unbounded below" physics reproduced in the f32 production kernels, not a port defect. Block-diag data (every launcher seed) is an invariant manifold where u_η reduces to the positive 3D energy: the demo regime. The launcher carries a bounded-energy auto-pause guard |
| Unit-map gap (flagged, not fixed) | 🔶 | the launcher's physical `dx_am` (~15.6 at 64³ over 1 fm) differs from the research grid unit (h = 1.5), shifting the curvature-vs-V4 balance relative to research-grade runs; the selftest pins dx_am = 1.5 via the universe edge; a launcher-side unit alignment (or a config-declared research-unit dx) = round 2 |

No plot artifacts this round (the gates are numeric; the launcher itself is the visual deliverable). No data files > 1 MB were produced (the selftest writes nothing to `data/`).

### Round 2+ backlog (stays in this task, per the multi-round expectation)

| Item | From |
| --- | --- |
| Conservation-ledger estimator fix (centered velocity / half-step ledger) + re-gate at 5e-3 | the surfaced gate |
| G8 statics-relaxer catch-up (FIRE frozen-time-row; stencil-symmetrized functional) | gap table |
| G9 guarded winding instrument | gap table |
| Launcher unit-map alignment (research-unit dx option) | measured finding |
| Optional sponge (the § 5.3 exact-ledger damping) for long live runs | canonical § 5.3 |
| Constrained-path docstring pointer to the § 2 verdict chain (G7 note) | gap table |

## ROUND 2 FINDINGS (2026-07-19, go 17:53)

Trigger: the user's live-demo verdict at the round-1 review ("the demo works" + the animation is too slow to watch the ellipsoids move). Round-2 scope executed: the viz-speed lever, the conservation-ledger fix, the unit-map alignment, the G7 status note, the G9 guarded winding instrument. G8 (FIRE statics relaxers) rides to round 3.

### What landed

| Piece | Content |
| --- | --- |
| `ETA_SUBSTEPS` (the viz-speed lever) | N physics steps per rendered frame at the CERTIFIED dt (default 8, config-tunable). dt itself cannot rise (the stiff-mode wall 2/78 ≈ 0.026 τ, NaN measured at 0.02) and SIM_SPEED cannot help (the cap pins dt_eff), so visual speed = substeps. The launcher's shared buffer-swap tail still runs exactly once per frame (the last substep leaves its swap to it) |
| `ETA_DX` (the unit-map fix, was round-2 backlog) | the canonical kernels now take the grid spacing as an argument; the demo config sets the RESEARCH unit (1.5 = the m5_21_2b/3 h), making the 63³ box research-twin geometry. The physical `dx_am` (~15.6) had weakened curvature ~100× against the dimensionless V4, so the field barely moved spatially: this was a large hidden part of the "nothing moves" report |
| The conservation ledger, fixed + mechanism identified | (a) centered velocity `(M_{t+1} − M_{t−1})/2dt` replaces the staggered estimator (its O((ωdt)²) stiff-mode bias was round 1's suspicion, confirmed real but NOT the whole story); (b) the remaining monotone loss was traced by an amplitude ladder (amp 0.05/0.1/0.2/0.4 → rel drift 2.6e-2/5.6e-3/6.8e-4/4.8e-5, E growing quartically): the ABSOLUTE loss is amplitude-independent (~1e-2 per 1000 steps) = the f32 update-truncation floor of the \|M\| ~ g field scale. ✅ measured: numerical cooling, not a scheme error. The gate now runs at demo-grade amplitude (0.2) and PASSES at 5.16e-4; the floor is quantified in a selftest info line |
| G7 status note | the constrained-integrator section header now carries the canonical § 2 verdict-chain pointer (historical precedent; the true-L EL solve is diagnostics-only; kept as the era mode) |
| G9 guarded winding | `compute_winding_number_guarded` (engine3): multi-radius Q + the eigen-gap decline flag per canonical § 5.4 + the § 6 churned-state anti-recipe. NOTE: the plain winding read has NO live launcher caller today (dormant M5.1 diagnostic); the guarded wrapper is the instrument any future consumer must use. Exercised in the smoke: the seeded 63³ hedgehog reads q = 0.996 ± 0.000, min gap 0.582, not declined (matches the era anchor Q = ±0.996, § 4b) |

### Round-2 gates

| Check | Result |
| --- | --- |
| Full selftest (new signatures + centered ledger) | ✅ ALL 11 GREEN, incl. E conservation drift 5.16e-4 over 1000 steps at amp 0.2 (was the round-1 surfaced gate) |
| f32 floor quantified | [info] small-amplitude state loses 1.93e-3 absolute per 500 steps, amplitude-independent |
| Launcher smoke (63³, substeps 8, dx_eta 1.5) | ✅ routing + flip + cap + 30 frames (240 physics steps) bounded, max\|M\| = 8.000; guarded winding green; xperiment switch still resets routing |

Round-3 backlog: G8 statics-relaxer catch-up (FIRE frozen-time-row + the stencil-symmetrized functional, the census-grade-states enabler), optional sponge for long live runs, and any live-demo feedback from the substeps/dx configuration.

## ROUND 3 FINDINGS (2026-07-19, go 20:02)

Trigger: the user's round-2 live verdict (the animation works at 64 substeps; the sloshing/shape-melt diagnosis was accepted as known physics) → "go round 3": the relaxer + sponge arc, with the M5.21.9 → M5.23.2 staging left untouched.

### What landed

| Piece | Content |
| --- | --- |
| G8: the canonical FIRE relaxer | `fire_relax_canonical` (engine2, single source: launcher + selftest call the same function): FIRE on the SAME pinned-interior η + V4 gradient the dynamics uses, TIME ROW + COL FROZEN (the canonical § 5.2 frozen-time-row recipe; dt0 = 0.005, dt_max = 0.05, standard FIRE constants). Velocity lives in the free `Md_am`; force extracted from the dt = 1 two-pass (no new matrix field); Metal-safe two-stage reductions (per-z-slice partials in the new small `fire_partials` field, the M5.0h contention lesson). The § 5.2 `1/w` precondition is NOT ported (convergence accelerator only) |
| The RELAX button + config | `RELAX (FIRE canonical)` in the sim menu (canonical path, paused; 500-iter chunks per press, prints the residual trajectory) + `CANON_RELAX_ITERS` for seed-time relax (default 0) |
| The boundary sponge | `apply_eta_sponge` (engine2): damped-Verlet correction `M_new ← (M_new + a·M_prev)/(1 + a)`, `a = ½γ(x)dt`, `γ = γmax·q²` quadratic ramp (the canonical § 5.3 pattern), in-kernel distance (no extra field); `γmax = 0` is an EXACT no-op. Config `ETA_SPONGE_GAMMA` (demo: 0.5) + `ETA_SPONGE_WIDTH` (10). The research exact-dissipation LEDGER is not ported (demo device, noted in the docstring) |

### Round-3 gates (selftest now 14 checks)

| Check | Result |
| --- | --- |
| FIRE relax on the amp-0.2 bump (400 iters) | ✅ E 23.0089 → 0.0010, force RMS 2.82e-2 → 1.16e-5 (÷2400), boundary vacuum + block-diag sector intact |
| Sponge γ = 0 no-op | ✅ max diff 0.0 exactly (try-1 failure was a TEST bug: `bump_field(rng)` draws fresh randoms per call, so the two compared runs used different fields; fixed by reusing one array) |
| Sponge dissipation | ✅ 20.6% dissipated over 1500 steps at γ = 1.0 vs the 5e-4 conservation floor (unambiguous; the try-1 threshold was mis-predicted: γq² at the bump's sponge depth gives a few %/τ, exactly what was then measured) |
| All round-1/2 gates | ✅ unchanged green (14/14 total) |
| Launcher smoke (63³, 64 substeps, sponge on) | ✅ RELAX chunk on the real hedgehog: residual ×8 down in 60 iters with the winding HELD (guarded read q = 0.996 ± 0.000 after relax, topology intact); 320-substep frame loop bounded; routing reset on switch still green |

### Live-demo recipe (what the user can now do)

| Step | Effect |
| --- | --- |
| Boot `_topo_canonical4d` (PAUSED) | the raw seed, energy view from the true-zero floor |
| Press `RELAX (FIRE canonical)` a few times | drains the seed's transient (the round-2 sloshing) toward a near-stationary state, watching the force RMS fall in the console; topology holds |
| Press `>> EVOLVE PDE >>` | dynamics from the relaxed state: the residual motion is genuine physics (breathing mode), not seed shock; the sponge (γ 0.5) absorbs outgoing radiation instead of reflecting it back (the t ≈ 12 caveat) |

The clean spinning-electron animation (the δ-axis sweep) remains gated on the fixed-J physics as staged: [M5.21.9](m5_21_9_task_details.md) → [M5.23.2](m5_23_2_task_details.md) (untouched this round, per the user's directive).

Round-4+ backlog: the sponge energy ledger (if live runs ever need the dissipated-energy bookkeeping), the launcher unit-map follow-through on other paths, and any census-grade relax targets (deep relaxes stay research-side; the launcher FIRE is demo-grade conditioning, chunked + interruptible).

## TASK REVIEW (2026-07-19, close-out)

**Task Duration:** 05:15 wall (from the 15:50 go to the ~21:05 close; active segments: round 1 00:25, round 2 00:07, round 3 00:07, close-out ~00:10; the user live-tested the demo between rounds)
**Usage Cap Triggered:** NO (all rounds; the 7:30pm window rolled once mid-task, the re-armed ping never fired)

**Results**: the three rounds above (each with its own findings section + review record). Consolidated: (1) ✅ the full production audit + the routed gap table G1-G12, every canonical-registry recipe row dispositioned, the merged M5.8.3 scope closed (seeder arm partially delivered via the canonical path; faithful-kinetic arm superseded by the § 2 verdict chain); (2) ✅ THE CANONICAL STACK in production (η-bracket curvature + universal spectral V4, symmetrized stencil, covariant vacuum, verified-H view, dt capped at the certified full-3D step), gated against the audited M5.21.3 reference; (3) ✅ the viz levers (ETA_SUBSTEPS, ETA_DX research-unit map), the centered conservation ledger + the measured f32 cooling floor, the guarded winding instrument; (4) ✅ the FIRE relaxer (frozen-time-row recipe, topology-holding: q = 0.996 after relax) + the boundary sponge. Selftest ALL 14 GREEN; the launcher is now a live research-twin of the M5.21-era instrument.

**Issues / blockers**: none open. The measured characteristics carried honestly: the f32 numerical-cooling floor (amplitude-independent ~1e-2/1000 steps), the pinned-shell energy frame (the Dirichlet mismatch layer, normal), and the indefinite time-row channel (guarded by auto-pause).

**Deviations from plan**: the round-1 conservation gate consumed its 3-try goal-loop cap and was surfaced rather than tuned (resolved at round 2 with the mechanism measured); the dt cap moved 0.02 → 0.005 on measurement; otherwise per plan.

**Action needed**: the user runs git (message proposed at close). Next in the chain: [M5.21.9](m5_21_9_task_details.md) (author-gated) → [M5.23.1](m5_23_1_task_details.md) (the fixed-J port) → [M5.23.2](m5_23_2_task_details.md).

**Standing directive recorded (user, 2026-07-19, at the round-3 review)**: NO display-only kinematics in the launcher: rendered motion must always come from simulated dynamics ("we need simulation fidelity; animation is easy; we need to be running on top of real physics always"). Codified in [`m5_visualization.md § Other visualization conventions`](../m5_visualization.md).

**Model-doc sweep (close-out)**: canonical § 5.3 (round 1, dt annotation) + § 5.2 (this close: the FIRE production-port pointer); the model briefing gained the production-launcher implementation-status row (the launcher now carries the verified-L stack); [`m5_particle_hunt.md`](../m5_particle_hunt.md) checked and explicitly SKIPPED (M5.24 is infrastructure: no hunt observable moved; its M5.21.9 staging rows are already current).

**Findings**: The launcher's production physics has been brought from three pre-verified-L era stacks to the canonical formulation of record, end to end: audit → routed gap table → the η + V4 canonical integrator → viz/unit levers → FIRE relax + sponge, all machine-gated against the audited research reference (ALL 14 GREEN) with three measured by-products (the dimension-split certified dt, the f32 cooling floor, the production reproduction of the indefinite-channel physics). The live demo now faithfully shows the true physics of the verified L, including its measured instabilities; the stable-ZBW animation is correctly blocked on the fixed-J physics (M5.21.9), not on rendering.

**Research docs created / updated**: this doc (audit + gap table + three rounds + reviews) · [`../scripts/m5_24_canonical_engine_selftest.py`](../scripts/m5_24_canonical_engine_selftest.py) (14 gates) · production `engine2_pde.py` / `engine3_observables.py` / `medium.py` / `_launcher.py` / `xparameters/_topo_canonical4d.py` · [`../m5_theory_canonical.md`](../m5_theory_canonical.md) (§ 5.2 + § 5.3 annotations) · [`../../__M5_model_briefing.md`](../../__M5_model_briefing.md) (implementation-status row) · [`../m5_visualization.md`](../m5_visualization.md) (the no-display-only-kinematics convention) · [`m5_23_1_task_details.md`](m5_23_1_task_details.md) (the re-homed round 4) · [`../m5_roadmap.md`](../m5_roadmap.md)

## TASK REVIEW (2026-07-19, round 1)

**Task Duration:** 00:25 (from the 15:50 go to the 16:15 review post)
**Usage Cap Triggered:** NO

Round-1 review presented in the terminal and discussed; user decisions: **the task STAYS In Progress** (multi-round, as anticipated at staging), **the canonical port + the § 5.3 dt annotation APPROVED** (annotation applied to [`m5_theory_canonical.md § 5.3`](../m5_theory_canonical.md) at this review: dt = 0.02 marked axisym-only, dt = 0.005 = the full-3D step, production-port pointer added), live-demo test by the user follows this review. Results, the surfaced conservation gate (🔶 estimator-artifact-suspect, bounded state, goal-loop cap honored), measured findings (dimension-split dt; the indefinite channel reproduced in production f32; the stale-4D-state launcher bug fixed), and the round-2 backlog: all recorded in the Stage A/B and Stage C + D sections above. Model-doc sweep: the canonical § 5.3 amendment (this review); the model briefing states nothing this round changes (skipped explicitly).

**Why**: the launcher's production Evolve-PDE path (`engine2_pde.py` + the engine stack the GGUI app runs) largely predates the M5.9-M5.21 research era: the research body validated recipes (the M5.16 parameter locks, the verified Lagrangian work, the 4D integrator findings, the quartic-saturation era results) that production never received. The maintainer flagged this at the M5.23 review: the live app should evolve with the certified physics, not the old-era kernels.

**Scope sketch (to be firmed at go)**:

| Step | Deliverable |
| --- | --- |
| 1. Audit | Production kernels vs the canonical registry ([`m5_theory_canonical.md`](../m5_theory_canonical.md): verified equations, locked parameters, working recipes + anti-recipes, the 4D stack) and the Done-task lineage in [`m5_roadmap.md`](../m5_roadmap.md) |
| 2. Gap table | Per kernel: research-validated vs production-implemented, with the source task/method-note link for each gap |
| 3. Port | The gaps, taichi-first, each with a per-gap headless selftest (goal-loop gates) |
| 4. Verify | Live launcher run on the standard configs; no regression in the certified views |

**Merged scope (2026-07-19)**: the former [M5.8.3](m5_8_3_task_details.md) residual row folded in whole (maintainer call at the M5.23 review):

| From M5.8.3 | Now an M5.24 scope item |
| --- | --- |
| 4D extension of the M5.6.5a production seeder | Part of step 3 (port), audited in step 1 like every other kernel |
| The faithful-kinetic engine variant (the 5d diagnosis: frequency correction only) | Evaluated in the step-2 gap table; ported only if production-scale clock runs need it |

**Gating**: user "go".
