# M5.11 vortex-loop findings , regularized, stable, stationary loop + mass (in progress)

> **Status: PENDING EXECUTION.** Skeleton to be filled phase by phase as M5.11 runs. Plan + physics:
> [`11a_vortex_loop.md`](11a_vortex_loop.md). This document mirrors the [`10e`](10e_findings_N4c.md) structure
> (headline, phases, tables, figures, caveats, artifacts) and is the canonical record of the dynamical
> vortex-loop simulation that answers Duda's 2026-06-22 critique. Code: `sandbox_v11/`.

## 0. Headline (to fill)

The deliverable: a CLOSED topological vortex loop, regularized by the full Landau-de Gennes potential, relaxed
to a STATIONARY solution, proven STABLE, with its MASS read off and the mixing re-derived on the real loops.
_Headline scorecard + figure pending P3-P5._

## 1. Phase results

| Phase | Result | Gate | Status |
| --- | --- | --- | --- |
| P0 infrastructure (engine + minimizer validation) | , | engine `V_M`/curvature vs analytic; monotone descent | 🚧 pending |
| P1 regularized core (`q=1/2`, size `ξ`, biaxial melt) | , | vs known LdG disclination core | 🚧 pending |
| P2 loop + collapse (`E(R)`, `dE/dL>0`, Derrick) | , | reproduces the shrink; `τ(R)` quantified | 🚧 pending |
| P3 stabilization (B: clock-breather / A: Hopfion) | , | stationary loop at finite radius | 🚧 pending |
| P4 stability proof (Hessian / real-time evolution) | , | no collapse over >> one clock period | 🚧 pending |
| P5 masses from loop-length-density | , | reproduce mass ratios + `Δm²` hierarchy | 🚧 pending |
| P6 mixing re-derived on real loops | , | PMNS from stable solutions (vs `10e`) | 🚧 pending |

## 2. The rounds (canonical implementation + result) , to fill per phase

_Each phase: the functional/method, the run, the validation against the known result, the figure._

## 3. Summary tables , to fill

## 4. Caveats, open questions , to fill (Derrick route taken; stable-vs-metastable; mass tension; the Duda question if asked)

## 5. Reproducibility , to fill (env, per-phase run commands, resolution-convergence, determinism)

## 6. Artifact index , to fill (`sandbox_v11/` scripts, summaries, figures, checkpoints)

## Cross-refs

[`11a_vortex_loop.md`](11a_vortex_loop.md) (plan) · [`10e_findings_N4c.md`](10e_findings_N4c.md) (the
symmetry/overlap result this makes dynamical) · [`sandbox_v8`](sandbox_v8/)/[`sandbox_vn`](sandbox_vn/) (the
M5.8 clock-breather machinery) · [#199](https://github.com/openwave-labs/openwave/issues/199) ·
[#236](https://github.com/openwave-labs/openwave/issues/236) (HELD).
