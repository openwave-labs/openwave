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
| P0 infrastructure (`V_M` ON, scales, energy MINIMIZER) | , | monotone descent to `\|δE/δM\|→0`; `V_M` vs analytic LdG | 🚧 pending |
| P1 reproduce FABER's electron (regularized hedgehog) | , | `∫½\|E\|² = 511 keV` at Faber core; running `α(d)`; Coulomb `E(d)` | 🚧 pending |
| P2 the vortex LOOP (seeder + relax under full functional) | , | `δE/δM→0` at finite `R`; no collapse (Skyrme evades Derrick) | 🚧 pending |
| P3 stability + the clock (Hessian / evolution + M5.8 dressing) | , | no collapse mode; loop persists; clock lowers energy, `ω` measured | 🚧 pending |
| P4 mass from the loop (field energy / loop-length density) | , | mass spectrum + `Δm²` hierarchy; the 6.2 pm scale | 🚧 pending |
| P5 parameter search (Higgs `A,B,C`/`Λ`, `g,δ`) , Duda's assignment | , | parameters reproducing masses + Faber electron | 🚧 pending |
| P6 mixing re-derived on real relaxed loops | , | PMNS from stable solutions (vs `10e`) | 🚧 pending |

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
