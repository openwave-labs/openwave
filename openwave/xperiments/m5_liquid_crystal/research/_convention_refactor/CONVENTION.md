# M5 index convention , CANONICAL note (the single source of truth)

> **The M5 engine stores the order parameter with the time/boost axis at array INDEX 0.**
> `D = diag(g, 1, δ, 0)`, `eta = diag(-1, 1, 1, 1)` (g and the Minkowski minus both at index 0).
> This is Duda's convention. Flipped from the legacy index-3 ordering on 2026-06-21.

## The convention (post 2026-06-21 flip)

| Array index | Eigenvalue | Physical axis |
| --- | --- | --- |
| 0 | g (~1e10) | time / boost (gets the `eta` minus) |
| 1 | 1 | EM / principal spatial (the director r̂) |
| 2 | δ (~ℏ) | QM / clock twist axis (e_Θ) |
| 3 | 0 | null axis (e_Φ) |

`D = diag(g, 1, δ, 0)`, `eta = diag(-1, 1, 1, 1)`. The spatial 3×3 block is indices `[1:4,1:4]`;
callers that need the nematic spatial order parameter (V_M, eigen_decompose, the amplitude tracker)
extract that block. `signed_dot4` puts its minus where exactly one index is 0.

## History (why there were two notations)

Before 2026-06-21 the engine STORED `D = diag(1, δ, 0, g)` with the time axis at index 3
(`eta = diag(1,1,1,-1)`), while the physics was always described in Duda's `diag(g,1,δ,0)`
notation. That split (storage index-3 vs notation index-0) is what Dr. Duda repeatedly flagged
(the eta-placement comments, 2026-06-20/21). The two are IDENTICAL physics under the cyclic
relabel `index k -> (k+1) mod 4` (time 3 -> 0); see `sandbox_v10/n0_engine_equivalence.py` Test B.
To end the confusion permanently, the storage was flipped to index-0 so storage and notation match.

## What was flipped, and the proof it is physics-neutral

| Flipped (index-0 now) | Frozen (still index-3, historical record) |
| --- | --- |
| `medium.py`, `engine1_seeds.py`, `engine2_pde.py`, `engine3_observables.py` | `sandbox_v1`..`sandbox_v5` (pre-4×4, 3×3), `sandbox_v7` (3×3 self-contained), the 3×3 files of `sandbox_v6` |
| `sandbox_v9` (#200), `sandbox_v10` (neutrino), **`sandbox_v8` + `sandbox_vn` + `sandbox_v6/m5_6_5c_prod_scale`** (2026-06-22 second-wave flip, see [`01_sandbox_v8_vn_flip.md`](01_sandbox_v8_vn_flip.md)) | (see `FROZEN_SANDBOXES.md`) |

**Proof:** `_convention_refactor/golden_master.py` froze 66 convention-invariant physical outputs
(seeder spectra, energies, stable-mask count, the 4D gradient flow, the constrained clock
integrator, AND the render path: eigen_decompose director + spatial eigenvalues + the amplitude
tracker) PRE-flip, then re-checked POST-flip: **PASS, worst rel diff 3.84e-07; the render
invariants bit-identical**. The #200 ledger re-runs unchanged (boost_GEM dressed = 1.549e-6,
undressed = 0 exactly). `n0_engine_equivalence.py` re-confirms engine == numpy port at index-0.

## For anyone editing the engine (e.g. Dr. Duda)

Write code in `diag(g, 1, δ, 0)` / `eta = diag(-1, 1, 1, 1)` , the array stores it that way.
The time/boost axis is index 0. The spatial nematic block is `[1:4, 1:4]`. To verify a change is
physics-neutral, run `python3 research/_convention_refactor/golden_master.py --check`.

Cross-refs: the flip plan + site checklist = [`00_plan.md`](00_plan.md); the frozen-sandbox list =
[`FROZEN_SANDBOXES.md`](FROZEN_SANDBOXES.md); the engine<->port equivalence =
[`../sandbox_v10/n0_engine_equivalence.py`](../sandbox_v10/n0_engine_equivalence.py).
