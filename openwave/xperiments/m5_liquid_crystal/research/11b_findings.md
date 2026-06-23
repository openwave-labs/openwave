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
| P0 infrastructure (`V_M` ON, scales, energy MINIMIZER) | gradcheck 1.4e-7 · `V_M`=engine formula to 3.5e-15 (g-decoupled exact) · φ⁴ kink → 0.94258 vs 0.94281 (monotone, profile=tanh to 1e-4) · 3D hedgehog relax drops `\|∇E\|` 5.2 decades | monotone descent to `\|δE/δM\|→0`; `V_M` vs analytic LdG | ✅ ALL PASS |
| P1a reproduce FABER's electron (single hedgehog rest energy) | from a GENERIC seed the minimizer → Faber `arctan(u)` to 5.3e-6; dimensionless `I=π/4` to **6e-6**; **E0=511.00 keV at r0=2.2132 fm**; `α_f ℏc=1.43996 MeV·fm`; FIRE=L-BFGS to 1e-11 | `I[a]→π/4` (non-circular); E0 at Faber `r0` | ✅ PASS |
| P1b-foundation (full 3D SU(2) curvature machinery) | 3D-lattice cumulative energy `r<5r0` converges `O(a²)` (−5.24→−3.07→−1.69→−0.90%), Richardson → **−0.016%** of exact; `Γ_i=q0∂q⃗−∂q0 q⃗+q⃗×∂q⃗`, `R_ij=Γ_i×Γ_j` validated (the loop + dipole foundation) | 3D `Γ/R` reproduces the radial energy | ✅ PASS |
| P1b-dipole Coulomb / running `α(d)` (two-soliton) | , | `α_sol ℏc≈1.4387 MeV·fm`; `α⁻¹≈137`; `α(d)` running | 🚧 next (axisymmetric 2D lattice) |
| P2 the vortex LOOP (seeder + relax under full functional) | , | `δE/δM→0` at finite `R`; no collapse (Skyrme evades Derrick) | 🚧 pending |
| P3 stability + the clock (Hessian / evolution + M5.8 dressing) | , | no collapse mode; loop persists; clock lowers energy, `ω` measured | 🚧 pending |
| P4 mass from the loop (field energy / loop-length density) | , | mass spectrum + `Δm²` hierarchy; the 6.2 pm scale | 🚧 pending |
| P5 parameter search (Higgs `A,B,C`/`Λ`, `g,δ`) , Duda's assignment | , | parameters reproducing masses + Faber electron | 🚧 pending |
| P6 mixing re-derived on real relaxed loops | , | PMNS from stable solutions (vs `10e`) | 🚧 pending |

## 2. The rounds (canonical implementation + result) , to fill per phase

_Each phase: the functional/method, the run, the validation against the known result, the figure._

### P0 , the minimizer + V_M/LdG validation ✅ ([`sandbox_v11/v11_p0_minimizer.py`](sandbox_v11/v11_p0_minimizer.py))

Built the piece the leapfrog engine lacks: a true static energy MINIMIZER (FIRE + scipy L-BFGS to `‖δE/δM‖→0`), with the production functional mirrored exactly in f64 numpy , `U(M)=c²·4Σ_{μ<ν}‖[∂_μM,∂_νM]‖²_F + V_M`, `V_M=a·Tr(M²)−b·Tr(M³)+c·(TrM²)²` (engine2_pde.py:280/291), flux `G_α=8Σ_ν[[M_α,M_ν],M_ν]` so `δU/δM=dV_M−c²·div(G)`. The relax route also sidesteps the V-on leapfrog instability (the known M5 roadblock). Four gates, all PASS:

| Gate | Result |
| --- | --- |
| gradcheck (analytic `δU/δM` vs finite-diff) | max rel err **1.4e-7** (25 checks) |
| `V_M` reproduces the engine LdG formula | indep einsum match 3.5e-15; g-axis decoupled exactly; hand ref 4.8 ✓ |
| φ⁴ kink (independent textbook minimizer check) | E→**0.94258** vs analytic `2√2/3=0.94281` (0.024%); profile=`tanh(x/√2)` to 1e-4; monotone |
| 3D hedgehog relax (engine end-to-end) | `‖∇E‖` drops **5.2 decades**, energy 65.8 → −12.4 |

### P1a , FABER's electron reproduced ✅ ([`sandbox_v11/v11_p1_faber_electron.py`](sandbox_v11/v11_p1_faber_electron.py), fig [`p1_faber_electron.png`](sandbox_v11/p1_faber_electron.png))

Faber & Golubich 2026 (arXiv:2604.12021) SU(2) soliton: `Q=cosα−iσ·n̂ sinα`, `ℒ=−(α_f ℏc/4π)(¼R⃗·R⃗+Λ)`, `Λ=q0⁶/r0⁴`, electron = hedgehog `n̂=r̂`, `α=arctan(r/r0)`, `E0=(α_f ℏc/r0)(π/4)` (Eq. 8), calibrated `r0=2.2132 fm → 511 keV` (Eq. 10). Derived the radial reduction by hand and verified it analytically:

```text
G_ij = Γ_i·Γ_j = α'² r̂_i r̂_j + (sin²α/r²)(δ_ij − r̂_i r̂_j)
u_curv = ¼[(Tr G)² − Tr(G²)] = α'² sin²α/r² + sin⁴α/(2r⁴)
E = (α_f ℏc/r0) ∫₀^∞ [ (dα/du)² sin²α + sin⁴α/(2u²) + u² cos⁶α ] du ,   u=r/r0
I[arctan u] = 2·(π/16) + ½·(π/4) = π/4   (exact)
```

The reproduction is **non-circular**: minimize `I[α]` from a GENERIC seed (a stretched tanh, NOT arctan), boundary-pinned, with the P0 relaxer. Result:

| Quantity | Value | Target |
| --- | --- | --- |
| relaxed profile vs `arctan(u)` | max err **5.3e-6** | Faber Eq. 7 |
| dimensionless `I_min` (FIRE) | **0.7854029** | `π/4 = 0.7853982` (6e-6) |
| FIRE vs L-BFGS agreement | 1e-11 | independent minimizers |
| descent | monotone, `I`: 0.880 (seed) → 0.78540 | , |
| `E0` at `r0=2.2132 fm` | **511.00 keV** | `m_e c²` (6e-6) |
| `r0` for 511 keV | **2.21322 fm** | Faber `2.21321 fm` |
| `α_f ℏc` (from constants) | **1.43996 MeV·fm** | `e0²/4πε0` |

The truncated Coulomb tail (the `1/(2u_max)≈0.31%` deficit) is added analytically, faithful to Faber's own exterior-energy term `H_out` (his Eq. 24). 511 keV at `r0=2.2132` is Faber's CALIBRATION; the physics content is the relaxed profile + `I=π/4` from the functional. The non-trivial fine-structure result (`α_sol`, `α⁻¹≈137`) is the two-soliton dipole → P1b.

### P1b-foundation , the full 3D SU(2) curvature machinery ✅ ([`sandbox_v11/v11_p1b_lattice.py`](sandbox_v11/v11_p1b_lattice.py))

P1a validated the radial REDUCTION; this validates the genuinely-3D curvature machinery the vortex LOOP (P2) and the dipole both need, with NO symmetry assumed. Build the smooth fields `q0=cosα`, `q⃗=n̂ sinα` on a 3D Cartesian grid, finite-difference, and form the singularity-free `Γ_i=q0 ∂_i q⃗ − (∂_i q0) q⃗ + q⃗×∂_i q⃗` (Faber Eq. 6), `R_ij=Γ_i×Γ_j`, `u_curv=¼[(TrG)²−TrG²]`. Tail-free check: the lattice cumulative energy inside `r<5 r0` vs the exact radial `4π·I_radial(<5)`:

| spacing `a` (r0) | lattice `J̃(<5)` | vs exact |
| --- | --- | --- |
| 0.40 | 8.13311 | −5.24% |
| 0.30 | 8.31923 | −3.07% |
| 0.22 | 8.43794 | −1.69% |
| 0.16 | 8.50555 | −0.90% |
| Richardson `O(a²)` extrap | **8.58146** | **−0.016%** (target 8.58286) |

Clean second-order convergence to the exact energy , the 3D `Γ/R` machinery is correct. (My 2nd-order central scheme UNDER-shoots by a few % at coarse `a`, the opposite sign of Faber's high-order-stencil OVERestimate, but the controlled `O(a²)` convergence is the validation.) Full soliton `J̃=π² → 510.9 keV`.

## 3. Summary tables , to fill

## 4. Caveats, open questions , to fill (Derrick route taken; stable-vs-metastable; mass tension; the Duda question if asked)

## 5. Reproducibility , to fill (env, per-phase run commands, resolution-convergence, determinism)

## 6. Artifact index

| Artifact | What |
| --- | --- |
| [`sandbox_v11/v11_p0_minimizer.py`](sandbox_v11/v11_p0_minimizer.py) | P0: FIRE + L-BFGS energy minimizer + the production functional in f64 numpy; gates gradcheck / vm_ldg / phi4 / hedgehog |
| [`sandbox_v11/v11_p1_faber_electron.py`](sandbox_v11/v11_p1_faber_electron.py) | P1a: Faber radial soliton, generic seed → `arctan`, `I→π/4`, 511 keV |
| [`sandbox_v11/v11_p1b_lattice.py`](sandbox_v11/v11_p1b_lattice.py) | P1b-foundation: full 3D SU(2) `Γ/R` curvature machinery, `O(a²)` convergence |
| [`sandbox_v11/p1_faber_electron.png`](sandbox_v11/p1_faber_electron.png) | P1a figure: seed → relaxed = arctan profile + energy-per-shell |
| `sandbox_v11/_checkpoints/p0_minimizer.json`, `p1_faber_electron.json`, `p1b_lattice.json` | per-phase result checkpoints (small JSON, regenerable) |

No raw data files > 1 MB were written (all outputs are small JSON + one PNG).

## Cross-refs

[`11a_vortex_loop.md`](11a_vortex_loop.md) (plan) · [`10e_findings_N4c.md`](10e_findings_N4c.md) (the
symmetry/overlap result this makes dynamical) · [`sandbox_v8`](sandbox_v8/)/[`sandbox_vn`](sandbox_vn/) (the
M5.8 clock-breather machinery) · [#199](https://github.com/openwave-labs/openwave/issues/199) ·
[#236](https://github.com/openwave-labs/openwave/issues/236) (HELD).
