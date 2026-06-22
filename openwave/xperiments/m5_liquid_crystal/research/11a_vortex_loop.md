# M5.11 , the regularized, stable, stationary topological vortex loop (the real simulation)

> **Purpose.** Build the simulation Dr. Jarek Duda expects for a neutrino: a CLOSED topological vortex loop in
> the Landau-de Gennes substrate, REGULARIZED by the full potential, RELAXED to a STATIONARY solution, proven
> STABLE, with its MASS read off from the loop , and only then the mixing recomputed on the real (relaxed)
> loops. This is the dynamical counterpart to the symmetry/overlap result of
> [`10e_findings_N4c.md`](10e_findings_N4c.md), and it directly answers Duda's 2026-06-22 critique (below).
> Findings as we execute: [`11b_findings.md`](11b_findings.md). Code + per-phase findings + checkpoints:
> `sandbox_v11/`. Convention: index-0 (`D = diag(g, 1, δ, 0)`, `eta = diag(-1,1,1,1)`),
> [`_convention_refactor/CONVENTION.md`](_convention_refactor/CONVENTION.md).

## 0. The challenge (Duda, 2026-06-22, verbatim)

> "I am trying to read these Python files, but they look very far from simulations I was expecting. E.g. for
> neutrinos require topological vortex loop configurations, with main problem being regularization requiring
> potential ... But I don't see anything like this there , for me the code is much too simple ... while these
> results look great, personally I am very far from trusting them."

He is right. The [`sandbox_v10`](sandbox_v10/) code computes flavour mixing from SEEDED loop ansatze via
symmetry + energy-overlap matrices (a Gram matrix of unrelaxed configurations); it never regularizes, relaxes,
or stabilizes a loop. The mixing angles turned out to be symmetry-determined (μτ + magic), substrate-independent
(N4b/N4c), so the code is genuinely simple , because the mixing does not need the dynamics. The HARD physics
he names , a regularized, stable, stationary loop under the full potential, and the mass that follows , is
exactly what M5.11 must do. **Goal: a simulation a working physicist trusts. No cut corners.**

## 1. The order parameter and the full energy functional

The substrate order parameter is the symmetric tensor field `M(x)` (index-0: `M = O D O^T`,
`D = diag(g, 1, δ, 0)`; the spatial nematic block is `M[1:4,1:4]`, the time/g axis is index 0). For the loop
problem the relevant sector is the spatial nematic block `Q(x)` (a real symmetric traceless-ish 3x3), with the
g/time axis carried along for the boost/clock sector (Phase 3B).

The energy functional we relax under (the "full potential" Duda means) is the standard Landau-de Gennes free
energy , elastic (Frank) + bulk (Landau) , plus, where Derrick requires it (§2), a higher-order term:

```text
E[Q] = INT_V  [  f_elastic  +  f_bulk  ( +  f_stab ) ]  dV

  f_elastic = (L1/2) (∂_k Q_ij)(∂_k Q_ij)  [ + L2 (∂_j Q_ij)(∂_k Q_ik) + L3 Q_kl (∂_k Q_ij)(∂_l Q_ij) ]
  f_bulk    = (A/2) Tr(Q^2)  −  (B/3) Tr(Q^3)  +  (C/4) (Tr Q^2)^2          (A = a0 (T − T*))
  f_stab    = (κ/4) (Faddeev-Skyrme term)  OR  the dynamical clock charge (§3)   ← the Derrick-evading term
```

**Regularization (Duda's "main problem").** The bulk potential is what regularizes the core. At a disclination
the director cannot wind smoothly through a point, so the gradient energy would diverge; the bulk term lets `Q`
**melt** at the core (drop toward the isotropic/uniaxial-to-biaxial state), trading gradient energy for bulk
energy and setting a finite core size, the nematic coherence length `ξ = sqrt(L/|A|)`. The core is biaxial (the
Schopohl-Sluckin / Lyuksyutov ring-disclination structure), not a true singularity. This is the regularized
core M5.11 must produce and validate against the known LdG disclination solution.

The defect: a `q = 1/2` disclination LINE (director winds by `π` around the core) bent into a CLOSED LOOP. Three
neutrino flavours = the same regularized loop at three SO(3) orientations (the [#199](https://github.com/openwave-labs/openwave/issues/199)
structure), but now REAL relaxed solutions, not seeds.

## 2. The central obstruction: Derrick's theorem (why the bare loop collapses)

For `E = INT (|∇Q|^2 + V(Q))` in 3D, under the scaling `x → λx`: the gradient energy scales as `λ^(3−2) = λ`,
the bulk energy as `λ^3`. `dE/dλ|_{λ=1} = E_grad + 3 E_bulk = 0` needs `E_grad = −3 E_bulk` , impossible for
`E_grad, E_bulk ≥ 0`. So **no stable static finite-size soliton exists from elastic + potential alone**; a free
disclination loop has positive line tension and collapses (precisely our measured `dE/dL = +6.74 > 0`,
[`sandbox_v10/n2_closed_loop.py`](sandbox_v10/n2_closed_loop.py)). This is not a bug , it is a theorem, and it
is the crux Duda is pointing at. A stable stationary loop REQUIRES a Derrick-evading mechanism:

| Route | Mechanism | Evades Derrick by | Particle picture |
| --- | --- | --- | --- |
| **A , static Hopfion** | add a Faddeev-Skyrme `(∂n × ∂n)^2` term (`f_stab`) | the 4th-order term scales as `λ^(−1)` , outward pressure balances collapse | a knotted/linked field with a Hopf charge `Q_H ∈ ℤ`; Vakulenko-Kapitanskii bound `E ≥ c\|Q_H\|^{3/4}` |
| **B , dynamical breather** | a conserved internal rotation/oscillation `Q = Q(x; ωt)` (the clock) | the `ω^2` kinetic term acts as outward pressure (Q-ball mechanism) | the loop CANNOT sit static, so it OSCILLATES (Zitterbewegung, `ω = 2mc²/ℏ`) , a time-crystal breather |

**Route B is our M5.8 result.** The whole [`sandbox_v8`](sandbox_v8/) + [`sandbox_vn`](sandbox_vn/) body (now
index-0) is the demonstration, at the energy level, that the boost-dressed oscillating defect (the negative-energy
`(0,α)` clock fuel) beats the static one , "a defect that can't relax → oscillates" (the M5 course, lesson 7).
M5.11 makes this rigorous in full 3D for a CLOSED LOOP. So we are not guessing the stabilization , we have a
physically-grounded candidate (B), and we will also test (A) as the static control.

## 3. The two stabilization routes (both built, honestly compared)

| | Route A (static Hopfion) | Route B (clock-breather) , the M5.8 route |
| --- | --- | --- |
| Add to functional | Faddeev-Skyrme `f_stab` | boost-dressing + the clock charge `ω` (M5.8.2a machinery, already index-0) |
| Stationary means | `δE/δQ = 0` (true static minimum at finite size) | stationary in the co-rotating/oscillating frame; `E(ω) = A + ω² C` minimized at finite `ω*` with the cap |
| Stability proof | Hessian `≥ 0`; survives perturbation | survives real-time Minkowski evolution over many clock periods without collapse/dispersal |
| Mass | `E[Hopfion]` | the dressed oscillating loop energy `E(ω*, b*)` |
| Risk | the Faddeev term may not be "the M5 potential" Duda means | the breather may be only metastable / hard to make truly stationary in 3D |

We build the regularized core + the loop + the collapse demonstration FIRST (route-independent), then test B
(our bet) and A (the control), and report which yields a genuine stable stationary loop.

## 4. The phased plan (each phase gated against a known result; no phase advances on an unvalidated prior)

| Phase | What | Validation gate (how we KNOW it is right) |
| --- | --- | --- |
| **P0 , infrastructure** | reuse the index-0 engine (`medium`/`engine2_pde` `V_M`, curvature, the 4D gradient flow) + the N1 graded-precision method + FIRE/L-BFGS minimizer. | the engine's `V_M`/curvature reproduce the analytic LdG free energy on a uniform + a single hedgehog (vs `n0`/`n1`); minimizer descends monotonically. |
| **P1 , regularized core** | the `q = 1/2` straight-disclination core: 2D radial relaxation under elastic + bulk → the biaxial melted core, size `ξ`. | matches the known LdG disclination core (biaxial ring, `ξ = sqrt(L/\|A\|)`, the Schopohl-Sluckin profile); core energy per length finite. |
| **P2 , the loop + the collapse** | seed a closed loop radius `R`, relax under the FULL functional (no `f_stab`); measure `E(R)`, the line tension, the shrink. | reproduces `dE/dL > 0` (Derrick); quantifies `τ(R)` and the collapse trajectory under gradient flow. This is the PROBLEM, rigorously stated. |
| **P3 , stabilization (the crux)** | Route B: boost-dress + clock, find the stationary oscillating loop (M5.8 mechanism in 3D). Route A: add the Faddeev-Skyrme term, relax to a static Hopfion. | a loop of FINITE radius that is stationary: `δE/δQ = 0` (A) or a stationary breather `∂E/∂ω = 0` at finite `ω*` (B). |
| **P4 , stability proof** | Hessian spectrum (route A) and/or real-time Minkowski evolution (route B) over many periods. | no negative collapse mode (A); the loop persists, does not shrink to 0 or disperse, over >> one clock period (B). THE proof Duda wants. |
| **P5 , masses from the loop** | `E[stable loop] = mass`. Vary the stabilizing charge (Hopf number `Q_H` / twist / `ω`-family) → a mass spectrum. | reproduce the lepton/neutrino mass RATIOS and the `Δm²` hierarchy , the N4c near-term tension (spectrum was ~6x too compressed); honest pass/fail vs data. |
| **P6 , close the loop with mixing** | recompute the PMNS overlap on the REAL relaxed stable loops at 3 SO(3) orientations (not seeds). | the mixing of [`10e`](10e_findings_N4c.md) re-derived from stable solutions; report whether the angles hold and whether the substrate now does work. |

P0-P2 are standard LdG numerics (tractable, days). **P3 is the research crux** , the honest unknown. P4-P6
follow only if P3 yields a stable loop.

## 5. Numerics and precision

| Item | Choice |
| --- | --- |
| Functional + fields | the index-0 M5 engine (`engine2_pde.V_M`, signed curvature) + the LdG elastic; the N1 graded-`δ` method for the `g/δ ~ 1e20` range where the boost sector enters (P3B). |
| Relaxation | accelerated gradient flow (FIRE) / L-BFGS to a stationary point; Newton polish near the solution; `\|δE/δQ\|` to tolerance. |
| Real-time (P4B) | the M5.8 constrained 4D Minkowski integrator (`sandbox_v8` constrained path), now index-0, over many clock periods. |
| Hardware | Taichi/Metal GPU for the 3D relax + evolution (the heavy phases); numpy f64 + the N1 method where precision dominates; full machine, multi-hour runs as needed. |
| Grids | core (P1) on a fine 2D mesh; loop (P2-P4) on `48³–96³` (resolution-converged: every reported number checked vs a finer grid). |
| Determinism | no `Date.now`/random; seeds + params logged; every artifact regenerable. |

## 6. Validation philosophy (the trust-rebuilder)

- Every phase is gated against an ANALYTIC or KNOWN result (the disclination core profile, the Derrick scaling,
  the Vakulenko-Kapitanskii bound, the M5.8 energy gates). A phase does not advance on an unvalidated prior.
- Resolution convergence on every reported number (coarse vs fine grid).
- Honest failure: if P3 yields no stable loop, we report that plainly (a real negative result), not a dressed-up
  partial. If the masses stay in tension (P5), we say so.
- No symmetry shortcuts standing in for dynamics: the loops in P3-P6 are RELAXED solutions, not seeds.

## 7. Do we ask Duda first, or figure it out ourselves?

**Recommendation: figure it out ourselves , then ask one sharp, competence-demonstrating question if P3 stays
open.** Reasoning:

- P0-P2 (regularized core + loop + collapse) are standard LdG numerics. Asking how to do them would reinforce
  the "can't do serious simulation" impression. DOING them , showing a real regularized disclination core that
  matches the known solution , flips that impression immediately. Show, don't ask.
- The ONE genuinely-open physics fork is the §2 Derrick choice: static-Hopfion (Skyrme term) vs dynamical-breather
  (the clock). We already have a grounded answer (Route B, our M5.8 result), and we will build BOTH. So we can
  proceed without him.
- IF, after P1-P3, the route remains genuinely ambiguous, the reserved question for Duda is sharp and PROVES we
  understand the problem: *"Derrick's theorem forbids a stable static loop from elastic + LdG bulk alone; we get
  the regularized core and confirm the collapse. Do you stabilize via a Faddeev-Skyrme term (static Hopfion) or
  via the internal Zitterbewegung oscillation (dynamical breather)? Our M5.8 work points to the latter."* That
  question, backed by working code, rebuilds trust far better than asking up front.

So: no question before we start. We build P0-P2, and the answer to "what stabilizes it" is the experiment.

## 8. Honest risks / unknowns

| Risk | Honest status |
| --- | --- |
| P3 may not yield a truly STABLE loop (only metastable / long-lived) | likely outcome for the breather; we will state metastable-vs-stable precisely (lifetime in clock periods). |
| "The full potential" Duda means may include a term we are not assuming (e.g. a chiral/Lifshitz or a Faddeev term) | we test both the plain LdG (Route B) and LdG + Skyrme (Route A); the reserved question pins his intent. |
| The masses (P5) may stay in tension with `Δm²` | the N4c spectrum was ~6x too compressed; the stable-loop family may or may not fix it. Reported honestly either way. |
| Cost: P3-P4 are multi-hour GPU research runs | accepted; full machine, run for the time it needs. |

## 9. Deliverables and structure

| Artifact | Where |
| --- | --- |
| This plan | `11a_vortex_loop.md` |
| Findings (filled per phase) | [`11b_findings.md`](11b_findings.md) (mirrors the `10b`/`10e` structure: headline, phases, tables, figures, caveats, artifacts) |
| Code (one script per phase) + per-phase findings + checkpoints | `sandbox_v11/` (`v11_p0_*`, `v11_p1_core`, `v11_p2_loop`, `v11_p3a_hopfion`, `v11_p3b_breather`, `v11_p4_stability`, `v11_p5_mass`, `v11_p6_mixing`) |
| Figures | core profile, `E(R)` collapse, the stationary loop, the stability evolution, the mass family, the re-derived mixing |

## Cross-refs

[`10e_findings_N4c.md`](10e_findings_N4c.md) (the symmetry/overlap result this makes dynamical) ·
[`10a_neutrino_oscillations.md`](10a_neutrino_oscillations.md) (master plan) ·
[`sandbox_v8`](sandbox_v8/) + [`sandbox_vn`](sandbox_vn/) (the M5.8 clock-breather machinery, index-0) ·
[#199](https://github.com/openwave-labs/openwave/issues/199) (the SO(3) loop structure) ·
[#236](https://github.com/openwave-labs/openwave/issues/236) (HELD). M5 course lesson 7 (the de Broglie
clock-engine, "can't relax → oscillates").
