# M5.47: Klein-Gordon: the dispersion of small perturbations in a time evolution

> Opened and staged 2026-11-19. Roadmap row: [`m5_roadmap.md`](../m5_roadmap.md). Thread record: [`m5_32_convo.md`](m5_32_convo.md). The cell it reads for: the quantum wave equation (Klein-Gordon) in the M5 table of [`MODELS.md`](../../../../../MODELS.md).

## TASK PLANNING (2026-11-19; the readings of § 5 are frozen before the run)

**Scope.** The run named by the way back of the Klein-Gordon cell, as the cell reads after the re-read of [T11](../../../../../dev_docs/tasks/t11_task_details.md): the dispersion of small perturbations about a converged relaxed state, read in a time evolution, giving ω² = k² + m². The row's test is an emergent ω² = k² + m² for matter waves, and its regime is `dynamic`. The earlier spectral read cited in the cell ([`m5_21_1_method_note.md`](../findings/m5_21_1_method_note.md)) used another action, toy parameters and a sliding background; this run uses the action of [M5.42](m5_42_task_details.md) with its time part, on the relaxed states of [M5.46](m5_46_task_details.md) with the shell held. The run reports; the cell is re-read afterwards, on the bar of `MODELS.md`, in the pull request that lands the record.

**Two limits, stated before the run.** (1) The time part is supplied by this plan (§ 1): the module of M5.42 codes its value on the twist only, so every label of § 5 is conditional on that reading of the action. (2) A property of the base states, computed at staging and derivable from the action for any unit of degree 1, already says that one law ω² = k² + m² with the vacuum mass cannot be met inside boxes of half-width 10 (§ 5, registered numbers). A label R1 NOT HELD on these boxes is that consequence returned by a time evolution; it is not evidence that the model has no Klein-Gordon law at large distance. What the run can show is which law small waves obey on the relaxed unit, and whether the linearized action accounts for it.

**Definition of done.** The time-stepper passes its known-answer controls (§ 4); every job of § 2 ends at its length with its Fourier sums saved; the vacuum jobs return the six branches; each 0.25 grid of the unit carries one label for each of R1 to R3 and each pair of grids one label for R4, or the words "no label" with the rule that withheld it; the findings embed the dispersion plots, the map of the kinetic factor and the energy series; an adversarial audit on a different model has tried to refute each label; the doc checker and the roadmap linter are clean.

**Gating.** The user's go. No answer of the author is waited for.

**Research body.** This document (plan, findings, review); scripts `m5_47_evolve.py`, `m5_47_waves.py`, `m5_47_reads.py` in [`../scripts/`](../scripts/); summaries `m5_47_*.json` in [`../data/`](../data/) (arrays local, indexed by the manifest); plots `m5_47_*.png` in [`../plots/`](../plots/); checkpoints in `../checkpoints/m5_47/`.

### 1. The action with its time part

The static part is the action of [M5.42](m5_42_task_details.md) as coded in [`m5_42_eq42.py`](../scripts/m5_42_eq42.py), at J = 0, with κ = v = 1, δ = 0.05, L₁ = 0.03, c_k = 1 + δ^k, the mirror-averaged stencil:

$$
U = h^3 \sum_n \Big[ L_1 \sum_i \lVert \partial_i M \rVert^2 + \kappa \sum_{i \lt j} \lVert [\partial_i M, \partial_j M] \rVert^2 + v \sum_{k=1}^{3} \big( \operatorname{tr} M^k - c_k \big)^2 \Big]
$$

The time part is the form whose value on the twist is the inertia of M5.42: the rate of M in the L₁ term and in the time-space pairs of the commutator term, with the weights of the same stencil, and no rate in the potential:

$$
T = h^3 \sum_n \Big[ L_1 \lVert \dot M \rVert^2 + \kappa \sum_i \lVert [\dot M, \partial_i M] \rVert^2 \Big]
$$

With the rate set to Ω [G, M] this is ½ I Ω², I the inertia that `m5_42_eq42.kernel` returns. The module of M5.42 codes the inertia and not T for a general rate; the form above is this plan's reading of it (blindspot pass, § 6). The equations of motion are those of T − U, in the momentum p conjugate to the six components of M:

$$
p = \frac{\partial T}{\partial \dot M} = \mathcal{A}(M) \dot M , \qquad \dot p = \frac{\partial T}{\partial M} - \frac{\partial U}{\partial M} , \qquad E = T + U
$$

The operator 𝒜(M) is local: a symmetric positive 6 × 6 matrix per node, built from the differences of M at that node, and solved at every node at every stage. In a uniform field it is 2 h³ L₁ times the metric of the six components, so with L₁ = 0 the vacuum has no time part at all: the run is made with L₁ = 0.03. The integrator is the classical fourth-order Runge-Kutta scheme on (M, p). The nodes of the outer shell are held at their starting values in every job, as in the descents that made the states.

**Small perturbations.** About a state M₀ at rest, M = M₀ + φ obeys, to first order, 𝒜(M₀) φ̈ = −H(M₀) φ, H the second variation of U. In the uniform vacuum this is φ̈ = Δ_h φ − (v V″ / 2 L₁) φ, with Δ_h the seven-point Laplacian of the grid and V″ the Hessian of the potential at the eigenvalues (1, δ, 0). Three of the six directions rotate the frame and cost nothing in the potential; the three others change the eigenvalues, with m² = 0.1057, 60.09 and 473.5 at these values. The lattice law is then exact:

$$
\omega^2 = \hat k^2 + m_\ell^2 , \qquad \hat k^2 = \frac{4}{h^2} \sum_i \sin^2 \frac{k_i h}{2}
$$

This follows from the declared action. The vacuum jobs check the instrument against it and carry no label for the cell.

### 2. The jobs

| Job | State | Half-box | Spacing (N) | Start | Step, length | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| V1 | uniform vacuum, its frame turned by a fixed rotation | 8 | 0.5 (33) | noise in all six components, amplitude 10⁻⁴ | 0.0125, 48 | the six branches; the estimator on a known answer |
| V2 | the same | 6 | 0.25 (49) | the same | 0.0125, 48 | the same at the second spacing |
| V3 | V1 with v = 2 | 8 | 0.5 (33) | the same | 0.0125, 48 | the read of the masses must move with the potential |
| Ua, Ub | base states B0a, B0b of M5.46 | 8 | 0.5 (33), 0.25 (65) | noise in the mid polarization (§ 3), amplitude 10⁻³ | 0.03, 48 | the read on the relaxed unit, two spacings |
| Uc, Ud | base states B0c, B0d | 10 | 0.5 (41), 0.25 (81) | the same | 0.03, 48 | the same on the larger box |
| Na to Nd | the four base states | as above | as above | no perturbation | 0.03, 48 | what the background does by itself |
| Ha, Hb | B0a, B0b | 8 | 0.5, 0.25 | the noise of Ua, Ub at half the amplitude | 0.03, 48 | linearity |
| Da | B0a | 8 | 0.5 | the noise of Ua | 0.015, 48 | the step |
| Sa, Sb | B0a, B0b | 8 | 0.5, 0.25 | a second draw of the noise | 0.03, 48 | the scatter of the read |
| E1 | B0a | 8 | 0.5 | at rest, with the rate Ω [G, M] (zero on the shell), T = 1 % of U | 0.03 and 0.015, 10, then back | energy and reversal at finite amplitude |

The noise is Gaussian white noise from a seeded generator, smoothed by a Gaussian of standard deviation 0.35 in length units, scaled to a unit root mean square over the interior, multiplied by the amplitude, and zero on the shell. The start has no rate (p = 0), except E1. Every wave job accumulates, at every step, the Fourier sums of the six components of M − M₀ at its target frequencies under a Hann window over the whole length (at N = 81 about 0.9 GB per job), and E, T and U every 20 steps. The targets on the unit are ω = 3 to 6 in steps of 0.25 and 6.5, 7, 7.5, 8. The vacuum jobs add 8.25, 8.5, 8.75, 9, 10 and 11, and keep at every step the time series of twenty sine modes of the box, sin(n_i π (x_i + L)/2L) on each axis with (n_x, n_y, n_z) = (1,1,1), (2,1,1), (2,2,1), (2,2,2), (3,1,1), (3,2,1), (3,3,1), (3,3,3), (4,1,1), (4,2,2), (4,4,1), (4,4,4), (5,1,1), (5,3,1), (5,5,2), (6,1,1), (6,3,2), (6,6,1), (7,2,1), (8,1,1). The jobs Ha, Hb, Da, Sa and Sb are read against the twins Na and Nb.

**Parallel shape.** One single-threaded process per job (thread variables set to 1 before NumPy loads), the largest grids first, at most twelve at once: sixteen wave jobs, then E1 on a free core. Inside a job the force, the gradient of T and 𝒜 are built in one pass over the nodes per stage. The reads run after the jobs, one process. The GPU is not used: the action needs f64. No job is sequential by default; the only serial step is the controls K1 to K5 before the pool.

### 3. The read on the relaxed unit

**The polarization.** n(x) is the eigenvector of the largest eigenvalue of the background. In the two directions E₁ = n nᵀ and E₂ = (1 − n nᵀ)/√2 the vacuum Hessian has the eigenvalues m² = 60.08 and 473.4, within 0.02 % of the exact mid and heavy branches; W(x) = 0.1278 E₁ − 0.9918 E₂ is the direction of the lower one, written with n alone because the two small eigenvalues are 0.05 apart and their axes are not a stable frame. The start of the unit jobs is the noise times W(x), and the read is the scalar a(x, ω) = ⟨W(x), A(x, ω)⟩, A the six-component Fourier sum of the perturbed job minus that of its unperturbed twin. The heavy branch is not read on the unit (at step 0.03 the scheme damps it), and the light branch is not read there either: its mass, 0.1057, is of the size of the 1/r² terms of a frame that turns once around the box, so these boxes cannot resolve it.

**The local wavenumber.** On a set B of nodes, with Δ_h the seven-point Laplacian:

$$
K^2(B, \omega) = - \frac{\sum_{x \in B} \operatorname{Re} \big( \bar a \quad \Delta_h a \big)}{\sum_{x \in B} \lvert a \rvert^2}
$$

It is a property of the pattern of the wave alone and uses nothing of the action beyond the grid. A fit point is a pair (B, ω) with 1 ≤ K² ≤ 4/h² (a third of the band top of the grid): 16 at spacing 0.5, 64 at 0.25. Its radial share is ρ = Σ_B |r̂ · ∇a|² / Σ_B |∇a|², with central differences and r̂ taken from the center of the loop; K_r² = ρ K² and K_t² = (1 − ρ) K².

**The read region and the bins.** The nodes at least 1.5 from the loop line and at least the loop's radius from its center (center, radius and normal from the reader of [M5.44](m5_44_task_details.md) on the base state), at least 1.5 inside every face, and with the two largest eigenvalues of the background at least 0.5 apart. In the vacuum jobs the set is every node at least 1.5 inside the faces, one bin. On the unit the nodes are binned by the kinetic factor q(x): the kinetic density of the rate W at the node, with the differences and weights of the kernel, over its vacuum value L₁,

$$
q(x) = 1 + \frac{\kappa}{L_1} \sum_j \tfrac{1}{2} \Big( \lVert [W, \partial_j^{+} M_0] \rVert^2 + \lVert [W, \partial_j^{-} M_0] \rVert^2 \Big)
$$

with fixed edges 1.25, 1.5, 1.75, 2, 2.25, 2.75, 3.25, 3.75, 4.25, 5, 6, 7, the same on every grid. A bin with fewer than 500 nodes on a grid is not read there. The same tables are also given on shells of r of width 0.5, with no label.

**The fits.** Per bin, the least-squares line ω² = c² K² + m² over its fit points, with four points at least; fewer is no fit. With six points or more, also the plane ω² = c_r² K_r² + c_t² K_t² + m². The product m² q̄, q̄ the mean of q over the bin, is reported per bin beside 60.09.

**What the linearized action says, for comparison.** For a wave a(x) cos ωt in the polarization W, with C_j = [W, ∂_j M₀] (central differences) and V″_WW the second derivative of the potential along W at the local M₀, the quadratic action gives the quotient

$$
\omega_{\mathrm{act}}^2(B) = \frac{\sum_{x \in B} \Big[ L_1 \lvert \nabla a \rvert^2 + \kappa \Big( \sum_j \lVert C_j \rVert^2 \lvert \nabla a \rvert^2 - \big\lVert \sum_j C_j \partial_j a \big\rVert^2 \Big) + \tfrac{1}{2} v V''_{WW} \lvert a \rvert^2 \Big]}{\sum_{x \in B} L_1 q \lvert a \rvert^2}
$$

so that a wave along a direction e has the speed c² = 1 − κ ‖Σ_j e_j C_j‖² / (L₁ q) and the local mass m² = v V″_WW / (2 L₁ q): the law is that of an effective metric, with one speed along the gradient of the frame and another across it. Left out of the quotient: the terms in the gradient of W, the curvature term [∂_i M₀, ∂_j M₀], and the exchange with the other polarizations. The last one is not small in the kinetic form: W and the heavy direction share the commutator with ∂_j M₀, and the two-direction problem in E₁, E₂ at q = 4 gives a local mass of 13.8 where the quotient gives 15.0. The quotient is therefore expected above ω² by up to about 6 % near a cutoff. The share of the Fourier power outside W is reported per target.

### 4. The controls

| Control | Input | The answer it must return | What a failure would look like |
| --- | --- | --- | --- |
| K1 | the seed and B0a, the rate Ω [G, M] | T = ½ I Ω² with the I of `m5_42_eq42.kernel`, to 10⁻¹² relative | a weight of the stencil or the commutator term missing: a relative gap of order 1 |
| K2 | B0a, a smooth random rate | ∂T/∂Ṁ and ∂T/∂M equal to central differences of T at 200 random interior entries, within 10⁻⁶ of the largest entry of each gradient | a sign or a neighbor missing in the gradient |
| K3 | B0a and the seed | 𝒜 symmetric and positive at every node; 𝒜 𝒜⁻¹ p = p to 10⁻¹² | a node with a negative or zero eigenvalue, a residual of the solve |
| K4 | E1 | with the drift defined as the largest \|E(t) − E(0)\| over the length: under 10⁻⁴ of the starting T at step 0.015, and the drift at 0.03 between 12 and 48 times that at 0.015 | a force that is not the gradient of the E it reports: a drift that does not fall with the step |
| K5 | E1 run back from its end with p reversed | the largest \|M − M(0)\| after the return at most 1 % of the largest displacement at the turn at step 0.015, and at least 8 times smaller than at step 0.03 | a state that does not return, or a return that does not improve with the step |
| K6 | V1, V2: the twenty sine modes, all six components | each mode projected on the six eigen-directions of the Hessian of the coded potential at the vacuum, taken with the metric diag(1, 1, 1, 2, 2, 2) of the six components; per direction one frequency from x(t + τ) + x(t − τ) = c x(t), τ one step for the heavy direction and sixteen steps for the five others; each ω² within 1 % of k̂² + m_ℓ², the m_ℓ² being the eigenvalues of that same Hessian over 2 L₁ | a branch off its line. V3: the three masses must read doubled, the three others unchanged |
| K7 | V1, V2: the estimator of § 3 with W on the vacuum set, targets 8 to 11 | ω² − K² within 1.0 of 60.09 at every fit point | the estimator biased by the window, the grid or the polarization |
| K8 | Na to Nd against Ua to Ud | in the bin of every fit point, the Fourier power of the twin at most 1 % of the power of the difference | a background that moves as much as the wave |
| K9 | Ha, Hb against Ua, Ub | a(ε) and 2 a(ε/2) differ by at most 2 % in norm over the bin of every fit point | a response that is not linear in the amplitude |
| K10 | Da against Ua | K² of every fit point changes by at most 2 % | a read set by the step |
| K11 | every perturbed job | E within 2 % of the perturbation's energy E(0) − U(M₀) over the length | energy lost to the scheme or gained from an instability |
| K12 | Ua to Ud, every fit point | Σ_B Re⟨Ā, H A⟩ / Σ_B Re⟨Ā, 𝒜(M₀) A⟩ within 3 % of ω², with A the six-component difference sum and H A the central difference of the coded gradient about M₀ | Fourier sums that are not solutions of the linearized equations of § 1 |

K1 to K5 run before the pool; K6 to K12 are read from it. The try cap on the controls is three; a change of the instrument between tries is recorded. What a failure costs: K1 to K5, no pool; K6, K7, K9 or K10, no label on any grid; K8, K11 or K12, no label on the grid that failed. K12 is the control that stands in for a known answer on the non-uniform background: it holds for any exact solution of the linearized equations, whatever the polarization.

### 5. The readings, frozen

| Reading | HELD | NOT HELD |
| --- | --- | --- |
| R1, one Klein-Gordon law on the unit | on a grid, with m² the median of ω² − K² over all fit points of all bins, every fit point has \|ω² − K² − m²\| ≤ 0.05 ω² | any fit point outside |
| R2, a local law | at least three bins are fitted, and in each every fit point lies within 0.05 ω² of the bin's plane (of its line, in a bin with four or five points) | a fitted bin with a point outside; fewer than three fitted bins is no label |
| R3, the linearized action in one polarization | ω²_act within 10 % of ω² at every fit point | any fit point outside |
| R4, two spacings and two boxes | for a pair of grids, at least three bins are compared and in each c² agrees within 0.05 and m² within 10 % | a compared bin outside; fewer than three compared bins is no label for the pair |

**The pairs of R4.** Two spacings: Ua with Ub, and Uc with Ud. Two boxes: Ub with Ud, at spacing 0.25. A bin is compared in a pair when both grids read it and at least four targets are fit points of that bin on both grids; c² and m² are those of the line refitted on those shared targets on each grid.

**Descriptors, with no label of their own.** Per bin: c² and m² of the line; c_r², c_t² and m² of the plane, beside the bin means of the two speeds of § 3; m² q̄; whether c² is within 0.05 of 1; whether m² is the same in every bin within 10 %. Where R2 is HELD with c_r² and c_t² more than 0.05 apart, the finding is worded as a local law with two speeds.

**The scatter rule.** On the fit points common to the two draws of a grid (Ub with Sb; Ua with Sa): if the median of \|K²(U) − K²(S)\|/ω² on the 0.25 grid passes 0.025, R1 and R2 carry no label on any grid; if the median of \|ω²_act(U) − ω²_act(S)\|/ω² there passes 0.05, R3 carries none; if, at a spacing, the median over the bins fitted in both draws of \|c²(U) − c²(S)\| passes 0.025 or that of the relative difference of m² passes 0.05, the pairs of R4 that use that spacing carry none. R1 to R3 are labeled on the 0.25 grids. Nothing in this section sets an icon.

**Registered before the run.** The kinetic factor q was computed on the four base states at staging, with the read region and the stencil of § 3. It runs from 1.69 to 6.6 on half-box 8 and from 1.40 to 6.5 on half-box 10; its far field falls as 1 + 88/r² on every grid (the median of (q − 1) r² beyond r = 5 is 87.8 to 88.0: the frame of a unit of degree 1 turns once around any sphere). The local mass of the mid branch by the formula of § 3 has the bin medians 10.7 to 31.4 on half-box 8 and 10.7 to 36.3 on half-box 10, against 60.09 in the vacuum; the bin means of the speed are 0.91 to 1.00 along r̂ and 0.64 to 0.80 across it. The expectation recorded with these numbers: R1 NOT HELD; R2 HELD, with c² of the line between 0.7 and 0.9 and m² between 10 and 40; R3 HELD by a margin of a few percent, the quotient above ω² near the cutoffs; R4 HELD where three bins are compared. If the tail holds, q − 1 falls under 0.05 only beyond r ≈ 42, outside any box of this run. Nothing in the reads script uses these numbers.

### 6. Blindspot pass, consumers, estimate

| Unknown | Guard |
| --- | --- |
| The module of M5.42 codes the inertia on the twist, not the time part for a general rate | § 1 states the form used and the Scope makes every label conditional on it; K1 ties it to the inertia; another time part is another model, and is the author's to state |
| No region of these boxes is vacuum: the kinetic density of the mid polarization is 1.4 to 6.6 times its vacuum value over the read region | R1 is read as written and its expected outcome is stated in the Scope; R2 and R3 say what holds in its place; m² q̄ is reported, and the radius at which the vacuum law would be reached is given from the tail, as an extrapolation |
| The law on the unit has two speeds, along the gradient of the frame and across it, and one scalar K² mixes them; near a cutoff the mix changes with ω | the radial share per fit point; the plane of R2; the line kept for R4 and as a descriptor |
| The wave does not stay in W: the kinetic form couples it to the heavy direction and to the rotations | the size is stated in § 3 (about 6 % in the local mass); the share of power outside W per target; K12 holds whatever the polarization, so a failure of R3 with K12 green is a failure of the one-polarization formula and is worded so |
| A local wavenumber on a non-uniform background has no known-answer control | K7 on the uniform vacuum; K12 on the unit |
| The base states are at the gate of M5.45, not at rest, and a background that slides makes a false branch | the shell is held; the unperturbed twins; the read uses the difference; K8 |
| A fourth-order scheme damps what it does not resolve | the start is smoothed; targets up to 8 at step 0.03; K10, K11; the heavy branch is not read on the unit |
| Spacing 0.5 has few cells per wavelength at these targets, and its K² range leaves some bins with fewer than four fit points | the K² range of § 3; R4 on shared targets with a floor of three compared bins; the 0.25 grids carry R1 to R3 |
| Standing waves in a held box, evanescent zones, few modes in a bin, adjacent targets that share modes under the window | K² is signed and a point enters a fit only above 1; the second draws and the scatter rule |
| The light branch, which carries the smallest mass, is not read on the unit | stated as a limit: it is read in the vacuum jobs only |
| An unvalidated instrument, an artifact read as a result (the families of the workflow) | K1 to K12; two spacings and two boxes; each control names its failure and what it costs |

**Consumers.** R1 to R4 are read by the re-read of the cell: they need K1 to K12 green and the audit. The time-stepper is reusable by a later run of a moving or clocked unit: it needs K1 to K5.

**Estimate.** Compute: the jobs at spacing 0.5 minutes each, those at 0.25 from under an hour (N = 49, 65) to about two hours and a half (N = 81), all in one pool inside the ceiling of twelve, so the wall time is that of the largest grid. With the scripts, the controls, the reads and two audits in parallel: about four hours from go to review. Model: the session's model for the run; the audit on Opus, a numbers check on Sonnet.

**Plan audit (2026-11-19, before the go).** An independent pass on a different model read this plan against the action code, the cell, the bar and the four base states, and recomputed its numbers. It confirmed the time part against the coded inertia, the equations of motion, the vacuum masses, the polarization and every formula of § 3, and the step sizes against the frequencies present. It found thirteen points, all folded above: registered local masses that covered one bin and not the range (recomputed here on the final read region); two definitions of R1 that disagreed; R4 and the scatter rule left to judgment where a bin has no fit on one grid; K6 reading the heavy branch at a stride that aliases it, with a job it named missing from the table; K7 without a set in the vacuum; no account of the two speeds of the law, nor of the size of the exchange with the heavy direction; controls applied at targets where the response is only window tails; tolerances of K2, K4 and K5 without a basis; the Scope silent on what R1 can and cannot show on these boxes; and an estimate too low by a factor of two. K12 was added in answer to the missing known answer on the unit.
