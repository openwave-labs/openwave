# M5.45: The inertia of the δ = 0.05 state without L₁ against the grid spacing

> Opened 2026-11-18 from the author's answer to the [M5.44](m5_44_task_details.md) results post on discussion 186 ([discussioncomment-19696575](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19696575)), which reads the inertia of the δ = 0.05 state without L₁ as set by the grid, records a prediction for the same state at spacing 0.25, and asks for that read and for the inertia by box of the L₁ = 0.1 ladder of [M5.42](m5_42_task_details.md). Roadmap row: [`m5_roadmap.md`](../m5_roadmap.md). Thread record: [`m5_32_convo.md`](m5_32_convo.md).

## TASK PLANNING (2026-11-18; a short relaxation run on seven grids and a desk read; the go is the user's)

**Scope.** (1) R1: the driver of this task on the evaluator of M5.42, with a restart after every end by the function tolerance, and its control (§ 3). (2) R2: the state at J = 0 with δ = 0.05, half-box 6, the mirror-averaged stencil and the chart fixed, relaxed from the seed without L₁ at spacings 0.5, 0.375, 0.25 and 0.1875, and with L₁ = 0.03 at spacings 0.5, 0.375 and 0.25. (3) R3: the reads of § 4 on every end state. (4) R4: the inertia by box of the J = 0 ladders of M5.42 at L₁ = 0.1 and 0.3, from its tracked reads; nothing is relaxed for it. Out of scope: any J above 0; another box; L₁ = 0.01 and its two states; the broken sector on half-boxes 12 and 18; a statement about the continuum; any icon or pointer of [`MODELS.md`](../../../../../MODELS.md).

**Definition of done.** The control of § 3 reported as met or as the exact failing check; each of the seven chains ended by its own rule or cut by the wall-time cap, with its words; the tables of § 4 written by the script from a tracked JSON; the readings of § 2 stated; an audit by an agent on a different model, with its own evaluation of U and I on the saved end states, with every claim standing or qualified; the task review presented in the terminal.

**Gating.** The user's go. The findings go to the author by mail before they enter the public record.

**Model and effort.** The session's model for the run; one audit agent on a different model, named at the spawn.

**Research body.** This document (`## FINDINGS` after the run); scripts `../scripts/m5_45_relax.py` and `../scripts/m5_45_reads.py`; data `../data/m5_45_summary.json` (tracked) and the end states `../data/m5_45_*.npz` (local-only, indexed by [`_DATASETS.md`](../data/_DATASETS.md)); plots `../plots/m5_45_*.png`, embedded in the findings; in-flight state in `../checkpoints/m5_45/` (local-only).

**Consumers.**

| Sub-result | Consumer | The check that consumer needs |
| --- | --- | --- |
| The inertia and the energy at spacing 0.25 | the author's recorded prediction (inertia above 3000, energy within 3 percent) | the words of § 2 row 2 on the state; the trajectory of the inertia, since a state can end on the optimizer's tolerance while its inertia still moves |
| The inertia against the spacing, without L₁ and with L₁ = 0.03 | the author's reading that without L₁ the grid sets the inertia; M5.42 § 15 and M5.44 § 15, whose numbers on these states were read at one spacing | the same driver and rule at every spacing; the control with L₁ = 0.03, where the author reports no dependence on the spacing |
| Where the inertia sits, and whether the gradients commute there | the author's mechanism (thin walls in which the commutator term costs nothing) | the definitions of § 2 row 4 fixed before the run; the same read on the L₁ = 0.03 states |
| The inertia by box of the two ladders | the author's request | the numbers taken from the tracked JSON of M5.42 by the script, with the label of each state |

### 1. What was found at plan time

| Source | State | U | I | I inside r < 2 | How it ended |
| --- | --- | --- | --- | --- | --- |
| M5.42, chain `r6_hb6_L0.0`, path `j0` | spacing 0.375, without L₁ | 14.7359 | 2096.0 | 521.1 | iteration 8364, by the optimizer's function tolerance, CONVERGED by the rule of M5.42; its inertia was 1212.6 at iteration 1000 and 1889.6 at iteration 3000 |
| A pilot of the driver of this task (one chain, not audited) | spacing 0.5, without L₁ | 14.4870 | 1188.6 | 321.9 | iteration 5563 by the function tolerance; the restart made 3 iterations; the inertia was 1189.6 at iteration 3000 |
| The author's post | spacings 0.5, 0.375, 0.25, without L₁ | not given | 1233, 2120, about 3460 (the last not converged and still rising) | 532 at spacing 0.375 | the author's code |
| The author's post | spacings 0.5 and 0.375, L₁ = 0.03 | not given | 119.0 and 119.2 | not given | a flat loop of radius 3.13 |

Between spacings 0.5 and 0.375 the inertia of the two states of this code grows by a factor 1.76, a local exponent of −1.97 in the spacing, and U rises by 1.7 percent.

### 2. The readings frozen before the run

| # | Item | The reading frozen here |
| --- | --- | --- |
| 1 | The words of a state | CONVERGED as in M5.42 § 2 row 4. STEADY: the inertia changed by less than 10⁻³ (relative) over the last tenth of the iterations. RISING: it grew by more than that. The words are printed for every state beside its inertia, with the inertia at every stop |
| 2 | The author's prediction at spacing 0.25 | "I above 3000, against your 2096 at 0.375, with the energy within 3 %". The energy is read as U against the 14.7359 of the state at spacing 0.375, a window of 14.294 to 15.178; this reading of "the energy" is ours and is stated with the score. HELD: the last stop has I above 3000 and U in the window (a RISING state above 3000 holds the inertia clause). FAILED: I at or under 3000 on a STEADY or CONVERGED state, or U outside the window. OPEN: I at or under 3000 on a RISING state at the cap |
| 3 | "Set by the grid" | Read on this code's states only. GROWS AS THE SPACING FALLS: each finer state without L₁ has an inertia more than 10 percent above the next coarser one; the local exponent in the spacing is printed for each pair. INDEPENDENT OF THE SPACING (the L₁ = 0.03 states): the three inertias within 1 percent. No limit at zero spacing is stated in either case, and an exponent from a RISING state is printed as a bound, not as a value |
| 4 | Where the inertia sits | With the inertia density of the functional per node: the smallest set of nodes W that holds 95 percent of the inertia, and its share of the nodes. Per node, with the differences of the functional's own stencil, q = Σ‖[∂ₐM, ∂_bM]‖² over the three pairs, divided by (Σₐ‖∂ₐM‖²)²; its mean weighted by the inertia density, on W and on the rest. The spacing times the sum of ‖∇M‖² over the box. The post does not give the author's definition of a wall, so these numbers are compared with the author's in kind and not digit for digit |
| 5 | The loop | The reader of M5.44 § 2 on every end state: loops, fragments, radius. Without L₁ no radius word is given (M5.44 § 15: a core the grid does not resolve) |

### 3. R1: the control of the driver (three tries)

| Control | What it is | Met when |
| --- | --- | --- |
| The saved state returned | the chain at spacing 0.375 without L₁, at its first end by the function tolerance | the same iteration count as chain `r6_hb6_L0.0` path `j0` of M5.42 (8364), with U and I equal to its record to 10 digits |
| The evaluator | unchanged from M5.42 (its controls: M5.42 § 8) | the audit's own NumPy evaluation of U and I on two end states of this run within 10⁻¹⁰ (relative) |

### 4. R2 to R4: the chains and the tables

| Chain | Grid | Spacing | L₁ |
| --- | --- | --- | --- |
| `h0500_L0` | 25³ | 0.5 | 0 |
| `h0375_L0` | 33³ | 0.375 | 0 |
| `h0250_L0` | 49³ | 0.25 | 0 |
| `h01875_L0` | 65³ | 0.1875 | 0 |
| `h0500_L0.03` | 25³ | 0.5 | 0.03 |
| `h0375_L0.03` | 33³ | 0.375 | 0.03 |
| `h0250_L0.03` | 49³ | 0.25 | 0.03 |

Each chain: L-BFGS-B from the seed, 20 correction pairs, function tolerance 10⁻¹⁵, the gradient stop off; after an end by the function tolerance a restart with an empty memory, until a restart ends within 25 iterations; a cap of 100000 iterations and a wall-time cap of two and a half hours; stops saved at 1000, 3000, 10000, 30000 and 100000; U and I recorded every 100 iterations.

| Table | Content |
| --- | --- |
| The inertia against the spacing | for the seven states: U and its three terms, I, I inside r < 2, the words of § 2 row 1, the iteration and the way of ending, the local exponents |
| The score | § 2 row 2 on the state at spacing 0.25 |
| Where the inertia sits | § 2 row 4 on the seven states |
| The loop | § 2 row 5 on the seven states |
| The inertia by box | J = 0 on half-boxes 6, 9, 12 and 18 of M5.42 at L₁ = 0.1 and 0.3 (mirror-averaged, spacing 0.375, δ = 0.3): I, I inside r < 2, between r = 2 and 4, from r = 4 to the half-box, and beyond the inscribed sphere, with the loop radius of M5.44 |

Two figures: the inertia against the spacing for the two values of L₁, and the inertia against the iteration for the four chains without L₁.

Recorded before the run. Line 1 follows the pilot and is not a prediction; the others are, and a failure is reported with the same weight.

1. At spacing 0.5 without L₁: I = 1188.6, STEADY.
2. At spacing 0.375 without L₁ the restarts move the inertia by less than 2 percent from 2096.0.
3. At spacing 0.25 without L₁ the inertia ends between 3400 and 4900 (the exponent of § 1 continued gives 4660), so the inertia clause of the author's prediction holds.
4. At spacing 0.25 without L₁ U is between 14.85 and 15.15, inside the author's 3 percent and within 0.3 percent of its upper edge at most.
5. With L₁ = 0.03 the three inertias are within 1 percent of each other and within 2 percent of the author's 119.0 and 119.2, each state with one loop of radius 3.0 to 3.3.
6. The set W is under 10 percent of the nodes without L₁ and its share falls with the spacing; q on W falls with the spacing and is at least ten times smaller than on the rest.
7. Not predicted: whether the chain at spacing 0.1875 ends inside the wall-time cap, and the share of the inertia inside r < 2 at spacing 0.25.

### 5. Blindspot pass, the audit and the parallel shape

| Risk | Where it enters | Guard |
| --- | --- | --- |
| A stop of the optimizer read as a state | the saved state at spacing 0.375 ended on the function tolerance with its inertia rising along the path | the restart rule; the trajectory; the words STEADY and RISING beside every inertia |
| A grid-set number read as a property | this is the question; the inertia of M5.42 and M5.44 on these states was read at one spacing | four spacings; no limit stated; the L₁ = 0.03 control on the same grids |
| A slow chain cut and compared with fast ones | the finest grids may not end inside the wall-time cap | a cut state is labeled, its inertia printed as a bound, and no exponent taken from it as a value |
| Our definition taken for the author's | the walls and the commutator ratio | § 2 row 4 fixed before the run and stated beside every number; compared in kind |
| One start taken for the state | every chain starts from the seed; the author reports two states by start at L₁ = 0.01 | named as not computed; the reads say "from the seed" |
| Unvalidated instrument | the new driver | the control of § 3 on a saved state of M5.42; the pilot of § 1 |

The audit: one agent on a different model evaluates U and I on the saved end states with its own NumPy code, recomputes the reads of § 2 row 4 from the definitions, checks every number of the findings against the JSON, and tries to refute each line.

The parallel shape: seven single-threaded processes at once, one per chain, all launched at the start (seven busy cores, under the ceiling of twelve). Inside a chain the descent is sequential by nature. No GPU: the kernel is compiled double-precision code on small grids. The reads run after the chains, on up to six workers.

What follows, each on its own go: the same ladder at a second box; the two states by start at L₁ = 0.01; J above 0 on the finer grids.
