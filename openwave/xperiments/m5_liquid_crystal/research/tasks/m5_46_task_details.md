# M5.46: Charge quantization: the relaxed unit pulled apart with free faces, the charge tracked by branch

> Opened and staged 2026-11-19. Roadmap row: [`m5_roadmap.md`](../m5_roadmap.md). Thread record: [`m5_32_convo.md`](m5_32_convo.md). The cell it reads for: charge quantization in the M5 table of [`MODELS.md`](../../../../../MODELS.md).

## TASK PLANNING (2026-11-19; the readings of § 4 are frozen before the run)

**Scope.** The run named by the way back of the charge-quantization cell, as the cell reads after the re-read of [T11](../../../../../dev_docs/tasks/t11_task_details.md): a free-boundary run in which the unit stays whole when its two +½ lines are pulled apart, the charge followed by an instrument that tracks the branch. The row's test is an integer topological charge with a split e → e/2 + e/2 impossible; its regime is `static`, so a descent is the process. The column's author proposed this run after reading the re-based cell; their own report of it is not public at this date and is not evidence here. The run reports; the cell is re-read afterwards, on the bar of `MODELS.md`, in the pull request that lands the record.

**Definition of done.** The instrument passes its known-answer controls (§ 3); every chain of § 2 ends at exit, at the gate or at its cap, with its reads saved; each chain carries one label per reading of § 4; the labels are compared across the two spacings; the findings embed the cross-sections of the base, pulled and end states and the two time series; an adversarial audit on a different model has tried to refute each label; the doc checker and the roadmap linter are clean.

**Gating.** The user's go. No answer of the author is waited for.

**Research body.** This document (plan, findings, review); scripts `m5_46_degree.py`, `m5_46_pull.py`, `m5_46_reads.py` in [`../scripts/`](../scripts/); summaries `m5_46_*.json` in [`../data/`](../data/) (arrays local, indexed by the manifest); plots `m5_46_*.png` in [`../plots/`](../plots/); checkpoints in `../checkpoints/m5_46/`.

### 1. The action, the state and the pull

The action is that of [M5.42](m5_42_task_details.md) as coded in [`m5_42_eq42.py`](../scripts/m5_42_eq42.py), at J = 0:

$$
U = h^3 \sum_n \Big[ L_1 \sum_i \lVert \partial_i M \rVert^2 + \kappa \sum_{i \lt j} \lVert [\partial_i M, \partial_j M] \rVert^2 + v \sum_{k=1}^{3} \big( \operatorname{tr} M^k - c_k \big)^2 \Big]
$$

with κ = v = 1, δ = 0.05, L₁ = 0.03, c_k = 1 + δ^k, the mirror-averaged stencil. The relaxed state of [M5.45](m5_45_task_details.md) at these values carries the half turn of its long axis on one closed loop ([M5.44](m5_44_task_details.md)); a plane through the loop's axis meets it in two +½ points, the two lines of the cell's clause. The pull is a dilation of the relaxed field about the loop's center, M′(x) = M(x/s), with s set so that the two points sit a chosen distance apart. The descent that follows runs with `free_faces=True`: every node is a variable and none is held.

### 2. The chains

| Chain | Half-box | Spacing (N) | Start | Faces | Purpose |
| --- | --- | --- | --- | --- | --- |
| B0a, B0b | 8 | 0.5 (33), 0.25 (65) | seed | shell held | the base state, to the gate of M5.45 |
| B0c | 10 | 0.5 (41) | seed | shell held | the base state on the larger box |
| F0a, F0b | 8 | 0.5, 0.25 | base | free | no pull: the window the free faces leave |
| P10a, P10b | 8 | 0.5, 0.25 | base pulled to 10 | free | the first separation, two spacings |
| P14a, P14b | 8 | 0.5, 0.25 | base pulled to 13.8 | free | the second separation, two spacings |
| P14c | 10 | 0.5 | base pulled to 13.8 | free | the second separation with room to contract |
| H10a | 8 | 0.5 | base pulled to 10 | shell held | the pull without the faces |
| G10a | 8 | 0.5 | base pulled to 10 | free | fixed-step steepest descent: the path |

The optimizer is L-BFGS-B as in M5.45, except G10a. A free chain stops when the box surface reads 0 on three reads in a row (exit), at the gate max |∂U/∂m|/h³ ≤ 10⁻³, at 20000 iterations or at one hour. Reads are taken every 10 iterations up to 300 and every 25 after.

**Parallel shape.** One single-threaded process per chain (thread variables set to 1 before NumPy loads): the three base chains first, then the nine others at once, inside the ceiling of twelve. The reads run inside each chain's callback. The GPU is not used: the action needs f64. No job is sequential by default; the only serial step is the base before its pulls.

### 3. The instruments and their controls

**The line.** [`m5_44_loops.py`](../scripts/m5_44_loops.py) as it stands: the plaquettes pierced by the half turn of the top eigenvector, grouped into loops (a set under 8 plaquettes is a fragment), with center, radius and normal per loop.

**The charge.** A single-purpose module, `m5_46_degree.py`. On a closed surface of plaquettes (nested cubes of half-width 2 to 7, and the surface one cell inside the faces) the top eigenvector n is lifted to a vector by a spanning-tree sign choice; a plaquette whose four sign products multiply to −1 is pierced. A surface with no pierced plaquette and a gap λ₁ − λ₂ above 0.2 at every node is liftable, and its degree is

$$
q(S) = \frac{1}{4\pi} \sum_{t \in S} \Omega_t , \qquad \tan \frac{\Omega_t}{2} = \frac{n_a \cdot (n_b \times n_c)}{1 + n_a \cdot n_b + n_b \cdot n_c + n_c \cdot n_a}
$$

over the two triangles of each plaquette. The sign of the lift is fixed on the outermost liftable surface (q ≥ 0) and carried inward. A whole number on a liftable surface is by construction; which whole number is the read. A pierced surface gets no charge: its count of piercings is reported.

| Control | Input | The answer it must return | What a failure would look like |
| --- | --- | --- | --- |
| K1 | the seed and the base state | 1 on every surface outside the loop, 0 inside | any other value |
| K2 | the vacuum frame alone | 0 everywhere | a nonzero degree |
| K3 | two base units placed side by side | 2 on the outer surface, 1 on a cube around each | 1 or 0 outside |
| K4 | a cube cut by the loop | not liftable, an even count of piercings | a charge assigned |
| K5 | two pulled loops in one box | the line reader returns two loops | one loop |
| K6 | the base state with the global sign of n flipped at random nodes | every read unchanged | a read that moves |

Each control states the input that makes it fail; the try cap on the controls is three.

### 4. The readings, frozen

| Reading | HELD | NOT HELD |
| --- | --- | --- |
| R1, the unit stays whole | at every read from the pull to the exit the line reader returns one loop, and after the first 50 iterations its radius does not grow | two loops of 8 plaquettes or more at any read, or a loop that ends on a face while its center is more than 2 from that face |
| R2, the charge stays on it | at every read each liftable surface that encloses the loop reads 1 and each that does not reads 0 | a liftable surface that reads another whole number, or the box surface at 0 while a loop is inside |
| R3, the exit is whole | the box surface goes from 1 to 0 within two reads, with the loop's center within 2 of a face | a box surface that reads 0, or is pierced, over more reads than that |
| R4, two spacings | R1 to R3 carry the same labels at spacings 0.5 and 0.25 | a label that differs |

H10a decides what the faces add: with the shell held the pulled unit is expected to contract and stay. G10a decides whether the L-BFGS path hides a split: its labels are read like the others. Fragments are reported at every read and enter no label. Nothing in this table sets an icon.

### 5. Blindspot pass, consumers, estimate

| Unknown | Guard |
| --- | --- |
| L-BFGS is not a gradient flow, and a long step could pass over a split | dense reads early, and G10a |
| A free box has no state at rest: the unit may leave before it contracts | F0 gives the window; P14c gives room; H10a removes the faces |
| The dilation interpolates the six components of M | the potential term of the pulled state is reported before the descent; the first reads show its relaxation |
| A cube within two cells of the loop may show piercings of the grid | the label uses liftable surfaces only; the count of surfaces dropped is reported |
| An unvalidated instrument, an artifact read as a result (the families of the workflow) | K1 to K6; two spacings; each control names its failing input |

**Consumers.** R1 to R4 are read by the re-read of the cell: they need K1 to K6 green and the audit. The base states feed [M5.47](m5_47_task_details.md): they need the gate of M5.45.

**Estimate.** Compute: the base chains minutes each, the free chains tens of minutes at spacing 0.25, about half an hour of wall time in all on up to nine cores. With the scripts, the controls, the reads and two audits in parallel: about an hour and a half from go to review. Model: the session's model for the run; the audit on Opus, a numbers check on Sonnet.
