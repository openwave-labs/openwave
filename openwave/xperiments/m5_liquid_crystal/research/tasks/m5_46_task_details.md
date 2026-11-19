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
| B0a, B0b | 8 | 0.5 (33), 0.25 (65) | seed | shell held at the seed | the base state, to the gate of M5.45 |
| B0c, B0d | 10 | 0.5 (41), 0.25 (81) | seed | shell held at the seed | the base state on the larger box |
| F0a, F0b | 8 | 0.5, 0.25 | base | free | no pull: what the free faces do to the unit by themselves |
| P10a, P10b | 8 | 0.5, 0.25 | base pulled to 10 | free | the first separation, two spacings |
| P14c, P14d | 10 | 0.5, 0.25 | base pulled to 13.8 | free | the second separation, two spacings, with surfaces left around the loop |
| P14a | 8 | 0.5 | base pulled to 13.8 | free | the second separation in the tight box (the loop 1.1 from the faces), reported as it comes |
| H10a | 8 | 0.5 | base pulled to 10 | shell held at the pulled field's own values | the pull without the faces |
| G10a | 8 | 0.5 | base pulled to 10 | free | fixed-step steepest descent: the path |

The optimizer is L-BFGS-B as in M5.45, except G10a. A free chain stops at exit (no loop set clear of the faces and the box surface at 0, on three reads in a row), at the gate max |∂U/∂m|/h³ ≤ 10⁻³, at 20000 iterations or at one hour. Reads are taken every 10 iterations up to 300 and every 25 after. Before any release, U is evaluated along the dilation, s from 1 to the largest pull: a reported curve, an upper bound on the cost of the separation, and no label.

**Parallel shape.** One single-threaded process per chain (thread variables set to 1 before NumPy loads): the four base chains first, then the nine others at once, inside the ceiling of twelve. The reads run inside each chain's callback. The GPU is not used: the action needs f64. No job is sequential by default; the only serial step is the base before its pulls.

### 3. The instruments and their controls

**The line.** The functions of [`m5_44_loops.py`](../scripts/m5_44_loops.py) called directly (`pierced_plaquettes`, `group`, `loop_stats`; its second method off): the plaquettes pierced by the half turn of the eigenvector of the largest eigenvalue, grouped into sets (a set under 8 plaquettes is a fragment), with center, radius and normal per set. Added in the M5.46 module: per set, whether it touches a face (a plaquette within one cell of it) and its least distance to the faces.

**The charge.** A single-purpose module, `m5_46_degree.py`. On a closed cube surface (half-widths 2 up to one cell inside the faces) each plaquette is cut into two triangles, and the eigenvector n of the largest eigenvalue is lifted to a vector by a spanning-tree sign choice, each surface by itself. A triangle whose three sign products multiply to −1 is pierced. A surface with no pierced triangle, and with the gap between the two largest eigenvalues above 0.2 at every node, is liftable, and its degree is

$$
q(S) = \frac{1}{4\pi} \sum_{t \in S} \Omega_t , \qquad \Omega_t = 2 \operatorname{atan2} \big( n_a \cdot (n_b \times n_c) , \quad 1 + n_a \cdot n_b + n_b \cdot n_c + n_c \cdot n_a \big)
$$

The sign of a lift is a convention, so the read is the absolute value of q on each surface. That q is a whole number holds on any surface with no antipodal edge and is no evidence; which whole number is the read. A pierced surface gets no charge: its count of piercings is reported.

| Control | Input | The answer it must return | What a failure would look like |
| --- | --- | --- | --- |
| K1 | the seed (a point hedgehog, no loop) | 1 on every cube, no pierced triangle | any other value |
| K2 | a uniform M | 0 on every cube | a nonzero degree |
| K3 | an analytic director of degree 2 (the map w → w² on the sphere) | 2 on every cube | 1 or 0 |
| K4 | an analytic hedgehog and anti-hedgehog on the axis | 0 on a cube around both, 1 on a cube around each | 1 outside, or 0 around one |
| K5 | cubes cut by the base loop | not liftable, the count of piercings equal to the crossings the line reader finds on that surface | a charge assigned, or a count that differs |
| K6 | an analytic director with one half-turn ring, then with two rings at z = ±3 | one loop at the ring's radius; two loops | a wrong count or radius |
| K7 | the base state, the sign of n flipped at random nodes before the lift | every read unchanged | a read that moves |

Each control states the input that makes it fail; the try cap on the controls is three.

### 4. The readings, frozen

| Reading | HELD | NOT HELD |
| --- | --- | --- |
| R1, the unit stays whole | at every read from the pull to the exit or the gate there is one loop set clear of the faces, and its radius falls from the pulled value toward the base radius | SPLIT: two sets of 8 plaquettes or more, both clear of the faces, at any read. EXPANDED: the radius grows from the pulled value, or the loop reaches two or more faces and is cut into arcs |
| R2, the charge stays on it | at every read with a liftable surface around the loop, each such surface reads 1, and no liftable surface reads anything but 0 or 1 | a liftable surface around the loop that reads another value, or any liftable surface outside 0 and 1 |
| R3, the exit is whole (chains that exit through one face) | at the last read with the box surface at 1 the loop lies within 2 of a face, and at no later read does a set stay clear of the faces | the box surface at 0 while a set clear of the faces remains |
| R4, two spacings | R1 to R3 carry the same labels at spacings 0.5 and 0.25 | a label that differs |

A read with no liftable surface around the loop is recorded as no read, and a chain with fewer than ten reads carries no R2 label. H10a decides what the faces add: with the shell held the pulled unit is expected to contract and stay. G10a decides whether the L-BFGS path hides a split: its labels are read like the others. Fragments, and the values on liftable surfaces that do not surround the loop, are reported at every read and enter no label. Nothing in this table sets an icon.

### 5. Blindspot pass, consumers, estimate

| Unknown | Guard |
| --- | --- |
| A dilation keeps the unit whole by construction, so the test is what the descent does next: pull itself together, split, or expand | R1 names the three outcomes; the free chains of [M5.42](m5_42_task_details.md) saw loops leave a free box, so EXPANDED can occur |
| A release reads the sign of the force at two separations; it does not hold a separation | stated as a limit of this run: a ladder with the separation held and raised in steps is the stronger run, and is not in this plan |
| L-BFGS is not a gradient flow, and a long step could pass over a split | dense reads early, and G10a |
| A free box has no state at rest: the unit may leave before it contracts | F0 shows the faces alone; the larger box gives room; H10a removes the faces |
| The dilation interpolates the six components of M | the potential term of the pulled state is reported before the descent; the first reads show its relaxation |
| A cube within two cells of the loop may show piercings of the grid, and a pull close to the faces may leave no liftable surface | the label uses liftable surfaces only; a read without one is no read; the count of surfaces dropped is reported |
| An unvalidated instrument, an artifact read as a result (the families of the workflow) | K1 to K7; two spacings; each control names its failing input |

**Consumers.** R1 to R4 are read by the re-read of the cell: they need K1 to K7 green and the audit. The base states feed [M5.47](m5_47_task_details.md): they need the gate of M5.45.

**Estimate.** Compute: the base chains minutes each, the free chains tens of minutes at spacing 0.25 (N = 81 on the larger box), about three quarters of an hour of wall time in all on up to nine cores. With the scripts, the controls, the reads and two audits in parallel: about two hours from go to review. Model: the session's model for the run; the audit on Opus, a numbers check on Sonnet.

**Plan audit (2026-11-19, before the go).** An independent pass on a different model read this plan against the action code, the line reader and the cell. It found eleven faults, all folded above: two controls whose stated answers were wrong for their inputs (the frame is a hedgehog, the seed has no loop), one control a correct instrument could fail, one that could not fail, readings that would have called a loop leaving the box a split, a wide pull that left no surface to read, the solid angle written with tan where atan2 is needed, a pierced test that missed the plaquette's diagonal, and the shell of H10a left unstated.
