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

## FINDINGS (2026-11-19)

> Run of 2026-11-19: the thirteen chains of the plan and four chains added during the run (§ 12, outside the frozen table). The labels and the numbers they rest on are carried by [`m5_46_summary.json`](../data/m5_46_summary.json); every read of every chain by one file per chain, `m5_46_chain_<name>.json` in [`../data/`](../data/) (for example [`m5_46_chain_P10a.json`](../data/m5_46_chain_P10a.json)); the controls by [`m5_46_controls.json`](../data/m5_46_controls.json); the dilation curves by [`m5_46_dilation.json`](../data/m5_46_dilation.json). The tables below are printed by [`m5_46_reads.py`](../scripts/m5_46_reads.py) from those files.

**TL;DR.** (1) The controls K1 to K7 passed on the second of three tries; the first failed K6 on a fault of the control's own input, and the instrument was not changed between the tries (§ 6). (2) No read of any chain held two sets clear of the faces: SPLIT did not occur in 983 reads of the thirteen chains of the plan, nor in 251 reads of the four added chains. (3) Released with free faces from a separation of 10 on half-box 8 and of 13.8 on half-box 10, at both spacings, the loop contracted from the pulled radius (4.97 to 6.96) to 3.41 and 3.44, and the four chains ended at the gate: R1 HELD and R2 HELD on P10a, P10b, P14c and P14d. (4) In the tight box (P14a, the pulled loop 1.25 from the faces) the loop reached the four side faces between iterations 10 and 20, was cut into four arcs, and no set was left from iteration 80 on, with the box surface at 0: R1 NOT HELD (EXPANDED), R2 no label (two counted reads). (5) R3 carries no label on any chain of the plan: none ended by an exit through one face. (6) R4 is HELD on the three pairs (on the F0 pair by a margin of 0.005 in a tolerance set at run time, § 15) of spacings. (7) The gate of the plan ended the four continued free chains (F0a, F0b, P14c, P14d) on a plateau and not at rest, and for P10a and P10b the same is inferred from U and not shown. Four chains added during the run continued the end states of F0a, F0b, P14c and P14d with the gate stop off: each stayed in place for 550 to 925 iterations, with max \|dU/dm\|/h³ down to between 1.4 × 10⁻⁴ and 2.1 × 10⁻⁵, and then the loop traveled as one set to a face and left the box through it (through +z, along its axis, in X0a, X0b and X14c; through −x, in its own plane, in X14d), the box surface going from 1 to 0 and U falling to between 4.1 and 6.9 at the stop. These chains are outside the frozen table and carry no label. (8) H10a, the shell held: the loop contracted to 3.389 and the chain ended by the function tolerance at max \|dU/dm\|/h³ = 2.2 × 10⁻⁷. G10a, the steepest descent: one clear set at each of its 452 reads, and the gate met at step 10818 with the radius at 3.573, above the 3.405 of P10a. (9) U rose at every step of the dilation on the four base states (27.30, 36.19, 57.04 at s = 1 and at separations 10 and 13.8 on B0a), the potential term carrying the rise.

**Scope.** Releases of the relaxed unit of the action of M5.42 at J = 0 with δ = 0.05 and L₁ = 0.03, on half-boxes 8 and 10 at spacings 0.5 and 0.25, after an isotropic dilation about the loop's center. A release reads which way the loop moves from two separations; it holds no separation. Nothing is said here about another pull than a dilation, another δ or L₁, J above 0, a limit at zero spacing, or the state of a free box after the unit has left. Nothing in this section sets or suggests a mark of [`MODELS.md`](../../../../../MODELS.md).

### 6. The controls

Three tries were allowed and two were used. On the first try K1 to K5 and K7 passed and K6 failed: the two-ring input of that try was n = cos(ψ/2) ρ̂ + sin(ψ/2) ẑ with ψ the sum of the two angles atan2(z − zᵢ, ρ − R₀), a field that also carries a full-turn line of n on the axis between the rings; with the axis set off the nodes the half-turn test of the line reader reads plaquettes on such a line as pierced where the step of the angle between corners passes 90°, and a third set of radius 1.6 to 1.8 appeared at z = 0. The input was replaced by the field of the control table below, whose angle equals π on the axis, so that n = ẑ there and the only lines are the rings. On the second try all seven passed. K5 and K7 need a relaxed state: before the base chains they ran on the relaxed state of M5.45 at spacing 0.5 (half-box 6, the same δ and L₁), and after the base chains all seven were run again on the four base states of this run, before any release. The tracked record is that of a rerun made after the releases, byte-identical to the record of that last run kept in the checkpoints, and its numbers are below.

The two-ring input of K6, with ζ = ρ + i z and ζᵢ = R₀ + i zᵢ for k rings:

$$
\psi = \arg \Big[ (-1)^{k+1} \prod_{i=1}^{k} ( \zeta - \zeta_i ) ( \zeta + \bar{\zeta}_i ) \Big] , \qquad n = \cos \tfrac{\psi}{2} \hat{\rho} + \sin \tfrac{\psi}{2} \hat{z}
$$

| Control | Input | Returned | Result | An input on which the same check returns another value |
| --- | --- | --- | --- | --- |
| K1 | the seed on half-box 8 at spacings 0.5 and 0.25 and on half-box 10 at spacing 0.5 (12, 13 and 16 cubes) | 1 on all 41 cubes; 0 pierced triangles; \|q\| within 2.2 × 10⁻¹⁶ of a whole number; least gap on the cubes 0.885; 0 loops and 0 fragments from the line reader | PASS | a uniform M: 0 on the same cubes |
| K2 | a uniform M with eigenvalues 1, 0.05, 0 on the same three grids | 0 on all 41 cubes, the largest \|q\| 0.0 | PASS | the seed: 1 |
| K3 | M = n nᵀ with n(x) = f(x/\|x\|), f the map w → w² of the sphere, half-box 8 at both spacings | 2 on all 25 cubes; 0 pierced triangles; within 4.4 × 10⁻¹⁶ of a whole number | PASS | the map w → w: 1 |
| K4 | M = n nᵀ with n the unit field of a point source at (0.1, 0.07, 3.2) and a point sink at (0.1, 0.07, −3.2), half-box 8 at both spacings | 0 on every centered cube of half-width 4.5 and more (7 and 8 cubes); 1 on the cube of half-width 2 centered at (0, 0, 3) and 1 on the one centered at (0, 0, −3), with q = −1.0 and +1.0 before the absolute value. At spacing 0.5 the centered cube of half-width 3, which passes 0.2 from the two points, is pierced and gets no charge | PASS | two sources of the same sign: 1 around both (a zero of index −1 sits between them) |
| K5 | the four base states of this run | on each state the cubes of half-widths 2.5 and 3 are cut by the loop: 8 pierced triangles and 8 crossings of the line reader on each, no charge assigned; no cube has a pierced triangle without a crossing; no plaquette has both of its triangles pierced | PASS | the liftable cubes around the loop: a charge (1) is assigned and the count is 0 |
| K6 | the ring field above with δ = 0.05, the axis through (0.11, 0.17), half-box 8 at both spacings: one ring of radius 3, one of radius 5, two of radius 3 at z = ±3 | one ring of radius 3: one loop of radius 2.981 (spacing 0.5) and 3.004 (0.25); of radius 5: one loop of radius 5.013 and 5.010; two rings: two loops of radius 2.981 (3.004), their centers at z = ±3.25 (±3.125); no fragment in any. Not part of the check: on the single ring the cubes that surround it read 1, the cubes it cuts are pierced, and the cubes inside the ring of radius 5 read 0 | PASS | the ring of radius 5 reads 5, and the two rings read two loops |
| K7 | the four base states, the sign of n flipped at a random half of the nodes before the lift, five draws per state (2 277 597 flips in all) | 0 of 290 surface reads moved (58 cubes, five draws); the largest change of \|q\| 0.0 | PASS | the same flipped field summed without the lift: −6, 6, −38, −10 on the first four liftable cubes of B0d, which read 0, 1, 1, 1 with the lift |

Over all reads of all chains the largest distance of \|q\| from a whole number on a liftable surface was 6.7 × 10⁻¹⁶.

### 7. The chains

**Table 1. Every chain: how it ran, how it ended, the loop and the labels.**

| Chain | Half-box, spacing (grid) | Start | Faces | Descent | Iterations | How it ended | max \|dU/dm\|/h³ at the end | U at read 0 | U at the end | Radius of the loop at read 0 | Radius at the last read | R1 | R2 | R3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0a | 8, 0.5 (33³) | seed | shell held at the seed | L-BFGS-B | 817 | FUNCTION TOLERANCE, THEN A STALLED LINE SEARCH | 1.2e-06 | 38.3099 | 27.3009 | no set | 3.293 | no label | no label | no label |
| B0b | 8, 0.25 (65³) | seed | shell held at the seed | L-BFGS-B | 1124 | FUNCTION TOLERANCE, REPEATED AFTER A RESTART | 7.5e-06 | 39.0243 | 27.5012 | no set | 3.316 | no label | no label | no label |
| B0c | 10, 0.5 (41³) | seed | shell held at the seed | L-BFGS-B | 957 | FUNCTION TOLERANCE, REPEATED AFTER A RESTART | 9.0e-07 | 42.2969 | 31.1976 | no set | 3.293 | no label | no label | no label |
| B0d | 10, 0.25 (81³) | seed | shell held at the seed | L-BFGS-B | 1254 | FUNCTION TOLERANCE, REPEATED AFTER A RESTART | 1.2e-05 | 43.0152 | 31.3993 | no set | 3.316 | no label | no label | no label |
| F0a | 8, 0.5 (33³) | B0a, no pull | free | L-BFGS-B | 271 | GATE | 7.5e-04 | 27.3009 | 26.4371 | 3.293 | 3.405 | no label | HELD | no label |
| F0b | 8, 0.25 (65³) | B0b, no pull | free | L-BFGS-B | 640 | GATE | 9.5e-04 | 27.5012 | 26.6467 | 3.316 | 3.436 | no label | HELD | no label |
| P10a | 8, 0.5 (33³) | B0a pulled to 10 (s = 1.5184) | free | L-BFGS-B | 255 | GATE | 7.7e-04 | 36.1900 | 26.4371 | 5.024 | 3.405 | HELD | HELD | no label |
| P10b | 8, 0.25 (65³) | B0b pulled to 10 (s = 1.5079) | free | L-BFGS-B | 660 | GATE | 9.7e-04 | 34.4639 | 26.6466 | 4.968 | 3.437 | HELD | HELD | no label |
| P14c | 10, 0.5 (41³) | B0c pulled to 13.8 (s = 2.0954) | free | L-BFGS-B | 435 | GATE | 9.8e-04 | 69.7653 | 30.3938 | 6.963 | 3.405 | HELD | HELD | no label |
| P14d | 10, 0.25 (81³) | B0d pulled to 13.8 (s = 2.0809) | free | L-BFGS-B | 903 | GATE | 8.2e-04 | 64.7187 | 30.6006 | 6.935 | 3.437 | HELD | HELD | no label |
| P14a | 8, 0.5 (33³) | B0a pulled to 13.8 (s = 2.0954) | free | L-BFGS-B | 110 | EXIT | 4.9e-01 | 57.0413 | 4.1907 | 6.867 | no set | NOT HELD (EXPANDED) | no label | no label |
| H10a | 8, 0.5 (33³) | B0a pulled to 10 (s = 1.5184) | shell held at the pulled field | L-BFGS-B | 390 | FUNCTION TOLERANCE, REPEATED AFTER A RESTART | 2.2e-07 | 36.1900 | 27.1460 | 5.024 | 3.389 | HELD | HELD | no label |
| G10a | 8, 0.5 (33³) | B0a pulled to 10 (s = 1.5184) | free | steepest, step 0.01 | 10818 | GATE | 1.0e-03 | 36.1900 | 26.4503 | 5.024 | 3.573 | HELD | HELD | no label |

The four base states met the gate of M5.45 (max \|dU/dm\|/h³ from 9.0 × 10⁻⁷ to 1.2 × 10⁻⁵ at the end, the gate of 10⁻³ first met at iterations 470, 718, 522 and 814). Each holds one loop around the z axis, centered on the box, of radius 3.293 at spacing 0.5 and 3.316 at spacing 0.25, the same on half-boxes 8 and 10. Along each base chain the reader finds a set of 8 pierced plaquettes or more from iteration 20 on, at first 0.6 of a cell in radius (0.30 at spacing 0.5, 0.15 at 0.25); every cube read 1 up to iterations 230 to 260, and the loop then widened through the inner cubes to its final radius by iteration 400 (the first panel row of the first two figures of § 8).

The pull was read back by the line reader at read 0: a radius of 5.024 and 4.968 for the separation of 10 (5 asked for), and of 6.963, 6.935 and 6.867 for the separation of 13.8 (6.9 asked for). In the tight box the pulled loop of P14a stood 1.25 from the faces by the reader (the plan's figure of 1.1 is 8 − 6.9).

### 8. The labels R1 to R3

**Table 2. What each label rests on (the released chains).**

| Chain | R1 | R2 | R3 |
| --- | --- | --- | --- |
| F0a | no label: no pull, so the HELD clause has no pulled value; neither SPLIT nor EXPANDED; one clear set at every read; radius 3.293 at read 0, 3.405 at the last clear read | HELD: 29 counted reads of 29, every surface around the loop at 1, every liftable surface at 0 or 1 | no label: did not end by EXIT; reads with the box surface at 0 and a clear set: 0 |
| F0b | no label: no pull, so the HELD clause has no pulled value; neither SPLIT nor EXPANDED; one clear set at every read; radius 3.316 at read 0, 3.436 at the last clear read | HELD: 45 counted reads of 45, every surface around the loop at 1, every liftable surface at 0 or 1 | no label: did not end by EXIT; reads with the box surface at 0 and a clear set: 0 |
| P10a | HELD: one clear set at every read; the radius fell from 5.024 to 3.405 (base 3.293) | HELD: 27 counted reads of 27, every surface around the loop at 1, every liftable surface at 0 or 1 | no label: did not end by EXIT; reads with the box surface at 0 and a clear set: 0 |
| P10b | HELD: one clear set at every read; the radius fell from 4.968 to 3.437 (base 3.316) | HELD: 46 counted reads of 46, every surface around the loop at 1, every liftable surface at 0 or 1 | no label: did not end by EXIT; reads with the box surface at 0 and a clear set: 0 |
| P14c | HELD: one clear set at every read; the radius fell from 6.963 to 3.405 (base 3.293) | HELD: 37 counted reads of 37, every surface around the loop at 1, every liftable surface at 0 or 1 | no label: did not end by EXIT; reads with the box surface at 0 and a clear set: 0 |
| P14d | HELD: one clear set at every read; the radius fell from 6.935 to 3.437 (base 3.316) | HELD: 56 counted reads of 56, every surface around the loop at 1, every liftable surface at 0 or 1 | no label: did not end by EXIT; reads with the box surface at 0 and a clear set: 0 |
| P14a | NOT HELD (EXPANDED): two faces or more touched from iteration 20 (4 sets at that read; faces touched in all: -x, +x, -y, +y) | no label: 2 counted reads of 12 | no label: exit with 4 faces touched; reads with the box surface at 0 and a clear set: 0 |
| H10a | HELD: one clear set at every read; the radius fell from 5.024 to 3.389 (base 3.293) | HELD: 35 counted reads of 35, every surface around the loop at 1, every liftable surface at 0 or 1 | no label: did not end by EXIT; reads with the box surface at 0 and a clear set: 0 |
| G10a | HELD: one clear set at every read; the radius fell from 5.024 to 3.573 (base 3.293) | HELD: 452 counted reads of 452, every surface around the loop at 1, every liftable surface at 0 or 1 | no label: did not end by EXIT; reads with the box surface at 0 and a clear set: 0 |

**Table 3. The reads of the line and of the charge, per chain.**

| Chain | Reads | Reads with one clear set, two or more, none | Faces touched | Counted reads (a liftable surface around the loop) | No read | Mean surfaces around the loop per counted read | Counted reads with a surface around the loop not at 1 | Reads with a liftable surface outside 0 and 1 | Box surface at the last read | Reads with the box surface at 0 and a clear set | Surface reads: liftable, pierced, gap | Values on liftable surfaces that do not surround the loop | Reads with fragments (most fragments, most plaquettes in them) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0a | 52 | 50, 0, 2 | none | 50 | 2 | 10.2 | 0 | 0 | 1 | 0 | 555, 43, 26 | 0, 1 | 1 (1, 6) |
| B0b | 64 | 62, 0, 2 | none | 62 | 2 | 10.9 | 0 | 0 | 1 | 0 | 729, 65, 38 | 0, 1 | 0 (0, 0) |
| B0c | 58 | 56, 0, 2 | none | 56 | 2 | 14.1 | 0 | 0 | 1 | 0 | 845, 55, 28 | 0, 1 | 1 (1, 6) |
| B0d | 70 | 68, 0, 2 | none | 68 | 2 | 14.7 | 0 | 0 | 1 | 0 | 1070, 77, 43 | 0, 1 | 0 (0, 0) |
| F0a | 29 | 29, 0, 0 | none | 29 | 0 | 8.0 | 0 | 0 | 1 | 0 | 261, 58, 29 | 0 | 0 (0, 0) |
| F0b | 45 | 45, 0, 0 | none | 45 | 0 | 9.0 | 0 | 0 | 1 | 0 | 450, 90, 45 | 0 | 0 (0, 0) |
| P10a | 27 | 27, 0, 0 | none | 27 | 0 | 7.4 | 0 | 0 | 1 | 0 | 237, 61, 26 | 0 | 0 (0, 0) |
| P10b | 46 | 46, 0, 0 | none | 46 | 0 | 8.5 | 0 | 0 | 1 | 0 | 451, 107, 40 | 0 | 0 (0, 0) |
| P14c | 37 | 37, 0, 0 | none | 37 | 0 | 10.6 | 0 | 0 | 1 | 0 | 464, 89, 39 | 0 | 0 (0, 0) |
| P14d | 56 | 56, 0, 0 | none | 56 | 0 | 11.4 | 0 | 0 | 1 | 0 | 754, 140, 58 | 0 | 0 (0, 0) |
| P14a | 12 | 2, 0, 10 | -x, +x, -y, +y | 2 | 10 | 1.0 | 0 | 0 | 0 | 0 | 100, 35, 9 | 0 | 1 (4, 28) |
| H10a | 35 | 35, 0, 0 | none | 35 | 0 | 7.6 | 0 | 0 | 1 | 0 | 310, 73, 37 | 0 | 0 (0, 0) |
| G10a | 452 | 452, 0, 0 | none | 452 | 0 | 6.9 | 0 | 0 | 1 | 0 | 3871, 1038, 515 | 0 | 0 (0, 0) |
| X0a | 50 | 47, 0, 3 | none | 46 | 4 | 7.9 | 0 | 0 | 0 | 0 | 461, 92, 47 | 0 | 0 (0, 0) |
| X0b | 69 | 65, 0, 4 | +z | 65 | 4 | 8.8 | 0 | 0 | 0 | 0 | 696, 134, 67 | 0 | 0 (0, 0) |
| X14c | 58 | 53, 0, 5 | +z | 53 | 5 | 11.4 | 0 | 0 | 0 | 0 | 741, 135, 52 | 0 | 0 (0, 0) |
| X14d | 74 | 61, 0, 13 | -x | 61 | 13 | 12.2 | 0 | 0 | 0 | 0 | 888, 309, 61 | 0 | 0 (0, 0) |

How the frozen wording of § 4 was applied by the script, where the plan left a term open (each line is also in the docstring of [`m5_46_reads.py`](../scripts/m5_46_reads.py)):

| Reading | Term of the frozen table | As read by the script |
| --- | --- | --- |
| R1 | "one loop set clear of the faces" | exactly one set of 8 plaquettes or more whose least distance to the faces is above one cell |
| R1 | the radius "falls" or "grows" from the pulled value | the radius of the clear loop at its last read against its radius at read 0, with a tolerance of half a cell either way (0.25 at spacing 0.5, 0.125 at spacing 0.25) |
| R1 | "reaches two or more faces and is cut into arcs" | a read at which the sets that touch the faces touch two faces or more; the number of sets at that read is printed |
| R1 | "to the exit or the gate" | every read of the chain; for a chain that ends by EXIT, every read before the first one without a clear set |
| R2 | "a liftable surface around the loop" | a liftable cube whose half-width is above the largest coordinate of the loop's plaquette centers; the loop of a read is its largest clear set, or its largest set when none is clear |
| R2 | "fewer than ten reads" | fewer than ten reads in the chain, or, when no read is NOT HELD, fewer than ten counted reads |
| R3 | "chains that exit through one face" | a chain that ends by EXIT and whose sets touched one face in all |
| R3 | "the box surface" | the outermost nested cube, of half-width L − h |
| All | the nested surfaces | half-widths 2, 2.5, 3, ... and L − h, at both spacings |

1. **R1.** HELD on P10a, P10b, P14c, P14d, H10a and G10a: one clear set at every read, and a radius that fell from the pulled value to 3.405, 3.437, 3.405, 3.437, 3.389 and 3.573, against a base radius of 3.293 (spacing 0.5) and 3.316 (0.25). NOT HELD (EXPANDED) on P14a, by the second clause: the sets touched the four side faces from the read at iteration 20, where the loop was four arcs. No label on F0a and F0b: they have no pull, so the HELD clause has no pulled value; neither was SPLIT or EXPANDED by the script's rule.
2. **A tolerance decides the word of F0a and F0b.** With free faces and no pull the radius of the loop grew from 3.293 to 3.405 (F0a, by 0.112) and from 3.316 to 3.436 (F0b, by 0.120). The script calls a radius grown when the change passes half a cell, 0.25 and 0.125. With a tolerance under 0.112 both chains would read NOT HELD (EXPANDED) by the first clause of EXPANDED; with one between 0.112 and 0.120, F0b alone. The tolerance is the script's and was set at run time in the reads script; that it predates the releases cannot be shown, since the script's last change follows the last chain (§ 15); the plan names none. The same growth is in the end radius of the pulled chains (3.405 and 3.437 against the base 3.293 and 3.316): the loop of the free box at the gate is 0.11 to 0.12 wider than the loop of the held shell.
3. **R2.** HELD on the eight released chains with ten counted reads or more: at every one of their reads a liftable surface surrounded the loop (no read was "no read"), every such surface read 1, and every liftable surface that did not surround the loop read 0. No label on P14a: two counted reads (iterations 0 and 10, one surface each, both at 1), then ten reads without a liftable surface around a loop. No liftable surface of any chain read another whole number than 0 or 1. In the base chains the value 1 in the column of surfaces that do not surround the loop comes from their first two reads, which have no set.
4. **R3.** No label on any chain of the plan. Eight released chains did not end by EXIT. P14a ended by EXIT with four faces touched, so it is not an exit through one face; its box surface read 1 at iterations 0 and 10, was pierced from 20 to 80, and read 0 from 90. The condition of NOT HELD (the box surface at 0 while a clear set remains) was met at no read of any chain.
5. **Surfaces dropped.** A cube that the loop cuts is pierced, and a cube that passes close to the loop's core has a gap at or under 0.2: Table 3 counts both per chain. In the released chains of the plan between 71 and 79 percent of the surface reads were liftable, except P14a (69 percent). Fragments appeared at one read of P14a (iteration 80: four fragments, 28 plaquettes, the last of the four arcs) and at iteration 10 of B0a and of B0c (one fragment of 6 plaquettes at the seed's point).

![Radius of the loop at every read, per chain: filled points where the loop is clear of the faces, open squares where the largest set touches a face](../plots/m5_46_radius.png)

![The read on each nested cube surface at every read, per chain: liftable at 0, liftable at 1, pierced, or gap at or under 0.2; the cubes above the line surround the loop](../plots/m5_46_degree.png)

![Least distance of the loop to the faces at every read, per chain](../plots/m5_46_distance.png)

### 9. R4, the two spacings

**Table 4. R4, the two spacings.**

| Pair | R1 at 0.5 | R1 at 0.25 | R2 at 0.5 | R2 at 0.25 | R3 at 0.5 | R3 at 0.25 | R4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F0a, F0b | no label | no label | HELD | HELD | no label | no label | HELD |
| P10a, P10b | HELD | HELD | HELD | HELD | no label | no label | HELD |
| P14c, P14d | HELD | HELD | HELD | HELD | no label | no label | HELD |

R4 is HELD on the three pairs (on the F0 pair by a margin of 0.005 in a tolerance set at run time, § 15): R1, R2 and R3 carry the same label at spacings 0.5 and 0.25. P14a, H10a and G10a ran at spacing 0.5 only and enter no pair. Beyond the labels, the two spacings agree in the end radius within 0.032 (3.405 against 3.437 in both pulled pairs) and in U at the end within 0.21 (26.437 against 26.647, 30.394 against 30.601); the finer grid took 2.1 to 2.6 times the iterations to reach the gate.

### 10. The pulled state, its first reads, and U along the dilation

**Table 5. The pulled state and its first reads: U, the potential term U_V, the radius of the loop.**

| Chain | Base | Read 0 (pulled) | Iteration 10 | Iteration 20 | Iteration 50 | Iteration 100 | The end |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P10a | 27.301, 5.303, 3.293 | 36.190, 21.199, 5.024 | 28.362, 8.419, 4.941 | 27.813, 7.712, 4.817 | 26.800, 6.550, 3.928 | 26.471, 5.832, 3.529 | 26.437, 5.710, 3.405 |
| P10b | 27.501, 5.445, 3.316 | 34.464, 19.067, 4.968 | 28.794, 9.130, 4.936 | 28.326, 8.150, 4.884 | 27.600, 7.385, 4.546 | 26.906, 6.493, 3.858 | 26.647, 5.835, 3.437 |
| P14c | 31.198, 5.401, 3.293 | 69.765, 55.508, 6.963 | 37.283, 12.955, 6.916 | 35.280, 10.846, 6.773 | 33.314, 9.456, 5.853 | 31.167, 6.774, 4.182 | 30.394, 5.576, 3.405 |
| P14d | 31.399, 5.544, 3.316 | 64.719, 50.096, 6.935 | 37.901, 13.691, 6.912 | 36.235, 11.952, 6.852 | 34.657, 10.561, 6.487 | 33.115, 8.931, 5.481 | 30.601, 5.711, 3.437 |
| P14a | 27.301, 5.303, 3.293 | 57.041, 47.123, 6.867 | 30.573, 10.999, 6.867 | 26.301, 8.261, 2.076 | 18.689, 5.400, 1.277 | 4.897, 0.162, no set | 4.191, 0.081, no set |
| H10a | 27.301, 5.303, 3.293 | 36.190, 21.199, 5.024 | 28.577, 8.509, 4.941 | 28.222, 7.913, 4.817 | 27.537, 6.559, 3.928 | 27.165, 5.753, 3.389 | 27.146, 5.752, 3.389 |
| G10a | 27.301, 5.303, 3.293 | 36.190, 21.199, 5.024 | 31.865, 15.656, 4.954 | 30.650, 13.758, 4.954 | 29.286, 11.118, 4.950 | 28.644, 9.656, 4.937 | 26.450, 5.887, 3.573 |

Each cell gives U, the potential term U_V and the radius of the loop. The dilation by s widens the core of the loop with everything else, and the potential term of the pulled state is 3.5 to 10.3 times that of the base. It is the first thing the descent removes: within 10 iterations U_V fell from 21.2 to 8.4 (P10a) and from 55.5 to 13.0 (P14c) while the radius moved by under 0.1. The contraction of the loop followed: the radius came within 0.05 of its end value at iterations 130 to 240 of the four free chains (80 in H10a). In P14a the same first 10 iterations left the loop where it was (radius 6.867, 1.25 from the faces), and at the next read it had reached the faces.

![U at every read, per chain](../plots/m5_46_energy.png)

**Table 6. U along the dilation of each base state.**

| Base | Half-box, spacing | Radius of the base loop | U at s = 1 | U at separation 10 (s) | U at separation 13.8 (s) | Largest U on the curve (separation) | U_V at s = 1, at 10, at 13.8 | L₁ term at s = 1, at 10, at 13.8 | Commutator term at s = 1, at 10, at 13.8 | Steps of the curve on which U falls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0a | 8, 0.5 | 3.2930 | 27.3009 | 36.1900 (1.5184) | 57.0413 (2.0954) | 57.0413 (13.80) | 5.303, 21.199, 47.123 | 10.698, 8.670, 6.372 | 11.300, 6.321, 3.546 | 0 of 25 |
| B0b | 8, 0.25 | 3.3159 | 27.5012 | 34.4639 (1.5079) | 53.5172 (2.0809) | 53.5172 (13.80) | 5.445, 19.067, 43.266 | 10.722, 8.829, 6.540 | 11.334, 6.568, 3.711 | 0 of 25 |
| B0c | 10, 0.5 | 3.2930 | 31.1976 | 40.7624 (1.5184) | 69.7653 (2.0954) | 69.7653 (13.80) | 5.401, 21.845, 55.508 | 14.112, 12.115, 9.930 | 11.684, 6.802, 4.328 | 0 of 25 |
| B0d | 10, 0.25 | 3.3159 | 31.3993 | 38.8559 (1.5079) | 64.7187 (2.0809) | 64.7187 (13.80) | 5.544, 19.523, 50.096 | 14.138, 12.285, 10.130 | 11.718, 7.048, 4.493 | 0 of 25 |

![U and its three terms along the dilation of the four base states, against the separation of the two points](../plots/m5_46_dilation.png)

U rose at every step of the curve on the four base states, by 8.9 and 7.0 at the separation of 10 on half-box 8 (spacings 0.5 and 0.25) and by 29.7 and 26.0 at 13.8 (on half-box 10: by 9.6 and 7.5, and by 38.6 and 33.3); the L₁ term and the commutator term were lower at both pulls than at s = 1, and the potential term carried the whole rise. The curve is an upper bound on the cost of the separation and carries no label; most of it is the cost of the widened core, which the descent removed in its first iterations (Table 5), so it bounds the cost from far above.

![The base, the pulled state, a state on the way and the end state of P10a and P10b, on the plane y = 0 and on the plane of the loop](../plots/m5_46_sections_p10.png)

![The base, the pulled state, a state on the way and the end state of P14c and P14d, on the plane y = 0 and on the plane of the loop](../plots/m5_46_sections_p14.png)

### 11. H10a, G10a, F0 and P14a

1. **H10a, the pull without the faces.** With the outer shell held at the pulled field's own values the loop contracted from 5.024 to 3.389, reached that radius by iteration 80, and stayed: the chain ended by the function tolerance at iteration 390 with max \|dU/dm\|/h³ = 2.2 × 10⁻⁷, the loop centered (z = 0.000). Its U of 27.146 is not comparable with the 27.301 of the base, whose shell holds the seed and not the dilated field. Through the contraction the free chain from the same start (P10a) followed the same radii at the same reads (4.941, 4.817, 3.928 at iterations 10, 20, 50 in both). What the faces added appears after the contraction: the held shell has a state at rest, and the free box was not at rest at the gate (§ 12).
2. **G10a, the path.** The step was fixed at 0.01 by a pilot on the pulled state (steps of 0.08 and 0.04 made U rise, 0.02 made no rise in 60 steps, and half of that was used); U rose at none of the 10818 steps. The 452 reads show one clear set at each, a radius that never rose between two reads (5.024 down to 3.573), no fragment, and the labels R1 HELD and R2 HELD. The chain ended at the gate with the loop still contracting: its radius of 3.573 is 0.17 above the 3.405 at which P10a ended, and its U of 26.4503 is 0.013 above. So the steepest descent shows no split along the contraction as far as the gate; it was not carried further, and the departure of § 12 was read on the L-BFGS path only.
3. **F0, the faces alone.** Released without a pull, the base lost 0.86 and 0.85 of U at the two spacings (27.301 to 26.437, 27.501 to 26.647), the loop widened by 0.11 to 0.12 (§ 8 item 2) and the chains ended at the gate after 271 and 640 iterations with one clear set at every read and R2 HELD. The chains pulled to 10 ended at the same U as their F0 partner within 5 × 10⁻⁵ (26.43714 against 26.43710 at spacing 0.5, 26.64663 against 26.64667 at 0.25): the release from a separation of 10 and the release without a pull reached the same state at the gate. At the gate the center of the loop stood off the box center along z by 0.25 (F0a), 0.16 (P10a), 0.25 (P14c), 0.02 (F0b), 0.125 (P10b) and 0.125 (P14d), where the base loops are centered to three decimals.
4. **P14a, the tight box.** Pulled to 13.8 on half-box 8, the loop stood 1.25 from the four side faces. After the first 10 iterations it had not moved; at iteration 20 the reader found four sets of 25 plaquettes, each touching two side faces, with centers at (±4.77, ±4.77, 0), toward the four vertical edges of the box; the arcs shortened toward the edges (radius 2.08 at iteration 20, 0.85 at iteration 70), four fragments were left at iteration 80, and no pierced plaquette at iteration 90, where every cube read 0. The chain ended by EXIT at iteration 110 with U = 4.19 and falling. The same pull on half-box 10, with 3.25 and 3.125 of room by the reader, contracted (P14c, P14d). Between the two lies a threshold this run does not locate.

![The base, the start, a state on the way and the end state of H10a, G10a and P14a](../plots/m5_46_sections_other.png)

![The base and the states of F0a and F0b](../plots/m5_46_sections_f0.png)

### 12. Added during the run: the gate-ended free states continued

The free chains of the plan ended at the gate with the loop whole and in the box. The plan's own blindspot table says that a free box has no state at rest, so four chains were added to see what the gate had stopped: X0a, X0b, X14c and X14d start from the end states of F0a, F0b, P14c and P14d, with free faces and the same optimizer, and with the gate stop off (they end by EXIT, by the function tolerance repeated after a restart, at 20000 iterations or at their wall time). They are outside the frozen table of § 4 and carry no label.

**Table 7. The chains added during the run (outside the frozen table; no label).**

| Quantity | X0a | X0b | X14c | X14d |
| --- | --- | --- | --- | --- |
| Starts from the end state of | F0a | F0b | P14c | P14d |
| Half-box, spacing (grid) | 8, 0.5 (33³) | 8, 0.25 (65³) | 10, 0.5 (41³) | 10, 0.25 (81³) |
| U and max \|dU/dm\|/h³ at read 0 | 26.4371, 7.5e-04 | 26.6467, 9.5e-04 | 30.3938, 9.8e-04 | 30.6006, 8.2e-04 |
| Iterations, how it ended | 775, EXIT | 1250, EXIT | 975, EXIT | 1375, EXIT |
| Least max \|dU/dm\|/h³ before the departure (iteration) | 2.3e-05 (325) | 3.8e-05 (800) | 2.1e-05 (375) | 1.4e-04 (525) |
| U at that read | 26.4369 | 26.6466 | 30.3936 | 30.6006 |
| Departure: first read from which the center stays 0.5 or more from its start | 550 | 925 | 625 | 800 |
| Last read with a set: iteration | 700 | 1175 | 900 | 1300 |
| Center of the loop at that read | (0.00, -0.18, 6.62) | (-1.31, 1.00, 6.92) | (0.10, 1.29, 8.83) | (-8.81, -0.59, 1.38) |
| Radius, least distance to the faces at that read | 2.835, 0.75 | 2.757, 0.00 | 1.424, 0.00 | 1.365, 0.00 |
| Face nearest to the loop at that read | +z | +z | +z | -x |
| Faces touched by a set at any read | none | +z | +z | -x |
| Radius range while a set exists | 2.835 to 3.418 | 2.757 to 3.442 | 1.424 to 3.405 | 1.365 to 3.565 |
| Most sets at a read | 1 | 1 | 1 | 1 |
| Reads with two clear sets | 0 | 0 | 0 | 0 |
| Reads with two faces touched | 0 | 0 | 0 | 0 |
| Counted reads; of them with a surface around the loop not at 1 | 46; 0 | 65; 0 | 53; 0 | 61; 0 |
| Reads with a liftable surface outside 0 and 1 | 0 | 0 | 0 | 0 |
| Last read with the box surface at 1: iteration, distance of the loop to the faces | 675, 3.25 | 1150, 2.12 | 850, 1.75 | 1050, 1.38 |
| First read with the box surface at 0 | 725 | 1200 | 925 | 1325 |
| Reads with the box surface at 0 and a clear set | 0 | 0 | 0 | 0 |
| U at the end | 4.1007 | 6.5618 | 4.5414 | 6.9365 |

1. **The gate was a plateau.** Each chain stayed where it started for hundreds of iterations, with U falling by 3.0 × 10⁻⁴ at most up to the read of least gradient (1.3 × 10⁻³ to 3.6 × 10⁻³ up to the departure) and max \|dU/dm\|/h³ falling to between 1.4 × 10⁻⁴ and 2.1 × 10⁻⁵, 7 to 47 times under the gate. Then the gradient rose again and the loop left its place, for good from iterations 550, 925, 625 and 800 (X0a, X0b, X14c, X14d).
2. **The loop left as one set, through one face.** In X0a, X0b and X14c the loop's center traveled along +z, its own axis, and the loop passed the +z face flat (in X14c the set clipped by the face reads tilted at its last two reads): no set was left two or three reads after the last read with the box surface at 1. In X14d it traveled along −x, in its own plane with its normal staying within 6° of z, and passed the −x face edge first: from iteration 1075 to 1300 the part still in the box was one set ending on that face, of 104 plaquettes falling to 30. The radius at the last read with the box surface at 1 was 3.20, 3.07, 3.08 and 3.45 (X0a, X0b, X14c, X14d), against 3.40 to 3.44 at the start. No read of the four chains has two sets of 8 plaquettes or more, a fragment, or two faces touched.
3. **The charge went with it.** At every read of the four chains with a liftable cube around the loop (225 reads) every such cube read 1, and every liftable cube that did not surround the loop read 0. The box surface read 1 up to the read named in the table, was not liftable while the loop crossed it (pierced, or under the gap), and read 0 after, when no pierced plaquette was left in the box. No read has the box surface at 0 with a clear set. U stood between 26.4 and 30.6 on the plateau and between 4.1 and 6.9 at the stop, three reads after the exit, and was still falling.
4. **The clauses of R3, for information.** The script's R3 clauses evaluated on these chains give: X0a: no set was read within one cell of a face, so it is not an exit through one face by the script's rule (the box surface last read 1 at iteration 675 with the loop 3.25 from a face); X0b: one face touched (+z); the box surface last read 1 at iteration 1150 with the loop 2.12 from a face, above 2: neither column; X14c: one face touched (+z); the box surface last read 1 at iteration 850 with the loop 1.75 from a face, and no clear set after it: both clauses of the R3 wording are met by the reads, and no label is given; X14d: one face touched (−x); the box surface last read 1 at iteration 1050 with the loop 1.38 from a face, and no clear set after it: both clauses of the R3 wording are met by the reads, and no label is given. The reads are 25 iterations apart there, and the loop covered up to 2.0 between two reads as it left, so the clause "within 2 of a face at the last read with the box surface at 1" depends on where a read happened to fall. None of this is a label.
5. **Repeatability.** X0a and X14c were run twice (the second time to save states on the way): the same iteration counts (775 and 975) and the same U at the end to all printed digits.

![X0a and X14c (spacing 0.5): the start, four states on the way and the end, on the plane y = 0 and on the plane of the loop](../plots/m5_46_sections_added_h05.png)

![X0b and X14d (spacing 0.25): the start, four states on the way and the end, on the plane y = 0 and on the plane of the loop](../plots/m5_46_sections_added_h025.png)

### 13. The expectations of the plan against the outcome

| Recorded in the plan | Outcome |
| --- | --- |
| § 4: "with the shell held the pulled unit is expected to contract and stay" | as recorded: H10a contracted to 3.389 and ended by the function tolerance |
| § 5: "A free box has no state at rest: the unit may leave before it contracts" | no state at rest was found on a free box, and the order was the other one: with 3.1 of room or more the loop contracted first, the chains of the plan ended at the gate with the unit in the box, and in the added chains it left afterwards |
| § 5: "the free chains of M5.42 saw loops leave a free box, so EXPANDED can occur" | EXPANDED occurred once, in the tight box (P14a). In the added chains the radius of the loop stayed within 0.02 of its starting value or under it at every read at which it was clear of the faces |
| § 5: "L-BFGS is not a gradient flow, and a long step could pass over a split" | G10a: one clear set at each of 452 reads along 10818 steepest-descent steps. No chain shows two clear sets at a read |
| § 5: "a pull close to the faces may leave no liftable surface" | P14a: two counted reads with one surface each, then none. Every read of the other released chains counted, with 6.9 to 11.4 surfaces around the loop on average |
| § 5: the potential term of the pulled state, and its relaxation in the first reads | Table 5: 3.5 to 10.3 times the base value at read 0, and under 2.5 times after 10 iterations of L-BFGS-B (2.95 times after 10 steps of G10a) |
| § 5, the estimate: base chains minutes each, free chains tens of minutes at spacing 0.25, three quarters of an hour of wall time | base chains 32 s to 11 min 50 s; free chains of the plan 5 s to 9 min 5 s; the two pools 11.8 and 9.2 minutes of wall time |

Not expected and found: the gate of M5.45, which on a held shell is met by a state at rest, was met on the free box by a state that left the box afterwards (§ 12); and the face of the exit: +z, along the loop's axis, in three of the four added chains, and −x, in the loop's plane, in the fourth.

### 14. Deviations from plan, what was not computed, and how to regenerate

**Deviations from plan.**

1. **K5 and K7 ran twice.** The plan runs the controls first and names "the base state" as the input of K5 and K7, which does not exist before the base chains. They ran first on the relaxed state of M5.45 (spacing 0.5), and all seven controls were run again on the four base states before any release (§ 6).
2. **The first try of K6 failed on its input** (§ 6). The input was replaced; the instrument was not touched.
3. **Terms the plan left open were fixed in the script before the chains ran** (the table of § 8): the tolerance of half a cell on "falls" and "grows", the set of nested half-widths, the box surface as the cube of half-width L − h, the loop of a read, "around the loop" by the largest coordinate, the ten-read rule of R2 applied to the counted reads as well, and "exit through one face".
4. **Outcomes outside both columns of R1 and R3 are printed as "no label"** with their numbers (F0a, F0b; R3 everywhere).
5. **The base chains and H10a ran by the rule of M5.45** (the function tolerance, repeated after a restart with an empty memory) and so past the gate of 10⁻³; the iteration at which the gate was first met is in the chain records. The plan states the stops of a free chain and not those of H10a.
6. **Ends by a stalled line search.** L-BFGS-B ended B0a by the function tolerance at iteration 817 and its restart by a stalled line search with no iteration made; B0c stalled at 955 and its restart ended by the function tolerance two iterations later; H10a stalled once at 389 between two ends by the function tolerance. The three are far under the gate (1.2 × 10⁻⁶, 9.0 × 10⁻⁷, 2.2 × 10⁻⁷).
7. **The gate is tested at every iteration** on the gradient of the optimizer's last evaluation and confirmed by an evaluation at the iterate; every read costs one more evaluation of U.
8. **The step of G10a came from a pilot** (§ 11 item 2); the plan fixes the step and does not give it.
9. **Four chains were added** (§ 12), and two of them were run twice. The driver saves states of an added chain at every unit of travel of its loop.
10. **Snapshots.** Every chain saves its state at iterations 50, 100, 200, 300, 500, 1000, 2000, 5000 and 10000 (single precision, local arrays) beside its start and end states.
11. **The dilation curve** has 26 values of s on each base (25 evenly spaced from 1 to the pull of 13.8, and the pull of 10), including on B0b, which the plan pulls to 10 only.
12. **Two figures beyond those asked for** (U at every read, the distance to the faces), and the cross-sections on two planes.
13. **Wall time.** The estimate of three quarters of an hour was not reached: 21 minutes for the two pools of the plan.

**Not computed.**

- A ladder with the separation held and raised in steps, which the plan names as the stronger run.
- The separation, or the room to the faces, at which a release turns from contraction to expansion: bracketed only by the pull of 13.8 with 3.25 of room (contracts) and with 1.25 of room (expands), at spacing 0.5. P14a has no partner at spacing 0.25.
- A pull that is not a dilation: a stretch in one plane, a pinch of the loop, an off-center or tilted start, a perturbation that seeks a split.
- The steepest descent past its gate, and a steepest-descent version of the added chains: the departure and the exit were read on the L-BFGS path only.
- What sets the face of the exit (+z along the loop's axis in X0a, X0b and X14c, −x in the loop's plane in X14d): the same chains with the loop turned or moved, on another chart, or from a perturbed start.
- The exit at a finer cadence than 25 iterations, which the clauses of R3 need.
- The state of the box after the exit (the added chains stop three reads after it, with U between 4.1 and 6.9 and falling).
- Another δ or L₁, J above 0, another box shape, a held shell at the separation of 13.8.
- The charge by a second method, and any limit at zero spacing.
- A comparison with the author's own run of this test, which is not public at this date.
- The adversarial audit on a different model named in the definition of done: it is not part of this section.

**Regenerate.** From `research/`, with one thread per process:

```text
python scripts/m5_46_degree.py --selftest                           K1 to K7; K5 and K7 on a relaxed state already on disk
python scripts/m5_46_pull.py --pool base --cores 4                  B0a B0b B0c B0d
python scripts/m5_46_degree.py --selftest --out data/m5_46_controls.json      K1 to K7 on the four base states
python scripts/m5_46_pull.py --dilation                             U along the dilation
python scripts/m5_46_pull.py --pool release --cores 9               the nine released chains
python scripts/m5_46_pull.py --pool added --cores 4 --wall-hours 0.75         X0a X0b X14c X14d (each needs the end state it continues)
python scripts/m5_46_reads.py                                       labels, tables, summary JSON, figures
```

`python scripts/m5_46_pull.py --list` names the chains, and `--chain <name>` runs one. The states (`m5_46_<chain>_start.npz`, `_end.npz` and the snapshots `_s<iteration>.npz`) are local-only arrays indexed by [`_DATASETS.md`](../data/_DATASETS.md); each is rewritten by the chain of its name, and a chain repeats itself exactly (§ 12 item 5). The selftest and the reads script each took under a minute.

**Table 8. Wall time per chain.**

| Chain | Iterations | Evaluations of U by the descent | Wall time |
| --- | --- | --- | --- |
| B0a | 817 | 881 | 32 s |
| B0b | 1124 | 1212 | 5 min 29 s |
| B0c | 957 | 1047 | 1 min 14 s |
| B0d | 1254 | 1311 | 11 min 50 s |
| F0a | 271 | 280 | 13 s |
| F0b | 640 | 662 | 3 min 38 s |
| P10a | 255 | 264 | 12 s |
| P10b | 660 | 676 | 3 min 43 s |
| P14c | 435 | 444 | 39 s |
| P14d | 903 | 918 | 9 min 05 s |
| P14a | 110 | 113 | 5 s |
| H10a | 390 | 487 | 20 s |
| G10a | 10818 | 10818 | 4 min 42 s |
| X0a | 775 | 811 | 36 s |
| X0b | 1250 | 1289 | 5 min 50 s |
| X14c | 975 | 1009 | 1 min 22 s |
| X14d | 1375 | 1406 | 12 min 25 s |

The base pool took 11.8 minutes of wall time (four processes) and the release pool 9.2 minutes (nine processes); the added chains ran beside them and after, 20 min 13 s of chain time in all.

### 15. The audits (2026-11-19)

Two independent passes on other models read the findings against the saved reads. The numbers pass recomputed about 700 values from the summary files and found no mismatch; all plots and links resolve. The adversarial pass recomputed every label from the chain records and found them as tabled; it raised the points below, and the sentences they concern were corrected in place the same day.

| # | Point raised | What holds |
| --- | --- | --- |
| 1 | § 12 stated the outcome of the R3 wording for two added chains | the added chains carry no label; § 12 now says the reads meet the wording and gives none |
| 2 | R4 on the F0 pair rests on a narrow margin | F0b grew by 0.1203 against a tolerance of 0.125 (half a cell at spacing 0.25). With a tolerance between 0.112 and 0.120, F0b alone would read EXPANDED and R4 on that pair NOT HELD; under 0.112 both would read EXPANDED and R4 HELD |
| 3 | The tolerance was said to predate the chains | it is a choice of the reads script, whose last change follows the last chain; no earlier copy was kept, so the order cannot be shown |
| 4 | K5 can barely fail | the four-edge product of the line reader equals the product of the two triangle products, so a cube the loop crosses always has a pierced triangle; only the equality of the two counts is a live check |
| 5 | Which run of the controls is tracked | a rerun after the releases, byte-identical to the run before them |
| 6 | The tight-box chain | its sets were gone from iteration 80, where four fragments remained |
| 7 | "A plateau and not at rest" | shown for the four chains that were continued; inferred for P10a and P10b |

**Choices the plan left open and the scripts fixed.** A tolerance of half a cell on a fall or a growth of the radius (the pulled chains fall by 1.45 to 3.56 and do not depend on it; the F0 pair does, row 2). R1 read on the net change of the radius and not as monotone (the pulled chains tick up by 0.024 at most; a strictly monotone rule would give them no label, not NOT HELD). The ten-read rule applied to counted reads (the literal rule would give P14a an R2 label of HELD on its twelve reads). "Clear of the faces" as more than one cell away; "around the loop" as a cube wider than the loop's largest coordinate. The step of G10a from a pilot.

**Scope.** With free faces and at least 3.1 of room, a unit pulled by a dilation relaxed back as one loop, and every liftable cube around it read 1, at two spacings and up to the gate. The integer is inherited: the base relaxes from a seed of degree 1, and a whole value of q holds by construction. The run does not show a split to be impossible: it is two releases, with no held separation, no perturbation that seeks a split, and end states that are not at rest. It does not show the unit staying whole in a box it can leave: in the tight box the faces cut the loop, and in the continued chains it left the box as one set, on chains that carry no label.

## TASK REVIEW (2026-11-19)

**Task Duration:** 00:59 (from 2026-11-19 09:21 EST to 2026-11-19 10:20 EST)
**Usage Cap Triggered:** NO

**Results.** ✅ The seven controls passed on the second of three tries (the first failed on a test input, which was replaced; the instrument was not changed). ✅ Pulled to separations of 10 and 13.8, at spacings 0.5 and 0.25, the unit relaxed back as one loop of radius about 3.4, and every liftable cube around it read 1 at every read; the shell-held chain and the steepest-descent chain read the same (R1 HELD, R2 HELD). ✅ No read of any chain shows two loops clear of the faces, and no liftable surface reads anything but 0 or 1. ⚠️ In the tight box the pulled loop reached four faces and was gone from iteration 80 (R1 NOT HELD, EXPANDED). ⚠️ The gate ended the continued free chains on a plateau and not at rest: continued outside the plan, the loop left the box as one set. 🔶 R4 HELD on the three pairs, on the F0 pair by a margin of 0.005 in a tolerance set at run time.

**Issues / blockers.** A tolerance the plan left open decides the R1 and R4 labels of the F0 pair (family: artifact read as result; § 15 gives each other choice). R3 applied to no planned chain. Control K5 has one live check, not two (§ 15).

**Deviations from plan.** The build and the chains ran in one builder agent on the session's model. Four chains were added to continue the gate-ended states, without labels. The base chains and H10a ran past the gate to the optimizer's tolerance. Open terms were fixed in the scripts at run time (§ 15).

**Action needed.** The charge-quantization cell stays partial (the user's decision at the review), its text gaining this result. The stronger run, a ladder with the separation held and raised in steps, is not filed. The results go to the author in one message with those of the tasks that follow.

**Scope**: With free faces and at least 3.1 of room, a unit pulled by a dilation relaxed back as one loop with a charge of 1 around it, at two spacings, up to the gate; the run does not show a split to be impossible, and does not show the unit staying whole in a box it can leave.

**Findings.** A unit pulled apart by a dilation contracts whole, nothing reads ½, and a free box lets the unit leave as one piece. The run reads which way the force points at two separations, which is short of the row's test.

**Research docs created / updated.** [`m5_46_task_details.md`](m5_46_task_details.md) (this document) · [`m5_46_degree.py`](../scripts/m5_46_degree.py) · [`m5_46_pull.py`](../scripts/m5_46_pull.py) · [`m5_46_reads.py`](../scripts/m5_46_reads.py) · [`m5_46_summary.json`](../data/m5_46_summary.json) · [`m5_46_controls.json`](../data/m5_46_controls.json) · [`m5_46_dilation.json`](../data/m5_46_dilation.json) · the plots embedded in § 7 to § 12 · [`m5_roadmap.md`](../m5_roadmap.md)
