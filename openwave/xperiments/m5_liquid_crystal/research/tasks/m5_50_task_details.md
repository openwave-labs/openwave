# M5.50: Charge quantization: the two +½ lines held apart on a ladder, in a larger box

> Opened 2026-11-19, staged 2026-11-20. Roadmap row: [`m5_roadmap.md`](../m5_roadmap.md). Thread record: [`m5_32_convo.md`](m5_32_convo.md). The cell it reads for: charge quantization in the M5 table of [`MODELS.md`](../../../../../MODELS.md).

## TASK PLANNING (2026-11-20; staged: the readings of § 4 are frozen before any rung is relaxed)

**Scope.** The run that [M5.46](m5_46_task_details.md) named as the stronger one and did not make: the relaxed unit with the separation of its two +½ lines held at a value and that value raised in steps, the energy and the charge read at every step. The row's test is an integer topological charge with a split e → e/2 + e/2 impossible; its regime is `static`, so a descent is the process. The run reports one word per reading of § 4, or "no label" with the rule that withheld it. Out of scope: two straight lines in place of the loop; another δ or L₁; J above 0; a limit at zero spacing; a box beyond half-box 14; any icon or pointer of `MODELS.md`.

**Where M5.46 left it.** Pulled by a dilation to separations 10 and 13.8 and released with free faces, the unit relaxed back as one loop and every liftable surface around it read 1; nothing read ½; in a box it could leave it left as one set. The author's own run, posted on discussion #186 on 2026-11-19 ([discussioncomment-19756045](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19756045); the row of that date in the [thread record](m5_32_convo.md)), reports the same on their code. The cell stayed partial on two limits that M5.46 states in its own scope: the integer on a liftable surface holds by construction, the base relaxing from a seed of degree 1; and a release reads the sign of the force at two separations without holding one, so a split is not shown impossible.

| # | What a ladder adds to a release | What it has to read |
| --- | --- | --- |
| 1 | The energy as a function of a separation that is held, not passed through | U at each held separation, the state relaxed under the constraint, converged by a stated gate |
| 2 | Whether the energy keeps rising as far as the box allows, or flattens (a split that costs nothing more) | the slope of U against the separation over the last steps, at two spacings and in two boxes |
| 3 | Whether the charge splits before the box does | the sets of the line and the charge on nested surfaces at every step, with the instrument of M5.46 |
| 4 | A perturbation that seeks a split, which a release never applied | a stated family of perturbations at the largest held separation, relaxed under the constraint |

**The clause the run reads.** The author posted on discussion #186 on 2026-11-20 a definition of "stays whole" for this row ([discussioncomment-19766100](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19766100); the row of that date in the thread record), in three parts: (a) every liftable surface enclosing the unit reads 1 from the pull until the exit; (b) no liftable surface ever reads ½; (c) the reads between the last 1 and the first 0 are exactly those in which the loop pierces the surface. A ladder in a box with held faces has no exit, and on it neither (a) nor (b) can fail for a state with one set (limit (2)): the informative read of the ladder is the number of clear sets, at rest under the hold and at rest after the hold is taken away. Parts (a) and (c) are tested on the chains that leave a box with free faces (R4 of § 4). M5.46 read (a) and (b) on its planned chains and not (c).

**Five limits, stated before the run.** (1) On a grid the two +½ lines of the clause are the two points at which a plane through the loop's axis meets one closed loop. The separation held here is the size of that loop, which separates the two points in every such plane at once. For a closed line with a positive energy per length, U rises with its size by geometry, so a rising U is the expected outcome; what the ladder can refute is that the cost per unit of separation stays positive and does not fall away as far as the box allows. (2) Part (b) cannot fail on this instrument: the degree of a liftable surface is a whole number by construction (M5.46 § 3), and it is read in the form M5.46 used, no liftable surface outside 0 and 1. With held faces part (a) cannot fail either for a state with one set: the faces hold a frame of degree 1, and a liftable cube around all of the line is joined to the faces by a shell that holds no line, so it reads the degree of the faces. No outcome of this run removes the first limit of the cell. (3) The reads along a chain are reads along an optimizer's path and not along a motion in time. (4) Two boxes and two spacings are not a continuum limit: every label is the label of named grids. (5) The room rule of § 4 is learned on loops of radius up to about 6.4, the largest that half-box 10 holds, and is applied as a room to the larger loops of half-box 14: an extrapolation in the size of the loop.

**Definition of done.** The controls of § 3 each reported as met or as the exact failing check; every chain of § 2 ended by its own rule or cut by a cap, with its word; the tables of § 4 written by the reads script from a tracked JSON; R1 to R5 each carrying one word or "no label" with its rule, per grid; the expectations of § 4 scored against the outcome; the findings embedding U and the force of the hold against the separation for the four ladders, the cross-sections of the base, the top rung and each end state of the perturbation family, and the surface reads of the exit chains; an adversarial audit on a different model, with its own evaluation of U and of the held quantity on saved end states, having tried to refute each word; the doc checker and the roadmap linter clean.

**Gating.** An audit of this plan by an independent model, folded here; then the user's go. No answer of the author is waited for, and nothing in this plan reads a result of [M5.49](m5_49_task_details.md).

**Research body.** This document (plan, findings, review); scripts `m5_50_hold.py` (the hold, the driver and the pool), `m5_50_reads.py` and `m5_50_report.py` in [`../scripts/`](../scripts/); summaries `m5_50_*.json` in [`../data/`](../data/) (tracked), the states `m5_50_*.npz` there (local-only, indexed by [`_DATASETS.md`](../data/_DATASETS.md)); plots `m5_50_*.png` in [`../plots/`](../plots/); checkpoints, chain records and progress notes in `../checkpoints/m5_50/` (local-only).

### 1. The action, the hold and the descent rule

The action is that of [M5.42](m5_42_task_details.md) as coded in [`m5_42_eq42.py`](../scripts/m5_42_eq42.py), at J = 0, unchanged:

$$
U = h^3 \sum_n \Big[ L_1 \sum_i \lVert \partial_i M \rVert^2 + \kappa \sum_{i \lt j} \lVert [\partial_i M, \partial_j M] \rVert^2 + v \sum_{k=1}^{3} \big( \operatorname{tr} M^k - c_k \big)^2 \Big]
$$

with κ = v = 1, δ = 0.05, L₁ = 0.03, c_k = 1 + δ^k, the mirror-averaged stencil, the chart fixed, the faces held to the frame except where a chain is named free.

**The hold.** The potential density is zero in the vacuum and largest on the core of the line, where the two largest eigenvalues meet. Its square is taken as a density of the line, and the separation held is twice the mean distance of that density from its own center:

$$
u_n = \sum_{k=1}^{3} \big( \operatorname{tr} M_n^k - c_k \big)^2 , \qquad w_n = u_n^2 , \qquad \bar{x} = \frac{\sum_n w_n x_n}{\sum_n w_n}
$$

$$
D[M] = \frac{2 \sum_n w_n \lVert x_n - \bar{x} \rVert}{\sum_n w_n}
$$

$$
E_\mu = U + \frac{\mu}{2} \big( D[M] - d \big)^2 , \qquad f = \mu \big( d - D[M] \big)
$$

A rung is a descent of E_μ at a target d. At a state at rest of E_μ the gradient of U equals f times the gradient of D: the state is a stationary point of U on a level set of D, whatever μ, and f is the slope of U against D along the family of such states. The hold energy is reported apart from U and is never part of it. Between two neighboring rungs k and k + 1 on one branch of states, the step identity and its error, the error always taken against the absolute value of the step in U:

$$
T_k = \tfrac{1}{2} \big( f_k + f_{k+1} \big) \big( D_{k+1} - D_k \big) , \qquad e_k = \frac{\lvert ( U_{k+1} - U_k ) - T_k \rvert}{\lvert U_{k+1} - U_k \rvert}
$$

| Choice | Reason |
| --- | --- |
| A functional of the field, not pinned nodes | pinned nodes fix a core profile by hand and their energy is not relaxed; here every interior node stays a variable of U |
| The square of the potential density | a faint halo over a large volume costs energy in proportion to u and gains weight in proportion to u², so the hold cannot be met cheaply by a halo. On the base states of M5.46 at spacing 0.25, 89 percent of the weight lies within 1.0 of the loop's circle (63 percent for the first power) |
| The distance from the weight's own center, in three dimensions | the hold names no point and no axis of the box, so a loop that moves or tilts gains nothing |
| The held value measured apart from the hold | the abscissa of every reading is the separation of the line reader of [M5.44](m5_44_task_details.md), twice the radius of the one clear set, and D is printed beside it. On the base states D is 6.87 and 6.88 against 6.59 and 6.63 from the reader |
| Two strengths, μ = 20 and μ = 80 | the points of both must lie on one curve U(D) (K8) |
| A second kind of hold, as a control | an obstacle on the director inside the loop, which presses on the line from the inside and not on its core (K9) |
| A release of every state in two pieces | the hold can be met by two pieces: with weights W₁ and W₂ at a distance s, D is about 4 W₁ W₂ s / (W₁ + W₂)², so the unit at its free size with a small second loop far from it meets a large target at a cost that does not grow with the target. A split found under the hold is therefore tested again with the hold taken away (below) |

**The release.** Every CONVERGED state with two clear sets or more, from a rung or from the family of § 4, is relaxed again with no hold, the faces held, by the same descent rule. Two words follow, and the reads of § 4 use them.

| Word | Rule |
| --- | --- |
| PERSISTENT SPLIT | the release is CONVERGED and ends with two clear sets or more: the line is in two pieces at rest with no hold |
| MADE BY THE HOLD | the release is CONVERGED and ends with one clear set. Printed with E_μ and U of the split state against the one-loop state at the same target, where one exists, and with the share of the weight that lies within 1.5 of each piece |

**The second kind of hold.** On the plane of the loop, inside it, the long axis stands along the loop's normal (on the base states M_zz is above 0.97 out to 2.0 from the center). With Δ(R_d) the nodes of the plane z = 0 at a distance of at most R_d from the axis:

$$
E_o = U + \frac{\mu_o}{2} h^3 \sum_{n \in \Delta(R_d)} \big( 1 - (M_n)_{zz} \big)^2
$$

with μ_o = 100. A loop that contracts into the disk pays; a loop outside it pays little.

**The descent rule** is that of M5.45 § 4 as [M5.48](m5_48_task_details.md) § 1 states it, with E_μ in place of U: L-BFGS-B over the six components of every free node, 20 correction pairs, function tolerance 10⁻¹⁵; after an end by the function tolerance a restart with an empty memory, until a restart ends within 25 iterations (40 restarts at most); a stalled line search restarted three times at most; a checkpoint at every stop, from which an interrupted chain continues with an empty memory, the resume recorded.

| Word | Rule |
| --- | --- |
| The gate | the largest entry of the gradient of E_μ over h³ at the end is at most the smaller of 10⁻³ and one tenth of the largest entry of the hold's own force there (f times the gradient of D, over h³), and not under 10⁻⁴, the bound that serves where no hold acts |
| CONVERGED | not cut by a cap, the gate met, and an end by the function tolerance repeated after a restart (or, for a chain with a stalled line search in its record, the gate alone, the driver's word printed beside it) |
| STALLED | an end by the function tolerance repeated within 25 iterations with the gate not met, restarted three times at most; a chain that ends so is never CONVERGED |
| EVADED | a CONVERGED rung whose end state has one clear set and either a reader's separation more than 0.75 from D, or less than 0.80 of the weight within 1.5 of the loop's plaquettes. The hold did not hold the line; the rung gives no point of the curve |

A chain that is not CONVERGED carries no value of U into any reading.

### 2. The ladder and the chains

Two boxes, half-box 10 and half-box 14, at spacings 0.5 and 0.25: grids of 41³ and 81³, 57³ and 113³. The base states of half-box 10 are those of M5.46 (B0c, B0d). The base states of half-box 14 are relaxed from the seed by the rule of M5.45 (B14a, B14b). A rung starts fresh: the base dilated about its loop's center by the ratio of the target to the base's D (the dilation of M5.46), with the faces set back to the frame. The targets rise in steps of 1.5 at spacing 0.25 and of 0.75 at spacing 0.5. At the top target the loop stands 3.6 from the nearest face on half-box 10 and 3.9 on half-box 14.

| Class | Chains | Grid | Start | Caps (iterations; wall time) |
| --- | --- | --- | --- | --- |
| Bases | `B14a`, `B14b` | 57³, 113³ | the seed, faces held | 20000; 1 hour, 2.5 hours |
| The free state (K6) | on each of the four grids: the base continued with no hold, and the rung whose target is the base's own D, μ = 20. The second is the origin of the ladder of its grid | all four | the base | as the rung of the grid |
| The force of the hold (K7) | four rungs on half-box 10 at spacing 0.5, with targets 0.1, 0.2, 0.3 and 0.4 above the base's D, μ = 20 | 41³ | the dilated base | 20000; 0.5 hours |
| Rungs, half-box 10 | targets 7.75 to 13 in steps of 0.75 (spacing 0.5), and 8.5, 10, 11.5, 13 (spacing 0.25); inward targets 5.5 and 4, descriptors; μ = 20 | 41³, 81³ | the dilated base | 20000; 0.5 hours, 2 hours |
| Rungs, half-box 14 | targets 7.75 to 20.5 in steps of 0.75 (spacing 0.5), and 8.5 to 20.5 in steps of 1.5 (spacing 0.25); μ = 20 | 57³, 113³ | the dilated base | 20000; 1 hour, 3 hours |
| The second strength | μ = 80 on every outward rung at spacing 0.5; at spacing 0.25 on targets 10 and 13 (half-box 10) and 14.5 (half-box 14) | as the rung | as the rung | as the rung |
| Bisections | by the rule of § 4: one rung at the middle target of a step that fails the identity, two per ladder at most, so eight at most | the grid of the step | the dilated base | as the rung |
| Branch checks | by the rule of § 4: one chain per step on which U falls between two eligible rungs; their number is not known before the run | the grid of the step | the end state of the upper rung, under the hold at the lower target | as the rung |
| Continuation | at spacing 0.5 on each box: one chain up the ladder, each rung from the end of the rung below, and one chain back down from the top | 41³, 57³ | the rung before | as the rung, per rung |
| The second hold | the obstacle at R_d = 1.5, 4.5, 5.5 (half-box 10) and 1.5, 5.5, 8 (half-box 14), spacing 0.5 | 41³, 57³ | the base dilated so that its loop stands 0.5 outside the disk; for R_d = 1.5 the base | as the rung |
| Perturbations | the family of § 4 on the end state of the top counted rung: six chains per grid, three on 113³ | all four | the perturbed state, under the hold of that rung | as the rung |
| Releases | by the rule of § 1: one per CONVERGED state with two clear sets or more, from a rung or from the family; their number is not known before the run | the grid of the state | that state, no hold, faces held | as the rung |
| Exits, the top rung released | `Z10a`, `Z10b`, `Z14a`, `Z14b`: the end state of the top counted rung with free faces and no hold, the gate stop off | all four | as named | 20000; as the rung |
| Exits, M5.46 again | `E0a`, `E0b`, `E14c`, `E14d`: the chains X0a, X0b, X14c and X14d of M5.46 § 12 run again from the same end states (F0a, F0b, P14c, P14d) | 33³, 65³, 41³, 81³ | as named | 20000; 0.5 hours, 1 hour, 0.5 hours, 2 hours |

**Reads and stops.** Reads are taken at the start, every 10 iterations up to 300, every 25 after, and at the end, with the instrument of M5.46: the cadence of M5.46. An exit chain is read in addition at every iteration from the first cadence read at which its loop stands within 2 of a face or the box surface does not read 1, whichever comes first. These dense reads are recorded and enter R4; no stop rule counts them. An exit chain stops by the EXIT rule of M5.46, counted on the cadence reads alone (no set clear of the faces and the box surface at 0 on three cadence reads in a row), by the descent rule, or at a cap.

**Parallel shape.** One single-threaded process per chain (the five thread variables set to 1 before NumPy loads). The pool holds ten processes at once, under the ceiling of twelve, and eight while the wave jobs of [M5.49](m5_49_task_details.md) run beside it. Order: the bases, the controls, the chains of K6 and K7 and the four `E` chains, which depend on nothing; then the whole of spacing 0.5 on both boxes, which gives a first reading on two boxes while the large grids run; then the rungs at spacing 0.25, the largest grid first. Dependencies: a rung on half-box 14 waits for its base; a bisection or a branch check waits for the two rungs of its step; the perturbations, the `Z` chain and the releases of a grid wait until the counted run of that grid is final, which needs every rung of both boxes at that spacing ended, their bisections and branch checks ended, and K8 read, since the room rule joins the two boxes; a release waits for the state it releases. The longest dependent path is `B14b`, the rungs of 113³, a bisection or a branch check on 113³ where a step calls for one, a member of the family or `Z14b` on 113³, and the release of a member that ended in two sets. Inside a chain the descent is sequential by nature. The GPU is not used: the kernel is double precision. The pool script holds the order and the dependencies and is restartable, a finished chain read from its record and an interrupted one continued from its checkpoint.

### 3. The instruments and their controls

The line and the charge are read by [`m5_46_degree.py`](../scripts/m5_46_degree.py), unchanged: the sets of pierced plaquettes with their place against the faces, and the absolute degree on every nested cube surface that is liftable. One addition: at a read with two sets or more, cubes centered on each set are read as well, so that the whole number around each piece is on record.

| Control | Input | The answer it must return | What a failure would look like |
| --- | --- | --- | --- |
| K1, the states read back | the end files of B0c and B0d of M5.46 | U of [`m5_46_summary.json`](../data/m5_46_summary.json) to 10⁻¹² (relative); one loop of radius 3.293 and 3.316 | a wrong δ, L₁ or file: a gap of order 1 |
| K2, the new bases | `B14a`, `B14b` | CONVERGED; one loop, centered within 0.01, its radius within 0.03 of the base of half-box 10 at the same spacing | no loop, two loops, or another radius |
| K3, the instrument of M5.46 | its seven controls on the four base states | as in M5.46 § 6 | as in M5.46 § 3 |
| K4, the held quantity | a base state shifted by whole cells along z and along x, the cells that the shift leaves empty at the faces filled with the frame; a base dilated by 1.5 and by 2; the base and the dilated states against the reader | D unchanged by the shift to 10⁻⁸; D in the ratio of the dilation within 2 percent; D above the reader's separation by 0.1 to 0.6 | the same sum taken about the box center moves by more than 0.1 under the shift |
| K5, the gradients | central differences at 200 random interior entries, the best of three steps, on a dilated base, which is not at rest, with the target 0.5 above its D so that the hold term is not zero: the gradient of D alone, the hold term alone, the whole E_μ, and the obstacle term | each within 10⁻⁶ of its largest entry | the term of the moving center left out, or a wrong sign |
| K6, the free state returned | on each grid, the rung whose target is the base's own D, against the base continued with no hold under the same rule and gate | U equal to 10⁻⁸ (relative), the hold energy under 10⁻¹², the loop unchanged | a target 0.5 above must raise U by more than 10⁻³ |
| K7, the force of the hold | the four rungs of § 2, all on one side of the free state | the error e of the step identity under 1 percent on each of the three steps between them | f with the other sign, or μ doubled: off by a factor |
| K8, two strengths | the points at μ = 80 against the cubic through the points and slopes at μ = 20 | within the larger of 2 percent of the rise of U from the free state and 0.005 | a hold energy counted into U: a gap that grows with μ |
| K9, the second hold | the obstacle at R_d = 1.5; then at the larger disks | at 1.5: the obstacle energy at the end not above its value on the base, U within 10⁻³ (relative) of the base, the loop's radius within half a cell. At the larger disks: U against the curve of the first hold at the same reader's separation, printed as a fraction of the rise and used by R1 | the disk laid on another plane than the loop's: the loop is pushed and U rises at 1.5 |
| K10, the readers of the labels | the made-up records of the table below, passed through the reads script | the word listed beside each record | a reader that returns HELD on every record |
| K11, the exits repeated | `E0a` and `E14c` against the records of M5.46 | 775 and 975 iterations, and U at the end 4.1007 and 4.5414 to the printed digits, with the dense reads taken | a dense read counted by the EXIT rule, or a read that changes the descent: another count |
| K12, the maps of the family | each map at zero amplitude; the pinch with two lobes at 0.5 on a base | the same field to 10⁻¹⁴; no source point outside the box; one set whose plaquettes span radii in the ratio 0.5, within 0.1 (0.52 and 0.48 on the two bases of half-box 10, evaluated at plan time) | a map applied about another axis than the loop's |

**The made-up records of K10.** The ladder of a record has six outward rungs at D = 8.5, 10, 11.5, 13, 14.5 and 16, the tolerance of spacing 0.25, every rung CONVERGED, not EVADED, with one clear set and inside the room rule, and U built from the step identity, except where the record says otherwise. The base ladder A has f = 0.40, 0.42, 0.44, 0.45, 0.46, 0.46.

| # | Record | Word it must return |
| --- | --- | --- |
| 1 | R1: ladder A | HELD, the run ending at 16, the top of the ladder |
| 2 | R1: f = 0.40, 0.30, 0.20, 0.10, 0.04, 0.02 | NOT HELD (LEVEL): the mean of the last two is 0.075 of the largest |
| 3 | R1: f = 0.40, 0.36, 0.30, 0.24, 0.18, 0.14 | no label: the mean of the last two is 0.40 of the largest |
| 4 | R1: ladder A with f = 0.005 at the fifth rung | NOT HELD (LEVEL): a rung at or under 0.02 of the largest |
| 5 | R1: ladder A with U at the fourth rung 0.3 under U at the third; the branch check ends CONVERGED with one clear set at the third rung's own U | NOT HELD (BRANCH) |
| 6 | R1: ladder A, the fifth rung not CONVERGED | HELD, the run ending at 13 on a rung that is not CONVERGED |
| 7 | R1: ladder A, the third to sixth rungs not CONVERGED | no label: two counted rungs |
| 8 | R1: record 7 with U at the second rung 0.1 under U at the first; the branch check ends CONVERGED with one clear set at the first rung's own U | NOT HELD (BRANCH): NOT HELD outranks no label |
| 9 | R1: ladder A, the fourth rung EVADED | HELD, the run ending at 11.5 on an EVADED rung |
| 10 | R1: ladder A, the sixth rung with less room than the room rule | HELD, the run ending at 14.5 by the room rule |
| 11 | R1: ladder A with U raised by 0.2 from the fourth rung on (the step from 11.5 to 13 fails, e = 0.23), and a bisection rung at 12.25 with f = 0.445 and U on the identity from 11.5 (the half step from 12.25 to 13 fails, e = 0.37) | HELD, the run ending at 11.5 on the identity, the step classed as two branches |
| 12 | R1: f = 0.40, 0.42, 0.44, 0.10, 0.10, 0.10, with the step from 11.5 to 13 equal to 0.4725 in U (e = 0.14), and a bisection rung at 12.25 with f = 0.36 whose two half steps are 0.30 and 0.1725 (both pass) | no label: the run goes through the bisected step to 16, and the mean of the last two is 0.23 of the largest |
| 13 | R1: ladder A with three failing steps, the first two passing on bisection | HELD, the run ending below the third failing step, which is not bisected |
| 14 | R1: ladder A, and a second hold whose U falls between its two larger disks | no label: the two kinds of hold disagree in sign |
| 15 | R1: ladder A, and a member of the family that ends with one clear set and U 0.05 under the top rung | HELD (LOWER STATE), read with that state in the rung's place, the first print beside it |
| 16 | R1 at spacing 0.5: ladder A, and a continuation rung that differs from its fresh partner by 2 percent of the rise | HELD, START-DEPENDENT |
| 17 | R1: ladder A with U at the first rung under U of the free state | no label: a rung under the free state |
| 18 | R2: every CONVERGED rung with one clear set, every surface read at 0 or 1 | HELD |
| 19 | R2: a CONVERGED rung above the counted run with two clear sets, 1 and 0 around them; its release CONVERGED with two clear sets | NOT HELD (PERSISTENT SPLIT) |
| 20 | R2: record 19 with a release that ends with one clear set | no label (MADE BY THE HOLD) |
| 21 | R2: record 19 with a release that is not CONVERGED | no label |
| 22 | R2: record 18 with a liftable surface that reads 2 at one read | NOT HELD |
| 23 | R2: records 20 and 22 together | NOT HELD |
| 24 | R2: record 18 with one outward rung chain of eight counted reads | no label |
| 25 | R3: six members CONVERGED, one clear set each | HELD |
| 26 | R3: one member with two clear sets, its release with two | NOT HELD (PERSISTENT SPLIT) |
| 27 | R3: one member with two clear sets, its release with one | no label (MADE BY THE HOLD) |
| 28 | R3: one member not CONVERGED, the others with one clear set | no label |
| 29 | R3: records 26 and 28 together | NOT HELD |
| 30 | R3: a grid with no counted rung | no label |
| 31 | R4, a chain that ends by EXIT through one face, the reads of a surface: 1, at the loop, 0 | HELD |
| 32 | R4: 1, then 0, with no read between | HELD |
| 33 | R4: 1, at the loop by a fragment alone, 0 | HELD |
| 34 | R4: 1, blind, 0 | NOT HELD |
| 35 | R4: 1, at the loop, 1, at the loop, 0 | NOT HELD |
| 36 | R4: 1, 0, 1 | NOT HELD |
| 37 | R4: a liftable surface that reads 2 | NOT HELD |
| 38 | R4: the sets at the last read with a set touch two faces | no label: cut by the box |
| 39 | R4: the sets at the last read with a set touch no face | no label: vanished inside |
| 40 | R4: a chain cut by a cap | no label |
| 41 | R5: R1 and R2 with the same words at the two spacings of a box, R3 the same on the members both grids ran | HELD |
| 42 | R5: R1 HELD at one spacing and NOT HELD at the other | NOT HELD, the reading named |
| 43 | R5: one grid with no label in R1 | no label |
| 44 | R1: ladder A with the fresh state of the third rung 0.9 above its U on the identity, so that U falls by 0.2325 from the third rung to the fourth; the branch check ends CONVERGED with one clear set at the U and f of ladder A at 11.5 | HELD (LOWER STATE at 11.5), the ladder read again as ladder A, the run ending at 16 |
| 45 | R1: record 44 with a branch check that is not CONVERGED | no label: the step from 11.5 to 13 is not decided, printed |

The try cap is three. Between tries the input of a control or a defect of a script may change. Of the instrument, only the pair of strengths (to 10 and 40, or 40 and 160) and the power of u in the weight (2 to 3) may change, and only after a failure of one of K4 to K9 or of the one-third rule below, each change logged with its reason; never a threshold or a definition of § 4, and never the list of targets. A ladder on which more than a third of the rungs are EVADED is an instrument failure and ends the try. The tables and the words of every try are kept and printed in the findings, the changed instrument named beside them. What a failure costs: K1 to K3, no ladder; K4 to K7, no rung; K8, no R1 on that grid; K9, R1 printed with the words "one kind of hold"; K10, no label; K11, no R4; K12, no R3.

### 4. The readings, frozen

**The quantities.** Per grid and per rung: the target d, the D reached, the reader's separation, U and its three terms, the force f, the hold energy, the rise ΔU = U − U of the free state (the rung of K6 on that grid), the room (the half-box less the loop's radius), the sets and the surface reads at every read of the chain. The free state is the origin of the ladder and not a rung of it: its f is zero by definition.

**Precedence.** In every reading NOT HELD outranks no label, and no label outranks HELD.

**The counted run of a ladder.** An outward rung is eligible when its chain is CONVERGED, it is not EVADED, its end state has one clear set, and its room is at least the room rule below. The counted run starts at the first outward target and is the unbroken run of eligible rungs from there. It ends before the first rung that is not eligible, or at a step between two eligible rungs whose error e is above 10 percent (5 percent at spacing 0.5) and which does not pass its bisection. The step from the free state to the first outward target is not tested by the identity: the trapezoid is not accurate across the foot of the curve, and that end is covered by K6 and K7. A rung above the counted run enters no label of R1. Every label of R1 names the separation at which its counted run ends and the rule that ended it: the top of the ladder, a rung that is not CONVERGED, an EVADED rung, two sets, the room rule, or the identity.

**The bisection.** A step that fails the identity is bisected once: a rung at the middle target, from a fresh start. If the middle rung is eligible and both half steps pass, the run continues through it and the middle rung is counted. If not, the run ends at the lower rung of the step, and the failing half step is printed with its class: a fall of the force inside the step when f at its upper rung is under 0.8 of f at its lower rung, two branches otherwise. Two bisections per ladder at most; a third failing step ends the run without one. The bisection is a rule of this plan and not a change of its targets.

**The check of a branch.** Before the word NOT HELD (BRANCH) is given for a step on which U falls between two eligible rungs, the end state of the upper rung is continued down to the lower target, under the hold of the lower rung, in one chain with the same descent rule and gate. A fresh start can land on a state that is not the lowest at its target, and a fall of U from such a state is start dependence and not a fall of the cost. If the continued chain is CONVERGED, ends with one clear set, and has a U lower than the lower rung's by more than 10⁻⁶ (relative, the margin of LOWER STATE), the lower rung takes that state, the change is printed as LOWER STATE, and the step and the counted run are read again. BRANCH is given only when U still falls after that: the continued chain CONVERGED with no lower state at the lower target. A continued chain that is not CONVERGED leaves the step at no label, printed. A continued chain that ends in two clear sets is released by the rule of § 1, and the lower rung stands.

**The room rule.** A shared target is one present on both boxes at one spacing with both rungs CONVERGED, not EVADED and with one clear set. On a shared target the rung of half-box 10 agrees when its rise is within 5 percent of the rise on half-box 14. The room rule is the smallest room of half-box 10 such that this rung and every shared rung with more room agree, and it is never under 2, the room that the maps of the family need. A rung of either box with less room is printed and not eligible. If no shared target agrees, R1 carries no label on either box at that spacing.

| Reading | HELD | NOT HELD | no label |
| --- | --- | --- | --- |
| R1, the cost of the separation, per grid | the counted run has three rungs or more, U rises from each to the next, f is above 0.02 of its largest value at every rung, and the mean of f over the last two rungs is at least half the largest f of the run | BRANCH: U falls between two rungs at neighboring targets that are both eligible, anywhere on the ladder and whatever the identity says, and still falls after the check of a branch above. LEVEL: a counted rung with f at or under 0.02 of the largest; or the mean of f over the last two counted rungs under one tenth of the largest | fewer than three counted rungs; the mean over the last two between one tenth and one half (a slope that falls: a rise without bound and a leveling are not told apart inside the box); a counted rung with U at or under the free state's; a step on which U falls and whose branch check is not CONVERGED; the second hold giving a fall of U between its two larger disks where the first hold gives a rise |
| R2, one line, per grid | every CONVERGED outward rung of the ladder, counted or not, ends with one clear set; at every read of every rung chain no liftable surface reads anything but 0 or 1, and each liftable surface around the loop reads 1 (which the held faces fix for a state with one set, limit (2)) | PERSISTENT SPLIT: a CONVERGED rung that ends with two clear sets or more and whose release keeps them, the whole number around each set printed; or any liftable surface outside 0 and 1, or one around the loop that does not read 1 | a rung in two sets that is MADE BY THE HOLD, or whose release is not CONVERGED; fewer than three counted rungs; an outward rung chain with fewer than ten counted reads (the rung of K6 is not such a chain) |
| R3, the family that seeks a split, per grid | every chain of the family is CONVERGED and ends with one clear set | PERSISTENT SPLIT: a CONVERGED chain of the family that ends with two clear sets or more and whose release keeps them, whatever its U | a member in two sets that is MADE BY THE HOLD, or whose release is not CONVERGED; a member that is not CONVERGED; a grid with no top counted rung |
| R4, the exit, per exit chain that ends by EXIT through one face | for every nested surface that is liftable and around the loop at the first read: its reads are a run of 1, then reads at the loop, then a run of 0, and no read at the loop lies outside that stretch (parts (a) and (c)); no liftable surface outside 0 and 1 | a blind read between the last 1 and the first 0; a read at the loop before the last 1; a 1 after a 0; a liftable surface outside 0 and 1 | a chain that does not end by EXIT (a `Z` chain cut by a cap among them); a chain cut by the box (two faces or more touched, as P14a of M5.46); a chain whose line vanished inside (no face touched), printed |
| R5, the two spacings of a box | R1 and R2 carry the same word at the two spacings, and R3 the same word on the members that both grids ran | a word that differs, named | a grid with no label |

**Terms of the table, fixed here.** A clear set is a set of 8 plaquettes or more whose least distance to the faces is above one cell (M5.46 § 8). A surface is around the loop when it is a liftable cube wider than the largest coordinate of the loop's plaquettes. A read of a surface is at the loop when the surface is not liftable and either holds a pierced triangle or has a pierced plaquette of the line reader within 1.0 of it, every pierced plaquette counting, those of fragments under 8 included; a read that is not liftable for another reason is blind. An exit is through one face, cut by the box or vanished inside by the faces that the sets touch at the last read that has a set: one, two or more, or none. The top counted rung of a grid is the last rung of its counted run. The two thresholds of R1, one half and one tenth, are this plan's.

**Two prints of R1.** R1 is printed on the counted run, which carries the label, and a second time on all CONVERGED rungs with one clear set and no cut by the identity or the room rule, as a descriptor with no word: a cut moves which rungs are the last two, and the second print shows what the cut removed. The agreement of the two boxes on the shared targets is a descriptor as well, since over that range the room rule builds part of it in.

**The family of R3.** Each map is applied to the end state of the top counted rung, the faces set back to the frame, and the result is relaxed under the hold of that rung. With (ρ, φ, z) the cylinder coordinates about the loop's axis, R the loop's radius and s(φ) = 1 − ε cos²(mφ/2), the pinch is a radial map that is the identity beyond R + 2: the new field at (ρ, φ, z) is the old field at (σ, φ, z), the six components sampled by trilinear interpolation,

$$
\sigma = \frac{\rho}{s} \quad \text{for } \rho \le s R , \qquad \sigma = R + \frac{2 ( \rho - s R )}{R + 2 - s R} \quad \text{for } s R \lt \rho \le R + 2 , \qquad \sigma = \rho \quad \text{beyond}
$$

The line at azimuth φ moves from R to s(φ) R. No source point lies outside the box when the room is at least 2, which the room rule guarantees.

| Member | Map | Amplitudes |
| --- | --- | --- |
| The pinch with two lobes | m = 2: two opposite points of the loop are brought toward the axis, to a fraction 1 − ε of the radius | ε = 0.5, 0.8, 0.95 (at 0.95 the two cores overlap on the axis) |
| The pinch with three lobes | m = 3 | ε = 0.8 |
| Noise | six independent white Gaussian fields, one per component, smoothed by a Gaussian of width 1, multiplied by the window exp(−d²/4.5) with d the distance of a node to the loop's fitted circle, and scaled so that the rms of the added field over the nodes with d under 1.5 is the amplitude. The window, in place of a cut, leaves the far field and the faces untouched without a step in the field | 0.05 (seed 5001) and 0.2 (seed 5002), in units of the largest eigenvalue of the frame, which is 1 |

On 113³ the family is the pinch with two lobes at 0.8 and 0.95 and the noise at 0.2. A CONVERGED chain of the family that ends with one clear set and a U lower than the rung's by more than 10⁻⁶ (relative) is printed as LOWER STATE: the label of R1 is then the one read with that state in the rung's place, and the first print stands beside it.

**The exit, from stored records and from new chains.** (i) A desk read, with no compute: the per-read records of the four added chains of M5.46 hold, for every nested surface at every read, its class (liftable with its value, pierced with the count, or under the gap) and the circle fitted to the loop, at a cadence of 25 iterations. They do not hold the plaquettes, so "at the loop" for a read under the gap is taken from the fitted circle. The table of R4 is filled for them and marked as a re-read: M5.46 § 12 item 3 already states the outcome on the box surface. (ii) `E0a`, `E0b`, `E14c` and `E14d`: the chains X0a, X0b, X14c and X14d again, with the dense reads through the exit and the plaquettes in hand. M5.46 repeated X0a and X14c exactly (its § 12 item 5); X0b and X14d are run a second time here for the first time. (iii) `Z10a` to `Z14b`: the top rung released, which reads the clause from the pull to the exit on one chain. R4 is given per chain of (ii) and (iii).

**What each outcome would and would not support.** The run reports; the cell is re-read by the repo owner on the bar of `MODELS.md`, in the pull request that lands the record.

| Outcome | It would support | It would not support |
| --- | --- | --- |
| R1 to R5 HELD | the second limit of the cell is answered inside the box: with the separation held and raised to the top of the counted run, the cost keeps rising, the line is one piece at every rung, and the family left no second piece at rest | "impossible": a finite ladder, one family, two boxes; a statement about the whole number around the line on the ladder, which follows from the held faces; and the first limit of the cell, which no read on a liftable surface can remove |
| R1 NOT HELD (LEVEL) | a separation inside the box beyond which pulling the two lines apart costs nothing more: the clause loses its energetic support | a statement about the number of pieces, which R2 reads |
| R1 NOT HELD (BRANCH) | a held state at a larger separation that costs less than every state found at the smaller one, the state continued down from it included | that no lower state exists at the smaller separation: one continuation is one more start |
| LOWER STATE by a branch check | that the fresh start of that rung did not reach the lowest state found at its target: start dependence, printed with both states | a fall of the cost: the ladder is read again with the lower state |
| R2 or R3 NOT HELD (PERSISTENT SPLIT) | a state at rest, with no hold, in which the line is in two pieces, with the whole number each piece carries | a charge of ½: the pieces carry whole numbers that add to 1 |
| MADE BY THE HOLD | that this hold can be met by two pieces at that target, with what it cost | a split of the unit: the pieces did not stay apart without the hold. A neutral loop shed by the unit shrinks under its own tension once released, so this run can find a PERSISTENT SPLIT only where both pieces stay at rest |
| R4 NOT HELD | a read between the last 1 and the first 0 that the loop's crossing does not explain | anything about the ladder |
| no label | the rule that withheld it, with the numbers | either side |

**Registered before the run.** Expectations, which no script uses. R1: f between 0.1 and 0.8 at every outward rung, the mean of the last two at least 0.8 of the largest, HELD on every grid whose top rungs converge. The basis is one number with another boundary: P14c of M5.46, with free faces, twenty iterations after a dilation and not at rest, stood 4.89 above its own end state, between separations 6.8 and 13.5, a mean slope of about 0.7; the ladder holds the faces. The two boxes agree on all shared targets, so the room rule is at most 3.6. No step needs a bisection. R2 HELD, with no rung in two sets. R3: the pinches at 0.5 and 0.8 and the noise return to one loop; the pinch at 0.95 is not predicted, and a split there is expected to be MADE BY THE HOLD. R4 HELD on the chains that exit through one face; whether `Z14a` and `Z14b` contract before they leave, or reach several faces, is not predicted. Not predicted: whether every rung on 113³ converges inside its cap; a rung that does not shortens the counted run and is never read as a failure of the unit.

Nothing in this section sets an icon.

### 5. Blindspot pass, consumers, estimate

Read against the recurring failure families of the records of M5.46 to [M5.48](m5_48_task_details.md), for the instruments of this task.

| Unknown | Guard |
| --- | --- |
| A check that cannot fail: U rises with the size of a closed line by geometry; a whole number on a liftable surface holds by construction; with held faces a cube around a one-set state reads the faces | limits (1) and (2); R1 reads the fall of the force and not the sign alone; R2 reads the number of sets; parts (a) and (c) are read where the faces are free; the table of outcomes names what HELD does not support |
| The hold carries the answer | the hold energy apart from U; the abscissa from the reader; K6 returns the free state; K8, two strengths on one curve; K9, a hold of another kind; EVADED |
| A hold that is met without holding the line: by a loop that moves aside, or by a second piece far away, which the form of D rewards (§ 1) | the moving center and the squared density against the first; against the second, every state in two pieces is released, and only a split that persists with no hold gives NOT HELD; a split that does not persist is printed as MADE BY THE HOLD with its cost and gives no label |
| An optimizer stall read as a state, a gate that tests nothing | the gate at most one tenth of the force it balances; the restart rule; STALLED is never CONVERGED; the step identity between rungs, which a stalled rung breaks |
| A number set by a cap, and a cut that moves a word | a chain that is not CONVERGED enters no reading. A cut of the counted run (a cap, the room rule, the identity, an EVADED rung) makes no word by itself, but it moves which rungs are the last two; so R1 is printed a second time with no cut, every label names where its run ends and why, a failing step is bisected before it cuts, and a fall of U between two eligible rungs that survives the check of a branch is NOT HELD whether or not the run was cut there |
| One start taken for the state | fresh starts carry the label; a fall of U between two rungs is checked by a continuation from the upper one before it gets a word; the continuation chains up and down at spacing 0.5 are tabled beside them, and a rung that differs from its fresh partner by more than 1 percent of the rise appends START-DEPENDENT to R1 |
| The box read as the unit | two boxes; the room rule with its eligibility; limit (5); the rungs with less room printed and not counted |
| A dilated start is not a state of the action (its core is widened) | it is a start only; no read of a start enters a label except the first read of R4's surfaces |
| The reader outside its regime: a pinched loop whose cores overlap, sets under 8 plaquettes | fragments are reported at every read; the labels of R1 to R3 are read on states at rest; in R4 a fragment's plaquettes count for "at the loop", so a late piece of the line is not read as a blind surface |
| A read of the exit that depends on where a read falls (M5.46 § 12 item 4), and dense reads that change when a chain stops | dense reads through the exit, which no stop rule counts; part (c) classes each read and does not count steps; K11 |
| The instrument tuned with outcomes in view | a change of the strengths or of the weight only after a failed control or the one-third rule; every try kept and printed |
| This plan's reading taken for the author's | the open points are this plan's and are named as such in the findings: the held quantity and its two strengths, the targets, the two thresholds of R1, the bisection, the check of a branch, the room rule, the family, the release and its two words, "at the loop" within 1.0, a split read as two pieces at rest with no hold |
| A file of M5.46 overwritten | the driver writes under `data/` only `m5_50_*`; the end files of M5.46 are opened read-only and their hashes compared before and after |
| An unvalidated instrument (the family of the workflow) | K1 to K12, each with the input that fails it |

**Consumers.** R1 to R5 are read by the re-read of the charge-quantization cell and by the results message of the repo owner: they need K1 to K12 met and the audit. The base states of half-box 14 and the held states are reusable by a later run: they need the word CONVERGED and K2.

**Estimate.** The clock of the run is set by the longest dependent path of § 2 on the 113³ grid; everything at spacing 0.5 is short beside it and reports first. The build of the hold and the controls overlaps the two base chains, which need no new code. The reads, the findings and two audits follow the pool. The estimate in hours is given at staging and is not part of this document. Model: the session's model for the plan and the build, in one sub-agent; the audit on Opus, a numbers check on Sonnet.

**Not planned.** Two straight +½ lines held apart in place of the loop; a continuation ladder at spacing 0.25; a hold at a separation beyond 20.5 or a box beyond half-box 14; a start that is already in two pieces; a steepest-descent version of any chain; the second hold at spacing 0.25.

**Plan audit (2026-11-20, before any rung was relaxed).** An independent pass on a different model read the draft of this plan against the action code, the instrument and the records of M5.46, the thread record and the cell, and ran its own checks on stored states and on model curves. It found eighteen faults, two critical, five high, eight medium and three low, all folded above. The two critical ones were controls that a correct build could not meet: K7 put its four targets on both sides of the free state, where the middle step crosses the minimum of U and the trapezoid fails on any curve (the targets are now on one side, and the error of the identity is defined against the step in U); and K11 asked for the iteration counts of M5.46 while reading at every iteration, where the EXIT rule of M5.46 counts reads (the dense reads are now extra reads that no stop rule counts). The faults that changed a reading: R2 could not be NOT HELD by a split, a rung in two sets never being counted (R2 now reads every CONVERGED rung, and NOT HELD outranks no label in every reading); a failed step of the identity turned a jump to a cheaper branch, or a fall of the force inside a step, into a shorter run read as HELD (R1 gained the word BRANCH, a failing step is bisected once, R1 is printed a second time with no cut, and every label names where its run ends and why); the step from the free state to the first target could fail the identity on a correct curve and leave no counted rung (it is no longer tested there); with held faces part (a) of the clause cannot fail for a state with one set (limit (2) says so, and the ladder's read is the number of sets); and the hold can be met by two pieces at a cost that does not grow with the target, so that a split could be made by the hold and read as NOT HELD (every state in two pieces is now released, and only a split that persists with no hold gives the word). The faults that changed a control, a rule or the job list: the map of the family took its source from outside the box at about half of the nodes (it is now radial and the identity beyond R + 2, checked on stored states); the released top rung was read at every iteration from its first read, and a late fragment made a blind read (the dense reads start within 2 of a face, and a fragment's plaquettes count); two tolerances failed correct code (10⁻⁸ for the shift of K4, 10⁻¹⁴ for the maps at zero amplitude); K10 listed no expected word (its records now do); the dependencies of the top counted rung were misstated (the family, the releases and the `Z` chains wait for the counted run of both boxes at their spacing); the room rule had no eligibility and extrapolates in the size of the loop (eligibility stated, limit (5) added, the agreement of the two boxes made a descriptor); the instrument could be changed between tries with outcomes in view (only after a failed control or the one-third rule, every try printed); the first bound of K9 was under the start value; two claims about M5.46 overreached (only two of the four exits were repeated there, and the rise of 4.1 compared a chain with free faces to a base with held faces); and a list of terms was left open (the denominator of the identity, the free state in R1, the reads rule of R2, R3 with no counted rung, an exit that touches no face, the tolerance of K8, the noise, LOWER STATE, the target of K5, the chains of K6 and K7 and the caps of the `E` chains in the job list, the members compared by R5). Two things were added at the fold and not from the audit. A counted rung with U at or under the free state's gives no label for R1. And, on the drafter's objection to the word BRANCH as the fold first had it, a step on which U falls is checked by continuing the upper rung's state down to the lower target before the word is given, since a fresh start that lands above the lowest state at the lower rung would otherwise read as a fall of U. The auditor found sound: D and the reader on the base states, the share of the weight near the loop, M_zz inside the loop, the dilation clauses of K4, the gradient of D against central differences, the gate, the margins of EVADED on the bases, the room at the top targets, the counts of targets, the algebra of the force of the hold, the numbers quoted from M5.46, the author's definition against the thread record, and the two limits of the cell against `MODELS.md` and M5.46 § 15. It did not check, and could not at plan time: any relaxation under the hold, so whether the rungs converge, the real curve U(D), whether the hold makes a split cheaper than one loop, and whether K6 and K9 are met; and it read the author's post through the thread record and not at its source.
