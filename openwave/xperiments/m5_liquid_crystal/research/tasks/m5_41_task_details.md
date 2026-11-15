# M5.41: The Card C background under the agreed rule, and the calibration at zero spacing

> Opened 2026-11-15 from the author's reply to the results post of [M5.40](m5_40_task_details.md) on discussion #186 ([discussioncomment-19617697](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19617697), 14 November 2026, 18:39 UTC), which answers the three questions of that post and states the rule for the next pass, and from the announcement of the capstone v3.9 ([discussioncomment-19626175](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19626175), 15 November 2026, 06:26 UTC; Zenodo record [24240337](https://doi.org/10.5281/zenodo.24240337)). Roadmap row: [`m5_roadmap.md`](../m5_roadmap.md). Thread record: [`m5_32_convo.md`](m5_32_convo.md). Sources: [`_CITATIONS.md`](../../theory/_CITATIONS.md) (the v3.9 rows). This is the task that [M5.40](m5_40_task_details.md) § 8 and § 15 name as the following one.

## TASK PLANNING (2026-11-15; the plan post went on the thread at 14:22 UTC, [discussioncomment-19633185](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19633185); the go is the user's, after the author's reply)

**Scope.** (1) The release v3.9 filed and the reply's numbers checked (§ 1, done at plan time). (2) R1: the instruments, namely the driver of M5.39 extended by a list of saved stops, a stop rule on the unit-sphere residual and an evaluation limit above the iteration cap, and the calibration relations coded against `calibration33.json` (§ 3). (3) R2: twelve routes on the mesh ladder at half-box 6.075 (three starts on each of 55³, 65³ and 77³, two on 45³, and the author's own field as a fourth route on 55³), each run to 30000 iterations with four saved stops (§ 4). (4) R3: the readouts of [M5.40](m5_40_task_details.md) § 13 at every stop, and the within-route criterion (§ 5). (5) R4: the spread between routes on each mesh, the zero-spacing line through the three finest meshes, and the calibration by the frozen rule with the route spread as its error (§ 5). Out of scope: the box ladder (the zero-spacing values of this task are values at half-box 6.075, and the box rise stays the number of [M5.39](m5_39_task_details.md)); the 33³ grid of the calibration file, whose half-box 3.6 is not on this ladder; the optional rung of the announcement (the energy per length of the three vortex types and the merge barrier of a split pair); the tilt reduction and the charged core ([M5.39](m5_39_task_details.md) § 2, rows 12 and 13); any icon or pointer of [`MODELS.md`](../../../../../MODELS.md), since every run is on the author's Card C and not on this column's certified action.

**Definition of done.** The instrument controls of § 3 each reported as met or as the exact failing check; the twelve routes run to their cap or to a logged end, every stop saved; every route labeled by the rule of § 2 (MEETS THE RULE, or the clause it fails); the spread, the zero-spacing line and the calibration reported under the wording rule of § 2 row 8; every number of the findings carried by a tracked JSON and a script; an audit by agents on a different model with every claim either standing or qualified (§ 6); the task review presented in the terminal. A control that fails after its three tries is reported as failed, and the reads that depend on it are kept as numbers without comparison words.

**Gating.** The author's rule (arrived, § 1). The author's reply to the plan post: the readings of § 2 go to the author before any route runs, and a correction is folded into § 2, with its date, before the go (the maintainer's decision of 2026-11-15). The user's go.

**Model and effort.** The session's model for the run; the audits on a different, smaller model (two agents in parallel), named at each spawn.

**Research body.** This document (`## FINDINGS` after the run); scripts `../scripts/m5_41_*.py` and the extended [`m5_39_relax.py`](../scripts/m5_39_relax.py); data `../data/m5_41_*.json` (tracked) and `../data/m5_41_*.npz` (local-only, indexed by [`_DATASETS.md`](../data/_DATASETS.md)); plots `../plots/m5_41_*.png`, embedded in the findings; in-flight state in `../checkpoints/m5_41/` (local-only).

**Consumers.**

| Sub-result | Consumer | The check that consumer needs |
| --- | --- | --- |
| The label of each route | the author's next pass, which counts only states that meet the rule | the stop rule and the gate reading fixed before the run (§ 2); the regression of the extended driver (§ 3) |
| The spread between routes | the fixed-mesh error of the calibration | starts of more than one seed family (§ 4); the center spectrum and the branch label of each route, since routes on different branches are not combined |
| The zero-spacing values | the calibration (e, the effective f_π, the action energy unit, g_A) | the readout code gated in [M5.40](m5_40_task_details.md) § 13; the fit-form spread stated beside the route error |
| The calibration numbers | the capstone's hadronic section and its pre-registered amplitudes | the relations reproduced on `calibration33.json` before they are applied (§ 3) |

### 1. The reply, the release and what was checked at plan time

| Item | Fact |
| --- | --- |
| The rule, as the reply states it | The readouts' own stability is accepted as the criterion, instead of a tighter bound on the energy, since the energy left open by the gate does not move the readouts. Within a route: the inertia, g_A e² and μ_V fixed to 10⁻⁴ relative between two stops at least a factor 10 apart in iterations, plus the unit-normalized residual gate and the lump check. Across routes at a fixed mesh: the spread of at least two independent routes is the fixed-mesh error; the reply names the 55³ pair of [M5.40](m5_40_task_details.md) § 12 item 5 (the `smooth` state of M5.39 against the continued field of the author), which differs by 0.004 in the center eigenvalues and by 5.7 × 10⁻⁴ in μ_V, more than the within-route criterion. The calibration: at zero spacing from the three finest rungs, with the route spread as its error. The reply adds that this shift (3 to 4 percent) dominates everything above, and that g_A at each grid's own coupling stays near 0.91 because the mesh and box corrections nearly cancel |
| μ_V | The reading of the scripts in [M5.40](m5_40_task_details.md) § 13 is the intended one: (4/3) m_p \|m_z\| / (e³ f_π) with m_z the mirror-averaged response of the gauged lattice energy. The sentence on the inertia quadrupole belonged to the transition quadrupole and is corrected in v3.9 |
| The release | v3.9 (406 pages; a PDF and a source archive of 1318 files), filed from the version record 24240337 of the concept DOI that the announcement gives (23994958); both files carry the md5 of that record's file list. Against v3.7: 1237 files in common, 5 of them changed (`README.md`, `sm_from_ldgs.tex`, `contribution_map.tex`, `checks/self_metric_cones.py` and its `.json`), 81 new; the 49 files whose paths name the Card C code, the frozen packets or the pre-registration are identical by hash |
| `calibration33.json` | In `frozen_packets/CardC_payload_manifest/` of the source; its sha256 equals the digest the reply posts. The three numbers that `static_preflight.py` reads: `effective_e` 4.127737520949882, `effective_f_pi_MeV` 95.33832953347283, `frequency_unit_MeV` 393.5316000000001 |
| The file's numbers obey closed relations | With L the file's `length_unit_fm` and I its inertia of the 33³ field (34.32633181038963): e f_π = ħc / L (393.5316000000001, the frequency unit); e³ f_π = the file's `inverse_inertia_unit_MeV` (6705.076813629442 against 6705.076813629443 computed); (3/2) e³ f_π / I = 293.0, the calibrated bare gap; f_π / e = the file's `action_energy_unit_MeV`. So e² = 2 I Δ L / (3 ħc), which is the frozen rule as coded in M5.40 (`frozen_rule_e`): on the author's 33³ field that code returns the file's `effective_e` in every printed digit |
| The author's axial script with the file | `static_preflight.py`, unmodified, in an isolated copy outside the repository, beside a file holding the three numbers of the reply: g_A 0.957190 (33³) and 0.985809 (55³), the independent values of [M5.40](m5_40_task_details.md) § 13 (0.95719 and 0.98581). Its tail couplings, which [M5.40](m5_40_task_details.md) § 15 could not compute without the file, come out as 9.51 to 9.66 (33³) and 9.74 to 9.93 (55³) over the four shells of its Dirichlet fit; they are numbers of this file's normalization, they are not compared with a printed value, and R1 repeats the run with the author's file itself and stores its output |
| The capstone's paragraph on M5.40 | Read against the record of M5.40: the digests, the two energies, the readouts, the two corrections, the continuation and the mesh trend are as reported. One digit differs: the paragraph gives the agreement of the deposited scripts as 2 × 10⁻¹¹, which is the figure of `zero_q2.py`; for `static_preflight.py` the record has 1.4 × 10⁻¹⁰ |
| `checks/self_metric_cones.py` | Rewritten in v3.9 with a request to rerun wherever the old script or its json was reused. No document of this column uses the script or its output, so nothing reruns; the thread record and the citations name it only as the announced rewrite |
| The states of M5.39 on the ladder | All residuals of this row are in the reading of M5.39, on the winding vector as stored. On each of 45³, 55³ and 65³ the `iso` and `chain` routes ended under the gate without a record lump (2153 to 6011 iterations); on 77³ neither did within 8000 iterations (7.8 × 10⁻³ and 2.3 × 10⁻³). The `smooth` start ended under the gate on 55³ (the lowest state of that mesh, 75.56175, no lump) and on 77³ (one lump at a corner node), stayed 0.077 above the other routes on 65³ after 8000 iterations, and ended on the split-center branch on 45³. The `iso` start is a closed form, `smooth` is the archived seed, and the `chain` starts are local-only arrays of M5.39's checkpoint folder, rebuilt by the pool of M5.39 ([M5.39](m5_39_task_details.md) § 17) |

### 2. The rule as it is frozen before the run

Each clause of the reply leaves a choice open. The readings below are fixed here, before any route runs, and go to the author with the plan.

| # | Clause | The reading frozen here |
| --- | --- | --- |
| 1 | "the unit-normalized residual gate" | The larger of two numbers, each the largest entry over the interior nodes divided by h³: the winding gradient projected on the unit sphere, and the factor gradient. At most 10⁻³. This is the projected reading of [M5.40](m5_40_task_details.md) § 10 (K3), under which the author's recorded residuals return; on fields stored at unit length it equals the stored reading. It is evaluated at every iteration |
| 2 | The stops | Four targets, counted in the route's own optimizer iterations from its start: 1000, 3000, 10000, 30000. The residual moves up and down along one optimizer path ([M5.40](m5_40_task_details.md) § 12: under the gate on 55 of 200 saved rows at 33³ and on 1 of 60 at 55³), so the state saved for a target is the first iterate at or after it that is under the gate. A target with no such iterate before the next target (before 33000 for the last) keeps the iterate at the target count (at 33000 for the last) and is labeled GATE NOT MET. The optimizer is never restarted at a stop, and the route ends when its last stop is saved |
| 3 | "two stops at least a factor 10 apart in iterations" | The pair that is read is the last stop and the latest earlier stop whose iteration count is at most a tenth of the last stop's. The counts are those of the iterates actually saved |
| 4 | "fixed to 10⁻⁴ (relative)" | For each of the inertia (the mean of the three diagonal entries), g_A e² and μ_V: FIXED when the relative difference from the last stop is at most 10⁻⁴ both for the stop of row 3 and for the stop of the 10000 target; otherwise NOT FIXED, with both drifts printed. The two radii, the center eigenvalues and the energy are given beside them at all four stops and carry no label |
| 5 | "the lump check" | `record_lumps` of [`m5_39_reads.py`](../scripts/m5_39_reads.py) unchanged: no node whose largest record eigenvalue stands more than 0.2 above the card's target scale. The known values of the largest excess: 0.012 on the author's fields, at most 0.16 on the lump-free states of M5.39, 0.96 and 0.98 on its two states with a lump |
| 6 | A route that meets the rule | MEETS THE RULE on a mesh when its last stop is under the gate, holds no lump, and its three readouts are FIXED. Any other route is named by the clause it fails and its numbers are kept. The label says that the rule's three checks pass; it does not say that the energy has stopped falling, and the energy's drift over the last tenth of the route is printed beside it |
| 7 | "at least two independent routes", "the spread" | Three starts: `iso` (the closed-form isotropic-center seed), `chain` (the refinement chain of M5.39) and `smooth` (the archived smooth seed). `iso` and `chain` are two starts of one seed family, since the chain's first stage and the `iso` seed are both built from the `linked` seed; `smooth` is a second family, and on 55³ the author's field is a third origin. The spread on a mesh is the half range, (max − min)/2 over the routes' last stops, relative to their mean, given over all routes, over the routes that meet the rule, and with one family left out. It is a spread inside the declared branch: the center spectrum of every route is printed, and routes that end on different branches by the label of M5.39 are not combined |
| 8 | "at zero spacing from the three finest rungs, with the route spread as its error" | A straight line in h² through 55³, 65³ and 77³ for each readout. The central value is the least-squares intercept on the means of the routes that meet the rule. The route error is reported as a range, not as a standard error: the half range of the intercepts over every combination of one such route per mesh (36 when all ten routes of these meshes meet the rule). The intercept weighs the three meshes by −1.23, 0.49 and 1.73, so a spread on 77³ enters 1.7 times. Stated beside it and not folded in: the intercepts of the two finest meshes alone and of all four meshes (the fit form), the intercepts from the 10000 stops (the drift), and the intercepts of each start family alone. The word calibration is used only when each of the three meshes has two routes that meet the rule; otherwise the same numbers are reported as a trend, with the failing clause named |

The calibration itself is the rule of the capstone's recalibration, with the inherited length unit L and the bare gap Δ held fixed:

```text
e0^2      = 2 I0 Delta L / (3 hbar c)        Delta = 293 MeV, hbar c / L = 393.5316 MeV
f_pi      = hbar c / (L e0)                  the effective f_pi of this convention
unit_E    = f_pi / e0                        the action energy unit
g_A       = (g_A e^2)_0 / e0^2               also read as the intercept of g_A at each mesh's own e
mu_V      = 2 m_p |m_z| / (I Delta)          per mesh: the script's value times its own gap over Delta,
                                             the gap of a mesh being (3/2)(4.259^3 x 92.4) / I; then the intercept
rigid     = m_p / Delta
```

I0 and (g_A e²)0 are the zero-spacing intercepts. Each output is given with the route range carried through the combinations of row 8, not through a linearized formula.

### 3. R1: the instruments and their controls

| Instrument | Change | Control (three tries each) |
| --- | --- | --- |
| The driver [`m5_39_relax.py`](../scripts/m5_39_relax.py) | A list of stops saved by the rule of § 2 row 2 (the single hand-off of M5.39 generalized); the gate of § 2 row 1 evaluated at every iteration (the driver records the unit-sphere winding residual every 50 iterations only, and traces the stored reading between); the optimizer's own gradient stop switched off (`--stop-gate 10⁻⁹`, the setting of the timing pilot), so that only the cap, a stop of § 2 or the optimizer's function tolerance ends a route; the optimizer's limit on function evaluations raised above the cap, since the library's default of 15000 would end a route near iteration 14600 and M5.40's longest run was 10000 | Regression: with the arguments of [M5.40](m5_40_task_details.md) § 16 on one thread, the first 3000 iterations of the 33³ continuation return the energies of the stored rows of `m5_40_r3_g33_cont.json` to 10⁻¹² relative. Start control: the first evaluation of every route returns the start energy below to 10⁻¹⁰ relative |
| The readouts [`m5_40_readouts.py`](../scripts/m5_40_readouts.py) | None | Gated in [M5.40](m5_40_task_details.md) § 13; rerun on the author's 55³ field at the head of this task's read, the stored values returned to 10⁻¹⁰ |
| The author's `static_preflight.py` | None; run unmodified in an isolated copy with the author's `calibration33.json` itself | Its output on the two deposited fields stored in a tracked JSON; g_A against [M5.40](m5_40_task_details.md) § 13 to 10⁻⁸ |
| The calibration relations, `m5_41_calibration.py` | New | From the inertia of the author's 33³ field: the file's `effective_e`, `effective_f_pi_MeV`, `inverse_inertia_unit_MeV` and `action_energy_unit_MeV` to 10⁻¹² relative, and its `minimal_muV_muN` (3.198542995087794) from the stored μ_V and gap of M5.40 to 10⁻⁹ relative. From the author's 55³ field: the printed e = 4.311 and f_π = 91.29 MeV to the printed digits |

### 4. R2: the routes

| Mesh | Spacing h | Route | Start | Start energy (the control of § 3) |
| --- | --- | --- | --- | --- |
| 45³ | 0.27614 | `iso` | closed form | 951.1869569484667 |
| 45³ | 0.27614 | `chain` | the hand-off of the 37³ stage | 76.17486674698928 |
| 55³ | 0.22500 | `iso` | closed form | 960.2457216651314 |
| 55³ | 0.22500 | `chain` | the hand-off of the 45³ chain | 76.10589655857467 |
| 55³ | 0.22500 | `smooth` | the archived seed | 931.1781080672494 |
| 55³ | 0.22500 | `author` | the author's deposited field | 75.56196067295708 |
| 65³ | 0.18984 | `iso` | closed form | 966.5931374279587 |
| 65³ | 0.18984 | `chain` | the hand-off of the 55³ chain | 75.97938170352795 |
| 65³ | 0.18984 | `smooth` | the archived seed | 939.0364814034779 |
| 77³ | 0.15987 | `iso` | closed form | 972.1235440994741 |
| 77³ | 0.15987 | `chain` | the hand-off of the 65³ chain | 76.09773221330995 |
| 77³ | 0.15987 | `smooth` | the archived seed | 945.730922837993 |

Every route runs the card's parameters and the optimizer of M5.39 (the same memory, line search and admissibility wall), with the two settings of § 3 that keep it from ending early. What ends a route: its last stop (§ 2 row 2), or the optimizer's function tolerance (an energy change under 10⁻¹⁵ relative), in which case the final iterate is the last stop and is labeled by the same rule. An abnormal end (a failed line search) is restarted from the last iterate at most three times; a restart, like a resume from the checkpoint after an interruption, clears the optimizer's memory and returns to the last recorded row, so each is logged with its iteration and the route is marked as restarted in every table.

The 45³ mesh is not one of the three finest; it is run as the fourth point of the fit-form check, with the two starts that stayed on the declared branch there (`smooth` ended on the split-center branch on 45³ in M5.39). The `chain` routes of different meshes share ancestry along the chain while the other starts do not, which is one reason the intercepts of each family alone are printed (§ 2 row 8). The `linked` seed itself is not a route: it was above the gate on 55³, 65³ and 77³ after 8000 iterations in M5.39, and its family is represented by `iso` and `chain`.

### 5. R3 and R4: the reads, and what is recorded before the run

At each of the 48 saved stops: the energy and its seven terms, the residual in both readings, the range of the winding vector's length, the center eigenvalues and the branch label, the lump count, and the readouts of [M5.40](m5_40_task_details.md) § 13 (the inertia, the two radii, μ_V with its rigid value, g_A e² with its two parts). From them: the labels of § 2, the spread per mesh, the lines per readout, the calibration, and one figure each for the routes' descent, the readouts against the stop, and the zero-spacing lines with their route range.

Recorded before the run, from the numbers of M5.39 and M5.40. Energies are in the card's units, as absolute differences. Each line is an expectation, and its failure is reported with the same weight.

1. **Energy.** Every route ends below its end state of M5.39. On 55³ the gate stops of M5.39 spread over 1.2 × 10⁻³ (75.56175 to 75.56298), while the continuation of M5.40 ended 4 × 10⁻⁶ from the lowest of them; the expectation is that all four 55³ routes end within 2 × 10⁻⁴ of 75.5617. A route that ends more than 5 × 10⁻⁴ above the lowest of its mesh has not reached the floor the others found; whether it is slow or in another valley is read from its center spectrum and its lump count, and the `smooth` route on 65³ is the candidate (0.077 above after 8000 iterations in M5.39).
2. **The gate at the last stop.** Not assured on any mesh. Under the gate were 55 of 200 saved rows of the 33³ continuation of M5.40 and 1 of 60 of the 55³ one, so the 3000 iterations between the last target and the end may hold no gate-passing iterate, on 55³ as on 77³. A mesh on which fewer than two routes pass withholds the word calibration (§ 2 row 8).
3. **Within a route.** By extension from the continuations of M5.40 (33³ and 55³, at most 10000 iterations, where the inertia, the radii and g_A e² moved by under 5 × 10⁻⁵): the inertia and g_A e² come out FIXED on 45³, 55³ and 65³. μ_V is the readout at the edge of the criterion: it moved by 3 × 10⁻⁵ and 7 × 10⁻⁵ over those continuations and differs by 5 to 6 × 10⁻⁴ between two states of one mesh. No continuation has been run on 45³, 65³ or 77³.
4. **Across routes.** The half range on a mesh is under 10⁻⁴ for the inertia and g_A e² (the `iso` and `chain` states of M5.39 give 7.9 × 10⁻⁵ on 55³ and 4.1 × 10⁻⁵ on 65³ for the inertia), and near 3 × 10⁻⁴ for μ_V.
5. **Zero spacing.** The intercepts land within half a percent of the trend of [M5.40](m5_40_task_details.md) § 13: inertia 35.92, g_A e² 16.28, μ_V 3.148, hence e near 4.222, an effective f_π near 93.2 MeV and g_A near 0.913. That trend used one state per mesh and its 77³ point is the `smooth` state that holds a lump. The route range of the intercepts is at least an order under the distance between the 55³ values and the intercepts, with the 77³ spread entering 1.7 times.

### 6. Blindspot pass, the audit and the records

Read against the recurring failure families of the workflow, for this task's instruments.

| Risk | Where it enters | Guard |
| --- | --- | --- |
| A criterion fitted after the fact | the stops, the pair that is read, the stop rule at a fluctuating residual | all fixed in § 2 before any route runs; every stop's numbers are printed, not only the pair that is read |
| Unfalsifiable check | two points 27000 iterations apart can agree while the readout wanders between them | the 10000 stop must agree as well (§ 2 row 4); the gate and the lump check are separate conditions; the label does not claim a floor (§ 2 row 6) |
| Unvalidated instrument | the extended driver; the new calibration code | the regression and the start controls of § 3; the relations reproduced on the author's file first |
| Artifact read as result | a record lump at a corner node; a growing length of the winding vector | the lump check at every stop; the residual read on the unit sphere and the length range recorded per row |
| One valley taken for the minimum | routes that agree share a valley and prove no more than that; two of the three starts are one seed family | the spread is called a spread inside the declared branch; a second family and the author's field are in the set, and the spread is given with one family left out (§ 2 row 7); a route ending high is reported (§ 5 item 1) |
| Correlated samples counted as independent | the `chain` routes share ancestry across meshes; the combinations of § 2 row 8 reuse the same routes | the route error is stated as a range with the mesh weights, never as a standard error; the per-family intercepts printed |
| Resolution mistaken for accuracy | three points and a straight line in h² | the fit-form spread printed beside the route error and never folded into it; the box named as the correction this task does not carry |
| A path that depends on the thread count or on an interruption | threaded kernels change the floating-point order; a resume returns to the last recorded row with an empty optimizer memory | each route is one path on one thread count; the only cross-run equality that is asked is the single-thread regression; every resume and restart is logged and shown in the tables (§ 4) |
| A run cut by power or by the optimizer | 30000 to 33000 iterations per route | the driver's resumable checkpoint; the restart rule of § 4, capped and logged |

The audit: two agents on a different model, in parallel, each asked to refute. One takes R1 to R3 (the driver's diff, the regression, every label recomputed from the stop files by its own code); the other takes R4 (the fits and the combinations recomputed, the calibration relations against the author's file, every number of the findings against the JSON). The thread record gets its row when the results are posted; the post is the maintainer's, after the review.

### 7. The parallel shape, the estimate and what follows

| Stage | Jobs | Shape |
| --- | --- | --- |
| Wave 1 | the three 77³ routes | four threads each in the compiled kernels, twelve busy cores, the one-thread pins of M5.39 on every other library. Measured in a 60-iteration pilot from the end states of M5.39, with twelve cores busy: 1.25 s per iteration on four threads, so 10.4 hours for 30000 iterations and 11.5 for 33000 |
| Wave 2 | the three 65³ routes | four threads each. Not measured on four threads: 1.11 s per iteration on two in the same pilot, 2.2 s on one in the pool of M5.39; about 0.7 s expected, 6 to 7 hours |
| Wave 3 | the four 55³ routes and the two 45³ routes | two threads each. 55³: 1.39 s on one thread in the pool of M5.39 and 0.38 s on six in M5.40, about 0.75 s expected, 6 to 7 hours; 45³: 0.75 s on one thread, under 4 hours |
| The reads | 48 stop states | one process pool over the states after the routes; the nine states of M5.40 took 7 minutes on four threads, so about half an hour |

No job runs on one core: the kernels thread inside a job and the waves fill the cores across jobs. The GPU is not used: every control of § 3 is an equality at 10⁻⁸ or tighter in double precision, and the kernels are compiled for the CPU.

Go to review: about a day, 24 to 28 hours. The three waves take 22 to 26 hours (the lower figure when every last stop lands at its target, the upper when the routes run to 33000); the instrument changes and their controls, the reads, the two audits in parallel and the write-up add about two hours. The pilot is short (60 iterations) and two of the three thread layouts are extrapolated, so the first rows of each wave are read against these figures and the estimate is restated in the checkpoint file if they differ. The run is unattended and loads the machine for its whole length.

What follows this task, each as its own plan: a mesh ladder at a second box, which is what a value at zero spacing and infinite volume needs; the optional rung of the announcement; the tilt reduction and the charged core.
