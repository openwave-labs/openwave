# M8.18: THE CENSUS'S TWO REMEDIES, a delta on M8.17's terms

Pre-registration, for review. It governs nothing until the maintainer's go. **Sealed until the freeze:** no solve runs at `N = 40` on a control or a target, at T4/R5's moved midpoint, or at a guard reading. § Known outcomes names what is already on record at `N = 40`.

## TASK PLANNING

### The question

**What M8.17 settled.** The maintainer adjudicated M8.17 on [#676](https://github.com/openwave-labs/openwave/pull/676#pullrequestreview-5691493287): it stands as recorded, and its close-out merged on [#677](https://github.com/openwave-labs/openwave/pull/677). T2 and T3, in both slots, meet an S4 completion condition. T1 and T4 in both slots, and T5 in `R2`, are not settled by that run.

**What was admitted.** The answer of 2026-10-29 ([19278753](https://github.com/openwave-labs/openwave/discussions/512#discussioncomment-19278753)) admits two remedies, and the answer of 2026-10-30 ([19296471](https://github.com/openwave-labs/openwave/discussions/512#discussioncomment-19296471)) fixes the first one's tested band. This registration holds both, as two independent arms:
- **Arm A, twelve points at `N = 40`:** T1/R4 at 1.41, 32, 45.25, 64 and 72; T1/R5 at 32; T4/R4 at 4, 8, 11.31, 64 and 72; and T5/R2 at 45.25. M8.17's rule read each of them Unresolved on `N = 16`, 24 and 32, with K1 and branch identity holding. This arm reads them on the ladder `N = 24`, 32 and 40, and locates any change of type the new readings bracket.
- **Arm B, T4/R5's bracket (8, 11.31):** every grid point is classified, and `n(L)` goes from 7 to 6 at `N = 24` and 32, but K1 refused both deciding bisections at the midpoint 8.721638. This arm runs the locator again, with one change: a midpoint where K1 fails is moved once.

T5/R2's 128 and 132 are not here. Branch identity between `N = 24` and 32 failed there, that check stays, and they go to S6 as M8.17 left them.

**What this task asks:** classify the twelve points with the frozen instrument and rule on the new ladder, once the controls have shown that the ladder reads them correctly; locate T4/R5's change, and any change the twelve bracket; and record where each of the five rows stands afterwards, stating separately whether it meets an S4 completion condition.

**Old and new evidence.** The instrument is deterministic, and every reading at `N = 24` and 32 on the paths is on record, so this task claims no new evidence from them. The new evidence is at `N = 40`, at T4/R5's moved midpoint, at the guard readings, and at the midpoints of any bracket this run creates (§ Known outcomes).

### Standing

- **The rulings it files under,** all on #512:
  - the answer of 2026-10-15 ([18955679](https://github.com/openwave-labs/openwave/discussions/512#discussioncomment-18955679)), whose § 3 is the boundary below;
  - the ruling of 2026-10-20 ([19071299](https://github.com/openwave-labs/openwave/discussions/512#discussioncomment-19071299)) and the answers of 2026-10-20 ([19076863](https://github.com/openwave-labs/openwave/discussions/512#discussioncomment-19076863)), § 2;
  - the answers of 2026-10-29 and 2026-10-30, above.
- **The conditions, and where they are met:**

| Condition | Where |
| --- | --- |
| Its own ID, the next free one; M8.17's outcomes stand | M8.18; ID, below |
| Twelve points, fixed before the go, nothing added after it | § ARM A, The points |
| The ladder, each cutoff on its own path; `N = 16` not read; its missing channels DERIVED; the choice disclosed | § THE INSTRUMENT AND THE LADDERS |
| The instrument byte for byte; `N = 40` an input the pinned code accepts | § THE INSTRUMENT AND THE LADDERS |
| The kernel at `N = 40`, measured first, on M8.16.1's controls, against K1 as frozen | § ARM A, The kernel |
| Each control reads its recorded type on the ladder before any target is read | § ARM A, The control gate |
| Branch identity between 24 and 32 and between 32 and 40; the type from `N = 40`, with its indices | § ARM A, Classifying a point |
| M8.17's locator, its deciding cutoffs fixed before the go | § ARM A, Locating a change of type |
| The level, where it enters, where it applies, the reading beside it, the reason, what a classified point claims, the disclosure | § THE TESTED BAND |
| T4/R5's bracket only, `N = 24` and 32 deciding, `N = 16` run and reported | § ARM B |
| The one change, its diff, and the design stated | § ARM B |
| What a location claims | § ARM A, Locating a change of type; § ARM B |
| The calibration flagged | § STATED LIMITS |
| Known outcomes, of both remedies and of the replay | § Known outcomes |
| A failure: nothing repaired and rerun, not offered again; S6 per item, on the census page | § THE ROW OUTCOMES |
| Within about 8,000 words | Word budget, below |
| Maintainer reproduction for any claim that frozen text is defective | not reached: this task claims none, and the band's exception is the maintainer's reading of two frozen texts |
| Both M8 status rows record the filing | § DEFINITION OF DONE |
| The census page's dated notes, before any terms are drafted | done, at MIT `2a4bccb` and `7a2c638` (§ V, step 5) |

- **Word budget.** About 7,750 words, inside § 12.2's cap of about 8,000.
- **Run-before-write** (`PR_REVIEW_STANDARDS.md` § 12.2) is met: M8.17 ran and was adjudicated (#676), and its records after the verdict landed (#677).
- **ID.** M8.18, which no M8 file or open pull request carries. The BACKLOG row the filing creates allocates it (`ROADMAP_STANDARDS.md` § 6).
- **Inherited.** Every rule of M8.17's terms that this document does not change governs here: the filed text, SHA-256 `667159c3…`, the 27,823-byte prefix of [`m8_17_task_details.md`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m8_mit/research/tasks/m8_17_task_details.md), and through it M8.16.1's and M8.16's. Inherited unchanged, in particular:
  - the instrument and the separation statement, the seeds and their check, continuation, the solve and its acceptance bound, and the linear record;
  - the cases' definitions, seeds and expected columns;
  - K1, applied here at every point;
  - THE CONVERGENCE RULE, as built and frozen, apart from the band's stated exception;
  - M8.17's locator, with Arm B's one change and Arm A's rules for a stored or borrowed end, and its unsuccessful stops, conditions for a location, name and resolution;
  - THE RUN's interruption and resume rules, and its power and sleep conditions;
  - validity, and no search;
  - STATED LIMITS, with the additions below;
  - the guard, the freeze block and provenance, with new pins.
- **What this task changes:**
  - the points, the ladder and the band's level (§ THE INSTRUMENT AND THE LADDERS; § THE TESTED BAND; § ARM A);
  - the controls' gate, and a second branch-identity pair (§ ARM A);
  - solving and reading as separate operations at `N = 40`, and a solve-state gate where no record exists (§ ARM A; § THE FREEZE, THE GUARD AND THE RUN);
  - the locator's deciding cutoffs and a bracket's borrowed end in Arm A, and its midpoint in Arm B (§ ARM A; § ARM B);
  - how the rows stand afterwards, under the ruling's reading of S6 (§ THE ROW OUTCOMES);
  - what ends a part of the run, and a count of each job's attempts (§ TWO ARMS, ONE REGISTRATION; § THE FREEZE, THE GUARD AND THE RUN);
  - no qualification verdict: this task qualifies nothing.

  Its code is new, in `run/census.py` and `run/census_lib.py`, in a folder of its own beside M8.17's, which it leaves unchanged. Their differences from M8.17's files are filed with the terms as a diff.

### The boundary (#512, § 3, verbatim)

> Admitted on #512: a harmonic Galerkin instrument on the slot action's nontrivial sectors, used only for stationary continuation of named branches and their linear-spectrum record, under a law it derives and pins itself, with nothing from `m8_5c/design_inputs/` or the C2 room. It is not a second qualification of the spectral chassis that `M8.5-C2-FAILED` closed: M8.5-C2 does not rerun, no M8.4 observable enters, `M8.4-R1` never files, and its results say nothing about that chassis in either direction. Outside this admission: time integration in a harmonic basis, which is the closed route's evolving half and needs a ruling on this thread; any use as a route for the two MODELS.md criteria gated on field dynamics; and the census's persistence tests, which need their own pre-registration.

Every point below runs inside it.

### What this task is not

- **No persistence test,** per the boundary. S5's obligation stays open, under a pre-registration of its own, and so does the census's close, which needs S5.
- **No new instrument, no revised instrument code, no new bar.** The instrument, K1 and branch identity's test are M8.16.1's, and the rule is M8.16.1's with the level the answer of 2026-10-30 fixes.
- **No cutoff but 24, 32 and 40 in Arm A, and none but 16, 24 and 32 in Arm B.** `N = 16` is not read in Arm A, C1's own check aside (§ ARM A, The kernel), and no cutoff is added after the freeze.
- **No grid point but the twelve is classified in Arm A.** `N = 40` solves the paths' other grid points and reads no spectrum there, apart from a bracket's end, which adds no classification (§ ARM A). M8.17's classifications at those points stand, and the rule does not read them on this ladder.
- **No moved midpoint but T4/R5's.** A bracket this run creates uses M8.17's locator, with no moved midpoint, under Arm A's rules for a stored or borrowed end (§ ARM A).
- **Not a rescoring of M8.15, M8.15.2, M8.16.1 or M8.17.** Their verdicts stand as adjudicated. A point M8.17 read Unresolved and this run classifies has two records, on two ladders, and neither corrects the other.
- **No point on T5J or T5's other members,** and **no M8.4 object, slot relation, mass or particle state,** as in M8.17. Nothing here moves M8.7's gate or a MODELS.md cell.

### Known outcomes, and what is new

These have outcomes on record. This task names them and claims no new evidence from them.
- **M8.17's readings of the twelve** ([`RECORD.md`](https://github.com/dmobius3/mode-identity-theory/blob/7b66f8a0e207888042e6bb4a9ab636f946cb2a81/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-17/RECORD.md), § 3): every reading it stored there, the 19 count changes among them, and T4/R4's two non-contractions, at 64 and 72.
- **M8.17's locator on (8, 11.31)** (`RECORD.md`, § 4): every midpoint it computed, at every cutoff.
- **The replay,** filed with its scripts (`dev/replay_band.py`, `dev/replay_14.py`, `dev/set_motion.py`), and every reading in it. It reads the frozen rule on the stored `N = 24` and 32 summaries alone, so it stays a two-cutoff preview.
- **M8.16.1's records of the controls,** at `N = 16`, 24 and 32, and C1's at `N = 40`.
- **M8.16's development record** holds `N = 40` on C2a, C2b and C3 at `ε` = 0.5, 1 and 2 (`DEVLOG.md`, entry 78), and at each ladder's top three points (entry 106), on the instrument as it then stood. Each was accepted with the expected kernel, type and indices, and branch identity held at the nine ladder points.

**New evidence:**
- the `N = 40` records at the twelve points, and what the rule reads from them;
- the six controls' readings on this ladder, which test the ladder and say nothing of a row;
- T4/R5's moved midpoint;
- each guard reading, the midpoints of any bracket this run creates, and the new record at a bracket's end outside the twelve.

### Ownership and run format

Author-proposed, author-run, maintainer-adjudicated, as M8.17. One registered run on the author's machine, on mains power with sleep off, after the maintainer's go and the freeze block's pull request.

## TWO ARMS, ONE REGISTRATION

**The order of the run:**
1. **Arm B,** to its record, written before any `N = 40` solve.
2. **Arm A's gates:** the controls' paths at `N = 24` and 32, each reproducing M8.16.1's record; then the kernel at `N = 40`; then the control gate.
3. **Arm A's targets,** in the order T5/R2, T1/R4, T4/R4, T1/R5: each path at `N = 24`, then 32, then 40.
4. **Arm A's classification,** and the locator on any new bracket.
5. **The record.**

**What stops what.** Each arm has its own gates and its own failure:
- The kernel check and the control gate are Arm A's: the answer of 2026-10-29 sets them in its § 1, for the reading at `N = 40`, and its § 2 sets none for the locator, which may file on its own. Arm B also runs first, so its record is written before either gate.
- An unsuccessful stop in Arm B ends Arm B only. Arm A goes on.
- A failure inside Arm A's gates ends Arm A. The twelve stay unclassified, and the record is written.
- **A reproduction failure is global.** A recomputed point that does not equal its record, in either arm, is evidence about the machine or the code and not about a branch, so it stops the run. Every record already written is kept, and no reading of Arm A is classified. Arm B's record, if its own path and midpoints reproduced, stands as written, with the stop named beside it.
- A scored reading whose stored band is not the frozen level's is a defect in the code: it stops the run, as a reproducibility failure does, and no reading of Arm A is classified.
- A read refused at `N = 40` is the same kind of defect, and stops the run the same way.
- **A job that runs out of memory, or that three launches start and leave unfinished, ends the part of the run it belongs to,** with the cause named in the record:
  - a control's `N = 40` path, read or identity, or C1: the kernel check fails, and Arm A ends;
  - a target's job: that row stays unread, and the run goes on to the next row;
  - a location: that bracket is not located, and its ends keep their classifications;
  - Arm B: Arm B ends, not located, and Arm A goes on;
  - any other job: the run stops.
- **Any other exception the run raises** is a defect in the code. It stops the run on its first launch, as a reproducibility failure does, so a defect is never counted as an interruption.

One registration means one adjudication, so a dispute over either arm holds up both.

## THE INSTRUMENT AND THE LADDERS

**The instrument** is M8.16.1's, byte for byte: each file under `instrument/` carries the SHA-256 it has at MIT `d1c0ec8`, under the manifest M8.16.1's pin `8cc2d7e2…` covers. `run/cases.py`, `run/seeds.py`, `run/checks.py` and `run/driver.py` are M8.16.1's, byte for byte. `N = 40` is an input the pinned code accepts, and no file under `instrument/` changes.

**The ladders.** Arm A's are `N = 24`, 32 and 40, each from its own seed, along its own continuation path. Arm B's are M8.17's, `N = 16`, 24 and 32.

**The solves.** Each solve uses the instrument's continuation and each read uses M8.16.1's `driver.read_point`, through M8.17's `census.Cutoff`, with its step split as § ARM A states. This run does not call `driver.run_case`, which passes its ladder's lowest cutoff to the rule and computes branch identity only between its two highest cutoffs. M8.17 showed that `Cutoff` reproduces `run_case` field for field: at low cutoffs on C2a in its development pass, and on every recomputed point of its two locator paths in the run.

**Why `N = 16` is left out (DERIVED).** Order `ε²` reaches only Lemma F's channels ([`soft-slot-branches.md`](https://github.com/dmobius3/mode-identity-theory/blob/7a2c638cd1b1bda1df57db1d47304d14aaa3249b/files/framework/files/working/files/soft-slot-branches.md), § III): levels 10, 14, 16 and 18 in `R4`; 8, 12, 14, 16 and 18 in `R5`; and 13, 17 and 19 in `R2`. `N = 16`'s truncated space stops at level 16, so it lacks level 18 in `R4` and `R5`, and levels 17 and 19 in `R2`. `N = 24`, 32 and 40 hold every one. The answers of 2026-10-20 (§ 1) recorded the same, and § III's check vi removes those channels to reproduce `N = 16`'s coefficients.

**The choice is made knowing the data.** It was made knowing M8.17's stored spectra. At the fourteen points M8.17 left open there are 22 count-change reasons, 20 of them distinct, and at every one `N = 16` holds the fewer members, while `N = 24` and 32 agree (`dev/replay_14.py`, which reproduces each recorded verdict before it reads a count; the maintainer's port confirms it). Only T4/R4 at 64 and T5/R2 at 132 have none.

**Branch identity** is M8.16.1's test on each pair: the lower cutoff's solution, embedded at the higher, lies within `10⁻²` of its norm once the orbit-tangent components are removed, and the two records agree in type, `n(L)`, `n(D)`, `K`, Krein counts, `V₀`'s dimension and ranks. Between `N = 24` and 32 it must reproduce M8.17's record.

## THE TESTED BAND: AN EXCEPTION, STATED

The rule tests every typed group below half the band edge. On this ladder, two frozen texts part over that edge:

| Text | What it says |
| --- | --- |
| The instrument's docstring, `instrument/convergence.py` | "the slot's lowest level above the lowest cutoff used" |
| [M8.16's terms, THE CONVERGENCE RULE](https://github.com/openwave-labs/openwave/blob/e8762263612a3e981f67009079c2cdae66a0071c/openwave/xperiments/m8_mit/research/tasks/m8_16_task_details.md#the-convergence-rule-as-built-frozen) | "the lowest truncation cutoff, `N = 16`", levels 17 and 18, and the floors 8.02, 10.03 and 5.58 |

While `N = 16` is read they agree. With it left out, the docstring moves the band and the terms' values do not. The answer of 2026-10-30 admits an exception, and this registration takes it: to the docstring's wording, and to the terms' reason for the level, "where the lowest cutoff's basis first lacks modes that the higher cutoffs carry". The level stays at 17 in `R2` and 18 in `R4` and `R5`, with the terms' floors. It is kept as a scope, the one under which M8.16.1 qualified the rule and M8.17's readings were adjudicated, and no longer as the place where the lowest cutoff's basis runs out of modes.

**The level is fixed for the whole registration.** If a control fails at this level, the gate stops the targets, and no other level is tried.

**Where the 16 enters.** `checks.reading` takes the cutoff as its argument `n_lo`, and `band_edge` derives the level from it. M8.17's `census.readings_of` passes the ladder's lowest cutoff, so on this ladder it would pass 24, and "byte for byte" does not reach this call. The change is at that call, in the census's own code, which moves to `census.rule`, with `BAND_N = 16` a frozen value and `BESIDE_N` the reading beside it:

```diff
-        reading = ck.reading([q['summary'] for q in pts], hi['eig'], hi['lam'], hi['record']['margin'], lib.CUTOFFS[0])
+    args = ([q['summary'] for q in pts], hi['eig'], hi['lam'], hi['record']['margin'])
+    return ck.reading(*args, lib.BAND_N), ck.reading(*args, lib.BESIDE_N)
```

No file under `instrument/` changes, and a test confirms that every scored reading's stored band is the frozen level's (§ THE TESTS BEFORE THE FREEZE).

**Where it applies:** every reading on this ladder: the controls, the twelve, and the guard readings of any location, since a guard is classified by the same rule. Arm B keeps `N = 16` among its cutoffs, so its band does not move.

**Reported beside it.** Each of those readings is also taken on the band above `N = 24`, from the same summaries, and its verdict and reasons go in the record beside the scored one. It neither classifies a point nor stops the gate.

**What a classified point claims.** The rule reads every Krein-negative and off-axis group wherever it lies, with each tested imaginary group's nearest group of opposite Krein sign. What the band leaves out is Krein-positive imaginary groups above half its edge, and a left-out group's own motion is not tested. On a Krein-definite branch none of them can change the type at the point read. Where a classified point holds a Krein-negative group, that argument does not reach it, so its record gives the distance from the left-out groups to the nearest Krein-negative group. A left-out group is one, at `N = 40`, that no row of the reading reads.

**The disclosure.** The choice was made knowing that it restores one control and four targets on the replay. On the stored `N = 24` and 32, the band above `N = 24` gives 23 of the 24 control points their recorded type and reads 7 of the twelve converged; the band above `N = 16` gives 24 of 24 and 11 of 12. T4/R4 at 72 reads Unresolved on both. Nothing here establishes what `N = 40` will read.

## ARM A: THE READING AT `N = 40`

### The points

| Row | The twelve | The path |
| --- | --- | --- |
| T1/R4 | 1.41, 32, 45.25, 64, 72 | the whole grid, to 72 |
| T4/R4 | 4, 8, 11.31, 64, 72 | the whole grid, to 72 |
| T1/R5 | 32 | the whole grid, to 32 |
| T5/R2 | 45.25 | the grid, to 45.25 |

**Each path** starts at the case's seed at `ε = 0.25` and continues along the grid, as M8.17's. A point where the solve fails or is not accepted ends its cutoff's path, as inherited, and every later point on that path is untested at that cutoff.

**Every solution is stored,** at every cutoff and grid point, with its `λ` and continuation slope, in the run's outputs. The maintainer can recompute both identities with no `N = 40` solve, and a bracket's end needs no recomputed path.

### Solving and reading

M8.17's `Cutoff.step` reads every point it solves. Here the two are separate operations:
- **A solve** is the continuation to a point, its acceptance check, and the stored solution. It computes no linear-spectrum record.
- **A read** is `driver.read_point` on a solution, as M8.17's `step` calls it.

At `N = 24` and 32 every grid point is solved and read, so that each can be compared with its record. At `N = 40` every grid point is solved, and a read happens only at:
- a control's registered point;
- one of the twelve;
- a bracket's end, as § Locating a change of type states;
- a location's midpoints and guard readings.

That scope is frozen, and the run refuses a read at `N = 40` anywhere else.

### The kernel at `N = 40`

It is measured first, before any of Arm A's targets at any cutoff, on M8.16.1's controls in its order: C1, C2a, C2b, C3, C2a/R5, the pyramid P5 and the C3 ray C3R/R4, at their registered points, `ε` = 0.25, 0.5, 1 and 2, and C1's 0.5, 1 and 2.

**It passes** when, at `N = 40`, each control's seed passes its check, every point is accepted, and K1 holds as M8.16.1's terms freeze it, with its calibration constants unchanged: `checks.k1` at each point, and for C1 `checks.check_c1` over its four registered cutoffs, which reproduces M8.16.1's record.

**If it fails at any control,** Arm A ends there. No further `N = 40` solve runs, no point is classified at `N = 40`, and the twelve stay unclassified. A control's `N = 40` path, read or identity, or C1, that runs out of memory, or that three launches start and leave unfinished, is a failure of this check, recorded with that cause. An interrupted run resumes under the run's rules.

### The control gate

With the kernel passed, each control but C1 is read on `N = 24`, 32 and 40 at its registered points, on the band above `N = 16`. Each point must read converged with the type and indices on its record (`checks.converged`, byte for byte), and branch identity must hold on both pairs. This asks more than the answer's "the type it has on the record": a control passes only where a target would be classified. C1 has no rule reading: its test is the check above.

**If a control does not pass, none of Arm A's targets is read,** at any cutoff, and no other level is tried. A failure in any slot stops all of them. That is stricter than M8.16.1, where C2a/R5 gated the `R5` rows alone, and it is how the answer of 2026-10-29 words it. All six controls are read and recorded before the gate decides, since by then the reading needs no further solve.

### Classifying a point

**A point is classified** when all of these hold:
- at `N = 24`, 32 and 40, it is accepted and K1 holds;
- the convergence rule, on those three cutoffs' summaries (`checks.reading`, with `n_lo = 16`), reads it converged. As in M8.17, "converged" is the rule's verdict alone;
- branch identity holds between `N = 24` and 32, and between 32 and 40.

The rule's type, with `N = 40`'s indices, is the point's. M8.15 decided none of the twelve, so M8.17's precedence between the instruments is not reached.

**A point is not classified** when it is untested at a cutoff, K1 fails, the rule reads it Unresolved, or either identity fails. Each such point is named in the record, with the rule's reasons or the failure.

### Locating a change of type

M8.17's locator, on this ladder:
- **A row's classified points** are M8.17's and this run's. Type brackets, index brackets and changes across unclassified points are as M8.17's terms define them.
- **The deciding cutoffs are `N = 32` and 40.** `N = 24` is bisected where its ends' `n(L)` is resolved and different, and reported otherwise. Where M8.17's text has `N = 24` and 32 deciding and `N = 16` reported, this ladder reads `N = 32` and 40 deciding and `N = 24` reported.
- **A bracket's end outside the twelve** has its `N = 40` record computed from its stored solution, and it keeps the classification M8.17 gave it. M8.17's locator reads two ends classified on one ladder, so before such an end enters a bracket it must meet that condition on this one: its `N = 40` record is accepted and K1 holds, its type, `n(L)`, `n(D)`, `K` and `k_i⁻` agree with M8.17's classification, and branch identity holds on both pairs. If any of these fails, the bracket is not located under this registration. An end M8.17 classified on M8.15's instrument alone holds no indices to agree with, and an end past its row's path has no stored solution, so neither can meet the condition. The end's adjudicated classification stays unchanged. Its `N = 40` record is evidence that the bracket is admissible, not a new classification, and the rule does not read it.
- **Each bisection starts from the ends' stored solutions,** since M8.17's terms recompute an end only "where an end has a known record but no stored solution vector". M8.17's code always recomputed, since only the pyramid's vectors were stored then, so this run's locator code takes the stored ends, shown in the diff. So no path is recomputed here, M8.17's comparison of a recomputed path does not run, and no reading on this ladder is compared with a record on M8.17's. Each stored solution passes the solve-state gate (§ THE FREEZE, THE GUARD AND THE RUN).
- **The guard readings** are read at all three cutoffs at one `ε`, and classified as above, with branch identity on both pairs.
- **The name** is M8.17's, with "at `N = 32` and 40". The location claims "a change of type located between the guard readings", and its identification with the genuine index crossing is not claimed, as the census page's step 2 note states.

## ARM B: THE REVISED LOCATOR ON T4/R5

**The bracket** is T4/R5's (8, 11.31), and no other. Its ends read Elliptic with `(n(L), n(D), K) = (7, 1, 6)` and Hyperbolic with `(6, 1, 5)`, at every cutoff. `N = 24` and 32 decide, and `N = 16` is run and reported.

**The path.** Each cutoff's registered path is recomputed from its seed to 11.31, and each recomputed point must equal its record in M8.17's outputs. Each of M8.17's midpoints, at each cutoff, must equal its stored record, timings aside.

**The one change.** Each midpoint where K1 fails is moved once, to the point 0.4 of the current interval above its lower end, and solved there from both ends. K1 failing at the moved point is an unsuccessful stop, as now. In `census_lib.bisect`, with `moved` frozen at 0.4 for Arm B and `None` for every other bracket:

```diff
         s_lo, s_hi = solve(end_a, m), solve(end_b, m)
         mids.append(dict(eps=m, from_lower=s_lo, from_upper=s_hi))
+        if moved is not None and s_lo['ok'] and s_hi['ok'] and not (s_lo['k1'] and s_hi['k1']):
+            m = a + moved * (b - a)
+            s_lo, s_hi = solve(end_a, m), solve(end_b, m)
+            mids.append(dict(eps=m, from_lower=s_lo, from_upper=s_hi, moved=True))
         stop = None
```

K1 is not bypassed: every midpoint the locator keeps passes it. The midpoint K1 refused stays in the record and decides nothing, and a midpoint where a solve fails is not moved. Nothing else changes: the two solves from each end, their agreement, `n(L)` by inertia, the completion width, the location's width, the guard readings and the name. The code is the census's own, and no file under `instrument/` changes.

**The design, stated.** The fraction and the direction were chosen from M8.17's locator record, and they decide in advance whether the bisection can complete. The record stops at the interval [8.718402, 8.724874] at both cutoffs, 6.472×10⁻³ wide, 7.4×10⁻⁴ of its lower end against the 5×10⁻⁴ that completes it. The values they were chosen from (`dev/moved_midpoint.py`):

| `ε` | `n(L)` | Nearest eigenvalue outside the zero cluster, `N = 24` | `N = 32` |
| --- | --- | --- | --- |
| 8.673097 | 7 | 9.4335×10⁻⁵ | 9.4456×10⁻⁵ |
| 8.698985 | 7 | 6.4790×10⁻⁵ | 6.4970×10⁻⁵ |
| 8.711930 | 7 | 4.2603×10⁻⁵ | 4.2879×10⁻⁵ |
| 8.718402 | 7 | 2.4808×10⁻⁵ | 2.5282×10⁻⁵ |
| 8.721638 | 7, K1 fails | 3.6937×10⁻⁶ | 6.1163×10⁻⁶ |
| 8.724874 | 6 | 2.4280×10⁻⁵ | 2.3785×10⁻⁵ |

K1's gray zone reaches up to three margins: 6.1×10⁻⁶ at `N = 24` and 8.2×10⁻⁶ at `N = 32`. The squares on the `n(L) = 7` side fall at 1.82 to 1.86×10⁻⁷ per unit of `ε` at both cutoffs. Extrapolating them (INFERRED), the pair reaches zero near 8.72171 at `N = 24` and 8.72184 at `N = 32`, and K1 refuses from about 8.7215.

The moved point is 8.720990, between two readings on the record. Interpolating their squares (INFERRED) puts the nearest eigenvalue there at about 1.16×10⁻⁵ and 1.26×10⁻⁵, 1.9 and 1.5 times K1's limit. The interval it leaves is 3.88×10⁻³ wide if `n(L)` reads 7 there, and 2.59×10⁻³ if it reads 6, each narrower than the 4.36×10⁻³ that completes the bisection.

**What stays open until the run:** whether K1 holds at the moved point; `n(L)` there, by inertia, at both cutoffs, which can read either 7 or 6; the two solves' agreement; and the guard readings.

**The guards.** M8.17's record puts `N = 16`'s change of `n(L)` in [8.851079, 8.854315], above both deciding cutoffs'. The upper guard at `δ = 0.01`, near 8.812, lies below it, so `N = 16` should read `n(L) = 7` there against 6 at `N = 24` and 32, and the rule's structure condition should leave that guard Unresolved (INFERRED). The retry at `δ = 0.05`, near 9.161, lies above it. That is the rule and the locator as they stand, and nothing is changed for it.

**What a location claims** is "a change of type located between the guard readings", with `n(L)` changing inside the interval at `N = 24` and 32. Its identification with the genuine index crossing is not claimed, and Corollary I is not applied.

**A failure.** The bracket stays not located, with the stop named. Nothing is repaired and rerun, and the bracket is not offered to this instrument again. A job that ends Arm B is named as that, in a record of its own kind: it is neither a stop of the locator nor a reproduction, and the record claims nothing of the path or the midpoints.

## THE ROW OUTCOMES

After the registered readings, each of the five rows receives one of M8.17's run outcomes, from M8.17's classified points, this run's, and the locations: Elliptic throughout; Each change of type located; or Not settled by this run. M8.17's outcomes stand as adjudicated, and these state where each row stands after this run. Whether a row has met an S4 completion condition is stated separately, in the record.

T5/R2 ends Not settled by this run whatever 45.25 reads, since 128 and 132 are not read here.

**A failure.** A point this run does not classify is accounted for as S4 and S6 read it, with its evidenced cause and no type claimed. Nothing is repaired and rerun under any ID, and the point is not offered to this instrument again.

**S6.** The record states each cause. The census page applies S6 to each item after the adjudication, with the cause the adjudication names, as the answer of 2026-10-29 reads it (§ 3):
- an item with no remedy admitted closes under S6's last sentence. T5/R2's 128 and 132 are the two such items, and S6 is applied to T5's row once 45.25 has been read again;
- a remedy that meets the same cause again is the second instrument-limited outcome with that cause, under S6's first sentence, and the item closes the same way;
- no ruling per cause is needed;
- a closure claims no type, and no shared cause with the finite-element instrument is claimed while § IV leaves that one unknown.

Closure is per item. Classified points keep their types, and S5 still applies to their Elliptic stretches.

**Where Arm A's gates stop the targets,** no target has been read. The record names the failed kernel or control gate and keeps each point's cause as M8.17 recorded it, and that is not a second outcome with that cause.

**Where a job ended a target,** that row stays unread, and the same holds: each of its twelve keeps M8.17's cause, with this run's cause named beside it, and that is not a second outcome with that cause. The other rows are read and classified as usual. A bracket a job ended stays not located, and its ends keep their classifications.

S4's third ending is not reached here, as in M8.17.

**For the catalog,** the census page's dated note after the adjudication gives each row's ending and moves its status, on MIT, and names each point's instrument and ladder.

## STATED LIMITS: WHAT IS ADDED

- **The calibration's range.** `N = 40` sits at the edge of K1's calibrated range: "Above `N = 40`, K1's margin is an extrapolation" (the answers of 2026-10-20, § 2). As in M8.17, the margin was fitted in `R2`, `R3` and `R4`, so `R5` lies outside the fitted slots.
- **The ladder is new.** The rule's contraction test reads the steps 24 to 32 and 32 to 40, not 16 to 24 and 24 to 32. The control gate shows that it reads the controls correctly at `ε ≤ 2`. The twelve reach `ε = 72`, and no registered control is read there on this ladder, so the gate is necessary, not sufficient. M8.16's development record read C2a, C2b and C3 at `N = 40` at their ladders' top three points, each accepted, with branch identity kept (`DEVLOG.md`, entry 106): unscored evidence above `ε = 2`, on the instrument as it then stood, and not part of the gate.
- **The band.** A left-out group's motion is untested, as § THE TESTED BAND states.
- **The count changes are not a window effect alone.** M8.17's stated limits describe most count changes at M8.16.1's Unresolved points as a same-sign eigenvalue just outside the closure's window. That fits C2a/R5 at 2 and T4/R4 at 4. It does not describe T1/R5 at 32, T1/R4 at 64 or T4/R4 at 64 on its own: those sets are still moving between `N = 24` and 32. The maintainer's measure, which is not the rule's, gives 5.3×10⁻³, 4.8×10⁻³, and 8.3×10⁻³ to 1.3×10⁻² there, against 2.3×10⁻⁵ and 7.0×10⁻⁵ (the answer of 2026-10-30, § 3). Each set is Krein-positive, and its motion is far below its distance to any Krein-negative group.
- **The other grid points.** M8.17's classifications there rest on `N = 16`, 24 and 32, and this run does not test them on the new ladder.
- **A new bracket has no moved midpoint.** A midpoint that lands in K1's gray zone near a slow pair's crossing is an unsuccessful stop, as T4/R5's was. T4/R4's change, across 4, is the likeliest such bracket: its ends carry T4/R5's indices, (7, 1, 6) and (6, 1, 5), so it is likely the same kind of crossing. The stop would leave it not located.
- **A location's resolution** is M8.17's: an `n(L)` interval and two guard readings, not a crossing point.

## THE TESTS BEFORE THE FREEZE

Each test is logged in this task's `DEVLOG.md` with its script's SHA-256. No `N = 40` solve runs before the freeze.
1. **Every constant.** Every numeric item in this document is classified as a literal the code reads and `FROZEN.json` holds, a value its calculation reproduces, or an unscored report, with M8.17's reverse scan.
2. **The reader, with no solve.** On M8.17's stored summaries, the reading code reproduces M8.17's reading at every stored point, on `N = 16`, 24 and 32. On a point with a cutoff missing, it refuses to classify.
3. **The band.** On stored records, every scored reading's band equals the frozen level's: 17 in `R2`, 18 in `R4` and `R5`. A mutation that passes 24 must fail the test. The run applies the same test to every scored reading before it writes the record. The reading beside it, on the band above `N = 24`, is recorded and changes nothing: on synthetic records where it reads Unresolved and the scored reading converged, the point classifies and the gate passes. At a point with a Krein-negative group, the recorded distance from the left-out groups to the nearest one equals a direct computation, and a mutation that omits it fails.
4. **The reproduction gate, on data on record, with no solve:** the exact comparison of a path at `N = 24` and 32 against M8.17's outputs and M8.16.1's controls, with P5's `k3_overlaps` at 0.25 left out since `Cutoff` does not compute them; the comparison of M8.17's midpoints; the solve-state gate, on a stored file; and a mutation of each compared field or file, which must fail it.
5. **Classification and identity, on synthetic records.** Each condition has an arm whose removal lets a counterexample through: K1 at each cutoff, the rule's verdict, each identity pair, and the type from `N = 40`. M8.17's stored distances at T5/R2's 128 and 132 are the counterexample for the first pair.
6. **The gates' scope, on synthetic records.** A kernel failure ends Arm A and leaves Arm B's record; a control failing the gate stops the targets; a control failure in `R5` stops `R4`'s targets too; an Arm B stop leaves Arm A running; a reproduction failure stops both. Each route of a job that runs out of memory, or that three launches start and leave unfinished, has its own scene, with no fourth start of the job: a control's `N = 40` job and C1 fail the kernel check and no target runs; a target's row stays unread, with the rows before and after it classified; a location's bracket is not located, and its ends keep their classifications; Arm B ends, naming the job, and Arm A goes on; any other job stops the run. Attempts are counted per job, a job interrupted fewer than three times completes, and a job out of memory is not started again on a later launch. Any other exception stops the run on its first launch. A reproduction failure after a target has finished still classifies nothing. A scored reading off the frozen band and a refused read each stop the run. A stop on record is read again on a resume. Arm B's bracket must equal the frozen one. With a spy on `driver.read_point` and stub solves through the run's whole order, a read at `N = 40` outside its frozen scope must be refused, and so must a `N = 40` solve started before Arm B's record is written. At the paths' other `N = 40` grid points the spy counts no call, and a solve not accepted there ends the path.
7. **The locator, on synthetic records.** M8.17's arms carry over, on both ladders. With `moved = None`, `bisect` must equal M8.17's on every arm. The moved midpoint has its own arms:
   - K1 fails at a midpoint and holds at the moved point: the moved point counts, and the refused midpoint stays in the record and decides nothing;
   - K1 fails at the moved point: an unsuccessful stop, with no second move;
   - a solve fails at a midpoint: no move;
   - K1 holds at a midpoint: no move;
   - M8.17's ten stored midpoints, replayed through a stub solver, reach the stop, and the moved point completes the bisection with `n(L)` 7 and with 6;
   - a mutation of the fraction or the end must fail.

   A bracket's end outside the twelve has its own arms: its `N = 40` record failing K1, failing either identity, or differing from M8.17's type or any index must leave the bracket not located, with the end's classification unchanged. An end with no indices on record, or past its row's path, is refused with that cause.
8. **Plumbing, with real solves,** on the GO-list control C2a at `N = 12`, 16 and 20, where the run never solves it: paths through `Cutoff` with a spectral record at chosen points only, stored solutions, a resume from them, both identity pairs, and the band's call.
9. **The freeze rehearsal,** as M8.17's.

## FROZEN VALUES: WHAT IS NEW

- the twelve points, the paths, and the targets' order;
- Arm A's ladder, 24, 32 and 40, and its deciding cutoffs, 32 and 40;
- Arm B's ladder, 16, 24 and 32, and its deciding cutoffs, 24 and 32;
- `BAND_N = 16`, and the band above `N = 24` as the unscored reading;
- the controls, their order and their registered points;
- `moved = 0.4`, from the lower end, for T4/R5's (8, 11.31) alone;
- `ATTEMPTS = 3`, the starts a job is allowed, counted per job from the run's log;
- M8.17's outputs, by the SHA-256 its run's log records (`run/out/RUNLOG.json`, `7d761506…`), and M8.16.1's controls, C3R/R4's among them, by its log's;
- the instrument's files at `d1c0ec8` and this task's manifest pin.

`w`, `δ`, the two-ends agreement and branch identity's `10⁻²` are M8.17's.

## THE FREEZE, THE GUARD AND THE RUN

As M8.17's: the guard writes `run/FROZEN.json` and its pin, the filed terms are copied to `run/m8_16_task_details.md`, the freeze block files on OpenWave before the run, and the run starts after the maintainer's go. Four additions:
- **The known points reproduce,** at the cutoffs that have records:
  - Arm A's paths at `N = 24` and 32 must equal M8.17's outputs, each cutoff's points field for field and the identity between them, timings aside. The controls' must equal M8.16.1's in the same way, and C1's at all four cutoffs. The rule's readings are not compared, since their ladder differs;
  - Arm B's path and M8.17's midpoints reproduce as § ARM B states.

  The first point that does not reproduce stops the run, as a reproducibility failure, and nothing is repaired or rerun.
- **The solve-state gate.** `N = 40` has no record to reproduce, so nothing at that cutoff is compared with one. The run's log records the SHA-256 of each path job's stored solutions when the job completes. A stored solution is used, by a read or by a bisection's continuation, only if its file still has that SHA-256 and its `λ` equals the job's record. One that fails stops the run, as a reproducibility failure.
- **Interruption.** M8.16.1's rules apply to each job in place of each case: Arm B, C1, a path at one cutoff, its read, an identity, a case's reading, the gate or a location. A resume refuses another tree's manifest, a stopped run and a completed job whose file differs, moves partial outputs aside, and restarts at the first incomplete job. Each job's start is logged before it solves, and its attempts are counted from those entries alone, so an interruption elsewhere, or a launch that never reaches the job, does not count. A completed job is never repeated. A job that ran out of memory is not started again, and a launch that finds a job started three times and not completed starts it no fourth time: each takes its route.
- **The estimate,** INFERRED from M8.16.1's and M8.17's timings and M8.16's at `N = 40`: about 55 to 70 hours of compute.
  - about 6 hours for Arm B;
  - 10 to 13 for the controls;
  - about 7 for the targets at `N = 24` and 32;
  - 31 to 44 for the targets' `N = 40` paths;
  - about 9 to 13 more for each new bracket in `R4`.

  The kernel check measures `N = 40`'s time and memory on the controls, before any of Arm A's targets: the run's log gives each job's time and the process's peak resident size. No target's path at `N = 40`, and no `R5` path at that cutoff, has run on this instrument.

## DEFINITION OF DONE

- **The record:**
  - the kernel check, and each control's reading on the ladder;
  - each of the twelve with its classification or its cause, on both bands, and the Krein-negative distance where it applies;
  - every bracket, with its location or the condition it failed;
  - each of the five rows' standing;
  - each stop, in either arm, and each job that ended a part of the run, with its cause.
- **The publication:** the record, `REPRODUCE.md`, the run's outputs and stored solutions, and the diff against M8.17's code, on MIT, in this task's folder.
- **The method note** on OpenWave, and the maintainers' adjudication.
- **The roadmap:** both M8 status rows record the filing, and the Done row states the adjudicated outcome.
- **After the adjudication,** a dated note on the census page with each row's standing and catalog status, S6 applied per item, and S5's obligation stated as open.
