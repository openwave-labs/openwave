# M8.16.1: THE SECOND AND LAST QUALIFICATION OF THE STATIONARY HARMONIC GALERKIN INSTRUMENT, a delta on M8.16's terms

Pre-registration, for review. It governs nothing until the maintainer's go. **No sealed case is solved before the freeze:** no census row, not the C3 ray in `R4`, and not T5's second member. None of them has ever been solved on this instrument (§ The split). Development ran only the GO list. Each run is logged with its script's SHA-256 in this task's `DEVLOG.md`, and disclosed here (§ The split, and the development record).

## TASK PLANNING

### The question

**What M8.16 left open.** M8.16 ran on 2026-10-18, and the maintainer adjudicated it on [#640](https://github.com/openwave-labs/openwave/pull/640#pullrequestreview-5575251218) on 2026-10-19: NOT QUALIFIED.
- The five controls passed every check.
- The pyramid control's quartet lay 5.76%, 5.78% and 5.78% from Theorem R's leading order at `ε = 0.25`, at `N = 16, 24` and 32, over the 5% bar. That failed K4, and the same reading failed K3(a)'s slow-band clause, so K3's outcome was "instrument failure: the diagnosis is untested".
- No row ran, and K5 and K6 have no outcome. The failure stands as registered.

**What has been derived since.** The Soft-Slot Branches, § III, gives the slow spectrum one order past Theorem R at a nondegenerate orbit: Proposition N, DERIVED ([§ III at `e7652a4`](https://github.com/dmobius3/mode-identity-theory/blob/e7652a447efde0c56aee2713298d3a042bef87b4/files/framework/files/working/files/soft-slot-branches.md#iii-the-small-amplitude-spectrum)). Each slow group's eigenvalue is `λ_g = εμ₀ + ε²c_g + O(ε³)`, with `μ₀ = κs_g` and `c_g` in closed form.
- With nothing fitted, `c_g` reproduces the next-order coefficients extrapolated from M8.16's readings at the pyramid and the four soft-slot controls, within their extrapolation spread. This holds at `N = 24` and 32, and at `N = 16` with the channels above that cutoff removed (INFERRED).
- At the pyramid, two terms put the quartet 5.91% from leading order at `ε = 0.25`, where the record reads 5.78%. Its departure over the bar is its own next-order term (INFERRED).
- M8.16's record, § 7, left "how a bar at the smallest amplitude allows for a mode's next-order term" to a later pre-registration.

**What this task asks:** does the instrument qualify, and for which rows, when the checks a mode's next-order term can break are scored against Proposition N? Everything else is M8.16's, as registered.

**Why it is the last.** #512's ruling of 2026-10-20 admits this task as "a second and last qualification" of the instrument. A row it does not qualify, or a member a row names that it does not carry, is not offered to this instrument again (§ THE VERDICT RULE: WHAT CHANGES).

**Old and new evidence.** The instrument is deterministic. With the instrument's code unchanged, the controls' and the pyramid's outcomes are on M8.16's record, under whichever bars apply to them (the ruling, § 1). This task names them and claims no new evidence from them (§ Known outcomes). The new evidence comes from the census rows, the C3 ray in `R4` and T5's second member.

### Standing

- **The rulings it files under,** all on #512:
  - the answer of 2026-10-15 ([18955679](https://github.com/openwave-labs/openwave/discussions/512#discussioncomment-18955679)), whose § 3 is the boundary below, as for M8.16;
  - the ruling of 2026-10-20 ([19071299](https://github.com/openwave-labs/openwave/discussions/512#discussioncomment-19071299)), which admits M8.16.1 and sets its conditions;
  - the answers of 2026-10-20 ([19076863](https://github.com/openwave-labs/openwave/discussions/512#discussioncomment-19076863)), on the frozen values at `N = 16`, the census run's adequacy tests, and a second member of T5.
- **The conditions, and where they are met:**

| Condition | Where |
| --- | --- |
| Identity: M8.16.1, the second and last qualification; the census run takes the next free ID | this task; ID, below |
| Boundary: the 2026-10-15 answer's § 3, verbatim | § The boundary |
| The instrument, byte for byte, as M8.16's pin `a5f45173…` covers it | § THE INSTRUMENT: UNCHANGED |
| Proposition N: one value per mode and cutoff, from one pinned script, and the C3 ray's prediction, frozen before the go; the agreement with M8.16's readings cited as INFERRED | § THE NEXT-ORDER BAR; § FROZEN VALUES: WHAT IS NEW |
| The tolerance: the points, the fit form, the spread, and its frozen ceiling at every cutoff | § THE NEXT-ORDER BAR |
| Known outcomes, named, with no new evidence claimed | § Known outcomes |
| The split: a GO list only; the rows and the C3 ray sealed; no credit; never solved here | § The split |
| T4 in `R5`: its resolution and accuracy check, and its reading if that fails | § THE NEXT-ORDER BAR, check 2 |
| A second member of T5: named, sealed, its own predictions, K5 and no K6, its outcome its own, DERIVED as not symmetry-equivalent to the registered fixed-`i` standing wave | § T5'S SECOND MEMBER |
| Run-before-write; the word budget; maintainer reproduction | below; inherited |
| The roadmap: both M8 status rows record the filing | the pull request that files this task |
| The census page: the dated note under step 4, before any terms | MIT `deec830` |

- **Run-before-write** (`PR_REVIEW_STANDARDS.md` § 12.2) is met. M8.16 ran and was adjudicated on 2026-10-19 (#640). Since then the column has landed only that run's records after the verdict ([#642](https://github.com/openwave-labs/openwave/pull/642)).
- **Word budget.** About 7,700 words, inside § 12.2's cap of about 8,000.
- **ID.** The ruling fixes M8.16.1, which no M8 file or open pull request carries. The BACKLOG row this pull request creates allocates it (`ROADMAP_STANDARDS.md` § 6).
- **Inherited.** Every rule of M8.16's terms that this document does not change governs here. M8.16's governing text is its filed pre-registration, SHA-256 `86992a5f…`: the 48,450-byte prefix of [`m8_16_task_details.md`](https://github.com/openwave-labs/openwave/blob/5b7a56b85d733d53e257cf7559b1b69ac05276a2/openwave/xperiments/m8_mit/research/tasks/m8_16_task_details.md) that its dated note of 2026-10-19 names. That note's reading of the verdict rule's K3 sentence, that K3's list governs, carries over, and is written out below. Inherited unchanged:
  - SETTING;
  - THE INSTRUMENT, AND THE SEPARATION STATEMENT: the basis, the quartic, the exact kernel, the solve and its acceptance bound, continuation, the seeds and their check, the linear record;
  - M8.16's fifteen cases and their expected columns;
  - K1, K2, K3(b) to (d) with K3's three outcomes, K4's items for C1, C2a, C2b, C3 and C2a/R5, K5's clauses other than its bar and its unscored reading at `ε = 0.5` and 1, which the next-order bar's check 1 now scores, and K6 with its branch identity and scoring;
  - THE CONVERGENCE RULE, as built and frozen;
  - THE RUN's points, its interruption and resume rules, and its power and sleep conditions;
  - validity, and no search;
  - STATED LIMITS, with the additions below;
  - FROZEN VALUES, with the additions below;
  - the guard, the freeze block and provenance, with new pins;
  - the definition of done, with the additions below.
- **What this task changes:**
  - the bars a next-order term can break, now scored against Proposition N (§ THE NEXT-ORDER BAR);
  - two sealed cases: the C3 ray in `R4`, a control that gates nothing, and T5's second member (§ THE CASES: WHAT IS ADDED; § T5'S SECOND MEMBER);
  - the verdict rule, for the stop and the new cases (§ THE VERDICT RULE: WHAT CHANGES);
  - three tests of the frozen text, before the freeze (§ THE TESTS BEFORE THE FREEZE);
  - one amendment to the split (§ The split);
  - the stated limits development found, on `R5`'s ladder (§ STATED LIMITS: WHAT IS ADDED).

  The scoring is new code in `run/`, in a folder of its own beside M8.16's, which it leaves unchanged.

### The boundary (#512, § 3, verbatim)

> Admitted on #512: a harmonic Galerkin instrument on the slot action's nontrivial sectors, used only for stationary continuation of named branches and their linear-spectrum record, under a law it derives and pins itself, with nothing from `m8_5c/design_inputs/` or the C2 room. It is not a second qualification of the spectral chassis that `M8.5-C2-FAILED` closed: M8.5-C2 does not rerun, no M8.4 observable enters, `M8.4-R1` never files, and its results say nothing about that chassis in either direction. Outside this admission: time integration in a harmonic basis, which is the closed route's evolving half and needs a ruling on this thread; any use as a route for the two MODELS.md criteria gated on field dynamics; and the census's persistence tests, which need their own pre-registration.

Every case below runs inside it.

### What this task is not

- **Not the census run.** It decides no census row, and it reads the spectrum only where M8.16 reads it. The census run's adequacy tests are its own (#512's answers of 2026-10-20, § 2), and nothing here registers them. The census run's pre-registration waits until this task has run and been adjudicated.
- **Not a rescoring of M8.16.** M8.16's verdict stands as adjudicated on #640. K5 and K6 never produced an outcome, and the controls' and the pyramid's outcomes here are named as known.
- **No time integration and no persistence test,** per the boundary.
- **No M8.4 object,** and **no slot relation, mass or particle state,** as in M8.16.
- **No re-test of theorems.** Theorems B, D and R, and Proposition N, are the author's. A disagreement with them is recorded as a failure of the check, and the adjudication says what it indicts: the instrument, or a frozen value's computation.
- **No re-reading of M8.15,** as in M8.16.

### The split, and the development record

**The split** is M8.16's, carried over by the ruling ("as M8.16's split does"): Blake's ruling of 2026-10-15 with its four amendments, in M8.16's [`BOUNDARY.md`](https://github.com/dmobius3/mode-identity-theory/blob/deec830b29f4bd6b45e58d2c333946dbbb60e964/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-16/BOUNDARY.md) (SHA-256 `a54619de…`). One amendment is added, in this task's own `BOUNDARY.md` (SHA-256 `934cca3d…`):
- **Amendment 5 (Blake's decision, 2026-10-20): `R5`'s coherent branch, along M8.15's grid to `ε = 32`, joins the GO list for development only.**
  - It runs at `N = 16, 24` and 32, is logged in `DEVLOG.md` and disclosed, and earns no credit.
  - Why: no `R5` ladder had run on this instrument, and the margin was fitted in `R2`, `R3` and `R4` (STATED LIMITS). This is the only evidence on `R5`'s acceptance and calibration along a ladder that unseals no row.
  - Its reading is amendment 3's note's, with one difference: no constant is refitted on it. The calibration's constants are in `instrument/linear.py`, which is fixed byte for byte (§ THE INSTRUMENT: UNCHANGED), so a refit would be a revised instrument, which files only under a ruling on #512. A failed calibration condition is preserved and returned to Blake and the units before further development.

**SEALED until M8.16.1's freeze,** at every amplitude:
- every census row, T1 to T4 in both slots, and T5;
- the C3 ray in `R4`;
- T5's second member.

None of them has ever been solved on this instrument. M8.16's run stopped at the pyramid before any row ([`RECORD.md`](https://github.com/dmobius3/mode-identity-theory/blob/87fc6439d8ebea228901b06323259c82327c18d7/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-16/RECORD.md), § 4). Neither the C3 ray nor T5's second member was on M8.16's GO list or among the fifteen seeds that amendment 2 checked (M8.16's `DEVLOG.md`). This task's seed check read both on the lowest level only, as amendment 2 scopes it, with nothing solved (`DEVLOG.md` entries 17 to 19). The pyramid is not sealed: its readings are on M8.16's record.

**Development** earns no qualification credit. Its registered reruns count as conformance and regression evidence, not as independent confirmation.

**What development ran before filing.** `DEVLOG.md` logs each run, with its script's SHA-256 and each output's. It is written by hand: `devtools.log_run`, M8.16's, would misnumber it, since it counts the log's headings and this log began at 1 (entry 9).
- **The pinned next-order script** (entries 2 to 7): Proposition N's values at every case, the sealed ones included, from the reduced quartic alone, with no solve on the instrument, as the ruling's Proposition N condition requires. Its checks A to F and X hold, and its nine arms are red (§ FROZEN VALUES: WHAT IS NEW).
- **The record's check 2 and the plan's facts** (entries 8 to 10): check 2's estimate, read from M8.16's record, and the facts the scoring code's plan states, read again from source. Both ran off the instrument, with nothing solved.
- **The scoring code and the three tests of the frozen text** (entries 11 to 16 and 23), with the tests' outcomes in § THE TESTS BEFORE THE FREEZE.
- **The record regression** (entries 12, 15, 16 and 23): this task's checks on M8.16's recorded outputs, with no solve. It reproduces the known outcomes below.
- **The plumbing pass** (entries 13 and 15): the driver's loop on the GO controls at cutoffs the run never uses, never scored. It ran twice, with identical records apart from their timings.
- **The seed checks of all seventeen seeds** (entries 17 to 19 and 23): `Q(u)`, criticality and `(k, r_u)` on the lowest level only, as amendment 2 scopes them, with no solve and no spectrum. The seed check's claim 7 tells each new seed from the orbit whose values it shares, by its Majorana root multiplicities.
- **T5's second member's exact checks** (entry 27; § T5'S SECOND MEMBER).
- **Amendment 5's ladder** (entries 20 to 26), with its findings in § STATED LIMITS: WHAT IS ADDED.

### Known outcomes

These checks have outcomes on M8.16's record under this task's bars. This task names them, and claims no new evidence from them. The registered run decides them formally.
- **C1**, Theorem D at `R3`: K1 and K4 passed.
- **C2a, C2b and C3:** K1 and K4 passed, and K2 for C2a and C2b.
- **C2a/R5:** K1 and K4 passed, with its largest deviation at `ε = 0.25` at 0.88%, 0.93% and 0.93%, at `N = 16, 24` and 32. They gate the `R5` rows.
- **The pyramid:**
  - K1 and K2 passed;
  - K3(b) to (d), on the record unscored: the in-span pair at 0.35% of `0.25 κ τ_in`, the overlaps 0.9970 and at most 3×10⁻¹⁰, the axis fraction 0.987, and the separation at 532 times its uncertainty (#640, § 2);
  - K3(a)'s slow-band clause and K4's pyramid item, under the next-order bar's checks 1, 3 and 4: the quartet and the Krein-positive pair agree with `c_g` within their spreads at all three cutoffs, `N = 16`'s with its channels removed (Proposition N's claim F). The quartet's spread, the largest on the record, is 6.0×10⁻⁴ of `|κs_g|`, against a ceiling of 0.0236;
  - check 2, whose estimate the record keeps: every projector is usable, and `E_c` lies below its trigger, `C_g/10`, by 4.8 to 5.7 decades (`DEVLOG.md` entry 8).
- **`N = 16`'s agreement** at the pyramid and the four soft-slot controls (claim F, INFERRED).

So the instrument's verdict on C1 to C3 and the pyramid is predicted from the record in full. The record regression reads these outcomes with this task's own code, with no solve, and reproduces each (`DEVLOG.md` entries 12, 15 and 23): K3 reads resolved, and K4 passes at the pyramid. What this task adds is the rows, the C3 ray and T5's second member.

### Ownership and run format

Inherited from M8.16. The frozen terms' SHA-256 lands on MIT `main`, on the census page, before any sealed case runs.

### Sources of record

| Source | What it supplies |
| --- | --- |
| M8.16's terms: the governing prefix, `86992a5f…`, of OpenWave's [`m8_16_task_details.md`](https://github.com/openwave-labs/openwave/blob/5b7a56b85d733d53e257cf7559b1b69ac05276a2/openwave/xperiments/m8_mit/research/tasks/m8_16_task_details.md) | everything inherited |
| M8.16's [`RECORD.md`](https://github.com/dmobius3/mode-identity-theory/blob/87fc6439d8ebea228901b06323259c82327c18d7/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-16/RECORD.md) at MIT `87fc643`, and [#640](https://github.com/openwave-labs/openwave/pull/640#pullrequestreview-5575251218) | the readings; the known outcomes; the K3 sentence |
| MIT's [The Soft-Slot Branches](https://github.com/dmobius3/mode-identity-theory/blob/e7652a447efde0c56aee2713298d3a042bef87b4/files/framework/files/working/files/soft-slot-branches.md#iii-the-small-amplitude-spectrum), § III, at `e7652a4`, with `scripts/soft-slot-branches/next-order/` | Proposition N; checks iv to vi; claim F |
| The same page, [§ IV](https://github.com/dmobius3/mode-identity-theory/blob/e7652a447efde0c56aee2713298d3a042bef87b4/files/framework/files/working/files/soft-slot-branches.md#iv-the-orbits)'s table | the C3 ray's representative, value, signature and `(k, r_u)` |
| MIT's [The Slot Dynamics Census](https://github.com/dmobius3/mode-identity-theory/blob/deec830b29f4bd6b45e58d2c333946dbbb60e964/files/framework/files/working/files/slot-dynamics-census.md#v-the-program), at `deec830` | step 4 and its notes; S4 to S6; T5's row; § II's rule on members |
| M8.16's [`BOUNDARY.md`](https://github.com/dmobius3/mode-identity-theory/blob/deec830b29f4bd6b45e58d2c333946dbbb60e964/files/framework/files/working/files/scripts/soft-slot-dynamics-m8-16/BOUNDARY.md) | the split, with amendments 1 to 4 |
| This task's `BOUNDARY.md` and `DEVLOG.md`, in its folder | amendment 5; the development record |
| This task's `run/next_order_values.json`, its cases section `45ad2371…` | the frozen next-order values |
| #512, comments 18955679, 19071299 and 19076863 | the boundary; the admission and its conditions; the answers |
| `registration/t5_second_member.md`, with its check | T5's second member, DERIVED (§ T5'S SECOND MEMBER) |

## THE INSTRUMENT: UNCHANGED

`instrument/` is M8.16's, byte for byte. Each of its files carries the SHA-256 it has in the manifest that M8.16's pin `a5f45173…` covers. The freeze block lists them again, and the guard refuses a tree where any differs. The law, the basis, the quartic, the kernel, the solve, continuation and the linear record are M8.16's. Only `run/` is new: the scoring, the cases, the guard and the tests. So M8.16.1 takes its own manifest pin, freeze and outputs, in a fresh run folder. The calibration is fixed with the instrument: no constant is refitted inside M8.16.1 (this task's `BOUNDARY.md`, amendment 5).

## THE NEXT-ORDER BAR

**Where it applies:**
- K3(a)'s slow-band clause at the pyramid: the quartet and the Krein-positive pair;
- K4's pyramid item, for the same two groups;
- K5, for every predicted slow group of every row and of T5's second member;
- the C3 ray's three groups.

Elsewhere the registered bars stand:
- K3(b)'s in-span mode, within 5% of `0.25 κ τ_in`;
- K4's controls, within 5% of `εκsτ` at `ε = 0.25`, with their deviation there smaller than at `ε = 2`;
- K2's band, a detection rule.

**The frozen value.** For each predicted slow group `g` and each cutoff `N`, `c_g(N)` is Proposition N's coefficient `ℓᵀT₁r`, from one pinned script and one formula, frozen before the go.
- At `N = 24` and 32: the full formula, computed for each cutoff on its own. Order `ε²` reaches only channels both cutoffs hold (check vi), so the two values must coincide. They do, at every case and group: the pinned script's check D.
- At `N = 16`: the same formula, with exactly the channels above that cutoff removed: level 18 in `R4` and `R5`, and levels 17 and 19 in `R2`. Nothing else is removed, and nothing is chosen case by case.
- A group of multiplicity 2 or 3 carries one value. At C3's triple, the correction is a multiple of the identity on its eigenspace (§ III). At the doubles of T2 and T3, the two coefficients coincide: the pinned script's check C finds the correction a multiple of the identity on every multiple group. The reading takes such a group's `λ_g` as the mean of its matched eigenvalues, its centre.

**The reading,** at each cutoff, as Proposition N's check iv defines it. At `ε = 0.25`, 0.5 and 1, take `h_g(ε) = (λ_g(ε)/ε − κs_g)/ε`, with `λ_g` the group's centre at the point's reached amplitude and `κs_g` the frozen leading-order value, which the values file holds to 25 digits. For an imaginary pair, the member at `+iτ` is read; for a quartet, its member in the first quadrant, as claim F reads them. With `x₁, x₂, x₃` the three reached amplitudes:
- `ĉ₂` is the linear extrapolation of `h_g` to `ε = 0` through `x₁` and `x₂`;
- `ĉ₃` is the quadratic extrapolation through `x₁`, `x₂` and `x₃`, with weights `w_i`, the Lagrange weights at 0 of the reached amplitudes;
- the spread is `|ĉ₃ − ĉ₂|`.

At the nominal amplitudes the weights are (2, −1) and (8/3, −2, 1/3). The reached amplitudes stand slightly off them: at `N = 32` the pyramid reached 0.24999967 and 0.49999952 (M8.16's record).

**The checks,** for each group at each cutoff:
1. **Matched.** At each of the three points:
   - the point is accepted, with its seed check passed;
   - `V₀` is as expected;
   - the point is converged, with the expected type and indices;
   - the group is matched as in K2, one to one by class and Krein sign with multiplicity, but against each predicted group's next-order value, `εκs_g + ε²c_g(N)`, in place of its leading order. At T5's two lowest groups, whose next-order shifts at `ε = 1` are about as large as their gap at leading order, a match against the leading order exchanges them.

   This is new: M8.16's K5 scored only `ε = 0.25`, and reported 0.5 and 1 unscored. On the ladder rows, K6 already scores acceptance and agreement at those points, but not the matching. T1, T4 and T5's second member have no ladder.

   **At K3(a),** check 1 asks only that the point be accepted with its seed checked, and that the slow band match outside the in-span pair's class, against the next-order values as above (`DEVLOG.md` entry 9). K3's other clauses read the kernel and the in-span pair. So a merged in-span pair stays K3(b)'s outcome, as K3's list registers it, and is not read as the instrument's failure under (a).
2. **Resolved.** The group's numerical estimate `E(ε) = ‖P_g‖ max(‖R_g‖, u‖A‖)`, as the convergence rule computes it, is computed at all three points, with the group's projector passing the convergence rule's frozen checks: the span at 10⁻⁶, the left residual at 10⁴u, the conditioning at 10⁻²/u, and `P X = X` at 10⁻⁶. It is read from the point's summary, whose entries are associated with the record's group by exact eigenvalue-index set, each consistent with it in centre, class, size and Krein count; `E` is their largest, and a missing, duplicated or inconsistent association reads Unresolved. The coefficient's propagated numerical-error estimate, `E_c = Σ |w_i| E(x_i)/x_i²`, is at most a tenth of the group's ceiling. At the nominal amplitudes, `E_c = (128/3) E(0.25) + 8 E(0.5) + E(1)/3`. `E` is the instrument's estimate, not a certified bound. The convergence rule's full uncertainty `U` is not used here. Its floor, the cluster margin of about 4.5×10⁻⁷ at `N = 32`, would fail this test at every slow group of the pyramid.
3. **Within the ceiling.** The spread is at most `C_g(N) = max(0.1 |c_g(N)|, 10⁻³ |κs_g|)`. A spread over the ceiling fails the check; it never widens the pass.
4. **Agreement.** `|c_g(N) − ĉ₃| ≤ max(|ĉ₃ − ĉ₂|, 0.1 C_g(N))`. This is M8.16.1's own acceptance rule, not Proposition N's check v. To see why the floor is needed, truncate the expansion at fifth order: `λ = μ₀ε + cε² + dε³ + eε⁴ + fε⁵`. At the reached amplitudes, `ĉ₃ − c = f x₁x₂x₃`, while `ĉ₃ − ĉ₂ = x₁x₂ (e + f (x₁ + x₂ + x₃))`. At the nominal amplitudes these are `0.125 f` and `0.125 e + 0.21875 f`, so the spread alone can fail an exact `c_g` through cancellation, whenever `|e + 1.75 f| < |f|`. The floor keeps a group from failing on that cancellation while `|f x₁x₂x₃|` stays under it, and on noise, which check 2 holds below `C_g(N)/10`. It still tests `c_g(N)` to a tenth of its ceiling.

**A group's reading:**
- it **passes** when 1 to 4 hold at every cutoff;
- it **fails** when 3 or 4 fails at a cutoff where 1 and 2 hold, whatever it reads at the other cutoffs;
- otherwise it is **Unresolved** when 1 or 2 fails at some cutoff, and the record names the failing check and point.

A row's group that does not pass fails the row's K5. At the pyramid, any reading but a pass fails K3(a) and K4.

**T4 in `R5`,** the ruling's case. Its smallest mode has `|κs_g| = 5.27×10⁻⁵` and `c_g/(κs_g) = +0.240` at `N = 24` and 32, and +0.234 at `N = 16` (the values file).
- Its ceiling is 1.27×10⁻⁶, so check 2 needs per-point estimates below about 2.5×10⁻⁹.
- On the controls, the estimate ran from 1.1×10⁻¹³ to 1.2×10⁻¹² (M8.16's `DEVLOG.md`, entry 165; INFERRED for this mode).
- T4's values come from the formula at 50 digits at the interval-certified point. The reduced model's extrapolation, kept as a cross-check, agrees within 7.1×10⁻¹⁵, relative, at every group and cutoff (the pinned script's check X; `DEVLOG.md` entries 6 and 7).
- If check 2 fails there, the mode reads Unresolved, as any mode does, and T4/`R5` is not qualified. No other reading is registered for it.

**Where the floor binds.** Where `|c_g|` is below 1% of `|κs_g|`, the ceiling is its floor, `10⁻³ |κs_g|`. Among the rows that holds at five groups at `N = 24` and 32, and at seven at `N = 16`. It also holds at every cutoff at the pyramid's Krein-positive pair and at the C3 ray's third group. The smallest among the rows is T2/R5's double, at 1.9×10⁻³ of `|κs_g|` at `N = 24` and 32 and 1.4×10⁻³ at `N = 16`, where check 4 tests `c_g` to about 7% of its value.

**Reported, unscored,** for every group and cutoff:
- `c_g`, `ĉ₂`, `ĉ₃`, the spread, the ceiling and `E_c`;
- the two-term residual at `ε = 0.25`, `|λ_g − εκs_g − ε²c_g| / (ε|κs_g|)`, beside 1%;
- the deviation from leading order, as M8.16 reports it.

## THE CASES: WHAT IS ADDED

Both are sealed until the freeze. Their expected columns follow M8.16's derivation for the soft-slot cases (§ THE CASES there).

| ID | slot | orbit (Condon–Shortley `v_m`) | `Q(u)` | `(n₋, n₀, n₊)` | `k, r_u` | expected `V₀`: dimension; ranks of `A`, `A²` | type; `(n(L), n(D), K)` at small amplitude |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C3R | `R4` | the C3 ray, `v₂ + 2v₋₁` | 77/65 | (5, 0, 4) | 4, 2 | 6; 2, 0 | Hyperbolic: one quartet, one Krein-negative pair, one Krein-positive pair; (5, 1, 4) |
| T5J | `R2` | T5's member with complex structure `j`, `√(3/20) v_{7/2} + i √(7/20) v_{3/2} + √(7/20) v_{−3/2} − i √(3/20) v_{−7/2}` | 22/13 | (9, 0, 0) | 6, 4 | 10; 4, 0 | Elliptic; (9, 3, 6) |

**The C3 ray** is the other orbit of `R4`'s census born with a complex quartet ([§ IV](https://github.com/dmobius3/mode-identity-theory/blob/e7652a447efde0c56aee2713298d3a042bef87b4/files/framework/files/working/files/soft-slot-branches.md#iv-the-orbits)). Its value of the quartic is the pyramid's, and so is its `Q`. It is a control whose every check runs and is scored, and it gates nothing (§ THE VERDICT RULE: WHAT CHANGES). Each row checks its own kernel and band under K5, so a gate here would add a way to lose every row without adding a check.

Its checks:
- K1 and K2, at `ε ∈ {0.25, 0.5, 1, 2}` and every cutoff, as for the pyramid;
- the next-order bar, on its three groups;
- **the off-ray discrimination.** Proposition N's off-ray part is large here. At the quartet, at `N = 24` and 32, it is `+0.292 + 0.115 i` of `κs_g`, against `−0.238 − 0.025 i` for the prediction without it, `c_g°`, and a total of `+0.054 + 0.090 i` (the values file). For each group and cutoff, the prediction with the off-ray part left out, `c_g°`, must fail check 4 at `c_g`'s tolerance, `max(|ĉ₃ − ĉ₂|, 0.1 C_g(N))`, with `C_g(N)` taken from `c_g`. Each group's off-ray part exceeds check 4's floor by a factor of at least 44 at every cutoff (the values file). So a pass at the C3 ray tests the off-ray part itself, as the landing showed at the pyramid, where without it every group falls outside its spread.

**T5's second member** is § T5'S SECOND MEMBER's. It carries K5 under the next-order bar, and no K6, since M8.15 ran no ladder for it.

**The points.** Both run at `ε ∈ {0.25, 0.5, 1, 2}`, continued in that order from the seed, at `N = 16, 24` and 32, each cutoff on its own path.

## T5'S SECOND MEMBER

**The member.** T5's registered case is the member of `u = √(3/10) v_{7/2} + √(7/10) v_{−3/2}` with complex structure `i`, in `R2` at `j = 7/2` (M8.16's terms, § THE CASES). `Sp(1)` acts on the fibre by right multiplication, with `i` the complex structure and `j` acting as `Θ`, up to a phase (The Soft-Slot Branches, § I). The second member is `u₂ = u p₀`, with `p₀ = (1 + k)/√2`, so that `p₀ i p₀⁻¹ = j`. That is `u₂ = (u − iΘu)/√2`, as in § THE CASES: WHAT IS ADDED. Read with `i`, as the instrument reads it, it is the member with complex structure `j`. The census names a member this way: "the row names the member tested" (§ II).

**What is claimed.** The two stationary states lie on one `Sp(1)` orbit, as every member of T5's family does. The claim concerns the standing wave with its complex structure, which is what the instrument linearizes.

**Proposition (DERIVED).** Let `Γ` be the frozen action's symmetry group: right `SU(2)` times `Sp(1)`, with time translations and time reversal. Every complete-level truncation keeps it exactly (the census, § V, step 3, condition (a)).
- **(a)** No element of `Γ` carries `u₂`'s standing wave to one of the registered member's, at the seed or along Lemma G's germ.
- **(b)** For sufficiently small positive `ε`, the two members' linearizations, formed with `i`, have disjoint nonzero slow spectra; both keep their exact structural zeros. So no linear map conjugates one into the other, including a symmetry of a truncation outside `Γ`.

*Proof, in outline.* An element carrying one `i`-standing wave to another has a fibre part `q` with `q⁻¹ i q = ±i`, since right multiplication by a quaternion is injective on a nonzero fibre vector. So it conjugates the two linearizations, up to sign. By Theorem R(a) and § III's "Families in `R₂`", the slow matrix of `u₂` read with `i` is similar to `Θ M̂_u`. The two slow polynomials, exact in `ν = s²`:
- the registered member's, `(20449ν + 153664)(418161601ν² + 6989958976ν + 28677390336)`, the terms' frozen T5 spectrum, with `τ` = 2.6886, 2.7413 and 3.0802;
- the second member's, `(1859ν + 15680)(418161601ν² + 6605190592ν + 25491013632)`, with `τ` = 2.5899, 2.9042 and 3.0147.

Their resultant is nonzero. That gives (b), and with it (a) at the seed. Lemma G's germ is `Γ`-equivariant, which carries (a) along the germ, wherever its isotropy group is the seed's. The full proof is `registration/t5_second_member.md`.

**A second, independent certificate.** Right `SU(2)` acts on a spin-7/2 state's Majorana polynomial by Möbius maps. A phase leaves the polynomial unchanged, and `Θ` sends its roots to their antipodes. Each preserves the multiplicities of the roots.
- The registered member's polynomial is `z⁷ − 7z²`, with a double root at 0.
- The second member's, `z⁷ + 7i z⁵ − 7z² + i`, has seven simple roots: its discriminant is nonzero.

So no element of the group that preserves `i` up to sign, which by the proof's first step is every element carrying one `i`-standing wave to another, carries one state to the other (DERIVED).

**The checks,** exact, ran before the freeze, and land with the terms in `registration/`:
- `t5_member_check.py`: both members' seed values, the polynomial identities, the resultant, and one mutation arm, `u` in place of `u₂`, which turns its claim red (`DEVLOG.md` entry 27);
- the Majorana root multiplicities: seven simple roots at T5J against T5's double root, in exact arithmetic on the seeds' own components (the seed check's claim 7, entry 19).

**What it does not give:**
- anything where a branch's isotropy group grows beyond the seed's, or past Lemma G's range;
- which equatorial member to take. The leading slow spectrum is the same at `j` and `k` exactly. The member's next-order coefficients come from the pinned script at this seed.

**Its outcome is its own.** It is scored for that member alone. A failure records it as not qualified, with its failing check named, and stops no other row or member. It neither carries the verdict on the member with complex structure `i`, nor takes it.

## THE VERDICT RULE: WHAT CHANGES

- **The instrument qualifies** if K1 to K4 pass on C1 to C3 and the pyramid, as registered, with K3(a)'s slow-band clause and K4's pyramid item read by the next-order bar. The C3 ray gates nothing.
- **A row qualifies,** for the census run, if the instrument qualifies and the row passes K5, read by the next-order bar, and K6 where it has a ladder. An `R5` row also needs C2a/R5's K1 and K4. T5's second member qualifies as a member, on its own K5.
- **The verdict:** QUALIFIED, with the rows and members it qualifies named; or NOT QUALIFIED, with the failing check named. At a stop, the verdict is NOT QUALIFIED.
- **K3's outcomes, written out** (M8.16's dated note of 2026-10-19). When (a) passes and (b), (c) or (d) fails, the kernel is exact and the in-span mode is not resolved: the diagnosis is wrong, and the record says so. When (a) fails, the outcome is the instrument's, and the diagnosis is untested. K3's list governs.
- **The diagnosis at the rows,** reported, unscored.
  - At T1, the finite-element instrument's failure came at the smallest amplitude. At T4, M8.15 reached no point, through a defect in Newton's stopping test, and the diagnosis predicts the same merger there (the census, § IV, INFERRED). An exact kernel and a matched slow band at those rows, under K5, are what the diagnosis predicts. The record reports them as evidence on the diagnosis there, for it or against it, and claims nothing beyond them.
  - At the prism in `R4`, the failure came near its crossing, which this task does not read. That is the census run's.
  - Where a row fails, the record names its failing check as the cause.
- **The stop** (the ruling, § 3):
  - M8.16.1 is the instrument's last qualification, under any ID, for the instrument and for each row.
  - A row it does not qualify, or a member a row names that it does not carry, is not offered to this instrument again.
  - A revised harmonic instrument for the rows files only under a ruling on #512.
  - For S6, an instrument that can no longer qualify, overall or for a row, is not one that removes a cause. What S6 then does with each row is the census's to apply, and an instrument-limited row claims no type.
- **A failure stands as registered.** There is no repair-and-rerun under the M8.16.1 identity or any other. A claim that frozen text is defective goes to maintainer reproduction. A ratified defect is recorded in the record, and does not by itself open a third qualification.

## STATED LIMITS: WHAT IS ADDED

**`R5`'s ladder** (amendment 5; `DEVLOG.md` entries 24 to 26). C2a/R5 ran along M8.15's grid to `ε = 32`, at `N = 16, 24` and 32, on 2026-10-22.
- **What held.** All 45 points were accepted. K1 held throughout, and every record read Elliptic, (0, 0, 0), with nothing unresolved. Through the sample's range, `ε ≤ 2`, K2 and the convergence rule held. The sample reproduced the registered run's, exactly at `ε = 0.25`, where both runs take the same first step from the germ.
- **Two failures at `ε = 32`,** both preserved, with no refit and no change to the instrument:
  - the convergence rule read Unresolved. A Krein-positive pair, 0.029 apart at `N = 24` and 32 and agreeing there to 1.3×10⁻⁶, could not be matched one to one to `N = 16`, whose values for it are off by more than its spacing;
  - the calibration's condition (a) failed at `N = 16`. The zero cluster lay 8.3 times below its gray zone's lower edge, against the condition's 10. It stayed separated, and K1 held.
- **The reading:** both are `N = 16`'s resolution limit at the grid's top (INFERRED). `N = 16`'s largest departure from `N = 32` on the tested groups grows steadily with `ε`, to 4.4×10⁻² at 32, where `N = 24`'s is 1.3×10⁻⁶. Why the cluster grows at `N = 16` is OPEN.
- **Condition (a) in `R5`.** At `N = 16` the cluster coefficient exceeds the controls' largest, 0.0568, on which the margin was fitted (M8.16's STATED LIMITS), at `ε` = 5.66, 22.6 and 32. There the cluster lies 11.6, 11.0 and 8.3 times below its gray zone's lower edge. At `N = 24` and 32 it lies at least 16 times below, along the whole ladder.

**For T2 and T3 in `R5`,** which run along this grid to `ε = 32` in the registered run:
- their grids, cutoffs and K6 are unchanged, and these failures do not pre-adjudicate them;
- an Unresolved reading at `ε = 32` is expected there (INFERRED). It is not a disagreement with M8.15, and it fails no row: K6 scores acceptance at every point, no disagreement, and agreement through `ε = 2`. It leaves that point unclassified.

**Qualification does not establish finite-amplitude adequacy** at the top of an `R5` ladder. A point the convergence rule leaves Unresolved is not labelled with the type each cutoff reads. The census run's adequacy tests still apply there (S4). A cutoff other than 16, 24 and 32 belongs to the census run's own pre-registration, with the structural kernel first measured on its controls at that cutoff, against K1 as frozen here, and the calibration unchanged ([19076863](https://github.com/openwave-labs/openwave/discussions/512#discussioncomment-19076863), § 2).

## THE TESTS BEFORE THE FREEZE

M8.16's frozen text carried three defects, each recorded after the run:
- the verdict rule's K3 sentence (#640, § 2);
- `FROZEN.json`'s two omitted values (`RECORD.md` § 5);
- the freeze block's seeds line (§ 6).

Under the stop, a rerun cannot repair a defect. So each class gets a scripted test, run before the freeze, each with a mutation arm that must turn it red:
1. **Every verdict path.** For each scored check, a synthetic record for each registered outcome, asserting the verdict text this document gives it:
   - K1 to K6, with K3's three outcomes;
   - the next-order bar's pass, Unresolved (check 1 or 2) and failure (check 3 or 4), with failure's precedence, and a group whose spread is tiny and whose difference lies between the spread and check 4's floor, which must pass;
   - two groups of one class and sign whose next-order shifts match their gap, on a spectrum exactly at the next order, as at T5's two lowest, which must pass;
   - the `R5` gate, and the C3 ray's gating nothing;
   - T5's second member's independence;
   - the stops, and the resume after an interruption;
   - cutoff matching.
2. **Every constant.** Every numeric item in both FROZEN VALUES sections is classified as a code-read literal held by `FROZEN.json`, a derived value reproduced by its registered calculation, or an explicitly unscored figure on an exclusion list with its reason. The reverse scan accounts for every effective scoring constant, including those brought forward from M8.16, and rejects an unregistered scoring literal.
3. **Every import.** Each file the seed check and the driver import is in the manifest and named in the freeze block.

**They ran in development** (`DEVLOG.md` entries 14 to 16, 23, 28 to 30 and 40), each with its arms red:
- the verdict paths: 57 checks, with six mutations of the verdict code;
- the constants, as test 2 states them, with the reverse scan's lists: registered, case data, a formula's own constant, a lookup or float guard, an unscored report, and a load-time check;
- the imports: the driver, `write_guard.py`, `freeze_block.py` and the seed check load only manifest files, each named in the freeze block, and the block's seeds line names exactly the files the seed check loaded, its child process's included.

They run again on the filed text before the go.

## FROZEN VALUES: WHAT IS NEW

**The new cases' slow spectra,** for unit `u` in `R4`'s `h` normalization, as the pinned script freezes them (its check A):
- C3R: a quartet at `τ = 0.0355667254 + 0.00986952877 i`; `τ = 0.453830235`, Krein −; `τ = 1.00180626`, Krein +;
- T5J: `τ² = 15680/1859`, and `τ² = −ν` at the two roots of `418161601ν² + 6605190592ν + 25491013632`, so `τ ≈` 2.5899, 2.9042 and 3.0147; all Krein −.

**Their smallest predicted modes** at `ε = 0.25`, as in M8.16's table (INFERRED, with `‖A‖ ≈ N(N+2)`):
- C3R at 1.41×10⁻⁴, about 310 margins outside the gray zone at `N = 32`, since its `Q` is the pyramid's;
- T5J at 6.03×10⁻³, about 13,700, with T5's `Q`.

**The next-order coefficients** `c_g(N)`, for every case, group and cutoff, come from the pinned script, `run/next_order_values.py`. It is Proposition N's landed check, reduced to the formula. Its checks: check i at every case, exact where the representative is algebraic and at 50 digits at T4, whose representative is the interval-certified point; the multiple groups' identity check; and check vi's coincidence at 24 and 32. Its output, `run/next_order_values.json` (`DEVLOG.md` entry 7), with the C3 ray's `c_g°(N)` beside its `c_g(N)` at every cutoff, is in `FROZEN.json` and in the freeze block.
- `FROZEN.json` pins its cases section by SHA-256, `45ad2371…`, which does not depend on the environment, with the environment beside it.
- The freeze checks its inventory of cases, cutoffs and groups against this document's cases.
- The loader refuses a file whose groups change order across the cutoffs, or whose bar membership differs from the registered rule: at the pyramid, the quartet and the Krein-positive pair; everywhere else, every predicted slow group.

The values, as `c_g/(κs_g)`, for every group under the bar, by ascending modulus of `s_g`, at `N = 24` and 32, where they coincide, and at `N = 16`. At the pyramid, those are its quartet and its Krein-positive pair:

| case | `N = 24` and 32: `c_g/(κs_g)` | `N = 16` |
| --- | --- | --- |
| P5 | −0.109831 − 0.20915 i; −0.00792389 | −0.112895 − 0.206203 i; −0.00460195 |
| T1/R4 | −0.0306976; −0.00416477; −0.0201379; −0.0330517; −0.0166264 | −0.0293849; −0.0026187; −0.018164; −0.0315831; −0.0149448 |
| T1/R5 | −0.0336794; −0.0426299; −0.0349632; −0.00780072; −0.0207945 | −0.0323578; −0.0410768; −0.0329856; −0.00632444; −0.0191069 |
| T2/R4 | −0.0532431 (double); −0.0172923 | −0.0502696 (double); −0.0149243 |
| T2/R5 | −0.00185628 (double); −0.0112677 | +0.00135837 (double); −0.00868442 |
| T3/R4 | +0.004905 (double); −0.0113879 | +0.00840142 (double); −0.008384 |
| T3/R5 | −0.0530714 (double); −0.0303976 | −0.0488139 (double); −0.0266862 |
| T4/R4 | −0.113972; +0.0151491; −0.0678847; −0.00801725 | −0.120026; +0.0175212; −0.0654372; −0.00495022 |
| T4/R5 | +0.23999; −0.0616253; +0.0143517; −0.0231059 | +0.233765; −0.058983; +0.0170735; −0.0197323 |
| T5/R2 | −0.0206729; −0.0197174; −0.017373 | −0.0202035; −0.0192194; −0.0168049 |
| T5J | −0.0214702; −0.018581; −0.0177909 | −0.0210247; −0.0180491; −0.0172354 |
| C3R | +0.0540938 + 0.0900495 i; −0.0189471; −0.00449803 | +0.055299 + 0.0902736 i; −0.0157961; −0.00136602 |

**New constants:**
- the ceiling's factors, 0.1 and 10⁻³;
- the resolution test's tenth, with its weights 128/3, 8 and 1/3;
- check 4's floor, a tenth of the ceiling;
- the extrapolation's points, 0.25, 0.5 and 1.

**Brought into FROZEN VALUES,** each stated in M8.16's terms outside that section and read by the scoring code (the constant test's reverse scan):
- K2's slow band, twice the largest predicted mode;
- the band-edge guard's half;
- K3's orthonormal cut, 10⁻⁶;
- K6's agreement at every read point with `ε ≤ 2`;
- C1's bounds: the residual at 10⁻¹⁵, the off-level part at 10⁻¹⁴, the spectrum at 10⁻¹⁰, and the readings unchanged in `N` to 10⁻¹².

**Every item's kind** is one of three: a literal the code reads and `FROZEN.json` holds; a value its calculation reproduces; or an unscored report (§ THE TESTS BEFORE THE FREEZE).

**Unchanged:** M8.16's FROZEN VALUES. These include the 5% bar where it still applies, the overlaps, the guards, the calibration, the solve, the grid and the cutoffs.

## THE FREEZE, THE GUARD AND THE RUN

**The freeze,** as M8.16's, with new pins:
- `pin_manifest.py` writes `run/MANIFEST_PIN`, over `instrument/`, byte for byte M8.16's, `run/` and `reference/m815`.
- The filed terms are copied to `run/m8_16_task_details.md`, the name by which the instrument's `devtools.frozen()` reads them, and `FROZEN.json` pins their SHA-256.
- The order is M8.16's: the pin is published on MIT `main` with the solver commit; the maintainer's go; `write_guard.py` writes `run/FROZEN.json` and `run/FROZEN_PIN`; the freeze block; the run.
- `FROZEN.json` holds:
  - the seventeen cases, M8.16's fifteen with C3R and T5J, each with its slot, representative, `Q`, `(k, r_u)`, expected `V₀`, type, indices and frozen spectrum;
  - every frozen value, M8.16's and the new, with the next-order coefficients by case, group and cutoff;
  - the terms' SHA-256, the go and the manifest's SHA-256.
- `run/freeze_block.py` writes the freeze block. It lists every file in the manifest with its SHA-256, both pins, the terms' SHA-256, the environment, every seed with its checks from the seed check's record, each file the seed check loaded with its SHA-256, and `FROZEN.json` in full.

**The guard** refuses as M8.16's does. It also refuses the C3 ray, T5J or any row before the gate record shows K1 to K4 passed. Its tests, M8.16's carried and extended, pass 126 of 126 (`DEVLOG.md` entry 23).

**The run's order:**
1. the freeze;
2. the controls: C1, C2a, C2b, C3, then C2a/R5. A failed check of C1 to C3 stops the run. C2a/R5's failure records the `R5` rows as not qualified, and stops nothing else;
3. the pyramid. A failed check stops the run. The driver then writes the gate record, with K1 to K4;
4. the C3 ray, scored, stopping nothing;
5. the rows and members: the samples of T1 and T4 in both slots, and of T5J; the ladders of T2 and T3 in both slots, and of T5.

**The cost** (INFERRED), on the recorded machine: M8.16's estimate with its rows, 13.6 to 17.3 h, plus about 0.5 h for the C3 ray and 0.2 h for T5J. Amendment 5's ladder, a development run, took 1.85 h (`DEVLOG.md` entry 24).

## DEFINITION OF DONE

As M8.16's, with these additions:
- **the record:** every group's next-order reading at every cutoff, with each Unresolved group's failing check and point named, and the C3 ray's discrimination;
- **the roadmap:** both M8 status rows record the filing, and the Done row states the adjudicated verdict;
- **after the verdict,** a dated note on the census page;
- **what follows:** the census run, under its own pre-registration, runs only the rows and members qualified here.
