# M10.3 maintainer review: the free pair recomputed, the pin adopted, row 4 at partial

> Task: [M10.3](../tasks/m10_3_task_details.md). The author's record is the
> [method note](m10_3_method_note.md) and the
> [reproduction record](../data/m10_3_reproduction.json). This note is the
> maintainers' side: an independent recomputation of the free two-load sector of the
> pinned law, and the two decisions of the review, the pin and row 4, with what would
> move the row. The review as posted is on
> [pull request #745](https://github.com/openwave-labs/openwave/pull/745), merged
> 2026-11-21.
>
> **Verdict.** Every number recomputed agrees with the author's, by a route that
> shares no code with the author's. The re-pin is adopted: the column is scored on
> Interval Echo at `4fe10f9a`, and the M10.2 candidate is kept as ancestry. Row 4
> moves to ⚠️ partial, not ✅. The pair's persistence comes out of the generator the
> author's spec states for the pinned law, and the law with the coupling removed
> loses it. One value of the law, the coupling, is a choice that answers to this row,
> and the published bar puts that case at ⚠️.
>
> **Scope**: an independent recomputation of the free two-load sector of the
> generator the candidate's spec states, on the 20-cell carrier, and the record of
> the review's two decisions; it licenses "the recomputed numbers are correct for
> that generator", and does not license the reduction from the seam law, a particle
> identification, a statement about the echo and memory law, or any other row. The
> ⚠️ is a decision from the published bar, not a result of the recompute.

## 1. The equations

The generator is the one the author's spec states for the pinned law on loads 0 and
1. It acts on two unit loads at distinct ports $a \lt b$ of the 240-port carrier,
28,680 configurations, with $e$ running over the 630 seams:

$$
H(\kappa) = \tfrac12 \sum_e (I - \mathrm{SWAP}_e) - 2\kappa \sum_e n_a n_b
$$

$\mathrm{SWAP}_e$ exchanges the occupations of the two ends of seam $e$, and
$n_a n_b$ is 1 when both loads sit on them. Write $K = \tfrac12 \sum_e (I - \mathrm{SWAP}_e)$
and $P$ for the projector on the 630 adjacent configurations, so $H = K - 2\kappa P$.

**The evolution.** The two loads start on the launch seam $(156, 157)$ in cell 13,
the state $\psi_0$. The route propagates with a Chebyshev expansion. With
$\tilde H = (H - c)/\rho$ mapping the spectrum into $[-1, 1]$ (the script leaves out the phase $e^{-ict}$,
which changes no probability):

$$
\psi(t) = e^{-itH} \psi_0 = e^{-ict} \sum_{k \ge 0} a_k (-i)^k J_k(\rho t) T_k(\tilde H) \psi_0, \qquad a_0 = 1, \quad a_k = 2 \quad (k \ge 1)
$$

**The observables.** The home cell is the launch cell, with its 30 internal seams:

$$
P_{\mathrm{home}}(t) = \sum_{s \in \mathrm{home}} \lvert \langle s \vert \psi(t) \rangle \rvert^2 \qquad P_{\mathrm{adj}}(t) = \sum_{s \in \mathrm{seams}} \lvert \langle s \vert \psi(t) \rangle \rvert^2
$$

$P_{\mathrm{adj}}$ is "adjacent anywhere": both loads on one seam, in any cell. The
energy spread of the launch state is

$$
\Delta H^2 = \langle \psi_0 \vert H^2 \vert \psi_0 \rangle - \langle \psi_0 \vert H \vert \psi_0 \rangle^2
$$

**The window bound.** The angle to the home subspace, $\arccos \sqrt{P_{\mathrm{home}}}$,
moves at most $\Delta H$ per unit exposure, and no time in $[0, 37]$ is further than
1/16 from the 1/8 grid, so for every $t$ in the window

$$
P_{\mathrm{home}}(t) \ge \cos^2 \Big( \arccos \sqrt{\min_{\mathrm{grid}} P_{\mathrm{home}} - 10^{-6}} + \Delta H / 16 \Big)
$$

**The pair records.** At each of exposures 1, 7 and 37, and for $\kappa = 8$ and
$\kappa = 0$, 4,096 categorical draws from the 31-outcome law (the 30 home seams and
quiet), with numpy's `default_rng([917421, arm, j])`, the author's seeds. A record is
a draw on a home seam. The author's simultaneous one-sided Hoeffding radius and the
common lower bound:

$$
\epsilon = \sqrt{\frac{\ln(8/0.01)}{2 \cdot 4096}} \qquad \mathrm{bound} = \frac{\min N_{\kappa = 8}}{4096} - \epsilon
$$

**Arming.** Before any comparison: the norm at exposure 37 within $10^{-10}$, the
energy conserved to $10^{-10}$ over $[0, 37]$, and the route's probabilities equal to
those of `scipy.sparse.linalg.expm_multiply` at exposure 1 within $10^{-12}$.

**What is reported.** The carrier and the launch seam; $P_{\mathrm{home}}$ at
exposures 1, 7 and 37 for $\kappa = 8$ and its minimum on the 1/8 grid; $\Delta H^2$;
$P_{\mathrm{home}}$ of cell 0 at exposure 150 for the older launch on $(0, 1)$; the
pair records, the radius, the bound and the window bound; with the author's receipt,
its 186 outcome probabilities and its counts. Then, not compared with a claim, the
coupling table and the longer times.

## 2. Equation-to-code map

One file,
[`m10_3_maintainer_recompute.py`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py).
Its docstring carries the same equations.

| Object | Function | Lines |
| --- | --- | --- |
| The author's numbers | `CLAIMED` | [77-90](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L77-L90) |
| The board, read as arrays after its SHA-256 | `load_board` | [97-107](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L97-L107) |
| The receipt on stdin, after its SHA-256 | `read_receipt` | [110-116](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L110-L116) |
| Ports, seams, degrees, the cells' joins | `structure` | [119-152](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L119-L152) |
| The configurations, $K$ and $P$ | `build` | [158-178](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L158-L178) |
| $H = K - 2\kappa P$ | `generator` | [181-183](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L181-L183) |
| The Chebyshev propagation | `chebyshev` | [186-226](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L186-L226) |
| $P_{\mathrm{home}}$ and $P_{\mathrm{adj}}$ | `probabilities` | [229-232](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L229-L232) |
| The home seams of a cell, the launch state | `Carrier` | [238-257](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L238-L257) |
| Norm, energy, $\Delta H^2$, the second algorithm | `arming` | [260-275](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L260-L275) |
| The free pair on the 1/8 grid over $[0, 37]$ | `free_pair` | [278-283](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L278-L283) |
| Cell 0 at exposure 150 | `cell0` | [286-291](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L286-L291) |
| The 31-outcome law and the seeded draws | `outcome_law`, `pair_records` | [294-309](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L294-L309) |
| The Hoeffding radius and the common bound | `hoeffding` | [312-315](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L312-L315) |
| The window bound | `window_bound` | [318-321](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L318-L321) |
| The receipt against this route | `receipt_comparison` | [324-343](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L324-L343) |
| The coupling table at exposure 37 (reported) | `coupling_table` | [346-353](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L346-L353) |
| The longer times at $\kappa = 8$ (reported) | `long_times` | [356-368](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L356-L368) |
| The arming checks, the comparisons, their wrong-value arms | `arming_checks`, `verdicts`, `arms` | [374-458](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py#L374-L458) |

## 3. Reproduction

The script reads one data file of the author's, the carrier's seam list
`openwave/interval_echo/row_04/FEDERATION_LEVEL0.npz` of the author package at commit
`4fe10f9ac3e81423155fb3ac7baaf0fdb4c0b11c` (tag `interval-echo-m10.3-source`), as
arrays (`allow_pickle=False`). Its SHA-256 is
`99870398e3d516f899c600e9bf25ba53c7f513142ba8facbe1cedd94b23f915b`, the value in the
author's `check_inputs.py`, and the script checks it before computing anything. With
`--receipt-stdin` it also reads `row_04_receipt.json` from the same folder at the same
commit, SHA-256 `7f6e458fd0b7cf9b9e50fb15a5c4b77f944a2457a9c53cdcca88efdf4d4c3bdd`. It
imports nothing from the author's package and runs none of it. A missing or different
board, or a different receipt, stops it with a nonzero exit. Without
`--receipt-stdin` the four receipt comparisons are not run, and the printed count
says so.

```sh
python3 -I openwave/xperiments/m10_oph/research/scripts/m10_3_maintainer_recompute.py \
  /path/to/FEDERATION_LEVEL0.npz --receipt-stdin --output /path/to/out.json \
  < /path/to/row_04_receipt.json
```

The graph file is not vendored: the author's record stays in the author's
repository, and the pin is the commit plus the hash. The run is recorded in
[`m10_3_maintainer_receipt.json`](../data/m10_3_maintainer_receipt.json).

## 4. Results

Stored in [`m10_3_maintainer_recompute.json`](../data/m10_3_maintainer_recompute.json).
Each line is compared with the author's value; each comparison turns false when it
is fed a wrong value (eighteen of eighteen, the three arming checks included).

| Claim | Result | How compared |
| --- | --- | --- |
| Carrier: 240 ports, 630 seams | 240 ports, 630 seams: 20 identical cells, 30 seams between cells forming a matching, three to a cell | ports and seams exact; the cell structure reported |
| Launch seam in cell 13: (156, 157) | (156, 157), degrees 5 and 6 | the seam exact; the degrees reported |
| Home-cell pair probability at exposures 1, 7, 37, κ = 8: 0.990982505, 0.989653268, 0.980271948 | the same nine digits | within 5e-10 plus 1e-11 |
| Minimum on the 1/8 grid over [0, 37]: 0.978883326 at 34.0 | 0.978883326 at 34.0 | within 5e-10 plus 1e-11; exact position |
| Energy variance of the launch state: 2.25 | 2.25 | within 1e-12 |
| Cell 0 at exposure 150, κ = 8 and κ = 0: 0.983880525, 0.149558605 | 0.983880525, 0.149558605 | within 5e-10 plus 1e-11 |
| Pair records of 4,096, κ = 8: 4,048, 4,052, 4,028 | the same, drawn with the author's seeds from the recomputed probabilities | exact |
| Pair records of 4,096, κ = 0: 1,780, 1,182, 824 | the same | exact |
| The receipt's source and its 30 addresses | the launch seam and the 30 home seams | exact |
| The 186 outcome probabilities of the receipt, nine digits each | largest difference 4.9e-10 | within 5e-10 plus 1e-11 |
| The receipt's record counts | equal to the recomputed draws, all six | exact |
| Hoeffding radius, about 0.02857 | 0.028566 | within 5e-6 |
| The common lower bound, above 95.48% | 4,028/4,096 less the radius is 0.95483 | above 0.9548 |
| The window floor, 94% | 0.9437 | at or above 0.94 |

Armed before any comparison: the norm at exposure 37 is off by 5e-14, the energy is
conserved to 5e-13, and the route agrees with `expm_multiply` at exposure 1 to 1e-14.

The 4,096 records per setting are seeded draws from the computed outcome law, as the
author's row-04 design says. Since the probabilities are computed exactly, 0.980
against 0.203, the counts and the Hoeffding gates check the sampler and not the law.

**What the result depends on.** Reported from the same script, not compared with a
claim. The rows at κ of 1 and below are the ones the decision uses.

| κ | Home cell at exposure 37 | Adjacent anywhere at exposure 37 |
| --- | --- | --- |
| 0 | 0.203 | 0.236 |
| 0.5 | 0.417 | 0.442 |
| 1 | 0.467 | 0.508 |
| 2 | 0.845 | 0.867 |
| 4 | 0.955 | 0.975 |
| 8 | 0.980 | 0.991 |
| 16 | 0.997 | 0.998 |
| −8 | 0.984 | 0.989 |

- **The reversed sign does not separate persistence.** At κ = −8 the home-cell
  probability at exposure 37 is 0.984. The author's reference spec's § 7 counts
  against the claim a changed law that gives the same answer "on the whole property",
  and the spectral part of the property does change: with the sign reversed the
  generator on distinct ports is $K + 16P$, which is positive, so no level there lies
  below two free loads at rest. In the full two-load sector of the spec's § 5 the
  lowest level is a doubled load, which the reversed sign lowers by 12κ. Either way
  the lowest level of two loads is not a pair on a seam, which is the spectral part
  of the property. The review asked the author which of the two the row claims.
- **A bound at every exposure.** The author's reference theorem bounds the
  probability of no pair on a seam at every exposure, by $40/(2\kappa - 6)^2$, which
  is 0.40 at κ = 8. It was not checked, and it does not enter the decision.
- **Past the window the pair stays bound and slowly leaves the cell.** Adjacency
  anywhere stays at or above 0.990 through exposure 2,000 (minimum 0.9904 on a grid
  of step 4). The home-cell probability is 0.969 at 300, 0.929 at 1,000 and 0.910 at
  2,000. The one-cell read measures binding and staying in the cell together, and
  binding alone is the larger of the two.
- **Context, from the author's reference spec.** It states that on loads 0 and 1
  the seam term is the ferromagnetic XXZ interaction, here with anisotropy
  $1 + 2\kappa = 17$, that the binding is of the droplet kind known from that
  literature, and that the construction claims no new interaction.

## 5. The two decisions

From the review as posted.

**The pin.** Adopted. The canonical says what changed and keeps the ancestry, which
is what term 6 asks. From the close-out on, the three M10.2 cells describe
`c330c6e7` as ancestry, as the other rows describe the admission construction, until
a task brings each row onto the new candidate. No icon changes from the re-pin
itself.

**Row 4.** ⚠️ partial. The bar, as [`MODELS.md`](../../../../../MODELS.md#summary-status)
states it for every column:

> **✅ validated**: the property comes out of that process and meets the row's test. **⚠️ partial**: it comes out with something still supplied (a value the model is calibrated to earns nothing by itself; a state the process returns once that value is set can), or with part of the row open.

and in the words of the
[admission reply](https://github.com/openwave-labs/openwave/discussions/712#discussioncomment-19595538),
which the canonical's § 8 records:

> What moves a row is its observable computed on the carrier's own process in-platform. With something still supplied, that is a case for ⚠️ with the caveat in the cell. With the source derived from the carrier, it meets the #482 bar for ✅.

Read against this result:

1. **The persistence is computed in-platform on the two-load generator the author's
   spec states for the pinned law.** The reduction from the 99-state seam law to that
   generator is the check of the author's `law.py` and `native_check.py`, which the
   maintainers did not rerun.
2. **The law with the coupling removed loses the pair.** With the same alphabet,
   routing and symmetry and κ = 0, the home-cell probability at exposure 37 is
   0.203, against 0.980. This is the control the
   [#723](https://github.com/openwave-labs/openwave/pull/723) reply named, and the
   row passes it. The reversed sign does not separate persistence (§ 4).
3. **One value of the law is a choice, and it answers to this row.** The #723 reply:
   "Where a choice answers to a target, the row is a calibration of that choice."
   The author's reference spec tests κ from 0 to 128 with 8 and 128 as headline
   values, and states its binding theorem for κ above 3. The structure admits the
   other values, so by the
   [parameter-count test](../../../../../ONBOARDING_MODELS.md#the-parameter-count-test)'s
   step 2 the value is a choice ("Count a choice as free if the model's own framework
   admits an inequivalent alternative"). At κ of 1 or below the home-cell probability
   at exposure 37 is 0.47 or less (§ 4), against 0.98 at κ = 8, so the row's outcome
   turns on the value. The author's row-04 design records that an earlier study fixed
   the strength at 8 and that the binding result was known before this one. Being
   declared does not by itself make the coupling a declared source for this row (the
   #723 reply, reading 1), which is why the row is not at 🚧. What makes it a supplied
   piece here is that it answers to this row's target, and the canonical § 1.2 also
   lists it as supplied: "Coupling 8 and dimensionless exposures are supplied." The
   pair is a state the process returns once that value is set, which the bar puts at
   ⚠️. The review's request asks for the record that would show the value fixed
   independently of this row.

The adjacent launch, the blank memories, the uniform exposure and the read window are
the experiment's design. They go in the cell's text, and they are not what holds the
row at ⚠️.

The same row in the columns that hold a state, as the cells stood at the review:

| Column | Cell | What it rests on |
| --- | --- | --- |
| M5 | ✅ | the electron of its verified-L stack persists in free dynamics, started from a mid-descent snapshot, over one window on one box |
| M7 | ⚠️ | a harmonic-frame soliton with a measured Derrick minimum, whose real-time vacuum is tachyonic, so persistence fails in about two periods |
| M4 | ⚠️ | a lock-in, from the Wolff-LaFreniere (M3) runs this column builds on, that holds at perfect placement and is fragile under perturbation |
| M6 | 🚧 | a localized state that has not been evolved in time |

What separates this row from M5's is item 3, on the record as it stands. The re-read
of M5 under this bar on 2026-11-18
([T11](../../../../../dev_docs/tasks/t11_task_details.md)) did not run the
parameter-count test on that cell, and its cited notes record no value of the stack
that the row turns on. Here the author's design and canonical name one, and the
coupling table shows the row turning on it. The same reading moves M5's cell if such a
value is shown there.

**What would move it to ✅.** In the reply's words, nothing supplied for the row. In
this candidate's terms: the coupling fixed by something independent of this row's
target, which is the parameter-count test's step 5 ("Genuine forcing means: fixed by
something independent of the target being reproduced"). The two records the #723
reply named are what show it: what was known beforehand for this row, and the laws set
aside (for this row, the values of κ among them). No icon is promised here; the cell
is decided from the artifact.

The cell also says, in the author's words, that the pair's charge, spin and species
are unassigned. That is a caveat in the text, not a condition.

## 6. Not computed, not run

- The author's package was not run. The author's three scripts in the pull request
  were read in full and not executed; the route above shares no code with them.
- The reduction from the 99-state seam law to the load generator above (the author's
  `native_check.py` and `law.py`), and with it everything that uses the echo and the
  memory. The run starts from blank memories, and the generator used here does not
  act on them.
- The author's reference theorem and its bound. The 80-cell carrier. The scorecard's
  seven checks. The clock, currents and consensus parts of the Interval Echo spec.
  The simulator and the animation. The evidence archive and its audit. The held
  release.

## 7. Minimal inspection set

1. [`m10_3_maintainer_recompute.py`](../scripts/m10_3_maintainer_recompute.py), the docstring first.
2. [`m10_3_maintainer_recompute.json`](../data/m10_3_maintainer_recompute.json).
3. The author's [method note](m10_3_method_note.md), for the experiment as the author states it.

## 8. Audit

The review as posted was checked by a second model before it went out, in two
parts. A recomputation by a script of its own, with a different propagation,
confirmed every number and raised two wording points: one was corrected, and the
other, a quotation, was confirmed against its source. A judgment audit in three
passes found nineteen points. The largest: the first draft grounded the partial mark
in reasons that hold for every column (the dependence on a coupling, a prepared
start state, one window), and it offered as a way forward a state the candidate's
free evolution cannot reach. The posted text grounds the mark in the published
sentences it quotes.

This note, its script and the cells were then audited again by a second model
before they landed. A numbers pass reran the script and compared its output leaf by
leaf, recomputed the tables by a route of its own, and ran 42 mutations of the
computation. Each turned a verdict false or stopped the run, except two that cannot
change a result: the 1e-6 margin inside the window bound, and the minimum against
the maximum of the records inside a one-sided bound, which the exact comparison of
the records already covers. The arms in the script test the comparisons, not the
computation. A record pass compared § 5 with the posted review sentence by sentence
and found no difference of substance. It found seven points elsewhere in the
close-out, among them a way-forward line in the cell that had drifted from the
review's wording. All were corrected before landing, except three lines of the
author's pages that the M10.2 decision had already overtaken, which the pull
request lists for the author.
