# M10.4 maintainer review: the connected release recomputed, the pin moved, row 4 at partial

> Task: [M10.4](../tasks/m10_4_task_details.md). The authors' record is the
> [method note](m10_4_method_note.md) and the
> [run summary](../data/m10_4_reproduction.json). This note is the maintainers' side:
> an independent recomputation of the connected two-load release on the written and
> the blank board, one exact check of the seam law, and the two decisions of the
> review, the pin and row 4, with what would move the row. The review as posted is on
> [pull request #749](https://github.com/openwave-labs/openwave/pull/749#pullrequestreview-6060088147),
> merged 2026-11-25. It was filed by the column's co-author on the author's behalf
> ([discussion #712](https://github.com/openwave-labs/openwave/discussions/712#discussioncomment-19859247)).
>
> **Verdict.** Every number recomputed agrees with the authors', by a route that shares
> no code with theirs. The pin moves to Interval Echo at `09f062f7`, whose law and
> board are the same bytes as at `4fe10f9a`. Row 4 stays at ⚠️ partial. The
> persistence is exact and holds at every coupling, which removes the M10.3 ground. It
> turns on the pattern of eleven written seam records, a pinned piece chosen with this
> row's criterion known. The reply on #723 reads such a choice as a calibration, and the
> admission reply puts a row computed with something still supplied at ⚠️ (§ 5).
>
> **Scope**: an independent recomputation of the free two-load sector on the written and
> the blank board, and of the acquisition model the method note states, with one exact
> check of the seam law on one seam, and the record of the review's two decisions; it
> licenses "the recomputed numbers are correct for that generator and that model", and
> does not license the law on the full board beyond what the run uses, the
> preparation's addressed operations as operations of the law, a particle
> identification, or any other row. The ⚠️ is a decision from the published bar, not a
> result of the recompute.

## 1. The equations

Two unit loads at distinct ports $i \lt j$ of the 240-port carrier, 28,680
configurations. A seam is live when its memory is blank; on a live seam a load moves
into an empty port. With $m_{ij}$ the number of such moves from $ij$, $L$ their
Laplacian and $\deg$ the board's original degrees:

$$
K = \tfrac12 L, \qquad C_{ij,ij} = m_{ij} + (6 - \deg i) + (6 - \deg j), \qquad H(\kappa) = K + \kappa C
$$

The written board leaves out the eleven seams that touch exactly one of the ports
$a = 157$, $b = 156$, $c = 166$, other than $(157, 163)$ and $(163, 166)$; the blank board
keeps all 630. The three source words are $X = ab$, $Y = ac$, $Z = bc$.

**The seam law, on one seam.** From the definitions in the docstring of the pinned
`law.py`: the write $W$, the echo $U$, the record generator $G_\kappa$ and the local term
$h_\kappa$,

```math
U = \begin{pmatrix} 0 & W^\dagger \\ W & I - W W^\dagger \end{pmatrix}, \qquad
G_\kappa = 0_{\rm blank} \oplus \left[ \tfrac12 (I - \mathcal R) + \kappa D^2 \right] \otimes I, \qquad
h_\kappa = U G_\kappa U^\dagger
```

The check, in exact arithmetic with
$\kappa$ a symbol: $h_\kappa$ has no element between blank and written memory, acts on
blank memory as $\tfrac12 (I - \mathrm{SWAP}) + \kappa (n_a - n_b)^2$, and annihilates the
neutral record (the equal coin superposition at $d = 0$) beside each of the nine load
words. That is the one fact the run uses: a written seam is switched off.

**The acquisition** (the method note's three-word model). On $(X, Y, Z)$,

```math
2 K_{\rm src} = \begin{pmatrix} 1 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 1 \end{pmatrix}
```

Three pulses of length $2\pi/3$, each followed by a read that removes $Y$,
give a failed weight $f$ per attempt and the accepted state $v$. With at most eight
attempts:

$$
\rho = \Big( \sum_{j=0}^{7} f^j \Big) v v^\dagger, \qquad p_{\rm exhausted} = f^8
$$

**The floor.** $D = (X - Z)/\sqrt2$ satisfies $2 K D = 2 D$ and $C D = 3 D$ on the
written board, and lies inside the 30 home seams $P$. For any $B \perp D$,
$\langle D \vert P e^{-itH} B \rangle = \langle e^{itH} D \vert B \rangle = 0$, so the
cross term vanishes and

$$
p_{\rm home}(\kappa, t) \ \ge\ \langle D \vert \rho \vert D \rangle = (1 - f^8) \cdot \tfrac{729}{730}
\qquad \text{for every real } \kappa \text{ and every } t
$$

**The four arms**, at $(\kappa, t) \in \lbrace 0, \tfrac12, 8 \rbrace \times \lbrace 1, 7, 37 \rbrace$
on both boards, with $(w_X, w_Z)$ the diagonal of $\rho$ and $U_t = e^{-itH}$:

$$
p^{\rm coh} = \sum_{s \in P} \Big\lvert \sqrt{w_X} \langle s \vert U_t \vert X \rangle + \sqrt{w_Z} \langle s \vert U_t \vert Z \rangle \Big\rvert^2, \qquad
p^{\rm deph} = \sum_{s \in P} \Big( w_X \lvert \langle s \vert U_t \vert X \rangle \rvert^2 + w_Z \lvert \langle s \vert U_t \vert Z \rangle \rvert^2 \Big)
$$

$$
p^{\rm swap} = \sum_{s \in P} \big( e^{-tK} (w_X X + w_Z Z) \big)_s, \qquad
p^{\rm unf} = \sum_{s \in P} \lvert \langle s \vert U_t \vert X \rangle \rvert^2
$$

with the amplitudes $\sqrt{w}$ carrying the signs of $v$. The propagation is
`scipy.sparse.linalg.expm_multiply`, with the mean diagonal removed (a common phase).

**The escape rank** of the block of $L$ from the 30 home configurations to the rest, by
exact integer elimination, on each board.

**The one-load table.** One load from $b$, two uniformly chosen attempts over its
seams, a written seam leaving the load in place, read after both attempts or stopped
at the first receipt at $163$; reported as (receiver, outside the four ports).

**Arming.** Before any comparison: the two flown states unit and orthogonal to
$10^{-10}$ at every setting; the energy of $X$ conserved over $[0, 37]$ to $10^{-10}$ of
the largest diagonal entry of $H$; and, on the blank board at $\kappa = 8$, the three
values on record for the M10.3 sharp pair, which were computed by a Chebyshev route
([`m10_3_maintainer_recompute.py`](../scripts/m10_3_maintainer_recompute.py)).

## 2. Equation-to-code map

One file,
[`m10_4_maintainer_recompute.py`](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_4_maintainer_recompute.py).
Its docstring carries the same equations. It imports nothing from the authors'
package or from the pull request's scripts.

| Object | Function | Lines |
| --- | --- | --- |
| The authors' numbers | `CLAIMED` | [101-130](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_4_maintainer_recompute.py#L101-L130) |
| The board and the run summary, each after its SHA-256 | `load_board`, `load_summary` | [141-161](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_4_maintainer_recompute.py#L141-L161) |
| The seam law on one seam, exact | `seam_fact` | [167-213](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_4_maintainer_recompute.py#L167-L213) |
| Degrees, the eleven written seams | `degrees`, `written_seams` | [219-229](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_4_maintainer_recompute.py#L219-L229) |
| $L$, $C$ on a board | `sector` | [232-253](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_4_maintainer_recompute.py#L232-L253) |
| The exact rank | `rank_exact` | [256-270](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_4_maintainer_recompute.py#L256-L270) |
| The acquisition, the floor | `acquisition`, `floor_exact` | [276-302](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_4_maintainer_recompute.py#L276-L302) |
| $e^{-itH}$ and the four arms | `flown`, `arms_at` | [305-352](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_4_maintainer_recompute.py#L305-L352) |
| The one-load table and facts | `one_load` | [358-412](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_4_maintainer_recompute.py#L358-L412) |
| The arming checks, the comparisons, their wrong-value arms | `arming_checks`, `verdicts`, `arms` | [418-525](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_4_maintainer_recompute.py#L418-L525) |
| The dark identity, the escape ranks, the run-summary comparison | `main` | [531-659](https://github.com/openwave-labs/openwave/blob/main/openwave/xperiments/m10_oph/research/scripts/m10_4_maintainer_recompute.py#L531-L659) |

## 3. Reproduction

The script reads one data file of the authors', the carrier's seam list
`model/executable_model/carriers/FEDERATION_LEVEL0.npz` of the author package at commit
`09f062f77f7f540f2783f128ae6314c5d8b818ea` (tag
`interval-echo-m10.4-connected-pair-proposal`), as arrays (`allow_pickle=False`). Its
SHA-256 is `99870398e3d516f899c600e9bf25ba53c7f513142ba8facbe1cedd94b23f915b`, the M10.3
board's, and the script checks it before computing anything. It also reads the merged
run summary [`m10_4_reproduction.json`](../data/m10_4_reproduction.json) as JSON, after
checking its SHA-256. It runs none of the authors' code. A missing or different input
stops it with a nonzero exit.

```sh
python3 -I openwave/xperiments/m10_oph/research/scripts/m10_4_maintainer_recompute.py \
  /path/to/FEDERATION_LEVEL0.npz --output /path/to/out.json
```

The graph file is not vendored: the authors' record stays in the authors' repository,
and the pin is the commit plus the hash. The run took about two minutes and is recorded
in [`m10_4_maintainer_receipt.json`](../data/m10_4_maintainer_receipt.json).

## 4. Results

Stored in [`m10_4_maintainer_recompute.json`](../data/m10_4_maintainer_recompute.json).
Each line is compared with the authors' value; each comparison turns false when it is
fed a wrong value (23 of 23, the three arming checks included).

| Claim | Result | How compared |
| --- | --- | --- |
| The seam law: no element between blank and written memory; blank memory gives exchange plus contrast | holds, exact, every κ | sympy, κ symbolic |
| A neutral record switches its seam off, beside each of the nine load words | holds, exact, every κ; a record at $d = 1$ does not | sympy, κ symbolic |
| The eleven written seams, two of them between cells | the same eleven; (152, 166) and (157, 170) join other cells | exact |
| Acquisition: failures 4/9, 4/81, 4/729; accepted $(-13 X + 14 Z)/27$; exhaustion 0.0038635883 | the same | within 1e-14, 1e-12 and 6e-11 |
| The floor, 0.994771841 | $(1 - (364/729)^8) \cdot 729/730 = 0.99477184127$, equal to the weight of $D$ in $\rho$ | exact, and within 1e-12 |
| $2 K D = 2 D$ and $C D = 3 D$ on the written board | holds; fails for $X + Z$ and on the blank board | integers |
| Escape ranks: 29 written, 30 blank | 29 and 30 | exact elimination |
| The method note's table at exposure 37, four arms at κ = 0, ½, 8 | the same twelve values | within 5e-10 plus 1e-11 |
| The coherent arm at or above the floor | its least value over the nine settings is 0.99520 | against the exact floor |
| Blank memories at κ = ½, exposure 37: coherent 0.440929105, unfiltered 0.416819359 | the same | within 5e-10 plus 1e-11 |
| Random swaps at exposure 1: 0.753088868, below the 0.94 gate | the same | within 5e-10 plus 1e-11 |
| Ten writes, coherent, exposure 37: 0.526840130, 0.940265361, 0.996086764 | the same | within 5e-10 plus 1e-11 |
| The one-load table, six (receiver, other) pairs | the same fractions | exact |
| The run summary's 72 home probabilities | largest difference 4.7e-10 | within 5e-10 plus 1e-11 |

Armed before any comparison: the flown states stay unit and orthogonal to 3.8e-11, the
energy is conserved to 3.8e-11 of the scale, and the three M10.3 values come back within
3.5e-10.

**What the result depends on.** Reported from the same script, not compared with a
claim. The coherent arm at exposure 37:

| κ | Written board | Blank board |
| --- | --- | --- |
| 0 | 0.995 | 0.271 |
| 0.5 | 0.996 | 0.441 |
| 1 | 0.996 | |
| 2 | 0.996 | |
| 4 | 0.996 | |
| 8 | 0.996 | 0.970 |
| 16 | 0.996 | |
| −8 | 0.996 | |

- **On the written board the result does not turn on the coupling**, as the floor says.
  On the blank board it turns on the coupling again, as in M10.3.
- **One load alone already persists on the written board.** $(a - c)/\sqrt2$, on ports 157
  and 166, satisfies $2 K_1 v = 2 v$ for the one-load move Laplacian of the written
  board, and not on the blank board: its two routes to 156 cancel, and so do its two
  routes to 163. Restoring one written seam on its own leaks $1/8$ of that state at first
  order for each of the eight at the tips, and nothing for the three at 156. At κ = 0
  the two loads do not interact. The review asked the authors which of the two is the
  row's particle; this does not bear on the mark.
- **Fifteen seams touch the three ports**, eleven written and four live, so the routing
  enumeration's 32,768 settings are $2^{15}$.

## 5. The two decisions

From the review as posted, addressed there to the authors.

**The pin.** Adopted at `09f062f7`. The canonical says what changed and keeps the
ancestry: the board, the seam law, the degree references and the detector are M10.3's,
and the preparation and the state at release are new. The law and the board have the
same git blob hashes at both pins. The M10.3 record stays reproducible from `4fe10f9a`.
Adopting the pin adopts § 1.3's candidate with its records as pinned memory, consistent
with the mark below in the way the coupling was: pinned, the carrier's own, and held at
⚠️ on this row because the choice answers to the row.

**Row 4.** ⚠️ partial. The bar, as [`MODELS.md`](../../../../../MODELS.md#summary-status)
states it for every column:

> **✅ validated**: the property comes out of that process and meets the row's test. **⚠️ partial**: it comes out with something still supplied (a value the model is calibrated to earns nothing by itself; a state the process returns once that value is set can), or with part of the row open.

and in the words of the
[admission reply](https://github.com/openwave-labs/openwave/discussions/712#discussioncomment-19595538):

> What moves a row is its observable computed on the carrier's own process in-platform. With something still supplied, that is a case for ⚠️ with the caveat in the cell. With the source derived from the carrier, it meets the #482 bar for ✅.

Read against this result:

1. **The persistence is computed in-platform, and it is exact.** It is an eigenvector
   identity on the full two-load sector, checked in integers, and the step from the seam
   law to the load generator is checked for the one fact the run uses.
2. **It no longer turns on the coupling.** The floor holds at every real κ, zero
   included. The M10.3 ground, the supplied value 8, does not hold this result at ⚠️.
3. **The written memory is part of the candidate, and as pinned it is the carrier's
   own.** Admission term 6: "A changed law, memory, state or decoder is a new candidate
   with its ancestry kept". The canonical's § 1.3 nominates "the pair and its eleven
   prepared neutral seam records". The reply on
   [#723](https://github.com/openwave-labs/openwave/pull/723#issuecomment-6920720626),
   reading 1: "Cells, law, memory, schedule and routing, as pinned, are the carrier's own
   process. Being declared does not by itself make one of them a declared source for a
   row." So the records do not send the row back to 🚧.
4. **The pattern of records is a choice that answers to this row.** The same reply:
   "Where a choice answers to a target, the row is a calibration of that choice." The
   M10.3 decision read a pinned piece of that kind as the piece "still supplied" of the
   admission reply, in these words: "What makes it a supplied piece here is that it
   answers to this row's target, and your canonical § 1.2 also lists it as supplied".
   That ground had two legs, and this one has the first only: § 1.3 does not list the
   records as supplied. The first is the one the sentence names as what makes the piece
   supplied, and the label came as "also". § 1.3 calls the preparation "designed
   apparatus" and records what was known when the source was chosen. The M10.3 decision
   said this of a value of the law, the coupling. This decision says it of pinned memory,
   for the first time, on two points:
   - **It is a choice.** The structure admits other patterns, and the authors' own
     controls use them: blank memories and the ten-write layout. That is the
     [parameter-count test](../../../../../ONBOARDING_MODELS.md#the-parameter-count-test)'s
     step 2 ("Count a choice as free if the model's own framework admits an inequivalent
     alternative"). The row's outcome turns on it. With the same accepted load state and
     the same detector at exposure 37, the home probability at κ = ½ is 0.996 on the
     eleven writes and 0.441 on blank memories. At κ = 0 it is 0.995 on the eleven
     writes, 0.271 on blank memories and 0.527 on ten writes. Without the records the
     outcome turns on the coupling again: on blank memories the same state returns 0.970
     at κ = 8.
   - **It answers to this row.** The canonical's § 1.3: "The 94% criterion and the
     interference solution were known when the eleven-record source was chosen."

The M10.3 decision listed the adjacent launch and the blank memories as the
experiment's design, and its record showed no choice of memory made for the row. Here
the record shows one, and the matched control shows the row turning on it. A prepared
state is not the ground; the two points of item 4, together, are. The acquisition
filter, the reads, the attempts, the exposures and the detector are the experiment's
design: they go in the cell's text and do not hold the row.

**The question the authors put to the review**, in the pull request's words: "whether
that persistent material belongs to the localized candidate or counts as a supplied
trap". The mark does not rule on it. What the row's particle is made of is the authors'
to say, and the nomination is recorded as theirs. Item 4 holds on either answer: taken
as part of the particle, the records are pinned memory chosen with the row's criterion
known; taken as apparatus, they are supplied beside it.

The same row in the columns that hold a state, as the cells stood at the review:

| Column | Cell | What it rests on |
| --- | --- | --- |
| M5 | ✅ | the electron of its verified-L stack persists in free dynamics, started from a mid-descent snapshot, over one window on one box |
| M7 | ⚠️ | a harmonic-frame soliton whose real-time vacuum is tachyonic, so persistence fails in about two periods |
| M4 | ⚠️ | a lock-in that holds at perfect placement and is fragile under perturbation |
| M6 | 🚧 | a localized state that has not been evolved in time |

**Item 4 against M5's cell.** M5's start state is a choice too, and its run turns on it:
[`m5_21_10_note.md`](../../../m5_liquid_crystal/research/findings/m5_21_10_note.md)
launches three seeds, each from its own mid-descent snapshot, and the electron holds
while the tau candidate disintegrates and the mu candidate sheds outward-moving features.
The first point alone would therefore move M5's cell, and it is not the ground. The
second point is what separates the two cells, and the review ran it on the two choices
M5's record shows:

- **The seed.** The note runs the electron seed as the control of a task whose two decay
  reads were the other seeds, and records no selection of it for persistence.
- **The box.** The cell's run is at one box, N = 64, and its text says the radius
  follows the box. The [task's plan](../../../m5_liquid_crystal/research/tasks/m5_21_10_task_details.md)
  logs why that box, before the run: grown from 48 to 64 at the same spacing, which
  "isolates the BOX variable (the named caveat on both M5.21.6 verdicts)". Those verdicts
  were on the other two seeds. The plan logs the box as chosen to test that caveat, and
  records no selection of it for persistence.

So item 4 leaves M5's cell where it is, on the record as it stands, with the box caveat
the cell already carries. The same reading moves M5's cell, through its author, if its
record shows a choice made for this row. The canonical's § 1.3 states what was known
when the records were chosen; that statement is what the rule asks for.

**What would move it to ✅.** In the reply's words, nothing supplied for the row. Two
routes, each decided from its artifact; no icon is promised here.

- **On blank memories.** The M10.3 preparation, which under term 6 is a different
  candidate from the one nominated here. Its route is the one the cell already named:
  the coupling fixed by something independent of this row's target. The authors' answer
  on κ = 8 is recorded as given: the design swept nine strengths (0, ½, 1, 2, 4, 8, 16,
  64 and 128) and holds no selection of 8 that is independent of persistence
  ([#745](https://github.com/openwave-labs/openwave/pull/745#issuecomment-7103977216);
  the [task](../tasks/m10_4_task_details.md)).
- **On written memories.** The pattern of records fixed by something independent of
  this row's target, the parameter-count test's step 5 ("Genuine forcing means: fixed by
  something independent of the target being reproduced"). For the filed pattern the
  criterion and the solution are known, and registering a rule now does not undo that.
  A rule counts when the record shows it was fixed without reference to this layout: by
  another row's task, or by a requirement written before the pair's layout was in view.
  The pattern it returns may be this one or another.

The one-load routing task does not count toward the second route as filed: "This
routing requirement was formulated with the pair layout known" (the method note).

## 6. Not computed, not run

- The authors' package and the pull request's two scripts were not run. The two
  scripts and the pinned `law.py` were read in full as text; the route above shares no
  code with them.
- The seam law beyond one seam: the degree references on written memory, and the echo
  and read of the terminal detector.
- The acquisition's addressed operations as operations of the law: the four swaps, the
  pulses and the vacancy reads. The three-word model the method note states was
  recomputed.
- The 1,116 per-address probabilities of the authors' archive `connected_records.json`.
  The 72 home sums of the run summary were compared.
- The $t^8/18432$ coefficient, the two minors of the escape block, and the pair's
  single-omission norms.
- The authors' tests, the Windows run, and the routing protocol at `5c915945`.

## 7. Minimal inspection set

1. [`m10_4_maintainer_recompute.py`](../scripts/m10_4_maintainer_recompute.py), the docstring first.
2. [`m10_4_maintainer_recompute.json`](../data/m10_4_maintainer_recompute.json).
3. The authors' [method note](m10_4_method_note.md), for the experiment as they state it.

## 8. Audit

The review as posted was checked by second models before it went out. A fact pass
reran the recompute, rechecked every quotation, link and hash, and found nothing
blocking; its one wording point was applied. A judgment audit ran four passes on the
decisions. It reached ⚠️ independently in each, and found the stated reasons wrong
twice over before the last pass found nothing blocking. The first draft said the bar
itself put the records at ⚠️, where the bar's clause names a calibrated value; it
compared the records with M5's stack, where the comparable choice is M5's start state;
it quoted the M10.3 decision's sentence without its second leg; it promised a later
test of M5's cell instead of running the test from M5's record; and it offered as a
way forward a rule registered now, which would carry the same defect the routing task
is not counted for. The posted text rests the mark on the published sentences it
quotes.

This note, its script and the cells were audited again by second models before they
landed. A numbers pass reran the script and matched its output leaf by leaf, recomputed
the headline values by a Chebyshev route of its own armed on the M10.3 values, and ran
17 mutations of the computation: 16 turned a verdict false or stopped the run at the
arming checks, and the one that survived changes the one-load leak table, which § 4
reports and does not compare. A record pass compared § 5 with the posted review
sentence by sentence. It found a lower bound rounded up in the cell, the experiment's
design missing from the cell, the authors' answer on κ = 8 not recorded, and § 5
shortened in places where the review gave its reasons; all were corrected before
landing.
