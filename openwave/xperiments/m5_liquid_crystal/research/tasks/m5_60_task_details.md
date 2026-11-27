# M5.60: the held ladder of two rings again: a hold accepted by its response, on the lowest branch

> Opened 2026-11-27 from the author's answer to the results of [M5.57](m5_57_task_details.md) on discussion #186 ([discussioncomment-19938140](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19938140), 2026-11-27 01:49 UTC). Roadmap row: [`m5_roadmap.md`](../m5_roadmap.md). Thread record: [`m5_32_convo.md`](m5_32_convo.md).

## TASK PLANNING (2026-11-27; opened, not staged: the plan is written and audited before any go)

**Scope.** [M5.57](m5_57_task_details.md) closed with neither of the author's two predictions scored: the control of its hold was not met, so no held ladder ran, and the free pair on larger boxes carried no label. The results post asked the author three questions; this task holds what their answer asks for the two rings. The action, the spacing parameter and the boxes are those of M5.57 unless the plan changes one and says so.

| # | What the answer asks | On record on OpenWave |
| --- | --- | --- |
| 1 | The same line moment $D'$ as the hold, with a changed acceptance rule: a held $E(d)$ is valid if (i) each held state is stationary under the constraint, (ii) the fitted separation is a monotone, reproducible function of the target, (iii) release returns to the free state | M5.57 § 10: held at 1.0 and 2.0, one ring met its moment with the fitted center at 0.851 and 1.683; both held states stationary by the audit's check; both releases back on the free ring's energy and center, one clause of the last-step check not met |
| 2 | The control repeated with the ring moved along its normal, which is the geometry of the ladder, and on the pair, scored on (i) to (iii) | M5.57 § 10: the control moved the ring in its own plane only. One held chain on the pair was stopped unconverged at its first target and decides nothing |
| 3 | The energy read against the fitted separation, not against the target | M5.57 § 4 already takes the reader's separation as the abscissa of its curve, with $D'$ printed beside |
| 4 | If the response is not monotone along the normal, a smaller $g_0$ (0.1 as an example), which concentrates the weight on the core | Every read of M5.57 used $g_0 = 0.2$ |
| 5 | The held ladder run "on the lowest-energy (possibly turned) branch" | M5.57 §§ 11 and 12: ends with a turned ring lie below the coaxial ones on half-boxes 24 and 30 at spacing 0.5, and the free pair at spacing 0.375 on half-box 18 holds both rings turned 36° |
| 6 | The free pair scored for OW57-1 on the lowest converged end found on each box, with the orientation angle reported as a read | M5.57 § 3 counts a box only when its two starts end alike, which a turned end beside a coaxial one cannot meet |

The author sets aside a hold on the fitted centers themselves, as not differentiable through the fit, and agrees with avoiding the degree moment, which moves in jumps on this lattice.

**What the answer states about the record (the author's reading; evidence, not resolution).** On the coaxial ends the free separation over the box width reads 0.709, 0.687 and 0.678 on half-boxes 18, 24 and 30 (the ratios hold against the record of M5.57), with the face distance growing. The author reads this as rings at a nearly fixed fraction of the box, the pattern OW57-1 predicted, and agrees that under the frozen rule of M5.57 it stays unscored. On the turned ring: relative orientation is a collective coordinate of the pair with its own potential, as for two Skyrmions, whose attractive channel is a relative rotation, so the coaxial branch need not be the lowest. Neither statement was tested on OpenWave.

**Nothing of M5.57 is scored again.** The acceptance rule and the scoring of OW57-1 above are rules for the runs of this task, written after the ends of M5.57 were read. The ends of M5.57 keep the words its frozen plan gave them.

**Open at the plan.** Each to be fixed with a threshold and a known-answer control before any chain is relaxed.

1. The three clauses as numbers: what "stationary under the constraint" is at an end that the optimizer leaves on an abnormal message at a small gradient; how many targets carry "monotone", and over what range; between which repeats "reproducible" is read (two starts, two values of the penalty, two tries).
2. What the plan does when the response along the normal is monotone and far from one, and when it is not monotone at $g_0 = 0.2$: whether the smaller $g_0$ is a second branch of the control or a new control.
3. What a curve read against the fitted separation bounds. A held state is a minimum of the energy at a fixed moment, not at a fixed fitted separation, so at its fitted separation it lies at or above the lowest energy with that separation. The plan states what a contact maximum read on such a curve says about OW57-2, whose strict scoring is that of the amendment of M5.57.
4. The transverse drift. In the control of M5.57 the fitted center also moved across the push (−0.118 and 0.128 in the reading of the record), which clauses (i) to (iii) do not name.
5. The lowest branch: how it is found on each box and spacing, how many starts, and what is run when the lowest end found is not converged.
6. Whether the free pair on larger boxes is run again in this task for OW57-1, and with which convergence rule.
7. The faults the review of M5.57 names, each with its rule: a control in the geometry the instrument is used in; a word for a chain that ends abnormally after its restarts at a small gradient; the sentence on restarts; the list of rows and the seeds written before the go; no smoke run on a field whose log could inform a control.

**Definition of done.** Written with the plan.

**Gating.** [M5.59](m5_59_task_details.md) first, by the order the author asked in the same post; then a plan with its audit, and the user's go.

**Blindspot pass, consumers, research body.** Written with the plan.
