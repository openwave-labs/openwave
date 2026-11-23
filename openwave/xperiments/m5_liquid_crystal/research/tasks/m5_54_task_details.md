# M5.54: the author's charge integral ported, and the ranges of rung T2 scored on the states of M5.51

> Staged 2026-11-23 from the author's post of 2026-11-23 20:05 UTC (rung T2; the row of that date in the [thread record](m5_32_convo.md)). Roadmap: [`m5_roadmap.md`](../m5_roadmap.md). It reads only the stored states of [M5.51](m5_51_task_details.md), the port that T2 describes, and runs no relaxation. Sister tasks of the same thread: [M5.52](m5_52_task_details.md) (the frequency of rung T4) and [M5.53](m5_53_task_details.md) (the scope of rung T3).

## TASK PLANNING (2026-11-23; staged)

**Scope.** In the post of 2026-11-23 ([discussioncomment-19840001](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19840001), 20:05 UTC, item 6) the author offered rung T2: an independent port of the ring of their post of 2026-11-20 with the midpoint weight, relaxed from the radial hedgehog on half-boxes 6 and 9 at spacings 0.5 and 0.375, with four reads stated as registered on 2026-11-23 at 20:53 CET, before any run, at a confidence of 80 percent. That port is [M5.51](m5_51_task_details.md), run under a plan frozen before the post; its plan holds no rule for T2's ranges and did not read Q/4π. This task does two things on M5.51's stored end states: it ports the author's charge integral Q/4π with its controls (§§ 1 to 2), and it turns M5.51's numbers and the new Q/4π into one word per grid and read against T2's ranges, by frozen arithmetic (§ 3). It reports words; no cell of `MODELS.md` changes by this plan.

| Read | T2 as stated (post of 2026-11-23, item 6) | Grids, in M5.51's names | The number scored |
| --- | --- | --- | --- |
| U | 20.50 (G1), 20.53 (G2), 21.50 (G4), each within 1 percent | G1, G2, G4; G5 tabled, no entry | M5.51, midpoint weight, $U$ |
| Ring radius | 2.76 to 2.86 | G1, G2, G4, G5 | M5.51's readers $R_L$, $R_A$ and the offset read $R_{\mathrm{rel}}$ of § 3 |
| Q/4π | at least 0.99 | G1, G2, G4, G5 | this task, § 1 |
| N/U | between 0.94 and 1.00 | G1, G2, G4, G5 | M5.51, N/U |

G1 is half-box 6 at spacing 0.5 ($25^3$), G2 half-box 6 at 0.375 ($33^3$), G4 half-box 9 at 0.5 ($37^3$), G5 half-box 9 at 0.375 ($49^3$). G3 and G6 of M5.51 are not T2 grids and are tabled only.

**The order of events, as the records show it.**

| Event | Time (UTC, 2026-11-23) | Record |
| --- | --- | --- |
| M5.51's plan frozen and pushed to the public branch | 18:22 | commit `e6da01ff` |
| M5.51's 24 chains and the re-relaxations of its K5 ended | by 19:01 | local file times of M5.51's chain folders |
| The four continuations of M5.51 § 17 ended | by 19:21 | the same |
| T2 registered, as the author states it | 19:53 (20:53 CET) | the post, item 6 |
| The post | 20:05 | the discussion |

M5.51's plan, with its registered expectations, was on the public branch from the push of `e6da01ff` at 18:22 UTC, 91 minutes before the stated registration; none of M5.51's results was public at either time. The order supports one claim: T2's ranges were stated with no access to M5.51's numbers (its plan could be read, its results could not), and M5.51's rules were frozen before T2 existed. It does not support three others. (a) A blind read of U, the radius and N/U: those numbers stood in M5.51's record when § 3 was written, so their words are arithmetic against the author's ranges, and the choices inside that arithmetic (which chain, which weight, which radius reader, the budgets, the withholding) were made with the numbers in view. The guard is that each choice takes the stricter of its readings in both directions and is named, not blindness. (b) A prediction made without data: "before any run" refers to the port, and the author's own runs with the midpoint weight (the fixed-field rises at δ = 0 and a relaxed state at δ = 0.05, post of 2026-11-21, item 5) came before it. As arithmetic facts, with no claim about how the numbers were chosen: T2's three U values equal the author's stored energies times the fixed-field rises that M5.51's K2 reproduced, to the digits given (20.280477 × 1.010709 = 20.498, 20.423145 × 1.005358 = 20.533, 21.287491 × 1.010084 = 21.502), and the release's radii 2.73 to 2.83 scaled by about 1.01 give 2.76 to 2.86. (c) Independent time marks for everything: the public marks are two, the push of `e6da01ff` (18:22 UTC) and the post (20:05 UTC); the registration time is the author's statement and M5.51's end times are local file times. Only Q/4π is read after its rule, and § 1 shows that its value is fixed by the two outer node layers.

**The inputs already in view** (M5.51 § 10 and [`m5_51_summary.json`](../data/m5_51_summary.json), midpoint weight; the builder reproduces this table from the summary, not by hand):

| Grid | Chain | U | $R_L$ | $R_A$ | N/U | M5.51's second step on U, $R_L$, N/U |
| --- | --- | --- | --- | --- | --- | --- |
| G1 | `G1_mid_a`, `G1_mid_b` | 20.482703, 20.482703 | 2.8140, 2.8140 | 2.8247, 2.8247 | 0.9345, 0.9345 | passes; withheld (earlier state = start) |
| G2 | `G2_mid_a`, `G2_mid_b` | 20.527989, 20.527962 | 2.8154, 2.8133 | 2.8146, 2.8192 | 0.9382, 0.9388 | withheld (earlier state = start; class and one ring too); passes |
| G4 | `G4_mid_a`, `G4_mid_b` | 21.487779, 21.487779 | 2.8516, 2.8516 | 2.8697, 2.8697 | 0.9558, 0.9558 | passes; withheld (earlier state = start) |
| G5 | `G5_mid_a`, `G5_mid_b` | 21.548266, 21.548266 | 2.8439, 2.8439 | 2.8484, 2.8484 | 0.9594, 0.9594 | passes; withheld (earlier state = start) |

**Words already fixed by M5.51's numbers.** The rules of § 3 applied to the inputs above, recomputed from the summary for this plan and by the plan audit; the builder recomputes them again. They are written here because they follow by arithmetic; the run adds the Q column, whose outcome § 1 shows to be fixed as well.

| Grid | U | Radius | N/U |
| --- | --- | --- | --- |
| G1 | INSIDE (−0.084 %) | no label (readers; $R_{\mathrm{rel}}$ = 2.7595) | OUTSIDE (0.9345) |
| G2 | no label (second step) | no label (second step; $R_{\mathrm{rel}}$ = 2.7620) | no label (second step; 0.9382 and 0.9388, both under 0.939) |
| G4 | INSIDE (−0.057 %) | no label (readers; $R_L$ 0.0084 inside against 0.01, $R_A$ 0.0097 outside against 0.020, $R_{\mathrm{rel}}$ = 2.8600) | INSIDE (0.9558) |
| G5 | tabled, no entry | no label (readers; $R_A$ 0.0116 inside against 0.033) | INSIDE (0.9594) |

**What the author's archive holds, read as text.** The checks archive attached to the author's post of 2026-11-20 19:28 UTC, and again to that of 2026-11-21 19:43 UTC, holds identical copies of the script `hodge.py`, its output `hodge.json`, the loop check `holonomy.py` and the registration `predictions_hodge_20nov.md` in the folders `v41rc5/soleve20b/` and `v41rc5/rethink20/`, and the notes `rethink_20nov.md` in `v41rc5/rethink20/` only. They were read as text and never run or imported; they were produced with AI assistance on the author's side and are evidence, not resolution, and § 1 re-derives what it uses. Four facts shape the plan. (a) The definition: the top eigenvector, lifted by the sign of its product with the position, the density $b$ of triple products, $\rho = \nabla \cdot b$ by second-order central differences inside and first-order one-sided differences on the faces, and Q summed over the nodes one layer in from the faces. (b) The script read three files named `Zz_L6_h5`, `Zz_L6_h375`, `Zz_L9_h5` and printed Q/4π = 0.9986, 0.9991, 0.9994 (16 digits each), and the shares of $\lvert \rho \rvert$ within distances 0.5, 1 and 1.5 of the line, with the distance taken about the z axis through the box center at the radius and height of the least top gap on one half-plane. (c) At plan time the gap read of that script, re-coded, returned on M5.51's stored array `checkpoints/m5_51/author/milestone_action_halfbox6_h0p5.npz` with its grid axes and tensor indices permuted cyclically (new x, y, z = old y, z, x) the least gap, radius and height that the archive prints, to every printed digit: the author's array of half-box 6 at 0.5 is read here as that lattice rotation of the stored array, which C1 tests on the whole field. No lattice rotation or reflection of the stored G2 or G4 arrays matches (the least gap differs in the fifth significant digit on G4 and in the third on G2, whose stored ring lies 17° off the nearest box axis). (d) The registration's item P-1 reads, verbatim, "total rho in the box (one-cell margin) = 4 pi within 5 %", scored met by the author at those three values; the post of 2026-11-20 (item 2) states Q/4π = 0.999 with 93 to 96 percent within one unit of the line.

**Definition of done.** The controls of § 2 each met or reported as the exact failing check, the regime test with its known answers included; the SHA-256 of `hodge.py` and `hodge.json` recorded; Q/4π at both surfaces of § 1, the regime test, the profile $Q(\ell)$ and the shares read on the 24 end states of M5.51, on their earlier states, and on the author's four stored arrays; every entry of § 3 carrying one outcome, or "no label" with the rule that withheld it; the table of words already fixed recomputed from the summary, any difference reported; the tally of § 3; findings that embed $Q(\ell)/4\pi$ against $\ell$ per state and $\rho$ on a meridian plane of the four T2 midpoint states; a numbers check on a different model recomputing every word from the summary JSON; an adversarial audit with its own code for Q on the stored states, trying to refute each word; the doc checker clean.

**Gating.** The repo owner's commit of this plan, which freezes it and is the go for its reads, under the pre-flight already armed for M5.51 and M5.52. No answer of the author is waited for. The plan audit and its fold are recorded at the end of this section.

### 1. The charge integral

λ₁ ≥ λ₂ ≥ λ₃ are the eigenvalues of $M$, $g_{12} = \lambda_1 - \lambda_2$, and $\hat n$ the top eigenvector, defined up to sign where $g_{12} > 0$. With $x$ the node position about the center node of the grid and $\hat r = x / \lvert x \rvert$, the oriented lift and the density are

$$
\tilde n = s \hat n , \qquad s = \operatorname{sgn}\big( \hat n \cdot x + 10^{-12} \big) , \qquad b_a = \tfrac12 \epsilon_{abc} \, \tilde n \cdot \big( \partial_b \tilde n \times \partial_c \tilde n \big) , \qquad \rho = \nabla \cdot b , \qquad Q = \int \rho \, d^3 x
$$

so $b_1 = \tilde n \cdot (\partial_2 \tilde n \times \partial_3 \tilde n)$ and cyclically; $b$ is the dual of the pulled-back area form $G$ of the post of 2026-11-20 (item 3). Where $\tilde n$ is smooth, $b$ is the pullback of the closed area form of the unit sphere, so $\nabla \cdot b = 0$: $\rho$ lives only on the ring line, where $g_{12} = 0$ and $\hat n$ turns by π about the line, and on the branch sheet where $\hat n \cdot x = 0$ and $\tilde n$ jumps to $-\tilde n$. For a closed surface $S$ on which $\tilde n$ is smooth, $\int_V \rho = \oint_S b \cdot dS = 4\pi \deg(\tilde n \vert_S)$. On a cube centered on the grid, $\hat n \cdot x \neq 0$ everywhere on $S$ gives $\tilde n \cdot \hat r > 0$ there, so $\tilde n$ deforms to $\hat r$ through nonzero fields and $\deg = 1$. ⭐ With this lift, every centered cube on which $\hat n \cdot x$ does not vanish reads Q/4π = 1, whatever lies inside it.

**The discretization (the archive's, ported).** $\partial_a$ is the second-order central difference at a node with both neighbors along $a$ and the first-order one-sided difference on a face; $\rho(n) = \sum_a \partial_a b_a(n)$; $Q = h^3 \sum_{\mathrm{inner}} \rho$ over the nodes off the faces. With $i = 0, \dots, N - 1$ the index along axis $a$ and $\sum_{\perp a}$ the sum over the inner indices of the other two axes, the sum telescopes exactly:

$$
Q = h^2 \sum_{a=1}^{3} \sum_{\perp a} \tfrac12 \big[ b_a(N - 1) + b_a(N - 2) - b_a(1) - b_a(0) \big]
$$

Since $b_a$ holds derivatives along the other two axes only, each term uses the nodes of one layer: Q is a function of the field on the two outermost node layers alone. On M5.51's states the outer layer is held at the seed $M_0$ (top eigenvector $\hat r$, lift $+\hat r$) and the next one is free.

**Two surfaces, and the profile.** For a half-width $\ell$ that is a multiple of $h$, $Q(\ell) = h^3 \sum_{\lVert x \rVert_\infty \le \ell - h} \rho$, the same two-layer flux at half-widths $\ell - h$ and $\ell$; $Q = Q(L)$. The second surface is $\ell_2 = L - 2h$, whose two layers are disjoint from those of the box and lie on free nodes. The profile $Q(\ell)/4\pi$ for $\ell$ from $2h$ to $L$ is a descriptor. Inside the ring a cube cuts the branch sheet, so there the profile is set by the lift, not by enclosed charge.

**Regime, and its test.** A lift flip between neighboring nodes $i$, $j$ is $\tilde n_i \cdot \tilde n_j \lt 0$. A Q word needs, on both two-layer shells (at $L$ and at $\ell_2$), no lift flip between any two neighboring nodes of the shell, $g_{12} > 0.2$ on every node of the shell, and $\ell_2 - h \ge R_L + 1.5$, so the second shell lies outside the tube that holds the charge on the line. The counts are printed. The test is on lift flips, not on signs: the sign of $\hat n \cdot x$ on the raw eigenvector is arbitrary per node (4436 neighbor pairs of opposite raw sign on the box shell of the author's G1 array, with no flip), and $\tilde n \cdot x \ge 0$ holds by construction, so a test on either could not fail. Its known answers: C4 (a) and the author's four arrays in regime (the plan audit read 0 flips and a least $g_{12}$ of 0.83 or more on the shells of all four arrays); C4 (b) and C4 (c) out of regime, since their branch sheets cross the faces.

**Error budget.** The face-flux discretization. The archive's three values sit $1.4 \times 10^{-3}$, $0.9 \times 10^{-3}$ and $0.6 \times 10^{-3}$ under 1, and fall between them by the factors 0.66 and 0.43 against 0.56 and 0.44 for an error in $(h/L)^2$. The budget is 0.002 at the box surface and 0.0044 at $\ell_2$: the largest archive value scaled by $(L/(\ell_2 - h/2))^2$ is 0.0022 on half-box 6 at 0.5, doubled; the deficits measured at $\ell_2$ on the seed and the author's arrays are 0.0020 to 0.0022. Q/4π is printed to four decimals.

**The values are known, and what they say.** The plan audit computed, with its own code:

| Field | G1 | G2 | G4 | G5 | G6 |
| --- | --- | --- | --- | --- | --- |
| The seed $M_0$, no ring, $Q(L)/4\pi$ | 0.998609 | 0.999237 | 0.999402 | 0.999669 | not read |
| The seed, $Q(\ell_2)/4\pi$ | 0.99795 | 0.99899 | 0.99924 | 0.99960 | not read |
| The author's stored array, $Q(L)/4\pi$ | 0.9986035 | 0.9992369 | 0.9994018 | no array | 0.9996680 |

On the author's arrays $Q(\ell_2)/4\pi$ runs from 0.99781 to 0.99960. The archive's value on half-box 6 at 0.5 equals the seed's within $5.7 \times 10^{-6}$. In its regime Q cannot read OUTSIDE: its value is fixed by the two outer layers (the outer held, the next free), and the seed with no ring reads the same. The Q read therefore has its own outcome, INSIDE (fixed by the held faces), which the tally counts in a column of its own, never among the INSIDE. The author's notes (`rethink_20nov.md`) read this number as the ring carrying the whole hedgehog charge; the read does not show that. What it shows of the ring lies in the shares and the profile, which carry no word.

**The shares (descriptors, no control).** With $d$ the distance to the circle fitted by M5.51's line reader on the same state (center $c_L$, normal $\nu_L$, radius $R_L$), over the inner nodes: $S_\delta = \sum_{d < \delta} \lvert \rho \rvert / \sum \lvert \rho \rvert$ for δ = 0.5, 1, 1.5; the signed share $\sum_{d<1} \rho / \sum \rho$; and the branch share, $\lvert \rho \rvert$ on nodes with $d \ge 1$ that have a lift flip with a neighbor. On the rotated G1 array the shares are printed with both distances, the fitted circle and the archive's, side by side, with no bound. On the line $\rho$ is a line charge whose nodal profile sharpens as $h$ falls (the archive's shares within 0.5: 42 percent at $h = 0.5$, 81 percent at 0.375), so the shares are read at their spacing. The posted 93 to 96 percent (end-weight arrays) is tabled beside, with no word.

### 2. The controls

| Control | Input | The answer it must return | An input that fails it |
| --- | --- | --- | --- |
| C1, the port against the archive | `checkpoints/m5_51/author/milestone_action_halfbox6_h0p5.npz`, rotated as in fact (c) | Q/4π, the least gap, the radius, the height and the four shares, with the archive's distance, each within $10^{-10}$ absolute of `hodge.json`; Q/4π of the unrotated array equal to the rotated one within $10^{-12}$ | the opposite lift (Q → −Q); the least eigenvector; ρ by forward differences; a sum that includes the face nodes; the shares on the unrotated array (the archive's distance assumes a z-aligned ring) |
| C2, the archive's other two arrays | the stored G2 and G4 arrays | Q/4π within 0.001 of the archive's 0.9991 and 0.9994; the shares tabled with no bound | the inputs of C1. The arrays differ (fact (c)), so a round-off match is not expected; both are relaxed states of one action under the same held faces, and Q depends on the two outer layers only |
| C3, the identity | every field read | the volume sum and the two-layer flux of § 1 equal within $10^{-12}$ of the sum of the absolute terms | a sum over the face nodes; a stencil that mixes orders |
| C4 (a), charge 1, in regime | the seed $M_0$ on the grids of G1, G2, G4, G5 | Q/4π within 0.002 and $Q(\ell_2)/4\pi$ within 0.0044 of 1; in regime | the opposite lift gives −1; one component of $b$ in the wrong cyclic order gives about 1/3; the least eigenvector (degenerate on $M_0$) |
| C4 (b), out of regime | $M = \hat z \hat z^{\mathsf T}$ on the same grids | out of regime: $\tilde n = \operatorname{sgn}(z) \hat z$ flips across the plane z = 0, which crosses the faces; Q (0, since $b = 0$) printed | a regime test on the raw sign of $\hat n \cdot x$ or on $\tilde n \cdot x$, which reads in regime |
| C4 (c), charge 1 − √2, out of regime | $M = 0.999 \hat n \hat n^{\mathsf T} + 0.001 \cdot \tfrac13 \mathbb{1}$, $\hat n = (-x, -y, z)/r$, on the same grids | Q/4π within 0.01 of $1 - \sqrt2$: the lift flips $\hat n$ on the band $\lvert z \rvert \lt \rho_{\mathrm{cyl}}$, so on the cube the caps are covered with orientation + and the band with orientation −, $[4\pi(1 - 1/\sqrt2) - 4\pi/\sqrt2]/4\pi$; out of regime | no lift (reads 0); the opposite lift (+0.41) |
| The regime on the author's arrays | the four stored arrays | in regime on both shells | the inputs of C4 (b) |

The tolerance of C4 (c) is under half of the gap (0.41) to the nearest wrong answer, 0; the plan-time reads of this field lie 0.008 from $1 - \sqrt2$ at spacing 0.5 and at most 0.0032 at 0.375 and 0.25. At the center node $\hat n \cdot x = 0$, so the lift keeps whatever sign the eigen-solver returns there, and two solvers may differ; Q does not see it (§ 1), the shares near the center can. A share that misses C1 only through that node is re-read on the next try with the node's sign flipped, both values printed.

No charge was computed for this plan on any chain state of M5.51. The plan audit computed it on the seed, on the author's four stored arrays and on the constructed fields, with the values of § 1; the drafter computed it on C4 (b) and C4 (c) only.

**Order, cap, cost.** C3, C4 and the regime row first, then C1 and C2, before any chain state is read. Try cap three per control; between tries only an input or a defect of a script may change, never a threshold, and each change is logged. A failure of C1, C3, C4 or the regime row withholds every Q outcome (Q tabled); a failure of C2 withholds the Q outcomes and is reported with the difference.

### 3. The words, frozen

**General rules, for every read.**

| Rule | Statement |
| --- | --- |
| The state | M5.51's midpoint chain `a` of the grid: from the radial hedgehog, the start T2 names, with M5.51's registered noise of $10^{-4}$ (its limit 8; the noise is M5.51's choice) |
| The second start vetoes | chain `b` (from the author's end-weight array on G1, G2, G4; from `G4_mid_a` interpolated on G5) never gives a word; when its end value, under the same rule without a second step, gives a different word, the entry reads "no label (starts disagree)" |
| Qualifying | CONVERGED, SMOOTH, one ring with `line_regime` and `halfturn_all` true, K5 met, as M5.51 § 8 records them; a state that does not qualify gives "no label (not qualifying)" |
| The second step | M5.51's withholdings carry over unchanged (on `G2_mid_a`: U, $R_L$, N/U, the class and one ring); a read whose input or class M5.51 withholds gives "no label (second step)". The new reads are taken at M5.51's earlier state of the chain: $R_A$ within 0.001 (the bar of $R_L$), Q/4π at both surfaces within $10^{-4}$. A Q outcome also needs the one-ring read to pass its second step, since the seed reads the same with no ring. A failed step gives "no label (second step)", the two values printed |
| The weight | the midpoint weight, the one T2 names; the end-weight states are tabled with no word |
| Orientation | T2's ranges name no lattice orientation, so no word requires one. On G2, M5.51 left the orientation unresolved (two midpoint states at different angles, U within $1.4 \times 10^{-6}$ relative, $R_L$ 0.0021 apart at the end and 0.0020 after the continuation of M5.51 § 17); a word there would apply to either, and the veto covers a disagreement |

**Per read.**

| Read | INSIDE | OUTSIDE | No label | Budget and its reason |
| --- | --- | --- | --- | --- |
| U (G1, G2, G4) | $\lvert U/U_{T2} - 1 \rvert \le 0.00966$ | $\lvert U/U_{T2} - 1 \rvert \ge 0.01034$ | between; G5 tabled, no entry | $3.4 \times 10^{-4}$: M5.51's second-step bar on U ($10^{-4}$) and the rounding of $U_{T2}$ = 20.50, 20.53, 21.50 to two decimals ($2.4 \times 10^{-4}$ relative) |
| Radius (G1, G2, G4, G5) | every available read inside [2.76, 2.86] by more than its budget | every available read outside it by more than its budget | otherwise, "no label (readers)" | $R_L$: 0.01 ($\tau_L$ of M5.51). $R_A$: the largest $\lvert R_A - R_0 \rvert$ of M5.51's K4 (a) at the spacing, for the class (box axis 0.020 at 0.5 and 0.033 at 0.375; body diagonal 0.089 and 0.048; neither, the largest over the three orientations, 0.123 and 0.048), recomputed by the builder with `m5_51_reads.py`, to agree within 0.001. $R_{\mathrm{rel}}$: 0.015 |
| Q/4π (G1, G2, G4, G5) | INSIDE (fixed by the held faces): $Q(L)/4\pi \ge 0.992$ and $Q(\ell_2)/4\pi \ge 0.9944$ | $Q(L)/4\pi \lt 0.988$ and $Q(\ell_2)/4\pi \lt 0.9856$, kept as written; it cannot be met in regime (§ 1) | otherwise, or out of regime | 0.002 at the box, 0.0044 at $\ell_2$, § 1 |
| N/U (G1, G2, G4, G5) | in [0.941, 0.999] | under 0.939 or above 1.001 | otherwise | 0.001, the tolerance to which M5.51's K4 (c) reproduced the posted values |

The available reads of the radius are $R_L$, $R_A$ when its node set has at least 4 nodes, and $R_{\mathrm{rel}}$ on G1, G2 and G4.

**The radius: three readers in the word, none chosen.** The post names no reader. M5.51 § 9 shows three readers giving three sets of radii on the author's own arrays: $R_L$ 2.7846, 2.7934, 2.8215, 2.9087; $R_A$ 2.7978, 2.7800, 2.8325, 2.8504; the release's 2.73, 2.74, 2.83, 2.82, apart by up to 0.09, against a range 0.10 wide. A word resting on one reader would choose the author's reader for them, so a word needs every available reader to agree, in both directions. The release's reader is not defined in the archive and cannot run on a new state; it enters through the offset read

$$
R_{\mathrm{rel}} = R_L(\text{state}) - \big[ R_L(A_g) - r_g \big]
$$

with $A_g$ the author's stored array of the grid and $r_g$ the release's radius (G1: 2.7846 and 2.73; G2: 2.7934 and 2.74; G4: 2.8215 and 2.83; M5.51 §§ 7, 9). Its assumption, the plan's: the offset between the two readers on the author's end-weight array carries to the midpoint state of the same grid. Its budget is the 0.01 of $R_L$ and the 0.005 of the release's rounding. It is not defined on G5, where two readers decide. A fourth reader, that of the archive's charge script (the least gap on one half-plane about the z axis, refined by a parabola), reads 2.7999, 2.7725 and 2.8591 on the author's three files against $R_L$ 2.7846, 2.7934 and 2.8215. It is tabled as a fourth reader, on the author's arrays and on the box-axis states after the lattice rotation that puts their normal on z, and does not enter the word: it is defined for a z-aligned ring on one half-plane and has no known-answer control.

**N/U.** M5.51's read, with the node weight $s(M)$ in the time part for both link weights (its limit 5). The archive's Laue script `laue12_now.py` takes $s$ at the nodes, so the read is the author's as far as the archive shows. $N_{\mathrm{vir}}/U$ is tabled with no word.

**The tally.** Fifteen entries (U on three grids, the other three reads on four), counted per read and in total in four columns: INSIDE, INSIDE (fixed by the held faces), OUTSIDE, no label. The plan gives no word to the rung as a whole, since the author did not state how the four reads combine; the 80 percent confidence is tabled, not scored.

**Tabled with no word.** The same reads on G3 and G6 and on every end-weight state; the author's four stored arrays (Q at both surfaces, the regime counts, the profile, the shares, beside the archive's values); the seed's values of § 1 beside every Q outcome; the shares of every state against the posted 93 to 96 percent.

**The plan's choices, named as such in the findings.** Chain `a` as the state and chain `b` as a veto; the midpoint weight; the three radius reads in the word, $R_{\mathrm{rel}}$ and its assumption, the rule that all agree, and the fourth reader left out of the word; every budget; the node weight of N; the second surface at $L - 2h$; the regime and its test on lift flips; the bars of the new second steps; the separate Q outcome; the tally with no rung word; the distance to M5.51's fitted circle for the shares.

### 4. Blindspot pass, consumers, research body, estimate

| Family or unknown | Where it lives here | Guard |
| --- | --- | --- |
| Unvalidated instrument | the Q read; the regime test; the archive's distance; the budgets of $R_A$ | C1 to round-off on a field the archive read; C4 with the answers 1 and $1 - \sqrt2$, each with its failing input; the regime test with known answers in and out of regime; C3 on every field; $R_A$'s budget from its own known-answer record |
| Artifact read as result | the spacing sets ρ's nodal profile on the line; the stop sets a value | the shares carry no word; the second step per read; Q at two surfaces |
| Unfalsifiable check | ⭐ Q/4π is fixed by the two outer layers (the outer held, the next free), and the seed with no ring reads the same values (§ 1); U on chain `b` descends from the author's array, whose fixed-field midpoint energy equals $U_{T2}$ to the digits given, so it lies at or under $U_{T2}$ by construction; a regime test on $\tilde n \cdot x$ could not fail | Q has its own outcome, INSIDE (fixed by the held faces), never counted as INSIDE, with the seed's values beside it; the ring's content lies in the shares, the profile and M5.51's one-ring and half-turn reads; the U word is stated as testing only that relaxing under the midpoint weight lowers U by less than 1 percent from the fixed-field value; the regime test on lift flips, with known answers both ways |
| Resolution mistaken for accuracy | Q printed to 16 digits in the archive; radii from readers 0.05 to 0.09 apart | a budget beside every rule; Q to four decimals; radius words need every reader in the word |
| Ceiling-to-point creep | "within 1 percent", the 80 percent confidence, M5.51's withheld reads | the confidence not scored; a withheld read stays withheld, its numbers printed |
| Wrong diagnostic for the regime | the archive's z-axis distance on rings of other orientations; Q where a shell holds a branch sheet | that distance only in C1 and beside the shares of the rotated G1 array; the fitted circle elsewhere; the regime test of § 1 |
| Post hoc choice | U, the radius and N/U known before § 3; M5.51's plan public before T2 | the order of events stated with both public marks; the words already fixed printed before the run; each choice the stricter in both directions, and named |
| Choosing among starts and states | chain `a` or `b`; the two G2 orientations | chain `a` gives, chain `b` vetoes; nothing averaged |

**Consumers.**

| Sub-result | Who reads it | The instrument check it needs |
| --- | --- | --- |
| The Q port, its controls, the regime test and the shares | the results post of the repo owner on discussion #186; any later read of the charge on the loop | C1 to C4 and the regime row |
| The words of § 3 and the tally | the same post, beside M5.51's words | the numbers check and the audit |
| The scorecard cells of M5 | only a later ruling of the repo owner | none from this plan |

**Research body.** This document (plan, findings, review). Scripts in [`../scripts/`](../scripts/): `m5_54_charge.py` (the read of § 1 and the controls of § 2, with `--selftest`) and `m5_54_score.py` (the reads on the stored states, the words of § 3, the tables, the summary and the plots); both import `m5_51_reads.py` read-only, for $R_A$, the line reader at earlier states and the constructed rings of K4 (a), and no file of M5.51 is modified. The summary `m5_54_summary.json` in [`../data/`](../data/) (tracked); control and per-state records in `../checkpoints/m5_54/` (local-only); plots `m5_54_*.png` in [`../plots/`](../plots/). The data manifest [`_DATASETS.md`](../data/_DATASETS.md) is regenerated at the end.

**Parallel shape and estimate.** One single-threaded process (the thread variables set to 1 before NumPy loads), sequential: the controls on constructed fields at four grids, then 24 end states, their earlier states and the four stored arrays, each one eigen-decomposition and a few stencils on at most $49^3$ nodes, with no iteration. The reads are short, so a pool would cost more than it gains; this is the plan's documented choice. No GPU: the read needs double precision. It takes one slot of the twelve. Sub-agents: one builder on Opus (the two scripts, the controls, the reads, the findings); one numbers check on Sonnet; one adversarial audit on Opus. The user's touchpoints: the commit of this plan and the review. Autonomy: unattended between them, unless a control reaches its try cap.

**Not computed.** Any relaxation; the other rungs of the post (T1; T3 is the scope of [M5.53](m5_53_task_details.md); T4 overlaps [M5.52](m5_52_task_details.md)); the archive's split of $b$ into its longitudinal and transverse parts; Q on free faces; the release's own radius reader; a spacing of 0.25 on half-box 9; any limit in an infinite box; a word on the shares.

**Deviations.** Anything forced off this plan takes the conservative option (the word withheld rather than given, the stricter of two readings) and is logged in this document as it happens.

**Plan audit (2026-11-23, before any read of a chain state).** An independent pass in a fresh context derived the telescoped flux and the topology of the lift, ported the read with its own NumPy code, reproduced the archive's values on the rotated G1 array to $1.1 \times 10^{-16}$ and C2 to $1.5 \times 10^{-4}$ and $6 \times 10^{-7}$, ran every constructed control, and recomputed the words that M5.51's numbers already fix. It found no blocker. It computed no charge on any chain state. The items folded above:

| Item | Fold |
| --- | --- |
| Q cannot read OUTSIDE in its regime, and its values are known | the outcome INSIDE (fixed by the held faces) in its own tally column; the seed's and the author's arrays' values in § 1; the author's reading of the number named and not supported |
| The regime test on a sign that is arbitrary or non-negative by construction | the test on lift flips $\tilde n_i \cdot \tilde n_j \lt 0$, the branch share on the same definition, with known answers in and out of regime |
| Words fixed by M5.51's numbers | tabled beside the order of events |
| M5.51's plan public before the registration | the push of `e6da01ff` named as a public mark; claim (c) corrected |
| T2's U values and radii equal scaled numbers of the author's | stated as arithmetic in claim (b); the content of the U word in the blindspot pass |
| The budget of $R_A$ for the class "neither" | the largest over the three orientations at the spacing |
| A reader whose input M5.51 withholds, a state that does not qualify | "no label (second step)" and "no label (not qualifying)" |
| Smaller items | C4 (b) made the out-of-regime control; C4 (c)'s failing input and tolerance; the budget at $\ell_2$; the rounding of $U_{T2}$; the fourth reader tabled as such; the citations, the file names and the wording of fact (d) |

The plan and its audit were written by instances of one model in separate contexts.
