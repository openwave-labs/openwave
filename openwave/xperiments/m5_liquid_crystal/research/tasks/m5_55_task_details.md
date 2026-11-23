# M5.55: the author's rung T4 scored on the stored driven fields of M5.52

> Staged 2026-11-23. Roadmap row: [`m5_roadmap.md`](../m5_roadmap.md). Thread record: [`m5_32_convo.md`](m5_32_convo.md). It scores the rung T4 that the author posted on 2026-11-23 20:05 UTC ([discussioncomment-19840001](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19840001)) on the lock-in fields at ω = 5.5 that the jobs of [M5.52](m5_52_task_details.md) store, with the read, the pieces and the budget of that task. It is a desk read: no job runs in this task. The cell it reads for: the quantum wave equation (Klein-Gordon) in the M5 table of [`MODELS.md`](../../../../../MODELS.md); this plan changes no cell.

## TASK PLANNING (2026-11-23; staged)

**Scope.** In their post of 2026-11-23 20:05 UTC (the last row of the [thread record](m5_32_convo.md)) the author states four rungs with predictions registered, by their statement, on 2026-11-23 at 20:53 CET, before any run. Rung T4, as the thread record gives it: a driven wave at ω = 5.5 near the core of the unit of M5.42, the phase read along the axis and in the ring plane; the phase advances inside the bands, the wave is evanescent beyond the outer edges (axis 8.4 to 8.8, plane 9.7 to 10.1), and the decay rate is √(m_eff² − ω²) within 15 percent (65 percent). Here m_eff is the local mass of their post of 2026-11-19 17:55 UTC, √(L₁ m² / I(r)), whose form on the background of OpenWave is F of [M5.49](m5_49_task_details.md) § 1. The plan of M5.52 was frozen before the post; its jobs drive ω = 5.5 on grids b, d and e, and its rules give no word to a literal phase or to a decay rate. This task scores T4 on the fields those jobs store, under rules frozen here:

| Entry | Clause of T4 | Read |
| --- | --- | --- |
| T4a | the phase advances inside the bands | the literal phase: a descriptor, no word (§ 3) |
| T4a′ | the same, in the form that can fail and tells a band from its outside | the status of the pieces inside the band of F: a word |
| T4b1 | the outer edges lie at axis 8.4 to 8.8, plane 9.7 to 10.1 | the read outer edge on three profiles, against the intervals: a word, expected "no label" (§ 3) |
| T4b2 | beyond them the wave is evanescent | the status of the pieces within 1.5 beyond the intervals: a word |
| T4c | the decay rate is √(m_eff² − ω²) within 15 percent | no word, decided at freeze by the budget (§ 3); the rate profile is a descriptor |

**Six limits, stated before the run.** (1) Every limit of M5.52 holds here: the time part of [M5.47](m5_47_task_details.md) § 1, the action of M5.42 and not that of the author's later posts, one spacing (0.25), the damping, the residual force and the source added by M5.52, and its limit (6) on the soft waves where the mid wave has decayed. (2) The source of M5.52 is two circles of the loop's radius at ±3 along its normal, inside both bands, not a source near the core as T4 states; the words rest on local reads of the piece estimator, which do not depend on where the source sits. (3) The literal phase cannot tell a band from its outside under the damping of the jobs (§ 3). (4) Within the region where the mid wave is read, a decay rate within 15 percent needs the local cutoff within about 1 to 2.5 percent of m_eff (§ 1), and M5.52's budget at ω = 5.5 is wider than the margin such a word needs on every piece: T4c carries no word, a decision made at freeze from the budget, not from a field. (5) Not blind, by grid (§ 5): grid b's fields had been read under M5.52's rules before this plan was written; grids d and e had been read by no one; M5.49 had read the cutoffs on the same pieces of grids d and e from free noise, and the expectations of § 6 come from that record. (6) The author's wording fixes neither the speed, nor the offset, nor where the rate is read, nor whether the intervals locate the edge or only bound where evanescence begins; the readings this plan adds are named in §§ 1 and 3.

**Definition of done.** The prediction file `m5_55_predicted.json` written and hashed before any read of grids d and e; the controls of § 4 each met, or reported with the exact failing check; T4a′, T4b1 and T4b2 each carrying one word, or "no label" with the rule that withheld it, per line and grid; the descriptors of § 3 tabled; the expectations of § 6 scored against the outcome; the findings embedding, per line of grid e, the profile of y with its three edges against the intervals, the rate profile κ_A and κ_P with ε beside κ_F and κ_C, the slope of the modulus and the literal phase; a record audit in a fresh context and a numbers check on a different model, each with its own code on the arrays; the doc checker clean.

**Gating.** The repo owner's commit of this plan, which freezes it and is the go for its reads, under the pre-flight already armed for M5.51 and M5.52. The order the record must show: the commit; `m5_55_predicted.json` written and its hashes in this task's run log; then the first read of grid d or e by either task. The reads run after M5.52's last job ends.

**Research body.** This document (plan, findings, review); in [`../scripts/`](../scripts/): `m5_55_reads.py` (the prediction file, the controls, the reads, the words, the descriptors and the plots). It imports [`m5_52_reads.py`](../scripts/m5_52_reads.py) and calls only its pure functions (`ratio_read`, `read_field`, `joined_pairs`, `edges0`, `bin_of`, `eps_of`, `kappa_bar`) and its read-only loader `load_lockin`; it never calls `read_job`, `check_hashes`, `words`, `control_c12` or any function that writes into M5.52's tree. Its inputs: the lock-in fields `lockin_w1.npz` and `lockin_w2.npz` of jobs Ge, Gd, Gb and VD1 to VD4, opened read-only in M5.52's job folders; the hashed prediction `m5_52_drive_predicted.json`; and the summary that M5.52 writes, `../data/m5_52_summary.json`, whose SHA-256 the run log records and which supplies b_γ (its C11 entry `b_gamma`), b_vac (its C9 entry `b_vac`) and M5.52's control outcomes. A term of ε missing from that summary makes every entry it enters "no label (budget input missing)". Geometry from [`m5_49_predict.py`](../scripts/m5_49_predict.py). Outputs: `../data/m5_55_predicted.json` and the summary `../data/m5_55_summary.json` (tracked, each under 1 MB); plots `m5_55_*.png` in [`../plots/`](../plots/); the run log and every intermediate file in `../checkpoints/m5_55/` (local-only, gitignored). Nothing is written under M5.52's folders.

### 1. What the decay rate is on OpenWave's equations

**The local law with damping.** For the mid wave about a point, M5.52 § 3 gives, with m the local cutoff, c the local speed along the wave and the uniform damping γ = 0.3 of the jobs:

$$
c^2 K^2 = \omega^2 - m^2 + i \omega \gamma = y + i \omega \gamma , \qquad y = \omega^2 - m^2
$$

A steady field at ω varies locally as $e^{i K s}$ with the root of positive imaginary part (the convention of M5.52 § 1, $M - M_0 = \operatorname{Re}(A e^{-i \omega t})$, under which a wave leaving the source has a phase that grows with s). Where the cutoff lies above ω (y < 0):

$$
\operatorname{Im} K = \frac{1}{c} \sqrt{\frac{\sqrt{y^2 + \omega^2 \gamma^2} - y}{2}} , \qquad \operatorname{Re} K = \frac{1}{c} \sqrt{\frac{\sqrt{y^2 + \omega^2 \gamma^2} + y}{2}}
$$

Im K is the decaying part and Re K the phase part; Re K > Im K exactly when y > 0. With $r = \omega \gamma / \lvert y \rvert$, $\operatorname{Im} K = (\sqrt{-y} / c) D(r)$ with $D(r) = \sqrt{(\sqrt{1 + r^2} + 1)/2}$, and $\operatorname{Re} K = \omega \gamma / (2 c \sqrt{-y})$ to first order in r.

**The author's expression and the terms that separate it from the physical rate.** The author's law has speed 1 and no loss, so its rate is $\kappa_{\mathrm{auth}} = \sqrt{m_{\mathrm{eff}}^2 - \omega^2}$. On OpenWave's damped equations:

$$
\operatorname{Im} K = \kappa_{\mathrm{auth}} \times \frac{1}{c} \times D(r) \times \sqrt{\frac{m^2 - \omega^2}{m_{\mathrm{eff}}^2 - \omega^2}}
$$

| Term | Size here | Where it comes from |
| --- | --- | --- |
| 1/c | 1.06 to 1.08 for c² of 0.86 to 0.89; the two speeds of a direction are not resolved | the slopes of the bins of M5.47 § 11 at spacing 0.25 |
| D(r) | 1.010 to 1.062 for $\lvert y \rvert$ from 5.8 down to 2.2 at ω = 5.5 (ωγ = 1.65) | the damping of the jobs |
| the offset of m under m_eff | $d \ln \kappa / d \ln m = m^2 / \lvert y \rvert$, about 6 to 15 beyond the outer edges inside M5.52's domain, so one percent in m is 6 to 15 percent in κ | C, the bin cutoffs of M5.47 § 11 as hashed, lies 2.5 to 3.9 percent under F over the 65 pieces with q̄ under 2.25 on grids b, d and e, all lines; the read cutoffs of M5.49 on grid e beyond the outer edges lie 1 to 2.5 percent under F |
| WKB | the local rate is the decay of the amplitude only where $w = \lvert d\kappa/ds \rvert / \kappa^2$ is small | a cutoff that changes along the line |

**The rate read without the damping.** The ratio of M5.52 § 3 gives on a piece $y_i = \omega^2 - m^2_{\mathrm{read}}$ and $c^2_{\mathrm{read}}$, exactly for one local law. From them:

$$
\kappa_{A,i} = \sqrt{-y_i} , \qquad \kappa_{P,i} = \sqrt{- \operatorname{Re} Z_i} = \frac{\sqrt{-y_i}}{c_{\mathrm{read}}}
$$

κ_A is the rate in the units of the author's law (speed 1, the convention of M5.52 for Φ and κ̄); κ_P is the rate an undamped field of OpenWave would show, the author's own read in their run of 2026-11-19 (the decay of the modulus, no loss inside their box).

**What the expression leaves unspecified.** Each reading is tabled; none carries a word (T4c, § 3).

| Unspecified | Readings tabled |
| --- | --- |
| The background of m_eff | F of the hashed file `m5_52_drive_predicted.json` (the author's rule on OpenWave's background); the author's own profiles are not used |
| The offset of the cutoffs | against $\kappa_F = \sqrt{m_F^2 - \omega^2}$ (λ = 1) and against $\kappa_C = \sqrt{C^2 - \omega^2}$, the formula with OpenWave's offset, which the author accepted for the edges on 2026-11-21 and which their T4 intervals carry (8.38 to 8.8 and 9.74 to 10.1 are their registered edges and up to about 5 percent outside) |
| The speed | κ_A (speed 1) and κ_P (physical, no loss) |
| Local or mean rate | per piece, and $\bar\kappa$ of M5.52 over the read pieces within 1.5 of the read outer edge |
| 15 percent of what | the ratios $\kappa_{A,i} / \kappa_{X,i}$ and $\kappa_{P,i} / \kappa_{X,i}$, X = F or C, printed with their budget |

### 2. The reads

**Read Z, the piece estimator** of M5.52 § 3 on window 2, unchanged: $Z_i$, $y_i$, $c^2_{\mathrm{read}}$, ε_i and the status of each piece (PROP when y − ε ≥ 0, EVAN when y + ε < 0, MARGIN otherwise), from `read_field` and `eps_of` on the stored fields (tone index 1, ω = 5.5; `Ar + i Ai`, six components, projected on W), with the ε terms of M5.52's summary. Every word of this task rests on read Z.

**The budget at ω = 5.5, known at freeze.** From the outcome of C11 that § 5 discloses (no piece of grid b in bin I, 0.48 in bin II, 3.44 in bin III) and the rule of `control_c11` (a bin with fewer than five pieces takes the largest value of all bins), the damping term of ε is 2 × 3.44 = 6.88 where $\lvert y \rvert$ < 1.65 (bin I) or $\lvert y \rvert$ ≥ 6.6 (bin III), and 2 × 0.48 = 0.96 where 1.65 ≤ $\lvert y \rvert$ < 6.6 (bin II); the floor (0.30), the windows, the filter and the vacuum term can only raise ε. So ε ≥ 0.96 on every read piece at 5.5. A piece with 0 < y < 1.65 or 6.6 ≤ y < 6.88 cannot be PROP; a piece with −1.65 < y < 0 or −6.88 < y ≤ −6.6 cannot be EVAN.

**Read S, the slope of the modulus: a descriptor, no word and no control of its own.** On every ray of a line (2 for A, 4 for PF, 4 for PD), the scalar a sampled at points h/2 apart by trilinear interpolation of its node values, $\kappa_S$ the least-squares slope of $- \ln \lvert a \rvert$ against s over the span of the T4b2 pieces of the line (9.5 to 10.5 on A, 10.5 to 11.5 on PF and PD), per ray, then averaged; tabled beside the mean of $\operatorname{Im} \sqrt{Z_i}$ over the same pieces. Its one check is the plan audit's exact lattice steady state of a blob source in the uniform vacuum with the held shell (below): in the evanescent case of VD1 the slope fell within 0.6 percent of that reference on A and PF and 0.3 percent on PD; in the propagating case of VD4 it read 3.4 and 4.5 percent high, of which 1.9 and 4.0 percent are images in the held faces of a half-box of 6; the phase slope of the lattice sits 2.4 percent under the continuum Re K. Its biases against Im K, none bounded by this plan:

| Bias | Size |
| --- | --- |
| Spreading | a pattern that spreads like a point source adds the least-squares slope of ln s over the span (0.3185 over s from 2 to 4.5 in the blob case, against 1/s̄ = 0.308) |
| WKB prefactor | $\kappa' / (2 \kappa)$ where the cutoff changes along the ray |
| Transverse structure of the tail | an angular order ℓ or a confinement $k_\perp$ steepens the modulus along a ray by about $k_\perp^2 / (2\kappa)$ or $\ell(\ell + 1) / (2 \kappa s^2)$; Z, which holds the whole Laplacian, does not show it |
| Geometry | Z averages a cone of radius 2.6 to 2.9 about the ray at s from 9.75 to 10.75; S reads on the ray |

**When a piece is NOT READ for T4b2 and the descriptors**, besides the rules of M5.52 (source, rms floor of 10⁻¹², c² outside 0.4 to 2.5, the soft rule):

| Rule | Bound | Reason |
| --- | --- | --- |
| Second window | $\lVert a_2 - a_1 \rVert / \lVert a_2 \rVert \le 0.02$ over the piece's nodes in D; for read S, $\lvert \ln \lvert a_2 \rvert - \ln \lvert a_1 \rvert \rvert \le 0.02$ at every sample of the span | on five samples over half a unit, 2 percent moves a least-squares slope by 0.096 in the worst case |
| Purity | the power of $A_2$ outside W under its power on W on the piece (the purity clause of M5.52 § 6) | beyond an edge the mid wave decays and a soft wave, which does not, takes over the projection and can read PROP with a wrong mass (M5.52 limit (6)) |
| The faces | $e^{-2 \kappa_F d_f} \le 0.01$, d_f the distance from the piece's outer end to the nearest face | the held shell reflects the decaying field |

**The reach of the read at 5.5.** The near-threshold domain of M5.52 (m_F² ≤ 1.2 ω²) is its closed rule, set at ω = 4, where it kept the axis stretch under 1.7 e-folds of predicted decay and rejected a bound reaching 2.61. At ω = 5.5 it does not carry that reason: from the outer edge of F to the end of the domain on grid e the predicted decay is 4.6 (A), 2.8 (PF) and 3.1 (PD) e-folds on F, and 3.6, 2.05 and 2.1 on the cutoffs that M5.49 read. M5.52's regime table limits its decay descriptor to within 1.5 of an outer edge. So the pieces of T4b2 stop 1.5 beyond the interval, and pieces farther out are tabled with the break of regime stated.

### 3. The entries, frozen

Words are given on grid e for T4b1 and T4b2, on grids d and e for T4a′; grid b carries tables only (§ 5). Every statement must be the same on window 1, otherwise it is no label. The plane is PF, the author's +x, with PD read against the same numbers as this plan's addition (the rule of M5.49): a plane entry is HELD when one line is HELD and none NOT HELD, NOT HELD when either is, no label otherwise.

**T4a, the literal phase: no word, a descriptor.** The phase $\varphi(s) = \arg a$ along each ray and its gradient are tabled. The clause can fail as written (a standing or an inward pattern inside a band gives a flat or falling phase), but its holding says nothing about a band: under γ = 0.3, Re K > 0 on both sides of a cutoff (§ 1; M5.52's Scope, fact 3); in the evanescent vacuum of VD1 (y = −3.025) Re K = 0.4587, and the audit's lattice steady state of a blob source there advances in phase at 2.4 percent under it. A word on it could not fail on the evanescent side.

**T4a′, inside the band the wave propagates.** By § 1, the phase part of the wave exceeds its decaying part exactly when y > 0. The scored pieces are the pieces inside the band of F at ω = 5.5 (y_F(1) > 0) whose centers lie at least 2Δ = 1 from each of its edges on the pieces form, read by the geometry of M5.49 and not excluded by the source, fixed from the prediction file:

| Grid, line | Scored pieces (centers) |
| --- | --- |
| e, A | 2.25 to 7.25 (11 pieces) |
| e, PF | 4.75 to 8.25 (8 pieces) |
| e, PD | 2.25, and 4.75 to 8.25 (9 pieces) |
| d, A | 2.25 to 8.25 (13 pieces) |
| d, PF | 4.75 to 8.25 (8 pieces) |
| d, PD | 2.25, and 4.25 to 8.25 (10 pieces) |

HELD: at least nine in ten of them read, and every read one PROP. NOT HELD: at least nine in ten read, and at least one EVAN, named. No label otherwise. By the budget above, a scored piece whose read y lies between 0 and 1.65, or between 6.6 and 6.88, cannot be PROP, and its line is then no label.

**T4b1, the outer edge in the author's interval.** The interval is 8.4 to 8.8 on A and 9.7 to 10.1 on PF and PD (grid e). Its window is the set of pieces with centers from 8.4 − Δ to 8.8 + Δ (8.25, 8.75, 9.25 on A) and from 9.7 − Δ to 10.1 + Δ (9.25, 9.75, 10.25 on PF and PD), covered when every one is read. An outer edge is that of M5.52 § 3 (`edges0`: a profile crossing 0 from positive to negative as s grows, between read pieces no more than 2Δ apart). On each of the three profiles y, y + ε, y − ε the outer edge nearest the interval is taken. HELD: covered, and all three inside the interval. NOT HELD: covered, and all three outside it on the same side. No label otherwise: the three straddle the interval, or a profile has no outer edge. **Known at freeze:** with ε ≥ 0.96 everywhere, the three edges fall inside an interval 0.4 wide only where the read y changes by more than 2 × 0.96 / 0.4 = 4.8 per unit across it, against 2.6 (A) and 3.4 (PF) for F there; with ε = 6.88 on the pieces next to an edge, where $\lvert y \rvert$ < 1.65, the three edges spread over more than a unit (§ 6). T4b1 cannot be HELD under the known budget and is expected "no label"; it is kept because NOT HELD stays possible.

**T4b2, the field within 1.5 beyond the interval.** The scored pieces have centers from the interval's top plus Δ to its top plus 1.5, fixed now: 9.75 and 10.25 on A; 10.75 and 11.25 on PF and on PD (grid e). A scored piece is NOT READ unless it meets the purity clause and the second-window rule of § 2. HELD: every scored piece read and EVAN. NOT HELD: every scored piece read, and at least one PROP, named. No label otherwise. The pieces read by the geometry beyond the top plus 1.5 (10.75 and 11.25 on A; 11.75 to 13.25 on PD; none on PF) are tabled with their status, under the break of regime of § 2. Grids d and b hold no scored piece of T4b1 or T4b2. T4b1 and T4b2 are two readings of the wording ("the edge lies in the interval", "beyond the interval the wave is evanescent"), tabled apart and never merged.

**T4c, the decay rate: no word, decided at freeze by the budget.** A word within 15 percent on a piece needs its budget under the tolerance's margin: half the narrower side of the band in y that 15 percent in κ allows ($1 - 0.85^2 = 0.2775$), so ε ≤ 0.139 $\lvert y_X \rvert$. Inside M5.52's domain beyond the outer edges $\lvert y_F \rvert$ ≤ 0.2 ω² = 6.05, so the margin is at most 0.139 × 6.05 = 0.84, under the least ε of 0.96 at ω = 5.5; outside the domain the pieces lie in bin III, with ε = 6.88. No piece can carry a word on F or on C. The descriptor, per piece beyond the outer edge of F on every line of grids d and e: y, ε, the bin, κ_A and κ_P, κ_F and κ_C, the four ratios, $\operatorname{Im} \sqrt{Z_i}$, the purity ratio, the second-window ratio, and read S per line.

**The prediction file.** `m5_55_reads.py --predict` writes `../data/m5_55_predicted.json` from the hashed blocks of `m5_52_drive_predicted.json` (it checks their hashes and stops on a mismatch) and the base states, and opens no lock-in amplitude: per line and grid at ω = 5.5, y_F, y_C, κ_F, κ_C, w, the scored pieces of T4a′ and T4b2, the windows of T4b1, the T4b2 spans of read S, the exact vacuum values of § 4 and every bound of §§ 2 and 3. Its block hashes (the rule of M5.49) are written in this task's run log before any read of grids d and e; the reads take them as arguments and stop on a mismatch.

### 4. The controls

| Control | Input | The answer it must return | What a failure would look like |
| --- | --- | --- | --- |
| K0, M5.52 at ω = 5.5 | the controls of M5.52 as its reads report them | a word of this task at 5.5 on a grid only where M5.52's own cost rules leave its labels at 5.5 there, with one exception, stated with its reason in § 5: the clause of C9 on the read speed (c² within 1 percent of 1) is not inherited. Every other clause of C9 at 5.5 is (the cutoff of each vacuum case within its bound, no edge, the statuses), and so are C8, C10, C12 and the binned damping term of C11 inside ε | a field that M5.52's controls do not stand behind |
| K1, the prediction first | `m5_55_predicted.json`, the run log, the reads | the hashes in this task's run log before any read of d or e, beside the SHA-256 of `m5_52_summary.json`; the reads stop on a mismatch | a scored set or a bound written after a read |
| K2, read Z on the evanescent vacuum (reported, no cost) | VD1 of M5.52 at ω = 5.5, window 2 (m² = 33.275, y = −3.025), the 26 read pieces of A, PF, PD | κ_A and κ_P within 2 percent of √3.025 = 1.7393; $\operatorname{Im} \sqrt{Z_i}$ within 2 percent of 1.7987, the imaginary part of $\sqrt{-3.025 + 1.65 i}$ | a ratio that does not remove the damping or the speed. Not fresh: it is algebra on C9's per-piece y and c² on VD1, computable from numbers the freezing session had; the worst innermost piece puts κ_P near 2.0 percent low, at the bar. It guards the descriptor only |
| K3, the rules on the vacuum (inherited) | VD4 (y = +1.815), VD2 and VD3 (y = +15.21 and +13.29), and VD1, at 5.5 | on VD4, VD2, VD3: no outer edge, no piece EVAN, every read piece PROP; on VD1: no edge, every read piece EVAN | an edge or a band where the medium has none. It is C9's status clause on the same fields, known met when this plan was frozen |
| K4, the rules on made-up profiles | profiles of y with budgets, pieces with set amplitudes and powers | each outcome once: T4a′, T4b1 and T4b2 each HELD, NOT HELD and no label by every rule (the cover, the three profiles straddling, a profile with no edge, the nine in ten, the purity, the second window, window 1, a missing budget term) | a rule that returns one word |

**Tries and costs.** The try cap of each control is three; between tries only the input of a control or a defect of the script may change, never a bound of §§ 2 to 4, each change logged in the run log. What a failure costs: K1 or K4, no word on any grid; K0, as M5.52's rules say for its own labels, with the exception above; K2, nothing (reported beside the descriptor); K3, no word of T4a′, T4b1 or T4b2.

### 5. Grids, blindness and the order of the record

| Grid | Words | Tables | Blind |
| --- | --- | --- | --- |
| e (half-box 13, job Ge) | T4a′, T4b1, T4b2 | every descriptor of §§ 2 and 3 | not read by anyone when this plan was written; M5.49 read its cutoffs on the same pieces from free noise (limit (5)) |
| d (half-box 10, job Gd) | T4a′ | the rate profile beyond PD's edge, read S where the line reaches, the literal phase | not read by anyone when this plan was written; M5.49 read its cutoffs from free noise on d as well |
| b (half-box 8, job Gb) | none | the same tables | no: read under M5.52's own rules before this plan was written |

The fields of grid b had been read under M5.52's rules before this plan was written. The drafter of this plan opened no file of M5.52's job outputs, reads, logs or findings; it designed from M5.52's plan, the source of its scripts, its hashed prediction file and the closed records of M5.47 and M5.49. A hold keeps grids d and e unread. The record must show, in this order: the commit of this plan; `m5_55_predicted.json` written and its hashes in this task's run log; then the first read of grid d or e by either task. The evidence of that order: the commit and its push time (the push is the one public mark), the file times of M5.52's reads of Gd and Ge, the lines of its hash log, and this task's run log, all local. A read of d or e found before the commit makes every word of this task on that grid "no label (not blind)".

**Known when this plan was frozen, and by whom.** The drafter knew none of M5.52's results. The session that folded the audit and froze the plan did know the outcome of M5.52's controls on the vacuum jobs and on grid b, which had ended and been read under M5.52's rules: C9 was not met at ω = 5.5 on one clause, the read speed on the evanescent case VD1 (c² up to 1.0244 on its innermost pieces, where the field is smallest, against 1 percent; window 1 reads 0.92 to 0.93 there), with every other clause of C9 at 5.5 met (the cutoff of VD1 within 0.0498 against 0.3025; VD4, VD2 and VD3 met); and C11 at ω = 5.5 was met with a 90th percentile of 0.48 in bin II and 3.44 in bin III, no piece of grid b lying in bin I. No field of grid d or e had been read by anyone.

With that knowledge, K0 was changed in one place from the drafter's text, which inherited every clause of C9: the speed clause is not inherited. What is right in the equations: for one local law the ratio gives $y = \omega\gamma \operatorname{Re} Z / \operatorname{Im} Z$, which holds no speed, so the statuses PROP and EVAN, the edges, T4a′, T4b1, T4b2 and κ_A do not depend on $c^2_{\mathrm{read}}$. What the failed clause shows: c² up to 1.0244 on the innermost pieces of VD1 against 0.92 to 0.93 in window 1 means that Z is not of the single-law form on those low-field pieces (a transient or a mixture), and the single-law form is the premise that makes y free of the speed. What licenses not inheriting the clause is C9's separate cutoff clause, met on the same pieces (0.0498 against 0.3025): there the y that the words use was read correctly although the speed was not. The pieces of T4b2 are the low-field analog of those VD1 pieces, which is why the second-window rule and the purity clause gate them. The speed enters this task only through κ_P, a descriptor. Under M5.52's own rule the clause withholds every label of M5.52 at 5.5, and that stays so in M5.52's record. The change is this plan's choice, made after the control's outcome was known, and the findings name it as such; had the clause been inherited, this task would give no word on any grid. The same outcome of C11 sets the budget of § 2, from which T4c's lack of a word and T4b1's expected "no label" follow at freeze.

### 6. Registered before the run

Expectations, which no script uses, from the free-noise cutoffs of M5.49 (`wc` in [`m5_49_reads.json`](../data/m5_49_reads.json)) put in place of the read, with ε from § 2. (1) T4a′: every scored piece PROP on A and PF of grid e and on A, PF and PD of grid d (on d, PD at 8.25 sits near y = 6.5, close under 6.6); on grid e PD at 8.25 sits near y = 6.7, where it cannot be PROP, so PD is no label and the plane entry rests on PF. Expected: HELD on the axis and the plane of grids d and e. (2) T4b1, recomputed here with the auditor's numbers reproduced: the outer edges of y − ε, y and y + ε at 7.89, 8.835 and 9.63 on A and at 9.39, 10.07 and 10.67 on PF and PD (on PD, y − ε crosses also at 8.21 and y + ε also at 12.26); the three straddle each interval: no label on both. (3) T4b2: y of −2.81 and −3.88 on A, −2.21 and −3.90 on PF, −2.20 and −4.00 on PD, all EVAN against ε = 0.96: HELD on both if the purity clause and the second window let the pieces be read. If the driven field on these pieces is no purer than M5.49's free noise was on the axis at ω = 4 (0.87 to 0.88 of the power outside W on grid e), they are not read and the entry is no label. (4) T4c: no word. The descriptor on grid e: κ_A/κ_F rising outward, 0 on A at 8.75 (the read edge lies beyond that piece), 0.73 at 9.25 and 0.89 to 0.93 from 9.75 to 10.75; 0.62 to 0.90 on PF and 0.58 to 0.95 on PD from 10.25 outward; κ_P about 7 percent higher; against C, about 1.0 to 1.3. (5) K2 met at its bar; K3 met (known). Nothing in this section sets an icon: the plan reports, and the cell is re-read later by the repo owner, by a ruling.

### 7. Blindspot pass

| Family | Guard here |
| --- | --- |
| Unvalidated instrument | read Z: C9 of M5.52 on its vacuum cases (cutoff and status clauses), K3; the rules: K4; read S has no control of its own and no word, its one check being the audit's lattice solve |
| Artifact read as result | window 1 repeats every statement; the second-window rule on the low-field pieces; the binned damping term in ε |
| Unfalsifiable check | the literal phase holds on the evanescent side, so it gets no word (T4a); T4a′ can fail by one EVAN piece; T4b1 keeps only the word it can still give (NOT HELD); K2 and K3 named as not fresh |
| Resolution mistaken for accuracy | ε beside every y; T4c withheld at freeze because ε exceeds the margin 15 percent needs; the sensitivity $m^2 / \lvert y \rvert$ stated (§ 1) |
| Ceiling-to-point creep | the offset is a second column of the descriptor (κ_C), never a correction of κ_F; the expectations carry no word |
| Wrong diagnostic for the regime | T4b2 stops 1.5 beyond the interval, the reach of M5.52's decay descriptor; the e-folds at 5.5 stated; the purity clause where the mid wave decays; read S's biases listed |
| The author's wording read one way after the run | T4b1 and T4b2 tabled apart; F and C, κ_A and κ_P tabled side by side; all decided here |
| This plan's choices taken for the author's | named in the findings: T4a′ in place of the literal phase; the windows of the intervals and the three-profile rule; the 1.5 of T4b2; the second-window floor; the purity clause on T4b2; C as the offset; the speed clause of C9 not inherited |

### 8. Consumers, parallel shape, estimate, not computed

**Consumers.** The words of § 3 are read by the repo owner's post to the author on discussion #186: they need K0, K1, K3 and K4 met and the two audits. The Klein-Gordon cell reads them only through a later ruling of the repo owner. The descriptor of T4c is reusable by a later driven read with a smaller budget beyond the edges; it would need its own control of read S there.

**Parallel shape.** No job runs. The reads are one single-threaded process after M5.52's last job ends, sequential by choice (three labeled grids and four vacuum cases, each read once, by a desk pass over stored arrays), counted in the ceiling of twelve processes shared with M5.51 and M5.52.

**Estimate.** Machine: one desk pass over stored lock-in fields. Sub-agents: the builder on Opus; a numbers check on Sonnet; a record audit on Opus in a fresh context. The user's touchpoints: the commit of this plan, and the review. Autonomy: unattended from the commit to the review.

**Deviations.** Anything forced off this plan takes the conservative option (a word withheld rather than given) and is logged in this document as it happens.

**Not computed.** Any new job; a source near the core; the author's absorbing layer; the rate on the author's own profiles; T4c as a word; read S as a word, and a control of read S on data; ω = 4; the inner edges; the literal phase as a word; a word beyond 1.5 past the intervals; the two speeds of a direction apart; another spacing; the action of the author's later posts; the rungs T1 to T3.

**Plan audit (2026-11-23, before any read of grids d and e).** An independent pass in a fresh context, blind to M5.52's outputs, derived § 1 again, solved the exact lattice steady state of a blob source in the uniform vacuum with the held shell, and worked out from the tracked files and the numbers disclosed in § 5 which entries could carry a word. It found three blockers and six major items, folded as follows; the plan and its audit were written by instances of one model in separate contexts.

| Item | Fault | Fold |
| --- | --- | --- |
| B1 | T4b1's NOT HELD fired when the band of the three profiles holds the interval | NOT HELD only with all three edges outside on one side; no label when they straddle or a profile has no edge; stated at freeze that it cannot be HELD; expectation (2) recomputed |
| B2, M1 | with ε ≥ 0.96 and a margin of at most 0.84, T4c could carry no word | T4c has no word, decided at freeze; the rate profile is a descriptor; the two vacuum jobs, their controls, the wrapper script and the agreement gate removed |
| B3 | T4b2's NOT HELD was the false mode of M5.52's limit (6); the domain's reason does not hold at 5.5 | scored pieces within 1.5 beyond the interval, gated by purity and the second window; the e-folds at 5.5 stated |
| M2 | the reason for not inheriting C9's speed clause was incomplete | the single-law premise, the cutoff clause as the license, and the low-field analog stated (§ 5) |
| M3 | K2 and K3 presented as fresh validation | named as computable from known numbers and as inherited; K2 without cost |
| M4 | the build could write into M5.52's tree and named no source for the ε terms | only pure functions and the read-only loader; outputs under this task's folder; `m5_52_summary.json` hashed as the budget input |
| M6 | the empty bin I's effect on T4a′ was not stated, and its rule was circular | the PROP gaps stated; scored pieces fixed from the prediction file and listed |
| Minor | the second-window bound, grid d's blindness, the evidence of the order, the offset's subset, K2's window | corrected in §§ 1, 2, 4 and 5 |
