# M5.59: the runs asked with capstone 4.7: a Coulomb-phase test on CP³, the linear cost of a unit Chern line

> Opened 2026-11-26 from the author's post on discussion #186 that releases capstone 4.7 ([discussioncomment-19930565](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19930565), 2026-11-26 18:48 UTC). Roadmap row: [`m5_roadmap.md`](../m5_roadmap.md). Thread record: [`m5_32_convo.md`](m5_32_convo.md).

## TASK PLANNING (2026-11-26; opened, not staged: the plan is written and audited before any go)

**Scope.** The post asks two runs of OpenWave, "if they fit your queue". Both sit in the coframe picture of capstone 4.7 (the DOI of all versions is [10.5281/zenodo.23994958](https://doi.org/10.5281/zenodo.23994958); the post names items 29 to 33), and neither sits in the actions of the M5 record.

| # | The run asked | What the post states | On record on OpenWave |
| --- | --- | --- | --- |
| 1 | An independent Coulomb-phase test of a pure Berry-plaquette CP³ lattice action with a fugacity on the charge of each cube, plus a short-range SU(4)-invariant disordering weight on neighbor overlaps | The author's own second round, on a $12 \times 6^3$ lattice with the fugacity from 1.4 to 3.8, found the charges suppressed and mixing between hot and cold starts, the line nearly ordered (order 0.24 to 0.28 against 0.067 for independent lines; static transverse flux ratio 2.3 against 1 for a photon), and "No Coulomb window yet". The score asked: order times $\sqrt{N}$ across two sizes, a flat transverse flux, and agreement of the charge density between starts | No CP³ field, and no lattice action sampled at a fugacity. The plaquettes of the M5 record are readers of the line of a relaxed field ([M5.45](m5_45_task_details.md)), not terms of an action |
| 2 | A replication of a linear cost for a unit Chern line under a quadratic stiffness $K_2 = f_\pi^2$ | The cost is $\pi K_2$ = 136 MeV per fm of radius, from the bound $\int \lvert D_\Omega z \rvert^2 \ge 2 \pi \lvert q \rvert$ on each sphere; without that stiffness B = 1 is left with no energy floor | No Chern line and no such bound |

The action of run 1 as the post writes it, the third term given there as an example with $c_0$ between 0.25 and 0.5:

```math
\beta_f \sum_{\square} \left( 1 - \cos f_{\square} \right) + \mu_m \sum_{\mathrm{cubes}} \lvert q \rvert + w \sum_{\langle x y \rangle} \left( \lvert z_x^{\dagger} z_y \rvert^2 - c_0 \right)^2
```

**What the capstone document adds (version 4.7 as deposited on 2026-11-26, listed in [`_CITATIONS.md`](../../theory/_CITATIONS.md); items 29 and 33; evidence, not resolution).** For run 1, item 29 reports a first screen of forty runs on $8 \times 6^3$ with the native action: two phases only, a disordered plasma of Chern monopoles and an ordered phase free of them, and no window that is disordered and free of monopoles. Item 33 reports the second round, which adds a cost on the monopoles: all 39 of its rows fail the kit's own gates; the charge density falls to zero near a fugacity of 3.5 while the order of the line stays at 0.24 to 0.28 whatever the fugacity; hot and cold starts agree on the charge density and disagree on the order. The action asked in the post is what the document names as the minimal change of the native one: a review it reports finds that on CP³ the commutator quartic of a rank-one projector holds a positive part besides the Berry flux, which favors order, while on CP¹ it is the Berry flux alone, and it names independent compact links as a fallback. The document also holds a criterion frozen on the author's side before the reading (a finite-size plateau across three sizes in place of a fixed cut on the order) and a registered prediction that a window found at one spacing shrinks or vanishes at half the spacing; the score asked of OpenWave is the one in the post.

For run 2, item 33 states the cost in a round on a coframe quotient: a pion stiffness that is not projected puts $K_2 / 2$ on the charged line, so by the Berry bound an isolated charge costs at least $\pi K_2$ = 135.9 MeV per fm of radius with $f_\pi$ = 92.4 MeV; projecting the stiffness off the charged directions removes the cost and leaves B = 1 without a uniform energy floor. Given the bound and $K_2 = f_\pi^2$, the number is arithmetic: $\pi f_\pi^2 / \hbar c$ = 135.9 MeV per fm with $\hbar c$ = 197.327 MeV fm, checked by hand at the filing. The bound and the placement of $K_2 / 2$ on the line were not checked.

**Evidence, not resolution.** The numbers and the analysis in the post and in the document are the author's side's; apart from the arithmetic above, nothing of them was recomputed on OpenWave.

**Open at the plan.** Read in full first: items 29 to 33 of capstone 4.7 and its sections on the coframe. To be fixed by the plan, each with a known-answer control: the definitions of $z$, of the plaquette flux $f_{\square}$ and of the cube charge $q$ on the lattice; the sampler and how thermalization is shown (the agreement between hot and cold starts is itself one of the scores asked); the order parameter and the transverse flux as readers, with a case where each returns the other answer (independent lines, a free photon); the two sizes, the couplings and the range of the fugacity; for run 2, the configuration space, the sphere and the units that carry 136 MeV per fm, and whether the replication is a derivation, a numerical minimum, or both. Run 1 is a sampling of a lattice action, a method the M5 record does not hold: its parallel shape and cost are stated at the plan.

**Definition of done.** Written with the plan.

**Gating.** A plan with its audit, and the user's go.

**Blindspot pass, consumers, research body.** Written with the plan.
