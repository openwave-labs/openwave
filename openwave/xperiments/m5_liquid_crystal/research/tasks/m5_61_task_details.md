# M5.61: the reads asked with capstone 4.8: a Chern density on stored states, a dual-link lattice, a frozen packet

> Opened 2026-11-27 from the author's post on discussion #186 that releases capstone 4.8 ([discussioncomment-19959770](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19959770), 2026-11-27 19:14 UTC). Roadmap row: [`m5_roadmap.md`](../m5_roadmap.md). Thread record: [`m5_32_convo.md`](m5_32_convo.md).

## TASK PLANNING (2026-11-27; opened, not staged: the plan is written and audited before any go)

**Scope.** The post sets a road for the single field of capstone 4.8, whose first step (stage 0) chooses the frame of the charge: topological, with charges as Chern classes and monopoles of a dual link, or Noether, with charges from a generator. It then names three places "where OpenWave could decide most", and asks that a plan be posted for any that fits, so that the author registers predictions before the reads.

| # | What the post asks | What the post states | On record on OpenWave |
| --- | --- | --- | --- |
| 1 | The test of stage 0 on stored configurations: in the relaxed B = 1 and ring states of the tensor, the Berry-flux (Chern) density of the time-like eigenvector, and whether any neutral state carries it distributed, zero in total, and rotating with the frame | The topological frame needs a neutral B = 1 core that carries such a density, giving the neutron's $r_n^2$ = −0.1155 fm² and $\mu_n$ = −1.913 $\mu_N$; the author's present core has none, so that frame "fails as it stands" | Relaxed rings and pairs of rings of a real symmetric 3 × 3 field ([M5.51](m5_51_task_details.md), [M5.56](m5_56_task_details.md), [M5.57](m5_57_task_details.md)); states of the real symmetric 4 × 4 tensor in the lepton series; the baryon analogs of [M5.22](m5_22_task_details.md), a 3 × 3 series, with a neutral state. The charge instrument of record is the dual curvature of the oriented long axis ([`m5_theory_canonical.md`](../m5_theory_canonical.md)), a density of a real unit vector. No reader of a Berry flux of a time-like eigenvector exists |
| 2 | A1 on a lattice: a dual U(1) link in Villain form whose monopoles are hedgehog worldlines with a tunable core action; the monopole density and the static $1/r^2$ flux read against the core action per link | In the topological frame a Coulomb phase is expected only above a core action of about ln 7 per link; the author would register the transition point and the flux before the run | No lattice action is sampled on OpenWave. Run 1 of [M5.59](m5_59_task_details.md) is a Coulomb-phase test asked the day before on another action (Berry plaquettes on CP³ with a cost on the charge of each cube) |
| 3 | "D1's frozen packet, if the external replication you offered still stands" | D1 is the first of the frozen predictions the road lists for a version 5.0: the tension and the stress profile of the fractional-charge string against lattice QCD | The integrity pass of [M5.39](m5_39_task_details.md) checked the checksums of the STRING packet and reproduced the value of its strand script (0.240 MeV per fm). By the author's post of 2026-11-13 the STRING calculation went to another external team, and OpenWave took the complement ([thread record](m5_32_convo.md)). No STRING ladder is planned on OpenWave |

**What the capstone document adds (version 4.8 as deposited on 2026-11-27, listed in [`_CITATIONS.md`](../../theory/_CITATIONS.md); items 36 and 45, the passage on FB-7 and the passages on the frozen packets; evidence, not resolution).** Item 45 orders the road as A (four definitions of the action), B (its stationary states), C (quantization) and D (blind predictions). In the deposited text it has no step named stage 0: the choice of the frame is the first decision of A1, "which identification". A hedgehog's Chern flux is magnetic for the connection of the time line; in the reading where its radial curvature is the electric field, the independent link is the dual potential and the charges are its monopoles. The post says the item is revised in a version 4.9, which was not read.

For read 1, item 36 reports a ledger (SL-1) on a coframe with a fixed metric, with three configurations: the charge as a Chern-1 hedgehog of the time line in the $(t, p)$ plane, the baryon as a B = 1 winding of the degenerate doublet, and the proton as both. On the baryon the curvature of the time line vanishes identically, and FB-7 draws the consequence the post states: a neutron with no charge distribution and no moment. The document lists the complex coframe, where its Chern-charge routes live, apart from the real coframe and from "the pure 2108 tensor M", which it calls a historical anchor.

For read 2, A1 puts the dual coupling near 21, a lattice coupling near 0.002, "deep in the regime where a Wilson-action link condenses its monopoles", and gives that as the reason why the earlier scans found only a monopole plasma or an ordered phase. A Coulomb phase needs a core action of the charges per cutoff length above the entropy of a loop. The pass of A1: on a lattice a static hedgehog sources a $1/r$ potential with its Chern number times the bare charge, then the retarded Green tensor against Maxwell's.

For run 2 of [M5.59](m5_59_task_details.md), the linear cost: SL-1 reports, on the charge, a stiffness acting on every line that grows linearly with the radius (slope 9.32 against 9.36), and item 36 then selects a covariant stiffness that is blind to the charge, "so it gives no 135.9 MeV/fm string"; the pass of B4 is a finite energy and no string.

For read 3, the document names STRING as the first of three frozen packets taken by an external team of the discussion. Its sector (a) rests on published formulas, with the strand value reproduced by the packet's script; for sector (b) the endpoint and profile arrays and the freeze certification are "still missing", and production waits on the completed packet.

The status of the single field, the fork on the spectator curvature and the road are the author's statements. Nothing of them was recomputed on OpenWave.

**Open at the plan.** Read in full first: the items of capstone 4.8 that item 45 rests on, and its revision when deposited. To be fixed by the plan, each with a known-answer control.

1. For read 1, the definition of the density on a real field. The eigenvectors of a real symmetric tensor are real, and the Berry curvature of a real line vanishes identically, so the read needs either the complex line the author means (and how it is built from the stored tensor and its velocity, if the rotation with the frame enters), or the real analog on record (the dual curvature of an oriented axis). The stored rings of M5.51 to M5.57 are 3 × 3 fields with no time-like eigenvector.
2. For read 1, which stored states count: what a B = 1 state and a neutral state are in the author's sense among the states on record, and what "rotating with the frame" reads on a static state.
3. For read 2, how it stands to run 1 of M5.59: one of the two, or both, and in which order. The definitions of the link, of the monopole current and of the core action; the sampler and how thermalization is shown; a control with a known transition.
4. For read 3, whether it is asked of OpenWave beside the other team, the completed packet with its inputs and scorer, and whether it is run as received or recomputed.
5. The place of these three in the order the author asked earlier the same day (M5.59, [M5.60](m5_60_task_details.md), [M5.58](m5_58_task_details.md)).

**Definition of done.** Written with the plan.

**Gating.** The author's word on the order, a plan with its audit, and the user's go.

**Blindspot pass, consumers, research body.** Written with the plan.
