# M5.58: the runs asked with capstone 4.6: a decay film, reconnection, clock locking

> Opened 2026-11-25 from the author's post on discussion #186 that releases capstone 4.6 ([discussioncomment-19899604](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19899604), 2026-11-25 19:31 UTC). Roadmap row: [`m5_roadmap.md`](../m5_roadmap.md). Thread record: [`m5_32_convo.md`](m5_32_convo.md).

## TASK PLANNING (2026-11-25; opened, not staged: the plan is written and audited before any go)

**Scope.** The post lists three runs asked of OpenWave, each with a prediction the author states as registered before anyone runs it. All three are dynamic runs on a kicked or rotating core, which is the ground of the lepton series ([M5.21.6](m5_21_6_task_details.md), [M5.21.10](m5_21_10_task_details.md), [M5.21.9](m5_21_9_task_details.md)), of the tracer of [M5.23.2](m5_23_2_task_details.md) and of the kicked neutron analog of [M5.22.2](m5_22_2_task_details.md).

| # | The run asked | The author's prediction | On record on OpenWave |
| --- | --- | --- | --- |
| 1 | Decay film: kick the heavy core and track the attached lines with the tracer of M5.23.2 | FD-1, 35 percent: the two ejecta carry different axis classes | [M5.21.6](m5_21_6_task_details.md): in a pinned arena every kick relaxes back; in a free arena the middle state of the census decays to the lowest, rotation-dominant. [M5.21.10](m5_21_10_task_details.md): a paired symmetric ejection, the count of two loops left open. [M5.23.2](m5_23_2_task_details.md): the core one closed torus throughout, the pair two open mirror filaments that evaporate. No reader of an axis class of an ejectum exists |
| 2 | Reconnection: a kicked heavy core, and the same test on the neutron analog (whether a decay to a proton analog, an electron and a loop opens) | RL-1, 35 percent: the kicked heavy core ends with at least one closed loop pinched off by reconnection, not only evaporating filaments | the tracer read of row 1 (no closed loop pinched off at that grid). [M5.22.2](m5_22_2_task_details.md): twenty kicks of three families, all returning to the neutral state |
| 3 | Clock locking | CL-1, 20 percent: the core's internal rotation locks at 1:3 with a slow precession mode, giving three discrete states | [M5.21.9](m5_21_9_task_details.md): the electron carried at fixed J on three rungs. No read of a precession mode or of a lock |

The post names one more read as a bonus: the split of the energy between the electron and the loops, against 7/20 to 13/20.

**What the capstone document adds (version 4.6 as deposited under the DOI of the post, items 9 and 28; evidence, not resolution).** Item 9 describes the lepton runs above, what they found and what failed, as read at a commit of 2026-11-24; the energy ratios it quotes for the three states are those of [M5.21.2b](m5_21_2b_task_details.md). It then places the decay film in the action of version 4.6 (the gradient term on time derivatives only, and a spectral potential) with the three states as three cores in one exterior, and says that the mechanism of the OpenWave runs carries over while their numbers do not. Neither that action nor a heavy core in the exterior of the lowest state is on OpenWave: the film of M5.21.6 and M5.21.10 ran on the action of that series, with each state in its own exterior. The document assigns its own round on these runs to the author's side, with two predictions (a core of the middle state's type in the exterior of the lowest state is a local minimum distinct from the lowest; a kicked heavy core decays to the lowest state and emits two compact closed loops of different classes). The label RL-1 of the post is not in the document; the second of those predictions is the nearest to it.

**Open at the plan, and gated by the author.** Which action the runs are in: the action on which the film exists, or that of version 4.6, which would first be ported from its equations, its parameters and a deposited reference state, as [M5.51](m5_51_task_details.md) ported the ring. Which state is the heavy core. The kick. What counts as a closed loop pinched off by reconnection, as a rule of the tracer with a known-answer control. What an axis class of an ejectum is, as a reader. Which mode is the slow precession, and how a 1:3 lock is told from a transient.

**Definition of done.** Written with the plan.

**Gating.** [M5.57](m5_57_task_details.md) closed, the author's answer on the action and the states, a plan with its audit, and the user's go.

**Blindspot pass, consumers, research body.** Written with the plan.
