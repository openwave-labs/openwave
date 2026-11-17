# M5.43: Card C at zero spacing: pair stop under the gate, two boxes, fit form

> Opened 2026-11-17 from the author's reply to the results of [M5.41](m5_41_task_details.md) on discussion #186 ([discussioncomment-19669234](https://github.com/vantasnerdan/substrate-framework/discussions/186#discussioncomment-19669234)). Roadmap row: [`m5_roadmap.md`](../m5_roadmap.md). Thread record: [`m5_32_convo.md`](m5_32_convo.md).

## TASK PLANNING (2026-11-17; opened, not staged: the plan is written and audited before any go)

**Scope.** The run the author asked for after M5.41, on the same Card C background and with the same instruments.

| # | What the author asked | Where it enters |
| --- | --- | --- |
| 1 | The pair stop is read as reading (ii) of M5.41 § 11: both stops of a pair under the gate. Where that leaves a mesh without a pair (77³ in M5.41), its routes are extended to a 100000 target; the rule is not relaxed. The three readings stay side by side | the rule of the route labels |
| 2 | The box: the 1.6 percent rise of the inertia from half-box 6.075 to 10.125 is a hundred times the route range. A value at infinite volume needs two half-boxes on at least the two finest meshes, or the exterior tail continued analytically | the meshes and boxes of the pool |
| 3 | The fit form: X₀ + b h² + c h⁴ on four meshes fits them to 3 × 10⁻⁵ (inertia 35.839, e 4.2177, f_π 93.31 MeV, g_A 0.91346, the author's numbers); a fit linear in h fails. A manufactured smooth field with a known continuum energy, on the same four meshes, fixes the stencil's error form independently of the relaxations | the zero-spacing read and a new control |
| 4 | The six-neighbor read of `m5_41_spikes.py` becomes part of the lump label | the lump check |
| 5 | "Second valley or slow descent": the energy along the straight path between an upper-group state (center eigenvalue 0.345) and a lower-group one (0.330) on the same mesh; a barrier means two valleys, a monotone fall a slow descent | a desk read on saved states of M5.41 |

**Definition of done.** Written with the plan.

**Gating.** A plan with its audit, and the user's go.

**Blindspot pass, consumers, research body.** Written with the plan. The numbers of row 3 are the author's and are recomputed from [`m5_41_summary.json`](../data/m5_41_summary.json) before they are used.
