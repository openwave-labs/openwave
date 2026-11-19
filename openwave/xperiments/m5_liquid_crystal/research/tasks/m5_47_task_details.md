# M5.47: Klein-Gordon: the dispersion of small perturbations in a time evolution

> Opened 2026-11-19. Roadmap row: [`m5_roadmap.md`](../m5_roadmap.md). Thread record: [`m5_32_convo.md`](m5_32_convo.md).

## TASK PLANNING (2026-11-19; opened, not staged: the plan is written and audited before any go)

**Scope.** The run named by the way back of the Klein-Gordon cell of the M5 column, as the cell reads after the re-read of [T11](../../../../../dev_docs/tasks/t11_task_details.md): the dispersion of small perturbations about a converged relaxed state, read in a time evolution, giving ω² = k² + m². The row's test is an emergent ω² = k² + m² for matter waves, and its regime is `dynamic`.

| # | Part | What it reads |
| --- | --- | --- |
| 1 | A time evolution of the action of [M5.42](m5_42_task_details.md) with its time part (the rate of M in the L₁ term and in the quartic) | the instrument, with an energy-conservation control |
| 2 | Small perturbations about the vacuum on a periodic box, all six components | six branches against the eigenvalues of the potential's Hessian at the vacuum; this part follows from the declared action and checks the instrument |
| 3 | A driven wave on the relaxed unit with L₁, absorbing faces | the local wavenumber against a position-dependent mass read off the background; this is the part the cell's clause asks for |

**Not planned at this date.** The grids, the amplitudes, the frozen readings, the controls and the parallel shape: they are written, and audited, when the task is staged.

**Gating.** [M5.46](m5_46_task_details.md) first, then a plan and the user's go.
