# M8.7: The 3D rendering port (LATER, gated)

> Roadmap row: [`../m8_roadmap.md`](../m8_roadmap.md) § LATER. Status: 🚧 GATED: do not
> start before field dynamics validates in-platform, with an audited method note. **The
> gate CONDITION is unchanged by the M8.4 closure; only the routing prose is repointed.**
> The M8.4 lineage that was to supply the dynamics CLOSED UNRESOLVED 2026-08-26 with no
> target configuration executed ([closeout](../findings/m8_4_closeout.md)). The route
> through M8.5-C then closed with the C2 verdict of 2026-08-31
> ([#506](https://github.com/openwave-labs/openwave/issues/506)): any route now runs
> through a future program under platform conditions, which nothing currently authorizes
> ([#512](https://github.com/openwave-labs/openwave/discussions/512)). This is a
> scaffold-stage planning aid written by the maintainers (2026-07-21); the author owns the
> column and may amend everything here.

## PLANNING

### Scope

Port the validated S³/2I dynamics into the M5-style interactive stack so the model
runs as a live 3D demo: per-model `_launcher.py`, the `engine1-4` split (seeds → PDE →
observables → render), and the shared GGUI rendering in `openwave/i_o/`. The complete
agent-ready port path, with file pointers, is
[`../m8_platform_pointers.md § 7`](../m8_platform_pointers.md); this doc adds the
planning frame.

### Why the gate is strict

Rendering an unvalidated dynamics showcases nothing and costs real engineering. The
platform rule (M5's, kept here): the launcher must run the SAME kernels the research
stack validates, so the interactive demo shows the physics of record, including its
instabilities. That is only meaningful once there IS a physics of record.

### The M8-specific design problem (new ground)

S³/2I is not a box, so the port is a design task, not a copy:

| Problem | Candidate approaches to evaluate |
| --- | --- |
| Rendering chart | draw a fundamental domain of the 120-fold quotient; or a stereographic projection of S³ with the identification indicated; or an embedded slice/tour |
| The Möbius edge + cone structure | must be visibly represented (it carries the model's Λ story), not clipped away by the chart |
| Defect visualization | reuse M5's defect-line/glyph machinery where the M8.4 dynamics produces defects; new glyphs for representation-theoretic labels if needed |

### Suggested definition of done

| # | Item |
| --- | --- |
| 1 | Chart design doc (the table above resolved, with screenshots/prototypes) |
| 2 | Launcher wired to the SAME validated kernels (production kernels Taichi-first per platform practice), gated against the research stack's reference outputs |
| 3 | Render features documented M5-style ([`../m8_platform_pointers.md § 7`](../m8_platform_pointers.md) step 3's worked example) |

### Blindspots

| Risk | Guard |
| --- | --- |
| Starting early "just to have visuals" | the gate is in the roadmap; visuals for talks can hotlink the author's existing assets instead |
| Demo kernels drifting from research kernels | the M5 gating pattern: launcher output checked against the research stack's reference results before merge |

### Ownership + gating

Author-driven with platform support. Gate: field dynamics validated in-platform
(M8.4-lineage, audited).

## DEVIATIONS LOG

(none)

## FINDINGS

(pending)
