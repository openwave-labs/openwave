# M10.3 · A freely evolving bound pair

In the two-unit sector the native seam and degree-reference terms reduce, up to an overall phase, to `H = L_moves/2 − 2κ P_adjacent`. Allowed swaps connect all distinct-port configurations. The contrast is 10 for adjacent units and 12 when separated. The observer occupies one echosahedron and returns the occupied internal seam or quiet. Its echo writes the pair flag and the inverse echo restores the state in each terminal branch. A fresh source is released for each read. [Native operator, full-matrix comparison and decoder](https://github.com/FloatingPragma/observer-patch-holography/tree/4fe10f9ac3e81423155fb3ac7baaf0fdb4c0b11c/openwave/interval_echo/row_04/native_check.py) · [Independent sparse evolution](../scripts/m10_3/row_04_check.py)

**Scope**: On the 20-cell carrier at the supplied coupling $\kappa = 8$, the free two-load generator gives the probability of detecting the prepared pair on a launch-cell seam through exposure 37, without establishing its charge, spin, species or lifetime beyond that window.

| Native exposure | Seeded draws on a home seam, κ = 8 | Exchange only, κ = 0 |
| ---: | ---: | ---: |
| 1 | 4,048 / 4,096 | 1,780 / 4,096 |
| 7 | 4,052 / 4,096 | 1,182 / 4,096 |
| 37 | 4,028 / 4,096 | 824 / 4,096 |

A changed-law run using the same source and detector rejects at the native persistence gate. Reproducing the older cell-0 experiment gives local probabilities 0.983880525 and 0.149558605 at exposure 150. The source, same-geometry control, complete simulated single-trial records and logic of the simultaneous four one-sided decisions were fixed for the reproduction at planning commit `484a96fd08f9e940a8239918341b88c0e324db56`. The strongest native lower confidence bound common to these three comparisons exceeds 95.48%. A separate Taylor evolution on a 1/8 mesh and a bound on between-mesh motion using ΔH = 1.5 give a downward-rounded lower bound of 94% on the local probability throughout the finite window. This bounds the ensemble probability at each possible terminal read. [Complete author records](https://github.com/FloatingPragma/observer-patch-holography/tree/4fe10f9ac3e81423155fb3ac7baaf0fdb4c0b11c/openwave/interval_echo) · [Mesh and interpolation code](../scripts/m10_3/row_04_window.py) · [Replay outputs and failing control](../data/m10_3_reproduction.json)

## Equation-to-code map

- [Full seam echo and its memory generator](https://github.com/FloatingPragma/observer-patch-holography/blob/4fe10f9ac3e81423155fb3ac7baaf0fdb4c0b11c/model/candidate_laws/contrast_memory_spherical_v1/law.py#L67)
- [Two-load Hamiltonian at either coupling](https://github.com/FloatingPragma/observer-patch-holography/blob/4fe10f9ac3e81423155fb3ac7baaf0fdb4c0b11c/model/candidate_laws/contrast_memory_spherical_v1/sectors.py#L74)
- [Independent configuration graph](https://github.com/FloatingPragma/observer-patch-holography/blob/4fe10f9ac3e81423155fb3ac7baaf0fdb4c0b11c/openwave/interval_echo/row_04/row_04_check.py#L23)
- [The local outcome probabilities](https://github.com/FloatingPragma/observer-patch-holography/blob/4fe10f9ac3e81423155fb3ac7baaf0fdb4c0b11c/openwave/interval_echo/row_04/row_04_check.py#L54)
- [Exponential time evolution](https://github.com/FloatingPragma/observer-patch-holography/blob/4fe10f9ac3e81423155fb3ac7baaf0fdb4c0b11c/openwave/interval_echo/row_04/row_04_check.py#L63)
- [All-time interpolation within the tested window](https://github.com/FloatingPragma/observer-patch-holography/blob/4fe10f9ac3e81423155fb3ac7baaf0fdb4c0b11c/openwave/interval_echo/row_04/row_04_window.py#L14)
- [Native generator and detector comparison](https://github.com/FloatingPragma/observer-patch-holography/blob/4fe10f9ac3e81423155fb3ac7baaf0fdb4c0b11c/openwave/interval_echo/row_04/native_check.py#L17)

## Requested row 4 cell

The paragraph below is the author's submitted cell. The [maintainer decision](m10_3_maintainer_review.md) awards partial credit.

✅ A prepared two-unit pair persists under Interval Echo on a 20-cell sphere. With all seams active its one-cell detection probability stays at least 94% through native exposure 37. At 37 the terminal receiver recorded 4,028/4,096 pairs, versus 824/4,096 under exchange alone. The adjacent launch, blank memories, uniform exposure and read window are supplied. Charge and spin are unassigned.

The free-persistence experiment supplies the initial adjacent pair. Its later spatial distribution is a consequence of the law. A separate encountered pair could give a different preparation yield. The row assigns neither a charged-lepton species nor an indefinite small-patch lifetime. The related simulator demonstrates local repair and recording in a teaching model. The numerical pair measurements use the scripts here. Maintainers decide the pin and cell. [Teaching repair view](https://simulation.floatingpragma.io/patches?scene=repair)

## Source and attribution

The replay scripts come from the RER author package, copyright 2026 Bernhard Mueller and the Observer Patch Holography contributors, under Apache-2.0. The checker reads the author fixture and two source receipts from the separate RER checkout at the proposed pin. The three input digests are committed in the checker. Author data follow the RER data terms. [Source terms](https://github.com/FloatingPragma/observer-patch-holography/blob/4fe10f9ac3e81423155fb3ac7baaf0fdb4c0b11c/LICENSE)

## Receiver records and candidate scorecard

The measured figure compares the pair counts at every enrolled exposure with the same detector under exchange alone. Its caption gives the independently calculated probability bound for the full window. The author’s scorecard describes the scope of the other finite tests on this pin. [Measured local detections](https://github.com/FloatingPragma/observer-patch-holography/blob/e93472c72821d992c9824a287b7062a377ceffad/openwave/interval_echo/review/visuals/row-04-persistence.png) · [Figure source and teaching views](https://github.com/FloatingPragma/observer-patch-holography/blob/e93472c72821d992c9824a287b7062a377ceffad/openwave/interval_echo/review/visuals/README.md) · [Code, board and raw records](https://github.com/FloatingPragma/observer-patch-holography/blob/e93472c72821d992c9824a287b7062a377ceffad/openwave/interval_echo/review/row-04/row-04-evidence.zip) · [Interval Echo scorecard](https://github.com/FloatingPragma/observer-patch-holography/blob/e93472c72821d992c9824a287b7062a377ceffad/openwave/interval_echo/review/scorecard/SUMMARY.md)

## Watch the free pair

The animation follows two loads initially placed on seam (0, 1) in cell 0. Choose the 20- or 80-cell screen and scrub forward. The upper panels show how often the two loads share any seam on the sphere. The lower panels show how often the birth cell detects them together. The left column uses Interval Echo’s contrast interaction and the right column uses exchange alone. The exposure-37 result in the table above uses a separate preparation on seam (156, 157) in cell 13. [Play the row 4 particle-stability comparison](https://floatingpragma.ai/oph/simulation/openwave/row-4/)
