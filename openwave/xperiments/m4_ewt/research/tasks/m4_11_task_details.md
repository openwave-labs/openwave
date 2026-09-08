# M4.11 - N as the Internal Lock-In Point: Drift, Resolution, Grouping

## Status
DONE (post-hoc)

## Source
Task inspired by:
https://github.com/openwave-labs/openwave/pull/526

## Criterion
`Gravity: Newton limit (GEM)` + `Gravity: local metric phenomena` (strength clause)

## Objective
Verify that \(N_{\text{geom}} = 778.8025\) is the internal lock-in point of
the Enhanced EWT model by measuring, for each sector, how quickly its
prediction drifts away from the value at \(N_{\text{geom}}\) when \(N\) is
changed, and by determining which sectors actually resolve \(N\) and which
do not.

## Method

1. Compute the reference value of each sector at \(N_{\text{geom}}\).
2. Scan \(N\) from 500 to 1100 with step 2.
3. For each \(N\), compute all sector values using the same functions as
   the M4.7 emergence engine.
4. Compute the relative drift
   \(|O(N) - O(N_{\text{geom}})| / |O(N_{\text{geom}})|\).
5. Compute the logarithmic derivative
   \(\beta = d\ln O / d\ln N\) at \(N_{\text{geom}}\).
6. Compute the resolution as
   \(\text{relative error} / |\beta|\).
7. Find the separately optimal \(N\) for each sector by log-space scan and
   bisection.
8. Group the sectors by resolution:
   - **PRIMARY** — resolution < 1%,
   - **WEAK** — resolution between 1% and 100%,
   - **NO CONSTRAINT** — resolution > 100%.

Only PDG quark masses are used for the Cabibbo sector. No EWT quark masses
enter the calculation.

## Artifacts

- `research/scripts/m4_11_lock_in_n_scan.py`
- `research/findings/m4_11_lock_in_n_scan.md`

## Reference

Enhanced EWT manuscript, version 5.0.0:
[DOI: 10.5281/zenodo.22540635](https://doi.org/10.5281/zenodo.22540635)