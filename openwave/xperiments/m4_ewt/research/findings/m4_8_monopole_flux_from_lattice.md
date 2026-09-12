# M4.8 Monopole Flux from Lattice Geometry with Derived Zeta

> Extension artifact: Enhanced EWT, Łukasz Smoliński ([PR #479](https://github.com/openwave-labs/openwave/pull/479)); manuscript v4.5.9 (Zenodo, [DOI 10.5281/zenodo.22110605](https://doi.org/10.5281/zenodo.22110605)). Contributed under the extension's name per [`dev_docs/CROSS_MODEL_TESTING.md`](../../../../../dev_docs/CROSS_MODEL_TESTING.md).

## Purpose

This artifact replaces the CODATA \(G\) in the M4.6 flux amplitude
with the geometric \(G\) chain of M4.7.

Previous artifacts M4.3-M4.6 established that the far-field profile

\[
\eta(r) = 1 - \frac{A}{r}
\]

reproduces light bending, gravitational redshift, and Shapiro delay
when the monopole amplitude \(A\) is known.

This artifact computes \(A = 2 G_{\text{EWT}} M / c^2\) with
\(G_{\text{EWT}}\) from the BCC lattice geometry instead of the CODATA
value (see the maintainer note below on where the measured \(G\)
still enters).

## What was computed

Two tests were performed:

### Test 1 : Pure BCC geometry

The ideal geometric stiffness was used:

\[
N_{\text{ideal}} = 8\pi^4 .
\]

This gave:

- \(G_{\text{EWT, pure}} = 6.662662892091293 \times 10^{-11}\)
- Amplitude \(A_{\text{pure}} = 2.948975829211922 \times 10^{3}\) m
- Difference from CODATA: \(1743.57\) ppm

This is the raw scale of the problem. The pure geometric limit is
close, but not sufficient.

### Test 2 : BCC geometry corrected by the packing impedance

The estimated packing impedance was used:

\[
\zeta_{\text{est}}
=
\frac{1-\eta_{\text{BCC}}}{\eta_{\text{BCC}}\,N_{\text{ideal}}}
\]

where

\[
\eta_{\text{BCC}} = \frac{\sqrt{3}\,\pi}{8}
\]

is the BCC sphere packing fraction.

This gave:

- \(N_{\zeta} = N_{\text{ideal}}(1-\zeta_{\text{est}})\)
- \(G_{\text{EWT, zeta}} = 6.674738142638409 \times 10^{-11}\)
- Amplitude \(A_{\zeta} = 2.954320482329130 \times 10^{3}\) m
- Difference from CODATA: \(65.65\) ppm

This is the key result. The packing impedance correction, estimated
from the BCC sphere-packing fraction, brings the amplitude to within
about \(66\) ppm of the measured value.

No calibration to the measured fine-structure constant is used
(\(N_{\text{ideal}}\) and \(\alpha_{\text{geo}}\) are the geometric
values). The measured \(G\) enters once, through the lattice length
\(\lambda_l\) (the Planck length); see the maintainer note.

## Role in the platform

In M4.6 the flux amplitude was the CODATA \(r_s = 2GM/c^2\). Here the
monopole amplitude \(A\) is computed from the M4.7 geometric chain
and then used as the boundary condition for the same Laplace solve.

The form of the flux condition (\(A = 2GM/c^2\)) and the encoding of
\(\eta\) into the index and the clock speed (M4.3 to M4.5) remain
manuscript-derived; what moves in-platform is the value of \(G\) in
the amplitude.

## Dependency on M4.7

The full geometric derivation of \(G_{\text{EWT}}\) is implemented
in:

- `m4_7_ewt_emergence_engine.py` (v5.0.0, since
  [PR #523](https://github.com/openwave-labs/openwave/pull/523), 2026-09-06;
  M4.8 ran on 2026-08-26 against the v4.5.2 port
  `m4_7_enhanced_ewt_geometric_consistency.py`, readable at commit
  `ec2564af`)

M4.8 does not re-derive the gravitational identity from scratch.
It uses the same BCC lattice parameters and the same geometric
chain, then extends it to the far-field monopole flux condition.

This keeps the artifact focused and traceable.

## Relation to criteria

This artifact is a foundational contribution to:

- `Gravity: local metric phenomena` (the flux amplitude of M4.6)
- `Gravity: Newton limit (GEM)`, the strength (\(G\)) clause added in
  [PR #480](https://github.com/openwave-labs/openwave/pull/480)

It is not itself a pass/fail validation against a single row: it links
the geometric derivation of \(G\) (M4.7) to the local metric
amplitude (M4.6). No cell change is proposed.

## Result summary

| Test | \(G_{\text{EWT}}\) | \(A\) | Difference from CODATA |
|---|---|---|---|
| Pure BCC geometry | \(6.662662892091293 \times 10^{-11}\) | \(2.948975829211922 \times 10^{3}\) m | \(1743.57\) ppm |
| BCC + packing \(\zeta\) | \(6.674738142638409 \times 10^{-11}\) | \(2.954320482329130 \times 10^{3}\) m | \(65.65\) ppm |

The radial Laplace equation was solved using the corrected amplitude.
The profile matched the analytic monopole form to a maximum
\(\eta\)-relative error of \(5.7 \times 10^{-14}\), and the asymptotic
Robin condition was satisfied to \(5.6 \times 10^{-14}\). As in M4.6
this is an integrator check (the initial data are the analytic
solution), not a physics test.

## Maintainer note (at merge, PR #479 review)

| Check | Result |
| --- | --- |
| Status of \(\lambda_l\) | Primary lattice ansatz, per the author ([PR #479 thread](https://github.com/openwave-labs/openwave/pull/479#issuecomment-5425514570)): set to the Planck length as the natural scale of the lattice, not fitted and not derived from geometry, and no derivation is claimed. The measured \(G\) enters the chain once, here; downstream of it the chain is fixed by BCC geometry and \(\zeta\) |
| Where the measured \(G\) enters | `lambda_l = 1.6162e-35` m is the Planck length \(\sqrt{\hbar G / c^3} = 1.616255 \times 10^{-35}\) m (the extension's own [`M4_k_selectivity_Formalization.md`](../M4_k_selectivity_Formalization.md) names it so). Through \(N_\nu \propto \lambda_l^{-3}\) and \(G_{\text{EWT}} \propto N_\nu^{-1/2}\), \(G_{\text{EWT}} \propto \lambda_l^{3/2} \propto G^{3/4}\); verified by scaling the input \(G\) by 0.5 and 2 (ratios 0.5946 and 1.6818, equal to \(f^{3/4}\)) |
| Fixed-point reading | The parameter-free statement of the chain is \(G_\ast = G_{\text{EWT}}^4 / G_{\text{CODATA}}^3\) (the value that reproduces itself through \(\lambda_l\)): \(6.6774 \times 10^{-11}\), 467 ppm above CODATA with \(\zeta_{\text{est}}\); \(6.6292 \times 10^{-11}\), 6753 ppm below, with pure geometry |
| Sensitivity to the 5-digit \(\lambda_l\) | With the CODATA Planck length \(1.616255 \times 10^{-35}\) m the \(\zeta\)-corrected \(G_{\text{EWT}}\) sits 117 ppm above CODATA (the 34 ppm truncation of \(\lambda_l\) moves \(G_{\text{EWT}}\) by 25 ppm) |
| \(\zeta_{\text{est}}\) vs the \(\alpha\)-implied \(\zeta\) | \(6.03 \times 10^{-4}\) vs \(5.83 \times 10^{-4}\) (M4.7's \(N_{\text{final}}\)), 3.4% apart; \(G_{\text{EWT}} \propto N^{-3}\) turns that into 60 of the 66 ppm |
| Measurement bar | CODATA 2022 \(G\) has a 22 ppm relative uncertainty; 66 ppm is outside it, while M4.7's \(\alpha\)-anchored route (4.78 ppm) is inside |

## Model assumptions

The model follows the Enhanced EWT manuscript, version 4.5.9 or later.

The lattice parameters used in the derivation chain are:

- the BCC coordination number,
- the ideal sphere-packing fraction,
- the geometric lattice projection factor \(L_p^{\text{geom}}\),
- the lattice length \(\lambda_l\) (Planck length) and the statutory
  neutrino radius \(r_\nu = r_e/100\), inherited from M4.7.

No free numerical parameters were introduced beyond those inputs.

## Reference

Full derivation in the Enhanced EWT manuscript, version 4.5.9 or later:
[DOI: 10.5281/zenodo.22110605](https://doi.org/10.5281/zenodo.22110605)

Relevant section:

- "From Microscopic EMC Displacement to the Gravitational Radius"
