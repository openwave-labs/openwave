# M4.10.2 Emergent Gravitational Force from Geometric Amplitude

> Extension artifact: Enhanced EWT, Łukasz Smoliński
> (manuscript v5.0.0, DOI 10.5281/zenodo.22540635)

## Criterion
`Gravity: Newton limit (GEM)` — strength (G) clause:
attractive 1/r² between masses, via the GEM route; the strength (G)
credited only when it follows from the model's own mechanism, never fitted.

## Status
Candidate for **validated in platform** under the strength (G) clause.

## What was computed

The gravitational force between the Sun and the Earth was computed from
geometric factors alone, with no G value entering the calculation.

The force kernel follows from the overlap of the two EMC density
deficits. Each soliton sources a monopole deficit δη_i(r) = −A_i/r
whose gradient is ∇(δη_i) = (A_i/r²) r̂. The angular integral of
∇(δη_1)·∇(δη_2) over the sphere of radius r gives exactly 4π/r²
for r ≥ R and zero for r < R (Newton's shell theorem analogue). The
radial integral over r ∈ [R, ∞) gives I(R) = 4π A_1 A_2 / R. The force
is −dI/dR = 4π A_1 A_2 / R², and the EMC field-energy normalization
K_emc turns the kernel into the physical force:

    F_grav = 4 π K_emc A_1 A_2 / R^2

The amplitudes and coupling are written in M4.7 chain factors:

    A_i   = (2 M_i r_e / m_e) * sqrt(X_eff)
            / (A_pi^4 N_geom^3 K_WC sqrt(N_nu_stat))

    K_emc = c^2 m_e A_pi^4 N_geom^3 K_WC sqrt(N_nu_eff) / (16 pi r_e)

Every factor on the right-hand side is either geometric (A_pi, N_geom,
K_WC, X_eff, N_nu_stat, N_nu_eff) or a dimensional anchor (r_e, m_e, c).
No G value appears in any of these expressions.

Derivation source: M4.10 artifact, sections "Angular Integral Correction
& Field Overlap Formulation" and "Physical Interaction Energy and Sign
Convention". Amplitude chain factors: M4.12 artifact.

## Result

    F_grav (geometric, no G) = 3.544208708919220e+22 N
    F_obs  (from G_CODATA)   = 3.542499577248244e+22 N
    relative difference      = 0.048246 %
    residual / CODATA unc.   = 21.9 x

The 0.048246 % agreement is reported for transparency. The criterion
does not require agreement within CODATA uncertainty; it requires that
the strength follows from the model's own mechanism.


## Mutation tests (geometric factors, not G)

| Perturbation    | Relative change in F_grav |
|-----------------|---------------------------|
| K_WC: 10 -> 9   | 11.11 %                   |
| N_geom: * 1.001 |  0.30 %                   |
| A_pi: * 1.01    |  3.90 %                   |
| r_e: * 1.01     |  0.00 %                   |

The r_e test is structural: A ~ r_e and K_emc ~ 1/r_e, so r_e cancels
identically in F_grav. The dimensional anchor r_e fixes units, not the
magnitude of the force.

## Criterion mapping

| Clause | Evidence |
|--------|----------|
| attractive 1/r² between masses | F_geom = 4 π A_1 A_2 / R^2, R² dependence confirmed |
| via the GEM route | A_i, K_emc from BCC lattice geometry (M4.7 chain) |
| strength (G) follows from mechanism | F_grav built entirely from A_i and K_emc in M4.7 chain factors; G is a consequence, not an input |
| never fitted | mutation tests on geometric factors, not on G (section 8) |

## Relation to M4.10

M4.10 is a normalization-consistency gate: G enters in both A and K_emc
and cancels identically. This artifact demonstrates that the force
strength follows from the model's own mechanism: the geometric factors
alone fix F_grav, and G is a derived consequence of them, not an input.

## Artifacts

- `research/scripts/m4_10_2_emergent_gravitational_force_geometric.py`

## Reference

Enhanced EWT manuscript, version 5.0.0 or leter:
[DOI: 10.5281/zenodo.22540635](https://doi.org/10.5281/zenodo.22540635)

Relevant manuscript section:
- "Newtonian Force from Interacting EMC Deficits"

Relevant M4 artifacts:
- M4.10 — angular integral, sign convention, K_emc normalization
- M4.12 — amplitude written in M4.7 chain factors
- M4.7  — geometric primitives (A_pi, N_geom, K_WC, X_eff,
  N_nu_stat, N_nu_eff)