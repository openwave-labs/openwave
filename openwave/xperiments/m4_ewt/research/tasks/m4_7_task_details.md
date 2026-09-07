# M4.7 - Enhanced EWT Geometric Emergence Engine (Zero-Calibration)

## Status
DONE (post-hoc)

## Purpose
Provide the OpenWave platform with a complete zero-calibration geometric
engine for the Enhanced EWT model, version 5.0.0. The engine produces the
gravitational constant, the fine-structure constant, the lepton anomalous
magnetic moments, particle masses, and atomic scales from the BCC lattice
geometry and four experimental anchors, without fitted parameters.

## Scope
This artifact is the platform's reference implementation for the v5.0.0
model. It provides the geometric constants (G, alpha, epsilon_M, lambda_l,
r_nu, g_v) and the recursive lepton hierarchy needed by later tasks.

The artifact consists of four runnable Python scripts:

  - m4_7_ewt_emergence_engine.py        (main engine: BCC geometry, G,
                                         alpha, r_nu, lambda_l, AMM,
                                         atomic scales, rigidity tests)
  - m4_7_ewt_amm_extended.py            (extended lepton AMM with internal
                                         shell references and consistency
                                         errors)
  - m4_7_ewt_amm_resonance_scanner.py   (Onion Model resonance scanner;
                                         generates four PDF figures)
  - m4_7_ewt_particle_masses.py         (spherical, orbital, meson mass
                                         modes; full PDG/CODATA scan)

The scripts are four of the six files of the package archived at
DOI: 10.5281/zenodo.22540262, content-identical after line-ending
normalization except for the module rename and the import line that follows
it (see the findings note). The corresponding manuscript is
DOI: 10.5281/zenodo.22540635 (version 5.0.0).

This artifact supersedes the v4.5.2 Scilab port merged 2026-08-25 in
[PR #477](https://github.com/openwave-labs/openwave/pull/477) (`ec2564af`);
the task keeps its ID. The v5.0.0 artifact was merged in
[PR #523](https://github.com/openwave-labs/openwave/pull/523).

## Method

1. Derive the ideal BCC stiffness N_ideal = 8*pi^4 from the coordination
   number and the four-dimensional saturation budget.
2. Compute the lattice impedance zeta from the BCC sphere-packing fraction
   eta_BCC = sqrt(3)*pi/8:
       zeta = (1 - eta_BCC) / (eta_BCC * N_ideal)
3. Obtain the effective geometric stiffness:
       N_geom = N_ideal * (1 - zeta)
4. Define the magnetic deficit:
       epsilon_M = 1 / (N_geom * pi^3)
5. Compute the geometric fine-structure constant:
       alpha_geom^-1 = A_pi - epsilon_M
       A_pi = 4*pi^3 + pi^2 + pi
6. Derive the neutrino radius from the geometric fixed point of g_v using
   the self-consistent quadratic equation, with q_P = e / sqrt(alpha_geom).
7. Derive the reduced Planck constant from alpha_geom and electron anchors:
       hbar_geom = m_e * c * r_e / alpha_geom
8. Derive the EMC lattice spacing lambda_l self-consistently from the
   Planck-length definition and the geometric expression for G.
9. Compute the gravitational constant:
       G_geom = (G_Base / A_pi) * (1 / (N_geom * A_pi))^3
                / (K_WC * sqrt(N_nu_eff))
   with K_WC = 10 and N_nu_eff = N_nu_stat / X_eff.
10. Compute lepton AMMs recursively with the Onion Model and the
    dimensional projection operators O_e = 1, O_mu = 1/(4*pi^2),
    O_tau = 1.
11. Compute particle masses in spherical, orbital, and meson modes.
12. Run rigidity tests by perturbing N_geom, L_p_geom, lambda_l, r_e,
    m_e, and epsilon_M to confirm the model is not a numerical
    coincidence.

## Result

Key geometric predictions from the zero-calibration engine:

| Quantity | Prediction | Target / reference | Rel. error |
| --- | --- | --- | --- |
| G_geom | 6.6775199755e-11 m^3 kg^-1 s^-2 | CODATA 6.674305e-11 | 0.048169% |
| alpha_geom^-1 | 137.036262364 | CODATA 137.035999084 | 0.000192% |
| a_e (full) | 1159.916228 ppm | CODATA 1159.652182 ppm | 0.022769% |
| a_mu (full) | 1166.212608 ppm | Experiment 1165.920610 ppm | 0.025044% |
| a_tau (full) | 1176.838130 ppm | SM prediction 1177.21 ppm (no measurement at this precision) | 0.031589% |
| lambda_l | 1.6166464066e-35 m | 1.6162e-35 m (rounded; CODATA 2018 1.616255e-35 m gives 0.024217%) | 0.027621% |
| r_nu | 2.8179354360e-17 m | r_e/100 = 2.8179403262e-17 m | 0.000174% |
| g_v (fixed point) | 0.9835944447 | phenomenological 0.98359223 | 0.000221% |

Atomic scales (from alpha_geom and r_e = 100*r_nu):

| Quantity | Prediction | CODATA 2022 | Rel. error |
| --- | --- | --- | --- |
| R_inf | 10973687.362034 m^-1 | 10973731.568157 m^-1 | 0.000403% |
| a0 | 5.291783259435e-11 m | 5.291772109030e-11 m | 0.000211% |
| lambda_C | 2.426310689655e-12 m | 2.426310238670e-12 m | 0.000019% |

The full AMM predictions for muon and tau use no lepton masses as inputs;
they are pure geometric quantities derived from epsilon_M and the
recursive nodal growth law K_n = K_{n-1} + round(10^(n-1) * 2*pi^2).

## Interpretation

This is a zero-calibration geometric derivation. The only numerical
inputs are the mathematical constants pi, e, sqrt(2), sqrt(3), the BCC
lattice geometry (coordination number 8, packing fraction), and four
experimental anchors: r_e, m_e, c, e.

The engine also establishes the geometric self-consistency of the
Planck-Gravity-Metric triangle: G_geom, hbar_geom, lambda_l, and c are
not independent constants but are linked by the closed condition

    lambda_l = sqrt(hbar_geom * G_geom / c^3)

with hbar_geom = m_e * c * r_e / alpha_geom. This closure follows from
the BCC lattice structure; the Planck scale, the reduced Planck
constant, and the gravitational constant are reconstructed as derived
quantities, not assumed inputs.

The model is not claimed to be absolutely parameter-free. The four
experimental anchors r_e, m_e, c, e are fixed by measurement and by
metric convention. They are not adjustable calibration parameters: if
any one of them were changed independently while keeping the BCC
geometry fixed, the entire set of geometric predictions would break.
The model therefore contains no free calibration constants; it is
zero-free-parameter in the operational sense. Of these anchors, c is a
metric conversion factor rather than a dynamical parameter. The
electron mass m_e may also be derivable in the future through the
E proportional to r^5 scaling law and the K_WC = 10 stability
condition. The true irreducible anchors would then reduce to r_e and e.

## Maintainer note (review of PR #523)

Edits applied at merge, announced in the review: import path in the three
companion modules (they raised `ImportError` in the repository after the
rename), byte-order mark and trailing newlines, the tau reference relabeled
as the Standard Model prediction, the `g_v` error cell in percent like the
rest of its column, the `lambda_l` reference identified as rounded, and the
scoping of "what enters as a number" to the engine chain, with the other
numbers the package carries tabulated in the findings note. The scripts are
deliberately not reformatted with `black`, so that they stay checkable
against the archive's checksums.

## Artifacts

- `research/scripts/m4_7_ewt_emergence_engine.py`
- `research/scripts/m4_7_ewt_amm_extended.py`
- `research/scripts/m4_7_ewt_amm_resonance_scanner.py`
- `research/scripts/m4_7_ewt_particle_masses.py`
- `research/findings/m4_7_enhanced_ewt_geometric_consistency.md`

## Reference

Manuscript: Enhanced EWT, version 5.0.0
DOI: 10.5281/zenodo.22540635

Source scripts:
DOI: 10.5281/zenodo.22540262
