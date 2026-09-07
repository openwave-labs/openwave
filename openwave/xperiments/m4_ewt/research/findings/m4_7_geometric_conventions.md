# M4.7 Geometric Conventions and Parameter Status

## Review Response Context

This document is the author's response to the review questions raised
in PR #523:

- https://github.com/openwave-labs/openwave/pull/523

It records the conceptual and dimensional conventions of the
zero-calibration geometric engine, and clarifies the status of the
constants that enter the model. It is written for reviewers and future
contributors who need to know what is derived, what is postulated, and
what is carried as an experimental anchor.

## Purpose

The M4.7 engine derives the gravitational constant, the fine-structure
constant, the lepton anomalous magnetic moments, particle masses, and
atomic scales from BCC lattice geometry and four experimental anchors.
This document makes the underlying conventions explicit, so that the
claim "zero-free-parameter" can be audited against a written record.

## Provenance

The manuscript v5.0.0 and the source package are registered in the M4
citation registry (`_CITATIONS.md`):

- Manuscript: Enhanced EWT, version 5.0.0,
  DOI: 10.5281/zenodo.22540635
- Source scripts: Enhanced EWT geometric emergence engine,
  DOI: 10.5281/zenodo.22540262

The primary Energy Wave Theory work on charge as wave amplitude is
already registered in the same citation registry as an EWT document:

- Yee, J., Gardi, L. (2019). The Relationship of Mass and Charge.
  DOI: 10.13140/RG.2.2.12645.45289

## 1. Charge as Wave Amplitude

The identification of charge with wave amplitude is a foundational
postulate of Energy Wave Theory, established in the primary EWT
literature:

- Yee, J., Gardi, L. (2019). The Relationship of Mass and Charge
  (revision 2). DOI: 10.13140/RG.2.2.12645.45289.
- Yee, J. What is Charge? Energy Wave Theory,
  https://energywavetheory.com/explanations/what-is-charge/

In that work, the authors show that the units of charge can be
expressed as wave amplitude, with units of distance:

  "As charge can be expressed as wave amplitude, the units for
  Coulombs (C) is corrected to be expressed as amplitude, which in
  SI units is measured as meters (m)."

The elementary charge of a single electron is the wave amplitude at
the first wavelength. Amplitude is the average displacement distance
of granules from equilibrium. Therefore the elementary charge
amplitude is

    e_geom = 1.602176634 x 10^-19 m

This is a direct dimensional mapping: the numerical value of the
CODATA elementary charge is preserved, while the dimension changes
from coulombs to metres.

The same convention is used in manuscript v5.0.0, Section "Planck
Charge from Geometric Alpha", where

    q_P = e_geom / sqrt(alpha_geom)

with e expressed in metres.

### 1.1 Manuscript references

In manuscript v5.0.0, the geometric charge convention is used in:

- Section "Planck Charge from Geometric Alpha"
  (label: sec:qP_from_alpha)

A separate section that makes the Coulomb-to-metre reduction explicit,
and that resolves the CGS / statcoulomb ambiguity, is proposed for a
future version of the manuscript. Until then, the primary source for
this convention remains the Energy Wave Theory literature cited above,
in particular:

- Yee, J., Gardi, L. (2019). The Relationship of Mass and Charge.
  DOI: 10.13140/RG.2.2.12645.45289

### 1.2 Distinction from CGS / statcoulomb

The geometric charge is a spatial displacement, not a force-derived
quantity. In Gaussian CGS units, the statcoulomb is defined via
Coulomb's law and carries a composite mechanical dimension. In the
geometric EWT convention, transforming the charge to CGS spatial units
means expressing the displacement in centimetres, not substituting the
statcoulomb value.

The numerical equality 1 C -> 1 m is anchored to the SI convention,
where spatial displacement is measured in metres. This convention is
part of the model's foundations, not an ad hoc assumption introduced
in v5.0.0.

## 2. Status of the Constants

The engine's header promises that every parameter is DERIVED,
POSTULATED, or CALIBRATION. The table below records the status of
each constant that appears in the four scripts, with explicit
references to the manuscript v5.0.0.

| Parameter | Value | Status | Justification | Manuscript reference |
| --- | --- | --- | --- | --- |
| pi, e, sqrt(2), sqrt(3) | mathematical | MATHEMATICAL | pure mathematical constants | Section "Global Inputs and Constants" |
| BCC coordination number | 8 | BCC_GEOMETRY | crystallographic fact | Section "BCC Lattice Geometry and the Geometric Ladder" |
| BCC packing fraction | sqrt(3)*pi/8 | BCC_GEOMETRY | geometric property of the BCC lattice | Section "BCC Lattice Geometry and the Geometric Ladder" |
| L_p_geom | 2/sqrt(3) | BCC_GEOMETRY | ideal BCC lattice projection factor; inverse of the dimensionless nearest-neighbour distance in the BCC lattice | Section "BCC Lattice Geometry and the Geometric Ladder" |
| K_WC | 10 | STRUCTURAL (geometric necessity) | electron wave center count from the 1-3-6 geometry; topological winding number Q=10; consistent with the decadic resonance r_e/r_nu = 100, which does not itself vary with K_WC (see section 2.1) | Sections "Structural Emergence of the 1-3-6 Geometry", "Topological Selection Rule and the Electron Ground State", "Decadic Resonance Test" |
| L_mu | 5 | STRUCTURAL (Fibonacci-Lucas) | F_5, the 5th Fibonacci number; 2D planar shell stability invariant | Sections "Structural Stability Invariants of the BCC Lattice", "The Geometric Stability of 3D Wave-Packing" |
| L_tau | 34 | STRUCTURAL (Fibonacci-Lucas) | F_9, the 9th Fibonacci number; 3D volumetric shell closure invariant | Sections "Structural Stability Invariants of the BCC Lattice", "The Geometric Stability of 3D Wave-Packing" |
| O_mu | 1/(4*pi^2) | DERIVED (double derivation) | (1) from M_mu*pi^3*epsilon_M identity, verified to 0.14%; (2) independently from the normalization of the inverse 2D Fourier transform | Section "Final Results and Experimental Correlation", subsection "Remark on the geometric origin of the muon projection operator"; Section "Dimensional Projection Operators" |
| O_tau | 1 | DERIVED | dimensional projection rule; 3D resonance matches observable space | Section "Final Results and Experimental Correlation", subsection "Dimensional Projection Operators" |
| +L_mu^2 | 25 | STRUCTURAL (interface tension) | equals L_mu^2 with L_mu=5; topological splitting of phase space by a genus-1 surface | Section "The Interface Tension (delta)" |
| RHO_A, A_LONG, L_LONG | Yee's wave constants | INPUT | original EWT constants from Jeff Yee's work | Section "EWT particle mass prediction"; Section "Global Inputs and Constants" |
| orbital amplitudes 185.68543, 3436.795 | mass-sector | CALIBRATION | mass-mode inputs; do not enter the geometric AMM derivation | Section "EWT particle mass prediction", subsection "Orbital Mode (Resonant Excitations)" |

### 2.1 Justification for K_WC = 10

K_WC = 10 is not a free parameter. It is the number of wave centers
in the electron, fixed by the 1-3-6 tetrahedral geometry:

1. The core (1) anchors the soliton at a BCC node.
2. The inner shell (3) defines the spin plane.
3. The outer shell (6) completes the octahedral coordination of the
   BCC unit cell.

This geometry is described in manuscript section "Structural
Emergence of the 1-3-6 Geometry".

The same value appears as the topological winding number Q=10 on the
spherical boundary S^2:

    Q = (1 / 4*pi) * integral over S^2 of Psi* omega = 10

This is presented in manuscript section "Topological Selection Rule
and the Electron Ground State".

The decadic resonance test is consistent with this geometry:

    r_e / r_nu = 100

    (r_e / r_nu)^5 = 10^10

which is the energy scaling between the electron (K_WC=10) and the
neutrino (K_WC=1). See manuscript section "Decadic Resonance Test:
r_e / r_nu ~ 100". K_WC=10 is therefore a geometric necessity, not a
fit.

Maintainer note (added at merge of PR #526), on what the decadic
resonance does and does not establish. r_nu is derived from the g_v
fixed point through alpha_geom and q_P, and K_WC enters none of those
steps: `derive_neutrino_radius(alpha_geom, q_P)` takes no K_WC and no
function it calls reads one. The ratio is therefore the same number
for K_WC = 9, 10 or 11, and cannot discriminate between them. The
second line is the fifth power of the first (`K_implied = ratio**5`
in `test_decadic_resonance`), so it adds no information to it. The
1-3-6 geometry and the winding number Q=10 above are the load-bearing
arguments for K_WC = 10; the ratio is a consistency observation,
measured at 100.000174 (0.000174%).

This structural necessity is a prediction of the model: the same
geometric configuration must be selected dynamically as the unique
stable ground state against neighbouring wave-center counts
(K_WC = 9 and K_WC = 11). Confirmation of this K-selectivity in the
OpenWave M4 solver remains an open task for the platform: it is
roadmap row [M4.1](../m4_roadmap.md) (harness merged in
[PR #205](https://github.com/openwave-labs/openwave/pull/205),
spec in `M4_k_selectivity_Formalization.md`), which is open and
unowned. Today's solver finds K = 2..10 degenerate.

### 2.2 Justification for L_mu, L_tau, and L_mu^2

The stability invariants are Fibonacci-Lucas numbers:

- L_mu = 5 = F_5 stabilizes the first shell (muon) as a 2D planar
  resonance.
- L_tau = 34 = F_9 stabilizes the second shell (tau) as a 3D
  volumetric resonance.

Their origin is the requirement that the wave-center configuration
close on the sphere according to the golden angle (~137.5 degrees)
and the principle of spherical phyllotaxis, as described in the
manuscript section "The Geometric Stability of 3D Wave-Packing".
Intermediate Fibonacci states do not provide a stable energy
distribution; only F_9 = 34 allows proper 3D closure on the BCC
vertices.

These invariants are introduced in manuscript section "Structural
Stability Invariants of the BCC Lattice".

The interface tension L_mu^2 = 25 is not an independent parameter.
It is the square of L_mu=5 and follows from the topological splitting
of phase space: a closed genus-1 surface embedded in a 3D sphere
divides the phase space into two equal regions. The muon core
therefore occupies half of the available geometric budget, which is
the structural origin of the A_pi/2 term in the tau shell coupling.
This mechanism is discussed in manuscript section "The Interface
Tension (delta)".

### 2.3 Justification for O_mu

The muon projection operator has a double derivation:

1. From the discrete lattice geometry:

   O_mu = M_mu * pi^3 * epsilon_M = 1/(4*pi^2)

   which follows from epsilon_M = 1/(8*pi^7*(1-zeta)) and the shell
   algorithm. The identity is satisfied to 0.14%.

2. From continuum Fourier theory:

   f(x) = (1/(2*pi)^2) * double integral of f(k) e^(i k x) d^2k

   where the normalization of the inverse 2D Fourier transform is
   exactly 1/(2*pi)^2.

This double convergence, one from discrete lattice geometry and one
from continuum Fourier analysis, is the signature of a structural
mathematical relation, not a fitted parameter.

The derivation is given in manuscript section "Final Results and
Experimental Correlation", in the remark titled "Remark on the
geometric origin of the muon projection operator", and in the
subsection "Dimensional Projection Operators".

## 3. Planck Self-Consistency and lambda_l

The reduced Planck constant, the gravitational constant, and the
lattice spacing are linked by the closed condition

    lambda_l = sqrt(hbar_geom * G_geom / c^3)

with

    hbar_geom = m_e * c * r_e / alpha_geom

This closure is not a definition imported from outside the model. It
is an internal algebraic constraint that simultaneously fixes the
gravitational scale, the quantum scale, and the fundamental lattice
spacing from the same BCC geometry.

The intended claim is that the fixed point exists and is unique, and
that the fixed point is a consequence of the BCC geometry. The
0.0276% deviation from the rounded CODATA Planck length is therefore a
consistency check, not an independent prediction of lambda_l.

In natural units (c = 1), the constraint reduces to

    lambda_l^2 = hbar_geom * G_geom

which is a purely geometric statement.

### 3.1 Manuscript references

- Section "Geometric Emergence of the Planck Length"
  (label: sec:lambda_l_geom)
- Section "Reduced Planck Constant from Geometric Alpha"
  (label: sec:hbar_from_alpha)
- Section "Geometric Self-Consistency of the Planck--Gravity--Metric
  Triangle" (label: sec:geometric_self_consistency)

## 4. Rigidity Tests and Falsifiability

The rigidity tests perturb N_geom, L_p_geom, lambda_l, r_e, m_e, and
epsilon_M to show that the model is sensitive to its inputs. Any
formula is sensitive; sensitivity alone is not evidence against
coincidence.

The falsifier is the absence of a shared lock-in point. If G, alpha,
and the AMMs required different values of N_geom to match their
respective targets, the unification would be spurious. The fact that
the same N_geom = 778.8025179 places all sectors on their
CODATA/experimental values is the non-coincidental feature.

A coincidence would likely manifest as different optimal N values for
different sectors, or as a flat response around the claimed lock-in
point.

### 4.1 Manuscript references

- Section "Robustness Analysis: Geometric Stability and Sensitivity"
  (label: sec:robustness)
- Section "Sensitivity of alpha^-1 to the Geometric Modulator
  N_geom" (label: sec:robustness_alpha)
- Section "Parametric Unification: The Stability Path of G and
  alpha" (label: sec:parametric_unification)
- Section "EWT Unification: Convergence of G and a_e on the Vacuum
  Stiffness Isentrope" (label: sec:unification_point_lock)

## 5. Atomic Scales as Consistency Identities

The atomic scales (R_inf, a0, lambda_C) are consistency identities
evaluated at alpha_geom and r_e = 100*r_nu. They are not independent
predictions. Their errors are propagated from alpha_geom and r_nu.

The independent predictions in the package are:

- alpha_geom (from A_pi and epsilon_M)
- G_geom (from the BCC chain)
- a_e, a_mu, a_tau (from epsilon_M and the recursive hierarchy)
- r_nu (from the g_v fixed point)

All other quantities are derived from these plus the four anchors.
lambda_l is deliberately not in this list: by section 3 it is the
fixed point of the Planck closure, a consistency check rather than an
independent prediction.

### 5.1 Manuscript references

- Section "The Rydberg Constant (R_inf)"
  (label: sec:rydberg_constant)
- Section "The Bohr Radius (a0)"
  (label: sec:bohr_radius)
- Section "The Electron Compton Wavelength (lambda_C)"
  (label: sec:compton_wavelength)
- Section "Consistency Across Atomic Scales"
  (label: sec:atomic_consistency)

## 6. Zero-Free-Parameter Statement

The model is not claimed to be absolutely parameter-free. It uses four
experimental anchors:

    r_e, m_e, c, e

These anchors are fixed by measurement and by metric convention. They
are not adjustable calibration parameters. If any one of them were
changed independently while keeping the BCC geometry fixed, the entire
set of geometric predictions would break.

The model therefore contains no free calibration constants in the
geometric sector; it is zero-free-parameter in the operational sense.

This statement is different from a claim of zero parameters. The
engine has no adjustable parameters, but it does have four measured
anchors and a number of structural constants. The distinction is:

- zero parameters: no inputs at all, not claimed here,
- zero free parameters: no fitted, calibrated, or adjustable inputs;
  fixed anchors and structural constants are not free parameters.

In this sense, the geometric sector of the model is zero-free-parameter:
no constant is tuned to improve agreement with data.

### 6.1 Manuscript references

- Section "Speed of Light as the Metric of Spacetime"
  (label: sec:c_as_metric)
- Section "The Geometric Status of Mass: Kilogram as a Derived Unit"
  (label: sec:mass_geometric_status)
- Section "Final Synthesis"
  (label: sec:final_synthesis)
- Section "Status of Inputs"
  (label: sec:inputs_planck)
