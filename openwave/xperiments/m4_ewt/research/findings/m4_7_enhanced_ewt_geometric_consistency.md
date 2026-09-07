# M4.7 - Enhanced EWT Geometric Emergence Engine (Zero-Calibration)

## Summary

The M4.7 artifact is the zero-calibration geometric engine of the
Enhanced EWT model, version 5.0.0. It derives the effective geometric
stiffness N_geom directly from the BCC lattice packing fraction and the
ideal stiffness 8*pi^4, with no fitted parameters. The gravitational
constant emerges from the same chain as the fine-structure constant and
the lepton anomalous magnetic moments.

The artifact consists of four scripts (see task details), four of the six
files of the package archived at DOI: 10.5281/zenodo.22540262
(`ewt_robustness_plots.py` and `ewt_electroweak_bosons.py` are not ported).
Each file is content-identical to the archived one after line-ending
normalization (the archive ships CRLF endings, and a byte-order mark on the
engine), except for the `ewt_*` to `m4_7_ewt_*` module rename and, in the
three companion modules, the one import line that follows it. Verified at
merge against the record's MD5 checksums (maintainer note below).

## What the scripts deliver

### 1. Emergence of G from BCC geometry (zero calibration)

The gravitational constant is not calibrated to CODATA. The derivation
chain is:

    BCC packing fraction eta_BCC = sqrt(3)*pi/8
    -> zeta = (1 - eta_BCC) / (eta_BCC * 8*pi^4)
    -> N_geom = 8*pi^4 * (1 - zeta) = 778.8025179
    -> epsilon_M = 1 / (N_geom * pi^3) = 4.141169769e-5
    -> alpha_geom^-1 = (4*pi^3 + pi^2 + pi) - epsilon_M
    -> q_P = e / sqrt(alpha_geom)
    -> r_nu = q_P * S_tot  (S_tot from the g_v fixed point)
    -> lambda_l self-consistently from the Planck condition
    -> G_geom = (c^2*r_e/m_e) / A_pi * (1/(N_geom*A_pi))^3
                / (10 * sqrt(N_nu_eff))

The result is G_geom = 6.6775199755e-11, which differs from CODATA by
0.048169%.

### 2. Zero-calibration fine-structure constant

alpha_geom^-1 = 137.036262364, relative error 0.000192% vs CODATA.
The value is computed from A_pi and epsilon_M only. No measured alpha is
used as input.

### 3. Lepton anomalous magnetic moments without mass inputs

The full AMM predictions are:

    a_e   = 1159.916228 ppm  (0.022769% vs CODATA)
    a_mu  = 1166.212608 ppm  (0.025044% vs experiment)
    a_tau = 1176.838130 ppm  (0.031589% vs the Standard Model prediction)

The tau reference 1177.21 ppm is the Standard Model prediction, not a
measurement: no experiment reaches this precision (the PDG bound is
-0.052 < a_tau < 0.013). The script's `A_TAU_EXP` label is the package's
own wording and is left as archived.

The muon and tau AMMs are computed entirely from epsilon_M, the
recursive nodal growth law

    K_n = K_{n-1} + round(10^(n-1) * 2*pi^2)

with K_1 = 10, K_2 = 207, K_3 = 2181, and the dimensional projection
rules

    O_e   = 1
    O_mu  = 1/(4*pi^2)
    O_tau = 1

The identity O_mu = M_mu * pi^3 * epsilon_M is satisfied to within
0.14%. No lepton mass enters the AMM calculation.

### 4. Self-consistent Planck length and neutrino anchor

lambda_l is not taken from CODATA. It is obtained by combining the
geometric hbar_geom, the geometric G_geom, and the Planck-length
definition

    lambda_l = sqrt(hbar_geom * G_geom / c^3)

The solution gives lambda_l = 1.6166464e-35 m, relative error 0.0276%
vs the CODATA Planck length. The neutrino radius r_nu = 2.8179354e-17 m
is derived from the fixed point of g_v = 0.9835944447, not from r_e/100
as an input.

### 5. Atomic scales from the same geometry

The Rydberg constant, Bohr radius, and Compton wavelength are computed
from alpha_geom and r_e = 100*r_nu. All three are sub-ppm:

    R_inf    : 0.000403% error
    a0       : 0.000211% error
    lambda_C : 0.000019% error

### 6. Geometric self-consistency of the Planck-Gravity-Metric triangle

A central result of the engine is that G_geom, hbar_geom, lambda_l,
and c do not form a hierarchy of independent constants. They are linked
by the closed geometric condition

    lambda_l = sqrt(hbar_geom * G_geom / c^3)

where

    hbar_geom = m_e * c * r_e / alpha_geom

    G_geom = (G_Base / A_pi)
             * (1 / (N_geom * A_pi))^3
             * (1 / (K_WC * sqrt(N_nu_eff)))

and the effective density N_nu_eff itself depends on lambda_l through
the statutory background,

    N_nu_eff = (1 / X_eff) * (r_nu / (2 * e * lambda_l))^3

where e in the denominator is Euler's number.

This closure is not a definition imported from outside the model. It is
an internal algebraic constraint that simultaneously fixes the
gravitational scale, the quantum scale, and the fundamental lattice
spacing from the same BCC geometry.

In natural units (c = 1), the constraint reduces to the purely
geometric statement

    lambda_l^2 = hbar_geom * G_geom

showing that the square of the fundamental EMC length is the product of
the geometric quantum of action and the geometric gravitational
coupling.

### 7. Final synthesis: zero-free-parameter in the operational sense

The Enhanced EWT model demonstrates that the fundamental constants are
integrated resonances of a single BCC substrate. The geometric vacuum
stiffness N_geom, the magnetic deficit epsilon_M, and the geometric
core A_pi together determine the fine-structure constant, the
gravitational constant, the lepton anomalous magnetic moments, and the
principal atomic scales.

The model is not claimed to be absolutely parameter-free. It currently
uses four experimental anchors:

    r_e, m_e, c, e

These anchors are fixed by measurement and by the metric convention.
They are not adjustable calibration parameters: if any one of them
were changed independently while keeping the BCC geometry fixed, the
entire set of geometric predictions would break. The model therefore
contains no free calibration constants; it is zero-free-parameter in
the operational sense.

Of these anchors, c is a metric conversion factor rather than a
dynamical parameter. The electron mass m_e may also be derivable in
the future through the E proportional to r^5 scaling law and the
K_WC = 10 stability condition. The true irreducible anchors would then
reduce to r_e and e.

## What enters as a number

The G, alpha and lepton-AMM chain of the emergence engine is
zero-calibration in the operational sense. Its numerical inputs are:

| Constant | Value | Kind |
| --- | --- | --- |
| pi | 3.141592653589793 | mathematical |
| e (Euler) | 2.718281828459045 | mathematical |
| sqrt(2) | 1.4142135623730951 | mathematical |
| sqrt(3) | 1.7320508075688772 | mathematical |
| BCC coordination | 8 | crystallographic |
| BCC packing fraction | sqrt(3)*pi/8 | crystallographic |
| c | 299792458 m/s | SI definition |
| m_e | 9.1093837015e-31 kg | CODATA 2022 |
| r_e | 2.8179403262e-15 m | CODATA 2022 |
| e_charge | 1.602176634e-19 C | CODATA 2022 |

No calibrated stiffness, no calibrated projection, and no measured
alpha are used in that chain. The other numbers the package carries are
listed in the maintainer note below.

## Maintainer note (review of PR #523, 2026-09-06)

Added at merge so a reader can tell a prediction from an identity and a
geometric constant from a calibration. The author's prose above is
unchanged; this section is the platform's reading of the artifact.

### Other numbers the package carries

| Constant | Value | Where | Role |
| --- | --- | --- | --- |
| `K_WC` (= `K_1`) | 10 | engine, `main()` and `get_AMMi_K` | wave-center count of the electron; enters G through `1/(K_WC sqrt(N_nu_eff))` and `X_eff`, and the AMM ladder through `M_mu`, `M_tau` |
| `L_mu_dim`, `L_tau_dim` | 5, 34 | engine, `compute_lepton_amms` | dimensional counts in the muon and tau shell terms; `L_mu_dim^2 = 25` ppm is added to the tau shell as the interface term |
| `O_mu`, `O_tau` | `1/(4 pi^2)`, 1 | engine, `compute_lepton_amms` | projection operators; with `O_tau = 1` the tau "full" value equals the shell total identically |
| Shell prefactors | `3 A_pi pi^3 / (2 L_mu^2)`, `3 A_pi pi^3 / (8 sqrt 2) + A_pi / 2` | engine, `compute_lepton_amms` | structure of the shell terms; whether these are derived or postulated in v5.0.0 is the author's to state |
| `Q_P_INPUT`, `LAMBDA_L` | 1.87554603778e-18, 1.6162e-35 | engine, section 1.2 | reference values only; `LAMBDA_L` is a rounded Planck length (CODATA 2018: 1.616255e-35 m, which puts the derived `lambda_l` at 0.0242%) |
| EWT shell references | 248.8, 1177.21 ppm | `m4_7_ewt_amm_extended.py`, `m4_7_ewt_amm_resonance_scanner.py` | comparison targets for the shell terms, not inputs to the engine chain |
| `RHO_A`, `A_LONG`, `L_LONG` | 3.8598e22, 9.2154e-19, 2.8541e-17 | `m4_7_ewt_particle_masses.py` | Jeff Yee's EWT wave constants, themselves fitted to the electron; the spherical-mode masses inherit them |
| Orbital amplitudes | 185.68543, 3436.795 | `m4_7_ewt_particle_masses.py`, `mass_orbital` | mass-sector calibrations, unchanged from the v4.5.2 port; their targets in the spherical table (0.09488543, 1.75619909 GeV) are EWT reference masses, not the PDG values used in the full scan |

### Identities and propagated errors

| Quantity | What it is | Consequence |
| --- | --- | --- |
| `hbar_geom = m_e c r_e / alpha_geom` | the definition of the classical electron radius solved for hbar | its 0.000192% deviation is alpha's, not a second result |
| `R_inf`, `a0`, `lambda_C` | the textbook identities `alpha^3/(4 pi r_e)`, `r_e/alpha^2`, `2 pi r_e/alpha`, evaluated at `alpha_geom` and `100 r_nu` | with the CODATA alpha and r_e they hold to 3e-10%; the quoted sub-ppm errors are the alpha and `r_nu` errors propagated |
| `lambda_l = sqrt(hbar_geom G_geom / c^3)` | the Planck-length definition, solved for `lambda_l` as the fixed point of `G_geom(lambda_l)` | the closure holds by construction; its 0.0276% is half of G's 0.048% plus the rounded reference above. What is checkable is that the fixed point exists and is unique, which it is (an independent fixed-point iteration converges to the same value) |
| `q_P / e` vs `1/sqrt(alpha_geom)` | the same expression on both sides (`q_P := e / sqrt(alpha_geom)`) | the printed difference is identically 0 and cannot fail |
| `r_nu = q_P S_tot` | a charge in coulombs read as a length in meters | the identification is unit-dependent (the same formula with e in statcoulombs gives 8.4e-8); see the review thread |

The independent content of the chain is `alpha_geom^-1 = A_pi - 1/(N_geom pi^3)`
with `N_geom = 8 pi^4 (1 - zeta)`, and `G_geom` from the formula above with
the structural constants in the first table. With `zeta = 0` (`N = 8 pi^4`),
`alpha^-1` is unchanged to the digits shown and G lands 0.674% from CODATA,
so the packing-impedance term is what carries G from 0.67% to 0.048%. For
comparison, the v4.5.2 port took the CODATA Planck length as an input and
reached 4.78 ppm on G; the v5.0.0 engine derives `lambda_l` instead and
reaches 482 ppm.

### Verification at merge

Every value in this document and in the task document was recomputed from
the formulas above with an independent script (no import of the contributed
module; `lambda_l` solved by fixed-point iteration instead of the author's
closed form) and agrees to every printed digit. All four scripts were run in
an isolated copy of the Zenodo package (exit 0; the scanner writes its four
PDFs beside itself, into `research/scripts/`, which is not a tracked plot
location). The four ported files match the record's MD5 checksums after
line-ending normalization. This artifact supersedes the v4.5.2 port merged
in [PR #477](https://github.com/openwave-labs/openwave/pull/477)
(`ec2564af`), whose calibration table remains readable at that commit.

## Reference

Manuscript: Enhanced EWT, version 5.0.0
DOI: 10.5281/zenodo.22540635

Source scripts:
DOI: 10.5281/zenodo.22540262
