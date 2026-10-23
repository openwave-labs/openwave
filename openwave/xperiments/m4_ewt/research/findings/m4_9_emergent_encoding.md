# M4.9 Emergent Metric Encodings from Lattice Dynamics

## Criterion
Gravity: local metric phenomena — foundational encoding derivation

## What was computed

A one-dimensional spring-mass lattice was simulated along a high-symmetry axis. The inter-site lattice spacing was treated as a physical length scale $a(\eta) = 1/\eta$, where $\eta$ represents the local linear site density along the propagation direction, and the per-site mass was kept fixed at $m_0$.

The local stiffness $k(\eta)$ was derived from a microscopic 1D pair potential

$$V(r) = \frac{V_0}{r^n}$$

evaluated at the constrained lattice spacing $a(\eta) = 1/\eta$ (for the standard EMC interaction, $n = 1$).

The resulting wave speed was measured in physical length units ($v_{\text{phys}} = a \cdot v_{\text{lattice}}$), not in discrete site-index units.

## Emergence chain

### 1. Microscopic potential to stiffness

The stiffness is the second derivative of the pair potential $V(r) = V_0 / r^n$ evaluated at the constrained lattice spacing $a(\eta) = 1/\eta$:

$$k(\eta) = \left. \frac{\mathrm{d}^2 V}{\mathrm{d}r^2} \right|_{r=1/\eta} = \frac{n(n+1) V_0}{a^{n+2}} \propto \eta^{n+2}$$

For the 1/r EMC pair potential ($n=1$):

$$k(\eta) \propto \eta^3$$

### 2. Stiffness and mass to physical wave speed

With fixed mass per lattice site $m_0$, the long-wavelength physical wave speed is

$$v_{\text{phys}} = a(\eta) \sqrt{\frac{k(\eta)}{m_0}}$$

For a general potential $V(r) \propto r^{-n}$ under linear spacing $a(\eta) = 1/\eta$, this gives:

$$v_{\text{phys}} \propto \eta^{-1} \cdot \sqrt{\eta^{n+2}} = \eta^{n/2}$$

For $n=1$, this scales as

$$v_{\text{phys}} \propto \eta^{+1/2}$$

### 3. Refractive index

The timelike wave speed is $v_{\text{phys}} \propto \eta^{+1/2}$; this is the clock exponent (M4.4: $v_{\text{clock}} = \sqrt{\eta}$).

The null-probe refractive index is not $1/v_{\text{phys}}$. A light ray samples both the spatial cell stretching ($\eta^{-1/2}$) and the temporal propagation-period dilation ($\eta^{-1/2}$) of the strained medium, giving

$$n_\gamma(\eta) = \sqrt{\frac{g_{rr}}{g_{00}}} = \frac{1}{\eta} = \left(1 - \frac{r_s}{r}\right)^{-1}$$

(manuscript v5.1.7, "Emergent Metric Encodings from Lattice Dynamics"; M4.3 and M4.5 use this form). This index is not measured by the script below; the script measures the timelike exponents $\alpha$ and $\beta$.

### 4. Clock frequency

For a standing-wave clock of fixed physical length $L$,

$$f \propto \frac{v_{\text{phys}}}{L} \propto \eta^{+1/2}$$

This is the same encoding used in M4.4.

## Measured results

### Summary

| Metric / Quantity | Model Relation | Calculated / Measured Exponent | Result / Status |
|---|---|---|---|
| Stiffness exponent $\alpha$ | $k(\eta) \propto \eta^3$ | $\alpha = 3.000002$ | **PASS** |
| Wave-speed exponent $\beta$ | $v_{\text{phys}}(\eta) \propto \eta^{0.5}$ | $\beta = 0.501714$ | **PASS** |
| Clock-frequency exponent $\gamma$ | $\gamma \equiv \beta$ | $\gamma = 0.501714$ | **Derived** |

### Stiffness exponent $\alpha$

| $\eta$ | $k$ |
|---|---|
| 0.4 | 0.128000 |
| 0.5 | 0.250000 |
| 0.6 | 0.432000 |
| 0.7 | 0.686001 |
| 0.8 | 1.024000 |
| 0.9 | 1.458000 |

Fitted exponent:

$$\alpha = 3.000002$$

Expected: $3.0$. This is the analytic derivative of the potential at the constrained spacing, not an independent simulation measurement.

### Wave-speed exponent $\beta$

| $\eta$ | $v_{\text{phys}}$ |
|---|---|
| 0.4 | 0.889680 |
| 0.5 | 0.998259 |
| 0.6 | 1.091577 |
| 0.7 | 1.180412 |
| 0.8 | 1.260294 |
| 0.9 | 1.337968 |

Fitted exponent:

$$\beta = 0.501714$$

Expected: $0.5$.

### Oscillator-frequency exponent $\gamma$

The oscillator frequency follows directly from $f = v_{\text{phys}}/L$. Therefore

$$\gamma = \beta = 0.501714$$

Expected: $0.5$.

Phase 3 is an analytical consistency check on phase 2 ($\gamma \equiv \beta$), not an independent simulation.

## Model assumptions

- **1D Axial Reduction:** One-dimensional reduction along a high-symmetry axis of the 3D BCC lattice, where $\eta$ represents the local linear site density along the line of propagation.
- **Axial Compression ($q = 1$):** purely axial lattice strain along the propagation vector, so the axial spacing is $a_\parallel(\eta) = \eta^{-1}$.
- **Fixed Site Mass:** Fixed mass per lattice site $m_0$.
- **EMC Pair Potential:** Microscopic pair potential $V(r) = V_0 / r$ ($n=1$).
- **Constrained Lattice Spacing:** Inter-site spacing $a(\eta) = 1/\eta$. Because $V(r)$ is purely repulsive ($V'(a) \neq 0$), the spacing $a$ is held by external lattice background pressure, not by an isolated potential minimum.

## Interpretation

Given the $1/r$ EMC pair potential ($n=1$) and the 1D linear spacing reduction $a(\eta) = 1/\eta$, the physical (timelike) wave speed scales as $v_{\text{phys}} \propto \eta^{n/2} = \eta^{+1/2}$.

The mapping to M4.3–M4.5 uses the null-probe index, not the timelike speed. In 1D axial strain ($q = 1$) the axial spacing is $a_\parallel(\eta) = \eta^{-1}$ and the timelike speed is $v_{\text{phys}} \propto \eta^{+1/2}$. The null-probe index is $n_\gamma = 1/v_\gamma = \sqrt{g_{rr}/g_{00}} = \eta^{-1}$, which is the encoding M4.3 and M4.5 use.

This establishes how the Schwarzschild encodings used in M4.3–M4.5 map back to microscopic lattice parameters:

- $v_{\text{clock}} \propto \sqrt{\eta}$ (timelike, $\beta = 0.5$)
- $n_\gamma = \eta^{-1}$ (null probe, from $\sqrt{g_{rr}/g_{00}}$, not from $1/v_{\text{phys}}$)

For a pair potential $V(r) \propto r^{-n}$ the timelike exponent is $\beta = n/2$ at $q = 1$. Under the physical assumption of a $1/r$ electrostatic-like EMC interaction, the timelike wave-speed exponent is $\beta = +1/2$ (clock, M4.4) and the null-probe index is $n_\gamma = \eta^{-1}$ (light, M4.3 and M4.5). The two exponents are distinct: a clock samples node availability ($\sqrt{\eta}$); a light ray samples the combined spatial and temporal strain ($\eta^{-1}$).

## Reference

Enhanced EWT manuscript, version 5.1.7:
[DOI: 10.5281/zenodo.23559858](https://doi.org/10.5281/zenodo.23559858)

Relevant section:

- „Emergent Metric Encodings from Lattice Dynamics"