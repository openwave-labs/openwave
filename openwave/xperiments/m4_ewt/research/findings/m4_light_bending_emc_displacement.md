# M4 Light Bending from EMC Density Gradient

## Criterion
Gravity: metric phenomena (light bending, with gravitational time
dilation treated as the same EMC-deformation effect; Lambda omitted)

## Status
⚠️ partial validation candidate

## Mechanism

In the Enhanced EWT framework, the speed of light is the structural
conversion factor between the spatial and temporal steps of the BCC
lattice:

\[
c \equiv \frac{\lambda_l}{t_p}.
\]

In the natural units of the lattice, \(c = 1\) and therefore
\([m] = [s]\). Consequently, a single EMC-density deformation
manifests simultaneously as:

- **light bending** — the ray follows the deformed lattice geometry
  produced by the EMC displacement field
  \(\vec{u}(r) = -\chi \nabla N_\nu(r)\),
- **gravitational time dilation** — a clock ticks more slowly
  because the geometric path required for each internal signal
  changes in the same density gradient.

Thus, within this model, a test of light bending is also a test of
the geometric mechanism underlying gravitational time dilation.
They are not independent phenomena.

## Method

- Assumed EMC density profile:
  \[
  N_\nu(r) = N_{\text{stat}} \left(1 - \frac{2r_s}{r}\right),
  \]
  where \(r_s = 2GM_{\odot}/c^2\).

- The displacement field is encoded by the scalar
  \[
  n(r) = 1 / \sqrt{1 - 2r_s/r},
  \]
  which is not an independent optical assumption but a convenient
  representation of the deformed EMC geometry.

- The bending angle is obtained from the standard ray integral in
  the variable \(u = R_{\odot}/r\):

  \[
  \Delta\theta
  = \frac{2r_s}{R_{\odot}}
    \int_0^1
    \frac{
      u\left(1 - \frac{2r_s u}{R_{\odot}}\right)^{-3/2}
    }{
      \sqrt{1-u^2}
    }
    \,du .
  \]

## Result

- Solar-limb bending angle:
  \(\Delta\theta = 1.751728\) arcsec
- Observed value: \(1.75\) arcsec
- Relative difference: \(0.099\%\)

## Free choices

- The profile \(N_\nu(r)\) is chosen as the simplest weak-field model
  consistent with the EMC push-out mechanism.
- \(n(r)\) is treated as the scalar encoding of the EMC displacement
  field; its direct derivation from BCC lattice elasticity remains
  open.

## Interpretation

The numerical result demonstrates that the EMC density-deficit
mechanism reproduces the observed solar light bending without
introducing spacetime curvature. Because the same mechanism also
defines the local clock geometry through \(c \equiv \lambda_l/t_p\),
the present test simultaneously covers the structural origin of
gravitational time dilation in the EWT framework.

Formally, the ray bends because of the gradient of the phase
velocity. In EWT, the physical carrier of this gradient is the
lattice deformation field \(\vec{u}(r)\), not an abstract optical
property.