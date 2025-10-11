# XPBD FINDINGS

Wavelength Validation:

- Visualize wave patterns (should see spherical/radial propagation)
- Check energy conservation (should be stable over time)
- I dont see waves propagating, maybe the energy injected is too small (just 1 granule, 1 granule mass, not much momentum/energy driving the waves)?

- Wave drivers frequency: f = QWAVE_SPEED / QWAVE_LENGTH ≈ 1.05e25 Hz (slowed by factor 1e25 → ~1 Hz visible)
- I noticed natural _frequency is showing a slight discrepancy from the injected frequency

considering the SLO_MO factor:

- Measure wave speed: Compare emergent propagation velocity to expected c = QWAVE_SPEED

- Measure wavelength: Track spatial period of oscillation, compare to λ = QWAVE_LENGTH (λ = c / f)
  - Method: Sample positions along radial line from vertex, measure distance between peaks, find spatial period
  - Expected: λ ≈ 2.854e-17 m (from constants)
  - Validates both stiffness k and lattice discretization
  - Relationship: λ = v / f, so if v ≈ c and f is correct → λ should match QWAVE_LENGTH

- This validates the entire physics model: correct k, m, lattice spacing, and wave equation

Success criteria: Wave speed ≈ c AND wavelength ≈ λ (within 5-10% tolerance), using real physics parameters (k, m, etc).
