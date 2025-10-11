# XPBD FINDINGS

## Simulation Validation

### Visualize wave patterns (should see spherical/radial propagation)

- Check energy conservation (should be stable over time)
- RESULT: I dont see waves propagating, maybe the energy injected is too small (just 1 granule, 1 granule mass, not much momentum/energy driving the waves or a stiff lattice)?

### Wave drivers

- frequency: f = QWAVE_SPEED / QWAVE_LENGTH ≈ 1.05e25 Hz (slowed by factor 1e25 → ~1 Hz visible)
- RESULT: I noticed natural _frequency is showing a slight discrepancy from the injected frequency

### Main Parameters to MATCH (considering the SLO_MO factor)

- Measure wave speed: Compare emergent propagation velocity to expected c = QWAVE_SPEED
  - RESULT: we need a way to measure that in the script (converted from SLO_MO)

- Measure wavelength: Track spatial period of oscillation, compare to λ = QWAVE_LENGTH (λ = c / f)
  - Suggested Method: Sample positions along radial line from vertex, measure distance between peaks, find spatial period
  - Expected: λ ≈ 2.854e-17 m (from constants)
  - Validates both stiffness k and lattice discretization
  - Relationship: λ = v / f, so if v ≈ c and f is correct → λ should match QWAVE_LENGTH
  - RESULT: we need a way to measure that in the script (converted from SLO_MO)

Success criteria: Wave speed ≈ c AND wavelength ≈ λ (within 5-10% tolerance), using real physics parameters (k, m, lattice sizes, wave equations).

- This will validate the entire physics model: correct k, m, lattice spacing, and wave equation
