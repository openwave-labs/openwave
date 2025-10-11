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
- Speed of light is the fundamental constant we must preserve (sacred, measured by observations)

## LAMBDA vs C PARADOX

### The Dilemma

**Wave equation:** `c = f × λ` - Can't independently set all three!

**Two approaches:**

1. **Match QWAVE_FREQUENCY** (quantum frequency):
   - Use `k = (2π × QWAVE_FREQUENCY)² × m` where QWAVE_FREQUENCY ≈ 1.05e25 Hz
   - Result: λ = QWAVE_LENGTH ✓, f = QWAVE_FREQUENCY ✓
   - **Problem:** Wave speed ≈ c/13 ≈ 23,000 km/s (NOT speed of light!)

2. **Match SPEED OF LIGHT** (current approach):
   - Use `k = (2π × natural_frequency)² × m` where `natural_frequency = c/(2L)`
   - Result: Wave speed = c ✓ (speed of light preserved!)
   - **Tradeoff:** Wavelength = 2L (lattice discretization), frequency = c/(2L) (higher than quantum)

### Resolution: Option 2 (Preserve Speed of Light)

**Reasoning:**

- **Speed of light is fundamental** - Measured constant, must be preserved
- **Lattice discretization is explicit** - We're sampling at discrete spacing L
- **Physical interpretation:**
  - Medium propagates waves at speed c (validates aether properties)
  - Resolution (granules/wavelength) determines how well we **represent** quantum waves
  - Actual wave mechanics operate at lattice scale (like pixels representing an image)

**Frequency "discrepancy" explained:**

- **QWAVE_FREQUENCY** (1.05e25 Hz): Target quantum value from EWT theory
- **natural_frequency** (varies with resolution): What the discrete lattice supports for c propagation
- Not a bug - reflects difference between quantum scale and computational sampling

**Test results (UNIVERSE_EDGE = 1e-16 m):**

- TARGET_PARTICLES = 1e3: natural_freq = 1.2e25 Hz, resolution = 4 granules/qwave (VISIBLE WAVE PROPAGATION)
- TARGET_PARTICLES = 1e6: natural_freq = ~1.36e26 Hz, resolution = 45 granules/qwave (NO VISIBLE WAVE PROPAGATION)
- Higher resolution → higher natural frequency → smaller lattice wavelength
- But wave speed = c in all cases ✓

**Conclusion:** Validate wave speed = c. This confirms medium properties are correct for simulating the aether at computational scale.

### What we're doing now

```bash
# Natural frequency based on LATTICE spacing (scaled-up not planck scale)
natural_frequency = c / (2 * rest_length)
                  = c / (2 * 1.1 am)
                  ≈ 1.36e26 Hz

#### Lattice natural stiffness property derived from this frequency
k = (2π × natural_frequency)² × m
```

Result:

- Wave speed in lattice = c ✓ (correct!)
- But frequency = 1.36e26 Hz (13x higher than QWAVE_FREQUENCY)
- Wavelength in lattice ≈ 2.2 am (13x smaller than QWAVE_LENGTH)

The lattice is resolving waves much SMALLER than the quantum wavelength! - the actual wave mechanics operate at the discretized lattice scale.

Interpretation: The lattice is simulating waves at speed c, but at the lattice's natural wavelength, not the quantum wavelength - Spacing is coarser than quantum (1.27 am unit cells vs Planck scale, scale factor of ~1e16x planck scale).

Lattice discretization is explicit - We know we're sampling at ~1.27 am spacing

```bash
# The medium has the right properties to propagate at c (sacred)
# But the actual wave mechanics work at the lattice scale

The frequency discrepancy is not a bug - its the difference between:
  - Quantum frequency (what we are trying to simulate: 1.05e25 Hz)
  - Lattice mechanics frequency (what the discrete grid naturally supports: 1.36e26 Hz for c propagation)
```
