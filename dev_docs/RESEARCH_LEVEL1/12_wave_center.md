# Wave Center Research - Standing Wave Formation

## Goals

1. **Make standing waves emerge** from energy waves bouncing in a simulation domain
1. **Isotropic wave field**: Waves have stable, diluted energy coming from all directions
1. **Wave center model**: Develop logic/script to make wave-center effects create standing wave patterns around the WC
   - High amplitude at r = 0, λ, 2λ, etc. (antinodes)
   - Zero amplitude at r = λ/4, 3λ/4, etc. (nodes)
1. **Particle formation**: WCs will be the building blocks for particle formation in EWT
1. **Visualization**: Clearly see standing waves (confirmed working via kinematic test)

## Standing Wave Physics Reference

Standing waves form from superposition of incoming + outgoing waves:

```text
ψ(r,t) = 2A·cos(ωt)·cos(kr)
```

Where:

- `k = 2π/λ` (wave number)
- Antinodes (max amplitude): r = 0, λ/2, λ, 3λ/2, ...
- Nodes (zero amplitude): r = λ/4, 3λ/4, 5λ/4, ...

**Key insight from PSU reference**: Standing waves require **two counter-propagating waves** of same frequency and amplitude. For a reflector:

- **Dirichlet (ψ=0)**: Hard boundary, phase inverts on reflection, node at boundary
- **Neumann (∂ψ/∂n=0)**: Soft boundary, no phase change, antinode at boundary

## Test Environment

- λ ≈ 30·dx (wavelength in grid units)
- Wave field charged from central oscillator, then stabilizes
- After ~2000 timesteps, energy is stable and isotropic
- Universe walls use Dirichlet BC (working correctly)

## Experiments Performed

### 1. Single Voxel Dirichlet (ψ=0)

**Approach**: Skip single voxel in propagation, force ψ=0

**Result**: Black dot visible, but no standing waves. Voxel too small relative to λ.

**Issue**: Single voxel (1dx) << λ (30dx), ineffective scatterer.

### 2. Single Voxel Signal Inversion

**Approach**: Invert displacement at single voxel after propagation

**Result**: Red-to-blue hue under redshift color, tiny black sphere (1 voxel radius).

**Issue**: Same size problem. Also, inverting after propagation creates discontinuities.

### 3. Spherical Dirichlet Boundary (r=8 voxels)

**Approach**: Skip propagation inside sphere, force ψ=0 inside

**Result**: Black sphere visible, but waves appear to ignore it. No standing waves.

**Issue**: In isotropic field, reflections from all directions cancel out in far field.

### 4. Spherical Neumann Boundary (∂ψ/∂n=0)

**Approach**: Skip propagation inside sphere, copy outer values to inner surface

**Result**: Same as Dirichlet — black sphere, no standing waves.

**Issue**: Same cancellation problem in isotropic field.

### 5. Cubic Dirichlet Boundary (16×16×16 voxels)

**Approach**: Cube aligned with grid axes, Dirichlet BC on all faces

**Result**: Black cube visible. During charging phase, wake/shadow visible (directional waves). After stabilization (isotropic field), no visible effect.

**Key observation**: Cube creates disturbance only with directional waves. In isotropic field, effects cancel.

### 6. Cubic Neumann Boundary

**Approach**: Same cube, but copy outer values to inner faces (zero gradient)

**Result**: Same as cubic Dirichlet — no standing waves in isotropic field.

### 7. Phase-Locking Single Point

**Approach**: Force WC voxel to oscillate coherently: `ψ = A·cos(ωt)`

**Result**: Something happens but no clear standing waves.

**Issue**: Single point doesn't create enough spatial structure.

### 8. Forced Standing Wave Pattern (Kinematic Test)

**Approach**: Directly enforce `ψ(r,t) = 2A·cos(ωt)·cos(kr)` within 2λ radius

**Result**: **SUCCESS** — Clear concentric rings visible. Confirms visualization works.

**Note**: This is forced/kinematic, not emergent from physics.

### 9. Lens Model - Multiplicative Amplification

**Approach**: Multiply displacement at WC by amplification factor each frame

**Result**: Unstable — explodes even at 1.37× amplification (4.1/3).

**Issue**: Exponential feedback: `amp^n` grows without bound.

### 10. Lens Model - Neighbor Average Amplification

**Approach**: Set WC = (average of 6 neighbors) × amplification

**Result**: Unstable — explodes at 3.0×, nothing at 2.5×.

**Issue**: Neighbors are affected by WC, creating feedback loop.

### 11. Lens Model - Tracker-Based Amplification

**Approach**: Set WC = (tracked amplitude EMA) × amplification × phase_sign

**Result**: Unstable — explodes at 2.5×.

**Issue**: Tracker at WC is affected by amplified WC value.

### 12. Lens Model - Clamp-Based (Amplitude Floor)

**Approach**: Only boost WC if |ψ| < min_amplitude, otherwise leave unchanged

**Result**: Stable, something happening, but no clear standing waves yet.

**Current state**: Most promising approach so far.

## Key Insights

### Why Wall Boundaries Work But WC Doesn't

**Walls (plane reflection)**:

- Incident wave: `sin(ωt - kx)` traveling in +x
- Reflected wave: `sin(ωt + kx)` traveling in -x
- Coherent interference → clear standing wave pattern

**Spherical/Cubic WC in isotropic field**:

- Waves arrive from ALL directions equally
- Each reflection is countered by wave from opposite direction
- Net effect: cancellation in far field
- WC becomes "invisible" to isotropic background

### The Isotropic Field Problem

In a uniform isotropic wave field:

- Every point already has waves passing through from all directions
- A reflector at any point reflects what's there back where it came from
- But there's always another wave coming from the opposite direction
- No NET change to the field structure

**This is why the cube creates wake/shadow during charging (directional) but disappears after stabilization (isotropic).**

### Jeff Yee's Two Models

From EWT author:

1. **Model #1 (Reflection)**: Wave center reflects incoming waves, creating standing waves
1. **Model #2 (Equilibrium)**: Wave center is simply the point where waves converge with equal amplitude; WC moves to stay at equilibrium

Both are mathematically equivalent in EWT. The question is implementation.

## Ideas for Future Testing

### 0. Spin Theory - Longitudinal to Transverse Conversion (HIGH PRIORITY)

**See `13_spin_theory.md` for full details.**

The missing physics may be **spin** — the conversion of longitudinal waves to transverse waves at the wave center. This could break the symmetry that causes cancellation in isotropic fields.

Key insight: Outgoing waves have different character (L+T) than incoming (pure L), so they don't cancel!

### 1. Return to Spherical Geometry

The cube may create strange disruptions since waves are spherical. Try:

- Spherical WC with radius ≈ λ/4 to λ/2
- More physically accurate for point-like particles

### 2. Active Source at WC (Not Just Passive)

Instead of reflecting/amplifying, make WC an active wave source:

- Emit spherical waves at frequency ω
- These interfere with background waves
- May need to balance emission to not add net energy

### 3. Modified Wave Equation at WC

Change the wave speed or add a potential well at WC:

- `c_local = c × factor` inside WC region
- Creates refractive index discontinuity
- Waves bend/focus around WC naturally

### 4. Resonant Cavity Approach

Place WC between two reflective boundaries:

- Standing waves form in the cavity
- WC becomes the "node organizer"

### 5. Multiple Wave Centers

Two or more WCs create standing waves between them:

- Like two mirrors forming a laser cavity
- May be necessary for stable standing wave formation

### 6. Non-Isotropic Initial Conditions

Test WC with directional wave field:

- Single plane wave hitting WC
- Should see clear reflection/standing wave
- Then gradually add more directions

### 7. Frequency-Locked Oscillation

Instead of amplitude manipulation, lock the PHASE at WC:

- WC always at phase φ = 0 (or some fixed value)
- Surrounding waves must conform to this phase anchor
- Different from amplitude amplification

### 8. Gradient-Based Attraction

WC creates amplitude gradient that "attracts" wave energy:

- Higher wave speed outside WC, lower inside
- Waves refract toward WC center
- Natural focusing without artificial amplification

## Current Best Approach: Clamp-Based Lens

```python
# Reference amplitude from wave field
ref_amplitude = base_amplitude_am * wave_field.scale_factor
min_amplitude = ref_amplitude * amplification

# Only boost if below minimum (preserve phase)
if ti.abs(current_val) < min_amplitude:
    phase_sign = 1.0 if current_val >= 0.0 else -1.0
    wave_field.displacement_am[cx, cy, cz] = phase_sign * min_amplitude
```

**Pros**:

- Stable (no exponential growth)
- Creates amplitude peak at WC
- Preserves wave phase

**Cons**:

- No clear standing waves yet
- May need larger effect region
- May need different physics approach entirely

## Next Steps

1. Try spherical geometry with clamp-based lens
1. Experiment with modified wave speed at WC (refractive approach)
1. Test with directional waves first, then gradually add isotropy
1. Consider multiple WCs for cavity-like standing wave formation
1. Research more on how EWT describes the actual mechanism of standing wave formation

## References

- [PSU Standing Wave Ratio Demo](https://www.acs.psu.edu/drussell/Demos/SWR/SWR.html)
- [PSU Reflection Demo](https://www.acs.psu.edu/drussell/Demos/reflect/reflect.html)
- EWT Papers in `/research_requirements/scientific_source/`
- `03_FUNDAMENTAL_PARTICLE.md` - Wave center architecture

---

**Status**: Research in progress

**Last Updated**: 2025-12-04

**Related Files**:

- `L1_wave_engine.py` - Wave propagation and WC test implementations
- `03_FUNDAMENTAL_PARTICLE.md` - Particle/WC theory
