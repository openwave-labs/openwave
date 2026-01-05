# PHASE SHIFT: KEY ASPECT

## Conceptual Clarity

Why Separate ωt and φ is Superior:

```python
Position: x(t) = x_eq + A·cos(ωt + φ)·direction
Velocity: v(t) = -A·ω·sin(ωt + φ)·direction

Phase: φ = -kr, where
  k = 2π/λ is the wave number,
  r is the radial distance from center.
  (φ represents spatial phase shift; negative creates outward propagation)
```

This makes it clear that we have two independent controls:

- Time evolution (frequency domain)
- Phase relationships (spatial domain)

### Future Flexibility (Your Key Point!)

With separate factors, you can:

```python
# Example: Phase manipulation without changing frequency
φ = -k * r  # Current: simple radial phase
φ = calculate_interference_phase(r, other_sources)  # Future: multi-source
φ = apply_phase_shift_from_particle(r, particle_position)  # Future: particle interaction
```

### EWT Alignment

EWT is fundamentally about phase relationships between waves:

- Particle formation from constructive/destructive interference = phase relationships
- Standing waves = specific phase patterns
- Wave centers (K) creating particles = phase synchronization

Keeping φ explicit makes this physics visible in the code!

### Standard Physics Convention

```python
Actually, separating them IS the standard form:
x(t) = A·cos(ωt + φ₀)  ← initial phase φ₀

In your case: φ = -kr (spatially-varying phase based on position)
```

This is:

- ✅ Clear separation of temporal and spatial terms
- ✅ Flexible for future phase manipulation
- ✅ Aligned with EWT's phase-centric view
- ✅ Standard physics notation

Phase control is the key to implementing wave interactions, interference, and particle formation in the future. Keep φ as an independent, first-class parameter!
