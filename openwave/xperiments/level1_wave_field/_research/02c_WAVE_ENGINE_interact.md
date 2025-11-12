# Wave Interaction (Reflection & Superposition)

## Boundary Reflection

Boundary walls emulate all matter in the universe reflecting the energy waves.
So, total energy is conserved inside that volume domain.

Boundary handling (Dirichlet boundary conditions):

1. Propagation loop: ti.ndrange((1, nx - 1), (1, ny - 1), (1, nz - 1))

- Only updates interior points (excludes boundaries at i=0, i=max, j=0, j=max, z=0, z=max)
- Boundary values remain at ψ = 0 (from initialization)

2. Laplacian operator:

- Accesses neighbors directly without bounds checking
- When called on interior points, it reads boundary values (which are always ψ = 0)

This creates :

- Fixed displacement ψ = 0 at all boundaries
- Acts like a rigid wall - waves should reflect back

The boundary behavior is NOT in the Laplacian itself - it's implemented through:

1. Keeping boundaries fixed at zero (never updated)
2. Interior points "see" zero at boundaries when computing Laplacian

## Wave Superposition

Wave superposition after reflection.
Superposition principle.

## CLAUDE REVIEW: OLD CONTENT Wave Interactions

### Boundary Reflections

**Reflection at Boundaries**: Waves reflect without energy loss.

**Boundary Types**:

1. **Hard boundaries** (universe walls):
   - Perfect reflection: `ψ_reflected = -ψ_incident`
   - Phase inversion for fixed boundaries
   - No phase inversion for free boundaries

2. **Wave centers** (particles):
   - Reflect waves like boundaries
   - Invert wavelet propagation direction
   - Source of emergent forces (MAP)

**Implementation**:

```python
# At boundary (e.g., i=0 face)
if i == 0:
    displacement[i,j,k] = -displacement[i+1,j,k]  # Hard reflection
```

### Reflection

**Wave Reflection** from boundaries and wave centers:

**At Universe Boundaries**:

- Waves reflect back into domain
- Energy conserved
- Creates standing wave patterns near walls

**At Wave Centers** (particles):

- Particles act as reflectors
- Inverts wave propagation direction
- Creates near-field wave patterns around particles
- Source of particle forces (MAP)

### Interference

**Superposition Principle**: Multiple waves combine linearly at each point.

**Types**:

- **Constructive**: Waves in phase → amplitude increases
  - `Δφ = 0, 2π, 4π, ...`
  - `A_total = A₁ + A₂`

- **Destructive**: Waves out of phase → amplitude decreases
  - `Δφ = π, 3π, 5π, ...`
  - `A_total = A₁ - A₂` (can cancel completely)

**Implementation**:

```python
# Multiple waves naturally interfere by summing contributions
for source in wave_sources:
    displacement[i,j,k] += source.contribution(i, j, k, t)
```
