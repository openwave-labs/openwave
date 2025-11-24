# Wave Interaction (Reflection & Superposition)

## Table of Contents

1. [Boundary Reflection](#boundary-reflection)
   - [Detailed Reflection](#detailed-reflection)
1. [Wave Superposition](#wave-superposition)

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

### Detailed Reflection

**Wave Reflection** from boundaries and wave centers:

**At Universe Boundaries**:

- Waves reflect back into domain
- Perfect reflection with phase inversion: `ψ_reflected = -ψ_incident`
- Energy conserved (no absorption)
- Creates standing wave patterns near walls

**At Wave Centers** (particles):

- Particles act as reflectors (like internal boundaries)
- Inverts wave propagation direction
- Creates near-field wave patterns around particles
- Source of particle forces (MAP)

**Boundary Types**:

1. **Fixed (Dirichlet) boundaries**: `ψ = 0` at wall → phase inversion
2. **Free boundaries**: No phase inversion (not used in LEVEL-1)

**Implementation Note**: Fixed boundaries are implemented by never updating boundary voxels, keeping them at `ψ = 0`. Interior voxels "see" this zero value when computing Laplacian, creating natural reflection.

## Wave Superposition

**Superposition Principle**: Multiple waves combine linearly at each point.

**Types**:

- **Constructive**: Waves in phase → amplitude increases
- **Destructive**: Waves out of phase → amplitude decreases/cancels

**Why No Explicit Implementation is Needed**:

The PDE solver (wave equation + Laplacian) handles superposition **automatically**:

1. **Single ψ field**: The displacement field `ψ[i,j,k]` represents the *total* displacement at each voxel - already the sum of all wave contributions
2. **Charging phase**: Multiple wave centers inject energy into the same ψ field (additive)
3. **Propagation phase**: The wave equation evolves the combined field:

   ```text
   ψ_new = 2ψ - ψ_old + (c·dt)²·∇²ψ
   ```

4. **Interference emerges**: Constructive/destructive patterns arise naturally from the field dynamics

**No phase tracking required** - the wave's phase information is encoded in the displacement value and its time history (ψ, ψ_old). When two wavefronts meet, their displacements simply add in the shared ψ field, and the Laplacian propagates the result.

This is fundamentally different from ray-tracing or analytical approaches where you'd explicitly sum contributions from each source.
