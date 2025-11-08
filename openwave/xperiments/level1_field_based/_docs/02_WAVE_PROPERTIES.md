# WAVE PROPERTIES

## Table of Contents

1. [Summary](#summary)
1. [Scalar Properties (Magnitude)](#scalar-properties-magnitude)
   - [Speed (c)](#speed-c)
   - [Amplitude (A)](#amplitude-a)
   - [Wavelength (λ)](#wavelength-λ)
   - [Frequency (f)](#frequency-f)
   - [Density](#density)
   - [Energy](#energy)
   - [Phase (φ)](#phase-φ)
1. [Vector Properties (Direction + Magnitude)](#vector-properties-direction--magnitude)
   - [Wave Propagation Direction](#wave-propagation-direction)
   - [Amplitude Direction (Wave Mode)](#amplitude-direction-wave-mode)
   - [Velocity (Granules/Particles Only)](#velocity-granulesparticles-only)
   - [Force](#force)
1. [Field Storage in Taichi](#field-storage-in-taichi)
   - [Complete WaveField Class Implementation](#complete-wavefield-class-implementation)
   - [Field Categories](#field-categories)
1. [Property Relationships](#property-relationships)
1. [LEVEL-0 vs LEVEL-1 Properties](#level-0-vs-level-1-properties)

## SUMMARY

Wave field attributes represent physical quantities and wave disturbances stored at each voxel in the field-based medium. Properties are categorized as **scalar** (magnitude only) or **vector** (magnitude + direction).

**IMPORTANT**: Field values like amplitude and wavelength use **attometer scaling** for numerical precision with f32 storage, following the same principle as LEVEL-0. This prevents catastrophic cancellation in gradient calculations and maintains 6-7 significant digits of precision. See the complete `WaveField` class implementation below for details.

### Wave Medium

- MEDIUM-DENSITY (ρ): propagates momentum, carries energy, defines wave-speed
- WAVE-SOURCE: defines frequency, rhythm, vibration and charges energy

### Wave Form

- WAVE-MODE: longitudinal, transverse (polarization)
- WAVE-TYPE: standing, traveling

### Wave Energy

- WAVE-SPEED (c): constant, has direction of propagation
- WAVE-LENGTH (λ): changes when moving particle (doppler)
- WAVE-AMP (A): falloff at 1/r, near/far fields (max = color, min = black, zero = white-lines)
- WAVE-ENERGY (E): constant, conserved property (unmanifested <> manifested states)

### Wave Rhythm

- FREQUENCY (f): c / λ (can change)
- TIME: the wave's frequency, rhythm

### Wave Interaction

- INTERFERENCE: amplitude combinations (resonance, superposition) [sources diffs: phase, motion, freq]
- REFLECTION: changes direction of propagation (velocity vector)

### Notation

- ρ (rho) = medium density
- c = wave speed (speed of light)
- λ (lambda) = wavelength
- A = wave amplitude
- f = frequency (c / λ)
- ω (omega) = angular frequency (2πf)
- ωt = temporal oscillation (controls rhythm, time-varying component)
- φ (phi) = spatial phase shift (controls phase shift, wave relationship, interference, position-dependent component)
- k = wave number (2π/λ)
- t = time

## Scalar Properties (Magnitude)

### Speed (c)

**Wave Speed** (constant, medium property):

- **For granules** (LEVEL-0): Speed varies as `sin(ωt)` (oscillation)
- **For waves** (LEVEL-1): Constant propagation speed through medium
  - Depends on medium density and properties
  - Different for wave types:
    - Standing waves: nodes fixed
    - Traveling waves: constant `c`
    - Transverse vs longitudinal modes

**Storage**: Typically derived from medium properties, not stored per-voxel

### Amplitude (A)

**Maximum Displacement/Disturbance**:

- **Granule-based** (LEVEL-0):
  - Displacement from equilibrium position
  - Density fluctuation `ρ / ρ_avg`
  - Phase `φ` per granule
- **Field-based** (LEVEL-1):
  - Amplitude at voxel
  - Proportional to density: `A ∝ ρ`
  - Proportional to pressure: `A ∝ P`
  - Represents energy density at that location

**Storage**: `ti.field(dtype=ti.f32)` per voxel

**Physical meaning**:

- **At maximum displacement (amplitude)**: Maximum potential energy, zero velocity
  - Force is maximum (pulling back toward equilibrium)
  - All energy stored as potential energy in displacement/compression
- **At equilibrium position (zero displacement)**: Maximum kinetic energy, maximum velocity
  - Granules/voxels moving fastest through equilibrium
  - All energy is kinetic (motion)
  - Zero restoring force at this instant
- **Energy oscillation**: Energy continuously converts between kinetic ↔ potential
- **Total amplitude** determines total energy in the wave: `E_total ∝ A²`
- Negative amplitude = displacement in opposite direction from positive

### Wavelength (λ)

**Spatial Period of Wave**:

- Distance between successive wave crests/troughs
- **Not stored directly** in fields
- **Derived/measured** from spatial patterns
- Used to calculate voxel resolution: `dx = λ / points_per_wavelength`

**Calculation**: Measure distance between amplitude maxima in field

### Frequency (f)

**Temporal Period of Wave**:

- `f = c / λ` (wave equation)
- **Spatial frequency**: `ξ = 1/λ` (inverse wavelength)
- Can be stored per-voxel if multiple wave sources with different frequencies

**Storage**: Optional `ti.field(dtype=ti.f32)` if needed for multi-frequency waves

### Density

**Medium Density at Voxel**:

- Energy density
- Mass density (for matter simulations)
- Related to amplitude via equation of state

**Storage**: `ti.field(dtype=ti.f32)` per voxel

**Physical meaning**:

- Represents compression/rarefaction of medium
- Higher density = wave compression
- Lower density = wave rarefaction

### Energy

**Energy Density at Voxel**:

- **Kinetic energy** (motion): `E_k ∝ v²`
  - Maximum at equilibrium position (zero displacement)
  - Zero at maximum displacement (turning points)
- **Potential energy** (compression/displacement): `E_p ∝ A²`
  - Maximum at maximum displacement
  - Zero at equilibrium position
- **Total energy**: `E_total = E_kinetic + E_potential = constant`
- **Energy oscillation**: `E_k ↔ E_p` (continuously converts)
- **Amplitude-energy relationship**: `E_total ∝ A²` (energy proportional to amplitude squared)

**Storage**: `ti.field(dtype=ti.f32)` per voxel (optional, can be computed)

**Conservation**: Total energy must be conserved across entire field

**Wave cycle**:

1. At `t=0` (equilibrium): Max velocity, max KE, zero PE
2. At `t=T/4` (max displacement): Zero velocity, zero KE, max PE
3. At `t=T/2` (equilibrium, opposite): Max velocity, max KE, zero PE
4. At `t=3T/4` (max displacement, opposite): Zero velocity, zero KE, max PE

### Phase (φ)

**Wave Phase at Voxel**:

- Position within wave cycle (0 to 2π)
- Critical for interference patterns
- Determines constructive/destructive interference

**Storage**: `ti.field(dtype=ti.f32)` per voxel

**Physical meaning**:

- Phase difference determines interference
- `Δφ = 0, 2π, ...` → constructive
- `Δφ = π, 3π, ...` → destructive

## Vector Properties (Direction + Magnitude)

### Wave Propagation Direction

**Direction of Wave Travel**:

- Unit vector indicating propagation direction
- Can vary spatially (curved wavefronts, reflections)
- For spherical waves: radial from source
- For plane waves: uniform direction

**Storage**: `ti.Vector.field(3, dtype=ti.f32)` per voxel

**Physical meaning**:

- Points toward energy flow direction
- Orthogonal to wavefronts (for isotropic media)

### Amplitude Direction (Wave Mode)

**Direction of Displacement/Oscillation**:

- **Longitudinal waves**: Parallel to propagation direction
  - Compression waves
  - Sound-like waves
- **Transverse waves**: Perpendicular to propagation direction
  - Shear waves
  - EM-like waves (in EWT context)

**Storage**: `ti.Vector.field(3, dtype=ti.f32)` per voxel

**Physical meaning**:

- Defines wave polarization
- Non-linear for multi-source spherical waves
- Can point in all directions depending on wave superposition

### Velocity (Granules/Particles Only)

**Rate of Position Change**:

- **LEVEL-0 only**: Granules have velocity vectors
- **LEVEL-1**: Field voxels don't move, but can store flow velocity
- Can represent momentum density in field

**Storage**: `ti.Vector.field(3, dtype=ti.f32)` per voxel (for momentum)

**Physical meaning**:

- For particles: actual motion velocity
- For fields: local flow/current of energy

### Force

**Force Vector at Voxel**:

- Derived from amplitude gradients: `F ∝ -∇A`
- Points toward minimum amplitude (MAP: Minimum Amplitude Principle)
- Drives particle motion in LEVEL-1

**Storage**: `ti.Vector.field(3, dtype=ti.f32)` per voxel (computed)

**Physical meaning**:

- Gradient of potential (amplitude)
- Determines particle acceleration
- Source of emergent forces (gravity, EM, etc.)

## Field Storage in Taichi

### Complete WaveField Class Implementation

```python
import taichi as ti
from openwave.common import constants

@ti.data_oriented
class WaveField:
    """
    Wave field simulation using cell-centered grid with attometer scaling.

    This class implements LEVEL-1 field-based wave propagation with:
    - Cell-centered cubic grid
    - Attometer scaling for numerical precision (f32 fields)
    - Computed positions from indices (memory efficient)
    - Wave properties stored at each voxel
    """

    def __init__(self, nx, ny, nz, wavelength_m, points_per_wavelength=40):
        """
        Initialize wave field grid.

        Args:
            nx, ny, nz: Grid dimensions (number of voxels per axis)
            wavelength_m: Wavelength in meters (e.g., 2.854096501e-17 for energy wave)
            points_per_wavelength: Sampling rate (voxels per wavelength, default: 40)
        """
        # Grid dimensions
        self.nx, self.ny, self.nz = nx, ny, nz

        # ATTOMETER SCALING (critical for f32 precision)
        self.wavelength_am = wavelength_m / constants.ATTOMETER
        self.dx_am = self.wavelength_am / points_per_wavelength  # Voxel size in am

        # Physical sizes in meters (for external reporting)
        self.dx_m = self.dx_am * constants.ATTOMETER
        self.domain_size_m = [nx * self.dx_m, ny * self.dx_m, nz * self.dx_m]

        # SCALAR FIELDS (values in attometers where applicable)
        self.amplitude_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # am
        self.phase = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # radians (no scaling)

        # SCALAR FIELDS (no attometer scaling needed)
        self.density = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # kg/m³
        self.energy = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # J

        # VECTOR FIELDS (directions normalized, magnitudes may need scaling)
        self.wave_direction = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # unit vector
        self.amplitude_direction = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # unit vector
        self.velocity_am = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # am/s
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # N

    @ti.func
    def get_position_am(self, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.math.vec3:
        """Get physical position of voxel center in attometers."""
        return ti.Vector([
            (i + 0.5) * self.dx_am,
            (j + 0.5) * self.dx_am,
            (k + 0.5) * self.dx_am
        ])

    @ti.func
    def get_position_m(self, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.math.vec3:
        """Get physical position of voxel center in meters (for external use)."""
        pos_am = self.get_position_am(i, j, k)
        return pos_am * ti.f32(constants.ATTOMETER)

    @ti.func
    def get_voxel_index(self, pos_am: ti.math.vec3) -> ti.math.ivec3:
        """
        Get voxel index from position in attometers.

        Inverse mapping: position → index
        Used for particle-field interactions.
        """
        return ti.Vector([
            ti.i32((pos_am[0] / self.dx_am) - 0.5),
            ti.i32((pos_am[1] / self.dx_am) - 0.5),
            ti.i32((pos_am[2] / self.dx_am) - 0.5)
        ])

    @ti.kernel
    def compute_amplitude_gradient(self):
        """
        Compute force from amplitude gradient using attometer-scaled values.

        Force follows MAP (Minimum Amplitude Principle): F = -∇A
        Particles move toward regions of lower amplitude.
        """
        for i, j, k in self.amplitude_am:
            if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
                # Gradient in attometer space (better precision)
                grad_x = (self.amplitude_am[i+1,j,k] - self.amplitude_am[i-1,j,k]) / (2.0 * self.dx_am)
                grad_y = (self.amplitude_am[i,j+1,k] - self.amplitude_am[i,j-1,k]) / (2.0 * self.dx_am)
                grad_z = (self.amplitude_am[i,j,k+1] - self.amplitude_am[i,j,k-1]) / (2.0 * self.dx_am)

                # Force proportional to negative gradient (MAP principle)
                # Note: gradient is in am/am = dimensionless
                # Force scaling applied separately based on physical constants
                self.force[i,j,k] = -ti.Vector([grad_x, grad_y, grad_z])

    @ti.kernel
    def compute_laplacian(self, output: ti.template()):  # type: ignore
        """
        Compute Laplacian operator for wave equation (6-connectivity).

        Laplacian: ∇²A = (∂²A/∂x² + ∂²A/∂y² + ∂²A/∂z²)
        Used in wave equation: ∂²A/∂t² = c²∇²A
        """
        for i, j, k in self.amplitude_am:
            if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
                # 6-connectivity stencil (face neighbors only)
                laplacian = (
                    self.amplitude_am[i+1, j, k] + self.amplitude_am[i-1, j, k] +
                    self.amplitude_am[i, j+1, k] + self.amplitude_am[i, j-1, k] +
                    self.amplitude_am[i, j, k+1] + self.amplitude_am[i, j, k-1] -
                    6.0 * self.amplitude_am[i, j, k]
                ) / (self.dx_am * self.dx_am)

                output[i, j, k] = laplacian
```

### Field Categories

**Scalar fields with attometer scaling:**

```python
amplitude_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Attometers
```

**Scalar fields without attometer scaling:**

```python
phase = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Radians
density = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # kg/m³
energy = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Joules
frequency = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Hz (if multi-frequency)
```

**Vector fields:**

```python
wave_direction = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # Unit vector
amplitude_direction = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # Unit vector
velocity_am = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # am/s
force = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # N
```

## Property Relationships

Key wave equation relationships:

```python
# Wave equation fundamentals
f = c / λ              # Frequency from speed and wavelength
ω = 2 * pi * f         # Angular frequency
k = 2 * pi / λ         # Wave number
xi = 1 / λ             # Spatial frequency

# Energy relationships
E_total = E_kinetic + E_potential   # Total energy (conserved)
E_kinetic ∝ v²                      # Kinetic energy from velocity
E_potential ∝ A²                    # Potential energy from displacement
E_total ∝ A²                        # Total energy proportional to amplitude squared

# Energy oscillation in time
# At equilibrium (A=0): E_kinetic = max, E_potential = 0
# At max displacement (A=max): E_kinetic = 0, E_potential = max

# Force from amplitude gradient
F = -gradient(amplitude)        # MAP: move toward lower amplitude

# Density-amplitude relationship (equation of state)
density ∝ amplitude             # For compression waves
```

## LEVEL-0 vs LEVEL-1 Properties

| Property | LEVEL-0 (Granule) | LEVEL-1 (Field) |
|----------|-------------------|-----------------|
| **Position** | Per-granule vector | Computed from index: `(i+0.5)*dx` |
| **Velocity** | Per-granule oscillation | Optional momentum density |
| **Displacement** | From equilibrium | Amplitude at voxel |
| **Density** | Count granules in region | Direct field value |
| **Phase** | Per-granule phase | Per-voxel phase |
| **Amplitude** | Displacement magnitude | Direct field value |
| **Wave Direction** | Inferred from motion | Stored vector field |
| **Forces** | Inter-granule forces | Computed from gradients |

**Key Difference**: LEVEL-1 stores properties directly at fixed grid locations, while LEVEL-0 computes from moving particles.

---

**Status**: Properties defined, ready for wave engine implementation

**Next Steps**: Implement wave propagation using these field properties
