# WAVE-FIELD PROPERTIES

## Table of Contents

1. [Summary](#summary)
1. [WAVE PROPERTIES Terminology & Notation](#wave-properties-terminology-and-notation)
1. [Scalar Properties (Magnitude)](#scalar-properties-magnitude)
   - [Speed (c)](#speed-c)
   - [Displacement (ψ)](#displacement-ψ)
   - [Amplitude (A)](#amplitude-a)
   - [Frequency (f)](#frequency-f)
   - [Wavelength (λ)](#wavelength-λ)
   - [Density](#density)
   - [Energy](#energy)
   - [Phase (φ)](#phase-φ)
1. [Vector Properties (Direction + Magnitude)](#vector-properties-direction--magnitude)
   - [Wave Propagation Direction](#wave-propagation-direction)
   - [Displacement Direction (Wave Mode)](#displacement-direction-wave-mode)
   - [Velocity (Granules/Particles Only)](#velocity-granulesparticles-only)
   - [Force](#force)
1. [Field Storage in Taichi](#field-storage-in-taichi)
   - [Complete WaveField Class Implementation](#complete-wavefield-class-implementation)
   - [Initialization Comparison: LEVEL-0 vs LEVEL-1](#initialization-comparison-level-0-vs-level-1)
   - [Field Categories](#field-categories)
1. [Property Relationships](#property-relationships)
1. [Notation Clarification: ψ vs A](#notation-clarification-ψ-vs-a)
1. [LEVEL-0 vs LEVEL-1 Properties](#level-0-vs-level-1-properties)

## SUMMARY

Wave field attributes represent physical quantities and wave disturbances stored at each voxel in the wave-field medium. Properties are categorized as **scalar** (magnitude only) or **vector** (magnitude + direction).

**IMPORTANT**: Field values use **scaled SI units** for numerical precision with f32 storage, following the same principle as LEVEL-0. This prevents catastrophic cancellation in gradient and derivative calculations and maintains 6-7 significant digits of precision. See the complete `WaveField` class implementation below for details.

**Initialization**: LEVEL-1 mirrors LEVEL-0's initialization strategy, computing voxel size from `config.TARGET_VOXELS = 1e9` (1 billion voxels, 1000× more than LEVEL-0's 1M granules). This is tractable because LEVEL-1 doesn't render every voxel individually - only field slices, isosurfaces, or sample points are visualized, eliminating the rendering bottleneck. Result: **10× better spatial resolution** per dimension while maintaining interactive performance.

**Scaling Constants**:

- Spatial: `ATTOMETER = 1e-18` (m) - attometer scale
- Temporal: `RONTOSECOND = 1e-27` (s) - rontosecond scale

**Naming Convention**:

- `dx_am`: spatial step in attometers (scaled), `dx`: in meters (SI)
- `dt_rs`: time step in rontoseconds (scaled), `dt`: in seconds (SI)
- Variable/field suffix `_am` → use `constants.ATTOMETER` to convert
- Variable/field suffix `_rs` → use `constants.RONTOSECOND` to convert

## WAVE PROPERTIES Terminology and Notation

- Refer to [`wave_notation.md`](../../openwave/common/wave_notation.md)

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

### Displacement (ψ)

- **ψ (displacement)**: Instantaneous wave displacement at high frequency (~10²⁵ Hz)
  - What propagates via wave equation: ∂²ψ/∂t² = c²∇²ψ
  - Oscillates rapidly between positive and negative
  - Used for wave propagation mechanics

### Amplitude (A)

**Maximum Displacement/Disturbance**:

- **A (amplitude)**: Envelope of |ψ| (slowly varying maximum)
  - A = max|ψ| over time (running maximum)
  - Used for energy density: u = ρ(fA)² (EWT, no ½ factor, frequency-based)
  - Used for forces: F = -2ρVfA × [f∇A + A∇f] (full form) or F = -2ρVf² × A∇A (monochromatic, ∇f = 0)
  - Particles respond to envelope A, not instantaneous ψ oscillations
  - Note: f = c/λ embeds wave speed c

**Field-based** (LEVEL-1):

- Store BOTH fields: `psiL_am` (ψ) and `amplitude_am` (A)
- Amplitude proportional to density: `A ∝ ρ`
- Amplitude proportional to pressure: `A ∝ P`
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

### Frequency (f)

**Temporal Period of Wave** (PRIMARY wave property):

- **Primary property**: Measured directly from temporal oscillations (f = 1/dt)
- **Embeds wave speed**: f = c/λ incorporates constant c
- **Natural pairing**: f (temporal) × A (spatial) in energy formula E = ρV(fA)²
- **Human-intuitive**: Radio (98.7 FM), audio (440 Hz), WiFi (2.4 GHz)
- **Planck alignment**: E = hf (energy proportional to frequency)
- **Spatial frequency**: ξ = 1/λ = f/c (inverse wavelength, derived)
- Can be stored per-voxel if multiple wave sources with different frequencies

**Storage**: Optional `ti.field(dtype=ti.f32)` if needed for multi-frequency waves

### Why Frequency-Centric? (Design Rationale)

This implementation uses **frequency (f) as the primary wave property** rather than wavelength (λ), with λ treated as a derived quantity. This design decision is based on multiple converging considerations:

#### 1. Mathematical Elegance

Energy formula becomes dramatically simpler:

```text
Frequency-centric: E = ρV(fA)²
vs.
Wavelength-based:  E = ρVc²(A/λ)²
```

The frequency-based formulation:

- Eliminates explicit c² factor (c is embedded in f via f = c/λ)
- Removes division by λ (cleaner arithmetic)
- More compact and readable

Force formula maintains equivalent complexity:

```text
Frequency-centric: F = -2ρVfA × [f∇A + A∇f]
Wavelength-based:  F = -2ρVc² × [A∇A/λ² - A²∇λ/λ³]
```

Both have two terms (amplitude gradient + wavelength/frequency gradient), but the frequency version:

- Uses multiplication instead of division
- Has simpler dimensional structure
- Monochromatic case: F = -2ρVf² × A∇A is symmetric and elegant

#### 2. Spacetime Coupling (Fundamental Physics Insight)

The product **fA represents natural spacetime coupling**:

- f: Temporal domain (frequency, 1/seconds)
- A: Spatial domain (amplitude, meters)
- fA: Spacetime product (Hz·m = m/s, related to velocity)

This mirrors fundamental physics concepts:

- Frequency describes "how fast" in time
- Amplitude describes "how much" space is displaced
- Their product fA captures the complete wave character
- Energy E ∝ (fA)² shows energy depends on spacetime coupling squared

This may relate to spacetime curvature: the Laplacian ∇²ψ in the wave equation measures local curvature of the displacement field, and the factor fA determines how this curvature "stretches" spacetime.

#### 3. Quantum Mechanics Alignment (Planck Relation)

Planck's quantum energy relation:

```text
E = hf    (energy proportional to frequency)
```

The frequency-centric formulation E = ρV(fA)² naturally aligns with this:

- Energy proportional to f² (frequency squared)
- NOT proportional to 1/λ² (inverse wavelength squared)
- Quantum mechanics uses frequency as fundamental, not wavelength
- Our classical wave formulation matches quantum intuition

#### 4. Human Intuition and Real-World Usage

Frequency is the natural parameter across multiple domains:

- **Radio/telecommunications**: FM 98.7 MHz (frequency, not wavelength)
- **Audio/acoustics**: A4 = 440 Hz (musical pitch by frequency)
- **WiFi/RF**: 2.4 GHz, 5 GHz (frequency bands)
- **Medical**: Ultrasound 1-10 MHz (frequency)
- **Brain waves**: Alpha 8-13 Hz, Beta 13-30 Hz (frequency)

**Radio analogy (double match)**:

- **AM/FM**: Amplitude Modulation / Frequency Modulation
- A (amplitude) and f (frequency) are the fundamental wave properties
- Radio controls map directly to wave parameters:
  - Volume knob = Sound Amplitude (A)
  - Station dial = Transmission Frequency (f)

People think in terms of frequency, not wavelength:

- "440 Hz A note" is intuitive
- "0.78 m wavelength sound" is not

#### 5. Direct Temporal Measurement

Frequency is measured directly from time-domain observations:

```python
# Measure period T between wave peaks
dt = time_peak_n - time_peak_(n-1)

# PRIMARY property computed immediately
f = 1.0 / dt    # Frequency (Hz)

# DERIVED properties
T = dt          # Period (same value, different name)
λ = c / f       # Wavelength (requires conversion)
```

This measurement hierarchy is natural:

1. Observe temporal oscillations (direct)
1. Measure time period dt (direct)
1. Compute frequency f = 1/dt (immediate)
1. Derive wavelength λ = c/f (when needed for spatial design)

#### 6. Frequency Domain Analysis

Multi-frequency superposition is natural in frequency space:

- Fourier decomposition operates in frequency domain
- Each mode characterized by (A_mode, f_mode) pair
- Energy per mode: E_mode = ρV(A_mode × f_mode)²
- Total energy: E_total = Σ E_mode

Frequency-centric formulation aligns with:

- Signal processing (frequency spectrum)
- Fourier analysis (frequency components)
- Harmonic decomposition (frequency modes)

#### 7. Information Content

Frequency carries more information than wavelength:

```text
f = c/λ  (frequency incorporates constant c)
```

Since c is constant in the medium:

- f uniquely determines λ via λ = c/f
- λ uniquely determines f via f = c/λ
- But f = c/λ shows f "already contains" the wave speed c
- f carries both temporal oscillation rate AND spatial periodicity (via c)

#### 8. Scientific Rigor Maintained

The frequency-centric formulation is **equally rigorous** as wavelength-based:

- f = c/λ embeds the speed of light (no information lost)
- All physics is preserved (force equations equivalent)
- Dimensional analysis consistent: [f²] = [1/s²] provides same scaling as [c²/λ²]
- Aligns with quantum mechanics convention (Planck uses f, not λ)

#### 9. Practical Implementation

Code becomes cleaner and more intuitive:

```python
# Energy calculation (frequency-centric)
energy = rho * V * (A * f)**2                    # Clean!

# vs. wavelength-based
energy = rho * V * c**2 * (A / wavelength)**2    # More complex

# Force calculation (monochromatic)
force_scale = 2.0 * rho * V * f**2               # Simple
F = -force_scale * A * grad_A

# vs. wavelength-based
force_scale = 2.0 * rho * V * c**2 / wavelength**2   # Division
F = -force_scale * A * grad_A
```

#### 10. Wavelength Used When Needed

Wavelength is still available when required for spatial design:

```python
# Convert to wavelength for spatial layout
wavelength = c / frequency

# Voxel spacing from wavelength
dx = wavelength / points_per_wavelength

# Spatial pattern design
def design_wave_pattern(frequency):
    wavelength = constants.EWAVE_SPEED / frequency
    return create_spatial_pattern(wavelength)
```

#### Summary: Why Frequency is Primary

| Aspect | Frequency (f) | Wavelength (λ) |
|--------|---------------|----------------|
| **Energy formula** | E = ρV(fA)² ✓ Elegant | E = ρVc²(A/λ)² (complex) |
| **Measurement** | Direct: f = 1/dt ✓ | Indirect: λ = c/f |
| **Information** | Embeds c: f = c/λ ✓ | Spatial only |
| **Spacetime** | A×f couples space×time ✓ | A/λ is less natural |
| **Quantum** | E = hf (Planck) ✓ | E ∝ 1/λ (derived) |
| **Human use** | Radio, audio, WiFi ✓ | Scientific contexts |
| **Domain** | Temporal/frequency ✓ | Spatial |
| **Rigor** | Full (f = c/λ) ✓ | Full |

**Conclusion**: Frequency-centric formulation is more elegant, more intuitive, aligns with quantum mechanics, and maintains full scientific rigor. Wavelength is computed as λ = c/f when needed for spatial design tasks.

### Wavelength (λ)

**Spatial Period of Wave** (DERIVED from frequency):

- Distance between successive wave crests/troughs
- **Derived property**: λ = c/f (computed from frequency)
- **Not stored directly** in fields (unless measured independently)
- Used to calculate voxel resolution: `dx = λ / points_per_wavelength`

**Calculation**: λ = c/f (from frequency) or measure distance between amplitude maxima in field

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
- **Potential energy** (compression/displacement): `E_p ∝ (fA)²`
  - Maximum at maximum displacement
  - Zero at equilibrium position
- **Total energy**: `E_total = E_kinetic + E_potential = constant`
- **Energy oscillation**: `E_k ↔ E_p` (continuously converts)
- **Amplitude-frequency relationship**: `E_total ∝ (fA)²` (energy proportional to frequency × amplitude squared)
- **EWT energy formula**: `E = ρV(fA)²` (frequency-centric, no ½ factor)

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

### Displacement Direction (Wave Mode)

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

- Derived from amplitude gradients (frequency-based formulation):
  - Full form: `F = -2ρVfA × [f∇A + A∇f]` (dual-term with amplitude and frequency gradients)
  - Monochromatic: `F = -2ρVf² × A∇A` (when ∇f = 0, single wave source)
- Points toward minimum amplitude (MAP: Minimum Amplitude Principle)
- Drives particle motion in LEVEL-1
- Frequency-centric: f² provides natural 1/s² dimensional scaling

**Storage**: `ti.Vector.field(3, dtype=ti.f32)` per voxel (computed)

**Physical meaning**:

- Gradient of energy potential u = ρ(fA)²
- Determines particle acceleration
- Source of emergent forces (gravity, EM, etc.)
- Dual-term structure captures both amplitude variation and frequency variation contributions

## Field Storage in Taichi

### Complete WaveField Class Implementation

```python
import taichi as ti
from openwave.common import config, constants

@ti.data_oriented
class WaveField:
    """
    Wave field simulation using cell-centered grid with attometer scaling.

    This class implements LEVEL-1 wave-field propagation with:
    - Cell-centered cubic grid
    - Attometer scaling for numerical precision (f32 fields)
    - Computed positions from indices (memory efficient)
    - Wave properties stored at each voxel
    - Asymmetric universe support (nx ≠ ny ≠ nz allowed)

    Initialization Strategy (mirrors LEVEL-0 BCCLattice):
    1. User specifies init_universe_size [x, y, z] in meters (can be asymmetric)
    2. Compute universe volume and target voxel count from config.TARGET_VOXELS
    3. Calculate cubic voxel size: dx = (volume / target_voxels)^(1/3)
    4. Compute grid dimensions: nx = int(x_size / dx), ny = int(y_size / dx), nz = int(z_size / dx)
    5. Recalculate actual universe size to fit integer voxel counts

    Alternative: User can specify points_per_wavelength instead, which determines dx from wavelength.
    """

    def __init__(self, init_universe_size):
        """
        Initialize WaveField from universe size with automatic voxel sizing
        and asymmetric universe support.

        Args:
            init_universe_size: List [x, y, z] in meters (can be asymmetric)

        Design:
            - Voxel size (dx) is CUBIC (same for all axes) - preserves wave physics
            - Grid counts (nx, ny, nz) can differ - allows asymmetric domain shapes

        Returns:
            WaveField instance with optimally sized voxels for target_voxels
        """

        if wavelength is None:
            wavelength = constants.EWAVE_LENGTH
        if target_voxels is None:
            target_voxels = config.TARGET_VOXELS

        # Compute universe volume
        init_volume = init_universe_size[0] * init_universe_size[1] * init_universe_size[2]

        # Calculate cubic voxel size from target voxel count
        # voxel_volume = universe_volume / target_voxels
        # dx³ = voxel_volume → dx = voxel_volume^(1/3)
        voxel_volume = init_volume / target_voxels
        dx = voxel_volume ** (1/3)  # Cubic voxel edge

        # Calculate grid dimensions (integer voxel counts per axis)
        nx = int(init_universe_size[0] / dx)
        ny = int(init_universe_size[1] / dx)
        nz = int(init_universe_size[2] / dx)

        # Compute points per wavelength (for consistency)
        points_per_wavelength = wavelength / dx

        # Grid dimensions (asymmetric support: nx ≠ ny ≠ nz allowed)
        self.nx, self.ny, self.nz = nx, ny, nz

        # ATTOMETER SCALING (critical for f32 precision)
        self.wavelength_am = wavelength_m / constants.ATTOMETER
        self.dx_am = self.wavelength_am / points_per_wavelength  # Cubic voxel size in am

        # Physical sizes in meters (for external reporting, asymmetric)
        self.dx_m = self.dx_am * constants.ATTOMETER
        self.domain_size_m = [nx * self.dx_m, ny * self.dx_m, nz * self.dx_m]  # Can differ per axis

        # SCALAR FIELDS (values in attometers where applicable)
        # Wave equation fields (leap-frog scheme requires three time levels)
        self.psiL_old_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # am (ψ at t-dt)
        self.psiL_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # am (ψ at t, instantaneous)
        self.psiL_new_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # am (ψ at t+dt)
        self.amplitude_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # am (envelope A = max|ψ|)
        self.phase = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # radians (no scaling)

        # SCALAR FIELDS (no attometer scaling needed)
        self.density = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # kg/m³
        self.nominal_energy = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # J

        # VECTOR FIELDS (directions normalized, magnitudes may need scaling)
        self.wave_direction = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # unit vector
        self.amplitude_direction = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # unit vector
        self.velocity_am = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # am/s
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # N

        # Compute actual universe size (rounded to integer voxel counts)
        self.actual_universe_size = [nx * self.dx_m, ny * self.dx_m, nz * self.dx_m]
        self.actual_voxel_count = nx * ny * nz

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
```

### Initialization Comparison: LEVEL-0 vs LEVEL-1

| Aspect | LEVEL-0 (BCCLattice) | LEVEL-1 (WaveField) |
|--------|---------------------|-------------------|
| **Config constant** | `TARGET_GRANULES = 1e6` | `TARGET_VOXELS = 1e9` |
| **Target count** | 1 million granules | 1 billion voxels (1000× more!) |
| **Performance rationale** | Every granule rendered as Taichi particle | Only field slices/surfaces rendered |
| **Initialization** | `BCCLattice(init_universe_size)` | `WaveField(init_universe_size)` |
| **Unit cell/voxel** | `unit_cell_edge` from cube root | `dx` from cube root |
| **Grid dimensions** | `grid_size = [nx, ny, nz]` (asymmetric) | `self.nx, self.ny, self.nz` (asymmetric) |
| **Cubic constraint** | Unit cell edge same for all axes | Voxel size same for all axes |
| **Actual size** | `[nx*edge, ny*edge, nz*edge]` | `[nx*dx, ny*dx, nz*dx]` |
| **Memory per unit** | 3D position + 3D velocity + displacement (7 floats) | Just field value (1 float scalar or 3 floats vector) |

**Why 1000× More Voxels in LEVEL-1?**

1. **Rendering bottleneck in LEVEL-0**:
   - Every granule = Taichi particle rendered individually
   - GPU particle rendering scales poorly beyond ~1M particles
   - Visual clutter with too many particles

2. **No rendering bottleneck in LEVEL-1**:
   - Voxels not rendered individually
   - Only field slices, isosurfaces, or sample points visualized
   - Can handle billions of voxels without rendering overhead

3. **Better spatial resolution**:
   - 1e9 voxels = ~1000³ = 1000 voxels per axis
   - 1e6 granules = ~100³ = 100 granules per axis (in BCC ~86³)
   - **10× better spatial resolution** in each dimension!

4. **Memory still tractable**:
   - 1e9 voxels × 4 bytes (f32) = 4 GB per scalar field
   - Modern GPUs handle this easily (8-24 GB VRAM typical)
   - LEVEL-0: 1e6 granules × 28 bytes = 28 MB (much less per granule, but fewer of them)

**Usage Example**:

```python
from openwave.common import config

# LEVEL-0: Granule-Motion (1M granules)
lattice = BCCLattice(
    init_universe_size=[250e-18, 250e-18, 125e-18]
)
# Result: ~86×86×43 granule grid, TARGET_GRANULES = 1e6

# LEVEL-1: Field-based (1B voxels)
wave_field = WaveField(
    init_universe_size=[250e-18, 250e-18, 125e-18]
)
# Result: ~1260×1260×630 voxel grid, TARGET_VOXELS = 1e9
# Spatial resolution: ~15× better per dimension!
```

### Field Categories

**Note on Asymmetric Universes**: All field declarations use `shape=(nx, ny, nz)` where `nx`, `ny`, `nz` can differ. This allows memory-efficient asymmetric domains (e.g., 350×350×175 for a flat universe) while maintaining cubic voxels (same `dx` for all axes).

**Scalar fields with attometer scaling:**

```python
psiL_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Attometers (instantaneous ψ)
amplitude_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Attometers (envelope A)
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

Key wave equation relationships (frequency-centric):

```python
# Wave equation fundamentals (frequency-centric)
f = c / λ              # Frequency from speed and wavelength (PRIMARY property)
λ = c / f              # Wavelength derived from frequency (DERIVED property)
ω = 2 * pi * f         # Angular frequency
k = 2 * pi / λ         # Wave number
xi = 1 / λ = f / c     # Spatial frequency (derived from frequency)

# Energy relationships (frequency-based)
E_total = E_kinetic + E_potential   # Total energy (conserved)
E_kinetic ∝ v²                      # Kinetic energy from velocity
E_potential ∝ (fA)²                 # Potential energy from frequency × amplitude
E_total ∝ (fA)²                     # Total energy proportional to (frequency × amplitude)²
E = ρV(fA)²                         # EWT energy formula (frequency-centric, no ½ factor)

# Energy oscillation in time
# At equilibrium (A=0): E_kinetic = max, E_potential = 0
# At max displacement (A=max): E_kinetic = 0, E_potential = max

# Force from amplitude gradient (frequency-based)
F = -2ρVfA × [f∇A + A∇f]       # Full form (dual-term with amplitude and frequency gradients)
F = -2ρVf² × A∇A                # Monochromatic (∇f = 0, single frequency)

# Density-amplitude relationship (equation of state)
density ∝ amplitude             # For compression waves
```

### Point Properties

**Stored at Each Voxel**:

- Displacement (ψ): Instantaneous oscillating value at [i,j,k]
- Amplitude (A): Envelope, running maximum of |ψ| at [i,j,k]
- Density: Local compression/rarefaction
- Speed: Oscillation velocity at point
- Direction: Wave propagation direction at point
- Phase: Position in wave cycle

**Direct Access**:

```python
psi = psiL_am[i, j, k]  # Instantaneous displacement
A = amplitude_am[i, j, k]        # Envelope (max|ψ|)
dir = wave_direction[i, j, k]
```

### Derived Properties

**Computed from Field**:

- **Frequency f**: PRIMARY property, measured from temporal oscillations
  - Measured directly: f = 1/dt (where dt is time between peaks)
  - If stored: propagates with wave
  - Aligns with Planck E = hf (energy proportional to frequency)

- **Wavelength λ**: DERIVED from frequency
  - Computed from frequency: λ = c/f
  - Can also be measured spatially: `λ = distance(amplitude_max[n], amplitude_max[n+1])`
  - Not typically stored (derived when needed)

- **Energy**: Integral of energy density (frequency-based)
  - Energy density: u = ρ(fA)² (EWT, frequency-centric, no ½ factor)
  - Total energy: `E_total = Σ u[i,j,k] * dx³ = Σ ρ(fA)²[i,j,k] * dx³`

**Measurement Algorithms**:

```python
@ti.kernel
def measure_wavelength() -> ti.f32:
    """Measure wavelength from spatial pattern."""
    # Find two successive amplitude maxima
    max_positions = find_amplitude_maxima()
    wavelength = distance(max_positions[0], max_positions[1])
    return wavelength
```

## Notation Clarification: ψ vs A

**Two Distinct Physical Quantities - Both Needed!**

The Physics Distinction. In wave physics, these symbols typically mean:

- ψ (psi): The wave field itself (displacement from equilibrium, instantaneous displacement at position x)
- A: Amplitude envelope (maximum of |ψ|)

For a sinusoidal wave: ψ(x,t) = A sin(kx - ωt)

- ψ varies between -A and +A
- A is the constant amplitude (maximum displacement)

| Computation            | Uses                          |
|------------------------|-------------------------------|
| Wave propagation (PDE) | ψ (psiL_am)           |
| Energy density         | A² (amplitude_am²)            |
| Force calculation, MAP | ∇A (gradient of amplitude_am) |
| Wave mode (long/trans) | ∇ψ (displacement direction)   |
| Phase                  | From ψ field                  |

- granule displacement (from rest, sine wave, localized)
- granule amplitude = granule max displacement (constant, localized)
- universe peak amplitude = max from all granule amplitude (constant)
- universe avg amplitude = avg from all granule amplitude (constant)

### 1. ψ (psi): Instantaneous Displacement

- **What it is**: The actual wave displacement at each instant in time
  - Oscillates rapidly at wave frequency (~10²⁵ Hz for energy waves)
  - Can be positive or negative
  - Varies: ψ(x,y,z,t)
  - **Propagates via wave equation**: ∂²ψ/∂t² = c²∇²ψ

- **In code**: `self.psiL_am[i,j,k]`
- **Used for**:
  - Wave propagation (PDEs, Laplacian)
  - Wave mode analysis (longitudinal vs transverse)
  - Phase calculations
  - Instantaneous field values

### 2. A: Amplitude Envelope

- **What it is**: The **maximum displacement** at each location (envelope)
  - For sinusoidal wave: ψ(x,t) = A(x) sin(kx - ωt)
  - A is the peak: |ψ|max = A
  - Always positive: A ≥ 0
  - Slowly varying (envelope of high-frequency oscillation)
  - **Tracked as running maximum** of |ψ| over time

- **In code**: `self.amplitude_am[i,j,k]`
- **Used for**:
  - **Energy density**: u = ρ(fA)² (EWT, no ½ factor, frequency-centric)
  - **Force calculation**: F = -2ρVfA×[f∇A + A∇f] or F = -2ρVf²×A∇A (MAP: Minimum **Amplitude** Principle)
  - Energy gradients
  - Pressure-like field that drives particle motion

### Why Two Fields Are Needed

**The High-Frequency Problem**:

- Energy waves oscillate at ~10²⁵ Hz (from EWT)
- Particles have mass/inertia - cannot respond to every oscillation
- Particles respond to **time-averaged** force = force from **envelope** (A)

**Analogy** (Speaker Diaphragm):

- **ψ**: Diaphragm position oscillating at audio frequency
- **A**: "Volume" setting - controls maximum displacement
- You feel air pressure from **A** (volume), not individual oscillations (ψ)

### Implementation Strategy

**Wave Equation** propagates ψ (displacement):

```python
# High-frequency oscillation (updated every timestep)
∂²ψ/∂t² = c²∇²ψ
self.psiL_am[i,j,k]  # Stores current ψ
```

**Amplitude Tracking** extracts envelope A from ψ:

```python
# Track maximum |ψ| over time (envelope extraction)
@ti.kernel
def track_amplitude_envelope(self):
    for i, j, k in self.psiL_am:
        disp_mag = ti.abs(self.psiL_am[i,j,k])
        ti.atomic_max(self.amplitude_am[i,j,k], disp_mag)
```

**Force Calculation** uses A (not ψ):

```python
# Particles respond to amplitude gradient (envelope)
F = -∇A  # Not -∇ψ !
F = -(∂A/∂x, ∂A/∂y, ∂A/∂z)
```

### Summary Table

| Property | ψ (Displacement) | A (Amplitude) |
|----------|------------------|---------------|
| **Field name** | `psiL_am[i,j,k]` | `amplitude_am[i,j,k]` |
| **Physics** | Instantaneous oscillation | Envelope (max \|ψ\|) |
| **Frequency** | High (~10²⁵ Hz) | Slowly varying |
| **Sign** | ± (positive/negative) | + (always positive) |
| **Propagation** | Wave equation (PDE) | Tracked from ψ |
| **Used for** | Wave dynamics, phase, mode | Forces, energy, MAP |
| **Formula** | ∂²ψ/∂t² = c²∇²ψ | A = max(\|ψ\|) over time |

**Critical Point**: Forces use **amplitude gradient** (∇A), not displacement gradient (∇ψ)! This is because MAP = "Minimum **Amplitude** Principle" - particles move toward regions of lower amplitude envelope, not lower instantaneous displacement.

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

## Wave Terminology

This section documents the standardized terminology used throughout OpenWave for describing wave properties, with particular attention to how OpenWave extends standard physics terminology.

### Standard Physics Terminology

#### 1. Wave Mode (Polarization)

**Definition**: Relationship between displacement direction and wave propagation direction.

**Standard Physics Usage**:

- **Longitudinal wave**: Displacement parallel to propagation (compression/rarefaction)
  - Examples: Sound waves, seismic P-waves
  - Pattern: Compression → rarefaction → compression
  - Medium particles oscillate along direction of wave travel

- **Transverse wave**: Displacement perpendicular to propagation
  - Examples: Light, EM waves, water surface waves, guitar strings
  - Pattern: Peaks and troughs perpendicular to motion
  - Medium particles oscillate perpendicular to direction of wave travel

**Physics Literature**: "Wave mode" primarily refers to polarization (longitudinal/transverse), though it can also refer to standing wave patterns (fundamental mode, 2nd mode, etc.). In OpenWave, "wave mode" always means polarization.

#### 2. Wave Type (Energy Transport)

**Definition**: Whether wave energy moves through space.

**Standard Physics Usage**:

- **Traveling wave**: Pattern moves through space, transports energy
  - Mathematical form: `ψ(x,t) = A sin(kx - ωt)` (pattern moves at velocity c)
  - Net energy flux present: `<S> ≠ 0`
  - Nodes and antinodes move with wave

- **Standing wave**: Pattern stationary in space, no net energy transport
  - Mathematical form: `ψ(x,t) = A sin(kx) cos(ωt)` (spatial pattern fixed)
  - No net energy flux on average: `<S> = 0`
  - Nodes (zero displacement) and antinodes (max displacement) remain fixed
  - Formed by interference of two identical waves traveling in opposite directions

**Physics Literature**: Also called "stationary wave" for standing waves.

#### 3. Other Standard Wave Classifications

**Wave Classification by Medium**:

- **Mechanical waves**: Require medium (sound, water waves, seismic)
- **Electromagnetic waves**: No medium required (light, radio, X-rays)
- **Matter waves**: Quantum mechanical (de Broglie waves)

**Wave Form/Shape** (Signal Processing):

- Describes temporal/spatial profile of the wave
- Examples: sine, square, triangle, sawtooth
- Used primarily in electronics, acoustics, signal processing
- Not a primary classification in wave physics

**Wave Pattern**:

- General term for spatial distribution
- Examples: "interference pattern," "standing wave pattern," "diffraction pattern"
- Too vague for precise technical use - prefer specific terms

**Wave State**:

- Not standard physics terminology
- Could mean: instantaneous configuration, superposition state, quantum state
- Avoid unless clearly defined in context

### OpenWave-Specific Terminology

OpenWave extends standard physics terminology with continuous measurements and specific physical interpretations.

#### 1. Wave Mode (OpenWave Extension)

**OpenWave Implementation**: Continuous scalar field `wave_mode[i,j,k]` ranging `[0, 1]`

```python
wave_mode[i,j,k] = continuous value [0, 1]

# Values:
1.0  # Pure longitudinal (displacement ∥ propagation)
0.8  # Mostly longitudinal (80% longitudinal, 20% transverse)
0.5  # Mixed (equal longitudinal and transverse components)
0.2  # Mostly transverse (20% longitudinal, 80% transverse)
0.0  # Pure transverse (displacement ⊥ propagation)
```

**Measurement Method**:

```python
# Compute wave_mode from field data
wave_mode = |k̂ · û|

where:
k̂ = wave propagation direction (from energy flux S)
û = displacement direction (from gradient ∇ψ)
· = dot product
```

**Physical Interpretation**:

- Measures how aligned displacement is with propagation
- Handles mixed/intermediate polarization states
- Reflects realistic wave superposition (not just binary states)

**Standard vs OpenWave**:

- **Standard Physics**: Binary choice (longitudinal OR transverse)
- **OpenWave**: Continuous spectrum allowing mixed states

#### 2. Wave Type (OpenWave Extension)

**OpenWave Implementation**: Continuous scalar field `wave_type[i,j,k]` ranging `[0, 1]`

```python
wave_type[i,j,k] = continuous value [0, 1]

# Values:
0.0  # Pure standing (nodes completely fixed)
0.3  # Mostly standing (weak energy flux)
0.5  # Quasi-standing (intermediate)
0.7  # Mostly traveling (strong energy flux)
1.0  # Pure traveling (free wave propagation)
```

**Measurement Method**:

```python
# Energy-based detection
# Standing waves: E_kinetic and E_potential oscillate 90° out of phase
# Traveling waves: E_kinetic = E_potential at all times

wave_type = f(E_kinetic / E_potential ratio, energy_flux)
```

**Physical Interpretation**:

- Measures degree of energy transport vs oscillation
- Accounts for partial standing wave behavior
- Near particle boundaries: gradual transition from traveling to standing

**Standard vs OpenWave**:

- **Standard Physics**: Binary distinction (standing OR traveling)
- **OpenWave**: Continuous spectrum for transitional states

#### 3. Wave Classification (4-Category System)

**Unique Feature**: Combined mode + type creates 4 fundamental physics categories

| Class | Mode | Type | wave_mode | wave_type | Physical Meaning |
|-------|------|------|-----------|-----------|------------------|
| **1** | Longitudinal | Traveling | > 0.7 | > 0.7 | **Gravitational radiation** (expanding gravity waves) |
| **2** | Longitudinal | Standing | > 0.7 | < 0.3 | **Particle mass** (trapped energy in standing pattern) |
| **3** | Transverse | Traveling | < 0.3 | > 0.7 | **EM radiation** (light, photons from accelerating charges) |
| **4** | Transverse | Standing | < 0.3 | < 0.3 | **Electron orbitals** (orbital structure, hypothesized) |
| **0** | Mixed | Mixed | intermediate | intermediate | **Transitional/mixed** (boundary regions) |

**Implementation Pattern**:

```python
# Classify wave based on mode and type
mode = wave_mode[i,j,k]   # 0=transverse, 1=longitudinal
wtype = wave_type[i,j,k]  # 0=standing, 1=traveling

if mode > 0.7 and wtype > 0.7:
    wave_class = 1  # Longitudinal traveling (gravity waves)
elif mode > 0.7 and wtype < 0.3:
    wave_class = 2  # Longitudinal standing (particle mass)
elif mode < 0.3 and wtype > 0.7:
    wave_class = 3  # Transverse traveling (EM radiation)
elif mode < 0.3 and wtype < 0.3:
    wave_class = 4  # Transverse standing (electron orbitals)
else:
    wave_class = 0  # Mixed/transitional
```

**Physical Context**:

This classification system reflects fundamental EWT physics:

1. **Gravitational radiation**: Longitudinal traveling waves expanding from particle motion
2. **Particle mass**: Longitudinal standing waves trapped around wave centers (reflective voxels)
3. **EM radiation**: Transverse traveling waves from accelerating charges
4. **Electron orbitals**: Transverse standing waves in orbital configurations

**Standard vs OpenWave**:

- **Standard Physics**: No equivalent combined classification system
- **OpenWave**: Unique physics model linking wave character to physical phenomena

#### 4. Wave Component Decomposition (Mixed States)

**OpenWave Feature**: A single voxel can carry BOTH longitudinal and transverse components simultaneously.

**Component Fields**:

```python
# Longitudinal component (parallel to propagation)
u_longitudinal = (u · k̂) k̂          # Projection onto propagation direction
longitudinal_amplitude[i,j,k]        # Magnitude
longitudinal_fraction[i,j,k]         # Energy fraction [0,1]

# Transverse component (perpendicular to propagation)
u_transverse = u - u_longitudinal    # Perpendicular projection
transverse_amplitude[i,j,k]          # Magnitude
transverse_fraction[i,j,k]           # Energy fraction [0,1]
```

**When This Occurs**:

- Wave interference (multiple sources with different propagation directions)
- Near particle boundaries (incident + reflected wave mixing)
- Spherical waves (radial propagation with tangential oscillations)
- EM wave generation from electrons (energy transformation)

**Physical Interpretation**:

- Total displacement = longitudinal + transverse components
- Energy distribution: `E_long + E_trans = E_total`
- Allows realistic modeling of complex wave superposition

#### 5.Wave Character

**Term Definition**: "Wave character" describes the fundamental nature of a wave through its mode (polarization) and type (energy transport).

**Usage in OpenWave**:

```text
### Wave Character

- WAVE-MODE: longitudinal, transverse (polarization)
- WAVE-TYPE: standing, traveling (energy transport)
```

**Why "Character"**:

- ✓ Concise (shorter than "characteristics")
- ✓ Descriptive (captures "essential nature/quality")
- ✓ Physics-appropriate (common in wave physics literature)
- ✓ Singular form (groups mode+type as aspects of overall wave character)
- ✓ Distinct from "Classification" (avoids confusion with 4-category system)

**Example Usage**: "What's the character of this wave?" → "It's longitudinal and traveling (gravitational radiation)"

**Alternative Terms Considered**:

- **Wave Classes**: Could confuse with 4-category classification system
- **Wave Characteristics**: Too verbose
- **Wave Profile**: Conflicts with standard usage (spatial/temporal shape)
- **Wave Categories**: Implies discrete bins (OpenWave uses continuous [0,1])
- **Wave Properties**: Too broad (already used for top-level section)

### Wave Profiling (Diagnostic Tool)

**Definition**: Comprehensive diagnostic function that measures all wave properties at a point or region.

**Purpose**: Analysis and measurement tool (NOT a property category)

**Conceptual Function**:

```python
def wave_profiling(field, position):
    """
    Comprehensive wave diagnostics tool.

    Measures all wave properties at specified location:

    Wave Character:
    - mode: longitudinal/transverse fraction [0,1]
    - type: standing/traveling fraction [0,1]
    - classification: category (1-4 or 0 for mixed)

    Wave Rhythm:
    - frequency (f): measured in Hz (PRIMARY property)
    - period (T): 1/f in seconds

    Wave Size:
    - amplitude (A): envelope magnitude in meters
    - wavelength (λ): c/f in meters (DERIVED from frequency)

    Wave Energy:
    - energy_density (u): ρ(fA)² in J/m³
    - energy_flux (S): Poynting-like vector in W/m²

    Wave Direction:
    - propagation_direction (k̂): unit vector
    - displacement_direction (û): unit vector

    Wave Phase:
    - phase (φ): position in wave cycle (radians)

    Returns:
        WaveProfile: dataclass containing all measured properties
    """
    pass
```

**Why "Wave Profiling"**:

- ✓ "Profiling" implies measurement/analysis (performance profiling analogy)
- ✓ Diagnostic connotation (gathering detailed information)
- ✓ Returns comprehensive "profile" of wave properties
- ✓ Distinct from "wave character" (which is just mode+type)

**Not Confused With**:

- **Wave profile** (standard physics): Spatial/temporal shape (Gaussian profile, exponential profile)
- **Wave character**: Categorical description (mode+type only)

**Usage Pattern**:

```python
# Diagnostic analysis at detector position
profile = wave_profiling(field, detector_position)

print(f"Wave Character: {profile.mode_name} + {profile.type_name}")
print(f"Classification: Class {profile.classification} ({profile.physics_meaning})")
print(f"Frequency: {profile.frequency:.3e} Hz")
print(f"Wavelength: {profile.wavelength:.3e} m")
print(f"Amplitude: {profile.amplitude:.3e} m")
print(f"Energy Density: {profile.energy_density:.3e} J/m³")
print(f"Propagation: {profile.propagation_direction}")
```

### Terminology Best Practices

**Use "Wave Character" when**:

- Describing fundamental wave nature (mode + type)
- Categorizing wave properties
- Organizing documentation sections

**Use "Wave Classification" when**:

- Referring to 4-category physics system
- Discussing physical phenomena (gravity, mass, EM, orbitals)
- Computing wave_class from mode+type

**Use "Wave Profiling" when**:

- Implementing diagnostic/measurement functions
- Comprehensive wave analysis at a point
- Returning multiple wave properties simultaneously

**Avoid**:

- "Wave profile" for character/classification (conflicts with shape/distribution)
- "Wave state" (not standard, ambiguous)
- "Wave pattern" without qualifier (too vague - use "interference pattern," "standing wave pattern," etc.)
- "Wave classes" in contexts where it could confuse with classification system

### Comparison: Standard Physics vs OpenWave

| Term | Standard Physics | OpenWave Extension | Key Difference |
|------|-----------------|---------------|----------------|
| **Wave Mode** | Binary (longitudinal OR transverse) | Continuous [0,1] (allows mixed) | OpenWave handles superposition |
| **Wave Type** | Binary (standing OR traveling) | Continuous [0,1] (allows quasi-standing) | OpenWave handles transitions |
| **Wave Classification** | By medium (mechanical, EM, etc.) | 4 physics categories (mode+type) | OpenWave-specific system |
| **Wave Character** | Not standard term | Mode + Type grouped | OpenWave organizational term |
| **Wave Profiling** | Not specific term | Diagnostic tool | OpenWave measurement function |
| **Component Decomposition** | Sometimes used | Standard feature in OpenWave | Mixed polarization states |

### Summary

**Key Terminology Decisions**:

1. **"Wave Character"** = Wave mode + wave type (fundamental nature)
2. **"Wave Classification"** = 4-category physics system (derived from character)
3. **"Wave Profiling"** = Diagnostic measurement tool (analyzes all properties)

**OpenWave Innovation**:

- Continuous wave_mode and wave_type fields [0,1] instead of binary
- Physically realistic for complex wave superposition
- 4-category classification linking wave character to physical phenomena

**Reserved Terms**:

- **"Wave profile"**: Spatial/temporal shape (Gaussian, exponential, etc.)
- **"Wave pattern"**: General spatial distribution (use with qualifier)
- **"Frequency (f)"**: PRIMARY measured property (not wavelength)
- **"Wavelength (λ)"**: DERIVED from frequency (λ = c/f)
