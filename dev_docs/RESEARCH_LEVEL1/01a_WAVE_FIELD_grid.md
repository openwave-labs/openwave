# WAVE FIELD REPRESENTATION (medium / grid)

## Table of Contents

1. [Summary](#summary)
1. [Grid Architecture: Cell-Centered Design](#grid-architecture-cell-centered-design)
   - [Computational Method](#computational-method)
   - [Index-to-Position Mapping (CRITICAL)](#index-to-position-mapping-critical)
   - [Why Cell-Centered (Not Vertex-Centered)?](#why-cell-centered-not-vertex-centered)
1. [Field Indexing: 1D vs 3D Arrays](#field-indexing-1d-vs-3d-arrays)
   - [LEVEL-0 Uses 1D Arrays (Particle-Based)](#level-0-uses-1d-arrays-particle-based)
   - [LEVEL-1 Uses 3D Arrays (Grid-Based)](#level-1-uses-3d-arrays-grid-based)
   - [Why 3D Arrays for LEVEL-1?](#why-3d-arrays-for-level-1)
   - [Performance Considerations](#performance-considerations)
1. [Scaled SI Units Strategy for LEVEL-1](#scaled-si-units-strategy-for-level-1)
   - [LEVEL-0 vs LEVEL-1 Comparison](#level-0-vs-level-1-comparison)
   - [Why LEVEL-1 Still Needs Scaled Units](#why-level-1-still-needs-scaled-units)
1. [Data Containers: Taichi Fields](#data-containers-taichi-fields)
   - [Field Categories](#field-categories)
   - [Example Field Declaration](#example-field-declaration)
1. [Resolution & Sampling](#resolution--sampling)
   - [Wave Sampling Requirements](#wave-sampling-requirements)
   - [Voxel Size Calculation](#voxel-size-calculation)
   - [Numerical Precision](#numerical-precision)
1. [Terminology & Best Practices](#terminology--best-practices)
   - [Standard Terms](#standard-terms)
   - [Naming Conventions](#naming-conventions)
1. [Lattice Type: Cubic vs Orthorhombic](#lattice-type-cubic-vs-orthorhombic)
   - [Cubic Lattice (Recommended)](#cubic-lattice-recommended-for-initial-implementation)
   - [Orthorhombic Lattice (Future)](#orthorhombic-lattice-future-extension)
1. [Voxel Neighbor Connectivity (STENCIL)](#voxel-neighbor-connectivity-stencil)
   - [3D Neighbor Classification](#3d-neighbor-classification)
   - [Configurable Connectivity Parameter](#configurable-connectivity-parameter)
   - [Distance-Based Weighting](#distance-based-weighting)
1. [Implementation Example](#implementation-example)
1. [Asymmetric Universe Support](#asymmetric-universe-support)
1. [Key Design Decisions](#key-design-decisions)

## SUMMARY

LEVEL-1 uses a **cell-centered grid** where field indices `[i,j,k]` represent the centers of cubic voxels in 3D space. This approach follows industry standards from Lattice QCD and Computational Fluid Dynamics, providing optimal memory efficiency and numerical accuracy. Unlike LEVEL-0's particle-based system which stores positions explicitly, LEVEL-1 computes positions from indices. However, **attometer scaling is still required** for field values (amplitude, wavelength, voxel size) to maintain numerical precision in f32 calculations, following the same precision principles as LEVEL-0.

## Grid Architecture: Cell-Centered Design

### Computational Method

- **Medium** = WAVE FIELD (the grid - not physical substrate, just information carrier)
- **Unit** = VOXEL (unit of volume, cell-centered grid point, industry standard)
- **Data structure** = 3D scalar/vector Taichi fields

### Index-to-Position Mapping (CRITICAL)

**Key Principle**: Grid indices `[i,j,k]` represent **voxel centers**, not corners.

```python
# Grid index [i,j,k] maps to physical position:
pos_x = (i + 0.5) * dx_am  # Position in attometers
pos_y = (j + 0.5) * dx_am
pos_z = (k + 0.5) * dx_am

# Where:
# - dx_am = voxel edge length in attometers (for f32 precision)
# - (i,j,k) = integer grid indices (0, 1, 2, ...)
# - 0.5 offset centers the point in the voxel
# - To convert to meters: pos_m = pos_am * ATTOMETER
```

**Voxel Geometry**:

- Voxel `[i,j,k]` occupies physical space from (in attometers):
  - `x: i*dx_am to (i+1)*dx_am`
  - `y: j*dx_am to (j+1)*dx_am`
  - `z: k*dx_am to (k+1)*dx_am`
- Voxel center at `((i+0.5)*dx_am, (j+0.5)*dx_am, (k+0.5)*dx_am)`
- Voxel boundaries at integer multiples of `dx_am`

### Why Cell-Centered (Not Vertex-Centered)?

**Industry Standard Practice**:

1. **Lattice QCD**: Fields defined at lattice sites (cell centers)
2. **Finite Volume Methods (FVM)**: Standard in CFD and physics sims
3. **Physical interpretation**: Values represent averages over volumes
4. **Symmetric neighbors**: Cleaner stencil calculations
5. **Taichi optimization**: Direct index mapping, better cache locality

## Field Indexing: 1D vs 3D Arrays

### LEVEL-0 Uses 1D Arrays (Particle-Based)

**Medium Object**: Granule (moves freely in space)

```python
# LEVEL-0: 1D array of granules with 3D position vectors
self.position_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
self.velocity_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
self.displacement_am = ti.field(dtype=ti.f32, shape=self.total_granules)

# Access granule 12345
pos = self.position_am[12345]  # Returns [x, y, z] in attometers
amp = self.displacement_am[12345]  # Scalar amplitude
```

**Why 1D is optimal for LEVEL-0:**

- ✅ **Granules move**: No fixed spatial relationship between index and position
- ✅ **Particle dynamics**: Iterate over all granules to update state
- ✅ **Cache-friendly**: Sequential access pattern
- ✅ **Dynamic topology**: Granules can rearrange, no grid constraints
- ✅ **Industry standard**: All particle-based engines use 1D arrays

**Spatial lookup challenge:**

```python
# Question: "What's at position (x, y, z)?"
# Answer: Must search ALL granules - O(N) complexity!

@ti.kernel
def find_granule_at_position(target_pos: ti.math.vec3) -> ti.i32:
    closest_idx = -1
    min_dist = 1e10

    for i in range(self.total_granules):  # O(N) search
        dist = (self.position_am[i] - target_pos).norm()
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    return closest_idx
```

This is acceptable in LEVEL-0 because **spatial queries are rare** - you mainly iterate over all granules.

### LEVEL-1 Uses 3D Arrays (Grid-Based)

**Medium Object**: Voxel (fixed position in grid)

```python
# LEVEL-1: 3D grid of field values
self.displacement_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
self.velocity_am = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))
self.phase = ti.field(dtype=ti.f32, shape=(nx, ny, nz))

# Access voxel [100, 200, 300]
amp = self.displacement_am[100, 200, 300]  # Direct O(1) access!
vel = self.velocity_am[100, 200, 300]   # 3D vector at this voxel
```

**Why 3D is optimal for LEVEL-1:**

- ✅ **Voxels don't move**: Fixed spatial relationship (index = position)
- ✅ **Spatial queries**: Constantly need "what's at (x,y,z)?" - O(1) lookup
- ✅ **Neighbor access**: Natural 6/18/26-connectivity via index offsets
- ✅ **Grid algorithms**: Laplacian, gradients, stencils are trivial
- ✅ **Readable code**: `displacement_am[i, j, k]` has clear spatial meaning

**Spatial lookup solution:**

```python
# Question: "What's the amplitude at position (x, y, z)?"
# Answer: O(1) direct index calculation!

@ti.func
def get_amplitude_at_position(pos_am: ti.math.vec3) -> ti.f32:
    # Convert position to voxel index (instant calculation)
    i = ti.i32((pos_am[0] / self.dx_am) - 0.5)
    j = ti.i32((pos_am[1] / self.dx_am) - 0.5)
    k = ti.i32((pos_am[2] / self.dx_am) - 0.5)

    # Direct O(1) lookup - no searching!
    return self.displacement_am[i, j, k]
```

### Why 3D Arrays for LEVEL-1?

**Gradient calculation example** (demonstrates the power of 3D indexing):

```python
# Computing amplitude gradient for forces: ∇A = [∂A/∂x, ∂A/∂y, ∂A/∂z]

# With 3D arrays (clean and efficient):
@ti.kernel
def compute_gradient_3D():
    for i, j, k in self.amplitude_am:  # Taichi auto-generates 3D loop
        if 0 < i < nx-1 and 0 < j < ny-1 and 0 < k < nz-1:
            # Neighbor access is trivial and readable
            grad_x = (amplitude_am[i+1, j, k] - amplitude_am[i-1, j, k]) / (2.0 * dx_am)
            grad_y = (amplitude_am[i, j+1, k] - amplitude_am[i, j-1, k]) / (2.0 * dx_am)
            grad_z = (amplitude_am[i, j, k+1] - amplitude_am[i, j, k-1]) / (2.0 * dx_am)

            force[i, j, k] = -ti.Vector([grad_x, grad_y, grad_z])

# With 1D arrays (complex and error-prone):
@ti.kernel
def compute_gradient_1D():
    for idx in range(total_voxels):
        # Convert 1D index to 3D coordinates (expensive)
        i = idx // (ny * nz)
        j = (idx % (ny * nz)) // nz
        k = idx % nz

        if 0 < i < nx-1 and 0 < j < ny-1 and 0 < k < nz-1:
            # Compute neighbor indices (6 complex calculations!)
            idx_xp = (i+1) * (ny * nz) + j * nz + k  # i+1
            idx_xm = (i-1) * (ny * nz) + j * nz + k  # i-1
            idx_yp = i * (ny * nz) + (j+1) * nz + k  # j+1
            idx_ym = i * (ny * nz) + (j-1) * nz + k  # j-1
            idx_zp = i * (ny * nz) + j * nz + (k+1)  # k+1
            idx_zm = i * (ny * nz) + j * nz + (k-1)  # k-1

            grad_x = (amplitude_am[idx_xp] - amplitude_am[idx_xm]) / (2.0 * dx_am)
            grad_y = (amplitude_am[idx_yp] - amplitude_am[idx_ym]) / (2.0 * dx_am)
            grad_z = (amplitude_am[idx_zp] - amplitude_am[idx_zm]) / (2.0 * dx_am)

            force[idx] = ti.Vector([grad_x, grad_y, grad_z])
```

The 3D version is **cleaner, more readable, and less error-prone**.

### Performance Considerations

**Memory Layout**: Taichi stores 3D arrays as 1D in memory (row-major order)

- `displacement_am[i, j, k]` is compiled to efficient 1D index calculation
- **You get clean syntax + compiler optimization**
- No performance penalty vs explicit 1D indexing

**Cache Locality**: 3D loops maintain good cache performance

```python
for i, j, k in displacement_am:
    # Taichi ensures k varies fastest (innermost loop)
    # = Sequential memory access pattern
    value = displacement_am[i, j, k]
```

**Summary Table**:

| Aspect | LEVEL-0 (1D Arrays) | LEVEL-1 (3D Arrays) |
|--------|---------------------|---------------------|
| **Medium object** | Granule (moves) | Voxel (fixed) |
| **Position** | Stored explicitly | Computed from index |
| **Access pattern** | Iterate all particles | Spatial queries by position |
| **Spatial lookup** | O(N) search required | O(1) index calculation |
| **Neighbor access** | Complex (requires search) | Trivial (`[i±1, j±1, k±1]`) |
| **Gradient/Laplacian** | N/A (particle-based) | Essential (field operators) |
| **Code readability** | `position_am[i]` | `displacement_am[i, j, k]` |
| **Best practice** | 1D ✓ (particle engines) | 3D ✓ (field solvers) |
| **Use case** | Moving particles, N-body | Fixed grid, PDEs, wave equations |

## Scaled SI Units Strategy for LEVEL-1

LEVEL-1 uses **scaled SI units** for both spatial and temporal values to maintain numerical precision with f32:

- **Spatial**: ATTOMETER = 10⁻¹⁸ m (variable/field suffix: `_am`)
- **Temporal**: RONTOSECOND = 10⁻²⁷ s (variable/field suffix: `_rs`)

### LEVEL-0 vs LEVEL-1 Comparison

| Aspect | LEVEL-0 (Granule-Based) | LEVEL-1 (Wave-Field) |
|--------|------------------------|----------------------|
| **Position Storage** | Stored in vector fields `pos_am[i] = [x, y, z]` | Computed from indices: `(i+0.5)*dx_am` |
| **Memory Usage** | 3 floats per granule × millions | 1 scalar `dx_am` (shared) |
| **Field Values** | displacement_am, velocity_am per granule | displacement_am per voxel |
| **Attometer Benefit** | Prevents catastrophic cancellation in distance calculations | Prevents precision loss in displacement gradients and wave calculations |
| **Code Complexity** | Conversion for positions and field values | Conversion for field values only (positions computed) |

### Why LEVEL-1 Still Needs Scaled Units

**Position Calculation** (computed, not stored):

```python
# LEVEL-1: Position computed from arithmetic using attometer-scaled dx
dx_am = 1.25  # attometers (f32 handles well: 1-100 range)
i, j, k = 100, 200, 300  # integers (exact)
pos_x_am = (i + 0.5) * dx_am  # Computed on-the-fly in attometers

# NO storage of individual coordinates (memory efficient ✓)
# Still benefits from attometer scaling for precision in calculations
```

**Field Values** (stored, need precision):

```python
# Critical: Amplitude, wavelength, and voxel size need attometer scaling
wavelength_am = 28.54096501  # attometers (vs 2.854e-17 m)
displacement_am[i,j,k] = 0.9215  # attometers (vs 9.215e-19 m)

# F32 precision preserved:
# - Amplitude gradients: ∇A computed with 6-7 significant digits
# - Wave number: k = 2π/λ_am with better precision
# - Force calculations: F = -2ρVfA×[f∇A + A∇f] maintains accuracy (frequency-based)
```

**Inverse Mapping** (Position → Index):

When implementing the wave engine with particles, you can efficiently find the voxel index from a particle's position using the inverse formula:

```python
# Given particle position (pos_x_am, pos_y_am, pos_z_am) in attometers
# Find containing voxel index [i, j, k]
i = int((pos_x_am / dx_am) - 0.5)
j = int((pos_y_am / dx_am) - 0.5)
k = int((pos_z_am / dx_am) - 0.5)

# Example:
pos_x_am = 125.0  # particle position in attometers (1.25e-16 m)
dx_am = 1.25      # voxel size in attometers (1.25e-18 m)
i = int((125.0 / 1.25) - 0.5) = int(100 - 0.5) = 99

# This is fast: O(1) lookup, no searching required!
```

**Use case**: When particles interact with the field (wave reflection, force calculation), you need to know which voxel contains the particle. This inverse mapping is instant and precise.

**Key Benefits of Attometer Scaling**:

- ✅ **Numerical precision**: Field values (amplitude, wavelength) in f32-friendly range (1-100 am)
- ✅ **Memory efficiency**: Positions computed from indices (not stored)
- ✅ **Gradient accuracy**: Amplitude gradients maintain 6-7 significant digits
- ✅ **Catastrophic cancellation prevented**: Same principle as LEVEL-0
- ✅ **Consistent with LEVEL-0**: Both levels use attometer units internally

## Data Containers: Taichi Fields

LEVEL-1 uses Taichi fields to store wave properties at each voxel. Fields are 3D arrays indexed by `[i,j,k]`.

### Field Categories

- **Scalar fields**: Store single values per voxel (amplitude, density, energy, phase, etc.)
- **Vector fields**: Store 3D vectors per voxel (wave direction, velocity, force, etc.)

**For detailed property descriptions**, see [`02_WAVE_PROPERTIES.md`](./02_WAVE_PROPERTIES.md)

### Example Field Declaration

```python
import taichi as ti
from openwave.common import constants

# Scalar fields with attometer scaling
displacement_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Attometers
phase = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Radians (no scaling)

# Scalar fields without attometer scaling
density = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # kg/m³
energy = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Joules

# Vector fields
wave_direction = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # Unit vector
velocity_am = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # am/s
```

## Resolution & Sampling

### Wave Sampling Requirements

- **Minimum**: >10 points/λ (voxels per wavelength)
- **Recommended**: >40 points/λ (for high accuracy)
- **Trade-off**: Resolution vs. computational cost

### Voxel Size Calculation

```python
from openwave.common import constants

# For wavelength λ and sampling rate
wavelength_m = 5.0e-17  # meters (neutrino scale)
wavelength_am = wavelength_m / constants.ATTOMETER  # Convert to attometers: 50.0 am

points_per_wavelength = 40
dx_am = wavelength_am / points_per_wavelength  # voxel edge length in attometers

# Example: λ=50 am, 40 pts/λ → dx_am=1.25 am (1.25e-18 m)
```

### Numerical Precision

- **Float precision**: `f32` (single precision) sufficient when using attometer scaling
  - Amplitude in attometers: 0.1-10 am (good f32 range, 6-7 significant digits)
  - Wavelength in attometers: 10-100 am (good f32 range)
  - Phase in radians: 0-2π (no scaling needed)
- **Index precision**: `i32` integers for grid indices (exact)
- **Physical scale**: Planck scale (10⁻³⁵m) to molecular (10⁻⁹m)
- **Attometer scaling**: Prevents catastrophic cancellation in gradient calculations

## Terminology & Best Practices

### Standard Terms

- **Field**: 3D array storing physical quantities
- **Voxel**: Volume element, cubic region `[i*dx to (i+1)*dx]³`
- **Cell**: Synonym for voxel in FVM context
- **Grid point / Lattice site**: Center of voxel at `[i,j,k]`
- **Stencil**: Pattern of neighbors used in computation

### Usage Example

```python
WaveField (=Grid + Properties, 3D array)
WaveField.universe_size [list]
WaveField.grid_size [list] (= nx, ny, nz) # compute integers

Voxel (=Cell, volume element)
WaveField.voxel_volume
WaveField.voxel_edge (= dx) # use descriptive var names, better for code maintenance

compute & store once:
- actual adjusted universe dimensions & voxel count
- resolutions
- total energy

field arrays (wave properties)
- SCALAR MEASURED: displacement, amplitude, frequency
- SCALAR COMPUTED: wavelength, period, phase, ...
- VECTOR MEASURED: wave_direction, wave_mode, wave_type

_am = attometer version
_rs = rontosecond version
```

### Naming Conventions

```python
# Recommended variable names
nx, ny, nz              # Grid dimensions (number of voxels)
dx, dy, dz              # Voxel edge lengths (meters, for external reporting)
dx_am, dy_am, dz_am     # Voxel edge lengths (attometers)
i, j, k                 # Grid indices (integers)
pos_am, position_am     # Physical coordinates (attometers)
pos_m, position_m       # Physical coordinates (meters, for external use)
displacement_am[i,j,k]  # Displacement field in attometers
phase[i,j,k]            # Phase field in radians
```

## Lattice Type: Cubic vs Orthorhombic

### Cubic Lattice (Recommended for Initial Implementation)

- **Symmetric**: `dx = dy = dz` (all edges equal)
- **Simpler logic**: Single `dx` parameter
- **Isotropic**: Same resolution in all directions
- **Similar to LEVEL-0**: Maintains consistency

```python
from openwave.common import constants

class CubicFieldMedium:
    def __init__(self, nx, ny, nz, wavelength_m, points_per_wavelength=40):
        self.nx, self.ny, self.nz = nx, ny, nz

        # Attometer scaling for precision
        self.wavelength_am = wavelength_m / constants.ATTOMETER
        self.dx_am = self.wavelength_am / points_per_wavelength  # Single edge length (am)

        # Meters for external reporting
        self.dx = self.dx_am * constants.ATTOMETER
```

### Orthorhombic Lattice (Future Extension)

- **Asymmetric**: `dx ≠ dy ≠ dz` (flexible dimensions)
- **More complex**: Need separate edge lengths
- **Anisotropic**: Different resolution per axis
- **Use case**: Non-uniform domains (e.g., thin film simulations)

```python
from openwave.common import constants

class OrthorhombicFieldMedium:
    def __init__(self, nx, ny, nz, dx, dy, dz):
        self.nx, self.ny, self.nz = nx, ny, nz

        # Attometer scaling for precision
        self.dx_am = dx / constants.ATTOMETER
        self.dy_am = dy / constants.ATTOMETER
        self.dz_am = dz / constants.ATTOMETER

        # Meters for external reporting
        self.dx, self.dy, self.dz = dx, dy, dz
```

**Recommendation**: Start with cubic for simplicity, extend to orthorhombic if needed.

## Voxel Neighbor Connectivity (STENCIL)

### 3D Neighbor Classification

From voxel `[i,j,k]`, neighbors classified by distance:

1. **Face neighbors (6)**: Distance = `dx`, strongest coupling
   - Offsets: `(±1,0,0)`, `(0,±1,0)`, `(0,0,±1)`

2. **Edge neighbors (12)**: Distance = `√2*dx`, medium coupling
   - Offsets: `(±1,±1,0)`, `(±1,0,±1)`, `(0,±1,±1)`

3. **Corner neighbors (8)**: Distance = `√3*dx`, weakest coupling
   - Offsets: `(±1,±1,±1)`

**Total**: 26 neighbors in full 3D connectivity

### Configurable Connectivity Parameter

Allow selection of neighbor set for performance tuning:

- **6-connectivity**: Face neighbors only (computationally efficient)
- **18-connectivity**: Face + edge neighbors (balanced)
- **26-connectivity**: All neighbors (maximum physical accuracy)

### Distance-Based Weighting

Coupling strength inversely proportional to distance:

- Face weight = `1.0` (distance = `dx`)
- Edge weight ≈ `0.707` (distance = `√2*dx`, weight = `1/√2`)
- Corner weight ≈ `0.577` (distance = `√3*dx`, weight = `1/√3`)
- Alternative: Inverse square distance for force fields

**Reference**: See [`03_WAVE_ENGINE.md`](./03_WAVE_ENGINE.md) for detailed propagation mechanics.

## Implementation Example

See [`02_WAVE_PROPERTIES.md`](./02_WAVE_PROPERTIES.md) for the complete `WaveField` class implementation with attometer scaling.

## Asymmetric Universe Support

**Design Principle** (following LEVEL-0's `BCCLattice` / `SCLattice`):

```python
# CRITICAL: Voxel size (dx) MUST remain cubic (same edge length on all axes)
# This preserves wave physics and numerical stability.
# Only the NUMBER of voxels varies per axis.

# Initialization with asymmetric universe size
init_universe_size = [x_size, y_size, z_size]  # meters, can be asymmetric

# Grid dimensions - different counts per axis
nx = int(init_universe_size[0] / dx)  # Number of voxels in x
ny = int(init_universe_size[1] / dx)  # Number of voxels in y
nz = int(init_universe_size[2] / dx)  # Number of voxels in z

# Actual universe dimensions (rounded to fit integer voxel counts)
universe_size = [nx * dx, ny * dx, nz * dx]  # meters

# Taichi field declaration (asymmetric shape)
self.displacement_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
self.amplitude_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
self.force = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))
```

### Example: Flat Universe (Different z)

```python
# User request: 250×250×125 attometer universe
init_universe_size = [250e-18, 250e-18, 125e-18]  # meters

# With points_per_wavelength = 40
dx = wavelength / 40  # Cubic voxel size (same for all axes)

# Grid dimensions
nx = int(250e-18 / dx) = 350  # voxels in x
ny = int(250e-18 / dx) = 350  # voxels in y
nz = int(125e-18 / dx) = 175  # voxels in z (half of x/y)

# Result: 350×350×175 grid with cubic voxels
# Total voxels: 21,437,500 (vs 42,875,000 for symmetric 350³)
# Memory saved: 50% for flat universe!
```

**Key Benefits:**

1. ✓ **Memory efficiency**: Match simulation domain to physical requirements
2. ✓ **Preserves wave physics**: Cubic voxels maintain isotropic wave propagation
3. ✓ **Consistent with LEVEL-0**: Same design philosophy as `BCCLattice`
4. ✓ **Boundary conditions**: Easy to implement reflecting/periodic boundaries per axis
5. ✓ **Visualization**: Natural for flat or elongated domains

**Implementation Notes:**

- All Laplacian stencils use same `dx` for all axes (isotropic)
- CFL condition computed using cubic voxel: `dt ≤ dx / (c√3)`
- Index ranges: `0 ≤ i < nx`, `0 ≤ j < ny`, `0 ≤ k < nz`
- Position mapping: `pos = [(i+0.5)*dx, (j+0.5)*dx, (k+0.5)*dx]`

## Key Design Decisions

1. ✅ **Cell-centered grid** (industry standard for field solvers)
2. ✅ **3D array indexing** `shape=(nx, ny, nz)` (optimal for grid-based algorithms, unlike LEVEL-0's 1D arrays)
3. ✅ **Asymmetric universe support** (nx ≠ ny ≠ nz allowed, following LEVEL-0 design)
4. ✅ **Cubic voxels required** (dx same for all axes, preserves wave physics)
5. ✅ **Attometer scaling for field values** (numerical precision, same principle as LEVEL-0)
6. ✅ **Computed positions from indices** (memory efficiency, no explicit position storage)
7. ✅ **Configurable connectivity** (6/18/26 neighbors)
8. ✅ **Distance-based weighting** (physical accuracy)
9. ✅ **Attometer units internally, meters for external reporting** (precision + clarity)
