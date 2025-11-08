# WAVE FIELD REPRESENTATION (medium / grid)

## Table of Contents

1. [Summary](#summary)
1. [Grid Architecture: Cell-Centered Design](#grid-architecture-cell-centered-design)
   - [Computational Method](#computational-method)
   - [Index-to-Position Mapping (CRITICAL)](#index-to-position-mapping-critical)
   - [Why Cell-Centered (Not Vertex-Centered)?](#why-cell-centered-not-vertex-centered)
1. [Attometer Scaling Strategy for LEVEL-1](#attometer-scaling-strategy-for-level-1)
   - [LEVEL-0 vs LEVEL-1 Comparison](#level-0-vs-level-1-comparison)
   - [Why LEVEL-1 Still Needs Attometer Scaling](#why-level-1-still-needs-attometer-scaling)
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
1. [Voxel Neighbor Connectivity](#voxel-neighbor-connectivity)
   - [3D Neighbor Classification](#3d-neighbor-classification)
   - [Configurable Connectivity Parameter](#configurable-connectivity-parameter)
   - [Distance-Based Weighting](#distance-based-weighting)
1. [Implementation Example](#implementation-example)
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

## Attometer Scaling Strategy for LEVEL-1

### LEVEL-0 vs LEVEL-1 Comparison

| Aspect | LEVEL-0 (Granule-Based) | LEVEL-1 (Field-Based) |
|--------|------------------------|----------------------|
| **Position Storage** | Stored in vector fields `pos_am[i] = [x, y, z]` | Computed from indices: `(i+0.5)*dx_am` |
| **Memory Usage** | 3 floats per granule × millions | 1 scalar `dx_am` (shared) |
| **Field Values** | amplitude_am, velocity_am per granule | amplitude_am per voxel |
| **Attometer Benefit** | Prevents catastrophic cancellation in distance calculations | Prevents precision loss in amplitude gradients and wave calculations |
| **Code Complexity** | Conversion for positions and field values | Conversion for field values only (positions computed) |

### Why LEVEL-1 Still Needs Attometer Scaling

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
amplitude_am[i,j,k] = 0.9215  # attometers (vs 9.215e-19 m)

# F32 precision preserved:
# - Amplitude gradients: ∇A computed with 6-7 significant digits
# - Wave number: k = 2π/λ_am with better precision
# - Force calculations: F = -∇A maintains accuracy
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
amplitude_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Attometers
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

- **Voxel**: Volume element, cubic region `[i*dx to (i+1)*dx]³`
- **Grid point / Lattice site**: Center of voxel at `[i,j,k]`
- **Cell**: Synonym for voxel in FVM context
- **Field**: 3D array storing physical quantities
- **Stencil**: Pattern of neighbors used in computation

### Naming Conventions

```python
# Recommended variable names
nx, ny, nz              # Grid dimensions (number of voxels)
dx_am, dy_am, dz_am     # Voxel edge lengths (attometers)
dx_m, dy_m, dz_m        # Voxel edge lengths (meters, for external reporting)
i, j, k                 # Grid indices (integers)
pos_am, position_am     # Physical coordinates (attometers)
pos_m, position_m       # Physical coordinates (meters, for external use)
amplitude_am[i,j,k]     # Amplitude field in attometers
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
        self.dx_m = self.dx_am * constants.ATTOMETER
```

### Orthorhombic Lattice (Future Extension)

- **Asymmetric**: `dx ≠ dy ≠ dz` (flexible dimensions)
- **More complex**: Need separate edge lengths
- **Anisotropic**: Different resolution per axis
- **Use case**: Non-uniform domains (e.g., thin film simulations)

```python
from openwave.common import constants

class OrthorhombicFieldMedium:
    def __init__(self, nx, ny, nz, dx_m, dy_m, dz_m):
        self.nx, self.ny, self.nz = nx, ny, nz

        # Attometer scaling for precision
        self.dx_am = dx_m / constants.ATTOMETER
        self.dy_am = dy_m / constants.ATTOMETER
        self.dz_am = dz_m / constants.ATTOMETER

        # Meters for external reporting
        self.dx_m, self.dy_m, self.dz_m = dx_m, dy_m, dz_m
```

**Recommendation**: Start with cubic for simplicity, extend to orthorhombic if needed.

## Voxel Neighbor Connectivity

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

## Key Design Decisions

1. ✅ **Cell-centered grid** (industry standard)
2. ✅ **Attometer scaling for field values** (numerical precision, same as LEVEL-0)
3. ✅ **Computed positions from indices** (memory efficiency)
4. ✅ **Cubic lattice initially** (symmetric, simpler)
5. ✅ **Configurable connectivity** (6/18/26 neighbors)
6. ✅ **Distance-based weighting** (physical accuracy)
7. ✅ **Attometer units internally, meters for external reporting** (precision + clarity)

---

**Status**: Architecture defined, ready for implementation

**Next Steps**: Implement wave propagation engine using this grid structure
