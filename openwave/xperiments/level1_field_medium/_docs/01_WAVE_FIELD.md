# WAVE FIELD REPRESENTATION (medium / grid)

## Table of Contents

1. [Summary](#summary)
1. [Grid Architecture: Cell-Centered Design](#grid-architecture-cell-centered-design)
   - [Computational Method](#computational-method)
   - [Index-to-Position Mapping (CRITICAL)](#index-to-position-mapping-critical)
   - [Why Cell-Centered (Not Vertex-Centered)?](#why-cell-centered-not-vertex-centered)
1. [Attometer Conversion: NOT NEEDED in LEVEL-1](#attometer-conversion-not-needed-in-level-1)
   - [LEVEL-0 vs LEVEL-1 Comparison](#level-0-vs-level-1-comparison)
   - [Why LEVEL-1 Avoids Attometer Conversion](#why-level-1-avoids-attometer-conversion)
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

LEVEL-1 uses a **cell-centered grid** where field indices `[i,j,k]` represent the centers of cubic voxels in 3D space. This approach follows industry standards from Lattice QCD and Computational Fluid Dynamics, providing optimal memory efficiency and numerical accuracy. Unlike LEVEL-0's particle-based system, LEVEL-1 eliminates the need for attometer conversion by computing positions from indices, significantly reducing code complexity while maintaining physical precision.

## Grid Architecture: Cell-Centered Design

### Computational Method

- **Medium** = information grid (not physical substrate)
- **Data structure** = 3D scalar/vector Taichi fields
- **Grid type** = Cell-centered (industry standard)

### Index-to-Position Mapping (CRITICAL)

**Key Principle**: Grid indices `[i,j,k]` represent **voxel centers**, not corners.

```python
# Grid index [i,j,k] maps to physical position:
pos_x = (i + 0.5) * dx
pos_y = (j + 0.5) * dx
pos_z = (k + 0.5) * dx

# Where:
# - dx = voxel edge length (meters, NOT attometers)
# - (i,j,k) = integer grid indices (0, 1, 2, ...)
# - 0.5 offset centers the point in the voxel
```

**Voxel Geometry**:

- Voxel `[i,j,k]` occupies physical space from:
  - `x: i*dx to (i+1)*dx`
  - `y: j*dx to (j+1)*dx`
  - `z: k*dx to (k+1)*dx`
- Voxel center at `((i+0.5)*dx, (j+0.5)*dx, (k+0.5)*dx)`
- Voxel boundaries at integer multiples of `dx`

### Why Cell-Centered (Not Vertex-Centered)?

**Industry Standard Practice**:

1. **Lattice QCD**: Fields defined at lattice sites (cell centers)
2. **Finite Volume Methods (FVM)**: Standard in CFD and physics sims
3. **Physical interpretation**: Values represent averages over volumes
4. **Symmetric neighbors**: Cleaner stencil calculations
5. **Taichi optimization**: Direct index mapping, better cache locality

## Attometer Conversion: NOT NEEDED in LEVEL-1

### LEVEL-0 vs LEVEL-1 Comparison

| Aspect | LEVEL-0 (Granule-Based) | LEVEL-1 (Field-Based) |
|--------|------------------------|----------------------|
| **Position Storage** | Stored in vector fields `pos[i] = [x, y, z]` | Computed from indices: `(i+0.5)*dx` |
| **Memory Usage** | 3 floats per granule × millions | 1 scalar `dx` (shared) |
| **Precision Issue** | Need attometers (10⁻¹⁸m) for each coordinate | `dx` stored in meters (f32 sufficient) |
| **Code Complexity** | Attometer conversion everywhere | No conversion needed |
| **Conversion Function** | `am_to_m()`, `m_to_am()` required | **NOT REQUIRED** |

### Why LEVEL-1 Avoids Attometer Conversion

**Position Calculation**:

```python
# LEVEL-1: Position computed from arithmetic
dx = 1.0e-18  # meters (f32 can handle this scalar)
i, j, k = 100, 200, 300  # integers (exact)
pos_x = (i + 0.5) * dx  # Computed on-the-fly

# NO storage of individual coordinates
# NO precision loss from storing millions of tiny values
# NO need for attometer integer conversion
```

**Inverse Mapping** (Position → Index):

When implementing the wave engine with particles, you can efficiently find the voxel index from a particle's position using the inverse formula:

```python
# Given particle position (pos_x, pos_y, pos_z) in meters
# Find containing voxel index [i, j, k]
i = int((pos_x / dx) - 0.5)  # or int(pos_x / dx - 0.5)
j = int((pos_y / dx) - 0.5)
k = int((pos_z / dx) - 0.5)

# Example:
pos_x = 1.25e-16  # particle position in meters
dx = 1.25e-18     # voxel size
i = int((1.25e-16 / 1.25e-18) - 0.5) = int(100 - 0.5) = 99

# This is fast: O(1) lookup, no searching required!
```

**Use case**: When particles interact with the field (wave reflection, force calculation), you need to know which voxel contains the particle. This inverse mapping is instant.

**Key Insight**: Only **one scalar** (`dx`) needs sub-nanometer precision, not millions of coordinates. A single `f32` value can represent `1e-18` adequately for this purpose.

**Benefits**:

- ✅ Simpler code (no conversion functions)
- ✅ Lower memory usage
- ✅ Easier for new developers
- ✅ Still maintains physical accuracy
- ✅ Direct SI units (meters) throughout

## Data Containers: Taichi Fields

LEVEL-1 uses Taichi fields to store wave properties at each voxel. Fields are 3D arrays indexed by `[i,j,k]`.

### Field Categories

- **Scalar fields**: Store single values per voxel (amplitude, density, energy, phase, etc.)
- **Vector fields**: Store 3D vectors per voxel (wave direction, velocity, force, etc.)

**For detailed property descriptions**, see [`02_WAVE_PROPERTIES.md`](./02_WAVE_PROPERTIES.md)

### Example Field Declaration

```python
import taichi as ti

# Scalar field example
amplitude = ti.field(dtype=ti.f32, shape=(nx, ny, nz))

# Vector field example
wave_direction = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))
```

## Resolution & Sampling

### Wave Sampling Requirements

- **Minimum**: >10 points/λ (voxels per wavelength)
- **Recommended**: >40 points/λ (for high accuracy)
- **Trade-off**: Resolution vs. computational cost

### Voxel Size Calculation

```python
# For wavelength λ and sampling rate
wavelength = 5.0e-17  # meters (neutrino scale)
points_per_wavelength = 40
dx = wavelength / points_per_wavelength  # voxel edge length

# Example: λ=5e-17m, 40 pts/λ → dx=1.25e-18m
```

### Numerical Precision

- **Float precision**: `f32` (single precision) sufficient for field values
- **Index precision**: `i32` integers for grid indices (exact)
- **Physical scale**: Planck scale (10⁻³⁵m) to molecular (10⁻⁹m)

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
nx, ny, nz          # Grid dimensions (number of voxels)
dx, dy, dz          # Voxel edge lengths (meters)
i, j, k             # Grid indices (integers)
pos, position       # Physical coordinates (meters)
field[i,j,k]        # Field value at voxel [i,j,k]
```

## Lattice Type: Cubic vs Orthorhombic

### Cubic Lattice (Recommended for Initial Implementation)

- **Symmetric**: `dx = dy = dz` (all edges equal)
- **Simpler logic**: Single `dx` parameter
- **Isotropic**: Same resolution in all directions
- **Similar to LEVEL-0**: Maintains consistency

```python
class CubicFieldMedium:
    def __init__(self, nx, ny, nz, dx):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = dx  # Single edge length for all axes
```

### Orthorhombic Lattice (Future Extension)

- **Asymmetric**: `dx ≠ dy ≠ dz` (flexible dimensions)
- **More complex**: Need separate edge lengths
- **Anisotropic**: Different resolution per axis
- **Use case**: Non-uniform domains (e.g., thin film simulations)

```python
class OrthorhombicFieldMedium:
    def __init__(self, nx, ny, nz, dx, dy, dz):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx, self.dy, self.dz = dx, dy, dz  # Per-axis edge lengths
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

```python
import taichi as ti

@ti.data_oriented
class CellCenteredFieldMedium:
    """Cell-centered grid for wave field simulation."""

    def __init__(self, nx, ny, nz, dx):
        # Grid parameters
        self.nx, self.ny, self.nz = nx, ny, nz
        self.dx = dx  # Voxel edge length (meters, NOT attometers)

        # Field allocations
        self.amplitude = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))
        self.density = ti.field(dtype=ti.f32, shape=(nx, ny, nz))

    @ti.func
    def get_position(self, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.math.vec3:
        """Get physical position of voxel center [i,j,k]."""
        return ti.Vector([
            (i + 0.5) * self.dx,
            (j + 0.5) * self.dx,
            (k + 0.5) * self.dx
        ])

    @ti.kernel
    def compute_laplacian(self):
        """Example: 6-connectivity Laplacian operator."""
        for i, j, k in self.amplitude:
            if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
                laplacian = (
                    self.amplitude[i+1, j, k] + self.amplitude[i-1, j, k] +
                    self.amplitude[i, j+1, k] + self.amplitude[i, j-1, k] +
                    self.amplitude[i, j, k+1] + self.amplitude[i, j, k-1] -
                    6.0 * self.amplitude[i, j, k]
                ) / (self.dx * self.dx)
                # Use laplacian in wave equation...
```

## Key Design Decisions

1. ✅ **Cell-centered grid** (industry standard)
2. ✅ **No attometer conversion** (computed positions)
3. ✅ **Cubic lattice initially** (symmetric, simpler)
4. ✅ **Configurable connectivity** (6/18/26 neighbors)
5. ✅ **Distance-based weighting** (physical accuracy)
6. ✅ **SI units (meters)** throughout code (clarity)

---

**Status**: Architecture defined, ready for implementation

**Next Steps**: Implement wave propagation engine using this grid structure
