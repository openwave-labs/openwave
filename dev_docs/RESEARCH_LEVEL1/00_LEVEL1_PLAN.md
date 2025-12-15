# LEVEL-1: DESIGN & PLAN

## Overview

LEVEL-1 is a wave-field based physics simulator that models Energy Wave Theory (EWT) at the wavelength scale, from fundamental particles to molecules. Unlike LEVEL-0's granule-based medium, LEVEL-1 uses efficient PDE-based wave propagation on 3D grids to simulate hundreds of particles with interactive performance.

**Core Innovation**: Waves are primary - particles and forces emerge from wave interference patterns.

## Quick Reference

| Aspect | LEVEL-0 (Granule-Based) | LEVEL-1 (Wave-Field) |
|--------|-------------------------|----------------------|
| **Scale** | Planck-scale to λ | λ-scale to molecules |
| **Medium** | 1M granules (particles) | 1B voxels (grid) |
| **Wave Engine** | Phase-shifted oscillations | PDE wave equation |
| **Particles** | N/A (made of granules) | Hundreds (wave centers) |
| **Resolution** | ~100³ granule grid | ~1000³ voxel grid (10× better) |
| **Use Case** | Educational, visualization | Scientific research, engineering |
| **Performance** | Rendering bottleneck | Field visualization (slices only) |
| **Timestep** | Variable (elapsed time) | Fixed (CFL-limited) |
| **Units** | Attometers (spatial only) | Attometers + Rontoseconds |

## Table of Contents

1. [Architecture Components](#architecture-components)
   - [1. Wave Field (Grid Medium)](#1-wave-field-grid-medium)
   - [2. Energy Charging](#2-energy-charging)
   - [3. Wave Engine (Propagation)](#3-wave-engine-propagation)
   - [4. Boundary Conditions](#4-boundary-conditions)
   - [5. Particles (Wave Centers)](#5-particles-wave-centers)
   - [6. Forces (Emergent)](#6-forces-emergent)
   - [7. Visualization](#7-visualization)
1. [Energy Evolution Sequence](#energy-evolution-sequence)
   - [Phase 1: Center-Concentrated Pulse Injection](#phase-1-center-concentrated-pulse-injection)
   - [Phase 2: Outward Propagation](#phase-2-outward-propagation)
   - [Phase 3: Boundary Reflections](#phase-3-boundary-reflections)
   - [Phase 4: Energy Dilution (Quasi-Equilibrium)](#phase-4-energy-dilution-quasi-equilibrium)
   - [Phase 5: Wave Center Insertion](#phase-5-wave-center-insertion)
   - [Phase 6: Standing Wave Emergence](#phase-6-standing-wave-emergence)
   - [Phase 7: Particle Formation](#phase-7-particle-formation)
1. [Implementation Details](#implementation-details)
   - [Initialization Strategy](#initialization-strategy)
   - [Timestep Strategy](#timestep-strategy)
   - [Numerical Precision](#numerical-precision)
1. [Physics Equations](#physics-equations)
   - [Energy Density (EWT - Frequency-Centric)](#energy-density-ewt---frequency-centric)
   - [Wave Propagation](#wave-propagation)
   - [Force Calculation](#force-calculation)
   - [Particle Motion](#particle-motion)
   - [Amplitude Envelope Tracking](#amplitude-envelope-tracking)
1. [Detailed Documentation](#detailed-documentation)
1. [Current Status & Next Steps](#current-status--next-steps)
   - [Validation Requirements](#validation-requirements)
   - [Known Challenges](#known-challenges)
   - [Implementation Roadmap](#implementation-roadmap)
   - [Next Immediate Actions](#next-immediate-actions)
1. [Key Design Decisions Summary](#key-design-decisions-summary)

## Architecture Components

### 1. Wave Field (Grid Medium)

**Purpose**: Store and propagate wave properties through 3D space

**Implementation**:

- **Cell-centered cubic grid**: Voxel `[i,j,k]` = center at `(i+0.5)*dx`
- **Shape**: `(nx, ny, nz)` - asymmetric universes supported
- **Resolution**: 1 billion voxels (~1000³) - 10× better than LEVEL-0
- **Voxel size**: `dx = wavelength / points_per_wavelength` (typically 40 points/λ)
- **Units**: Attometer scaling for spatial (f32 precision)

**Key Fields**:

- `psiL_am[i,j,k]`: Instantaneous ψ (high-frequency oscillation)
- `amplitude_am[i,j,k]`: Envelope A = max|ψ| (slowly varying)
- `force[i,j,k]`: Force vector from amplitude gradient (Newtons)
- `wave_direction[i,j,k]`: Energy flux direction (unit vector)

**See**: [`01a_WAVE_FIELD_grid.md`](./01a_WAVE_FIELD_grid.md), [`01b_WAVE_FIELD_properties.md`](./01b_WAVE_FIELD_properties.md)

### 2. Energy Charging

**Purpose**: Initialize wave field with correct total energy

**Method**: Spherical Gaussian pulse at universe center

```python
# Phase 1: Concentrated pulse (single injection)
E_total = equations.compute_energy_wave_equation(volume)  # EWT equation
charge_spherical_gaussian(center, E_total, width=3λ)
```

**Energy Equation** (EWT, frequency-centric):

```text
E = ρV(fA)²    (no ½ factor - total energy, not time-averaged)
```

**See**: [`02a_WAVE_ENGINE_charge.md`](./02a_WAVE_ENGINE_charge.md)

### 3. Wave Engine (Propagation)

**Purpose**: Evolve wave field using PDE physics

**Core Equation**:

```text
∂²ψ/∂t² = c²∇²ψ    (Classical 3D wave equation)
```

**Numerical Method**:

- **Leap-frog scheme**: `ψ_new = 2ψ - ψ_old + (cdt/dx)²∇²ψ`
- **Stability**: CFL condition `dt ≤ dx/(c√3)` for 3D
- **Timestep**: Fixed (not elapsed time!) - typically ~2e-27 s
- **Scaling**: Rontosecond (10⁻²⁷ s) for temporal precision

```python
# Temporal: RONTOSECOND = 1e-27 s
#   - Period: ~95.2 rs (vs 9.52e-26 s)
#   - Timestep: ~2.4 rs (vs 2.4e-27 s)
#   - Naming: variables/fields with suffix '_rs'
# RONTOSECOND = 1e-27  # s, rontosecond time scale
```

**Laplacian Operator** (6-connectivity):

```python
∇²ψ[i,j,k] = (
    ψ[i+1,j,k] + ψ[i-1,j,k] +
    ψ[i,j+1,k] + ψ[i,j-1,k] +
    ψ[i,j,k+1] + ψ[i,j,k-1] -
    6 × ψ[i,j,k]
) / dx²
```

**See**: [`02_WAVE_ENGINE.md`](./02_WAVE_ENGINE.md), [`02b_WAVE_ENGINE_propagate.md`](./02b_WAVE_ENGINE_propagate.md)

### 4. Boundary Conditions

**Universe Walls**: Fixed boundaries at grid edges

- **Dirichlet**: `ψ = 0` at all boundaries (never updated)
- **Effect**: Waves reflect back into domain (phase inversion)
- **Energy**: Total energy conserved (no absorption at walls)

**Wave Centers** (Particles): Reflective voxels inside domain

- **Behavior**: `ψ = 0` always at wave center position
- **Effect**: Create standing wave patterns via interference
- **Formation**: Insert after energy dilution (Phase 5)

**See**: [`02c_WAVE_ENGINE_interact.md`](./02c_WAVE_ENGINE_interact.md)

### 5. Particles (Wave Centers)

**What They Are**: Reflective points that create standing waves

**Count**: Hundreds to thousands (not millions like LEVEL-0 granules)

- 1 neutrino = 1 wave center
- 1 electron = 10 wave centers ("lock" together)
- 1 proton = complex multi-center structure

**Properties**:

- Position, velocity, mass (computed from trapped energy)
- Wave reflection creates standing wave pattern
- Standing wave radius: r = nλ/2 (nodes at half-wavelengths)
- Mass = E/c² where E = trapped standing wave energy

**See**: [`03_FUNDAMENTAL_PARTICLE.md`](./03_FUNDAMENTAL_PARTICLE.md), [`05_MATTER.md`](./05_MATTER.md)

### 6. Forces (Emergent)

**Fundamental Principle**: All forces from amplitude gradients

**Force Formula** (EWT frequency-centric):

```text
Energy density: u = ρ(fA)²
Force: F = -∇E = -2ρVfA × [f∇A + A∇f]    (dual-term)
Monochromatic: F = -2ρVf² × A∇A          (∇f = 0)

Units: [N] = [kg⋅m/s²]
```

**MAP (Minimum Amplitude Principle)**: Particles move toward lower amplitude

**Emergent Force Types**:

- **Gravity**: Amplitude shading from trapped energy (mass)
- **Electric**: Different wave reflection patterns (charge types)
- **Magnetic**: Moving wave patterns (velocity-dependent)
- **Strong**: Near-field standing wave coupling

**See**: [`04_FORCE_MOTION.md`](./04_FORCE_MOTION.md)

### 7. Visualization

**Primary Methods**:

1. **FLUX Detector planes**: 2D slices through 3D field (Taichi meshes)
1. **Wall painting**: Universe boundaries show wave properties
1. **Particle spray**: Tiny particles at wave nodes/antinodes
1. **Vector fields**: Force arrows (Taichi lines)
1. **Streamlines**: Energy flow paths

**NOT rendering all voxels**: Only field slices, isosurfaces, sample points

**See**: [`07_VISUALIZATION.md`](./07_VISUALIZATION.md)

## Energy Evolution Sequence

LEVEL-1 follows a 7-phase natural evolution from pulse to particle:

### Phase 1: Center-Concentrated Pulse Injection

- Single Gaussian pulse at universe center
- Total energy = `equations.compute_energy_wave_equation(volume)`
  - `E = ρV(fA)²`
- Width = 3× wavelength
- Implementation: `charge_spherical_gaussian()`

### Phase 2: Outward Propagation

- Wave equation `∂²ψ/∂t² = c²∇²ψ` governs evolution
- Spherical wave fronts expand at speed c
- 1/r amplitude falloff emerges naturally (no manual scaling)
- Energy conserved by wave equation physics

### Phase 3: Boundary Reflections

- Waves reach universe walls (ψ = 0 boundaries)
- Reflect back into domain (phase inversion)
- Create interference patterns
- Total energy remains constant

### Phase 4: Energy Dilution (Quasi-Equilibrium)

- Multiple reflections distribute energy throughout field
- Reaches stable distributed state after ~10,000 timesteps
- Energy density relatively uniform
- System ready for wave center insertion

### Phase 5: Wave Center Insertion

- Insert reflective voxels at specific positions
- `ψ = 0` always at wave centers (never changes)
- Function like internal boundary walls
- Create local reflection sites

### Phase 6: Standing Wave Emergence

- Reflected waves (OUT) interfere with incoming waves (IN)
- Constructive/destructive interference → nodes and antinodes
- Pattern: Φ = Φ₀ e^(iωt) sin(kr)/r (Wolff's solution)
- Steady-state standing wave forms around center

### Phase 7: Particle Formation

- Standing wave boundary defines particle extent
- Energy trapped within pattern
- **Particle mass = total trapped energy / c²**
- Fundamental particle successfully formed

## Implementation Details

### Initialization Strategy

**Mirrors LEVEL-0** `BCCLattice` approach:

```python
# User specifies desired universe size (can be asymmetric)
init_universe_size = [250e-18, 250e-18, 125e-18]  # meters

# Compute from TARGET_VOXELS = 1e9 (config constant)
volume = init_universe_size[0] * [1] * [2]
dx = (volume / TARGET_VOXELS)**(1/3)  # Cubic voxel size

# Grid dimensions (integer voxel counts, can differ per axis)
nx = int(init_universe_size[0] / dx)  # ~1260
ny = int(init_universe_size[1] / dx)  # ~1260
nz = int(init_universe_size[2] / dx)  # ~630

# Result: ~1B voxels, 10× better resolution than LEVEL-0 per dimension
```

**Why 1000× more voxels?**

- LEVEL-0: Every granule rendered → 1M limit
- LEVEL-1: Only slices rendered → 1B tractable
- Result: 10× better spatial resolution in each dimension

### Timestep Strategy

**CRITICAL**: Must use fixed timesteps (not elapsed time)

**Why?**

```text
CFL requirement: dt ≤ dx/(c√3) ≈ 1.2e-26 s (for dx = 6 am, 1B voxels)
Frame time: ~0.016 s (60 FPS)
Violation: 10²⁴× over limit → INSTANT NUMERICAL EXPLOSION!
```

**Solution**: Apply SLO_MO to wave speed (not timestep)

```python
# Slow the wave speed by SLO_MO factor
c_slo = c / SLO_MO  # SLO_MO = 1.05×10²⁵

# New CFL critical timestep
dt_critical = dx / (c_slo * √3) ≈ 0.121 s

# Frame timestep (fixed)
dt_frame = 1/60  # 0.0167 s

# Check: 0.0167 s < 0.121 s → STABLE! ✓
```

### Numerical Precision

**Spatial Scaling Only**:

- **Spatial**: ATTOMETER = 10⁻¹⁸ m (field suffix: `_am`)

**Why attometer scaling?**

- Maintain 6-7 significant digits in f32 calculations
- Prevent catastrophic cancellation in spatial derivatives
- Values in optimal f32 range (1-100)

**No temporal scaling needed**: With SLO_MO, timesteps are ~0.016 s (milliseconds), already in good f32 range.

**Example** (6 fm³ universe, 1B voxels):

```python
# Physical values (SI units)
dx = 6e-18 m      # voxel edge
dt = 0.0167 s     # frame time (60 FPS)

# Scaled spatial values only
dx_am = 6         # attometers (for precision)
dt = 0.0167       # seconds (no scaling needed)
```

## Physics Equations

### Energy Density (EWT - Frequency-Centric)

```text
u = ρ(fA)²    [J/m³]

where:
ρ = 3.860×10²² kg/m³ (medium density)
f = c/λ (frequency, Hz)
A = amplitude envelope (meters)

Note: No ½ factor - total energy capacity, not time-averaged
```

### Wave Propagation

```text
∂²ψ/∂t² = c²∇²ψ

where:
ψ = displacement field (meters, scaled to attometers)
c = 2.998×10⁸ m/s (speed of light)
∇² = Laplacian operator (measures neighbor deviation)
```

### Force Calculation

```text
Full form (dual-term):
F = -∇E = -2ρVfA × [f∇A + A∇f]    [N]

Monochromatic (single frequency):
F = -2ρVf² × A∇A                   [N]

where:
V = dx³ (voxel volume, m³)
f∇A = amplitude gradient term (primary)
A∇f = frequency gradient term (secondary, for multi-frequency)
```

**Why frequency-centric?**

- **Elegance**: E = ρV(fA)² simpler than E = ρVc²(A/λ)²
- **Planck alignment**: E = hf (energy ∝ frequency)
- **Human intuition**: Radio (98.7 FM), audio (440 Hz), WiFi (2.4 GHz)
- **Spacetime coupling**: f (temporal) × A (spatial) = natural pairing
- **Direct measurement**: f = 1/dt from timing peaks

### Particle Motion

```text
Newton's 2nd Law:
F = ma
a = F/m

Euler Integration:
v_new = v_old + a × dt
x_new = x_old + v × dt

where:
m = E_trapped / c² (mass from standing wave energy)
```

### Amplitude Envelope Tracking

```text
Two distinct fields needed:

ψ(t) = instantaneous displacement (fast oscillation ~10²⁵ Hz)
A(t) = amplitude envelope = max|ψ| (slow variation)

Wave equation propagates ψ:
∂²ψ/∂t² = c²∇²ψ

Envelope tracked from ψ:
A[i,j,k] = max_over_time(|ψ[i,j,k]|)

Forces use A (not ψ):
F = -2ρVf² × A∇A    (MAP: particles respond to envelope)
```

## Detailed Documentation

### Core Architecture

1. **[01a_WAVE_FIELD_grid.md](./01a_WAVE_FIELD_grid.md)** - Grid architecture, cell-centered design, index-to-position mapping
1. **[01b_WAVE_FIELD_properties.md](./01b_WAVE_FIELD_properties.md)** - Scalar/vector properties, field storage, WaveField class

### Wave Engine

1. **[02_WAVE_ENGINE.md](./02_WAVE_ENGINE.md)** - Overview, wave mechanics, energy evolution phases
1. **[02a_WAVE_ENGINE_charge.md](./02a_WAVE_ENGINE_charge.md)** - Initial energy charging methods
1. **[02b_WAVE_ENGINE_propagate.md](./02b_WAVE_ENGINE_propagate.md)** - PDE propagation, Laplacian, CFL stability
1. **[02c_WAVE_ENGINE_interact.md](./02c_WAVE_ENGINE_interact.md)** - Boundary reflections, superposition

### Particles, Forces & Visualization

1. **[03_FUNDAMENTAL_PARTICLE.md](./03_FUNDAMENTAL_PARTICLE.md)** - Wave centers, reflection behavior, mass accumulation
1. **[04_FORCE_MOTION.md](./04_FORCE_MOTION.md)** - Force calculation, MAP principle, emergent fields
1. **[05_MATTER.md](./05_MATTER.md)** - Composite particles, electron formation, binding
1. **[06_PHOTON_HEAT.md](./06_PHOTON_HEAT.md)** - (placeholder for future content)
1. **[07_VISUALIZATION.md](./07_VISUALIZATION.md)** - Detector planes, wall painting, particle spray, vector fields
1. **[08_NUMERICAL_ANALYSIS.md](./08_NUMERICAL_ANALYSIS.md)** - (placeholder for data sampling, plotting)

### Supporting Material

Located in `support_material/`:

- **[L1field_QFT.md](./support_material/L1field_QFT.md)** - Connections to QFT and lattice QCD
- **[phase_shift.md](./support_material/phase_shift.md)** - Phase shift wave mechanics
- **[wave_exponential.md](./support_material/wave_exponential.md)** - Exponential wave solutions
- **[wave_spherical.md](./support_material/wave_spherical.md)** - Spherical wave solutions
- **[wave_vs_heat_equation.md](./support_material/wave_vs_heat_equation.md)** - Wave vs heat equation comparison
- **[taichi_sparse.md](./support_material/taichi_sparse.md)** - Why sparse data structures not used
- **[taichi_autodiff.md](./support_material/taichi_autodiff.md)** - Why autodiff not used

### Reference

1. **[09_OTHER_METHODS.md](./09_OTHER_METHODS.md)** - Evaluation of alternative numerical methods

## Current Status & Next Steps

### Validation Requirements

**NOT YET VALIDATED**: Wave propagation physics needs experimental verification

**Success Criteria**:

- Wave speed ≈ c (within 5-10% tolerance)
- Wavelength ≈ λ (within 5-10% tolerance)
- Energy conservation (< 1% drift over 1M timesteps)
- Standing wave formation at correct radii
- MAP force behavior (particles → amplitude minimum)

**Test Cases**:

1. Single point source → spherical wave propagation
1. Plane wave → reflection from boundaries
1. Two sources → interference patterns
1. Wave center → standing wave formation
1. Multiple particles → force-driven motion

### Known Challenges

**High Energy Density**:

- Energy waves carry enormous energy
- High forces and momentum from wave interactions
- Numerical stability critical (CFL condition)
- Careful parameter tuning required

**Approach**:

- Start simple
- Validate at each step
- Use stable integration (Verlet/leapfrog)
- Monitor energy conservation continuously

### Implementation Roadmap

#### Phase A: Core Wave Engine (Current Priority)

1. ✅ Grid architecture designed
1. ✅ Wave properties defined
1. ✅ Force calculation specified
1. ⬜ Implement wave equation solver
1. ⬜ Verify energy conservation
1. ⬜ Test boundary reflections

#### Phase B: Energy Charging

1. ⬜ Implement Gaussian pulse injection
1. ⬜ Verify total energy matches EWT equation
1. ⬜ Test propagation and dilution
1. ⬜ Measure wave speed and wavelength

#### Phase C: Wave Centers

1. ⬜ Implement reflective voxels
1. ⬜ Test standing wave formation
1. ⬜ Measure particle mass from trapped energy
1. ⬜ Verify MAP force behavior

#### Phase D: Particle Dynamics

1. ⬜ Implement force interpolation
1. ⬜ Test particle motion (Newton's laws)
1. ⬜ Verify multi-particle interactions
1. ⬜ Test particle formation ("lock" events)

#### Phase E: Visualization

1. ⬜ Detector plane rendering
1. ⬜ Wall painting
1. ⬜ Particle spray for standing waves
1. ⬜ Vector field display

#### Phase F: Scientific Analysis

1. ⬜ Data sampling and export
1. ⬜ Plot generation (seaborn)
1. ⬜ Offline video rendering
1. ⬜ Quantitative measurements

### Next Immediate Actions

1. **Implement basic wave equation solver** (02b_WAVE_ENGINE_propagate.md)
   - Leap-frog scheme
   - 6-connectivity Laplacian
   - CFL stability check
   - Energy monitoring

1. **Create minimal test case**
   - Small grid (50³)
   - Single Gaussian pulse
   - Verify propagation
   - Check energy conservation

1. **Validate against analytical solution**
   - Compare to Wolff's spherical wave
   - Measure wave speed
   - Verify 1/r amplitude falloff

## Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Grid type** | Cell-centered cubic | Industry standard, optimal for PDEs |
| **Array indexing** | 3D `[nx,ny,nz]` | Natural for grid operations, not 1D |
| **Asymmetric support** | Yes (nx≠ny≠nz) | Memory efficiency, follows LEVEL-0 design |
| **Voxel size** | Cubic (same dx all axes) | Preserves wave physics, isotropic |
| **Spatial scaling** | Attometers (10⁻¹⁸ m) | f32 precision, prevents cancellation |
| **Temporal scaling** | Rontoseconds (10⁻²⁷ s) | f32 precision, CFL timesteps |
| **Timestep strategy** | Fixed (not elapsed) | Numerical stability, CFL requirement |
| **Wave equation** | PDE (not Huygens) | 10× faster, naturally conserves energy |
| **Connectivity** | 6 face neighbors | Simplest, sufficient for isotropy |
| **Boundary conditions** | Dirichlet (ψ=0) | Reflective walls, energy conservation |
| **Energy formulation** | Frequency-centric E=ρV(fA)² | Elegant, aligns with Planck E=hf |
| **Force formula** | Dual-term F=-2ρVfA[f∇A+A∇f] | Complete, handles multi-frequency |
| **Primary property** | Frequency f (not λ) | Direct measurement, human-intuitive |
| **Sparse grids** | No | Only 0.5% of simulation sparse |
| **Autodiff** | No | Forward physics, Metal incompatible |
| **Target voxels** | 1×10⁹ | 1000× more than LEVEL-0 granules |

---

**Document Version**: 2.0 (Post-Reorganization)

**Last Updated**: 2025-11-12

**Status**: Architecture complete, implementation in progress
