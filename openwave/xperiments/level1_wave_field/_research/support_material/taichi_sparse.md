# TAICHI SPARSE DATA STRUCTURES FOR OPENWAVE

## Overview

This document evaluates **Taichi's Spatially Sparse Data Structures** ([docs](https://docs.taichi-lang.org/docs/sparse)) for potential use in OpenWave LEVEL-1 wave field simulation.

**Conclusion**: Sparse data structures are **NOT recommended** for OpenWave. While theoretically promising for initial wave propagation, the cost-benefit analysis shows marginal gains (~1.05x speedup) for significant implementation complexity. Dense grids remain the optimal choice.

## Table of Contents

1. [What Are Taichi Sparse Data Structures?](#what-are-taichi-sparse-data-structures)
2. [Theoretical Pros: Where Sparse Could Help](#theoretical-pros-where-sparse-could-help)
3. [Practical Cons: Why It's Problematic](#practical-cons-why-its-problematic)
4. [Reality Check: Wave Expansion Timeline](#reality-check-wave-expansion-timeline)
5. [Cost-Benefit Analysis](#cost-benefit-analysis)
6. [Final Verdict](#final-verdict)
7. [Better Optimization Strategies](#better-optimization-strategies)

## What Are Taichi Sparse Data Structures?

Taichi provides hierarchical sparse data structures that only allocate memory for **active voxels** rather than the entire dense grid.

### Key Features

- **Pointer fields**: Linked blocks, only allocate where data exists
- **Bitmasked fields**: Bit-level occupancy tracking
- **Dynamic allocation**: Runtime activation/deactivation of voxels
- **Hierarchical structure**: Multi-level grids (like octrees)

### Example Usage

```python
import taichi as ti

# Dense field (current approach)
wave_field_dense = ti.field(dtype=ti.f32, shape=(nx, ny, nz))

# Sparse field (theoretical alternative)
wave_field_sparse = ti.field(dtype=ti.f32)
sparse_tree = ti.root.pointer(ti.ijk, 16)  # 16³ blocks
sparse_tree.dense(ti.ijk, 8).place(wave_field_sparse)  # 8³ cells per block

# Only allocate blocks where wave exists
ti.activate(wave_field_sparse.parent(), [i, j, k])
```

## Theoretical Pros: Where Sparse Could Help

### 1. Memory Savings (Initially)

For wave propagation from a localized source:

```text
Phase 1-2: Energy concentrated at center
- Dense grid: 100³ voxels = 1M voxels stored (100% memory)
- Sparse: Only ~1000 active voxels initially (0.1% memory!)

Memory reduction: 1000x during initial timesteps
```

### 2. Computational Efficiency (Initially)

Only iterate over active voxels:

```python
# Dense (current approach)
for i, j, k in ti.ndrange(nx, ny, nz):  # Iterate ALL voxels
    propagate_wave(i, j, k)  # Most voxels have ψ ≈ 0 (wasted work)

# Sparse (theoretical)
for i, j, k in wave_field:  # Only iterate ACTIVE voxels
    propagate_wave(i, j, k)  # Every iteration does useful work
```

**Potential speedup**: 10x - 100x for localized waves (during sparse phase only)

### 3. Automatic Boundary Handling

- Unallocated voxels implicitly ψ = 0
- No need to store/iterate over empty space
- Could simplify boundary condition logic

### 4. Cache Efficiency

- Active data is contiguous in memory
- Better cache utilization than striding through mostly-zero dense arrays

## Practical Cons: Why It's Problematic

### 1. Laplacian Operator Complexity ⚠️

The wave equation ∂²ψ/∂t² = c²∇²ψ requires computing ∇²ψ at EVERY active voxel.

**Problem**: ∇²ψ needs 6 neighbors (±x, ±y, ±z)

```python
# Dense (simple and fast)
laplacian = (psi[i+1,j,k] + psi[i-1,j,k] +
             psi[i,j+1,k] + psi[i,j-1,k] +
             psi[i,j,k+1] + psi[i,j,k-1] - 6*psi[i,j,k]) / dx²

# Sparse (complex and slower)
@ti.func
def compute_laplacian_sparse(i, j, k):
    """Compute Laplacian with neighbor existence checking"""
    laplacian = -6.0 * wave_field[i, j, k]

    # Check each neighbor, use 0 if unallocated
    if ti.is_active(wave_field.parent(), [i+1, j, k]):
        laplacian += wave_field[i+1, j, k]
    else:
        laplacian += 0.0  # Unallocated = implicit zero

    if ti.is_active(wave_field.parent(), [i-1, j, k]):
        laplacian += wave_field[i-1, j, k]
    # ... repeat for all 6 neighbors (6 conditionals per voxel!)

    return laplacian / (dx * dx)
```

**Overhead**: 6 conditional checks per voxel per timestep

### 2. Wave Propagation Fills Space Rapidly

Unlike particle simulations (stay sparse), **waves expand to fill available volume**:

```text
t = 0:    ●              (1 voxel active, 0.001%)
t = 1:    ◯●◯            (7 voxels active)
t = 10:   ◯◯◯            (~500 voxels, 0.05%)
t = 100:  ◯◯◯◯◯          (~5,000 voxels, 0.5%)
t = 1000: ◯◯◯◯◯◯◯        (~50,000 voxels, 5%)
t = 5000: ◯◯◯◯◯◯◯◯◯◯    (~500,000 voxels, 50%) ← Crossover point
t = 10000: ◯◯◯◯◯◯◯◯◯◯◯  (1M voxels, 100% - fully dense)
```

**Result**: After initial expansion, the entire grid becomes active anyway!

### 3. Wavefront Growth Requires Neighbor Activation

Any active voxel propagating waves needs its neighbors to become active:

```python
@ti.kernel
def propagate_sparse_wave(dt: ti.f32):
    for i, j, k in wave_field:  # Active voxels only
        # Compute wave evolution
        laplacian = compute_laplacian_sparse(i, j, k)
        wave_field_new[i, j, k] = verlet_step(wave_field[i, j, k], laplacian, dt)

        # Auto-activate neighbors if wave is strong enough
        if abs(wave_field[i, j, k]) > activation_threshold:
            activate_neighbors(i, j, k)  # Add 6 neighbors to active set
```

**Problem**: Activation overhead + rapid growth = sparse advantage vanishes quickly

### 4. Implementation Complexity

Sparse fields require:

- Dynamic activation/deactivation logic
- Neighbor existence checking (6 conditionals per voxel)
- Activation heuristics (when to activate? how far ahead?)
- GPU synchronization (threads activating new voxels concurrently)
- Debugging difficulty (sparse iteration harder to visualize)
- Transition logic (sparse → dense when beneficial)

**Current dense approach**: Simple, direct, debuggable, fast

**Sparse approach**: Complex, indirect, harder to debug, overhead

### 5. Boundary Conditions

OpenWave has **ψ = 0 at walls** (Dirichlet boundaries).

- **Dense**: Explicitly set boundary voxels to 0 (simple)
- **Sparse**: Unallocated = implicit 0, but need to ensure neighbors of active voxels near boundaries behave correctly

Not impossible, but adds edge case complexity.

### 6. Force Calculation Requirements

F = -2ρVf² × A∇A requires amplitude gradients ∇A at particle positions:

```python
# Need neighbors to compute gradient
grad_A_x = (A[i+1,j,k] - A[i-1,j,k]) / (2*dx)
grad_A_y = (A[i,j+1,k] - A[i,j-1,k]) / (2*dy)
grad_A_z = (A[i,j,k+1] - A[i,j,k-1]) / (2*dz)
```

**Sparse issue**: What if neighbor is unallocated? Need fallback logic or pre-activation.

## Reality Check: Wave Expansion Timeline

### OpenWave Physical Parameters

Given typical OpenWave constants:

- **Wave speed**: c = 2.998 × 10⁸ m/s
- **Universe size**: ~10⁻¹⁴ m (femtometer scale for particle simulation)
- **Voxel resolution**: ~10⁻¹⁸ m (attometer scale)

### Time to Fill Universe

```text
t_fill = universe_size / wave_speed
t_fill = 10⁻¹⁴ m / 2.998×10⁸ m/s
t_fill ≈ 3.3×10⁻²³ seconds
```

### Number of Timesteps Required

With CFL stability condition (dt < dx/c for numerical stability):

```text
dx = 10⁻¹⁸ m (attometer voxel size)
dt_max = dx / c ≈ 3.3×10⁻²⁷ seconds

Timesteps to fill = t_fill / dt
Timesteps to fill = 3.3×10⁻²³ / 3.3×10⁻²⁷
Timesteps to fill ≈ 10,000 timesteps
```

### Sparsity Evolution Over Time

Starting from center point source (100³ grid = 1M voxels):

| Timestep | Active Voxels | Occupancy | Status |
|----------|---------------|-----------|--------|
| t = 0 | 1 | 0.0001% | Extremely sparse |
| t = 1 | 7 | 0.0007% | Extremely sparse |
| t = 10 | ~500 | 0.05% | Very sparse |
| t = 100 | ~5,000 | 0.5% | Sparse |
| t = 1,000 | ~50,000 | 5% | Moderately sparse |
| t = 5,000 | ~500,000 | 50% | **Crossover point** ⚠️ |
| t = 10,000 | ~1,000,000 | 100% | Fully dense |

**By timestep 5,000-10,000, the grid is essentially fully populated.**

## Cost-Benefit Analysis

### Sparse Benefits (First ~5,000 Timesteps Only)

**Computation savings during sparse phase**:

- Average sparsity: ~10-20% occupancy over sparse phase
- Potential speedup: ~5-10x for these timesteps
- **BUT**: These are only 5,000 out of potentially millions of total timesteps!

**Example**: If total simulation runs for 1,000,000 timesteps:

- Sparse phase: 5,000 timesteps (0.5% of total simulation)
- Dense phase: 995,000 timesteps (99.5% of total simulation)

**Overall speedup calculation**:

```text
Sparse phase speedup: 10x for 5,000 timesteps
Dense phase speedup: 1x for 995,000 timesteps

Time_sparse_with_optimization = 5,000 / 10 = 500 timesteps worth of work
Time_dense = 995,000 timesteps worth of work
Total_time = 500 + 995,000 = 995,500 timesteps worth of work

Overall speedup = 1,000,000 / 995,500 = 1.0045x (0.45% faster)
```

**Effective overall speedup: ~1.005x - 1.01x** (essentially negligible!)

### Sparse Costs

**Implementation complexity**:

- Neighbor existence checking (6 conditionals per voxel per timestep)
- Dynamic activation logic
- Activation heuristics (when to activate ahead of wavefront?)
- GPU synchronization for concurrent activation
- Debugging sparse iteration patterns
- Transition logic (sparse → dense at crossover)
- Edge case handling (boundaries, forces, gradients)

**Development time estimate**: Several days to weeks

**Runtime overhead**:

- Memory management (activation/deactivation operations)
- Sparse iteration overhead (pointer chasing, less cache-friendly than dense)
- Conditional branches (neighbor checking hurts GPU performance)
- Synchronization costs (concurrent activation)

**Maintenance burden**: Ongoing complexity in debugging and extending

### Cost-Benefit Summary

| Metric | Value | Assessment |
|--------|-------|------------|
| **Overall speedup** | ~1.005x - 1.01x | Negligible |
| **Speedup during sparse phase** | ~5-10x | Good, but irrelevant |
| **Sparse phase duration** | 0.5% of simulation | Too short to matter |
| **Implementation time** | Days to weeks | High cost |
| **Code complexity** | High | Ongoing burden |
| **Debugging difficulty** | Significantly harder | Maintenance cost |

**Cost-benefit ratio**: ~1.005x speedup / weeks of dev time = **NOT WORTH IT**

## Final Verdict

### ❌ Sparse Data Structures NOT Recommended

**Reasoning**:

1. **Short sparse phase**: Only ~5,000-10,000 timesteps before grid fills (0.5% of total)
2. **Negligible overall benefit**: ~0.5-1% total speedup despite 10x during sparse phase
3. **High complexity cost**: Days/weeks of development + ongoing maintenance burden
4. **Wave physics reality**: Waves naturally expand to fill space (unlike particle simulations that stay sparse throughout)
5. **GPU overhead**: Conditionals and memory management hurt parallel efficiency

### When Sparse WOULD Be Valuable

Sparse data structures are valuable when:

- **Permanently sparse**: Data stays sparse throughout simulation (e.g., particles in mostly empty space)
- **Long sparse phase**: Majority of simulation time spent in sparse state (>50%)
- **Extreme sparsity**: <1% occupancy maintained for most of simulation
- **Localized features only**: No global wave propagation filling space

**OpenWave doesn't fit any of these criteria** ❌

### ✅ Stick with Dense Grids

**Advantages of current dense approach**:

- Simple implementation ✓
- Fast on GPU (no conditionals, perfect parallelization) ✓
- Easy to debug ✓
- Covers 99%+ of simulation time ✓
- No complexity overhead ✓
- Direct memory access patterns ✓

## Better Optimization Strategies

Instead of sparse data structures, focus on optimizations with better ROI:

### 1. Efficient Dense Implementation (Current Approach)

```python
# Use Taichi's optimized dense fields (what you already have)
wave_field = ti.field(dtype=ti.f32, shape=(nx, ny, nz))

# GPU parallelization handles all voxels efficiently
@ti.kernel
def propagate():
    for i, j, k in wave_field:  # Parallel across all voxels
        # Simple Laplacian (no conditionals, no overhead)
        laplacian = (psi[i+1,j,k] + psi[i-1,j,k] +
                     psi[i,j+1,k] + psi[i,j-1,k] +
                     psi[i,j,k+1] + psi[i,j,k-1] - 6*psi[i,j,k]) / dx²
        # Verlet time integration
        psi_new[i,j,k] = 2*psi[i,j,k] - psi_old[i,j,k] + c²*dt²*laplacian
```

**Benefits**: Simple, fast, proven

### 2. Higher-Order Time Stepping

```python
# 4th order Runge-Kutta (RK4) allows larger timesteps
# Fewer timesteps needed for same accuracy
# More impact than sparse optimization!

@ti.kernel
def rk4_step(dt: ti.f32):
    # RK4 stages
    k1 = compute_derivative(psi)
    k2 = compute_derivative(psi + 0.5*dt*k1)
    k3 = compute_derivative(psi + 0.5*dt*k2)
    k4 = compute_derivative(psi + dt*k3)

    psi_new = psi + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
```

**Impact**: 2-4x fewer timesteps, more accuracy

### 3. GPU Memory Optimization

```python
# Use single precision (half the memory bandwidth)
wave_field = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # Not ti.f64

# Shared memory for Laplacian stencils (reduce global memory access)
@ti.kernel
def propagate_optimized():
    ti.block_dim = 64  # Optimize thread block size
    # Use shared memory for neighborhood access
```

**Impact**: 2x memory bandwidth improvement

### 4. Multi-Grid for Later Phases (Phase 5+)

Once particles exist, use adaptive resolution:

```python
# Fine grid near particles (high detail needed)
# Coarse grid far from particles (low detail sufficient)

# This is where hierarchical structures WOULD help
# (But for particle-field interaction, not initial wave propagation)
```

**Impact**: Significant for long-term particle dynamics

### 5. Spectral Methods for Phases 1-4 (Optional)

```python
# FFT-based propagation during smooth wave expansion
# Exact in Fourier space, faster than finite difference

if simulation_phase <= 4:
    psi_k = fft3d(psi)
    psi_k_new = propagate_spectral(psi_k, dt)
    psi = ifft3d(psi_k_new)
```

**Impact**: 2-5x faster during Phases 1-4 (0.5% of total), but simpler than sparse

### 6. Higher-Order Spatial Stencils

```python
# 4th or 6th order finite difference for Laplacian
# Better accuracy = larger timesteps possible

# 4th order (13-point stencil):
laplacian_4th_order = (
    -1/12 * (psi[i+2,j,k] + psi[i-2,j,k] + ...) +
    4/3 * (psi[i+1,j,k] + psi[i-1,j,k] + ...) +
    -5/2 * psi[i,j,k]
) / dx²
```

**Impact**: Better accuracy, still simple finite difference

## Conclusion

**Taichi sparse data structures** are a powerful tool, but **not appropriate for OpenWave LEVEL-1**. The wave physics (rapid expansion to fill space) and simulation timeline (sparse phase is <1% of total) make the cost-benefit analysis unfavorable.

**Recommendation**: Stick with dense grids and focus on optimizations that provide better ROI (higher-order methods, GPU optimization, spectral methods for smooth phases).

**Key takeaway**: Not every advanced feature is beneficial - sometimes the simple approach is optimal. Dense grids are the right choice for wave field simulation.
