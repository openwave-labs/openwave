# TAICHI DIFFERENTIABLE PROGRAMMING FOR OPENWAVE

## Overview

This document evaluates **Taichi's Differentiable Programming** ([docs](https://docs.taichi-lang.org/docs/differentiable_programming)) for potential use in OpenWave LEVEL-1 wave field simulation.

**Conclusion**: Autodiff is **NOT recommended** for OpenWave. The combination of no applicable use cases (forward physics simulation, not inverse problems), critical Metal backend incompatibility (Apple Silicon), and already-correct manual gradient computation makes autodiff both unnecessary and problematic.

## Table of Contents

1. [What Is Taichi Differentiable Programming?](#what-is-taichi-differentiable-programming)
2. [Potential Use Cases Analysis](#potential-use-cases-analysis)
   - [Inverse Problems](#1-inverse-problems)
   - [Parameter Optimization](#2-parameter-optimization)
   - [Force Gradient Optimization](#3-force-gradient-optimization)
   - [Neural Network Surrogates](#4-neural-network-surrogates)
3. [Critical Issue: Metal Backend Incompatibility](#critical-issue-metal-backend-incompatibility)
4. [Current Approach: Manual Gradients](#current-approach-manual-gradients)
5. [Cost-Benefit Analysis](#cost-benefit-analysis)
6. [Final Verdict](#final-verdict)
7. [When Autodiff WOULD Be Useful](#when-autodiff-would-be-useful)

## What Is Taichi Differentiable Programming?

Taichi's autodiff feature enables automatic computation of gradients (∂output/∂input) for optimization, inverse problems, and machine learning applications.

### Key Features

- **Forward mode**: Compute gradients alongside function evaluation
- **Reverse mode**: Backpropagation (similar to PyTorch)
- Built-in with `@ti.kernel` decorator using `ti.ad` API
- Supports gradient computation through Taichi kernels

### Example Usage

```python
import taichi as ti

# Define loss function
loss = ti.field(dtype=ti.f32, shape=())
x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def compute_loss():
    loss[None] = x[None]**2

# Compute gradient automatically
with ti.ad.Tape(loss=loss):
    compute_loss()

# x.grad now contains ∂loss/∂x = 2x
print(f"Gradient: {x.grad[None]}")
```

### Common Applications

- **Machine learning**: Training neural networks
- **Inverse problems**: Finding inputs that produce desired outputs
- **Parameter optimization**: Tuning model parameters to fit data
- **Sensitivity analysis**: Understanding parameter influence on outcomes

## Potential Use Cases Analysis

### 1. Inverse Problems

**Problem**: Given desired wave pattern at time T, what initial conditions produce it?

**Example Implementation**:

```python
@ti.kernel
def forward_simulation(A0: ti.f32) -> ti.f32:
    """
    Run wave propagation from initial amplitude A0.
    Return loss = how far result is from target pattern.
    """
    # Initialize wave with amplitude A0
    initialize_wave(A0)

    # Run wave propagation
    for step in range(num_timesteps):
        propagate_wave()

    # Compute loss: difference from target pattern
    loss = 0.0
    for i, j, k in wave_field:
        diff = wave_field[i,j,k] - target_pattern[i,j,k]
        loss += diff * diff

    return loss

# Use autodiff to find optimal initial amplitude
with ti.ad.Tape(loss=loss):
    loss[None] = forward_simulation(A0)
    # A0.grad now contains ∂loss/∂A0

# Gradient descent
optimal_A0 = A0[None] - learning_rate * A0.grad[None]
```

**Applicability to OpenWave**: ⚠️ **Limited**

**Reasoning**:

- EWT is a **forward simulation**: given initial conditions → predict evolution
- Not solving inverse problems (outcome → initial conditions)
- Scientific goal is understanding emergent physics, not control/design
- Initial conditions are typically simple (center-concentrated pulse, single particle, etc.)

**Verdict**: Not needed for core EWT physics simulation

---

### 2. Parameter Optimization

**Problem**: Find optimal EWT constants (ρ, c, f, A) that best match experimental observations

**Example Implementation**:

```python
# Optimizable parameters
medium_density = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
wave_speed = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

@ti.kernel
def simulate_particle(rho: ti.f32, c: ti.f32) -> ti.f32:
    """
    Run simulation with given parameters.
    Return loss = deviation from experimental data.
    """
    # Set up field with parameters
    setup_wave_field(rho, c)

    # Run simulation
    for step in range(num_steps):
        propagate_waves()
        apply_forces()
        update_particles()

    # Compare to experimental observations
    loss = 0.0
    for i in range(num_observations):
        predicted = get_observable(i)
        measured = experimental_data[i]
        loss += (predicted - measured)**2

    return loss

# Gradient descent on parameters
with ti.ad.Tape(loss=loss):
    loss[None] = simulate_particle(medium_density[None], wave_speed[None])

    # Update parameters
    medium_density[None] -= learning_rate * medium_density.grad[None]
    wave_speed[None] -= learning_rate * wave_speed.grad[None]
```

**Applicability to OpenWave**: ❌ **Not Applicable**

**Reasoning**:

- EWT constants are **derived from fundamental physics**, not tunable parameters:
  - ρ = 3.860×10²² kg/m³ (from energy density equations, not adjustable)
  - c = 2.998×10⁸ m/s (speed of light, fundamental constant)
  - f = 1.050×10²⁵ Hz (from Planck constants and wavelength)
  - A = varies by wave, but relationships are physics-based
- This is **not a machine learning problem** with free parameters
- Constants must match fundamental physics, not be fitted to data

**Verdict**: EWT constants are physics-determined, not optimized

---

### 3. Force Gradient Optimization

**Problem**: Compute higher-order force derivatives (∂²F/∂A², Hessian, etc.) for advanced particle dynamics

**Example Implementation**:

```python
@ti.kernel
def compute_force_from_amplitude(A_field):
    """Compute force from amplitude field"""
    # F = -2ρVf² × A∇A
    grad_A = compute_gradient(A_field)
    F = -2 * rho * V * f*f * A_field * grad_A
    return F

# First derivative: ∂F/∂A (force gradient)
force_gradient = ti.ad.grad(compute_force_from_amplitude)

# Second derivative: ∂²F/∂A² (Hessian, curvature)
force_hessian = ti.ad.grad(force_gradient)
```

**Applicability to OpenWave**: ❌ **Not Needed**

**Reasoning**:

- EWT force law: F = -2ρVf² × A∇A (first-order gradient sufficient)
- Particle acceleration: a = F/m (Newton's second law, first-order)
- EWT does not require higher-order force derivatives for physics
- First-order gradients (∇A) are computed manually and efficiently
- Higher-order derivatives would add complexity without physics benefit

**Verdict**: Not needed for EWT force calculations

---

### 4. Neural Network Surrogates

**Problem**: Train neural network to approximate wave propagation for faster inference

**Example Implementation**:

```python
# Neural network approximating wave equation
@ti.kernel
def neural_wave_propagator(input_state, weights) -> output_state:
    """
    NN: current wave state → next wave state
    Trained to approximate ∂²ψ/∂t² = c²∇²ψ
    """
    # Forward pass through neural network
    hidden1 = relu(matmul(input_state, weights1))
    hidden2 = relu(matmul(hidden1, weights2))
    output_state = matmul(hidden2, weights3)
    return output_state

# Training with autodiff
with ti.ad.Tape(loss=loss):
    predicted = neural_wave_propagator(input_state, weights)
    actual = exact_wave_propagation(input_state)
    loss[None] = (predicted - actual).norm_sqr()

    # Update weights via gradient descent
    for w in weights:
        w -= learning_rate * w.grad
```

**Applicability to OpenWave**: ❌ **Not Applicable**

**Reasoning**:

- OpenWave is **first-principles physics simulation**, not data-driven modeling
- We want **exact** solutions to wave equation, not approximations
- Neural networks introduce:
  - Approximation errors incompatible with precise physics
  - Training overhead (data generation, convergence)
  - Generalization issues (fails outside training distribution)
  - Loss of physical interpretability
- Wave equation solver is already fast on GPU (no need for surrogate)

**Verdict**: Wrong approach for fundamental physics simulation

---

## Critical Issue: Metal Backend Incompatibility

### The Problem

Taichi autodiff has **known compatibility issues with Metal backend** (Apple Silicon GPUs):

**From Taichi GitHub Issues and Documentation**:

- Autodiff kernels may fail to compile on Metal
- Gradient computation can crash or produce incorrect results
- Metal's shader language (MSL) has limitations with complex autodiff operations
- Reverse-mode autodiff particularly problematic on Metal
- No clear timeline for full Metal autodiff support

**Example Issue**:

```python
# This may fail or produce wrong results on Metal backend
@ti.kernel
def compute_gradients():
    # Autodiff operations
    pass

# Error on Metal:
# "Metal backend does not fully support autodiff"
# or produces silently incorrect gradients
```

### Why This Is Critical for OpenWave

**OpenWave's Target Platform**:

- Primary development: **macOS with Apple Silicon** (M1/M2/M3/M4)
- GPU backend: **Metal** (Apple's GPU API)
- Alternative backends (CUDA) not available on Mac
- CPU backend defeats purpose of GPU acceleration

**Metal backend incompatibility = dealbreaker** even if autodiff were needed

### Potential Workarounds (None Acceptable)

#### 1. Use CUDA Backend

```python
ti.init(arch=ti.cuda)  # Use NVIDIA GPU
```

**Problem**: Not available on macOS/Apple Silicon

#### 2. Use CPU Backend

```python
ti.init(arch=ti.cpu)  # Fall back to CPU
```

**Problem**: Defeats entire purpose of GPU acceleration, massive performance loss

#### 3. Manual Gradients

```python
# Implement chain rule manually
grad_output_wrt_input = compute_manually()
```

**Problem**: Complex, error-prone, maintenance burden

#### 4. Wait for Taichi Fix

**Problem**: Uncertain timeline, no guarantee of full Metal support

**None** of these workarounds are acceptable for OpenWave development

---

## Current Approach: Manual Gradients

OpenWave already computes gradients correctly using manual finite differences:

### Amplitude Gradient (for Force Calculation)

```python
@ti.func
def compute_amplitude_gradient(i, j, k):
    """
    Compute ∇A using central finite differences.

    This is the gradient needed for force calculation:
    F = -2ρVf² × A∇A
    """
    # Central difference: (A[i+1] - A[i-1]) / (2*dx)
    grad_A_x = (amplitude[i+1, j, k] - amplitude[i-1, j, k]) / (2.0 * dx)
    grad_A_y = (amplitude[i, j+1, k] - amplitude[i, j-1, k]) / (2.0 * dy)
    grad_A_z = (amplitude[i, j, k+1] - amplitude[i, j, k-1]) / (2.0 * dz)

    return ti.Vector([grad_A_x, grad_A_y, grad_A_z])
```

### Force Calculation

```python
@ti.kernel
def compute_force_field():
    """
    Compute force at each voxel: F = -2ρVf² × A∇A
    """
    for i, j, k in ti.ndrange(nx, ny, nz):
        # Get amplitude at this voxel
        A = amplitude[i, j, k]

        # Get amplitude gradient
        grad_A = compute_amplitude_gradient(i, j, k)

        # Compute force (monochromatic case: ∇f = 0)
        rho = constants.MEDIUM_DENSITY
        V = dx * dy * dz  # Voxel volume
        f = constants.EWAVE_FREQUENCY

        force[i, j, k] = -2.0 * rho * V * f * f * A * grad_A
```

### Why Manual Gradients Are Better

| Aspect | Manual Gradients | Autodiff |
|--------|------------------|----------|
| **Metal compatibility** | ✅ Works perfectly | ❌ Broken/unreliable |
| **Simplicity** | ✅ Simple, explicit | ❌ Complex, opaque |
| **Debuggability** | ✅ Easy to trace | ❌ Hard to debug |
| **Performance** | ✅ Optimized for case | ❌ Generic overhead |
| **Correctness** | ✅ Verified correct | ⚠️ May be wrong on Metal |
| **Maintenance** | ✅ Straightforward | ❌ Dependency on Taichi |

**Manual gradients are superior** for this use case

---

## Cost-Benefit Analysis

| Factor | Assessment |
|--------|------------|
| **Applicable use cases** | ❌ None identified for forward physics |
| **Metal compatibility** | ❌ Broken on Apple Silicon (primary platform) |
| **Physics requirements** | ❌ Not needed for EWT simulation |
| **Current gradients** | ✅ Already implemented correctly |
| **Development cost** | High (learning curve, debugging, workarounds) |
| **Maintenance burden** | High (Metal compatibility tracking) |
| **Benefit for OpenWave** | Zero (no use cases apply) |
| **Risk** | High (wrong gradients, crashes, platform lock-in) |

**Cost-benefit ratio**: High cost + High risk / Zero benefit = **STRONGLY NOT RECOMMENDED**

---

## Final Verdict

### ❌ Taichi Autodiff NOT Recommended for OpenWave

**Primary Reasons**:

1. **No applicable use cases**:
   - OpenWave is forward simulation (initial conditions → evolution)
   - Not solving inverse problems (outcome → initial conditions)
   - Not optimizing parameters (EWT constants are physics-based)
   - Not training ML models (first-principles physics, not data-driven)

2. **Metal backend incompatibility** ⚠️ **CRITICAL**:
   - Primary development platform: macOS with Apple Silicon
   - Metal backend has known autodiff issues
   - No acceptable workarounds (CUDA unavailable, CPU too slow)
   - Risk of silent gradient errors
   - Uncertain timeline for full Metal support

3. **Manual gradients already correct**:
   - F = -2ρVf² × A∇A computed with finite differences
   - Simple, explicit, debuggable
   - Works perfectly on Metal
   - Computationally efficient
   - Matches EWT physics exactly

4. **Adds complexity without benefit**:
   - Learning curve for autodiff API
   - Debugging opaque gradient computations
   - Maintenance burden tracking Metal support
   - Zero identified benefit for OpenWave

### ✅ Stick with Manual Gradient Computation

**Current implementation is optimal**:

```python
# What OpenWave already does (and should continue doing)

# 1. Compute amplitude gradient manually
grad_A = compute_amplitude_gradient(i, j, k)

# 2. Compute force from gradient
F = -2 * rho * V * f² * A * grad_A

# Simple, explicit, correct, Metal-compatible
```

**Advantages**:

- ✅ Works on Metal backend (primary platform)
- ✅ Simple and explicit (easy to understand)
- ✅ Easy to debug and verify
- ✅ Computationally efficient
- ✅ Matches EWT physics exactly
- ✅ No external dependencies
- ✅ Future-proof (not tied to Taichi autodiff roadmap)

---

## When Autodiff WOULD Be Useful

Taichi autodiff is valuable for problems involving:

### 1. Machine Learning Applications

**Use cases**:

- Training neural networks for pattern recognition
- Gradient-based optimization of model parameters
- Reinforcement learning (policy gradients)
- Generative models (GANs, VAEs)

**Why**: Automatic backpropagation through complex model architectures

**OpenWave fit**: ❌ Not ML, first-principles physics

### 2. Inverse Problems

**Use cases**:

- Design optimization (given desired outcome, find optimal input)
- Optimal control (find control sequence to reach target state)
- Medical imaging (reconstruct internal structure from measurements)
- Seismic inversion (infer Earth structure from wave measurements)

**Why**: Gradient descent to minimize difference between prediction and target

**OpenWave fit**: ❌ Forward simulation, not inverse/control problems

### 3. Parameter Estimation and Data Assimilation

**Use cases**:

- Fit model parameters to experimental measurements
- Calibrate simulations to real-world observations
- Uncertainty quantification via gradient-based sampling
- Data assimilation (combine model and observations)

**Why**: Optimize free parameters to match data

**OpenWave fit**: ❌ EWT constants are physics-derived, not free parameters

### 4. Sensitivity Analysis

**Use cases**:

- Understand how output varies with input parameters
- Identify most influential parameters (gradient magnitude)
- Robust design (minimize sensitivity to perturbations)
- Uncertainty propagation

**Why**: Compute ∂output/∂parameter efficiently

**OpenWave fit**: ⚠️ Possible research use, but not core simulation need

### 5. Differentiable Physics for Control

**Use cases**:

- Robot motion planning through physics simulator
- Trajectory optimization with physics constraints
- Learning control policies via gradient descent
- Optimal manipulation under physical laws

**Why**: Gradients through physics enable gradient-based control optimization

**OpenWave fit**: ❌ Understanding physics, not controlling systems

---

## Summary

**Taichi autodiff** is a powerful tool for machine learning, inverse problems, and parameter optimization. However, **it is not appropriate for OpenWave** due to:

1. **No applicable use cases** (forward physics, not inverse/ML/optimization)
2. **Metal backend incompatibility** (critical for macOS/Apple Silicon)
3. **Manual gradients already optimal** (simple, correct, Metal-compatible)
4. **Zero benefit for high cost** (complexity, risk, maintenance)

**Recommendation**: Continue using manual finite difference gradients for force calculations. This is the right approach for OpenWave's physics simulation needs.

**Key Takeaway**: Advanced features should only be adopted when they solve actual problems. Manual gradients are simpler, more reliable, and perfectly suited for OpenWave's requirements.

---

## Related Documentation

- [Comparing Other Numerical Methods](./10_OTHER_METHODS.md) - Analysis of alternative approaches
- [Taichi Sparse Data Structures](./taichi_sparse.md) - Analysis of sparse grids
- [Wave Engine Force Calculation](./03b_WAVE_ENGINE_B.md) - Current gradient implementation
