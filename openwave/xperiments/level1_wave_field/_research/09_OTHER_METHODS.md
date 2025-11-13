# COMPARING OTHER NUMERICAL METHODS FOR OPENWAVE

This document evaluates alternative numerical methods that could potentially be used for simulating wave field dynamics in OpenWave LEVEL-1. Each method is assessed for its applicability, advantages, disadvantages, and practical fit with Energy Wave Theory (EWT) physics requirements.

**Conclusion**: The current **finite difference + wave equation PDE** approach is optimal for OpenWave. None of the alternative methods offer compelling advantages given our specific physics requirements (scalar wave field, discontinuous wave centers, real-space forces, GPU parallelization).

## Table of Contents

1. [Plane-Wave Eigenmodes](#1-plane-wave-eigenmodes)
2. [Lattice Boltzmann Methods (LBM)](#2-lattice-boltzmann-methods-lbm)
3. [Path Integral Monte Carlo (PIMC)](#3-path-integral-monte-carlo-pimc)
4. [Spectral Methods](#4-spectral-methods)
5. [Schrödinger Equation](#5-schrödinger-equation)
6. [Dirac Equation](#6-dirac-equation)
7. [Summary Table](#summary-table)
8. [Recommendation for OpenWave LEVEL-1](#recommendation-for-openwave-level-1)
   - [Stick with current approach: Finite Difference + Wave Equation PDE](#stick-with-current-approach-finite-difference--wave-equation-pde)
   - [Possible Enhancement: Hybrid Spectral for Phase 1-4](#possible-enhancement-hybrid-spectral-for-phase-1-4)
9. [Final Answer](#final-answer)
10. [Additional Considerations](#additional-considerations)
    - [Methods Not Evaluated (but not recommended)](#methods-not-evaluated-but-not-recommended)
    - [Performance Optimization Strategies (within finite difference framework)](#performance-optimization-strategies-within-finite-difference-framework)

## 1. Plane-Wave Eigenmodes

What it is: Decompose the field into a superposition of plane waves (Fourier modes), each with definite k-vector and frequency.

Applicability to OpenWave: ⚠️ Limited

Pros:

- Natural for analyzing wave propagation in infinite/periodic domains
- FFT makes it computationally efficient for certain operations
- Good for spectral analysis of multi-frequency fields

Cons:

- Poor for localized features: Particles (wave centers) are point-like reflectors - require infinite plane waves to represent
- Boundary conditions: Our universe has walls (ψ = 0 at boundaries), not periodic - plane waves don't naturally fit
- Nonlinearity: Wave-particle interactions, reflections from wave centers are nonlinear - plane waves best for linear systems
- Real-space forces: F = -2ρVf² × A∇A requires real-space gradients, not k-space

Verdict: ❌ Not recommended for primary simulation. Could be useful as a diagnostic/analysis tool to measure frequency content, but not
for time-stepping the core physics.

## 2. Lattice Boltzmann Methods (LBM)

What it is: Simulate fluid dynamics using discrete velocity distributions on a lattice. Particles move and collide according to kinetic
theory.

Applicability to OpenWave: ⚠️ Interesting but Complex

Pros:

- Natural for emergent fluid behavior from microscopic rules
- Handles complex geometries well
- Inherently parallel (local collisions)
- Could model medium "flow" if EWT medium has fluid-like properties

Cons:

- Different physics: LBM simulates Navier-Stokes fluid dynamics, not wave equations
- No direct wave propagation: Would need custom collision operators to recover ∂²ψ/∂t² = c²∇²ψ
- Overhead: More complex than direct PDE solving
- Unclear mapping: How do LBM velocity distributions map to scalar wave displacement ψ?

Verdict: ❌ Not recommended. OpenWave's wave equation ∂²ψ/∂t² = c²∇²ψ is already well-suited to direct finite difference methods. LBM
is overkill and doesn't naturally express wave physics.

## 3. Path Integral Monte Carlo (PIMC)

What it is: Quantum Monte Carlo method that samples Feynman path integrals to compute thermal/quantum properties.

Applicability to OpenWave: ❌ Not Applicable

Pros:

- Exact quantum statistical mechanics (for bosons)
- Could handle quantum effects if needed at extreme scales

Cons:

- Wrong regime: OpenWave simulates classical wave fields, not quantum path integrals
- Stochastic: Monte Carlo introduces noise - we want deterministic wave evolution
- Computational cost: Extremely expensive, requires sampling many paths
- No real-time dynamics: PIMC computes equilibrium properties, not time evolution

Verdict: ❌ Not applicable. OpenWave is a classical wave field simulator. PIMC is for quantum many-body systems at finite temperature.

## 4. Spectral Methods

What it is: Represent fields using global basis functions (Fourier, Chebyshev, Legendre), compute derivatives in spectral space.

Applicability to OpenWave: ✅ Potentially Useful (with caveats)

Pros:

- Exponential convergence: For smooth fields, spectral accuracy is extremely high
- Exact derivatives: Derivatives in Fourier space are just multiplication by ik
- Efficient for uniform grids: FFT makes O(N log N) transforms fast
- Energy conservation: Spectral methods naturally conserve energy in wave equations

Cons:

- Discontinuities: Wave centers (reflective voxels with ψ = 0) create discontinuities - spectral methods struggle with sharp features
(Gibbs phenomenon)
- Boundary conditions: Non-periodic boundaries require special treatment (Chebyshev for walls)
- Nonlinearity: Particle-wave interactions may require real-space operations anyway
- Memory: Need to store Fourier/spectral coefficients + real-space values

Potential Use Cases:

1. Initial energy distribution phase (Phases 1-4): While energy dilutes and no wave centers exist, spectral methods could efficiently
propagate smooth waves
2. Hybrid approach: Spectral for wave propagation, switch to real-space near wave centers
3. High-accuracy validation: Use spectral method as reference solution to validate finite difference

Verdict: ⚠️ Possibly useful as hybrid/validation, but not for primary particle simulation. Once wave centers (particles) are inserted,
discontinuities make spectral methods less attractive.

## 5. Schrödinger Equation

What it is: Quantum wave equation: iℏ∂ψ/∂t = -ℏ²/(2m)∇²ψ + Vψ

Applicability to OpenWave: ⚠️ Conceptually Related, Practically Different

Similarities:

- Both are wave equations with Laplacian ∇²ψ
- Both produce wavelike behavior and interference
- Standing waves in both systems

Key Differences:

| Aspect     | OpenWave (EWT)             | Schrödinger                               |
|------------|----------------------------|-------------------------------------------|
| Equation   | ∂²ψ/∂t² = c²∇²ψ            | iℏ∂ψ/∂t = -ℏ²/(2m)∇²ψ                     |
| Order      | 2nd order in time          | 1st order in time                         |
| Wave speed | Finite c                   | Infinite (dispersion relation ω = ℏk²/2m) |
| ψ meaning  | Real displacement          | Complex probability amplitude             |
| Energy     | E = ρV(fA)² (field energy) | E = ⟨ψ\|Ĥ\|ψ⟩ (expectation)                 |
| Particles  | Emergent from wave centers | Fundamental (wave-particle duality)       |

Could we use Schrödinger numerics?

- Time-stepping: No - Schrödinger uses 1st order (exp(-iHt) propagator), OpenWave needs 2nd order (verlet/leapfrog)
- Dispersion: Schrödinger has dispersion (different k travel at different speeds), EWT is non-dispersive (c constant)
- Complex vs Real: Schrödinger ψ is complex, EWT ψ is real scalar

Verdict: ❌ Different physics, can't directly use Schrödinger methods. However, conceptual parallels exist (standing waves =
particles).

## 6. Dirac Equation

What it is: Relativistic quantum wave equation for spin-½ particles: (iℏγ^μ∂_μ - mc)ψ = 0

Applicability to OpenWave: ❌ Not Applicable

Pros:

- Relativistic (consistent with c being fundamental)
- Naturally includes spin
- Predicts antimatter

Cons:

- Spinor field: ψ is a 4-component spinor, not scalar displacement
- Quantum: Treats ψ as quantum probability amplitude, not classical field
- Complexity: First-order in spacetime but coupled 4-component system
- Wrong regime: OpenWave models classical wave medium, not relativistic quantum fermions

Verdict: ❌ Not applicable. Dirac equation is for quantum spin-½ fermions. OpenWave simulates classical scalar wave fields in a medium.

## Summary Table

| Method                    | Applicability | Recommendation                                     |
|---------------------------|---------------|----------------------------------------------------|
| Plane-wave eigenmodes     | ⚠️ Limited    | Analysis tool only, not for time-stepping          |
| Lattice Boltzmann         | ⚠️ Complex    | Not recommended - wrong physics paradigm           |
| Path Integral Monte Carlo | ❌ No          | Quantum statistical mechanics, not classical waves |
| Spectral methods          | ✅ Possibly    | Hybrid approach for smooth wave propagation phase  |
| Schrödinger equation      | ⚠️ Conceptual | Different physics, can't use directly              |
| Dirac equation            | ❌ No          | Quantum spinors, not classical scalar waves        |

## Recommendation for OpenWave LEVEL-1

### Stick with current approach: Finite Difference + Wave Equation PDE

✅ Why it's optimal:

1. Direct match to physics: ∂²ψ/∂t² = c²∇²ψ is exactly what EWT describes
2. Handles discontinuities: Wave centers (ψ = 0 reflective voxels) are naturally represented
3. Local operations: Laplacian is nearest-neighbor, perfect for GPU parallelization (Taichi)
4. Real-space forces: F = -2ρVf² × A∇A computed directly from real-space gradients
5. Stable and well-understood: Leapfrog/Verlet time-stepping is stable, energy-conserving
6. Boundary conditions: Dirichlet (ψ = 0 at walls) trivial to implement

### Possible Enhancement: Hybrid Spectral for Phase 1-4

During initial energy distribution (before wave centers inserted), consider:

```python
# Phase 1-4: Smooth wave propagation (no particles yet)
if simulation_phase <= 4:
    # Use FFT-based spectral propagation (higher accuracy, faster)
    psi_k = fft3d(psi)
    psi_k_new = propagate_spectral(psi_k, dt)  # Exact in Fourier space
    psi = ifft3d(psi_k_new)

# Phase 5+: Particles inserted (discontinuities)
else:
    # Use finite difference (handles ψ = 0 wave centers)
    psi_new = propagate_finite_difference(psi, dt)
```

But even this is optional - finite difference works fine throughout.

## Final Answer

None of the listed methods offer compelling advantages over your current finite difference wave equation approach. Spectral methods could provide a marginal accuracy boost during the smooth propagation phase (Phases 1-4), but it's not necessary. Stick with what you have! 🎯

## Additional Considerations

### Methods Not Evaluated (but not recommended)

- **Finite Element Methods (FEM)**: Better for irregular geometries, but OpenWave uses regular grids - finite difference is simpler and faster
- **Discontinuous Galerkin**: Handles discontinuities well but adds complexity without clear benefits over finite difference with proper boundary handling
- **Smoothed Particle Hydrodynamics (SPH)**: Lagrangian method for fluids, not suitable for Eulerian wave fields on fixed grids
- **Cellular Automata**: Too coarse for continuous wave equations, better suited for discrete rule-based systems

### Performance Optimization Strategies (within finite difference framework)

Rather than switching methods, consider these optimizations:

1. **Adaptive timestep**: Adjust dt dynamically based on CFL condition and energy conservation
2. **Multi-grid methods**: Coarse-grain smooth regions, fine-grain near wave centers
3. **GPU optimization**: Maximize Taichi parallelization, memory coalescing, shared memory usage
4. **Higher-order stencils**: 4th or 6th order finite difference for better accuracy (still finite difference!)
5. **Energy drift correction**: Periodic global energy renormalization to maintain E = ρV(fA)²

These optimizations maintain the simplicity and directness of finite difference while improving performance and accuracy.
