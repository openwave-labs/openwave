# WAVE EQUATION VS HEAT EQUATION: FUNDAMENTAL COMPARISON

## Overview

This document compares and contrasts the **wave equation** (used in OpenWave) and the **heat equation** (diffusion process), exploring their mathematical structure, physical meaning, and fundamental differences in behavior.

**Key Insight**: These equations differ by one derivative order in time, leading to profoundly different physics: waves propagate and oscillate (reversible), while heat diffuses and smooths (irreversible).

## Table of Contents

1. [The Equations](#the-equations)
2. [Mathematical Structure Comparison](#mathematical-structure-comparison)
3. [Physical Interpretation](#physical-interpretation)
4. [Fundamental Differences](#fundamental-differences)
5. [Solution Behavior](#solution-behavior)
6. [Energy and Information](#energy-and-information)
7. [Numerical Methods Comparison](#numerical-methods-comparison)
8. [Relationship and Connections](#relationship-and-connections)
9. [Relevance to OpenWave](#relevance-to-openwave)
10. [Summary Table](#summary-table)

## The Equations

### Wave Equation (OpenWave)

**Standard form**:

```text
∂²ψ/∂t² = c²∇²ψ
```

**Components**:

- ψ(x,y,z,t): Wave displacement field
- c: Wave propagation speed (constant, e.g., speed of light)
- ∇²ψ: Laplacian (spatial curvature)
- ∂²/∂t²: **Second-order** time derivative (acceleration)

**1D form**:

```text
∂²ψ/∂t² = c² ∂²ψ/∂x²
```

**3D form (OpenWave)**:

```text
∂²ψ/∂t² = c²(∂²ψ/∂x² + ∂²ψ/∂y² + ∂²ψ/∂z²)
```

### Heat Equation (Diffusion)

**Standard form**:

```text
∂u/∂t = α∇²u
```

**Components**:

- u(x,y,z,t): Temperature field (or concentration)
- α: Thermal diffusivity (material property)
- ∇²u: Laplacian (spatial curvature)
- ∂/∂t: **First-order** time derivative (rate of change)

**1D form**:

```text
∂u/∂t = α ∂²u/∂x²
```

**3D form**:

```text
∂u/∂t = α(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
```

## Mathematical Structure Comparison

### Order of Time Derivative (CRITICAL DIFFERENCE)

| Equation | Time Derivative | Order | Meaning |
|----------|----------------|-------|---------|
| **Wave** | ∂²ψ/∂t² | 2nd order | Acceleration, oscillation |
| **Heat** | ∂u/∂t | 1st order | Rate of change, relaxation |

**This single difference causes profoundly different behavior!**

### Spatial Derivative (SAME)

Both equations use the **Laplacian** ∇²:

```text
∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
```

**Physical meaning**: Measures local curvature

- Positive curvature (∇²f > 0): Field curves upward, concave
- Negative curvature (∇²f < 0): Field curves downward, convex
- Zero curvature (∇²f = 0): Field is flat (harmonic)

### Dimensional Analysis

**Wave equation**:

```text
[∂²ψ/∂t²] = [c²∇²ψ]
[1/s²] = [m²/s² × 1/m²]
[1/s²] = [1/s²]  ✓ Consistent
```

**Heat equation**:

```text
[∂u/∂t] = [α∇²u]
[K/s] = [m²/s × K/m²]
[K/s] = [K/s]  ✓ Consistent
```

Both are dimensionally consistent, but with different units reflecting different physics.

## Physical Interpretation

### Wave Equation: Propagating Oscillations

**Physical systems**:

- Sound waves (air pressure oscillations)
- Water waves (surface height oscillations)
- Electromagnetic waves (electric/magnetic field oscillations)
- Seismic waves (ground displacement oscillations)
- **Energy waves (EWT)**: Medium displacement oscillations

**Key characteristics**:

- **Propagation**: Disturbances travel at speed c
- **Oscillation**: Solutions oscillate in time
- **Energy transfer**: Energy moves through medium
- **Reversible**: Can reverse time and recover initial state
- **No dissipation**: Energy conserved (lossless)

**Intuition**: "Throw a rock in a pond" → ripples propagate outward

### Heat Equation: Diffusive Smoothing

**Physical systems**:

- Heat conduction (temperature spreads)
- Molecular diffusion (concentration spreads)
- Momentum diffusion (viscosity smooths flow)
- Random walks (probability density spreads)

**Key characteristics**:

- **Diffusion**: Temperature/concentration spreads out
- **Smoothing**: Sharp features blur over time
- **Equilibration**: System approaches uniform state
- **Irreversible**: Cannot reverse time (entropy increases)
- **Dissipation**: Gradients decay, energy spreads

**Intuition**: "Hot coffee cools" → heat diffuses to environment

## Fundamental Differences

### 1. Reversibility vs Irreversibility

**Wave equation** (Reversible):

```text
Time forward:  ψ(t) → ψ(t+dt)
Time backward: ψ(t) → ψ(t-dt)  ✓ Both valid solutions
```

- Replace t with -t: equation unchanged (time-reversal symmetric)
- Can "rewind" the wave and recover initial state
- **Microscopically reversible** physics

**Heat equation** (Irreversible):

```text
Time forward:  u(t) → u(t+dt)  ✓ Valid solution (smooths)
Time backward: u(t) → u(t-dt)  ✗ Invalid (creates sharp features)
```

- Replace t with -t: equation changes sign (time-reversal asymmetric)
- Cannot "unheat" a system to recover initial sharp gradients
- **Thermodynamic arrow of time** (entropy increases)

**Deep connection**: 2nd law of thermodynamics embedded in 1st-order time derivative

### 2. Oscillation vs Decay

**Wave equation** (Oscillatory):

```text
Solutions: ψ(x,t) = A sin(kx - ωt)
           ψ(x,t) = A cos(kx - ωt)

Amplitude remains constant: |ψ| = A (no decay)
Oscillates forever at frequency ω
```

**Heat equation** (Exponential decay):

```text
Solutions: u(x,t) = u₀ e^(-αk²t) sin(kx)

Amplitude decays: u(t) = u₀ e^(-t/τ)
Approaches zero: u(t→∞) → 0
Decay time: τ = 1/(αk²)
```

**Visualization**:

```text
Wave (t=0 to t=∞):
t=0:  ∿∿∿∿∿∿    (amplitude A)
t=1:  ∿∿∿∿∿∿    (amplitude A, shifted)
t=2:  ∿∿∿∿∿∿    (amplitude A, shifted)
→ Propagates unchanged

Heat (t=0 to t=∞):
t=0:  ∿∿∿∿∿∿    (amplitude u₀)
t=1:  ∼∼∼∼∼∼    (amplitude u₀/2, smoother)
t=2:  ~~~~~~    (amplitude u₀/4, very smooth)
→ Decays and smooths
```

### 3. Propagation Speed

**Wave equation** (Finite speed):

```text
Speed = c (constant, independent of wavelength)
Disturbance at x=0 reaches x=L at time t = L/c
Causal: Information cannot travel faster than c
```

**Heat equation** (Infinite speed!)**:

```text
Speed = ∞ (disturbance felt instantly everywhere)
Hot spot at x=0 immediately affects all x (exponentially small but non-zero)
Non-causal: Instantaneous influence (unphysical for fundamental physics)
```

**Why infinite speed?**: First-order time derivative allows instantaneous response

**Physical reality**: Heat equation is **approximation** valid when diffusion slow compared to speed of sound

### 4. Dispersion

**Wave equation** (Non-dispersive):

```text
Dispersion relation: ω = ck (linear)
All wavelengths travel at same speed c
Wave packets maintain shape (no spreading)
```

**Heat equation** (Dispersive decay):

```text
Decay rate: λ(k) = αk² (quadratic)
Short wavelengths decay faster
Wave packets spread and decay
```

### 5. Energy Conservation

**Wave equation** (Energy conserved):

```text
Total energy: E = ∫ [½(∂ψ/∂t)² + ½c²(∇ψ)²] dV = constant

Kinetic + Potential = constant
Energy oscillates between kinetic and potential
No dissipation
```

**Heat equation** (Energy dissipated):

```text
Total "energy" (variance): E = ∫ u² dV = decreasing

Energy spreads from concentrated → uniform
Gradients decay (entropy increases)
Irreversible dissipation
```

## Solution Behavior

### Wave Equation Solutions

**1. Traveling waves**:

```text
ψ(x,t) = f(x - ct) + g(x + ct)

f(x - ct): Wave traveling right at speed c
g(x + ct): Wave traveling left at speed c
```

**Example**: Gaussian pulse

```text
t=0: ψ(x,0) = e^(-x²)    (centered at x=0)
t=1: ψ(x,1) = e^(-(x-c)²)  (moved to x=c, same shape)
```

**2. Standing waves** (superposition of left + right):

```text
ψ(x,t) = A sin(kx) cos(ωt)

Nodes: Points where ψ = 0 always (x = nπ/k)
Antinodes: Points of maximum oscillation
Used in EWT for particle formation!
```

**3. Spherical waves** (3D):

```text
ψ(r,t) = (A/r) sin(kr - ωt)

Amplitude decays as 1/r (energy conservation over expanding sphere)
Used in OpenWave for wave propagation from center
```

### Heat Equation Solutions

**1. Gaussian spreading**:

```text
u(x,t) = (1/√(4παt)) e^(-x²/(4αt))

Width grows as √t (diffusive spreading)
Height decreases as 1/√t (conservation of "mass")
```

**Example**: Initial delta function

```text
t=0:  δ(x)         (infinitely sharp spike)
t=0.1: Narrow Gaussian
t=1:  Wide Gaussian
t→∞: Uniform (flat)
```

**2. Exponential modes**:

```text
u(x,t) = e^(-αk²t) sin(kx)

Each mode decays exponentially
High k (short wavelength) decays fastest
Low k (long wavelength) decays slowest
```

**3. Separation of variables**:

```text
u(x,t) = Σ aₙ e^(-αλₙt) φₙ(x)

Sum of decaying eigenmodes
All modes → 0 as t → ∞
System reaches equilibrium
```

## Energy and Information

### Wave Equation: Energy Transport

**Energy density** (classical wave):

```text
Kinetic: ε_k = ½ρ(∂ψ/∂t)²
Potential: ε_p = ½ρc²(∇ψ)²
Total: ε = ε_k + ε_p
```

**Energy flux** (Poynting-like vector):

```text
S = -ρc²(∂ψ/∂t)∇ψ

Energy flows at speed c
Carries information at speed c
Reversible transport
```

**OpenWave (EWT)**:

```text
Energy density: u = ρ(fA)²  (no ½ factor)
Force: F = -∇E (energy gradient)
Energy conserved during propagation
```

### Heat Equation: Entropy Increase

**Entropy**:

```text
S = -k_B ∫ u ln(u) dx

Entropy always increases: dS/dt ≥ 0
Equilibrium: Maximum entropy (uniform u)
Irreversible process
```

**Information loss**:

```text
Sharp features smooth out (lose fine detail)
Cannot recover initial high-frequency modes
Information flows from system to environment
```

**No energy transport** (just spreading):

```text
Heat spreads from hot → cold
No net energy flow in isolated system
Energy redistributes (doesn't propagate)
```

## Numerical Methods Comparison

### Wave Equation (OpenWave Uses This)

**Finite difference (explicit)**:

```python
# Leapfrog / Verlet scheme (2nd order time)
psi_new[i,j,k] = 2*psi[i,j,k] - psi_old[i,j,k] + c²*dt²*laplacian

# Requires TWO time levels: psi and psi_old
# Stability: CFL condition dt < dx/c
```

**Stability condition**:

```text
CFL: dt ≤ dx/c (Courant-Friedrichs-Lewy)
Timestep limited by wave speed and grid spacing
```

**Properties**:

- Energy-conserving (symplectic)
- Time-reversible
- Stable for dt < dx/c

### Heat Equation

**Finite difference (explicit)**:

```python
# Forward Euler (1st order time)
u_new[i,j,k] = u[i,j,k] + α*dt*laplacian

# Requires ONE time level: u only
# Stability: dt < dx²/(2αd) where d=dimension
```

**Stability condition**:

```text
dt ≤ dx²/(2α × dimensionality)
Much more restrictive! (scales as dx² not dx)
```

**Implicit method** (preferred for heat):

```python
# Backward Euler (unconditionally stable)
(I - α*dt*L) u_new = u

# Requires solving linear system
# Stable for any dt (no CFL limit)
# Used when accuracy > speed
```

**Properties**:

- Dissipative (smooths errors)
- Irreversible
- Can use implicit methods efficiently

### Key Numerical Differences

| Aspect | Wave Equation | Heat Equation |
|--------|---------------|---------------|
| **Time levels** | 2 (ψ, ψ_old) | 1 (u) |
| **Stability (explicit)** | dt < dx/c | dt < dx²/(2α) |
| **Stiffness** | Not stiff | Stiff (small dt) |
| **Energy** | Conserved | Dissipated |
| **Errors** | Oscillatory | Smoothed |
| **Implicit methods** | Complex | Common |
| **Long-time behavior** | Oscillates | Decays to equilibrium |

## Relationship and Connections

### Telegraph Equation: Bridge Between Them

The **telegraph equation** interpolates between wave and heat:

```text
τ ∂²u/∂t² + ∂u/∂t = c² ∇²u
```

**Limits**:

- τ → 0: Reduces to heat equation (diffusion limit)
- ∂u/∂t → 0: Reduces to wave equation (wave limit)

**Physical meaning**: Damped wave equation (wave + friction)

### Wick Rotation: Imaginary Time

**Mathematical connection** via imaginary time:

```text
Wave equation: ∂²ψ/∂t² = c²∇²ψ

Replace t → it (imaginary time):
∂²ψ/∂(it)² = c²∇²ψ
-∂²ψ/∂t² = c²∇²ψ
∂²ψ/∂t² = -c²∇²ψ  (changes sign!)

Further replace ∂/∂t → ∂²/∂t:
∂ψ/∂t = -c²∇²ψ  (heat equation with negative diffusivity)
```

**Significance**: Relates quantum mechanics (wave) to statistical mechanics (diffusion)

### Quantum Mechanics Connection

**Schrödinger equation**:

```text
iℏ ∂ψ/∂t = -ℏ²/(2m) ∇²ψ
```

**Structure**:

- 1st order in time (like heat equation)
- But imaginary coefficient i (makes it wave-like!)
- Complex ψ allows oscillatory solutions
- Unitary evolution (probability conserved)

**Relation**:

- Real time + imaginary i: Schrödinger (wave-like)
- Imaginary time: Diffusion equation (heat-like)
- **EWT wave equation**: Real, 2nd order (classical wave)

### Random Walks and Diffusion

**Heat equation emerges from random walks**:

```text
Particle position: x(t) = random walk
Probability density: u(x,t)

Limit of many small random steps → heat equation!
∂u/∂t = D∇²u (D = diffusion coefficient)
```

**Wave equation**: Does NOT emerge from random walks (needs deterministic rules)

## Relevance to OpenWave

### Why OpenWave Uses Wave Equation

**EWT physics requires**:

1. **Reversible dynamics**: Microscopic physics is time-reversible
2. **Energy conservation**: No dissipation in fundamental medium
3. **Finite wave speed**: Causality (c = speed of light)
4. **Oscillatory behavior**: Standing waves form particles
5. **Propagation**: Energy transport through space

**Heat equation would NOT work**:

- ❌ Irreversible (violates microscopic reversibility)
- ❌ Dissipative (energy lost, not conserved)
- ❌ Infinite speed (violates causality)
- ❌ No oscillations (cannot form standing wave particles)
- ❌ Smooths everything (destroys structure)

### Wave Equation in OpenWave

```python
# Core physics: 2nd order wave equation
@ti.kernel
def propagate_wave():
    for i, j, k in wave_field:
        # Laplacian (spatial curvature)
        laplacian = compute_laplacian(i, j, k)

        # Verlet integration (2nd order time)
        psi_new[i,j,k] = 2*psi[i,j,k] - psi_old[i,j,k] + c²*dt²*laplacian

        # Store for next step
        psi_old[i,j,k] = psi[i,j,k]
        psi[i,j,k] = psi_new[i,j,k]
```

**Key features implemented**:

- ✅ Energy conserved (verified in simulation)
- ✅ Finite propagation speed c
- ✅ Reversible (can reverse time in principle)
- ✅ Supports standing waves (particles)
- ✅ Wave reflection and interference

### When Heat Equation Might Appear

**NOT in core physics**, but potentially in:

1. **Numerical damping** (artificial viscosity for stability):

   ```python
   # Add small dissipation to stabilize numerics
   psi_new += damping * laplacian  # Heat-like term
   ```

2. **Energy dissipation** (if modeling non-ideal medium):

   ```python
   # Add friction/resistance
   ∂²ψ/∂t² + γ ∂ψ/∂t = c²∇²ψ  # Telegraph equation
   ```

3. **Thermal effects** (heat as emergent, not fundamental):

   ```python
   # Temperature from energy density fluctuations
   # But EWT doesn't use heat equation for primary physics
   ```

**In pure EWT**: Heat equation NOT used (wave equation only)

## Summary Table

| Property | Wave Equation | Heat Equation |
|----------|---------------|---------------|
| **Time derivative** | 2nd order (∂²/∂t²) | 1st order (∂/∂t) |
| **Solutions** | Oscillatory, propagating | Decaying, diffusing |
| **Speed** | Finite (c) | Infinite |
| **Reversibility** | Reversible (time-symmetric) | Irreversible (entropy increases) |
| **Energy** | Conserved | Dissipated |
| **Dispersion** | Non-dispersive | Dispersive (k² decay) |
| **Causality** | Causal (c limit) | Non-causal (instant) |
| **Information** | Preserved | Lost (smoothing) |
| **Physical systems** | Waves, oscillations, light | Heat, diffusion, viscosity |
| **Numerical method** | 2 time levels, CFL dt<dx/c | 1 time level, dt<dx²/α |
| **Stability** | CFL-limited | More restrictive (or implicit) |
| **Long-time behavior** | Oscillates forever | Approaches equilibrium |
| **Standing waves** | Yes (superposition) | No (decay) |
| **OpenWave use** | ✅ Core physics | ❌ Not used |

## Key Takeaways

1. **One derivative makes all the difference**: 2nd vs 1st order in time completely changes behavior

2. **Wave = Reversible, Heat = Irreversible**: Wave equation conserves information, heat equation increases entropy

3. **Speed**: Waves propagate at finite c (causal), heat diffuses infinitely fast (acausal approximation)

4. **Energy**: Waves conserve energy (oscillates), heat dissipates energy (smooths)

5. **OpenWave needs waves**: EWT physics requires oscillatory, reversible, energy-conserving behavior → wave equation is correct choice

6. **Heat emerges from waves**: Thermodynamics (heat) emerges from statistical mechanics of many wave/particle interactions, but fundamental physics is wave-based

**The wave equation is fundamental to OpenWave because energy wave theory describes reversible, oscillatory, energy-conserving physics at the microscopic level.**
