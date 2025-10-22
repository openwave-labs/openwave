# Experimental Evaluation of Numerical Methods for Planck-Scale Wave Simulation

**Date:** October 2025

**Project:** OpenWave Energy Wave Dynamics Simulator Development

**Keywords:** Numerical methods, spring-mass dynamics, position-based dynamics, CFL condition, stiffness problems, wave simulation, physics-based simulation

---

## Important Disclaimer

**This is experimental work during simulator development, not peer-reviewed research.** This document describes technical experiments conducted while developing the OpenWave energy wave dynamics simulator, exploring numerical integration challenges when simulating extremely stiff spring-mass systems. The work is based on the Energy Wave Theory (EWT).

This report documents practical rediscovery of known concepts from numerical methods and computational physics (particularly the CFL condition for stiff systems) encountered during simulator development. It is shared to document technical implementation details and lessons learned.

---

## Abstract

This report documents experimental comparisons of numerical methods conducted during development of the OpenWave simulator, focusing on wave propagation in extremely stiff spring-mass lattices inspired by Energy Wave Theory's "aether medium" concept. We implemented and tested four approaches: explicit force-based integrators (semi-implicit Euler and Leapfrog), a constraint-based solver (Extended Position-Based Dynamics/XPBD), and a PSHO method that directly evaluates the wave equation ("phase-synchronized harmonic oscillation").

The experiments confirmed known limitations of explicit integrators for stiff systems: the Courant-Friedrichs-Lewy (CFL) stability condition requires impractically small timesteps for the high spring stiffness values needed to propagate waves at realistic speeds. This is standard knowledge in numerical analysis—the value here is documenting practical experience in a GPU-accelerated simulation context.

XPBD achieved stability with realistic stiffness values but exhibited reduced wave speeds compared to theoretical predictions, likely due to approximations inherent in iterative constraint solvers.

The PSHO approach succeeded: we directly compute particle positions from the analytical wave equation solution. This method achieves "exact" wave propagation by construction.

**Key Lessons:** This experimental work reinforced that numerical method selection must match the problem's mathematical characteristics. For extremely stiff systems, explicit methods fail due to fundamental timestep constraints, constraint-based methods offer stability with accuracy trade-offs, and analytical solutions (when available) bypass integration challenges.

---

## 1. Introduction

### 1.1 Development Context

During development of the OpenWave energy wave dynamics simulator, we encountered the need to evaluate numerical methods for simulating wave propagation in extremely stiff spring-mass systems. The target parameters, inspired by Energy Wave Theory (EWT) [1-3] created a challenging test case for method selection. EWT proposes that a dense "aether medium" made of Planck-scale particles could explain quantum phenomena through classical wave mechanics. EWT provides extreme parameter values useful for stress-testing numerical methods.

The hypothetical parameters from EWT create an unusually stiff numerical scenario:

- Spring stiffness coefficient: k ≈ 5.56×10^44 N/m (at Planck length)
- Wave frequency: f ≈ 1.05×10^25 Hz
- Wave propagation velocity: c = 2.998×10^8 m/s
- Granule mass: m ≈ 2.17×10^-8 kg (Planck mass)

These extreme values proved useful for understanding the practical limits of different numerical integration methods during simulator development.

### 1.2 Experimental Objectives

This experimental evaluation aimed to gain practical experience with:

1. **Stiff system numerics**: Encountering the CFL (Courant-Friedrichs-Lewy) stability condition and observing why explicit integrators fail for stiff systems
2. **GPU-accelerated simulation**: Implementing methods using Taichi Lang for parallel computing
3. **Constraint-based methods**: Applying XPBD (Extended Position-Based Dynamics) from computer graphics literature to an extreme stiffness regime
4. **Trade-offs in method selection**: Comparing accuracy, stability, and performance across different approaches

The challenges encountered are well-documented in numerical analysis textbooks and computational physics literature. This work represents practical rediscovery and demonstration of these concepts through hands-on implementation.

### 1.3 Questions Explored

This experimental project investigated:

1. How do explicit integrators (Euler, Leapfrog) perform with extremely high spring stiffness?
2. What timestep would be needed to maintain stability, and is it computationally feasible?
3. Can XPBD achieve stability while maintaining the target stiffness values?
4. How do wave speeds in XPBD compare to theoretical predictions?
5. Can PSHO methods bypass the integration challenges?

### 1.4 What This Experimental Work Demonstrates

This experimental evaluation during simulator development:

1. **Confirms the CFL condition in practice**: Explicit integrators require impractically small timesteps for stiff systems—a result from numerical analysis, encountered here in a GPU simulation context.

2. **Illustrates the "stiffness problem"**: We visualize the three-way trade-off between stiffness, stability, and visualization timescales (informally termed the "Impossible Triangle"). This is analogous to challenges in molecular dynamics, cloth simulation, and other stiff systems.

3. **Tests XPBD on extreme parameters**: We apply a constraint-based solver from computer graphics (designed for games and real-time applications) to an unusually stiff problem, observing stability but reduced wave speeds—consistent with XPBD being an approximate method.

4. **Implements PSHO wave oscillation**: We develop a PSHO approach that directly evaluates the wave equation, similar to ocean shader techniques in graphics. This achieves "perfect" waves by construction, bypassing integration methods.

**Note**: These are practical rediscoveries of established concepts encountered during development, not novel research contributions. The value is in documenting implementation details and lessons learned for GPU-accelerated simulation using Taichi Lang.

---

## 2. Terminology and Notation

### 2.1 Physical Constants (EWT Parameters)

| Symbol | Description | Value |
|--------|-------------|-------|
| c | Speed of light (wave speed) | 2.998×10^8 m/s |
| λ | Energy Wave length | 2.854×10^-17 m (28.54 am) |
| f | Energy Wave frequency | c/λ ≈ 1.05×10^25 Hz |
| A | Energy Wave amplitude | 9.215×10^-19 m |
| l_p | Planck length | 1.616×10^-35 m |
| m_p | Planck mass | 2.17×10^-8 kg |
| ρ | Medium density | 5.16×10^96 kg/m³ |

### 2.2 Lattice Parameters

| Symbol | Description |
|--------|-------------|
| L | Rest length (BCC nearest neighbor distance) |
| L_0 | Unit cell edge length |
| k | Spring stiffness |
| m | Granule mass (scaled from Planck mass) |
| dt | Timestep (frame or substep) |
| N_sub | Number of substeps per frame |

### 2.3 Key Equations

**Spring Force:**

$$F = -k(x - L)$$

**Natural Frequency:**

$$\omega_n = \sqrt{k/m}$$

$$f_n = \omega_n/(2\pi)$$

**Wave Speed (Spring-Mass Lattice):**

$$v_{wave} = \sqrt{k/m} \times L$$

**Stability Condition (Explicit Methods):**

$$dt < 2/\omega_n = 2/\sqrt{k/m}$$

**XPBD Compliance:**

$$\tilde{\alpha} = 1/(k \cdot dt^2)$$

**Phase-Synchronized Position:**

$$x(t) = x_{eq} + A(r) \cdot \cos(\omega t - kr) \cdot \hat{d}$$

where $k = 2\pi/\lambda$ (wave number), $r$ = radial distance from wave source, $\hat{d}$ = direction unit vector, and $A(r) = A_0 \cdot (r_0/r)$ for spherical wave energy conservation.

**Amplitude Falloff (Spherical Waves):**

$$A(r) = A_0 \cdot \frac{r_0}{r} \quad \text{for } r \geq r_{min}$$

where $r_0$ = reference radius (1 wavelength), $r_{min}$ = minimum safe radius (1 wavelength from wave source).

---

## 3. Background

### 3.1 Energy Wave Theory (EWT)

EWT [1-3] proposes that spacetime emerges from a dense aether medium aether composed of fundamental granules at or near Planck scale. These granules:

- Possess mass (Planck mass corrected by medium density [4])
- Are connected by elastic interactions (quantifiable as spring constant)
- Oscillate harmonically to propagate energy wave
- Form particles through standing wave interference patterns

**Key Physical Relationships:**

From EWT papers [4]:

$$k = (2\pi f_n)^2 \times m \quad \text{(Spring constant from natural frequency)}$$

$$f_n = c/(2L) \quad \text{(Natural frequency for wave speed c)}$$

$$v_{wave} = c \quad \text{(Wave propagation at speed of light)}$$

### 3.2 BCC Lattice Structure

We model the aether medium aether as a Body-Centered Cubic (BCC) lattice:

- **Topology:** 8-way connectivity (each interior granule connected to 8 neighbors)
- **Rest length:** $L = L_0 \cdot \sqrt{3}/2$ (where $L_0$ is unit cell edge)
- **Resolution:** Granules per wavelength = $\lambda_q/(L_0/2)$ (factor 2 accounts for BCC having 2 granules per unit cell)
- **Vertices:** 8 corner granules act as wave sources (harmonic oscillators)

### 3.3 Previous Work

**Classical Numerical Integration:**

- Euler method: 1st-order accuracy, conditionally stable
- Leapfrog (Velocity Verlet): 2nd-order accuracy, symplectic, conditionally stable
- Both methods share same stability limit: $dt < 2/\omega_n$

**Position-Based Dynamics:**

- Müller et al. [5-6] introduced XPBD (Extended Position-Based Dynamics)
- "Small Steps" paper [5] demonstrates many small substeps with single iteration outperforms few substeps with many iterations
- "Unified Particle Physics" [6] provides GPU-friendly Jacobi iteration with constraint averaging

---

## 4. Experimental Methodology

### 4.1 Computational Framework

The experimental investigation was conducted using the OpenWave simulator, a specialized computational framework developed for quantum-scale wave dynamics simulation. The implementation utilizes Python with Taichi Lang GPU acceleration to achieve the computational throughput necessary for large-scale granular simulations.

The simulator architecture comprises five primary computational modules:

- `aether_granule.py`: Implements BCC lattice topology construction and granule initialization procedures
- `energy_wave_springeuler.py`: Provides force-based spring-mass dynamics using semi-implicit Euler integration
- `energy_wave_springleap.py`: Implements symplectic Leapfrog (Velocity Verlet) integration
- `energy_wave_xpbd.py`: Contains the XPBD constraint-based solver implementation
- `energy_wave_radial.py`: Implements phase-synchronized harmonic oscillation methodology

![OpenWave Demo 2](images/demo2.gif)
![OpenWave Demo 3](images/demo3.gif)

### 4.2 Experimental Parameters

To achieve computationally feasible simulations while maintaining physical relevance, we employed a scaled lattice configuration with the following specifications:

**Spatial Scaling:**

- Simulation domain: 1×10^-16 m (100 attometers)
- Scale factor: approximately 10^19 × Planck length
- Lattice configurations: 8³ (1,024 granules), 37³ (101,306 granules), and 79³ (984,064 granules)
- Unit cell dimensions: 1.25×10^-18 m, 2.70×10^-18 m, and 1.27×10^-18 m, respectively

**Temporal Scaling:**

To enable visualization of quantum-scale dynamics, we implemented a temporal scaling factor:

- Slow-motion factor: 1×10^25
- Effective frequency reduction: 1.05×10^25 Hz → ~1 Hz
- Rendering framerate: 30-60 frames per second

### 4.3 Experimental Configurations

#### Configuration A: Force-Based (Euler)

- Integration: Semi-implicit Euler
- Substeps: 30-1000 per frame
- Damping: 0.1-1% per substep
- Stiffness: Variable (1×10^-10 to 1×10^-13 N/m)

#### Configuration B: Force-Based (Leapfrog)

- Integration: Velocity Verlet (symplectic)
- Substeps: 30-1000 per frame
- Damping: 0.1-1% per substep
- Stiffness: Variable (1×10^-10 to 1×10^-13 N/m)

#### Configuration C: XPBD Constraints

- Solver: Jacobi iteration with constraint averaging
- Substeps: 100 per frame, 1 iteration each
- SOR parameter: ω = 1.5
- Damping: 0.999 per substep
- Stiffness: physical value (k ≈ 2.66×10^23 N/m for 79×79×79 grid)

#### Configuration D: Phase-Synchronized Harmonic Oscillators (PSHO)

- Method: Direct harmonic oscillation equation
- No integration (analytical position calculation)
- All granules oscillate radially with phase φ = -kr
- Amplitude falloff: A(r) = A₀(r₀/r) for spherical wave energy conservation
- Minimum safe radius: r_min = 1λ (based on EWT neutrino boundary)
- Stiffness: N/A (no springs, pure wave mechanics)
- All granules oscillate radially along their direction vectors to wave source
- Phase determined by radial distance from wave source, creating outward-propagating spherical wavefronts

---

## 5. Results and Analysis

### 5.1 Force-Based Methods: Stability Analysis and Failure Modes

#### 5.1.1 Theoretical Stability Constraints

The stability analysis for force-based methods reveals fundamental limitations arising from the extreme stiffness requirements. For a representative scaled lattice configuration with spring stiffness $k = 1 \times 10^{-13}$ N/m and granule mass $m = 1.753 \times 10^{-32}$ kg, the system natural frequency is calculated as:

$$\omega_n = \sqrt{k/m} = \sqrt{10^{-13} / 1.753 \times 10^{-32}} = 2.388 \times 10^9 \text{ rad/s}$$

$$f_n = \omega_n/(2\pi) = 3.801 \times 10^8 \text{ Hz}$$

The critical timestep for numerical stability, derived from the Courant-Friedrichs-Lewy condition for explicit integration schemes, yields:

$$dt_{critical} = 2/\omega_n = 8.374 \times 10^{-10} \text{ s}$$

For real-time visualization at 30 frames per second, the frame timestep is:

$$dt_{frame} = 1/30 = 0.0333 \text{ s}$$

Consequently, maintaining numerical stability requires:

$$N_{sub} = dt_{frame} / dt_{critical} \approx 4.0 \times 10^7 \text{ substeps per frame}$$

This requirement exceeds practical computational limits by several orders of magnitude. Experimental validation confirms that even with 1,000 substeps per frame (representing 6,000 iterations per second), numerical instability manifests within 0.4 seconds of simulation time.

#### 5.1.2 Experimental Results Table

| Configuration | k (N/m) | Substeps | Damping | Result | Time to Failure |
|---------------|---------|----------|---------|--------|-----------------|
| Euler | 1×10^-10 | 30 | 0% | NaN | 0.1 s |
| Euler | 1×10^-10 | 200 | 0.1% | NaN | 0.3 s |
| Euler | 1×10^-10 | 1000 | 1% | NaN | 0.5 s |
| Euler | 1×10^-13 | 1000 | 1% | NaN | 0.4 s |
| Leapfrog | 1×10^-13 | 200 | 0.1% | NaN | 0.4 s |
| Leapfrog | 1×10^-13 | 1000 | 1% | NaN | 0.6 s |

**Conclusion:** Force-based methods fundamentally unstable at required stiffness.

![Euler Experiment](images/x_euler.png)
![Leapfrog Experiment](images/x_leap.png)

### 5.2 The Frequency Mismatch Problem: Visualizing the Stiffness Challenge

#### 5.2.1 Understanding the CFL Stability Constraint

Our experiments demonstrate the classic incompatibility between stiff systems and explicit integrators—a well-documented phenomenon in numerical analysis known as the CFL (Courant-Friedrichs-Lewy) condition. The frequency disparity in our setup can be quantified as follows:

- Vertex driving frequency (with temporal scaling): f_drive ≈ 1 Hz
- Spring natural frequency: f_n = 3.8 × 10^8 Hz
- Frequency ratio: approximately 380,000,000:1

This extreme frequency mismatch illustrates what we informally call the "Impossible Triangle" (a visualization of the stiffness problem)—wherein three simulation requirements conflict:

```text
         Realistic Stiffness
                / \
               /   \
              /     \
        Stability --- Human-Visible Motion
```

This is analogous to challenges in molecular dynamics, real-time cloth simulation, and other domains where stiff systems must be visualized. The conflict is standard in numerical methods literature.

#### 5.2.2 Understanding the Three-Way Trade-Off

The three constraints represent competing requirements common to stiff system simulation:

1. **Realistic Stiffness** (k ≈ 10^44 N/m at Planck scale)
   - **Physical requirement**: Fidelity to EWT-specified parameters
   - **Mathematical consequence**: Determines natural frequency ω_n = √(k/m)

2. **Numerical Stability** (dt < 2/ω_n)
   - **Mathematical requirement**: Courant-Friedrichs-Lewy (CFL) condition for convergence
   - **Computational constraint**: Bounds maximum permissible timestep for explicit integration

3. **Human-Visible Motion** (temporal scaling factor = 10^25)
   - **Visualization requirement**: Rendering at 30-60 FPS for observable dynamics
   - **Practical necessity**: Reduces driving frequency from 10^25 Hz to approximately 1 Hz

#### 5.2.3 Causal Analysis of the Frequency Mismatch

The fundamental incompatibility arises through the following causal chain:

```text
Visualization Requirement (human perception constraints)
    ↓
Temporal scaling factor of 10^25 required
    ↓
Frequency mismatch ratio: f_drive ≈ 1 Hz versus f_n = 3.8 × 10^8 Hz
    ↓
Stability criterion demands dt < 10^-10 s (requiring 4 × 10^7 substeps per frame)
    ↓
Mathematical intractability and computational infeasibility
```

#### 5.2.4 Analysis of Vertex Independence

The independence of these constraints merits careful examination:

- **Stiffness versus Stability**: While high stiffness precipitates stability challenges, these represent distinct concepts. Stiffness is a physical parameter governing wave propagation velocity, whereas stability is a mathematical property of the numerical integration scheme. The causal relationship is unidirectional: stiffness affects stability requirements, but they remain conceptually separate.

- **Visualization versus Computational Feasibility**: The visualization requirement constitutes the root cause that necessitates temporal scaling. This scaling subsequently induces the frequency mismatch that renders timesteps computationally infeasible. These are causally linked but represent different aspects of the simulation challenge: perceptual requirements versus computational constraints.

- **Stability versus Feasible Timesteps**: Although both are connected through the CFL condition, stability concerns the mathematical convergence of the integration scheme, while feasible timesteps relate to practical computational limits imposed by the visualization constraint. The distinction is between mathematical necessity and practical possibility.

#### 5.2.5 Experimental Manifestation of Constraints

The constraints manifest experimentally as follows:

1. **Physically realistic stiffness** (k ≈ 10^44 N/m): Results in immediate numerical divergence, with system instability occurring within milliseconds due to violation of stability criteria.

2. **Artificially reduced stiffness** (k ≈ 10^-31 N/m): Achieves numerical stability but yields wave propagation velocities of approximately 10^-24 × c, requiring geological timescales for waves to traverse individual lattice spacings—effectively producing no observable motion.

3. **Temporal scaling for visualization**: Introduces an insurmountable frequency disparity of 360 million to one between the visualization-constrained driving frequency (1 Hz) and the physically required natural frequency (380 MHz).

This frequency gap represents a fundamental mathematical barrier for explicit integration methods, transcending mere computational resource limitations.

#### 5.2.6 Experimental Validation

**Configuration 1: High Stiffness Regime** ($k = 1 \times 10^{-13}$ N/m)

Experimental outcome:

- Numerical divergence occurred after 0.4 seconds of simulation time
- Natural frequency: ω_n = 2.4 × 10^9 rad/s
- Analysis: Despite using 1,000 substeps, the effective timestep remains nine orders of magnitude larger than the stability threshold

**Configuration 2: Frequency-Matched Regime** ($k = 6.9 \times 10^{-31}$ N/m)

Target parameters:

- Matched frequencies: f_n = f_drive = 1 Hz
- Result: Numerical stability achieved, but negligible granule motion observed

Quantitative analysis of motion:

- Spring force for 10 attometer displacement: F = k × Δx = 6.9 × 10^-31 × 10^-17 = 6.9 × 10^-48 N
- Resulting acceleration: a = F/m = 6.9 × 10^-48 / 1.753 × 10^-32 = 3.9 × 10^-16 m/s²
- Position change per frame (dt = 0.033s): Δx = 0.5 × a × dt² = 2.1 × 10^-19 m (0.21 attometers)
- Wave propagation velocity: v = √(k/m) × L = 6.3 × 10^-16 m/s ≈ 2 × 10^-24 × c

The resulting wave speed is twenty-four orders of magnitude below the speed of light, requiring billions of years for waves to traverse a single lattice spacing.

**Learning Point**: These experiments confirm what numerical analysis textbooks teach: explicit integrators cannot handle extremely stiff systems when real-time visualization is required. The 360-million-fold frequency disparity demonstrates the severity of the CFL constraint for this parameter regime.

### 5.3 XPBD: Stability Achieved, But Wave Speed Anomaly

#### 5.3.1 Implementation Success

XPBD achieved numerical stability with physically realistic stiffness values:

Configuration parameters:

- Grid dimensions: 79×79×79 (984,064 granules)
- Spring stiffness: k = 2.66×10^23 N/m (unreduced physical value)
- Substeps: 100 per frame
- Successive over-relaxation parameter: ω = 1.5
- Damping coefficient: 0.999 per substep

Result: Stable wave propagation achieved without numerical divergence.

![XPBD Experiment](images/x_xpbd.png)

#### 5.3.2 Wave Speed Measurements

| Particles | Grid | Resolution (granules/λ) | Measured v | Expected c | Ratio | Error |
|-----------|------|------------------------|------------|------------|-------|-------|
| 1×10³ | 8³ | 4.6 | 1.624×10^7 m/s | 2.998×10^8 m/s | 0.054 | 94.6% |
| 1×10^5 | 37³ | 21.1 | 3.750×10^7 m/s | 2.998×10^8 m/s | 0.125 | 87.5% |

**Analysis:**

Wave propagation velocity improved by a factor of 2.3 with increased spatial resolution. However, at 101,306 particles (21.1 granules per wavelength, exceeding the Nyquist criterion of 10 granules per wavelength), the measured wave speed remained approximately eight-fold below the theoretical value.

**XPBD Compliance Analysis:**

For the 101,306-particle configuration:

- Spring stiffness: k = 2.962×10^23 N/m
- Substep duration: dt_sub = 1.466×10^-4 s (100 substeps at 30 FPS)
- Compliance parameter: α̃ = 1/(k×dt²) = 1.570×10^-16
- Inverse mass: w = 1/m = 1.956×10^31 kg^-1
- Normalized compliance: α̃/(2w) = 4.013×10^-48 << 1

The extremely small normalized compliance value indicates near-rigid constraint enforcement. Nevertheless, the observed wave speed remained significantly below the speed of light.

### 5.4 Phase-Synchronized Harmonic Oscillation

#### 5.4.1 From Force-Simulation to Harmonic Oscillators

The "phase-synchronized" approach eliminates springs and constraints entirely, replacing force-based simulation $(F \rightarrow a \rightarrow v \rightarrow x)$ with direct evaluation of the wave equation:

$$x(t) = x_{eq} + A(r) \cdot \cos(\omega t - kr) \cdot \hat{d}$$

$$v(t) = -A(r) \cdot \omega \cdot \sin(\omega t - kr) \cdot \hat{d}$$

where:

- $\omega = 2\pi f$ (angular frequency)
- $k = 2\pi/\lambda$ (wave number)
- $r$ = radial distance from wave source
- $\hat{d}$ = unit vector from granule to wave source
- $A(r) = A_0 \cdot (r_0/r)$ = distance-dependent amplitude for energy conservation

**Implementation approach:** We directly compute particle positions as functions of time using the analytical wave equation solution. Each particle oscillates radially from a central source with phase $\phi = -kr$ creating the appearance of outward-propagating spherical waves—similar to procedural ocean wave shaders in computer graphics.

**Spherical Wave Energy Conservation (for physical correctness):**

For spherical waves propagating from a point source, total energy must be conserved as the wave expands. The energy of a wave system is given by:

$$E = \rho V \left(\frac{c}{\lambda} \times A\right)^2$$

As a spherical wave propagates through a uniform medium, wavelength $\lambda$ and wave speed $c$ remain constant, requiring amplitude $A$ to decrease with distance. Energy density integrated over expanding spherical shells must remain constant:

$$E_{total} = \int A^2(r) \times 4\pi r^2 \, dr = \text{constant}$$

This requires $A^2(r) \times r^2 = \text{constant}$, yielding the amplitude falloff law:

$$A(r) = A_0 \cdot \frac{r_0}{r}$$

where $r_0$ is the reference radius (set to one wavelength). This ensures energy conservation across all radial distances from the wave source.

**Near-Field vs Far-Field Regions:**

Based on electromagnetic wave theory and EWT specifications, the region around the wave source divides into three zones:

1. **Near field** ($r < \lambda$ from wave source): Source region where wave structure is forming. Amplitude is clamped to prevent singularity at $r \to 0$.

2. **Transition zone** ($\lambda < r < 2\lambda$ from wave source): Wave fronts organize into spherical geometry, transitioning toward far-field behavior.

3. **Far field** ($r > 2\lambda$ from wave source): Fully formed spherical waves with clean $A \propto 1/r$ energy conservation law.

The implementation uses $r_{min} = 1\lambda$ (one wavelength from wave source) as the minimum safe radius, based on:

- EWT neutrino boundary specification at $r = 1\lambda$
- EM theory transition to radiative fields around $\lambda$
- Prevention of singularity at $r \to 0$
- Numerical stability and physical wave behavior

#### 5.4.2 Implementation Detail: Separating Temporal and Spatial Phase Terms

A design choice in our implementation is maintaining **separate, independent factors** for temporal oscillation and spatial phase:

$$x(t) = x_{eq} + A \cdot \cos(\omega t + \phi) \cdot \hat{d}$$

where $\omega t$ (temporal) and $\phi$ (spatial) remain distinct, rather than collapsing them into a single term like $\cos(kr - \omega t)$.

**Rationale for Separation:**

1. **Conceptual Clarity**
   - $\omega t$ = temporal oscillation (controls rhythm, time-varying component)
   - $\phi = -kr$ = spatial phase shift (controls phase relationships, interference, position-dependent component)

   This separation makes explicit that we control two independent aspects of wave behavior.

2. **EWT Alignment**

   Energy Wave Theory is fundamentally about **phase relationships** between waves:
   - Particle formation via constructive/destructive interference = phase relationships
   - Standing waves = specific phase patterns
   - Wave centers creating particles = phase synchronization

   Keeping $\phi$ as a first-class, independent parameter makes this physics visible in the code and allows direct manipulation of the core mechanism in EWT.

3. **Future Flexibility**

   With separate factors, we can independently control and manipulate phase without affecting frequency:

   ```python
   # Current: Simple radial phase
   φ = -k * r

   # Future: Multi-source interference
   φ = calculate_interference_phase(r, source_array)

   # Future: Particle-induced phase shifts
   φ = apply_phase_shift_from_particle(r, particle_position)

   # Future: Standing wave patterns
   φ = create_standing_wave_phase(r, boundary_conditions)
   ```

   This design enables future implementation of wave interactions, interference patterns, and particle formation without restructuring the fundamental wave equation.

4. **Standard Physics Convention**

   The separated form $x(t) = A \cdot \cos(\omega t + \phi_0)$ is actually standard in physics, where $\phi_0$ represents initial or spatial phase. Our implementation extends this by making $\phi$ spatially-varying based on position.

**Why This Design Choice:**

Keeping phase as a separate, first-class parameter makes the code more flexible for future extensions (multiple wave sources, interference patterns, standing waves). It also makes the physics more explicit in the code structure, which is helpful for learning and experimentation.

#### 5.4.3 Implementation (ewave_radial.py)

```python
@ti.kernel
def oscillate_granules(
    positions, velocities, equilibrium, directions,
    radial_distances, t, slow_mo, freq_boost, amp_boost
):
    f_slowed = frequency / slow_mo * freq_boost
    omega = 2.0 * ti.math.pi * f_slowed
    k = 2.0 * ti.math.pi / wavelength_am  # Wave number

    # Reference radius for amplitude normalization (one wavelength from wave source)
    r_reference = wavelength_am  # attometers

    for idx in position:
        direction = directions[idx]
        r = radial_distances[idx]  # distance from wave source
        phase = -k * r  # Outward propagating wave

        # Amplitude falloff for spherical wave energy conservation: A(r) = A₀(r₀/r)
        # Uses r_min = 1λ based on EWT neutrino boundary and EM near-field physics
        r_safe = ti.max(r, r_reference * 1)  # minimum 1 wavelength from wave source
        amplitude_falloff = r_reference / r_safe

        # Total amplitude at distance r from wave source
        amplitude_at_r = amplitude_am * amplitude_falloff * amp_boost

        # Direct position calculation (no integration!)
        displacement = amplitude_at_r * ti.cos(omega * t + phase)
        position[idx] = equilibrium[idx] + displacement * direction

        # Velocity from derivative
        velocity_mag = -amplitude_at_r * omega * ti.sin(omega * t + phase)
        velocity[idx] = velocity_mag * direction
```

![Radial Wave Experiment](images/x_wave.png)

#### 5.4.4 Results from PSHO Approach

**Visual Observations:**

- Clean spherical wavefronts propagating outward from center
- Wavelength λ = 2π/k by construction (we set k directly)
- Frequency f by construction (we set ω directly)
- No numerical artifacts or instabilities

**Wave Speed:**

$$\text{By construction: } v = f \times \lambda = (c/\lambda_q) \times \lambda_q = c \quad \checkmark$$

**Wavelength:**

$$\text{By construction: } k = 2\pi/\lambda \rightarrow \lambda = 2\pi/k = \lambda_q \quad \checkmark$$

Again, exact by construction—we define k in the code.

**Stability:**

- Unconditionally stable
- Runs indefinitely without issues

**Practical Advantages of This Approach:**

1. **Perfect wave propagation** - by construction
2. **Exact wavelength** - by construction
3. **Energy conservation** - Amplitude falloff A ∝ 1/r implemented directly
4. **Unconditional stability** - No CFL constraint
5. **Computational efficiency** - Just trigonometric evaluations, very fast
6. **Simple implementation** - Clean, readable code

#### 5.4.5 Comparison Summary

| Method | Wave Speed | Wavelength | Stability | Realistic k? | Notes |
|--------|-----------|------------|-----------|--------------|-------|
| Euler | N/A (crashes) | N/A | Unstable | No (10^-10×) | CFL violation |
| Leapfrog | N/A (crashes) | N/A | Unstable | No (10^-10×) | CFL violation |
| XPBD | 0.125c (at 1e5) | Not measured | Stable | Yes | Approx. solver |
| PSHO | c (exact) | λ (exact) | Unconditional | N/A | Expected |

---

## 6. Discussion

### 6.1 Understanding Explicit Integrator Limitations for Stiff Systems

This project demonstrates the limitations of explicit numerical methods when applied to extremely stiff systems. The CFL stability condition creates a direct relationship between stiffness and required timestep:

$$\text{High frequency} \rightarrow \text{Extreme stiffness} \rightarrow \text{Prohibitive timestep constraints}$$

For the EWT parameters, the wave frequency is:

$$f = 1.05 \times 10^{25} \text{ Hz}$$

This yields a spring stiffness coefficient of:

$$k = (2\pi f)^2 \times m \approx 5.56 \times 10^{44} \text{ N/m}$$

The CFL-limited timestep becomes:

$$dt_{max} < 2/\omega = 2/(2\pi f) \approx 3 \times 10^{-26} \text{ s}$$

Consequently, simulating one second of physical time would require:

$$N_{steps} > 1 / (3 \times 10^{-26}) = 3.3 \times 10^{25} \text{ integration steps}$$

This is a classic stiffness problem. In real computational physics, this situation is typically addressed using:

- **Implicit integrators** (e.g., Backward Euler, BDF methods) that are unconditionally stable
- **Specialized solvers** for the specific physics (e.g., spectral methods for wave equations)
- **Reduced models** or coarse-graining approaches
- **Analytical solutions** where available

### 6.2 Why XPBD Shows Reduced Wave Speed

XPBD achieves stability through **compliance parameter** $\tilde{\alpha}$:

$$\Delta\lambda = -C / (w_i + w_j + \tilde{\alpha})$$

where $\tilde{\alpha} = 1/(k \cdot dt^2)$

Even with $\tilde{\alpha}$ extremely small ($\tilde{\alpha}/(2w) \approx 10^{-47}$), the iterative constraint satisfaction process may introduce **effective damping** or **phase lag** that slows wave propagation.

**Hypothesis:** XPBD's Jacobi iteration with constraint averaging distributes corrections differently than instantaneous spring forces, creating **dispersion** (frequency-dependent wave speed).

At low resolution (4.6 granules/λ), this effect is severe (5% of c). At higher resolution (21 granules/λ), improves to 12.5% of c, but gap remains.

### 6.3 Why PSHO Bypasses Integration Challenges

We directly evaluate the analytical wave equation solution.

**Force-Based Approach (What We Tried to Simulate):**

The classical simulation chain: Forces → Acceleration → Velocity → Position (F → a → v → x)

This sequential integration process:

- Accumulates numerical error at each step
- Requires restrictive timestep constraints (CFL condition)
- Fails at extreme stiffness values (as we demonstrated)

**PSHO Approach (What Actually Worked):**

Direct evaluation: Wave Equation → Position and Velocity (Analytical Solution)

This approach:

- Bypasses numerical integration entirely
- Has no stability constraints
- Achieves "exact" results

This is analogous to techniques used in computer graphics:

- Ocean wave shaders (Tessendorf 2001) directly evaluate Fourier wave sums
- Procedural animation of cloth or hair using analytical curves
- Any "physics-based" visual effect that directly evaluates equations

The "phase-synchronized" method is essentially the same idea applied to a spring-mass lattice.

### 6.4 Reflections on Method Selection

This project reinforced several lessons from computational physics and graphics:

1. **Match method to problem**: Explicit integrators are great for many problems but fundamentally unsuitable for extremely stiff systems without massive computational resources
2. **Understand trade-offs**: XPBD trades accuracy for stability and speed (appropriate for games, questionable for physics validation)
3. **Analytical solutions are best when available**: If you know the answer (wave equation), directly evaluating it beats integration methods
4. **Missing methods matter**: We didn't test implicit integrators, spectral methods, or FDTD—all standard approaches for wave problems in computational physics

---

## 7. Limitations and Future Work

### 7.1 Current Limitations

1. **Phase-synchronized method** limited to simple wave patterns (spherical propagation from center)
2. **Cannot model** complex interactions like particle formation (requires interference, nonlinearity)
3. **XPBD wave speed anomaly** not fully explained (requires deeper investigation)
4. **Wavelength measurement** O(N²) complexity prevents validation at high resolution

### 7.2 Future Directions

#### 7.2.1 Hybrid Approaches

Combine phase synchronization with constraint-based methods:

- Use phase-synchronized oscillation for wave injection
- Use XPBD for particle interactions and nonlinear effects
- Investigate if XPBD wave speed approaches c with higher substeps

#### 7.2.2 Alternative Wave Solvers

- **Spectral methods** (FFT-based): Solve wave equation in frequency domain
- **Finite difference** (FDTD): Direct discretization of wave PDE
- **Lattice Boltzmann**: Mesoscopic approach for wave propagation

#### 7.2.3 Adaptive Resolution

- Coarse grid in free-space regions
- Fine grid near particles/interactions
- Multi-scale methods for bridging Planck to macroscopic scales

#### 7.2.4 GPU Optimization

- Investigate Taichi GPU performance for 10^9+ granules
- Spatial hashing for wavelength measurement at scale
- Memory-efficient data structures for petascale simulation

---

## 8. Conclusions and Lessons Learned

This experimental evaluation during OpenWave simulator development explored numerical methods for simulating wave propagation in extremely stiff spring-mass lattices, with parameters inspired by the Energy Wave Theory framework. The experiments confirmed principles from numerical analysis and computational physics:

1. **Explicit integrators and the CFL condition**: Semi-implicit Euler and symplectic Leapfrog methods both failed at high stiffness values, as expected from the CFL stability condition. The timestep required for stability (~10^-26 seconds for hypothetical Planck-scale parameters) would demand 10^25 steps per second—confirming textbook predictions about stiff system behavior. Our informal "Impossible Triangle" visualization helped us understand the three-way conflict between stiffness, stability, and visualization timescales—a challenge well-documented in molecular dynamics, cloth simulation, and other stiff-system domains.

2. **XPBD as a practical compromise**: The Extended Position-Based Dynamics solver (borrowed from computer graphics literature) achieved stability at target stiffness values but exhibited reduced wave speeds (12.5-5% of theoretical). This is consistent with XPBD being an approximate, compliance-based method designed for speed and stability in real-time applications (games), not physical accuracy. The wave speed reduction likely stems from dispersion in the iterative constraint solver—a characteristic of position-based methods.

**Practical Outcomes:**

This experimental work successfully:

- Implemented GPU-accelerated physics simulation using Taichi Lang
- Gained practical experience with the CFL stability constraint in stiff systems
- Compared explicit integrators, constraint solvers, and PSHO methods hands-on
- Demonstrated the importance of matching numerical methods to problem characteristics
- Showed why analytical solutions (when available) outperform integration methods

The progression through different methods during development:

- Force-based integration (Euler): Encountered CFL stability barrier ✓
- Symplectic integration (Leapfrog): Same CFL limitation ✓
- Constraint-based dynamics (XPBD): Stability with accuracy trade-offs ✓
- PSHO approach: Bypassed integration methods ✓

This experimental evaluation served its purpose during OpenWave development, providing practical insights into GPU computing, numerical methods, and physics-based simulation. The documented code and experimental results may be useful for others working with Taichi Lang or exploring similar challenges in stiff-system simulation.

---

## Acknowledgments

This experimental work was conducted using the OpenWave simulator (available at <https://github.com/openwave-labs/openwave>), implemented with the Taichi Lang GPU acceleration framework. Assistance from Claude AI (Anthropic) throughout this development process include code development, debugging, experimental design, data analysis, interpretation of results, and document preparation.

Special thanks to the computer graphics and computational physics communities whose published work informed this experimental evaluation:

- Matthias Müller and colleagues for their seminal work on Position-Based Dynamics and XPBD [5-6, 8], which provided clear documentation for implementing constraint-based solvers
- Miles Macklin et al. for "Unified Particle Physics for Real-Time Applications" [7]
- Robert Bridson et al. for foundational work on numerical simulation [8]
- Jerry Tessendorf for ocean wave simulation techniques that inspired the PSHO approach
- The Taichi Lang development team for creating an accessible GPU computing framework

This is experimental work during simulator development, not peer-reviewed research, and should be evaluated as such.

---

## References

[1] Yee, J. (2019). "The Geometry of Spacetime and the Unification of the Electromagnetic, Gravitational and Strong Forces." Energy Wave Theory. <https://www.researchgate.net/publication/334316805>

[2] Yee, J. (2020). "The Physics of Subatomic Particles and their Behavior Modeled with Classical Laws." Energy Wave Theory. <https://www.researchgate.net/publication/338634046>

[3] Yee, J. (2019). "The Geometry of Particles and the Explanation of their Creation and Decay." Energy Wave Theory. <https://www.researchgate.net/publication/335101008>

[4] Yee, J. "Relationship of the Speed of Light to Aether Density." Energy Wave Theory. (Contains Planck mass correction affecting granule mass calculations)

[5] Müller, M., et al. (2020). "Small Steps in Physics Simulation." SCA '20: ACM SIGGRAPH/Eurographics Symposium on Computer Animation. <https://matthias-research.github.io/pages/publications/smallsteps.pdf>

[6] Macklin, M., et al. (2014). "Unified Particle Physics for Real-Time Applications." ACM Transactions on Graphics (TOG), 33(4).

[7] Bridson, R., et al. (2002). "Robust Treatment of Collisions, Contact and Friction for Cloth Animation." ACM SIGGRAPH 2002.

[8] Müller, M., et al. (2007). "Position Based Dynamics." Journal of Visual Communication and Image Representation, 18(2), 109-118.

---

## Appendix A: Code Snippets

### A.1 Force-Based Spring-Mass (Euler) - UNSTABLE

```python
@ti.kernel
def compute_spring_forces(position, equilibrium, forces, links,
                          links_count, rest_length, stiffness):
    for i in range(position.shape[0]):
        force = ti.Vector([0.0, 0.0, 0.0])
        for j in range(links_count[i]):
            neighbor = links[i, j]

            # Spring force: F = -k(x - L0)
            d = position[neighbor] - position[i]
            distance = d.norm()
            displacement = distance - rest_length
            force_mag = -stiffness * displacement
            force += (force_mag / distance) * d

        forces[i] = force

@ti.kernel
def integrate_euler(position, velocity, forces, mass, dt, damping):
    for i in range(position.shape[0]):
        a = forces[i] / mass
        velocity[i] += a * dt
        velocity[i] *= damping
        position[i] += velocity[i] * dt
```

### A.2 XPBD Constraint Solver - STABLE

```python
@ti.kernel
def solve_distance_constraints(position, neighbors, masses,
                                rest_length, compliance, dt, omega):
    # Phase 1: Accumulate position deltas (Jacobi iteration)
    for i in range(position.shape[0]):
        delta[i] = ti.Vector([0.0, 0.0, 0.0])
        count[i] = 0

        for j in range(8):  # 8 BCC neighbors
            neighbor = neighbors.links[i, j]

            # Constraint: C = ||xi - xj|| - L0
            d = position[neighbor] - position[i]
            distance = d.norm()
            C = distance - rest_length

            # Gradient & Lagrange multiplier
            grad = d / distance
            alpha_tilde = 1.0 / (stiffness * dt * dt)
            w_sum = (1/masses[i]) + (1/masses[neighbor])
            delta_lambda = -C / (w_sum + alpha_tilde)

            # Position correction
            delta_xi = -(1/masses[i]) * grad * delta_lambda
            delta[i] += delta_xi
            count[i] += 1

    # Phase 2: Apply with SOR and constraint averaging
    for i in range(position.shape[0]):
        if count[i] > 0:
            position[i] += (omega / count[i]) * delta[i]
```

### A.3 Phase-Synchronized Harmonic - PERFECT

```python
@ti.kernel
def oscillate_granules(position, velocity, equilibrium, direction,
                       radial_distance, t, slow_mo, freq_boost, amp_boost):
    """Phase-synchronized harmonic oscillation (ewave_radial.py)"""
    f_slowed = frequency / slow_mo * freq_boost
    omega = 2.0 * ti.math.pi * f_slowed
    k = 2.0 * ti.math.pi / wavelength_am  # Wave number

    # Reference radius for amplitude normalization (one wavelength from wave source)
    r_reference = wavelength_am  # attometers

    for idx in range(position.shape[0]):
        direction = directions[idx]
        r = radial_distances[idx]  # distance from wave source
        phase = -k * r  # Outward propagating wave

        # Amplitude falloff for spherical wave energy conservation: A(r) = A₀(r₀/r)
        # Uses r_min = 1λ based on EWT neutrino boundary and EM near-field physics
        r_safe = ti.max(r, r_reference * 1)  # minimum 1 wavelength from wave source
        amplitude_falloff = r_reference / r_safe

        # Total amplitude at distance r from wave source
        amplitude_at_r = amplitude_am * amplitude_falloff * amp_boost

        # Direct analytical solution (no integration!)
        displacement = amplitude_at_r * ti.cos(omega * t + phase)
        position[idx] = equilibrium[idx] + displacement * direction

        velocity_magnitude = -amplitude_at_r * omega * ti.sin(omega * t + phase)
        velocity[idx] = velocity_magnitude * direction
```

---

## Appendix B: Experimental Data

### B.1 Force-Based Stability Tests

| Test | k (N/m) | m (kg) | ω_n (rad/s) | dt_critical (s) | dt_sub (s) | N_sub | Result |
|------|---------|--------|-------------|-----------------|------------|-------|--------|
| 1 | 1e-10 | 1.753e-32 | 2.39e9 | 8.37e-10 | 1.11e-3 | 30 | NaN at 0.1s |
| 2 | 1e-10 | 1.753e-32 | 2.39e9 | 8.37e-10 | 1.67e-4 | 200 | NaN at 0.3s |
| 3 | 1e-13 | 1.753e-32 | 2.39e9 | 8.37e-10 | 3.33e-5 | 1000 | NaN at 0.4s |
| 4 | 6.9e-31 | 1.753e-32 | 6.28 | 0.318 | 1.11e-3 | 30 | Stable, no motion |

### B.2 XPBD Wave Speed Measurements

| Particles | Grid | L (am) | k (N/m) | Resolution | v_measured | v_expected | Error |
|-----------|------|--------|---------|------------|------------|------------|-------|
| 1e3 | 8³ | 10.83 | 2.71e23 | 4.6 | 1.624e7 m/s | 2.998e8 m/s | 94.6% |
| 1e5 | 37³ | 2.34 | 2.96e23 | 21.1 | 3.750e7 m/s | 2.998e8 m/s | 87.5% |

---

End of Report
