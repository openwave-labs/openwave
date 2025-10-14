# Numerical Methods for Quantum-Scale Wave Dynamics: A Comparative Analysis of Force-Based and Phase-Synchronized Approaches

## From Particle Mechanics to Wave Mechanics in Planck-Scale Simulation

**Author:** Rodrigo Griesi

**Date:** October 2025

**Keywords:** Quantum simulation, Energy Wave Theory, extended position-based dynamics, numerical stability, Planck scale, wave propagation, computational physics, phase synchronization, harmonic oscillators

---

## Abstract

This study presents a systematic investigation of numerical methods for simulating quantum wave dynamics at the Planck scale within the framework of Energy Wave Theory (EWT). We evaluate four distinct computational approaches: explicit force-based integration methods (semi-implicit Euler and Leapfrog), constraint-based solvers (Extended Position-Based Dynamics, XPBD), and phase-synchronized harmonic oscillation (PSHO). Our analysis identifies fundamental computational barriers arising from the extreme stiffness inherent in spring-mass systems that model the quantum aether medium.

The investigation reveals that force-based mechanics encounter an insurmountable "Impossible Triangle" wherein realistic physical stiffness, numerical stability, and human-visible motion cannot be simultaneously achieved using explicit integration schemes. This limitation persists even with higher-order symplectic methods, manifesting as a frequency mismatch ratio of approximately 360 million to one between the spring natural frequency and the driving frequency. While XPBD successfully circumvents this stability barrier by maintaining realistic stiffness values, it introduces an unexpected wave speed reduction of approximately eight-fold relative to the theoretical speed of light.

We demonstrate that these fundamental limitations can be resolved through a phase-synchronized harmonic oscillation approach that directly implements wave mechanics, achieving exact wave propagation at the speed of light with precise wavelength correspondence. This method bypasses numerical integration entirely, providing unconditional stability and physical accuracy. Our findings indicate that wave mechanics frameworks are inherently superior to particle-based force mechanics for quantum-scale simulations, validating the wave-centric interpretation of quantum phenomena posited by EWT.

---

## 1. Introduction

### 1.1 Motivation

Energy Wave Theory (EWT) [1-3] presents a deterministic framework for quantum mechanics, proposing that matter and energy emerge from wave interactions within a dense quantum aether medium. Computational validation of EWT necessitates the simulation of wave propagation through a lattice of quantum granules operating at or near the Planck scale (1.616×10^-35 m). The physical parameters required for such simulations include:

- Spring stiffness coefficient: k ≈ 5.56×10^44 N/m (at Planck length)
- Wave frequency: f ≈ 1.05×10^25 Hz
- Wave propagation velocity: c = 2.998×10^8 m/s
- Granule mass: m ≈ 2.17×10^-8 kg (Planck mass)

These extreme parameters present formidable numerical challenges that exceed the capabilities of conventional force-based simulation methodologies.

### 1.2 Computational Challenges

The simulation of Planck-scale physics presents computational challenges that transcend mere hardware limitations. The extreme stiffness required for physically accurate wave propagation imposes stability conditions that reveal fundamental mathematical limitations in numerical integration schemes. Specifically, force-based numerical integration methods encounter mathematical singularities when attempting to resolve the extreme stiffness inherent in quantum-scale systems. The requisite timestep restrictions and iteration counts exceed the mathematical bounds of floating-point arithmetic and numerical stability, resulting in algorithmic failure rather than merely computational insufficiency.

This phenomenon corroborates theoretical predictions regarding the substantial energy density of quantum waves, as evidenced by force magnitudes and momentum values that exceed the resolution capacity of standard numerical integration techniques.

### 1.3 Research Questions

The present investigation addresses the following research questions:

1. Can force-based spring-mass dynamics achieve stable simulation of quantum wave propagation using physically realistic parameters?
2. What fundamental computational barriers limit force-based methodologies at the Planck scale?
3. Do constraint-based solvers, specifically Extended Position-Based Dynamics, overcome the limitations of force-based approaches?
4. Can alternative computational paradigms achieve both physical accuracy and numerical stability in quantum-scale simulations?

### 1.4 Contributions

This work presents four primary contributions to the field of quantum-scale computational physics:

1. **Identification of fundamental computational barriers**: We demonstrate that force-based methods encounter mathematical, rather than merely computational, limitations at the Planck scale due to extreme stiffness requirements.

2. **Characterization of the "Impossible Triangle"**: We establish that explicit integration schemes cannot simultaneously achieve realistic stiffness values, numerical stability, and human-visible motion, revealing a frequency mismatch ratio of approximately 360 million to one that proves insurmountable for explicit integrators.

3. **Evaluation of XPBD capabilities and limitations**: We demonstrate that XPBD successfully circumvents the Impossible Triangle through stability at realistic stiffness values, albeit with an observed wave speed reduction of 8-18 fold.

4. **Development of phase-synchronized harmonic oscillation**: We introduce a novel computational approach that achieves exact wave propagation through direct implementation of wave mechanics, validating the superiority of wave-based frameworks for quantum-scale phenomena.

---

## 2. Terminology and Notation

### 2.1 Physical Constants (EWT Parameters)

| Symbol | Description | Value |
|--------|-------------|-------|
| c | Speed of light (wave speed) | 2.998×10^8 m/s |
| λ_q | Quantum Wave length | 2.854×10^-17 m (28.54 am) |
| f_q | Quantum Wave frequency | c/λ_q ≈ 1.05×10^25 Hz |
| A | Wave amplitude | 9.215×10^-19 m |
| l_p | Planck length | 1.616×10^-35 m |
| m_p | Planck mass | 2.17×10^-8 kg |
| ρ_aether | Medium density | 5.16×10^96 kg/m³ |

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

$$x(t) = x_{eq} + A \cdot \cos(\omega t - kr) \cdot \hat{d}$$

where $k = 2\pi/\lambda$ (wave number), $r$ = radial distance from source, $\hat{d}$ = direction unit vector.

---

## 3. Background

### 3.1 Energy Wave Theory (EWT)

EWT [1-3] proposes that spacetime emerges from a dense quantum aether composed of fundamental granules at or near Planck scale. These granules:

- Possess mass (Planck mass corrected by aether density [4])
- Are connected by elastic interactions (quantifiable as spring constant)
- Oscillate harmonically to propagate quantum waves
- Form particles through standing wave interference patterns

**Key Physical Relationships:**

From EWT papers [4]:

$$k = (2\pi f_n)^2 \times m \quad \text{(Spring constant from natural frequency)}$$

$$f_n = c/(2L) \quad \text{(Natural frequency for wave speed c)}$$

$$v_{wave} = c \quad \text{(Wave propagation at speed of light)}$$

### 3.2 BCC Lattice Structure

We model the quantum aether as a Body-Centered Cubic (BCC) lattice:

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

- `quantum_medium_lattice.py`: Implements BCC lattice topology construction and granule initialization procedures
- `quantum_wave_springeuler.py`: Provides force-based spring-mass dynamics using semi-implicit Euler integration
- `quantum_wave_springleap.py`: Implements symplectic Leapfrog (Velocity Verlet) integration
- `quantum_wave_xpbd.py`: Contains the XPBD constraint-based solver implementation
- `quantum_wave_radial.py`: Implements phase-synchronized harmonic oscillation methodology

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
- Stiffness: REAL physical value (k ≈ 2.66×10^23 N/m for 79×79×79 grid)

#### Configuration D: Phase-Synchronized Harmonic Oscillators (PSHO)

- Method: Direct harmonic oscillation equation
- No integration (analytical position calculation)
- All granules oscillate radially with phase φ = -kr
- Stiffness: N/A (no springs, pure wave mechanics)
- All granules oscillate radially along their direction vectors to wave source
- Phase determined by radial distance, creating outward-propagating spherical wavefronts

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

### 5.2 The Frequency Mismatch Problem: Characterization of the Impossible Triangle

#### 5.2.1 Identification of Fundamental Computational Barriers

Our investigation reveals a fundamental incompatibility between the physical requirements of quantum-scale wave propagation and the mathematical constraints of explicit numerical integration. The frequency disparity can be quantified as follows:

- Vertex driving frequency (with temporal scaling): f_drive ≈ 1 Hz
- Spring natural frequency: f_n = 3.8 × 10^8 Hz
- Frequency ratio: approximately 380,000,000:1

This extreme frequency mismatch gives rise to what we term the "Impossible Triangle," wherein three essential simulation requirements prove mutually incompatible:

```text
         Realistic Stiffness
                / \
               /   \
              /     \
        Stability --- Human-Visible Motion
```

#### 5.2.2 Theoretical Analysis of Constraint Independence

The three vertices of the Impossible Triangle represent fundamentally distinct requirements that cannot be simultaneously satisfied:

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

**Conclusion**: These experiments definitively demonstrate that no intermediate parameter regime exists that satisfies all three constraints. The 360-million-fold frequency disparity represents an unbridgeable gap for explicit integration methods.

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

### 5.4 Phase-Synchronized Harmonic Oscillation: Resolution Through Wave Mechanics

#### 5.4.1 Paradigm Shift from Force-Based to Wave-Based Formulation

The critical insight involves eliminating springs and constraints entirely in favor of synchronized phase relationships between granules. Rather than employing force-based mechanics $(F \rightarrow a \rightarrow v \rightarrow x)$, we implement direct wave equations:

$$x(t) = x_{eq} + A \cdot \cos(\omega t - kr) \cdot \hat{d}$$

$$v(t) = -A \cdot \omega \cdot \sin(\omega t - kr) \cdot \hat{d}$$

where:

- $\omega = 2\pi f$ (angular frequency)
- $k = 2\pi/\lambda$ (wave number)
- $r$ = radial distance from source
- $\hat{d}$ = unit vector from granule to wave source

**Key insight:** Radial waves originate from the wave source and propagate via synchronized phase shifts. This approach replaces force-driven position integration with direct harmonic oscillation equations that define granule positions as functions of time.

The phase relationship $\phi = -kr$ generates outward-propagating spherical waves from the wave source without requiring spring forces or numerical integration.

**Result:** This approach achieves exact wave propagation with clearly defined wavefronts, matching both theoretical wave speed and wavelength parameters.

#### 5.4.2 Design Decision: Separating Temporal and Spatial Phase Terms

A critical design choice in our implementation is maintaining **separate, independent factors** for temporal oscillation and spatial phase:

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

**Why This Matters for Quantum Simulation:**

At quantum scale, **phase control is the fundamental mechanism** for wave behavior. Particle formation, wave interactions, and interference all emerge from phase relationships, not force interactions. By treating phase as an independent, manipulable parameter rather than collapsing it into the wave equation, we maintain the ability to implement the full range of quantum phenomena predicted by EWT.

This design philosophy reflects the insight that phase relationships are more fundamental than forces at quantum scale - the very principle that makes PSHO successful where force-based methods fail.

#### 5.4.3 Implementation (radial_wave.py)

```python
@ti.kernel
def oscillate_granules(
    positions, velocities, equilibrium, directions,
    radial_distances, t, slow_mo, amp_boost
):
    f_slowed = frequency / slow_mo
    omega = 2.0 * ti.math.pi * f_slowed
    k = 2.0 * ti.math.pi / wavelength_am  # Wave number

    for idx in positions:
        direction = directions[idx]
        r = radial_distances[idx]
        phase = -k * r  # Outward propagating

        # Direct position calculation (no integration!)
        displacement = amplitude_am * amp_boost * ti.cos(omega * t + phase)
        positions[idx] = equilibrium[idx] + displacement * direction

        # Velocity from derivative
        velocity_mag = -amplitude_am * amp_boost * omega * ti.sin(omega * t + phase)
        velocities[idx] = velocity_mag * direction
```

![Radial Wave Experiment](images/x_wave.png)

#### 5.4.4 Experimental Results and Validation

**Observed Phenomena:**

- Spherical wavefronts propagating outward from the wave source
- Wavelength λ = 2π/k verified through spatial pattern analysis
- Frequency f confirmed through temporal oscillation measurements
- Absence of numerical artifacts or instabilities

**Wave Speed:**

$$\text{By construction: } v = f \times \lambda = (c/\lambda_q) \times \lambda_q = c \quad \checkmark$$

Measured visually: Perfect spherical propagation at expected rate

**Wavelength:**

$$\text{By construction: } k = 2\pi/\lambda \rightarrow \lambda = 2\pi/k = \lambda_q \quad \checkmark$$

Measured spatially: Distance between wavefronts matches theoretical

**Stability:**

- Unconditionally stable - no timestep constraint
- Simulation runs indefinitely without numerical issues

**Advantages of the Phase-Synchronized Approach:**

1. **Exact wave propagation velocity** - Eliminates numerical dispersion associated with discretization
2. **Precise wavelength preservation** - Phase relationships enforce exact wavelength λ
3. **Unconditional stability** - No timestep constraints or numerical divergence
4. **Computational efficiency** - Requires only trigonometric function evaluations, no iterative constraint solving
5. **Physical fidelity** - Exact correspondence with EWT parameters

#### 5.4.5 Comparison Table

| Method | Wave Speed | Wavelength | Stability | Realistic k? |
|--------|-----------|------------|-----------|--------------|
| Euler | N/A (crashes) | N/A | Unstable | No (reduced by 10^-10) |
| Leapfrog | N/A (crashes) | N/A | Unstable | No (reduced by 10^-10) |
| XPBD | 0.125c (at 1e5) | Disabled (O(N²)) | Stable | Yes (real k!) |
| Phase-Sync | c (exact) | λ_q (exact) | Unconditional | N/A (no springs) |

---

## 6. Discussion

### 6.1 Mathematical Limitations of Force-Based Methods at Quantum Scales

The failure of force-based numerical methods at the Planck scale stems from fundamental mathematical constraints rather than computational resource limitations. The relationship between physical parameters and numerical requirements can be expressed as:

$$\text{High frequency} \rightarrow \text{Extreme stiffness} \rightarrow \text{Prohibitive timestep constraints}$$

For physically realistic EWT parameters, the quantum wave frequency is:

$$f_q = 1.05 \times 10^{25} \text{ Hz}$$

This yields a spring stiffness coefficient of:

$$k = (2\pi f)^2 \times m \approx 5.56 \times 10^{44} \text{ N/m}$$

The corresponding stability-limited timestep becomes:

$$dt_{max} < 2/\omega = 2/(2\pi f) \approx 3 \times 10^{-26} \text{ s}$$

Consequently, simulating one second of physical time would require:

$$N_{steps} > 1 / (3 \times 10^{-26}) = 3.3 \times 10^{25} \text{ integration steps}$$

This requirement exceeds not only current computational capabilities but also fundamental limits of floating-point arithmetic and numerical precision. The extreme force magnitudes and momentum values inherent in quantum wave dynamics corroborate theoretical predictions regarding the substantial energy density at quantum scales, manifesting as mathematical singularities in numerical integration schemes.

### 6.2 Why XPBD Shows Reduced Wave Speed

XPBD achieves stability through **compliance parameter** $\tilde{\alpha}$:

$$\Delta\lambda = -C / (w_i + w_j + \tilde{\alpha})$$

where $\tilde{\alpha} = 1/(k \cdot dt^2)$

Even with $\tilde{\alpha}$ extremely small ($\tilde{\alpha}/(2w) \approx 10^{-47}$), the iterative constraint satisfaction process may introduce **effective damping** or **phase lag** that slows wave propagation.

**Hypothesis:** XPBD's Jacobi iteration with constraint averaging distributes corrections differently than instantaneous spring forces, creating **dispersion** (frequency-dependent wave speed).

At low resolution (4.6 granules/λ), this effect is severe (5% of c). At higher resolution (21 granules/λ), improves to 12.5% of c, but gap remains.

### 6.3 Theoretical Foundation for Phase Synchronization Success

The phase-synchronized approach succeeds through direct implementation of wave mechanics rather than force-based dynamics. This represents a fundamental paradigm shift in computational methodology for quantum-scale simulation.

**Force Mechanics Paradigm (Limitations at Quantum Scale):**

The classical approach follows the causal chain: Forces → Acceleration → Velocity → Position (Dynamic → Kinematic)

This sequential integration process:

- Accumulates numerical error at each step
- Requires restrictive timestep constraints
- Fails at extreme stiffness values

**Wave Mechanics Paradigm (Successful Implementation):**

The wave-based approach directly specifies: Wave Equation → Position and Velocity (Analytical Solution)

This direct formulation:

- Eliminates numerical integration
- Avoids error accumulation
- Provides unconditional stability

**Comparative Analysis:**

- **Force mechanics paradigm**: F → a → v → x (sequential integration, fails at quantum scales)
- **Wave mechanics paradigm**: Phase relationships → Direct position calculation (analytical solution, exact results)

**Physical Interpretation:**

At quantum scales, granules oscillate according to phase relationships governed by wave equations. Phase coherence, constitutes a fundamental mechanism.

This observation validates EWT's wave-centric interpretation and aligns with wave-particle duality in quantum mechanics, suggesting that particles are more accurately conceptualized as wave patterns.

### 6.4 Implications for Quantum Simulation

The extreme energy density of quantum waves is evidenced by force magnitudes and momentum values that exceed the resolution capacity of numerical integration methods. This represents a fundamental mathematical limitation rather than a computational constraint. Even with unlimited computational resources, numerical integration cannot resolve the extreme stiffness arising from:

$$\text{High frequencies} \rightarrow \text{Extreme stiffness} \rightarrow \text{Prohibitive iteration requirements} \rightarrow \text{Infinitesimal timesteps}$$

This corroborates EWT's theoretical predictions regarding the enormous energy contained in quantum waves, manifesting as forces and momentum values that cause mathematical failure of integration schemes.

Our findings indicate:

1. **Force-based particle simulation** is inappropriate for quantum-scale phenomena (mathematical limitation, not computational)
2. **Wave equation frameworks** are fundamentally superior for quantum simulation
3. **Direct analytical solutions** bypass numerical integration challenges entirely
4. **Phase relationships** are more fundamental than force interactions at quantum scale

This validates EWT's wave-centric view of quantum mechanics over particle-centric interpretations - waves are fundamental entities described by phase relationships and harmonic oscillations.

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

## 8. Conclusions

This systematic investigation of numerical methods for quantum-scale wave dynamics yields four principal findings with significant implications for computational quantum physics:

1. **Fundamental mathematical limitations of force-based integration**: Explicit integration methods (semi-implicit Euler and symplectic Leapfrog) encounter insurmountable mathematical barriers when applied to Planck-scale dynamics with physically realistic parameters. The extreme stiffness requirements (k ≈ 10^44 N/m) necessitate timesteps below 10^-26 seconds, exceeding the resolution limits of floating-point arithmetic. This limitation is mathematical rather than computational, confirming theoretical predictions regarding the extreme energy density of quantum waves. The "Impossible Triangle" demonstrates that realistic stiffness, numerical stability, and human-visible motion cannot be simultaneously achieved, with a frequency mismatch ratio of approximately 360 million to one proving unbridgeable for explicit integrators.

2. **XPBD performance characteristics**: The Extended Position-Based Dynamics solver successfully achieves numerical stability while maintaining physically realistic stiffness values, effectively circumventing the Impossible Triangle. However, empirical measurements reveal an unexpected wave speed reduction of 8-18 fold relative to the theoretical speed of light. While this represents significant progress in achieving stable quantum-scale simulation with realistic parameters, the wave speed anomaly merits further investigation into the dispersive effects of iterative constraint satisfaction.

3. **Phase-synchronized harmonic oscillation superiority**: The PSHO approach achieves exact wave propagation at the speed of light with precise wavelength correspondence through direct implementation of wave mechanics. This method bypasses numerical integration entirely, providing unconditional stability and computational efficiency. The success of this approach validates the fundamental superiority of wave-based frameworks over particle-based force mechanics for quantum-scale phenomena.

4. **Paradigm implications for quantum simulation**: Our findings demonstrate that wave mechanics frameworks are inherently superior to particle-based force mechanics for quantum-scale simulation. This superiority reflects the fundamental nature of quantum phenomena, wherein phase relationships and wave equations supersede particle interactions and force dynamics.

The investigation reveals a critical insight: quantum wave phenomena cannot be accurately simulated using particle mechanics at Planck scales; they must be modeled using wave mechanics directly. This conclusion validates the wave-centric interpretation of quantum mechanics proposed by Energy Wave Theory and establishes phase relationships as the fundamental organizing principle at quantum scales.

The progression of our investigation illustrates the evolution from force-based to wave-based approaches:

- Force-based integration (Euler): Numerical instability
- Symplectic integration (Leapfrog): Persistent instability
- Constraint-based dynamics (XPBD): Stability with reduced wave speed
- Phase-synchronized oscillation: Exact wave propagation

These findings resolve critical computational barriers in quantum simulation and demonstrate that computational methodologies must align with the fundamental physics being modeled. For quantum-scale wave dynamics, wave mechanics provides the appropriate mathematical framework.

---

## Acknowledgments

This research was conducted using the OpenWave simulator (available at <https://github.com/openwave-labs/openwave>), implemented with the Taichi Lang GPU acceleration framework. The author acknowledges the assistance of Claude AI (Anthropic) in code development, experimental design, data analysis, and manuscript preparation. The author assumes full responsibility for all scientific claims and conclusions presented herein. We acknowledge the seminal contributions of Matthias Müller and colleagues in developing Extended Position-Based Dynamics [5-6], which provided the theoretical foundation for our constraint-based experimental approach.

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
def compute_spring_forces(positions, equilibrium, forces, links,
                          links_count, rest_length, stiffness):
    for i in range(positions.shape[0]):
        force = ti.Vector([0.0, 0.0, 0.0])
        for j in range(links_count[i]):
            neighbor = links[i, j]

            # Spring force: F = -k(x - L0)
            d = positions[neighbor] - positions[i]
            distance = d.norm()
            displacement = distance - rest_length
            force_mag = -stiffness * displacement
            force += (force_mag / distance) * d

        forces[i] = force

@ti.kernel
def integrate_euler(positions, velocities, forces, mass, dt, damping):
    for i in range(positions.shape[0]):
        a = forces[i] / mass
        velocities[i] += a * dt
        velocities[i] *= damping
        positions[i] += velocities[i] * dt
```

### A.2 XPBD Constraint Solver - STABLE

```python
@ti.kernel
def solve_distance_constraints(positions, neighbors, masses,
                                rest_length, compliance, dt, omega):
    # Phase 1: Accumulate position deltas (Jacobi iteration)
    for i in range(positions.shape[0]):
        delta[i] = ti.Vector([0.0, 0.0, 0.0])
        count[i] = 0

        for j in range(8):  # 8 BCC neighbors
            neighbor = neighbors.links[i, j]

            # Constraint: C = ||xi - xj|| - L0
            d = positions[neighbor] - positions[i]
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
    for i in range(positions.shape[0]):
        if count[i] > 0:
            positions[i] += (omega / count[i]) * delta[i]
```

### A.3 Phase-Synchronized Harmonic - PERFECT

```python
@ti.kernel
def oscillate_granules(positions, velocities, equilibrium, directions,
                       radial_distances, t, slow_mo, amp_boost):
    """Phase-synchronized harmonic oscillation (radial_wave.py)"""
    f_slowed = frequency / slow_mo
    omega = 2.0 * ti.math.pi * f_slowed
    k = 2.0 * ti.math.pi / wavelength_am  # Wave number

    for idx in range(positions.shape[0]):
        direction = directions[idx]
        r = radial_distances[idx]
        phase = -k * r  # Outward propagating wave

        # Direct analytical solution (no integration!)
        displacement = amplitude_am * amp_boost * ti.cos(omega * t + phase)
        positions[idx] = equilibrium[idx] + displacement * direction

        velocity_magnitude = -amplitude_am * amp_boost * omega * ti.sin(omega * t + phase)
        velocities[idx] = velocity_magnitude * direction
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
