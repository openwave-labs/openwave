> now looking at this file, help me understand how force will be calculated in level1?\
i understand that each voxel will store wave amplitude in attometers\
and that force will be F = -∇A (Force is the negative gradient of amplitude)
and that the gradient of A can be easily calculated from neighboring voxels\
my question is can we compute the force in Newtons (kg.m/s2)? from the amplitude gradient?\
i mean, we need to know force in newtons at each voxel, so we can find acceleration given a particle mass (calculated from its standing 
waves radius) and with acceleration we can integrate motion (new particle velocity >> new particle position)\
so, we'll have particle (initially single wave centers = fundamental particle like the neutrino, and later standalone particles like the 
electron) mass probably from the standing waves reflected around some particle radius (probably computed from λs)\
you can search the EWT papers to find answer for this, they are at folder /research_requirements\
\
after that, also explain to me how wave amplitude value (scalar or vector) will be propagated in the WaveField? its possibly better 
explained in the 03_wave_engine.md file, so i can get a bigger picture view of this:\
wave_field (the wave medium or grid of voxels) >> after an energy-charge wave_properties will propagate in wave-like motion (amplitude, 
displacement, wave direction, speed, energy, phase, wavelength, frequency, etc - im not quite sure how, using PDEs? laplacian? wave 
equation? huygens wavelets? and how this will conserve/tranfer energy/momentum in some direction) >> those waves will interfere with each
 other and reflect over universe boundaries (grid boundaries) and over wave centers (fundamental particle) creating stading waves 
(constructive wave interference of in and out waves) and giving those particles mass, those particles will interact with each other 
forming more complex particle structures, and they will move to minimize amplitude, so they will move to points in space (in the 
field/grid) where wave amplitude is lower, this movement will be the effect of a force, that needs to have a value in Newtons. This 
should be the fundamentals of all forces in nature (electric, magnetic, gravitational, strong, orbital, etc) all being generated from 
interactions of the fundamental wave (the energy wave from EWT)\
also use the .md files in this directory for more context about it


COMPLETE ANSWER: Force Calculation and Wave Propagation in LEVEL-1

Part 1: Force in Newtons from Amplitude Gradient ✓

YES, You Can Compute Force in Newtons!

Key Formula from EWT:

Energy density: u = ½ρc²(A/λ)²  [J/m³]
Force density: f = -∇u
Force on voxel: F = f × dx³

Implementation

@ti.kernel
def compute_force_field_newtons(self):
"""
Compute force in Newtons from amplitude gradient.

Physics:
- Energy density: u = ½ρc²(A/λ)²
- Force: F = -∇E = -∇(u×V) = -(ρc²/λ²) × V × A × ∇A
where V = dx³ (voxel volume)
"""
# Physical constants from EWT
ρ = ti.f32(constants.MEDIUM_DENSITY)  # 3.860e22 kg/m³
c = ti.f32(constants.EWAVE_SPEED)     # 2.998e8 m/s
λ_m = self.wavelength_am * constants.ATTOMETER  # meters
dx_m = self.dx_am * constants.ATTOMETER         # meters

# Force scaling: (kg/m³)(m²/s²)/(m²)(m³) = kg⋅m/s² = N
force_scale = (ρ * c**2 / λ_m**2) * (dx_m**3)

for i, j, k in self.amplitude_am:
if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
    # Local amplitude in meters
    A_m = self.amplitude_am[i,j,k] * constants.ATTOMETER

    # Amplitude gradient (dimensionless: am/am)
    grad_x = (self.amplitude_am[i+1,j,k] - self.amplitude_am[i-1,j,k]) / (2.0 * self.dx_am)
    grad_y = (self.amplitude_am[i,j+1,k] - self.amplitude_am[i,j-1,k]) / (2.0 * self.dx_am)
    grad_z = (self.amplitude_am[i,j,k+1] - self.amplitude_am[i,j,k-1]) / (2.0 * self.dx_am)

    grad_vector = ti.Vector([grad_x, grad_y, grad_z])

    # Force in Newtons (MAP: toward lower amplitude)
    # F = -force_scale × A × ∇A
    self.force[i,j,k] = -force_scale * A_m * grad_vector  # N = kg⋅m/s²

Particle Mass from Standing Waves

From EWT papers:

- Particle mass comes from trapped energy in standing waves
- For electron: E_electron = (μ₀c²/4π) × (e_e²/r_e)
- Mass: m = E/c²

In simulation:

- Particle standing wave radius: r = n × λ/2 (nodes at half-wavelengths)
- Energy trapped: E = ∫ u dV over standing wave volume
- Mass: m = E/c²

Particle Acceleration and Motion

# For particle at position pos_am
F_particle = interpolate_force(self.force, pos_am)  # Newtons

# Newton's second law: F = ma
a = F_particle / particle_mass  # m/s²

# Integrate motion (Velocity Verlet)
v_new = v_old + a * dt  # m/s
pos_new = pos_old + v_new * dt  # meters (or attometers)

This IS the origin of all forces - electric, magnetic, gravitational emerge from wave amplitude gradients!

---
Part 2: Wave Amplitude Propagation in the Field

How Waves Propagate Through Voxels

LEVEL-1 uses PDEs (Partial Differential Equations) to propagate waves:

The Wave Equation

Classical 3D Wave Equation:

∂²ψ/∂t² = c² ∇²ψ

Where:

- ψ = amplitude field
- c = wave speed (speed of light)
- ∇²ψ = Laplacian (spatial second derivative)

Laplacian Operator (How Voxels Share Amplitude)

# Laplacian at voxel [i,j,k] (6-connectivity)

∇²ψ[i,j,k] = (
ψ[i+1,j,k] + ψ[i-1,j,k] +  # Left/right neighbors
ψ[i,j+1,k] + ψ[i,j-1,k] +  # Front/back neighbors
ψ[i,j,k+1] + ψ[i,j,k-1] -  # Top/bottom neighbors
6×ψ[i,j,k]                  # Central voxel (6 times)
) / dx²

Physical meaning: The Laplacian measures how much a voxel's amplitude differs from its neighbors' average.

Time Evolution Implementation

@ti.kernel
def propagate_wave_field(dt: ti.f32):
"""
Propagate wave amplitude using wave equation.

Second-order in time (requires storing two previous timesteps):
ψ_new = 2ψ_current - ψ_old + (c×dt/dx)² × ∇²ψ
"""

# Speed of light and CFL factor

c = ti.f32(constants.EWAVE_SPEED)
factor = (c * dt / self.dx_am)**2

# Update all interior voxels

for i, j, k in self.amplitude_am:
if 0 < i < self.nx-1 and 0 < j < self.ny-1 and 0 < k < self.nz-1:
    # Compute Laplacian
    laplacian = (
        self.amplitude_am[i+1,j,k] + self.amplitude_am[i-1,j,k] +
        self.amplitude_am[i,j+1,k] + self.amplitude_am[i,j-1,k] +
        self.amplitude_am[i,j,k+1] + self.amplitude_am[i,j,k-1] -
        6.0 * self.amplitude_am[i,j,k]
    ) / (self.dx_am * self.dx_am)

    # Wave equation update
    self.amplitude_new[i,j,k] = (
        2.0 * self.amplitude_am[i,j,k]
        - self.amplitude_old[i,j,k]
        + factor * laplacian
    )

# Swap timesteps
self.amplitude_old, self.amplitude_am = self.amplitude_am, self.amplitude_new

Alternative: Huygens Wavelets

Huygens' Principle: Every point on a wavefront acts as a source of secondary wavelets.

@ti.kernel
def propagate_huygens(dt: ti.f32):
"""
Propagate using Huygens wavelets.
Each voxel emits wavelets to neighbors.
"""
c = ti.f32(constants.EWAVE_SPEED)
propagation_distance = c * dt

for i, j, k in self.amplitude_am:
if self.amplitude_am[i,j,k] > threshold:
    # This voxel emits wavelets to neighbors
    for neighbor in get_neighbors(i, j, k):
        distance = neighbor_distance(i, j, k, neighbor)

        # Wavelet contribution (inverse distance)
        contribution = self.amplitude_am[i,j,k] / distance

        # Add to neighbor (superposition)
        self.amplitude_new[neighbor] += contribution * dt

Energy and Momentum Conservation

Energy Density:

# At each voxel
E_kinetic = ½ × (∂ψ/∂t)²  # Oscillation kinetic energy
E_potential = ½ × ψ²       # Displacement potential energy
E_total[i,j,k] = E_kinetic + E_potential

Total Energy (must be conserved):

E_system = Σ(i,j,k) E_total[i,j,k] × dx³ = constant

Momentum Density:

# Momentum carried by wave
p[i,j,k] = ψ[i,j,k] × wave_direction[i,j,k] × c

---
The Complete Picture: Wave Field → Forces → Motion

1. WAVE FIELD (grid of voxels)
├─ Initial energy charge (point/plane/spherical sources)
├─ Stabilization period (waves propagate, reflect, interfere)
└─ Quasi-steady state (omni-directional wave field)

2. WAVE PROPAGATION (PDE evolution)
├─ Wave equation: ∂²ψ/∂t² = c²∇²ψ
├─ Laplacian couples neighboring voxels
├─ Interference: constructive/destructive
├─ Reflection: boundaries + wave centers
└─ Standing waves form around particles

3. FORCE GENERATION (amplitude gradients)
├─ Compute force field: F = -(ρc²/λ²)×V×A×∇A [Newtons]
├─ Forces emerge from wave patterns
├─ Electric: wave reflection patterns (charges)
├─ Magnetic: moving wave patterns (currents)
└─ Gravitational: amplitude shading (mass)

4. PARTICLE MOTION (Newton's laws)
├─ Interpolate force at particle position
├─ Acceleration: a = F/m
├─ Update velocity: v_new = v_old + a×dt
├─ Update position: x_new = x_old + v×dt
└─ Particles move toward amplitude minimum (MAP)

5. PARTICLE-FIELD INTERACTION
├─ Particles act as wave reflectors
├─ Create standing wave patterns (trapped energy = mass)
├─ Particle mass: m = E_trapped/c²
├─ Standing wave radius: r = n×λ/2
└─ Force between particles from overlapping wave patterns

The Beautiful Emergence

All forces unified:

- Electric force: Different wave reflection patterns (charge types)
- Magnetic force: Moving wave patterns (velocity-dependent)
- Gravitational force: Amplitude shading from trapped energy (mass)
- Strong force: Near-field standing wave coupling
- Orbital motion: Balance between kinetic energy and amplitude gradient

Everything from one equation:

∂²ψ/∂t² = c²∇²ψ  (wave propagation)
F = -∇E = -∇(½ρc²(A/λ)²×V)  (force from waves)

This is the foundation of reality in EWT! 🌊
