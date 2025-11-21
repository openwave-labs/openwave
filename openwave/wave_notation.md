
# WAVE NOTATION GUIDE

OpenWave uses wave equations & terminology based on standardized physics notation, these are best practices used in scientific literature.

## Wave Medium

- MEDIUM-DENSITY: `ρ = 3.86e22 [kg/m³]`, propagates momentum, defines c (EWT constant)
- WAVE-SPEED: `c = 3e8 [m/s]`, constant defined by medium property, √(medium elasticity/
density)
- WAVE-SOURCE: defines amplitude and frequency (rhythm, vibration), charges energy

## Wave Character (WIP equation derivations)

- WAVE-MODE: `cos(θ) = k̂·û`, longitudinal / transverse polarity (fraction [0,1])
- WAVE-TYPE: standing / traveling (fraction [0,1])
- WAVE-FORM: temporal / spatial profile (sine, square, triangle, sawtooth)
- WAVE-DIRECTION: `k̂ = S / |S|`, unit vector (from energy flux)
- DISPLACEMENT-DIRECTION: `û = ∇ψt / ∇ψl`, unit vector (from displacement gradient)

## Wave Rhythm

- WAVE-FREQUENCY: `f = c/λ [Hz]`, temporal oscillation rate (can change locally, multiple FT decomposition). Defines TIME = the wave frequency, rhythm
  - angular frequency: `ω = 2πf [rad/s]`
  - ωt = temporal phase (controls rhythm, time-varying component)
- WAVE-PERIOD: `T = 1/f [s]`, time for one complete cycle
- WAVE-PHASE: `φ [radians]`, 0 to 2π, position in wave cycle, interference patterns

## Wave Size

- WAVE-DISPLACEMENT: `ψl & ψt [m]`, harmonic oscillation in 2 polarities
- WAVE-AMPLITUDE: `Al & At = max|ψ| [m]`, displacement envelope (running RMS, falloff at 1/r, near/far fields, force/pressure/density)
- WAVE-LENGTH: `λ = c/f [m]`, spatial period (changes when moving particle, doppler, wave fronts distance)
  - spatial frequency: `ξ = 1/λ [1/m]`
  - angular wave number: `k = 2π/λ [rad/m]`
  - kr = spatial phase

## Wave Energy

- WAVE-ENERGY: `E = ρV(fA)² [J]`, total energy (EWT equation, conserved property)
- ENERGY-FLUX: `S = -c²ρ(∂ψ/∂t)∇ψ [W/m²]`, Poynting-like vector (energy flux density)
- MOMENTUM-DENSITY: `g = S/c² = -ρ(∂ψ/∂t)∇ψ [kg/(m²·s)]`, momentum per unit volume (vector field)

## Wave Motion

- WAVE-FUNCTION: `ψ(r,t) = A·e^(iωt)·sin(kr)/r [m]`, exponential spherical wave
- WAVE-PROPAGATION: `ψ̈ = c²∇²ψ [m/s²]`, wave equation (propagates instantaneous oscillating scalar value)

## Wave Interaction

- REFLECTION: changes direction of propagation (`ψ = 0` Dirichlet boundary conditions)
- SUPERPOSITION: amplitude modulation/combination (constructive / destructive interference)
- RESONANCE: harmonics, coherence, influence on [phase, motion, frequency]

## Wave Profiling

- Diagnostic tool (measures wave properties per voxel, comprehensive wave data)
