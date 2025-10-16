"""
QUANTUM-WAVE
(AKA: PRANA @yoga, QI @daoism, JEDI FORCE @starwars)

Wave Physics Engine @spacetime module.
Wave dynamics and motion.

Radial harmonic oscillation for BCC lattice granules.
All granules oscillate toward/away from the lattice center along
their individual direction vectors, creating spherical wave interference patterns.
"""

import taichi as ti

from openwave.common import constants

# ================================================================
# Quantum-Wave Oscillation Parameters
# ================================================================
amplitude_am = constants.QWAVE_AMPLITUDE / constants.ATTOMETTER  # am, oscillation amplitude
wavelength_am = constants.QWAVE_LENGTH / constants.ATTOMETTER  # in attometers
frequency = constants.QWAVE_SPEED / constants.QWAVE_LENGTH  # Hz, quantum-wave frequency


# ================================================================
# Quantum-Wave Source Kernel (energy injection, harmonic oscillation, rhythm)
# ================================================================


@ti.kernel
def oscillate_granules(
    position: ti.template(),  # type: ignore
    equilibrium: ti.template(),  # type: ignore
    velocity: ti.template(),  # type: ignore
    center_direction: ti.template(),  # type: ignore
    center_distance: ti.template(),  # type: ignore
    t: ti.f32,  # type: ignore
    slow_mo: ti.f32,  # type: ignore
    freq_boost: ti.f32,  # type: ignore
    amp_boost: ti.f32,  # type: ignore
):
    """Injects energy into all granules using harmonic oscillation (wave source, rhythm).

    All granules oscillate radially along their direction vectors toward/away from the
    wave source (lattice center). Phase is determined by radial distance from wave source,
    creating outward-propagating spherical wave fronts. Granules at similar distances from
    wave source form oscillating shell-like structures.

    For spherical waves, amplitude decreases as 1/r to conserve total energy.
    This ensures that energy density integrated over expanding spherical shells
    remains constant: E_total = ∫(A²r²)dΩ = constant, requiring A ∝ 1/r.

    Near-Field vs Far-Field Regions (distance from wave source):
        - Near field (r < λ): Source region, wave structure forming, A held constant
        - Transition zone (λ < r < 2λ): Wave fronts organizing into spherical geometry
        - Far field (r > 2λ): Fully formed spherical waves, clean A ∝ 1/r falloff

    Waves are considered "fully formed" in the far-field region (r > 2λ from wave source).

    Position: x(t) = x_eq + A(r)·cos(ωt + φ)·direction
    Velocity: v(t) = -A(r)·ω·sin(ωt + φ)·direction (derivative of position)
    Amplitude: A(r) = A₀·(r₀/r), where r₀ is reference radius (1λ)
    Phase: φ = -kr, where
        k = 2π/λ is the wave number,
        r is the radial distance from wave source.
        (φ represents spatial phase shift; negative creates outward propagation)

    Energy conservation: E = ρV(c/λ × A)²
        - λ remains constant (wavelength unchanged in uniform medium)
        - c remains constant (wave speed unchanged in uniform medium)
        - A decreases as 1/r (amplitude falls off with distance from wave source)

    Implementation uses r_min = 1λ (one wavelength from wave source) based on:
        - EWT neutrino boundary specification at r = 1λ
        - EM theory transition to radiative fields around λ
        - Prevents singularity at r → 0
        - Ensures numerical stability and physical wave behavior

    Args:
        position: Position field for all granules
        velocity: Velocity field for all granules
        equilibrium: Equilibrium position of all granules
        center_direction: Normalized direction vectors from all granules toward wave source
        center_distance: Distance from each granule to wave source (in attometers)
        t: Current simulation time (accumulated)
        slow_mo: Slow motion factor (divides frequency for visualization)
        freq_boost: Frequency multiplier
        amp_boost: Multiplier for oscillation amplitude (for visibility in scaled lattices)
    """
    f_slowed = frequency / slow_mo * freq_boost
    omega = 2.0 * ti.math.pi * f_slowed  # angular frequency

    # Wave number k = 2π/λ (for spatial phase variation)
    # Using quantum wavelength to determine how phase changes with distance
    k = 2.0 * ti.math.pi / wavelength_am  # wave number (radians per attometer)

    # Reference radius for amplitude normalization (one wavelength from wave source)
    # This prevents singularity at r=0 and provides physically meaningful normalization
    r_reference = wavelength_am  # attometers

    # Process all granules in the lattice
    for idx in range(position.shape[0]):
        direction = center_direction[idx]

        # Phase determined by radial distance from wave source
        # Negative k·r creates outward propagating wave
        # Granules at same distance r from wave source oscillate in phase (shell-like behavior)
        r = center_distance[idx]  # distance from wave source in attometers
        phase = -k * r  # phase shift based on distance from wave source (outward propagating)

        # Amplitude falloff for spherical wave energy conservation: A(r) = A₀(r₀/r)
        # Prevents division by zero and non-physical amplitudes very close to wave source
        # Uses r_min = 1λ based on EWT neutrino boundary and EM near-field physics
        r_safe = ti.max(r, r_reference * 1)  # minimum 1 wavelength from wave source
        amplitude_falloff = r_reference / r_safe

        # Total amplitude at distance r from wave source
        # Includes energy conservation (amplitude_falloff) and visualization scaling (amp_boost)
        amplitude_at_r = amplitude_am * amplitude_falloff * amp_boost

        # Position: x(t) = x_eq + A(r)·cos(ωt + φ)·direction
        displacement = amplitude_at_r * ti.cos(omega * t + phase)
        position[idx] = equilibrium[idx] + displacement * direction

        # Velocity: v(t) = -A(r)·ω·sin(ωt + φ)·direction (derivative of position)
        velocity_magnitude = -amplitude_at_r * omega * ti.sin(omega * t + phase)
        velocity[idx] = velocity_magnitude * direction
