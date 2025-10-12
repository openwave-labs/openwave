"""
QUANTUM-WAVE DYNAMICS
(AKA: PRANA @yoga, QI @daoism, JEDI FORCE @starwars)

Radial harmonic oscillation for BCC lattice granules.
All granules oscillate toward/away from the lattice center along
their individual direction vectors, creating spherical wave interference patterns.

Wave dynamics and motion physics for spacetime.
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
# Quantum-Wave Driver Kernel (energy injection, harmonic oscillation, rhythm)
# ================================================================


@ti.kernel
def oscillate_granules(
    positions: ti.template(),  # type: ignore
    velocities: ti.template(),  # type: ignore
    equilibrium: ti.template(),  # type: ignore
    directions: ti.template(),  # type: ignore
    radial_distances: ti.template(),  # type: ignore
    t: ti.f32,  # type: ignore
    slow_mo: ti.f32,  # type: ignore
):
    """Injects energy into all granules using harmonic oscillation (wave drivers, rhythm).

    All granules oscillate radially along their direction vectors to lattice center.
    Phase is determined by radial distance from center, creating outward-propagating
    spherical wave fronts. Granules at similar distances form oscillating shell-like
    structures, with the wave originating from the lattice center.

    Position: x(t) = x_eq + A·cos(ωt - kr + φ₀)·direction
    Velocity: v(t) = -A·ω·sin(ωt - kr + φ₀)·direction (derivative of position)

    where k = 2π/λ is the wave number, r is the radial distance from center.

    Args:
        positions: Position field for all granules
        velocities: Velocity field for all granules
        equilibrium: Equilibrium positions of all granules
        directions: Normalized direction vectors from all granules to center
        radial_distances: Distance from each granule to lattice center (in attometers)
        t: Current simulation time (accumulated)
        slow_mo: Slow motion factor (divides frequency for visualization)
    """
    f_slowed = frequency / slow_mo
    omega = 2.0 * ti.math.pi * f_slowed  # angular frequency

    # Wave number k = 2π/λ (for spatial phase variation)
    # Using quantum wavelength to determine how phase changes with distance
    k = 2.0 * ti.math.pi / wavelength_am  # wave number (radians per attometer)

    # Process all granules in the lattice
    for idx in range(positions.shape[0]):
        direction = directions[idx]

        # Phase determined by radial distance from center
        # Negative k·r creates outward propagating wave (wave starts at center)
        # Granules at same distance r oscillate in phase (shell-like behavior)
        r = radial_distances[idx]  # distance to center in attometers
        phase = -k * r  # phase shift based on distance (outward propagating)

        # Position: x(t) = x_eq + A·cos(ωt + φ)·direction
        displacement = amplitude_am * ti.cos(omega * t + phase)
        positions[idx] = equilibrium[idx] + displacement * direction

        # Velocity: v(t) = -A·ω·sin(ωt + φ)·direction (derivative of position)
        velocity_magnitude = -amplitude_am * omega * ti.sin(omega * t + phase)
        velocities[idx] = velocity_magnitude * direction
