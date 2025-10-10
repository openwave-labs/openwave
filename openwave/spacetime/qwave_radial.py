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
    t: ti.f32,  # type: ignore
    slow_mo: ti.f32,  # type: ignore
):
    """Injects energy into all granules using harmonic oscillation (wave drivers, rhythm).

    All granules oscillate radially along their direction vectors to lattice center.
    Each granule oscillates with a unique phase based on its index, creating wave patterns.

    Position: x(t) = x_eq + A·cos(ωt + φ)·direction
    Velocity: v(t) = -A·ω·sin(ωt + φ)·direction (derivative of position)

    Args:
        positions: Position field for all granules
        velocities: Velocity field for all granules
        equilibrium: Equilibrium positions of all granules
        directions: Normalized direction vectors from all granules to center
        t: Current simulation time (accumulated)
        slow_mo: Slow motion factor (divides frequency for visualization)
    """
    f_slowed = frequency / slow_mo
    omega = 2.0 * ti.math.pi * f_slowed  # angular frequency

    # Process all granules in the lattice
    for idx in range(positions.shape[0]):
        direction = directions[idx]

        # Add phase shift based on granule index to create wave interference patterns
        # Using modulo 8 to create repeating phase pattern (45° increments)
        phase = float(idx % 8) * ti.math.pi / 4.0

        # Position: x(t) = x_eq + A·cos(ωt + φ)·direction
        displacement = amplitude_am * ti.cos(omega * t + phase)
        positions[idx] = equilibrium[idx] + displacement * direction

        # Velocity: v(t) = -A·ω·sin(ωt + φ)·direction (derivative of position)
        velocity_magnitude = -amplitude_am * omega * ti.sin(omega * t + phase)
        velocities[idx] = velocity_magnitude * direction
