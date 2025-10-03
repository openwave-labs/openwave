"""
QUANTUM-WAVE
(AKA: PRANA @yoga, QI @daoism, JEDI FORCE @starwars)

Wave dynamics and motion physics for spacetime granules.
OpenWave proprietary physics engine.
"""

import taichi as ti

from openwave.common import constants

# ================================================================
# Quantum-Wave Oscillation Parameters
# ================================================================
amplitude = constants.QWAVE_AMPLITUDE  # m, quantum-wave amplitude (equilibrium-to-peak)
frequency = constants.QWAVE_SPEED / constants.QWAVE_LENGTH  # Hz, quantum-wave frequency
# slow-motion factor (divides frequency for human-visible motion)
slow_mo = 1e25  # (1 = real-time, 10 = 10x slower, 1e25 = 10 * trillions * trillions FPS)


# ================================================================
# Quantum-Wave Maker Kernel (harmonic oscillations)
# ================================================================


@ti.kernel
def oscillate_vertex(
    positions: ti.template(),  # type: ignore
    velocities: ti.template(),  # type: ignore
    vertex_indices: ti.template(),  # type: ignore
    vertex_equilibrium: ti.template(),  # type: ignore
    vertex_directions: ti.template(),  # type: ignore
    t: ti.f32,  # type: ignore
):
    """Update 8 vertex positions and velocities using harmonic oscillation (wave makers).

    Vertices oscillate radially along direction vectors toward/away from lattice center.
    Position: x(t) = x_eq + A·cos(ωt)·direction
    Velocity: v(t) = -A·ω·sin(ωt)·direction (derivative of position)

    Args:
        positions: Position field for all granules
        velocities: Velocity field for all granules
        vertex_indices: Indices of 8 corner vertices
        vertex_equilibrium: Equilibrium positions of 8 vertices
        vertex_directions: Normalized direction vectors from vertices to center
        t: Current simulation time (accumulated)
    """
    f_slowed = frequency / slow_mo
    omega = 2.0 * ti.math.pi * f_slowed  # angular frequency

    for v in range(8):
        idx = vertex_indices[v]
        direction = vertex_directions[v]

        # Position: x(t) = x_eq + A·cos(ωt)·direction
        displacement = amplitude * ti.cos(omega * t)
        positions[idx] = vertex_equilibrium[v] + displacement * direction

        # Velocity: v(t) = -A·ω·sin(ωt)·direction (derivative of position)
        velocity_magnitude = -amplitude * omega * ti.sin(omega * t)
        velocities[idx] = velocity_magnitude * direction
