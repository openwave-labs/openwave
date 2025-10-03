"""
QUANTUM-WAVE
(AKA: PRANA @yoga, QI @daoism, JEDI FORCE @starwars)

Wave dynamics and motion physics for spacetime granules.
"""

import taichi as ti


@ti.kernel
def oscillate_vertex(
    lattice_positions: ti.template(),  # type: ignore
    lattice_velocities: ti.template(),  # type: ignore
    vertex_indices: ti.template(),  # type: ignore
    vertex_equilibrium: ti.template(),  # type: ignore
    vertex_directions: ti.template(),  # type: ignore
    t: ti.f32,  # type: ignore
    amplitude: ti.f32,  # type: ignore
    frequency: ti.f32,  # type: ignore
    slow_mo: ti.f32,  # type: ignore
):
    """Update 8 vertex positions and velocities using harmonic oscillation.

    Vertices oscillate radially along direction vectors toward/away from lattice center.
    Position: x(t) = x_eq + A·cos(ωt)·direction
    Velocity: v(t) = -A·ω·sin(ωt)·direction (derivative of position)

    Args:
        lattice_positions: Position field for all granules
        lattice_velocities: Velocity field for all granules
        vertex_indices: Indices of 8 corner vertices
        vertex_equilibrium: Equilibrium positions of 8 vertices
        vertex_directions: Normalized direction vectors from vertices to center
        t: Current simulation time (accumulated)
        amplitude: Oscillation amplitude (equilibrium to peak)
        frequency: Base frequency in Hz
        slow_mo: Slow-motion factor (divides frequency for human-visible motion)
    """
    f_slowed = frequency / slow_mo
    omega = 2.0 * ti.math.pi * f_slowed  # angular frequency

    for v in range(8):
        idx = vertex_indices[v]
        direction = vertex_directions[v]

        # Position: x(t) = x_eq + A·cos(ωt)·direction
        displacement = amplitude * ti.cos(omega * t)
        lattice_positions[idx] = vertex_equilibrium[v] + displacement * direction

        # Velocity: v(t) = -A·ω·sin(ωt)·direction (derivative of position)
        velocity_magnitude = -amplitude * omega * ti.sin(omega * t)
        lattice_velocities[idx] = velocity_magnitude * direction
