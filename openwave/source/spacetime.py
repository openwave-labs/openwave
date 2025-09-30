"""
SPACETIME
(AKA: AKASHA @yoga, WUJI @daoism, AETHER @classical)

SPACETIME is a Wave Medium and Propagates Wave Motions (QUANTUM-WAVE).
Modeled as a particle-based elastic medium that allows energy
to transfer from one point to the next.

"Aether" can refer to the personification of the bright upper sky in Greek mythology,
the classical fifth element or quintessence filling the universe,
or a hypothetical substance once thought to carry light and other electromagnetic waves.
"""

import numpy as np
import taichi as ti

import openwave.common.config as config
import openwave.common.constants as constants
import openwave.common.equations as equations

ti.init(arch=ti.gpu)


# ==================================================================
# Physics Engine
# ==================================================================


class Granule:
    """
    Granule Model: The aether consists of "granules".
    Fundamental units that vibrate in harmony and create wave patterns.
    Their collective motion at Planck scale creates all observable phenomena.
    Each granule has a defined radius and mass.
    """

    def __init__(self, unit_cell_edge: float):
        """Initialize scaled-up granule properties based on scaled-up unit cell edge length.

        Args:
            unit_cell_edge: Edge length of the BCC unit-cell in meters.
        """
        self.radius = unit_cell_edge / (2 * np.e)  # radius = unit cell edge / 2e
        self.mass = (
            constants.SPACETIME_DENSITY * unit_cell_edge**3 / 2
        )  # mass = spacetime density * scaled unit cell volume / 2 granules per BCC unit-cell


@ti.data_oriented
class Lattice:
    """
    3D Body-Centered Cubic (BCC) lattice for spacetime simulation.
    BCC topology: cubic unit cells with an additional granule at the center.
    More efficient packing than simple cubic (68% vs 52% space filling).

    Performance Design: 1D Arrays with 3D Vectors
    - Memory: Contiguous layout, perfect cache line utilization (64-byte alignment)
    - Compute: Single loop parallelization, no index arithmetic (vs i*dim²+j*dim+k)
    - GPU: Direct thread mapping (thread_i→granule_i), coalesced memory access
    - BCC Lattice: Uniform treatment of corner+center granules in single array
    Benefits:
    - Simpler updates: One kernel updates all particles
    - Cleaner code: No need to manage multiple arrays
    - Movement-Ready: Velocity field prepared for dynamics,
    particles can move freely without grid remapping constraints

    This is why high-performance physics engines (molecular dynamics, N-body simulations)
    universally use 1D arrays for particle data, regardless of spatial dimensionality.
    """

    def __init__(self, universe_edge: float):
        """
        Initialize BCC lattice and compute scaled-up unit-cell spacing.
        Universe size (arg) and computing capacity (config.py) are used to define
        scaled-up unit-cell properties and scale factor.

        Args:
            universe_edge: Edge length of the cubic universe in meters
        """
        # Compute lattice total energy from quantum-wave equation
        self.energy = equations.energy_wave_equation(universe_edge**3)  # in Joules
        self.energy_kWh = equations.J_to_kWh(self.energy)  # in KWh
        self.energy_years = self.energy_kWh / (183230 * 1e9)  # global energy use

        # Get max granule count from computing capacity resolution
        target_granules = config.SPACETIME_RES

        # Set universe properties
        self.universe_edge = universe_edge
        universe_volume = universe_edge**3

        # Compute initial unit-cell properties (before rounding and lattice symmetry)
        # BCC has 2 granules per unit cell (8 corners shared + 1 center)
        init_unit_cell_volume = universe_volume / (target_granules / 2)
        init_unit_cell_edge = init_unit_cell_volume ** (1 / 3)  # unit cell edge (a^3 = volume)

        # Calculate grid dimensions (number of unit cells per dimension)
        # Round to nearest odd integer for symmetric grid
        self.raw_size = universe_edge / init_unit_cell_edge
        floor = int(self.raw_size)
        self.grid_size = floor if floor % 2 == 1 else floor + 1

        # Recompute unit-cell edge length based on rounded grid size and scale factor
        self.unit_cell_edge = universe_edge / self.grid_size  # adjusted unit cell edge length
        self.scale_factor = self.unit_cell_edge / (
            2 * constants.PLANCK_LENGTH * np.e
        )  # linear scale factor from Planck length, increases computability

        # Compute quantum-wave linear resolution, sampling rate
        # granules per wavelength, should be >2 for Nyquist
        self.qwave_res = constants.QWAVE_LENGTH / self.unit_cell_edge * 2
        # Compute universe linear resolution, qwavelengths per universe edge
        self.uni_res = universe_edge / constants.QWAVE_LENGTH

        # Total granules: corners + centers
        # Corners: (grid_size + 1)^3, Centers: grid_size^3
        corner_count = (self.grid_size + 1) ** 3
        center_count = self.grid_size**3
        self.total_granules = corner_count + center_count

        # Initialize position and velocity 1D arrays
        # 1D array design: Better memory locality, simpler kernels, future-ready for dynamics
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.front_octant = ti.field(dtype=ti.i32, shape=self.total_granules)

        # Populate the lattice
        self.populate_lattice()
        self.find_front_octant()

    @ti.kernel
    def populate_lattice(self):
        """Populate BCC lattice positions in a 1D array.
        Kernel is properly optimized for Taichi's parallel execution:
        1. Single outermost loop - for idx in range() allows full GPU parallelization
        2. Index decoding - Converts linear index to 3D coordinates using integer division/modulo
        3. No nested loops - All granules computed in parallel across GPU threads
        4. Efficient branching - Simple if/else to determine corner vs center granules
        This structure ensures maximum parallelization on GPU, as each thread independently
        computes one granule's position without synchronization overhead.
        """
        # Parallelize over all granules using single outermost loop
        for idx in range(self.total_granules):
            # Determine if this is a corner or center granule
            corner_count = (self.grid_size + 1) ** 3

            if idx < corner_count:
                # Corner granule: decode 3D position from linear index
                grid_dim = self.grid_size + 1
                i = idx // (grid_dim * grid_dim)
                j = (idx % (grid_dim * grid_dim)) // grid_dim
                k = idx % grid_dim

                self.positions[idx] = ti.Vector(
                    [i * self.unit_cell_edge, j * self.unit_cell_edge, k * self.unit_cell_edge]
                )
            else:
                # Center granule: decode position with offset
                center_idx = idx - corner_count
                i = center_idx // (self.grid_size * self.grid_size)
                j = (center_idx % (self.grid_size * self.grid_size)) // self.grid_size
                k = center_idx % self.grid_size

                offset = self.unit_cell_edge / 2
                self.positions[idx] = ti.Vector(
                    [
                        i * self.unit_cell_edge + offset,
                        j * self.unit_cell_edge + offset,
                        k * self.unit_cell_edge + offset,
                    ]
                )

            # Initialize velocity to zero for all granules
            self.velocities[idx] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def find_front_octant(self):
        """."""
        for i in range(self.total_granules):
            # Mark if granule is in the front 1/8th block, > halfway on all axes
            # 0 = not in front octant, 1 = in front octant
            self.front_octant[i] = (
                1
                if (
                    self.positions[i][0] > self.universe_edge / 2
                    and self.positions[i][1] > self.universe_edge / 2
                    and self.positions[i][2] > self.universe_edge / 2
                )
                else 0
            )

    @ti.kernel
    def update_positions(self, dt: ti.f32):  # type: ignore
        """Update granule positions based on velocities."""
        for i in self.positions:
            self.positions[i] += self.velocities[i] * dt
