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
        self.granule_type = ti.field(dtype=ti.i32, shape=self.total_granules)
        self.granule_color = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.front_octant = ti.field(dtype=ti.i32, shape=self.total_granules)

        # Populate the lattice & index granule types
        self.populate_lattice()
        self.index_granule_type()
        self.set_granule_colors()
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
    def index_granule_type(self):
        """Classify each granule by its position in the BCC lattice structure.

        Classification:
        - VERTEX (0): 8 corner vertices of the cubic lattice boundary
        - EDGE (1): Granules on the 12 edges (but not corners)
        - FACE (2): Granules on the 6 faces (but not on edges/corners)
        - CORE (3): All other interior granules (not on boundary)
        - CENTRAL (4): Single granule at exact center of lattice
        """
        corner_count = (self.grid_size + 1) ** 3
        half_grid = self.grid_size // 2

        for idx in range(self.total_granules):
            if idx < corner_count:
                # Corner granule: decode 3D grid position
                grid_dim = self.grid_size + 1
                i = idx // (grid_dim * grid_dim)
                j = (idx % (grid_dim * grid_dim)) // grid_dim
                k = idx % grid_dim

                # Count how many coordinates are at boundaries (0 or grid_size)
                at_boundary = 0
                if i == 0 or i == self.grid_size:
                    at_boundary += 1
                if j == 0 or j == self.grid_size:
                    at_boundary += 1
                if k == 0 or k == self.grid_size:
                    at_boundary += 1

                if at_boundary == 3:
                    self.granule_type[idx] = config.TYPE_VERTEX
                elif at_boundary == 2:
                    self.granule_type[idx] = config.TYPE_EDGE
                elif at_boundary == 1:
                    self.granule_type[idx] = config.TYPE_FACE
                else:
                    self.granule_type[idx] = config.TYPE_CORE
            else:
                # Center granule: decode position with offset
                center_idx = idx - corner_count
                i = center_idx // (self.grid_size * self.grid_size)
                j = (center_idx % (self.grid_size * self.grid_size)) // self.grid_size
                k = center_idx % self.grid_size

                # Check if this is the exact central granule (only for odd grid_size)
                if i == half_grid and j == half_grid and k == half_grid:
                    self.granule_type[idx] = config.TYPE_CENTRAL
                else:
                    # Center granules are always in core (offset by 0.5 means never on boundary)
                    self.granule_type[idx] = config.TYPE_CORE

    @ti.kernel
    def set_granule_colors(self):
        """Assign colors to granules based on their classified type."""
        # Color lookup table (type index -> RGB color)
        color_lut = ti.Matrix(
            [
                config.COLOR_VERTEX[1],  # TYPE_VERTEX = 0
                config.COLOR_EDGE[1],  # TYPE_EDGE = 1
                config.COLOR_FACE[1],  # TYPE_FACE = 2
                config.COLOR_CORE[1],  # TYPE_CORE = 3
                config.COLOR_CENTRAL[1],  # TYPE_CENTRAL = 4
            ]
        )

        for i in range(self.total_granules):
            granule_type = self.granule_type[i]
            if 0 <= granule_type <= 4:
                self.granule_color[i] = ti.Vector(
                    [
                        color_lut[granule_type, 0],
                        color_lut[granule_type, 1],
                        color_lut[granule_type, 2],
                    ]
                )
            else:
                self.granule_color[i] = ti.Vector([1.0, 0.0, 1.0])  # Magenta for undefined

    @ti.kernel
    def find_front_octant(self):
        """Mark granules in the front octant (for block-slicing visualization).

        Front octant = granules where x, y, z > universe_edge/2
        Used for rendering: 0 = render, 1 = skip (for see-through effect)
        """
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


@ti.data_oriented
class Spring:
    """
    Spring couplings between granules in BCC lattice.
    Models elastic connections with 8-way, 4-way, 2-way, or 1-way topology
    depending on granule type (core/central, face, edge, vertex).

    In BCC lattice, each interior granule has 8 nearest neighbors at distance a*√3/2)
    where a is the unit cell edge length.
    """

    def __init__(self, lattice: Lattice):
        """
        Initialize spring connections for the lattice.

        Args:
            lattice: Lattice instance containing granule positions and types
            stiffness: Spring constant k (default 1.0)
        """
        self.lattice = lattice
        self.stiffness = 1.0  # Spring constant k

        # Rest length for BCC nearest neighbor connections
        # In BCC, nearest neighbor distance = a * sqrt(3) / 2
        # Note: rest_length is a scalar distance, not a vector
        # Direction vectors will be computed dynamically between connected granules
        self.rest_length = lattice.unit_cell_edge * np.sqrt(3) / 2

        # Connection topology: [granule_idx] -> [8 possible neighbors]
        # Value -1 indicates no connection (for boundary granules)
        # Max 8 neighbors for BCC structure
        self.links = ti.field(dtype=ti.i32, shape=(lattice.total_granules, 8))

        # Number of active links per granule (for optimization)
        self.links_count = ti.field(dtype=ti.i32, shape=lattice.total_granules)

        # Build the connectivity graph using simple distance-based approach
        self.build_links_simple()

    @ti.kernel
    def build_links_simple(self):
        """
        Connect each granule to its nearest neighbors based on distance.
        This is a brute-force O(N²) approach, but acceptable for small lattices.
        For larger lattices, a spatial partitioning method (e.g., grid or tree)
        would be more efficient."""

        # Initialize all links
        for i in range(self.lattice.total_granules):
            for j in ti.static(range(8)):
                self.links[i, j] = -1
            self.links_count[i] = 0

        # For each granule, find and connect to nearest neighbors
        for i in range(self.lattice.total_granules):
            pos_i = self.lattice.positions[i]
            granule_type = self.lattice.granule_type[i]
            neighbor_count = 0

            # Determine max neighbors based on type
            max_neighbors = 8  # Default for CORE/CENTRAL
            if granule_type == config.TYPE_VERTEX:
                max_neighbors = 1
            elif granule_type == config.TYPE_EDGE:
                max_neighbors = 2
            elif granule_type == config.TYPE_FACE:
                max_neighbors = 4

            # Search through all granules (simplified for small lattices)
            for j in range(self.lattice.total_granules):
                if i != j and neighbor_count < max_neighbors:
                    pos_j = self.lattice.positions[j]

                    # Calculate distance
                    dx = pos_i[0] - pos_j[0]
                    dy = pos_i[1] - pos_j[1]
                    dz = pos_i[2] - pos_j[2]
                    dist_sq = dx * dx + dy * dy + dz * dz

                    # Check if within neighbor distance (with small tolerance)
                    max_dist_sq = (self.rest_length * 1.1) ** 2

                    if dist_sq < max_dist_sq:
                        self.links[i, neighbor_count] = j
                        neighbor_count += 1
                        self.links_count[i] = neighbor_count
