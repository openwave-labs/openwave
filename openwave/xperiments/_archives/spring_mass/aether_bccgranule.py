"""
AETHER-MEDIUM
(AKA: AKASHA @yoga, WUJI @daoism, AETHER @classical)

GRANULE-BASED MEDIUM

Objects Engine @spacetime module.

AETHER is a Wave Medium and Propagates Wave Motion (ENERGY-WAVE).
Modeled as an fluid medium that allows energy to transfer from one point to the next.

"Aether" can refer to the personification of the bright upper sky in Greek mythology,
the classical fifth element or quintessence filling the universe,
or a hypothetical substance once thought to carry light and other electromagnetic waves.
"""

import random

import taichi as ti

from openwave.common import config
from openwave.common import constants
from openwave.common import equations


class BCCGranule:
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
        self.radius = unit_cell_edge / (2 * ti.math.e)  # radius = unit cell edge / 2e
        self.mass = (
            constants.MEDIUM_DENSITY * unit_cell_edge**3 / 2
        )  # mass = medium density * scaled unit cell volume / 2 granules per BCC unit-cell


@ti.data_oriented
class BCCLattice:
    """
    Body-Centered Cubic (BCC) lattice for spacetime simulation.
    BCC topology: cubic unit cells with an additional granule at the center.
    More efficient packing than simple cubic (68% vs 52% space filling).

    Performance Design: 1D Arrays with 3D Vectors
    - Memory: Contiguous layout, perfect cache line utilization (64-byte alignment)
    - Compute: Single loop parallelization, no index arithmetic (vs i*dim²+j*dim+k)
    - GPU: Direct thread mapping (thread_i→granule_i), coalesced memory access
    - BCC Lattice: Uniform treatment of corner+center granules in single array
    Benefits:
    - Simpler updates: One kernel updates all granules
    - Cleaner code: No need to manage multiple arrays
    - Movement-Ready: Velocity field ready for dynamics,
    granules can move freely without grid remapping constraints

    This is why high-performance physics engines (molecular dynamics, N-body simulations)
    universally use 1D arrays for granule data, regardless of spatial dimensionality.
    """

    def __init__(self, universe_edge):
        """
        Initialize BCC lattice and compute scaled-up unit-cell spacing.
        Universe edge (size) and target granules are used to define
        scaled-up unit-cell properties and scale factor.

        Args:
            universe_edge: Simulation domain size, edge length of the cubic universe in meters
        """
        # Compute lattice total energy from energy-wave equation
        self.energy = equations.energy_wave_equation(universe_edge**3)  # in Joules
        self.energy_kWh = equations.J_to_kWh(self.energy)  # in KWh
        self.energy_years = self.energy_kWh / (183230 * 1e9)  # global energy use

        # Set universe properties (simulation domain)
        self.target_granules = config.TARGET_GRANULES
        self.universe_edge = universe_edge
        self.universe_edge_am = universe_edge / constants.ATTOMETTER  # in attometers
        universe_volume = universe_edge**3

        # Compute initial unit-cell properties (before rounding and lattice symmetry)
        # BCC has 2 granules per unit cell (8 corners shared + 1 center)
        init_unit_cell_volume = universe_volume / (self.target_granules / 2)
        init_unit_cell_edge = init_unit_cell_volume ** (1 / 3)  # unit cell edge (a^3 = volume)

        # Calculate grid dimensions (number of unit cells per dimension)
        # Round to nearest odd integer for symmetric grid
        self.raw_size = universe_edge / init_unit_cell_edge
        floor = int(self.raw_size)
        self.grid_size = floor if floor % 2 == 1 else floor + 1

        # Recompute unit-cell edge length based on rounded grid size and scale factor
        self.unit_cell_edge = universe_edge / self.grid_size  # adjusted unit cell edge length
        self.unit_cell_edge_am = self.unit_cell_edge / constants.ATTOMETTER  # in attometers
        self.scale_factor = self.unit_cell_edge / (
            2 * ti.math.e * constants.PLANCK_LENGTH
        )  # linear scale factor from Planck length, increases computability

        # Compute energy-wave linear resolution, sampling rate
        # granules per wavelength, should be >2 for Nyquist
        self.ewave_res = ti.math.round(constants.EWAVE_LENGTH / self.unit_cell_edge * 2)
        # Compute universe linear resolution, ewavelengths per universe edge
        self.uni_res = universe_edge / constants.EWAVE_LENGTH

        # Total granules: corners + centers
        # Corners: (grid_size + 1)^3, Centers: grid_size^3
        corner_count = (self.grid_size + 1) ** 3
        center_count = self.grid_size**3
        self.total_granules = corner_count + center_count

        # Initialize position and velocity 1D arrays
        # 1D array design: Better memory locality, simpler kernels, ready for dynamics
        # position, velocity and acceleration in attometers for f32 precision
        # This scales 1e-17 m values to ~10 am, well within f32 range
        self.position_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.equilibrium_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)  # rest
        self.velocity_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.acceleration_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.granule_type = ti.field(dtype=ti.i32, shape=self.total_granules)
        self.center_direction = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.center_distance_am = ti.field(dtype=ti.f32, shape=self.total_granules)
        self.vertex_index = ti.field(dtype=ti.i32, shape=8)  # indices of 8 corner vertices
        self.vertex_equilibrium_am = ti.Vector.field(3, dtype=ti.f32, shape=8)  # rest position
        self.vertex_center_direction = ti.Vector.field(3, dtype=ti.f32, shape=8)
        self.front_octant = ti.field(dtype=ti.i32, shape=self.total_granules)
        self.granule_color = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)

        # Populate the lattice & index granule types
        self.populate_lattice()  # initialize position and velocity
        self.build_granule_type()  # classifies granules
        self.build_center_vectors()  # builds direction vectors for all granules to center
        self.build_vertex_data()  # builds the 8-element vertex data (indices, equilibrium, directions)
        self.find_front_octant()  # for block-slicing visualization
        self.set_granule_color()  # colors based on granule_type
        self.set_sliced_plane_objects()  # set near/far-fields & random probes on sliced planes

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

                # Store positions in attometers
                self.position_am[idx] = ti.Vector(
                    [
                        i * self.unit_cell_edge_am,
                        j * self.unit_cell_edge_am,
                        k * self.unit_cell_edge_am,
                    ]
                )
            else:
                # Center granule: decode position with offset
                center_idx = idx - corner_count
                i = center_idx // (self.grid_size * self.grid_size)
                j = (center_idx % (self.grid_size * self.grid_size)) // self.grid_size
                k = center_idx % self.grid_size

                offset = self.unit_cell_edge_am / 2
                # Store positions in attometers
                self.position_am[idx] = ti.Vector(
                    [
                        (i * self.unit_cell_edge_am + offset),
                        (j * self.unit_cell_edge_am + offset),
                        (k * self.unit_cell_edge_am + offset),
                    ]
                )

            self.equilibrium_am[idx] = self.position_am[idx]  # set equilibrium position

            # Initialize velocity to zero for all granules
            self.velocity_am[idx] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def build_granule_type(self):
        """Classify each granule by its position in the BCC lattice structure.

        Classification:
        - VERTEX (0): 8 corner vertices of the cubic lattice boundary
        - EDGE (1): Granules on the 12 edges (but not corners)
        - FACE (2): Granules on the 6 faces (but not on edges/corners)
        - CORE (3): All other interior granules (not on boundary)
        - CENTER (4): Single granule at exact center of lattice
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
                    self.granule_type[idx] = 4
                else:
                    # Center granules are always in core (offset by 0.5 means never on boundary)
                    self.granule_type[idx] = config.TYPE_CORE

    @ti.kernel
    def build_center_vectors(self):
        """Compute distance & direction vectors from all granules to lattice center.

        For each granule in the position_am array, computes a normalized direction vector
        pointing from the granule's position toward the geometric center of the lattice.
        The lattice center is at (0.5, 0.5, 0.5) in normalized coordinates.

        Direction vectors are stored in the directions field and used for radial operations
        such as compression/expansion forces or wave propagation from the center.

        Special case: The central granule (TYPE_CENTER) has zero distance and a default
        direction vector, ensuring it remains stationary in radial wave patterns.
        """
        # Lattice center in normalized coordinates (0.5, 0.5, 0.5)
        lattice_center = ti.Vector([0.5, 0.5, 0.5])

        # Process all granules in the lattice
        for idx in range(self.total_granules):
            # Special case: Central granule should have zero distance and no direction
            if self.granule_type[idx] == 4:
                # Central granule: zero distance, arbitrary direction (won't be used)
                self.center_direction[idx] = ti.Vector([0.0, 0.0, 0.0])
                self.center_distance_am[idx] = 0.0
            else:
                # Convert granule position from attometers to normalized coordinates [0, 1]
                # by dividing by the total universe edge length in attometers
                pos_normalized = ti.Vector(
                    [
                        self.position_am[idx][0] / self.universe_edge_am,
                        self.position_am[idx][1] / self.universe_edge_am,
                        self.position_am[idx][2] / self.universe_edge_am,
                    ]
                )

                # Compute direction vector from granule position to lattice center
                direction = lattice_center - pos_normalized

                # Normalize and store the direction and distance to center vectors
                self.center_direction[idx] = direction.normalized()
                self.center_distance_am[idx] = (
                    direction.norm() * self.universe_edge_am
                )  # in attometers

    @ti.kernel
    def build_vertex_data(self):
        """Directly compute indices of 8 corner vertices and their direction vectors to center.
        Uses the corner granule indexing formula: idx = i*(grid_dim^2) + j*grid_dim + k
        where grid_dim = grid_size + 1, and i,j,k ∈ {0, grid_size}
        Also computes normalized direction vectors from each vertex to lattice center (0.5, 0.5, 0.5).
        """
        grid_dim = self.grid_size + 1
        lattice_center = ti.Vector([0.5, 0.5, 0.5])

        # Map each of 8 vertices (binary encoding of corner position)
        for v in range(8):
            i = self.grid_size if (v & 4) else 0
            j = self.grid_size if (v & 2) else 0
            k = self.grid_size if (v & 1) else 0
            idx = i * (grid_dim * grid_dim) + j * grid_dim + k
            self.vertex_index[v] = idx

            # Store equilibrium position for this vertex
            self.vertex_equilibrium_am[v] = self.position_am[idx]

            # Compute normalized direction from vertex to center
            # Vertex position in normalized coordinates (0 or 1 in each dimension)
            vertex_pos = ti.Vector(
                [float(i) / self.grid_size, float(j) / self.grid_size, float(k) / self.grid_size]
            )
            direction = lattice_center - vertex_pos
            self.vertex_center_direction[v] = direction.normalized()

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
                    self.position_am[i][0] > self.universe_edge_am / 2
                    and self.position_am[i][1] > self.universe_edge_am / 2
                    and self.position_am[i][2] > self.universe_edge_am / 2
                )
                else 0
            )

    @ti.kernel
    def set_granule_color(self):
        """Assign colors to granules based on their classified type."""
        # Color lookup table (type index -> RGB color)
        color_lut = ti.Matrix(
            [
                config.COLOR_VERTEX[1],  # TYPE_VERTEX = 0
                config.COLOR_EDGE[1],  # TYPE_EDGE = 1
                config.COLOR_FACE[1],  # TYPE_FACE = 2
                config.COLOR_CORE[1],  # TYPE_CORE = 3
                config.BLACK[1],  # TYPE_CENTER = 4
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

    def set_sliced_plane_objects(self, num_circles=0, num_probes=3):
        """Select random granules from each of the 3 planes exposed by the front octant slice.

        Uses hybrid approach: Python for probe selection, GPU kernel for field circles.

        Args:
            num_circles: Number of concentric circles (1λ, 2λ, 3λ, etc.) to create.
            num_probes: Number of random probe granules per plane.
        """
        # Quick plane collection (small lists, Python is acceptable)
        corner_count = (self.grid_size + 1) ** 3
        half_grid = self.grid_size // 2
        grid_dim = self.grid_size + 1

        yz_plane, xz_plane, xy_plane = [], [], []

        # Corner granules on exposed planes
        for i in range(grid_dim):
            for j in range(grid_dim):
                for k in range(grid_dim):
                    idx = i * (grid_dim**2) + j * grid_dim + k
                    if i == half_grid + 1 and j > half_grid and k > half_grid:
                        yz_plane.append(idx)
                    if j == half_grid + 1 and i > half_grid and k > half_grid:
                        xz_plane.append(idx)
                    if k == half_grid + 1 and i > half_grid and j > half_grid:
                        xy_plane.append(idx)

        # Center granules on exposed planes
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    idx = corner_count + i * self.grid_size**2 + j * self.grid_size + k
                    if i == half_grid and j >= half_grid and k >= half_grid:
                        yz_plane.append(idx)
                    if j == half_grid and i >= half_grid and k >= half_grid:
                        xz_plane.append(idx)
                    if k == half_grid and i >= half_grid and j >= half_grid:
                        xy_plane.append(idx)

        # Select and mark random probes
        probe_color = config.COLOR_PROBE[1]
        for plane in [yz_plane, xz_plane, xy_plane]:
            if len(plane) >= num_probes:
                for idx in random.sample(plane, num_probes):
                    if self.granule_type[idx] != 4:
                        self.granule_color[idx] = ti.Vector(
                            [probe_color[0], probe_color[1], probe_color[2]]
                        )

        # Convert energy wavelength and call GPU kernel for field circles
        wavelength_am = constants.EWAVE_LENGTH / constants.ATTOMETTER
        self._mark_sliced_plane_objects(wavelength_am, num_circles)

    @ti.kernel
    def _mark_sliced_plane_objects(self, wavelength_am: ti.f32, num_circles: ti.i32):  # type: ignore
        """GPU kernel to mark field circles on exposed planes.

        This kernel processes all granules in parallel, marking:
        - Field circles at 1λ, 2λ, etc. from the center granule
        """
        corner_count = (self.grid_size + 1) ** 3
        half_grid = self.grid_size // 2
        half_universe = self.universe_edge_am / 2.0
        plane_tolerance = self.unit_cell_edge_am * 0.6
        circle_tolerance = wavelength_am * 0.05

        # Calculate center granule index
        center_idx = (
            corner_count
            + half_grid * self.grid_size * self.grid_size
            + half_grid * self.grid_size
            + half_grid
        )
        center_pos = self.position_am[center_idx]

        # Colors
        field_color = ti.Vector(config.COLOR_FIELDS[1])

        # Process all granules in parallel
        for idx in range(self.total_granules):
            if self.granule_type[idx] != 4:
                pos = self.position_am[idx]

                # Check if granule is on one of the exposed planes and mark field circles
                # YZ plane (x at boundary)
                if (
                    ti.abs(pos[0] - half_universe) < plane_tolerance
                    and pos[1] > half_universe
                    and pos[2] > half_universe
                ):
                    # Calculate 2D distance in YZ plane
                    dy = pos[1] - center_pos[1]
                    dz = pos[2] - center_pos[2]
                    dist_2d = ti.sqrt(dy * dy + dz * dz)

                    # Check if on a target circle
                    for n in range(1, num_circles + 1):
                        if ti.abs(dist_2d - ti.f32(n) * wavelength_am) <= circle_tolerance:
                            self.granule_color[idx] = field_color
                            break

                # XZ plane (y at boundary)
                elif (
                    ti.abs(pos[1] - half_universe) < plane_tolerance
                    and pos[0] > half_universe
                    and pos[2] > half_universe
                ):
                    # Calculate 2D distance in XZ plane
                    dx = pos[0] - center_pos[0]
                    dz = pos[2] - center_pos[2]
                    dist_2d = ti.sqrt(dx * dx + dz * dz)

                    # Check if on a target circle
                    for n in range(1, num_circles + 1):
                        if ti.abs(dist_2d - ti.f32(n) * wavelength_am) <= circle_tolerance:
                            self.granule_color[idx] = field_color
                            break

                # XY plane (z at boundary)
                elif (
                    ti.abs(pos[2] - half_universe) < plane_tolerance
                    and pos[0] > half_universe
                    and pos[1] > half_universe
                ):
                    # Calculate 2D distance in XY plane
                    dx = pos[0] - center_pos[0]
                    dy = pos[1] - center_pos[1]
                    dist_2d = ti.sqrt(dx * dx + dy * dy)

                    # Check if on a target circle
                    for n in range(1, num_circles + 1):
                        if ti.abs(dist_2d - ti.f32(n) * wavelength_am) <= circle_tolerance:
                            self.granule_color[idx] = field_color
                            break


@ti.data_oriented
class BCCNeighbors:
    """
    8-way neighbors couplings between Granules in lattice.
    Models connections with 8-way, 4-way, 2-way, or 1-way topology
    depending on granule type (core/center, face, edge, vertex).

    In BCC lattice, each interior granule has 8 nearest neighbors at distance a*√3/2)
    where a is the unit cell edge length.
    """

    def __init__(self, lattice: BCCLattice):
        """
        Initialize neighbors links for the BCC Lattice.

        Args:
            lattice: Lattice instance containing granule position and types
        """
        self.lattice = lattice

        # Rest length for BCC nearest neighbor connections
        # In BCC, nearest neighbor distance = a * sqrt(3) / 2
        # Note: rest_length is a scalar distance, not a vector
        # Direction vectors will be computed dynamically between connected granules
        self.rest_length = lattice.unit_cell_edge * ti.math.sqrt(3) / 2
        # For link building, use attometer-scaled rest_length to match position units
        self.rest_length_am = lattice.unit_cell_edge_am * ti.math.sqrt(3) / 2

        # Natural frequency for wave propagation at speed of light
        # For wave speed c in lattice with spacing L: f_n = c / (2L)
        # where λ_lattice ≈ 2L (minimum resolvable wavelength in discrete lattice)
        self.natural_frequency = constants.EWAVE_SPEED / (2 * self.rest_length)  # Hz

        # Connection topology: [granule_idx] -> [8 possible neighbors]
        # Value -1 indicates no connection (for boundary granules)
        # Max 8 neighbors for BCC structure
        self.links = ti.field(dtype=ti.i32, shape=(lattice.total_granules, 8))

        # Number of active links per granule (for optimization)
        self.links_count = ti.field(dtype=ti.i32, shape=lattice.total_granules)

        # Build the connectivity graph using appropriate method based on lattice size
        if lattice.total_granules < 1000:
            # For small lattices, use simple distance-based search
            self.build_links_simple()
        else:
            # For large lattices, use structured BCC approach
            self.build_links_structured()

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
            pos_i = self.lattice.position_am[i]
            granule_type = self.lattice.granule_type[i]
            neighbor_count = 0

            # Determine max neighbors based on type
            max_neighbors = 8  # Default for CORE/CENTER
            if granule_type == config.TYPE_VERTEX:
                max_neighbors = 1
            elif granule_type == config.TYPE_EDGE:
                max_neighbors = 2
            elif granule_type == config.TYPE_FACE:
                max_neighbors = 4

            # Search through all granules (simplified for small lattices)
            for j in range(self.lattice.total_granules):
                if i != j and neighbor_count < max_neighbors:
                    pos_j = self.lattice.position_am[j]

                    # Calculate distance
                    dx = pos_i[0] - pos_j[0]
                    dy = pos_i[1] - pos_j[1]
                    dz = pos_i[2] - pos_j[2]
                    dist_sq = dx * dx + dy * dy + dz * dz

                    # Check if within neighbor distance (with small tolerance)
                    # Use scaled rest_length to match attometer position units
                    max_dist_sq = (self.rest_length_am * 1.1) ** 2

                    if dist_sq < max_dist_sq:
                        self.links[i, neighbor_count] = j
                        neighbor_count += 1
                        self.links_count[i] = neighbor_count

    @ti.kernel
    def build_links_structured(self):
        """
        Efficient O(n) connection building using BCC lattice structure.

        This method directly computes neighbor indices based on the BCC topology
        rather than searching through all granules. Much faster for large lattices.

        In BCC, each granule has 8 nearest neighbors at specific offsets:
        - Corner granules connect to 4 centers in adjacent cells
        - Center granules connect to 8 corners of their cell
        """
        corner_count = (self.lattice.grid_size + 1) ** 3
        grid_size = self.lattice.grid_size

        # Initialize all links
        for i in range(self.lattice.total_granules):
            for j in ti.static(range(8)):
                self.links[i, j] = -1
            self.links_count[i] = 0

        # Process each granule
        for idx in range(self.lattice.total_granules):
            granule_type = self.lattice.granule_type[idx]
            neighbor_count = 0

            if idx < corner_count:
                # Corner granule: decode 3D grid position
                grid_dim = grid_size + 1
                i = idx // (grid_dim * grid_dim)
                j = (idx % (grid_dim * grid_dim)) // grid_dim
                k = idx % grid_dim

                # For corner granules, connect to adjacent center granules
                # Determine max connections based on granule type
                max_neighbors = 8
                if granule_type == config.TYPE_VERTEX:
                    max_neighbors = 1
                elif granule_type == config.TYPE_EDGE:
                    max_neighbors = 2
                elif granule_type == config.TYPE_FACE:
                    max_neighbors = 4

                # Each center is offset by (-0.5, -0.5, -0.5) from corners
                for di in ti.static(range(2)):
                    for dj in ti.static(range(2)):
                        for dk in ti.static(range(2)):
                            if neighbor_count < max_neighbors:
                                # Calculate center cell indices
                                ci = i + di - 1
                                cj = j + dj - 1
                                ck = k + dk - 1

                                # Check bounds for center cell
                                if (
                                    ci >= 0
                                    and ci < grid_size
                                    and cj >= 0
                                    and cj < grid_size
                                    and ck >= 0
                                    and ck < grid_size
                                ):
                                    # Calculate center granule index
                                    center_idx = (
                                        corner_count
                                        + ci * grid_size * grid_size
                                        + cj * grid_size
                                        + ck
                                    )

                                    self.links[idx, neighbor_count] = center_idx
                                    neighbor_count += 1
                                    self.links_count[idx] = neighbor_count
            else:
                # Center granule: decode position
                center_idx = idx - corner_count
                i = center_idx // (grid_size * grid_size)
                j = (center_idx % (grid_size * grid_size)) // grid_size
                k = center_idx % grid_size

                # For center granules, connect to 8 surrounding corners
                for di in ti.static(range(2)):
                    for dj in ti.static(range(2)):
                        for dk in ti.static(range(2)):
                            # Calculate corner indices
                            ci = i + di
                            cj = j + dj
                            ck = k + dk

                            # Corner index calculation
                            grid_dim = grid_size + 1
                            corner_idx = ci * grid_dim * grid_dim + cj * grid_dim + ck

                            # Centers are always CORE or CENTER type, so connect to all 8
                            self.links[idx, neighbor_count] = corner_idx
                            neighbor_count += 1
                            self.links_count[idx] = neighbor_count


if __name__ == "__main__":
    print("\n================================================================")
    print("SMOKE TEST: AETHER-MEDIUM MODULE")
    print("================================================================")

    import time

    ti.init(arch=ti.gpu)

    # ================================================================
    # Parameters & Subatomic Objects Instantiation
    # ================================================================

    UNIVERSE_EDGE = (
        4 * constants.EWAVE_LENGTH
    )  # m, simulation domain, edge length of cubic universe

    lattice = BCCLattice(UNIVERSE_EDGE)
    start_time = time.time()
    lattice_time = time.time() - start_time

    print(f"\nLattice Statistics:")
    print(f"  Universe edge: {UNIVERSE_EDGE:.1e} m")
    print(f"  Granule count: {lattice.total_granules:,}")
    print(f"  Grid size: {lattice.grid_size}x{lattice.grid_size}x{lattice.grid_size}")
    print(f"  Unit cell edge: {lattice.unit_cell_edge:.2e} m")
    print(f"  Scale factor: {lattice.scale_factor:.2e} x Planck Length")
    print(f"  Creation time: {lattice_time:.3f} seconds")

    # Create granule
    granule = BCCGranule(lattice.unit_cell_edge)
    print(f"\nGranule Properties:")
    print(f"  Radius: {granule.radius:.2e} m")
    print(f"  Mass: {granule.mass:.2e} kg")

    # Create links
    print(f"\nBuilding neighbor connections...")
    start_time = time.time()
    neighbors = BCCNeighbors(lattice)
    neighbor_time = time.time() - start_time

    print(f"Neighbor Statistics:")
    print(f"  Rest length: {neighbors.rest_length:.2e} m")
    print(
        f"  Build method: {'distance-based' if lattice.total_granules < 1000 else 'structured BCC'}"
    )
    print(f"  Build time: {neighbor_time:.3f} seconds")

    # Sample connections (avoiding slice notation)
    print(f"\nSample Connections:")
    sample_indices = [0, 1, lattice.total_granules // 2, lattice.total_granules - 1]
    for idx in sample_indices:
        if idx < lattice.total_granules:
            count = neighbors.links_count[idx]
            granule_type = lattice.granule_type[idx]
            type_names = {0: "VERTEX", 1: "EDGE", 2: "FACE", 3: "CORE", 4: "CENTER"}

            # Get first few connections without slicing
            connections = []
            for j in range(min(count, 3)):  # Show first 3 connections
                connections.append(neighbors.links[idx, j])

            print(
                f"  Granule {idx:6d} ({type_names.get(granule_type, 'UNKNOWN'):7s}): "
                f"{count} links -> {connections}{'...' if count > 3 else ''}"
            )

    # Overall statistics - with progress for large lattices
    print(f"\nComputing connection statistics...")

    if lattice.total_granules > 10000:
        # For large lattices, sample instead of checking all
        print(f"  (Sampling 1000 granules from {lattice.total_granules:,} total)")
        sample_size = 1000
        import random

        sample_indices = random.sample(range(lattice.total_granules), sample_size)

        total_connections = 0
        max_connections = 0
        min_connections = 1000

        for i in sample_indices:
            count = neighbors.links_count[i]
            total_connections += count
            max_connections = max(max_connections, count)
            min_connections = min(min_connections, count)

        # Estimate total connections
        avg_per_granule = total_connections / sample_size
        estimated_total = int(avg_per_granule * lattice.total_granules)

        print(f"\nConnection Summary (estimated from sample):")
        print(f"  Est. total connections: ~{estimated_total:,}")
        print(f"  Est. average per granule: {avg_per_granule:.2f}")
        print(f"  Sample min/max connections: {min_connections}/{max_connections}")
    else:
        # For small lattices, check all
        total_connections = 0
        max_connections = 0
        min_connections = 1000

        for i in range(lattice.total_granules):
            count = neighbors.links_count[i]
            total_connections += count
            max_connections = max(max_connections, count)
            min_connections = min(min_connections, count)

        print(f"\nConnection Summary:")
        print(f"  Total connections: {total_connections:,}")
        print(f"  Average per granule: {total_connections/lattice.total_granules:.2f}")
        print(f"  Min/Max connections: {min_connections}/{max_connections}")

    print(f"  Total build time: {lattice_time + neighbor_time:.3f} seconds")

    print("\n================================================================")
    print("END SMOKE TEST: AETHER-MEDIUM MODULE")
    print("================================================================")

    # Properly exit
    import sys

    sys.exit(0)
