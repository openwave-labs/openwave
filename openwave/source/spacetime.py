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


# ================================================================
# Physics Engine
# ================================================================


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
    - Movement-Ready: Velocity field ready for dynamics,
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
        # 1D array design: Better memory locality, simpler kernels, ready for dynamics
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.granule_type = ti.field(dtype=ti.i32, shape=self.total_granules)
        self.vertex_indices = ti.field(dtype=ti.i32, shape=8)  # indices of 8 corner vertices
        self.vertex_equilibrium = ti.Vector.field(3, dtype=ti.f32, shape=8)  # rest positions
        self.vertex_directions = ti.Vector.field(3, dtype=ti.f32, shape=8)  # direction to center
        self.granule_color = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.front_octant = ti.field(dtype=ti.i32, shape=self.total_granules)

        # Populate the lattice & index granule types
        self.populate_lattice()  # initialize positions and velocities
        self.build_granule_type()  # classifies granules
        self.build_vertex_data()  # builds the 8-element vertex data (indices, equilibrium, directions)
        self.set_granule_colors()  # colors based on granule_type
        self.find_front_octant()  # for block-slicing visualization

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
    def build_granule_type(self):
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
            self.vertex_indices[v] = idx

            # Store equilibrium position for this vertex
            self.vertex_equilibrium[v] = self.positions[idx]

            # Compute normalized direction from vertex to center
            # Vertex position in normalized coordinates (0 or 1 in each dimension)
            vertex_pos = ti.Vector(
                [float(i) / self.grid_size, float(j) / self.grid_size, float(k) / self.grid_size]
            )
            direction = lattice_center - vertex_pos
            self.vertex_directions[v] = direction.normalized()

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

                            # Centers are always CORE or CENTRAL type, so connect to all 8
                            self.links[idx, neighbor_count] = corner_idx
                            neighbor_count += 1
                            self.links_count[idx] = neighbor_count


if __name__ == "__main__":
    print("\n================================================================")
    print("START SMOKE TEST: SPACETIME MODULE")
    print("================================================================")

    import time

    # Create lattice
    universe_edge = 3e-16  # m (default 300 attometers)
    print(f"Creating lattice with universe edge: {universe_edge:.1e} m")

    start_time = time.time()
    lattice = Lattice(universe_edge)
    lattice_time = time.time() - start_time

    print(f"\nLattice Statistics:")
    print(f"  Grid size: {lattice.grid_size}x{lattice.grid_size}x{lattice.grid_size}")
    print(f"  Total granules: {lattice.total_granules:,}")
    print(f"  Unit cell edge: {lattice.unit_cell_edge:.2e} m")
    print(f"  Scale factor: {lattice.scale_factor:.2e} x Planck Length")
    print(f"  Creation time: {lattice_time:.3f} seconds")

    # Create granule
    granule = Granule(lattice.unit_cell_edge)
    print(f"\nGranule Properties:")
    print(f"  Radius: {granule.radius:.2e} m")
    print(f"  Mass: {granule.mass:.2e} kg")

    # Create springs
    print(f"\nBuilding spring connections...")
    start_time = time.time()
    springs = Spring(lattice)
    spring_time = time.time() - start_time

    print(f"Spring Statistics:")
    print(f"  Rest length: {springs.rest_length:.2e} m")
    print(
        f"  Build method: {'distance-based' if lattice.total_granules < 1000 else 'structured BCC'}"
    )
    print(f"  Build time: {spring_time:.3f} seconds")

    # Sample connections (avoiding slice notation)
    print(f"\nSample Connections:")
    sample_indices = [0, 1, lattice.total_granules // 2, lattice.total_granules - 1]
    for idx in sample_indices:
        if idx < lattice.total_granules:
            count = springs.links_count[idx]
            granule_type = lattice.granule_type[idx]
            type_names = {0: "VERTEX", 1: "EDGE", 2: "FACE", 3: "CORE", 4: "CENTRAL"}

            # Get first few connections without slicing
            connections = []
            for j in range(min(count, 3)):  # Show first 3 connections
                connections.append(springs.links[idx, j])

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
            count = springs.links_count[i]
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
            count = springs.links_count[i]
            total_connections += count
            max_connections = max(max_connections, count)
            min_connections = min(min_connections, count)

        print(f"\nConnection Summary:")
        print(f"  Total connections: {total_connections:,}")
        print(f"  Average per granule: {total_connections/lattice.total_granules:.2f}")
        print(f"  Min/Max connections: {min_connections}/{max_connections}")

    print(f"  Total build time: {lattice_time + spring_time:.3f} seconds")

    print("\n================================================================")
    print("END SMOKE TEST: SPACETIME MODULE")
    print("================================================================")

    # Properly exit
    import sys

    sys.exit(0)
