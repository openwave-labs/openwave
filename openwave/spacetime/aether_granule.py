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
        self.front_octant = ti.field(dtype=ti.i32, shape=self.total_granules)
        self.granule_color = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)

        # Populate the lattice & index granule types
        self.populate_latticeBCC()  # initialize position and velocity
        self.build_granule_typeBCC()  # classifies granules
        self.find_front_octantBCC()  # for block-slicing visualization
        self.set_granule_colorBCC()  # colors based on granule_type
        self.set_sliced_plane_objectsBCC()  # set near/far-fields & random probes on sliced planes

    @ti.kernel
    def populate_latticeBCC(self):
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
    def build_granule_typeBCC(self):
        """Classify each granule by its position in the BCC lattice structure.

        Classification:
        - VERTEX (0): 8 corner vertices of the cubic lattice boundary
        - EDGE (1): Granules on the 12 edges (but not corners)
        - FACE (2): Granules on the 6 faces (but not on edges/corners)
        - CORE (3): All other interior granules (not on boundary)
        """
        corner_count = (self.grid_size + 1) ** 3

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
                # Center granules are always in core (offset by 0.5 means never on boundary)
                self.granule_type[idx] = config.TYPE_CORE

    @ti.kernel
    def find_front_octantBCC(self):
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
    def set_granule_colorBCC(self):
        """Assign colors to granules based on their classified type."""
        # Color lookup table (type index -> RGB color)
        color_lut = ti.Matrix(
            [
                config.COLOR_VERTEX[1],  # TYPE_VERTEX = 0
                config.COLOR_EDGE[1],  # TYPE_EDGE = 1
                config.COLOR_FACE[1],  # TYPE_FACE = 2
                config.COLOR_CORE[1],  # TYPE_CORE = 3
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

    def set_sliced_plane_objectsBCC(self, num_circles=0, num_probes=3):
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


class SCGranule:
    """
    Granule Model: The aether consists of "granules".
    Fundamental units that vibrate in harmony and create wave patterns.
    Their collective motion at Planck scale creates all observable phenomena.
    Each granule has a defined radius and mass.
    """

    def __init__(self, unit_cell_edge: float):
        """Initialize scaled-up granule properties based on scaled-up unit cell edge length.

        Args:
            unit_cell_edge: Edge length of the SC unit-cell in meters.
        """
        self.radius = unit_cell_edge / (2 * ti.math.e)  # radius = unit cell edge / 2e
        self.mass = constants.MEDIUM_DENSITY * unit_cell_edge**3  # medium density * cell volume


@ti.data_oriented
class SCLattice:
    """
    Simple Cubic (SC) lattice for spacetime simulation.
    SC topology: cubic unit cells with one granule at each corner.
    Less efficient packing than body-centered cubic (52% vs 68% space filling).

    Performance Design: 1D Arrays with 3D Vectors
    - Memory: Contiguous layout, perfect cache line utilization (64-byte alignment)
    - Compute: Single loop parallelization, no index arithmetic (vs i*dim²+j*dim+k)
    - GPU: Direct thread mapping (thread_i→granule_i), coalesced memory access
    - SC Lattice: Uniform treatment of corner+center granules in single array
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
        Initialize SC lattice and compute scaled-up unit-cell spacing.
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
        # SC has 1 granule per unit cell (each corner is shared by 8 cells = 1/8 per cell * 8 corners)
        init_unit_cell_volume = universe_volume / self.target_granules
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
        self.ewave_res = ti.math.round(constants.EWAVE_LENGTH / self.unit_cell_edge)
        # Compute universe linear resolution, ewavelengths per universe edge
        self.uni_res = universe_edge / constants.EWAVE_LENGTH

        # Total granules: corners only (no centers for SC)
        # Corners: (grid_size + 1)^3
        self.total_granules = (self.grid_size + 1) ** 3

        # Initialize position and velocity 1D arrays
        # 1D array design: Better memory locality, simpler kernels, ready for dynamics
        # position, velocity and acceleration in attometers for f32 precision
        # This scales 1e-17 m values to ~10 am, well within f32 range
        self.position_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.equilibrium_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)  # rest
        self.velocity_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.acceleration_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.granule_type = ti.field(dtype=ti.i32, shape=self.total_granules)
        self.front_octant = ti.field(dtype=ti.i32, shape=self.total_granules)
        self.granule_color = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)

        # Populate the lattice & index granule types
        self.populate_latticeSC()  # initialize position and velocity
        self.build_granule_typeSC()  # classifies granules
        self.find_front_octantSC()  # for block-slicing visualization
        self.set_granule_colorSC()  # colors based on granule_type
        self.set_sliced_plane_objectsSC()  # set near/far-fields & random probes on sliced planes

    @ti.kernel
    def populate_latticeSC(self):
        """Populate SC lattice positions in a 1D array.
        Kernel is properly optimized for Taichi's parallel execution:
        1. Single outermost loop - for idx in range() allows full GPU parallelization
        2. Index decoding - Converts linear index to 3D coordinates using integer division/modulo
        3. No nested loops - All granules computed in parallel across GPU threads
        This structure ensures maximum parallelization on GPU, as each thread independently
        computes one granule's position without synchronization overhead.

        Simple Cubic: Only corner granules are created (no center granules).
        """
        # Parallelize over all granules using single outermost loop
        for idx in range(self.total_granules):
            # SC lattice: only corner granules
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

            self.equilibrium_am[idx] = self.position_am[idx]  # set equilibrium position

            # Initialize velocity to zero for all granules
            self.velocity_am[idx] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def build_granule_typeSC(self):
        """Classify each granule by its position in the SC lattice structure.

        Classification:
        - VERTEX (0): 8 corner vertices of the cubic lattice boundary
        - EDGE (1): Granules on the 12 edges (but not corners)
        - FACE (2): Granules on the 6 faces (but not on edges/corners)
        - CORE (3): All other interior granules (not on boundary)

        Simple Cubic: All granules are corners, so classification is based on grid position only.
        """
        for idx in range(self.total_granules):
            # SC lattice: all granules are corner granules
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

    @ti.kernel
    def find_front_octantSC(self):
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
    def set_granule_colorSC(self):
        """Assign colors to granules based on their classified type."""
        # Color lookup table (type index -> RGB color)
        color_lut = ti.Matrix(
            [
                config.COLOR_VERTEX[1],  # TYPE_VERTEX = 0
                config.COLOR_EDGE[1],  # TYPE_EDGE = 1
                config.COLOR_FACE[1],  # TYPE_FACE = 2
                config.COLOR_CORE[1],  # TYPE_CORE = 3
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

    def set_sliced_plane_objectsSC(self, num_circles=0, num_probes=3):
        """Select random granules from each of the 3 planes exposed by the front octant slice.

        Uses hybrid approach: Python for probe selection, GPU kernel for field circles.

        Args:
            num_circles: Number of concentric circles (1λ, 2λ, 3λ, etc.) to create.
            num_probes: Number of random probe granules per plane.
        """
        # Quick plane collection (small lists, Python is acceptable)
        half_grid = self.grid_size // 2
        grid_dim = self.grid_size + 1

        yz_plane, xz_plane, xy_plane = [], [], []

        # SC lattice: only corner granules on exposed planes
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

        # Select and mark random probes
        probe_color = config.COLOR_PROBE[1]
        for plane in [yz_plane, xz_plane, xy_plane]:
            if len(plane) >= num_probes:
                for idx in random.sample(plane, num_probes):
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

        Simple Cubic: Center is calculated from corner granule positions.
        """
        grid_dim = self.grid_size + 1
        half_grid = self.grid_size // 2
        half_universe = self.universe_edge_am / 2.0
        plane_tolerance = self.unit_cell_edge_am * 0.6
        circle_tolerance = wavelength_am * 0.05

        # Calculate center granule index (corner granule at center of lattice)
        # For SC lattice, this is the corner at (half_grid+1, half_grid+1, half_grid+1)
        center_idx = (
            (half_grid + 1) * grid_dim * grid_dim + (half_grid + 1) * grid_dim + (half_grid + 1)
        )
        center_pos = self.position_am[center_idx]

        # Colors
        field_color = ti.Vector(config.COLOR_FIELDS[1])

        # Process all granules in parallel
        for idx in range(self.total_granules):
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

    print("\n================================================================")
    print("END SMOKE TEST: AETHER-MEDIUM MODULE")
    print("================================================================")

    # Properly exit
    import sys

    sys.exit(0)
