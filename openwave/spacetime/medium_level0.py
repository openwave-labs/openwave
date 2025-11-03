"""
WAVE-MEDIUM

LEVEL-0: GRANULE-BASED MEDIUM

Objects Engine @spacetime module.

WAVE-MEDIUM propagates Wave Motion (ENERGY-WAVE).
Modeled as a fluid-like medium that allows energy to transfer from one point to the next.
"""

import random

import taichi as ti

from openwave.common import config
from openwave.common import constants
from openwave.common import equations


class BCCGranule:
    """
    Granule Model: The Medium consists of "granules".
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

    def __init__(self, init_universe_size, theme="OCEAN"):
        """
        Initialize BCC lattice and compute scaled-up unit-cell spacing.
        Universe size and target granules are used to define
        scaled-up unit-cell properties and scale factor.

        Args:
            init_universe_size: Requested simulation domain size [x, y, z] in meters (can be asymmetric)
                Will be rounded to fit integer number of cubic unit-cells.
            theme: Color theme name from config.py (OCEAN, DESERT, FOREST, etc.)
        """
        # Compute initial lattice properties (before rounding and lattice symmetry)
        init_universe_volume = (
            init_universe_size[0] * init_universe_size[1] * init_universe_size[2]
        )
        self.target_granules = config.TARGET_GRANULES

        # Calculate unit cell properties
        # CRITICAL: Unit cell must remain cubic (same edge length on all axes)
        # This preserves crystal structure. Only the NUMBER of cells varies per axis.
        unit_cell_volume = init_universe_volume / (self.target_granules / 2)  # BCC = 2 /unit-cell
        self.unit_cell_edge = unit_cell_volume ** (1 / 3)  # a^3 = volume
        self.unit_cell_edge_am = self.unit_cell_edge / constants.ATTOMETER

        # Calculate grid dimensions (number of unit cells per dimension) - asymmetric
        self.raw_size = [
            init_universe_size[0] / self.unit_cell_edge,
            init_universe_size[1] / self.unit_cell_edge,
            init_universe_size[2] / self.unit_cell_edge,
        ]
        # Round to nearest odd integer for symmetric grid per axis
        self.grid_size = [
            int(self.raw_size[0]) if int(self.raw_size[0]) % 2 == 1 else int(self.raw_size[0]) + 1,
            int(self.raw_size[1]) if int(self.raw_size[1]) % 2 == 1 else int(self.raw_size[1]) + 1,
            int(self.raw_size[2]) if int(self.raw_size[2]) % 2 == 1 else int(self.raw_size[2]) + 1,
        ]

        # Recompute actual universe dimensions to fit integer number of cubic unit cells
        self.universe_size = [
            self.grid_size[0] * self.unit_cell_edge,
            self.grid_size[1] * self.unit_cell_edge,
            self.grid_size[2] * self.unit_cell_edge,
        ]
        self.universe_size_am = [
            self.universe_size[0] / constants.ATTOMETER,
            self.universe_size[1] / constants.ATTOMETER,
            self.universe_size[2] / constants.ATTOMETER,
        ]
        self.max_universe_edge = max(
            self.grid_size[0] * self.unit_cell_edge,
            self.grid_size[1] * self.unit_cell_edge,
            self.grid_size[2] * self.unit_cell_edge,
        )
        self.max_universe_edge_am = self.max_universe_edge / constants.ATTOMETER
        self.max_grid_size = max(
            self.grid_size[0],
            self.grid_size[1],
            self.grid_size[2],
        )
        self.universe_volume = (
            self.universe_size[0] * self.universe_size[1] * self.universe_size[2]
        )

        # Scale factor based on cubic unit cell edge
        self.scale_factor = self.unit_cell_edge / (
            2 * ti.math.e * constants.PLANCK_LENGTH
        )  # linear scale factor from Planck length, increases computability

        # Compute energy-wave linear resolution, sampling rate
        # granules per wavelength, should be >10 for Nyquist (same for all axes with cubic cells)
        self.ewave_res = ti.math.round(constants.EWAVE_LENGTH / self.unit_cell_edge * 2)
        # Compute universe linear resolution, ewavelengths per universe edge (per axis - can differ)
        self.max_uni_res = self.max_universe_edge / constants.EWAVE_LENGTH

        # Total granules: corners + centers (asymmetric grid)
        # Corners: (grid_size[0] + 1) * (grid_size[1] + 1) * (grid_size[2] + 1)
        # Centers: grid_size[0] * grid_size[1] * grid_size[2]
        corner_count = (self.grid_size[0] + 1) * (self.grid_size[1] + 1) * (self.grid_size[2] + 1)
        center_count = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.total_granules = corner_count + center_count

        # Compute lattice total energy from energy-wave equation
        self.energy = equations.energy_wave_equation(self.universe_volume)  # in Joules
        self.energy_kWh = equations.J_to_kWh(self.energy)  # in KWh
        self.energy_years = self.energy_kWh / (183230 * 1e9)  # global energy use

        # Initialize position and velocity 1D arrays
        # 1D array design: Better memory locality, simpler kernels, ready for dynamics
        # position, velocity in attometers for f32 precision
        # This scales 1e-17 m values to ~10 am, well within f32 range
        self.position_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.equilibrium_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)  # rest
        self.velocity_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.granule_type = ti.field(dtype=ti.i32, shape=self.total_granules)
        self.front_octant = ti.field(dtype=ti.i32, shape=self.total_granules)
        self.granule_type_color = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.granule_var_color = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)

        # Populate the lattice & index granule types
        self.populate_latticeBCC()  # initialize position and velocity
        self.build_granule_typeBCC()  # classifies granules
        self.find_front_octantBCC()  # for block-slicing visualization
        self.set_granule_type_colorBCC(theme)  # colors based on granule_type
        self.set_sliced_plane_objectsBCC()  # set near/far-fields & random probes on sliced planes

    @ti.kernel
    def populate_latticeBCC(self):
        """Populate BCC lattice positions in a 1D array with asymmetric grid support.
        Kernel is properly optimized for Taichi's parallel execution:
        1. Single outermost loop - for idx in range() allows full GPU parallelization
        2. Index decoding - Converts linear index to 3D coordinates using integer division/modulo
        3. No nested loops - All granules computed in parallel across GPU threads
        4. Efficient branching - Simple if/else to determine corner vs center granules
        This structure ensures maximum parallelization on GPU, as each thread independently
        computes one granule's position without synchronization overhead.

        Asymmetric grid: Different grid sizes for x, y, z dimensions.
        """
        # Parallelize over all granules using single outermost loop
        for idx in range(self.total_granules):
            # Determine if this is a corner or center granule
            corner_count = (
                (self.grid_size[0] + 1) * (self.grid_size[1] + 1) * (self.grid_size[2] + 1)
            )

            if idx < corner_count:
                # Corner granule: decode 3D position from linear index (asymmetric)
                grid_dim_x = self.grid_size[0] + 1
                grid_dim_y = self.grid_size[1] + 1
                grid_dim_z = self.grid_size[2] + 1

                i = idx // (grid_dim_y * grid_dim_z)
                j = (idx % (grid_dim_y * grid_dim_z)) // grid_dim_z
                k = idx % grid_dim_z

                # Store positions in attometers (cubic unit cells)
                self.position_am[idx] = ti.Vector(
                    [
                        i * self.unit_cell_edge_am,
                        j * self.unit_cell_edge_am,
                        k * self.unit_cell_edge_am,
                    ]
                )
            else:
                # Center granule: decode position with offset (asymmetric grid)
                center_idx = idx - corner_count
                i = center_idx // (self.grid_size[1] * self.grid_size[2])
                j = (center_idx % (self.grid_size[1] * self.grid_size[2])) // self.grid_size[2]
                k = center_idx % self.grid_size[2]

                # Store positions in attometers (cubic unit cells with offset)
                offset = self.unit_cell_edge_am / 2
                self.position_am[idx] = ti.Vector(
                    [
                        i * self.unit_cell_edge_am + offset,
                        j * self.unit_cell_edge_am + offset,
                        k * self.unit_cell_edge_am + offset,
                    ]
                )

            self.equilibrium_am[idx] = self.position_am[idx]  # set equilibrium position

            # Initialize velocity to zero for all granules
            self.velocity_am[idx] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def build_granule_typeBCC(self):
        """Classify each granule by its position in the BCC lattice structure.

        Granule Type:
        - VERTEX (0): 8 corner vertices of the lattice boundary
        - EDGE (1): Granules on the 12 edges (but not corners)
        - FACE (2): Granules on the 6 faces (but not on edges/corners)
        - CORE (3): All other interior granules (not on boundary)

        Asymmetric grid: Works with different grid sizes for x, y, z dimensions.
        """
        corner_count = (self.grid_size[0] + 1) * (self.grid_size[1] + 1) * (self.grid_size[2] + 1)

        for idx in range(self.total_granules):
            if idx < corner_count:
                # Corner granule: decode 3D grid position (asymmetric)
                grid_dim_y = self.grid_size[1] + 1
                grid_dim_z = self.grid_size[2] + 1

                i = idx // (grid_dim_y * grid_dim_z)
                j = (idx % (grid_dim_y * grid_dim_z)) // grid_dim_z
                k = idx % grid_dim_z

                # Count how many coordinates are at boundaries (0 or grid_size per axis)
                at_boundary = 0
                if i == 0 or i == self.grid_size[0]:
                    at_boundary += 1
                if j == 0 or j == self.grid_size[1]:
                    at_boundary += 1
                if k == 0 or k == self.grid_size[2]:
                    at_boundary += 1

                if at_boundary == 3:
                    self.granule_type[idx] = 0  # Granule Type: VERTEX (0)
                elif at_boundary == 2:
                    self.granule_type[idx] = 1  # Granule Type: EDGE (1)
                elif at_boundary == 1:
                    self.granule_type[idx] = 2  # Granule Type: FACE (2)
                else:
                    self.granule_type[idx] = 3  # Granule Type: CORE (3)
            else:
                # Center granules are always in core (offset by 0.5 means never on boundary)
                self.granule_type[idx] = 3  # Granule Type: CORE (3)

    @ti.kernel
    def find_front_octantBCC(self):
        """Mark granules in the front octant (for block-slicing visualization).

        Front octant = granules where x, y, z > universe_size/2 (per axis)
        Used for rendering: 0 = render, 1 = skip (for see-through effect)

        Asymmetric grid: Uses different thresholds for x, y, z dimensions.
        """
        for i in range(self.total_granules):
            # Mark if granule is in the front 1/8th block, > halfway on all axes
            # 0 = not in front octant, 1 = in front octant
            self.front_octant[i] = (
                1
                if (
                    self.position_am[i][0] > self.universe_size_am[0] / 2
                    and self.position_am[i][1] > self.universe_size_am[1] / 2
                    and self.position_am[i][2] > self.universe_size_am[2] / 2
                )
                else 0
            )

    def set_granule_type_colorBCC(self, theme="OCEAN"):
        """Assign colors to granules based on their classified type and color theme.

        Args:
            theme: Color theme name from config.py (OCEAN, DESERT, FOREST, etc.)
        """
        # Get theme configuration from config module
        theme_config = getattr(config, theme, config.OCEAN)

        # Extract color values from theme
        color_vertex = ti.Vector(theme_config["COLOR_VERTEX"][1])
        color_edge = ti.Vector(theme_config["COLOR_EDGE"][1])
        color_face = ti.Vector(theme_config["COLOR_FACE"][1])
        color_core = ti.Vector(config.COLOR_MEDIUM[1])

        # Call GPU kernel with theme colors
        self._apply_granule_type_colorsBCC(color_vertex, color_edge, color_face, color_core)

    @ti.kernel
    def _apply_granule_type_colorsBCC(
        self,
        color_vertex: ti.types.vector(3, ti.f32),  # type: ignore
        color_edge: ti.types.vector(3, ti.f32),  # type: ignore
        color_face: ti.types.vector(3, ti.f32),  # type: ignore
        color_core: ti.types.vector(3, ti.f32),  # type: ignore
    ):
        """GPU kernel to apply colors based on granule type.

        Args:
            color_vertex: RGB color for VERTEX granules (type 0)
            color_edge: RGB color for EDGE granules (type 1)
            color_face: RGB color for FACE granules (type 2)
            color_core: RGB color for CORE granules (type 3)
        """
        # Color lookup table (type index -> RGB color)
        color_lut = ti.Matrix(
            [
                [color_vertex[0], color_vertex[1], color_vertex[2]],  # VERTEX (0)
                [color_edge[0], color_edge[1], color_edge[2]],  # EDGE (1)
                [color_face[0], color_face[1], color_face[2]],  # FACE (2)
                [color_core[0], color_core[1], color_core[2]],  # CORE (3)
            ]
        )

        for i in range(self.total_granules):
            granule_type = self.granule_type[i]
            if 0 <= granule_type <= 3:
                self.granule_type_color[i] = ti.Vector(
                    [
                        color_lut[granule_type, 0],
                        color_lut[granule_type, 1],
                        color_lut[granule_type, 2],
                    ]
                )
            else:
                self.granule_type_color[i] = ti.Vector([0.1, 0.6, 0.9])  # Light Blue for undefined

    def set_sliced_plane_objectsBCC(self, num_circles=0, num_probes=3):
        """Select random granules from each of the 3 planes exposed by the front octant slice.

        Uses hybrid approach: Python for probe selection, GPU kernel for field circles.

        Args:
            num_circles: Number of concentric circles (1λ, 2λ, 3λ, etc.) to create.
            num_probes: Number of random probe granules per plane.

        Asymmetric grid: Works with different grid sizes for x, y, z dimensions.
        """
        # Quick plane collection (small lists, Python is acceptable)
        corner_count = (self.grid_size[0] + 1) * (self.grid_size[1] + 1) * (self.grid_size[2] + 1)
        half_grid = [self.grid_size[0] // 2, self.grid_size[1] // 2, self.grid_size[2] // 2]
        grid_dim = [self.grid_size[0] + 1, self.grid_size[1] + 1, self.grid_size[2] + 1]

        yz_plane, xz_plane, xy_plane = [], [], []

        # Corner granules on exposed planes (asymmetric indexing)
        for i in range(grid_dim[0]):
            for j in range(grid_dim[1]):
                for k in range(grid_dim[2]):
                    idx = i * (grid_dim[1] * grid_dim[2]) + j * grid_dim[2] + k
                    if i == half_grid[0] + 1 and j > half_grid[1] and k > half_grid[2]:
                        yz_plane.append(idx)
                    if j == half_grid[1] + 1 and i > half_grid[0] and k > half_grid[2]:
                        xz_plane.append(idx)
                    if k == half_grid[2] + 1 and i > half_grid[0] and j > half_grid[1]:
                        xy_plane.append(idx)

        # Center granules on exposed planes (asymmetric indexing)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                for k in range(self.grid_size[2]):
                    idx = (
                        corner_count
                        + i * (self.grid_size[1] * self.grid_size[2])
                        + j * self.grid_size[2]
                        + k
                    )
                    if i == half_grid[0] and j >= half_grid[1] and k >= half_grid[2]:
                        yz_plane.append(idx)
                    if j == half_grid[1] and i >= half_grid[0] and k >= half_grid[2]:
                        xz_plane.append(idx)
                    if k == half_grid[2] and i >= half_grid[0] and j >= half_grid[1]:
                        xy_plane.append(idx)

        # Select and mark random probes
        probe_color = ti.Vector(config.COLOR_PROBE[1])
        for plane in [yz_plane, xz_plane, xy_plane]:
            if len(plane) >= num_probes:
                for idx in random.sample(plane, num_probes):
                    self.granule_type_color[idx] = probe_color

        # Convert energy wavelength and call GPU kernel for field circles
        wavelength_am = constants.EWAVE_LENGTH / constants.ATTOMETER
        self._mark_sliced_plane_objects(wavelength_am, num_circles)

    @ti.kernel
    def _mark_sliced_plane_objects(self, wavelength_am: ti.f32, num_circles: ti.i32):  # type: ignore
        """GPU kernel to mark field circles on exposed planes.

        This kernel processes all granules in parallel, marking:
        - Field circles at 1λ, 2λ, etc. from the center granule

        Asymmetric grid: Uses different spacing and thresholds for x, y, z dimensions.
        """
        corner_count = (self.grid_size[0] + 1) * (self.grid_size[1] + 1) * (self.grid_size[2] + 1)
        half_grid = [self.grid_size[0] // 2, self.grid_size[1] // 2, self.grid_size[2] // 2]
        half_universe = [
            self.universe_size_am[0] / 2.0,
            self.universe_size_am[1] / 2.0,
            self.universe_size_am[2] / 2.0,
        ]
        # Plane tolerance based on cubic unit cell edge
        plane_tolerance = self.unit_cell_edge_am * 0.6
        circle_tolerance = wavelength_am * 0.05

        # Calculate center granule index (asymmetric)
        center_idx = (
            corner_count
            + half_grid[0] * self.grid_size[1] * self.grid_size[2]
            + half_grid[1] * self.grid_size[2]
            + half_grid[2]
        )
        center_pos = self.position_am[center_idx]

        # Colors
        field_color = ti.Vector(config.COLOR_FIELD[1])

        # Process all granules in parallel
        for idx in range(self.total_granules):
            pos = self.position_am[idx]

            # Check if granule is on one of the exposed planes and mark field circles
            # YZ plane (x at boundary)
            if (
                ti.abs(pos[0] - half_universe[0]) < plane_tolerance
                and pos[1] > half_universe[1]
                and pos[2] > half_universe[2]
            ):
                # Calculate 2D distance in YZ plane
                dy = pos[1] - center_pos[1]
                dz = pos[2] - center_pos[2]
                dist_2d = ti.sqrt(dy * dy + dz * dz)

                # Check if on a target circle
                for n in range(1, num_circles + 1):
                    if ti.abs(dist_2d - ti.f32(n) * wavelength_am) <= circle_tolerance:
                        self.granule_type_color[idx] = field_color
                        break

            # XZ plane (y at boundary)
            elif (
                ti.abs(pos[1] - half_universe[1]) < plane_tolerance
                and pos[0] > half_universe[0]
                and pos[2] > half_universe[2]
            ):
                # Calculate 2D distance in XZ plane
                dx = pos[0] - center_pos[0]
                dz = pos[2] - center_pos[2]
                dist_2d = ti.sqrt(dx * dx + dz * dz)

                # Check if on a target circle
                for n in range(1, num_circles + 1):
                    if ti.abs(dist_2d - ti.f32(n) * wavelength_am) <= circle_tolerance:
                        self.granule_type_color[idx] = field_color
                        break

            # XY plane (z at boundary)
            elif (
                ti.abs(pos[2] - half_universe[2]) < plane_tolerance
                and pos[0] > half_universe[0]
                and pos[1] > half_universe[1]
            ):
                # Calculate 2D distance in XY plane
                dx = pos[0] - center_pos[0]
                dy = pos[1] - center_pos[1]
                dist_2d = ti.sqrt(dx * dx + dy * dy)

                # Check if on a target circle
                for n in range(1, num_circles + 1):
                    if ti.abs(dist_2d - ti.f32(n) * wavelength_am) <= circle_tolerance:
                        self.granule_type_color[idx] = field_color
                        break


class SCGranule:
    """
    Granule Model: The Medium consists of "granules".
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
    SC topology: cubic unit cells with one granule at each corner (no center granules).
    Less efficient packing than body-centered cubic (52% vs 68% space filling).

    Performance Design: 1D Arrays with 3D Vectors
    - Memory: Contiguous layout, perfect cache line utilization (64-byte alignment)
    - Compute: Single loop parallelization, no index arithmetic (vs i*dim²+j*dim+k)
    - GPU: Direct thread mapping (thread_i→granule_i), coalesced memory access
    - SC Lattice: All granules stored in single 1D array (corner granules only)
    Benefits:
    - Simpler updates: One kernel updates all granules
    - Cleaner code: No need to manage multiple arrays
    - Movement-Ready: Velocity field ready for dynamics,
    granules can move freely without grid remapping constraints

    This is why high-performance physics engines (molecular dynamics, N-body simulations)
    universally use 1D arrays for granule data, regardless of spatial dimensionality.
    """

    def __init__(self, init_universe_size, theme="OCEAN"):
        """
        Initialize SC lattice and compute scaled-up unit-cell spacing.
        Universe size and target granules are used to define
        scaled-up unit-cell properties and scale factor.

        Args:
            init_universe_size: Requested simulation domain size [x, y, z] in meters (can be asymmetric)
                Will be rounded to fit integer number of cubic unit-cells.
            theme: Color theme name from config.py (OCEAN, DESERT, FOREST, etc.)
        """
        # Compute initial lattice properties (before rounding and lattice symmetry)
        init_universe_volume = (
            init_universe_size[0] * init_universe_size[1] * init_universe_size[2]
        )
        self.target_granules = config.TARGET_GRANULES

        # Calculate unit cell properties
        # CRITICAL: Unit cell must remain cubic (same edge length on all axes)
        # This preserves crystal structure. Only the NUMBER of cells varies per axis.
        unit_cell_volume = init_universe_volume / self.target_granules  # SC = 1 /unit-cell
        self.unit_cell_edge = unit_cell_volume ** (1 / 3)  # a^3 = volume
        self.unit_cell_edge_am = self.unit_cell_edge / constants.ATTOMETER

        # Calculate grid dimensions (number of unit cells per dimension) - asymmetric
        self.raw_size = [
            init_universe_size[0] / self.unit_cell_edge,
            init_universe_size[1] / self.unit_cell_edge,
            init_universe_size[2] / self.unit_cell_edge,
        ]
        # Round to nearest odd integer for symmetric grid per axis
        self.grid_size = [
            int(self.raw_size[0]) if int(self.raw_size[0]) % 2 == 1 else int(self.raw_size[0]) + 1,
            int(self.raw_size[1]) if int(self.raw_size[1]) % 2 == 1 else int(self.raw_size[1]) + 1,
            int(self.raw_size[2]) if int(self.raw_size[2]) % 2 == 1 else int(self.raw_size[2]) + 1,
        ]

        # Recompute actual universe dimensions to fit integer number of cubic unit cells
        self.universe_size = [
            self.grid_size[0] * self.unit_cell_edge,
            self.grid_size[1] * self.unit_cell_edge,
            self.grid_size[2] * self.unit_cell_edge,
        ]
        self.universe_size_am = [
            self.universe_size[0] / constants.ATTOMETER,
            self.universe_size[1] / constants.ATTOMETER,
            self.universe_size[2] / constants.ATTOMETER,
        ]
        self.max_universe_edge = max(
            self.grid_size[0] * self.unit_cell_edge,
            self.grid_size[1] * self.unit_cell_edge,
            self.grid_size[2] * self.unit_cell_edge,
        )
        self.max_universe_edge_am = self.max_universe_edge / constants.ATTOMETER
        self.max_grid_size = max(
            self.grid_size[0],
            self.grid_size[1],
            self.grid_size[2],
        )
        self.universe_volume = (
            self.universe_size[0] * self.universe_size[1] * self.universe_size[2]
        )

        # Scale factor based on cubic unit cell edge
        self.scale_factor = self.unit_cell_edge / (
            2 * ti.math.e * constants.PLANCK_LENGTH
        )  # linear scale factor from Planck length, increases computability

        # Compute energy-wave linear resolution, sampling rate
        # granules per wavelength, should be >10 for Nyquist (same for all axes with cubic cells)
        self.ewave_res = ti.math.round(constants.EWAVE_LENGTH / self.unit_cell_edge)
        # Compute universe linear resolution, ewavelengths per universe edge (per axis - can differ)
        self.max_uni_res = self.max_universe_edge / constants.EWAVE_LENGTH

        # Total granules: corners only (no centers for SC) - asymmetric grid
        # Corners: (grid_size[0] + 1) * (grid_size[1] + 1) * (grid_size[2] + 1)
        self.total_granules = (
            (self.grid_size[0] + 1) * (self.grid_size[1] + 1) * (self.grid_size[2] + 1)
        )

        # Compute lattice total energy from energy-wave equation
        self.energy = equations.energy_wave_equation(self.universe_volume)  # in Joules
        self.energy_kWh = equations.J_to_kWh(self.energy)  # in KWh
        self.energy_years = self.energy_kWh / (183230 * 1e9)  # global energy use

        # Initialize position and velocity 1D arrays
        # 1D array design: Better memory locality, simpler kernels, ready for dynamics
        # position, velocity in attometers for f32 precision
        # This scales 1e-17 m values to ~10 am, well within f32 range
        self.position_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.equilibrium_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)  # rest
        self.velocity_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.granule_type = ti.field(dtype=ti.i32, shape=self.total_granules)
        self.front_octant = ti.field(dtype=ti.i32, shape=self.total_granules)
        self.granule_type_color = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.granule_var_color = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)

        # Populate the lattice & index granule types
        self.populate_latticeSC()  # initialize position and velocity
        self.build_granule_typeSC()  # classifies granules
        self.find_front_octantSC()  # for block-slicing visualization
        self.set_granule_type_colorSC(theme)  # colors based on granule_type
        self.set_sliced_plane_objectsSC()  # set near/far-fields & random probes on sliced planes

    @ti.kernel
    def populate_latticeSC(self):
        """Populate SC lattice positions in a 1D array with asymmetric grid support.
        Kernel is properly optimized for Taichi's parallel execution:
        1. Single outermost loop - for idx in range() allows full GPU parallelization
        2. Index decoding - Converts linear index to 3D coordinates using integer division/modulo
        3. No nested loops - All granules computed in parallel across GPU threads
        This structure ensures maximum parallelization on GPU, as each thread independently
        computes one granule's position without synchronization overhead.

        Simple Cubic: Only corner granules are created (no center granules).
        Asymmetric grid: Different grid sizes for x, y, z dimensions.
        """
        # Parallelize over all granules using single outermost loop
        for idx in range(self.total_granules):
            # SC lattice: only corner granules (asymmetric)
            grid_dim_y = self.grid_size[1] + 1
            grid_dim_z = self.grid_size[2] + 1

            i = idx // (grid_dim_y * grid_dim_z)
            j = (idx % (grid_dim_y * grid_dim_z)) // grid_dim_z
            k = idx % grid_dim_z

            # Store positions in attometers (cubic unit cells)
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

        Granule Type:
        - VERTEX (0): 8 corner vertices of the lattice boundary
        - EDGE (1): Granules on the 12 edges (but not corners)
        - FACE (2): Granules on the 6 faces (but not on edges/corners)
        - CORE (3): All other interior granules (not on boundary)

        Simple Cubic: All granules are corners, so classification is based on grid position only.
        Asymmetric grid: Works with different grid sizes for x, y, z dimensions.
        """
        for idx in range(self.total_granules):
            # SC lattice: all granules are corner granules (asymmetric)
            grid_dim_y = self.grid_size[1] + 1
            grid_dim_z = self.grid_size[2] + 1

            i = idx // (grid_dim_y * grid_dim_z)
            j = (idx % (grid_dim_y * grid_dim_z)) // grid_dim_z
            k = idx % grid_dim_z

            # Count how many coordinates are at boundaries (0 or grid_size per axis)
            at_boundary = 0
            if i == 0 or i == self.grid_size[0]:
                at_boundary += 1
            if j == 0 or j == self.grid_size[1]:
                at_boundary += 1
            if k == 0 or k == self.grid_size[2]:
                at_boundary += 1

            if at_boundary == 3:
                self.granule_type[idx] = 0  # Granule Type: VERTEX (0)
            elif at_boundary == 2:
                self.granule_type[idx] = 1  # Granule Type: EDGE (1)
            elif at_boundary == 1:
                self.granule_type[idx] = 2  # Granule Type: FACE (2)
            else:
                self.granule_type[idx] = 3  # Granule Type: CORE (3)

    @ti.kernel
    def find_front_octantSC(self):
        """Mark granules in the front octant (for block-slicing visualization).

        Front octant = granules where x, y, z > universe_size/2 (per axis)
        Used for rendering: 0 = render, 1 = skip (for see-through effect)

        Asymmetric grid: Uses different thresholds for x, y, z dimensions.
        """
        for i in range(self.total_granules):
            # Mark if granule is in the front 1/8th block, > halfway on all axes
            # 0 = not in front octant, 1 = in front octant
            self.front_octant[i] = (
                1
                if (
                    self.position_am[i][0] > self.universe_size_am[0] / 2
                    and self.position_am[i][1] > self.universe_size_am[1] / 2
                    and self.position_am[i][2] > self.universe_size_am[2] / 2
                )
                else 0
            )

    def set_granule_type_colorSC(self, theme="OCEAN"):
        """Assign colors to granules based on their classified type and color theme.

        Args:
            theme: Color theme name from config.py (OCEAN, DESERT, FOREST, etc.)
        """
        # Get theme configuration from config module
        theme_config = getattr(config, theme, config.OCEAN)

        # Extract color values from theme
        color_vertex = ti.Vector(theme_config["COLOR_VERTEX"][1])
        color_edge = ti.Vector(theme_config["COLOR_EDGE"][1])
        color_face = ti.Vector(theme_config["COLOR_FACE"][1])
        color_core = ti.Vector(config.COLOR_MEDIUM[1])

        # Call GPU kernel with theme colors
        self._apply_granule_type_colorsSC(color_vertex, color_edge, color_face, color_core)

    @ti.kernel
    def _apply_granule_type_colorsSC(
        self,
        color_vertex: ti.types.vector(3, ti.f32),  # type: ignore
        color_edge: ti.types.vector(3, ti.f32),  # type: ignore
        color_face: ti.types.vector(3, ti.f32),  # type: ignore
        color_core: ti.types.vector(3, ti.f32),  # type: ignore
    ):
        """GPU kernel to apply colors based on granule type.

        Args:
            color_vertex: RGB color for VERTEX granules (type 0)
            color_edge: RGB color for EDGE granules (type 1)
            color_face: RGB color for FACE granules (type 2)
            color_core: RGB color for CORE granules (type 3)
        """
        # Color lookup table (type index -> RGB color)
        color_lut = ti.Matrix(
            [
                [color_vertex[0], color_vertex[1], color_vertex[2]],  # VERTEX (0)
                [color_edge[0], color_edge[1], color_edge[2]],  # EDGE (1)
                [color_face[0], color_face[1], color_face[2]],  # FACE (2)
                [color_core[0], color_core[1], color_core[2]],  # CORE (3)
            ]
        )

        for i in range(self.total_granules):
            granule_type = self.granule_type[i]
            if 0 <= granule_type <= 3:
                self.granule_type_color[i] = ti.Vector(
                    [
                        color_lut[granule_type, 0],
                        color_lut[granule_type, 1],
                        color_lut[granule_type, 2],
                    ]
                )
            else:
                self.granule_type_color[i] = ti.Vector([0.1, 0.6, 0.9])  # Light Blue for undefined

    def set_sliced_plane_objectsSC(self, num_circles=0, num_probes=3):
        """Select random granules from each of the 3 planes exposed by the front octant slice.

        Uses hybrid approach: Python for probe selection, GPU kernel for field circles.

        Args:
            num_circles: Number of concentric circles (1λ, 2λ, 3λ, etc.) to create.
            num_probes: Number of random probe granules per plane.

        Asymmetric grid: Works with different grid sizes for x, y, z dimensions.
        """
        # Quick plane collection (small lists, Python is acceptable)
        half_grid = [self.grid_size[0] // 2, self.grid_size[1] // 2, self.grid_size[2] // 2]
        grid_dim = [self.grid_size[0] + 1, self.grid_size[1] + 1, self.grid_size[2] + 1]

        yz_plane, xz_plane, xy_plane = [], [], []

        # SC lattice: only corner granules on exposed planes (asymmetric indexing)
        for i in range(grid_dim[0]):
            for j in range(grid_dim[1]):
                for k in range(grid_dim[2]):
                    idx = i * (grid_dim[1] * grid_dim[2]) + j * grid_dim[2] + k
                    if i == half_grid[0] + 1 and j > half_grid[1] and k > half_grid[2]:
                        yz_plane.append(idx)
                    if j == half_grid[1] + 1 and i > half_grid[0] and k > half_grid[2]:
                        xz_plane.append(idx)
                    if k == half_grid[2] + 1 and i > half_grid[0] and j > half_grid[1]:
                        xy_plane.append(idx)

        # Select and mark random probes
        probe_color = ti.Vector(config.COLOR_PROBE[1])
        for plane in [yz_plane, xz_plane, xy_plane]:
            if len(plane) >= num_probes:
                for idx in random.sample(plane, num_probes):
                    self.granule_type_color[idx] = probe_color

        # Convert energy wavelength and call GPU kernel for field circles
        wavelength_am = constants.EWAVE_LENGTH / constants.ATTOMETER
        self._mark_sliced_plane_objects(wavelength_am, num_circles)

    @ti.kernel
    def _mark_sliced_plane_objects(self, wavelength_am: ti.f32, num_circles: ti.i32):  # type: ignore
        """GPU kernel to mark field circles on exposed planes.

        This kernel processes all granules in parallel, marking:
        - Field circles at 1λ, 2λ, etc. from the center granule

        Simple Cubic: Center is calculated from corner granule positions.
        Asymmetric grid: Uses different spacing and thresholds for x, y, z dimensions.
        """
        grid_dim = [self.grid_size[0] + 1, self.grid_size[1] + 1, self.grid_size[2] + 1]
        half_grid = [self.grid_size[0] // 2, self.grid_size[1] // 2, self.grid_size[2] // 2]
        half_universe = [
            self.universe_size_am[0] / 2.0,
            self.universe_size_am[1] / 2.0,
            self.universe_size_am[2] / 2.0,
        ]
        # Plane tolerance based on cubic unit cell edge
        plane_tolerance = self.unit_cell_edge_am * 0.6
        circle_tolerance = wavelength_am * 0.05

        # Calculate center granule index (corner granule at center of lattice) - asymmetric
        # For SC lattice, this is the corner at (half_grid+1, half_grid+1, half_grid+1)
        center_idx = (
            (half_grid[0] + 1) * grid_dim[1] * grid_dim[2]
            + (half_grid[1] + 1) * grid_dim[2]
            + (half_grid[2] + 1)
        )
        center_pos = self.position_am[center_idx]

        # Colors
        field_color = ti.Vector(config.COLOR_FIELD[1])

        # Process all granules in parallel
        for idx in range(self.total_granules):
            pos = self.position_am[idx]

            # Check if granule is on one of the exposed planes and mark field circles
            # YZ plane (x at boundary)
            if (
                ti.abs(pos[0] - half_universe[0]) < plane_tolerance
                and pos[1] > half_universe[1]
                and pos[2] > half_universe[2]
            ):
                # Calculate 2D distance in YZ plane
                dy = pos[1] - center_pos[1]
                dz = pos[2] - center_pos[2]
                dist_2d = ti.sqrt(dy * dy + dz * dz)

                # Check if on a target circle
                for n in range(1, num_circles + 1):
                    if ti.abs(dist_2d - ti.f32(n) * wavelength_am) <= circle_tolerance:
                        self.granule_type_color[idx] = field_color
                        break

            # XZ plane (y at boundary)
            elif (
                ti.abs(pos[1] - half_universe[1]) < plane_tolerance
                and pos[0] > half_universe[0]
                and pos[2] > half_universe[2]
            ):
                # Calculate 2D distance in XZ plane
                dx = pos[0] - center_pos[0]
                dz = pos[2] - center_pos[2]
                dist_2d = ti.sqrt(dx * dx + dz * dz)

                # Check if on a target circle
                for n in range(1, num_circles + 1):
                    if ti.abs(dist_2d - ti.f32(n) * wavelength_am) <= circle_tolerance:
                        self.granule_type_color[idx] = field_color
                        break

            # XY plane (z at boundary)
            elif (
                ti.abs(pos[2] - half_universe[2]) < plane_tolerance
                and pos[0] > half_universe[0]
                and pos[1] > half_universe[1]
            ):
                # Calculate 2D distance in XY plane
                dx = pos[0] - center_pos[0]
                dy = pos[1] - center_pos[1]
                dist_2d = ti.sqrt(dx * dx + dy * dy)

                # Check if on a target circle
                for n in range(1, num_circles + 1):
                    if ti.abs(dist_2d - ti.f32(n) * wavelength_am) <= circle_tolerance:
                        self.granule_type_color[idx] = field_color
                        break


if __name__ == "__main__":
    print("\n================================================================")
    print("SMOKE TEST: WAVE-MEDIUM MODULE")
    print("================================================================")

    import time

    ti.init(arch=ti.gpu)

    # ================================================================
    # Parameters & Subatomic Objects Instantiation
    # ================================================================

    UNIVERSE_SIZE = [
        2e-16,
        2e-16,
        1e-16,
    ]  # m, simulation domain [x, y, z] dimensions (can be asymmetric)

    lattice = BCCLattice(UNIVERSE_SIZE)
    start_time = time.time()
    lattice_time = time.time() - start_time

    print(f"\nLattice Statistics:")
    print(
        f"  Requested universe: [{UNIVERSE_SIZE[0]:.1e}, {UNIVERSE_SIZE[1]:.1e}, {UNIVERSE_SIZE[2]:.1e}] m"
    )
    print(
        f"  Actual universe: [{lattice.universe_size[0]:.1e}, {lattice.universe_size[1]:.1e}, {lattice.universe_size[2]:.1e}] m"
    )
    print(
        f"  Grid size: {lattice.grid_size[0]}x{lattice.grid_size[1]}x{lattice.grid_size[2]} unit cells"
    )
    print(f"  Unit cell edge: {lattice.unit_cell_edge:.2e} m (cubic - same for all axes)")
    print(f"  Granule count: {lattice.total_granules:,}")
    print(f"  Scale factor: {lattice.scale_factor:.2e} x Planck Length")
    print(f"  Creation time: {lattice_time:.3f} seconds")

    # Resolutions
    print(f"\nLattice Linear Resolutions:")
    print(f"  Energy-wave resolution: {lattice.ewave_res:.2f} granules per wavelength")
    print(
        f"  Max universe resolution: {lattice.max_uni_res:.2f} ewavelengths per max universe edge"
    )

    # Create granule
    granule = BCCGranule(lattice.unit_cell_edge)
    print(f"\nGranule Properties:")
    print(f"  Radius: {granule.radius:.2e} m")
    print(f"  Mass: {granule.mass:.2e} kg")

    print("\n================================================================")
    print("END SMOKE TEST: WAVE-MEDIUM MODULE")
    print("================================================================")

    # Properly exit
    import sys

    sys.exit(0)
