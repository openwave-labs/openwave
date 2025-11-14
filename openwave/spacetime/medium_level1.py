"""
WAVE-MEDIUM

LEVEL-1: WAVE-FIELD MEDIUM

Object Classes @spacetime module.

WAVE-MEDIUM propagates Wave Motion (ENERGY-WAVE).
Modeled as a fluid-like medium that allows energy to transfer from one point to the next.
"""

import taichi as ti

from openwave.common import config, constants, equations


@ti.data_oriented
class WaveField:
    """
    Wave field simulation using cell-centered grid with attometer scaling.

    This class implements LEVEL-1 wave-field propagation with:
    - Cell-centered cubic grid
    - Attometer scaling for numerical precision (f32 fields)
    - Computed positions from indices (memory efficient)
    - Wave properties stored at each voxel
    - Asymmetric universe support (nx ≠ ny ≠ nz allowed)

    Initialization Strategy (mirrors LEVEL-0 BCCLattice):
    1. User specifies init_universe_size [x, y, z] in meters (can be asymmetric)
    2. Compute universe volume and target voxel count from config.TARGET_VOXELS
    3. Calculate cubic voxel size: dx = (volume / target_voxels)^(1/3)
    4. Compute grid dimensions: nx = int(x_size / dx), ny = int(y_size / dx), nz = int(z_size / dx)
    5. Recalculate actual universe size to fit integer voxel counts
    6. Initialize scalar and vector fields with attometer scaling for f32 precision
    """

    def __init__(self, init_universe_size):
        """
        Initialize WaveField from universe size with automatic voxel sizing
        and asymmetric universe support.

        Args:
            init_universe_size: simulation domain size [x, y, z], m (can be asymmetric)
                Will be rounded to fit integer number of voxels.

        Design:
            - Voxel size (dx) is CUBIC (same for all axes) - preserves wave physics
            - Grid counts (nx, ny, nz) can differ - allows asymmetric domain shapes

        Returns:
            WaveField instance with optimally sized voxels for target_voxels
        """
        # Compute initial lattice properties (before rounding and lattice symmetry)
        init_universe_volume = (
            init_universe_size[0] * init_universe_size[1] * init_universe_size[2]
        )
        target_voxels = config.TARGET_VOXELS

        # Calculate unit cell properties
        # CRITICAL: Unit cell must remain cubic (same edge length on all axes)
        # This preserves crystal structure. Only the NUMBER of cells varies per axis.
        unit_cell_volume = init_universe_volume / (target_voxels / 2)  # BCC = 2 /unit-cell
        self.unit_cell_edge = unit_cell_volume ** (1 / 3)  # a^3 = volume
        self.unit_cell_edge_am = self.unit_cell_edge / constants.ATTOMETER

        # Calculate grid dimensions (number of complete unit cells per dimension) - asymmetric
        # int() is required because:
        # 1. User-specified universe size is arbitrary (any float value)
        # 2. unit_cell_edge comes from cube root, rarely divides evenly into universe size
        # 3. Ensures integer count needed for array indexing and loop bounds
        # 4. Rounds down to fit only complete unit cells (actual universe size recalculated below)
        self.grid_size = [
            int(init_universe_size[0] / self.unit_cell_edge),
            int(init_universe_size[1] / self.unit_cell_edge),
            int(init_universe_size[2] / self.unit_cell_edge),
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

        # Total granules: corners + centers (asymmetric grid)
        # Corners: (grid_size[0] + 1) * (grid_size[1] + 1) * (grid_size[2] + 1)
        # Centers: grid_size[0] * grid_size[1] * grid_size[2]
        corner_count = (self.grid_size[0] + 1) * (self.grid_size[1] + 1) * (self.grid_size[2] + 1)
        center_count = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.total_granules = corner_count + center_count

        # Scale factor based on cubic unit cell edge
        self.scale_factor = self.unit_cell_edge / (
            2 * ti.math.e * constants.PLANCK_LENGTH
        )  # linear scale factor from Planck length, increases computability

        # Compute energy-wave linear resolution, sampling rate
        # granules per wavelength, should be >10 for Nyquist (same for all axes with cubic cells)
        self.ewave_res = ti.math.round(constants.EWAVE_LENGTH / self.unit_cell_edge * 2)
        # Compute universe linear resolution, ewavelengths per universe edge (per axis - can differ)
        self.max_uni_res = self.max_universe_edge / constants.EWAVE_LENGTH

        # Compute lattice total energy from energy-wave equation
        self.energy = equations.energy_wave_equation(self.universe_volume)  # in Joules
        self.energy_kWh = equations.J_to_kWh(self.energy)  # in KWh
        self.energy_years = self.energy_kWh / (183230 * 1e9)  # global energy use

        # Initialize position and velocity 1D arrays
        # 1D array design: Better memory locality, simpler kernels, ready for dynamics
        # position, velocity in attometers for f32 precision
        # This avoids catastrophic cancellation in difference calculations
        # This scales 1e-17 m values to ~10 am, well within f32 range
        self.position_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.position_screen = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.equilibrium_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)  # rest
        self.amplitude_am = ti.field(dtype=ti.f32, shape=self.total_granules)  # granule amplitude
        self.velocity_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.granule_type_color = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.granule_var_color = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)

        # Populate the lattice & index granule types
        self.populate_latticeBCC()  # initialize position and velocity
        self.normalize_to_screen(0)  # initial normalization for rendering

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
    def normalize_to_screen(self, enable_slice: ti.i32):  # type: ignore
        """Normalize lattice positions to 0-1 range for GGUI rendering."""
        for i in range(self.total_granules):
            # Normalize to 0-1 range (positions are in attometers, scale them back)
            if enable_slice == 1:
                # Block-slicing enabled: hide front octant granules by moving to origin
                # hide front 1/8th of the lattice for see-through effect
                self.position_screen[i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                # Normal rendering: normalize to 0-1 range
                self.position_screen[i] = self.position_am[i] / self.max_universe_edge_am


@ti.data_oriented
class PROPOSEDWaveField:
    def __init__(self, init_universe_size):
        if wavelength is None:
            wavelength = constants.EWAVE_LENGTH
        if target_voxels is None:
            target_voxels = config.TARGET_VOXELS

        # Compute universe volume
        init_volume = init_universe_size[0] * init_universe_size[1] * init_universe_size[2]

        # Calculate cubic voxel size from target voxel count
        # voxel_volume = universe_volume / target_voxels
        # dx³ = voxel_volume → dx = voxel_volume^(1/3)
        voxel_volume = init_volume / target_voxels
        dx = voxel_volume ** (1 / 3)  # Cubic voxel edge

        # Calculate grid dimensions (integer voxel counts per axis)
        nx = int(init_universe_size[0] / dx)
        ny = int(init_universe_size[1] / dx)
        nz = int(init_universe_size[2] / dx)

        # Compute points per wavelength (for consistency)
        points_per_wavelength = wavelength / dx

        # Grid dimensions (asymmetric support: nx ≠ ny ≠ nz allowed)
        self.nx, self.ny, self.nz = nx, ny, nz

        # ATTOMETER SCALING (critical for f32 precision)
        self.wavelength_am = wavelength_m / constants.ATTOMETER
        self.dx_am = self.wavelength_am / points_per_wavelength  # Cubic voxel size in am

        # Physical sizes in meters (for external reporting, asymmetric)
        self.dx_m = self.dx_am * constants.ATTOMETER
        self.domain_size_m = [
            nx * self.dx_m,
            ny * self.dx_m,
            nz * self.dx_m,
        ]  # Can differ per axis

        # SCALAR FIELDS (values in attometers where applicable)
        # Wave equation fields (leap-frog scheme requires three time levels)
        self.displacement_old_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # am (ψ at t-dt)
        self.displacement_am = ti.field(
            dtype=ti.f32, shape=(nx, ny, nz)
        )  # am (ψ at t, instantaneous)
        self.displacement_new_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # am (ψ at t+dt)
        self.amplitude_am = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # am (envelope A = max|ψ|)
        self.phase = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # radians (no scaling)

        # SCALAR FIELDS (no attometer scaling needed)
        self.density = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # kg/m³
        self.energy = ti.field(dtype=ti.f32, shape=(nx, ny, nz))  # J

        # VECTOR FIELDS (directions normalized, magnitudes may need scaling)
        self.wave_direction = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # unit vector
        self.amplitude_direction = ti.Vector.field(
            3, dtype=ti.f32, shape=(nx, ny, nz)
        )  # unit vector
        self.velocity_am = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # am/s
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny, nz))  # N

        # Compute actual universe size (rounded to integer voxel counts)
        self.actual_universe_size = [nx * self.dx_m, ny * self.dx_m, nz * self.dx_m]
        self.actual_voxel_count = nx * ny * nz

    @ti.func
    def get_position_am(self, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.math.vec3:
        """Get physical position of voxel center in attometers."""
        return ti.Vector([(i + 0.5) * self.dx_am, (j + 0.5) * self.dx_am, (k + 0.5) * self.dx_am])

    @ti.func
    def get_position_m(self, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.math.vec3:
        """Get physical position of voxel center in meters (for external use)."""
        pos_am = self.get_position_am(i, j, k)
        return pos_am * ti.f32(constants.ATTOMETER)

    @ti.func
    def get_voxel_index(self, pos_am: ti.math.vec3) -> ti.math.ivec3:
        """
        Get voxel index from position in attometers.

        Inverse mapping: position → index
        Used for particle-field interactions.
        """
        return ti.Vector(
            [
                ti.i32((pos_am[0] / self.dx_am) - 0.5),
                ti.i32((pos_am[1] / self.dx_am) - 0.5),
                ti.i32((pos_am[2] / self.dx_am) - 0.5),
            ]
        )


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

        # Calculate grid dimensions (number of complete unit cells per dimension) - asymmetric
        # int() is required because:
        # 1. User-specified universe size is arbitrary (any float value)
        # 2. unit_cell_edge comes from cube root, rarely divides evenly into universe size
        # 3. Ensures integer count needed for array indexing and loop bounds
        # 4. Rounds down to fit only complete unit cells (actual universe size recalculated below)
        self.grid_size = [
            int(init_universe_size[0] / self.unit_cell_edge),
            int(init_universe_size[1] / self.unit_cell_edge),
            int(init_universe_size[2] / self.unit_cell_edge),
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

        # Total granules: corners + centers (asymmetric grid)
        # Corners: (grid_size[0] + 1) * (grid_size[1] + 1) * (grid_size[2] + 1)
        # Centers: grid_size[0] * grid_size[1] * grid_size[2]
        corner_count = (self.grid_size[0] + 1) * (self.grid_size[1] + 1) * (self.grid_size[2] + 1)
        center_count = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.total_granules = corner_count + center_count

        # Scale factor based on cubic unit cell edge
        self.scale_factor = self.unit_cell_edge / (
            2 * ti.math.e * constants.PLANCK_LENGTH
        )  # linear scale factor from Planck length, increases computability

        # Compute energy-wave linear resolution, sampling rate
        # granules per wavelength, should be >10 for Nyquist (same for all axes with cubic cells)
        self.ewave_res = ti.math.round(constants.EWAVE_LENGTH / self.unit_cell_edge * 2)
        # Compute universe linear resolution, ewavelengths per universe edge (per axis - can differ)
        self.max_uni_res = self.max_universe_edge / constants.EWAVE_LENGTH

        # Compute lattice total energy from energy-wave equation
        self.energy = equations.energy_wave_equation(self.universe_volume)  # in Joules
        self.energy_kWh = equations.J_to_kWh(self.energy)  # in KWh
        self.energy_years = self.energy_kWh / (183230 * 1e9)  # global energy use

        # Initialize position and velocity 1D arrays
        # 1D array design: Better memory locality, simpler kernels, ready for dynamics
        # position, velocity in attometers for f32 precision
        # This avoids catastrophic cancellation in difference calculations
        # This scales 1e-17 m values to ~10 am, well within f32 range
        self.position_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.position_screen = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.equilibrium_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)  # rest
        self.amplitude_am = ti.field(dtype=ti.f32, shape=self.total_granules)  # granule amplitude
        self.velocity_am = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.granule_type_color = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.granule_var_color = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)

        # Populate the lattice & index granule types
        self.populate_latticeBCC()  # initialize position and velocity
        self.normalize_to_screen(0)  # initial normalization for rendering

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
    def normalize_to_screen(self, enable_slice: ti.i32):  # type: ignore
        """Normalize lattice positions to 0-1 range for GGUI rendering."""
        for i in range(self.total_granules):
            # Normalize to 0-1 range (positions are in attometers, scale them back)
            if enable_slice == 1:
                # Block-slicing enabled: hide front octant granules by moving to origin
                # hide front 1/8th of the lattice for see-through effect
                self.position_screen[i] = ti.Vector([0.0, 0.0, 0.0])
            else:
                # Normal rendering: normalize to 0-1 range
                self.position_screen[i] = self.position_am[i] / self.max_universe_edge_am


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

    print("\n================================================================")
    print("END SMOKE TEST: WAVE-MEDIUM MODULE")
    print("================================================================")

    # Properly exit
    import sys

    sys.exit(0)
