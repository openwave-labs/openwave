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
        # Compute initial grid properties (before rounding and grid symmetry)
        init_universe_volume = (
            init_universe_size[0] * init_universe_size[1] * init_universe_size[2]
        )
        target_voxels = config.TARGET_VOXELS

        # Calculate cubic voxel size from target voxel count
        # CRITICAL: voxels must remain cubic (same edge length on all axes)
        # This preserves crystal structure. Only the NUMBER of voxels varies per axis.
        voxel_volume = init_universe_volume / target_voxels  # cubic voxels
        self.voxel_edge = voxel_volume ** (1 / 3)  # same as dx, dx³ = voxel volume
        self.voxel_edge_am = self.voxel_edge / constants.ATTOMETER  # in attometers

        # Calculate grid dimensions (number of complete voxels per dimension) - asymmetric
        # int() is required because:
        # 1. User-specified universe size is arbitrary (any float value)
        # 2. voxel_edge comes from cube root, rarely divides evenly into universe size
        # 3. Ensures integer count needed for array indexing and loop bounds
        # 4. Rounds to fit only complete voxels (actual universe size recalculated below)
        self.grid_size = [
            int(init_universe_size[0] / self.voxel_edge),
            int(init_universe_size[1] / self.voxel_edge),
            int(init_universe_size[2] / self.voxel_edge),
        ]  # same as (nx, ny, nz)

        # Compute total voxels (asymmetric grid)
        self.voxel_count = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # Recompute actual universe dimensions to fit integer number of cubic voxels
        self.universe_size = [
            self.grid_size[0] * self.voxel_edge,
            self.grid_size[1] * self.voxel_edge,
            self.grid_size[2] * self.voxel_edge,
        ]
        self.universe_size_am = [
            self.universe_size[0] / constants.ATTOMETER,
            self.universe_size[1] / constants.ATTOMETER,
            self.universe_size[2] / constants.ATTOMETER,
        ]
        self.max_universe_edge = max(
            self.grid_size[0] * self.voxel_edge,
            self.grid_size[1] * self.voxel_edge,
            self.grid_size[2] * self.voxel_edge,
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

        # Compute energy-wave linear resolution, sampling rate
        # voxels per wavelength, should be >10 for Nyquist (same for all axes with cubic cells)
        self.ewave_res = ti.math.round(constants.EWAVE_LENGTH / self.voxel_edge)
        # Compute universe linear resolution, ewavelengths per universe edge (per axis - can differ)
        self.max_uni_res = self.max_universe_edge / constants.EWAVE_LENGTH

        # Compute grid total energy from energy-wave equation
        self.energy = equations.energy_wave_equation(self.universe_volume)  # in Joules
        self.energy_kWh = equations.J_to_kWh(self.energy)  # in KWh
        self.energy_years = self.energy_kWh / (183230 * 1e9)  # global energy use

        # MEASURED SCALAR FIELDS (values in attometers for f32 precision)
        # This avoids catastrophic cancellation in difference calculations
        # This scales 1e-17 m values to ~10 am, well within f32 range
        # Wave equation fields (leap-frog scheme requires three time levels)
        self.displacement_new_am = ti.field(dtype=ti.f32, shape=self.grid_size)  # am (ψ at t+dt)
        self.displacement_am = ti.field(dtype=ti.f32, shape=self.grid_size)  # am (ψ at t)
        self.displacement_old_am = ti.field(dtype=ti.f32, shape=self.grid_size)  # am (ψ at t-dt)
        self.amplitude_am = ti.field(dtype=ti.f32, shape=self.grid_size)  # am, envelope A = max|ψ|
        self.frequency = ti.field(dtype=ti.f32, shape=self.grid_size)  # Hz, wave rhythm

        # DERIVED SCALAR FIELDS
        # wavelength, period, phase, energy, momentum

        # DERIVED VECTOR FIELDS (directions normalized to unit vectors)
        # energy_flux, wave_direction, displacement_direction, wave_mode, wave_type

        # Grid Visualization Fields
        # Wire-frame: unique edges of voxel grid for rendering
        # Each voxel edge connects two grid vertices, stored as vertex pairs for scene.lines()
        # Total edges = nx×(ny+1)×(nz+1) + (nx+1)×ny×(nz+1) + (nx+1)×(ny+1)×nz
        # Each edge needs 2 vertices for rendering
        self.edge_count = (
            self.grid_size[0] * (self.grid_size[1] + 1) * (self.grid_size[2] + 1)  # X-edges
            + (self.grid_size[0] + 1) * self.grid_size[1] * (self.grid_size[2] + 1)  # Y-edges
            + (self.grid_size[0] + 1) * (self.grid_size[1] + 1) * self.grid_size[2]  # Z-edges
        )
        self.wire_frame = ti.Vector.field(3, dtype=ti.f32, shape=self.edge_count * 2)

        # Populate the grid wire_frame with taichi lines positions
        self.populate_wire_frame()  # initialize grid lines position
        self.normalize_wire_frame()  # normalize wire-frame for rendering

    @ti.kernel
    def populate_wire_frame(self):
        """
        Create a wave field wire-frame for GGUI rendering and visualization.
        Stores all unique voxel edges as vertex pairs for scene.lines() rendering.

        The wire-frame visualizes the 3D grid structure by drawing edges of all voxels.
        Each edge is shared between neighboring voxels and stored only once.

        For a 2×2×2 voxel grid:
        - 27 unique vertices (3×3×3 grid points)
        - 54 unique edges (18 per axis direction)
        - 108 line vertices for rendering (54 edges × 2 vertices)

        Edge calculation:
        - X-direction: nx × (ny+1) × (nz+1) edges
        - Y-direction: (nx+1) × ny × (nz+1) edges
        - Z-direction: (nx+1) × (ny+1) × nz edges

        Positions stored in attometers, normalized later for rendering.
        """
        # Calculate number of edges per direction
        nx, ny, nz = self.grid_size[0], self.grid_size[1], self.grid_size[2]
        x_edges = nx * (ny + 1) * (nz + 1)
        y_edges = (nx + 1) * ny * (nz + 1)
        z_edges = (nx + 1) * (ny + 1) * nz

        # Parallelize over all edges using single outermost loop
        for edge_idx in range(self.edge_count):
            vertex_idx = edge_idx * 2  # Each edge has 2 vertices

            if edge_idx < x_edges:
                # X-direction edge: decode 3D position from linear index
                temp = edge_idx
                i = temp // ((ny + 1) * (nz + 1))
                temp = temp % ((ny + 1) * (nz + 1))
                j = temp // (nz + 1)
                k = temp % (nz + 1)

                # Start vertex at (i, j, k), End vertex at (i+1, j, k)
                self.wire_frame[vertex_idx] = ti.Vector(
                    [i * self.voxel_edge_am, j * self.voxel_edge_am, k * self.voxel_edge_am]
                )
                self.wire_frame[vertex_idx + 1] = ti.Vector(
                    [(i + 1) * self.voxel_edge_am, j * self.voxel_edge_am, k * self.voxel_edge_am]
                )

            elif edge_idx < x_edges + y_edges:
                # Y-direction edge: decode 3D position
                temp = edge_idx - x_edges
                i = temp // (ny * (nz + 1))
                temp = temp % (ny * (nz + 1))
                j = temp // (nz + 1)
                k = temp % (nz + 1)

                # Start vertex at (i, j, k), End vertex at (i, j+1, k)
                self.wire_frame[vertex_idx] = ti.Vector(
                    [i * self.voxel_edge_am, j * self.voxel_edge_am, k * self.voxel_edge_am]
                )
                self.wire_frame[vertex_idx + 1] = ti.Vector(
                    [i * self.voxel_edge_am, (j + 1) * self.voxel_edge_am, k * self.voxel_edge_am]
                )

            else:
                # Z-direction edge: decode 3D position
                temp = edge_idx - x_edges - y_edges
                i = temp // ((ny + 1) * nz)
                temp = temp % ((ny + 1) * nz)
                j = temp // nz
                k = temp % nz

                # Start vertex at (i, j, k), End vertex at (i, j, k+1)
                self.wire_frame[vertex_idx] = ti.Vector(
                    [i * self.voxel_edge_am, j * self.voxel_edge_am, k * self.voxel_edge_am]
                )
                self.wire_frame[vertex_idx + 1] = ti.Vector(
                    [i * self.voxel_edge_am, j * self.voxel_edge_am, (k + 1) * self.voxel_edge_am]
                )

    @ti.kernel
    def normalize_wire_frame(self):
        """
        Normalize wire-frame vertex positions to 0-1 range for GGUI rendering.

        Converts positions from attometers to normalized coordinates by dividing
        by the maximum universe edge dimension. This ensures the wire-frame fits
        within the rendering viewport regardless of actual physical scale.
        """
        for i in range(self.edge_count * 2):
            # Normalize each vertex position to 0-1 range
            self.wire_frame[i] = self.wire_frame[i] / self.max_universe_edge_am

    # @ti.func
    # def get_position(self, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.math.vec3:  # type: ignore
    #     """Get physical position of voxel center in meters (for external use)."""
    #     pos_am = self.get_position_am(i, j, k)
    #     return pos_am * ti.f32(constants.ATTOMETER)

    # @ti.func
    # def get_position_am(self, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.math.vec3:  # type: ignore
    #     """Get physical position of voxel center in attometers."""
    #     return ti.Vector([(i + 0.5) * self.dx_am, (j + 0.5) * self.dx_am, (k + 0.5) * self.dx_am])

    # @ti.func
    # def get_voxel_index(self, pos_am: ti.math.vec3) -> ti.math.ivec3:  # type: ignore
    #     """
    #     Get voxel index from position in attometers.

    #     Inverse mapping: position → index
    #     Used for particle-field interactions.
    #     """
    #     return ti.Vector(
    #         [
    #             ti.i32((pos_am[0] / self.dx_am) - 0.5),
    #             ti.i32((pos_am[1] / self.dx_am) - 0.5),
    #             ti.i32((pos_am[2] / self.dx_am) - 0.5),
    #         ]
    #     )


if __name__ == "__main__":
    print("\n================================================================")
    print("SMOKE TEST: WAVE-MEDIUM MODULE")
    print("================================================================")

    ti.init(arch=ti.gpu)

    # ================================================================
    # Parameters & Subatomic Objects Instantiation
    # ================================================================

    UNIVERSE_SIZE = [
        1e-16,
        1e-16,
        1e-16,
    ]  # m, simulation domain [x, y, z] dimensions (can be asymmetric)

    wave_field = WaveField(UNIVERSE_SIZE)

    print(f"\nGrid Statistics:")
    print(
        f"  Requested universe: [{UNIVERSE_SIZE[0]:.1e}, {UNIVERSE_SIZE[1]:.1e}, {UNIVERSE_SIZE[2]:.1e}] m"
    )
    print(
        f"  Actual universe: [{wave_field.universe_size[0]:.1e}, {wave_field.universe_size[1]:.1e}, {wave_field.universe_size[2]:.1e}] m"
    )
    print(
        f"  Grid size: {wave_field.grid_size[0]} x {wave_field.grid_size[1]} x {wave_field.grid_size[2]} voxels"
    )
    print(f"  Voxel edge: {wave_field.voxel_edge:.2e} m (cubic - same for all axes)")
    print(f"  Voxel count: {wave_field.voxel_count:,}")
    print(f"  Total energy: {wave_field.energy:.2e} J")

    # Resolutions
    print(f"\nGrid Linear Resolutions:")
    print(f"  Energy-wave resolution: {wave_field.ewave_res:.2f} voxels per wavelength")
    print(
        f"  Max universe resolution: {wave_field.max_uni_res:.2f} ewavelengths per max universe edge"
    )

    print("\n================================================================")
    print("END SMOKE TEST: WAVE-MEDIUM MODULE")
    print("================================================================")

    # Properly exit
    import sys

    sys.exit(0)
