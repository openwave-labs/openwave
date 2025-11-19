"""
WAVE-MEDIUM

LEVEL-1: WAVE-FIELD MEDIUM

Object Classes @spacetime module.

WAVE-MEDIUM propagates Wave Motion (ENERGY-WAVE).
Modeled as a fluid-like medium that allows energy to transfer from one point to the next.
"""

import taichi as ti

from openwave.common import constants, equations, utils


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
    2. Compute universe volume and target voxel count
    3. Calculate cubic voxel size: dx = (volume / target_voxels)^(1/3)
    4. Compute grid dimensions: nx = int(x_size / dx), ny = int(y_size / dx), nz = int(z_size / dx)
    5. Recalculate actual universe size to fit integer voxel counts
    6. Initialize scalar and vector fields with attometer scaling for f32 precision
    """

    def __init__(self, init_universe_size, target_voxels):
        """
        Initialize WaveField from universe size with automatic voxel sizing
        and asymmetric universe support.

        Args:
            init_universe_size: simulation domain size [x, y, z], m (can be asymmetric)
                Will be rounded to fit integer number of voxels.
            target_voxels: desired total voxel count (impacts performance)

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

        # Calculate cubic voxel size from target voxel count
        # CRITICAL: voxels must remain cubic (same edge length on all axes)
        # This preserves crystal structure. Only the NUMBER of voxels varies per axis.
        self.voxel_volume = init_universe_volume / target_voxels  # cubic voxels
        self.voxel_edge = self.voxel_volume ** (1 / 3)  # same as dx, dx³ = voxel volume
        self.voxel_edge_am = self.voxel_edge / constants.ATTOMETER  # in attometers
        self.dx = self.voxel_edge
        self.dx_am = self.voxel_edge_am

        # Calculate grid dimensions (number of complete voxels per dimension) - asymmetric
        # Uses nearest odd integer to ensure grid symmetry with unique central voxel:
        # 1. User-specified universe size is arbitrary (any float value)
        # 2. voxel_edge comes from cube root, rarely divides evenly into universe size
        # 3. Ensures integer count needed for array indexing and loop bounds
        # 4. Rounds to nearest odd integer for symmetric grid with central voxel
        # 5. Actual universe size recalculated below to fit integer voxel count
        self.grid_size = [
            utils.round_to_nearest_odd(init_universe_size[0] / self.dx),
            utils.round_to_nearest_odd(init_universe_size[1] / self.dx),
            utils.round_to_nearest_odd(init_universe_size[2] / self.dx),
        ]  # same as (nx, ny, nz)
        self.nx, self.ny, self.nz = self.grid_size[0], self.grid_size[1], self.grid_size[2]
        self.max_grid_size = max(self.nx, self.ny, self.nz)

        # Compute total voxels (asymmetric grid)
        self.voxel_count = self.nx * self.ny * self.nz

        # Recompute actual universe dimensions to fit integer number of cubic voxels
        self.universe_size = [self.nx * self.dx, self.ny * self.dx, self.nz * self.dx]
        self.universe_size_am = [size / constants.ATTOMETER for size in self.universe_size]
        self.max_universe_edge = max(self.nx * self.dx, self.ny * self.dx, self.nz * self.dx)
        self.max_universe_edge_am = self.max_universe_edge / constants.ATTOMETER
        self.universe_volume = self.voxel_count * self.voxel_volume

        # Compute energy-wave linear resolution, sampling rate
        # voxels per wavelength, should be >10 for Nyquist (same for all axes with cubic cells)
        self.ewave_res = ti.math.round(constants.EWAVE_LENGTH / self.dx)  # in voxels / λ
        # Compute universe linear resolution, ewavelengths per universe edge (per axis - can differ)
        self.max_uni_res = self.max_universe_edge / constants.EWAVE_LENGTH

        # Compute grid total energy from energy-wave equation
        self.energy = equations.energy_wave_equation(self.universe_volume)  # in Joules
        self.energy_kWh = self.energy * utils.J2KWH  # in KWh
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

        # ================================================================
        # Grid Visualization: data structures & initialization
        # ================================================================
        # Grid: optimized grid lines for rendering
        # Each line spans the entire grid dimension (e.g., from x=0 to x=1 in normalized coords)

        # Line count per direction:
        # - X-direction (parallel to X): (ny+1) × (nz+1) lines
        # - Y-direction (parallel to Y): (nx+1) × (nz+1) lines
        # - Z-direction (parallel to Z): (nx+1) × (ny+1) lines
        # Total vertices = 2 × (sum of lines)
        self.line_count = (
            (self.ny + 1) * (self.nz + 1)  # X-parallel lines
            + (self.nx + 1) * (self.nz + 1)  # Y-parallel lines
            + (self.nx + 1) * (self.ny + 1)  # Z-parallel lines
        )
        self.grid_lines = ti.Vector.field(3, dtype=ti.f32, shape=self.line_count * 2)

        # Populate the grid with normalized positions (ready for rendering)
        self.populate_grid_lines()  # initialize grid lines (already normalized)

        # ================================================================
        # Flux Mesh: data structures & initialization
        # ================================================================
        # Three orthogonal meshes intersecting at universe center
        # Each flux mesh has resolution matching simulation voxel grid
        # Vertices positioned at voxel centers for direct property sampling

        # XY Flux Mesh (at z = center): spans (0→1, 0→1, 0.5)
        # - Vertices: nx × ny grid points
        # - Indices: (nx-1) × (ny-1) quads × 6 vertices (2 triangles each)
        # nx, ny, nz = self.grid_size[0], self.grid_size[1], self.grid_size[2]
        self.fluxmesh_xy_vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.nx, self.ny))
        self.fluxmesh_xy_colors = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.nx, self.ny))
        self.fluxmesh_xy_indices = ti.field(dtype=ti.i32, shape=(self.nx - 1, self.ny - 1, 6))

        # XZ Flux Mesh (at y = center): spans (0→1, 0.5, 0→1)
        # - Vertices: nx × nz grid points
        # - Indices: (nx-1) × (nz-1) quads × 6 vertices (2 triangles each)
        self.fluxmesh_xz_vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.nx, self.nz))
        self.fluxmesh_xz_colors = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.nx, self.nz))
        self.fluxmesh_xz_indices = ti.field(dtype=ti.i32, shape=(self.nx - 1, self.nz - 1, 6))

        # YZ Flux Mesh (at x = center): spans (0.5, 0→1, 0→1)
        # - Vertices: ny × nz grid points
        # - Indices: (ny-1) × (nz-1) quads × 6 vertices (2 triangles each)
        self.fluxmesh_yz_vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.ny, self.nz))
        self.fluxmesh_yz_colors = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.ny, self.nz))
        self.fluxmesh_yz_indices = ti.field(dtype=ti.i32, shape=(self.ny - 1, self.nz - 1, 6))

        # Initialize flux mesh (vertices, indices, colors)
        self.create_flux_mesh()

    @ti.kernel
    def populate_grid_lines(self):
        """
        Create optimized grid lines for GGUI rendering and visualization.

        Draws continuous lines across the entire grid face-to-face.

        For a 2×2×2 voxel grid (3×3×3 grid points):
        - 9 lines parallel to X-axis (one per (j,k) pair on YZ plane)
        - 9 lines parallel to Y-axis (one per (i,k) pair on XZ plane)
        - 9 lines parallel to Z-axis (one per (i,j) pair on XY plane)
        - Total: 27 lines × 2 vertices = 54 vertices (vs 108 in non-optimized version)

        Line calculation:
        - X-direction: (ny+1) × (nz+1) lines
        - Y-direction: (nx+1) × (nz+1) lines
        - Z-direction: (nx+1) × (ny+1) lines

        Positions stored directly in normalized coordinates (0-1 range) ready for rendering.
        Uses max_grid_size for uniform normalization across asymmetric grids.
        """
        # Grid dimensions and normalization factor
        max_dim = ti.cast(self.max_grid_size, ti.f32)

        # Calculate line counts per direction
        x_lines = (self.ny + 1) * (self.nz + 1)
        y_lines = (self.nx + 1) * (self.nz + 1)
        # z_lines = (self.nx + 1) * (self.ny + 1)  # implicit, computed as remainder

        # Parallelize over all lines using single outermost loop
        for line_idx in range(self.line_count):
            vertex_idx = line_idx * 2  # Each line has 2 vertices

            if line_idx < x_lines:
                # X-parallel lines: decode (j, k) position
                temp = line_idx
                j = temp // (self.nz + 1)
                k = temp % (self.nz + 1)

                # Line from x=0 to x=nx (normalized to 0 to nx/max)
                y_norm = ti.cast(j, ti.f32) / max_dim
                z_norm = ti.cast(k, ti.f32) / max_dim

                self.grid_lines[vertex_idx] = ti.Vector([0.0, y_norm, z_norm])
                self.grid_lines[vertex_idx + 1] = ti.Vector(
                    [ti.cast(self.nx, ti.f32) / max_dim, y_norm, z_norm]
                )

            elif line_idx < x_lines + y_lines:
                # Y-parallel lines: decode (i, k) position
                temp = line_idx - x_lines
                i = temp // (self.nz + 1)
                k = temp % (self.nz + 1)

                # Line from y=0 to y=ny (normalized to 0 to ny/max)
                x_norm = ti.cast(i, ti.f32) / max_dim
                z_norm = ti.cast(k, ti.f32) / max_dim

                self.grid_lines[vertex_idx] = ti.Vector([x_norm, 0.0, z_norm])
                self.grid_lines[vertex_idx + 1] = ti.Vector(
                    [x_norm, ti.cast(self.ny, ti.f32) / max_dim, z_norm]
                )

            else:
                # Z-parallel lines: decode (i, j) position
                temp = line_idx - x_lines - y_lines
                i = temp // (self.ny + 1)
                j = temp % (self.ny + 1)

                # Line from z=0 to z=nz (normalized to 0 to nz/max)
                x_norm = ti.cast(i, ti.f32) / max_dim
                y_norm = ti.cast(j, ti.f32) / max_dim

                self.grid_lines[vertex_idx] = ti.Vector([x_norm, y_norm, 0.0])
                self.grid_lines[vertex_idx + 1] = ti.Vector(
                    [x_norm, y_norm, ti.cast(self.nz, ti.f32) / max_dim]
                )

    @ti.kernel
    def create_flux_mesh(self):
        """
        Initialize normalized flux mesh for all three orthogonal planes.

        Creates vertex positions and triangle indices for XY, XZ, and YZ flux mesh
        positioned at the universe center. Each plane is a 2D mesh that samples wave
        properties from the voxel grid.

        Coordinate system:
        - Positions stored in normalized coordinates [0, 1] for rendering
        - Each vertex corresponds to a voxel center for direct property sampling
        - Meshes intersect at [0.5, 0.5, 0.5] (universe center)

        Mesh structure:
        - Vertices: Grid of 3D positions matching voxel resolution
        - Indices: Triangle pairs forming quads (2 triangles × 3 vertices = 6 indices)
        - Colors: Initialized to black, updated by update_flux_mesh_colors()
        """
        max_dim = ti.cast(self.max_grid_size, ti.f32)

        # Center position in normalized coordinates
        center_x = ti.cast(self.nx, ti.f32) / (2.0 * max_dim)
        center_y = ti.cast(self.ny, ti.f32) / (2.0 * max_dim)
        center_z = ti.cast(self.nz, ti.f32) / (2.0 * max_dim)

        # ================================================================
        # XY Plane (at z = center): spans (0→1, 0→1, 0.5)
        # ================================================================
        for i, j in ti.ndrange(self.nx, self.ny):
            # Normalized coordinates for rendering
            x_norm = (ti.cast(i, ti.f32) + 0.5) / max_dim
            y_norm = (ti.cast(j, ti.f32) + 0.5) / max_dim

            # Vertex position
            self.fluxmesh_xy_vertices[i, j] = ti.Vector([x_norm, y_norm, center_z])

            # Initialize color to black (will be updated by update_flux_mesh_colors)
            self.fluxmesh_xy_colors[i, j] = ti.Vector([0.0, 0.0, 0.0])

        # Triangle indices for XY plane
        for i, j in ti.ndrange(self.nx - 1, self.ny - 1):
            # Each quad = 2 triangles
            # Triangle 1: (i,j) → (i+1,j) → (i,j+1)
            self.fluxmesh_xy_indices[i, j, 0] = i * self.ny + j
            self.fluxmesh_xy_indices[i, j, 1] = (i + 1) * self.ny + j
            self.fluxmesh_xy_indices[i, j, 2] = i * self.ny + (j + 1)

            # Triangle 2: (i+1,j) → (i+1,j+1) → (i,j+1)
            self.fluxmesh_xy_indices[i, j, 3] = (i + 1) * self.ny + j
            self.fluxmesh_xy_indices[i, j, 4] = (i + 1) * self.ny + (j + 1)
            self.fluxmesh_xy_indices[i, j, 5] = i * self.ny + (j + 1)

        # ================================================================
        # XZ Plane (at y = center): spans (0→1, 0.5, 0→1)
        # ================================================================
        for i, k in ti.ndrange(self.nx, self.nz):
            # Normalized coordinates for rendering
            x_norm = (ti.cast(i, ti.f32) + 0.5) / max_dim
            z_norm = (ti.cast(k, ti.f32) + 0.5) / max_dim

            # Vertex position
            self.fluxmesh_xz_vertices[i, k] = ti.Vector([x_norm, center_y, z_norm])

            # Initialize color to black
            self.fluxmesh_xz_colors[i, k] = ti.Vector([0.0, 0.0, 0.0])

        # Triangle indices for XZ plane
        for i, k in ti.ndrange(self.nx - 1, self.nz - 1):
            # Each quad = 2 triangles
            # Triangle 1: (i,k) → (i+1,k) → (i,k+1)
            self.fluxmesh_xz_indices[i, k, 0] = i * self.nz + k
            self.fluxmesh_xz_indices[i, k, 1] = (i + 1) * self.nz + k
            self.fluxmesh_xz_indices[i, k, 2] = i * self.nz + (k + 1)

            # Triangle 2: (i+1,k) → (i+1,k+1) → (i,k+1)
            self.fluxmesh_xz_indices[i, k, 3] = (i + 1) * self.nz + k
            self.fluxmesh_xz_indices[i, k, 4] = (i + 1) * self.nz + (k + 1)
            self.fluxmesh_xz_indices[i, k, 5] = i * self.nz + (k + 1)

        # ================================================================
        # YZ Plane (at x = center): spans (0.5, 0→1, 0→1)
        # ================================================================
        for j, k in ti.ndrange(self.ny, self.nz):
            # Normalized coordinates for rendering
            y_norm = (ti.cast(j, ti.f32) + 0.5) / max_dim
            z_norm = (ti.cast(k, ti.f32) + 0.5) / max_dim

            # Vertex position
            self.fluxmesh_yz_vertices[j, k] = ti.Vector([center_x, y_norm, z_norm])

            # Initialize color to black
            self.fluxmesh_yz_colors[j, k] = ti.Vector([0.0, 0.0, 0.0])

        # Triangle indices for YZ plane
        for j, k in ti.ndrange(self.ny - 1, self.nz - 1):
            # Each quad = 2 triangles
            # Triangle 1: (j,k) → (j+1,k) → (j,k+1)
            self.fluxmesh_yz_indices[j, k, 0] = j * self.nz + k
            self.fluxmesh_yz_indices[j, k, 1] = (j + 1) * self.nz + k
            self.fluxmesh_yz_indices[j, k, 2] = j * self.nz + (k + 1)

            # Triangle 2: (j+1,k) → (j+1,k+1) → (j,k+1)
            self.fluxmesh_yz_indices[j, k, 3] = (j + 1) * self.nz + k
            self.fluxmesh_yz_indices[j, k, 4] = (j + 1) * self.nz + (k + 1)
            self.fluxmesh_yz_indices[j, k, 5] = j * self.nz + (k + 1)

    @ti.func
    def get_position(self, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.math.vec3:  # type: ignore
        """Get physical position of voxel center in meters (for external use)."""
        pos_am = self.get_position_am(i, j, k)
        return pos_am * ti.f32(constants.ATTOMETER)

    @ti.func
    def get_position_am(self, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.math.vec3:  # type: ignore
        """Get physical position of voxel center in attometers."""
        return ti.Vector([(i + 0.5) * self.dx_am, (j + 0.5) * self.dx_am, (k + 0.5) * self.dx_am])

    @ti.func
    def get_voxel_index(self, pos_am: ti.math.vec3) -> ti.math.ivec3:  # type: ignore
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


if __name__ == "__main__":
    print("\n================================================================")
    print("SMOKE TEST: WAVE-MEDIUM MODULE")
    print("================================================================")

    ti.init(arch=ti.gpu)

    # ================================================================
    # Parameters & Subatomic Objects Instantiation
    # ================================================================

    UNIVERSE_SIZE = [
        6e-15,
        6e-15,
        6e-15,
    ]  # m, simulation domain [x, y, z] dimensions (can be asymmetric)

    wave_field = WaveField(UNIVERSE_SIZE, target_voxels=1e9)

    print(f"\nGrid Statistics:")
    print(
        f"  Requested universe: [{UNIVERSE_SIZE[0]:.1e}, {UNIVERSE_SIZE[1]:.1e}, {UNIVERSE_SIZE[2]:.1e}] m"
    )
    print(
        f"  Actual universe: [{wave_field.universe_size[0]:.1e}, {wave_field.universe_size[1]:.1e}, {wave_field.universe_size[2]:.1e}] m"
    )
    print(f"  Grid size: {wave_field.nx} x {wave_field.ny} x {wave_field.nz} voxels")
    print(f"  Voxel edge: {wave_field.dx:.2e} m (cubic - same for all axes)")
    print(f"  Voxel count: {wave_field.voxel_count:,}")
    print(f"  Total energy: {wave_field.energy:.2e} J")

    # Resolutions
    print(f"\nGrid Linear Resolutions:")
    print(f"  Energy-wave resolution: {wave_field.ewave_res:.2f} voxels per lambda")
    if wave_field.ewave_res < 10:
        print(f"  *** WARNING: Undersampling! ***")

    print(f"  Max universe resolution: {wave_field.max_uni_res:.2f} lambda per max universe edge")

    print("\n================================================================")
    print("END SMOKE TEST: WAVE-MEDIUM MODULE")
    print("================================================================")

    # Properly exit
    import sys

    sys.exit(0)
