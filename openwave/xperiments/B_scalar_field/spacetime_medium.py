"""
SCALAR-FIELD Method Medium Data-Grid

Object Classes @spacetime module.

SCALAR-FIELD propagates Wave Motion (ENERGY-WAVE).
Modeled as a wave-field that allows energy to transfer from one point to the next.
"""

import taichi as ti

from openwave.common import colormap, constants, equations, utils


@ti.data_oriented
class WaveField:
    """
    Wave field simulation using cell-centered grid with attometer scaling.

    This class implements SCALAR-FIELD propagation with:
    - Cell-centered cubic grid
    - Attometer scaling for numerical precision (f32 fields)
    - Computed positions from indices (memory efficient)
    - Wave properties stored at each voxel
    - Asymmetric universe support (nx ≠ ny ≠ nz allowed)

    Initialization Strategy (mirrors GRANULE Method):
    1. User specifies init_universe_size [x, y, z] in meters (can be asymmetric)
    2. Compute universe volume and target voxel count
    3. Calculate cubic voxel size: dx = (volume / target_voxels)^(1/3)
    4. Compute grid dimensions: nx = int(x_size / dx), ny = int(y_size / dx), nz = int(z_size / dx)
    5. Recalculate actual universe size to fit integer voxel counts
    6. Initialize scalar and vector fields with attometer scaling for f32 precision
    """

    def __init__(self, init_universe_size, target_voxels, flux_mesh_planes=[0.5, 0.5, 0.5]):
        """
        Initialize WaveField from universe size with automatic voxel sizing.

        Args:
            init_universe_size: Simulation domain size [x, y, z] in meters.
                Can be asymmetric. Will be adjusted to fit integer voxel counts.
            target_voxels: Desired total voxel count (impacts memory and performance).

        Note:
            Voxel size (dx) is cubic (same for all axes) to preserve wave physics.
            Grid counts (nx, ny, nz) can differ for asymmetric domain shapes.
        """
        # Compute initial grid properties (before rounding and grid symmetry)
        init_universe_volume = (
            init_universe_size[0] * init_universe_size[1] * init_universe_size[2]
        )

        # Calculate cubic voxel size from target voxel count
        # CRITICAL: voxels must remain cubic (same edge length on all axes)
        # This preserves wave physics isotropy. Only the NUMBER of voxels varies per axis.
        self.voxel_volume = init_universe_volume / target_voxels  # cubic voxels
        self.voxel_edge = self.voxel_volume ** (1 / 3)  # same as dx, dx³ = voxel volume
        self.voxel_edge_am = self.voxel_edge / constants.ATTOMETER  # in attometers
        self.dx = self.voxel_edge  # additional alias for simplicity
        self.dx_am = self.voxel_edge_am  # additional alias for simplicity

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
        self.nx = self.grid_size[0]  # additional alias for simplicity
        self.ny = self.grid_size[1]  # additional alias for simplicity
        self.nz = self.grid_size[2]  # additional alias for simplicity
        self.max_grid_size = max(self.nx, self.ny, self.nz)
        self.min_grid_size = min(self.nx, self.ny, self.nz)

        # Compute total voxels (asymmetric grid)
        self.voxel_count = self.nx * self.ny * self.nz

        # Recompute actual universe dimensions to fit integer number of cubic voxels
        self.universe_size = [self.nx * self.dx, self.ny * self.dx, self.nz * self.dx]
        self.universe_size_am = [size / constants.ATTOMETER for size in self.universe_size]
        self.max_universe_edge = max(self.nx * self.dx, self.ny * self.dx, self.nz * self.dx)
        self.max_universe_edge_am = self.max_universe_edge / constants.ATTOMETER
        self.max_universe_edge_lambda = self.max_universe_edge / constants.EWAVE_LENGTH  # λ / edge
        self.universe_volume = self.voxel_count * self.voxel_volume

        # Compute SCALE FACTOR
        # Will be applied to wave amplitude & wavelength, preserving wave steepness
        min_sampling = 12  # voxels per wavelength for adequate sampling (stable ~12)
        self.scale_factor = max(
            min_sampling / (constants.EWAVE_LENGTH / self.dx), 1
        )  # linear scale factor, for computation tractability

        # Compute simulation resolution
        # Voxels per wavelength, should be >10 for adequate sampling (same for all axes)
        self.ewave_res = constants.EWAVE_LENGTH / self.dx * self.scale_factor  # voxels / λ

        # Compute grid nominal energy from energy-wave equation
        self.nominal_energy = equations.compute_energy_wave_equation(self.universe_volume)  # J
        self.nominal_energy_kWh = self.nominal_energy * utils.J2KWH  # KWh
        self.nominal_energy_years = self.nominal_energy_kWh / (183230 * 1e9)  # years

        # ================================================================
        # DATA STRUCTURE & INITIALIZATION
        # ================================================================
        # PROPAGATED SCALAR FIELDS (values in attometers for f32 precision)
        # This avoids catastrophic cancellation in difference calculations
        # Scales 1e-17 m values to ~10 am, well within f32 range
        # Wave equation fields (leap-frog scheme requires three time levels)
        # Longitudinal wave scalar field (ψl)
        self.psiL_new_am = ti.field(dtype=ti.f32, shape=self.grid_size)  # am, ψl at t+dt
        self.psiL_am = ti.field(dtype=ti.f32, shape=self.grid_size)  # am, ψl at t
        self.psiL_old_am = ti.field(dtype=ti.f32, shape=self.grid_size)  # am, ψl at t-dt
        # Transverse wave scalar field (ψt)
        self.psiT_new_am = ti.field(dtype=ti.f32, shape=self.grid_size)  # am, ψt at t+dt
        self.psiT_am = ti.field(dtype=ti.f32, shape=self.grid_size)  # am, ψt at t
        self.psiT_old_am = ti.field(dtype=ti.f32, shape=self.grid_size)  # am, ψt at t-dt

        # TODO: Implement DERIVED SCALAR FIELDS
        # wavelength, period, phase, energy, momentum

        # TODO: Implement DERIVED VECTOR FIELDS (directions normalized to unit vectors)
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
        self.edge_lines = ti.Vector.field(3, dtype=ti.f32, shape=24)  # 12 edges, 2 vertices each

        # Populate the grid and edge lines with normalized positions (ready for rendering)
        self.populate_grid_lines()  # initialize grid lines (already normalized)
        self.populate_edge_lines()  # initialize edge lines (already normalized)

        # ================================================================
        # Flux Mesh: data structures & initialization
        # ================================================================
        # Three orthogonal meshes intersecting at universe center
        # Each flux mesh has resolution matching simulation voxel grid
        # Vertices positioned at voxel centers for direct property sampling

        # XY Flux Mesh (at z = center): spans x and y dimensions
        # - Vertices: nx × ny grid points
        # - Indices: (nx-1) × (ny-1) quads × 6 indices (2 triangles each)
        self.fluxmesh_xy_vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.nx, self.ny))
        self.fluxmesh_xy_colors = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.nx, self.ny))
        self.fluxmesh_xy_indices = ti.field(dtype=ti.i32, shape=(self.nx - 1, self.ny - 1, 6))

        # XZ Flux Mesh (at y = center): spans x and z dimensions
        # - Vertices: nx × nz grid points
        # - Indices: (nx-1) × (nz-1) quads × 6 indices (2 triangles each)
        self.fluxmesh_xz_vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.nx, self.nz))
        self.fluxmesh_xz_colors = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.nx, self.nz))
        self.fluxmesh_xz_indices = ti.field(dtype=ti.i32, shape=(self.nx - 1, self.nz - 1, 6))

        # YZ Flux Mesh (at x = center): spans y and z dimensions
        # - Vertices: ny × nz grid points
        # - Indices: (ny-1) × (nz-1) quads × 6 indices (2 triangles each)
        self.fluxmesh_yz_vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.ny, self.nz))
        self.fluxmesh_yz_colors = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.ny, self.nz))
        self.fluxmesh_yz_indices = ti.field(dtype=ti.i32, shape=(self.ny - 1, self.nz - 1, 6))

        # Initialize flux mesh (vertices, indices, colors)
        self.flux_mesh_planes = flux_mesh_planes  # positions in normalized coords
        self.fm_plane_x_idx = int(flux_mesh_planes[0] * self.nx)  # x index of flux mesh plane
        self.fm_plane_y_idx = int(flux_mesh_planes[1] * self.ny)  # y index of flux mesh plane
        self.fm_plane_z_idx = int(flux_mesh_planes[2] * self.nz)  # z index of flux mesh plane
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
        # z_lines = (self.nx + 1) * (self.ny + 1)  # implicit, noted as reminder

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

    def populate_edge_lines(self):
        """
        Populate bounding box edge lines for GGUI wireframe rendering.

        Creates the 12 edges of the grid bounding box, each defined by 2 vertices.
        Total: 12 edges × 2 vertices = 24 vertices stored in self.edge_lines.

        Edge layout:
        - Edges 0-3: X-parallel (4 edges at YZ face corners)
        - Edges 4-7: Y-parallel (4 edges at XZ face corners)
        - Edges 8-11: Z-parallel (4 edges at XY face corners)

        Coordinates normalized by max_grid_size for uniform rendering of
        asymmetric grids (nx ≠ ny ≠ nz supported).

        Note: CPU assignment used instead of GPU kernel - for 24 fixed values,
        kernel launch overhead exceeds computation time.
        """
        # Normalized extents for asymmetric grid
        max_dim = float(self.max_grid_size)
        x_max = self.nx / max_dim
        y_max = self.ny / max_dim
        z_max = self.nz / max_dim

        # Define 12 edges as (start, end) vertex pairs
        edges = [
            # X-parallel edges (at 4 YZ corners)
            ([0, 0, 0], [x_max, 0, 0]),
            ([0, y_max, 0], [x_max, y_max, 0]),
            ([0, 0, z_max], [x_max, 0, z_max]),
            ([0, y_max, z_max], [x_max, y_max, z_max]),
            # Y-parallel edges (at 4 XZ corners)
            ([0, 0, 0], [0, y_max, 0]),
            ([x_max, 0, 0], [x_max, y_max, 0]),
            ([0, 0, z_max], [0, y_max, z_max]),
            ([x_max, 0, z_max], [x_max, y_max, z_max]),
            # Z-parallel edges (at 4 XY corners)
            ([0, 0, 0], [0, 0, z_max]),
            ([x_max, 0, 0], [x_max, 0, z_max]),
            ([0, y_max, 0], [0, y_max, z_max]),
            ([x_max, y_max, 0], [x_max, y_max, z_max]),
        ]

        # Populate field directly (no kernel needed for 24 values)
        for i, (start, end) in enumerate(edges):
            self.edge_lines[i * 2] = start
            self.edge_lines[i * 2 + 1] = end

    @ti.kernel
    def create_flux_mesh(self):
        """
        Initialize normalized flux mesh for all three orthogonal planes.

        Creates vertex positions and triangle indices for XY, XZ, and YZ flux mesh
        positioned at the xperiment parameter. Each plane is a 2D mesh that samples wave
        properties from the voxel grid.

        Coordinate system:
        - Positions stored in normalized coordinates [0, 1] for rendering
        - Each vertex corresponds to a voxel center for direct property sampling
        - Meshes intersect at xperiment parameter

        Mesh structure:
        - Vertices: Grid of 3D positions matching voxel resolution
        - Indices: Triangle pairs forming quads (2 triangles × 3 vertices = 6 indices)
        - Colors: Initialized to colormap.COLOR_FLUXMESH, updated by update_flux_mesh_values()
        """

        # Planes position in normalized coordinates
        max_dim = ti.cast(self.max_grid_size, ti.f32)
        fm_plane_x = self.flux_mesh_planes[0] * (self.nx / max_dim)  # x position normalized
        fm_plane_y = self.flux_mesh_planes[1] * (self.ny / max_dim)  # y position normalized
        fm_plane_z = self.flux_mesh_planes[2] * (self.nz / max_dim)  # z position normalized

        # ================================================================
        # XY Plane (at z = center): spans (0→1, 0→1, 0.5)
        # ================================================================
        for i, j in ti.ndrange(self.nx, self.ny):
            # Normalized coordinates for rendering
            x_norm = (ti.cast(i, ti.f32) + 0.5) / max_dim
            y_norm = (ti.cast(j, ti.f32) + 0.5) / max_dim

            # Vertex position
            self.fluxmesh_xy_vertices[i, j] = ti.Vector([x_norm, y_norm, fm_plane_z])

            # Initialize color (will be changed by update_flux_mesh_values)
            self.fluxmesh_xy_colors[i, j] = ti.Vector(colormap.COLOR_FLUXMESH[1])

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
            self.fluxmesh_xz_vertices[i, k] = ti.Vector([x_norm, fm_plane_y, z_norm])

            # Initialize color (will be changed by update_flux_mesh_values)
            self.fluxmesh_xz_colors[i, k] = ti.Vector(colormap.COLOR_FLUXMESH[1])

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
            self.fluxmesh_yz_vertices[j, k] = ti.Vector([fm_plane_x, y_norm, z_norm])

            # Initialize color (will be changed by update_flux_mesh_values)
            self.fluxmesh_yz_colors[j, k] = ti.Vector(colormap.COLOR_FLUXMESH[1])

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


@ti.data_oriented
class Trackers:
    """
    Wave property trackers for each voxel.
    Use separate class to avoid overloading WaveField class.

    Tracks amplitude envelope and frequency at each grid point using
    per-voxel fields and grid-wide averages for visualization scaling.
    """

    def __init__(self, grid_size, scale_factor):
        """
        Initialize tracker fields for wave property monitoring.

        Args:
            grid_size: Grid dimensions [nx, ny, nz] matching WaveField.
        """
        # TRACKED FIELDS
        # 2 polarities to track: longitudinal & transverse
        # Amplitude envelope tracks A via EMA of |ψ| and RMS calculation
        # Frequency tracks local oscillation rate via zero-crossing detection
        self.ampL_am = ti.field(dtype=ti.f32, shape=grid_size)  # am, longitudinal amp
        self.rms_ampL_am = ti.field(dtype=ti.f32, shape=())  # RMS all voxels
        self.ampT_am = ti.field(dtype=ti.f32, shape=grid_size)  # am, transverse amp
        self.rms_ampT_am = ti.field(dtype=ti.f32, shape=())  # RMS all voxels
        self.last_crossing = ti.field(dtype=ti.f32, shape=grid_size)  # rs, last zero crossing
        self.freq_rHz = ti.field(dtype=ti.f32, shape=grid_size)  # rHz, local frequency
        self.avg_freq_rHz = ti.field(dtype=ti.f32, shape=())  # avg frequency all voxels

        # Assign default values for visualization scaling
        # baseline to allow wave peaks to rise without color saturation
        self.rms_ampL_am[None] = constants.EWAVE_AMPLITUDE / constants.ATTOMETER * scale_factor
        self.rms_ampT_am[None] = constants.EWAVE_AMPLITUDE / constants.ATTOMETER * scale_factor
        self.avg_freq_rHz[None] = constants.EWAVE_FREQUENCY * constants.RONTOSECOND * scale_factor


if __name__ == "__main__":
    print("\n================================================================")
    print("SMOKE TEST: DATA-GRID MODULE")
    print("================================================================")

    ti.init(arch=ti.gpu)

    # ================================================================
    # Parameters & Subatomic Objects Instantiation
    # ================================================================

    UNIVERSE_SIZE = [
        2e-15,
        2e-15,
        2e-15,
    ]  # m, simulation domain [x, y, z] dimensions (can be asymmetric)

    wave_field = WaveField(
        UNIVERSE_SIZE, target_voxels=3.5e8
    )  # 350M voxels (~14GB), 1B voxels (~40GB)

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
    print(f"  Total energy: {wave_field.nominal_energy:.2e} J")

    # Resolutions
    print(f"\nGrid Linear Resolutions:")
    print(f"  Energy-wave resolution: {wave_field.ewave_res:.2f} voxels per lambda")
    if wave_field.ewave_res < 10:
        print(f"  *** WARNING: Undersampling! ***")

    print(
        f"  Max universe resolution: {wave_field.max_universe_edge_lambda:.2f} lambda per max universe edge"
    )

    print("\n================================================================")
    print("END SMOKE TEST: DATA-GRID MODULE")
    print("================================================================")

    # Properly exit
    import sys

    sys.exit(0)
