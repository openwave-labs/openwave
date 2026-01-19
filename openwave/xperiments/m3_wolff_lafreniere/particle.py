"""
PARTICLE MODULE

Matter modules for OpenWave Energy Wave Theory simulations.
This module defines the WaveCenter class for simulating fundamental particle.

This package contains the fundamental components:
- fundamental_particle: Fundamental particle representations and calculations
- standalone_particle: Standalone particle representations and calculations
- composite_particle: Composite particle representations and calculations
- atom: Atom representations and calculations
- molecule: Molecule representations and calculations
"""

import taichi as ti

from openwave.common import constants


@ti.data_oriented
class WaveCenter:
    """
    WaveCenter represents a fundamental particle as a localized wave deflector.

    Wave centers generate spherical waves and respond to forces from energy gradients.
    The phase offset models electrostatic charge (0 = electron, pi = positron).
    """

    def __init__(self, grid_size, num_sources, sources_position, sources_offset_deg):
        """
        Initialize the WaveCenter with given source positions and phase offsets.

        Args:
            grid_size: Tuple (nx, ny, nz) of grid dimensions
            num_sources: Number of wave center sources
            sources_position: List of normalized positions [(x, y, z), ...] in [0, 1]
            sources_offset_deg: List of phase offsets in degrees (0 = electron, 180 = positron)
        """

        self.num_sources = num_sources

        # ================================================================
        # Position fields (grid indices)
        # ================================================================
        # Integer grid position for wave generation (voxel indices)
        self.position_grid = ti.Vector.field(3, dtype=ti.i32, shape=num_sources)
        # Float grid position for smooth motion integration
        self.position_float = ti.Vector.field(3, dtype=ti.f32, shape=num_sources)

        # ================================================================
        # Motion fields (OpenWave scaled units for f32 precision)
        # ================================================================
        # Velocity in attometers per rontosecond (am/rs)
        # Sublight speeds: 0.0001 to 0.3 am/rs (c = 0.3 am/rs)
        self.velocity_amrs = ti.Vector.field(3, dtype=ti.f32, shape=num_sources)

        # Force in Newtons (SI) - kept in SI for physics accuracy
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=num_sources)

        # DEBUG: Intermediate values for force calculation inspection
        self.debug_A_center = ti.field(dtype=ti.f32, shape=num_sources)
        self.debug_dA_dx = ti.field(dtype=ti.f32, shape=num_sources)
        self.debug_force_scale = ti.field(dtype=ti.f32, shape=num_sources)

        # Mass in quectogram (qg) - initialized to neutrino mass, f32 friendly
        self.mass_qg = ti.field(dtype=ti.f32, shape=num_sources)

        # ================================================================
        # Wave properties
        # ================================================================
        # Phase offset in radians (charge analog: 0 = electron, pi = positron)
        self.offset = ti.field(dtype=ti.f32, shape=num_sources)

        # ================================================================
        # Activity state
        # ================================================================
        # Per-WC flag (1 = active, 0 = inactive/annihilated)
        self.active = ti.field(dtype=ti.i32, shape=num_sources)

        # Convert phase from degrees to radians
        sources_offset_rad = [deg * ti.math.pi / 180 for deg in sources_offset_deg]

        # Copy source data to Taichi fields
        for i in range(num_sources):
            # Compute grid position from normalized coordinates
            pos_i = int(sources_position[i][0] * grid_size[0])
            pos_j = int(sources_position[i][1] * grid_size[1])
            pos_k = int(sources_position[i][2] * grid_size[2])

            # Initialize integer and float positions
            self.position_grid[i] = [pos_i, pos_j, pos_k]
            self.position_float[i] = [float(pos_i), float(pos_j), float(pos_k)]

            # Initialize motion fields (at rest, no force)
            self.velocity_amrs[i] = [0.0, 0.0, 0.0]
            self.force[i] = [0.0, 0.0, 0.0]  # Newtons
            self.mass_qg[i] = constants.NEUTRINO_MASS_QG  # Default to neutrino mass, qg

            # Set phase offset (charge)
            self.offset[i] = sources_offset_rad[i]

            # Initialize activity state (active)
            self.active[i] = 1
