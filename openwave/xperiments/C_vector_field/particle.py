"""
PARTICLE MODULE

This module defines the WaveCenter class for simulating fundamental particle
"""

import taichi as ti


@ti.data_oriented
class WaveCenter:
    """
    WaveCenter represents a fundamental particle as a localized wave deflector.
    """

    def __init__(self, grid_size, num_sources, sources_position, sources_offset_deg):
        """
        Initialize the WaveCenter with given source positions and phase offsets.
        """
        # ================================================================
        # DATA STRUCTURE & INITIALIZATION
        # ================================================================

        self.num_sources = num_sources

        # Initialize Taichi fields for kernel access, converted from Python lists
        self.position_grid = ti.Vector.field(3, dtype=ti.i32, shape=num_sources)  # grid indices
        # TODO: review velocity units
        self.velocity_gridrs = ti.Vector.field(3, dtype=ti.f32, shape=num_sources)  # indices/rs
        self.offset = ti.field(dtype=ti.f32, shape=num_sources)

        # Convert phase from degrees to radians
        sources_offset_rad = [deg * ti.math.pi / 180 for deg in sources_offset_deg]

        # Copy source data to Taichi fields
        for i in range(num_sources):
            self.position_grid[i] = [
                int(sources_position[i][0] * grid_size[0]),
                int(sources_position[i][1] * grid_size[1]),
                int(sources_position[i][2] * grid_size[2]),
            ]
            self.offset[i] = sources_offset_rad[i]

        # TODO: remove debug prints
        print("Position grid shape:", self.position_grid.shape)
        print(
            "Position grid data:",
            [self.position_grid[i] for i in range(self.num_sources)],
        )
        print("Phase offset shape:", self.offset.shape)
        print("Phase offset data:", [self.offset[i] for i in range(self.num_sources)])
