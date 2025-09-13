"""
QUANTUM SPACE
(AKA: AKASHA @yoga, WUJI @taoism, AETHER @ancient)
QUANTUM SPACE is a Wave-Medium and Propagates Wave-Motions (QUANTUM WAVE).
Modeled as an elastic fluid structure (compressible),
that allows energy to transfer from one point to the next.
"""

import numpy as np
import taichi as ti

import openwave.core.config as config
import openwave.core.constants as constants

ti.init(arch=ti.gpu)

# ==================================================================
# Physics Engine
# ==================================================================


class Granule:
    # Granule Model: The aether consists of "granules".
    # Fundamental units that vibrate and create wave patterns.
    # Their collective motion at Planck scale creates all observable phenomena.

    og_granule_radius = constants.PLANCK_LENGTH

    def __init__(self, granule_radius=og_granule_radius):
        self.radius = granule_radius  # m


@ti.data_oriented
class Lattice:
    """
    3D Body-Centered Cubic (BCC) lattice for quantum space simulation.
    BCC topology: cubic unit cells with an additional granule at the center.
    More efficient packing than simple cubic (68% vs 52% space filling).
    Uses spherical universe boundary with buffer zone for proper wave physics.
    """

    def __init__(self, universe_radius: float):
        """
        Initialize BCC lattice within a spherical boundary.

        Args:
            universe_radius: Radius of the spherical universe in meters
        """
        # Compute total volume and granule count from resolution
        total_granules = config.QSPACE_RES

        # Scale to attometers to avoid float32 precision issues
        # This keeps position values in a reasonable range (e.g., 1000 instead of 1e-15)
        universe_radius = universe_radius / constants.ATTO_PREFIX  # Convert meters to attometers

        # Define boundary zones
        self.universe_radius = universe_radius
        self.active_radius = universe_radius * 0.95  # Active simulation zone
        self.buffer_thickness = universe_radius * 0.05  # Buffer zone thickness

        # Use sphere volume for granule density calculation
        sphere_volume = (4 / 3) * np.pi * (universe_radius**3)

        # For lattice generation, we still need a bounding cube
        # The cube edge should be 2*radius to fully contain the sphere
        bounding_cube_edge = 2 * universe_radius
        bounding_volume = bounding_cube_edge**3

        # BCC has 2 granules per unit cell (8 corners shared + 1 center)
        # We want approximately total_granules within the sphere volume
        # Since only ~52% of bounding cube granules will be in sphere,
        # we need to adjust the density accordingly

        # Target density in sphere
        target_density = total_granules / sphere_volume

        # Volume per unit cell to achieve this density
        # (2 granules per unit cell in BCC)
        unit_cell_volume = 2.0 / target_density

        # Unit cell edge length (a^3 = volume)
        self.unit_cell_edge = unit_cell_volume ** (1 / 3)
        self.bounding_cube_edge = bounding_cube_edge

        # Calculate grid dimensions for the bounding cube
        # Need enough cells to fill the bounding cube
        self.grid_size = int(bounding_cube_edge / self.unit_cell_edge)

        # Maximum possible granules in bounding cube
        max_corner_count = (self.grid_size + 1) ** 3
        max_center_count = self.grid_size**3
        max_total = max_corner_count + max_center_count

        # We'll allocate for max but only use granules within sphere
        # Initialize position and velocity fields with max size
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=max_total)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=max_total)

        # Granule zone classification: 0=outside, 1=active, 2=buffer
        self.zone_type = ti.field(dtype=ti.i32, shape=max_total)

        # Track actual number of granules within sphere
        self.actual_granules = ti.field(dtype=ti.i32, shape=())

        # Populate the lattice within spherical boundary
        self._populate_spherical_bcc_lattice()

        # Get the actual count after population
        self.total_granules = self.actual_granules[None]

    @ti.kernel
    def _populate_spherical_bcc_lattice(self):
        """Populate BCC lattice positions within spherical boundary."""
        # Center of the sphere (in attometers)
        center = ti.Vector([self.universe_radius, self.universe_radius, self.universe_radius])

        valid_count = ti.i32(0)
        corner_count = (self.grid_size + 1) ** 3
        max_granules = corner_count + self.grid_size**3

        # Process all potential granule positions
        for idx in range(max_granules):
            position = ti.Vector([0.0, 0.0, 0.0])

            if idx < corner_count:
                # Corner granule: decode 3D position from linear index
                grid_dim = self.grid_size + 1
                i = idx // (grid_dim * grid_dim)
                j = (idx % (grid_dim * grid_dim)) // grid_dim
                k = idx % grid_dim

                position = ti.Vector(
                    [i * self.unit_cell_edge, j * self.unit_cell_edge, k * self.unit_cell_edge]
                )
            else:
                # Center granule: decode position with offset
                center_idx = idx - corner_count
                i = center_idx // (self.grid_size * self.grid_size)
                j = (center_idx % (self.grid_size * self.grid_size)) // self.grid_size
                k = center_idx % self.grid_size

                offset = self.unit_cell_edge / 2.0
                position = ti.Vector(
                    [
                        i * self.unit_cell_edge + offset,
                        j * self.unit_cell_edge + offset,
                        k * self.unit_cell_edge + offset,
                    ]
                )

            # Check if position is within sphere
            distance = (position - center).norm()

            if distance <= self.universe_radius:
                # Use atomic add to safely increment counter in parallel
                current_idx = ti.atomic_add(self.actual_granules[None], 1)

                # Store position at current_idx (compact storage)
                self.positions[current_idx] = position
                self.velocities[current_idx] = ti.Vector([0.0, 0.0, 0.0])

                # Classify zone: active or buffer
                if distance <= self.active_radius:
                    self.zone_type[current_idx] = 1  # Active zone
                else:
                    self.zone_type[current_idx] = 2  # Buffer zone
                    # self.positions[current_idx] = ti.Vector([0.0, 0.0, 0.0])  # reset buffer zone
            # Positions outside sphere are ignored (not stored)

    @ti.kernel
    def update_positions(self, dt: ti.f32):  # type: ignore
        """Update granule positions based on velocities (for future dynamics)."""
        for i in self.positions:
            self.positions[i] += self.velocities[i] * dt

    def get_stats(self):
        """Return lattice statistics."""
        # Count granules by zone - only count valid granules (zones 1 and 2)
        zone_counts = self.zone_type.to_numpy()[: self.total_granules]
        active_count = int(np.sum(zone_counts == 1))
        buffer_count = int(np.sum(zone_counts == 2))

        return {
            "universe_radius": self.universe_radius
            * constants.ATTO_PREFIX,  # convert back to meters
            "active_radius": self.active_radius * constants.ATTO_PREFIX,
            "buffer_thickness": self.buffer_thickness * constants.ATTO_PREFIX,
            "unit_cell_edge": self.unit_cell_edge
            * constants.ATTO_PREFIX,  # convert back to meters
            "grid_size": self.grid_size,
            "total_granules": self.total_granules,
            "active_granules": active_count,
            "buffer_granules": buffer_count,
            "sphere_volume": (4 / 3) * np.pi * (self.universe_radius * constants.ATTO_PREFIX) ** 3,
            "efficiency": self.total_granules
            / ((self.grid_size + 1) ** 3 + self.grid_size**3)
            * 100,
        }


# ==================================================================
# Rendering Engine
# ==================================================================
# TODO: Implement Rendering Engine in a separate module?


def render_lattice(lattice_instance=None):
    """
    Render 3D BCC lattice within spherical boundary using GGUI's 3D scene.

    Args:
        lattice_instance: Optional Lattice instance to render. If None, creates default.
    """
    print("Initializing 3D spherical lattice render...")

    # Use provided lattice or create default
    if lattice_instance is None:
        universe_radius = 0.5e-15  # 0.5 femtometer radius sphere
        lattice = Lattice(universe_radius)
    else:
        lattice = lattice_instance

    # Get lattice statistics for display
    stats = lattice.get_stats()
    print("3D Spherical BCC Lattice Render initialized.")
    print(f"Active granules: {stats['active_granules']:,}")
    print(f"Buffer granules: {stats['buffer_granules']:,}")
    print(f"Space efficiency: {stats['efficiency']:.1f}%")

    # Create GGUI window with 3D scene
    window = ti.ui.Window(
        "3D Spherical BCC Quantum Lattice (GGUI)",
        (config.SCREEN_RES[0], config.SCREEN_RES[1]),
        vsync=True,
    )
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    # Initial camera setup (will be overridden by orbit parameters below)
    camera.up(0, 1, 0)  # Y-axis up

    # Normalize positions for rendering (0-1 range for GGUI)
    # Create a field for normalized positions
    normalized_positions = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)

    @ti.kernel
    def normalize_positions():
        """Normalize lattice positions to 0-1 range for GGUI rendering."""
        # Center positions around 0.5 for proper sphere visualization
        center_offset = ti.Vector(
            [lattice.universe_radius, lattice.universe_radius, lattice.universe_radius]
        )
        scale = 2.0 * lattice.universe_radius  # Diameter for normalization

        for i in range(lattice.total_granules):
            # Normalize from attometer scale to 0-1 range, centered at 0.5
            normalized_positions[i] = lattice.positions[i] / scale

    # Prepare colors and radius
    granule_color = config.COLOR_GRANULE[2]  # Blue color for granules
    bkg_color = config.COLOR_SPACE[2]  # Black background

    # Granule radius as fraction of universe diameter
    normalized_radius = lattice.unit_cell_edge / (
        2 * lattice.universe_radius * 2 * np.e
    )  # radius = unit cell edge / (2e * diameter)

    # Ensure minimum radius for visibility
    min_radius = 0.0001  # Minimum 0.01% of screen
    normalized_radius = max(normalized_radius, min_radius)

    print(f"Normalized granule radius: {normalized_radius:.6f}")
    print("_______________________________")
    print(f"Creating GGUI 3D window: {config.SCREEN_RES[0]}x{config.SCREEN_RES[1]}")
    print("Starting 3D render loop...")
    print("Controls: Right-click drag to rotate, Q/A keys to zoom in/out")
    print(f"Sphere radius: {lattice.universe_radius * constants.ATTO_PREFIX:.2e} m")
    print(
        f"Active zone: 0 to {lattice.active_radius / lattice.universe_radius * 100:.0f}% of radius"
    )
    print(
        f"Buffer zone: {lattice.active_radius / lattice.universe_radius * 100:.0f}% to 100% of radius"
    )

    # Normalize positions once before render loop
    normalize_positions()

    # # Debug: Print sample positions to verify they're in correct range
    # sample_positions = normalized_positions.to_numpy()[: min(5, lattice.total_granules)]
    # print(f"Sample normalized positions (first {len(sample_positions)}):")
    # for i, pos in enumerate(sample_positions):
    #     print(f"  Granule {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # Camera orbit parameters - matching initial position looking at center
    orbit_center = [0.5, 0.5, 0.5]  # Center of the lattice

    # Calculate initial angles from the desired initial position
    # close-up start (1.5, 1.2, 1.5) >> relative to center: (1.0, 0.7, 1.0)
    # far-away start (3.67, 2.72, 3.67) >> relative to center: (3.17, 2.22, 3.17)
    initial_rel_x = 3.67 - orbit_center[0]
    initial_rel_y = 2.72 - orbit_center[1]
    initial_rel_z = 3.67 - orbit_center[2]

    # Calculate initial orbit parameters
    orbit_radius = np.sqrt(initial_rel_x**2 + initial_rel_y**2 + initial_rel_z**2)  # ~1.5
    orbit_theta = np.arctan2(initial_rel_z, initial_rel_x)  # 45 degrees
    orbit_phi = np.arccos(initial_rel_y / orbit_radius)  # angle from vertical

    mouse_sensitivity = 0.5
    last_mouse_pos = None

    while window.running:
        # Handle mouse input for orbiting
        mouse_pos = window.get_cursor_pos()

        # Check for right mouse button drag
        if window.is_pressed(ti.ui.RMB):
            if last_mouse_pos is not None:
                # Calculate mouse delta
                dx = mouse_pos[0] - last_mouse_pos[0]
                dy = mouse_pos[1] - last_mouse_pos[1]

                # Update orbit angles
                orbit_theta += dx * mouse_sensitivity * 2 * np.pi
                # Allow nearly full vertical rotation (almost 180 degrees, small margin to prevent gimbal lock)
                orbit_phi = np.clip(orbit_phi + dy * mouse_sensitivity * np.pi, 0.01, np.pi - 0.01)

            last_mouse_pos = mouse_pos
        else:
            last_mouse_pos = None

        # Handle keyboard input for zoom
        if window.is_pressed("q"):  # Zoom in
            orbit_radius *= 0.98
            orbit_radius = np.clip(orbit_radius, 0.5, 5.0)
        if window.is_pressed("a"):  # Zoom out
            orbit_radius *= 1.02
            orbit_radius = np.clip(orbit_radius, 0.5, 5.0)

        # Calculate camera position from spherical coordinates
        cam_x = orbit_center[0] + orbit_radius * np.sin(orbit_phi) * np.cos(orbit_theta)
        cam_y = orbit_center[1] + orbit_radius * np.cos(orbit_phi)
        cam_z = orbit_center[2] + orbit_radius * np.sin(orbit_phi) * np.sin(orbit_theta)

        # Update camera
        camera.position(cam_x, cam_y, cam_z)
        camera.lookat(orbit_center[0], orbit_center[1], orbit_center[2])
        camera.up(0, 1, 0)
        scene.set_camera(camera)

        # Set background color
        canvas.set_background_color(bkg_color)

        # Add ambient and directional lighting
        scene.ambient_light((0.1, 0.1, 0.15))  # Slight blue ambient
        scene.point_light(
            pos=(0.5, 1.5, 0.5), color=(1.0, 1.0, 1.0)  # Light from above center  # White light
        )
        scene.point_light(
            pos=(1.0, 0.5, 1.0),  # Secondary light from corner
            color=(0.5, 0.5, 0.5),  # Dimmer white light
        )

        # Create color field for zone visualization
        # Active zone: brighter blue, Buffer zone: reddish
        granule_colors_field = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)

        @ti.kernel
        def set_granule_colors():
            for i in range(lattice.total_granules):
                if lattice.zone_type[i] == 1:  # Active zone
                    granule_colors_field[i] = ti.Vector([0.1, 0.6, 1.0])  # Bright blue
                else:  # Buffer zone (zone_type == 2)
                    granule_colors_field[i] = ti.Vector([1.0, 0.3, 0.3])  # Red/pink for visibility

        set_granule_colors()

        # Render granules as particles (spheres) with zone-based coloring
        scene.particles(
            normalized_positions, radius=normalized_radius, per_vertex_color=granule_colors_field
        )

        # Render the scene to canvas
        canvas.scene(scene)
        window.show()


# ==================================================================
# Main calls
# ==================================================================
if __name__ == "__main__":
    print("\n===============================")
    print("SIMULATION DATA")
    print("===============================")

    # Test the new 3D spherical BCC lattice
    print("\n--- 3D Spherical BCC Lattice Test ---")
    universe_radius = 0.5e-15  # 0.5 femtometer radius sphere
    lattice = Lattice(universe_radius)
    stats = lattice.get_stats()

    print(f"Universe radius: {stats['universe_radius']:.2e} m")
    print(f"Active radius: {stats['active_radius']:.2e} m")
    print(f"Buffer thickness: {stats['buffer_thickness']:.2e} m")
    print(f"Unit cell edge: {stats['unit_cell_edge']:.2e} m")
    print(
        f"Grid size: {stats['grid_size']}x{stats['grid_size']}x{stats['grid_size']} (bounding cube)"
    )
    print(f"Total granules in sphere: {stats['total_granules']:,}")
    print(f"  - Active zone: {stats['active_granules']:,}")
    print(f"  - Buffer zone: {stats['buffer_granules']:,}")
    print(f"Sphere volume: {stats['sphere_volume']:.2e} m³")
    print(f"Space efficiency: {stats['efficiency']:.1f}% (vs bounding cube)")

    # Render the 3D spherical lattice
    print("\n--- 3D Spherical Lattice Rendering ---")
    render_lattice(lattice)  # Pass the already created lattice instance
