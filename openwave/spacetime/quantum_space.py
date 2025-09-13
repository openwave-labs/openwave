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
    """

    def __init__(self, universe_edge: float):
        """
        Initialize BCC lattice with computed unit-cell spacing.

        Args:
            universe_edge: Edge length of the cubic universe in meters
        """
        # Compute total volume and granule count from resolution
        total_granules = config.QSPACE_RES

        # Scale to attometers to avoid float32 precision issues
        # This keeps position values in a reasonable range (e.g., 1000 instead of 1e-15)
        universe_edge = universe_edge / constants.ATTOMETER_SCALE  # Convert meters to attometers
        universe_volume = universe_edge**3

        # BCC has 2 granules per unit cell (8 corners shared + 1 center)
        # Volume per unit cell = universe_volume / (total_granules / 2)
        unit_cell_volume = universe_volume / (total_granules / 2)

        # Unit cell edge length (a^3 = volume)
        self.unit_cell_edge = unit_cell_volume ** (1 / 3)
        self.universe_edge = universe_edge

        # Calculate grid dimensions (number of unit cells per dimension)
        self.grid_size = int(universe_edge / self.unit_cell_edge)

        # Total granules: corners + centers
        # Corners: (grid_size + 1)^3, Centers: grid_size^3
        corner_count = (self.grid_size + 1) ** 3
        center_count = self.grid_size**3
        self.total_granules = corner_count + center_count

        # Initialize position and velocity fields
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.total_granules)

        # Populate the lattice
        self._populate_bcc_lattice()

    @ti.kernel
    def _populate_bcc_lattice(self):
        """Populate BCC lattice positions in a single field."""
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

                offset = self.unit_cell_edge / 2.0
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
    def update_positions(self, dt: ti.f32):  # type: ignore
        """Update granule positions based on velocities (for future dynamics)."""
        for i in self.positions:
            self.positions[i] += self.velocities[i] * dt

    def get_stats(self):
        """Return lattice statistics."""
        return {
            "universe_edge": self.universe_edge
            * constants.ATTOMETER_SCALE,  # convert back to meters
            "unit_cell_edge": self.unit_cell_edge
            * constants.ATTOMETER_SCALE,  # convert back to meters
            "grid_size": self.grid_size,
            "total_granules": self.total_granules,
            "corner_granules": (self.grid_size + 1) ** 3,
            "center_granules": self.grid_size**3,
        }


# ==================================================================
# Rendering Engine
# ==================================================================
# TODO: Implement Rendering Engine in a separate module?


def render_lattice(lattice_instance=None):
    """
    Render 3D BCC lattice using GGUI's 3D scene.

    Args:
        lattice_instance: Optional Lattice instance to render. If None, creates default.
    """
    print("Initializing 3D lattice render...")

    # Use provided lattice or create default
    if lattice_instance is None:
        universe_edge = 1e-15  # 1 femtometer cube (will be scaled to attometers internally)
        lattice = Lattice(universe_edge)
    else:
        lattice = lattice_instance

    # Get lattice statistics for display
    stats = lattice.get_stats()
    print("3D BCC Lattice Render initialized.")

    # Create GGUI window with 3D scene
    window = ti.ui.Window(
        "3D BCC Quantum Lattice (GGUI)", (config.SCREEN_RES[0], config.SCREEN_RES[1]), vsync=True
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
        for i in range(lattice.total_granules):
            # Normalize from attometer scale to 0-1 range
            normalized_positions[i] = lattice.positions[i] / lattice.universe_edge

    # Prepare colors and radius
    granule_color = config.COLOR_GRANULE[2]  # Blue color for granules
    bkg_color = config.COLOR_SPACE[2]  # Black background

    # Granule radius as fraction of unit cell
    normalized_radius = lattice.unit_cell_edge / (
        lattice.universe_edge * 2 * np.e
    )  # radius = unit cell edge / (2e)

    # Ensure minimum radius for visibility
    min_radius = 0.0001  # Minimum 0.01% of screen
    normalized_radius = max(normalized_radius, min_radius)

    print(f"Normalized granule radius: {normalized_radius:.6f}")
    print("_______________________________")
    print(f"Creating GGUI 3D window: {config.SCREEN_RES[0]}x{config.SCREEN_RES[1]}")
    print("Starting 3D render loop...")
    print("Controls: Right-click drag to rotate, Q/A keys to zoom in/out")

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

        # Render granules as particles (spheres)
        scene.particles(normalized_positions, radius=normalized_radius, color=granule_color)

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

    # Test the new 3D BCC lattice
    print("\n--- 3D BCC Lattice Test ---")
    universe_edge = 1e-15  # 1 femtometer cube
    lattice = Lattice(universe_edge)
    stats = lattice.get_stats()

    print(f"Universe edge: {stats['universe_edge']:.2e} m")
    print(f"Unit cell edge: {stats['unit_cell_edge']:.2e} m")
    print(f"Grid size: {stats['grid_size']}x{stats['grid_size']}x{stats['grid_size']}")
    print(f"Total granules: {stats['total_granules']:,}")
    print(f"  - Corner granules: {stats['corner_granules']:,}")
    print(f"  - Center granules: {stats['center_granules']:,}")

    # Render the 3D lattice
    print("\n--- 3D Lattice Rendering ---")
    render_lattice(lattice)  # Pass the already created lattice instance
