"""
Run sample xperiments shipped with the OpenWave package, tweak them, or create your own

eg. Tweak this xperiment script changing universe_edge = 0.1 m at __main__ entry point
(the approximate size of a tesseract) and simulate this artifact energy content,
sourced from the element aether.
"""

import taichi as ti
import time

import openwave.common.config as config
import openwave.common.render as render
import openwave.spacetime.space_medium_latticebcc as space_medium
import openwave.spacetime.quantum_wave_springmass as qwave

# Define the architecture to be used by Taichi (GPU vs CPU)
ti.init(arch=ti.gpu)  # Use GPU if available, else fallback to CPU

# ================================================================
# Xperiment Parameters & Quantum Objects Instantiation
# ================================================================

universe_edge = 3e-16  # m (default 300 attometers, contains ~10 qwaves per linear edge)
target_particles = 1e6  # target particle count, granularity (impacts performance)

lattice = space_medium.LatticeBCC(universe_edge, target_particles)
granule = space_medium.Granule(lattice.unit_cell_edge)
neighbors = space_medium.NeighborsBCC(lattice)  # Create neighbor links between granules

# Note:
# This is a scaled value for computational feasibility
# Real physical stiffness causes timestep requirements beyond computational feasibility
stiffness = 1e-12  # N/m, spring stiffness (tuned for stability and wave speed)
# stiffness = constants.COULOMB_CONSTANT / constants.PLANCK_LENGTH
# stiffness = constants.COULOMB_CONSTANT / granule.radius
# stiffness = lattice.scale_factor * constants.COULOMB_CONSTANT


# ================================================================
# Xperiment UI and overlay windows
# ================================================================

render.init_UI()  # Initialize the GGUI window


def xperiment_specs():
    """Display xperiment definitions & specs."""
    with render.gui.sub_window("XPERIMENT: Spring-Mass", 0.01, 0.01, 0.20, 0.14) as sub:
        sub.text("Medium: 3D BCC lattice")
        sub.text("Granule Type: Point Mass")
        sub.text("Coupling: 8-way neighbors springs")
        sub.text("QWave Driver: 8 Vertex Oscillators")
        sub.text("QWave Propagation: Spring-Mass Dynamics")


def data_dashboard():
    """Display simulation data dashboard."""
    with render.gui.sub_window("DATA-DASHBOARD", 0.01, 0.16, 0.20, 0.41) as sub:
        sub.text("--- SPACE-MEDIUM ---")
        sub.text(f"Universe Edge: {lattice.universe_edge:.1e} m")
        sub.text(f"Particle Count: {lattice.total_granules:,} granules")
        sub.text(f"Spring Stiffness: {stiffness:.1e} N/m")

        sub.text("")
        sub.text("--- Scaling-Up (for computation) ---")
        sub.text(f"Factor: {lattice.scale_factor:.1e} x Planck Scale magnified")
        sub.text(f"Unit-Cells per Lattice Edge: {lattice.grid_size:,}")
        sub.text(f"BCC Unit-Cell Edge: {lattice.unit_cell_edge:.2e} m")
        sub.text(f"Granule Radius: {granule.radius:.2e} m")
        sub.text(f"Granule Mass: {granule.mass:.2e} kg")

        sub.text("")
        sub.text("--- Simulation Resolution (linear) ---")
        sub.text(f"QWave: {lattice.qwave_res:.0f} granules/qwavelength (min 2)")
        if lattice.qwave_res < 2:
            sub.text(f"*** WARNING: Undersampling! ***", color=(1.0, 0.0, 0.0))
        sub.text(f"Universe: {lattice.uni_res:.1f} qwaves/universe-edge")

        sub.text("")
        sub.text("--- Universe Lattice Wave Energy ---")
        sub.text(f"Energy: {lattice.energy:.1e} J ({lattice.energy_kWh:.1e} KWh)")
        sub.text(f"{lattice.energy_years:,.1e} Years of global energy use")


def controls():
    """Render the controls UI overlay."""
    global block_slice, granule_type, show_links, radius_factor

    # Create overlay windows for controls
    with render.gui.sub_window("CONTROLS", 0.01, 0.58, 0.20, 0.16) as sub:
        block_slice = sub.checkbox("Block Slice", block_slice)
        granule_type = sub.checkbox("Granule Type Color", granule_type)
        show_links = sub.checkbox("Show Links (if <1k granules)", show_links)
        radius_factor = sub.slider_float("Granule", radius_factor, 0.0, 2.0)
        if sub.button("Reset Granule"):
            radius_factor = 1.0


# ================================================================
# Xperiment Normalization to Screen Coordinates
# ================================================================


@ti.kernel
def normalize_lattice(enable_slice: ti.i32):  # type: ignore
    """Normalize lattice positions to 0-1 range for GGUI rendering."""
    for i in range(lattice.total_granules):
        # Normalize to 0-1 range (positions are in attometers, scale them back)
        if enable_slice == 1 and lattice.front_octant[i] == 1:
            # Block-slicing enabled: hide front octant granules by moving to origin
            normalized_positions[i] = ti.Vector([0.0, 0.0, 0.0])
        else:
            # Normal rendering: normalize to 0-1 range
            normalized_positions[i] = lattice.positions[i] / (
                lattice.universe_edge * lattice.UNIT_SCALE
            )


def normalize_granule():
    """Normalize granule radius to 0-1 range for GGUI rendering"""

    global normalized_radius

    normalized_radius = max(
        granule.radius / lattice.universe_edge, 0.0001
    )  # Ensure minimum 0.01% of screen radius for visibility


def normalize_neighbors_links():
    """Create & Normalize links to 0-1 range for GGUI rendering"""

    global link_lines

    # Prepare link line endpoints
    max_connections = 0

    # Count total connections for line buffer
    for i in range(lattice.total_granules):
        max_connections += neighbors.links_count[i]
    if max_connections > 0:
        # Allocate line endpoint buffer (2 points per line)
        link_lines = ti.Vector.field(3, dtype=ti.f32, shape=max_connections * 2)

    # Create a field to track line index atomically
    line_counter = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def build_link_lines():
        """Build line endpoints for BBC granule connections."""
        # Reset counter
        line_counter[None] = 0

        # Build lines with atomic indexing to ensure correct ordering
        for i in range(lattice.total_granules):
            num_links = neighbors.links_count[i]
            if num_links > 0:
                # Normalized position (scale back from attometers)
                pos_i = lattice.positions[i] / (lattice.universe_edge * lattice.UNIT_SCALE)

                for j in range(num_links):
                    neighbor_idx = neighbors.links[i, j]
                    if neighbor_idx >= 0:  # Valid connection
                        pos_j = lattice.positions[neighbor_idx] / (
                            lattice.universe_edge * lattice.UNIT_SCALE
                        )

                        # Get current line index atomically
                        line_idx = ti.atomic_add(line_counter[None], 1)

                        # Add line endpoints (from i to j)
                        if line_idx < max_connections:  # Safety check
                            link_lines[line_idx * 2] = pos_i
                            link_lines[line_idx * 2 + 1] = pos_j

    # Build link lines
    if max_connections > 0:
        build_link_lines()


# ================================================================
# Xperiment Rendering
# ================================================================


def render_lattice(lattice, granule, neighbors):
    """
    Render 3D BCC lattice using GGUI's 3D scene.

    Args:
        lattice: Lattice instance containing positions and universe parameters.
                 Expected to have attributes: positions, total_granules, universe_edge
        granule: Granule instance for size reference.
                 Expected to have attribute: radius
        neighbors: NeighborsBCC instance containing connectivity information (optional)
    """
    global block_slice, granule_type, show_links, radius_factor
    global link_lines
    global normalized_positions

    # Initialize variables
    block_slice = False  # Block-slicing toggle
    granule_type = False  # Granule type coloring toggle
    show_links = False  # link visualization toggle
    radius_factor = 1.0  # Initialize granule size factor
    link_lines = None  # Link line buffer

    # Time tracking for harmonic oscillation
    t = 0.0
    last_time = time.time()

    # Initialize normalized positions (0-1 range for GGUI) & block-slicing
    # block-slicing: hide front 1/8th of the lattice for see-through effect
    normalized_positions = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)
    normalize_granule()
    if target_particles <= 1e3:
        normalize_neighbors_links()  # Skip neighbors for very high resolutions to save memory

    while render.window.running:
        # Render UI overlay windows
        render.cam_instructions()
        controls()
        data_dashboard()
        xperiment_specs()

        # Calculate actual elapsed time (real-time tracking)
        current_time = time.time()
        dt_real = current_time - last_time
        last_time = current_time
        t += dt_real  # Use real elapsed time instead of fixed DT

        # Update wave propagation (spring-mass dynamics with vertex wave makers)
        qwave.propagate_qwave(lattice, granule, neighbors, stiffness, t, dt_real, substeps=30)

        # Update normalized positions for rendering (must happen after position updates)
        # with optional block-slicing (see-through effect)
        normalize_lattice(1 if block_slice else 0)

        # Render granules with optional type-coloring
        if granule_type:
            render.scene.particles(
                normalized_positions,
                radius=normalized_radius * radius_factor,
                per_vertex_color=lattice.granule_color,
            )
        else:
            render.scene.particles(
                normalized_positions,
                radius=normalized_radius * radius_factor,
                color=config.COLOR_GRANULE[1],
            )

        # Render spring links if enabled and available
        if show_links and link_lines is not None:
            render.scene.lines(link_lines, width=5, color=config.COLOR_INFRA[1])

        # Render the scene to canvas
        render.show_scene()


# ================================================================
# Main calls
# ================================================================
if __name__ == "__main__":

    # Debug: Check positions after lattice init
    print(f"[INIT] pos[0]={lattice.positions[0]}, pos[513]={lattice.positions[513]}")

    # Render the 3D lattice
    render_lattice(lattice, granule, neighbors)
