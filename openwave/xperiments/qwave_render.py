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
import openwave.source.spacetime as spacetime
import openwave.source.quantum_wave as qwave

# Define the architecture to be used by Taichi (GPU vs CPU)
ti.init(arch=ti.gpu)  # Use GPU if available, else fallback to CPU

render.init_UI()  # Initialize the GGUI window


def data_dashboard():
    """Display simulation data dashboard."""
    with render.gui.sub_window("DATA-DASHBOARD", 0.01, 0.01, 0.22, 0.43) as sub:
        sub.text("--- SPACETIME ---")
        sub.text("Topology: 3D BCC lattice")
        sub.text(f"Total Granules: {lattice.total_granules:,} (config.py)")
        sub.text(f"  - Corner granules: {(lattice.grid_size + 1) ** 3:,}")
        sub.text(f"  - Center granules: {lattice.grid_size ** 3:,}")
        sub.text(f"Universe Lattice Edge: {lattice.universe_edge:.1e} m")
        sub.text(f"Unit-Cells per Lattice Edge: {lattice.grid_size:,}")

        sub.text("")
        sub.text("--- Scaling-Up (for computation) ---")
        sub.text(f"Factor: {lattice.scale_factor:.1e} x Planck Scale (magnified)")
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
    global block_slice, granule_type, show_springs, radius_factor

    # Create overlay windows for controls
    with render.gui.sub_window("CONTROLS", 0.01, 0.45, 0.20, 0.16) as sub:
        block_slice = sub.checkbox("Block Slice", block_slice)
        granule_type = sub.checkbox("Granule Type Color", granule_type)
        show_springs = sub.checkbox("Show Springs (if <1k granules)", show_springs)
        radius_factor = sub.slider_float("Granule", radius_factor, 0.0, 2.0)
        if sub.button("Reset Granule"):
            radius_factor = 1.0


@ti.kernel
def normalize_positions():
    """Normalize lattice positions to 0-1 range for GGUI rendering."""
    for i in range(lattice.total_granules):
        # Normalize to 0-1 range (positions are in attometers, scale them back)
        normalized_positions[i] = lattice.positions[i] / (
            lattice.universe_edge * lattice.UNIT_SCALE
        )


@ti.kernel
def normalize_positions_sliced():
    """Normalize lattice positions to 0-1 and apply block-slicing."""
    for i in range(lattice.total_granules):
        # Normalize to 0-1 range
        # And hide front 1/8th of the lattice for see-through effect (block-slicing)
        # 0 = not in front octant (render it), 1 = in front octant (skip it)
        # Currently block-slicing don't hide granules, just move them to origin (0,0,0)
        if lattice.front_octant[i] == 0:
            normalized_positions_sliced[i] = lattice.positions[i] / (
                lattice.universe_edge * lattice.UNIT_SCALE
            )


def normalize_lattice():
    """
    Normalize granule positions for rendering (0-1 range for GGUI) & block-slicing
    block-slicing: hide front 1/8th of the lattice for see-through effect
    """

    global normalized_positions, normalized_positions_sliced

    normalized_positions = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)
    normalized_positions_sliced = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)

    # Normalize positions once before render loop
    normalize_positions()
    normalize_positions_sliced()


def normalize_granule():
    """Normalize granule radius to 0-1 range for GGUI rendering"""

    global normalized_radius

    normalized_radius = max(
        granule.radius / lattice.universe_edge, 0.0001
    )  # Ensure minimum 0.01% of screen radius for visibility


def normalize_springs():
    """Create & Normalize springs to 0-1 range for GGUI rendering"""

    global spring_lines

    # Prepare spring line endpoints if springs provided
    max_connections = 0
    spring_lines = None
    if springs is not None:
        # Count total connections for line buffer
        for i in range(lattice.total_granules):
            max_connections += springs.links_count[i]

        if max_connections > 0:
            # Allocate line endpoint buffer (2 points per line)
            spring_lines = ti.Vector.field(3, dtype=ti.f32, shape=max_connections * 2)

    # Create a field to track line index atomically
    line_counter = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def build_spring_lines():
        """Build line endpoints for spring connections."""
        # Reset counter
        line_counter[None] = 0

        # Build lines with atomic indexing to ensure correct ordering
        for i in range(lattice.total_granules):
            num_links = springs.links_count[i]
            if num_links > 0:
                # Normalized position (scale back from attometers)
                pos_i = lattice.positions[i] / (lattice.universe_edge * lattice.UNIT_SCALE)

                for j in range(num_links):
                    neighbor_idx = springs.links[i, j]
                    if neighbor_idx >= 0:  # Valid connection
                        pos_j = lattice.positions[neighbor_idx] / (
                            lattice.universe_edge * lattice.UNIT_SCALE
                        )

                        # Get current line index atomically
                        line_idx = ti.atomic_add(line_counter[None], 1)

                        # Add line endpoints (from i to j)
                        if line_idx < max_connections:  # Safety check
                            spring_lines[line_idx * 2] = pos_i
                            spring_lines[line_idx * 2 + 1] = pos_j

    # Build spring lines if springs provided
    if springs is not None and max_connections > 0:
        build_spring_lines()


def render_lattice(lattice, granule, springs=None):
    """
    Render 3D BCC lattice using GGUI's 3D scene.

    Args:
        lattice: Lattice instance containing positions and universe parameters.
                 Expected to have attributes: positions, total_granules, universe_edge
        granule: Granule instance for size reference.
                 Expected to have attribute: radius
        springs: Spring instance containing connectivity information (optional)
    """
    global block_slice, granule_type, show_springs, radius_factor

    # Initialize variables
    block_slice = False  # Block-slicing toggle
    granule_type = False  # Granule type coloring toggle
    show_springs = False  # spring visualization toggle
    radius_factor = 1.0  # Initialize granule size factor

    # Time tracking for harmonic oscillation
    t = 0.0
    last_time = time.time()

    normalize_lattice()
    normalize_granule()
    normalize_springs()

    while render.window.running:
        # Render UI overlay windows
        data_dashboard()
        controls()

        # Calculate actual elapsed time (real-time tracking)
        current_time = time.time()
        dt_real = current_time - last_time
        last_time = current_time
        t += dt_real  # Use real elapsed time instead of fixed DT

        # Update wave propagation (spring-mass dynamics with vertex wave makers)
        if springs is not None:
            qwave.propagate_qwave(lattice, springs, granule, t, dt_real, substeps=30)
        else:
            # Fallback to vertex oscillation only if no springs
            qwave.oscillate_vertex(
                lattice.positions,
                lattice.velocities,
                lattice.vertex_indices,
                lattice.vertex_equilibrium,
                lattice.vertex_directions,
                t,
            )

        # Update normalized positions for rendering (must happen after position updates)
        normalize_positions()
        normalize_positions_sliced()

        # Render granules with optional block-slicing and type-coloring
        centers = normalized_positions_sliced if block_slice else normalized_positions

        if granule_type:
            render.scene.particles(
                centers,
                radius=normalized_radius * radius_factor,
                per_vertex_color=lattice.granule_color,
            )
        else:
            render.scene.particles(
                centers,
                radius=normalized_radius * radius_factor,
                color=config.COLOR_GRANULE[1],
            )

        # Render springs if enabled and available
        if show_springs and springs is not None and spring_lines is not None:
            render.scene.lines(spring_lines, width=5, color=config.COLOR_INFRA[1])

        # Render the scene to canvas
        render.show_scene()


# ================================================================
# Main calls
# ================================================================
if __name__ == "__main__":

    # Quantum objects instantiation
    universe_edge = 3e-16  # m (default 300 attometers, contains ~10 qwaves per linear edge)
    lattice = spacetime.Lattice(universe_edge)
    granule = spacetime.Granule(lattice.unit_cell_edge)

    # Debug: Check positions after lattice init
    print(f"[INIT] pos[0]={lattice.positions[0]}, pos[513]={lattice.positions[513]}")
    if config.SPACETIME_RES <= 10000:
        springs = spacetime.Spring(lattice, granule)  # Create spring links between granules
        # Debug: Check positions after spring init
        print(f"Spring constant k: {springs.stiffness:.2e} N/m")
        print(f"[AFTER_SPRING] pos[0]={lattice.positions[0]}, pos[513]={lattice.positions[513]}")
    else:
        springs = None  # Skip springs for very high resolutions to save memory

    # Render the 3D lattice
    render_lattice(lattice, granule, springs)
