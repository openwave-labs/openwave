"""
XPERIMENT: XPBD Quantum-Wave Oscillation
Run sample XPERIMENTS shipped with the OpenWave package or create your own
Tweak universe_edge and other parameters to explore different scales.
"""

import taichi as ti
import time

from openwave.common import config
from openwave.common import constants
from openwave.common import equations
from openwave.common import render

import openwave.spacetime.medium_bcclattice as medium
import openwave.spacetime.qwave_xpbd as qwave

# Define the architecture to be used by Taichi (GPU vs CPU)
ti.init(arch=ti.gpu)  # Use GPU if available, else fallback to CPU

# ================================================================
# Xperiment Parameters & Quantum Objects Instantiation
# ================================================================

UNIVERSE_EDGE = 1e-16  # m (default 300 attometers, contains ~10 qwaves per linear edge)
TARGET_PARTICLES = 1e4  # target particle count, granularity (impacts performance)

# slow-motion (divides frequency for human-visible motion, time microscope)
SLOW_MO = 1e25  # (1 = real-time, 10 = 10x slower, 1e25 = 10 * trillion * trillions FPS)

lattice = medium.BCCLattice(UNIVERSE_EDGE, TARGET_PARTICLES)
granule = medium.Granule(lattice.unit_cell_edge)
neighbors = medium.BCCNeighbors(lattice)  # Create neighbor links between granules

# PHYSICAL STIFFNESS (calculated for speed of light wave propagation)
# With XPBD, we can finally use REAL physical values!
# Using EWT spring-mass equation: k = (2πf_n)² * m
# where f_n = natural frequency = c/λ_lattice
# For wave propagation at speed c in lattice with spacing L:
#   λ_lattice ≈ 2L (minimum resolvable wavelength)
#   f_n = c / (2L)
# For 79x79x79 grid (1M particles): k ≈ 2.66e21 N/m
STIFFNESS = equations.stiffness_from_frequency(neighbors.natural_frequency, granule.mass)


# ================================================================
# Xperiment UI and overlay windows
# ================================================================

render.init_UI()  # Initialize the GGUI window


def xperiment_specs():
    """Display xperiment definitions & specs."""
    with render.gui.sub_window("XPERIMENT: XPBD Quantum-Wave", 0.00, 0.00, 0.19, 0.14) as sub:
        sub.text("Medium: BCC lattice")
        sub.text("Granule Type: Point Mass")
        sub.text("Coupling: 8-way distance constraints")
        sub.text("QWave Source: 8 Vertex Oscillators")
        sub.text("QWave Propagation: XPBD")


def data_dashboard():
    """Display simulation data dashboard."""
    with render.gui.sub_window("DATA-DASHBOARD", 0.00, 0.55, 0.19, 0.45) as sub:
        sub.text("--- SPACETIME-MEDIUM ---")
        sub.text(f"Universe Edge: {lattice.universe_edge:.1e} m")
        sub.text(f"Granule Count: {lattice.total_granules:,} particles")
        sub.text(f"Medium Density: {constants.MEDIUM_DENSITY:.1e} kg/m³")
        sub.text(f"Natural frequency: {neighbors.natural_frequency:.1e} Hz")
        sub.text(f"Spring Stiffness: {STIFFNESS:.1e} N/m")

        sub.text("")
        sub.text("--- Scaling-Up (for computation) ---")
        sub.text(f"Factor: {lattice.scale_factor:.1e} x Planck Scale")
        sub.text(f"Unit-Cells per Lattice Edge: {lattice.grid_size:,}")
        sub.text(f"BCC Unit-Cell Edge: {lattice.unit_cell_edge:.2e} m")
        sub.text(f"Granule Radius: {granule.radius:.2e} m")
        sub.text(f"Granule Mass: {granule.mass:.2e} kg")

        sub.text("")
        sub.text("--- Simulation Resolution (linear) ---")
        sub.text(f"QWave: {lattice.qwave_res:.0f} granules/qwavelength (>10)")
        if lattice.qwave_res < 10:
            sub.text(f"*** WARNING: Undersampling! ***", color=(1.0, 0.0, 0.0))
        sub.text(f"Universe: {lattice.uni_res:.1f} qwaves/universe-edge")

        sub.text("")
        sub.text("--- Universe Lattice Wave Energy ---")
        sub.text(f"Energy: {lattice.energy:.1e} J ({lattice.energy_kWh:.1e} KWh)")
        sub.text(f"{lattice.energy_years:,.1e} Years of global energy use")


def controls():
    """Render the controls UI overlay."""
    global block_slice, granule_type, show_links, radius_factor, slomo_factor, amplitude_boost

    # Create overlay windows for controls
    with render.gui.sub_window("CONTROLS", 0.85, 0.00, 0.15, 0.23) as sub:
        render.show_axis = sub.checkbox("Axis", render.show_axis)
        block_slice = sub.checkbox("Block Slice", block_slice)
        granule_type = sub.checkbox("Granule Type Color", granule_type)
        show_links = sub.checkbox("Show Links (<1k granules)", show_links)
        radius_factor = sub.slider_float("Granule", radius_factor, 0.0, 2.0)
        # if sub.button("Reset Granule"):
        #     radius_factor = 1.0
        slomo_factor = sub.slider_float("Speed", slomo_factor, 0.1, 10.0)
        amplitude_boost = sub.slider_float("Amp Boost", amplitude_boost, 1.0, 5.0)


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
            normalized_positions[i] = lattice.positions_am[i] / lattice.universe_edge_am


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
        """Build line endpoints for BCC granule connections."""
        # Reset counter
        line_counter[None] = 0

        # Build lines with atomic indexing to ensure correct ordering
        for i in range(lattice.total_granules):
            num_links = neighbors.links_count[i]
            if num_links > 0:
                # Normalized position (scale back from attometers)
                pos_i = lattice.positions_am[i] / lattice.universe_edge_am

                for j in range(num_links):
                    neighbor_idx = neighbors.links[i, j]
                    if neighbor_idx >= 0:  # Valid connection
                        pos_j = lattice.positions_am[neighbor_idx] / lattice.universe_edge_am

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


def render_xperiment(lattice, granule, neighbors):
    """
    Render 3D BCC lattice using GGUI's 3D scene.

    Args:
        lattice: Lattice instance containing positions and universe parameters.
        granule: Granule instance for size reference.
        neighbors: BCCNeighbors instance containing connectivity information (optional)
    """
    global block_slice, granule_type, show_links, radius_factor, slomo_factor, amplitude_boost
    global link_lines
    global normalized_positions

    # Initialize variables
    block_slice = False  # Block-slicing toggle
    granule_type = True  # Granule type coloring toggle
    show_links = True  # link visualization toggle
    radius_factor = 0.5  # Initialize granule size factor
    slomo_factor = 1.0  # Initialize slow motion factor
    amplitude_boost = 1.0  # Initialize amplitude boost factor
    link_lines = None  # Link line buffer

    # Time tracking for harmonic oscillation
    t = 0.0
    last_time = time.time()

    # Initialize wave diagnostics
    qwave.init_wave_diagnostics(measurement_interval=1.0)

    # Initialize normalized positions (0-1 range for GGUI) & block-slicing
    # block-slicing: hide front 1/8th of the lattice for see-through effect
    normalized_positions = ti.Vector.field(3, dtype=ti.f32, shape=lattice.total_granules)
    normalize_granule()
    if TARGET_PARTICLES <= 1e3:
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

        # Update wave propagation using XPBD constraint solver
        # XPBD = Extended Position-Based Dynamics (unconditionally stable!)
        # Following "Small Steps" + "Unified Particle Physics" papers:
        # - 100 substeps with 1 iteration each (error scales as Δt²)
        # - Jacobi iteration with constraint averaging (parallel GPU-friendly)
        # - SOR omega=1.5 for faster convergence
        # - Damping=0.999 per substep (explicit dissipation)
        qwave.propagate_qwave(
            lattice,
            granule,
            neighbors,
            STIFFNESS,  # REAL physical value!
            t,
            dt_real,
            substeps=100,  # 100 recommended from papers
            slow_mo=SLOW_MO / slomo_factor,
            damping=0.999,  # 0.1% energy loss per substep
            omega=1.5,  # SOR parameter for faster convergence
            amplitude_boost=amplitude_boost,  # Visibility boost for scaled lattices
        )

        # Update normalized positions for rendering (must happen after position updates)
        # with optional block-slicing (see-through effect)
        normalize_lattice(1 if block_slice else 0)

        # Probe wave diagnostics (measurements happen automatically at configured interval)
        qwave.probe_wave_diagnostics(
            lattice,
            neighbors,
            t,
            current_time,
            SLOW_MO / slomo_factor,
        )

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

    # Render the 3D lattice
    render_xperiment(lattice, granule, neighbors)
