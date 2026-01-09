"""
Xperiment Launcher

Unified launcher for granule motion xperiments featuring:
- UI-based xperiment selection and switching
- Single source of truth for rendering and UI code
- Xperiment-specific parameters in /xparameters directory
"""

import webbrowser
import time
import importlib
import sys
import os
from pathlib import Path

import taichi as ti

from openwave.common import colormap, constants
from openwave.i_o import render, video

import openwave.xperiments.a_granule_motion.spacetime_medium as medium
import openwave.xperiments.a_granule_motion.spacetime_ewave as ewave
import openwave.xperiments.a_granule_motion.instrumentation as instrument

# ================================================================
# XPERIMENT PARAMETERS MANAGEMENT
# ================================================================


class XperimentManager:
    """Manages loading and switching between xperiment parameters."""

    def __init__(self):
        self.available_xperiments = self._discover_xperiments()
        self.xperiment_display_names = {}  # Cache display names from meta
        self.current_xperiment = None
        self.current_xparameters = None

    def _discover_xperiments(self):
        """Discover all available xperiment parameters in /xparameters directory."""
        parameters_dir = Path(__file__).parent / "xparameters"

        if not parameters_dir.exists():
            return []

        xperiment_files = [
            file.stem for file in parameters_dir.glob("*.py") if file.name != "__init__.py"
        ]

        return sorted(xperiment_files)

    def load_xperiment(self, xperiment_name):
        """Load xperiment parameters by name.

        Args:
            xperiment_name: Parameter file name without .py extension

        Returns:
            dict: Parameters dictionary or None if loading fails
        """
        try:
            module_path = f"openwave.xperiments.a_granule_motion.xparameters.{xperiment_name}"
            parameters_module = importlib.import_module(module_path)
            importlib.reload(parameters_module)  # Reload for fresh parameters

            self.current_xperiment = xperiment_name
            self.current_xparameters = parameters_module.XPARAMETERS

            # Cache display name from meta
            self.xperiment_display_names[xperiment_name] = parameters_module.XPARAMETERS["meta"][
                "X_NAME"
            ]

            return self.current_xparameters

        except Exception as e:
            print(f"Error loading xperiment '{xperiment_name}': {e}")
            return None

    def get_xperiment_display_name(self, xperiment_name):
        """Get display name from parameter meta or fallback to filename conversion."""
        if xperiment_name in self.xperiment_display_names:
            return self.xperiment_display_names[xperiment_name]

        # Fallback: try to load just for the name
        try:
            module_path = f"openwave.xperiments.a_granule_motion.xparameters.{xperiment_name}"
            parameters_module = importlib.import_module(module_path)
            display_name = parameters_module.XPARAMETERS["meta"]["X_NAME"]
            self.xperiment_display_names[xperiment_name] = display_name
            return display_name
        except:
            # Final fallback: convert filename
            return " ".join(word.capitalize() for word in xperiment_name.split("_"))


# ================================================================
# SIMULATION STATE
# ================================================================


class SimulationState:
    """Manages the state of the simulation."""

    def __init__(self):
        self.lattice = None
        self.granule = None
        self.elapsed_t = 0.0
        self.last_time = time.time()
        self.frame = 1
        self.peak_amplitude = constants.EWAVE_AMPLITUDE

        # Current xperiment parameters
        self.X_NAME = ""
        self.CAM_INIT = [2.00, 1.50, 1.75]
        self.UNIVERSE_SIZE = []
        self.TARGET_GRANULES = 1e6
        self.NUM_SOURCES = 1
        self.SOURCES_POSITION = []
        self.SOURCES_OFFSET_DEG = []
        self.IN_WAVE_TOGGLE = 1
        self.OUT_WAVE_TOGGLE = 1

        # UI control variables
        self.SHOW_AXIS = False
        self.TICK_SPACING = 0.25
        self.BLOCK_SLICE = False
        self.SHOW_SOURCES = False
        self.RADIUS_FACTOR = 0.5
        self.FREQ_BOOST = 10.0
        self.AMP_BOOST = 1.0
        self.PAUSED = False

        # Color control variables
        self.COLOR_THEME = "OCEAN"
        self.COLOR_PALETTE = 0

        # Data Analytics & video export toggles
        self.INSTRUMENTATION = False
        self.EXPORT_VIDEO = False
        self.VIDEO_FRAMES = 24

    def reset(self):
        """Reset simulation state for a new xperiment."""
        self.elapsed_t = 0.0
        self.last_time = time.time()
        self.frame = 1
        self.peak_amplitude = constants.EWAVE_AMPLITUDE

    def apply_xparameters(self, params):
        """Apply parameters from xperiment parameter dictionary."""
        # Meta
        self.X_NAME = params["meta"]["X_NAME"]

        # Camera
        self.CAM_INIT = params["camera"]["INITIAL_POSITION"]

        # Universe
        universe = params["universe"]
        self.UNIVERSE_SIZE = list(universe["SIZE"])
        self.TARGET_GRANULES = universe["TARGET_GRANULES"]

        # Wave sources
        sources = params["wave_sources"]
        self.NUM_SOURCES = sources["COUNT"]
        self.SOURCES_POSITION = sources["POSITION"]
        self.SOURCES_OFFSET_DEG = sources["PHASE_OFFSETS_DEG"]
        self.IN_WAVE_TOGGLE = sources["IN_WAVE_TOGGLE"]
        self.OUT_WAVE_TOGGLE = sources["OUT_WAVE_TOGGLE"]

        # UI defaults
        ui = params["ui_defaults"]
        self.SHOW_AXIS = ui["SHOW_AXIS"]
        self.TICK_SPACING = ui["TICK_SPACING"]
        self.BLOCK_SLICE = ui["BLOCK_SLICE"]
        self.SHOW_SOURCES = ui["SHOW_SOURCES"]
        self.RADIUS_FACTOR = ui["RADIUS_FACTOR"]
        self.FREQ_BOOST = ui["FREQ_BOOST"]
        self.AMP_BOOST = ui["AMP_BOOST"]
        self.PAUSED = ui["PAUSED"]

        # Color defaults
        color = params["color_defaults"]
        self.COLOR_THEME = color["COLOR_THEME"]
        self.COLOR_PALETTE = color["COLOR_PALETTE"]

        # Data Analytics & video export toggles
        diag = params["analytics"]
        self.INSTRUMENTATION = diag["INSTRUMENTATION"]
        self.EXPORT_VIDEO = diag["EXPORT_VIDEO"]
        self.VIDEO_FRAMES = diag["VIDEO_FRAMES"]

    def initialize_lattice(self):
        """Initialize or reinitialize the lattice and granule objects."""
        self.lattice = medium.BCCLattice(
            self.UNIVERSE_SIZE, self.TARGET_GRANULES, self.COLOR_THEME
        )
        self.granule = medium.BCCGranule(
            self.lattice.unit_cell_edge, self.lattice.max_universe_edge
        )


# ================================================================
# UI OVERLAY WINDOWS
# ================================================================


def display_xperiment_launcher(xperiment_mgr, state):
    """Display xperiment launcher UI with selectable xperiments.

    Args:
        xperiment_mgr: XperimentManager instance
        state: SimulationState instance

    Returns:
        str or None: Selected xperiment name or None
    """
    selected_xperiment = None

    with render.gui.sub_window("XPERIMENT LAUNCHER", 0.00, 0.00, 0.14, 0.33) as sub:
        sub.text("(needs window reload)", color=colormap.LIGHT_BLUE[1])
        for xp_name in xperiment_mgr.available_xperiments:
            display_name = xperiment_mgr.get_xperiment_display_name(xp_name)
            is_current = xp_name == xperiment_mgr.current_xperiment

            if sub.checkbox(display_name, is_current) and not is_current:
                selected_xperiment = xp_name

        if sub.button("Close Launcher (esc)"):
            render.window.running = False

    return selected_xperiment


def display_controls(state):
    """Display the controls UI overlay."""
    with render.gui.sub_window("CONTROLS", 0.00, 0.34, 0.16, 0.27) as sub:
        state.SHOW_AXIS = sub.checkbox(f"Axis (ticks: {state.TICK_SPACING})", state.SHOW_AXIS)
        state.BLOCK_SLICE = sub.checkbox("Block Slice", state.BLOCK_SLICE)
        state.SHOW_SOURCES = sub.checkbox("Show Wave Centers (sources)", state.SHOW_SOURCES)
        state.IN_WAVE_TOGGLE = sub.checkbox("Incoming Wave (Full Amp)", state.IN_WAVE_TOGGLE)
        state.OUT_WAVE_TOGGLE = sub.checkbox("Outgoing Wave (Amp Falloff)", state.OUT_WAVE_TOGGLE)
        state.RADIUS_FACTOR = sub.slider_float("Granule", state.RADIUS_FACTOR, 0.1, 2.0)
        state.FREQ_BOOST = sub.slider_float("f Boost", state.FREQ_BOOST, 0.1, 10.0)
        state.AMP_BOOST = sub.slider_float("Amp Boost", state.AMP_BOOST, 0.1, 5.0)
        if state.PAUSED:
            if sub.button(">> PROPAGATE EWAVE >>"):
                state.PAUSED = False
        else:
            if sub.button("Pause"):
                state.PAUSED = True


def display_wave_menu(state):
    """Display wave properties selection menu."""
    with render.gui.sub_window("WAVE MENU", 0.00, 0.73, 0.14, 0.14) as sub:
        if sub.checkbox("Displacement (orange)", state.COLOR_PALETTE == 3):
            state.COLOR_PALETTE = 3
        if sub.checkbox("Amplitude (ironbow)", state.COLOR_PALETTE == 5):
            state.COLOR_PALETTE = 5
        if sub.checkbox("Granule Type Color", state.COLOR_PALETTE == 0):
            state.COLOR_PALETTE = 0
        if sub.checkbox("Default Color", state.COLOR_PALETTE == 99):
            state.COLOR_PALETTE = 99
        if state.COLOR_PALETTE == 6:  # Display orange gradient palette
            render.canvas.triangles(og_palette_vertices, per_vertex_color=og_palette_colors)
            with render.gui.sub_window("displacement", 0.00, 0.67, 0.08, 0.06) as sub:
                sub.text(f"0       {state.peak_amplitude:.0e}m")
        if state.COLOR_PALETTE == 3:  # Display ironbow gradient palette
            render.canvas.triangles(ib_palette_vertices, per_vertex_color=ib_palette_colors)
            with render.gui.sub_window("amplitude", 0.00, 0.67, 0.08, 0.06) as sub:
                sub.text(f"0       {state.peak_amplitude:.0e}m")


def display_level_specs(state, level_bar_vertices):
    """Display OpenWave level specifications overlay."""
    render.canvas.triangles(level_bar_vertices, color=colormap.WHITE[1])
    with render.gui.sub_window("GRANULE-MOTION METHOD", 0.84, 0.01, 0.16, 0.16) as sub:
        sub.text("Medium: Granule-Based Lattice")
        sub.text("Data-Structure: Vector Field")
        sub.text(f"Source: {state.NUM_SOURCES} Harmonic Oscillators")
        sub.text("Coupling: Phase Sync")
        sub.text("Propagation: Analytical Function")
        if sub.button("Wave Notation Guide"):
            webbrowser.open(
                "https://github.com/openwave-labs/openwave/blob/main/openwave/common/wave_notation.md"
            )


def display_data_dashboard(state):
    """Display simulation data dashboard."""
    with render.gui.sub_window("DATA-DASHBOARD", 0.84, 0.44, 0.16, 0.56) as sub:
        sub.text("--- SPACETIME ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Medium Density: {constants.MEDIUM_DENSITY:.1e} kg/m³")
        sub.text(f"eWAVE Speed (c): {constants.EWAVE_SPEED:.1e} m/s")

        sub.text("\n--- SIMULATION DOMAIN ---", color=colormap.LIGHT_BLUE[1])
        sub.text(
            f"Universe: {state.lattice.max_universe_edge:.1e} m ({state.lattice.max_universe_edge_lambda:.1f} waves)"
        )
        sub.text(f"Granule Count: {state.lattice.granule_count:,}")
        sub.text(
            f"Grid Size: {state.lattice.grid_size[0]} x {state.lattice.grid_size[1]} x {state.lattice.grid_size[2]}"
        )
        sub.text(f"Unit-Cell Edge: {state.lattice.unit_cell_edge:.2e} m")

        sub.text("\n--- RESOLUTION (scaled-up) ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Scale-up Factor: {state.lattice.scale_factor:.1e}x")
        sub.text(f"eWave: {state.lattice.ewave_res:.0f} granules/wave (>10)")
        if state.lattice.ewave_res < 10:
            sub.text(f"*** WARNING: Undersampling! ***", color=(1.0, 0.0, 0.0))
        sub.text(f"Granule Radius: {state.granule.radius * state.RADIUS_FACTOR:.2e} m")
        sub.text(f"Granule Mass: {state.granule.mass:.2e} kg")

        sub.text("\n--- ENERGY-WAVE ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"eWAVE Amplitude: {constants.EWAVE_AMPLITUDE:.1e} m")
        sub.text(f"eWAVE Frequency: {constants.EWAVE_FREQUENCY:.1e} Hz")
        sub.text(f"eWAVE Wavelength: {constants.EWAVE_LENGTH:.1e} m")
        sub.text(
            f"ENERGY: {state.lattice.nominal_energy:.1e} J ({state.lattice.nominal_energy_kWh:.1e} KWh)"
        )

        sub.text("\n--- TIME MICROSCOPE ---", color=colormap.LIGHT_BLUE[1])
        slowed_mo = constants.EWAVE_FREQUENCY / state.FREQ_BOOST
        fps = 0 if state.elapsed_t == 0 else state.frame / state.elapsed_t
        sub.text(f"Timesteps (frames): {state.frame}")
        sub.text(f"Sim Time: {state.elapsed_t / slowed_mo:.2e}s ({fps * slowed_mo:.0e} FPS)")
        sub.text(f"Clock Time: {state.elapsed_t:.2f}s ({fps:.0f} FPS)")
        sub.text(f"(1s sim time takes {slowed_mo / (60*60*24*365):.0e}y)")


# ================================================================
# XPERIMENT RENDERING
# ================================================================


def initialize_xperiment(state):
    """Initialize color palettes, wave sources and instrumentation.

    Args:
        state: SimulationState instance with xperiment parameters
    """
    global og_palette_vertices, og_palette_colors
    global ib_palette_vertices, ib_palette_colors
    global level_bar_vertices

    # Initialize color palettes for gradient rendering and level indicator
    og_palette_vertices, og_palette_colors = colormap.get_palette_scale(
        colormap.orange, 0.00, 0.66, 0.079, 0.01
    )
    ib_palette_vertices, ib_palette_colors = colormap.get_palette_scale(
        colormap.ironbow, 0.00, 0.66, 0.079, 0.01
    )
    level_bar_vertices = colormap.get_level_bar_geometry(0.84, 0.00, 0.159, 0.01)

    # Initialize wave sources
    ewave.build_source_vectors(
        state.NUM_SOURCES, state.SOURCES_POSITION, state.SOURCES_OFFSET_DEG, state.lattice
    )

    if state.INSTRUMENTATION:
        instrument.print_initial_parameters()


def compute_wave_motion(state):
    """Compute granule oscillations from wave superposition.

    Args:
        state: SimulationState instance with xperiment parameters
    """
    # Apply wave oscillations from all sources
    ewave.oscillate_granules(
        state.lattice.position_am,
        state.lattice.equilibrium_am,
        state.lattice.amplitude_am,
        state.lattice.velocity_am,
        state.lattice.granule_var_color,
        state.FREQ_BOOST,
        state.AMP_BOOST,
        state.COLOR_PALETTE,
        state.NUM_SOURCES,
        state.IN_WAVE_TOGGLE,
        state.OUT_WAVE_TOGGLE,
        state.elapsed_t,
    )

    # Update normalized positions for rendering (with optional block-slicing)
    state.lattice.normalize_to_screen(1 if state.BLOCK_SLICE else 0)

    # IN-FRAME DATA SAMPLING & INSTRUMENTATION ==================================
    # Sample peak amplitude periodically (every 30 frames)
    if state.frame % 30 == 0:
        state.peak_amplitude = ewave.peak_amplitude_am[None] * constants.ATTOMETER
        ewave.update_lattice_energy(state.lattice)

    if state.INSTRUMENTATION:
        instrument.print_wave_diagnostics(state.elapsed_t, state.frame, print_interval=100)


def render_elements(state):
    """Render granules and wave sources with appropriate coloring."""
    radius_render = state.granule.radius_screen * state.RADIUS_FACTOR

    # Render granules with color scheme
    if state.COLOR_PALETTE == 0:
        render.scene.particles(
            state.lattice.position_screen,
            radius=radius_render,
            per_vertex_color=state.lattice.granule_type_color,
        )
    elif state.COLOR_PALETTE == 3 or state.COLOR_PALETTE == 5:
        render.scene.particles(
            state.lattice.position_screen,
            radius=radius_render,
            per_vertex_color=state.lattice.granule_var_color,
        )
    else:
        render.scene.particles(
            state.lattice.position_screen,
            radius=radius_render,
            color=colormap.COLOR_MEDIUM[1],
        )

    # Render wave sources
    if state.SHOW_SOURCES:
        render.scene.particles(
            centers=ewave.sources_pos_field,
            radius=state.granule.radius_screen * 5,
            color=colormap.COLOR_SOURCE[1],
        )


# ================================================================
# MAIN LOOP
# ================================================================


def main():
    """Main entry point for xperiment launcher."""
    # Parse command-line argument for xperiment selection
    selected_xperiment_arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Initialize Taichi
    ti.init(arch=ti.gpu, log_level=ti.WARN)  # GPU preferred, suppress info logs

    # Initialize xperiment manager and state
    xperiment_mgr = XperimentManager()
    state = SimulationState()

    # Load xperiment (from CLI arg or default)
    default_xperiment = selected_xperiment_arg or "spacetime_vibration"
    if default_xperiment not in xperiment_mgr.available_xperiments:
        print(f"Error: Xperiment '{default_xperiment}' not found!")
        return

    params = xperiment_mgr.load_xperiment(default_xperiment)
    if not params:
        return

    state.apply_xparameters(params)
    state.initialize_lattice()
    initialize_xperiment(state)

    # Initialize GGUI rendering
    render.init_UI(state.UNIVERSE_SIZE, state.TICK_SPACING, state.CAM_INIT)

    # Main rendering loop
    while render.window.running:
        render.init_scene(state.SHOW_AXIS)  # Initialize scene with lighting and camera

        # Handle ESC key for window close
        if render.window.is_pressed(ti.ui.ESCAPE):
            render.window.running = False
            break

        # Display UI overlays
        new_xperiment = display_xperiment_launcher(xperiment_mgr, state)
        display_controls(state)

        # Handle xperiment switching via process replacement
        if new_xperiment:
            print("\n================================================================")
            print("XPERIMENT LAUNCH")
            print(f"Now running: {new_xperiment}\n")

            sys.stdout.flush()
            sys.stderr.flush()
            render.window.running = False

            # os.execv replaces current process (macOS may show harmless warning)
            os.execv(sys.executable, [sys.executable, __file__, new_xperiment])

        if not state.PAUSED:
            # Update elapsed time and run simulation step
            current_time = time.time()
            state.elapsed_t += current_time - state.last_time  # Real-time delta accumulation
            state.last_time = current_time

            compute_wave_motion(state)
            state.frame += 1
        else:
            # Prevent time jump on resume
            state.last_time = time.time()

        # Render scene elements
        render_elements(state)

        # Display additional UI elements and scene
        display_wave_menu(state)
        display_data_dashboard(state)
        display_level_specs(state, level_bar_vertices)
        render.show_scene()

        # Capture frame for video export (stops after VIDEO_FRAMES)
        if state.EXPORT_VIDEO:
            video.export_frame(state.frame, state.VIDEO_FRAMES)


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    main()
