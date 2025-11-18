"""
L0 XPERIMENT LAUNCHER

Unified launcher for Level-0 granule-based xperiments featuring:
- UI-based xperiment selection and switching
- Single source of truth for rendering and UI code
- Xperiment-specific parameters in /_xparameters directory
"""

import taichi as ti
import time
import importlib
import sys
import os
from pathlib import Path

from openwave.common import colormap, constants
from openwave._io import render, video

import openwave.spacetime.medium_level0 as medium
import openwave.spacetime.wave_engine_level0 as ewave
import openwave.validations.wave_diagnostics as diagnostics

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
        """Discover all available xperiment parameters in /_xparameters directory."""
        parameters_dir = Path(__file__).parent / "_xparameters"

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
            module_path = (
                f"openwave.xperiments.4_level0_granule_based._xparameters.{xperiment_name}"
            )
            parameters_module = importlib.import_module(module_path)
            importlib.reload(parameters_module)  # Reload for fresh parameters

            self.current_xperiment = xperiment_name
            self.current_xparameters = parameters_module.XPARAMETERS

            # Cache display name from meta
            self.xperiment_display_names[xperiment_name] = parameters_module.XPARAMETERS["meta"][
                "name"
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
            module_path = (
                f"openwave.xperiments.4_level0_granule_based._xparameters.{xperiment_name}"
            )
            parameters_module = importlib.import_module(module_path)
            display_name = parameters_module.XPARAMETERS["meta"]["name"]
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
        self.frame = 0
        self.peak_amplitude = 0.0

        # Current xperiment parameters
        self.X_NAME = ""
        self.CAM_INIT = [2.00, 1.50, 1.75]
        self.UNIVERSE_SIZE = []
        self.TARGET_GRANULES = 1e6
        self.TICK_SPACING = 0.25
        self.NUM_SOURCES = 1
        self.SOURCES_POSITION = []
        self.SOURCES_PHASE_DEG = []
        self.COLOR_THEME = "OCEAN"

        # UI control variables
        self.show_axis = False
        self.block_slice = False
        self.show_sources = False
        self.radius_factor = 0.5
        self.freq_boost = 10.0
        self.amp_boost = 1.0
        self.paused = False
        self.color_palette = 0
        self.var_amp = False

        # Diagnostics & video export toggles
        self.WAVE_DIAGNOSTICS = False
        self.EXPORT_VIDEO = False
        self.VIDEO_FRAMES = 24

    def reset(self):
        """Reset simulation state for a new xperiment."""
        self.elapsed_t = 0.0
        self.last_time = time.time()
        self.frame = 0
        self.peak_amplitude = 0.0

    def apply_xparameters(self, params):
        """Apply parameters from xperiment parameter dictionary."""
        # Meta
        self.X_NAME = params["meta"]["name"]

        # Camera
        self.CAM_INIT = params["camera"]["initial_position"]

        # Universe
        universe = params["universe"]
        self.UNIVERSE_SIZE = list(universe["size"])
        self.TARGET_GRANULES = universe["target_granules"]
        self.TICK_SPACING = universe["tick_spacing"]
        self.COLOR_THEME = universe["color_theme"]

        # Wave sources
        sources = params["wave_sources"]
        self.NUM_SOURCES = sources["count"]
        self.SOURCES_POSITION = sources["positions"]
        self.SOURCES_PHASE_DEG = sources["phase_offsets_deg"]

        # UI defaults
        ui = params["ui_defaults"]
        self.show_axis = ui["show_axis"]
        self.block_slice = ui["block_slice"]
        self.show_sources = ui["show_sources"]
        self.radius_factor = ui["radius_factor"]
        self.freq_boost = ui["freq_boost"]
        self.amp_boost = ui["amp_boost"]
        self.paused = ui["paused"]
        self.color_palette = ui["color_palette"]
        self.var_amp = ui["var_amp"]

        # Diagnostics
        diag = params["diagnostics"]
        self.WAVE_DIAGNOSTICS = diag["wave_diagnostics"]
        self.EXPORT_VIDEO = diag["export_video"]
        self.VIDEO_FRAMES = diag["video_frames"]

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


def xperiment_launcher(xperiment_mgr, state):
    """Display xperiment launcher UI with selectable xperiments.

    Args:
        xperiment_mgr: XperimentManager instance
        state: SimulationState instance (unused but kept for consistency)

    Returns:
        str or None: Selected xperiment name or None
    """
    selected_xperiment = None

    with render.gui.sub_window("XPERIMENT LAUNCHER L0", 0.00, 0.00, 0.13, 0.33) as sub:
        sub.text("(needs window reload)", color=colormap.LIGHT_BLUE[1])
        for xp_name in xperiment_mgr.available_xperiments:
            display_name = xperiment_mgr.get_xperiment_display_name(xp_name)
            is_current = xp_name == xperiment_mgr.current_xperiment

            if sub.checkbox(display_name, is_current) and not is_current:
                selected_xperiment = xp_name

        if sub.button("Close Launcher (esc)"):
            render.window.running = False

    return selected_xperiment


def controls(state):
    """Render the controls UI overlay."""
    # Create overlay windows for controls
    with render.gui.sub_window("CONTROLS", 0.00, 0.34, 0.15, 0.22) as sub:
        state.show_axis = sub.checkbox(f"Axis (ticks: {state.TICK_SPACING})", state.show_axis)
        state.block_slice = sub.checkbox("Block Slice", state.block_slice)
        state.show_sources = sub.checkbox("Show Wave Sources", state.show_sources)
        state.radius_factor = sub.slider_float("Granule", state.radius_factor, 0.1, 2.0)
        state.freq_boost = sub.slider_float("f Boost", state.freq_boost, 0.1, 10.0)
        state.amp_boost = sub.slider_float("Amp Boost", state.amp_boost, 0.1, 5.0)
        if state.paused:
            if sub.button("Continue"):
                state.paused = False
        else:
            if sub.button("Pause"):
                state.paused = True


def color_menu(state):
    """Render color selection menu."""
    tracker = "amplitude" if state.var_amp else "displacement"
    with render.gui.sub_window("COLOR MENU", 0.00, 0.70, 0.14, 0.17) as sub:
        if sub.checkbox("Displacement (ironbow)", state.color_palette == 1 and not state.var_amp):
            state.color_palette = 1
            state.var_amp = False
        if sub.checkbox("Amplitude (ironbow)", state.color_palette == 1 and state.var_amp):
            state.color_palette = 1
            state.var_amp = True
        if sub.checkbox("Amplitude (blueprint)", state.color_palette == 2 and state.var_amp):
            state.color_palette = 2
            state.var_amp = True
        if sub.checkbox("Granule Type Color", state.color_palette == 0):
            state.color_palette = 0
            state.var_amp = True
        if sub.checkbox("Medium Default Color", state.color_palette == 99):
            state.color_palette = 99
            state.var_amp = True
        if state.color_palette == 1:  # Display ironbow gradient palette
            # ironbow: black -> dark blue -> magenta -> red-orange -> yellow-white
            render.canvas.triangles(ib_palette_vertices, per_vertex_color=ib_palette_colors)
            with render.gui.sub_window(tracker, 0.00, 0.64, 0.08, 0.06) as sub:
                sub.text(f"0       {state.peak_amplitude:.0e}m")
        if state.color_palette == 2:  # Display blueprint gradient palette
            # blueprint: dark blue -> medium blue -> blue -> light blue -> extra-light blue
            render.canvas.triangles(bp_palette_vertices, per_vertex_color=bp_palette_colors)
            with render.gui.sub_window(tracker, 0.00, 0.64, 0.08, 0.06) as sub:
                sub.text(f"0       {state.peak_amplitude:.0e}m")


def level_specs(state, level_bar_vertices):
    """Display OpenWave level specifications overlay."""
    render.canvas.triangles(level_bar_vertices, color=colormap.WHITE[1])
    with render.gui.sub_window("LEVEL-0: GRANULE-BASED MEDIUM", 0.82, 0.01, 0.18, 0.10) as sub:
        sub.text(f"Wave Source: {state.NUM_SOURCES} Harmonic Oscillators")
        sub.text("Coupling: Phase Sync")
        sub.text("Propagation: Radial from Source")


def data_dashboard(state):
    """Display simulation data dashboard."""
    with render.gui.sub_window("DATA-DASHBOARD", 0.82, 0.41, 0.18, 0.59) as sub:
        sub.text("--- eWAVE-MEDIUM ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Universe Size: {state.lattice.max_universe_edge:.1e} m (max edge)")
        sub.text(f"Granule Count: {state.lattice.granule_count:,} particles")
        sub.text(f"Medium Density: {constants.MEDIUM_DENSITY:.1e} kg/m³")
        sub.text(f"eWAVE Speed (c): {constants.EWAVE_SPEED:.1e} m/s")

        sub.text("\n--- Scaling-Up (for computation) ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Factor: {state.lattice.scale_factor:.1e} x Planck Scale")
        sub.text(f"Unit-Cells per Max Edge: {state.lattice.max_grid_size:,}")
        sub.text(f"Unit-Cell Edge: {state.lattice.unit_cell_edge:.2e} m")
        sub.text(f"Granule Radius: {state.granule.radius * state.radius_factor:.2e} m")
        sub.text(f"Granule Mass: {state.granule.mass:.2e} kg")

        sub.text("\n--- Sim Resolution (linear) ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"eWave: {state.lattice.ewave_res:.0f} granules/lambda (>10)")
        if state.lattice.ewave_res < 10:
            sub.text(f"*** WARNING: Undersampling! ***", color=(1.0, 0.0, 0.0))
        sub.text(f"Universe: {state.lattice.max_uni_res:.1f} lambda/universe-edge")

        sub.text("\n--- ENERGY-WAVE ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"eWAVE Frequency (f): {constants.EWAVE_FREQUENCY:.1e} Hz")
        sub.text(f"eWAVE Amplitude (A): {constants.EWAVE_AMPLITUDE:.1e} m")
        sub.text(f"eWAVE Wavelength (lambda): {constants.EWAVE_LENGTH:.1e} m")

        sub.text("\n--- Sim Universe Wave Energy ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Energy: {state.lattice.energy:.1e} J ({state.lattice.energy_kWh:.1e} KWh)")

        sub.text("\n--- TIME MICROSCOPE ---", color=colormap.LIGHT_BLUE[1])
        slowed_mo = constants.EWAVE_FREQUENCY / state.freq_boost
        fps = 0 if state.elapsed_t == 0 else state.frame / state.elapsed_t
        sub.text(f"Frames Rendered: {state.frame}")
        sub.text(f"Real Time: {state.elapsed_t / slowed_mo:.2e}s ({fps * slowed_mo:.0e} FPS)")
        sub.text(f"(1 real second = {slowed_mo / (60*60*24*365):.0e}y of sim time)")
        sub.text(f"Sim Time (slow-mo): {state.elapsed_t:.2f}s ({fps:.0f} FPS)")


# ================================================================
# XPERIMENT RENDERING
# ================================================================


def initialize_xperiment(state):
    """Initialize wave sources and diagnostics (called once after lattice init).

    Args:
        state: SimulationState instance with xperiment parameters
    """
    global ib_palette_vertices, ib_palette_colors
    global bp_palette_vertices, bp_palette_colors
    global level_bar_vertices

    # Initialize color palettes for gradient rendering and level indicator (after ti.init)
    ib_palette_vertices, ib_palette_colors = colormap.palette_scale(
        colormap.ironbow, 0.00, 0.63, 0.079, 0.01
    )
    bp_palette_vertices, bp_palette_colors = colormap.palette_scale(
        colormap.blueprint, 0.00, 0.63, 0.079, 0.01
    )
    level_bar_vertices = colormap.level_bar_geometry(0.82, 0.00, 0.179, 0.01)

    # Initialize wave sources
    ewave.build_source_vectors(
        state.SOURCES_POSITION, state.SOURCES_PHASE_DEG, state.NUM_SOURCES, state.lattice
    )

    if state.WAVE_DIAGNOSTICS:
        diagnostics.print_initial_parameters()


def compute_motion(state):
    """Compute lattice motion from wave superposition and update visualization data.

    Args:
        state: SimulationState instance with xperiment parameters
    """
    # Apply wave oscillations from all sources (creates interference patterns)
    ewave.oscillate_granules(
        state.lattice.position_am,
        state.lattice.equilibrium_am,
        state.lattice.amplitude_am,
        state.lattice.velocity_am,
        state.lattice.granule_var_color,
        state.freq_boost,
        state.amp_boost,
        state.color_palette,
        state.var_amp,
        state.NUM_SOURCES,
        state.elapsed_t,
    )

    # Update normalized positions for rendering with optional block-slicing
    state.lattice.normalize_to_screen(1 if state.block_slice else 0)

    # IN-FRAME DATA SAMPLING & DIAGNOSTICS ==================================
    # Update data sampling every 30 frames
    if state.frame % 30 == 0:
        state.peak_amplitude = ewave.peak_amplitude_am[None] * constants.ATTOMETER
        ewave.update_lattice_energy(state.lattice)  # Update energy based on updated wave amplitude

    if state.WAVE_DIAGNOSTICS:
        diagnostics.print_wave_diagnostics(state.elapsed_t, state.frame, print_interval=100)


def render_elements(state):
    """Render granules and wave sources with appropriate coloring."""
    radius_render = state.granule.radius_screen * state.radius_factor

    # Render granules with color scheme
    if state.color_palette == 0:
        render.scene.particles(
            state.lattice.position_screen,
            radius=radius_render,
            per_vertex_color=state.lattice.granule_type_color,
        )
    elif state.color_palette == 1 or state.color_palette == 2:
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
    if state.show_sources:
        render.scene.particles(
            centers=ewave.sources_pos_field,
            radius=state.granule.radius_screen * 2,
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
    ti.init(arch=ti.gpu, log_level=ti.WARN)  # Use GPU if available, suppress info logs

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
        render.init_scene(state.show_axis)  # Initialize scene with lighting and camera

        # Handle ESC key for window close
        if render.window.is_pressed(ti.ui.ESCAPE):
            render.window.running = False
            break

        # Render UI overlays
        new_xperiment = xperiment_launcher(xperiment_mgr, state)
        controls(state)

        # Handle xperiment switching via process replacement
        if new_xperiment:
            print("\n================================================================")
            print("XPERIMENT LAUNCH")
            print(f"Now running: {new_xperiment}\n")

            sys.stdout.flush()
            sys.stderr.flush()
            render.window.running = False

            # os.execv replaces current process (macOS shows harmless warning, Cmd+Q broken)
            os.execv(sys.executable, [sys.executable, __file__, new_xperiment])

        if not state.paused:
            # Update elapsed time and run simulation step
            current_time = time.time()
            state.elapsed_t += current_time - state.last_time  # Elapsed time instead of fixed dt
            state.last_time = current_time

            compute_motion(state)
            state.frame += 1
        else:
            # Prevent time jump on resume
            state.last_time = time.time()

        # Render scene elements
        render_elements(state)

        # Render additional UI elements and scene
        color_menu(state)
        data_dashboard(state)
        level_specs(state, level_bar_vertices)
        render.show_scene()

        # Capture frame for video export (finalizes and stops at set VIDEO_FRAMES)
        if state.EXPORT_VIDEO:
            video.export(state.frame, state.VIDEO_FRAMES)


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    main()
