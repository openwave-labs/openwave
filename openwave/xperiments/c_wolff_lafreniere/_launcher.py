"""
Xperiment Launcher

Unified launcher for wave-field xperiments featuring:
- UI-based xperiment selection and switching
- Single source of truth for rendering and UI code
- Xperiment-specific parameters in /xparameters directory
"""

import webbrowser
import importlib
import sys
import os
import time
from pathlib import Path

import taichi as ti
import numpy as np

from openwave.common import colormap, constants
from openwave.i_o import flux_mesh, render, video

import openwave.xperiments.c_wolff_lafreniere.spacetime_medium as medium
import openwave.xperiments.c_wolff_lafreniere.spacetime_ewave as ewave
import openwave.xperiments.c_wolff_lafreniere.particle as particle
import openwave.xperiments.c_wolff_lafreniere.force_motion as force_motion
import openwave.xperiments.c_wolff_lafreniere.instrumentation as instrument

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
            module_path = f"openwave.xperiments.c_wolff_lafreniere.xparameters.{xperiment_name}"
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
            module_path = f"openwave.xperiments.c_wolff_lafreniere.xparameters.{xperiment_name}"
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
        self.wave_field = None
        self.trackers = None
        self.elapsed_t_rs = 0.0
        self.clock_start_time = time.time()
        self.frame = 1
        self.ampL_global_rms = constants.EWAVE_AMPLITUDE
        self.ampT_global_rms = 0.0
        self.freq_global_avg = constants.EWAVE_FREQUENCY
        self.wavelength_global_avg = constants.EWAVE_LENGTH

        # Current xperiment parameters
        self.X_NAME = ""
        self.CAM_INIT = [2.00, 1.50, 1.75]
        self.UNIVERSE_SIZE = []
        self.TARGET_VOXELS = 1e8
        self.NUM_SOURCES = 1
        self.SOURCES_POSITION = []
        self.SOURCES_OFFSET_DEG = []
        self.APPLY_MOTION = True

        # UI control variables
        self.SHOW_AXIS = False
        self.TICK_SPACING = 0.25
        self.SHOW_GRID = False
        self.SHOW_EDGES = False
        self.FLUX_MESH_PLANES = [0.5, 0.5, 0.5]
        self.SHOW_FLUX_MESH = 0
        self.WARP_MESH = 300
        self.PARTICLE_SHELL = False
        self.TIMESTEP = 0.0
        self.PAUSED = False

        # Color control variables
        self.COLOR_THEME = "OCEAN"
        self.WAVE_MENU = 1

        # Data Analytics & video export toggles
        self.INSTRUMENTATION = False
        self.EXPORT_VIDEO = False
        self.VIDEO_FRAMES = 24

    def apply_xparameters(self, params):
        """Apply parameters from xperiment parameter dictionary."""
        # Meta
        self.X_NAME = params["meta"]["X_NAME"]

        # Camera
        self.CAM_INIT = params["camera"]["INITIAL_POSITION"]

        # Universe
        universe = params["universe"]
        self.UNIVERSE_SIZE = list(universe["SIZE"])
        self.TARGET_VOXELS = universe["TARGET_VOXELS"]

        # Wave Centers
        sources = params["wave_centers"]
        self.NUM_SOURCES = sources["COUNT"]
        self.SOURCES_POSITION = sources["POSITION"]
        self.SOURCES_OFFSET_DEG = sources["PHASE_OFFSETS_DEG"]
        self.APPLY_MOTION = sources["APPLY_MOTION"]

        # UI defaults
        ui = params["ui_defaults"]
        self.SHOW_AXIS = ui["SHOW_AXIS"]
        self.TICK_SPACING = ui["TICK_SPACING"]
        self.SHOW_GRID = ui["SHOW_GRID"]
        self.SHOW_EDGES = ui["SHOW_EDGES"]
        self.FLUX_MESH_PLANES = ui["FLUX_MESH_PLANES"]
        self.SHOW_FLUX_MESH = ui["SHOW_FLUX_MESH"]
        self.WARP_MESH = ui["WARP_MESH"]
        self.PARTICLE_SHELL = ui["PARTICLE_SHELL"]
        self.TIMESTEP = ui["TIMESTEP"]
        self.PAUSED = ui["PAUSED"]

        # Color defaults
        color = params["color_defaults"]
        self.COLOR_THEME = color["COLOR_THEME"]
        self.WAVE_MENU = color["WAVE_MENU"]

        # Data Analytics & video export toggles
        diag = params["analytics"]
        self.INSTRUMENTATION = diag["INSTRUMENTATION"]
        self.EXPORT_VIDEO = diag["EXPORT_VIDEO"]
        self.VIDEO_FRAMES = diag["VIDEO_FRAMES"]

    def initialize_grid(self):
        """Initialize or reinitialize the wave-field grid and wave-centers."""
        self.wave_field = medium.WaveField(
            self.UNIVERSE_SIZE, self.TARGET_VOXELS, self.FLUX_MESH_PLANES
        )
        self.trackers = medium.Trackers(self.wave_field.grid_size, self.wave_field.scale_factor)

        # Initialize wave-centers
        self.wave_center = particle.WaveCenter(
            self.wave_field.grid_size,
            self.NUM_SOURCES,
            self.SOURCES_POSITION,
            self.SOURCES_OFFSET_DEG,
        )

    def reset_sim(self):
        """Reset simulation state."""
        self.wave_field = None
        self.trackers = None
        self.elapsed_t_rs = 0.0
        self.clock_start_time = time.time()
        self.frame = 1
        self.ampL_global_rms = constants.EWAVE_AMPLITUDE
        self.ampT_global_rms = 0.0
        self.freq_global_avg = constants.EWAVE_FREQUENCY
        self.wavelength_global_avg = constants.EWAVE_LENGTH
        self.initialize_grid()
        initialize_xperiment(self)


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

    with render.gui.sub_window("XPERIMENT LAUNCHER", 0.00, 0.00, 0.14, 0.32) as sub:
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
    with render.gui.sub_window("CONTROLS", 0.00, 0.33, 0.16, 0.27) as sub:
        state.SHOW_AXIS = sub.checkbox(f"Axis (ticks: {state.TICK_SPACING})", state.SHOW_AXIS)
        state.SHOW_EDGES = sub.checkbox("Sim Universe Edges", state.SHOW_EDGES)
        state.SHOW_FLUX_MESH = sub.slider_int("Flux Mesh", state.SHOW_FLUX_MESH, 0, 3)
        state.WARP_MESH = sub.slider_int("Warp Mesh", state.WARP_MESH, 0, 500)
        state.PARTICLE_SHELL = sub.checkbox("Particle Shell", state.PARTICLE_SHELL)
        state.TIMESTEP = sub.slider_float("Timestep", state.TIMESTEP, 0.1, 30.0)
        state.APPLY_MOTION = sub.checkbox("Apply Motion", state.APPLY_MOTION)
        if state.PAUSED:
            if sub.button(">> PROPAGATE EWAVE >>"):
                state.PAUSED = False
        else:
            if sub.button("Pause"):
                state.PAUSED = True
        if sub.button("Reset Simulation"):
            state.reset_sim()


def display_wave_menu(state):
    """Display wave properties selection menu."""
    with render.gui.sub_window("WAVE MENU", 0.00, 0.70, 0.15, 0.17) as sub:
        if sub.checkbox("Displacement (Longitudinal)", state.WAVE_MENU == 1):
            state.WAVE_MENU = 1
        if sub.checkbox("Amplitude (Longitudinal)", state.WAVE_MENU == 3):
            state.WAVE_MENU = 3
        if sub.checkbox("Envelope (Longitudinal)", state.WAVE_MENU == 4):
            state.WAVE_MENU = 4
        # Display gradient palette with 2× average range for headroom (allows peak visualization)
        if state.WAVE_MENU == 1:  # Displacement (Longitudinal) on greenyellow gradient
            render.canvas.triangles(gy_palette_vertices, per_vertex_color=gy_palette_colors)
            with render.gui.sub_window("displacement", 0.00, 0.64, 0.08, 0.06) as sub:
                sub.text(
                    f"{-state.ampL_global_rms*2/state.wave_field.scale_factor:.0e}  {state.ampL_global_rms*2/state.wave_field.scale_factor:.0e}m"
                )
        if state.WAVE_MENU == 3:  # Amplitude (Longitudinal) on viridis gradient
            render.canvas.triangles(vr_palette_vertices, per_vertex_color=vr_palette_colors)
            with render.gui.sub_window("amplitude", 0.00, 0.64, 0.08, 0.06) as sub:
                sub.text(f"0       {state.ampL_global_rms*2/state.wave_field.scale_factor:.0e}m")
        if state.WAVE_MENU == 4:  # Envelope (Longitudinal) on greenyellow gradient
            render.canvas.triangles(gy_palette_vertices, per_vertex_color=gy_palette_colors)
            with render.gui.sub_window("envelope", 0.00, 0.64, 0.08, 0.06) as sub:
                sub.text(
                    f"{-state.ampL_global_rms*2/state.wave_field.scale_factor:.0e}  {state.ampL_global_rms*2/state.wave_field.scale_factor:.0e}m"
                )


def display_level_specs(state, level_bar_vertices):
    """Display OpenWave level specifications overlay."""
    render.canvas.triangles(level_bar_vertices, color=colormap.DARK_BLUE[1])
    with render.gui.sub_window("WOLFF-LAFRENIERE METHOD", 0.84, 0.01, 0.16, 0.16) as sub:
        sub.text("Medium: Indexed Voxel Grid")
        sub.text("Data-Structure: Scalar Field")
        sub.text("Coupling: Phase Sync")
        sub.text("Propagation: Analytical Function")
        sub.text("Boundary: Open (Non-Reflective)")
        if sub.button("Wave Notation Guide"):
            webbrowser.open(
                "https://github.com/openwave-labs/openwave/blob/main/openwave/common/wave_notation.md"
            )


def display_data_dashboard(state):
    """Display simulation data dashboard."""
    clock_time = time.time() - state.clock_start_time
    sim_time_years = clock_time / (state.elapsed_t_rs * constants.RONTOSECOND or 1) / 31_536_000

    with render.gui.sub_window("DATA-DASHBOARD", 0.84, 0.45, 0.16, 0.55) as sub:
        state.INSTRUMENTATION = sub.checkbox("Instrumentation ON/OFF", state.INSTRUMENTATION)
        sub.text("--- SPACETIME ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Medium Density: {constants.MEDIUM_DENSITY:.1e} kg/m³")
        sub.text(f"eWAVE Speed (c): {constants.EWAVE_SPEED:.1e} m/s")

        sub.text("\n--- SIMULATION DOMAIN ---", color=colormap.LIGHT_BLUE[1])
        sub.text(
            f"Universe: {state.wave_field.max_universe_edge:.1e} m ({state.wave_field.max_universe_edge_lambda:.0f} waves)"
        )
        sub.text(f"Voxel Count: {state.wave_field.voxel_count:,}")
        sub.text(
            f"Grid Size: {state.wave_field.nx} x {state.wave_field.ny} x {state.wave_field.nz}"
        )
        sub.text(f"Voxel Edge: {state.wave_field.dx:.2e} m")

        sub.text("\n--- RESOLUTION (scaled-up) ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Scale-up Factor: {state.wave_field.scale_factor:.1f}x")
        sub.text(f"eWave: {state.wave_field.ewave_res:.1f} voxels/wave (~12)")
        if state.wave_field.ewave_res < 10:
            sub.text(f"*** WARNING: Undersampling! ***", color=(1.0, 0.0, 0.0))

        sub.text("\n--- ENERGY-WAVE ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Amp Longitudinal: {state.ampL_global_rms/state.wave_field.scale_factor:.1e} m")
        sub.text(f"Amp Transverse: {state.ampT_global_rms/state.wave_field.scale_factor:.1e} m")
        sub.text(f"Frequency: {state.freq_global_avg*state.wave_field.scale_factor:.1e} Hz")
        sub.text(f"Wavelength: {state.wavelength_global_avg/state.wave_field.scale_factor:.1e} m")

        sub.text("\n--- TIME MICROSCOPE ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Timestep: {state.TIMESTEP:.2f} rs")
        sub.text(f"Sim Steps (frames): {state.frame:,}")
        sub.text(f"Sim Time: {state.elapsed_t_rs:,.0f} rs")
        sub.text(f"Clock Time: {clock_time:.2f} s")
        sub.text(f"(1s sim time takes {sim_time_years:.0e}y)")


# ================================================================
# XPERIMENT RENDERING
# ================================================================


def initialize_xperiment(state):
    """Initialize color palettes, wave charger and instrumentation.

    Args:
        state: SimulationState instance with xperiment parameters
    """
    global gy_palette_vertices, gy_palette_colors
    global vr_palette_vertices, vr_palette_colors
    global level_bar_vertices

    # Initialize color palette scales for gradient rendering and level indicator
    gy_palette_vertices, gy_palette_colors = colormap.get_palette_scale(
        colormap.greenyellow, 0.00, 0.63, 0.079, 0.01
    )
    vr_palette_vertices, vr_palette_colors = colormap.get_palette_scale(
        colormap.viridis, 0.00, 0.63, 0.079, 0.01
    )
    level_bar_vertices = colormap.get_level_bar_geometry(0.84, 0.00, 0.159, 0.01)

    if state.INSTRUMENTATION:
        print("\n" + "=" * 64)
        print("INSTRUMENTATION ENABLED")
        print("=" * 64)


def compute_wave_motion(state):
    """Compute wave propagation, reflection, superposition and update tracker averages."""

    ewave.propagate_wave(
        state.wave_field,
        state.trackers,
        state.wave_center,
        state.TIMESTEP,
        state.elapsed_t_rs,
    )

    # IN-FRAME DATA SAMPLING & ANALYTICS ==================================
    # Frame skip reduces GPU->CPU transfer overhead
    if state.frame % 60 == 0 or state.frame == 10:
        ewave.sample_avg_trackers(state.wave_field, state.trackers)
    state.ampL_global_rms = state.trackers.ampL_global_rms_am[None] * constants.ATTOMETER  # in m
    state.ampT_global_rms = state.trackers.ampT_global_rms_am[None] * constants.ATTOMETER  # in m
    state.freq_global_avg = state.trackers.freq_global_avg_rHz[None] / constants.RONTOSECOND
    state.wavelength_global_avg = constants.EWAVE_SPEED / (
        state.freq_global_avg or 1
    )  # prevents 0 div

    if state.INSTRUMENTATION:
        instrument.log_timestep_data(state.frame, state.wave_field, state.trackers)
        if state.frame == 500:
            instrument.plot_probe_wave_profile(state.wave_field)


def compute_force_motion(state):
    """
    Compute forces and update particle motion.

    Physics:
    - Force = -grad(E) where E = rho * V * (f * A)^2
    - Motion: Euler integration of F = m * a

    Phases:
    - Phase 1 (SMOKE_TEST=True): Hardcoded force for testing motion integration
    - Phase 3+ (SMOKE_TEST=False): Force computed from energy gradient

    See research/02_force_motion.md for detailed documentation.
    """

    # Compute force from energy gradient, then integrate motion
    force_motion.compute_force_vector(
        state.wave_field,
        state.trackers,
        state.wave_center,
    )
    if state.APPLY_MOTION:
        force_motion.integrate_motion_euler(
            state.wave_field,
            state.wave_center,
            state.TIMESTEP,
        )
    else:
        # Zero-out velocities if not integrating force to motion
        for wc_idx in range(state.wave_center.num_sources):
            state.wave_center.velocity_amrs[wc_idx] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)

    # Annihilation naturally occurs from wave physics, but needs numerical precision check
    # Detect and handle particle annihilation (opposite phase WCs meeting)
    # Threshold: WCs can be at grid diagonal positions and TIMESTEP may cause larger jumps
    force_motion.detect_annihilation(state.wave_center, 5.0)


def render_elements(state):
    """Render grid, edges, flux mesh and test particles."""
    if state.SHOW_GRID:
        render.scene.lines(state.wave_field.grid_lines, width=1, color=colormap.COLOR_MEDIUM[1])

    if state.SHOW_EDGES:
        render.scene.lines(state.wave_field.edge_lines, width=1, color=colormap.COLOR_MEDIUM[1])

    if state.SHOW_FLUX_MESH > 0:
        ewave.update_flux_mesh_values(
            state.wave_field,
            state.trackers,
            state.WAVE_MENU,
            state.WARP_MESH,
        )
        flux_mesh.render_flux_mesh(render.scene, state.wave_field, state.SHOW_FLUX_MESH)

    if state.PARTICLE_SHELL:
        # Convert wave-centers positions from [ijk] to [screen_normalization]
        # Use position_float for smooth rendering (position_grid is integer, causes jumpy motion)
        # Normalize by max_grid_size to respect asymmetric universes (like flux_mesh does)
        max_dim = float(state.wave_field.max_grid_size)
        for wc_idx in range(state.wave_center.num_sources):
            # Skip inactive (annihilated) WCs
            if state.wave_center.active[wc_idx] == 0:
                continue

            wc_pos_screen = ti.Vector(
                [
                    state.wave_center.position_float[wc_idx][0] / max_dim,
                    state.wave_center.position_float[wc_idx][1] / max_dim,
                    state.wave_center.position_float[wc_idx][2] / max_dim,
                ],
                dt=ti.f32,
            )
            position = np.array(
                [[wc_pos_screen[0], wc_pos_screen[1], wc_pos_screen[2]]], dtype=np.float32
            )
            radius = (
                constants.EWAVE_LENGTH
                / state.wave_field.max_universe_edge
                * state.wave_field.scale_factor
                * 0.75  # adjusted for taichi particle rendering perspective projection
            )
            color = (
                colormap.COLOR_PARTICLE[1]
                if state.SOURCES_OFFSET_DEG[wc_idx] == 180
                else colormap.COLOR_ANTI[1]
            )
            # Render particle shell at wave-center position
            render.scene.particles(position, radius, color=color)


# ================================================================
# MAIN LOOP
# ================================================================


def main():
    """Main entry point for xperiment launcher."""
    selected_xperiment_arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Initialize Taichi
    ti.init(arch=ti.gpu, log_level=ti.WARN)  # GPU preferred, suppress info logs

    # Initialize xperiment manager and state
    xperiment_mgr = XperimentManager()
    state = SimulationState()

    # Load xperiment from CLI argument or default
    default_xperiment = selected_xperiment_arg or "electric_attraction"
    if default_xperiment not in xperiment_mgr.available_xperiments:
        print(f"Error: Xperiment '{default_xperiment}' not found!")
        return

    params = xperiment_mgr.load_xperiment(default_xperiment)
    if not params:
        return

    state.apply_xparameters(params)
    state.initialize_grid()
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
            # Run simulation step and update time
            compute_wave_motion(state)
            compute_force_motion(state)
            state.elapsed_t_rs += state.TIMESTEP  # Accumulate simulation time
            state.frame += 1

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

    if state.INSTRUMENTATION:
        instrument.generate_plots()


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    main()
