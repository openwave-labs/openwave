"""
L1 XPERIMENT LAUNCHER

Unified launcher for Level-1 wave-field xperiments featuring:
- UI-based xperiment selection and switching
- Single source of truth for rendering and UI code
- Xperiment-specific parameters in /_xparameters directory
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
from openwave._io import flux_mesh, render, video

import openwave.spacetime.L1_field_grid as data_grid
import openwave.spacetime.L1_wave_engine as ewave
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
            module_path = f"openwave.xperiments.5_level1_field_based._xparameters.{xperiment_name}"
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
            module_path = f"openwave.xperiments.5_level1_field_based._xparameters.{xperiment_name}"
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
        self.c_amrs = 0.0
        self.dt_rs = 0.0
        self.cfl_factor = 0.0
        self.elapsed_t_rs = 0.0
        self.clock_start_time = time.time()
        self.frame = 1
        self.avg_amplitude = constants.EWAVE_AMPLITUDE
        self.avg_frequency = constants.EWAVE_FREQUENCY
        self.avg_wavelength = constants.EWAVE_LENGTH
        self.avg_energy = 0.0
        self.charge_level = 0.0
        self.charging = True
        self.damping = False

        # Current xperiment parameters
        self.X_NAME = ""
        self.CAM_INIT = [2.00, 1.50, 1.75]
        self.UNIVERSE_SIZE = []
        self.TARGET_VOXELS = 1e8

        # UI control variables
        self.SHOW_AXIS = False
        self.TICK_SPACING = 0.25
        self.SHOW_GRID = False
        self.FLUX_MESH_OPTION = 0
        self.SIM_SPEED = 1.0
        self.PAUSED = False

        # Color control variables
        self.COLOR_THEME = "OCEAN"
        self.COLOR_PALETTE = 1  # Color palette index

        # Diagnostics & video export toggles
        self.WAVE_DIAGNOSTICS = False
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

        # UI defaults
        ui = params["ui_defaults"]
        self.SHOW_AXIS = ui["SHOW_AXIS"]
        self.TICK_SPACING = ui["TICK_SPACING"]
        self.SHOW_GRID = ui["SHOW_GRID"]
        self.FLUX_MESH_OPTION = ui["FLUX_MESH_OPTION"]
        self.SIM_SPEED = ui["SIM_SPEED"]
        self.PAUSED = ui["PAUSED"]

        # Color defaults
        color = params["color_defaults"]
        self.COLOR_THEME = color["COLOR_THEME"]
        self.COLOR_PALETTE = color["COLOR_PALETTE"]

        # Diagnostics
        diag = params["diagnostics"]
        self.WAVE_DIAGNOSTICS = diag["WAVE_DIAGNOSTICS"]
        self.EXPORT_VIDEO = diag["EXPORT_VIDEO"]
        self.VIDEO_FRAMES = diag["VIDEO_FRAMES"]

    def initialize_grid(self):
        """Initialize or reinitialize the wave field grid."""
        self.wave_field = data_grid.WaveField(self.UNIVERSE_SIZE, self.TARGET_VOXELS)
        self.trackers = data_grid.Trackers(self.wave_field.grid_size)

    def compute_timestep(self):
        """Compute timestep from CFL stability condition.

        CFL Condition: dt ≤ dx / (c × √3) for 3D wave equation.
        SIM_SPEED scales wave velocity for visualization control.
        """
        self.c_amrs = (
            constants.EWAVE_SPEED / constants.ATTOMETER * constants.RONTOSECOND * self.SIM_SPEED
        )  # am/rs
        self.dt_rs = self.wave_field.dx_am / (self.c_amrs / self.SIM_SPEED * (3**0.5))  # rs
        self.cfl_factor = round((self.c_amrs * self.dt_rs / self.wave_field.dx_am) ** 2, 7)

    def restart_sim(self):
        """Restart simulation state."""
        self.wave_field = None
        self.trackers = None
        self.c_amrs = 0.0
        self.dt_rs = 0.0
        self.cfl_factor = 0.0
        self.elapsed_t_rs = 0.0
        self.clock_start_time = time.time()
        self.frame = 1
        self.avg_amplitude = constants.EWAVE_AMPLITUDE
        self.avg_frequency = constants.EWAVE_FREQUENCY
        self.avg_wavelength = constants.EWAVE_LENGTH
        self.avg_energy = 0.0
        self.charge_level = 0.0
        self.charging = True
        self.damping = False
        self.initialize_grid()
        self.compute_timestep()
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

    with render.gui.sub_window("XPERIMENT LAUNCHER (L1)", 0.00, 0.00, 0.13, 0.33) as sub:
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
    with render.gui.sub_window("CONTROLS", 0.00, 0.34, 0.16, 0.17) as sub:
        state.SHOW_AXIS = sub.checkbox(f"Axis (ticks: {state.TICK_SPACING})", state.SHOW_AXIS)
        state.FLUX_MESH_OPTION = sub.slider_int("Flux Mesh", state.FLUX_MESH_OPTION, 0, 3)
        state.SIM_SPEED = sub.slider_float("Speed", state.SIM_SPEED, 0.5, 1.0)
        if state.PAUSED:
            if sub.button("Propagate eWave"):
                state.PAUSED = False
        else:
            if sub.button("Pause"):
                state.PAUSED = True
        if sub.button("Restart Sim"):
            state.restart_sim()


def display_color_menu(state):
    """Display color selection menu."""
    with render.gui.sub_window("COLOR MENU", 0.00, 0.75, 0.14, 0.12) as sub:
        if sub.checkbox("Displacement (redshift)", state.COLOR_PALETTE == 1):
            state.COLOR_PALETTE = 1
        if sub.checkbox("Amplitude (ironbow)", state.COLOR_PALETTE == 2):
            state.COLOR_PALETTE = 2
        if sub.checkbox("Frequency (blueprint)", state.COLOR_PALETTE == 3):
            state.COLOR_PALETTE = 3
        # Display gradient palette with 2× average range for headroom (allows peak visualization)
        if state.COLOR_PALETTE == 1:  # Display redshift gradient palette
            render.canvas.triangles(rs_palette_vertices, per_vertex_color=rs_palette_colors)
            with render.gui.sub_window("displacement", 0.00, 0.69, 0.08, 0.06) as sub:
                sub.text(f"{-state.avg_amplitude * 2:.0e}  {state.avg_amplitude * 2:.0e}m")
        if state.COLOR_PALETTE == 2:  # Display ironbow gradient palette
            render.canvas.triangles(ib_palette_vertices, per_vertex_color=ib_palette_colors)
            with render.gui.sub_window("amplitude", 0.00, 0.69, 0.08, 0.06) as sub:
                sub.text(f"0       {state.avg_amplitude * 2:.0e}m")
        if state.COLOR_PALETTE == 3:  # Display blueprint gradient palette
            render.canvas.triangles(bp_palette_vertices, per_vertex_color=bp_palette_colors)
            with render.gui.sub_window("frequency", 0.00, 0.69, 0.08, 0.06) as sub:
                sub.text(f"0       {state.avg_frequency * 2:.0e}Hz")


def display_level_specs(state, level_bar_vertices):
    """Display OpenWave level specifications overlay."""
    render.canvas.triangles(level_bar_vertices, color=colormap.LIGHT_BLUE[1])
    with render.gui.sub_window("LEVEL-1: FIELD-BASED METHOD", 0.82, 0.01, 0.18, 0.10) as sub:
        sub.text("Coupling: Phase Sync")
        sub.text("Propagation: Radial from Source")
        if sub.button("Wave Notation Guide"):
            webbrowser.open(
                "https://github.com/openwave-labs/openwave/blob/main/openwave/wave_notation.md"
            )


def display_data_dashboard(state):
    """Display simulation data dashboard."""
    clock_time = time.time() - state.clock_start_time
    sim_time_years = clock_time / (state.elapsed_t_rs * constants.RONTOSECOND or 1) / 31_536_000

    with render.gui.sub_window("DATA-DASHBOARD", 0.82, 0.39, 0.18, 0.61) as sub:
        sub.text("--- SPACETIME ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Universe Size: {state.wave_field.max_universe_edge:.1e} m (max edge)")
        sub.text(f"Medium Density: {constants.MEDIUM_DENSITY:.1e} kg/m³")
        sub.text(f"eWAVE Speed (c): {constants.EWAVE_SPEED:.1e} m/s")

        sub.text("\n--- eWAVE-FIELD ---", color=colormap.LIGHT_BLUE[1])
        sub.text(
            f"Grid Size: {state.wave_field.nx} x {state.wave_field.ny} x {state.wave_field.nz}"
        )
        sub.text(f"Voxel Count: {state.wave_field.voxel_count:,}")
        sub.text(f"Voxel Edge: {state.wave_field.dx:.2e} m")

        sub.text("\n--- Sim Resolution (linear) ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"eWave: {state.wave_field.ewave_res:.1f} voxels/lambda (>10)")
        if state.wave_field.ewave_res < 10:
            sub.text(f"*** WARNING: Undersampling! ***", color=(1.0, 0.0, 0.0))
        sub.text(f"Universe: {state.wave_field.max_uni_res:.1f} lambda/edge")

        sub.text("\n--- eWAVE-SAMPLING ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Avg Amplitude (A): {state.avg_amplitude:.1e} m")
        sub.text(f"Avg Frequency (f): {state.avg_frequency:.1e} Hz")
        sub.text(f"Avg Wavelength (lambda): {state.avg_wavelength:.1e} m")
        sub.text(f"Avg Energy: {state.avg_energy:.1e} J")
        sub.text(
            f"Charge Level: {state.charge_level:.0%} {"...CHARGING..." if state.charging else "...DAMPING..." if state.damping else "(target)"}",
            color=(
                colormap.ORANGE[1]
                if state.charging
                else colormap.RED[1] if state.damping else colormap.GREEN[1]
            ),
        )

        sub.text("\n--- TIME MICROSCOPE ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Frames Rendered: {state.frame}")
        sub.text(f"Simulation Time: {state.elapsed_t_rs:.2e} rs")
        sub.text(f"Clock Time: {clock_time:.2f} s")
        sub.text(f"(1s sim time takes {sim_time_years:.0e}y)")

        sub.text("\n--- TIMESTEP ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"c_amrs: {state.c_amrs:.3f} am/rs")
        sub.text(
            f"dt_rs: {state.dt_rs:.3f} rs",
            color=(1.0, 1.0, 1.0) if state.cfl_factor <= (1 / 3) else (1.0, 0.0, 0.0),
        )
        sub.text(
            f"CFL Factor: {state.cfl_factor:.3f} (target < 1/3)",
            color=((1.0, 1.0, 1.0) if state.cfl_factor <= (1 / 3) else (1.0, 0.0, 0.0)),
        )


# ================================================================
# XPERIMENT RENDERING
# ================================================================


def initialize_xperiment(state):
    """Initialize color palettes, wave charger and diagnostics.

    Args:
        state: SimulationState instance with xperiment parameters
    """
    global ib_palette_vertices, ib_palette_colors
    global rs_palette_vertices, rs_palette_colors
    global bp_palette_vertices, bp_palette_colors
    global level_bar_vertices

    # Initialize color palette scales for gradient rendering and level indicator
    ib_palette_vertices, ib_palette_colors = colormap.get_palette_scale(
        colormap.ironbow, 0.00, 0.68, 0.079, 0.01
    )
    bp_palette_vertices, bp_palette_colors = colormap.get_palette_scale(
        colormap.blueprint, 0.00, 0.68, 0.079, 0.01
    )
    rs_palette_vertices, rs_palette_colors = colormap.get_palette_scale(
        colormap.redshift, 0.00, 0.68, 0.079, 0.01
    )
    level_bar_vertices = colormap.get_level_bar_geometry(0.82, 0.00, 0.179, 0.01)

    # STATIC CHARGER methods (one-time initialization pattern)
    # Uncomment to test different initial wave configurations
    # ewave.charge_full(state.wave_field, state.dt_rs)
    # ewave.charge_gaussian(state.wave_field)
    # NO: ewave.charge_falloff(state.wave_field, state.dt_rs)
    # NO: ewave.charge_1lambda(state.wave_field, state.dt_rs)
    # TODO: code toggle to plot initial charging pattern
    # ewave.plot_charge_profile(state.wave_field)

    if state.WAVE_DIAGNOSTICS:
        diagnostics.print_initial_parameters()


def compute_wave_motion(state):
    """Compute wave propagation, reflection, superposition and update tracker averages.

    Args:
        state: SimulationState instance with xperiment parameters
    """
    # DYNAMIC CHARGER methods (oscillating source pattern during simulation)
    # Charger runs BEFORE propagation to inject energy into displacement_am
    if state.charging:
        ewave.charge_oscillator_sphere(state.wave_field, state.elapsed_t_rs)  # energy injection

    ewave.propagate_ewave(
        state.wave_field,
        state.trackers,
        state.c_amrs,
        state.dt_rs,
        state.elapsed_t_rs,
    )

    # DYNAMIC DAMPING runs AFTER propagation to damp displacement values
    if state.damping:
        ewave.damp_energy_full(state.wave_field, 0.99)  # energy absorption

    # IN-FRAME DATA SAMPLING & DIAGNOSTICS ==================================
    # Frame skip reduces GPU->CPU transfer overhead
    if state.frame % 60 == 0:
        ewave.sample_avg_trackers(state.wave_field, state.trackers)
        state.avg_amplitude = state.trackers.avg_amplitudeL_am[None] * constants.ATTOMETER  # in m
        state.avg_frequency = state.trackers.avg_frequency_rHz[None] / constants.RONTOSECOND
        state.avg_wavelength = constants.EWAVE_SPEED / (state.avg_frequency or 1)  # prevents 0 div
        state.avg_energy = (
            constants.MEDIUM_DENSITY
            * state.wave_field.universe_volume
            * (state.avg_frequency * state.avg_amplitude) ** 2
        )
        state.charge_level = state.avg_energy / state.wave_field.energy
        state.charging = state.charge_level < 0.9  # stop charging, seeks energy stabilization
        state.damping = state.charge_level > 1.1  # start damping, seeks energy stabilization


def render_elements(state):
    """Render grid, flux mesh and test particles."""
    if state.SHOW_GRID:
        render.scene.lines(state.wave_field.grid_lines, width=1, color=colormap.COLOR_MEDIUM[1])

    if state.FLUX_MESH_OPTION > 0:
        ewave.update_flux_mesh_colors(state.wave_field, state.trackers, state.COLOR_PALETTE)
        flux_mesh.render_flux_mesh(render.scene, state.wave_field, state.FLUX_MESH_OPTION)

    # TODO: remove test particles for visual reference
    # position1 = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    # render.scene.particles(position1, radius=0.01, color=colormap.COLOR_PARTICLE[1])
    # position2 = np.array([[0.5, 0.7, 0.5]], dtype=np.float32)
    # render.scene.particles(position2, radius=0.01, color=colormap.COLOR_ANTI[1])


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
    default_xperiment = selected_xperiment_arg or "energy_wave"
    if default_xperiment not in xperiment_mgr.available_xperiments:
        print(f"Error: Xperiment '{default_xperiment}' not found!")
        return

    params = xperiment_mgr.load_xperiment(default_xperiment)
    if not params:
        return

    state.apply_xparameters(params)
    state.initialize_grid()
    state.compute_timestep()
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
            state.compute_timestep()
            compute_wave_motion(state)
            state.elapsed_t_rs += state.dt_rs  # Accumulate simulation time
            state.frame += 1

        # Render scene elements
        render_elements(state)

        # Display additional UI elements and scene
        display_color_menu(state)
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
