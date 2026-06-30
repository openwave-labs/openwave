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

import openwave.xperiments.m4_ewt.medium as medium
import openwave.xperiments.m4_ewt.particle as particle
import openwave.xperiments.m4_ewt.wave_engine as ewave
import openwave.xperiments.m4_ewt.force_motion as force_motion
import openwave.xperiments.m4_ewt.instrumentation as instrument

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
            module_path = f"openwave.xperiments.m4_ewt.xparameters.{xperiment_name}"
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
            module_path = f"openwave.xperiments.m4_ewt.xparameters.{xperiment_name}"
            parameters_module = importlib.import_module(module_path)
            display_name = parameters_module.XPARAMETERS["meta"]["X_NAME"]
            self.xperiment_display_names[xperiment_name] = display_name
            return display_name
        except:
            # Final fallback: convert filename
            return " ".join(word.capitalize() for word in xperiment_name.split("_"))


# ================================================================
# ENGINE DEFAULTS (fallback when an xperiment omits the "engine" section)
# ================================================================
# The seed / potential / WC-interaction knobs are per-xperiment xparameters (the
# "engine" section of each xparameter file). These defaults apply only when a file
# does not specify a given key.
ENGINE_DEFAULTS = {
    # Base wave seed (P1)
    "SEED_MODE": 2,  # 0 = gaussian pulse, 1 = radial cosine, 2 = full (domain-filling base wave)
    "SEED_BOOST": 1.0,  # seed amplitude multiplier
    # Non-linear potential V(ψ) (P2)
    "V_MODE": 0,  # 0 = linear/off, 1 = cubic ψ³, 2 = saturating, 3 = double-well
    "V_C1": 0.0,  # primary coefficient (k for modes 1/2, a for mode 3); c1 < 0 = focusing
    "V_C2": 0.0,  # secondary coefficient (q for mode 2, b for mode 3)
    # Wave-center interaction (P3)
    "WC_INTERACT_MODE": 0,  # 0 = free, 1 = dirichlet, 2 = neumann, 3 = soft
    "WC_BOOST": 1.0,  # WC drive amplitude multiplier
    "WC_RADIUS": 2,  # WC drive ball radius (voxels)
    "WC_SIGMA": 1.5,  # soft-mode Gaussian width (voxels)
}


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
        self.amp_global_rms = constants.EWAVE_AMPLITUDE
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
        self.INIT_VELOCITY = None
        self.APPLY_MOTION = True

        # UI control variables
        self.SHOW_AXIS = False
        self.TICK_SPACING = 0.25
        self.SHOW_GRID = False
        self.SHOW_EDGES = False
        self.FLUX_MESH_PLANES = [0.5, 0.5, 0.5]
        self.SHOW_FLUX_MESH = 0
        self.WARP_MESH = 300
        self.SHOW_GRANULES = False
        self.PARTICLE_SHELL = False
        # PDE solver: CFL-derived timestep + wave speed (set in initialize_grid).
        # SIM_SPEED scales the rendered wave speed (c_amrs); dt stays at the CFL bound.
        self.SIM_SPEED = 1.0
        self.c_amrs = 0.0
        self.dt_rs = 0.0
        self.cfl_factor = 0.0
        self.PAUSED = False

        # Color control variables
        self.COLOR_THEME = "OCEAN"
        self.WAVE_MENU = 1

        # Data Analytics & video export toggles
        self.INSTRUMENTATION = False
        self.EXPORT_VIDEO = False
        self.VIDEO_FRAMES = 24

        # Engine config (from the xperiment "engine" section; see ENGINE_DEFAULTS)
        self.SEED_MODE = ENGINE_DEFAULTS["SEED_MODE"]
        self.SEED_BOOST = ENGINE_DEFAULTS["SEED_BOOST"]
        self.V_MODE = ENGINE_DEFAULTS["V_MODE"]
        self.V_C1 = ENGINE_DEFAULTS["V_C1"]
        self.V_C2 = ENGINE_DEFAULTS["V_C2"]
        self.WC_INTERACT_MODE = ENGINE_DEFAULTS["WC_INTERACT_MODE"]
        self.WC_BOOST = ENGINE_DEFAULTS["WC_BOOST"]
        self.WC_RADIUS = ENGINE_DEFAULTS["WC_RADIUS"]
        self.WC_SIGMA = ENGINE_DEFAULTS["WC_SIGMA"]

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
        self.INIT_VELOCITY = sources.get("INIT_VELOCITY", None)
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
        self.SHOW_GRANULES = ui.get("SHOW_GRANULES", False)
        self.PARTICLE_SHELL = ui["PARTICLE_SHELL"]
        self.SIM_SPEED = ui.get("SIM_SPEED", 1.0)  # PDE wave-speed scale (TIMESTEP retired)
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

        # Engine config (seed / potential / WC interaction); falls back to ENGINE_DEFAULTS
        engine = {**ENGINE_DEFAULTS, **params.get("engine", {})}
        self.SEED_MODE = engine["SEED_MODE"]
        self.SEED_BOOST = engine["SEED_BOOST"]
        self.V_MODE = engine["V_MODE"]
        self.V_C1 = engine["V_C1"]
        self.V_C2 = engine["V_C2"]
        self.WC_INTERACT_MODE = engine["WC_INTERACT_MODE"]
        self.WC_BOOST = engine["WC_BOOST"]
        self.WC_RADIUS = engine["WC_RADIUS"]
        self.WC_SIGMA = engine["WC_SIGMA"]

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
            self.INIT_VELOCITY,
        )

        # Derive the CFL-safe PDE timestep (needs dx_am), then seed the base wave once.
        # The base wave is the medium's always-on ground-state vibration (EWT); it is
        # sourced from the domain center, NOT from the wave centers (see seed_wave).
        self._compute_timestep()
        ewave.seed_wave(self.wave_field, self.SEED_MODE, self.SEED_BOOST, self.dt_rs)

    def _compute_timestep(self):
        """Derive the CFL-safe PDE timestep and wave speed (am/rs).

        dt is set just inside the 3D Courant limit dt ≤ dx/(c·√3). SIM_SPEED scales
        the rendered wave speed (c_amrs) without changing dt, so SIM_SPEED ≤ 1 stays stable.
        """
        CFL_SAFETY = 0.95  # margin below the 3D Courant boundary (1/√3)
        c_phys_amrs = constants.WAVE_SPEED / constants.ATTOMETER * constants.RONTOSECOND
        self.c_amrs = c_phys_amrs * self.SIM_SPEED
        self.dt_rs = CFL_SAFETY * self.wave_field.dx_am / (c_phys_amrs * (3**0.5))
        self.cfl_factor = round((self.c_amrs * self.dt_rs / self.wave_field.dx_am) ** 2, 7)

    def reset_sim(self):
        """Reset simulation state."""
        self.wave_field = None
        self.trackers = None
        self.elapsed_t_rs = 0.0
        self.clock_start_time = time.time()
        self.frame = 1
        self.amp_global_rms = constants.EWAVE_AMPLITUDE
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

    with render.gui.sub_window("XPERIMENT LAUNCHER", 0.00, 0.00, 0.14, 0.37) as sub:
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
    with render.gui.sub_window("CONTROLS", 0.00, 0.37, 0.16, 0.33) as sub:
        state.SHOW_AXIS = sub.checkbox(f"Axis (ticks: {state.TICK_SPACING})", state.SHOW_AXIS)
        state.SHOW_EDGES = sub.checkbox("Sim Universe Edges", state.SHOW_EDGES)
        state.SHOW_FLUX_MESH = sub.slider_int("Flux Mesh", state.SHOW_FLUX_MESH, 0, 3)
        state.WARP_MESH = sub.slider_int("Warp Mesh", state.WARP_MESH, 0, 300)
        state.SHOW_GRANULES = sub.checkbox("Show Granule Motion", state.SHOW_GRANULES)
        state.PARTICLE_SHELL = sub.checkbox("Particle Shell", state.PARTICLE_SHELL)
        state.APPLY_MOTION = sub.checkbox("Apply Motion", state.APPLY_MOTION)
        state.SIM_SPEED = sub.slider_float("Sim Speed", state.SIM_SPEED, 0.5, 1.0)
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
    with render.gui.sub_window("WAVE MENU", 0.00, 0.80, 0.15, 0.20) as sub:
        if sub.checkbox("Displacement (Magnitude)", state.WAVE_MENU == 1):
            state.WAVE_MENU = 1
        if sub.checkbox("Amplitude (EMA RMS)", state.WAVE_MENU == 2):
            state.WAVE_MENU = 2
        if sub.checkbox("Frequency (L&T)", state.WAVE_MENU == 3):
            state.WAVE_MENU = 3
        if sub.checkbox("ENERGY (Field)", state.WAVE_MENU == 4):
            state.WAVE_MENU = 4
        # Display gradient palette with 2× average range for headroom (allows peak visualization)
        if state.WAVE_MENU == 1:  # Displacement on orange gradient
            render.canvas.triangles(og_palette_vertices, per_vertex_color=og_palette_colors)
            with render.gui.sub_window("displacement", 0.00, 0.74, 0.08, 0.06) as sub:
                sub.text(f"0       {state.amp_global_rms*2/state.wave_field.scale_factor:.0e}m")
        if state.WAVE_MENU == 2:  # Amplitude (EMA RMS) on ironbow gradient
            render.canvas.triangles(ib_palette_vertices, per_vertex_color=ib_palette_colors)
            with render.gui.sub_window("amplitude", 0.00, 0.74, 0.08, 0.06) as sub:
                sub.text(f"0       {state.amp_global_rms*2/state.wave_field.scale_factor:.0e}m")
        if state.WAVE_MENU == 3:  # Frequency (L&T) on blueprint gradient
            render.canvas.triangles(bp_palette_vertices, per_vertex_color=bp_palette_colors)
            with render.gui.sub_window("frequency", 0.00, 0.74, 0.08, 0.06) as sub:
                sub.text(f"0       {state.freq_global_avg*2*state.wave_field.scale_factor:.0e}Hz")
        if state.WAVE_MENU == 4:  # Energy on ironbow gradient
            render.canvas.triangles(ib_palette_vertices, per_vertex_color=ib_palette_colors)
            with render.gui.sub_window("energy", 0.00, 0.74, 0.08, 0.06) as sub:
                sub.text(f"0       {state.energy_global_avg*2:.0e}J")


def display_model_specs(state, model_bar_vertices):
    """Display OpenWave model specifications overlay."""
    render.canvas.triangles(model_bar_vertices, color=colormap.YELLOW[1])
    with render.gui.sub_window("EWT MODEL (M4)", 0.84, 0.01, 0.16, 0.16) as sub:
        sub.text("Medium: Indexed Voxel Grid")
        sub.text("Data-Structure: Vector Field")
        sub.text("Coupling: Non-linear Lagrangian")
        sub.text("Propagation: PDE Solver")
        sub.text("Boundary: Dirichlet Condition")
        if sub.button("Wave Notation Guide"):
            webbrowser.open(
                "https://github.com/openwave-labs/openwave/blob/main/openwave/wave_notation.md"
            )


def display_data_dashboard(state):
    """Display simulation data dashboard."""
    clock_time = time.time() - state.clock_start_time
    sim_time_years = clock_time / (state.elapsed_t_rs * constants.RONTOSECOND or 1) / 31_536_000

    with render.gui.sub_window("DATA-DASHBOARD", 0.84, 0.17, 0.16, 0.55) as sub:
        state.INSTRUMENTATION = sub.checkbox("Instrumentation ON/OFF", state.INSTRUMENTATION)
        sub.text("--- SPACETIME ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Medium Density: {constants.MEDIUM_DENSITY:.1e} kg/m³")
        sub.text(f"eWAVE Speed (c): {constants.WAVE_SPEED:.1e} m/s")

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
        sub.text(f"Amplitude: {state.amp_global_rms/state.wave_field.scale_factor:.1e} m")
        sub.text(f"Frequency: {state.freq_global_avg*state.wave_field.scale_factor:.1e} Hz")
        sub.text(f"Wavelength: {state.wavelength_global_avg/state.wave_field.scale_factor:.1e} m")

        sub.text("\n--- TIME MICROSCOPE ---", color=colormap.LIGHT_BLUE[1])
        sub.text(f"Sim Speed: {state.SIM_SPEED:.2f}x")
        sub.text(f"Timestep (dt): {state.dt_rs:.2f} rs  (CFL² {state.cfl_factor:.2f})")
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
    global og_palette_vertices, og_palette_colors
    global ib_palette_vertices, ib_palette_colors
    global bp_palette_vertices, bp_palette_colors
    global model_bar_vertices

    # Initialize color palette scales for gradient rendering and level indicator
    og_palette_vertices, og_palette_colors = colormap.get_palette_scale(
        colormap.orange, 0.00, 0.73, 0.079, 0.01
    )
    ib_palette_vertices, ib_palette_colors = colormap.get_palette_scale(
        colormap.ironbow, 0.00, 0.73, 0.079, 0.01
    )
    bp_palette_vertices, bp_palette_colors = colormap.get_palette_scale(
        colormap.blueprint, 0.00, 0.73, 0.079, 0.01
    )
    model_bar_vertices = colormap.get_model_bar_geometry(0.84, 0.00, 0.159, 0.01)

    if state.INSTRUMENTATION:
        print("\n" + "=" * 64)
        print("INSTRUMENTATION ENABLED")
        print("=" * 64)


def compute_wave_oscillation(state):
    """Compute wave propagation, reflection, superposition and update tracker averages."""

    # Live sim-speed: scale the rendered wave speed; dt stays fixed at the CFL bound
    c_phys = constants.WAVE_SPEED / constants.ATTOMETER * constants.RONTOSECOND
    state.c_amrs = c_phys * state.SIM_SPEED
    state.cfl_factor = round((state.c_amrs * state.dt_rs / state.wave_field.dx_am) ** 2, 7)

    # PDE solver: leapfrog over all voxels every step (no selective-voxel shortcut)
    ewave.propagate_wave(
        state.wave_field,
        state.trackers,
        state.c_amrs,
        state.dt_rs,
        state.elapsed_t_rs,
        state.V_MODE,
        state.V_C1,
        state.V_C2,
    )

    # Re-drive the wave centers on top of the base wave (P3). Mode 0 = free (no re-drive).
    if state.WC_INTERACT_MODE == 1:
        ewave.interact_wc_dirichlet(
            state.wave_field,
            state.wave_center,
            state.elapsed_t_rs,
            state.dt_rs,
            state.WC_BOOST,
            state.WC_RADIUS,
        )
    elif state.WC_INTERACT_MODE == 2:
        ewave.interact_wc_neumann(
            state.wave_field,
            state.wave_center,
            state.elapsed_t_rs,
            state.dt_rs,
            state.WC_BOOST,
            state.WC_RADIUS,
        )
    elif state.WC_INTERACT_MODE == 3:
        ewave.interact_wc_soft(
            state.wave_field,
            state.wave_center,
            state.elapsed_t_rs,
            state.WC_BOOST,
            state.WC_SIGMA,
            state.WC_RADIUS,
        )

    # IN-FRAME DATA SAMPLING & ANALYTICS ==================================
    # Frame skip reduces GPU->CPU transfer overhead
    if state.frame % 60 == 0 or state.frame == 10:
        ewave.sample_avg_trackers(state.wave_field, state.trackers)
    state.amp_global_rms = state.trackers.amp_global_emarms_am[None] * constants.ATTOMETER  # m
    state.freq_global_avg = state.trackers.freq_global_avg_rHz[None] / constants.RONTOSECOND  # Hz
    state.energy_global_avg = state.trackers.energy_global_avg_aJ[None] * constants.ATTOJOULE  # J
    state.wavelength_global_avg = constants.WAVE_SPEED / (
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
        force_motion.integrate_motion_leapfrog(
            state.wave_field,
            state.wave_center,
            state.dt_rs,
        )
    else:
        # Zero-out velocities if not integrating force to motion
        for wc_idx in range(state.wave_center.num_sources):
            state.wave_center.velocity_amrs[wc_idx] = ti.Vector([0.0, 0.0, 0.0], dt=ti.f32)

    # Annihilation naturally occurs from wave physics, but needs numerical precision check
    # Detect and handle particle annihilation (opposite phase WCs meeting)
    # Threshold: WCs can be at grid diagonal positions and dt may cause larger jumps
    annihilation_threshold = state.wave_field.ewave_res / 2.0  # in voxels
    force_motion.detect_annihilation(state.wave_center, annihilation_threshold)


def render_elements(state):
    """Render grid, edges, flux mesh and test particles."""
    if state.SHOW_GRID:
        render.scene.lines(state.wave_field.grid_lines, width=1, color=colormap.COLOR_MEDIUM[1])

    if state.SHOW_EDGES:
        render.scene.lines(state.wave_field.edge_lines, width=1, color=colormap.COLOR_MEDIUM[1])

    if state.SHOW_FLUX_MESH > 0 and state.SHOW_GRANULES == False:
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

    # Render granule positional displacement (only when flux mesh is active, since position
    # is sampled from full-grid displacement data)
    if state.SHOW_GRANULES and state.SHOW_FLUX_MESH > 0:
        max_particles = 401
        granule_radius = 0.001  # in screen space (relative to max universe edge and scale factor)
        amp_boost = state.WARP_MESH  # Boost granule displacement for better visibility
        nx, ny = state.wave_field.nx, state.wave_field.ny
        stride = max(1, int(np.ceil(np.sqrt(nx * ny / max_particles))))
        sampled_nx = (nx + stride - 1) // stride
        sampled_ny = (ny + stride - 1) // stride
        num_render = min(sampled_nx * sampled_ny, max_particles)
        ewave.sample_position_to_render(state.wave_field, amp_boost, stride, num_render)
        pos_np = state.wave_field.position_render.to_numpy()[:num_render]
        render.scene.particles(pos_np, granule_radius, color=colormap.COLOR_MEDIUM[1])


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
    default_xperiment = selected_xperiment_arg or "annihilation1"
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
            compute_wave_oscillation(state)
            compute_force_motion(state)
            state.elapsed_t_rs += state.dt_rs  # Accumulate simulation time (PDE dt)
            state.frame += 1

        # Render scene elements
        render_elements(state)

        # Display additional UI elements and scene
        display_wave_menu(state)
        display_data_dashboard(state)
        display_model_specs(state, model_bar_vertices)
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
