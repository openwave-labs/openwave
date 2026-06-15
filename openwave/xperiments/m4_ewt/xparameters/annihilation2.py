"""
XPERIMENT PARAMETERS

This XPERIMENT showcases:
- Opposite-phase WC annihilation (diagonal approach, ~1λ separation)
- WCs placed at ~1λ apart (crosses one barrier at λ/2 with initial velocity)
- Tests whether KE overcomes one barrier before r=0 well captures them

PHYSICS: At ~1λ separation, there is one barrier at λ/2 between the WCs and r=0.
  The initial velocity provides KE to cross it. If they get past the λ/2 barrier,
  the r=0 well captures them for annihilation.
"""

UNIVERSE_EDGE = 1e-15  # m, universe edge length in meters
TARGET_VOXELS = 100_000_000  # Target voxel count (impacts performance)

XPARAMETERS = {
    "meta": {
        "X_NAME": f"  /Annihilation 2",
        "DESCRIPTION": "Opposite-phase WC annihilation",
    },
    "camera": {
        "INITIAL_POSITION": [1.64, 0.88, 0.61],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "SIZE": [UNIVERSE_EDGE, UNIVERSE_EDGE, UNIVERSE_EDGE],  # m, simulation domain [x, y, z]
        "TARGET_VOXELS": TARGET_VOXELS,  # Simulation voxel count (impacts performance)
    },
    "wave_centers": {
        "COUNT": 2,  # Number of wave-centers for this xperiment
        # Wave-Center positions: normalized coordinates (0-1 range, relative to max universe edge)
        "POSITION": [
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.25],
        ],
        # Phase offsets for each wave-center (integer degrees, converted to radians internally)
        "PHASE_OFFSETS_DEG": [0, 180],
        # Initial velocity [vx, vy, vz] in am/rs (c = 0.3 am/rs)
        "INIT_VELOCITY": [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        "APPLY_MOTION": True,  # Toggle to apply motion at wave-centers, from force at each iteration
    },
    "ui_defaults": {
        "SHOW_AXIS": True,  # Toggle to show/hide axis lines
        "TICK_SPACING": 0.25,  # Axis tick marks spacing for position reference
        "SHOW_GRID": False,  # Toggle to show/hide the voxel data-grid
        "SHOW_EDGES": True,  # Toggle to show/hide universe edges
        "FLUX_MESH_PLANES": [0.5, 0.5, 0.5],  # [x, y, z] positions relative to universe size
        "SHOW_FLUX_MESH": 2,  # Flux Mesh toggle, 0: none, 1: xy, 2: xy+xz, 3: xy+xz+yz
        "WARP_MESH": 300,  # Visual warp mesh effect intensity
        "SHOW_GRANULES": False,  # Toggle to show/hide granule particles (rendered as points)
        "PARTICLE_SHELL": True,  # Toggle to enable/disable particle shell rendering
        "TIMESTEP": 15.0,  # Simulation timestep in rontoseconds (10-27s)
        "PAUSED": False,  # Pause/Start simulation toggle
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
        "WAVE_MENU": 2,  # Check _launcher.py display_wave_menu() for wave_menu indexing
    },
    "analytics": {
        "INSTRUMENTATION": False,  # Toggle data acquisition and analytics
        "EXPORT_VIDEO": False,  # Toggle frame image export to video directory
        "VIDEO_FRAMES": 24,  # Target frame number to stop recording and finalize video export
    },
}
