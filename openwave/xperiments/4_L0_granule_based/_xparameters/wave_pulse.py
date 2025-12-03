"""
XPERIMENT PARAMETERS: Wave Pulse

Demonstrates radiation from a single wave source.
A single source generates spherical longitudinal waves that propagate radially outward.

This XPERIMENT showcases:
- Single wave source at center
- Wave pulse radiation
- Wave diagnostics enabled for speed and wavelength measurements
"""

XPARAMETERS = {
    "meta": {
        "X_NAME": "Wave Pulse",
        "DESCRIPTION": "Single source radiation with wave diagnostics",
    },
    "camera": {
        "INITIAL_POSITION": [1.35, 0.91, 0.68],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "SIZE": [1e-16, 1e-16, 1e-16],  # m, simulation domain [x, y, z]
        "TARGET_GRANULES": 1e6,  # Simulation particle count (impacts performance)
    },
    "wave_sources": {
        "COUNT": 1,  # Number of wave sources for this xperiment
        # Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
        # Single source at center for pure spherical wave pulse radiation
        "POSITIONS": [[0.5, 0.5, 0.5]],  # Wave Source position - Center
        # Phase offset in degrees (0° = in phase with base frequency)
        "PHASE_OFFSETS_DEG": [0],
    },
    "ui_defaults": {
        "SHOW_AXIS": True,  # Toggle to show/hide axis lines
        "TICK_SPACING": 0.25,  # Axis tick marks spacing for position reference
        "BLOCK_SLICE": False,  # Block-slicing toggle
        "SHOW_SOURCES": False,  # Toggle to show/hide wave source markers
        "RADIUS_FACTOR": 0.1,  # Granule radius scaling factor
        "FREQ_BOOST": 10.0,  # Frequency boost multiplier
        "AMP_BOOST": 5.0,  # Amplitude boost multiplier
        "PAUSED": False,  # Pause/Start simulation toggle
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
        "COLOR_PALETTE": 99,  # Color palette list: default (99), granule-type (0), redshift (1), ironbow (2), blueprint (3), viridis (4)
    },
    "analytics": {
        "ANALYTICS": True,  # Toggle data analytics (speed & wavelength measurements)
        "EXPORT_VIDEO": False,  # Toggle frame image export to video directory
        "VIDEO_FRAMES": 24,  # Target frame number to stop recording and finalize video export
    },
}
