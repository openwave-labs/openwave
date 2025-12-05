"""
XPERIMENT PARAMETERS

Demonstrates wave interference from multiple sources in the lattice.
Each source generates spherical longitudinal waves that superpose at each granule,
creating constructive and destructive interference patterns.

This XPERIMENT showcases:
- Single wave source at center
- Pure spherical wave propagation
- Radial wave patterns
"""

XPARAMETERS = {
    "meta": {
        "X_NAME": "3D Spherical Wave",
        "DESCRIPTION": "Single source spherical wave demonstrating radial propagation",
    },
    "camera": {
        "INITIAL_POSITION": [0.97, 2.06, 0.82],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "SIZE": [1e-16, 1e-16, 1e-16],  # m, simulation domain [x, y, z]
        "TARGET_GRANULES": 1e6,  # Simulation particle count (impacts performance)
    },
    "wave_sources": {
        "COUNT": 1,  # Number of wave sources for this xperiment
        # Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
        # Single source at center for pure spherical wave propagation
        "POSITIONS": [[0.5, 0.5, 0.5]],  # Wave Source position - Center
        # Phase offset in degrees (0° = in phase with base frequency)
        "PHASE_OFFSETS_DEG": [0],
    },
    "ui_defaults": {
        "SHOW_AXIS": True,  # Toggle to show/hide axis lines
        "TICK_SPACING": 0.25,  # Axis tick marks spacing for position reference
        "BLOCK_SLICE": True,  # Block-slicing toggle
        "SHOW_SOURCES": False,  # Toggle to show/hide wave source markers
        "RADIUS_FACTOR": 0.4,  # Granule radius scaling factor
        "FREQ_BOOST": 0.5,  # Frequency boost multiplier
        "AMP_BOOST": 5.0,  # Amplitude boost multiplier
        "PAUSED": False,  # Pause/Start simulation toggle
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
        "COLOR_PALETTE": 99,  # Color palette list: default (99), granule-type (0), redshift (1), ironbow (2), blueprint (3), viridis (4)
    },
    "analytics": {
        "ANALYTICS": False,  # Toggle data analytics (speed & wavelength measurements)
        "EXPORT_VIDEO": False,  # Toggle frame image export to video directory
        "VIDEO_FRAMES": 24,  # Target frame number to stop recording and finalize video export
    },
}
