"""
XPERIMENT PARAMETERS

Demonstrates wave interference from multiple sources in the lattice.
Each source generates spherical longitudinal waves that superpose at each granule,
creating constructive and destructive interference patterns.

This XPERIMENT showcases:
- Multiple wave sources (9 sources: center + 8 corners)
- Wave superposition and interference patterns
- Phase control between sources (constructive/destructive interference)
"""

XPARAMETERS = {
    "meta": {
        "X_NAME": "Spacetime Vibration",
        "DESCRIPTION": "Harmonic Oscillations with 9 sources demonstrating wave interference",
    },
    "camera": {
        "INITIAL_POSITION": [2.00, 1.50, 1.75],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "SIZE": [1e-16, 1e-16, 1e-16],  # m, simulation domain [x, y, z]
        "TARGET_GRANULES": 1e6,  # Simulation particle count (impacts performance)
    },
    "wave_sources": {
        "COUNT": 9,  # Number of wave sources for this xperiment
        # Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
        # Each row represents [x, y, z] coordinates for one source (Z-up coordinate system)
        "POSITIONS": [
            [0.5, 0.5, 0.5],  # Wave Source 0 - Center
            [0.0, 1.0, 1.0],  # Wave Source 1 - Back-top-left corner
            [1.0, 0.0, 1.0],  # Wave Source 2 - Front-top-right corner
            [0.0, 1.0, 0.0],  # Wave Source 3 - Back-bottom-left corner
            [1.0, 0.0, 0.0],  # Wave Source 4 - Front-bottom-right corner
            [0.0, 0.0, 1.0],  # Wave Source 5 - Front-top-left corner
            [1.0, 1.0, 1.0],  # Wave Source 6 - Back-top-right corner
            [0.0, 0.0, 0.0],  # Wave Source 7 - Front-bottom-left corner
            [1.0, 1.0, 0.0],  # Wave Source 8 - Back-bottom-right corner
        ],
        # Phase offsets for each source (integer degrees, converted to radians internally)
        # Center source at 180° creates destructive interference with corner sources at 0°
        "PHASE_OFFSETS_DEG": [0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "ui_defaults": {
        "SHOW_AXIS": False,  # Toggle to show/hide axis lines
        "TICK_SPACING": 0.25,  # Axis tick marks spacing for position reference
        "BLOCK_SLICE": False,  # Block-slicing toggle
        "SHOW_SOURCES": False,  # Toggle to show/hide wave source markers
        "RADIUS_FACTOR": 0.5,  # Granule radius scaling factor
        "FREQ_BOOST": 10.0,  # Frequency boost multiplier
        "AMP_BOOST": 1.0,  # Amplitude boost multiplier
        "PAUSED": False,  # Pause/Start simulation toggle
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
        "COLOR_PALETTE": 0,  # default (99), granule-type (0), ironbow (3), orange (6)
    },
    "analytics": {
        "INSTRUMENTATION": False,  # Toggle data collection (speed & wavelength measurements)
        "EXPORT_VIDEO": False,  # Toggle frame image export to video directory
        "VIDEO_FRAMES": 24,  # Target frame number to stop recording and finalize video export
    },
}
