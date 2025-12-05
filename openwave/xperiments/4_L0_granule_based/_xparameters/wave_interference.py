"""
XPERIMENT PARAMETERS

Demonstrates wave interference from three sources arranged in an equilateral triangle.
Each source generates spherical longitudinal waves that superpose at each granule,
creating constructive and destructive interference patterns.

This XPERIMENT showcases:
- 3 wave sources in equilateral triangle pattern
- Wave superposition and interference patterns
- Thin Z dimension for 2.5D visualization
"""

import numpy as np

# Calculate equilateral triangle positions for symmetric interference pattern
EQUILATERAL = np.sqrt(3) / 6  # Height factor for equilateral triangle geometry
Z_POSITION = 0.05  # Z-axis position for all sources

XPARAMETERS = {
    "meta": {
        "X_NAME": "Wave Interference",
        "DESCRIPTION": "3 sources in triangular pattern demonstrating wave superposition",
    },
    "camera": {
        "INITIAL_POSITION": [0.25, 0.91, 1.00],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "SIZE": [2e-16, 1.5e-16, 0.1e-16],  # m, simulation domain [x, y, z]
        "TARGET_GRANULES": 1e6,  # Simulation particle count (impacts performance)
    },
    "wave_sources": {
        "COUNT": 3,  # Number of wave sources for this xperiment
        # Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
        # Arranged in equilateral triangle for symmetric interference pattern
        "POSITIONS": [
            [0.5, 0.45 - EQUILATERAL, Z_POSITION],  # center
            [0.25, 0.45 + EQUILATERAL / 2, Z_POSITION],  # left
            [0.75, 0.45 + EQUILATERAL / 2, Z_POSITION],  # right
        ],
        # Phase offsets for each source (integer degrees, converted to radians internally)
        # All sources in phase (0°) to create symmetric interference
        "PHASE_OFFSETS_DEG": [0, 0, 0],
    },
    "ui_defaults": {
        "SHOW_AXIS": False,  # Toggle to show/hide axis lines
        "TICK_SPACING": 0.25,  # Axis tick marks spacing for position reference
        "BLOCK_SLICE": False,  # Block-slicing toggle
        "SHOW_SOURCES": True,  # Toggle to show/hide wave source markers
        "RADIUS_FACTOR": 1.0,  # Granule radius scaling factor
        "FREQ_BOOST": 1.0,  # Frequency boost multiplier
        "AMP_BOOST": 1.0,  # Amplitude boost multiplier
        "PAUSED": False,  # Pause/Start simulation toggle
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
        "COLOR_PALETTE": 1,  # Color palette list: default (99), granule-type (0), redshift (1), ironbow (2), blueprint (3), viridis (4)
    },
    "analytics": {
        "INSTRUMENTATION": False,  # Toggle data collection (speed & wavelength measurements)
        "EXPORT_VIDEO": False,  # Toggle frame image export to video directory
        "VIDEO_FRAMES": 24,  # Target frame number to stop recording and finalize video export
    },
}
