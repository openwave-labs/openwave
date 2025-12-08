"""
XPERIMENT PARAMETERS: Wave Sourced Magnetic Field-Like Interference

Demonstrates wave interference from 2 sources in a linear arrangement.
Each source generates spherical longitudinal waves that superpose at each granule,
creating constructive and destructive interference patterns.

This XPERIMENT showcases:
- 2 wave sources pattern creating a magnetic field-like interference
- Wave superposition and interference patterns
- Thin Z dimension for 2.5D visualization
"""

# Generate 2 sources in a linear pattern
Z_POSITION = 0.07  # Z-axis position for all sources

XPARAMETERS = {
    "meta": {
        "X_NAME": "Wave Magnet",
        "DESCRIPTION": "2 sources in linear pattern demonstrating wave superposition",
    },
    "camera": {
        "INITIAL_POSITION": [0.44, 0.94, 1.22],  # [x, y, z] in normalized coordinates
    },
    "universe": {
        "SIZE": [5e-16, 5e-16, 0.1e-16],  # m, simulation domain [x, y, z]
        "TARGET_GRANULES": 1e6,  # Simulation particle count (impacts performance)
    },
    "wave_sources": {
        "COUNT": 2,  # Number of wave sources for this xperiment
        # Wave Source positions: normalized coordinates (0-1 range, relative to max universe edge)
        # Arranged in equilateral triangle for symmetric interference pattern
        "POSITIONS": [
            [0.25, 0.50, Z_POSITION],  # left
            [0.75, 0.50, Z_POSITION],  # right
        ],
        # Phase offsets for each source (integer degrees, converted to radians internally)
        # All sources in phase (0°) to create symmetric interference
        "PHASE_OFFSETS_DEG": [0, 0],
    },
    "ui_defaults": {
        "SHOW_AXIS": False,  # Toggle to show/hide axis lines
        "TICK_SPACING": 0.25,  # Axis tick marks spacing for position reference
        "BLOCK_SLICE": False,  # Block-slicing toggle
        "SHOW_SOURCES": False,  # Toggle to show/hide wave source markers
        "RADIUS_FACTOR": 1.0,  # Granule radius scaling factor
        "FREQ_BOOST": 1.0,  # Frequency boost multiplier
        "AMP_BOOST": 1.0,  # Amplitude boost multiplier
        "PAUSED": False,  # Pause/Start simulation toggle
    },
    "color_defaults": {
        "COLOR_THEME": "OCEAN",  # Choose color theme for rendering (OCEAN, DESERT, FOREST)
        "COLOR_PALETTE": 2,  # Color palette list: default (99), granule-type (0), redshift (1), ironbow (2), blueprint (3), viridis (4), orange (5)
    },
    "analytics": {
        "INSTRUMENTATION": False,  # Toggle data collection (speed & wavelength measurements)
        "EXPORT_VIDEO": False,  # Toggle frame image export to video directory
        "VIDEO_FRAMES": 24,  # Target frame number to stop recording and finalize video export
    },
}
