"""
XPERIMENT PARAMETERS: Crossing Waves

Demonstrates wave interference from multiple sources in the lattice.
Each source generates spherical longitudinal waves that superpose at each granule,
creating constructive and destructive interference patterns with crossing wave fronts.

This XPERIMENT showcases:
- Multiple wave sources (9 sources: center + 8 corners)
- Crossing wave patterns
- Phase control between sources (constructive/destructive interference)
- DESERT color theme for different visual experience
- No spring coupling (pure wave propagation)
"""

PARAMETERS = {
    "meta": {
        "name": "Crossing Waves",
        "description": "Crossing Waves Harmonic Oscillations with 9 sources",
    },
    "camera": {
        "initial_position": [2.00, 1.50, 1.75],
    },
    "universe": {
        "size_multipliers": [4, 4, 4],  # Multiplies EWAVE_LENGTH
        "tick_spacing": 0.25,
        "color_theme": "DESERT",  # Only xperiment using DESERT theme
    },
    "wave_sources": {
        "count": 9,
        "positions": [
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
        "phase_offsets_deg": [180, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "ui_defaults": {
        "show_axis": False,
        "block_slice": False,
        "show_sources": True,
        "radius_factor": 1.0,
        "freq_boost": 1.0,
        "amp_boost": 5.0,
        "paused": False,
        "granule_type": True,
        "ironbow": False,
        "blueprint": False,
        "var_displacement": True,
    },
    "diagnostics": {
        "wave_diagnostics": False,
        "export_video": False,
        "video_frames": 24,
    },
}
