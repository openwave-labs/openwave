"""
XPERIMENT PARAMETERS: Wave Pulse

Demonstrates radiation from a single wave source.
A single source generates spherical longitudinal waves that propagate radially outward.

This XPERIMENT showcases:
- Single wave source at center
- Wave pulse radiation
- Wave diagnostics enabled for speed and wavelength measurements
- No spring coupling (pure wave propagation)
"""

PARAMETERS = {
    "meta": {
        "name": "Wave Pulse",
        "description": "Single source radiation with wave diagnostics",
    },
    "camera": {
        "initial_position": [1.35, 0.91, 0.68],
    },
    "universe": {
        "size": [1e-16, 1e-16, 1e-16],  # m
        "tick_spacing": 0.25,
        "color_theme": "OCEAN",
    },
    "wave_sources": {
        "count": 1,
        "positions": [[0.5, 0.5, 0.5]],  # Wave Source position - Center
        "phase_offsets_deg": [0],
    },
    "ui_defaults": {
        "show_axis": True,
        "block_slice": False,
        "show_sources": False,
        "radius_factor": 0.1,
        "freq_boost": 10.0,
        "amp_boost": 5.0,
        "paused": False,
        "granule_type": False,
        "ironbow": False,
        "blueprint": False,
        "var_displacement": True,
    },
    "diagnostics": {
        "wave_diagnostics": True,  # Only xperiment with diagnostics enabled by default
        "export_video": False,
        "video_frames": 24,
    },
}
