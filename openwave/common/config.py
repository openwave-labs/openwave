"""
Configuration settings for the OpenWave project.

This module provides global configuration parameters for OpenWave simulations:

- Resolution & Magnification Settings
- Color Schemes (RGBA hex)

Includes commented thermal imaging palette definitions for future use.
"""

import taichi as ti

from openwave.common import constants

# ================================================================
# Resolution & Magnification Settings
# ================================================================
TARGET_GRANULES = 1e6  # target particle count, granularity (impacts performance)
SLOW_MO = constants.EWAVE_FREQUENCY  # slows frequency down to 1Hz for human visibility

# ================================================================
# Color Definitions: [0] = hex RGBA, [1] = (R,G,B) float tuple
# ================================================================
BLACK = ["#000000ff", (0.0, 0.0, 0.0)]
WHITE = ["#ffffffff", (1.0, 1.0, 1.0)]
LIGHT_BLUE = ["#1a99e6ff", (0.1, 0.6, 0.9)]
MAGENTA = ["#ff00eeff", (1.0, 0.0, 0.93)]
CYAN = ["#00ffffff", (0.0, 1.0, 1.0)]
DARK_BLUE = ["#1a3366ff", (0.1, 0.2, 0.4)]
ORANGE = ["#ff7b00ff", (1.0, 0.5, 0.0)]
GREEN = ["#028800ff", (0.0, 0.53, 0.0)]
YELLOW = ["#ffea00ff", (1.0, 0.92, 0.0)]
RED = ["#ff0000ff", (1.0, 0.0, 0.0)]
PURPLE = ["#8b00ffff", (0.55, 0.0, 0.85)]

# ================================================================
# Color Settings
# ================================================================
COLOR_SPACE = BLACK  # background, void, emptiness
COLOR_INFRA = WHITE  # wire-framing, grid, links
COLOR_MEDIUM = LIGHT_BLUE  # medium, granules
COLOR_FIELD = CYAN  # fields, field lines
COLOR_PROBE = RED  # probes, sensors
COLOR_SOURCE = ORANGE  # wave source

# ================================================================
# Color Themes
# ================================================================
OCEAN = {
    "COLOR_VERTEX": BLACK,  # vertices
    "COLOR_EDGE": BLACK,  # edges
    "COLOR_FACE": DARK_BLUE,  # faces
}

DESERT = {
    "COLOR_VERTEX": WHITE,  # vertices
    "COLOR_EDGE": WHITE,  # edges
    "COLOR_FACE": ORANGE,  # faces
}

FOREST = {
    "COLOR_VERTEX": BLACK,  # vertices
    "COLOR_EDGE": BLACK,  # edges
    "COLOR_FACE": GREEN,  # faces
}

# ================================================================
# Ironbow Thermal Imaging Palette
# ================================================================
# Simplified thermal imaging palette (5-color)
ironbow5 = [
    "#00000A",  # black
    "#20008A",  # dark blue
    "#91009C",  # magenta
    "#E64616",  # red-orange
    "#FFFFF6",  # yellow-white
]

# Thermal imaging palette (7-color)
ironbow7 = [
    "#000000ff",  # black
    "#20008aff",  # dark blue
    "#cc0077ff",  # magenta
    "#ff0000ff",  # red
    "#ff7b00ff",  # orange
    "#ffcc00ff",  # yellow
    "#FFFFFF",  # white
]


@ti.func
def get_ironbow_color(value, min_value, max_value, saturation):
    """Maps a numerical value to an IRONBOW thermal camera gradient color.

    IRONBOW gradient: black → dark blue → magenta → red-orange → yellow-white
    Used for thermal visualization where cold = black/blue, hot = yellow/white.

    Args:
        value: The displacement magnitude to visualize
        min_value: Minimum displacement in range (typically 0.0)
        max_value: Maximum displacement observed (from exponential moving average tracker)
        saturation: Headroom factor to prevent color saturation (e.g., 3.0 = use only 1/3 of range)
                   Higher values = darker colors, lower values = brighter colors

    Returns:
        ti.Vector([r, g, b]): RGB color in range [0.0, 1.0] for each component

    Example:
        color = get_ironbow_color(displacement=50, min_value=0, max_value=100, saturation=3.0)
        # Returns blue-ish color since 50/(100*3) = 0.167 is in the low range
    """

    # Compute normalized scale range with saturation headroom
    # saturation > 1 leaves headroom (prevents saturation at max displacement)
    scale = (max_value - min_value) * saturation

    # Normalize color by scale range [0.0 - 1.0]
    norm_color = ti.math.clamp(value / scale, 0.0, 1.0)

    # Compute color as IRONBOW gradient for visualization
    # ironbow5 gradient with key colors (interpolated)
    r, g, b = 0.0, 0.0, 0.0

    if norm_color < 0.25:  # black to dark blue
        blend = norm_color / 0.25
        r = 0.0 * (1.0 - blend) + 0.125 * blend  # #00000A to #20008A
        g = 0.0 * (1.0 - blend) + 0.0 * blend
        b = 0.04 * (1.0 - blend) + 0.54 * blend
    elif norm_color < 0.5:  # dark blue to magenta
        blend = (norm_color - 0.25) / 0.25
        r = 0.125 * (1.0 - blend) + 0.57 * blend  # #20008A to #91009C
        g = 0.0 * (1.0 - blend) + 0.0 * blend
        b = 0.54 * (1.0 - blend) + 0.61 * blend
    elif norm_color < 0.75:  # magenta to red-orange
        blend = (norm_color - 0.5) / 0.25
        r = 0.57 * (1.0 - blend) + 0.90 * blend  # #91009C to #E64616
        g = 0.0 * (1.0 - blend) + 0.27 * blend
        b = 0.61 * (1.0 - blend) + 0.09 * blend
    else:  # red-orange to yellow-white
        blend = (norm_color - 0.75) / 0.25
        r = 0.90 * (1.0 - blend) + 1.0 * blend  # #E64616 to #FFFFF6
        g = 0.27 * (1.0 - blend) + 1.0 * blend
        b = 0.09 * (1.0 - blend) + 0.96 * blend

    ironbow_color = ti.Vector([r, g, b])

    return ironbow_color


# ================================================================
# FUTURE COLOR PALETTES
# COLOR_EWAVE = ORANGE  # energy-wave, wave functions
# COLOR_MATTER = DARK_BLUE  # matter, particles
# COLOR_ANTIMATTER = MAGENTA  # antimatter, antiparticles
# COLOR_MOTION = GREEN  # motion, velocity vectors
# COLOR_PHOTON = YELLOW  # photons
# COLOR_HEAT = RED  # heat, thermal energy
# COLOR_ENERGY = PURPLE  # energy, energy packets
