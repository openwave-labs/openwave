"""
Configuration settings for the OpenWave project.

This module provides global configuration parameters for OpenWave simulations:

- Resolution & Magnification Settings
- Color Schemes (RGBA hex)

Includes commented thermal imaging palette definitions for future use.
"""

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
# FUTURE COLOR PALETTES
# COLOR_EWAVE = ORANGE  # energy-wave, wave functions
# COLOR_MATTER = DARK_BLUE  # matter, particles
# COLOR_ANTIMATTER = MAGENTA  # antimatter, antiparticles
# COLOR_MOTION = GREEN  # motion, velocity vectors
# COLOR_PHOTON = YELLOW  # photons
# COLOR_HEAT = RED  # heat, thermal energy
# COLOR_ENERGY = PURPLE  # energy, energy packets

# THERMAL IMAGING PALETTE
# https://stackoverflow.com/questions/28495390/thermal-imaging-palette
# ironbow5 = ["#00000A", "#20008A", "#91009C", "#E64616", "#FFFFF6"]
# ironbow7 = ["#000000ff", "#20008aff", "#cc0077ff", "#ff0000ff", "#ff7b00ff", "#ffcc00ff", "#FFFFFF"]
