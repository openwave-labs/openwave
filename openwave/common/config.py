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
TARGET_GRANULES = 1e6  # target particle count, wave granularity (impacts performance)
TARGET_VOXELS = 1e3  # target voxel count, wave granularity (impacts performance)
SLOW_MO = constants.EWAVE_FREQUENCY  # slows frequency down to 1Hz for human visibility

# ================================================================
# Color Definitions: [0] = hex RGBA, [1] = (R,G,B) float tuple
# ================================================================
BLACK = ["#000000", (0.0, 0.0, 0.0)]
WHITE = ["#FFFFFF", (1.0, 1.0, 1.0)]
LIGHT_BLUE = ["#1A99E6", (0.1, 0.6, 0.9)]
MAGENTA = ["#FF00EE", (1.0, 0.0, 0.93)]
CYAN = ["#00FFFF", (0.0, 1.0, 1.0)]
DARK_BLUE = ["#192C64", (0.1, 0.17, 0.39)]
ORANGE = ["#FF7B00", (1.0, 0.5, 0.0)]
GREEN = ["#028800", (0.0, 0.53, 0.0)]
YELLOW = ["#FFEA00", (1.0, 0.92, 0.0)]
RED = ["#FF0000", (1.0, 0.0, 0.0)]
PURPLE = ["#8B00FF", (0.55, 0.0, 0.85)]

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
    ["#000000", (0.0, 0.0, 0.0)],  # black
    ["#20008A", (0.125, 0.0, 0.54)],  # dark blue
    ["#91009C", (0.57, 0.0, 0.61)],  # magenta
    ["#E64616", (0.90, 0.27, 0.09)],  # red-orange
    ["#FFFFF6", (1.0, 1.0, 0.96)],  # yellow-white
]


@ti.func
def get_ironbow_color(value, min_value, max_value, saturation=1.0):
    """Maps a numerical value to an IRONBOW thermal camera gradient color.

    IRONBOW gradient: black → dark blue → magenta → red-orange → yellow-white
    Used for thermal visualization where cold = black/blue, hot = yellow/white.

    Optimized for maximum performance with millions of particles.
    Uses hardcoded if-elif branches for fastest execution (no matrix lookups).

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
        r = 0.0 * (1.0 - blend) + 0.125 * blend  # #000000 to #20008A
        g = 0.0 * (1.0 - blend) + 0.0 * blend
        b = 0.0 * (1.0 - blend) + 0.54 * blend
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


def ironbow_palette(x, y, width, height):
    """Generate ironbow palette indicator as horizontal gradient using triangles.

    Creates a horizontal color bar from all 5 ironbow colors (black -> yellow-white).
    Each color band is made of 2 triangles forming a rectangle.
    Canvas coordinates: (0,0) at bottom-left, X increases to the right.

    Returns:
        tuple: (vertices_field, colors_field) for rendering with canvas.triangles()
    """
    # Calculate number of vertices needed: 4 color bands × 2 triangles × 3 vertices = 24
    num_bands = len(ironbow5) - 1  # 4 color transitions
    num_vertices = num_bands * 6  # Each band = 2 triangles × 3 vertices

    # Create Taichi fields for triangle vertices and colors
    palette_vertices = ti.Vector.field(2, dtype=ti.f32, shape=num_vertices)
    palette_colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)

    # Position parameters (screen coordinates)
    x_left = x
    x_right = x + width
    y_top = 1 - y
    y_bottom = 1 - (y + height)
    band_width = (x_right - x_left) / num_bands

    # Generate triangles for each color band
    for i in range(num_bands):
        # Calculate X positions for this band
        x_start = x_left + (i * band_width)
        x_end = x_left + ((i + 1) * band_width)

        # Get colors from ironbow5 (index 1 is the RGB tuple)
        color_left = ti.Vector(ironbow5[i][1])  # Cold color (left)
        color_right = ti.Vector(ironbow5[i + 1][1])  # Hot color (right)

        # Vertex indices for this band's two triangles
        v_idx = i * 6

        # First triangle (bottom-left, bottom-right, top-left)
        palette_vertices[v_idx + 0] = ti.Vector([x_start, y_bottom])
        palette_vertices[v_idx + 1] = ti.Vector([x_end, y_bottom])
        palette_vertices[v_idx + 2] = ti.Vector([x_start, y_top])
        palette_colors[v_idx + 0] = color_left
        palette_colors[v_idx + 1] = color_right
        palette_colors[v_idx + 2] = color_left

        # Second triangle (top-left, bottom-right, top-right)
        palette_vertices[v_idx + 3] = ti.Vector([x_start, y_top])
        palette_vertices[v_idx + 4] = ti.Vector([x_end, y_bottom])
        palette_vertices[v_idx + 5] = ti.Vector([x_end, y_top])
        palette_colors[v_idx + 3] = color_left
        palette_colors[v_idx + 4] = color_right
        palette_colors[v_idx + 5] = color_right

    return palette_vertices, palette_colors


# ================================================================
# Blueprint Imaging Palette
# ================================================================
# Simplified blueprint imaging palette (4-color)
blueprint4 = [
    ["#192C64", (0.1, 0.17, 0.39)],  # dark blue
    ["#405CB1", (0.25, 0.36, 0.69)],  # medium blue
    ["#98AEDD", (0.6, 0.68, 0.87)],  # light blue
    ["#E4EAF6", (0.9, 0.94, 0.98)],  # extra-light blue
]


@ti.func
def get_blueprint_color(value, min_value, max_value):
    """Maps a numerical value to a BLUEPRINT color gradient.

    BLUEPRINT gradient: dark blue → medium blue → light blue → extra-light blue
    Used for blueprint-style visualization where low = dark, high = light.

    Args:
        value: The displacement magnitude to visualize
        min_value: Minimum displacement in range (typically 0.0)
        max_value: Maximum displacement observed
    """
    # Compute normalized scale range
    scale = max_value - min_value

    # Normalize color by scale range [0.0 - 1.0]
    norm_color = ti.math.clamp(value / scale, 0.0, 1.0)

    # Compute color as BLUEPRINT gradient for visualization
    # blueprint4 gradient with key colors (interpolated)
    r, g, b = 0.0, 0.0, 0.0
    if norm_color < 0.33:  # dark blue to medium blue
        blend = norm_color / 0.33
        r = 0.1 * (1.0 - blend) + 0.25 * blend  # #192C64 to #405CB1
        g = 0.17 * (1.0 - blend) + 0.36 * blend
        b = 0.39 * (1.0 - blend) + 0.69 * blend
    elif norm_color < 0.66:  # medium blue to light blue
        blend = (norm_color - 0.33) / 0.33
        r = 0.25 * (1.0 - blend) + 0.6 * blend  # #405CB1 to #98AEDD
        g = 0.36 * (1.0 - blend) + 0.68 * blend
        b = 0.69 * (1.0 - blend) + 0.87 * blend
    else:  # light blue to extra-light blue
        blend = (norm_color - 0.66) / 0.34
        r = 0.6 * (1.0 - blend) + 0.9 * blend  # #98AEDD to #E4EAF6
        g = 0.68 * (1.0 - blend) + 0.94 * blend
        b = 0.87 * (1.0 - blend) + 0.98 * blend

    blueprint_color = ti.Vector([r, g, b])

    return blueprint_color


def blueprint_palette(x, y, width, height):
    """Generate blueprint palette indicator as horizontal gradient using triangles.

    Creates a horizontal color bar from all 4 blueprint colors (dark blue -> extra-light blue).
    Each color band is made of 2 triangles forming a rectangle.
    Canvas coordinates: (0,0) at bottom-left, X increases to the right.

    Returns:
        tuple: (vertices_field, colors_field) for rendering with canvas.triangles()
    """
    # Calculate number of vertices needed: 3 color bands × 2 triangles × 3 vertices = 18
    num_bands = len(blueprint4) - 1  # 3 color transitions
    num_vertices = num_bands * 6  # Each band = 2 triangles × 3 vertices

    # Create Taichi fields for triangle vertices and colors
    palette_vertices = ti.Vector.field(2, dtype=ti.f32, shape=num_vertices)
    palette_colors = ti.Vector.field(3, dtype=ti.f32, shape=num_vertices)

    # Position parameters (screen coordinates)
    x_left = x
    x_right = x + width
    y_top = 1 - y
    y_bottom = 1 - (y + height)
    band_width = (x_right - x_left) / num_bands

    # Generate triangles for each color band
    for i in range(num_bands):
        # Calculate X positions for this band
        x_start = x_left + (i * band_width)
        x_end = x_left + ((i + 1) * band_width)

        # Get colors from blueprint4 (index 1 is the RGB tuple)
        color_left = ti.Vector(blueprint4[i][1])  # Dark color (left)
        color_right = ti.Vector(blueprint4[i + 1][1])  # Light color (right)

        # Vertex indices for this band's two triangles
        v_idx = i * 6

        # First triangle (bottom-left, bottom-right, top-left)
        palette_vertices[v_idx + 0] = ti.Vector([x_start, y_bottom])
        palette_vertices[v_idx + 1] = ti.Vector([x_end, y_bottom])
        palette_vertices[v_idx + 2] = ti.Vector([x_start, y_top])
        palette_colors[v_idx + 0] = color_left
        palette_colors[v_idx + 1] = color_right
        palette_colors[v_idx + 2] = color_left

        # Second triangle (top-left, bottom-right, top-right)
        palette_vertices[v_idx + 3] = ti.Vector([x_start, y_top])
        palette_vertices[v_idx + 4] = ti.Vector([x_end, y_bottom])
        palette_vertices[v_idx + 5] = ti.Vector([x_end, y_top])
        palette_colors[v_idx + 3] = color_left
        palette_colors[v_idx + 4] = color_right
        palette_colors[v_idx + 5] = color_right

    return palette_vertices, palette_colors


# ================================================================
# Level Bar Geometry - visual indicator for xperiment Level
# ================================================================


def level_bar_geometry(x, y, width, height):
    """Generate level bar geometry as two triangles forming a rectangle.

    Creates a horizontal bar at specified screen coordinates.
    Canvas coordinates: (0,0) at bottom-left, X increases to the right.

    Returns:
        ti.Vector.field: vertices_field for rendering with canvas.triangles()
    """
    # Create Taichi field for triangle vertices
    level_bar_vertices = ti.Vector.field(2, dtype=ti.f32, shape=6)

    # Position parameters (screen coordinates)
    x_left = x
    x_right = x + width
    y_top = 1 - y
    y_bottom = 1 - (y + height)

    # First triangle (bottom-left, bottom-right, top-left)
    level_bar_vertices[0] = ti.Vector([x_left, y_bottom])
    level_bar_vertices[1] = ti.Vector([x_right, y_bottom])
    level_bar_vertices[2] = ti.Vector([x_left, y_top])

    # Second triangle (top-left, bottom-right, top-right)
    level_bar_vertices[3] = ti.Vector([x_left, y_top])
    level_bar_vertices[4] = ti.Vector([x_right, y_bottom])
    level_bar_vertices[5] = ti.Vector([x_right, y_top])

    return level_bar_vertices


# ================================================================
# FUTURE COLOR PALETTES
# COLOR_EWAVE = ORANGE  # energy-wave, wave functions
# COLOR_MATTER = DARK_BLUE  # matter, particles
# COLOR_ANTIMATTER = MAGENTA  # antimatter, antiparticles
# COLOR_MOTION = GREEN  # motion, velocity vectors
# COLOR_PHOTON = YELLOW  # photons
# COLOR_HEAT = RED  # heat, thermal energy
# COLOR_ENERGY = PURPLE  # energy, energy packets

# Thermal imaging palette (7-color)
# ironbow7 = [
#     "#000000ff",  # black
#     "#20008aff",  # dark blue
#     "#cc0077ff",  # magenta
#     "#ff0000ff",  # red
#     "#ff7b00ff",  # orange
#     "#ffcc00ff",  # yellow
#     "#FFFFFF",  # white
# ]
