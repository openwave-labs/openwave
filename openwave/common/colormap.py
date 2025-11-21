"""
Color settings for the OpenWave project.

This module provides global color parameters for OpenWave simulations:

- Color Schemes (RGBA hex)
"""

import taichi as ti

# ================================================================
# Color Definitions: [0] = hex RGBA, [1] = (R,G,B) float tuple
# ================================================================
BLACK = ["#000000", (0.0, 0.0, 0.0)]
WHITE = ["#FFFFFF", (1.0, 1.0, 1.0)]
LIGHT_BLUE = ["#1A99E6", (0.1, 0.6, 0.9)]
BLUE = ["#4E6BC1", (0.306, 0.42, 0.757)]
DARK_BLUE = ["#192C64", (0.1, 0.17, 0.39)]
MAGENTA = ["#FF00EE", (1.0, 0.0, 0.93)]
CYAN = ["#00FFFF", (0.0, 1.0, 1.0)]
ORANGE = ["#FF7B00", (1.0, 0.5, 0.0)]
GREEN = ["#028800", (0.0, 0.53, 0.0)]
YELLOW = ["#FFEA00", (1.0, 0.92, 0.0)]
RED = ["#FF0000", (1.0, 0.0, 0.0)]
PURPLE = ["#8B00FF", (0.55, 0.0, 0.85)]
LIGHT_GRAY = ["#7A7A7A", (0.478, 0.478, 0.478)]
DARK_GRAY = ["#262626", (0.149, 0.149, 0.149)]

# ================================================================
# Color Settings
# ================================================================
COLOR_SPACE = BLACK  # background, void, emptiness
COLOR_INFRA = WHITE  # wire-framing, grid, links
COLOR_FLUXMESH = DARK_GRAY  # flux mesh grid
COLOR_MEDIUM = LIGHT_BLUE  # medium, granules
COLOR_FIELD = CYAN  # fields, field lines
COLOR_PROBE = RED  # probes, sensors
COLOR_SOURCE = ORANGE  # wave source
COLOR_PARTICLE = BLUE  # particles, matter
COLOR_ANTI = PURPLE  # antiparticles, antimatter

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
# GRADIENT COLOR PALETTES
# ================================================================
# NOTE: Color getter functions (get_*_color) are intentionally kept separate
# rather than consolidated into a single generic function. These functions are
# called millions of times per frame in tight simulation loops, and any
# branching or indirection would add measurable overhead. The current approach
# allows Taichi's JIT compiler to optimally inline and vectorize each function.
# Performance-critical code > DRY principle in this case.

# ================================================================
# Ironbow Gradient: PALETTE [used in get_ironbow_color()]
# ================================================================
# Simplified thermal imaging palette (5-color)
# "color_palette" = 1
ironbow_palette = [
    ["#000000", (0.0, 0.0, 0.0)],  # black
    ["#20008A", (0.125, 0.0, 0.541)],  # dark blue
    ["#91009C", (0.569, 0.0, 0.612)],  # magenta
    ["#E64616", (0.902, 0.275, 0.086)],  # red-orange
    ["#FFFFF6", (1.0, 1.0, 0.965)],  # yellow-white
]

# ================================================================
# Blueprint Gradient: PALETTE [used in get_blueprint_color()]
# ================================================================
# Simplified blueprint imaging palette (5-color)
# "color_palette" = 2
blueprint_palette = [
    ["#192C64", (0.098, 0.173, 0.392)],  # dark blue
    ["#405CB1", (0.251, 0.361, 0.694)],  # medium blue
    ["#607DBD", (0.376, 0.490, 0.741)],  # blue
    ["#98AEDD", (0.596, 0.682, 0.867)],  # light blue
    ["#E4EAF6", (0.894, 0.918, 0.965)],  # extra-light blue
]

# ================================================================
# Redshift Gradient: PALETTE [used in get_redshift_color()]
# ================================================================
# Simplified redshift gradient palette (5-color)
# "color_palette" = 3
redshift_palette = [
    ["#FF6347", (1.0, 0.388, 0.278)],  # red-orange
    ["#8B0000", (0.545, 0.0, 0.0)],  # dark red
    ["#1C1C1C", (0.110, 0.110, 0.110)],  # dark gray
    ["#00008B", (0.0, 0.0, 0.545)],  # dark blue
    ["#4169E1", (0.255, 0.412, 0.882)],  # bright blue
]

# ================================================================
# Viridis Gradient: PALETTE [used in get_viridis_color()]
# ================================================================
# Perceptually uniform colormap for scientific visualization
# "color_palette" = 4
viridis_palette = [
    ["#440154", (0.267, 0.004, 0.329)],  # dark purple
    ["#31688E", (0.192, 0.408, 0.557)],  # blue-green
    ["#35B779", (0.208, 0.718, 0.475)],  # green
    ["#BDD93A", (0.741, 0.851, 0.227)],  # yellow-green
    ["#FDE724", (0.992, 0.906, 0.141)],  # bright yellow
]


# ================================================================
# Ironbow Gradient: FUNCTION
# ================================================================

# Taichi-compatible constants for use inside @ti.func
# Extracts RGB tuples from palette for use in both Python and Taichi scopes
ironbow = [color[1] for color in ironbow_palette]
IRONBOW_0 = ti.Vector([ironbow[0][0], ironbow[0][1], ironbow[0][2]])
IRONBOW_1 = ti.Vector([ironbow[1][0], ironbow[1][1], ironbow[1][2]])
IRONBOW_2 = ti.Vector([ironbow[2][0], ironbow[2][1], ironbow[2][2]])
IRONBOW_3 = ti.Vector([ironbow[3][0], ironbow[3][1], ironbow[3][2]])
IRONBOW_4 = ti.Vector([ironbow[4][0], ironbow[4][1], ironbow[4][2]])


@ti.func
def get_ironbow_color(value, min_value, max_value):
    """Maps a numerical value to an IRONBOW thermal camera gradient color.

    IRONBOW gradient: black → dark blue → magenta → red-orange → yellow-white
    Used for thermal visualization where cold = black/blue, hot = yellow/white.

    Optimized for maximum performance with millions of particles.
    Uses palette-derived constants for maintainability with if-elif branches for performance.

    Args:
        value: The displacement magnitude to visualize
        min_value: Minimum displacement in range
        max_value: Maximum displacement observed (from exponential moving average tracker)

    Returns:
        ti.Vector([r, g, b]): RGB color in range [0.0, 1.0] for each component

    Example:
        color = get_ironbow_color(displacement=50, min_value=0, max_value=100)
        # Returns blue-ish color since 50/100 = 0.5 is in the low range
    """

    # Compute normalized scale range with saturation headroom
    scale = max_value - min_value

    # Normalize color by scale range [0.0 - 1.0]
    norm_color = ti.math.clamp((value - min_value) / scale, 0.0, 1.0)

    # Compute color as gradient for visualization with key colors (interpolated)
    r, g, b = 0.0, 0.0, 0.0

    if norm_color < 0.25:
        blend = norm_color / 0.25
        r = IRONBOW_0[0] * (1.0 - blend) + IRONBOW_1[0] * blend
        g = IRONBOW_0[1] * (1.0 - blend) + IRONBOW_1[1] * blend
        b = IRONBOW_0[2] * (1.0 - blend) + IRONBOW_1[2] * blend
    elif norm_color < 0.5:
        blend = (norm_color - 0.25) / 0.25
        r = IRONBOW_1[0] * (1.0 - blend) + IRONBOW_2[0] * blend
        g = IRONBOW_1[1] * (1.0 - blend) + IRONBOW_2[1] * blend
        b = IRONBOW_1[2] * (1.0 - blend) + IRONBOW_2[2] * blend
    elif norm_color < 0.75:
        blend = (norm_color - 0.5) / 0.25
        r = IRONBOW_2[0] * (1.0 - blend) + IRONBOW_3[0] * blend
        g = IRONBOW_2[1] * (1.0 - blend) + IRONBOW_3[1] * blend
        b = IRONBOW_2[2] * (1.0 - blend) + IRONBOW_3[2] * blend
    else:
        blend = (norm_color - 0.75) / 0.25
        r = IRONBOW_3[0] * (1.0 - blend) + IRONBOW_4[0] * blend
        g = IRONBOW_3[1] * (1.0 - blend) + IRONBOW_4[1] * blend
        b = IRONBOW_3[2] * (1.0 - blend) + IRONBOW_4[2] * blend

    ironbow_color = ti.Vector([r, g, b])

    return ironbow_color


# ================================================================
# Blueprint Gradient: FUNCTION
# ================================================================

# Taichi-compatible constants for use inside @ti.func
# Extracts RGB tuples from palette for use in both Python and Taichi scopes
blueprint = [color[1] for color in blueprint_palette]
BLUEPRINT_0 = ti.Vector([blueprint[0][0], blueprint[0][1], blueprint[0][2]])
BLUEPRINT_1 = ti.Vector([blueprint[1][0], blueprint[1][1], blueprint[1][2]])
BLUEPRINT_2 = ti.Vector([blueprint[2][0], blueprint[2][1], blueprint[2][2]])
BLUEPRINT_3 = ti.Vector([blueprint[3][0], blueprint[3][1], blueprint[3][2]])
BLUEPRINT_4 = ti.Vector([blueprint[4][0], blueprint[4][1], blueprint[4][2]])


@ti.func
def get_blueprint_color(value, min_value, max_value):
    """Maps a numerical value to a BLUEPRINT color gradient.

    BLUEPRINT gradient: dark blue → medium blue → blue → light blue → extra-light blue
    Used for blueprint-style visualization where low = dark, high = light.

    Optimized for maximum performance with millions of voxels.
    Uses palette-derived constants for maintainability with if-elif branches for performance.

    Args:
        value: The displacement magnitude to visualize
        min_value: Minimum displacement in range
        max_value: Maximum displacement observed

    Returns:
        ti.Vector([r, g, b]): RGB color in range [0.0, 1.0] for each component

    Example:
        color = get_blueprint_color(value=50, min_value=0, max_value=100)
        # Returns blue since 50/100 = 0.5 is in the middle range
    """

    # Compute normalized scale range with saturation headroom
    scale = max_value - min_value

    # Normalize color by scale range [0.0 - 1.0]
    norm_color = ti.math.clamp((value - min_value) / scale, 0.0, 1.0)

    # Compute color as gradient for visualization with key colors (interpolated)
    r, g, b = 0.0, 0.0, 0.0

    if norm_color < 0.25:
        blend = norm_color / 0.25
        r = BLUEPRINT_0[0] * (1.0 - blend) + BLUEPRINT_1[0] * blend
        g = BLUEPRINT_0[1] * (1.0 - blend) + BLUEPRINT_1[1] * blend
        b = BLUEPRINT_0[2] * (1.0 - blend) + BLUEPRINT_1[2] * blend
    elif norm_color < 0.5:
        blend = (norm_color - 0.25) / 0.25
        r = BLUEPRINT_1[0] * (1.0 - blend) + BLUEPRINT_2[0] * blend
        g = BLUEPRINT_1[1] * (1.0 - blend) + BLUEPRINT_2[1] * blend
        b = BLUEPRINT_1[2] * (1.0 - blend) + BLUEPRINT_2[2] * blend
    elif norm_color < 0.75:
        blend = (norm_color - 0.5) / 0.25
        r = BLUEPRINT_2[0] * (1.0 - blend) + BLUEPRINT_3[0] * blend
        g = BLUEPRINT_2[1] * (1.0 - blend) + BLUEPRINT_3[1] * blend
        b = BLUEPRINT_2[2] * (1.0 - blend) + BLUEPRINT_3[2] * blend
    else:
        blend = (norm_color - 0.75) / 0.25
        r = BLUEPRINT_3[0] * (1.0 - blend) + BLUEPRINT_4[0] * blend
        g = BLUEPRINT_3[1] * (1.0 - blend) + BLUEPRINT_4[1] * blend
        b = BLUEPRINT_3[2] * (1.0 - blend) + BLUEPRINT_4[2] * blend

    blueprint_color = ti.Vector([r, g, b])

    return blueprint_color


# ================================================================
# Redshift Gradient: FUNCTION
# ================================================================

# Taichi-compatible constants for use inside @ti.func
# Extracts RGB tuples from palette for use in both Python and Taichi scopes
redshift = [color[1] for color in redshift_palette]
REDSHIFT_0 = ti.Vector([redshift[0][0], redshift[0][1], redshift[0][2]])
REDSHIFT_1 = ti.Vector([redshift[1][0], redshift[1][1], redshift[1][2]])
REDSHIFT_2 = ti.Vector([redshift[2][0], redshift[2][1], redshift[2][2]])
REDSHIFT_3 = ti.Vector([redshift[3][0], redshift[3][1], redshift[3][2]])
REDSHIFT_4 = ti.Vector([redshift[4][0], redshift[4][1], redshift[4][2]])


@ti.func
def get_redshift_color(value, min_value, max_value):
    """Maps a signed numerical value to a REDSHIFT gradient color.

    REDSHIFT gradient: red-orange → dark red → gray → dark blue → bright blue
    Used for displacement visualization where negative = red, zero = gray, positive = blue.

    Optimized for maximum performance with millions of voxels.
    Uses palette-derived constants for maintainability with if-elif branches for performance.

    Args:
        value: The signed displacement value to visualize (can be negative or positive)
        min_value: Minimum displacement in range
        max_value: Maximum displacement in range

    Returns:
        ti.Vector([r, g, b]): RGB color in range [0.0, 1.0] for each component

    Example:
        color = get_redshift_color(displacement=-50, min_value=-100, max_value=100)
        # Returns red-ish color since -50 is in the negative range
    """

    # Compute normalized scale range with saturation headroom
    scale = max_value - min_value

    # Normalize color by scale range [0.0 - 1.0]
    norm_color = ti.math.clamp((value - min_value) / scale, 0.0, 1.0)

    # Compute color as gradient for visualization with key colors (interpolated)
    r, g, b = 0.0, 0.0, 0.0

    if norm_color < 0.25:
        blend = norm_color / 0.25
        r = REDSHIFT_0[0] * (1.0 - blend) + REDSHIFT_1[0] * blend
        g = REDSHIFT_0[1] * (1.0 - blend) + REDSHIFT_1[1] * blend
        b = REDSHIFT_0[2] * (1.0 - blend) + REDSHIFT_1[2] * blend
    elif norm_color < 0.5:
        blend = (norm_color - 0.25) / 0.25
        r = REDSHIFT_1[0] * (1.0 - blend) + REDSHIFT_2[0] * blend
        g = REDSHIFT_1[1] * (1.0 - blend) + REDSHIFT_2[1] * blend
        b = REDSHIFT_1[2] * (1.0 - blend) + REDSHIFT_2[2] * blend
    elif norm_color < 0.75:
        blend = (norm_color - 0.5) / 0.25
        r = REDSHIFT_2[0] * (1.0 - blend) + REDSHIFT_3[0] * blend
        g = REDSHIFT_2[1] * (1.0 - blend) + REDSHIFT_3[1] * blend
        b = REDSHIFT_2[2] * (1.0 - blend) + REDSHIFT_3[2] * blend
    else:
        blend = (norm_color - 0.75) / 0.25
        r = REDSHIFT_3[0] * (1.0 - blend) + REDSHIFT_4[0] * blend
        g = REDSHIFT_3[1] * (1.0 - blend) + REDSHIFT_4[1] * blend
        b = REDSHIFT_3[2] * (1.0 - blend) + REDSHIFT_4[2] * blend

    redshift_color = ti.Vector([r, g, b])

    return redshift_color


# ================================================================
# Viridis Gradient: FUNCTION
# ================================================================

# Taichi-compatible constants for use inside @ti.func
# Extracts RGB tuples from palette for use in both Python and Taichi scopes
viridis = [color[1] for color in viridis_palette]
VIRIDIS_0 = ti.Vector([viridis[0][0], viridis[0][1], viridis[0][2]])
VIRIDIS_1 = ti.Vector([viridis[1][0], viridis[1][1], viridis[1][2]])
VIRIDIS_2 = ti.Vector([viridis[2][0], viridis[2][1], viridis[2][2]])
VIRIDIS_3 = ti.Vector([viridis[3][0], viridis[3][1], viridis[3][2]])
VIRIDIS_4 = ti.Vector([viridis[4][0], viridis[4][1], viridis[4][2]])


@ti.func
def get_viridis_color(value, min_value, max_value):
    """Maps a signed numerical value to a VIRIDIS gradient color.

    VIRIDIS gradient: dark purple → blue-green → green → yellow-green → bright yellow
    Used for displacement visualization with perceptually uniform color transitions.

    Valley (negative): dark purple (shadowed depth) → green (neutral)
    Hill (positive): green (neutral) → bright yellow (illuminated peak)

    Optimized for maximum performance with millions of voxels.
    Uses palette-derived constants for maintainability with if-elif branches for performance.

    Args:
        value: The signed displacement value to visualize (can be negative or positive)
        min_value: Minimum displacement in range
        max_value: Maximum displacement in range

    Returns:
        ti.Vector([r, g, b]): RGB color in range [0.0, 1.0] for each component

    Example:
        color = get_viridis_color(value=-50, min_value=-100, max_value=100)
        # Returns purple-ish color since -50 is in the negative range
    """

    # Compute normalized scale range with saturation headroom
    scale = max_value - min_value

    # Normalize color by scale range [0.0 - 1.0]
    norm_color = ti.math.clamp((value - min_value) / scale, 0.0, 1.0)

    # Compute color as gradient for visualization with key colors (interpolated)
    r, g, b = 0.0, 0.0, 0.0

    if norm_color < 0.25:
        blend = norm_color / 0.25
        r = VIRIDIS_0[0] * (1.0 - blend) + VIRIDIS_1[0] * blend
        g = VIRIDIS_0[1] * (1.0 - blend) + VIRIDIS_1[1] * blend
        b = VIRIDIS_0[2] * (1.0 - blend) + VIRIDIS_1[2] * blend
    elif norm_color < 0.5:
        blend = (norm_color - 0.25) / 0.25
        r = VIRIDIS_1[0] * (1.0 - blend) + VIRIDIS_2[0] * blend
        g = VIRIDIS_1[1] * (1.0 - blend) + VIRIDIS_2[1] * blend
        b = VIRIDIS_1[2] * (1.0 - blend) + VIRIDIS_2[2] * blend
    elif norm_color < 0.75:
        blend = (norm_color - 0.5) / 0.25
        r = VIRIDIS_2[0] * (1.0 - blend) + VIRIDIS_3[0] * blend
        g = VIRIDIS_2[1] * (1.0 - blend) + VIRIDIS_3[1] * blend
        b = VIRIDIS_2[2] * (1.0 - blend) + VIRIDIS_3[2] * blend
    else:
        blend = (norm_color - 0.75) / 0.25
        r = VIRIDIS_3[0] * (1.0 - blend) + VIRIDIS_4[0] * blend
        g = VIRIDIS_3[1] * (1.0 - blend) + VIRIDIS_4[1] * blend
        b = VIRIDIS_3[2] * (1.0 - blend) + VIRIDIS_4[2] * blend

    viridis_color = ti.Vector([r, g, b])

    return viridis_color


# ================================================================
# Generic Palette Scale Function
# ================================================================


def get_palette_scale(color_palette, x, y, width, height):
    """Generate palette scale indicator with geometry and colors as horizontal gradient.

    Generic function for creating palette display. Works with any color palette
    (ironbow, blueprint, redshift, viridis, etc.).

    Creates a horizontal color bar with gradient transitions between colors.
    Each color band is made of 2 triangles forming a rectangle.
    Canvas coordinates: (0,0) at bottom-left, X increases to the right.

    Args:
        color_palette: List of RGB tuples extracted from palette definition
        x: Left edge x-coordinate (normalized 0-1)
        y: Top edge y-coordinate (normalized 0-1)
        width: Width of palette bar (normalized 0-1)
        height: Height of palette bar (normalized 0-1)

    Returns:
        tuple: (vertices_field, colors_field) for rendering with canvas.triangles()

    Example:
        vertices, colors = get_palette_scale(ironbow, 0.02, 0.02, 0.15, 0.02)
        canvas.triangles(vertices, per_vertex_color=colors)
    """
    # Calculate number of vertices needed
    num_bands = len(color_palette) - 1  # N colors = N-1 transitions
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

        # Get colors for this transition
        color_left = ti.Vector(color_palette[i])
        color_right = ti.Vector(color_palette[i + 1])

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


def get_level_bar_geometry(x, y, width, height):
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
# COLOR_MOTION = GREEN  # motion, velocity vectors
# COLOR_PHOTON = YELLOW  # photons
# COLOR_HEAT = RED  # heat, thermal energy
# COLOR_ENERGY = PURPLE  # energy, energy packets
