"""
Configuration settings for the OpenWave project.

This module provides global configuration parameters for OpenWave simulations:

- Color Schemes (RGBA hex)

Includes commented thermal imaging palette definitions for future use.
"""

# ================================================================
# Color Settings (RGBA hex)
# ================================================================
BLACK = ["#000000ff", (0.0, 0.0, 0.0)]
WHITE = ["#ffffffff", (1.0, 1.0, 1.0)]
LIGHT_BLUE = ["#1a99e6ff", (0.1, 0.6, 0.9)]
MAGENTA = ["#ff00eeff", (1.0, 0.0, 0.93)]
CYAN = ["#00ffffff", (0.0, 1.0, 1.0)]
DARK_BLUE = ["#0000ffff", (0.0, 0.0, 1.0)]
ORANGE = ["#ff7f00ff", (1.0, 0.5, 0.0)]
GREEN = ["#04ff00ff", (0.25, 1.0, 0.25)]
YELLOW = ["#ffea00ff", (1.0, 0.92, 0.0)]
RED = ["#ff0000ff", (1.0, 0.0, 0.0)]
PURPLE = ["#8b00ffff", (0.55, 0.0, 0.85)]

# ================================================================
# Color Palette
# ================================================================
COLOR_SPACE = BLACK  # background, void, emptiness
COLOR_INFRA = WHITE  # wire-framing, grids, links
COLOR_MEDIUM = LIGHT_BLUE  # medium, granules
COLOR_QWAVES = ORANGE  # quantum-waves, wave functions
COLOR_FIELDS = CYAN  # fields, field lines
COLOR_MATTER = DARK_BLUE  # matter, particles
COLOR_ANTIMATTER = MAGENTA  # antimatter, antiparticles
COLOR_MOTION = GREEN  # motion, velocity vectors
COLOR_PHOTON = YELLOW  # photons, light particles
COLOR_HEAT = RED  # heat, thermal energy
COLOR_ENERGY = PURPLE  # energy, energy packets


# ================================================================
# Granule Type Classification & Colors
# ================================================================
# Granule types identify position within BCC lattice structure
# Used for visualization, analysis, and debugging lattice topology

# Type constants (integer values for GPU compatibility)
TYPE_VERTEX = 0  # 8 corner vertices of the lattice cube
TYPE_EDGE = 1  # 12 edges of the lattice cube
TYPE_FACE = 2  # 6 faces of the lattice cube
TYPE_CORE = 3  # Interior granules (not on boundary)
TYPE_CENTER = 4  # Single central granule (lattice center)

# Type-specific colors [hex_rgba, (r, g, b)]
COLOR_VERTEX = YELLOW  # cube vertices
COLOR_EDGE = WHITE  # cube edges
COLOR_FACE = ORANGE  # cube faces
COLOR_CORE = COLOR_MEDIUM  # interior granules
COLOR_CENTER = BLACK  # central granule
COLOR_PROBE = RED  # probe granule


# # FUTURE THERMAL IMAGING PALLETTE
# #https://stackoverflow.com/questions/28495390/thermal-imaging-palette

# ironbow        = ["#000000ff", "#20008aff", "#cc0077ff", "#ff0000ff", "#ff7b00ff", "#ffcc00ff", "#FFFFFF"]

# iron_palette = [
# "#00000a","#000014","#00001e","#000025","#00002a","#00002e","#000032","#000036",
# "#00003a","#00003e","#000042","#000046","#00004a","#00004f","#000052","#010055",
# "#010057","#020059","#02005c","#03005e","#040061","#040063","#050065","#060067",
# "#070069","#08006b","#09006e","#0a0070","#0b0073","#0c0074","#0d0075","#0d0076",
# "#0e0077","#100078","#120079","#13007b","#15007c","#17007d","#19007e","#1b0080",
# "#1c0081","#1e0083","#200084","#220085","#240086","#260087","#280089","#2a0089",
# "#2c008a","#2e008b","#30008c","#32008d","#34008e","#36008e","#38008f","#390090",
# "#3b0091","#3c0092","#3e0093","#3f0093","#410094","#420095","#440095","#450096",
# "#470096","#490096","#4a0096","#4c0097","#4e0097","#4f0097","#510097","#520098",
# "#540098","#560098","#580099","#5a0099","#5c0099","#5d009a","#5f009a","#61009b",
# "#63009b","#64009b","#66009b","#68009b","#6a009b","#6c009c","#6d009c","#6f009c",
# "#70009c","#71009d","#73009d","#75009d","#77009d","#78009d","#7a009d","#7c009d",
# "#7e009d","#7f009d","#81009d","#83009d","#84009d","#86009d","#87009d","#89009d",
# "#8a009d","#8b009d","#8d009d","#8f009c","#91009c","#93009c","#95009c","#96009b",
# "#98009b","#99009b","#9b009b","#9c009b","#9d009b","#9f009b","#a0009b","#a2009b",
# "#a3009b","#a4009b","#a6009a","#a7009a","#a8009a","#a90099","#aa0099","#ab0099",
# "#ad0099","#ae0198","#af0198","#b00198","#b00198","#b10197","#b20197","#b30196",
# "#b40296","#b50295","#b60295","#b70395","#b80395","#b90495","#ba0495","#ba0494",
# "#bb0593","#bc0593","#bd0593","#be0692","#bf0692","#bf0692","#c00791","#c00791",
# "#c10890","#c10990","#c20a8f","#c30a8e","#c30b8e","#c40c8d","#c50c8c","#c60d8b",
# "#c60e8a","#c70f89","#c81088","#c91187","#ca1286","#ca1385","#cb1385","#cb1484",
# "#cc1582","#cd1681","#ce1780","#ce187e","#cf187c","#cf197b","#d01a79","#d11b78",
# "#d11c76","#d21c75","#d21d74","#d31e72","#d32071","#d4216f","#d4226e","#d5236b",
# "#d52469","#d62567","#d72665","#d82764","#d82862","#d92a60","#da2b5e","#da2c5c",
# "#db2e5a","#db2f57","#dc2f54","#dd3051","#dd314e","#de324a","#de3347","#df3444",
# "#df3541","#df363d","#e0373a","#e03837","#e03933","#e13a30","#e23b2d","#e23c2a",
# "#e33d26","#e33e23","#e43f20","#e4411d","#e4421c","#e5431b","#e54419","#e54518",
# "#e64616","#e74715","#e74814","#e74913","#e84a12","#e84c10","#e84c0f","#e94d0e",
# "#e94d0d","#ea4e0c","#ea4f0c","#eb500b","#eb510a","#eb520a","#eb5309","#ec5409",
# "#ec5608","#ec5708","#ec5808","#ed5907","#ed5a07","#ed5b06","#ee5c06","#ee5c05",
# "#ee5d05","#ee5e05","#ef5f04","#ef6004","#ef6104","#ef6204","#f06303","#f06403",
# "#f06503","#f16603","#f16603","#f16703","#f16803","#f16902","#f16a02","#f16b02",
# "#f16b02","#f26c01","#f26d01","#f26e01","#f36f01","#f37001","#f37101","#f37201",
# "#f47300","#f47400","#f47500","#f47600","#f47700","#f47800","#f47a00","#f57b00",
# "#f57c00","#f57e00","#f57f00","#f68000","#f68100","#f68200","#f78300","#f78400",
# "#f78500","#f78600","#f88700","#f88800","#f88800","#f88900","#f88a00","#f88b00",
# "#f88c00","#f98d00","#f98d00","#f98e00","#f98f00","#f99000","#f99100","#f99200",
# "#f99300","#fa9400","#fa9500","#fa9600","#fb9800","#fb9900","#fb9a00","#fb9c00",
# "#fc9d00","#fc9f00","#fca000","#fca100","#fda200","#fda300","#fda400","#fda600",
# "#fda700","#fda800","#fdaa00","#fdab00","#fdac00","#fdad00","#fdae00","#feaf00",
# "#feb000","#feb100","#feb200","#feb300","#feb400","#feb500","#feb600","#feb800",
# "#feb900","#feb900","#feba00","#febb00","#febc00","#febd00","#febe00","#fec000",
# "#fec100","#fec200","#fec300","#fec400","#fec500","#fec600","#fec700","#fec800",
# "#fec901","#feca01","#feca01","#fecb01","#fecc02","#fecd02","#fece03","#fecf04",
# "#fecf04","#fed005","#fed106","#fed308","#fed409","#fed50a","#fed60a","#fed70b",
# "#fed80c","#fed90d","#ffda0e","#ffda0e","#ffdb10","#ffdc12","#ffdc14","#ffdd16",
# "#ffde19","#ffde1b","#ffdf1e","#ffe020","#ffe122","#ffe224","#ffe226","#ffe328",
# "#ffe42b","#ffe42e","#ffe531","#ffe635","#ffe638","#ffe73c","#ffe83f","#ffe943",
# "#ffea46","#ffeb49","#ffeb4d","#ffec50","#ffed54","#ffee57","#ffee5b","#ffee5f",
# "#ffef63","#ffef67","#fff06a","#fff06e","#fff172","#fff177","#fff17b","#fff280",
# "#fff285","#fff28a","#fff38e","#fff492","#fff496","#fff49a","#fff59e","#fff5a2",
# "#fff5a6","#fff6aa","#fff6af","#fff7b3","#fff7b6","#fff8ba","#fff8bd","#fff8c1",
# "#fff8c4","#fff9c7","#fff9ca","#fff9cd","#fffad1","#fffad4","#fffbd8","#fffcdb",
# "#fffcdf","#fffde2","#fffde5","#fffde8","#fffeeb","#fffeee","#fffef1","#fffef4",
# "#fffff6"]
