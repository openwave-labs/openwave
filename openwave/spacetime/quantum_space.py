import numpy as np
import taichi as ti

import openwave.core.config as config
import openwave.core.constants as constants
import openwave.core.equations as equations

ti.init(arch=ti.gpu)


class Granule:
    def __init__(self, position):
        self.position = position


N = 2  # config.UNIVERSE_SIZE
x = ti.Vector.field(2, dtype=float, shape=(N, N * 2))

# print(x)


# universe size, m [knob]
# granule object: radius, m
# lattice object: separation length, m
# resolution factor (computational efficiency) [knob]
# number of granules = calculated from resolution factor
# display (convert to screen coordinates/size, positioning, color)
# GUI widgets = knobs
