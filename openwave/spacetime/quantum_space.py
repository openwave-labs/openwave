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
