#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from python_inferno.inferno import sigmoid

if __name__ == "__main__":
    factor = 0.05
    centre = 2000
    shape = 9.2
    x = np.linspace(0, 1422, 400, dtype=np.float32)
    plt.figure()
    plt.plot(x, sigmoid(x, factor, centre, shape))
    plt.show()
