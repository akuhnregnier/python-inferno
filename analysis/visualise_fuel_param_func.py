#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from python_inferno.inferno import sigmoid

if __name__ == "__main__":
    factor = -1.1
    centre = -0.98
    shape = 1.2
    x = np.linspace(-1, 1, 100)
    plt.figure()
    plt.plot(x, sigmoid(x, factor, centre, shape))
    plt.show()
