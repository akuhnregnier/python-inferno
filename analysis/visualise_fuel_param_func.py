# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from python_inferno.inferno import fuel_param

if __name__ == "__main__":
    factor = -1.1
    centre = -0.98
    x = np.linspace(-1, 1, 100)
    plt.figure()
    plt.plot(x, fuel_param(x, factor, centre))
    plt.show()
