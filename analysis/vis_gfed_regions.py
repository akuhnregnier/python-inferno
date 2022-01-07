#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from wildfires.analysis import cube_plotting

from python_inferno.plotting import get_gfed_regions

if __name__ == "__main__":
    gfed_regions, N_regions = get_gfed_regions()
    cube_plotting(gfed_regions, boundaries=np.arange(0, N_regions + 1) + 0.5)
    plt.show()
