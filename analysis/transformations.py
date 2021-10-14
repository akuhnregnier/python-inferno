#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import partial
from warnings import filterwarnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jules_output_analysis.data import regrid_to_n96e
from wildfires.data import GFEDv4
from wildfires.utils import get_land_mask, match_shape

if __name__ == "__main__":
    filterwarnings("ignore", ".*divide by zero.*")
    filterwarnings("ignore", ".*invalid units.*")
    filterwarnings("ignore", ".*may not be fully.*")
    filterwarnings("ignore", ".*axes.*")
    filterwarnings("ignore")
    mpl.rc_file("matplotlibrc")

    gfed = GFEDv4()
    gfed = gfed.get_climatology_dataset(gfed.min_time, gfed.max_time)
    print(gfed)

    gfed.cube.data.mask |= match_shape(get_land_mask(), gfed.cube.shape)
    print("shape:", gfed.cube.shape)

    data = regrid_to_n96e(gfed.cube).data

    valid = data.data[~data.mask]

    xs = np.geomspace(1e-9, 1e10, 1000)
    plt.plot(xs, np.arcsinh(xs), label="arcsinh")
    plt.plot(xs, np.log(xs), label="log")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()

    def arcsinh_func(data, factor):
        return np.arcsinh(factor * data)

    for transformation, title in [
        (lambda data: data, "Raw BA Data (GFED4 Clim)"),
        (lambda data: np.log(data[data > 1e-9]), "Log (BA>1e-9)"),
        *(
            (
                partial(arcsinh_func, factor=factor),
                f"Inverse Hyperbolic Sine (data x {factor:0.1e})",
            )
            for factor in np.geomspace(1e4, 1e8, 3)
        ),
    ]:
        plt.figure()
        plt.hist(transformation(valid), bins="auto")
        plt.title(title)
        plt.yscale("log")

    plt.show()
