#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from warnings import filterwarnings

import iris
import matplotlib.pyplot as plt
import numpy as np
from jules_output_analysis.data import get_n96e_land_mask, regrid_to_n96e
from scipy.stats import lognorm
from wildfires.configuration import DATA_DIR
from wildfires.utils import match_shape

from python_inferno.plotting import use_style

if __name__ == "__main__":
    filterwarnings("ignore", ".*divide by zero.*")
    filterwarnings("ignore", ".*invalid units.*")
    filterwarnings("ignore", ".*may not be fully.*")
    filterwarnings("ignore", ".*axes.*")
    filterwarnings("ignore")
    use_style()

    cube_2d = regrid_to_n96e(
        iris.load_cube(str(Path(DATA_DIR) / "GFED4_climatology.nc"))
    )
    cube_2d.data.mask |= match_shape(
        ~get_n96e_land_mask(),
        cube_2d.shape,
    )

    data = np.ma.getdata(cube_2d.data)[~np.ma.getmaskarray(cube_2d.data)]
    params = lognorm.fit(data)
    params

    xs = np.geomspace(1e-8, 1, 1000000)
    plt.figure()
    plt.plot(xs, lognorm.pdf(xs, *params))
    plt.yscale("log")
    plt.xscale("log")

    plt.figure()
    plt.hist(
        np.ma.getdata(cube_2d.data)[~np.ma.getmaskarray(cube_2d.data)],
        bins=np.append([0], np.geomspace(1e-4, 1, 100)),
    )
    plt.yscale("log")
    plt.xscale("log")

    plt.show()
